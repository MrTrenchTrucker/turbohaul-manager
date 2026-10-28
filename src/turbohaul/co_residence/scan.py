"""Finding the waiting request the re-check should act on.

Demand waits in two places: the staging queue (held aside, for example for a grace window)
and the inbox of a busy engine (parked behind it by the arrival path). This file lists the
requests that have waited long enough, and picks the one to act on. It only reads; taking
the request out is the caller's job, apart from the inbox helper below.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Sequence
from typing import NamedTuple

from turbohaul import queue_liveness

from .decision import RecheckInputs, decide_recheck
from .host import CoResidenceHost, _ResidentView, _SlotView
from .state import RecheckState
from .window import demand_is_eligible


class Candidate(NamedTuple):
    """One waiting request: its age, source rank (0 staged, 1 inbox), source, slot and owner."""

    age: float
    rank: int
    source: str
    slot: _SlotView
    owner: _ResidentView | None


def inbox_slots(r: _ResidentView) -> tuple:
    """The slots waiting in ``r``'s inbox right now, oldest first.

    Read-only: a snapshot of the backing deque. Synchronous and await-free.
    """
    return tuple(getattr(getattr(r, "inbox", None), "_queue", ()) or ())


def take_from_inbox(r: _ResidentView, slot: _SlotView) -> _SlotView | None:
    """Take ``slot`` (matched by identity) out of ``r``'s inbox; None when it is not there.

    The other slots stay in their original order. Synchronous and await-free, so the
    engine's driver cannot interleave: it drains with ``get_nowait`` and puts the rest back.
    """
    inbox = getattr(r, "inbox", None)
    if inbox is None:
        return None
    drained = []
    while True:
        try:
            drained.append(inbox.get_nowait())
        except asyncio.QueueEmpty:
            break
    taken = None
    for s in drained:
        if taken is None and s is slot:
            taken = s
        else:
            inbox.put_nowait(s)
    return taken


def gather_candidates(host: CoResidenceHost, now: float, grace_pairs: Collection[tuple[str, str]]) -> list[Candidate]:
    """Every waiting request old enough to count, oldest first, staged before inbox on a tie.

    A staged request counts only for a tag with a live engine. An inbox request counts only
    when its engine is ACTIVE (an idle engine serves its inbox at once) and it is for that
    engine's model. A request whose thread an engine's grace loop still holds, or that is the
    engine's own idle thread, is protected and must wait longer. No await happens here.
    """
    cands: list[Candidate] = []
    for s in host.queue.staging_snapshot():
        if queue_liveness.client_gone(s) or not host._instances_for(s.model_tag):
            continue
        age = now - s.created_at
        if demand_is_eligible(age_s=age, protected=(s.thread_id, s.model_tag) in grace_pairs):
            cands.append(Candidate(age, 0, "staged", s, None))
    for r in host._model_residents():
        if r.state.name != "ACTIVE":
            continue
        for s in inbox_slots(r):
            if queue_liveness.client_gone(s) or s.model_tag != r.model_tag:
                continue
            protected = (s.thread_id, s.model_tag) in grace_pairs or bool(
                s.thread_id and s.thread_id == r.idle_thread_id
            )
            age = now - s.created_at
            if demand_is_eligible(age_s=age, protected=protected):
                cands.append(Candidate(age, 1, "inbox", s, r))
    cands.sort(key=lambda c: (-c.age, c.rank))
    return cands


def pick_candidate(
    host: CoResidenceHost, state: RecheckState, cands: Sequence[Candidate], now: float
) -> Candidate | None:
    """The oldest candidate of the first tag the per-tag decision lets run, or None.

    ``cands`` must be in the order ``gather_candidates`` returns. A tag with a live
    no-room memo entry, or with no live engine, is passed over.
    """
    counts: dict[str, int] = {}
    oldest: dict[str, Candidate] = {}
    for c in cands:
        tag = c.slot.model_tag
        counts[tag] = counts.get(tag, 0) + 1
        oldest.setdefault(tag, c)
    for tag, c in oldest.items():
        decision = decide_recheck(
            RecheckInputs(
                tag_live_instances=len(host._instances_for(tag)),
                eligible_demand=counts[tag],
                memo_blocked=state.norom.blocked(tag, now),
            )
        )
        if decision.run:
            return c
    return None
