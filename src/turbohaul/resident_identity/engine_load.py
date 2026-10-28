"""The load of one engine: the number the manager uses to pick an engine for a new request.

An engine's load is the work it has pending: the riders in flight, the request it is running
right now, and the requests waiting in its inbox. An engine in its grace loop keeps the slot of
the last request it served, and that slot is in the finished state: nothing is running, so the
slot is not load. Counting it would make a waiting engine look as busy as one in the middle of a turn,
so a new request could be sent to the busy engine while the waiting one sat unused. An engine
in its grace loop that is serving a follow-up turn is running, so its slot still counts.
"""

from __future__ import annotations

from typing import Protocol

_FINISHED_STATE = "GRACE"


class _SlotLike(Protocol):
    """The request in a resident's active slot. ``state`` is optional: a slot without it counts as running."""

    state: object  # optional: compared with the plain string of the finished state


class LoadLike(Protocol):
    """One engine's registry entry: the attributes the load reads.

    ``in_grace_loop`` and ``inbox`` are optional: an object that lacks them reads as not in its
    grace loop and as having no inbox. ``inflight`` and ``active_slot`` must be present. The
    active slot is read for its optional ``state`` only.
    """

    inflight: list
    active_slot: _SlotLike | None
    in_grace_loop: bool  # optional: read with a default of False
    inbox: object | None  # optional: an object with qsize(); read with a default of None


def pending_load(r: LoadLike) -> int:
    """The work pending on engine ``r``: riders in flight, plus one for a running turn, plus the inbox depth.

    The running-turn term is one when the engine has an active slot, unless the engine is in its
    grace loop and that slot is in the finished state ("GRACE"): then the slot is the request it
    already served, nothing is running, and it adds nothing. Both conditions are needed. A slot
    that is in the grace loop but serving a follow-up turn, a slot outside the grace loop, and a
    slot with no state all count. The state is compared with a plain string, which a string enum
    equals, so this module imports nothing. Riders in flight and requests in the inbox always
    count. Read-only.
    """
    slot = r.active_slot
    finished = getattr(r, "in_grace_loop", False) and getattr(slot, "state", None) == _FINISHED_STATE
    running = 1 if slot is not None and not finished else 0
    inbox = getattr(r, "inbox", None)
    queued = inbox.qsize() if inbox is not None else 0
    return len(r.inflight) + running + queued
