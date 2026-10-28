"""The periodic re-check: start a further engine for waiting demand when it fits.

A request that arrived while an engine was busy waits in that engine's inbox (or in the
staging queue). Nothing asks again whether another engine would now fit. This is that
question, asked on a bounded schedule from the dispatch loop: one attempt per interval for
the whole box, one probe per tag per no-room window.

Three phases, with the lock order queue lock then registry lock, never the reverse:

* A: probe the fit under the registry lock. No request is taken yet, so a no-fit verdict
  costs nothing and leaves the request where it was.
* B: take the request out of its container, with no registry lock held.
* C: reserve the engine directly on the card the fit check chose. The request is never
  sent back through the arrival path: that path enforces the box budget and may evict an
  idle engine to make room, which would turn co-residence into a swap.
"""

from __future__ import annotations

import asyncio
import logging

from turbohaul import queue_liveness

from .constants import RECHECK_MIN_INTERVAL_S
from .host import CoResidenceHost
from .lines import format_recheck_action
from .scan import gather_candidates, pick_candidate, take_from_inbox
from .verdict import fit_verdict
from .window import recheck_window

_log = logging.getLogger(__name__)


async def run_recheck(host: CoResidenceHost) -> None:
    """Run one re-check pass. Never raises, except for a cancellation."""
    try:
        await _run(host)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - the dispatch loop must go on whatever happens here
        _log.exception("co-residence re-check failed (dispatch continues)")


async def _run(host: CoResidenceHost) -> None:
    state = host._co_residence_state()
    now = host._co_residence_now()
    gate = recheck_window(now=now, last_recheck_at=state.last_recheck_at, min_interval_s=RECHECK_MIN_INTERVAL_S)
    if not gate.open:
        return
    # Attempts are counted, so a scan that finds nothing is debounced too.
    state.last_recheck_at = now
    grace_pairs = await host._grace_active_exclusions()
    # No await between the snapshot and the pick.
    cand = pick_candidate(host, state, gather_candidates(host, now, grace_pairs), now)
    if cand is None:
        return
    age, _rank, source, cslot, owner = cand
    tag = cslot.model_tag

    # Phase A: probe under the registry lock, no request taken yet.
    async with host._registry_lock:
        insts = host._instances_for(tag)
        if not insts:
            return
        n_insts = len(insts)
        verdict = await fit_verdict(host, state, tag, insts, now)
    if not verdict.admit:
        state.norom.note(tag, now)
        return

    # Phase B: take the request out of its container, with no registry lock held.
    _log.info(
        "%s",
        format_recheck_action(
            tag=tag, source=source, slot_id=cslot.slot_id, waited_s=age,
            instances=n_insts, residents=len(host._model_residents()),
        ),
    )
    if source == "staged":
        taken = await host.queue.remove(cslot.slot_id)
    else:
        taken = take_from_inbox(owner, cslot)
        if taken is not None:
            host._unpark_fastlane_claim(taken)
    if taken is None:
        return  # the dispatcher or the engine got it first
    if queue_liveness.client_gone(taken):
        host._handle_unloaded_slot(taken)
        return

    # Phase C: reserve directly with the card the fit check chose.
    admitted = False
    try:
        # The marker names the in-flight reservation for the status display. It is set before
        # the lock is taken and cleared by the context manager on every exit.
        with host._reserving(tag):
            async with host._registry_lock:
                new = await host._reserve_and_start_locked(
                    taken, main_gpu_override=verdict.card, split_mode_override=verdict.split
                )
                if new is not None:
                    host._remember_affinity(tag, host._session_key_for(taken), new.resident_key)
                    admitted = True
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - fail this one request, keep the dispatch loop alive
        _log.exception("co-residence re-check reserve of slot %s failed", taken.slot_id)
        host._fail_completion_future(taken, RuntimeError("dispatch routing failed"))
    if admitted:
        state.norom.clear(tag)
    else:
        state.norom.note(tag, now)
