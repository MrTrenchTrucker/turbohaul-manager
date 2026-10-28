"""Taking a waiting request whose client hung up out of the system, at the source.

``evict_unstarted`` removes one waiting request from wherever it waits. ``notify_client_gone``
is the synchronous entry a route calls when a disconnect is seen: it sets the client's event
and schedules the eviction of every waiting request that carries that event.
"""

from __future__ import annotations

import asyncio
import logging

from .liveness import _fail_once
from .liveness_host import QueueLivenessHost, ResidentLike, SlotLike

_log = logging.getLogger(__name__)

# The slot states before an engine turn has begun, compared by name (no enum import).
_UNSTARTED = frozenset({"RECEIVED", "ACCEPT_BUFFER", "STAGED"})


def _mid_turn(r: ResidentLike) -> bool:
    """True when the resident's driver is busy with a turn (so it will come back to its loop)."""
    return r.state.name == "ACTIVE" and r.active_slot is not None


def _drain(inbox) -> list:
    items: list = []
    while True:
        try:
            items.append(inbox.get_nowait())
        except asyncio.QueueEmpty:
            return items


def _inbox_items(inbox) -> list:
    """The slots in an inbox, oldest first. Await-free: they are put straight back in order."""
    items = _drain(inbox)
    for s in items:
        inbox.put_nowait(s)
    return items


def _inbox_take(inbox, slot: SlotLike) -> bool:
    """Take ``slot`` (matched by identity) out of an inbox, leaving the rest in order.

    Await-free, so the engine's driver cannot interleave. True when the slot was there.
    """
    taken = False
    for s in _drain(inbox):
        if not taken and s is slot:
            taken = True
        else:
            inbox.put_nowait(s)
    return taken


async def evict_unstarted(host: QueueLivenessHost, slot: SlotLike) -> bool:
    """Remove one waiting request at the source: flag it, free its claim, drop it from the queues, fail it.

    Returns True when this call evicted the slot, False when it did nothing: the slot has
    already begun, or it was already failed (a second call is a no-op).

    The flags are set before any await, so no take path can pick the slot meanwhile. The slot
    leaves a resident's inbox only when that resident is mid-turn (see ``_mid_turn``). A parked
    driver must not lose its last queued item: a producer flips such a resident to ACTIVE before
    it puts, and only the driver's own loop flips it back to idle, so an emptied inbox would
    leave it ACTIVE for good. For a parked or loading resident the slot stays in the inbox,
    flagged and failed, and the driver discards it. The claim is released in every case.
    """
    future = slot.completion_future
    if slot.state.name not in _UNSTARTED or (future is not None and future.done()):
        return False
    if slot.disconnect_event is not None:
        slot.disconnect_event.set()
    slot.is_evicted = True
    async with host._registry_lock:
        for r in host._model_residents():
            if _mid_turn(r) and r.inbox is not None and _inbox_take(r.inbox, slot):
                break
        host._release_fastlane_claim_locked(slot, "disconnected")
    await host.queue.remove(slot.slot_id)
    if slot.state.name not in _UNSTARTED:
        return False
    return _fail_once(host, slot)


def _waiting_with_event(host: QueueLivenessHost, event) -> list:
    """Every waiting slot (queue, acceptance buffer, any resident's inbox) that carries ``event``."""
    candidates = list(host.queue.waiting_snapshot())
    for r in host._model_residents():
        if r.inbox is not None:
            candidates.extend(_inbox_items(r.inbox))
    found: list = []
    for s in candidates:
        if s.disconnect_event is event and not any(s is f for f in found):
            found.append(s)
    return found


async def _evict_for_event(host: QueueLivenessHost, event) -> None:
    """Evict every waiting slot whose disconnect event is ``event``; one failure does not stop the rest."""
    for slot in _waiting_with_event(host, event):
        try:
            await evict_unstarted(host, slot)
        except Exception:  # noqa: BLE001 - one failed eviction must not stop the others
            _log.warning("evicting a waiting request after a client hang-up failed", exc_info=True)


def notify_client_gone(host: QueueLivenessHost, event) -> None:
    """Tell the system a client has gone: set its event and schedule the eviction of its waiting requests.

    Synchronous and safe to call from a task that is being cancelled (it never awaits). It is
    idempotent. When no waiting slot carries the event yet (the client left before the manager
    attached it), the event is set and nothing else happens; the take paths drop the slot later.
    """
    event.set()
    coro = _evict_for_event(host, event)
    try:
        host._spawn_bg(coro)
    except Exception:  # noqa: BLE001 - scheduling must never raise out of the caller
        coro.close()
        _log.warning("could not schedule the eviction after a client hang-up", exc_info=True)
