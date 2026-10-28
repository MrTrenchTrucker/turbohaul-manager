"""Is the client of a waiting request still there? The shared answer, and the one-shot failure.

Everything here is await-free and touches no queue, inbox or claim, so it can be called from
any synchronous stretch of a take path.
"""

from __future__ import annotations

from .liveness_host import QueueLivenessHost, SlotLike


def client_gone(slot: SlotLike) -> bool:
    """True when the request's client has gone: the slot is already flagged evicted, or its
    disconnect event is set. A pure read; it changes nothing."""
    event = slot.disconnect_event
    return bool(slot.is_evicted) or (event is not None and event.is_set())


def stamp_if_disconnected(slot: SlotLike) -> bool:
    """Flag the slot evicted when its disconnect event is set, and say whether its client is gone.

    The flag is only ever raised here, never cleared, and a slot whose event is not set is left
    exactly as it was. The queue's take paths call this check.
    """
    event = slot.disconnect_event
    if event is not None and event.is_set():
        slot.is_evicted = True
    return client_gone(slot)


def _fail_once(host: QueueLivenessHost, slot: SlotLike) -> bool:
    """Fail the slot through the host unless its completion future is already done.

    Returns True when this call failed it. A request with no completion future cannot be told
    apart from a fresh one, so it is failed on every call.
    """
    future = slot.completion_future
    if future is not None and future.done():
        return False
    host._handle_unloaded_slot(slot)
    return True


def drop_if_gone(host: QueueLivenessHost, slot: SlotLike) -> bool:
    """Flag the slot if its client has gone and, if so, fail it once. True iff the client is gone.

    Idempotent: a second call, or a call racing an eviction, finds the completion future done
    and does not fail or audit the slot again. It does not touch queues, inboxes or claims;
    the caller removes the slot from wherever it waits and releases its claim.
    """
    if not stamp_if_disconnected(slot):
        return False
    _fail_once(host, slot)
    return True
