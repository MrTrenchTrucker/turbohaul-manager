"""`queue.enqueue_tail(slot)` -- a proper FIFO-tail sibling
to `enqueue_head`, built for `_requeue_slots_tail_or_fail` (manager.py) to
requeue a DESIGNATED VICTIM's own displaced (already-STAGED) inbox backlog
without cutting to the front of the line.

Uses REAL `Slot` + `TurbohaulQueue` objects throughout, not a duck-typed fake
-- the ORIGINAL bug in `_requeue_slots_tail_or_fail` (InvalidTransition:
STAGED -> STAGED on every real, already-staged slot) was invisible to that
file's own unit tests precisely because they used `SimpleNamespace` fakes
with no `.state` attribute at all, so the real FSM guard was never exercised.
Rule of thumb: write the test RED-first, with a fake that HAS a real
.state.
"""
import asyncio

import pytest

from turbohaul.fsm import InvalidTransition, transition as fsm_transition
from turbohaul.queue import QueueFull, TurbohaulQueue
from turbohaul.slot import Slot, SlotState


def _staged_slot(model_tag="m", thread_id="t"):
    """A slot in the exact state every real inbox-drained slot is actually
    in -- RECEIVED (Slot.new's default) fast-forwarded to STAGED, mirroring
    the real dispatcher chain (pop_next pops an already-STAGED slot,
    _route_or_reserve hands it to r.inbox with no transition in between)."""
    slot = Slot.new(model_tag, prompt="p", thread_id=thread_id)
    fsm_transition(slot, SlotState.STAGED)
    return slot


@pytest.mark.asyncio
class TestEnqueueTailAlreadyStagedSlot:
    async def test_already_staged_slot_lands_at_tail_without_raising(self):
        """The exact real-world input that broke plain enqueue(): a slot
        that is ALREADY STAGED. Must succeed, must NOT re-transition."""
        q = TurbohaulQueue(staging_max=10)
        existing = Slot.new("m", prompt="p", thread_id="t-existing")
        await q.enqueue(existing)

        victim_followup = _staged_slot(thread_id="t-followup")
        await q.enqueue_tail(victim_followup)

        assert victim_followup.state is SlotState.STAGED
        staging = list(q._staging)
        assert staging == [existing, victim_followup], (
            "tail-append must preserve arrival order, landing BEHIND "
            "what was already staged"
        )

    async def test_multiple_already_staged_slots_preserve_relative_order(self):
        """No reversal trick needed here (unlike enqueue_head's own
        documented appendleft-per-call inversion) -- enqueue_tail APPENDS,
        so iterating a batch in given order already lands it in that order."""
        q = TurbohaulQueue(staging_max=10)
        first = _staged_slot(thread_id="t-1")
        second = _staged_slot(thread_id="t-2")
        await q.enqueue_tail(first)
        await q.enqueue_tail(second)
        assert list(q._staging) == [first, second]


@pytest.mark.asyncio
class TestEnqueueTailFreshSlot:
    async def test_fresh_received_slot_is_transitioned_to_staged(self):
        """The guard is `if slot.state is not STAGED: transition` -- a
        not-yet-staged slot must still genuinely transition, not silently
        skip admission."""
        q = TurbohaulQueue(staging_max=10)
        fresh = Slot.new("m", prompt="p", thread_id="t-fresh")
        assert fresh.state is SlotState.RECEIVED
        await q.enqueue_tail(fresh)
        assert fresh.state is SlotState.STAGED
        assert fresh in q._staging


@pytest.mark.asyncio
class TestEnqueueTailCapacity:
    async def test_staging_full_raises_queuefull_no_accept_buffer_detour(self):
        """Deliberately DROPS the accept-buffer fallback plain enqueue() has
        -- staging full must raise QueueFull directly, never fall through to
        _accept_buf (which would reproduce the same illegal-transition bug
        one hop downstream at a totally different call site)."""
        q = TurbohaulQueue(staging_max=1)
        await q.enqueue(Slot.new("m", prompt="p", thread_id="t-fills-it"))

        overflow = _staged_slot(thread_id="t-overflow")
        with pytest.raises(QueueFull):
            await q.enqueue_tail(overflow)

        assert overflow not in q._staging
        assert overflow not in q._accept_buf, (
            "an already-STAGED slot must never be parked in _accept_buf -- "
            "the 4+ accept_buf-drain sites elsewhere assume everything "
            "there is genuinely ACCEPT_BUFFER"
        )
        assert len(q._accept_buf) == 0


@pytest.mark.asyncio
class TestEnqueueTailWakeHookAndLogging:
    async def test_invoke_on_enqueue_fires(self):
        """The dispatcher wake hook -- the exact thing hand-appending
        to _staging would have silently skipped, so
        enqueue_tail must fire it."""
        fired = []
        q = TurbohaulQueue(staging_max=10, on_enqueue=lambda: fired.append(True))
        await q.enqueue_tail(_staged_slot())
        assert fired == [True]

    async def test_via_enqueue_tail_logged(self, caplog):
        import logging
        q = TurbohaulQueue(staging_max=10)
        with caplog.at_level(logging.INFO, logger="turbohaul.queue"):
            await q.enqueue_tail(_staged_slot())
        assert any("via=enqueue_tail" in r.message for r in caplog.records)


@pytest.mark.asyncio
class TestEnqueueTailClosedQueue:
    async def test_closed_queue_raises_queueclosed(self):
        from turbohaul.queue import QueueClosed
        q = TurbohaulQueue(staging_max=10)
        q._closed = True
        with pytest.raises(QueueClosed):
            await q.enqueue_tail(_staged_slot())
