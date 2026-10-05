"""`_requeue_slots_tail_or_fail`: re-queue a displaced batch at the queue tail.

Sibling of `_requeue_slots_or_fail` (see the requeue-order-inversion test),
which is what non-victim inbox-drain uses (the head path
serves non-victim evictions). This is the TAIL half, for a designated victim's own
displaced work.

Unlike the head sibling, `enqueue_tail` APPENDS -- so no iteration-reversal trick is
needed or correct here. The control test below proves that directly: reversing the
iteration (copying the head-sibling's trick onto this function) INVERTS the batch,
which is exactly the bug this shape must not have.

Unit-tests the METHOD in isolation (not a full TurbohaulManager) via a minimal stand-in
object carrying only `.queue.enqueue_tail` and `._fail_completion_future` -- the two
things `_requeue_slots_tail_or_fail` actually touches. Calling an unbound method with a
duck-typed `self` is deliberate here, matching how narrowly this helper's own contract
is scoped (queue.py:queue.enqueue_tail + _fail_completion_future, nothing else). Fake
slots are `SimpleNamespace(slot_id=...)`, not bare strings -- the real code logs
`slot.slot_id` on the failure path, so a bare string would AttributeError there.

WARNING: these fakes carry NO real `.state` and never exercise the FSM guard --
`queue.enqueue_tail`'s own STAGED-guard/capacity contract against REAL `Slot`/
`TurbohaulQueue` objects is covered separately by
the enqueue-tail queue test. This file exists ONLY for
`_requeue_slots_tail_or_fail`'s own generic ordering/failure-handling shape
(order-preservation, QueueFull/partial-failure/CancelledError semantics) --
exactly the class of gap that could let a real InvalidTransition bug ship
undetected. `_requeue_slots_tail_or_fail` calls `queue.enqueue_tail`, not plain
`queue.enqueue`, and these fakes are named to match that call site.
"""
import asyncio
from types import SimpleNamespace

import pytest

from turbohaul.manager import TurbohaulManager


def _slot(name):
    return SimpleNamespace(slot_id=name)


class _FakeQueueFull(Exception):
    """Stand-in for turbohaul.queue.QueueFull -- avoids needing a full queue import."""


class _FakeQueueTail:
    """Minimal stand-in: only `enqueue_tail` (tail-append), the one behaviour under test."""

    def __init__(self, fail_on=(), fail_with=None):
        self.landed = []
        self._fail_on = set(fail_on)
        self._fail_with = fail_with or RuntimeError("enqueue_tail failed")

    async def enqueue_tail(self, slot):
        if slot.slot_id in self._fail_on:
            raise self._fail_with
        self.landed.append(slot)


class _FakeSelf:
    def __init__(self, queue):
        self.queue = queue
        self.failed_futures = []

    def _unpark_fastlane_claim(self, slot):
        """No-op: this fake has no claim registry, so there is nothing to un-park."""

    def _fail_completion_future(self, slot, exc):
        self.failed_futures.append((slot, exc))


@pytest.mark.asyncio
class TestRequeueTailOrder:
    async def test_batch_lands_at_tail_in_original_order(self):
        """MUST FAIL if a reversal trick (the head-sibling's fix) is copied onto this
        tail function -- enqueue APPENDS, so forward iteration already preserves order."""
        q = _FakeQueueTail()
        fake = _FakeSelf(q)
        batch = [_slot("oldest"), _slot("middle"), _slot("newest")]

        await TurbohaulManager._requeue_slots_tail_or_fail(fake, batch)

        assert [s.slot_id for s in q.landed] == ["oldest", "middle", "newest"], (
            "the batch must reach the tail oldest-first, in its ORIGINAL order"
        )
        assert fake.failed_futures == []

    async def test_control_reversed_iteration_would_invert(self):
        """MUST-FAIL CONTROL: reproduces the wrong-fix shape directly against the
        real fake queue, asserting the defect it would produce. Proves reversal is
        actively WRONG here, not merely unnecessary."""
        q = _FakeQueueTail()
        batch = [_slot("oldest"), _slot("middle"), _slot("newest")]

        for slot in reversed(batch):  # the WRONG shape for a tail-append
            await q.enqueue_tail(slot)

        assert [s.slot_id for s in q.landed] == ["newest", "middle", "oldest"], (
            "reversing the iteration over an APPEND-based enqueue inverts the batch"
        )
        assert [s.slot_id for s in q.landed] != ["oldest", "middle", "newest"]


@pytest.mark.asyncio
class TestRequeueTailFailureHandling:
    async def test_queue_full_fails_the_future_not_swallowed(self):
        """The named failure mode is QueueFull -- for the real
        `queue.enqueue_tail`, raised when staging alone is at capacity (it
        deliberately has no accept-buffer fallback, see its own docstring);
        this fake only needs to prove the generic failure-handling shape."""
        q = _FakeQueueTail(fail_on={"stuck"}, fail_with=_FakeQueueFull("both buffers full"))
        fake = _FakeSelf(q)

        await TurbohaulManager._requeue_slots_tail_or_fail(fake, [_slot("stuck")])

        assert q.landed == []
        assert len(fake.failed_futures) == 1
        slot, exc = fake.failed_futures[0]
        assert slot.slot_id == "stuck"
        assert isinstance(exc, _FakeQueueFull)

    async def test_partial_failure_still_processes_remaining_slots(self):
        """A non-cancellation failure on one slot does not abort the rest of the
        batch -- matches _requeue_slots_or_fail's own generic-Exception semantics."""
        q = _FakeQueueTail(fail_on={"bad"})
        fake = _FakeSelf(q)
        batch = [_slot("good1"), _slot("bad"), _slot("good2")]

        await TurbohaulManager._requeue_slots_tail_or_fail(fake, batch)

        assert [s.slot_id for s in q.landed] == ["good1", "good2"]
        assert len(fake.failed_futures) == 1
        assert fake.failed_futures[0][0].slot_id == "bad"

    async def test_cancelled_error_fails_the_raising_slot_and_everything_after(self):
        """A CancelledError raised WHILE awaiting slot i's own enqueue() means slot i
        was NEVER successfully enqueued (the await did not complete) -- so slots[i:]
        (the raising slot itself, plus everything not yet attempted) must all be
        failed, and everything BEFORE i (already landed) must not be. Mirrors the
        head-requeue sibling's own semantics exactly (its ``ordered[i:]`` also
        includes the slot whose await raised, not just the ones after it)."""
        class _CancellingQueue:
            def __init__(self):
                self.landed = []

            async def enqueue_tail(self, slot):
                if slot.slot_id == "cancel-here":
                    raise asyncio.CancelledError()
                self.landed.append(slot)

        fake = _FakeSelf(_CancellingQueue())
        batch = [_slot("good1"), _slot("cancel-here"), _slot("not-yet-1"), _slot("not-yet-2")]

        with pytest.raises(asyncio.CancelledError):
            await TurbohaulManager._requeue_slots_tail_or_fail(fake, batch)

        assert [s.slot_id for s in fake.queue.landed] == ["good1"]
        failed_slots = [s.slot_id for s, _ in fake.failed_futures]
        assert failed_slots == ["cancel-here", "not-yet-1", "not-yet-2"], (
            "must fail the slot whose await raised PLUS everything not yet "
            "attempted, not just the ones strictly after it, and not 'good1' "
            "(already successfully landed before the cancellation)"
        )
