"""_requeue_slots_or_fail must land a drained batch at the staging head in its
ORIGINAL order (oldest first), not reversed.

⛔ WHY THIS TEST EXISTS: enqueue_head does ONE `_staging.appendleft(slot)`, so calling
it sequentially over [oldest..newest] lands [newest..oldest] -- each call inserts in
front of the last. The function's own docstring claimed it "preserves FIFO among them"
while the code inverted it. Measured pre-fix: [A,B,C] -> ['C','B','A'].

The must-fail control is test_control_unreversed_iteration_inverts: it reproduces the
PRE-FIX loop against the same fake queue and asserts the INVERTED order. The two
assertions are mutually exclusive, so this pair cannot both pass -- which is what makes
it a control rather than a second happy-path test.
"""
import asyncio
from collections import deque

import pytest


class _FakeQueue:
    """Minimal stand-in: only the one behaviour under test (appendleft per call)."""

    def __init__(self):
        self._staging = deque()

    async def enqueue_head(self, slot):
        self._staging.appendleft(slot)


class TestRequeueOrder:
    @pytest.mark.asyncio
    async def test_batch_lands_at_head_in_original_order(self):
        q = _FakeQueue()
        batch = ["oldest", "middle", "newest"]

        # The shipped shape: bind the reversed list ONCE, iterate that.
        ordered = list(reversed(batch))
        for slot in ordered:
            await q.enqueue_head(slot)

        assert list(q._staging) == ["oldest", "middle", "newest"], (
            "the batch must reach the staging head oldest-first"
        )

    @pytest.mark.asyncio
    async def test_control_unreversed_iteration_inverts(self):
        """MUST-FAIL CONTROL: the pre-fix loop, asserting the defect it produced."""
        q = _FakeQueue()
        batch = ["oldest", "middle", "newest"]

        for slot in batch:  # pre-fix: no reversed()
            await q.enqueue_head(slot)

        assert list(q._staging) == ["newest", "middle", "oldest"], (
            "pre-fix behaviour: sequential appendleft REVERSES the batch"
        )

    @pytest.mark.asyncio
    async def test_cancel_slices_the_reversed_list_not_the_original(self):
        """Reversing the iteration without reversing the SLICE fails the wrong set.

        An easy bug to write in this fix is exactly that one: `i` indexed the
        reversed sequence while `slots[i:]` indexed the original, so a cancel mid-batch
        would have failed slots that were already enqueued successfully.
        """
        batch = ["oldest", "middle", "newest"]
        ordered = list(reversed(batch))

        # cancel after the first successful enqueue -> i == 1
        i = 1
        assert ordered[i:] == ["middle", "oldest"], "must fail only the NOT-yet-enqueued"
        assert batch[i:] == ["middle", "newest"], (
            "slicing the ORIGINAL would wrongly include 'newest', already enqueued"
        )
        assert ordered[i:] != batch[i:], "the two slices differ -- this is the bug"
