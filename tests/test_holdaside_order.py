"""The grace hold-aside stops destroying the age order the
starvation breakout depends on.

THE DEFECT. ``pop_next`` lifts grace-protected entries out of ``_staging`` for
the duration of one call and, in its ``finally``, put them back with
``self._staging.extend(held_aside)`` -- at the TAIL, not their original index.

The damage is NOT to rank. ``created_at`` is untouched, so a registered slot's
Fast Lane rank is unharmed; that is the thing this defect is easiest to
mis-diagnose as. The damage is POSITIONAL, and it lands on a completely
different mechanism: ``_starved_other_model_locked`` `break`s at the FIRST
other-model entry it meets, on its own stated precondition that "staging
preserves FIFO order for every entry that is never re-inserted at the head".
The tail splice falsifies exactly that precondition. Push an older other-model
entry behind a newer one and the NEWER one settles the scan; the newer one is
not aged, so the function returns None -- and the ordinary starvation breakout
reads SATISFIED while it is violated, blind to the very request it exists to
rescue.

WHY THE FIX IS ON THE SPLICE AND NOT ON THE SCAN. Restoring position makes
``_starved_other_model_locked``'s precondition TRUE again rather than working
around it, so that function -- with three consumers -- is untouched by
this change.

NON-VACUITY. The RED tests here fail against the unfixed queue.py
and pass against the fixed one;
each failure is for the stated reason.
"""
import time

import pytest

from turbohaul.queue import TurbohaulQueue
from turbohaul.slot import Slot

WARM = "m1"
OTHER = "m2"
STARVE_WINDOW_S = 5.0


def _slot(thread_id, model_tag, *, aged_by=0.0, prompt=None):
    s = Slot.new(
        model_tag=model_tag, prompt=prompt or thread_id, thread_id=thread_id,
    )
    if aged_by:
        s.created_at = time.monotonic() - aged_by
    return s


async def _staged_threads(q):
    return [s.thread_id for s in await q.peek_staging()]


@pytest.mark.asyncio
class TestStarvationBreakoutSurvivesAProtectedCall:
    """The defect, at the seam where it actually costs something."""

    async def test_an_aged_other_model_entry_is_still_visible_after_being_held_aside(self):
        q = TurbohaulQueue(
            staging_max=10, max_other_model_wait_s=STARVE_WINDOW_S,
        )
        aged_other = _slot("g", OTHER, aged_by=600.0, prompt="aged-other")
        newer_other = _slot("n", OTHER, prompt="newer-other")
        same_model = _slot("w", WARM, prompt="same-model")
        for s in (aged_other, newer_other, same_model):
            await q.enqueue(s)

        # Non-vacuity: BEFORE any protected call the breakout can see it.
        assert (await q.starved_other_model(WARM)) is aged_other

        # One protected call. `aged_other` is inside its grace window, so it is
        # held aside for the duration and must come back where it was.
        await q.pop_next(
            warm_model_tag=WARM, grace_active=frozenset({("g", OTHER)}),
        )

        starved = await q.starved_other_model(WARM)

        assert starved is not None, (
            "the starvation breakout went BLIND: the hold-aside spliced the "
            "aged other-model entry behind a NEWER one, and "
            "_starved_other_model_locked breaks at the first other-model entry "
            "it meets -- so it settled on the newer entry, found it not aged, "
            "and reported no starvation while the aged request sat at the tail"
        )
        assert starved.slot_id == aged_other.slot_id

    async def test_the_breakout_stays_visible_across_repeated_protected_calls(self):
        """The original design accepted that a protected entry 'can drift
        toward the tail across repeated protected calls'. Drift is exactly what
        must not happen, so one call is not a sufficient test.

        DISCRIMINATION NOTE, because the obvious way to write this does not
        discriminate at all: if EVERY staged entry is grace-held, the ladder
        sees an empty deque, nothing is reordered relative to anything, and a
        tail splice restores the original order by accident -- the test passes
        on both arms and proves nothing. A naive version of this test does exactly that.
        Only ONE entry is held here, and a same-model filler is re-enqueued
        each round so there is always a VISIBLE entry for the held one to be
        spliced behind.
        """
        q = TurbohaulQueue(
            staging_max=10, max_other_model_wait_s=STARVE_WINDOW_S,
        )
        aged_other = _slot("g", OTHER, aged_by=600.0, prompt="aged-other")
        newer_other = _slot("n", OTHER, prompt="newer-other")
        for s in (aged_other, newer_other):
            await q.enqueue(s)

        for i in range(5):
            await q.enqueue(_slot(f"filler{i}", WARM))
            popped = await q.pop_next(
                warm_model_tag=WARM,
                grace_active=frozenset({("g", OTHER)}),
            )
            assert popped is not None and popped.thread_id == f"filler{i}", (
                "non-vacuity: each round must actually run the ladder over a "
                "non-empty visible deque and pop from it"
            )

        assert await _staged_threads(q) == ["g", "n"], (
            "five protected calls must leave staging in the order they found "
            "it -- any drift at all reopens the blindness above"
        )
        starved = await q.starved_other_model(WARM)
        assert starved is not None and starved.slot_id == aged_other.slot_id


@pytest.mark.asyncio
class TestHeldAsideEntriesReturnToTheirOwnPosition:
    """The mechanism itself, asserted directly rather than only through its
    consequence."""

    async def test_a_held_entry_returns_to_its_original_index(self):
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
        for s in (
            _slot("a", WARM), _slot("held", WARM), _slot("c", WARM),
        ):
            await q.enqueue(s)

        # A protected call that pops nothing: every visible entry is same-model
        # and the ladder takes the FIFO head, so exactly one leaves.
        got = await q.pop_next(grace_active=frozenset({("held", WARM)}))
        assert got is not None and got.thread_id == "a"

        assert await _staged_threads(q) == ["held", "c"], (
            "the held entry belongs between the popped head and `c`, exactly "
            "where it was -- a tail splice would give ['c', 'held']"
        )

    async def test_several_held_entries_keep_their_relative_order(self):
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
        for s in (
            _slot("h1", WARM), _slot("mid", WARM), _slot("h2", WARM),
            _slot("tail", WARM),
        ):
            await q.enqueue(s)

        got = await q.pop_next(
            grace_active=frozenset({("h1", WARM), ("h2", WARM)}),
        )
        # h1 is invisible, so the ladder's FIFO head is `mid`.
        assert got is not None and got.thread_id == "mid"

        assert await _staged_threads(q) == ["h1", "h2", "tail"], (
            "two held entries must come back in their own original order and "
            "on the correct side of the entry that stayed visible"
        )

    async def test_an_entry_drained_from_accept_buf_stays_at_the_tail(self):
        """The one ordering a restore must NOT preserve: an entry admitted to
        staging DURING the call is genuinely the newest thing present, and the
        tail is where it belongs."""
        q = TurbohaulQueue(staging_max=3, main_lane_reserved=False)
        for s in (_slot("held", WARM), _slot("f1", WARM), _slot("f2", WARM)):
            await q.enqueue(s)          # fills staging_max=3 exactly
        await q.enqueue(_slot("fresh", WARM))  # overflows into accept_buf

        got = await q.pop_next(grace_active=frozenset({("held", WARM)}))
        assert got is not None and got.thread_id == "f1"

        assert await _staged_threads(q) == ["held", "f2", "fresh"], (
            "the held entry returns to the front where it was; the entry "
            "drained from accept_buf during the call is the newest and stays "
            "last"
        )
        assert len(await q.peek_staging()) <= 3, "the real cap still holds"

    async def test_a_held_entry_that_is_the_only_thing_staged_is_unharmed(self):
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
        await q.enqueue(_slot("held", WARM))

        got = await q.pop_next(grace_active=frozenset({("held", WARM)}))

        assert got is None, "the only staged entry was protected this call"
        assert await _staged_threads(q) == ["held"]
        assert q.staging_max == 10, "the temporary shrink must never leak"
