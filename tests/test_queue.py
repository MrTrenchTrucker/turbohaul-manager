"""Tests for TurbohaulQueue + GraceTimer + IdleHotTimer (v0.2 §5/§6)."""
import asyncio
import time

import pytest

from turbohaul.queue import (
    GraceTimer,
    IdleHotTimer,
    QueueClosed,
    QueueFull,
    TurbohaulQueue,
)
from turbohaul.slot import Slot, SlotState


@pytest.mark.asyncio
class TestTurbohaulQueue:
    async def test_enqueue_pop_basic(self):
        q = TurbohaulQueue(staging_max=10, acceptance_max=100)
        s = Slot.new("m")
        await q.enqueue(s)
        d = q.depth()
        assert d["staging_queue_depth"] + d["acceptance_buffer_depth"] == 1
        popped = await q.pop_next()
        assert popped is not None
        assert popped.slot_id == s.slot_id

    async def test_fifo_ordering(self):
        q = TurbohaulQueue(staging_max=10)
        slots = [Slot.new("m") for _ in range(5)]
        for s in slots:
            await q.enqueue(s)
        for expected in slots:
            popped = await q.pop_next()
            assert popped.slot_id == expected.slot_id

    async def test_acceptance_buffer_holds_when_staging_full(self):
        q = TurbohaulQueue(staging_max=2, acceptance_max=100)
        for _ in range(5):
            await q.enqueue(Slot.new("m"))
        d = q.depth()
        assert d["staging_queue_depth"] == 2
        assert d["acceptance_buffer_depth"] == 3

    async def test_acceptance_buffer_full_raises(self):
        q = TurbohaulQueue(staging_max=1, acceptance_max=2)
        await q.enqueue(Slot.new("m"))
        await q.enqueue(Slot.new("m"))
        await q.enqueue(Slot.new("m"))
        with pytest.raises(QueueFull):
            await q.enqueue(Slot.new("m"))

    async def test_pop_drains_buffer_to_staging(self):
        q = TurbohaulQueue(staging_max=1, acceptance_max=10)
        slots = [Slot.new("m") for _ in range(3)]
        for s in slots:
            await q.enqueue(s)
        # staging=1, buffer=2
        p1 = await q.pop_next()
        d = q.depth()
        # After pop, replenished from buffer
        assert d["staging_queue_depth"] == 1
        p2 = await q.pop_next()
        p3 = await q.pop_next()
        ids = {p1.slot_id, p2.slot_id, p3.slot_id}
        assert ids == {s.slot_id for s in slots}

    async def test_enqueue_head_for_matched_thread(self):
        q = TurbohaulQueue(staging_max=10)
        s1 = Slot.new("m")
        s2 = Slot.new("m")
        s_head = Slot.new("m", thread_id="thr-1")
        await q.enqueue(s1)
        await q.enqueue(s2)
        await q.enqueue_head(s_head)
        popped = await q.pop_next()
        assert popped.slot_id == s_head.slot_id

    async def test_find_matched_thread(self):
        q = TurbohaulQueue(staging_max=10)
        s1 = Slot.new("model-a", thread_id="thr-x")
        await q.enqueue(s1)
        found = await q.find_matched_thread("thr-x", "model-a")
        assert found is not None
        assert found.slot_id == s1.slot_id

    async def test_find_matched_thread_no_match(self):
        q = TurbohaulQueue(staging_max=10)
        s1 = Slot.new("model-a", thread_id="thr-x")
        await q.enqueue(s1)
        assert await q.find_matched_thread("thr-x", "model-b") is None
        assert await q.find_matched_thread("thr-y", "model-a") is None
        assert await q.find_matched_thread("", "model-a") is None  # empty thread_id

    async def test_pop_empty_returns_none(self):
        q = TurbohaulQueue()
        assert await q.pop_next() is None

    async def test_remove(self):
        q = TurbohaulQueue(staging_max=10)
        s1 = Slot.new("m")
        s2 = Slot.new("m")
        await q.enqueue(s1)
        await q.enqueue(s2)
        removed = await q.remove(s1.slot_id)
        assert removed is not None
        assert removed.slot_id == s1.slot_id
        d = q.depth()
        assert d["staging_queue_depth"] + d["acceptance_buffer_depth"] == 1

    async def test_remove_nonexistent(self):
        q = TurbohaulQueue()
        assert await q.remove("not-here") is None

    async def test_close_clears_and_blocks(self):
        q = TurbohaulQueue()
        await q.enqueue(Slot.new("m"))
        await q.close()
        with pytest.raises(QueueClosed):
            await q.enqueue(Slot.new("m"))

    async def test_state_transitions_on_enqueue(self):
        q = TurbohaulQueue(staging_max=1)
        s1 = Slot.new("m")
        await q.enqueue(s1)
        assert s1.state == SlotState.STAGED
        s2 = Slot.new("m")
        await q.enqueue(s2)
        # Staging full, s2 lands in accept buffer
        assert s2.state == SlotState.ACCEPT_BUFFER


@pytest.mark.asyncio
class TestModelAffinityPop:
    """Single-mutator-safe parallelism support: model-affinity pop_next.

    All behavior is OFF unless ``warm_model_tag`` is passed AND the queue was
    constructed with batching enabled (cap > 1) / a non-zero starvation window.
    The defaults (cap=1, wait=0.0) collapse to strict FIFO even with a tag.
    """

    async def test_affinity_prefers_warm_model(self):
        # cap=3 allows a run of 3; large wait => no starvation interference.
        q = TurbohaulQueue(
            staging_max=10,
            max_consecutive_same_model=3,
            max_other_model_wait_s=10_000.0,
        )
        a1 = Slot.new("A")
        b1 = Slot.new("B")
        a2 = Slot.new("A")
        b2 = Slot.new("B")
        a3 = Slot.new("A")
        for s in (a1, b1, a2, b2, a3):
            await q.enqueue(s)
        tags = []
        for _ in range(5):
            popped = await q.pop_next(warm_model_tag="A")
            tags.append(popped.model_tag)
        # The three A's cluster ahead of the B's (warm-slot reuse), bounded by
        # the consecutive cap which then forces the FIFO head (a B).
        assert tags[:3] == ["A", "A", "A"]
        assert tags[3:] == ["B", "B"]
        # Queue fully drained — no loss / duplication.
        assert await q.pop_next(warm_model_tag="A") is None

    async def test_affinity_respects_consecutive_cap(self):
        # cap=3, 5xA + 1xB, warm='A' => exactly 3 A then forced B then remaining A.
        q = TurbohaulQueue(
            staging_max=10,
            max_consecutive_same_model=3,
            max_other_model_wait_s=10_000.0,  # cap (not starvation) forces the swap
        )
        # B enqueued 2nd so it reaches the FIFO head once the early A's are
        # affinity-clustered ahead of it; the cap then forces that head (the B).
        a1 = Slot.new("A")
        b1 = Slot.new("B")
        a2 = Slot.new("A")
        a3 = Slot.new("A")
        a4 = Slot.new("A")
        a5 = Slot.new("A")
        for s in (a1, b1, a2, a3, a4, a5):
            await q.enqueue(s)
        tags = []
        for _ in range(6):
            popped = await q.pop_next(warm_model_tag="A")
            tags.append(popped.model_tag)
        # Exactly 3 consecutive A's, then the cap forces the FIFO head (the B),
        # then the remaining A's drain.
        assert tags == ["A", "A", "A", "B", "A", "A"], tags

    async def test_starvation_by_age_forces_swap(self):
        # One AGED B + many fresh A; small wait => aged B popped despite affinity.
        q = TurbohaulQueue(
            staging_max=10,
            max_consecutive_same_model=1000,  # cap won't fire
            max_other_model_wait_s=0.05,
        )
        aged_b = Slot.new("B")
        # Backdate the B's monotonic creation so it is "starved" immediately.
        aged_b.created_at = time.monotonic() - 10.0
        fresh_a = [Slot.new("A") for _ in range(4)]
        # B is the FIFO head; affinity would skip it, but starvation forces it.
        for s in [aged_b, *fresh_a]:
            await q.enqueue(s)
        first = await q.pop_next(warm_model_tag="A")
        assert first.model_tag == "B", (
            "aged FIFO-head other-model request must be force-popped on starvation"
        )
        assert first.slot_id == aged_b.slot_id
        # Remaining are the fresh A's.
        rest = []
        for _ in range(4):
            rest.append((await q.pop_next(warm_model_tag="A")).model_tag)
        assert rest == ["A", "A", "A", "A"]

    async def test_no_preference_is_strict_fifo(self):
        # pop_next() no-arg AND pop_next(warm_model_tag=None) == current FIFO.
        for warm_kwargs in ({}, {"warm_model_tag": None}):
            q = TurbohaulQueue(
                staging_max=10,
                max_consecutive_same_model=3,
                max_other_model_wait_s=10_000.0,
            )
            order = [Slot.new("A"), Slot.new("B"), Slot.new("A"), Slot.new("B")]
            for s in order:
                await q.enqueue(s)
            got = []
            for _ in range(4):
                got.append((await q.pop_next(**warm_kwargs)).slot_id)
            assert got == [s.slot_id for s in order], (
                f"strict FIFO violated with warm_kwargs={warm_kwargs}: {got}"
            )

    async def test_cap_one_disables_batching(self):
        # cap=1 => strict FIFO even with warm tag (batch_cap_hit fires every pop).
        q = TurbohaulQueue(
            staging_max=10,
            max_consecutive_same_model=1,
            max_other_model_wait_s=10_000.0,
        )
        order = [Slot.new("A"), Slot.new("B"), Slot.new("A"), Slot.new("B"), Slot.new("A")]
        for s in order:
            await q.enqueue(s)
        got = []
        for _ in range(5):
            got.append((await q.pop_next(warm_model_tag="A")).slot_id)
        assert got == [s.slot_id for s in order], (
            f"cap=1 must collapse to strict FIFO: {got}"
        )

    async def test_matching_pop_honors_eviction(self):
        # Same-model match with disconnect_event SET is flagged is_evicted.
        q = TurbohaulQueue(
            staging_max=10,
            max_consecutive_same_model=3,
            max_other_model_wait_s=10_000.0,
        )
        b_head = Slot.new("B")  # FIFO head, other model — affinity skips it
        a_evicted = Slot.new("A")
        ev = asyncio.Event()
        ev.set()
        a_evicted.disconnect_event = ev
        await q.enqueue(b_head)
        await q.enqueue(a_evicted)
        popped = await q.pop_next(warm_model_tag="A")
        # Affinity picks the matching A; its disconnect_event is set => evicted.
        assert popped.slot_id == a_evicted.slot_id
        assert popped.is_evicted is True

    async def test_affinity_scan_bounded(self):
        # staging full of non-matching => matching helper scans <= max_scan and
        # returns None, so pop_next falls back to FIFO head (a non-match).
        q = TurbohaulQueue(
            staging_max=5,
            max_consecutive_same_model=3,
            max_other_model_wait_s=10_000.0,
        )
        # Fill staging with 5 B's (staging_max=5 = the scan bound).
        b_slots = [Slot.new("B") for _ in range(5)]
        for s in b_slots:
            await q.enqueue(s)
        # No "A" anywhere. Affinity helper scans <= max_scan, finds nothing,
        # returns None; pop_next falls back to FIFO head.
        popped = await q.pop_next(warm_model_tag="A")
        assert popped is not None
        assert popped.model_tag == "B"
        assert popped.slot_id == b_slots[0].slot_id  # FIFO head fallback


class TestGraceTimer:
    def test_start_then_expire(self):
        g = GraceTimer(grace_seconds=0.05, max_extensions=5)
        g.start("thr-1", "m")
        assert not g.expired()
        time.sleep(0.1)
        assert g.expired()

    def test_matches(self):
        g = GraceTimer(grace_seconds=10, max_extensions=5)
        g.start("thr-1", "model-a")
        assert g.matches("thr-1", "model-a")
        assert not g.matches("thr-2", "model-a")
        assert not g.matches("thr-1", "model-b")

    def test_restart_for_followup_extends_count(self):
        g = GraceTimer(grace_seconds=10, max_extensions=3)
        g.start("thr-1", "m")
        assert g.restart_for_followup() is True
        assert g.extension_count == 1
        assert g.restart_for_followup() is True
        assert g.restart_for_followup() is True
        assert g.extension_count == 3
        assert g.restart_for_followup() is False  # over cap

    def test_reset(self):
        g = GraceTimer(grace_seconds=10)
        g.start("thr-1", "m")
        g.reset()
        assert g.thread_id is None
        assert g.model_tag is None
        assert g.extension_count == 0
        assert g.expired()

    def test_remaining_s_decreases(self):
        g = GraceTimer(grace_seconds=1.0)
        g.start("thr-1", "m")
        r1 = g.remaining_s()
        time.sleep(0.05)
        r2 = g.remaining_s()
        assert r2 < r1


class TestIdleHotTimer:
    def test_start_then_expire(self):
        h = IdleHotTimer(idle_seconds=0.05)
        h.start("model-a")
        assert h.matches_same_model("model-a")
        time.sleep(0.1)
        assert not h.matches_same_model("model-a")

    def test_matches_same_model(self):
        h = IdleHotTimer(idle_seconds=10)
        h.start("model-a")
        assert h.matches_same_model("model-a")
        assert not h.matches_same_model("model-b")

    def test_reset(self):
        h = IdleHotTimer(idle_seconds=10)
        h.start("model-a")
        h.reset()
        assert h.model_tag is None
        assert h.expired()
