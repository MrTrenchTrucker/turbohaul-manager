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
