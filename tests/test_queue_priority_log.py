"""QUEUE_PRIORITY log line — WHO WAS SERVED and WHO WAS SKIPPED TO DO IT
Mirrors tests/test_queue.py's conventions
(Slot.new + TurbohaulQueue directly, no manager construction needed) so
this file drops into the same test session with no new fixtures.

Observability only: every test here also asserts the pre-existing pop_next
CONTRACT (which slot.slot_id comes back, in what order) is unchanged --
a log line is worthless as an acceptance arm if it could pass while the
underlying serve decision silently changed.
"""
import asyncio
import logging

import pytest

from turbohaul.queue import TurbohaulQueue
from turbohaul.slot import Slot


def _queue_priority_lines(caplog) -> list:
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("QUEUE_PRIORITY")]


@pytest.mark.asyncio
class TestQueuePriorityCompression:
    """The core acceptance arm: a test that FAILS without the new
    log line and PASSES with it, asserting the parsed fields -- not just
    that *a* line was emitted."""

    async def test_compression_serve_reports_exact_skip_count_and_classes(
        self, caplog,
    ):
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(staging_max=10)
        subs = [
            Slot.new(f"sub{i}", client_meta={"is_sub_agent": True})
            for i in range(3)
        ]
        comp = Slot.new(
            "comp", client_meta={"is_compression": True}, thread_id="thr-comp",
        )
        for s in subs:
            await q.enqueue(s)
        await q.enqueue(comp)

        popped = await q.pop_next()
        assert popped.slot_id == comp.slot_id  # pre-existing behavior, unchanged

        lines = _queue_priority_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "served_class=compression" in line
        assert "reason=compression" in line
        # THE load-bearing assertion: the exact count, not just "a" skip.
        assert "skipped=3" in line
        assert "skipped_classes=sub-agent,sub-agent,sub-agent" in line

    async def test_compression_present_but_first_in_line_reports_zero_skipped(
        self, caplog,
    ):
        # "compression happened to be first" must be DISTINGUISHABLE from
        # "compression jumped the queue" -- the whole point of the field.
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(staging_max=10)
        comp = Slot.new("comp", client_meta={"is_compression": True})
        sub = Slot.new("sub", client_meta={"is_sub_agent": True})
        await q.enqueue(comp)
        await q.enqueue(sub)

        popped = await q.pop_next()
        assert popped.slot_id == comp.slot_id

        line = _queue_priority_lines(caplog)[0]
        assert "skipped=0" in line
        assert "skipped_classes=" in line
        assert "skipped_classes=sub-agent" not in line


@pytest.mark.asyncio
class TestQueuePriorityMainLane:
    async def test_main_lane_serve_reports_skipped_aux(self, caplog):
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=True)
        aux = Slot.new("aux", client_meta={"is_sub_agent": True})
        main = Slot.new("main", client_meta={"is_main": True})
        await q.enqueue(aux)
        await q.enqueue(main)

        popped = await q.pop_next()
        assert popped.slot_id == main.slot_id

        line = _queue_priority_lines(caplog)[0]
        assert "served_class=main" in line
        assert "reason=main-lane" in line
        assert "skipped=1" in line
        assert "skipped_classes=sub-agent" in line


@pytest.mark.asyncio
class TestQueuePriorityFifo:
    async def test_plain_fifo_serve_is_always_zero_skipped(self, caplog):
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(staging_max=10)
        a = Slot.new("a")
        b = Slot.new("b")
        await q.enqueue(a)
        await q.enqueue(b)

        popped = await q.pop_next()
        assert popped.slot_id == a.slot_id

        line = _queue_priority_lines(caplog)[0]
        assert "reason=fifo" in line
        assert "skipped=0" in line
        assert "skipped_classes=" in line

    async def test_fifo_after_buffer_drain_retry_also_logs(self, caplog):
        # The retry-after-drain branch (staging was empty, accept_buf feeds
        # it, then the helper is called a second time) is a SEPARATE call
        # site from the immediate-hit branch -- both must log.
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(staging_max=1, acceptance_max=10)
        a = Slot.new("a")
        b = Slot.new("b")
        await q.enqueue(a)
        await q.enqueue(b)  # staging full at 1 -> b lands in accept_buf

        await q.pop_next()  # drains a, replenishes staging from accept_buf
        caplog.clear()
        popped = await q.pop_next()
        assert popped.slot_id == b.slot_id

        lines = _queue_priority_lines(caplog)
        assert len(lines) == 1
        assert "reason=fifo" in lines[0]
        assert "skipped=0" in lines[0]


@pytest.mark.asyncio
class TestServedClassLabelOnlyResolution:
    """A known limitation: served_class is a labels-only subset of
    the manager's classify_event-based resolved_class, and CAN diverge for
    the same request. Pinning the case most likely to diverge: unlabeled."""

    async def test_unlabeled_slot_defaults_served_class_to_main(self, caplog):
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new("m")  # no client_meta at all
        await q.enqueue(s)
        await q.pop_next()

        line = _queue_priority_lines(caplog)[0]
        assert "served_class=main" in line

    async def test_curator_label_outranks_main_lane_reason(self, caplog):
        # A slot can carry BOTH is_main and is_curator (verified
        # behavior per kv_classify's _class_from_label docstring). It still
        # matches _is_main_lane (reason=main-lane fires), but served_class
        # must resolve via the SAME curator>compression>sub-agent>main
        # priority the rest of the system uses -- reason and served_class
        # are independent axes and can legitimately disagree.
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=True)
        s = Slot.new("m", client_meta={"is_main": True, "is_curator": True})
        await q.enqueue(s)
        await q.pop_next()

        line = _queue_priority_lines(caplog)[0]
        assert "reason=main-lane" in line
        assert "served_class=curator" in line


@pytest.mark.asyncio
class TestSkippedCountsPriorityPassoversOnly:
    """Semantic pin: skipped counts entries passed over for
    PRIORITY only -- never evicted/dead entries. An evicted-but-non-matching
    entry sitting in the scan path is still a genuine priority-passover
    (it is not removed, not specially excluded) and must be counted
    exactly like any other skip -- the count must not be silently altered
    by an unrelated entry's eviction state in either direction."""

    async def test_evicted_nonmatching_entry_still_counts_as_skipped(
        self, caplog,
    ):
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=True)
        dead = Slot.new("dead", client_meta={"is_sub_agent": True})
        dead.disconnect_event = asyncio.Event()
        dead.disconnect_event.set()  # client disconnected while queued
        alive_aux = Slot.new("alive", client_meta={"is_sub_agent": True})
        main = Slot.new("main", client_meta={"is_main": True})
        await q.enqueue(dead)
        await q.enqueue(alive_aux)
        await q.enqueue(main)

        popped = await q.pop_next()
        assert popped.slot_id == main.slot_id

        line = _queue_priority_lines(caplog)[0]
        assert "skipped=2" in line
        assert "skipped_classes=sub-agent,sub-agent" in line
        # the evicted-but-skipped entry is untouched by this scan -- still
        # queued, not silently dropped or double-counted.
        assert dead in q._staging


@pytest.mark.asyncio
class TestQueuePriorityDepthsAndThread:
    async def test_depths_reflect_post_removal_pre_replenish_state(
        self, caplog,
    ):
        # staging_max=1: after popping the head, staging is momentarily
        # empty, THEN pop_next replenishes from accept_buf. The log must
        # capture the momentary post-removal/pre-replenish depth (0), not
        # the post-replenish depth (1), and must do so identically for
        # every reason.
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(staging_max=1, acceptance_max=10)
        a = Slot.new("a")
        b = Slot.new("b")
        await q.enqueue(a)
        await q.enqueue(b)  # staging=[a], accept_buf=[b]

        await q.pop_next()

        line = _queue_priority_lines(caplog)[0]
        assert "staging_depth=0" in line
        assert "accept_depth=1" in line

    async def test_thread_field_is_a_short_hash_not_the_raw_thread_id(
        self, caplog,
    ):
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(staging_max=10)
        raw = "session-router-ip-fingerprint-do-not-leak"
        s = Slot.new("m", thread_id=raw)
        await q.enqueue(s)
        await q.pop_next()

        line = _queue_priority_lines(caplog)[0]
        assert raw not in line
        import re
        m = re.search(r"thread=([0-9a-f]{12})$", line)
        assert m is not None, line


@pytest.mark.asyncio
class TestQueuePriorityEvictedInFlightStillLogs:
    async def test_evicted_in_flight_serve_still_emits_the_line(self, caplog):
        # Suppressing the line here would reintroduce the exact "absent
        # line is ambiguous" defect this line exists to remove (eviction vs.
        # broken logging would be indistinguishable).
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new("m")
        s.disconnect_event = asyncio.Event()
        s.disconnect_event.set()
        await q.enqueue(s)

        popped = await q.pop_next()
        assert popped.is_evicted is True  # pre-existing behavior, unchanged

        lines = _queue_priority_lines(caplog)
        assert len(lines) == 1
        assert "reason=fifo" in lines[0]


@pytest.mark.asyncio
class TestAffinityPathReservedUnemitted:
    async def test_affinity_served_slot_emits_no_queue_priority_line(
        self, caplog,
    ):
        # reason=affinity is a RESERVED, deliberately UNEMITTED value of the
        # reason field (separate feature area). Pinned here so a future
        # reader can trust the comment: this absence is intentional.
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        q = TurbohaulQueue(
            staging_max=10,
            max_consecutive_same_model=100,
            max_other_model_wait_s=999.0,
        )
        a1 = Slot.new("model-a")
        b1 = Slot.new("model-b")
        await q.enqueue(a1)
        await q.enqueue(b1)

        popped = await q.pop_next(warm_model_tag="model-a")
        assert popped.slot_id == a1.slot_id  # confirms the affinity branch fired

        assert _queue_priority_lines(caplog) == []
