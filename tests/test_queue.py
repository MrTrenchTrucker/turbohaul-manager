"""Tests for TurbohaulQueue + GraceTimer + IdleHotTimer."""
import ast
import asyncio
import inspect
import textwrap
import time

import pytest

from turbohaul.fastlane import FastLaneMatch
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

    async def test_main_lane_reservation_preempts_queued_auxiliary_work(self):
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=True)
        aux = Slot.new("aux", client_meta={"is_sub_agent": True})
        main = Slot.new("main", client_meta={"is_main": True})
        await q.enqueue(aux)
        await q.enqueue(main)
        assert (await q.pop_next()).slot_id == main.slot_id
        assert (await q.pop_next()).slot_id == aux.slot_id

    async def test_disabled_main_reservation_preserves_fifo(self):
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
        aux = Slot.new("aux", client_meta={"is_sub_agent": True})
        main = Slot.new("main", client_meta={"is_main": True})
        await q.enqueue(aux)
        await q.enqueue(main)
        assert (await q.pop_next()).slot_id == aux.slot_id

    # === Compression priority-skip ============================

    async def test_compression_skips_ahead_of_queued_sub_agent(self):
        # A compression request enqueued AFTER a sub-agent request must pop
        # first (skip the queue) so context can't grow unchecked while the
        # harness waits on parallel sub-agents. The sub-agent then pops next.
        q = TurbohaulQueue(staging_max=10)
        sub = Slot.new("sub", client_meta={"is_sub_agent": True})
        comp = Slot.new("comp", client_meta={"is_compression": True})
        await q.enqueue(sub)
        await q.enqueue(comp)
        assert (await q.pop_next()).slot_id == comp.slot_id
        assert (await q.pop_next()).slot_id == sub.slot_id

    async def test_compression_priority_without_compression_is_fifo(self):
        # No compression-tagged slot present -> strict FIFO preserved (no
        # behaviour change for ordinary queues).
        q = TurbohaulQueue(staging_max=10)
        a = Slot.new("a")
        b = Slot.new("b")
        await q.enqueue(a)
        await q.enqueue(b)
        assert (await q.pop_next()).slot_id == a.slot_id
        assert (await q.pop_next()).slot_id == b.slot_id

    # === Residency ===================================

    async def test_head_model_tag_peek_is_non_destructive(self):
        q = TurbohaulQueue(staging_max=10)
        assert await q.head_model_tag() is None  # empty -> None
        a = Slot.new("model-a")
        b = Slot.new("model-b")
        await q.enqueue(a)
        await q.enqueue(b)
        before = q.depth()
        assert await q.head_model_tag() == "model-a"
        # Idempotent + non-destructive: repeat peek, depth + head unchanged.
        assert await q.head_model_tag() == "model-a"
        assert q.depth() == before
        # FIFO order is intact after peeking.
        assert (await q.pop_next()).slot_id == a.slot_id
        assert await q.head_model_tag() == "model-b"

    async def test_batch_cap_does_not_force_avoidable_swap(self):
        """After max_consecutive_same_model same-model pops, a NON-starved
        other-model head must NOT force a swap while same-model work is still
        queued. FAILS before the fix (4th pop returned 'N'); PASSES after."""
        q = TurbohaulQueue(
            staging_max=100,
            max_consecutive_same_model=3,
            max_other_model_wait_s=3600.0,  # huge -> head never 'starved'
        )
        # FIFO: M M M N M  (other-model N buried behind 3 M's, one more M after)
        for _ in range(3):
            await q.enqueue(Slot.new("M"))
        n = Slot.new("N")
        await q.enqueue(n)
        m_tail = Slot.new("M")
        await q.enqueue(m_tail)

        for _ in range(3):  # drain 3 M's -> consecutive-same run hits the cap
            p = await q.pop_next(warm_model_tag="M")
            assert p.model_tag == "M"

        # 4th pop: head is now N (different, NOT starved) but m_tail is queued.
        p4 = await q.pop_next(warm_model_tag="M")
        assert p4.model_tag == "M"          # was "N" before the fix (avoidable swap)
        assert p4.slot_id == m_tail.slot_id
        # N is not starved -> served next (never dropped).
        p5 = await q.pop_next(warm_model_tag="M")
        assert p5.model_tag == "N"

    async def test_starved_other_model_still_forces_swap(self):
        """Genuine head-starvation (aged past max_other_model_wait_s) STILL forces
        the swap even with same-model affinity active -> no other-model starvation."""
        q = TurbohaulQueue(
            staging_max=100,
            max_consecutive_same_model=1000,  # count cap effectively off
            max_other_model_wait_s=10.0,
        )
        n = Slot.new("N")
        n.created_at = time.monotonic() - 100.0  # aged well past the window
        await q.enqueue(n)
        await q.enqueue(Slot.new("M"))
        # resident=M, but the aged N head forces the swap.
        p = await q.pop_next(warm_model_tag="M")
        assert p.model_tag == "N"

    # === Starvation scan beyond index 0 ==================================

    async def test_starved_other_model_at_index_ge1_is_served_not_invisible(self):
        """A starved other-model entry BURIED BEHIND
        a same-model head (index >= 1, not index 0) must still be found and
        drained. A scan that only inspects staging[0] for
        head_is_other/head_starved would miss it -- since a same-model follow-up is
        appendleft'd to the head on every grace-match (manager.py's
        enqueue_head), index 0 is almost always same-model, so a starved
        other-model entry sitting anywhere behind it would be structurally
        invisible: max_other_model_wait_s could never fire for it, no matter
        how long it aged.

        A head-only scan fails here -- the aged N never
        forces, M is popped instead, both assertions below fail. The
        bounded scan finds N specifically (not index 0) and
        drains it, leaving M (the actual FIFO head) queued for next."""
        q = TurbohaulQueue(
            staging_max=100,
            max_consecutive_same_model=1000,  # count cap effectively off
            max_other_model_wait_s=10.0,
        )
        m_head = Slot.new("M")
        await q.enqueue(m_head)
        n = Slot.new("N")
        n.created_at = time.monotonic() - 100.0  # aged well past the window
        await q.enqueue(n)
        # staging = [M (head, index 0), N (starved, index 1)]
        assert q._staging[0].model_tag == "M"
        assert q._staging[1].model_tag == "N"

        p = await q.pop_next(warm_model_tag="M")
        assert p.slot_id == n.slot_id, (
            "starved N at index 1 must be served, not the same-model head"
        )
        assert p.model_tag == "N"
        # M (the real FIFO head) is untouched, still queued, served next.
        p2 = await q.pop_next(warm_model_tag="M")
        assert p2.slot_id == m_head.slot_id

    async def test_strict_fifo_unchanged_when_warm_model_tag_is_none(self):
        """warm_model_tag=None (the default for every existing caller) must
        stay byte-identical to plain FIFO regardless of model_tag mix or age
        -- the starvation scan lives entirely inside the
        warm_model_tag-supplied affinity branch and must never run on this
        path. Settings below would starve/force-swap EVERYTHING instantly if
        the scan were (wrongly) consulted here -- proving it isn't."""
        q = TurbohaulQueue(
            staging_max=100,
            max_consecutive_same_model=1,
            max_other_model_wait_s=0.0,
        )
        slots = [Slot.new("M"), Slot.new("N"), Slot.new("M"), Slot.new("N")]
        for s in slots:
            await q.enqueue(s)
        for expected in slots:
            popped = await q.pop_next(warm_model_tag=None)
            assert popped.slot_id == expected.slot_id


@pytest.mark.asyncio
class TestOnEnqueueHook:
    """The dispatcher-wake hook: fired after every successful append, on both
    enqueue() and enqueue_head() -- the latter is the requeue-after-backoff
    re-entry point, so hooking enqueue() alone would leave the dispatcher
    unable to notice a slot returning from its own defer/backoff cycle."""

    async def test_no_hook_is_a_safe_no_op(self):
        q = TurbohaulQueue(staging_max=10)  # on_enqueue defaults to None
        await q.enqueue(Slot.new("m"))  # must not raise

    async def test_fires_on_enqueue_staging_branch(self):
        calls = []
        q = TurbohaulQueue(staging_max=10, on_enqueue=lambda: calls.append(1))
        await q.enqueue(Slot.new("m"))
        assert calls == [1]

    async def test_fires_on_enqueue_accept_buf_branch(self):
        """A slot landing in the acceptance buffer (staging full) is still a
        real arrival the dispatcher should be told about."""
        calls = []
        q = TurbohaulQueue(staging_max=0, acceptance_max=10, on_enqueue=lambda: calls.append(1))
        await q.enqueue(Slot.new("m"))
        assert calls == [1]

    async def test_fires_on_enqueue_head(self):
        calls = []
        q = TurbohaulQueue(staging_max=10, on_enqueue=lambda: calls.append(1))
        await q.enqueue_head(Slot.new("m"))
        assert calls == [1]

    async def test_fires_after_the_append_not_before(self):
        """The hook must observe the slot ALREADY landed -- a wake that fires
        before the append could race a dispatcher into finding nothing."""
        depths_seen_by_hook = []
        q = TurbohaulQueue(
            staging_max=10,
            on_enqueue=lambda: depths_seen_by_hook.append(len(q._staging)),
        )
        await q.enqueue(Slot.new("m"))
        assert depths_seen_by_hook == [1]

    async def test_raising_hook_is_caught_and_logged_not_propagated(self, caplog):
        import logging

        def boom():
            raise RuntimeError("hook exploded")

        q = TurbohaulQueue(staging_max=10, on_enqueue=boom)
        with caplog.at_level(logging.ERROR):
            await q.enqueue(Slot.new("m"))  # must NOT raise
        assert any(
            "QUEUE_WAKE_HOOK_FAILED" in rec.message and rec.levelname == "ERROR"
            for rec in caplog.records
        ), caplog.records
        # The append itself must have succeeded despite the hook raising.
        d = q.depth()
        assert d["staging_queue_depth"] + d["acceptance_buffer_depth"] == 1

    async def test_raising_hook_on_enqueue_head_is_also_caught(self, caplog):
        import logging

        q = TurbohaulQueue(staging_max=10, on_enqueue=lambda: (_ for _ in ()).throw(RuntimeError("x")))
        with caplog.at_level(logging.ERROR):
            await q.enqueue_head(Slot.new("m"))  # must NOT raise
        assert any(rec.levelname == "ERROR" for rec in caplog.records)


class TestDepthCountsInboxWaiters:
    """manager._route_or_reserve's HIT route
    hands a request straight to a busy resident's inbox and returns without
    ever calling this module's enqueue() (see _log_admitted_without_pick's
    docstring) -- such a request never enters _staging/_accept_buf and was
    silently invisible to the reported depth. queue.py has no visibility
    into resident inboxes; the caller (manager.py) supplies the live count."""

    async def test_depth_counts_inbox_waiters_invisible_to_staging_and_accept_buf(self):
        q = TurbohaulQueue(staging_max=10, acceptance_max=100)
        # No slot is ever enqueued -- exactly the HIT-route bypass: staging
        # and the acceptance buffer stay empty while requests are genuinely
        # waiting in a resident's inbox.
        d = q.depth(inbox_waiting=3)
        assert d["staging_queue_depth"] == 0
        assert d["acceptance_buffer_depth"] == 0
        # Previously this read 0.
        assert d["queue_depth_total"] == 3

    async def test_depth_zero_control_reads_zero_with_nothing_waiting(self):
        """Zero-control: without this, queue_depth_total could be
        hardcoded non-zero (or always echo some unrelated constant) and
        still pass the arm above for the wrong reason."""
        q = TurbohaulQueue(staging_max=10, acceptance_max=100)
        d = q.depth()
        assert d["queue_depth_total"] == 0

    async def test_depth_total_sums_all_three_sources(self):
        q = TurbohaulQueue(staging_max=10, acceptance_max=100)
        await q.enqueue(Slot.new("m"))
        d = q.depth(inbox_waiting=2)
        assert d["staging_queue_depth"] == 1
        assert d["acceptance_buffer_depth"] == 0
        assert d["queue_depth_total"] == 3

    async def test_depth_back_compat_existing_fields_unchanged(self):
        """The two pre-existing depth fields (and the two *_max fields) must
        read exactly what they read before -- queue_depth_total is
        an ADDITIONAL key, not a redefinition of either."""
        q = TurbohaulQueue(staging_max=5, acceptance_max=50)
        await q.enqueue(Slot.new("m"))
        await q.enqueue(Slot.new("m"))
        d = q.depth(inbox_waiting=7)
        assert d["staging_queue_depth"] == 2
        assert d["acceptance_buffer_depth"] == 0
        assert d["staging_queue_max"] == 5
        assert d["acceptance_buffer_max"] == 50
        # No-arg call (every existing caller in the codebase) still works
        # and still omits nothing that was there before.
        d_default = q.depth()
        assert {
            "acceptance_buffer_depth", "staging_queue_depth",
            "staging_queue_max", "acceptance_buffer_max",
        } <= set(d_default)
        assert d_default["queue_depth_total"] == 2  # inbox_waiting defaults to 0

    async def test_depth_counts_claims_waiting_not_in_staging_or_accept_buf(self):
        """A live Fast Lane claim parked in the
        manager's _fastlane_claims registry (during the _defer_unroutable
        backoff requeue window) is invisible to staging + accepted +
        inbox_waiting. claims_waiting closes that blind spot. Default 0
        keeps the field absent for callers that do not supply it.

        Non-vacuity control: point at KNOWN-ZERO --
        depth() with no claims yields exactly staging+accepted+inbox."""
        q = TurbohaulQueue(staging_max=10, acceptance_max=100)
        await q.enqueue(Slot.new("m"))
        # No-arg call: claims_waiting defaults to 0, same as before.
        d0 = q.depth(inbox_waiting=2)
        assert d0["queue_depth_total"] == 3  # staging 1 + inbox 2 + claims 0

        # claims_waiting=2: staging 1 + inbox 2 + claims 2 = 5
        d1 = q.depth(inbox_waiting=2, claims_waiting=2)
        assert d1["queue_depth_total"] == 5
        assert d1["staging_queue_depth"] == 1  # unchanged
        assert d1["acceptance_buffer_depth"] == 0  # unchanged


@pytest.mark.asyncio
class TestQueueSnapshot:
    """TurbohaulQueue.queue_snapshot(limit=50)
    -- the waiting-request surface behind /status's queue.waiting[]."""

    async def test_includes_both_staging_and_accept_buffer_state_distinguishes_them(self):
        """An accept-buffer slot IS included, not just _staging -- the snapshot
        covers waiting requests the whole time, and
        slot.fastlane is already resolved before either buffer decision.
        `state` is what tells the two populations apart -- no new field."""
        q = TurbohaulQueue(staging_max=1, acceptance_max=10)
        s1 = Slot.new("m")
        s2 = Slot.new("m")
        await q.enqueue(s1)  # lands in _staging (room)
        await q.enqueue(s2)  # staging full -> lands in _accept_buf
        rows = q.queue_snapshot()
        assert [r["slot_id"] for r in rows] == [s1.slot_id, s2.slot_id]
        assert rows[0]["state"] == "STAGED"
        assert rows[1]["state"] == "ACCEPT_BUFFER"

    async def test_position_is_index_in_combined_snapshot(self):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        for _ in range(3):
            await q.enqueue(Slot.new("m"))
        rows = q.queue_snapshot()
        assert [r["position"] for r in rows] == [0, 1, 2]

    async def test_limit_truncates_the_combined_list(self):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        for _ in range(5):
            await q.enqueue(Slot.new("m"))
        rows = q.queue_snapshot(limit=2)
        assert len(rows) == 2
        assert [r["position"] for r in rows] == [0, 1]

    async def test_empty_queue_returns_empty_list(self):
        """Zero-control, same reasoning as depth()'s own zero
        control just above: without this, queue_snapshot could fabricate
        rows and still pass every arm above for the wrong reason."""
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        assert q.queue_snapshot() == []

    async def test_fastlane_none_for_an_unlisted_slot(self):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        await q.enqueue(Slot.new("m"))  # slot.fastlane defaults to None
        rows = q.queue_snapshot()
        assert rows[0]["fastlane"] is None

    async def test_fastlane_dict_shape_for_a_listed_slot(self):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        s = Slot.new("m")
        s.fastlane = FastLaneMatch(
            rule_index=2, raw_address="10.0.0.9", label="ops-box",
            effective_tag="main", rank=1,
        )
        await q.enqueue(s)
        rows = q.queue_snapshot()
        assert rows[0]["fastlane"] == {
            "rule_index": 2, "rank": 1, "label": "ops-box",
            "fastlane_rule": "10.0.0.9",
        }

    async def test_floor_promoted_passthrough(self):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        s = Slot.new("m")
        s.floor_promoted = True
        await q.enqueue(s)
        rows = q.queue_snapshot()
        assert rows[0]["floor_promoted"] is True

    async def test_waited_s_reflects_elapsed_time_since_created_at(self):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        s = Slot.new("m")
        await q.enqueue(s)
        s.created_at = time.monotonic() - 5.0
        rows = q.queue_snapshot()
        assert 4.9 <= rows[0]["waited_s"] <= 5.2

    async def test_redaction_shape_never_leaks_full_thread_id_or_client_meta(self):
        """SECURITY CHECK. A slot carrying a full thread_id
        and a client_meta with a raw ip must surface ONLY thread_id_prefix
        (8 chars) on the row -- never the full thread_id, never client_meta,
        never an 'ip' key/value anywhere in the row. A version that leaks the
        full thread_id fails this test; it is kept as the
        permanent
        guard against any such
        regression."""
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        s = Slot.new("m", thread_id="thread-abcdefghijklmnop-full-id")
        s.client_meta = {"ip": "203.0.113.7", "session_id": "sess-1"}
        await q.enqueue(s)
        rows = q.queue_snapshot()
        row = rows[0]
        assert row["thread_id_prefix"] == "thread-a"
        assert "thread_id" not in row
        assert "client_meta" not in row
        row_text = repr(row)
        assert "thread-abcdefghijklmnop-full-id" not in row_text
        assert "203.0.113.7" not in row_text
        assert "session_id" not in row_text


class TestQueueSnapshotStillAwaitFree:
    """queue_snapshot must have zero
    suspension points, always.

    AST-based, not substring-based -- modelled on
    ``TestPriorityAdmitFromInboxStillAwaitFree``
    (in the turn-boundary handoff test). See also
    ``TestQueueSurfaceStillAwaitFree`` in ``test_manager.py`` for the other
    two functions this same invariant covers (``status_snapshot``,
    ``_likely_unload_target_model_tag``) -- being sync with zero suspension points
    is what makes this lock-free read atomic on the single-threaded event
    loop; a single `await` added here would silently reintroduce a
    torn-view race between a mutating deque and this read.
    """

    def test_queue_snapshot_has_zero_suspension_points(self):
        """Mutant this kills: turning queue_snapshot into `async def` and
        adding a real `await asyncio.sleep(0)` anywhere in its body --
        `offenders` would become non-empty and the assertion would name
        queue_snapshot."""
        src = textwrap.dedent(inspect.getsource(TurbohaulQueue.queue_snapshot))
        tree = ast.parse(src)
        offenders = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith))
        ]
        assert offenders == [], (
            f"queue_snapshot must have zero suspension points -- found {len(offenders)}"
        )


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
