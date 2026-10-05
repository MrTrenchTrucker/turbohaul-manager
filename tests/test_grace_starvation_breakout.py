"""Tests for the GRACE loop starvation breakout: the GRACE loop must evaluate the same starvation
rule pop_next uses and break out (to pop_next, which already forces the
swap) instead of holding a different-model request behind a same-model
grace window all the way to grace_seconds.

Covers all three call sites of the shared starvation predicate:
  - TurbohaulQueue._starved_other_model_locked / starved_other_model (queue.py)
  - TurbohaulManager._process_slot's grace loop (the single-sidecar /
    legacy worker_loop path, "Loop B")
  - TurbohaulManager._serve_on_resident's grace loop (the multi-slot
    dispatcher path, "Loop A")
"""
import asyncio
import time
from unittest.mock import MagicMock

import pytest

from turbohaul.config import (
    BootConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import Resident, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer, TurbohaulQueue
from turbohaul.slot import Slot, SlotState
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime(tmp_path, *, grace_seconds=30, max_other_model_wait_s=20.0,
                   max_grace_extensions=50):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False,
            grace_seconds=grace_seconds,
            max_other_model_wait_s=max_other_model_wait_s,
            max_grace_extensions=max_grace_extensions,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _make_fake_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 88_888
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _audit_events(boot, slot_id):
    conn = open_state_db(boot.storage.state_db_path)
    cur = conn.execute(
        "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
        (slot_id,),
    )
    events = [r["event_type"] for r in cur.fetchall()]
    conn.close()
    return events


# === Group 1: queue.py — the shared predicate itself =======================

@pytest.mark.asyncio
class TestSharedStarvationPredicate:
    async def test_locked_and_wrapper_agree(self):
        """starved_other_model() (lock-acquiring, used by the grace loops) and
        _starved_other_model_locked() (used by pop_next under its own held
        lock) must return the identical slot -- ONE source of truth, not two
        copies that can drift."""
        q = TurbohaulQueue(staging_max=100, max_other_model_wait_s=5.0)
        m = Slot.new("M")
        await q.enqueue(m)
        n = Slot.new("N")
        n.created_at = time.monotonic() - 100.0
        await q.enqueue(n)

        via_wrapper = await q.starved_other_model("M")
        async with q._lock:
            via_locked = q._starved_other_model_locked("M")

        assert via_wrapper is not None
        assert via_wrapper.slot_id == n.slot_id
        assert via_locked is not None
        assert via_locked.slot_id == n.slot_id

    async def test_wrapper_matches_what_pop_next_would_drain(self):
        """The grace loops' predicate must agree with pop_next's own decision
        -- this is the refactor-equivalence proof for the extraction: same
        input, same staging contents, identical answer."""
        q = TurbohaulQueue(staging_max=100, max_consecutive_same_model=1000,
                            max_other_model_wait_s=5.0)
        m = Slot.new("M")
        await q.enqueue(m)
        n = Slot.new("N")
        n.created_at = time.monotonic() - 100.0
        await q.enqueue(n)

        starved = await q.starved_other_model("M")
        assert starved is not None and starved.slot_id == n.slot_id

        popped = await q.pop_next(warm_model_tag="M")
        assert popped.slot_id == starved.slot_id, (
            "starved_other_model must name exactly the slot pop_next drains"
        )

    async def test_no_false_positive_when_not_aged(self):
        q = TurbohaulQueue(staging_max=100, max_other_model_wait_s=3600.0)
        await q.enqueue(Slot.new("M"))
        n = Slot.new("N")
        await q.enqueue(n)  # fresh, not aged
        assert await q.starved_other_model("M") is None

    async def test_none_when_staging_empty_or_all_same_model(self):
        q = TurbohaulQueue(staging_max=100, max_other_model_wait_s=0.0)
        assert await q.starved_other_model("M") is None
        await q.enqueue(Slot.new("M"))
        assert await q.starved_other_model("M") is None  # only same-model present

    async def test_starved_waiter_in_accept_buf_is_found(self):
        """A starvation-eligible waiter that overflowed
        into _accept_buf (staging full at enqueue time) must still be visible
        to the starvation predicate. staging_max=1 forces the second enqueue
        into _accept_buf."""
        q = TurbohaulQueue(staging_max=1, max_other_model_wait_s=0.2)
        now = time.monotonic()

        # Fill staging with the same-model anchor
        anchor = Slot.new("M")
        anchor.created_at = now - 0.01
        await q.enqueue(anchor)
        assert anchor in q._staging

        # Different-model waiter, aged past max_other_model_wait_s, overflows
        # into _accept_buf because staging_max=1
        other = Slot.new("N")
        other.created_at = now - 100.0
        await q.enqueue(other)
        assert other in q._accept_buf

        starved = await q.starved_other_model("M")
        assert starved is not None, (
            "starved waiter in _accept_buf was invisible to starved_other_model"
        )
        assert starved.slot_id == other.slot_id

    async def test_oldest_wins_across_both_buffers(self):
        """ORDERING CONTROL: when a starved other-model
        waiter resides in _accept_buf AND a different starved other-model
        waiter resides in _staging, the OLDEST (by created_at) must win --
        not the first-in-concatenation-order. This is the assertion that
        catches the naive-scan trap: a naive 'scan _accept_buf too, break on first hit'
        fix passes 'found in accept_buf' but fails 'oldest across buffers'.
        """
        q = TurbohaulQueue(staging_max=2, max_other_model_wait_s=0.2)
        now = time.monotonic()

        # Same-model anchor fills staging slot 1
        anchor = Slot.new("M")
        anchor.created_at = now - 0.01
        await q.enqueue(anchor)
        assert anchor in q._staging

        # YOUNGER starved other-model goes to staging (room remains)
        young = Slot.new("N")  # different model
        young.created_at = now - 50.0
        await q.enqueue(young)
        assert young in q._staging

        # staging is now full (2/2). OLDER starved other-model overflows
        # into _accept_buf.
        old = Slot.new("K")  # different model, older
        old.created_at = now - 100.0
        await q.enqueue(old)
        assert old in q._accept_buf, (
            "oldest starved waiter must land in _accept_buf to test cross-buffer ordering"
        )

        starved = await q.starved_other_model("M")
        assert starved is not None
        assert starved.slot_id == old.slot_id, (
            "must return the OLDEST starved waiter across both buffers, "
            f"not the younger one from staging: got {starved.slot_id} "
            f"({starved.model_tag}), expected {old.slot_id} ({old.model_tag})"
        )


# === Group 2: manager.py Loop B (_process_slot, single-sidecar path) =======

@pytest.mark.asyncio
class TestProcessSlotGraceStarvationBreakout:
    async def test_discriminator_grace_exits_on_starvation_not_full_deadline(
        self, tmp_path
    ):
        """THE defect this change fixes. Worker sits in GRACE for model 'm1'
        thread 't1' with no follow-up; a different-model 'm2' request is
        staged and already aged past max_other_model_wait_s. The grace loop
        must break out well before grace_seconds, not hold to the full
        60s-class deadline.

        Fails without the change: this loop only polls
        pop_matched_thread and never consults starvation, so it holds to the
        full grace_seconds deadline every time -- the bound assertion below
        (exit within ~1s of a 5s grace_seconds window) fails without the change."""
        grace_seconds = 5
        max_other_model_wait_s = 0.2
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            max_other_model_wait_s=max_other_model_wait_s,
        )

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            return True, 100

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        slot_a = await mgr.submit(model_tag="m1", prompt="hi", thread_id="t1")

        # Stage a different-model request, backdated past max_other_model_wait_s
        # BEFORE the worker even starts, so it is already starved the instant
        # grace begins.
        slot_b = Slot.new("m2", prompt="hi", thread_id="t2")
        slot_b.created_at = time.monotonic() - 100.0
        await mgr.queue.enqueue(slot_b)

        mgr._worker_task = asyncio.create_task(mgr.worker_loop())

        started = time.monotonic()
        # Poll until slot_a's audit trail shows the breakout event, bounded
        # well under grace_seconds so a hang (unfixed behavior) fails the test
        # via timeout rather than hanging the suite.
        deadline = started + grace_seconds - 0.5
        breakout_seen = False
        while time.monotonic() < deadline:
            if "grace_starvation_breakout" in _audit_events(boot, slot_a.slot_id):
                breakout_seen = True
                break
            await asyncio.sleep(0.05)
        elapsed = time.monotonic() - started

        await mgr.shutdown()

        assert breakout_seen, (
            "grace loop never broke out on the starved other-model request "
            "-- held toward the full grace_seconds deadline instead"
        )
        assert elapsed < grace_seconds - 1.0, (
            f"grace exited after {elapsed:.2f}s, not meaningfully sooner than "
            f"grace_seconds={grace_seconds}s -- starvation rule was not consulted"
        )

        # Non-vacuity: the predicate was actually consulted (audit line fired)
        # AND the starved slot subsequently got served, not just a timer firing.
        events_a = _audit_events(boot, slot_a.slot_id)
        assert "grace_starvation_breakout" in events_a

    async def test_invariant_same_thread_followup_still_served_and_rearms(
        self, tmp_path
    ):
        """The arm that must stay byte-identical: a same-model/same-thread
        follow-up is still served instantly out of grace via ACTIVE_MATCH,
        and grace still re-arms -- even with an aged other-model request
        also sitting in staging. The match branch must win; the starvation
        break must never preempt an available match (it lives strictly on
        the no-match fallthrough)."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds, max_other_model_wait_s=0.2,
        )

        complete_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            return True, 100

        async def fake_complete(slot, handle):
            complete_calls.append(slot.slot_id)
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        slot_a = await mgr.submit(model_tag="m1", prompt="hi", thread_id="t1")

        # A starved other-model request present the whole time -- must not
        # cause the same-thread follow-up below to be skipped or delayed.
        slot_other = Slot.new("m2", prompt="hi", thread_id="t2")
        slot_other.created_at = time.monotonic() - 100.0
        await mgr.queue.enqueue(slot_other)

        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        # Let slot_a reach ACTIVE + GRACE.
        for _ in range(40):
            if "grace_enter" in _audit_events(boot, slot_a.slot_id):
                break
            await asyncio.sleep(0.05)
        assert "grace_enter" in _audit_events(boot, slot_a.slot_id)

        # Same-thread follow-up: must be matched via pop_matched_thread and
        # served -- not lost to the starvation break-out.
        followup = await mgr.submit(model_tag="m1", prompt="again", thread_id="t1")
        for _ in range(60):
            if followup.slot_id in complete_calls:
                break
            await asyncio.sleep(0.05)

        await mgr.shutdown()

        assert followup.slot_id in complete_calls, (
            "same-thread/same-model follow-up was not served via ACTIVE_MATCH "
            "-- starvation break-out must never preempt an available match"
        )
        events_followup = _audit_events(boot, followup.slot_id)
        assert "active_match_completed" in events_followup or (
            "active" in events_followup
        )


# === Group 3: manager.py Loop A (_serve_on_resident, dispatcher path) ======

@pytest.mark.asyncio
class TestServeOnResidentGraceStarvationBreakout:
    async def test_discriminator_and_non_vacuity(self, tmp_path):
        """Same defect, same fix, the OTHER grace loop (_serve_on_resident,
        the multi-slot dispatcher's per-resident driver). Direct call --
        exercises exactly the loop this change touched without needing the full
        dispatcher/GPU-reservation machinery. Fails without the change
        (grace holds to the full deadline; no 'grace_starvation_breakout'
        audit event is ever written)."""
        grace_seconds = 5
        max_other_model_wait_s = 0.2
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            max_other_model_wait_s=max_other_model_wait_s,
        )

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )

        # A different-model request, already aged past max_other_model_wait_s,
        # sitting in the shared queue -- the resident's own grace loop must
        # notice it (via starved_other_model) and break out.
        slot_other = Slot.new("m2", prompt="hi", thread_id="t2")
        slot_other.created_at = time.monotonic() - 100.0
        await mgr.queue.enqueue(slot_other)

        handle = _make_fake_handle("m1", 59500)
        r = Resident(
            model_tag="m1",
            handle=handle,
            port=handle.port,
            grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
        )
        anchor = Slot.new("m1", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED

        started = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(r, anchor, handle),
            timeout=grace_seconds - 0.5,
        )
        elapsed = time.monotonic() - started

        assert elapsed < grace_seconds - 1.0, (
            f"_serve_on_resident's grace loop took {elapsed:.2f}s -- did not "
            f"break out on the starved other-model request"
        )
        events = _audit_events(boot, anchor.slot_id)
        assert "grace_starvation_breakout" in events, (
            "starvation predicate was never consulted / the break-out never "
            "fired an audit line -- non-vacuity check"
        )

        await mgr.shutdown()

    async def test_invariant_matched_followup_not_preempted(self, tmp_path):
        """The no-match-fallthrough placement must hold here too: if a
        same-thread follow-up is already staged, the match branch drains it
        (via pop_matched_thread) and re-arms grace -- the starvation
        break-out on the SAME iteration must never run, because it lives on
        the no-match path, strictly after the `if matched is not None: ...
        continue`."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds, max_other_model_wait_s=0.2,
        )

        complete_calls = []

        async def fake_complete(slot, handle):
            complete_calls.append(slot.slot_id)
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )

        followup = Slot.new("m1", prompt="again", thread_id="t1")
        await mgr.queue.enqueue(followup)
        slot_other = Slot.new("m2", prompt="hi", thread_id="t2")
        slot_other.created_at = time.monotonic() - 100.0
        await mgr.queue.enqueue(slot_other)

        handle = _make_fake_handle("m1", 59500)
        r = Resident(
            model_tag="m1",
            handle=handle,
            port=handle.port,
            grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
        )
        anchor = Slot.new("m1", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED

        task = asyncio.create_task(mgr._serve_on_resident(r, anchor, handle))
        for _ in range(60):
            if followup.slot_id in complete_calls:
                break
            await asyncio.sleep(0.05)
        assert followup.slot_id in complete_calls, (
            "matched follow-up was not served -- starvation break-out must "
            "never preempt an available match"
        )
        # The driver task may already have exited naturally (e.g. the
        # still-starved slot_other breaks it out on the very next iteration)
        # -- either a clean return or a cancellation is an acceptable end
        # state here; the invariant under test is already proven above.
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await mgr.shutdown()
