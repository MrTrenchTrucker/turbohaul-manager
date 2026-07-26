"""Tests for TurbohaulManager (mocked subprocess/GPU; foundations only - worker_loop covered separately)."""
import asyncio
from pathlib import Path

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
from turbohaul.manager import (
    TurbohaulManager,
    Resident,
    _SINGLETON_RESIDENT_KEY,
)
from turbohaul.slot import SlotState
from turbohaul.state import open_state_db


@pytest.fixture
def boot_and_runtime(tmp_path):
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
            llama_server_binary=tmp_path / "fake_llama_server",  # nonexistent but unused in tests
            default_port_base=59500,  # nothing on this range
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    return boot, runtime


class TestConstructor:
    def test_init_wires_subsystems(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        assert mgr.queue is not None
        assert mgr.grace.grace_seconds == runtime.queue.grace_seconds
        assert mgr.idle.idle_seconds == runtime.queue.idle_hot_load_seconds
        assert mgr._active_slot is None
        assert mgr._active_handle is None

    def test_cold_start_idle_client_meta_initialized(self, boot_and_runtime):
        """Cold-start safety: _idle_client_meta must be initialized to None.
        
        Caught in early smoke test: first request after fresh manager startup would
        hit AttributeError when reading _idle_client_meta at line 3643, causing the
        request to be dropped. Root cause: __init__ read and wrote the attribute via
        _set_idle_holder() but never initialized it.
        
        Fix: Initialize to None alongside other _idle_* fields (lines 903-917).
        """
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Critical: attribute exists and is None (cold-start state, no idle holder yet)
        assert hasattr(mgr, "_idle_client_meta"), "Missing initialization in __init__"
        assert mgr._idle_client_meta is None, "Should be None on cold start"


class TestAllocPort:
    """P1a: resident-registry-aware port allocation (max=1 identical to base)."""

    def test_returns_base_when_no_port_held(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        base = boot.runtime.default_port_base
        # Phase-0 singleton resident exists but its .port is the placeholder
        # (None) -> the window is empty of holds -> base is returned verbatim,
        # identical to the deployed hard-coded default_port_base.
        assert mgr._residents[_SINGLETON_RESIDENT_KEY].port is None
        assert mgr._alloc_port() == base

    def test_skips_a_held_port(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        base = boot.runtime.default_port_base
        # A second (Phase-1-style) resident holding the base port forces the
        # allocator to skip it and return the next free port in the window.
        mgr._residents["held-base"] = Resident(model_tag="held", port=base)
        assert mgr._alloc_port() == base + 1

    def test_skips_lower_held_returns_lowest_free(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        base = boot.runtime.default_port_base
        # Hold base+1 but leave base free -> lowest free is still base.
        mgr._residents["held-base1"] = Resident(model_tag="held", port=base + 1)
        assert mgr._alloc_port() == base
        # Now also hold base -> lowest free climbs to base+2.
        mgr._residents["held-base"] = Resident(model_tag="held2", port=base)
        assert mgr._alloc_port() == base + 2


class TestActiveSpawnSeq:
    """P1a: per-resident spawn_seq mirrors the global at max=1 (identical)."""

    def test_active_spawn_seq_mirrors_global_after_bump(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Fresh manager: both the global and the active resident start at 0.
        assert mgr._spawn_seq == 0
        assert mgr._active_spawn_seq() == 0
        # _bump_spawn_seq advances the global AND mirrors to the active resident
        # in lock-step, so the live-monitor generation_id input is unchanged.
        mgr._bump_spawn_seq()
        assert mgr._spawn_seq == 1
        assert mgr._residents[_SINGLETON_RESIDENT_KEY].spawn_seq == 1
        assert mgr._active_spawn_seq() == 1
        mgr._bump_spawn_seq()
        assert mgr._active_spawn_seq() == mgr._spawn_seq == 2


class TestBootReconcile:
    def test_boot_reconcile_returns_summary(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Boot reconcile uses /proc + nvidia-smi which are present on Linux server
        result = mgr.boot_reconcile()
        assert "orphans_reaped" in result
        assert "foreign_gpu_apps" in result
        assert "slots_reconciled_to_cold" in result

    def test_boot_reconcile_marks_stale_slots_cold(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Pre-populate state.sqlite with a fake-active slot whose pid is dead
        from turbohaul.state import upsert_slot
        conn = open_state_db(boot.storage.state_db_path)
        upsert_slot(
            conn,
            {
                "slot_id": "stale-1",
                "model_tag": "m",
                "state": "ACTIVE",
                "pid": 999_999_999,  # never alive
            },
        )
        conn.close()

        result = mgr.boot_reconcile()
        assert result["slots_reconciled_to_cold"] >= 1

        # Verify slot now COLD
        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute("SELECT state, end_reason FROM slots WHERE slot_id='stale-1'")
        row = cur.fetchone()
        assert row["state"] == "COLD"
        assert "boot-reconcile" in row["end_reason"]
        conn.close()

    def test_boot_reconcile_with_injected_pid_check(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Inject custom pid-alive function that says nothing is alive
        result = mgr.boot_reconcile(pid_is_alive_fn=lambda pid: False)
        assert "slots_reconciled_to_cold" in result


class TestVerifyBinary:
    def test_empty_sha_skips(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # boot.runtime.llama_server_binary_sha256 = "" by default → skip-OK
        assert mgr.verify_binary() is True

    def test_wrong_sha_fails(self, tmp_path):
        bin_path = tmp_path / "fake"
        bin_path.write_bytes(b"x" * 100)
        boot = BootConfig(
            server=ServerConfig(),
            storage=StorageConfig(
                blob_store_path=tmp_path / "b",
                manifests_path=tmp_path / "m",
                import_allowed_root=tmp_path / "i",
                state_db_path=tmp_path / "s.sqlite",
            ),
            runtime=RuntimePathsConfig(
                llama_server_binary=bin_path,
                llama_server_binary_sha256="deadbeef" * 8,  # wrong
                default_port_base=59500,
            ),
            ui=UIConfig(static_path=tmp_path / "ui"),
        )
        runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
        mgr = TurbohaulManager(boot, runtime)
        assert mgr.verify_binary() is False


@pytest.mark.asyncio
class TestSubmit:
    async def test_submit_returns_slot_with_auto_thread_id(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        slot = await mgr.submit(model_tag="demo-model", prompt="hello world")
        assert slot.slot_id.startswith("slot-")
        assert slot.thread_id.startswith("auto-")
        assert slot.model_tag == "demo-model"
        # Audit-logged
        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute("SELECT state FROM slots WHERE slot_id=?", (slot.slot_id,))
        row = cur.fetchone()
        assert row is not None
        assert row["state"] in ("STAGED", "ACCEPT_BUFFER")
        conn.close()

    async def test_submit_preserves_explicit_thread_id(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        slot = await mgr.submit(model_tag="demo-model", prompt="hi", thread_id="custom-thread")
        assert slot.thread_id == "custom-thread"

    async def test_submit_with_grace_match_enqueues_head(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Prime: fill queue
        s_first = await mgr.submit(model_tag="demo-model", prompt="prior", thread_id="thr-x")
        s_other = await mgr.submit(model_tag="demo-model", prompt="other")
        # Manually start grace timer to simulate active slot popped + in grace
        mgr.grace.start("thr-x", "demo-model")
        # Now follow-up should land at head
        s_followup = await mgr.submit(model_tag="demo-model", prompt="next-turn", thread_id="thr-x")
        # Pop from queue → should be follow-up first (head)
        popped = await mgr.queue.pop_next()
        assert popped.slot_id == s_followup.slot_id

    async def test_submit_audit_logs(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        slot = await mgr.submit(model_tag="m", prompt="hi")
        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute(
            "SELECT event_type FROM audit_events WHERE slot_id=?", (slot.slot_id,)
        )
        events = [row["event_type"] for row in cur.fetchall()]
        assert "submit" in events
        conn.close()


class TestStatusSnapshot:
    def test_empty_status(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        snap = mgr.status_snapshot()
        assert snap["queue"]["acceptance_buffer_depth"] == 0
        assert snap["queue"]["staging_queue_depth"] == 0
        assert snap["active"] is None
        assert snap["grace"] is None
        assert snap["idle_hot"] is None
        assert snap["parallel_slots"]["used"] == 0
        assert snap["parallel_slots"]["max"] == runtime.queue.max_parallel_sidecars

    def test_grace_state_reflected(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr.grace.start("thr-abc12345", "example-35b-moe")
        snap = mgr.status_snapshot()
        assert snap["grace"] is not None
        assert snap["grace"]["model_tag"] == "example-35b-moe"
        # Redaction: only first 8 chars exposed
        assert snap["grace"]["thread_id_prefix"] == "thr-abc1"

    def test_idle_state_reflected(self, boot_and_runtime):
        """P12 idle-status fix: legacy IdleHotTimer alone must NOT surface as
        idle_hot in /status. Only a real idle holder (_idle_handle + unexpired
        _idle_expires_at) reports idle_hot."""
        import time
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Legacy timer started but no real holder -> idle_hot must be None.
        mgr.idle.start("example-coder")
        snap = mgr.status_snapshot()
        assert snap["idle_hot"] is None, (
            "legacy IdleHotTimer must not surface as idle_hot without a real holder"
        )
        # With a real holder, idle_hot is reported.
        mgr._idle_handle = object()
        mgr._idle_expires_at = time.monotonic() + 300
        mgr._idle_model_tag = "example-coder"
        snap = mgr.status_snapshot()
        assert snap["idle_hot"] is not None
        assert snap["idle_hot"]["model_tag"] == "example-coder"


@pytest.mark.asyncio
class TestShutdown:
    async def test_shutdown_closes_queue(self, boot_and_runtime):
        from turbohaul.queue import QueueClosed

        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        await mgr.submit(model_tag="m", prompt="hi")
        await mgr.shutdown()
        # After shutdown queue should refuse new submits
        with pytest.raises(QueueClosed):
            await mgr.submit(model_tag="m", prompt="another")

    async def test_shutdown_cancels_worker_task(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Start worker loop as a task
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.sleep(0.05)
        await mgr.shutdown()
        # worker task should be done
        assert mgr._worker_task.done() or mgr._worker_task.cancelled()


@pytest.mark.asyncio
class TestWorkerLoopSkeleton:
    async def test_worker_loop_consumes_queue(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        slot = await mgr.submit(model_tag="m", prompt="hi")
        # Run worker loop briefly
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.sleep(0.2)
        await mgr.shutdown()
        # Slot should have been consumed + marked COLD by skeleton
        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute("SELECT state FROM slots WHERE slot_id=?", (slot.slot_id,))
        row = cur.fetchone()
        assert row is not None
        assert row["state"] == "COLD"
        conn.close()


import pytest
from turbohaul.manager import TurbohaulManager
from turbohaul.config import BootConfig, ServerConfig, StorageConfig, RuntimePathsConfig, UIConfig, RuntimeConfig, QueueConfig, PullConfig


@pytest.mark.asyncio
class TestShutdownFailsPending:
    """Shutdown must fail pending completion_futures."""

    async def test_shutdown_fails_staged_slot_completion_futures(self, tmp_path):
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
                default_port_base=59700,
            ),
            ui=UIConfig(static_path=tmp_path / "ui_dist"),
        )
        runtime = RuntimeConfig(
            queue=QueueConfig(grace_seconds=0, idle_hot_load_seconds=0),
            pull=PullConfig(),
        )
        mgr = TurbohaulManager(boot, runtime)
        # Submit a slot via submit_and_wait WITHOUT starting worker_loop.
        # The slot sits in staging with an unresolved completion_future.
        # Wrap in a task so we can await shutdown concurrently.
        caller_task = asyncio.create_task(
            mgr.submit_and_wait("anymodel", "hi")
        )
        # Give the submit_and_wait task time to register the future
        await asyncio.sleep(0.05)
        assert not caller_task.done(), (
            "caller_task should be blocked on completion_future before shutdown"
        )
        # Shutdown should fail the pending future, not let the caller hang.
        await mgr.shutdown()
        # Caller now completes (with exception)
        with pytest.raises((asyncio.CancelledError, RuntimeError)):
            await asyncio.wait_for(caller_task, timeout=2.0)


class TestIdleWindowSeconds:
    """Per-resident idle window: _idle_window_seconds helper."""

    def test_none_keep_alive_uses_default(self, boot_and_runtime):
        """When keep_alive is None, the default (per-model or global) is used."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # With keep_alive=None, default=300 -> returns 300
        assert mgr._idle_window_seconds(None, 300) == 300

    def test_negative_keep_alive_pins(self, boot_and_runtime):
        """keep_alive < 0 -> pin-warm (KEEP_ALIVE_MAX_S)."""
        from turbohaul.config import KEEP_ALIVE_MAX_S
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._idle_window_seconds(-1, 120) == KEEP_ALIVE_MAX_S

    def test_positive_keep_alive_caps_at_max(self, boot_and_runtime):
        """keep_alive > KEEP_ALIVE_MAX_S is capped."""
        from turbohaul.config import KEEP_ALIVE_MAX_S
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._idle_window_seconds(999_999, 120) == KEEP_ALIVE_MAX_S

    def test_zero_keep_alive_disables_idle(self, boot_and_runtime):
        """keep_alive=0 -> idle_window=0 (unload immediately)."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._idle_window_seconds(0, 300) == 0


class TestPerModelIdleTimeout:
    """Per-model sleep_idle_seconds from manifest.

    A 35B model was evicting after 120s because the global
    idle_hot_load_seconds was used for everything. Each model's manifest
    sleep_idle_seconds should govern its own idle timeout.
    """

    def test_manifest_shorter_than_global_evicts_at_manifest(self, boot_and_runtime):
        """Model with sleep_idle_seconds=60 evicts at 60s, not the global 120s."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Simulate a resident with a short per-model timeout
        r = Resident(model_tag="short-idle", sleep_idle_seconds=60)
        # The driver resolves: 60 > 0 -> per_model_idle = 60
        per_model_idle = (
            mgr.runtime.queue.idle_hot_load_seconds
            if r.sleep_idle_seconds == 0
            else r.sleep_idle_seconds
        )
        idle_window = mgr._idle_window_seconds(None, per_model_idle)
        assert idle_window == 60, "Should use manifest 60s, not global default"

    def test_manifest_longer_than_global_uses_manifest(self, boot_and_runtime):
        """Model with sleep_idle_seconds=600 stays warm 600s, not global 120s."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        r = Resident(model_tag="long-idle", sleep_idle_seconds=600)
        per_model_idle = (
            mgr.runtime.queue.idle_hot_load_seconds
            if r.sleep_idle_seconds == 0
            else r.sleep_idle_seconds
        )
        idle_window = mgr._idle_window_seconds(None, per_model_idle)
        assert idle_window == 600, "Should use manifest 600s, not global 120s"

    def test_manifest_pin_negative_ones_stays_warm(self, boot_and_runtime):
        """Model with sleep_idle_seconds=-1 is pinned (KEEP_ALIVE_MAX_S)."""
        from turbohaul.config import KEEP_ALIVE_MAX_S
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        r = Resident(model_tag="pinned", sleep_idle_seconds=-1)
        # -1 resolves to KEEP_ALIVE_MAX_S
        per_model_idle = (
            KEEP_ALIVE_MAX_S
            if r.sleep_idle_seconds == -1
            else (
                mgr.runtime.queue.idle_hot_load_seconds
                if r.sleep_idle_seconds == 0
                else r.sleep_idle_seconds
            )
        )
        idle_window = mgr._idle_window_seconds(None, per_model_idle)
        assert idle_window == KEEP_ALIVE_MAX_S, "Pinned model should stay warm"

    def test_manifest_zero_falls_back_to_global(self, boot_and_runtime):
        """Model with sleep_idle_seconds=0 (unset) uses global default."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        r = Resident(model_tag="default-idle", sleep_idle_seconds=0)
        # 0 means "use global default"
        per_model_idle = mgr.runtime.queue.idle_hot_load_seconds
        idle_window = mgr._idle_window_seconds(None, per_model_idle)
        assert idle_window == mgr.runtime.queue.idle_hot_load_seconds

    def test_request_keep_alive_overrides_manifest(self, boot_and_runtime):
        """A request's keep_alive_s overrides the manifest default.
        
        If the manifest says 60s but the client sends keep_alive=300,
        the 300s wins for that request cycle.
        """
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        r = Resident(model_tag="short-idle", sleep_idle_seconds=60)
        per_model_idle = r.sleep_idle_seconds  # 60s from manifest
        # Client sends keep_alive=300 -> should use 300, not manifest 60
        idle_window = mgr._idle_window_seconds(300, per_model_idle)
        assert idle_window == 300, "Request keep_alive should override manifest"

    def test_stale_keep_alive_no_leak_after_manifest_reset(self, boot_and_runtime):
        """B-after-A: B uses the manifest default, not A's stale keep_alive.
        
        Request A sends keep_alive=300, request B sends no keep_alive.
        B should use the manifest's sleep_idle_seconds, not A's 300.
        """
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        r = Resident(model_tag="short-idle", sleep_idle_seconds=60)
        per_model_idle = r.sleep_idle_seconds  # 60s from manifest
        # Request A: keep_alive=300 -> idle_window = 300
        idle_a = mgr._idle_window_seconds(300, per_model_idle)
        assert idle_a == 300
        # Request B: no keep_alive (None) -> idle_window = manifest 60, NOT 300
        idle_b = mgr._idle_window_seconds(None, per_model_idle)
        assert idle_b == 60, "B should use manifest 60s, not A's stale 300s"
