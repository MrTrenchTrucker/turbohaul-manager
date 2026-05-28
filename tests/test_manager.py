"""Tests for TurbohaulManager (mocked subprocess/GPU; foundations only - worker_loop in )."""
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
from turbohaul.manager import TurbohaulManager
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
        slot = await mgr.submit(model_tag="qwen", prompt="hello world")
        assert slot.slot_id.startswith("slot-")
        assert slot.thread_id.startswith("auto-")
        assert slot.model_tag == "qwen"
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
        slot = await mgr.submit(model_tag="qwen", prompt="hi", thread_id="custom-thread")
        assert slot.thread_id == "custom-thread"

    async def test_submit_with_grace_match_enqueues_head(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Prime: fill queue
        s_first = await mgr.submit(model_tag="qwen", prompt="prior", thread_id="thr-x")
        s_other = await mgr.submit(model_tag="qwen", prompt="other")
        # Manually start grace timer to simulate active slot popped + in grace
        mgr.grace.start("thr-x", "qwen")
        # Now follow-up should land at head
        s_followup = await mgr.submit(model_tag="qwen", prompt="next-turn", thread_id="thr-x")
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
        mgr.grace.start("thr-abc12345", "qwen3.6-35b-moe")
        snap = mgr.status_snapshot()
        assert snap["grace"] is not None
        assert snap["grace"]["model_tag"] == "qwen3.6-35b-moe"
        # Redaction: only first 8 chars exposed
        assert snap["grace"]["thread_id_prefix"] == "thr-abc1"

    def test_idle_state_reflected(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr.idle.start("qwen-coder")
        snap = mgr.status_snapshot()
        assert snap["idle_hot"] is not None
        assert snap["idle_hot"]["model_tag"] == "qwen-coder"


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
    """NEMO V2 2.1: shutdown must fail pending completion_futures."""

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

