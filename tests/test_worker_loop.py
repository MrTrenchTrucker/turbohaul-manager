"""Integration tests for the full worker_loop FSM cycle (Phase 2 Wave 7).

Uses DI to inject mocks for spawn / health / sigterm / vram / complete so no real
llama-server is spawned. Phase 6 smoke E2E uses the real backend.
"""
import asyncio
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
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import SlotState
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime(tmp_path, grace_seconds=0, idle_hot_load_seconds=0):
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
        queue=QueueConfig(
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=idle_hot_load_seconds,
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


@pytest.mark.asyncio
class TestWorkerLoopFullCycle:
    async def test_full_cycle_happy_path(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=0)
        spawn_calls = []
        sigterm_calls = []
        vram_calls = []
        complete_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv):
            spawn_calls.append({"model_tag": model_tag, "port": port, "argv": argv})
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, *, drained_window_s, is_active, **kwargs):
            sigterm_calls.append({"model_tag": handle.model_tag, "is_active": is_active})
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            vram_calls.append(kwargs)
            return True, 100

        async def fake_complete(slot, handle):
            complete_calls.append(slot.slot_id)

        mgr = TurbohaulManager(
            boot,
            runtime,
            spawn_fn=fake_spawn,
            health_fn=fake_health,
            sigterm_fn=fake_sigterm,
            vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        slot = await mgr.submit(model_tag="m1", prompt="hi")
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        # Allow enough time for: pop → load → active → complete → grace(0s) → pop → idle
        await asyncio.sleep(0.5)
        await mgr.shutdown()

        assert len(spawn_calls) == 1
        assert spawn_calls[0]["model_tag"] == "m1"
        assert spawn_calls[0]["port"] == boot.runtime.default_port_base
        assert len(complete_calls) == 1
        assert complete_calls[0] == slot.slot_id
        assert len(sigterm_calls) == 1
        assert len(vram_calls) == 1

        # Verify slot ended COLD via teardown
        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute(
            "SELECT state, end_reason FROM slots WHERE slot_id=?", (slot.slot_id,)
        )
        row = cur.fetchone()
        assert row["state"] == "COLD"
        assert "grace-expired" in row["end_reason"]
        conn.close()

    async def test_full_cycle_records_fsm_transitions(self, tmp_path):
        """Verify the FSM transition events land in audit_events table."""
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=0)

        def fake_spawn(binary, gguf, port, model_tag, argv):
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            return True, 100

        async def fake_complete(slot, handle):
            pass

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        slot = await mgr.submit(model_tag="m1", prompt="hi")
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.sleep(0.5)
        await mgr.shutdown()

        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute(
            "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
            (slot.slot_id,),
        )
        events = [r["event_type"] for r in cur.fetchall()]
        conn.close()

        assert "submit" in events
        assert "stage_to_loading" in events
        assert "active" in events
        assert "grace_enter" in events
        assert "teardown" in events
        assert "idle_hot_enter" in events

    async def test_loading_fail_health_timeout_pops(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=0)

        def fake_spawn(binary, gguf, port, model_tag, argv):
            return _make_fake_handle(model_tag, port)

        async def fake_health_timeout(port, timeout_s, **kwargs):
            return False  # never healthy

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            return True, 100

        async def fake_complete(slot, handle):
            pass  # never reached

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health_timeout,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        slot = await mgr.submit(model_tag="m1", prompt="hi")
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.sleep(0.5)
        await mgr.shutdown()

        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute(
            "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
            (slot.slot_id,),
        )
        events = [r["event_type"] for r in cur.fetchall()]
        assert "loading_fail_health_timeout" in events

        cur2 = conn.execute("SELECT state, end_reason FROM slots WHERE slot_id=?", (slot.slot_id,))
        row = cur2.fetchone()
        assert row["state"] == "COLD"
        assert "loading-fail" in row["end_reason"]
        conn.close()

    async def test_two_slots_processed_sequentially(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=0)
        spawn_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv):
            spawn_calls.append(model_tag)
            return _make_fake_handle(model_tag, port)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(*a, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            await asyncio.sleep(0.01)

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        s1 = await mgr.submit(model_tag="m1", prompt="first")
        s2 = await mgr.submit(model_tag="m2", prompt="second")
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.sleep(0.8)
        await mgr.shutdown()

        assert spawn_calls == ["m1", "m2"]  # FIFO order

    async def test_worker_loop_exits_on_shutdown(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        mgr = TurbohaulManager(boot, runtime)
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.sleep(0.1)
        await mgr.shutdown()
        assert mgr._worker_task.done() or mgr._worker_task.cancelled()
