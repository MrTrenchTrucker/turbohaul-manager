"""Gate removal for parallel>=2 idle teardown.

Goal: ONE test that proves a parallel>=2 resident at idle
teardown POPULATES idle_client_meta and idle_admission_ctx_len from
grace_tip rather than leaving them None/0.

RED: re-add the `if int(getattr(r, "parallel", 1) or 1) == 1:` gate →
     parallel>=2 resident's idle_client_meta stays None (verified).
GREEN: gate removed → idle_client_meta populated from grace_tip.
Control: parallel==1 must pass on BOTH arms (gate never blocked it).
"""
import asyncio
import time

import pytest
from unittest.mock import MagicMock, patch


async def _wait_until(predicate, *, timeout=5.0, interval=0.01):
    """Wait for predicate to return True."""
    t0 = time.monotonic()
    while True:
        if predicate():
            return
        if time.monotonic() - t0 > timeout:
            raise AssertionError(f"predicate never became true within {timeout}s")
        await asyncio.sleep(interval)


def _boot_runtime_parallel2(tmp_path):
    """Minimal boot+runtime for driving _drive_resident directly."""
    from turbohaul.config import (
        BootConfig, PullConfig, QueueConfig, RuntimeConfig,
        RuntimePathsConfig, ServerConfig, StorageConfig, UIConfig,
    )
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
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, max_parallel_sidecars=2,
            grace_seconds=2, max_grace_extensions=50,
            idle_hot_load_seconds=0,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag, *, main_gpu=0):
    import yaml
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 0,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "llama_server_flags": {"split_mode": "none", "main_gpu": main_gpu},
    }))


def _make_fakes_parallel2():
    """Fake sidecar fakes for driving _drive_resident without a real engine."""
    from turbohaul.subprocess_mgr import SidecarHandle

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        proc = MagicMock()
        proc.pid = 99001
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return fake_spawn, fake_health, fake_sigterm, fake_vram, fake_complete


@pytest.mark.asyncio
async def test_parallel2_idle_teardown_populates_idle_client_meta(tmp_path, monkeypatch):
    """With the parallel==1 gate REMOVED, a parallel>=2 resident
    at idle teardown (idle_hot_load_seconds=0 -> immediate eviction) must
    populate r.idle_client_meta and r.idle_admission_ctx_len.

    RED (gate re-added): both stay None/0 — the stash never fires.
    GREEN (gate removed): both are populated from grace_tip.

    Also includes a parallel==1 control that must pass on BOTH arms
    (the gate never blocked parallel==1).
    """
    from turbohaul.manager import Resident, ResidentState, TurbohaulManager
    from turbohaul.queue import GraceTimer, IdleHotTimer
    from turbohaul.slot import Slot, SlotState

    boot, runtime = _boot_runtime_parallel2(tmp_path)
    _seed_manifest(boot, "qwen3.6-27b-q4")

    fake_spawn, fake_health, fake_sigterm, fake_vram, fake_complete = _make_fakes_parallel2()
    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
        sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
    )

    # Mock VRAM admission (high free) so spawn succeeds
    with patch("turbohaul.manager._read_free_vram_all_mib", return_value=[80000, 80000]):

        # Stub out _serve_on_resident so the slot is "served" immediately
        async def fake_serve(r, slot, handle):
            return None
        monkeypatch.setattr(mgr, "_serve_on_resident", fake_serve)

        # Stub out _begin_unload_locked so it doesn't try to actually kill
        async def fake_begin_evict(r):
            r.state = ResidentState.DEAD
            r.handle = None
            r.idle_handle = None
        monkeypatch.setattr(mgr, "_begin_unload_locked", fake_begin_evict)

        client_meta = {"ip": "1.2.3.4", "messages": [{"role": "user", "content": "hi"}]}

        # --- parallel >= 2 resident: should populate idle_client_meta ---
        anchor_slot = Slot.new(
            "qwen3.6-27b-q4", prompt="hello", thread_id="t-parallel2",
            admission_ctx_len=50, client_meta=client_meta,
        )
        anchor_slot.state = SlotState.STAGED
        resident_p2 = Resident(
            model_tag="qwen3.6-27b-q4",
            resident_key="qwen3.6-27b-q4",
            handle=None, port=59901,
            grace=GraceTimer(grace_seconds=2, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
            inbox=asyncio.Queue(),
            rank_client_meta=client_meta,
            last_active_monotonic=1.0,
            parallel=2,
            reserved_need_mib=0,
        )
        # Newer code has grace_tip as a Resident field; older code
        # does not — the code reads from `slot` directly. Use hasattr for portability.
        if hasattr(resident_p2, "grace_tip"):
            setattr(resident_p2, "grace_tip", anchor_slot)  # type: ignore[attr-defined]
        assert resident_p2.inbox is not None  # type assertion
        resident_p2.inbox.put_nowait(anchor_slot)
        mgr._residents["qwen3.6-27b-q4"] = resident_p2

        task = asyncio.create_task(mgr._drive_resident(resident_p2))
        await _wait_until(lambda: resident_p2.state is ResidentState.DEAD, timeout=5.0)
        task.cancel()

        # GREEN: idle_client_meta must be populated (gate removed)
        assert resident_p2.idle_client_meta is not None, \
            "parallel>=2 resident's idle_client_meta was NOT populated — " \
            "the parallel==1 gate is still blocking the stash (RED)."
        assert resident_p2.idle_admission_ctx_len > 0, \
            "parallel>=2 resident's idle_admission_ctx_len was 0 — gate still blocking (RED)."

        # --- parallel == 1 control: must pass on BOTH arms ---
        anchor_slot1 = Slot.new(
            "qwen3.6-27b-q4", prompt="hello", thread_id="t-parallel1",
            admission_ctx_len=50, client_meta=client_meta,
        )
        anchor_slot1.state = SlotState.STAGED
        resident_p1 = Resident(
            model_tag="qwen3.6-27b-q4-p1",
            resident_key="qwen3.6-27b-q4-p1",
            handle=None, port=59902,
            grace=GraceTimer(grace_seconds=2, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
            inbox=asyncio.Queue(),
            rank_client_meta=client_meta,
            last_active_monotonic=1.0,
            parallel=1,
            reserved_need_mib=0,
        )
        if hasattr(resident_p1, "grace_tip"):
            setattr(resident_p1, "grace_tip", anchor_slot1)  # type: ignore[attr-defined]
        assert resident_p1.inbox is not None  # type assertion
        resident_p1.inbox.put_nowait(anchor_slot1)
        mgr._residents["qwen3.6-27b-q4-p1"] = resident_p1

        task1 = asyncio.create_task(mgr._drive_resident(resident_p1))
        await _wait_until(lambda: resident_p1.state is ResidentState.DEAD, timeout=5.0)
        task1.cancel()

        # Control must pass on both patched and unpatched (parallel==1 was always allowed)
        assert resident_p1.idle_client_meta is not None, \
            "parallel==1 resident's idle_client_meta should be populated (control)."
        assert resident_p1.idle_admission_ctx_len > 0, \
            "parallel==1 resident's idle_admission_ctx_len should be >0 (control)."
