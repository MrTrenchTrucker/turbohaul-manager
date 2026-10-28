"""Builders shared by the dead-client take-path tests.

Nothing here decides an outcome: it only builds a manager over a fake engine, a
resident, and the safety-gate fakes the spawn path needs, so that a test can set
up a waiting request and then observe what the shipped code does with it.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import yaml

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
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.safety import GateResult
from turbohaul.slot import Slot, SlotState
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle


def boot_and_runtime(tmp_path, *, spawn_reclaim_wait_max_s=10.0):
    """A real boot/runtime config pair with the host safety gates switched on."""
    root = tmp_path / "state"
    root.mkdir()
    for sub in ("blobs", "manifests", "import-staging"):
        (root / sub).mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs",
            manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging",
            state_db_path=root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59600,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            max_parallel_sidecars=2,
            grace_seconds=0,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
            spawn_reclaim_wait_max_s=spawn_reclaim_wait_max_s,
        ),
        pull=PullConfig(),
    )
    # Initialise the state database once, so that the audit writes the code under
    # test fires in the background never race to create it.
    open_state_db(boot.storage.state_db_path).close()
    return boot, runtime


def seed_single_card_manifest(boot, model_tag, *, main_gpu=0):
    path = boot.storage.manifests_path / f"{model_tag}.yaml"
    path.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 4096,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "auto_place": False,
        "llama_server_flags": {"split_mode": "none", "main_gpu": main_gpu},
    }))


def manager_with_spawn_log(boot, runtime):
    """A manager whose engine spawn is a fake that records every call.

    Returns ``(manager, spawn_log)``; ``spawn_log`` holds one model tag per engine
    started.
    """
    spawn_log: list[str] = []
    pid = [81000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        spawn_log.append(model_tag)
        pid[0] += 1
        proc = MagicMock()
        proc.pid = pid[0]
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
        sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
    )
    return mgr, spawn_log


def loading_resident(model_tag, *, main_gpu=0):
    return Resident(
        resident_key=model_tag,
        model_tag=model_tag,
        state=ResidentState.RESERVED_LOADING,
        main_gpu=main_gpu,
        split_mode="none",
        reserved_need_mib=0,
        port=59600,
    )


def waiting_slot(model_tag, *, name, disconnect_event=None):
    slot = Slot(
        slot_id=name,
        model_tag=model_tag,
        state=SlotState.LOADING,
        thread_id=f"thread-{name}",
        created_at=time.monotonic(),
    )
    slot.completion_future = asyncio.get_running_loop().create_future()
    slot.disconnect_event = disconnect_event
    return slot


def vram_refusal():
    return GateResult(
        name="vram", ok=False, detail="refused", blocked_on="vram",
        required_mib=9000, available_mib=1000,
    )


def vram_clear():
    return GateResult("vram", True, "ok")
