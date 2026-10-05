"""Cross-model waiter starves at VRAM gate because idle holder
is NOT a resident. The fix tears down the idle holder immediately when
VRAM is insufficient, so the waiter gets admitted instead of starving.

Test (i): Cross-model waiter behind the idle holder gets ADMITTED.
Test (ii): Negative control — same-model waiter still warm-inherits,
           holder is NOT torn down.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime(tmp_path, *, max_parallel_sidecars=2, grace_seconds=60):
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
        queue=QueueConfig(
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=grace_seconds,
            max_grace_extensions=50,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=False, rules=[]),
    )
    return boot, runtime


def _mocks(vram_full=False):
    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        if vram_full:
            return False, 0  # No VRAM available
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return dict(
        spawn_fn=lambda *a, **k: None,
        health_fn=fake_health,
        sigterm_fn=fake_sigterm,
        vram_fn=fake_vram,
        complete_fn=fake_complete,
    )


def _make_handle(model_tag="modela"):
    proc = MagicMock()
    proc.is_alive.return_value = True
    proc.pid = 12345
    return SidecarHandle(proc=proc, port=59700, model_tag=model_tag)


def _make_resident(model_tag="modela", state=ResidentState.IDLE_EVICTABLE):
    now = time.monotonic()
    handle = _make_handle(model_tag)
    return Resident(
        model_tag=model_tag,
        resident_key=model_tag,
        state=state,
        main_gpu=0,
        split_mode="none",
        reserved_need_mib=10000,
        last_active_monotonic=now,
        rank_client_meta={"ip": "10.0.0.1"},
        handle=handle,
        grace=GraceTimer(grace_seconds=60, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=0),
    )


@pytest.mark.asyncio
async def test_cross_model_waiter_admitted_with_idle_holder(tmp_path):
    """Cross-model waiter behind the idle holder gets ADMITTED instead of
    starving. The idle holder is torn down immediately when VRAM is
    insufficient, freeing space for the new model."""
    boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=2)
    mgr = TurbohaulManager(boot, runtime, **_mocks(vram_full=True))
    try:
        # Set up an idle holder for modela (different from the incoming modelb)
        handle_a = _make_handle("modela")
        mgr._set_idle_holder(handle_a, "modela", 0.0, "thread-A", 0, {"ip": "10.0.0.1"})

        # Incoming request for modelb (different model)
        slot_b = Slot.new("slot-B", prompt="hi", thread_id="thread-B",
                          client_meta={"ip": "10.0.0.2"})
        slot_b.model_tag = "modelb"

        # Mock _resolve_placement_locked to avoid manifest validation error
        def fake_resolve(tag):
            return (5000, 1, 0, "none", 0, True, None)

        # Mock _vram_admits_locked to always report insufficient VRAM
        def fake_vram_admits(*args, **k):
            return False

        with patch.object(mgr, '_resolve_placement_locked', side_effect=fake_resolve), \
             patch.object(mgr, '_vram_admits_locked', side_effect=fake_vram_admits):
            # Route the slot — should NOT starve, should tear down idle holder
            await mgr._route_or_reserve(slot_b)

        # The idle holder should have been torn down
        assert mgr._idle_handle is None, (
            "Idle holder should be torn down for cross-model request "
            "when VRAM is insufficient"
        )
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_same_model_waiter_warm_inherits_holder_preserved(tmp_path):
    """Negative control: same-model follow-up still warm-inherits and the
    idle holder is NOT torn down. The fix only triggers on insufficient
    VRAM, and same-model requests HIT the resident and bypass
    the VRAM gate entirely."""
    boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=2)
    mgr = TurbohaulManager(boot, runtime, **_mocks(vram_full=True))
    try:
        # Set up an idle holder for modela
        handle_a = _make_handle("modela")
        mgr._set_idle_holder(handle_a, "modela", 0.0, "thread-A", 0, {"ip": "10.0.0.1"})

        # Also set up a parked resident for modela (HIT path)
        resident_a = _make_resident("modela", state=ResidentState.IDLE_EVICTABLE)
        mgr._residents["modela"] = resident_a

        # Incoming request for the SAME model (modela)
        slot_a = Slot.new("slot-A", prompt="hi again", thread_id="thread-A",
                          client_meta={"ip": "10.0.0.1"})
        slot_a.model_tag = "modela"

        # Route the slot — should HIT the resident, NOT tear down holder
        await mgr._route_or_reserve(slot_a)

        # The idle holder should still be present (warm-inherit preserved)
        assert mgr._idle_handle is handle_a, (
            "Idle holder should NOT be torn down for same-model follow-up "
            "(warm-inherit must be preserved)"
        )
        # The resident should have been activated
        assert resident_a.state == ResidentState.ACTIVE, (
            f"Same-model request should HIT and activate resident, "
            f"got state={resident_a.state}"
        )
    finally:
        await mgr.shutdown()
