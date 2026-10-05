"""Load-time visibility + per-Resident config refresh.

(a) manager.py's /status elapsed_s computation reads
`slot.started_loading_at` via getattr — a phantom field, referenced nowhere
else in the package and never assigned, so elapsed_s stayed 0.0 forever
(confirmed empirically against the real unmodified base: 0.0 even after a
real 300ms sleep, not an AttributeError). Fix: Slot gains the field, stamped
at the REAL stage_to_loading call site inside `_process_slot` (there are two
`_audit_async(slot, "stage_to_loading")` calls in the file; the other is a
manifest-not-found bail that dies within 3 lines and never really loads).

(b) `_process_slot` (the cap<=1, default path) uses `self.grace`,
which PUT /api/config already refreshes. The SEPARATE cap>=2 multi-slot
dispatcher path (`_drive_resident`/`_serve_on_resident`) uses each
Resident's OWN `r.grace`/`r.idle`, snapshotted once at Resident construction
and never refreshed — confirmed these are two genuinely distinct code paths
("the cap<=1 path above is UNTOUCHED (byte-identical)",
manager.py's own comment). This fix is therefore only reachable with
max_parallel_sidecars>=2 configured; tests below use that config explicitly,
matching test_multislot_concurrency.py's own convention.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this file> -v
"""
from __future__ import annotations

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
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle


# --- shared fixtures / helpers --------------------------------------------------

def _boot_runtime(tmp_path, *, max_parallel_sidecars=1, grace_seconds=30,
                   idle_hot_load_seconds=600, max_grace_extensions=5):
    storage_root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (storage_root / sub).mkdir(parents=True)
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
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=idle_hot_load_seconds,
            max_grace_extensions=max_grace_extensions,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _fake_handle(model_tag: str, port: int, pid: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _seed_manifest(boot, model_tag, *, split_mode="none", main_gpu=0):
    import yaml
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 0,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


def _mocks(spawn_delay_s=0.0, complete_capture=None):
    pid = [90000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        if spawn_delay_s:
            await asyncio.sleep(spawn_delay_s)
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        if complete_capture is not None:
            complete_capture.append(slot)
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


def _mk(boot, runtime, **mocks):
    mgr = TurbohaulManager(boot, runtime, **mocks)
    mgr.runtime.queue.safety_enabled = False
    return mgr


# ================================================================================
# (a) — Consumer correctness: once started_loading_at
# is set, elapsed_s reflects real time and increases monotonically.
# ================================================================================

def test_elapsed_s_nonzero_and_increasing_once_stamped(tmp_path):
    boot, runtime = _boot_runtime(tmp_path)
    mgr = _mk(boot, runtime, **_mocks())

    slot = Slot.new("qwen3.6-27b", thread_id="t1")
    slot.state = SlotState.LOADING
    slot.started_loading_at = time.monotonic()
    mgr._active_slot = slot

    snap1 = mgr.status_snapshot()
    time.sleep(0.2)
    snap2 = mgr.status_snapshot()

    e1 = snap1["loading"]["elapsed_s"]
    e2 = snap2["loading"]["elapsed_s"]
    assert e1 >= 0.0
    assert e2 > e1, f"elapsed_s did not increase: {e1} -> {e2}"
    assert e2 >= 0.15, f"elapsed_s should reflect ~0.2s real sleep, got {e2}"


def test_elapsed_s_stays_zero_without_the_stamp(tmp_path):
    """Sanity control: WITHOUT started_loading_at set, elapsed_s is 0.0 even
    after real time passes -- this is exactly base's behavior, confirmed
    empirically (not AttributeError) before writing this test."""
    boot, runtime = _boot_runtime(tmp_path)
    mgr = _mk(boot, runtime, **_mocks())

    slot = Slot.new("qwen3.6-27b", thread_id="t1")
    slot.state = SlotState.LOADING
    # started_loading_at left at its dataclass default (0.0) -- getattr(...)
    # would find a real attribute now (unlike true base), but 0.0 itself is
    # falsy, so the `or getattr(slot, "received_at", None)` fallback still
    # applies and `started` still resolves the same way base's missing-
    # attribute None does. Documenting the boundary, not just assuming it.
    mgr._active_slot = slot
    time.sleep(0.2)
    snap = mgr.status_snapshot()
    assert snap["loading"]["elapsed_s"] == 0.0


# ================================================================================
# (a) — producer/wiring correctness AND the actual acceptance-criterion
# observable: the REAL stage_to_loading call site (inside _process_slot, the
# cap<=1 default path) makes elapsed_s non-zero WHILE the slot is
# still loading, measured via status_snapshot -- an observable that exists
# on BOTH arms (unlike probing slot.started_loading_at directly, which
# would AttributeError on base for the same reason a new-kwarg TypeError
# would: proving the attribute is absent, not that the behavior differs).
# ================================================================================

@pytest.mark.asyncio
async def test_process_slot_makes_elapsed_s_nonzero_while_loading(tmp_path):
    boot, runtime = _boot_runtime(tmp_path)
    _seed_manifest(boot, "m1")
    mgr = _mk(boot, runtime, **_mocks(spawn_delay_s=0.3))

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        submit_task = asyncio.create_task(
            mgr.submit_and_wait("m1", "hello", thread_id="t1")
        )
        await asyncio.sleep(0.15)  # mid-flight: fake_health is still sleeping

        snap = mgr.status_snapshot()
        loading = snap.get("loading")
        assert loading is not None, "expected the slot to still be in a loading state"
        assert loading["elapsed_s"] > 0.0, (
            "elapsed_s was still 0.0 partway through a real load -- this is "
            "exactly the observable base always produces, as seen in a real "
            "long stall (elapsed_s stayed 0.0 throughout)"
        )

        await asyncio.wait_for(submit_task, timeout=5)
    finally:
        await mgr.shutdown()


# ================================================================================
# (b) — Only reachable via the SEPARATE cap>=2
# multi-slot dispatcher path (_drive_resident/_serve_on_resident) -- the
# cap<=1 path never constructs a Resident/GraceTimer pair at all.
# ================================================================================

@pytest.mark.asyncio
async def test_resident_timers_refresh_at_next_turn(tmp_path):
    boot, runtime = _boot_runtime(
        tmp_path, max_parallel_sidecars=2, grace_seconds=0,
        idle_hot_load_seconds=600, max_grace_extensions=5,
    )
    _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
    mgr = _mk(boot, runtime, **_mocks())

    from unittest.mock import patch
    with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000]):
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            # First turn: constructs the Resident, r.grace.max_extensions=5
            # (the value captured at construction).
            await asyncio.wait_for(
                mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5
            )

            resident = mgr._model_residents()[0]
            assert resident.grace.max_extensions == 5, "sanity: captured at construction"

            # Simulate PUT /api/config changing max_grace_extensions -- same
            # effect api/config_put.py has (mgr.runtime replaced), WITHOUT
            # spinning up the HTTP layer for a unit test.
            new_queue = runtime.queue.model_copy(update={"max_grace_extensions": 9})
            mgr.runtime = runtime.model_copy(update={"queue": new_queue})

            # Second turn on the SAME (already-constructed) Resident.
            await asyncio.wait_for(
                mgr.submit_and_wait("m1", "b", thread_id="t2"), timeout=5
            )

            assert resident.grace.max_extensions == 9, (
                "the already-constructed Resident's grace timer did not pick up "
                "the new max_grace_extensions at its next turn"
            )
        finally:
            await mgr.shutdown()
