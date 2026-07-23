"""Smart GPU auto-placer tests (manifest.auto_place + _auto_pick_gpu).

Mirrors the conventions in tests/test_multislot_concurrency.py (same
_boot_runtime_multislot / _seed_manifest / _mk / _mocks / _high_vram shapes) so
this file drops into the same test session without new fixtures. nvidia-smi is
absent in the test env; tests patch turbohaul.safety._read_free_vram_all_mib
directly, same as the existing multislot suite.

Two-GPU tests use free-VRAM lists like [10000, 20000] (MiB) with small
(3000-4000 MiB expected_vram, ~3050-4070 MiB actual need after the KV-cache
estimate at ctx_size=2048/f16) models so KV-cache-estimate rounding never
flips a pass/fail boundary — the margins are deliberately generous, not tight
edge cases.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
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
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle

# pyproject.toml sets asyncio_mode = "auto" (confirmed against the reference
# project config) -- no pytestmark needed, matching test_multislot_concurrency.py.


def _boot_runtime_multislot(tmp_path, *, max_parallel_sidecars=2,
                            grace_seconds=0, idle_hot_load_seconds=0,
                            safety_min_free_vram_mib=4096):
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
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=idle_hot_load_seconds,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
            safety_min_free_vram_mib=safety_min_free_vram_mib,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag, *, expected_vram_mib=0,
                    split_mode="none", main_gpu=0, auto_place=False):
    """Write a minimal manifest. auto_place=True + split_mode='none' routes
    through _auto_pick_gpu; main_gpu is then only the degrade-open fallback
    (probe unreadable / nothing fits)."""
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": expected_vram_mib * 1024 * 1024,
        "context_size": 2048,
        "expected_vram_bytes": expected_vram_mib * 1024 * 1024,
        "auto_place": auto_place,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


def _fake_handle(model_tag: str, port: int, pid: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


@contextmanager
def _vram(vals):
    with patch("turbohaul.safety._read_free_vram_all_mib", return_value=vals), \
         patch("turbohaul.manager._read_free_vram_all_mib", return_value=vals):
        yield


def _mocks_capturing_argv(spawn_calls, *, health_gate=None):
    """Like test_multislot_concurrency._mocks, but spawn_calls entries also
    carry argv so tests can assert the ACTUAL spawn command's --main-gpu/
    --split-mode, not just the VRAM-gate's decision.

    health_gate (optional asyncio.Event): when given, fake_health blocks until
    it's set. This holds every spawned resident in RESERVED_LOADING on
    purpose — the auto-placer's per-card "spread" decision is driven by
    RESERVED_LOADING siblings (ACTIVE siblings are assumed already reflected
    in the live nvidia-smi reading, which the mocked probe here does NOT
    simulate since it returns a fixed value). Without the gate, a test that
    submits N models one at a time would let each finish (RESERVED_LOADING ->
    ACTIVE) before the next reserves, so the static mocked probe would look
    identical for every submission and ties would deterministically resolve
    to gpu0 — masking the spread behavior instead of exercising it.
    """
    pid = [90000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        spawn_calls.append({"model_tag": model_tag, "port": port, "argv": list(argv)})
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        if health_gate is not None:
            await health_gate.wait()
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


def _mk(boot, runtime, **mocks):
    mgr = TurbohaulManager(boot, runtime, **mocks)
    mgr.runtime.queue.safety_enabled = False
    return mgr


def _argv_value(argv: list[str], flag: str) -> "str | None":
    key = "--" + flag.replace("_", "-")
    if key not in argv:
        return None
    return argv[argv.index(key) + 1]


class TestAutoPickGpuUnit:
    """Direct unit tests of _auto_pick_gpu — no dispatcher/worker loop needed."""

    def test_picks_most_free_card_when_both_fit(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(tmp_path)
        mgr = _mk(boot, runtime, **_mocks_capturing_argv([]))
        with _vram([10000, 20000]):
            gpu, split = mgr._auto_pick_gpu(5000)
        assert (gpu, split) == (1, "none"), "gpu1 has more free -> must be picked"

    def test_picks_the_only_fitting_card(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(
            tmp_path, safety_min_free_vram_mib=4096,
        )
        mgr = _mk(boot, runtime, **_mocks_capturing_argv([]))
        with _vram([5000, 30000]):
            # gpu0: 5000-4096=904 avail, doesn't fit 5000MiB need. gpu1 does.
            gpu, split = mgr._auto_pick_gpu(5000)
        assert (gpu, split) == (1, "none")

    def test_layer_fallback_when_no_single_card_fits_but_aggregate_does(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(
            tmp_path, safety_min_free_vram_mib=1000,
        )
        mgr = _mk(boot, runtime, **_mocks_capturing_argv([]))
        with _vram([6000, 6000]):
            # Neither card alone: 6000-1000=5000 < 9000 need. Aggregate:
            # (6000-1000)+(6000-1000)=10000 >= 9000 -> layer-split fallback.
            gpu, split = mgr._auto_pick_gpu(9000)
        assert (gpu, split) == (0, "layer")

    def test_none_when_nothing_fits_anywhere(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(
            tmp_path, safety_min_free_vram_mib=1000,
        )
        mgr = _mk(boot, runtime, **_mocks_capturing_argv([]))
        with _vram([2000, 2000]):
            gpu, split = mgr._auto_pick_gpu(50000)
        assert gpu is None

    def test_none_when_probe_unavailable(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(tmp_path)
        mgr = _mk(boot, runtime, **_mocks_capturing_argv([]))
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=None), \
             patch("turbohaul.manager._read_free_vram_all_mib", return_value=None):
            gpu, split = mgr._auto_pick_gpu(5000)
        assert gpu is None, "unreadable probe -> caller keeps the manifest pin"

    def test_accounts_for_reserved_loading_siblings_on_same_card(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(
            tmp_path, safety_min_free_vram_mib=1000,
        )
        mgr = _mk(boot, runtime, **_mocks_capturing_argv([]))
        from turbohaul.manager import Resident
        # gpu0 has a RESERVED_LOADING sibling already claiming 8000 MiB.
        mgr._residents["booting-model"] = Resident(
            model_tag="booting-model",
            state=ResidentState.RESERVED_LOADING,
            reserved_need_mib=8000,
            main_gpu=0,
            split_mode="none",
        )
        with _vram([10000, 10000]):
            # gpu0: 10000-8000-1000=1000 avail, doesn't fit 5000. gpu1:
            # 10000-0-1000=9000 avail, fits.
            gpu, split = mgr._auto_pick_gpu(5000)
        assert (gpu, split) == (1, "none")

    def test_safety_floor_excludes_a_card_that_would_go_below_it(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(
            tmp_path, safety_min_free_vram_mib=4096,
        )
        mgr = _mk(boot, runtime, **_mocks_capturing_argv([]))
        with _vram([8000, 8000]):
            # Each card: 8000-4096=3904 avail, doesn't fit the 4000 need on
            # its own -> the floor correctly excludes both single cards.
            # Aggregate (3904*2=7808) does cover 4000, so the layer-split
            # fallback fires instead of admitting a placement that would push
            # a card below the configured safety margin.
            gpu, split = mgr._auto_pick_gpu(4000)
        assert split == "layer", "must not silently violate the safety floor on a single card"


class TestAutoPlaceIntegration:
    """Full dispatcher path: submit_and_wait through _reserve_and_start_locked."""

    async def test_spreads_four_small_models_two_per_card(self, tmp_path):
        """Submits all 4 concurrently and holds each in RESERVED_LOADING via
        health_gate until all 4 have reserved — see _mocks_capturing_argv's
        docstring for why sequential (fully-completed-before-the-next)
        submissions would NOT exercise the spread logic against this static
        VRAM mock."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=4, safety_min_free_vram_mib=1000,
        )
        for tag in ("m1", "m2", "m3", "m4"):
            _seed_manifest(boot, tag, expected_vram_mib=3000, auto_place=True)
        spawn_calls = []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks_capturing_argv(spawn_calls, health_gate=gate))
        with _vram([20000, 20000]):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                futures = [
                    asyncio.create_task(mgr.submit_and_wait(tag, f"p{i}", thread_id=f"t{i}"))
                    for i, tag in enumerate(("m1", "m2", "m3", "m4"))
                ]
                for _ in range(250):
                    await asyncio.sleep(0.02)
                    if len(mgr._model_residents()) >= 4:
                        break
                assert len(mgr._model_residents()) == 4, (
                    "all 4 must have reserved (RESERVED_LOADING) before the health "
                    "gate opens -- otherwise this test isn't exercising the spread"
                )
                by_gpu = {0: 0, 1: 0}
                for r in mgr._model_residents():
                    by_gpu[r.main_gpu] = by_gpu.get(r.main_gpu, 0) + 1
                assert by_gpu == {0: 2, 1: 2}, (
                    f"4 auto_place models on 2 equal cards must split 2/2, got {by_gpu}"
                )
                gate.set()
                await asyncio.wait_for(asyncio.gather(*futures), timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()

    async def test_spawn_argv_reflects_auto_picked_gpu_not_manifest_default(self, tmp_path):
        """The critical fix: _spawn_for_resident must NOT blindly re-derive
        --main-gpu/--split-mode from the manifest's raw flags once auto-place
        has chosen a different card, or the real llama-server process would
        bind the manifest's default (gpu0) while the VRAM gate checked gpu1 —
        silently defeating the whole feature."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, safety_min_free_vram_mib=1000,
        )
        # manifest says main_gpu=0, but gpu0 is starved and gpu1 is wide open
        # -> auto-placer must choose gpu1.
        _seed_manifest(boot, "m1", expected_vram_mib=3000, auto_place=True, main_gpu=0)
        spawn_calls = []
        mgr = _mk(boot, runtime, **_mocks_capturing_argv(spawn_calls))
        with _vram([500, 20000]):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(
                    mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
                )
                assert len(spawn_calls) == 1
                argv = spawn_calls[0]["argv"]
                assert _argv_value(argv, "main_gpu") == "1", (
                    f"spawn argv must reflect the auto-picked gpu1, got {argv}"
                )
                assert _argv_value(argv, "split_mode") == "none"
                r = mgr._model_residents()[0]
                assert r.main_gpu == 1, "the Resident record itself must also show gpu1"
            finally:
                await mgr.shutdown()

    async def test_auto_place_false_stays_pinned_explicit_main_gpu(self, tmp_path):
        """Back-compat: auto_place=False (the default) must be byte-behavior-
        identical to today — honor the manifest's explicit main_gpu verbatim,
        even when the auto-placer would have picked a different card."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, safety_min_free_vram_mib=1000,
        )
        # gpu0 starved, gpu1 wide open -- an auto-placer WOULD pick gpu1, but
        # auto_place is False (default) so the explicit pin to 1 must hold
        # regardless of which card is actually least-loaded.
        _seed_manifest(boot, "m1", expected_vram_mib=3000, main_gpu=1)
        spawn_calls = []
        mgr = _mk(boot, runtime, **_mocks_capturing_argv(spawn_calls))
        with _vram([500, 20000]):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(
                    mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
                )
                r = mgr._model_residents()[0]
                assert r.main_gpu == 1
                argv = spawn_calls[0]["argv"]
                assert _argv_value(argv, "main_gpu") == "1"
            finally:
                await mgr.shutdown()

    async def test_auto_place_refused_when_both_cards_full(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(
            tmp_path, safety_min_free_vram_mib=1000,
        )
        _seed_manifest(boot, "m1", expected_vram_mib=9000, auto_place=True)
        mgr = _mk(boot, runtime, **_mocks_capturing_argv([]))
        with _vram([1500, 1500]):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                with pytest.raises(RuntimeError):
                    await asyncio.wait_for(
                        mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
                    )
            finally:
                await mgr.shutdown()

    async def test_each_used_card_keeps_safety_floor_free(self, tmp_path):
        """Same gate reasoning as test_spreads_four_small_models_two_per_card:
        m2 must see m1 as a RESERVED_LOADING sibling (not yet ACTIVE) so the
        spread decision — and this floor check — reflects the intended
        one-per-card placement rather than an artifact of the static mock."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=4, safety_min_free_vram_mib=2000,
        )
        for tag in ("m1", "m2"):
            _seed_manifest(boot, tag, expected_vram_mib=4000, auto_place=True)
        spawn_calls = []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks_capturing_argv(spawn_calls, health_gate=gate))
        with _vram([10000, 10000]):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                futures = [
                    asyncio.create_task(mgr.submit_and_wait(tag, f"p{i}", thread_id=f"t{i}"))
                    for i, tag in enumerate(("m1", "m2"))
                ]
                for _ in range(250):
                    await asyncio.sleep(0.02)
                    if len(mgr._model_residents()) >= 2:
                        break
                assert len(mgr._model_residents()) == 2
                by_gpu: dict[int, int] = {}
                for r in mgr._model_residents():
                    by_gpu[r.main_gpu] = by_gpu.get(r.main_gpu, 0) + r.reserved_need_mib
                for gpu, used in by_gpu.items():
                    assert 10000 - used >= 2000, (
                        f"gpu{gpu} used {used} of 10000, violates the "
                        f"2000 MiB safety floor"
                    )
                gate.set()
                await asyncio.wait_for(asyncio.gather(*futures), timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()
