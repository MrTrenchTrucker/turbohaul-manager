"""Multi-slot dispatcher concurrency tests (cap==2).

These EXPLICITLY override max_parallel_sidecars=2 (the deployed default stays 1,
so the rest of the suite is byte-identical). They exercise the cap>=2 dispatcher
path: 2-model co-residence, per-resident zero-cross-contamination, the
cross-resident VRAM gate, LRU-idle-only eviction, no registry race, dup-model-tag
single-spawn, driver-death inbox-drain, torn_down exactly-once, booting_pid window.

nvidia-smi is absent in the test env, so the cross-resident VRAM gate would
refuse-blind a 2nd co-resident; tests patch turbohaul.safety._read_free_vram_all_mib
to supply a free-VRAM value (low to force a refusal, high to admit).
"""
from __future__ import annotations

import asyncio
import signal
from unittest.mock import MagicMock, patch

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
from turbohaul.live_monitor import LiveResidentsSupervisor, ResidentSlotsPoller
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime_multislot(tmp_path, *, max_parallel_sidecars=2,
                            grace_seconds=0, idle_hot_load_seconds=0):
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
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _fake_handle(model_tag: str, port: int, pid: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None  # is_alive() -> True
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _high_vram():
    # TWO GPUs, each ample. N1 co-residence requires DISTINCT cards (split=none on
    # gpu0 + gpu1), so a single-card probe couldn't represent a co-resident layout.
    return patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000])


def _mocks(spawn_calls, sigterm_calls=None, complete_gate=None,
           gate_model=None, raise_model=None):
    pid = [90000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        spawn_calls.append({"model_tag": model_tag, "port": port})
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        if sigterm_calls is not None:
            sigterm_calls.append(handle.model_tag)
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        if complete_gate is not None and (
            gate_model is None or handle.model_tag == gate_model
        ):
            await complete_gate.wait()
        if raise_model is not None and handle.model_tag == raise_model:
            raise RuntimeError("boom in complete")
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


def _mk(boot, runtime, **mocks):
    mgr = TurbohaulManager(boot, runtime, **mocks)
    # The cross-resident gate is not gated by safety_enabled; skip only the
    # per-spawn host gate so tests focus on the dispatcher (VRAM tests re-enable).
    mgr.runtime.queue.safety_enabled = False
    return mgr


class TestFanOutParallel:
    """LAYER-2 per-model parallelism: a parallel:N resident serves N same-model
    requests CONCURRENTLY (riders pulled from r.inbox, pipe kept full to n_parallel),
    not serialized; plus the cancel-mid-burst teardown path."""

    @staticmethod
    def _p2_spawn(spawn_calls):
        pid = [70000]

        def spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append(model_tag)
            pid[0] += 1
            proc = MagicMock()
            proc.pid = pid[0]
            proc.poll.return_value = None
            return SidecarHandle(
                proc=proc, port=port, model_tag=model_tag, parallel=2
            )

        return spawn

    @staticmethod
    def _gated_mocks(gate):
        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(h, **k):
            return True, "clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            await gate.wait()
            return {"ok": True}

        return fake_health, fake_sigterm, fake_vram, fake_complete

    async def test_two_same_model_requests_run_concurrently(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        gate = asyncio.Event()
        h, sg, vr, cp = self._gated_mocks(gate)
        inflight_max = [0]
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=self._p2_spawn([]), health_fn=h,
            sigterm_fn=sg, vram_fn=vr, complete_fn=cp,
        )
        mgr.runtime.queue.safety_enabled = False
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
            f2 = asyncio.create_task(mgr.submit_and_wait("m1", "b", thread_id="t2"))
            for _ in range(250):
                await asyncio.sleep(0.02)
                for r in mgr._model_residents():
                    if r.model_tag == "m1":
                        inflight_max[0] = max(inflight_max[0], len(r.inflight))
                if inflight_max[0] >= 2:
                    break
            assert inflight_max[0] >= 2, (
                "parallel:2 resident must serve 2 same-model requests concurrently, "
                f"got max inflight {inflight_max[0]}"
            )
            gate.set()
            await asyncio.wait_for(asyncio.gather(f1, f2), timeout=5)
            await mgr.shutdown()

    async def test_cancel_mid_burst_fails_both_futures(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        gate = asyncio.Event()  # never set -> both park in complete
        h, sg, vr, cp = self._gated_mocks(gate)
        inflight_max = [0]
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=self._p2_spawn([]), health_fn=h,
            sigterm_fn=sg, vram_fn=vr, complete_fn=cp,
        )
        mgr.runtime.queue.safety_enabled = False
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
            f2 = asyncio.create_task(mgr.submit_and_wait("m1", "b", thread_id="t2"))
            for _ in range(250):
                await asyncio.sleep(0.02)
                for r in mgr._model_residents():
                    if r.model_tag == "m1":
                        inflight_max[0] = max(inflight_max[0], len(r.inflight))
                if inflight_max[0] >= 2:
                    break
            assert inflight_max[0] >= 2
            await mgr.shutdown()  # cancels the driver mid-burst
        for f in (f1, f2):
            with pytest.raises((asyncio.CancelledError, RuntimeError)):
                await asyncio.wait_for(f, timeout=5)


class TestCoResidence:
    async def test_two_model_coresidence(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)
        spawn_calls = []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls))
        assert mgr.runtime.queue.max_parallel_sidecars == 2
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.wait_for(mgr.submit_and_wait("m2", "b", thread_id="t2"), timeout=5)
                await asyncio.sleep(0.1)
                live = {r.model_tag for r in mgr._model_residents()}
                assert live == {"m1", "m2"}, f"both models co-resident, got {live}"
                # follow-up on m1 must NOT re-spawn (warm reuse via HIT route)
                await asyncio.wait_for(mgr.submit_and_wait("m1", "c", thread_id="t3"), timeout=5)
                models = [c["model_tag"] for c in spawn_calls]
                assert models.count("m1") == 1, f"no swap-thrash, got {models}"
                assert models.count("m2") == 1
            finally:
                await mgr.shutdown()

    async def test_zero_cross_contamination(self, tmp_path):
        """2 residents ACTIVE at once each read/write ONLY their own
        r.* — no cross-contamination via a shared global/singleton."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)
        spawn_calls = []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, complete_gate=gate))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                f2 = asyncio.create_task(mgr.submit_and_wait("m2", "b", thread_id="t2"))
                await asyncio.sleep(0.2)  # both reach ACTIVE, blocked on gate
                r1 = mgr._residents.get("m1")
                r2 = mgr._residents.get("m2")
                assert r1 is not None and r2 is not None
                # each resident holds ONLY its own handle/active_slot — distinct pids
                assert r1.handle is not None and r2.handle is not None
                assert r1.handle.pid != r2.handle.pid
                assert r1.handle.model_tag == "m1" and r2.handle.model_tag == "m2"
                assert r1.active_slot.model_tag == "m1"
                assert r2.active_slot.model_tag == "m2"
                assert r1.spawn_seq == 1 and r2.spawn_seq == 1
                gate.set()
                await asyncio.wait_for(asyncio.gather(f1, f2), timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()

    async def test_dup_model_tag_single_spawn(self, tmp_path):
        """Two near-simultaneous submits for the SAME new model => exactly ONE
        resident / ONE spawn (the atomic get-then-reserve under the lock)."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        spawn_calls = []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, complete_gate=gate))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                f2 = asyncio.create_task(mgr.submit_and_wait("m1", "b", thread_id="t2"))
                await asyncio.sleep(0.2)
                assert [c["model_tag"] for c in spawn_calls].count("m1") == 1
                assert len(mgr._model_residents()) == 1
                gate.set()
                await asyncio.wait_for(asyncio.gather(f1, f2), timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()


class TestVramGate:
    async def test_refuses_cross_resident_overcommit(self, tmp_path):
        """2nd co-resident refused when its OWN card can't fit it. m1 pins GPU0, m2
        pins GPU1 (N1: split=none/distinct-card); GPU1 has only 10 GiB free so the
        18 GiB m2 is refused by the per-card budget. safety_enabled stays True so the
        real cross-resident + per-spawn gates run (m1's GPU0 has 22 GiB -> admits)."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        # Real footprints; pin to distinct cards so co-residence is N1-admissible and
        # the refusal comes from the per-card VRAM budget, not the N1 topology gate.
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", expected_vram_mib=18000, split_mode="none", main_gpu=1)
        spawn_calls = []
        gate = asyncio.Event()
        mgr = TurbohaulManager(boot, runtime, **_mocks(spawn_calls, complete_gate=gate))
        # GPU0: 22 GiB free (m1's 18 GiB fits); GPU1: 10 GiB free (m2's 18 GiB does NOT).
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[22000, 10000]):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                await asyncio.sleep(0.2)
                assert mgr._residents.get("m1") is not None
                with pytest.raises((RuntimeError,)):
                    await asyncio.wait_for(mgr.submit_and_wait("m2", "b", thread_id="t2"), timeout=5)
                assert mgr._residents.get("m2") is None
                assert [c["model_tag"] for c in spawn_calls] == ["m1"]
                gate.set()
                await asyncio.wait_for(f1, timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()

    async def test_footprint_cpu_moe_trusts_measured_expected_vram(self, tmp_path):
        # _read_model_footprint: a cpu-moe (n_cpu_moe) manifest with a measured
        # expected_vram LOWER than gguf+kv must reserve the MEASURED value, not the
        # over-counted max(expected, gguf+kv). (guards the reserve-gate for a large
        # MoE model whose measured footprint is below its gguf+kv upper bound.)
        import yaml
        boot, runtime = _boot_runtime_multislot(tmp_path)
        p = boot.storage.manifests_path / "cm.yaml"
        p.write_text(yaml.safe_dump({
            "model_tag": "cm", "gguf_blob_sha256": "a" * 64,
            "gguf_size_bytes": 20 * 1024 * 1024 * 1024,
            "context_size": 500000,
            "expected_vram_bytes": 19 * 1024 * 1024 * 1024,
            "llama_server_flags": {
                "split_mode": "none", "main_gpu": 1, "n_cpu_moe": 10,
                "parallel": 2, "kv_unified": True, "ctx_size": 500000,
                "cache_type_k": "turbo2"},
        }))
        mgr = TurbohaulManager(boot, runtime, **_mocks([]))
        need, parallel, main_gpu, split_mode, _sleep, _auto = mgr._read_model_footprint("cm")
        assert need == 19 * 1024 + (2 - 1) * 256, f"cpu-moe must trust measured, got {need}"
        assert parallel == 2 and main_gpu == 1 and split_mode == "none"
        await mgr.shutdown()


def _seed_manifest(boot, model_tag, *, expected_vram_mib=0,
                   split_mode="none", main_gpu=0):
    """Write a minimal manifest. split_mode/main_gpu drive the N1 co-residence gate:
    co-residence is admitted ONLY for split_mode='none' models on DISTINCT main_gpu,
    so co-residing tests pin gpu0/gpu1 + split='none'."""
    import yaml
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": expected_vram_mib * 1024 * 1024,
        "context_size": 2048,
        "expected_vram_bytes": expected_vram_mib * 1024 * 1024,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


class TestLRU:
    async def test_evicts_idle_only(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120
        )
        # N1: split=none on distinct cards so co-residence is admitted. m3 reuses
        # gpu1 (m2's freed card) once m2 is LRU-evicted.
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)
        _seed_manifest(boot, "m3", split_mode="none", main_gpu=1)
        spawn_calls = []
        sigterm_calls = []
        gate = asyncio.Event()
        # m1 BUSY (its complete blocks on the gate); m2 completes -> IDLE_EVICTABLE.
        mgr = _mk(boot, runtime, **_mocks(
            spawn_calls, sigterm_calls=sigterm_calls,
            complete_gate=gate, gate_model="m1",
        ))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(
                    mgr.submit_and_wait("m1", "a", thread_id="t1")
                )
                await asyncio.sleep(0.2)  # m1 ACTIVE, blocked on gate
                await asyncio.wait_for(
                    mgr.submit_and_wait("m2", "b", thread_id="t2"), timeout=5
                )
                await asyncio.sleep(0.15)  # m2 -> IDLE_EVICTABLE
                r2 = mgr._residents.get("m2")
                assert r2 is not None and r2.state is ResidentState.IDLE_EVICTABLE
                # capacity full (m1 busy + m2 idle); m3 -> evict IDLE m2, NOT busy m1
                f3 = asyncio.create_task(
                    mgr.submit_and_wait("m3", "c", thread_id="t3")
                )
                await asyncio.sleep(0.4)
                assert mgr._residents.get("m2") is None, "idle m2 evicted"
                assert mgr._residents.get("m1") is not None, "busy m1 NOT evicted"
                assert "m2" in sigterm_calls and "m1" not in sigterm_calls
                assert mgr._residents.get("m3") is not None, "m3 took the freed slot"
                gate.set()
                await asyncio.wait_for(asyncio.gather(f1, f3), timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()


class TestRace:
    async def test_no_registry_race(self, tmp_path):
        """Concurrent same+different-model submits: registry never exceeds the cap,
        no double-spawn of a model (the _registry_lock is what makes this pass)."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120
        )
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)
        spawn_calls = []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, complete_gate=gate))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                tasks = [
                    asyncio.create_task(
                        mgr.submit_and_wait(m, "x", thread_id=f"t{i}")
                    )
                    for i, m in enumerate(["m1", "m1", "m2", "m2", "m1"])
                ]
                await asyncio.sleep(0.3)
                assert len(mgr._model_residents()) <= 2
                spawned = [c["model_tag"] for c in spawn_calls]
                assert spawned.count("m1") == 1 and spawned.count("m2") == 1
                gate.set()
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=5
                )
            finally:
                gate.set()
                await mgr.shutdown()


class TestDriverDeath:
    async def test_driver_death_fails_future_and_reaps(self, tmp_path):
        """A driver that dies mid-serve (complete raises) => supervisor fails the
        in-flight future + deregisters the resident."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=0
        )
        spawn_calls = []
        sigterm_calls = []
        mgr = _mk(boot, runtime, **_mocks(
            spawn_calls, sigterm_calls=sigterm_calls, raise_model="m1",
        ))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                with pytest.raises(RuntimeError):
                    await asyncio.wait_for(
                        mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5
                    )
                await asyncio.sleep(0.2)
                assert mgr._residents.get("m1") is None, "dead resident deregistered"
            finally:
                await mgr.shutdown()


class TestBootingPid:
    async def test_booting_pid_in_live_handle_pids(self, tmp_path):
        """While RESERVED_LOADING (spawned, handle not yet published) the resident's
        booting_pid is in _live_handle_pids so a sibling teardown's intra_lifetime
        scan won't reap the still-booting sidecar."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120
        )
        spawn_calls = []
        health_gate = asyncio.Event()

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append({"model_tag": model_tag})
            return _fake_handle(model_tag, port, 91234)

        async def fake_health(*a, **k):
            await health_gate.wait()
            return True

        async def fake_sigterm(*a, **k):
            return True, "clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(
                    mgr.submit_and_wait("m1", "a", thread_id="t1")
                )
                await asyncio.sleep(0.2)  # spawned, blocked in health -> booting_pid set
                r = mgr._residents.get("m1")
                assert r is not None and r.state is ResidentState.RESERVED_LOADING
                assert r.booting_pid == 91234
                assert 91234 in mgr._live_handle_pids(), "booting pid in reaper union"
                health_gate.set()
                await asyncio.wait_for(f1, timeout=5)
            finally:
                health_gate.set()
                await mgr.shutdown()


class TestSpawnFail:
    async def test_health_timeout_reaps_no_leak(self, tmp_path):
        """Health-timeout => the spawned sidecar IS reaped (sigterm) + the resident
        is deregistered + booting_pid cleared (no PID/VRAM leak). The 8 routing
        tests use fake_health=True, so this guards the high-severity leak fix."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=0
        )
        spawn_calls = []
        sigterm_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append({"model_tag": model_tag})
            return _fake_handle(model_tag, port, 95000)

        async def fake_health(*a, **k):
            return False  # health timeout

        async def fake_sigterm(handle, **k):
            sigterm_calls.append(handle.model_tag)
            return True, "clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                with pytest.raises(RuntimeError):
                    await asyncio.wait_for(
                        mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5
                    )
                await asyncio.sleep(0.2)
                assert spawn_calls and spawn_calls[0]["model_tag"] == "m1"
                assert "m1" in sigterm_calls, "unhealthy sidecar reaped (no leak)"
                assert mgr._residents.get("m1") is None, "resident deregistered"
                assert 95000 not in mgr._live_handle_pids(), "booting_pid cleared"
            finally:
                await mgr.shutdown()


class TestVramLoadStateAware:
    async def test_admits_second_model_after_sibling_loads(self, tmp_path):
        """B1: once a sibling's weights load, they DROP OUT of the live nvidia-smi
        free reading. The cross-resident gate must NOT also subtract the sibling's
        reserved footprint (double-count) -- a 2nd split=none model on a DISTINCT
        card must still ADMIT. The other tests' constant-VRAM mock masks this; here
        GPU0's free COLLAPSES once m1 reaches ACTIVE while GPU1 stays free, exactly
        the steady-state two-warm-model case the feature exists to serve."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", expected_vram_mib=18000, split_mode="none", main_gpu=1)
        spawn_calls = []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, complete_gate=gate))

        def live_probe():
            # GPU0 free collapses once m1's weights are resident (state ACTIVE);
            # GPU1 stays ample. The OLD double-counting gate would compute
            # free_fit(GPU1)=24G - m1.reserve(18G) = 6G < 18G and WRONG-REFUSE m2.
            m1 = mgr._residents.get("m1")
            g0 = 4000 if (m1 is not None and m1.state is ResidentState.ACTIVE) else 24000
            return [g0, 24000]

        with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=live_probe):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                await asyncio.sleep(0.2)  # m1 ACTIVE (blocked on gate) -> GPU0 probe low
                r1 = mgr._residents.get("m1")
                assert r1 is not None and r1.state is ResidentState.ACTIVE
                f2 = asyncio.create_task(mgr.submit_and_wait("m2", "b", thread_id="t2"))
                await asyncio.sleep(0.2)
                r2 = mgr._residents.get("m2")
                assert r2 is not None, "loaded sibling on GPU0 must NOT block m2 on GPU1"
                assert {r.model_tag for r in mgr._model_residents()} == {"m1", "m2"}
                gate.set()
                await asyncio.wait_for(asyncio.gather(f1, f2), timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()


class TestN1LayerSplit:
    async def test_refuses_layer_split_coresidence(self, tmp_path):
        """N1 interim: co-residence is supported ONLY for single-GPU-pinned
        (split_mode='none') models on DISTINCT cards. A layer-split 2nd model spans
        every card, so it is refuse-blinded until per-card layer-split accounting
        lands -- EVEN with ample free VRAM (the refusal is topology, not budget)."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="layer", main_gpu=0)  # spans all cards
        spawn_calls = []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, complete_gate=gate))
        with _high_vram():  # plenty free on both cards -> refusal is N1, not budget
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                await asyncio.sleep(0.2)
                assert mgr._residents.get("m1") is not None
                with pytest.raises(RuntimeError):
                    await asyncio.wait_for(
                        mgr.submit_and_wait("m2", "b", thread_id="t2"), timeout=5
                    )
                assert mgr._residents.get("m2") is None, "layer-split 2nd model refused"
                gate.set()
                await asyncio.wait_for(f1, timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()


class TestInboxDrain:
    async def test_inbox_drain_on_spawn_fail(self, tmp_path):
        """B3: while a resident is RESERVED_LOADING, a 2nd same-model submit HIT-routes
        into r.inbox. If the load then FAILS, the driver finally must drain r.inbox
        back to the main queue -- else the 2nd slot is PERMANENTLY LOST (the client
        hangs until upstream timeout). With the fix the 2nd slot's future RESOLVES
        (re-served on a fresh resident, then fails its own spawn) instead of hanging.
        The single-slot routing tests never queue a 2nd same-model slot, so they mask
        this."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=0
        )
        spawn_calls = []
        sigterm_calls = []
        health_gate = asyncio.Event()
        pid = [96000]

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append({"model_tag": model_tag})
            pid[0] += 1
            return _fake_handle(model_tag, port, pid[0])

        async def fake_health(*a, **k):
            # First check blocks until released; thereafter ALL checks fail (the
            # load is genuinely broken) so every spawn attempt times out -> no
            # infinite respawn (each slot's future fails fast on its own gate).
            await health_gate.wait()
            return False

        async def fake_sigterm(handle, **k):
            sigterm_calls.append(handle.model_tag)
            return True, "clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                await asyncio.sleep(0.2)  # m1 RESERVED_LOADING, blocked in health
                r = mgr._residents.get("m1")
                assert r is not None and r.state is ResidentState.RESERVED_LOADING
                # 2nd same-model submit HIT-routes into the loading resident's inbox.
                f2 = asyncio.create_task(mgr.submit_and_wait("m1", "b", thread_id="t2"))
                await asyncio.sleep(0.2)
                assert not r.inbox.empty(), "2nd slot HIT-routed into the loading inbox"
                health_gate.set()  # release -> health FAILS -> spawn fails -> finally drains
                results = await asyncio.wait_for(
                    asyncio.gather(f1, f2, return_exceptions=True), timeout=5
                )
                # BOTH futures resolve (as errors); f2 NOT hanging proves it wasn't lost.
                assert all(isinstance(x, Exception) for x in results), results
                await asyncio.sleep(0.1)
                assert mgr._residents.get("m1") is None, "dead resident deregistered"
            finally:
                health_gate.set()
                await mgr.shutdown()


class TestCancelDuringLoading:
    async def test_cancel_during_loading_reaps_booting_pid(self, tmp_path, monkeypatch):
        """B2: a driver cancelled while RESERVED_LOADING (sidecar spawned, handle not
        yet published) must reap the live process by booting_pid -- nothing else can,
        because _live_handle_pids PROTECTS booting_pid from the orphan reaper. Assert
        os.kill(booting_pid, SIGTERM) fires on driver cancel, and the anchor slot's
        future is failed (not lost). The routing tests use fake_health=True so the
        driver never sits in the RESERVED_LOADING cancel window."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120
        )
        killed = []

        def rec_kill(pid, sig):
            killed.append((pid, sig))  # record; do NOT signal the fake pid

        monkeypatch.setattr("turbohaul.manager.os.kill", rec_kill)

        # _reap_booting_pid now waitpid-checks ownership BEFORE signalling.
        # The fake pid 93777 isn't a real child, so report it ALIVE at the pre-check
        # (so SIGTERM fires) then exited (so the reap returns) without a 3s grace spin.
        wp_calls = [0]

        def fake_waitpid(p, f):
            wp_calls[0] += 1
            return (0, 0) if wp_calls[0] == 1 else (p, 0)

        monkeypatch.setattr("turbohaul.manager.os.waitpid", fake_waitpid)

        spawn_calls = []
        health_gate = asyncio.Event()

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append({"model_tag": model_tag})
            return _fake_handle(model_tag, port, 93777)

        async def fake_health(*a, **k):
            await health_gate.wait()  # park in RESERVED_LOADING until cancelled
            return True

        async def fake_sigterm(*a, **k):
            return True, "clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                await asyncio.sleep(0.2)  # spawned -> booting_pid set, blocked in health
                r = mgr._residents.get("m1")
                assert r is not None and r.state is ResidentState.RESERVED_LOADING
                assert r.booting_pid == 93777
                r.driver_task.cancel()  # cancel the driver mid-load (teardown/shutdown)
                await asyncio.sleep(0.2)  # finally + supervisor reaper run
                assert (93777, signal.SIGTERM) in killed, f"booting_pid reaped, got {killed}"
                # the anchor slot's future must be failed, never lost (hang).
                with pytest.raises((asyncio.CancelledError, RuntimeError)):
                    await asyncio.wait_for(f1, timeout=2)
            finally:
                health_gate.set()
                await mgr.shutdown()


# ============================================================================
# P1e: per-resident live monitor + status residents[]/vram[] +
# shutdown-sweep + 2 folded follow-ups (waitpid reap, bg_tasks drain).
# ============================================================================
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _FakeSlotsClient:
    """Stand-in for httpx.AsyncClient: .get returns a canned /slots payload, raises a
    seeded exception, or fires an on_get hook (used to mutate identity mid-await)."""

    def __init__(self, payload=None, *, raise_exc=None, on_get=None):
        self._payload = payload if payload is not None else []
        self._raise = raise_exc
        self._on_get = on_get
        self.closed = False

    async def get(self, url):
        if self._on_get is not None:
            self._on_get(url)
        if self._raise is not None:
            raise self._raise
        return _FakeResp(self._payload)

    async def aclose(self):
        self.closed = True


def _slots_processing(n_decoded=10, max_tokens=100, stream=True):
    """A llama.cpp /slots payload with one actively-processing slot."""
    return [{
        "id_task": 0,
        "is_processing": True,
        "next_token": [{
            "n_decoded": n_decoded,
            "n_remain": max_tokens - n_decoded,
            "has_next_token": True,
        }],
        "n_prompt_tokens": 5,
        "n_prompt_tokens_processed": 5,
        "n_ctx": 4096,
        "params": {"max_tokens": max_tokens, "stream": stream},
    }]


class TestStatusResidentsVram:
    async def test_residents_and_vram_in_snapshot(self, tmp_path):
        """P1e: status_snapshot emits residents[] (per live model) + vram[] (cached)
        while keeping the `generation` back-compat alias. Await-free + lock-free."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks([], complete_gate=gate))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                f2 = asyncio.create_task(mgr.submit_and_wait("m2", "b", thread_id="t2"))
                await asyncio.sleep(0.2)  # both ACTIVE
                # simulate the supervisor's per-resident gen + per-GPU vram cache
                mgr.live_generations = {
                    "m1": {"state": "generating", "generation_id": "aa", "tok_s": 5.0},
                    "m2": {"state": "generating", "generation_id": "bb", "tok_s": 7.0},
                }
                mgr._vram_free_mib = [1111, 2222]
                snap = mgr.status_snapshot()
                residents = {r["model_tag"]: r for r in snap["residents"]}
                assert set(residents) == {"m1", "m2"}
                assert residents["m1"]["state"] == "ACTIVE"
                assert residents["m1"]["port"] is not None
                assert residents["m1"]["pid"] is not None
                assert residents["m1"]["split_mode"] == "none"
                assert residents["m1"]["main_gpu"] == 0 and residents["m2"]["main_gpu"] == 1
                assert residents["m1"]["generation"]["generation_id"] == "aa"
                assert snap["vram"] == [1111, 2222]
                assert "generation" in snap, "back-compat alias preserved"
                gate.set()
                await asyncio.wait_for(asyncio.gather(f1, f2), timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()

    async def test_residents_empty_and_vram_null_at_cap1(self, tmp_path):
        """cap<=1: residents[] is EMPTY (the legacy singleton is excluded — the
        active/loading/grace fields carry single-sidecar state) and vram[] is null
        (no supervisor populates the cache). The `generation` alias is unchanged."""
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=1)
        mgr = _mk(boot, runtime, **_mocks([]))
        snap = mgr.status_snapshot()
        assert snap["residents"] == [], "singleton excluded at cap<=1"
        assert snap["vram"] is None
        assert "generation" in snap
        await mgr.shutdown()


class TestLiveSupervisor:
    async def test_supervisor_writes_per_resident_generations(self, tmp_path):
        """P1e: ONE supervisor tick polls EVERY live resident's /slots and writes a
        per-model generation into live_generations, mirrors the primary into the
        live_generation alias, and refreshes the per-GPU VRAM cache."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks([], complete_gate=gate))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                f2 = asyncio.create_task(mgr.submit_and_wait("m2", "b", thread_id="t2"))
                await asyncio.sleep(0.2)  # both ACTIVE, blocked on gate
                sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
                with patch("turbohaul.live_monitor.httpx.AsyncClient",
                           return_value=_FakeSlotsClient(_slots_processing())), \
                     patch("turbohaul.live_monitor._read_free_vram_all_mib",
                           return_value=[111, 222]):
                    await sup._tick()
                    assert set(mgr.live_generations) == {"m1", "m2"}
                    assert mgr.live_generations["m1"]["state"] in (
                        "generating", "prefill", "finishing", "stalled")
                    assert mgr.live_generations["m1"]["generation_id"] is not None
                    assert mgr.live_generation is not None, "primary alias mirrored"
                    assert mgr._vram_free_mib == [111, 222], "vram cache refreshed"
                    snap = mgr.status_snapshot()
                    assert {r["model_tag"] for r in snap["residents"]} == {"m1", "m2"}
                    assert snap["vram"] == [111, 222]
                    await sup._close_all()
                gate.set()
                await asyncio.wait_for(asyncio.gather(f1, f2), timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()

    async def test_supervisor_gcs_vanished_resident(self, tmp_path):
        """P1e: when a resident vanishes (evicted/dead), the supervisor GCs its poller
        (closing the httpx client) AND its live_generations entry — no leak."""
        boot, runtime = _boot_runtime_multislot(tmp_path)
        mgr = _mk(boot, runtime, **_mocks([]))
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        with patch("turbohaul.live_monitor.httpx.AsyncClient",
                   return_value=_FakeSlotsClient()):
            poller = ResidentSlotsPoller(mgr, interval_s=1.0)
            sup._pollers["ghost"] = poller
            mgr.live_generations["ghost"] = {"state": "idle"}
            await sup._gc_vanished(live_tags=set())  # nothing live anymore
            assert "ghost" not in sup._pollers, "vanished poller GC'd"
            assert "ghost" not in mgr.live_generations, "vanished generation GC'd"
            assert poller._client.closed, "vanished poller's httpx client closed"
        await mgr.shutdown()

    async def test_supervisor_poll_error_is_transitioning_not_crash(self, tmp_path):
        """P1e (forced failure path): a /slots GET that RAISES must yield a
        'transitioning' generation, never crash the supervisor tick (pure observer)."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks([], complete_gate=gate))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                await asyncio.sleep(0.2)
                sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
                with patch("turbohaul.live_monitor.httpx.AsyncClient",
                           return_value=_FakeSlotsClient(raise_exc=RuntimeError("boom"))), \
                     patch("turbohaul.live_monitor._read_free_vram_all_mib",
                           return_value=None):
                    await sup._tick()  # must NOT raise
                    assert mgr.live_generations["m1"]["state"] == "transitioning"
                    assert mgr._vram_free_mib is None, "probe-down -> null, not stale"
                    await sup._close_all()
                gate.set()
                await asyncio.wait_for(f1, timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()

    async def test_poll_skips_write_if_resident_evicted_midpoll(self, tmp_path):
        """Forced path: a resident that goes DEAD DURING the /slots await
        must NOT get a ZOMBIE generation written — _poll_one re-checks liveness after
        the poll. poll_once's re-validate catches a handle SWAP but not a same-handle
        DEAD transition, so this is the gap that needed the explicit liveness re-check."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks([], complete_gate=gate))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                await asyncio.sleep(0.2)
                r = mgr._residents["m1"]
                sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
                poller = ResidentSlotsPoller(mgr, interval_s=1.0)

                def evict_midpoll(url):
                    r.state = ResidentState.DEAD  # evicted/dead mid-await

                poller._client = _FakeSlotsClient(
                    _slots_processing(), on_get=evict_midpoll)
                await sup._poll_one("m1", poller, r)
                assert "m1" not in mgr.live_generations, "no zombie gen for dead resident"
                await poller._client.aclose()
                gate.set()
                await asyncio.gather(f1, return_exceptions=True)
            finally:
                gate.set()
                await mgr.shutdown()


class TestResidentPollerRevalidate:
    async def test_poll_once_revalidates_spawn_seq_swap(self, tmp_path):
        """P1e: poll_once must REJECT a stale /slots reading if the resident's
        spawn_seq advanced across the await (fixed-port sidecar reuse) -> the gen is
        'transitioning', never gen-A's tok/s attributed to gen-B."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks([], complete_gate=gate))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                await asyncio.sleep(0.2)
                r = mgr._residents["m1"]
                poller = ResidentSlotsPoller(mgr, interval_s=1.0)

                def bump_spawn(url):
                    r.spawn_seq += 1  # sidecar swapped mid-await

                poller._client = _FakeSlotsClient(_slots_processing(), on_get=bump_spawn)
                gen = await poller.poll_once(r)
                assert gen["state"] == "transitioning", "stale /slots rejected"
                await poller._client.aclose()
                gate.set()
                await asyncio.wait_for(f1, timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()


class TestReapBootingPidWaitpid:
    async def test_reap_alive_then_sigterm_reaps(self, tmp_path, monkeypatch):
        """An ALIVE booting child is SIGTERM'd, then waitpid-reaped
        when it exits (so it doesn't linger as a zombie holding the PID). No SIGKILL
        when SIGTERM is honored. The pre-signal waitpid reports it ALIVE
        first, then 'exited' after the SIGTERM."""
        boot, runtime = _boot_runtime_multislot(tmp_path)
        mgr = _mk(boot, runtime, **_mocks([]))
        killed, waited = [], []
        calls = [0]
        monkeypatch.setattr("turbohaul.manager.os.kill",
                            lambda p, s: killed.append((p, s)))

        def fake_waitpid(p, f):
            waited.append((p, f))
            calls[0] += 1
            # pre-check -> (0,0)=alive; after SIGTERM the loop's WNOHANG -> (p,0)=exited+reaped
            return (0, 0) if calls[0] == 1 else (p, 0)

        monkeypatch.setattr("turbohaul.manager.os.waitpid", fake_waitpid)
        await mgr._reap_booting_pid(40404)
        assert (40404, signal.SIGTERM) in killed, "alive child gets SIGTERM"
        assert (40404, signal.SIGKILL) not in killed, "no SIGKILL when SIGTERM honored"
        assert len(waited) >= 2, "post-SIGTERM waitpid reaped the zombie"
        await mgr.shutdown()

    async def test_reap_not_our_child_never_signals(self, tmp_path, monkeypatch):
        """PID-RECYCLE GUARD: if the pre-signal waitpid raises ECHILD (the pid
        is no longer OUR child — already reaped, possibly recycled to a foreign process)
        we must NOT fire SIGTERM/SIGKILL at it. Zero signals."""
        boot, runtime = _boot_runtime_multislot(tmp_path)
        mgr = _mk(boot, runtime, **_mocks([]))
        killed = []
        monkeypatch.setattr("turbohaul.manager.os.kill",
                            lambda p, s: killed.append((p, s)))

        def waitpid_echild(p, f):
            raise ChildProcessError()  # ECHILD: not our child

        monkeypatch.setattr("turbohaul.manager.os.waitpid", waitpid_echild)
        await mgr._reap_booting_pid(40405)  # must NOT raise
        assert killed == [], "a not-our-child (recycled) pid is NEVER signaled"
        await mgr.shutdown()

    async def test_reap_already_exited_at_precheck_no_sigterm(self, tmp_path, monkeypatch):
        """If the pre-signal waitpid reaps a ZOMBIE (child already exited) there is
        nothing to kill -> no SIGTERM/SIGKILL, just the reap."""
        boot, runtime = _boot_runtime_multislot(tmp_path)
        mgr = _mk(boot, runtime, **_mocks([]))
        killed = []
        monkeypatch.setattr("turbohaul.manager.os.kill",
                            lambda p, s: killed.append((p, s)))
        # pre-check WNOHANG returns (pid, 0) = our child, just exited, reaped here.
        monkeypatch.setattr("turbohaul.manager.os.waitpid", lambda p, f: (p, 0))
        await mgr._reap_booting_pid(40406)
        assert killed == [], "already-exited child reaped at pre-check -> no signal"
        await mgr.shutdown()

    async def test_reap_escalates_to_sigkill(self, tmp_path, monkeypatch):
        """Forced failure path: a sidecar that IGNORES SIGTERM must be
        SIGKILL'd then final-reaped. WNOHANG always reports 'alive' so the grace window
        exhausts and escalates (time.sleep patched out for speed)."""
        boot, runtime = _boot_runtime_multislot(tmp_path)
        mgr = _mk(boot, runtime, **_mocks([]))
        killed, waited = [], []
        monkeypatch.setattr("turbohaul.manager.os.kill",
                            lambda p, s: killed.append((p, s)))

        def fake_waitpid(p, f):
            waited.append((p, f))
            # WNOHANG (f != 0) -> (0, 0) = still alive; blocking (f == 0) after SIGKILL
            # -> (p, 0) = reaped.
            return (0, 0) if f else (p, 0)

        monkeypatch.setattr("turbohaul.manager.os.waitpid", fake_waitpid)
        monkeypatch.setattr("turbohaul.manager.time.sleep", lambda _s: None)
        await mgr._reap_booting_pid(50505)
        assert (50505, signal.SIGTERM) in killed, "SIGTERM first"
        assert (50505, signal.SIGKILL) in killed, "escalated to SIGKILL after the poll window"
        assert any(f == 0 for (_p, f) in waited), "final BLOCKING waitpid reaped the killed child"
        await mgr.shutdown()


class TestShutdownSweep:
    async def test_shutdown_drains_bg_tasks(self, tmp_path):
        """shutdown() must AWAIT in-flight _spawn_bg tasks so an
        in-flight teardown can't leak past process exit (not GC/cancel them mid-run)."""
        boot, runtime = _boot_runtime_multislot(tmp_path)
        mgr = _mk(boot, runtime, **_mocks([]))
        done = []

        async def slow():
            await asyncio.sleep(0.1)
            done.append(True)

        mgr._spawn_bg(slow())
        await mgr.shutdown()
        assert done == [True], "in-flight bg task drained (awaited), not dropped"

    async def test_shutdown_cancels_supervisor_and_reaps_active_driver(self, tmp_path):
        """P1e shutdown-sweep: OBSERVERS cancelled first, then the DRIVERS — an ACTIVE
        resident's driver is cancelled and its sidecar reaped (sigterm) with the
        resident deregistered (no leak)."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        sigterm_calls = []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks(
            [], sigterm_calls=sigterm_calls, complete_gate=gate))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            mgr._live_supervisor = LiveResidentsSupervisor(mgr, interval_s=10.0)
            mgr._live_supervisor_task = asyncio.create_task(mgr._live_supervisor.run())
            f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
            await asyncio.sleep(0.2)  # m1 ACTIVE, driver blocked in complete(gate)
            sup_task = mgr._live_supervisor_task
            driver = mgr._residents["m1"].driver_task
            await mgr.shutdown()  # never set the gate -> sweep cancels the active driver
            assert sup_task.done(), "supervisor (observer) cancelled by the sweep"
            assert driver.done(), "active driver cancelled by the sweep"
            assert "m1" in sigterm_calls, "active resident's sidecar reaped on shutdown"
            assert mgr._residents.get("m1") is None, "resident deregistered (no leak)"
        with pytest.raises((asyncio.CancelledError, RuntimeError)):
            await asyncio.wait_for(f1, timeout=2)

    async def test_shutdown_parallelizes_driver_teardowns(self, tmp_path):
        """The sweep awaits driver teardowns via asyncio.gather (PARALLEL),
        not a sequential for-await — so N slow-SIGTERM sidecars don't serialize to N x
        the per-driver reap window. Proven deterministically: a sequential await would
        only ever have ONE sigterm in flight; gather has both (>=2)."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)
        gate = asyncio.Event()
        inflight = [0]
        max_inflight = [0]
        sigterm_calls = []
        pid = [70000]

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            pid[0] += 1
            return _fake_handle(model_tag, port, pid[0])

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(handle, **k):
            inflight[0] += 1
            max_inflight[0] = max(max_inflight[0], inflight[0])
            await asyncio.sleep(0.15)  # hold so concurrent teardowns overlap observably
            inflight[0] -= 1
            sigterm_calls.append(handle.model_tag)
            return True, "clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            await gate.wait()  # both residents park ACTIVE until shutdown cancels them
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
            f2 = asyncio.create_task(mgr.submit_and_wait("m2", "b", thread_id="t2"))
            await asyncio.sleep(0.2)  # both ACTIVE (blocked in complete on gate)
            assert {r.model_tag for r in mgr._model_residents()} == {"m1", "m2"}
            await mgr.shutdown()  # sweep cancels both drivers -> both reap in PARALLEL
            assert max_inflight[0] >= 2, "driver teardowns gathered (parallel), not serial"
            assert set(sigterm_calls) == {"m1", "m2"}, "both sidecars reaped"
        for f in (f1, f2):
            with pytest.raises((asyncio.CancelledError, RuntimeError)):
                await asyncio.wait_for(f, timeout=2)


class TestSupervisorPrimaryAlias:
    async def test_alias_picks_most_recently_active_resident(self, tmp_path):
        """_update_primary_alias mirrors the MOST-RECENTLY-ACTIVE resident's generation
        into the live_generation back-compat alias (what /status + live_stream follow
        by default at cap>=2). Order-independent; None when no resident has a gen."""
        from turbohaul.manager import Resident
        boot, runtime = _boot_runtime_multislot(tmp_path)
        mgr = _mk(boot, runtime, **_mocks([]))
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        r_old = Resident(model_tag="old", state=ResidentState.ACTIVE,
                         last_active_monotonic=10.0)
        r_new = Resident(model_tag="new", state=ResidentState.ACTIVE,
                         last_active_monotonic=20.0)
        mgr.live_generations = {"old": {"generation_id": "OLD"},
                                "new": {"generation_id": "NEW"}}
        sup._update_primary_alias([r_old, r_new])
        assert mgr.live_generation["generation_id"] == "NEW", "alias = most-recent"
        sup._update_primary_alias([r_new, r_old])  # order-independent
        assert mgr.live_generation["generation_id"] == "NEW"
        # a resident with NO generation yet is skipped (not chosen just for recency)
        mgr.live_generations = {"old": {"generation_id": "OLD"}}
        sup._update_primary_alias([r_old, r_new])
        assert mgr.live_generation["generation_id"] == "OLD"
        # no live generation at all -> alias None (status_snapshot falls back to idle)
        mgr.live_generations = {}
        sup._update_primary_alias([r_old, r_new])
        assert mgr.live_generation is None
        await mgr.shutdown()


class TestSupervisorRunLoop:
    async def test_run_loop_ticks_then_cancel_closes_clients(self, tmp_path):
        """The supervisor run() LOOP (not just _tick) writes per-resident gens ~1Hz
        and, on cancel, its finally _close_all closes EVERY per-resident httpx client
        (the observer-leak guard that a resource-lifecycle review demands)."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks([], complete_gate=gate))
        clients = []

        def make_client(*a, **k):
            c = _FakeSlotsClient(_slots_processing())
            clients.append(c)
            return c

        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                await asyncio.sleep(0.2)  # m1 ACTIVE
                with patch("turbohaul.live_monitor.httpx.AsyncClient",
                           side_effect=make_client), \
                     patch("turbohaul.live_monitor._read_free_vram_all_mib",
                           return_value=[9]):
                    sup = LiveResidentsSupervisor(mgr, interval_s=0.05)
                    task = asyncio.create_task(sup.run())
                    await asyncio.sleep(0.25)  # several ticks
                    assert mgr.live_generations.get("m1") is not None, "loop wrote a gen"
                    assert clients, "a per-resident httpx client was created"
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    assert all(c.closed for c in clients), "run() finally closed all clients"
                gate.set()
                await asyncio.wait_for(f1, timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()


class TestCreateAppCapGate:
    """The load-bearing monitor wiring in create_app() (api/main.py) had NO direct
    test (a pre-release review flagged this as the one real coverage gap).
    Asserts cap<=1 wires the legacy single LiveSlotsPoller (byte-identical) and cap>=2
    wires the LiveResidentsSupervisor. Sync test (drives the FastAPI lifespan via
    TestClient, which starts/cancels the monitor tasks)."""

    def _make_dirs(self, tmp_path, name):
        d = tmp_path / name
        d.mkdir()
        return d

    def test_cap1_wires_legacy_poller(self, tmp_path):
        from fastapi.testclient import TestClient

        from turbohaul.api.main import create_app
        boot, runtime = _boot_runtime_multislot(
            self._make_dirs(tmp_path, "c1"), max_parallel_sidecars=1)
        app = create_app(boot, runtime, auto_start_worker=True,
                         auto_boot_reconcile=False)
        with TestClient(app):
            mgr = app.state.manager
            assert mgr._live_poller_task is not None, "cap<=1 wires the legacy poller"
            assert mgr._live_supervisor_task is None, "cap<=1 does NOT wire the supervisor"

    def test_cap2_wires_supervisor(self, tmp_path):
        from fastapi.testclient import TestClient

        from turbohaul.api.main import create_app
        boot, runtime = _boot_runtime_multislot(
            self._make_dirs(tmp_path, "c2"), max_parallel_sidecars=2)
        app = create_app(boot, runtime, auto_start_worker=True,
                         auto_boot_reconcile=False)
        with TestClient(app):
            mgr = app.state.manager
            assert mgr._live_supervisor_task is not None, "cap>=2 wires the supervisor"
            assert mgr._live_poller_task is None, "cap>=2 does NOT wire the legacy poller"


class TestIdleSaveRework:
    """Verify the idle-holder teardown reroutes through _save_slot_kv
    with thread_id_override and admission_ctx_len_override, and that the engine
    POST uses 'filename' (not 'thread_id') and metadata prompt_len reflects
    admission_ctx_len."""

    async def test_idle_holder_teardown_posts_filename_not_thread_id(self, tmp_path):
            """When _teardown_idle_holder runs, it calls _save_slot_kv which POSTs
            {'filename': '...'} to the engine, NOT {'thread_id': '...'}."""
            boot, runtime = _boot_runtime_multislot(
                tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=1
            )
            _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
            _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)

            post_payloads = []

            pid = [90000]

            def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
                pid[0] += 1
                return _fake_handle(model_tag, port, pid[0])

            async def fake_health(*a, **k):
                return True

            async def fake_sigterm(handle, **k):
                return True, "sigterm-clean"

            async def fake_vram(**k):
                return True, 100

            async def fake_complete(slot, handle):
                return {"ok": True, "model": handle.model_tag}

            mgr = TurbohaulManager(
                boot, runtime,
                spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete,
            )
            mgr.runtime.queue.safety_enabled = False

            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            import os
            os.makedirs(SLOT_SAVE_DIR, exist_ok=True)

            # Mock httpx.AsyncClient to capture POST payloads
            class _FakeSaveClient:
                def __init__(self, *args, **kwargs):
                    self.closed = False
                    self._slots_payload = [{
                        "id": 0,
                        "n_prompt_tokens": 5,
                        "id_task": 0,
                    }]

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return False

                async def get(self, url):
                    if "/slots" in url and "action=save" not in url:
                        return _FakeResp(self._slots_payload)
                    return _FakeResp({})

                async def post(self, url, json=None):
                    post_payloads.append({"url": url, "json": json})
                    return _FakeResp({"status": "ok"})

                async def aclose(self):
                    self.closed = True

            with _high_vram():
                with patch("httpx.AsyncClient", _FakeSaveClient):
                    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
                    try:
                        # Manually set up idle holder (simulating what worker_loop does)
                        import time
                        from turbohaul.slot import Slot
                        handle = _fake_handle("m1", 59500, 90001)
                        slot = Slot.new("m1", prompt="hello", thread_id="t1", admission_ctx_len=5)
                        mgr._set_idle_holder(handle, "m1", time.monotonic() + 10, "t1", 5)

                        # Verify idle holder is set up
                        assert mgr._idle_handle is not None
                        assert mgr._idle_thread_id == "t1"
                        assert mgr._idle_admission_ctx_len == 5

                        # Now trigger _teardown_idle_holder
                        await mgr._teardown_idle_holder("test_reason")

                        # Verify POST was made with 'filename' not 'thread_id'
                        save_posts = [p for p in post_payloads if "action=save" in p["url"]]
                        assert save_posts, "save POST should have been made"
                        for p in save_posts:
                            assert "filename" in p["json"], f"POST should have 'filename', got {p['json']}"
                            assert "thread_id" not in p["json"], f"POST should NOT have 'thread_id', got {p['json']}"

                    finally:
                        await mgr.shutdown()

    async def test_save_slot_kv_metadata_prompt_len_uses_admission_ctx_len(self, tmp_path):
        """_save_slot_kv with admission_ctx_len_override writes metadata with
        prompt_len = admission_ctx_len_override, not computed from prompt."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120
        )
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)

        pid = [90000]

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            pid[0] += 1
            return _fake_handle(model_tag, port, pid[0])

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(handle, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            return {"ok": True, "model": handle.model_tag}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False

        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
        import os
        os.makedirs(SLOT_SAVE_DIR, exist_ok=True)

        # Track the metadata written
        written_meta = {}

        class _FakeSaveClient:
            def __init__(self, *args, **kwargs):
                self.closed = False
                self._slots_payload = [{
                    "id": 0,
                    "n_prompt_tokens": 100,
                    "id_task": 0,
                }]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url):
                if "/slots" in url and "action=save" not in url:
                    return _FakeResp(self._slots_payload)
                return _FakeResp({})

            async def post(self, url, json=None):
                # Simulate engine creating the temp file
                if "action=save" in url and json and "filename" in json:
                    import os
                    tmp_path = os.path.join(SLOT_SAVE_DIR, json["filename"])
                    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                    with open(tmp_path, "wb") as f:
                        f.write(b"dummy kv cache data")
                return _FakeResp({"status": "ok"})

            async def aclose(self):
                self.closed = True

        with _high_vram():
            with patch("turbohaul.manager.httpx.AsyncClient", _FakeSaveClient):
                mgr._worker_task = asyncio.create_task(mgr.worker_loop())
                try:
                    # Manually trigger _save_slot_kv with override
                    from turbohaul.slot import Slot
                    slot = Slot.new("m1", prompt="x" * 1000, thread_id="t-test", admission_ctx_len=42)

                    await mgr._save_slot_kv(
                        59500, "m1", slot,
                        thread_id_override="t-test",
                        admission_ctx_len_override=42,
                    )

                    # Verify metadata was written with prompt_len = 42 (admission_ctx_len_override)
                    thread_hash = mgr._thread_hash("t-test")
                    from turbohaul.kv_policy import kv_meta_fn
                    meta_fn = kv_meta_fn("m1", 0, thread_hash, 59500)
                    meta_path = os.path.join(SLOT_SAVE_DIR, meta_fn)
                    assert os.path.exists(meta_path), "metadata file should exist"
                    import json
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    assert meta["prompt_len"] == 42, f"prompt_len should be 42 (admission_ctx_len_override), got {meta['prompt_len']}"
                    assert meta["thread_id"] == "t-test", "thread_id in metadata should match override"

                finally:
                    await mgr.shutdown()
