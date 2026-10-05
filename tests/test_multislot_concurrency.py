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
from turbohaul.live_monitor import LiveResidentsSupervisor, LiveSlotsPoller, ResidentSlotsPoller
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import SlotState
from turbohaul.slot import VramOverCommitError
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


class TestKeepAliveActiveMatchGuard:
    """Keep-alive guard, cap>=2 path (_serve_on_resident's GRACE-loop
    ACTIVE_MATCH promotion). A same-thread follow-up that omits
    keep_alive_s must not overwrite an already-stored EXPLICIT keep_alive
    intent — last-EXPLICIT-writer-wins, not last-writer-wins."""

    async def test_omitted_keep_alive_preserves_anchor_intent(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(
            tmp_path, grace_seconds=5, idle_hot_load_seconds=120,
        )
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        spawn_calls = []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m1", "anchor", thread_id="t1",
                        client_meta={"keep_alive_s": 0},
                    ),
                    timeout=5,
                )
                r = mgr._residents.get("m1")
                assert r is not None
                assert r.latest_keep_alive_s == 0, (
                    "anchor's explicit keep_alive=0 must be captured"
                )
                await asyncio.sleep(0.05)  # breather: let the GRACE loop start polling

                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m1", "followup", thread_id="t1",
                        client_meta={},  # no keep_alive_s
                    ),
                    timeout=5,
                )
                assert r.latest_keep_alive_s == 0, (
                    "ACTIVE_MATCH follow-up with omitted keep_alive_s must NOT "
                    f"reset the anchor's explicit intent; got {r.latest_keep_alive_s!r}"
                )
                assert [c["model_tag"] for c in spawn_calls].count("m1") == 1, (
                    f"expected warm ACTIVE_MATCH reuse, no second spawn; got {spawn_calls}"
                )
            finally:
                await mgr.shutdown()

    async def test_explicit_keep_alive_still_overwrites(self, tmp_path):
        """Regression guard: an EXPLICIT follow-up keep_alive_s must still win
        over the anchor's. Already correct; must stay correct."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, grace_seconds=5, idle_hot_load_seconds=120,
        )
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        spawn_calls = []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m1", "anchor", thread_id="t1",
                        client_meta={"keep_alive_s": 0},
                    ),
                    timeout=5,
                )
                r = mgr._residents.get("m1")
                assert r is not None
                assert r.latest_keep_alive_s == 0
                await asyncio.sleep(0.05)

                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m1", "followup", thread_id="t1",
                        client_meta={"keep_alive_s": 600},
                    ),
                    timeout=5,
                )
                assert r.latest_keep_alive_s == 600, (
                    "explicit follow-up keep_alive_s=600 must overwrite the "
                    f"anchor's 0; got {r.latest_keep_alive_s!r}"
                )
            finally:
                await mgr.shutdown()


class TestVramGate:
    async def test_second_co_resident_that_can_never_fit_stays_queued_not_refused(
        self, tmp_path, monkeypatch,
    ):
        """Scenario: a request pinned to a card that can never fit it. The
        request stays queued instead of being refused with a 503; this
        replaces an earlier 503-based version of the test, with the same
        real-footprint scenario: m1 pins GPU0, m2
        pins GPU1 (split_mode=none on distinct cards); GPU1 has only 10 GiB
        free so the 18 GiB m2 genuinely can never fit there -- real
        physics (per-card VRAM), not a race. m1 occupies GPU0, not GPU1,
        so nothing about m1's own state ever frees the card m2 needs;
        m2's starvation here is PERMANENT for as long as this scenario
        holds.

        Behaviour: there is no exhaustion path --
        once a request is queued, it stays queued and is not
        cancelled unless the client cancels it.
        m2 does not 503 after burning its
        defer budget -- it stays QUEUED forever instead, same as every
        other unroutable-for-now slot. This is an intentional
        operational consequence, not a bug: a request pinned
        (main_gpu, no auto_place) to a card that can never fit it, with
        nothing ever freeing that card, now hangs from the client's own
        perspective until the client disconnects, rather than getting a
        503. The design decision on the adjacent "what happens to
        a model that can never fit" question
        puts the fix for THIS shape in manifest configuration
        (auto_place / a different main_gpu pin), not in the routing layer
        refusing on the client's behalf. The backoff AND the derived cap
        are both monkeypatched small -- backoff alone is not enough: the
        cap is a real ~60-derived value at grace_seconds=0, so
        a bounded wait window here needs the cap forced small too, or the
        run never reaches exhaustion inside the
        window and this test would pass for the wrong reason
        (the wait window must actually reach
        the defer limit)."""
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_BACKOFF_S", 0.01)
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_MIN_DEFERS", 20)
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_MAX_DEFERS", 20)
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        # Real footprints; pin to distinct cards so co-residence is N1-admissible and
        # the refusal comes from the per-card VRAM budget, not the N1 topology gate.
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", expected_vram_mib=18000, split_mode="none", main_gpu=1)
        spawn_calls = []
        gate = asyncio.Event()
        mgr = TurbohaulManager(boot, runtime, **_mocks(spawn_calls, complete_gate=gate))
        # GPU0: 22 GiB free (m1's 18 GiB fits); GPU1: 10 GiB free (m2's 18 GiB does NOT).
        # safety_enabled stays True on purpose (this test exercises the
        # real gate), so the cpu leg must be forced healthy too, the same way the
        # VRAM leg already is -- otherwise the outcome depends on host load, not
        # just the VRAM budget this test is actually about. The gate's
        # cpu probe is check_cpu_util (real /proc/stat busy%), not the retired
        # load-average check -- patch _read_stat_cpu_jiffies, its own probe, not
        # os.getloadavg (that patch is inert under the current gate and silently
        # falls through to real host CPU).
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[22000, 10000]), \
                patch("turbohaul.manager._read_free_vram_all_mib", return_value=[22000, 10000]), \
                patch("turbohaul.safety._read_stat_cpu_jiffies", return_value=None):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            f2 = None
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                # Wait for the worker loop to create the m1 resident. A fixed
                # 0.2s sleep was flaky on loaded runs: when the
                # worker was slow, the assertion fired with no m1 resident and
                # spawn_calls empty — the subsequent m2 submit never triggered
                # the refusal path the test is checking.
                for _ in range(50):  # up to 1.0s
                    if mgr._residents.get("m1") is not None:
                        break
                    await asyncio.sleep(0.02)
                assert mgr._residents.get("m1") is not None
                # m1's completion is parked on the gate. Set the gate and await f1
                # BEFORE shutdown so the slot completes normally and the driver
                # teardown does not fire the "driver exited before serve" error on
                # the orphaned f1 future (the original flaky mode).
                gate.set()
                await asyncio.wait_for(f1, timeout=5)

                f2 = asyncio.create_task(mgr.submit_and_wait("m2", "b", thread_id="t2"))
                try:
                    await asyncio.wait_for(asyncio.shield(f2), timeout=2)
                except asyncio.TimeoutError:
                    pass  # expected -- still queued, no exhaustion
                else:
                    pytest.fail(
                        "m2 resolved instead of staying queued -- it can "
                        "never fit on GPU1 in this scenario"
                    )
                assert not f2.done(), (
                    "m2 must stay QUEUED forever (never refused "
                    "via the old exhaustion 503) since GPU1 can genuinely "
                    "never fit it and nothing frees it here"
                )
                assert mgr._residents.get("m2") is None
                assert [c["model_tag"] for c in spawn_calls] == ["m1"]
            finally:
                gate.set()
                if f2 is not None and not f2.done():
                    f2.cancel()
                    try:
                        await f2
                    except BaseException:
                        pass
                await mgr.shutdown()

    async def test_footprint_cpu_moe_trusts_measured_expected_vram(self, tmp_path):
        # _read_model_footprint: a cpu-moe (n_cpu_moe) manifest with a measured
        # expected_vram LOWER than gguf+kv must reserve the MEASURED value, not the
        # over-counted max(expected, gguf+kv). (end-to-end 35b reserve-gate regression.)
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
        # This test never spawns (only _read_model_footprint is called
        # below), so the safety gate is never actually reached and this line cannot
        # change today's pass/fail. Set defensively for consistency with every other
        # manager fixture in the suite -- left at the True default here, this is the
        # one shape a future reader could "fix" by copying, which is the direction
        # that would actually be wrong.
        mgr.runtime.queue.safety_enabled = False
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


class TestTwoResidentCapIdleDeadClassify:
    """Cap>=2 twin of the dead-idle attribution test: classify_idle_dead_holder's only call site
    (worker_loop's proactive sweep) is UNREACHABLE once
    ``max_parallel_sidecars >= 2`` -- worker_loop branches straight into
    _dispatch_loop above that. So the whole dead-idle attribution feature
    was dead code on any cap>=2 configuration. These tests drive the cap>=2 dispatcher
    (``_unload_teardown`` and ``_drive_resident``'s own ``finally``, the two
    exactly-once teardown-claim sites that can discover a resident's handle
    already dead) and prove IDLE_DEAD_CLASSIFY fires at either site,
    reusing the SAME line name/fields the cap<=1 sweep already uses.

    max_parallel_sidecars is set EXPLICITLY (2) in both tests -- a deployment may
    run 2 but the test environment advertises 1.
    """

    async def test_evict_teardown_classifies_externally_discovered_death(
        self, tmp_path, caplog,
    ):
        """Drives the ``_unload_teardown`` fallthrough directly: a resident goes
        IDLE_EVICTABLE, its handle is killed, then teardown is triggered from
        OUTSIDE the driver task (mirrors the existing gate-1 test's
        own technique of calling ``_begin_unload_locked`` directly) -- since the
        driver task itself is parked elsewhere, ``_unload_teardown`` (the
        spawned background task) is the one that wins the torn_down claim."""
        import logging
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120,
        )
        _seed_manifest(boot, "m1")
        spawn_calls, sigterm_calls = [], []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                with caplog.at_level(logging.WARNING, logger="turbohaul.manager"):
                    await asyncio.wait_for(
                        mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5
                    )
                    await asyncio.sleep(0.15)  # m1 -> IDLE_EVICTABLE
                    r1 = mgr._residents.get("m1")
                    assert r1 is not None and r1.state is ResidentState.IDLE_EVICTABLE
                    held = r1.handle
                    engine_log = tmp_path / "two_resident_cap_fault_evict.log"
                    engine_log.write_text(
                        "H.01.000.000 I main: loading model\n"
                        "H.01.480.000 I slot: context checkpoint eviction\n"
                        "H.01.481.000 E ggml_backend_cuda_cpy_tensor_async: "
                        "CUDA error: unspecified launch failure\n"
                    )
                    held.engine_log_path = str(engine_log)
                    held.proc.poll.return_value = -11  # dead

                    async with mgr._registry_lock:
                        mgr._begin_unload_locked(r1)

                    for _ in range(200):
                        await asyncio.sleep(0.02)
                        if mgr._residents.get("m1") is None:
                            break
                lines = [
                    rec.message for rec in caplog.records
                    if "IDLE_DEAD_CLASSIFY" in rec.message
                ]
                assert lines, (
                    "cap>=2 _unload_teardown never emitted IDLE_DEAD_CLASSIFY "
                    "on an externally-discovered dead handle"
                )
                assert any("class=fault_signature" in l for l in lines)
                assert any("CUDA error" in l for l in lines)
                assert any(f"port={held.port}" in l for l in lines)
                assert any("model=m1" in l for l in lines)

                # Recovery unchanged: resident torn down, and sigterm must NOT
                # fire on a handle that was already dead (nothing to kill) --
                # classification is attribution only, not a behavior change.
                assert mgr._residents.get("m1") is None, "dead resident never torn down"
                assert "m1" not in sigterm_calls, (
                    "sigterm must not fire on an already-dead handle -- "
                    "classification must not perturb teardown/recovery"
                )
            finally:
                await mgr.shutdown()

    async def test_drive_resident_finally_classifies_natural_idle_timeout_death(
        self, tmp_path, caplog,
    ):
        """Drives ``_drive_resident``'s OWN finally fallthrough: same setup, but
        teardown is triggered by letting the real idle-timeout elapse INSIDE
        the driver's own loop (no external _begin_unload_locked call) -- the
        driver task's own finally wins the torn_down claim in this shape
        (no scheduling gap for a competing background task to win it first),
        exercising the OTHER of the two capture sites."""
        import logging
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=1,
        )
        _seed_manifest(boot, "m1")
        spawn_calls, sigterm_calls = [], []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                with caplog.at_level(logging.WARNING, logger="turbohaul.manager"):
                    await asyncio.wait_for(
                        mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5
                    )
                    await asyncio.sleep(0.1)  # m1 -> IDLE_EVICTABLE (well under 1s window)
                    r1 = mgr._residents.get("m1")
                    assert r1 is not None and r1.state is ResidentState.IDLE_EVICTABLE
                    held = r1.handle
                    engine_log = tmp_path / "two_resident_cap_fault_finally.log"
                    engine_log.write_text(
                        "H.01.000.000 I main: loading model\n"
                        "H.01.480.000 I slot: context checkpoint eviction\n"
                        "H.01.481.000 E ggml_backend_cuda_cpy_tensor_async: "
                        "CUDA error: unspecified launch failure\n"
                    )
                    held.engine_log_path = str(engine_log)
                    held.proc.poll.return_value = -11  # dead

                    # Let the real idle_window (1s) elapse naturally inside
                    # the driver's own wait_for -- no direct manager call.
                    for _ in range(200):
                        await asyncio.sleep(0.02)
                        if mgr._residents.get("m1") is None:
                            break
                lines = [
                    rec.message for rec in caplog.records
                    if "IDLE_DEAD_CLASSIFY" in rec.message
                ]
                assert lines, (
                    "cap>=2 _drive_resident's finally never emitted "
                    "IDLE_DEAD_CLASSIFY on a naturally-discovered dead handle "
                    "at idle-timeout"
                )
                assert any("class=fault_signature" in l for l in lines)
                assert any(f"port={held.port}" in l for l in lines)
                assert any("model=m1" in l for l in lines)
                assert mgr._residents.get("m1") is None, "dead resident never torn down"
                assert "m1" not in sigterm_calls, (
                    "sigterm must not fire on an already-dead handle"
                )
            finally:
                await mgr.shutdown()


class TestVramFitEvictAndQueue:
    """Make-room decoupled from the count cap — driven by VRAM-fit."""

    async def test_vram_full_idle_evictable_queues_then_evicts_then_loads(self, tmp_path):
        """(a) 1 resident under cap>=2 (count cap NOT the blocker), VRAM full, main
        IDLE_EVICTABLE. A request for model B does NOT silently fail: B queues, the
        idle main is evicted, B loads + serves. No future-fail."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120)
        # Same card: m2 can co-fit ONLY after m1 is evicted.
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        spawn_calls, sigterm_calls = [], []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))

        def live_probe():
            # gpu0 full while m1 resident (m1 in _residents); frees once evicted.
            m1 = mgr._residents.get("m1")
            return [4000, 4000] if m1 is not None else [30000, 30000]

        with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=live_probe):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                # m1 loads + completes (not gated) -> IDLE_EVICTABLE
                await asyncio.wait_for(mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.sleep(0.15)
                r1 = mgr._residents.get("m1")
                assert r1 is not None and r1.state is ResidentState.IDLE_EVICTABLE
                # m2 can't co-fit -> queue + evict idle m1 + load m2 (NO exception)
                slot, res = await asyncio.wait_for(
                    mgr.submit_and_wait("m2", "b", thread_id="t2"), timeout=10)
                assert res == {"ok": True, "model": "m2"}
                assert "m1" in sigterm_calls, "idle m1 evicted to make room"
                assert mgr._residents.get("m2") is not None
                assert mgr._residents.get("m1") is None
                assert [c["model_tag"] for c in spawn_calls] == ["m1", "m2"]
            finally:
                await mgr.shutdown()

    async def test_genuinely_unfittable_stays_queued_not_silent_200(self, tmp_path, monkeypatch):
        """This test replaces an earlier one whose 503 outcome was
        removed by an intentional behaviour change; the scenario is
        unchanged: the request is neither refused nor silently served.
        Scenario: a model too big for any card,
        nothing idle-evictable. Behaviour: once a request is queued,
        it stays queued and is not cancelled unless the
        client cancels it --
        the slot does not exhaust a bounded budget and fail with a
        VramOverCommitError. It stays QUEUED forever instead -- still
        never a silent success/None either. The design decision on the
        adjacent "what happens to a model that can never fit" question
        puts the actual fix for this shape in
        manifest configuration (offload knobs) or a real OOM, never a
        queue-layer refusal. The backoff AND the derived cap are both
        monkeypatched small -- backoff alone is not enough: the derived
        cap is a real ~60-derived value at grace_seconds=0, so a bounded
        wait window needs the cap forced small too, or the run never
        reaches exhaustion inside the window and the test would pass for
        the wrong reason (the wait window must actually reach the defer
        limit).
        """
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_BACKOFF_S", 0.01)
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_MIN_DEFERS", 20)
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_MAX_DEFERS", 20)
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=2)
        _seed_manifest(boot, "big", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        spawn_calls = []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls))
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[4000, 4000]):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            f = None
            try:
                f = asyncio.create_task(mgr.submit_and_wait("big", "x", thread_id="t1"))
                try:
                    await asyncio.wait_for(asyncio.shield(f), timeout=2)
                except asyncio.TimeoutError:
                    pass  # expected -- still queued, no exhaustion
                else:
                    pytest.fail(
                        "the slot resolved instead of staying queued -- it "
                        "can never fit any card in this scenario"
                    )
                assert not f.done(), (
                    "an unfittable-anywhere slot must stay "
                    "QUEUED forever now, never refused via the old "
                    "exhaustion 503"
                )
                assert mgr._residents.get("big") is None
                assert spawn_calls == [], "never spawned an over-committing sidecar"
            finally:
                if f is not None and not f.done():
                    f.cancel()
                    try:
                        await f
                    except BaseException:
                        pass
                await mgr.shutdown()

    async def test_count_cap_path_unchanged(self, tmp_path):
        """(c) With ample VRAM and the registry at the COUNT cap, make-room still
        evicts the IDLE resident (not the busy one) and reserves in-band — the
        previous count-cap behavior, byte-preserved."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)
        _seed_manifest(boot, "m3", split_mode="none", main_gpu=1)
        spawn_calls, sigterm_calls = [], []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks(
            spawn_calls, sigterm_calls=sigterm_calls, complete_gate=gate, gate_model="m1"))
        with _high_vram():  # VRAM never the blocker -> COUNT cap is the trigger
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                await asyncio.sleep(0.2)  # m1 ACTIVE, busy on gate
                await asyncio.wait_for(mgr.submit_and_wait("m2", "b", thread_id="t2"), timeout=5)
                await asyncio.sleep(0.15)  # m2 -> IDLE_EVICTABLE ; registry now == cap
                f3 = asyncio.create_task(mgr.submit_and_wait("m3", "c", thread_id="t3"))
                await asyncio.sleep(0.4)
                assert mgr._residents.get("m2") is None, "idle m2 evicted (count cap)"
                assert mgr._residents.get("m1") is not None, "busy m1 NOT evicted"
                assert "m2" in sigterm_calls and "m1" not in sigterm_calls
                assert mgr._residents.get("m3") is not None, "m3 took the freed slot"
                gate.set()
                await asyncio.wait_for(asyncio.gather(f1, f3), timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()

    async def test_hit_warm_reuse_and_flip_unchanged(self, tmp_path):
        """(d) HIT / lost-slot-race path preserved: a follow-up to an
        IDLE_EVICTABLE resident is served via warm HIT reuse (flip IDLE->ACTIVE +
        inbox.put co-located under the lock) with NO re-spawn and NO evict/probe."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        spawn_calls, sigterm_calls = [], []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.sleep(0.15)
                r = mgr._residents.get("m1")
                assert r is not None and r.state is ResidentState.IDLE_EVICTABLE
                # follow-up HIT-routes: reused, flipped back to ACTIVE, never re-spawned
                await asyncio.wait_for(mgr.submit_and_wait("m1", "b", thread_id="t2"), timeout=5)
                assert [c["model_tag"] for c in spawn_calls].count("m1") == 1, "warm reuse, no re-spawn"
                assert sigterm_calls == [], "HIT path never evicts"
            finally:
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
    async def test_layer_split_coresidence_miss_stays_queued_not_refused(
        self, tmp_path, monkeypatch,
    ):
        """This test replaces an earlier one whose 503 outcome was
        removed by an intentional behaviour change; the scenario is
        unchanged and is a split-mode topology scenario:
        co-residence is
        supported ONLY for single-GPU-pinned (split_mode='none') models on
        DISTINCT cards. A layer-split 2nd model spans every card, so it is
        refuse-blinded until per-card layer-split accounting lands -- EVEN
        with ample free VRAM (the refusal is topology, not budget). This
        structural refusal is inside the SAME `_vram_admits_locked` gate
        `_reserve_and_start_locked` re-checks (manager.py, "Structural
        refusals -- crediting VRAM cannot change them"), so the
        refuse-to-requeue conversion applies here too, uniformly -- there
        is no carve-out in the policy for a structural vs. a
        VRAM-headroom miss.

        Behaviour: m2 never exhausts the evict-pending budget and
        503s. It stays QUEUED instead, for as long as m1 (never idle-
        evictable here, gated) stays resident -- a genuinely permanent
        block for this scenario's own duration, same as the
        general rule that a queued request stays queued. Correctness
        does not depend on m1's own eventual departure firing the
        requeue wake signal specifically for a topology change --
        the requeue loop's own periodic backoff re-checks
        _vram_admits_locked regardless, the wake is a latency optimization
        only. The backoff AND the derived cap are both monkeypatched
        small -- backoff alone is not enough: the derived cap is a real
        ~60-derived value at grace_seconds=0, so a bounded wait window
        needs the cap forced small too, or the run
        never reaches exhaustion inside the window (the wait window
        must actually reach the defer limit).
        """
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_BACKOFF_S", 0.01)
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_MIN_DEFERS", 20)
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_MAX_DEFERS", 20)
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="layer", main_gpu=0)  # spans all cards
        spawn_calls = []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, complete_gate=gate))
        with _high_vram():  # plenty free on both cards -> refusal is N1, not budget
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            f2 = None
            try:
                f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                await asyncio.sleep(0.2)
                assert mgr._residents.get("m1") is not None
                f2 = asyncio.create_task(mgr.submit_and_wait("m2", "b", thread_id="t2"))
                try:
                    await asyncio.wait_for(asyncio.shield(f2), timeout=2)
                except asyncio.TimeoutError:
                    pass  # expected -- still queued, no exhaustion
                else:
                    pytest.fail(
                        "m2 resolved instead of staying queued -- it can "
                        "never co-reside with a layer-split-incompatible "
                        "sibling still resident"
                    )
                assert not f2.done(), (
                    "a topology-blocked co-residence miss must "
                    "stay QUEUED forever too, never refused via the old "
                    "exhaustion 503 -- the requeue conversion applies "
                    "uniformly to every _vram_admits_locked miss, not just "
                    "the VRAM-headroom race"
                )
                assert mgr._residents.get("m2") is None, "layer-split 2nd model still not admitted"
                gate.set()
                await asyncio.wait_for(f1, timeout=5)
            finally:
                gate.set()
                if f2 is not None and not f2.done():
                    f2.cancel()
                    try:
                        await f2
                    except BaseException:
                        pass
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
# Per-resident live monitor + status residents[]/vram[] +
# shutdown-sweep + the waitpid reap and bg_tasks drain.
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
        (the observer-leak guard for resource lifecycles)."""
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


class TestLiveMonitorOneResidentCapPublishesRealState:
    """Pins the PUBLISHED-STATE contract --
    distinct from TestCreateAppCapGate below, which pins the WIRING choice
    (which class create_app's lifespan constructs at each cap). This class
    proves what the two classes api/main.py chooses BETWEEN actually publish,
    given a REAL, ACTIVELY-SERVING resident. Not driven through create_app:
    create_app takes no spawn_fn/health_fn DI seam (TurbohaulManager gets
    constructed with only complete_fn overridden), so a real subprocess would
    be needed to get a resident to ACTIVE that way, and nothing else in this
    test suite does that for this concern. Instead this uses the SAME
    DI-based harness TestSupervisorRunLoop above already uses: a real
    TurbohaulManager, a real submit_and_wait blocked ACTIVE on a gate, and a
    fake httpx client standing in for a real llama.cpp /slots response.

    Both tests below construct the SAME classes (LiveSlotsPoller,
    LiveResidentsSupervisor), imported from the same module, with the SAME
    constructor api/main.py's lifespan uses -- not test doubles. Read
    together with TestCreateAppCapGate's cap-1 test (which proves cap<=1
    SELECTS LiveResidentsSupervisor through the real create_app
    lifespan), these two facts ARE the contract: (wiring)
    cap<=1 selects
    LiveResidentsSupervisor, not LiveSlotsPoller; (publication, this class) LiveSlotsPoller
    publishes 'idle' regardless of a live resident, LiveResidentsSupervisor
    publishes the real generation."""

    async def test_legacy_poller_reports_idle_despite_a_live_resident_RED(
        self, tmp_path,
    ):
        """Legacy cap<=1 wiring, demonstrated:
        LiveSlotsPoller -- the class create_app's lifespan constructed at
        cap<=1 -- reads mgr._active_slot / mgr._active_handle,
        which nothing writes any more (see _legacy_wire_live_poller,
        api/main.py, for the full trace: their only writer is
        _process_slot, whose only caller has been unreachable since
        the retirement of the single-slot path). With a REAL resident ACTIVELY SERVING (driven via
        submit_and_wait, blocked in complete_fn on a gate -- non-vacuity
        asserted below BEFORE touching the monitor), this publishes 'idle'
        -- the failure mode of the legacy wiring."""
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=1)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        spawn_calls = []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, complete_gate=gate))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(
                    mgr.submit_and_wait("m1", "a", thread_id="t1")
                )
                await asyncio.sleep(0.2)  # m1 -> ACTIVE, blocked in complete on gate
                # NON-VACUITY: a real resident really is live and serving
                # before we assert anything about the monitor.
                residents = mgr._model_residents()
                assert len(residents) == 1 and residents[0].model_tag == "m1", (
                    "setup did not produce a live serving resident -- the "
                    "rest of this test would prove nothing"
                )
                assert residents[0].state == ResidentState.ACTIVE
                assert residents[0].active_slot is not None
                assert residents[0].handle is not None
                # the manager-global fields the legacy poller reads: proven
                # None even though a resident is genuinely live, matching
                # the failure mode's own root cause, not merely asserted.
                assert mgr._active_slot is None
                assert mgr._active_handle is None

                with patch(
                    "turbohaul.live_monitor.httpx.AsyncClient",
                    return_value=_FakeSlotsClient(_slots_processing()),
                ):
                    poller = LiveSlotsPoller(mgr, interval_s=1.0)
                    await poller._tick()
                    await poller._client.aclose()
                assert mgr.live_generation["state"] == "idle", (
                    f"expected the legacy failure mode (idle while a resident is live), "
                    f"got {mgr.live_generation!r}"
                )
            finally:
                gate.set()
                await mgr.shutdown()

    async def test_supervisor_reports_real_state_for_a_live_resident_GREEN(
        self, tmp_path,
    ):
        """Target state: LiveResidentsSupervisor -- the
        class create_app's lifespan constructs at EVERY cap, including
        cap<=1 -- walks mgr._model_residents() (the real
        registry), not the dead manager-global fields. Same live-resident
        setup as the legacy-wiring test above; the ONLY variable is which observer
        class is driven."""
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=1)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        spawn_calls = []
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, complete_gate=gate))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(
                    mgr.submit_and_wait("m1", "a", thread_id="t1")
                )
                await asyncio.sleep(0.2)
                residents = mgr._model_residents()
                assert len(residents) == 1 and residents[0].model_tag == "m1", (
                    "setup did not produce a live serving resident -- the "
                    "rest of this test would prove nothing"
                )
                assert residents[0].state == ResidentState.ACTIVE

                with patch(
                    "turbohaul.live_monitor.httpx.AsyncClient",
                    return_value=_FakeSlotsClient(_slots_processing()),
                ):
                    sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
                    await sup._tick()
                    # Assert BEFORE any teardown: _close_all() deliberately
                    # clears mgr.live_generations / mgr.live_generation as
                    # its own documented post-shutdown behavior (so a
                    # dangling stream can't anchor-follow a dead generation)
                    # -- calling it first would wipe the very state under
                    # test, not a race.
                    gen = mgr.live_generations.get("m1")
                    assert gen is not None, (
                        "supervisor published nothing for the live resident"
                    )
                    assert gen["state"] != "idle", (
                        f"expected REAL state for a live resident, got {gen!r}"
                    )
                    assert mgr.live_generation is not None
                    assert mgr.live_generation["state"] != "idle"
                    await sup._close_all()
            finally:
                gate.set()
                await mgr.shutdown()


class TestCreateAppCapGate:
    """The load-bearing monitor wiring in create_app() (api/main.py) had NO direct
    test (the one real coverage gap in the monitor wiring).
    Asserts cap<=1 wires the legacy single LiveSlotsPoller (byte-identical) and cap>=2
    wires the LiveResidentsSupervisor. Sync test (drives the FastAPI lifespan via
    TestClient, which starts/cancels the monitor tasks)."""

    def _make_dirs(self, tmp_path, name):
        d = tmp_path / name
        d.mkdir()
        return d

    def test_one_resident_cap_wires_supervisor_not_legacy_poller(self, tmp_path):
        """The cap split is gone: EVERY cap, including cap<=1, wires
        LiveResidentsSupervisor and never the legacy LiveSlotsPoller.
        The legacy poller publishes 'idle' forever, because LiveSlotsPoller
        reads mgr._active_slot / mgr._active_handle, which nothing has
        written since the single-slot path was retired
        (see _legacy_wire_live_poller, api/main.py, for
        the full trace) -- so a cap-based split would never be a genuine
        behavioral choice, only an accident of build order (the same
        shape as the neighboring cap-gated
        task). The test is named for the wiring that holds under
        every legal cap, not for a
        cap-gated choice. A cap<=1 assertion that the legacy poller is wired would
        describe a wiring that does not exist under any legal cap,
        so this test asserts the opposite: the supervisor is wired and
        the legacy poller task is absent at every cap.
        The two assertions below pin exactly that wiring
        (supervisor present, legacy poller absent).

        This test pins the WIRING CHOICE ONLY (which class create_app's
        lifespan constructs) -- it does NOT duplicate
        TestLiveMonitorOneResidentCapPublishesRealState above, which pins the
        PUBLISHED STATE a live resident produces once wired. Different
        claims: this one can be fully satisfied by a task-object existing
        with no resident ever spawned; that one requires a REAL resident
        actively serving before it asserts anything about the monitor.

        A failure here means create_app wires the wrong observer at cap<=1:
        the supervisor task must exist and the legacy poller task must be
        absent (`mgr._live_supervisor_task is not None` and
        `mgr._live_poller_task is None`). The check is on the wiring itself,
        not on a collection or setup error,
        so a red result always points at the
        create_app wiring."""
        from fastapi.testclient import TestClient

        from turbohaul.api.main import create_app
        boot, runtime = _boot_runtime_multislot(
            self._make_dirs(tmp_path, "c1"), max_parallel_sidecars=1)
        app = create_app(boot, runtime, auto_start_worker=True,
                         auto_boot_reconcile=False)
        with TestClient(app):
            mgr = app.state.manager
            assert mgr._live_supervisor_task is not None, (
                "cap<=1 wires the supervisor (no more cap split)"
            )
            assert mgr._live_poller_task is None, (
                "cap<=1 does NOT wire the legacy poller any more"
            )

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


class TestOrphanSlotDisposition:
    """Driver teardown must never leave a caller waiting on a future nobody will
    resolve. The disposition table is written as 'states the serve path owns'
    rather than 'states to fail', so an unrecognised state fails LOUDLY instead
    of silently hanging the caller."""

    def test_served_states_are_left_alone(self):
        from turbohaul.manager import _SERVED_SLOT_STATES, _orphan_slot_disposition

        assert _SERVED_SLOT_STATES, "served-state set must not be empty"
        for state in _SERVED_SLOT_STATES:
            should_fail, unexpected = _orphan_slot_disposition(state)
            assert should_fail is False, f"{state.value} is served; must not be failed"
            assert unexpected is False, f"{state.value} is a known state"

    def test_pre_serve_states_fail_quietly(self):
        """A driver that died before serving must fail its anchor slot - this is
        the case the teardown exists for - without logging a table warning."""
        from turbohaul.manager import _orphan_slot_disposition

        for state in (SlotState.STAGED, SlotState.LOADING, SlotState.LOADING_FAIL):
            should_fail, unexpected = _orphan_slot_disposition(state)
            assert should_fail is True, f"{state.value} never served; must be failed"
            assert unexpected is False, f"{state.value} is a known pre-serve state"

    def test_unrecognised_states_fail_loudly(self):
        """States outside both tables still fail, and are flagged so the
        occurrence reaches the log rather than disappearing."""
        from turbohaul.manager import _orphan_slot_disposition

        for state in (SlotState.RECEIVED, SlotState.ACCEPT_BUFFER, SlotState.COLD):
            should_fail, unexpected = _orphan_slot_disposition(state)
            assert should_fail is True, f"{state.value} must fail, not hang"
            assert unexpected is True, f"{state.value} should be flagged for the log"

    def test_no_state_can_silently_hang(self):
        """THE INVARIANT. Every member of the enum - including any added later -
        must either be an explicitly-served state or be failed. A state that is
        neither would leave the completion future unresolved forever."""
        from turbohaul.manager import _SERVED_SLOT_STATES, _orphan_slot_disposition

        for state in SlotState:
            should_fail, _ = _orphan_slot_disposition(state)
            assert should_fail or state in _SERVED_SLOT_STATES, (
                f"{state.value} is neither failed nor explicitly served - a slot "
                f"in this state would hang its caller at driver teardown"
            )
def test_over_eviction_guard_is_per_card_accounting_no_fixed_settle():
    """Over-eviction guard reconciliation: the guard is the per-card
    pending-reclaim accounting, NOT a fixed settle timer. A fixed settle
    would BLOCK a proven-necessary multi-victim eviction into
    a false 503, so there is no settle timer and per-card accounting governs. (The teardown margin that
    the defer budget is sized against stays intact.)"""
    from turbohaul import manager as m
    assert not hasattr(m, "_EVICT_SETTLE_S"), "fixed settle timer must be gone (per-card accounting governs eviction)"
    assert m._VRAM_DEFER_TEARDOWN_MARGIN_S >= 30.0


def test_capacity_error_shape_unified():
    """The streaming SSE error frame and the non-streaming 503 body
    carry the SAME nested capacity-error shape (error.{type,cause,message,retry_after})
    so a client has ONE parse rule."""
    import json
    from turbohaul.api.chat_completion import (
        _stream_error_frame, capacity_unavailable_error_body,
    )
    e = VramOverCommitError("no room", retry_after_s=10)
    body_dict = capacity_unavailable_error_body(e)
    frame = _stream_error_frame(
        body_dict["type"], body_dict["message"],
        cause=body_dict["cause"], retry_after=body_dict["retry_after"])
    body = json.loads(frame.decode().split("data: ", 1)[1].strip())
    stream_err = body["error"]
    # Build the non-stream half from THE SAME production function the
    # routes call (capacity_unavailable_error_body -- openai_chat_completions,
    # ollama_chat, embeddings via _raise_capacity_or_routing_failure), NOT from
    # a literal.
    #
    # ⛔ WHY THE HELPER IS USED, because a literal looks identical and is not: if this
    # assertion compared the frame against a HARDCODED DICT LITERAL
    # written in the test, annotated "exactly what openai_chat_completions /
    # ollama_chat build". That annotation would be a claim, not a link -- nothing tied
    # the literal to the routes. Such a comparison is tautological:
    # renaming retry_after -> retryAfter inside the real shared
    # helper would drift EVERY non-stream 503 in the app, and this test would still PASS.
    # A contract test that cannot fail when the contract breaks is not enforcing
    # the contract. Built from the helper, the same mutation fails it.
    nonstream_err = capacity_unavailable_error_body(e)
    assert stream_err["type"] == nonstream_err["type"] == "capacity_unavailable"
    assert set(stream_err) == set(nonstream_err)  # identical inner shape both sides
    assert stream_err == nonstream_err  # and identical VALUES, not just key sets
    assert stream_err["retry_after"] == 10
    assert stream_err["cause"] == "vram_over_commit"


class TestMultislotConcurrencyFixes:
    """The 6 turbohaul capacity and eviction fixes."""

    async def test_count_cap_overcommit_defers_not_inband_reserve(self, tmp_path):
        """The make-room DECISION under a model-COUNT cap AND a cross-resident VRAM
        over-commit routes to the evict-pending QUEUE — NOT the in-band reserve that
        503s on the async-free race and kills a warm resident. Spy on the
        two make-room paths and drive one _route_or_reserve pass with the registry at
        the count cap (2 idle residents) + gpu0 over-committed for m3."""
        from types import SimpleNamespace
        import time as _t
        from turbohaul.manager import Resident
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=2)
        _seed_manifest(boot, "m3", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime, **_mocks([]))
        now = _t.monotonic()
        # Two IDLE residents -> registry AT the count cap (2). m1 is on the TARGET card
        # (gpu0); mA is off-card (gpu1) and older/LRU-adjacent to prove card-awareness.
        mgr._residents["m1"] = Resident(
            model_tag="m1", state=ResidentState.IDLE_EVICTABLE, main_gpu=0,
            split_mode="none", reserved_need_mib=18000, last_active_monotonic=now - 10)
        mgr._residents["mA"] = Resident(
            model_tag="mA", state=ResidentState.IDLE_EVICTABLE, main_gpu=1,
            split_mode="none", reserved_need_mib=18000, last_active_monotonic=now - 20)
        reserved, deferred, evicted = [], [], []

        async def spy_reserve(slot):
            reserved.append(slot.model_tag)

        def spy_defer(slot, *, evict_pending=False):
            deferred.append((slot.model_tag, evict_pending))

        def spy_evict(r):
            evicted.append(r.model_tag)

        mgr._reserve_and_start_locked = spy_reserve
        mgr._defer_unroutable = spy_defer
        mgr._begin_unload_locked = spy_evict
        # fastlane=None mirrors the real Slot
        # dataclass's own default -- _route_or_reserve now reads it
        # unconditionally via _shield_carveout_tags before this test's own
        # scope even begins.
        # fastlane_relocated_main_gpu/_split_mode=None
        # mirror the real Slot dataclass's own new defaults, same reason --
        # _route_or_reserve now reads the first one unconditionally right
        # after placement resolution, before this test's own scope begins.
        slot = SimpleNamespace(
            model_tag="m3", fastlane=None,
            fastlane_relocated_main_gpu=None, fastlane_relocated_split_mode=None,
        )
        # gpu0 over-committed for m3 (4000 < 18000); gpu1 ample.
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[4000, 30000]):
            await mgr._route_or_reserve(slot)
        assert reserved == [], "must NOT reserve in-band on an async-free over-commit"
        assert deferred == [("m3", True)], "routed to the evict-pending QUEUE"
        assert evicted == ["m1"], "evicted the ON-CARD victim (gpu0), never off-card mA"

    async def test_card_aware_never_evicts_off_card(self, tmp_path, monkeypatch):
        """This test replaces the second half of an earlier test whose 503
        outcome was removed by an intentional behaviour change.
        Card-awareness itself
        is the subject: the target card (gpu1)
        is full; the only idle victim (m1) is on gpu0. Card-aware selection
        must NOT evict the off-card m1 (it wouldn't relieve gpu1).

        What matters is the OUTCOME for m2: there is no
        evict-pending exhaustion path -- once a request is queued, it stays
        queued and is not cancelled unless the client
        cancels it -- so m2 does not
        503 once its budget would have run out. It stays QUEUED
        instead, genuinely forever in this scenario, since nothing here
        ever frees gpu1 (card-awareness forbids evicting the only idle resident because
        it is off-card). The warm off-card resident still survives
        untouched -- that half of the card-awareness guarantee is exactly as
        load-bearing as the rest of the test. The backoff AND the derived cap are both
        monkeypatched small -- backoff alone is not enough: the derived
        cap is a real ~60-derived value at grace_seconds=0, so a bounded
        wait window needs the cap forced small too, or the run never
        reaches exhaustion inside the window and the test would pass for the
        wrong reason (the wait window must actually reach the defer limit).
        """
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_BACKOFF_S", 0.01)
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_MIN_DEFERS", 15)
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_MAX_DEFERS", 15)
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", expected_vram_mib=18000, split_mode="none", main_gpu=1)
        spawn_calls, sigterm_calls = [], []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        # gpu0 ample (m1 fits + idle), gpu1 full (m2's 18000 can't fit its 4000 free).
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[30000, 4000]):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            f2 = None
            try:
                await asyncio.wait_for(mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.sleep(0.15)
                assert mgr._residents["m1"].state is ResidentState.IDLE_EVICTABLE
                f2 = asyncio.create_task(mgr.submit_and_wait("m2", "b", thread_id="t2"))
                try:
                    await asyncio.wait_for(asyncio.shield(f2), timeout=2)
                except asyncio.TimeoutError:
                    pass  # expected -- still queued, no exhaustion
                else:
                    pytest.fail(
                        "m2 resolved instead of staying queued -- gpu1 "
                        "never frees in this scenario"
                    )
                assert not f2.done(), (
                    "m2 must stay QUEUED forever now, never "
                    "refused via the old exhaustion 503"
                )
                assert "m1" not in sigterm_calls  # never evicted the OFF-CARD idle m1
                assert mgr._residents.get("m1") is not None  # still warm + resident
            finally:
                if f2 is not None and not f2.done():
                    f2.cancel()
                    try:
                        await f2
                    except BaseException:
                        pass
                await mgr.shutdown()

    async def test_pending_reclaim_bump_credit_and_drop(self, tmp_path):
        """_begin_unload_locked bumps per-card pending-reclaim; the EVICT-decision
        fit check credits it (so N concurrent misses don't each evict a resident) while
        the ACTUAL reserve gate does NOT; and _unload_teardown's finally drops it."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        spawn_calls, sigterm_calls = [], []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        with _high_vram():  # [80000, 80000]
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.sleep(0.15)
                r1 = mgr._residents.get("m1")
                assert r1 is not None and r1.state is ResidentState.IDLE_EVICTABLE
                need_r1 = r1.reserved_need_mib
                assert need_r1 > 0
                assert mgr._pending_reclaim_mib.get(0, 0) == 0  # nothing pending yet
                async with mgr._registry_lock:
                    mgr._begin_unload_locked(r1)
                    # bumped by the victim's reserved_need on its card
                    assert mgr._pending_reclaim_mib.get(0, 0) == need_r1
                    # a need above real free (80000) but within free + credit: the
                    # ACTUAL reserve gate refuses; the EVICT-decision gate admits.
                    big = 90000
                    assert not mgr._vram_admits_locked(big, 1, 0, "none")
                    assert mgr._vram_admits_locked(big, 1, 0, "none", credit_pending=True)
                # the detached teardown's finally drops the credit
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    if mgr._pending_reclaim_mib.get(0, 0) == 0:
                        break
                assert mgr._pending_reclaim_mib.get(0, 0) == 0  # dropped on teardown
                assert "m1" in sigterm_calls
            finally:
                await mgr.shutdown()

    async def test_confirmed_cleared_drops_full_credit(self, tmp_path):
        """Pass case: _vram_verify confirms cleared (>=90% drop)
        -> the FULL credited amount is dropped, same end-state as the sibling
        pending-reclaim test above, but EXPLICITLY gated on a real verify call instead
        of firing unconditionally right after reap (_vram_verify must be
        called from _unload_teardown -- verify_calls records it)."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        spawn_calls, sigterm_calls = [], []
        verify_calls = []

        async def fake_vram_confirmed(*, expected_drop_mib, timeout_s, device_index=0):
            verify_calls.append(device_index)
            return True, 100  # confirmed cleared

        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        mgr._vram_verify = fake_vram_confirmed
        mgr._gpu_used_mib = lambda *, device_index=0: 18100  # pre-teardown snapshot
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.sleep(0.15)
                r1 = mgr._residents.get("m1")
                need_r1 = r1.reserved_need_mib
                async with mgr._registry_lock:
                    mgr._begin_unload_locked(r1)
                    assert mgr._pending_reclaim_mib.get(0, 0) == need_r1
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    if mgr._pending_reclaim_mib.get(0, 0) == 0:
                        break
                assert mgr._pending_reclaim_mib.get(0, 0) == 0
                assert verify_calls == [0], (
                    f"expected _vram_verify called once for card 0; got {verify_calls}"
                )
            finally:
                await mgr.shutdown()

    async def test_timeout_reconciles_to_measured_delta(self, tmp_path):
        """Verify TIMES OUT (cleared_ok=False) with
        only a PARTIAL measured drop. The finally must not drop the
        FULL credit regardless of whether VRAM actually cleared (that would
        over-release). Only the measured delta is released; the
        un-freed remainder stays honestly credited -- and (a consequence of
        that) the EVICT-decision gate still sees that
        remainder as legitimately pending, not zeroed-out early."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        spawn_calls, sigterm_calls = [], []
        SHORTFALL = 500
        PRE_SNAPSHOT = 50000
        cell = {}

        def fake_gpu_used_mib(*, device_index=0):
            return PRE_SNAPSHOT

        async def fake_vram_timeout(*, expected_drop_mib, timeout_s, device_index=0):
            need = cell["need_r1"]
            current = PRE_SNAPSHOT - (need - SHORTFALL)  # measured delta = need - SHORTFALL
            return False, current

        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        mgr._vram_verify = fake_vram_timeout
        mgr._gpu_used_mib = fake_gpu_used_mib
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.sleep(0.15)
                r1 = mgr._residents.get("m1")
                need_r1 = r1.reserved_need_mib
                cell["need_r1"] = need_r1
                async with mgr._registry_lock:
                    mgr._begin_unload_locked(r1)
                    assert mgr._pending_reclaim_mib.get(0, 0) == need_r1
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    if mgr._pending_reclaim_mib.get(0, 0) <= SHORTFALL:
                        break
                assert mgr._pending_reclaim_mib.get(0, 0) == SHORTFALL, (
                    f"expected exactly {SHORTFALL} MiB left honestly credited "
                    f"(the un-measured remainder); got "
                    f"{mgr._pending_reclaim_mib.get(0, 0)}"
                )
                # The remaining SHORTFALL must still read as a
                # legitimate pending credit, not silently vanished.
                async with mgr._registry_lock:
                    assert mgr._vram_admits_locked(
                        SHORTFALL, 1, 0, "none", credit_pending=True
                    ), "the remaining SHORTFALL must still read as credited"
            finally:
                await mgr.shutdown()

    async def test_timeout_then_late_clear_releases_remainder(
        self, tmp_path,
    ):
        """Stuck-credit case: the
        FIRST verify times out with delta 0 (dec=0, the exact unexercised
        case the timeout test above does not cover -- it only tests a PARTIAL delta). Without a
        re-verify, nothing would ever re-check the card again -- stuck
        forever. Expected: a delayed re-verify at grace_seconds fires
        exactly once; when IT confirms clearance, the remaining credit is
        released."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120,
            grace_seconds=1,
        )
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        spawn_calls, sigterm_calls = [], []
        calls = []

        async def fake_vram_staged(*, expected_drop_mib, timeout_s, device_index=0, **_ignored):
            # **_ignored absorbs the settle_floor_s kwarg --
            # this fixture is wired to _vram_verify and this test's delayed
            # re-verify call (_late_vram_reconcile) now passes it explicitly
            # (settle_floor_s=0.0). Not part of what this test pins.
            calls.append(expected_drop_mib)
            if len(calls) == 1:
                return False, 0  # first check: not cleared, MEASURED delta 0 (dec=0)
            return True, 0  # delayed re-verify: now confirmed cleared

        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        mgr._vram_verify = fake_vram_staged
        # Constant, unchanging reading -> pre/post delta computes to exactly 0
        # on the first (not-cleared) call, producing dec=0 (the stuck shape).
        mgr._gpu_used_mib = lambda *, device_index=0: 0
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.sleep(0.15)
                r1 = mgr._residents.get("m1")
                need_r1 = r1.reserved_need_mib
                async with mgr._registry_lock:
                    mgr._begin_unload_locked(r1)
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    if len(calls) >= 1 and mgr._pending_reclaim_mib.get(0, 0) == need_r1:
                        break
                assert len(calls) == 1, "first verify must have fired"
                assert mgr._pending_reclaim_mib.get(0, 0) == need_r1, (
                    "dec=0 on the first timeout -- credit must stay fully alive "
                    "(the exact stuck-credit shape), not stuck "
                    "forever: a delayed re-verify must be pending"
                )
                # Wait past grace_seconds for the delayed re-verify to fire and land.
                for _ in range(200):
                    await asyncio.sleep(0.02)
                    if len(calls) >= 2 and mgr._pending_reclaim_mib.get(0, 0) == 0:
                        break
                assert len(calls) == 2, (
                    f"expected exactly one delayed re-verify (2 total calls); "
                    f"got {len(calls)}"
                )
                assert mgr._pending_reclaim_mib.get(0, 0) == 0, (
                    "late clear confirmed -> the remaining credit must be released"
                )
            finally:
                await mgr.shutdown()

    async def test_timeout_never_clears_exactly_one_reverify(
        self, tmp_path, caplog,
    ):
        """The delayed re-verify ALSO fails to confirm
        clearance. The credit must stay (safety: never release un-freed
        VRAM), exactly ONE delayed re-verify must have fired (not a retry
        loop -- proves the 'one-shot' bound, not just 'eventually clears'),
        and the WARN must fire exactly once."""
        import logging
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120,
            grace_seconds=1,
        )
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        spawn_calls, sigterm_calls = [], []
        calls = []

        async def fake_vram_never_clears(*, expected_drop_mib, timeout_s, device_index=0, **_ignored):
            # **_ignored absorbs the settle_floor_s kwarg, same
            # reason as fake_vram_staged above (this test's delayed re-verify
            # also runs through _late_vram_reconcile).
            calls.append(expected_drop_mib)
            return False, 0  # never confirms (measured delta stays 0), on every call

        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        mgr._vram_verify = fake_vram_never_clears
        mgr._gpu_used_mib = lambda *, device_index=0: 0
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                with caplog.at_level(logging.WARNING, logger="turbohaul.manager"):
                    await asyncio.wait_for(
                        mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                    await asyncio.sleep(0.15)
                    r1 = mgr._residents.get("m1")
                    need_r1 = r1.reserved_need_mib
                    async with mgr._registry_lock:
                        mgr._begin_unload_locked(r1)
                    # Wait comfortably past grace_seconds so the ONE delayed
                    # re-verify has definitely fired and settled, then keep
                    # watching a while longer to prove NOTHING fires a third time.
                    for _ in range(200):
                        await asyncio.sleep(0.02)
                        if len(calls) >= 2:
                            break
                    await asyncio.sleep(0.3)  # margin: a retry-loop bug would fire again here
                assert len(calls) == 2, (
                    f"expected EXACTLY one delayed re-verify (2 total calls, no "
                    f"retry loop); got {len(calls)}"
                )
                assert mgr._pending_reclaim_mib.get(0, 0) == need_r1, (
                    "never-clears -> the full credit must stay (safety: never "
                    "release VRAM that was never confirmed free)"
                )
                unreconciled_warns = [
                    rec for rec in caplog.records
                    if "unreconciled" in rec.message
                ]
                assert len(unreconciled_warns) == 1, (
                    f"expected exactly one 'unreconciled' WARN; got "
                    f"{len(unreconciled_warns)}: {[r.message for r in caplog.records]}"
                )
                assert unreconciled_warns[0].levelno == logging.WARNING
            finally:
                await mgr.shutdown()

    # -----------------------------------------------------------------
    # Trust-the-kill dev-tolerance pins.
    # All three of these paths are accepted as releasing the FULL credit
    # as-is (documented dev-tolerance, matching verify_vram_cleared's own
    # philosophy) -- these are PINS ONLY, not behavior changes. Each one
    # should already pass on today's code; their job is to fail loudly if
    # a future change silently drops the dev-tolerance default.
    # -----------------------------------------------------------------

    async def test_pre_snapshot_none_trusts_the_kill(self, tmp_path):
        """Pin: nvidia-smi unavailable BEFORE SIGTERM (_gpu_used_mib returns
        None) -> no pre-teardown baseline to measure a delta against -> the
        FULL credit is released even though verify itself reports
        cleared_ok=False. Mutation check: if this default were removed (e.g.
        measured_delta defaulted to 0 instead of reclaim_mib when the
        pre-snapshot is unavailable), this test would fail with credit stuck
        at need_r1 instead of released to 0."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        spawn_calls, sigterm_calls = [], []

        async def fake_vram_not_cleared(*, expected_drop_mib, timeout_s, device_index=0):
            return False, 12345  # a real reading exists post-verify...

        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        mgr._vram_verify = fake_vram_not_cleared
        mgr._gpu_used_mib = lambda *, device_index=0: None  # ...but no PRE snapshot
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.sleep(0.15)
                r1 = mgr._residents.get("m1")
                async with mgr._registry_lock:
                    mgr._begin_unload_locked(r1)
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    if mgr._pending_reclaim_mib.get(0, 0) == 0:
                        break
                assert mgr._pending_reclaim_mib.get(0, 0) == 0, (
                    "no pre-teardown snapshot -> trust the kill, full credit "
                    "released despite cleared_ok=False"
                )
            finally:
                await mgr.shutdown()

    async def test_current_reading_none_trusts_the_kill(self, tmp_path):
        """Pin: pre-snapshot IS available, but verify's own current_used_mib
        reading is None (nvidia-smi went unavailable mid-poll) -> no delta
        can be measured -> the FULL credit is released. Same dev-tolerance
        family as the pre-snapshot-None case, different half of the pair."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        spawn_calls, sigterm_calls = [], []

        async def fake_vram_current_none(*, expected_drop_mib, timeout_s, device_index=0):
            return False, None  # not cleared, and no current reading either

        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        mgr._vram_verify = fake_vram_current_none
        mgr._gpu_used_mib = lambda *, device_index=0: 50000  # a real pre-snapshot exists
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.sleep(0.15)
                r1 = mgr._residents.get("m1")
                async with mgr._registry_lock:
                    mgr._begin_unload_locked(r1)
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    if mgr._pending_reclaim_mib.get(0, 0) == 0:
                        break
                assert mgr._pending_reclaim_mib.get(0, 0) == 0, (
                    "verify's current reading unavailable -> trust the kill, "
                    "full credit released despite cleared_ok=False"
                )
            finally:
                await mgr.shutdown()

    async def test_verify_raises_trusts_the_kill(self, tmp_path):
        """Pin: _vram_verify itself raises (best-effort try/except in
        _unload_teardown's finally) -> caught, cleared_ok stays at its
        initial True default -> the FULL credit is released. Mutation check:
        if the except block instead re-raised or defaulted cleared_ok to
        False, this test would fail with credit stuck instead of released."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", expected_vram_mib=18000, split_mode="none", main_gpu=0)
        spawn_calls, sigterm_calls = [], []

        async def fake_vram_raises(*, expected_drop_mib, timeout_s, device_index=0):
            raise RuntimeError("simulated nvidia-smi subprocess failure")

        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        mgr._vram_verify = fake_vram_raises
        mgr._gpu_used_mib = lambda *, device_index=0: 50000
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.sleep(0.15)
                r1 = mgr._residents.get("m1")
                async with mgr._registry_lock:
                    mgr._begin_unload_locked(r1)
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    if mgr._pending_reclaim_mib.get(0, 0) == 0:
                        break
                assert mgr._pending_reclaim_mib.get(0, 0) == 0, (
                    "_vram_verify raised -> caught, trust the kill, full "
                    "credit released"
                )
            finally:
                await mgr.shutdown()

    async def test_auto_place_still_evicts_off_card_victim(self, tmp_path):
        """Card-filter refinement: the card filter must NOT apply
        to an auto_place model. Reaching the over-commit quadrant with auto_place=True
        means _auto_pick_gpu already returned None (no card fits), so main_gpu is just
        the manifest fallback pin — freeing ANY card lets the re-route auto-placer
        relocate the model. So the OFF-CARD idle victim IS evicted (not a needless
        503). The on-card (gpu0) is busy (not idle-evictable); the only idle victim is
        on gpu1."""
        from types import SimpleNamespace
        import time as _t
        import yaml
        from turbohaul.manager import Resident
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=3)
        (boot.storage.manifests_path / "x.yaml").write_text(yaml.safe_dump({
            "model_tag": "x", "gguf_blob_sha256": "a" * 64,
            "gguf_size_bytes": 18000 * 1024 * 1024, "context_size": 2048,
            "expected_vram_bytes": 18000 * 1024 * 1024, "auto_place": True,
            "llama_server_flags": {"split_mode": "none", "main_gpu": 0}}))
        mgr = _mk(boot, runtime, **_mocks([]))
        now = _t.monotonic()
        mgr._residents["busy0"] = Resident(
            model_tag="busy0", state=ResidentState.ACTIVE, main_gpu=0, split_mode="none",
            reserved_need_mib=18000, active_slot=object(), last_active_monotonic=now)
        mgr._residents["idle1"] = Resident(
            model_tag="idle1", state=ResidentState.IDLE_EVICTABLE, main_gpu=1,
            split_mode="none", reserved_need_mib=18000, last_active_monotonic=now - 10)
        reserved, deferred, evicted = [], [], []

        async def spy_reserve(slot):
            reserved.append(slot.model_tag)

        def spy_defer(slot, *, evict_pending=False):
            deferred.append((slot.model_tag, evict_pending))

        def spy_evict(r):
            evicted.append(r.model_tag)

        mgr._reserve_and_start_locked = spy_reserve
        mgr._defer_unroutable = spy_defer
        mgr._begin_unload_locked = spy_evict
        # fastlane=None mirrors the real Slot
        # dataclass's own default -- _route_or_reserve now reads it
        # unconditionally via _shield_carveout_tags before this test's own
        # scope even begins.
        # fastlane_relocated_main_gpu/_split_mode=None
        # mirror the real Slot dataclass's own new defaults, same reason --
        # _route_or_reserve now reads the first one unconditionally right
        # after placement resolution, before this test's own scope begins.
        slot = SimpleNamespace(
            model_tag="x", fastlane=None,
            fastlane_relocated_main_gpu=None, fastlane_relocated_split_mode=None,
        )
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[4000, 4000]):
            await mgr._route_or_reserve(slot)
        assert reserved == []
        assert deferred == [("x", True)]
        assert evicted == ["idle1"], "auto_place must evict the off-card idle victim, not 503"

    async def test_relocatable_credits_aggregate_pending(self, tmp_path):
        """Pending-reclaim aggregation: a relocatable (auto_place) model is evicted GLOBAL-LRU, so its
        in-flight reclaim can land on any card — the EVICT-decision credit must therefore
        be AGGREGATE (matching the eviction scope), else concurrent auto_place misses
        wouldn't see each other's pending and would over-evict. A pinned model still
        credits only the target card. The ACTUAL reserve gate never credits pending."""
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=2)
        mgr = _mk(boot, runtime, **_mocks([]))
        mgr._pending_reclaim_mib = {1: 5000}  # 5000 MiB pledged-to-free on card 1
        # The relocatable+split='none' path now also
        # reads the raw per-card array directly in manager.py (to check each
        # CANDIDATE card individually, not just aggregate blindly against
        # main_gpu's own free reading) -- patch both binding points, the
        # established dual-patch pattern this suite already uses elsewhere
        # (test_auto_placer.py and similar placement tests) for code
        # that reads this function from manager.py's own imported name.
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[10000, 10000]), \
             patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10000, 10000]):
            need = 13000  # > card0 free (10000), <= card0 free + card1 pending (15000)
            # pinned split='none' -> credits only card0 (no pending there) -> refuse
            assert not mgr._vram_admits_locked(
                need, 1, 0, "none", credit_pending=True, relocatable=False)
            # relocatable (auto_place) -> credits AGGREGATE pending (incl card1) -> admit
            assert mgr._vram_admits_locked(
                need, 1, 0, "none", credit_pending=True, relocatable=True)
            # the ACTUAL reserve gate NEVER credits pending (would spawn on not-yet-free VRAM)
            assert not mgr._vram_admits_locked(need, 1, 0, "none")

    async def test_per_card_credit_not_summed_across_cards(self, tmp_path):
        """A relocatable model's footprint can NEVER be
        assembled by summing two DIFFERENT cards' pending reclaim -- it is
        still a single-card pin once placed. Summing would false-
        positive: main_gpu(0)'s own free (4000) plus the AGGREGATE credit
        from BOTH cards (4000+4000=8000) = 12000 >= need, even though
        NEITHER card individually has enough (4000 free + 4000 own credit =
        8000 < 12000 on each). Expected: refused, because no
        single card can ever reach 12000."""
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=2)
        mgr = _mk(boot, runtime, **_mocks([]))
        mgr._pending_reclaim_mib = {0: 4000, 1: 4000}  # split evenly, neither card enough
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[4000, 4000]), \
             patch("turbohaul.manager._read_free_vram_all_mib", return_value=[4000, 4000]):
            need = 12000  # each card: 4000 free + 4000 own credit = 8000, short by 4000
            assert not mgr._vram_admits_locked(
                need, 1, 0, "none", credit_pending=True, relocatable=True
            ), (
                "no single card's own free+credit reaches need (8000 < 12000 on "
                "both) -- summing unrelated cards' credit together must not "
                "manufacture a fit that doesn't physically exist"
            )
            # A genuinely reachable per-card case must still admit: give card1
            # enough of its OWN credit to clear the bar alone.
            mgr._pending_reclaim_mib = {0: 4000, 1: 8000}
            assert mgr._vram_admits_locked(
                need, 1, 0, "none", credit_pending=True, relocatable=True
            ), "card1 alone (4000 free + 8000 own credit = 12000) must admit"

    async def test_probe_stall_credit_aware_fallback(self, tmp_path):
        """During a probe stall (nvidia-smi unreadable),
        the eviction-DECISION gate (credit_pending=True) must not blindly
        refuse and trigger an unnecessary extra eviction when the ALREADY-
        KNOWN pending credit alone already covers the need. Without this, a
        probe-unavailable co-residence call would always return False regardless
        of credit -- a per-tick eviction storm during any stall. Instead,
        the known credit alone (never a guessed free-VRAM substitute) can
        short-circuit to admit; when the credit alone is insufficient, the
        refuse-blind doctrine is preserved."""
        from turbohaul.manager import Resident
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=2)
        mgr = _mk(boot, runtime, **_mocks([]))
        # A sibling resident must exist -- the probe-unavailable branch only
        # refuses when co-residence (siblings) is in play; a lone spawn
        # always degrades open regardless of credit (unaffected, untested
        # here since it's not what this test is about).
        mgr._residents["sib"] = Resident(
            model_tag="sib", state=ResidentState.IDLE_EVICTABLE, main_gpu=0,
            split_mode="none", reserved_need_mib=8000,
        )
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=None):
            need = 10000
            # No credit at all -- must still refuse-blind (unchanged doctrine).
            mgr._pending_reclaim_mib = {}
            assert not mgr._vram_admits_locked(
                need, 1, 0, "none", credit_pending=True, relocatable=False
            )
            # Credit alone insufficient -- still refuses.
            mgr._pending_reclaim_mib = {0: 4000}
            assert not mgr._vram_admits_locked(
                need, 1, 0, "none", credit_pending=True, relocatable=False
            )
            # Credit alone already covers the need -- admits WITHOUT a probe,
            # avoiding an unnecessary extra eviction during the stall.
            mgr._pending_reclaim_mib = {0: 10000}
            assert mgr._vram_admits_locked(
                need, 1, 0, "none", credit_pending=True, relocatable=False
            )
            # credit_pending=False (the ACTUAL reserve gate) must NEVER use this
            # fallback -- still refuse-blind even with ample "credit" sitting
            # in the dict, probe or no probe.
            assert not mgr._vram_admits_locked(need, 1, 0, "none")

    async def test_per_card_probe_flake_retries_not_over_refuses(
        self, tmp_path,
    ):
        """The per-card loop's OWN
        nvidia-smi read (a SECOND, independent call in the same admission
        tick -- the FIRST already happened inside _vram_budget above it and
        succeeded) can itself flake to None on a transient hiccup. Without a retry,
        that flake alone -> vals=[] -> the loop never iterates -> return
        False, an over-refusal even though the primary read just showed
        room. A single retry of the same primary probe recovers."""
        from turbohaul.manager import Resident
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=2)
        mgr = _mk(boot, runtime, **_mocks([]))
        mgr._residents["vic-a"] = Resident(
            model_tag="vic-a", state=ResidentState.IDLE_EVICTABLE, main_gpu=0,
            split_mode="none", reserved_need_mib=8000,
        )
        mgr._pending_reclaim_mib = {0: 8000}
        need = 12000  # card 0: 4000 free + 8000 credit = 12000, exactly enough
        with patch(
            "turbohaul.safety._read_free_vram_all_mib", return_value=[4000],
        ), patch(
            "turbohaul.manager._read_free_vram_all_mib",
            side_effect=[None, [4000]],  # first call flakes, retry succeeds
        ) as mock_read:
            assert mgr._vram_admits_locked(
                need, 1, 0, "none", credit_pending=True, relocatable=True,
            ), (
                "a flaked second read must retry the primary probe, not "
                "over-refuse a request the primary read already showed fits"
            )
            assert mock_read.call_count == 2, (
                f"expected exactly one retry (2 calls total); got "
                f"{mock_read.call_count}"
            )

    async def test_per_card_double_flake_still_refuses(
        self, tmp_path,
    ):
        """Companion to the retry test: if BOTH the fresh read and the
        retry flake, the loop correctly falls through to vals=[] and
        refuses -- the retry is a bounded, one-shot recovery, not a
        guarantee, and must not mask a genuinely unreadable probe."""
        from turbohaul.manager import Resident
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=2)
        mgr = _mk(boot, runtime, **_mocks([]))
        mgr._residents["vic-a"] = Resident(
            model_tag="vic-a", state=ResidentState.IDLE_EVICTABLE, main_gpu=0,
            split_mode="none", reserved_need_mib=8000,
        )
        mgr._pending_reclaim_mib = {0: 8000}
        need = 12000
        with patch(
            "turbohaul.safety._read_free_vram_all_mib", return_value=[4000],
        ), patch(
            "turbohaul.manager._read_free_vram_all_mib",
            side_effect=[None, None],  # both the fresh read AND the retry flake
        ) as mock_read:
            assert not mgr._vram_admits_locked(
                need, 1, 0, "none", credit_pending=True, relocatable=True,
            ), "a double-flake must still refuse-blind, not silently admit"
            assert mock_read.call_count == 2


class TestParallelSlotsUsedCountsEveryServingResident:
    """parallel_slots.used must count EVERY serving resident at cap>=2.

    Symptom when BOTH sidecars stream real traffic:
    residents[] has len=2, both ACTIVE with real ports+pids, but parallel_slots
    reports {'used': 0, 'max': 2}. The FE renders that verbatim --
    the frontend's Queue component shows "{used} / {max}" -- so the queue
    tab would read "0 / 2" while two sidecars serve. That is a user-visible
    frontend display bug.

    A naive count has a ceiling of 1: `used` falls through to
    `1 if self._resolve_top_level_active_handle() else 0`, and that resolver returns
    ONE handle (`return best_r.handle`), while `self._inflight` is
    permanently unwritten at cap>=2 so the len() branch is never taken. Ceiling of 1.
    """

    async def test_used_reports_two_when_two_residents_are_serving(self, tmp_path):
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)
        gate = asyncio.Event()  # never set -> both park in complete, both stay ACTIVE
        h, sg, vr, cp = TestFanOutParallel._gated_mocks(gate)
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=TestFanOutParallel._p2_spawn([]), health_fn=h,
            sigterm_fn=sg, vram_fn=vr, complete_fn=cp,
        )
        mgr.runtime.queue.safety_enabled = False
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
            f2 = asyncio.create_task(mgr.submit_and_wait("m2", "b", thread_id="t2"))
            try:
                serving = 0
                for _ in range(250):
                    await asyncio.sleep(0.02)
                    serving = sum(
                        1 for r in mgr._model_residents()
                        if getattr(getattr(r.active_slot, "state", None), "value", None)
                        in ("ACTIVE", "ACTIVE_MATCH")
                    )
                    if serving >= 2:
                        break

                # PRECONDITION: the state must ARISE from real traffic, never be
                # hand-set. Assigning a scalar that production
                # cannot reach at the cap this test sets would prove nothing; this asserts the opposite,
                # so a green below cannot come from a fixture that built an impossible state.
                assert serving == 2, (
                    f"precondition unmet: need 2 residents ACTIVE, got {serving} -- "
                    "the assertion below would be vacuous"
                )

                snap = mgr.status_snapshot()
                used = snap["parallel_slots"]["used"]
                assert used == 2, (
                    f"parallel_slots.used reported {used} while TWO residents were "
                    "ACTIVE. The Queue component renders this straight to the queue tab, so "
                    "an under-count is user-visible. The count caps at 1 because: "
                    "_resolve_top_level_active_handle returns a single handle and "
                    "self._inflight is never written at cap>=2."
                )
            finally:
                gate.set()
                for f in (f1, f2):
                    f.cancel()
                await asyncio.gather(f1, f2, return_exceptions=True)
                await mgr.shutdown()
