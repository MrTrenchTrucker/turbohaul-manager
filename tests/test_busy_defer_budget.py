"""_defer_unroutable's busy regime (all residents busy,
no idle victim -- the model FITS, it's just not this moment's turn) had a fixed
50-defer x 50ms = 2.5s budget before refusing with a 503, while the HARDER
evict-pending regime (real VRAM over-commit) got a derived 60s-30min window via
_max_vram_defers. The intended behaviour: this scenario should QUEUE,
not refuse.

⛔ THE OBSERVABLE IS THE POINT: a nicer error is NOT the fix. The gate is that a
request which WOULD have been refused under the old 2.5s cap is now SERVED (a
real completion, no exception) once a resident frees. TestBusyRegimeServesInsteadOfRefusing
drives this via mgr.submit_and_wait -- the EXACT function chat_completion.py's
routes await and turn into HTTP 200/503/504 -- with a wait DELIBERATELY longer
than the old fixed cap.

The original refusal is hard to reproduce on real hardware, so these are
the primary proof of this fix -- forced directly at the three
named conditions, asserting slot._dispatch_defer_count itself, not just an
end-state.
"""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from turbohaul.api.main import create_app
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
    ResidentState,
    TurbohaulManager,
    _DISPATCH_DEFER_BACKOFF_S,
    _VRAM_DEFER_BACKOFF_S,
    _VRAM_DEFER_MAX_DEFERS,
    _VRAM_DEFER_MIN_DEFERS,
    _VRAM_DEFER_TEARDOWN_MARGIN_S,
)
from turbohaul.slot import Slot
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime(tmp_path, *, max_parallel_sidecars=2, grace_seconds=0,
                   max_grace_extensions=5, idle_hot_load_seconds=0):
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
            default_port_base=59600,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False,
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=grace_seconds,
            max_grace_extensions=max_grace_extensions,
            idle_hot_load_seconds=idle_hot_load_seconds,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _write_manifest_yaml(manifests_root, tag: str):
    (manifests_root / f"{tag}.yaml").write_text(
        f"""model_tag: {tag}
gguf_blob_sha256: "{'a' * 64}"
gguf_size_bytes: 1000
"""
    )


def _seed_manifest(boot, model_tag, *, split_mode="none", main_gpu=0):
    """Mirrors test_multislot_concurrency.py's own helper: co-residence needs
    split_mode='none' + DISTINCT main_gpu per model (the co-residence gate)."""
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


def _fake_handle(model_tag: str, port: int, pid: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _high_vram():
    return patch(
        "turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000],
    )


# ============================================================================
# 1. The derivation itself -- forced directly, _dispatch_defer_count asserted.
# ============================================================================


class TestMaxBusyDefersDerivation:
    def test_wall_clock_parity_with_the_harder_evict_pending_regime(self, tmp_path):
        """The design framing: busy (the EASIER case) must not get
        LESS real time than evict-pending (the HARDER case). Assert the two
        regimes' DERIVED WINDOWS (defer_count * that regime's own backoff)
        land within one busy-tick of each other -- not their raw counts,
        which differ by exactly the backoff ratio (20x) and would look like a
        mismatch if compared directly."""
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=30, max_grace_extensions=5,
        )
        mgr = TurbohaulManager(boot, runtime)
        busy_window_s = mgr._max_busy_defers() * _DISPATCH_DEFER_BACKOFF_S
        vram_window_s = mgr._max_vram_defers() * _VRAM_DEFER_BACKOFF_S
        assert abs(busy_window_s - vram_window_s) <= _DISPATCH_DEFER_BACKOFF_S, (
            f"busy window ({busy_window_s}s) and evict-pending window "
            f"({vram_window_s}s) should be at wall-clock parity; got a gap "
            f"of {abs(busy_window_s - vram_window_s)}s"
        )

    def test_the_pitfall_reusing_the_count_verbatim_would_have_bought_3s_not_60s(
        self, tmp_path,
    ):
        """⭐ THE SHARPEST PART: at the minimum-clamp config (grace_
        seconds=0), _max_vram_defers() returns exactly _VRAM_DEFER_MIN_DEFERS
        (60). Naively reusing that COUNT at busy's own faster 0.05s backoff
        buys only 60 * 0.05s = 3 SECONDS -- barely better than the original
        bug's 2.5s. The actual fix reuses the WINDOW (60 * 1.0s = 60s) and
        re-derives busy's count at ITS OWN backoff, buying the full ~60s."""
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=0, max_grace_extensions=0,
        )
        mgr = TurbohaulManager(boot, runtime)
        vram_defers = mgr._max_vram_defers()
        assert vram_defers == _VRAM_DEFER_MIN_DEFERS  # sanity: floor clamp hit

        naive_reuse_window_s = vram_defers * _DISPATCH_DEFER_BACKOFF_S
        assert naive_reuse_window_s == pytest.approx(3.0), (
            f"the naive (wrong) derivation this test documents as a trap "
            f"should compute ~3s; got {naive_reuse_window_s}s -- if this "
            f"assertion changed, the trap's own arithmetic changed"
        )

        actual_busy_window_s = mgr._max_busy_defers() * _DISPATCH_DEFER_BACKOFF_S
        assert actual_busy_window_s >= _VRAM_DEFER_MIN_DEFERS * _VRAM_DEFER_BACKOFF_S, (
            f"the SHIPPED derivation must buy the full ~60s window, not the "
            f"naive ~3s a count-reuse would have bought; got "
            f"{actual_busy_window_s}s"
        )
        assert actual_busy_window_s > naive_reuse_window_s * 15, (
            "shipped window should be over an order of magnitude larger than "
            "the naive count-reuse trap"
        )

    def test_pathological_grace_config_still_clamps_to_the_same_ceiling(self, tmp_path):
        """A pathological grace config can't make busy defer for days either
        -- same _VRAM_DEFER_MAX_DEFERS * _VRAM_DEFER_BACKOFF_S (1800s) ceiling
        VRAM itself is clamped to."""
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=3600, max_grace_extensions=1000,
        )
        mgr = TurbohaulManager(boot, runtime)
        ceiling_s = _VRAM_DEFER_MAX_DEFERS * _VRAM_DEFER_BACKOFF_S
        busy_window_s = mgr._max_busy_defers() * _DISPATCH_DEFER_BACKOFF_S
        assert busy_window_s <= ceiling_s + _DISPATCH_DEFER_BACKOFF_S
        vram_window_s = mgr._max_vram_defers() * _VRAM_DEFER_BACKOFF_S
        assert vram_window_s == ceiling_s  # VRAM itself pinned at the ceiling too

    @pytest.mark.asyncio
    async def test_dispatch_defer_count_climbs_past_the_old_derived_cap_without_ever_tripping(
        self, tmp_path,
    ):
        """Design rule: once a request is queued, it stays queued and the server
        does not cancel it unless the client explicitly cancels it.
        The rule applies to the busy regime as well.
        _defer_unroutable's exhaustion branch is gone -- re-points this test
        (was test_dispatch_defer_count_climbs_and_trips_exactly_at_the_
        derived_cap, which asserted the OLD fixed-cap 503). That test broke
        due to an intentional behavior change, so it is replaced rather
        than deleted. Same forced-direct scenario, same derived
        cap as the reference point -- new assertion: driven to 3x the OLD
        cap, _dispatch_defer_count keeps climbing and the future never
        trips, because there is no cap left to trip."""
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=0, max_grace_extensions=0,
        )
        mgr = TurbohaulManager(boot, runtime)
        max_defers = mgr._max_busy_defers()
        assert max_defers > 50  # strictly more patient than the retired fixed cap

        slot = Slot.new("unlisted-model")
        slot.completion_future = asyncio.get_event_loop().create_future()
        for i in range(1, (max_defers * 3) + 1):
            mgr._defer_unroutable(slot)
            assert slot._dispatch_defer_count == i
            assert not slot.completion_future.done(), (
                f"slot must never be failed by _defer_unroutable any more -- "
                f"tripped at defer {i} of {max_defers * 3} (old cap was "
                f"{max_defers})"
            )


# ============================================================================
# 2. Non-regression -- evict-pending regime untouched (by design).
# ============================================================================


class TestEvictPendingRegimeUnchanged:
    @pytest.mark.asyncio
    async def test_max_vram_defers_formula_unchanged(self, tmp_path):
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=30, max_grace_extensions=5,
        )
        mgr = TurbohaulManager(boot, runtime)
        grace_cycle_s = 30 * (1 + 5)
        window_s = grace_cycle_s + _VRAM_DEFER_TEARDOWN_MARGIN_S
        import math
        expected = max(
            _VRAM_DEFER_MIN_DEFERS,
            min(math.ceil(window_s / _VRAM_DEFER_BACKOFF_S), _VRAM_DEFER_MAX_DEFERS),
        )
        assert mgr._max_vram_defers() == expected

    @pytest.mark.asyncio
    async def test_evict_pending_defer_still_uses_its_own_counter_and_never_trips(self, tmp_path):
        """Re-points test_evict_
        pending_defer_still_uses_its_own_counter_and_backoff, which asserted
        the OLD fixed-cap 503 on this regime too -- for the same reason, same
        scenario (own counter, own backoff, floor-clamped derivation all
        unchanged), new assertion: driven to 3x the derived floor, the
        counter keeps its own identity (still independent of
        _dispatch_defer_count) and the future never trips."""
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=0, max_grace_extensions=0)
        mgr = TurbohaulManager(boot, runtime)
        slot = Slot.new("m")
        slot.completion_future = asyncio.get_event_loop().create_future()
        max_defers = mgr._max_vram_defers()
        assert max_defers == _VRAM_DEFER_MIN_DEFERS  # 60, unchanged floor
        for i in range(1, (max_defers * 3) + 1):
            mgr._defer_unroutable(slot, evict_pending=True)
            assert slot._vram_defer_count == i
            assert not hasattr(slot, "_dispatch_defer_count") or slot._dispatch_defer_count == 0
            assert not slot.completion_future.done(), (
                f"evict-pending slot must never be failed by "
                f"_defer_unroutable any more -- tripped at defer {i} of "
                f"{max_defers * 3} (old floor cap was {max_defers})"
            )


# ============================================================================
# 3. THE OBSERVABLE -- served (not refused) end-to-end via submit_and_wait,
#    the exact function every chat_completion.py route awaits.
# ============================================================================


@pytest.mark.asyncio
class TestBusyRegimeServesInsteadOfRefusing:
    async def test_third_model_queued_past_the_old_25s_cap_is_served_once_a_resident_frees(
        self, tmp_path,
    ):
        boot, runtime = _boot_runtime(
            tmp_path, max_parallel_sidecars=2, grace_seconds=0,
            max_grace_extensions=0, idle_hot_load_seconds=0,
        )
        pid = [80000]
        busy_gate = asyncio.Event()

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            pid[0] += 1
            return _fake_handle(model_tag, port, pid[0])

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(*a, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            if handle.model_tag in ("m1", "m2"):
                await busy_gate.wait()
            return {"ok": True, "model": handle.model_tag}

        _seed_manifest(boot, "m1", main_gpu=0)
        _seed_manifest(boot, "m2", main_gpu=1)
        _seed_manifest(boot, "m3", main_gpu=0)
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False

        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                # Occupy BOTH cap=2 residents with m1 and m2, both hung on
                # busy_gate -- ACTIVE, not idle, so no LRU victim exists.
                t1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                t2 = asyncio.create_task(mgr.submit_and_wait("m2", "a", thread_id="t2"))
                for _ in range(200):
                    await asyncio.sleep(0.02)
                    residents = mgr._model_residents()
                    if (
                        len(residents) == 2
                        and all(r.state is ResidentState.ACTIVE for r in residents)
                    ):
                        break
                assert len(mgr._model_residents()) == 2, "both m1/m2 residents ACTIVE"

                # m3 needs a THIRD resident: count-cap full, no idle victim ->
                # busy-defer path. Submitted via mgr.submit() directly (not
                # submit_and_wait) so slot3._dispatch_defer_count can be
                # polled directly -- a STRUCTURAL proof, not a wall-clock
                # race: real per-iteration overhead in a test environment
                # (logging, task scheduling) makes racing a real-time margin
                # against the retired fixed cap's nominal 2.5s unreliable
                # (an earlier wall-clock version of this approach was observed to
                # PASS on unamended, unfixed code, exactly the
                # vacuity trap this test guards against). Polling the counter
                # itself is unambiguous regardless of environment speed.
                slot3 = await mgr.submit(
                    model_tag="m3", prompt="a", thread_id="t3",
                    wait_for_completion=True,
                )
                for _ in range(3000):
                    await asyncio.sleep(0.01)
                    if getattr(slot3, "_dispatch_defer_count", 0) > 50:
                        break
                assert getattr(slot3, "_dispatch_defer_count", 0) > 50, (
                    "slot3 never deferred past the OLD fixed 50-defer cap -- "
                    "test setup problem, not proof of anything about the fix"
                )
                assert not slot3.completion_future.done(), (
                    "slot3 must still be PENDING past the exact defer count "
                    "(50) where the OLD fixed cap would already have failed "
                    "it -- this is the actual observable proof, not a nicer "
                    "error message"
                )

                busy_gate.set()  # m1/m2 complete -> residents free
                result3 = await asyncio.wait_for(slot3.completion_future, timeout=10)
                assert result3 == {"ok": True, "model": "m3"}, (
                    "m3 must be SERVED once a resident frees, not refused -- "
                    "this is the actual observable gate, not a nicer error"
                )
                await asyncio.wait_for(t1, timeout=5)
                await asyncio.wait_for(t2, timeout=5)
            finally:
                busy_gate.set()
                await mgr.shutdown()


# ============================================================================
# 4. Client-boundedness proof -- no NEW code in _defer_unroutable, but a
#    disconnected caller's busy-deferred slot IS still cut loose promptly, via
#    the pre-existing queue.pop_next -> is_evicted -> _handle_unloaded_slot path.
# ============================================================================


@pytest.mark.asyncio
class TestBusyRegimeDisconnectBoundedness:
    async def test_disconnected_caller_is_cut_loose_well_before_the_new_larger_cap(
        self, tmp_path,
    ):
        boot, runtime = _boot_runtime(
            tmp_path, max_parallel_sidecars=2, grace_seconds=30,
            max_grace_extensions=5, idle_hot_load_seconds=0,
        )
        pid = [81000]
        busy_gate = asyncio.Event()

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            pid[0] += 1
            return _fake_handle(model_tag, port, pid[0])

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(*a, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            if handle.model_tag in ("m1", "m2"):
                await busy_gate.wait()
            return {"ok": True}

        _seed_manifest(boot, "m1", main_gpu=0)
        _seed_manifest(boot, "m2", main_gpu=1)
        _seed_manifest(boot, "m3", main_gpu=0)
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False

        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                t1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
                t2 = asyncio.create_task(mgr.submit_and_wait("m2", "a", thread_id="t2"))
                for _ in range(200):
                    await asyncio.sleep(0.02)
                    residents = mgr._model_residents()
                    if (
                        len(residents) == 2
                        and all(r.state is ResidentState.ACTIVE for r in residents)
                    ):
                        break
                assert len(mgr._model_residents()) == 2

                disconnect_event = asyncio.Event()
                t3 = asyncio.create_task(
                    mgr.submit_and_wait(
                        "m3", "a", thread_id="t3", disconnect_event=disconnect_event,
                    )
                )
                await asyncio.sleep(0.1)  # let m3 land in the busy-defer loop
                assert not t3.done()

                # The caller hangs up. This new derived cap is ~240s at this
                # config (grace 30, ext 5) -- if disconnect-boundedness were
                # NOT already in place, t3 would still be pending here.
                disconnect_event.set()

                from turbohaul.slot import SlotEvictedError
                with pytest.raises(SlotEvictedError):
                    await asyncio.wait_for(t3, timeout=2.0)
            finally:
                busy_gate.set()
                await mgr.shutdown()
