"""Cross-resident VRAM budget double-count fix.

Problem: check_free_vram/check_kv_cache_fit only checked nvidia-smi free
VRAM, blind to VRAM other live residents hold, so two heavy spawns could be
approved concurrently -> cudaMalloc OOM.
TurbohaulManager._occupied_vram_mib() + occupied_vram_mib threading address
this. The pinned behaviours are: the filter must not charge EVERY live resident on
ANY card (only still-loading siblings on the SAME card) -- the exact
double-count doc SAFETY_GATE_VRAM_MATH.md SS5.5 forbids; check_kv_cache_fit
must use the occupied_vram_mib it accepts; and
all_safety_gates() itself must accept the
parameter, since otherwise both manager.py call sites crash with TypeError on
every single spawn attempt. This file covers four areas:
  1. _occupied_vram_mib(main_gpu) — TestOccupiedVramMib
  2. check_free_vram wiring (docstring truthfulness, existing logic kept) —
     TestCheckFreeVramOccupied
  3. check_kv_cache_fit genuinely subtracts occupied_vram_mib — it is wired
     in — TestCheckKvCacheFitOccupied
  4. all_safety_gates accepts the parameter and threads it to both gates,
     including the exact real-call-path regression arm that would have
     caught the TypeError — TestAllSafetyGates
"""
from __future__ import annotations

from unittest.mock import patch

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
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.safety import all_safety_gates, check_free_vram, check_kv_cache_fit


# === shared manager-construction fixture (mirrors tests/test_auto_placer.py's
# _boot_runtime_multislot / _mk so this file drops into the same test session
# without new shared fixtures, same convention that file documents) =========

def _boot_runtime(tmp_path, *, safety_min_free_vram_mib=4096):
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
            max_parallel_sidecars=2,
            grace_seconds=0,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
            safety_min_free_vram_mib=safety_min_free_vram_mib,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _mk(boot, runtime):
    mgr = TurbohaulManager(boot, runtime)
    mgr.runtime.queue.safety_enabled = False
    return mgr


def _resident(tag, *, state, main_gpu=0, reserved_need_mib=0, split_mode="none"):
    return Resident(
        model_tag=tag, state=state, main_gpu=main_gpu,
        reserved_need_mib=reserved_need_mib, split_mode=split_mode,
    )


# === 1. _occupied_vram_mib(main_gpu) — the filter itself ====================

class TestOccupiedVramMib:
    def test_no_other_residents_returns_zero(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        assert mgr._occupied_vram_mib(main_gpu=0) == 0

    def test_active_sibling_same_card_not_charged(self, tmp_path):
        # ACTIVE VRAM is already reflected in the live nvidia-smi free
        # reading -- charging it again is the double-count
        # the filter must avoid.
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        mgr._residents["sib"] = _resident(
            "sib", state=ResidentState.ACTIVE, main_gpu=0, reserved_need_mib=24000)
        assert mgr._occupied_vram_mib(main_gpu=0) == 0

    def test_grace_sibling_same_card_not_charged(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        mgr._residents["sib"] = _resident(
            "sib", state=ResidentState.GRACE, main_gpu=0, reserved_need_mib=24000)
        assert mgr._occupied_vram_mib(main_gpu=0) == 0

    def test_idle_evictable_sibling_same_card_not_charged(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        mgr._residents["sib"] = _resident(
            "sib", state=ResidentState.IDLE_EVICTABLE, main_gpu=0,
            reserved_need_mib=24000)
        assert mgr._occupied_vram_mib(main_gpu=0) == 0

    def test_reserved_loading_sibling_same_card_charged(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        mgr._residents["sib"] = _resident(
            "sib", state=ResidentState.RESERVED_LOADING, main_gpu=0,
            reserved_need_mib=9500)
        assert mgr._occupied_vram_mib(main_gpu=0) == 9500

    def test_reserved_loading_sibling_different_card_not_charged(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        mgr._residents["sib"] = _resident(
            "sib", state=ResidentState.RESERVED_LOADING, main_gpu=1,
            reserved_need_mib=9500)
        assert mgr._occupied_vram_mib(main_gpu=0) == 0
        assert mgr._occupied_vram_mib(main_gpu=1) == 9500

    def test_dead_sibling_not_charged(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        mgr._residents["sib"] = _resident(
            "sib", state=ResidentState.DEAD, main_gpu=0, reserved_need_mib=9500)
        assert mgr._occupied_vram_mib(main_gpu=0) == 0

    def test_mixed_siblings_only_same_card_reserved_loading_summed(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        mgr._residents["active-same"] = _resident(
            "active-same", state=ResidentState.ACTIVE, main_gpu=0,
            reserved_need_mib=24000)
        mgr._residents["loading-same-1"] = _resident(
            "loading-same-1", state=ResidentState.RESERVED_LOADING, main_gpu=0,
            reserved_need_mib=4000)
        mgr._residents["loading-same-2"] = _resident(
            "loading-same-2", state=ResidentState.RESERVED_LOADING, main_gpu=0,
            reserved_need_mib=3000)
        mgr._residents["loading-other-card"] = _resident(
            "loading-other-card", state=ResidentState.RESERVED_LOADING, main_gpu=1,
            reserved_need_mib=8000)
        mgr._residents["grace-same"] = _resident(
            "grace-same", state=ResidentState.GRACE, main_gpu=0,
            reserved_need_mib=5000)
        assert mgr._occupied_vram_mib(main_gpu=0) == 7000  # only the two "same"
        assert mgr._occupied_vram_mib(main_gpu=1) == 8000

    def test_singleton_resident_excluded(self, tmp_path):
        # The legacy PHASE-0 singleton key must not leak into this sum
        # (mirrors _model_residents' own exclusion, exercised through the
        # real manager, not re-implemented here).
        from turbohaul.manager import _SINGLETON_RESIDENT_KEY
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        mgr._residents[_SINGLETON_RESIDENT_KEY] = _resident(
            "singleton", state=ResidentState.RESERVED_LOADING, main_gpu=0,
            reserved_need_mib=9500)
        assert mgr._occupied_vram_mib(main_gpu=0) == 0

    def test_excludes_self_when_own_reservation_already_registered(self, tmp_path):
        # Regression for a defect where
        # _run_spawn_safety_gate re-checks a
        # resident AFTER _reserve_and_start_locked has already inserted it
        # into self._residents as RESERVED_LOADING on its own card. Without
        # exclude_model_tag, a spawn's own reservation gets summed as if it
        # were a sibling's, charging it against itself.
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        mgr._residents["self-model"] = _resident(
            "self-model", state=ResidentState.RESERVED_LOADING, main_gpu=0,
            reserved_need_mib=18000)
        assert mgr._occupied_vram_mib(
            main_gpu=0, exclude_model_tag="self-model") == 0
        # A genuine same-card sibling (different model_tag) is still charged.
        mgr._residents["real-sibling"] = _resident(
            "real-sibling", state=ResidentState.RESERVED_LOADING, main_gpu=0,
            reserved_need_mib=5000)
        assert mgr._occupied_vram_mib(
            main_gpu=0, exclude_model_tag="self-model") == 5000

    def test_no_exclude_model_tag_is_backward_compatible(self, tmp_path):
        # exclude_model_tag defaults to None -- a caller that never passes it
        # (the manager never omits it, but the default must still be
        # safe) sees ordinary summing behavior, matching every other test in
        # this class that doesn't pass it.
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        mgr._residents["sib"] = _resident(
            "sib", state=ResidentState.RESERVED_LOADING, main_gpu=0,
            reserved_need_mib=9500)
        assert mgr._occupied_vram_mib(main_gpu=0) == 9500


# === 2. check_free_vram wiring (the logic is correct; docstring
# truthfulness + a direct regression test) ===

class TestCheckFreeVramOccupied:
    def test_zero_default_unaffected(self):
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[10_000]):
            r = check_free_vram(min_free_mib=1000)
        assert r.ok
        assert "10000 MiB effective free" in r.detail

    def test_occupied_reduces_effective_free_causes_refusal(self):
        # 10000 free, 9500 reserved by a same-card still-loading sibling ->
        # 500 effective, below the 1000 floor -> refuse.
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[10_000]):
            r = check_free_vram(min_free_mib=1000, occupied_vram_mib=9500)
        assert not r.ok
        assert "9500" in r.detail

    def test_occupied_negative_effective_free_refuses_with_clear_message(self):
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[5_000]):
            r = check_free_vram(min_free_mib=1000, occupied_vram_mib=6_000)
        assert not r.ok
        assert "effective free negative" in r.detail

    def test_zero_occupied_admits_when_fits(self):
        # Positive control: no same-card sibling -> unaffected, still admits.
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[10_000]):
            r = check_free_vram(min_free_mib=1000, occupied_vram_mib=0)
        assert r.ok


# === 3. check_kv_cache_fit — the occupied-VRAM wiring (a deliberate
# decision: this is the precise body+KV+overhead gate, and it must not stay
# blind while the coarse check_free_vram gate is fixed) =====================

class TestCheckKvCacheFitOccupied:
    def test_default_zero_matches_pre_existing_behavior(self):
        # Same fixture as test_safety.py::test_fails_when_kv_exceeds_free —
        # proves occupied_vram_mib=0 is byte-identical to the pre-fix gate.
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[22_000]):
            r = check_kv_cache_fit(65536, 17 * 1024 * 1024 * 1024, kv_cache_quant="f16")
        assert not r.ok
        assert "22000 MiB free" in r.detail

    def test_occupied_vram_refuses_a_spawn_that_would_otherwise_fit(self):
        # Explicit arm: fit-gate REFUSES when a RESERVED_LOADING
        # same-card sibling holds VRAM, where it would otherwise ADMIT.
        # q4_0 64K Qwen27B fits comfortably on 22000 MiB free (see
        # the quantised-KV fit tests in test_safety.py) but a
        # ~18000 MiB same-card still-loading sibling eats enough of it that
        # the closed-form no longer fits.
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[22_000]):
            r_no_sibling = check_kv_cache_fit(
                65536, 17 * 1024 * 1024 * 1024, kv_cache_quant="q4_0",
                occupied_vram_mib=0)
            r_with_sibling = check_kv_cache_fit(
                65536, 17 * 1024 * 1024 * 1024, kv_cache_quant="q4_0",
                occupied_vram_mib=18_000)
        assert r_no_sibling.ok, f"positive control should fit: {r_no_sibling.detail}"
        assert not r_with_sibling.ok, (
            f"same-card RESERVED_LOADING sibling must shrink effective free "
            f"enough to refuse: {r_with_sibling.detail}"
        )

    def test_zero_occupied_admits_positive_control(self):
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[22_000]):
            r = check_kv_cache_fit(
                65536, 17 * 1024 * 1024 * 1024, kv_cache_quant="q4_0",
                occupied_vram_mib=0)
        assert r.ok

    def test_occupied_irrelevant_when_probe_unavailable(self):
        # free_mib is None (probe unreadable) -> occupied_vram_mib has
        # nothing to subtract against; parallel:1 keeps the historical
        # degrade-open doctrine regardless of occupied_vram_mib's value.
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=None):
            r = check_kv_cache_fit(
                65536, 17 * 1024 * 1024 * 1024, occupied_vram_mib=999_999)
        assert r.ok
        assert r.detail == "passed-no-probe"


# === 4. all_safety_gates — the missing middle: without the parameter this crashed on
# EVERY spawn, and nothing else in the
# repo covered it. The arm below calls all_safety_gates
# through the REAL call path with the kwarg set -- that is the difference
# between "it parses" and "it runs". ========================================

class TestAllSafetyGatesOccupiedVramMib:
    def _run(self, occupied_vram_mib, free_vram_mib_list=(22_000,)):
        # os.getloadavg/os.cpu_count patched here for the OLD
        # check_load_avg; dead now that the gate list calls check_cpu_util
        # (reads /proc/stat directly) -- replaced with a patch on that
        # gate's own probe, same fail-open idiom as the iowait patch on
        # the next line.
        with patch("turbohaul.safety._read_free_vram_all_mib",
                    return_value=list(free_vram_mib_list)), \
             patch("turbohaul.safety._read_meminfo_kib", return_value={}), \
             patch("turbohaul.safety._read_stat_cpu_jiffies", return_value=None), \
             patch("turbohaul.safety._read_stat_iowait_jiffies", return_value=None):
            return all_safety_gates(
                min_free_ram_mib=1024, min_free_vram_mib=1024,
                max_cpu_busy_percent=99.0, max_iowait_percent=30.0,
                ctx_size=65536, gguf_size_bytes=17 * 1024 * 1024 * 1024,
                kv_cache_quant="q4_0",
                occupied_vram_mib=occupied_vram_mib,
            )

    def test_real_call_path_with_kwarg_set_does_not_raise(self):
        # This exact call (occupied_vram_mib as a kwarg to the real
        # all_safety_gates) raises TypeError on every spawn attempt if
        # the parameter is missing -- visible by import-and-call and
        # independently by AST signature inspection.
        # This is the regression guard for exactly that defect class.
        results = self._run(occupied_vram_mib=1234)
        assert len(results) == 7  # gate count unaffected by the new param

    def test_occupied_vram_threads_into_the_vram_gate(self):
        results = self._run(occupied_vram_mib=0, free_vram_mib_list=(10_000,))
        vram_gate = next(g for g in results if g.name == "vram")
        assert vram_gate.ok
        results = self._run(occupied_vram_mib=9_500, free_vram_mib_list=(10_000,))
        vram_gate = next(g for g in results if g.name == "vram")
        assert not vram_gate.ok, "same-card reservation must reach the vram gate"

    def test_occupied_vram_threads_into_the_kv_cache_fit_gate(self):
        results = self._run(occupied_vram_mib=0, free_vram_mib_list=(22_000,))
        kv_gate = next(g for g in results if g.name == "kv_cache_fit")
        assert kv_gate.ok
        results = self._run(occupied_vram_mib=18_000, free_vram_mib_list=(22_000,))
        kv_gate = next(g for g in results if g.name == "kv_cache_fit")
        assert not kv_gate.ok, "same-card reservation must reach the kv_cache_fit gate"

    def test_default_zero_omitted_kwarg_still_works(self):
        # Every existing caller that doesn't know about occupied_vram_mib
        # (manager.py passes it everywhere, but the
        # parameter must stay optional for any future/test caller).
        # os.getloadavg/os.cpu_count patches were for the OLD
        # check_load_avg and went dead under check_cpu_util -- replaced
        # with a patch on that gate's own probe (same fail-open idiom as
        # the iowait patch on the next line).
        with patch("turbohaul.safety._read_meminfo_kib", return_value={}), \
             patch("turbohaul.safety._read_free_vram_mib", return_value=None), \
             patch("turbohaul.safety._read_free_vram_all_mib", return_value=None), \
             patch("turbohaul.safety._read_stat_iowait_jiffies", return_value=None), \
             patch("turbohaul.safety._read_stat_cpu_jiffies", return_value=None):
            results = all_safety_gates(
                min_free_ram_mib=1024, min_free_vram_mib=512,
                max_cpu_busy_percent=99.0, max_iowait_percent=30.0,
                iowait_sample_window_s=0.01, ctx_size=65536,
                gguf_size_bytes=17 * 1024 * 1024 * 1024, kv_cache_quant="f16",
            )
        assert len(results) == 7
