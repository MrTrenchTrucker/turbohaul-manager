"""Tests for safety guardrails, including the KV-cache fit gate."""
from unittest.mock import patch

from turbohaul.safety import (
    GateResult,
    all_safety_gates,
    check_free_ram,
    check_kv_cache_fit,
    check_load_avg,
    check_iowait,
    estimate_kv_cache_mib,
)


class TestCheckFreeRam:
    def test_passes_when_above_threshold(self):
        with patch(
            "turbohaul.safety._read_meminfo_kib",
            return_value={"MemAvailable": 4 * 1024 * 1024},  # 4 GiB
        ):
            r = check_free_ram(min_free_mib=1024)
        assert r.ok
        assert r.name == "ram"

    def test_fails_when_below_threshold(self):
        with patch(
            "turbohaul.safety._read_meminfo_kib",
            return_value={"MemAvailable": 256 * 1024},  # 256 MiB
        ):
            r = check_free_ram(min_free_mib=1024)
        assert not r.ok
        assert "only 256 MiB free" in r.detail

    def test_passes_no_probe_when_meminfo_unavailable(self):
        with patch("turbohaul.safety._read_meminfo_kib", return_value={}):
            r = check_free_ram(min_free_mib=1024)
        assert r.ok
        assert r.detail == "passed-no-probe"


class TestCheckLoadAvg:
    def test_passes_when_load_low(self):
        with patch("os.getloadavg", return_value=(0.5, 0.3, 0.2)):
            with patch("os.cpu_count", return_value=8):
                r = check_load_avg(max_per_core=0.9)
        assert r.ok

    def test_fails_when_load_high(self):
        with patch("os.getloadavg", return_value=(16.0, 12.0, 8.0)):
            with patch("os.cpu_count", return_value=8):
                r = check_load_avg(max_per_core=0.9)
        assert not r.ok
        assert "2.00" in r.detail or "2.0" in r.detail


class TestCheckIowait:
    def test_passes_no_probe_when_proc_stat_missing(self):
        with patch(
            "turbohaul.safety._read_stat_iowait_jiffies", return_value=None,
        ):
            r = check_iowait(max_percent=30.0, sample_window_s=0.01)
        assert r.ok
        assert "passed-no-probe" in r.detail

    def test_passes_when_iowait_low(self):
        samples = [(1000, 10), (1100, 12)]  # 2/100 = 2%
        with patch(
            "turbohaul.safety._read_stat_iowait_jiffies",
            side_effect=samples,
        ):
            r = check_iowait(max_percent=30.0, sample_window_s=0.01)
        assert r.ok
        assert "iowait 2.0%" in r.detail

    def test_fails_when_iowait_high(self):
        samples = [(1000, 100), (1100, 200)]  # delta 100/100 = 100%
        with patch(
            "turbohaul.safety._read_stat_iowait_jiffies",
            side_effect=samples,
        ):
            r = check_iowait(max_percent=30.0, sample_window_s=0.01)
        assert not r.ok
        assert "100.0%" in r.detail


class TestEstimateKvCacheMib:
    def test_zero_inputs_return_zero(self):
        assert estimate_kv_cache_mib(0, 0) == 0
        assert estimate_kv_cache_mib(4096, 0) == 0
        assert estimate_kv_cache_mib(0, 17000000000) == 0

    def test_f16_27b_at_64k_in_expected_range(self):
        # 17 GB gguf, 64K ctx, f16 → expect ~9000-10000 MiB (calibration)
        kv = estimate_kv_cache_mib(65536, 17 * 1024 * 1024 * 1024, "f16")
        # Allow wide band — formula is heuristic, exact varies per model
        assert 5000 < kv < 18000, f"expected 5-18 GB band, got {kv} MiB"

    def test_q8_0_halves_f16(self):
        f16 = estimate_kv_cache_mib(32768, 17 * 1024 * 1024 * 1024, "f16")
        q8 = estimate_kv_cache_mib(32768, 17 * 1024 * 1024 * 1024, "q8_0")
        # Allow some int-truncation rounding error
        assert q8 < f16 // 2 + 50

    def test_scales_linearly_with_ctx(self):
        small = estimate_kv_cache_mib(4096, 17 * 1024 * 1024 * 1024, "f16")
        big = estimate_kv_cache_mib(65536, 17 * 1024 * 1024 * 1024, "f16")
        # 16x ctx → ~16x KV (allow rounding)
        ratio = big / max(small, 1)
        assert 14 < ratio < 18, f"expected ~16x scaling, got {ratio:.2f}x"


class TestCheckKvCacheFit:
    # P1a fix: the dual-GPU gate patch re-routed ``check_kv_cache_fit`` to read
    # ``_vram_budget(...)`` -> ``_read_free_vram_all_mib()`` (per-GPU free list),
    # NOT the legacy GPU0-only ``_read_free_vram_mib``. These tests previously
    # mocked the legacy fn, which the function under test no longer calls, so the
    # mocks were inert and the assertions fell through to the real host GPU.
    # Mock ``_read_free_vram_all_mib`` with a single-element list (one card ->
    # ``_vram_budget`` returns ``(X, X, 1)``, identical to the legacy single-GPU
    # budget) so the gate is deterministic and host-GPU independent again. Same
    # intent asserted: body + KV + overhead vs free VRAM (and host-RAM KV fit).
    def test_passes_no_probe_when_nvidia_smi_unavailable(self):
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=None):
            r = check_kv_cache_fit(65536, 17 * 1024 * 1024 * 1024)
        assert r.ok
        assert r.detail == "passed-no-probe"

    def test_passes_when_fits(self):
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80_000]):
            r = check_kv_cache_fit(65536, 17 * 1024 * 1024 * 1024)
        assert r.ok
        assert "need" in r.detail

    def test_cpu_moe_trusts_measured_expected_vram(self):
        # A cpu-moe (n_cpu_moe) config whose CLOSED-FORM body+KV would NOT fit, but
        # whose MEASURED expected_vram (experts offloaded to RAM) does. The cpu-moe
        # branch trusts the measured value and PASSES; the same config WITHOUT
        # cpu_moe_offload refuses on the over-counted body. (regression case for a 35B MoE model.)
        gguf = 20 * 1024 * 1024 * 1024  # 20 GiB body -> closed-form blows the budget
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[23_700]):
            r_closed = check_kv_cache_fit(
                500_000, gguf, kv_cache_quant="turbo2", parallel=2,
                split_mode="none", main_gpu=0)
            assert not r_closed.ok, "closed-form over-counts offloaded body -> refuse"
            r_cpumoe = check_kv_cache_fit(
                500_000, gguf, kv_cache_quant="turbo2", parallel=2,
                split_mode="none", main_gpu=0,
                expected_vram_mib=20_000, cpu_moe_offload=True)
        assert r_cpumoe.ok, "cpu-moe trusts the measured 20 GiB footprint -> fits"
        assert "cpu-moe measured" in r_cpumoe.detail

    def test_cpu_moe_still_refuses_when_measured_exceeds_free(self):
        # cpu-moe trusts the measured value but it must STILL fit free VRAM.
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[18_000]):
            r = check_kv_cache_fit(
                500_000, 20 * 1024 * 1024 * 1024, kv_cache_quant="turbo2", parallel=2,
                split_mode="none", main_gpu=0,
                expected_vram_mib=22_000, cpu_moe_offload=True)
        assert not r.ok, "measured 22 GiB + overhead > 18 GiB free -> refuse"

    def test_fails_when_kv_exceeds_free(self):
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[22_000]):
            # 17 GB body + ~9.5 GB KV at 64K + 1 GB overhead = ~27.5 GB > 22 GB
            r = check_kv_cache_fit(65536, 17 * 1024 * 1024 * 1024, kv_cache_quant="f16")
        assert not r.ok
        assert "need" in r.detail and "22000 MiB free" in r.detail

    def test_q4_lets_64k_27b_fit_on_blackwell(self):
        # 24 GB Blackwell with ~22 GB free. q4_0 at 64K on a 27B model:
        # 17 GB body + ~2.4 GB KV + 1 GB overhead = ~20.4 GB → fits
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[22_000]):
            r = check_kv_cache_fit(65536, 17 * 1024 * 1024 * 1024, kv_cache_quant="q4_0")
        assert r.ok, f"q4_0 64K 27B model should fit on 22 GB free, got: {r.detail}"

    def test_q8_64k_27b_refused_on_blackwell(self):
        # Same hardware + q8_0 instead of q4_0. q8_0 ≈ 4.9 GB KV → 17+4.9+1 = ~23 GB > 22 GB
        # safety_gate CORRECTLY refuses (correct behavior — q8_0 needs bigger GPU)
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[22_000]):
            r = check_kv_cache_fit(65536, 17 * 1024 * 1024 * 1024, kv_cache_quant="q8_0")
        assert not r.ok, f"q8_0 64K 27B model should refuse on 22 GB Blackwell, got: {r.detail}"

    def test_passes_when_inputs_unknown(self):
        r = check_kv_cache_fit(0, 0)
        assert r.ok
        assert "passed-insufficient-input" in r.detail

    def test_no_kv_offload_excludes_kv_from_vram(self):
        # Same f16 64K case that FAILS in VRAM (test_fails_when_kv_exceeds_free)
        # now PASSES with --no-kv-offload: the KV moves to host RAM, so the VRAM
        # need is just body(17 GB) + overhead(1 GB) = ~18 GB < 22 GB free.
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[22_000]), \
             patch(
                 "turbohaul.safety._read_meminfo_kib",
                 return_value={"MemAvailable": 60 * 1024 * 1024},  # ~60 GiB free RAM
             ):
            r = check_kv_cache_fit(
                65536, 17 * 1024 * 1024 * 1024,
                kv_cache_quant="f16", no_kv_offload=True,
            )
        assert r.ok, f"no_kv_offload should fit (KV in host RAM), got: {r.detail}"
        assert "host RAM" in r.detail

    def test_no_kv_offload_fails_if_ram_insufficient(self):
        # --no-kv-offload but host RAM cannot hold the KV cache → refuse.
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[22_000]), \
             patch(
                 "turbohaul.safety._read_meminfo_kib",
                 return_value={"MemAvailable": 100 * 1024},  # ~100 MiB free RAM
             ):
            r = check_kv_cache_fit(
                65536, 17 * 1024 * 1024 * 1024,
                kv_cache_quant="f16", no_kv_offload=True,
            )
        assert not r.ok
        assert "host" in r.detail and "RAM" in r.detail

    def test_no_kv_offload_fails_if_body_exceeds_vram(self):
        # --no-kv-offload still refuses if the model BODY alone won't fit VRAM.
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[10_000]), \
             patch(
                 "turbohaul.safety._read_meminfo_kib",
                 return_value={"MemAvailable": 60 * 1024 * 1024},
             ):
            r = check_kv_cache_fit(
                65536, 17 * 1024 * 1024 * 1024,
                kv_cache_quant="f16", no_kv_offload=True,
            )
        assert not r.ok
        assert "VRAM" in r.detail


class TestAllGatesAggregate:
    def test_returns_5_gates(self):
        with patch("turbohaul.safety._read_meminfo_kib", return_value={}):
            with patch(
                "turbohaul.safety._read_free_vram_mib", return_value=None,
            ):
                with patch(
                    "turbohaul.safety._read_stat_iowait_jiffies",
                    return_value=None,
                ):
                    with patch("os.getloadavg", return_value=(0.1, 0.1, 0.1)):
                        with patch("os.cpu_count", return_value=4):
                            results = all_safety_gates(
                                min_free_ram_mib=1024,
                                min_free_vram_mib=512,
                                max_load_per_core=0.9,
                                max_iowait_percent=30.0,
                                iowait_sample_window_s=0.01,
                                ctx_size=65536,
                                gguf_size_bytes=17 * 1024 * 1024 * 1024,
                                kv_cache_quant="f16",
                            )
        assert len(results) == 5
        names = {g.name for g in results}
        assert names == {"ram", "vram", "kv_cache_fit", "cpu_load", "iowait"}
