"""Tests for safety guardrails ( KV-cache fit gate)."""
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

    def test_f16_qwen27b_at_64k_in_expected_range(self):
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
    def test_passes_no_probe_when_nvidia_smi_unavailable(self):
        with patch("turbohaul.safety._read_free_vram_mib", return_value=None):
            r = check_kv_cache_fit(65536, 17 * 1024 * 1024 * 1024)
        assert r.ok
        assert r.detail == "passed-no-probe"

    def test_passes_when_fits(self):
        with patch("turbohaul.safety._read_free_vram_mib", return_value=80_000):
            r = check_kv_cache_fit(65536, 17 * 1024 * 1024 * 1024)
        assert r.ok
        assert "need" in r.detail

    def test_fails_when_kv_exceeds_free(self):
        with patch("turbohaul.safety._read_free_vram_mib", return_value=22_000):
            # 17 GB body + ~9.5 GB KV at 64K + 1 GB overhead = ~27.5 GB > 22 GB
            r = check_kv_cache_fit(65536, 17 * 1024 * 1024 * 1024, kv_cache_quant="f16")
        assert not r.ok
        assert "need" in r.detail and "22000 MiB free" in r.detail

    def test_q4_lets_64k_qwen27b_fit_on_blackwell(self):
        # 24 GB Blackwell with ~22 GB free. q4_0 at 64K on Qwen27B:
        # 17 GB body + ~2.4 GB KV + 1 GB overhead = ~20.4 GB → fits
        with patch("turbohaul.safety._read_free_vram_mib", return_value=22_000):
            r = check_kv_cache_fit(65536, 17 * 1024 * 1024 * 1024, kv_cache_quant="q4_0")
        assert r.ok, f"q4_0 64K Qwen27B should fit on 22 GB free, got: {r.detail}"

    def test_q8_64k_qwen27b_refused_on_blackwell(self):
        # Same hardware + q8_0 instead of q4_0. q8_0 ≈ 4.9 GB KV → 17+4.9+1 = ~23 GB > 22 GB
        # safety_gate CORRECTLY refuses (correct behavior — q8_0 needs bigger GPU)
        with patch("turbohaul.safety._read_free_vram_mib", return_value=22_000):
            r = check_kv_cache_fit(65536, 17 * 1024 * 1024 * 1024, kv_cache_quant="q8_0")
        assert not r.ok, f"q8_0 64K Qwen27B should refuse on 22 GB Blackwell, got: {r.detail}"

    def test_passes_when_inputs_unknown(self):
        r = check_kv_cache_fit(0, 0)
        assert r.ok
        assert "passed-insufficient-input" in r.detail


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
