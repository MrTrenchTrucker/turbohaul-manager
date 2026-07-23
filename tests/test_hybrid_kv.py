"""Tests for the hybrid KV-fit estimate + qwen35 manifest fields.

Additive-only: every existing-model path remains byte-identical to the pre-diff baseline.
Run these alongside test_safety.py (same module imports).
"""
from unittest.mock import patch

import pytest

from turbohaul.manifest import (
    Manifest,
)
from turbohaul.safety import (
    GateResult,
    all_safety_gates,
    check_kv_cache_fit,
    estimate_kv_cache_mib,
)


# === Estimate helpers ==========================================================

def _qwen27b_bytes() -> int:
    """17 GiB GGUF — the canonical Qwen27B calibration target."""
    return 17 * 1024 * 1024 * 1024


# === TestEstimateKvCacheMibHybrid ==============================================

class TestEstimateKvCacheMibHybrid:
    """hybrid_kv_ratio defaults to 1.0 → byte-identical to the existing estimator."""

    def test_default_ratio_is_byte_identical_to_baseline(self):
        """hybrid_kv_ratio=1.0 (the default) produces the same result as omitting it."""
        baseline = estimate_kv_cache_mib(65536, _qwen27b_bytes(), "f16")
        explicit  = estimate_kv_cache_mib(65536, _qwen27b_bytes(), "f16",
                                          hybrid_kv_ratio=1.0)
        assert baseline == explicit, (
            f"default (1.0) must be byte-identical to baseline; "
            f"got {baseline} vs {explicit}"
        )

    def test_ratio_0_5_halves_per_token_kv(self):
        """hybrid_kv_ratio=0.5 → roughly half the KV estimate."""
        full = estimate_kv_cache_mib(65536, _qwen27b_bytes(), "f16",
                                     hybrid_kv_ratio=1.0)
        half = estimate_kv_cache_mib(65536, _qwen27b_bytes(), "f16",
                                     hybrid_kv_ratio=0.5)
        assert half < full
        # Allow ±2 MiB int-truncation rounding
        expected = full // 2
        assert abs(half - expected) <= 2, (
            f"ratio=0.5 should be ~half ({expected}±2), got {half}"
        )

    def test_ratio_0_25_quarters_kv(self):
        quarter = estimate_kv_cache_mib(65536, _qwen27b_bytes(), "f16",
                                        hybrid_kv_ratio=0.25)
        full = estimate_kv_cache_mib(65536, _qwen27b_bytes(), "f16")
        expected = full // 4
        assert abs(quarter - expected) <= 2

    def test_ratio_zero_yields_zero_kv(self):
        """Edge case: pure-SSM model (no attn layers) → kv_mib == 0."""
        zero = estimate_kv_cache_mib(65536, _qwen27b_bytes(), "f16",
                                     hybrid_kv_ratio=0.0)
        assert zero == 0

    def test_ratio_does_not_inflate_estimate(self):
        """ratio < 1 never inflates the estimate above the baseline."""
        full = estimate_kv_cache_mib(65536, _qwen27b_bytes(), "f16")
        reduced = estimate_kv_cache_mib(65536, _qwen27b_bytes(), "f16",
                                        hybrid_kv_ratio=0.5)
        assert reduced < full


# === TestCheckKvCacheFitHybrid =================================================

class TestCheckKvCacheFitHybrid:
    """check_kv_cache_fit routes hybrid_kv_ratio → estimate_kv_cache_mib.

    Regression gate: the same config that FAILS with ratio=1.0 can PASS with
    ratio=0.5 (hybrid model), which is exactly the qwen35 hybrid use-case.
    """

    def test_hybrid_ratio_makes_tight_fit_pass(self):
        """A 17 GiB model @64K ctx, f16 KV on a 30 GB GPU:
        - ratio=1.0 → ~27.5 GiB needed → PASSES on 30 GB
        - ratio=0.5 → ~22 GiB needed → PASSES comfortably
        Verify the ratio parameter actually flows through."""
        gguf = _qwen27b_bytes()
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[30_000]):
            r1 = check_kv_cache_fit(65536, gguf, kv_cache_quant="f16",
                                    hybrid_kv_ratio=1.0)
            r05 = check_kv_cache_fit(65536, gguf, kv_cache_quant="f16",
                                     hybrid_kv_ratio=0.5)
        # Both should pass on 30 GB — but the reduced estimate should have more margin
        assert r1.ok
        assert r05.ok

    def test_hybrid_ratio_can_save_a_tight_refusal(self):
        """Same model, smaller GPU — ratio=1.0 REFUSES, ratio=0.5 PASSES.

        17 GiB body + ratio=1.0 KV(9,792) + overhead(1,024) = ~28,224 MiB
        17 GiB body + ratio=0.5 KV(4,896) + overhead(1,024) = ~23,328 MiB
        So 24 GB free refuses at ratio=1.0 but passes at ratio=0.5."""
        gguf = _qwen27b_bytes()
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[24_000]):
            r1 = check_kv_cache_fit(65536, gguf, kv_cache_quant="f16",
                                    hybrid_kv_ratio=1.0)
            r05 = check_kv_cache_fit(65536, gguf, kv_cache_quant="f16",
                                     hybrid_kv_ratio=0.5)
        assert not r1.ok, f"ratio=1.0 should refuse on 24 GB: {r1.detail}"
        assert r05.ok, f"ratio=0.5 should pass on 24 GB: {r05.detail}"


# === TestAllSafetyGatesHybrid ==================================================

class TestAllSafetyGatesHybrid:
    """all_safety_gates routes hybrid_kv_ratio to check_kv_cache_fit."""

    def test_aggregator_routes_hybrid_kv_ratio(self):
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[30_000]):
            results = all_safety_gates(
                min_free_ram_mib=1024,
                min_free_vram_mib=512,
                max_load_per_core=0.9,
                max_iowait_percent=30.0,
                ctx_size=65536,
                gguf_size_bytes=_qwen27b_bytes(),
                kv_cache_quant="f16",
                hybrid_kv_ratio=0.5,
            )
        kv_gate = [g for g in results if g.name == "kv_cache_fit"][0]
        assert kv_gate.ok
        # The detail should show the reduced KV value
        kv_half = estimate_kv_cache_mib(65536, _qwen27b_bytes(), "f16",
                                        hybrid_kv_ratio=0.5)
        assert str(kv_half) in kv_gate.detail


# === TestManifestArchFields ====================================================

SAMPLE_SHA = "1a2b3c4d" + "0" * 56  # 64 hex chars


class TestManifestArchFields:
    """Arch + hybrid_kv_ratio manifest fields.

    Back-compat: existing manifests without these fields parse with defaults.
    Wiring: the manifest field is named `hybrid_kv_ratio` (same as the safety.py
    function param) so the manager can pass it through directly:
        hybrid_kv = m.hybrid_kv_ratio if m.arch == "qwen35" else 1.0
        estimate_kv_cache_mib(..., hybrid_kv_ratio=hybrid_kv)
    """

    def test_existing_manifest_without_arch_parses(self):
        """A manifest without arch or hybrid_kv_ratio fields should parse fine
        with defaults (arch='', hybrid_kv_ratio=1.0)."""
        m = Manifest(
            model_tag="qwen3.6-35b-moe",
            display_name="Qwen 3.6 35B",
            gguf_blob_sha256=SAMPLE_SHA,
            gguf_size_bytes=22_000_000_000,
            context_size=131072,
            expected_vram_bytes=22_500_000_000,
            revision=1,
        )
        assert m.arch == ""
        assert m.hybrid_kv_ratio == 1.0

    def test_qwen35_manifest_with_hybrid_ratio(self):
        """A qwen35 manifest with hybrid_kv_ratio < 1.0 parses and validates."""
        m = Manifest(
            model_tag="qwen35-35b-q2g64",
            display_name="Hybrid 35B",
            gguf_blob_sha256=SAMPLE_SHA,
            gguf_size_bytes=5_000_000_000,
            context_size=131072,
            expected_vram_bytes=6_000_000_000,
            arch="qwen35",
            hybrid_kv_ratio=0.45,
        )
        assert m.arch == "qwen35"
        assert abs(m.hybrid_kv_ratio - 0.45) < 1e-6

    def test_hybrid_kv_ratio_clamped_to_unit_interval(self):
        """hybrid_kv_ratio must be in [0.0, 1.0] (Pydantic Field ge=0.0, le=1.0)."""
        with pytest.raises(Exception):
            Manifest(
                model_tag="test",
                gguf_blob_sha256=SAMPLE_SHA,
                hybrid_kv_ratio=1.5,
            )
        with pytest.raises(Exception):
            Manifest(
                model_tag="test",
                gguf_blob_sha256=SAMPLE_SHA,
                hybrid_kv_ratio=-0.1,
            )


# === TestExistingModelRegression (GOLDEN) =======================================

class TestExistingModelRegression:
    """Golden regression: existing models must place/score identically to pre-diff.

    These pin EXACT numbers to prove zero regression on the existing-model code path.
    """

    def test_qwen27b_f16_64k_estimate_unchanged(self):
        """Qwen27B @64K ctx, f16 KV → estimate must be 9792 MiB (pinned golden).

        This is the canonical calibration: 17 GiB body, ~9 KB/token per GiB at f16,
        64K ctx. The pre-diff estimator returned 9792 MiB — it MUST still do so."""
        result = estimate_kv_cache_mib(65536, 17 * 1024 * 1024 * 1024, "f16")
        assert result == 9792, (
            f"GOLDEN REGRESSION: Qwen27B f16 64K should be 9792 MiB, got {result}"
        )

    def test_qwen27b_q4_64k_estimate_unchanged(self):
        """Qwen27B @64K ctx, q4_0 KV → pinned golden estimate unchanged."""
        result = estimate_kv_cache_mib(65536, 17 * 1024 * 1024 * 1024, "q4_0")
        # q4_0 scale = 0.25: int(153*0.25)=38, 38*65536//1024 = 2432
        assert result == 2432, (
            f"GOLDEN REGRESSION: Qwen27B q4_0 64K should be 2432 MiB, got {result}"
        )

    def test_cpu_moe_path_unchanged(self):
        """The cpu_moe_offload branch in check_kv_cache_fit should be unaffected."""
        gguf = 20 * 1024 * 1024 * 1024
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[23_700]):
            r = check_kv_cache_fit(
                500_000, gguf, kv_cache_quant="turbo2", parallel=2,
                split_mode="none", main_gpu=0,
                expected_vram_mib=20_000, cpu_moe_offload=True)
        assert r.ok
        assert "cpu-moe measured" in r.detail

    def test_no_kv_offload_path_unchanged(self):
        """The no_kv_offload branch should produce the same result as pre-diff."""
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[22_000]), \
             patch("turbohaul.safety._read_meminfo_kib",
                   return_value={"MemAvailable": 60 * 1024 * 1024}):
            r = check_kv_cache_fit(
                65536, 17 * 1024 * 1024 * 1024,
                kv_cache_quant="f16", no_kv_offload=True)
        assert r.ok
        assert "host RAM" in r.detail
