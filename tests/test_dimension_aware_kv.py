"""Dimension-aware / measured-override KV-fit.

The file-size KV heuristic under-counts ultra-low-bit hybrids (the low-bit hybrid
g64) ~6x, so the KV-fit gate can admit an over-commit that OOMs at high ctx.
This adds two higher-precedence, ADDITIVE paths — measured override and parsed
GGUF dims — while leaving every existing model byte-identical.

Calibration ground truth (nvidia-smi on the manager container, measured):
  a 27B qwen35 hybrid, cache K=turbo3 / V=turbo2, 250k ctx →
  MEASURED marginal ~3,295 MiB per 250k context.
Measured dims (VERIFIED from the GGUF bytes): block_count=64,
  full_attention_interval=4 → 16 attention layers; head_count_kv=4;
  key_length=value_length=256.
"""
import struct

import pytest

from turbohaul._gguf_meta import KVDims, read_kv_dims
from turbohaul.safety import (
    all_safety_gates,
    check_kv_cache_fit,
    estimate_kv_cache_mib,
)

# --- Low-bit hybrid fixtures ----------------------------------------------------------
HYBRID_BYTES = 7233 * 1024 * 1024          # file body ≈ 7233 MiB (weights)
HYBRID_CTX = 250_000
MEASURED_MARGINAL_MIB = 3295               # nvidia-smi, per 250k slot


def _hybrid_dims() -> KVDims:
    return KVDims(
        arch="qwen35",
        block_count=64,
        full_attention_interval=4,
        n_head_kv=4,
        key_length=256,
        value_length=256,
    )


# === Calibration ==============================================================

class TestCalibration:
    def test_dims_path_first_principles(self):
        """16 attn layers · 4 KV heads · (256+256) · 2B, K=turbo3/V=turbo2, no
        hybrid multiply → 2,441 MiB (first-principles floor, 4.4x the 549 the
        file-size heuristic gives — the safety improvement)."""
        kv = estimate_kv_cache_mib(HYBRID_CTX, HYBRID_BYTES, "turbo3", "turbo2",
                                   attn_dims=_hybrid_dims())
        assert kv == 2441, kv

    def test_measured_override_reproduces_nvidia_smi(self):
        """Operator-measured override (≈13.5 KiB/token = 13824 B) reproduces the
        MEASURED 3,295 MiB within ±15% (in fact exactly)."""
        kv = estimate_kv_cache_mib(HYBRID_CTX, HYBRID_BYTES, "turbo3", "turbo2",
                                   kv_bytes_per_token=13824.0)
        lo, hi = MEASURED_MARGINAL_MIB * 0.85, MEASURED_MARGINAL_MIB * 1.15
        assert lo <= kv <= hi, f"{kv} not within ±15% of {MEASURED_MARGINAL_MIB}"
        assert kv == 3295, kv

    def test_legacy_heuristic_undercounts_as_documented(self):
        """The file-size + hybrid(0.25) path gives 549 MiB — the 6x under-count
        this RC fixes. Pinned to prove the legacy path is unchanged."""
        legacy = estimate_kv_cache_mib(HYBRID_CTX, HYBRID_BYTES, "turbo3",
                                       "turbo2", hybrid_kv_ratio=0.25)
        assert legacy == 549, legacy
        # The dims/override paths are both far higher → catch the over-commit.
        assert estimate_kv_cache_mib(HYBRID_CTX, HYBRID_BYTES, "turbo3", "turbo2",
                                     attn_dims=_hybrid_dims()) > legacy * 4


# === Precedence + no-compound (the reconciliation guard) ======================

class TestPrecedenceAndReconciliation:
    def test_override_outranks_dims(self):
        """Measured override wins over parsed dims (a live measurement beats a
        first-principles estimate)."""
        kv = estimate_kv_cache_mib(HYBRID_CTX, HYBRID_BYTES, "turbo3", "turbo2",
                                   attn_dims=_hybrid_dims(),
                                   kv_bytes_per_token=13824.0)
        assert kv == 3295, kv  # override, NOT the dims 2441

    def test_dims_path_ignores_hybrid_kv_ratio(self):
        """RECONCILIATION: dims path must NOT also apply hybrid_kv_ratio (16 attn
        layers already encode the hybrid 1/4 — re-applying would 4x under-count)."""
        d_default = estimate_kv_cache_mib(HYBRID_CTX, HYBRID_BYTES, "turbo3",
                                          "turbo2", attn_dims=_hybrid_dims())
        d_quarter = estimate_kv_cache_mib(HYBRID_CTX, HYBRID_BYTES, "turbo3",
                                          "turbo2", hybrid_kv_ratio=0.25,
                                          attn_dims=_hybrid_dims())
        assert d_default == d_quarter == 2441

    def test_override_path_ignores_hybrid_kv_ratio(self):
        o_default = estimate_kv_cache_mib(HYBRID_CTX, HYBRID_BYTES, "turbo3",
                                          "turbo2", kv_bytes_per_token=13824.0)
        o_quarter = estimate_kv_cache_mib(HYBRID_CTX, HYBRID_BYTES, "turbo3",
                                          "turbo2", hybrid_kv_ratio=0.25,
                                          kv_bytes_per_token=13824.0)
        assert o_default == o_quarter == 3295


# === Backward compatibility (byte-identical legacy) ===========================

class TestLegacyByteIdentical:
    def test_no_dims_no_override_is_unchanged(self):
        """Neither new arg set → the exact legacy result."""
        assert estimate_kv_cache_mib(65536, 17 * 1024 * 1024 * 1024, "f16") == 9792
        assert estimate_kv_cache_mib(65536, 17 * 1024 * 1024 * 1024, "q4_0") == 2432

    def test_hybrid_ratio_still_works_on_legacy_path(self):
        full = estimate_kv_cache_mib(65536, 17 * 1024 * 1024 * 1024, "f16")
        half = estimate_kv_cache_mib(65536, 17 * 1024 * 1024 * 1024, "f16",
                                     hybrid_kv_ratio=0.5)
        assert abs(half - full // 2) <= 2

    def test_zero_guards_unchanged(self):
        assert estimate_kv_cache_mib(0, HYBRID_BYTES, attn_dims=_hybrid_dims()) == 0
        assert estimate_kv_cache_mib(HYBRID_CTX, 0, attn_dims=_hybrid_dims()) == 0


# === Gate integration (the safety win) ========================================

class TestGateCatchesOvercommitDimsMisses:
    def test_dims_gate_refuses_where_legacy_admits(self):
        """On a GPU where body+legacy_KV fits but body+dims_KV does not, the
        dims-aware gate REFUSES the over-commit the file-size gate would admit.
          body 7233 + overhead 1024 = 8257
          + legacy KV 549  = 8806  → PASS on 9000
          + dims   KV 2441 = 10698 → REFUSE on 9000
        """
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("turbohaul.safety._read_free_vram_all_mib",
                       lambda *a, **k: [9000])
            legacy = check_kv_cache_fit(HYBRID_CTX, HYBRID_BYTES,
                                        kv_cache_quant="turbo3",
                                        kv_cache_quant_v="turbo2",
                                        hybrid_kv_ratio=0.25)
            dims = check_kv_cache_fit(HYBRID_CTX, HYBRID_BYTES,
                                      kv_cache_quant="turbo3",
                                      kv_cache_quant_v="turbo2",
                                      attn_dims=_hybrid_dims())
        assert legacy.ok, f"legacy heuristic should admit (under-count): {legacy.detail}"
        assert not dims.ok, f"dims gate should refuse over-commit: {dims.detail}"

    def test_all_safety_gates_threads_override(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("turbohaul.safety._read_free_vram_all_mib",
                       lambda *a, **k: [40_000])
            results = all_safety_gates(
                min_free_ram_mib=1024, min_free_vram_mib=512,
                max_load_per_core=99.0, max_iowait_percent=99.0,
                ctx_size=HYBRID_CTX, gguf_size_bytes=HYBRID_BYTES,
                kv_cache_quant="turbo3", kv_cache_quant_v="turbo2",
                kv_bytes_per_token=13824.0,
            )
        kv_gate = [g for g in results if g.name == "kv_cache_fit"][0]
        assert kv_gate.ok
        assert "3295" in kv_gate.detail, kv_gate.detail


# === _gguf_meta parser ========================================================

def _gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(k: str, v: str) -> bytes:
    return _gguf_string(k) + struct.pack("<I", 8) + _gguf_string(v)


def _kv_u32(k: str, v: int) -> bytes:
    return _gguf_string(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def _kv_arr_u32(k: str, vals: list) -> bytes:
    out = _gguf_string(k) + struct.pack("<I", 9)  # ARRAY
    out += struct.pack("<I", 4) + struct.pack("<Q", len(vals))  # elem u32 + count
    for v in vals:
        out += struct.pack("<I", v)
    return out


def _build_gguf(kvs: list, tensor_count: int = 0) -> bytes:
    body = b"GGUF" + struct.pack("<I", 3)
    body += struct.pack("<Q", tensor_count) + struct.pack("<Q", len(kvs))
    return body + b"".join(kvs)


class TestGgufMetaParser:
    def test_parses_hybrid_header(self, tmp_path):
        p = tmp_path / "hybrid.gguf"
        p.write_bytes(_build_gguf([
            _kv_str("general.architecture", "qwen35"),
            _kv_arr_u32("qwen35.some_array", [1, 2, 3]),   # exercise array-skip
            _kv_u32("qwen35.block_count", 64),
            _kv_u32("qwen35.full_attention_interval", 4),
            _kv_u32("qwen35.attention.head_count_kv", 4),
            _kv_u32("qwen35.attention.key_length", 256),
            _kv_u32("qwen35.attention.value_length", 256),
        ], tensor_count=851))
        d = read_kv_dims(p)
        assert d is not None
        assert d.arch == "qwen35"
        assert d.block_count == 64
        assert d.n_head_kv == 4
        assert d.key_length == 256 and d.value_length == 256
        assert d.n_attn_layers == 16
        assert d.is_usable()

    def test_head_dim_fallback_from_embedding(self, tmp_path):
        """key/value_length absent → head_dim = embedding_length // head_count."""
        p = tmp_path / "fallback.gguf"
        p.write_bytes(_build_gguf([
            _kv_str("general.architecture", "qwen35"),
            _kv_u32("qwen35.block_count", 64),
            _kv_u32("qwen35.full_attention_interval", 4),
            _kv_u32("qwen35.attention.head_count_kv", 4),
            _kv_u32("qwen35.embedding_length", 5120),
            _kv_u32("qwen35.attention.head_count", 20),  # 5120//20 = 256
        ]))
        d = read_kv_dims(p)
        assert d is not None
        assert d.key_length == 256 and d.value_length == 256

    def test_malformed_returns_none(self, tmp_path):
        p = tmp_path / "bad.gguf"
        p.write_bytes(b"NOPEyadda")
        assert read_kv_dims(p) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert read_kv_dims(tmp_path / "does_not_exist.gguf") is None

    def test_missing_attention_keys_returns_none(self, tmp_path):
        p = tmp_path / "noattn.gguf"
        p.write_bytes(_build_gguf([
            _kv_str("general.architecture", "llama"),
            _kv_u32("llama.block_count", 32),
        ]))
        assert read_kv_dims(p) is None

    def test_n_attn_layers_full_when_no_interval(self):
        d = KVDims("qwen35", 64, 0, 4, 256, 256)
        assert d.n_attn_layers == 64  # conservative: all layers attention

    def test_non_path_arg_never_raises(self):
        """Contract: read_kv_dims returns None (never raises) even for a
        non-path-like arg (None/list/object) reaching open()."""
        assert read_kv_dims(None) is None
        assert read_kv_dims([1, 2]) is None
        assert read_kv_dims(object()) is None
