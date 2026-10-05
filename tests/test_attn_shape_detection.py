"""Derive attention shape (sliding-window vs SSM-hybrid vs plain) from GGUF
header metadata, with no architecture name in any conditional.

Fixtures are hand-built synthetic GGUF headers (real magic/version/KV-entry
bytes, no tensor data) reproducing the shapes of three kinds of model: a
sliding-window MoE (12 global + 36 sliding layers, window 512), a
dense model and a small model (both plain, scalar head_count, no
sliding_window key). Passing here proves the derivation logic; the parser is
separately validated against the real model files, which are not reachable
from the test environment.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from turbohaul._gguf_meta import read_kv_dims, _derive_full_attention_interval


# === Synthetic GGUF header builder =============================================

_VT_UINT32, _VT_FLOAT32, _VT_STRING, _VT_ARRAY, _VT_UINT64 = 4, 6, 8, 9, 10


def _enc_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _enc_kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _enc_str(key) + struct.pack("<I", vtype) + payload


def _enc_u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _enc_u32_array(values) -> bytes:
    return struct.pack("<I", _VT_UINT32) + struct.pack("<Q", len(values)) + b"".join(
        struct.pack("<I", v) for v in values
    )


def _build_gguf(kv_entries: list[bytes]) -> bytes:
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv_entries))
    return header + b"".join(kv_entries)


def _write(tmp_path, name: str, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


# === Fixtures, reproducing values measured from the real model files ===========

def _laguna_bytes(arch: str = "laguna") -> bytes:
    head_count = [48, 72, 72, 72] * 12  # period 4 over 48 blocks: 12 global, 36 swa
    assert len(head_count) == 48
    entries = [
        _enc_kv("general.architecture", _VT_STRING, _enc_str(arch)),
        _enc_kv(f"{arch}.block_count", _VT_UINT32, _enc_u32(48)),
        _enc_kv(f"{arch}.attention.head_count", _VT_ARRAY, _enc_u32_array(head_count)),
        _enc_kv(f"{arch}.attention.head_count_kv", _VT_UINT32, _enc_u32(8)),
        _enc_kv(f"{arch}.attention.sliding_window", _VT_UINT32, _enc_u32(512)),
        _enc_kv(f"{arch}.attention.key_length", _VT_UINT32, _enc_u32(128)),
        _enc_kv(f"{arch}.attention.value_length", _VT_UINT32, _enc_u32(128)),
    ]
    return _build_gguf(entries)


def _qwen36_bytes() -> bytes:
    arch = "qwen3.6-27b-dense"
    # GGUF key namespaces don't contain dots for the arch segment in real files;
    # the module treats arch purely as an opaque string prefix either way, so
    # this exercises that no special-casing of the literal is assumed.
    entries = [
        _enc_kv("general.architecture", _VT_STRING, _enc_str(arch)),
        _enc_kv(f"{arch}.block_count", _VT_UINT32, _enc_u32(65)),
        _enc_kv(f"{arch}.attention.head_count", _VT_UINT32, _enc_u32(24)),
        _enc_kv(f"{arch}.attention.head_count_kv", _VT_UINT32, _enc_u32(4)),
        _enc_kv(f"{arch}.attention.key_length", _VT_UINT32, _enc_u32(128)),
        _enc_kv(f"{arch}.attention.value_length", _VT_UINT32, _enc_u32(128)),
    ]
    return _build_gguf(entries)


def _tinyllama_bytes() -> bytes:
    arch = "tinyllama"
    entries = [
        _enc_kv("general.architecture", _VT_STRING, _enc_str(arch)),
        _enc_kv(f"{arch}.block_count", _VT_UINT32, _enc_u32(22)),
        _enc_kv(f"{arch}.attention.head_count", _VT_UINT32, _enc_u32(32)),
        _enc_kv(f"{arch}.attention.head_count_kv", _VT_UINT32, _enc_u32(4)),
        _enc_kv(f"{arch}.attention.key_length", _VT_UINT32, _enc_u32(64)),
        _enc_kv(f"{arch}.attention.value_length", _VT_UINT32, _enc_u32(64)),
    ]
    return _build_gguf(entries)


# === Tests =======================================================================

class TestLagunaSliding:
    def test_full_shape(self, tmp_path):
        path = _write(tmp_path, "laguna.gguf", _laguna_bytes())
        dims = read_kv_dims(path)
        assert dims is not None
        assert dims.sliding_window == 512
        assert dims.block_count == 48
        assert dims.full_attention_interval == 4
        assert dims.n_attn_layers == 12
        assert dims.n_swa_layers == 36
        assert dims.n_attn_layers + dims.n_swa_layers == dims.block_count
        assert dims.n_head_kv == 8

    def test_no_architecture_literal_involved(self, tmp_path):
        # Identical shape, built under a made-up arch name from scratch (not a
        # post-hoc byte patch -- length-prefixed KV strings would desync the
        # parse) -- proves no literal arch string is special-cased anywhere.
        renamed = _laguna_bytes(arch="totally-unknown-arch-xyz")
        path = _write(tmp_path, "laguna_renamed.gguf", renamed)
        dims = read_kv_dims(path)
        assert dims is not None
        assert dims.arch == "totally-unknown-arch-xyz"
        assert (dims.full_attention_interval, dims.n_attn_layers, dims.n_swa_layers) == (4, 12, 36)


class TestPlainModelsByteIdenticalBehaviour:
    def test_qwen36_dense_no_sliding(self, tmp_path):
        path = _write(tmp_path, "qwen36.gguf", _qwen36_bytes())
        dims = read_kv_dims(path)
        assert dims is not None
        assert dims.sliding_window is None
        assert dims.block_count == 65
        assert dims.full_attention_interval == 0  # no key, scalar head_count -> underivable
        assert dims.n_attn_layers == 65            # conservative: every layer growing
        assert dims.n_swa_layers == 0

    def test_tinyllama_no_sliding(self, tmp_path):
        path = _write(tmp_path, "tinyllama.gguf", _tinyllama_bytes())
        dims = read_kv_dims(path)
        assert dims is not None
        assert dims.sliding_window is None
        assert dims.block_count == 22
        assert dims.n_attn_layers == 22
        assert dims.n_swa_layers == 0


class TestMalformedNeverGuesses:
    def test_truncated_file_returns_none(self, tmp_path):
        path = _write(tmp_path, "truncated.gguf", _laguna_bytes()[:40])
        assert read_kv_dims(path) is None

    def test_not_a_gguf_file_returns_none(self, tmp_path):
        path = _write(tmp_path, "not_gguf.bin", b"NOTGGUF!" * 8)
        assert read_kv_dims(path) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert read_kv_dims(str(tmp_path / "does_not_exist.gguf")) is None

    def test_no_sliding_window_key_is_none_not_zero(self, tmp_path):
        # None must mean "absent", never coerced to a falsy-but-wrong 0/False.
        path = _write(tmp_path, "tinyllama.gguf", _tinyllama_bytes())
        dims = read_kv_dims(path)
        assert dims.sliding_window is None
        assert dims.sliding_window is not False
        assert dims.sliding_window != 0


class TestIntervalDerivationGuard:
    """_derive_full_attention_interval directly: prove it refuses to guess."""

    def test_laguna_pattern_derives_4(self):
        head_count = [48, 72, 72, 72] * 12
        assert _derive_full_attention_interval(head_count, 48) == 4

    def test_scalar_head_count_underivable(self):
        assert _derive_full_attention_interval(24, 65) == 0

    def test_wrong_length_underivable(self):
        assert _derive_full_attention_interval([48, 72, 72, 72] * 12, 47) == 0

    def test_three_distinct_values_underivable(self):
        # Not a clean two-tier split -- must not guess an interval.
        pattern = ([48, 72, 96, 72] * 12)
        assert _derive_full_attention_interval(pattern, 48) == 0

    def test_non_periodic_minority_underivable(self):
        # Two values, right counts, but NOT evenly spaced -- must not guess.
        pattern = [48, 48] + [72] * 46  # minority clustered at the front, not periodic
        assert _derive_full_attention_interval(pattern, 48) == 0

    def test_empty_or_zero_block_count_underivable(self):
        assert _derive_full_attention_interval([], 0) == 0
        assert _derive_full_attention_interval(None, 48) == 0
