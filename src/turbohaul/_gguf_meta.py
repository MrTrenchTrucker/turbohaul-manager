"""Minimal, stdlib-only GGUF *metadata* reader — attention dims for the KV-fit
dimension-aware estimate.

Why a hand-rolled reader (not gguf-py): the engine serves models whose weight
tensors use engine-custom quant type-ids that are ABSENT from gguf-py's
``GGMLQuantizationType`` enum, so ``GGUFReader`` raises on them.
This reader touches ONLY the GGUF key/value header (never a tensor-info ``type``
field and never any tensor data), so it is immune to custom quant ids — it reads
the scalar/string KV entries it needs and returns a small ``KVDims``.

Everything here is READ-ONLY and best-effort: any malformed / unexpected input
makes ``read_kv_dims`` return ``None`` and the caller falls back to the legacy
file-size KV heuristic (byte-identical behaviour). It never raises to the caller.

GGUFv3 header layout (little- or big-endian, auto-detected):
  magic "GGUF" · version(u32 ∈ {2,3}) · tensor_count(u64) · kv_count(u64) ·
  kv_count × [ key(gguf_string) · value_type(u32) · value ]  ·  <tensor infos…>
gguf_string = len(u64) + UTF-8 bytes.  We stop after the KV block.
"""
from __future__ import annotations

import struct
from typing import NamedTuple

_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = (2, 3)

# GGUFValueType enum (0-12). ARRAY=9 carries an element type + count.
_VT_UINT8, _VT_INT8 = 0, 1
_VT_UINT16, _VT_INT16 = 2, 3
_VT_UINT32, _VT_INT32 = 4, 5
_VT_FLOAT32 = 6
_VT_BOOL = 7
_VT_STRING = 8
_VT_ARRAY = 9
_VT_UINT64, _VT_INT64 = 10, 11
_VT_FLOAT64 = 12

# Fixed-width scalar value types → struct format char (endianness prepended).
_SCALAR_FMT = {
    _VT_UINT8: "B", _VT_INT8: "b",
    _VT_UINT16: "H", _VT_INT16: "h",
    _VT_UINT32: "I", _VT_INT32: "i",
    _VT_FLOAT32: "f",
    _VT_BOOL: "B",
    _VT_UINT64: "Q", _VT_INT64: "q",
    _VT_FLOAT64: "d",
}

# Defensive caps: a genuine GGUF header is small. Refuse absurd counts/lengths so
# a corrupt or hostile file can't drive a huge allocation or a long scan.
_MAX_KV_COUNT = 1 << 20          # 1,048,576 KV entries
_MAX_STRING_LEN = 64 * 1024 * 1024
_MAX_ARRAY_COUNT = 1 << 26
_MAX_ARRAY_DEPTH = 64


class KVDims(NamedTuple):
    """Attention dims needed to size a growing per-token KV cache.

    ``key_length`` / ``value_length`` are the per-head K/V dimensions.
    ``full_attention_interval`` (qwen35 hybrid) means every Nth layer is a
    full-attention layer that keeps a *growing* KV cache; the others are SSM
    (fixed recurrent state, no per-token KV growth).
    """

    arch: str
    block_count: int
    full_attention_interval: int
    n_head_kv: int
    key_length: int
    value_length: int

    @property
    def n_attn_layers(self) -> int:
        """Layers that contribute a growing per-token KV cache.

        qwen35 hybrid: block_count // full_attention_interval (e.g. 64 // 4 = 16).
        When the interval is absent/0 we conservatively count EVERY layer as
        attention (over-estimates KV → never under-reserves)."""
        if self.full_attention_interval and self.full_attention_interval > 0:
            return max(1, self.block_count // self.full_attention_interval)
        return self.block_count

    def is_usable(self) -> bool:
        return (
            self.block_count > 0
            and self.n_head_kv > 0
            and self.key_length > 0
            and self.value_length > 0
        )


class _Reader:
    """Sequential struct reader over a file object, endian-aware."""

    def __init__(self, f, endian: str):
        self._f = f
        self._e = endian  # "<" or ">"

    def _read(self, n: int) -> bytes:
        b = self._f.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return b

    def u32(self) -> int:
        return struct.unpack(self._e + "I", self._read(4))[0]

    def u64(self) -> int:
        return struct.unpack(self._e + "Q", self._read(8))[0]

    def gguf_string(self) -> str:
        n = self.u64()
        if n > _MAX_STRING_LEN:
            raise ValueError(f"gguf_string length {n} exceeds cap")
        return self._read(n).decode("utf-8", "replace")

    def value(self, vtype: int, depth: int = 0):
        fmt = _SCALAR_FMT.get(vtype)
        if fmt is not None:
            size = struct.calcsize(fmt)
            v = struct.unpack(self._e + fmt, self._read(size))[0]
            if vtype == _VT_BOOL:
                return bool(v)
            return v
        if vtype == _VT_STRING:
            return self.gguf_string()
        if vtype == _VT_ARRAY:
            if depth >= _MAX_ARRAY_DEPTH:
                raise ValueError("array nesting too deep")
            elem_type = self.u32()
            count = self.u64()
            if count > _MAX_ARRAY_COUNT:
                raise ValueError(f"array count {count} exceeds cap")
            # Consume every element so the stream stays aligned. We do not use
            # array values here, but must not desync the KV walk.
            for _ in range(count):
                self.value(elem_type, depth + 1)
            return None
        raise ValueError(f"unknown GGUF value type {vtype}")


def _detect_endian(f) -> str:
    """Return '<' or '>' after validating magic + version. Leaves the file
    positioned right after the 4-byte version field."""
    magic = f.read(4)
    if magic != _MAGIC:
        raise ValueError(f"not a GGUF file (magic={magic!r})")
    raw = f.read(4)
    if len(raw) != 4:
        raise EOFError("truncated version")
    for endian in ("<", ">"):
        ver = struct.unpack(endian + "I", raw)[0]
        if ver in _SUPPORTED_VERSIONS:
            return endian
    raise ValueError(f"unsupported GGUF version bytes {raw!r}")


def read_kv_dims(path) -> "KVDims | None":
    """Parse the GGUF KV header at ``path`` and return attention ``KVDims``.

    Returns None (never raises) when the file is missing/malformed, is not a
    qwen35 model, or lacks the attention keys — the caller then uses the legacy
    file-size KV heuristic, which is the safe, byte-identical fallback.
    """
    try:
        with open(path, "rb") as f:
            endian = _detect_endian(f)
            r = _Reader(f, endian)
            _tensor_count = r.u64()          # not needed; we stop before tensors
            kv_count = r.u64()
            if kv_count > _MAX_KV_COUNT:
                return None
            kv: dict[str, object] = {}
            for _ in range(kv_count):
                key = r.gguf_string()
                vtype = r.u32()
                val = r.value(vtype)
                # Keep only scalar str/int entries we might consult (arrays → None).
                if isinstance(val, (str, int)) and not isinstance(val, bool):
                    kv[key] = val
    except (OSError, EOFError, ValueError, TypeError, struct.error):
        # TypeError covers a non-path-like arg (None/list/object) reaching open();
        # keep the documented "never raises, returns None on bad input" contract.
        return None

    arch = kv.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None

    def _int(suffix: str, default: int = 0) -> int:
        v = kv.get(f"{arch}.{suffix}")
        return v if isinstance(v, int) else default

    block_count = _int("block_count")
    full_attn_interval = _int("full_attention_interval")
    n_head_kv = _int("attention.head_count_kv")
    key_length = _int("attention.key_length")
    value_length = _int("attention.value_length")

    # Fallback for head dims: key/value_length absent → embedding_length // head_count.
    if key_length <= 0 or value_length <= 0:
        embed = _int("embedding_length")
        n_head = _int("attention.head_count")
        if embed > 0 and n_head > 0:
            derived = embed // n_head
            if key_length <= 0:
                key_length = derived
            if value_length <= 0:
                value_length = derived

    dims = KVDims(
        arch=arch,
        block_count=block_count,
        full_attention_interval=full_attn_interval,
        n_head_kv=n_head_kv,
        key_length=key_length,
        value_length=value_length,
    )
    return dims if dims.is_usable() else None
