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
    ``full_attention_interval`` means every Nth layer is a full-attention
    layer that keeps a *growing* KV cache. What the OTHER layers are depends
    on ``sliding_window``:

    - ``sliding_window is None`` (e.g. qwen35): the other layers are SSM —
      fixed recurrent state, ZERO per-token KV growth.
    - ``sliding_window is not None`` (e.g. a hybrid SWA model): the other layers are
      sliding-window attention — CAPPED at ``min(ctx, sliding_window)`` per
      token, small but not zero. Do not conflate the two: an SSM layer is
      free, a SWA layer merely stops growing past the window.
    """

    arch: str
    block_count: int
    full_attention_interval: int
    n_head_kv: int
    key_length: int
    value_length: int
    sliding_window: int | None = None
    nextn_predict_layers: int = 0
    """MTP (multi-token-prediction speculative decode) draft-head layer
    count, e.g. qwen35(moe).nextn_predict_layers (read from the GGUF
    header). 0 = absent/non-MTP model, the default (no MTP head).
    Trailing field with a default so positional KVDims(...) construction
    (as in test_dimension_aware_kv.py) keeps working; it is declared last
    so the earlier positional arguments keep their positions."""

    @property
    def n_attn_layers(self) -> int:
        """Layers that contribute a growing (uncapped) per-token KV cache.

        block_count // full_attention_interval (e.g. 64 // 4 = 16, or
        48 // 4 = 12). When the interval is absent/underivable we conservatively
        count EVERY layer as attention (over-estimates KV → never under-reserves)."""
        if self.full_attention_interval and self.full_attention_interval > 0:
            return max(1, self.block_count // self.full_attention_interval)
        return self.block_count

    @property
    def n_swa_layers(self) -> int:
        """Sliding-window layers: capped, not zero, not growing.

        Derived from ``n_attn_layers`` rather than stored independently, so
        the two can never disagree — always 0 for non-sliding (SSM or plain)
        models, by construction, regardless of ``full_attention_interval``."""
        if self.sliding_window is None:
            return 0
        return max(0, self.block_count - self.n_attn_layers)

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

    def value(self, vtype: int, depth: int = 0, capture_array: bool = False):
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
            # Every element must still be walked so the stream stays aligned,
            # even when the caller doesn't want it (e.g. a multi-hundred-
            # thousand-entry tokenizer vocab array). Only materialize a list
            # when the caller opted in via capture_array=True for THIS key
            # (per-layer attention.head_count) — every other array-valued key
            # is walked and discarded at no extra cost.
            if capture_array:
                return [self.value(elem_type, depth + 1) for _ in range(count)]
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


def _derive_full_attention_interval(head_count, block_count: int) -> int:
    """Infer a per-layer full-attention period from a per-layer
    ``attention.head_count`` array, for models that publish no explicit
    ``full_attention_interval`` key (e.g. a hybrid SWA model).

    Returns 0 ("underivable", same convention as the explicit-key miss) unless
    ALL of the following hold, so a coincidental or unrelated per-layer
    head_count variation can never be mistaken for a real hybrid layout:

    - ``head_count`` is a per-layer array (list), one entry per block.
    - it contains exactly two distinct values (a two-tier split).
    - the minority value's positions form a perfect arithmetic progression
      starting at index 0 — i.e. genuinely periodic, not just proportional.

    No architecture name is consulted; this is pure shape inspection.
    """
    if not isinstance(head_count, list) or block_count <= 0:
        return 0
    if len(head_count) != block_count:
        return 0
    distinct = set(head_count)
    if len(distinct) != 2:
        return 0
    counts = {v: head_count.count(v) for v in distinct}
    minority_value = min(counts, key=lambda v: counts[v])
    minority_count = counts[minority_value]
    if minority_count <= 0 or block_count % minority_count != 0:
        return 0
    interval = block_count // minority_count
    expected_positions = set(range(0, block_count, interval))
    actual_positions = {i for i, v in enumerate(head_count) if v == minority_value}
    if actual_positions != expected_positions:
        return 0
    return interval


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
                # Only ".attention.head_count" is ever captured as an array —
                # it may be per-layer (hybrid SWA models) rather than uniform, and that's
                # the one array this module needs. Arch-agnostic suffix test
                # (no architecture name compared), decided before we know arch
                # since GGUF doesn't guarantee general.architecture comes first.
                val = r.value(vtype, capture_array=key.endswith(".attention.head_count"))
                # Keep scalar str/int entries, plus the one captured array.
                if isinstance(val, (str, int)) and not isinstance(val, bool):
                    kv[key] = val
                elif isinstance(val, list):
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

    def _opt_int(suffix: str) -> int | None:
        v = kv.get(f"{arch}.{suffix}")
        return v if isinstance(v, int) else None

    block_count = _int("block_count")
    n_head_kv = _int("attention.head_count_kv")
    key_length = _int("attention.key_length")
    value_length = _int("attention.value_length")
    sliding_window = _opt_int("attention.sliding_window")
    # MTP draft-head layer count, e.g.
    # qwen35(moe).nextn_predict_layers. 0 (default) on every model that
    # lacks the key — the vast majority.
    nextn_predict_layers = _int("nextn_predict_layers")

    full_attn_interval = _int("full_attention_interval")
    if full_attn_interval <= 0:
        # Explicit key absent (or not an int) — try to derive the period from
        # a per-layer head_count array. Returns 0 ("underivable") unless the
        # array is a genuine, strictly periodic two-tier split; see
        # _derive_full_attention_interval's docstring for the exact guard.
        full_attn_interval = _derive_full_attention_interval(
            kv.get(f"{arch}.attention.head_count"), block_count
        )

    # Fallback for head dims: key/value_length absent → embedding_length // head_count.
    if key_length <= 0 or value_length <= 0:
        embed = _int("embedding_length")
        n_head = _int("attention.head_count")  # scalar-only; list → default 0
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
        sliding_window=sliding_window,
        nextn_predict_layers=nextn_predict_layers,
    )
    return dims if dims.is_usable() else None


class ModelCard(NamedTuple):
    """Best-effort GGUF general/model-card metadata for FE display + sort
    (frontend model list). Independent of ``KVDims``
    — this reads a different field set (general.* + {arch}.expert_count +
    vision markers) for display/sort, not KV-fit sizing.

    All fields are ``None``/unknown on missing-or-malformed input; nothing
    here ever raises to the caller (same contract as ``read_kv_dims``).
    """

    architecture: str | None
    parameter_count: int | None
    size_label: str | None
    expert_count: int | None
    is_moe: bool | None
    has_vision_kv: bool


def read_model_card(path) -> "ModelCard | None":
    """Parse the GGUF KV header at ``path`` and return a best-effort ``ModelCard``.

    Returns None (never raises) when the file is missing/malformed. Reuses
    the same ``_Reader``/``_detect_endian`` primitives as ``read_kv_dims``
    but reads a different key set and does not modify or call into
    ``read_kv_dims`` — the two are independent, parallel readers over the
    same KV header.
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
            keys_seen: set[str] = set()
            for _ in range(kv_count):
                key = r.gguf_string()
                keys_seen.add(key)
                vtype = r.u32()
                val = r.value(vtype)
                if isinstance(val, (str, int)) and not isinstance(val, bool):
                    kv[key] = val
    except (OSError, EOFError, ValueError, TypeError, struct.error):
        return None

    arch = kv.get("general.architecture")
    arch = arch if isinstance(arch, str) and arch else None

    def _opt_int(key: str) -> int | None:
        v = kv.get(key)
        return v if isinstance(v, int) else None

    def _opt_str(key: str) -> str | None:
        v = kv.get(key)
        return v if isinstance(v, str) and v else None

    parameter_count = _opt_int("general.parameter_count")
    size_label = _opt_str("general.size_label")
    expert_count = _opt_int(f"{arch}.expert_count") if arch else None
    is_moe = (expert_count > 0) if expert_count is not None else None
    has_vision_kv = any(".vision." in k or k.startswith("clip.") for k in keys_seen)

    return ModelCard(
        architecture=arch,
        parameter_count=parameter_count,
        size_label=size_label,
        expert_count=expert_count,
        is_moe=is_moe,
        has_vision_kv=has_vision_kv,
    )
