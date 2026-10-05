"""Minimal, stdlib-only GGUF *tensor-info* reader -- exact per-tensor byte
sizes for the MoE expert-placement estimate.

Separate module from ``_gguf_meta.py`` on purpose: that module deliberately
stops at the end of the KV block and never reads tensor-info, because the
engine ships custom quant type-ids (e.g. an ultra-low-bit quantised type) that standard
readers choke on decoding. That caution is about the tensor-info ``type``
field specifically -- this module reads tensor-info too, but only ever
consumes each tensor's ``offset`` (a plain, quant-agnostic u64). It never
decodes ``type`` and never reads past the tensor-info block into tensor data
itself, so it inherits none of the risk the caution above is about.

Mechanism: consecutive tensors' data offsets give exact byte sizes
(``size(i) = offset(i+1) - offset(i)``, last tensor closes against
``file_size - data_start``). Sizes are offset differences, so each size
includes any padding up to the next tensor; the closure check below rejects
any layout whose sizes do not sum to ``file_size - data_start``.

Everything here is READ-ONLY and best-effort: any malformed / unexpected
input, or a failed closure assertion, makes ``read_expert_layout`` return
``None`` and the caller falls back to the existing behaviour (byte-identical).
It never raises to the caller.
"""
from __future__ import annotations

import os
import re
import struct
from typing import NamedTuple

from turbohaul._gguf_meta import _Reader, _detect_endian

# Defensive caps, same spirit as _gguf_meta.py: a genuine GGUF header is
# small; refuse absurd counts so a corrupt/hostile file can't drive a huge
# allocation or an unbounded scan.
_MAX_KV_COUNT = 1 << 20
_MAX_TENSOR_COUNT = 1 << 20
_MAX_TENSOR_DIMS = 1 << 8
_DEFAULT_ALIGNMENT = 32

_EXPERT_MARKER = "_exps"
_BLOCK_INDEX_RE = re.compile(r"blk\.(\d+)\.")


class ExpertLayout(NamedTuple):
    """Exact byte layout of a GGUF's tensors, split into per-block expert
    weights and everything else. Closure-verified: the caller can trust
    these numbers sum to the file's real data-region size."""

    expert_bytes_by_block: "dict[int, int]"
    non_expert_bytes: int
    fsize: int
    data_start: int

    @property
    def total_expert_bytes(self) -> int:
        return sum(self.expert_bytes_by_block.values())


def read_expert_layout(path) -> "ExpertLayout | None":
    """Parse the GGUF tensor-info block at ``path`` and return an
    ``ExpertLayout``, or ``None`` on any failure (missing file, malformed
    header, non-positive derived size, or a failed closure assertion).

    Reads the KV block (for ``general.alignment`` only) plus the full
    tensor-info block -- never tensor data. Never decodes a tensor's
    ``type`` field; only ``offset`` is consulted, which is quant-agnostic.
    """
    try:
        fsize = os.path.getsize(path)
        with open(path, "rb") as f:
            endian = _detect_endian(f)
            r = _Reader(f, endian)
            n_tensors = r.u64()
            kv_count = r.u64()
            if kv_count > _MAX_KV_COUNT or n_tensors > _MAX_TENSOR_COUNT:
                return None
            alignment = _DEFAULT_ALIGNMENT
            for _ in range(kv_count):
                key = r.gguf_string()
                vtype = r.u32()
                val = r.value(vtype)
                if key == "general.alignment" and isinstance(val, int) and not isinstance(val, bool):
                    if val > 0:
                        alignment = val

            tinfo: "list[tuple[str, int]]" = []
            for _ in range(n_tensors):
                name = r.gguf_string()
                n_dims = r.u32()
                if n_dims > _MAX_TENSOR_DIMS:
                    return None
                for _ in range(n_dims):
                    r.u64()  # dims -- unused, must still be consumed to stay aligned
                _ttype = r.u32()  # NEVER decoded/interpreted -- offset only
                offset = r.u64()
                tinfo.append((name, offset))
            ti_end = f.tell()
    except (OSError, EOFError, ValueError, TypeError, struct.error):
        return None

    if not tinfo:
        return None

    data_start = ti_end + ((-ti_end) % alignment)
    order = sorted(range(len(tinfo)), key=lambda i: tinfo[i][1])
    sizes = [0] * len(tinfo)
    for j, idx in enumerate(order):
        off = tinfo[idx][1]
        nxt = tinfo[order[j + 1]][1] if j + 1 < len(order) else (fsize - data_start)
        sizes[idx] = nxt - off

    if any(s <= 0 for s in sizes):
        return None
    # ★ Closure assertion -- what makes this trustworthy rather than plausible.
    if sum(sizes) != fsize - data_start:
        return None

    expert_bytes_by_block: "dict[int, int]" = {}
    non_expert_bytes = 0
    for (name, _off), sz in zip(tinfo, sizes):
        if _EXPERT_MARKER in name:
            m = _BLOCK_INDEX_RE.search(name)
            if not m:
                # An expert-marked tensor with no recognizable block index.
                # Refuse the whole derivation rather than silently miscount
                # or misattribute its bytes -- unknown shape must fall back,
                # never guess.
                return None
            blk = int(m.group(1))
            expert_bytes_by_block[blk] = expert_bytes_by_block.get(blk, 0) + sz
        else:
            non_expert_bytes += sz

    return ExpertLayout(
        expert_bytes_by_block=expert_bytes_by_block,
        non_expert_bytes=non_expert_bytes,
        fsize=fsize,
        data_start=data_start,
    )


def split_expert_bytes(layout: "ExpertLayout", n_cpu_moe: int) -> "tuple[int, int]":
    """Return ``(offloaded_bytes, resident_bytes)`` for the expert tensors
    only (``non_expert_bytes`` is not included in either -- the caller adds
    it to the VRAM/resident side itself).

    ``--n-cpu-moe N`` keeps the first N *block indices'* expert weights on
    CPU (a layer count, not an expert count and not a byte count) -- so the
    split compares each block's own index against ``n_cpu_moe``, not its
    position within the expert-bearing set. A model whose block 0 carries no
    experts (a dense first block) is handled correctly by construction: block 0
    simply isn't a key in ``expert_bytes_by_block``.
    """
    offloaded = sum(
        b for idx, b in layout.expert_bytes_by_block.items() if idx < n_cpu_moe
    )
    resident = layout.total_expert_bytes - offloaded
    return offloaded, resident
