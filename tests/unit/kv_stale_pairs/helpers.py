"""Shared helpers for the kv_stale_pairs unit tests: real files in a temp directory."""

import sys
from pathlib import Path

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.kv_stale_pairs import drop_stale_pairs  # noqa: E402

CARD = "see src/turbohaul/kv_stale_pairs/AGENTS.md"
TAG = "qwen3.6-27b"
TH = "a1b2c3d4e5f6"
OTHER_TH = "ffffffffffff"
NEW_PORT = 11501
LANDED = 1000


def bin_name(port, thread_hash=TH, slot=0, tag=TAG):
    return f"{tag}.p{port}.{thread_hash}.slot{slot}.bin"


def write_pair(save_dir, port, *, thread_hash=TH, slot=0, tag=TAG, bin_bytes=10, ckpt_bytes=5,
               ckpt=True, meta=True):
    """Write a bin (plus its .ckpt and .json sidecars) and return the bin file name."""
    name = bin_name(port, thread_hash, slot, tag)
    (Path(save_dir) / name).write_bytes(b"b" * bin_bytes)
    if ckpt:
        (Path(save_dir) / (name + ".ckpt")).write_bytes(b"c" * ckpt_bytes)
    if meta:
        (Path(save_dir) / (name[:-4] + ".json")).write_text("{}")
    return name


def listing(save_dir):
    return sorted(p.name for p in Path(save_dir).iterdir())


def call(save_dir, *, tag=TAG, thread_hash=TH, new_port=NEW_PORT, live_turns=(), landed=LANDED,
         n_written=LANDED, **kw):
    """drop_stale_pairs with a clean landed check and no live turns unless a test says otherwise."""
    return drop_stale_pairs(str(save_dir), tag, thread_hash, new_port, live_turns, landed, n_written, **kw)
