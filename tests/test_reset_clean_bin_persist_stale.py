"""Kill the RAM-delete/SSD-resurrect loop.

`_reset_clean_bin` deleted a clean bin from SLOT_SAVE_DIR (RAM tier) only.
`_hydrate_ram_from_persist` then copied the byte-identical file straight back
from SLOT_PERSIST_DIR (SSD tier) on the next scan, reproducing the same
verdict -> deleted again -> forever (resulting in pointless repeated SSD
reads, with resets and hydrates in near lock-step).

Fix: `_reset_clean_bin` now stale-marks the SLOT_PERSIST_DIR counterpart of
each sidecar it removes, BEFORE removing it, via the
`_stale_mark_persist_sidecar` helper — same stamp shape
`_mark_main_bin_stale_for_session` already used (`stale`/`stale_reason`/
`stale_at`, atomic tmp+replace), but written to the correct tier
(SLOT_PERSIST_DIR, which `_hydrate_ram_from_persist`'s stale check
actually reads — `_mark_main_bin_stale_for_session` writes
SLOT_SAVE_DIR, a different directory, so its stamp had no effect on hydrate)
and matched by the exact filename `_reset_clean_bin` already matched (covers
every identity it can remove — main, sub_agent, curator, legacy — not just
`_mark_main_bin_stale_for_session`'s main-only `sess:<id>:main` identity).

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this test file> -v
"""
from __future__ import annotations

import json
import os

import pytest

import turbohaul.manager as manager_mod
import turbohaul.subprocess_mgr as subprocess_mgr
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
from turbohaul.kv_policy import kv_meta_fn
from turbohaul.manager import TurbohaulManager

_QWEN = "qwen3.6-27b"
_PORT = 59500


# --- fixtures (mirror test_wave_return.py) --------------------------------------

@pytest.fixture
def mgr(tmp_path):
    storage_root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (storage_root / sub).mkdir(parents=True)
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
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    """Isolated SLOT_SAVE_DIR (RAM tier)."""
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture
def persist_dir(tmp_path, monkeypatch):
    """Isolated SLOT_PERSIST_DIR (SSD tier)."""
    d = tmp_path / "kvcache_persist"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_PERSIST_DIR", str(d))
    return d


def _write_triplet(root, model_tag, port, thread_id, sid, *, clean=True, role=None,
                    stale=None):
    """Write a .bin + .json (+ .bin.ckpt, matching the real triplet shape) into
    `root` (either kv_dir or persist_dir). Mirrors test_wave_return.py's
    _write_bin helper, extended with a `role` stamp (this scope test
    needs non-main identities) and an optional pre-set `stale` flag."""
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (root / bin_fn).write_bytes(b"\x00" * 64)
    (root / (bin_fn + ".ckpt")).write_bytes(b"\x00" * 8)
    meta = {
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": 12345,
        "prompt_len": 40000, "n_context_turns": 3, "hash_chain": ["a", "b", "c"],
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": clean,
    }
    if role is not None:
        meta["role"] = role
    if stale is not None:
        meta["stale"] = stale
    (root / meta_fn).write_text(json.dumps(meta))
    return bin_fn, meta_fn


# ================================================================================
# Case 1 — a reset must not be silently resurrected by hydrate.
# This test fails on the unmodified code.
# ================================================================================

def test_reset_then_hydrate_does_not_resurrect(mgr, kv_dir, persist_dir):
    thread_id = "t-main-1"
    bin_fn, meta_fn = _write_triplet(kv_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)
    # Simulate "already persisted to SSD before the compression event" — the
    # exact precondition of the loop (resets and hydrates in lock-step).
    _write_triplet(persist_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)

    mgr._reset_clean_bin(_PORT, _QWEN, thread_id, bin_fn, requester_is_labeled_main=True)

    # RAM tier: gone, as before.
    assert not (kv_dir / bin_fn).exists()
    assert not (kv_dir / meta_fn).exists()

    mgr._hydrate_ram_from_persist(str(kv_dir), str(persist_dir))

    # THE BUG: pre-fix, this next line resurrects the byte-identical file.
    assert not (kv_dir / bin_fn).exists(), (
        "hydrate resurrected the dropped clean bin from the persist tier — "
        "the exact RAM-delete/SSD-resurrect loop this fix targets"
    )
    assert not (kv_dir / meta_fn).exists()


# ================================================================================
# Case 2 — the marker written is the exact shape hydrate reads.
# Read directly, not assumed: manager.py's stale check is
# `json.load(_hf).get("stale")` against the file at os.path.join(persist_dir,
# _meta_fn) — so assert on that literal file/field, not a proxy for it.
# ================================================================================

def test_stale_stamp_is_the_exact_shape_hydrate_reads(mgr, kv_dir, persist_dir):
    thread_id = "t-main-2"
    bin_fn, meta_fn = _write_triplet(kv_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)
    _write_triplet(persist_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)

    mgr._reset_clean_bin(_PORT, _QWEN, thread_id, bin_fn, requester_is_labeled_main=True)

    persisted_meta = json.loads((persist_dir / meta_fn).read_text())
    assert persisted_meta.get("stale") is True
    assert "stale_reason" in persisted_meta
    assert "stale_at" in persisted_meta


# ================================================================================
# The scope gap: _mark_main_bin_stale_for_session only matches
# thread_id == "sess:<sid>:main" — a sub_agent/curator bin would have silently
# kept resurrecting under a narrower, main-only fix. Prove the actual
# fix (matched by filename, inside _reset_clean_bin itself) covers it.
# ================================================================================

def test_reset_covers_non_main_identity_sub_agent(mgr, kv_dir, persist_dir):
    thread_id = "sess:s1:sub-agent:abcd1234"  # NOT "sess:s1:main"
    bin_fn, meta_fn = _write_triplet(
        kv_dir, _QWEN, _PORT, thread_id, sid=0, clean=True, role="sub-agent")
    _write_triplet(
        persist_dir, _QWEN, _PORT, thread_id, sid=0, clean=True, role="sub-agent")

    mgr._reset_clean_bin(_PORT, _QWEN, thread_id, bin_fn, requester_is_labeled_main=False)

    mgr._hydrate_ram_from_persist(str(kv_dir), str(persist_dir))

    assert not (kv_dir / bin_fn).exists(), (
        "a non-main (sub_agent) bin was resurrected — this is exactly the gap "
        "a main-only fix [_mark_main_bin_stale_for_session] would have "
        "left open"
    )


# ================================================================================
# A persist sidecar marked stale must not stay dead forever — verify the
# next REAL save clears it (otherwise this trades a resurrect loop for a
# permanently dead SSD backup, which is worse).
# ================================================================================

def test_stale_mark_is_cleared_by_next_real_persist_save(mgr, kv_dir, persist_dir):
    thread_id = "t-main-3"
    bin_fn, meta_fn = _write_triplet(kv_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)
    _write_triplet(persist_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)

    mgr._reset_clean_bin(_PORT, _QWEN, thread_id, bin_fn, requester_is_labeled_main=True)

    assert json.loads((persist_dir / meta_fn).read_text()).get("stale") is True

    # The next turn's scale-up probe re-anchors and re-saves a FRESH clean bin
    # under the SAME deterministic filename (per _reset_clean_bin's own
    # docstring) — simulate that fresh RAM write, then the real SSD persist call.
    fresh_bin_fn, fresh_meta_fn = _write_triplet(
        kv_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)
    assert fresh_meta_fn == meta_fn, "sanity: same deterministic filename"

    th = TurbohaulManager._thread_hash(thread_id)
    mgr._persist_clean_bin_to_ssd_by_hash(_QWEN, th, _PORT, trigger="ownership_transfer")

    cleared_meta = json.loads((persist_dir / meta_fn).read_text())
    assert not cleared_meta.get("stale"), (
        "the next real persist save must overwrite the stale-marked sidecar — "
        "otherwise this fix trades a resurrect loop for a permanently dead "
        "SSD backup, which is worse"
    )


# ================================================================================
# Best-effort edge case: a bin that was never persisted to SSD before reset
# (no SLOT_PERSIST_DIR counterpart) must not raise.
# ================================================================================

def test_reset_with_no_persist_counterpart_does_not_raise(mgr, kv_dir, persist_dir):
    thread_id = "t-never-persisted"
    bin_fn, meta_fn = _write_triplet(kv_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)

    # No corresponding write to persist_dir — best-effort no-op path.
    mgr._reset_clean_bin(_PORT, _QWEN, thread_id, bin_fn, requester_is_labeled_main=True)

    assert not (kv_dir / bin_fn).exists()
    assert not (kv_dir / meta_fn).exists()
    assert list(persist_dir.iterdir()) == []
