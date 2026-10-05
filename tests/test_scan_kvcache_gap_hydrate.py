"""Hydrate must be gap-triggered, not per-scan.

`_scan_kvcache_if_stale` must not call `_hydrate_ram_from_persist` (a full-directory
SSD sweep + potential multi-GB copy2) unconditionally on EVERY call, before
even checking whether the mtime-based scan cache could short-circuit with
zero I/O. It takes an optional `requested_key` parameter. None (default) keeps
the unconditional sweep byte-identical (nothing currently calls it that
way, but the fallback exists for anyone who does). When a key
is given, hydrate is gap-triggered and key-scoped via
`_hydrate_ram_from_persist_for_key`: a RAM hit for that key never touches
SLOT_PERSIST_DIR at all, and a genuine RAM-empty gap hydrates that key's
full triplet (.bin/.json/.bin.ckpt) at most once until RAM is repopulated by
any means (a per-key latch cleared via `_kvcache_scan_cache_invalidate`, the
same chokepoint every real RAM-tier write already calls).

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


# --- fixtures (mirror test_wave_return.py / the stale-mark test file) -------------

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
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture
def persist_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache_persist"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_PERSIST_DIR", str(d))
    return d


def _write_triplet(root, model_tag, port, thread_id, sid, *, clean=True, stale=None):
    """Write .bin + .bin.ckpt + .json into `root`. Returns (bin_fn, ckpt_fn, meta_fn)."""
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    ckpt_fn = bin_fn + ".ckpt"
    (root / bin_fn).write_bytes(b"\x00" * 64)
    (root / ckpt_fn).write_bytes(b"\x00" * 8)
    meta = {
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": 12345,
        "prompt_len": 40000, "n_context_turns": 3, "hash_chain": ["a", "b", "c"],
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": clean,
    }
    if stale is not None:
        meta["stale"] = stale
    (root / meta_fn).write_text(json.dumps(meta))
    return bin_fn, ckpt_fn, meta_fn


# ================================================================================
# Acceptance criterion 1 — a RAM hit must perform NO persist-dir read.
# This test FAILS on the unmodified (pre-fix) code.
# ================================================================================

def test_ram_hit_never_touches_persist_dir(mgr, kv_dir, persist_dir, monkeypatch):
    thread_id = "t-hit-1"
    _write_triplet(kv_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)
    _write_triplet(persist_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)

    listed_dirs = []
    real_listdir = os.listdir

    def spy_listdir(path):
        listed_dirs.append(str(path))
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", spy_listdir)

    th = TurbohaulManager._thread_hash(thread_id)
    key = (_QWEN, th, _PORT)
    mgr._scan_kvcache_if_stale(requested_key=key)

    assert str(persist_dir) not in listed_dirs, (
        "a RAM hit for the requested key touched persist_dir — acceptance "
        "criterion 1 requires zero persist-dir reads on a RAM hit"
    )


# ================================================================================
# Acceptance criterion 2 — a genuine RAM-empty gap still hydrates once.
# This is the arm that makes test 1 non-vacuous (proves hydrate CAN still fire).
# Also proves the FULL TRIPLET arrives, not just the basename.
# ================================================================================

def test_genuine_gap_hydrates_full_triplet_once(mgr, kv_dir, persist_dir):
    thread_id = "t-gap-1"
    bin_fn, ckpt_fn, meta_fn = _write_triplet(
        persist_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)
    # RAM tier is empty for this key — a genuine gap.

    th = TurbohaulManager._thread_hash(thread_id)
    key = (_QWEN, th, _PORT)
    mgr._scan_kvcache_if_stale(requested_key=key)

    assert (kv_dir / bin_fn).exists(), "the .bin did not arrive"
    assert (kv_dir / ckpt_fn).exists(), "the .bin.ckpt did not arrive — engine-state sidecar lost"
    assert (kv_dir / meta_fn).exists(), "the .json did not arrive — bin invisible to the scan loop"


def test_gap_hydrate_makes_the_key_findable_by_the_scan(mgr, kv_dir, persist_dir):
    """Not just "files exist" — the scan's own clean_bins dict must find it,
    proving the hydrated files are structurally correct (right dir, right
    names) and the mtime-cache doesn't skip re-scanning after the copy."""
    thread_id = "t-gap-2"
    _write_triplet(persist_dir, _QWEN, _PORT, thread_id, sid=0, clean=True)

    th = TurbohaulManager._thread_hash(thread_id)
    key = (_QWEN, th, _PORT)
    clean_bins, _ = mgr._scan_kvcache_if_stale(requested_key=key)

    assert key in clean_bins
    assert clean_bins[key] is not None


def test_second_call_same_gap_does_not_relist_persist_dir(mgr, kv_dir, persist_dir, monkeypatch):
    """At most once per RAM-empty gap: if persist genuinely has nothing
    hydratable (e.g. a different, unrelated identity only), repeated lookups
    for the SAME still-empty key must not repeat the persist_dir listdir."""
    thread_id = "t-gap-3"
    th = TurbohaulManager._thread_hash(thread_id)
    key = (_QWEN, th, _PORT)
    # persist_dir has nothing for this key at all — genuinely empty gap.

    listed_persist_count = [0]
    real_listdir = os.listdir

    def spy_listdir(path):
        if str(path) == str(persist_dir):
            listed_persist_count[0] += 1
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", spy_listdir)

    mgr._scan_kvcache_if_stale(requested_key=key)
    mgr._scan_kvcache_if_stale(requested_key=key)
    mgr._scan_kvcache_if_stale(requested_key=key)

    assert listed_persist_count[0] == 1, (
        "persist_dir was listed more than once for a key whose RAM-empty gap "
        "never changed — the per-gap latch should have suppressed the repeats"
    )


# ================================================================================
# The double-suppression hazard against the stale-mark reset:
# stale-mark -> fresh save -> new gap -> hydrate DOES fetch.
# ================================================================================

def test_latch_does_not_permanently_suppress_a_since_cleared_bin(mgr, kv_dir, persist_dir):
    thread_id = "t-hazard-1"
    th = TurbohaulManager._thread_hash(thread_id)
    key = (_QWEN, th, _PORT)

    # 1. Simulate _reset_clean_bin having stale-marked the persist
    #    sidecar (RAM tier is empty — this IS the reset's outcome).
    bin_fn, ckpt_fn, meta_fn = _write_triplet(
        persist_dir, _QWEN, _PORT, thread_id, sid=0, clean=True, stale=True)

    # 2. First gap: hydrate must respect the stale mark — nothing copied.
    mgr._scan_kvcache_if_stale(requested_key=key)
    assert not (kv_dir / bin_fn).exists(), "stale-marked bin was hydrated — should be skipped"
    assert key in mgr._kvcache_hydrate_attempted, "sanity: the attempt should be latched"

    # 3. Simulate the legitimate clear: a fresh real save overwrites the SAME
    #    deterministic filename in persist_dir with a non-stale meta (this is
    #    exactly what _persist_clean_bin_to_ssd_by_hash does on the next real
    #    save — proven in the stale-mark tests, test_stale_mark_is_cleared_by_next_real_
    #    persist_save). That real save ALSO writes to SLOT_SAVE_DIR, which
    #    calls _kvcache_scan_cache_invalidate() — simulate that side effect
    #    explicitly rather than re-testing _save_slot_kv itself here.
    _write_triplet(persist_dir, _QWEN, _PORT, thread_id, sid=0, clean=True, stale=None)
    mgr._kvcache_scan_cache_invalidate()

    # 4. New gap (RAM still empty for this key — e.g. another swap happened
    #    before the new bin got parked back to RAM). Hydrate must now fetch
    #    the since-cleared bin — proving the latch from step 2 did NOT survive.
    mgr._scan_kvcache_if_stale(requested_key=key)
    assert (kv_dir / bin_fn).exists(), (
        "a legitimately-cleared bin was still suppressed — the per-key latch "
        "outlived the RAM-empty gap it should have been scoped to, exactly "
        "the double-suppression hazard against the stale-mark reset"
    )
    assert (kv_dir / ckpt_fn).exists()
    assert (kv_dir / meta_fn).exists()
    cleared_meta = json.loads((kv_dir / meta_fn).read_text())
    assert not cleared_meta.get("stale")
