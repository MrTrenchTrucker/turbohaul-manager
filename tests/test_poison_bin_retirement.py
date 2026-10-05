"""Tripwire: POISON-BIN RETIREMENT.

A completed generation whose engine heartbeat reports reused-to-zero
(``n_prompt_cache == 0``) on a CLEAN-STAMPED restored bin is the reuse=0
signature of a poisoned bin: the bin restored, the engine found nothing to
reuse — a post-generation dump stamped clean. The fix retires that bin:

  * stale-mark BOTH tiers (SLOT_SAVE_DIR + SLOT_PERSIST_DIR sidecars —
    never delete; the hydrate path's stale check reads the persist tier),
  * invalidate the in-memory kvcache scan cache (so a stale ring entry
    cannot re-offer the bin),
  * remove the bin's entry from the durable-ring mirror,
  * poison the bin's context chain-fp so the save gate will not
    re-stamp the same context,
  * and attribute all of it in the RETAINED KV_REUSE_OUTCOME line
    (``bin_retired=`` ``poison_chain_fp=`` appended only on fire).

Every guard is pinned here: non-clean bins are never retired (the
normalization re-probe owns the dirty-bin lane), a non-zero reuse figure
leaves a clean bin untouched, missing heartbeat / missing port / missing
sidecars are best-effort no-ops, and the outcome line's new fields appear
ONLY when a retirement fired.

These tripwires FAIL on the pre-fix base (no retirement path exists — the
line carries no bin_retired= fields and no sidecar is ever stamped) and
PASS on the fix.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this file> -v
"""
from __future__ import annotations

import json
import types

import logging

import pytest

import turbohaul.manager as manager_mod  # noqa: F401  (module under test)
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
from turbohaul.live_monitor import compute_generation_id
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot

_QWEN = "qwen3.6-27b"
_PORT = 59500
_PID = 4242

# NOTE: _log_kv_reuse_outcome is a sync method (the async lives in its
# callers' completion paths), so these tests are plain sync functions — no
# asyncio module mark.


# --- fixtures (mirror the reset-clean-bin persist-stale test) ---

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
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False), pull=PullConfig(),
    )
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


# --- helpers -----------------------------------------------------------------

def _bin_fn(model_tag=_QWEN, sid=0, thread_id="t1", port=_PORT):
    fn = kv_meta_fn(model_tag, sid, TurbohaulManager._thread_hash(thread_id), port)
    return fn[:-5] + ".bin"


def _write_pair(root, model_tag=_QWEN, sid=0, thread_id="t1", port=_PORT, *,
                clean=True, chain=None):
    """Write a .bin + .json sidecar pair into `root` (one tier)."""
    fn = kv_meta_fn(model_tag, sid, TurbohaulManager._thread_hash(thread_id), port)
    (root / (fn[:-5] + ".bin")).write_bytes(b"dummy kv cache data")
    meta = {
        "clean_prefix": clean,
        "prompt_tokens": 81902,
        "bin_provenance": "clean_prefill" if clean else "slot_n_prompt_tokens",
    }
    if chain is not None:
        meta["hash_chain"] = chain
    (root / fn).write_text(json.dumps(meta))
    return fn[:-5] + ".bin"


def _read_meta(root, model_tag=_QWEN, sid=0, thread_id="t1", port=_PORT):
    fn = kv_meta_fn(model_tag, sid, TurbohaulManager._thread_hash(thread_id), port)
    return json.loads((root / fn).read_text())


def _handle(pid=_PID, port=_PORT):
    return types.SimpleNamespace(pid=pid, port=port)


def _slot(session_id="sess-1"):
    return Slot.new(model_tag=_QWEN, thread_id="t1", context=None,
                    client_meta={"session_id": session_id})


def _prime_heartbeat(mgr, slot_id, n_prompt_cache, *, pid=_PID, spawn_seq=0,
                     model_tag=_QWEN):
    """Make the poller's heartbeat look like it already observed THIS exact
    request's generation (mirrors test_kv_reuse_outcome's idiom)."""
    gen_id = compute_generation_id(pid, spawn_seq, slot_id)
    heartbeat = {"generation_id": gen_id, "n_prompt_cache": n_prompt_cache}
    mgr.live_generations[model_tag] = heartbeat
    mgr.live_generation = heartbeat
    return heartbeat


def _outcome_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("KV_REUSE_OUTCOME")]


def _fire(mgr, slot, handle, n_prompt_cache, caplog):
    """Prime the matching generation sample and fire the outcome path."""
    _prime_heartbeat(mgr, slot.slot_id, n_prompt_cache)
    with caplog.at_level(logging.INFO):
        mgr._log_kv_reuse_outcome(slot, handle, wait_state="completed")
    return _outcome_lines(caplog)


# ================================================================================
# 1 — the main event: reuse-zero on a clean-stamped restored bin retires it
# ================================================================================
def test_reuse_zero_clean_bin_is_retired_on_both_tiers(mgr, kv_dir, persist_dir, caplog):
    """Both tiers are stale-marked (stale_reason=reuse_zero), the scan cache
    is invalidated (best-effort, no raise), and the retained outcome line
    carries bin_retired= + poison_chain_fp= for the retired bin."""
    chain = ["h1", "h2", "h3"]
    bfn = _write_pair(kv_dir, clean=True, chain=chain)
    _write_pair(persist_dir, clean=True, chain=chain)
    mgr._last_restored_bin[_PORT] = bfn

    lines = _fire(mgr, _slot(), _handle(), 0, caplog)

    assert len(lines) == 1, lines
    assert f"bin_retired={bfn}" in lines[0], lines[0]
    from turbohaul.manager import _chain_fp
    assert f"poison_chain_fp={_chain_fp(chain)}" in lines[0], lines[0]

    ram = _read_meta(kv_dir)
    assert ram.get("stale") is True
    assert ram.get("stale_reason") == "reuse_zero"
    assert "stale_at" in ram
    ssd = _read_meta(persist_dir)
    assert ssd.get("stale") is True
    assert ssd.get("stale_reason") == "reuse_zero"
    assert "stale_at" in ssd


# ================================================================================
# 2 — the ring mirror loses the bin; the poison set gains the context fp
# ================================================================================
def test_reuse_zero_ring_entry_removed_and_poison_set_populated(mgr, kv_dir, persist_dir, caplog):
    """The in-memory durable-ring index (a mirror of on-disk truth) must lose
    the retired bin's entry — and only that entry — while the context's chain
    fp lands in the per-port poison set the save gate consults."""
    chain = ["h1", "h2"]
    bfn = _write_pair(kv_dir, clean=True, chain=chain)
    _write_pair(persist_dir, clean=True, chain=chain)
    mgr._last_restored_bin[_PORT] = bfn
    ring_key = ("main", "sess-1")
    mgr._durable_ring_index[ring_key] = [
        {"bin_fn": bfn, "chain": chain},
        {"bin_fn": "keep.bin", "chain": ["h9"]},
    ]

    _fire(mgr, _slot(), _handle(), 0, caplog)

    from turbohaul.manager import _chain_fp
    assert mgr._durable_ring_index[ring_key] == [{"bin_fn": "keep.bin", "chain": ["h9"]}]
    assert mgr._poisoned_chain_fps[_PORT] == {_chain_fp(chain)}


# ================================================================================
# 3 — non-clean restored bin: NO retirement (the dirty lane is out of scope)
# ================================================================================
def test_reuse_zero_on_non_clean_bin_does_not_retire(mgr, kv_dir, persist_dir, caplog):
    """A reused-to-zero outcome on a NON-clean-stamped bin is the
    normalization re-probe's territory, not the retirement path's: no stale mark on either
    tier, no poison entry, and the outcome line stays unextended."""
    bfn = _write_pair(kv_dir, clean=False)
    _write_pair(persist_dir, clean=False)
    mgr._last_restored_bin[_PORT] = bfn

    lines = _fire(mgr, _slot(), _handle(), 0, caplog)

    assert len(lines) == 1, lines
    assert "bin_retired=" not in lines[0], lines[0]
    assert "poison_chain_fp=" not in lines[0], lines[0]
    assert _read_meta(kv_dir).get("stale") is None
    assert _read_meta(persist_dir).get("stale") is None
    assert mgr._poisoned_chain_fps.get(_PORT) is None


# ================================================================================
# 4 — non-zero reuse on a clean bin: everything stays put
# ================================================================================
def test_reuse_nonzero_clean_bin_is_untouched(mgr, kv_dir, persist_dir, caplog):
    """engine_reused_tokens > 0 on a clean-stamped bin is the healthy reuse
    path: no retirement, no poison, no bin_retired= fields on the line."""
    bfn = _write_pair(kv_dir, clean=True, chain=["h1"])
    _write_pair(persist_dir, clean=True, chain=["h1"])
    mgr._last_restored_bin[_PORT] = bfn

    lines = _fire(mgr, _slot(), _handle(), 12345, caplog)

    assert len(lines) == 1, lines
    assert "engine_reused_tokens=12345" in lines[0], lines[0]
    assert "bin_retired=" not in lines[0], lines[0]
    assert _read_meta(kv_dir).get("stale") is None
    assert _read_meta(persist_dir).get("stale") is None
    assert mgr._poisoned_chain_fps.get(_PORT) is None


# ================================================================================
# 5 — no matching heartbeat sample: honest unavailable, no retirement
# ================================================================================
def test_no_matching_sample_is_unavailable_and_inert(mgr, kv_dir, persist_dir, caplog):
    """Generation matching is the trust gate: with no sample proven to belong
    to THIS request there is no reuse figure at all — and certainly no
    retirement side effects."""
    bfn = _write_pair(kv_dir, clean=True, chain=["h1"])
    _write_pair(persist_dir, clean=True, chain=["h1"])
    mgr._last_restored_bin[_PORT] = bfn
    # Heartbeat present but for a DIFFERENT slot -> not trusted for this
    # request. (Primed DIRECTLY — not via _fire — because _fire would
    # re-prime a matching sample for this slot by construction.)
    _prime_heartbeat(mgr, "slot-someone-elses-request", 0)
    slot = _slot()
    with caplog.at_level(logging.INFO):
        mgr._log_kv_reuse_outcome(slot, _handle(), wait_state="completed")
    lines = _outcome_lines(caplog)

    assert len(lines) == 1, lines
    assert "engine_reused_tokens=unavailable" in lines[0], lines[0]
    assert "bin_retired=" not in lines[0], lines[0]
    assert _read_meta(kv_dir).get("stale") is None


# ================================================================================
# 6 — handle without a port: best-effort no-op, never raises
# ================================================================================
def test_handle_without_port_is_best_effort_noop(mgr, kv_dir, persist_dir, caplog):
    """The retirement must be fail-soft: a handle without a usable port yields
    no side effects and no exception (the outcome line still reports the
    honest figure)."""
    bfn = _write_pair(kv_dir, clean=True, chain=["h1"])
    _write_pair(persist_dir, clean=True, chain=["h1"])
    mgr._last_restored_bin[_PORT] = bfn

    lines = _fire(mgr, _slot(), types.SimpleNamespace(pid=_PID), 0, caplog)

    assert len(lines) == 1, lines
    assert "bin_retired=" not in lines[0], lines[0]
    assert _read_meta(kv_dir).get("stale") is None
    assert _read_meta(persist_dir).get("stale") is None


# ================================================================================
# 7 — SSD sidecar missing: RAM tier is still retired, no crash
# ================================================================================
def test_missing_persist_sidecar_still_retires_ram(mgr, kv_dir, persist_dir, caplog):
    """A bin whose SSD counterpart is already gone (e.g. pruned) is still
    stale-marked on the RAM tier and still poisons the context — the
    two-tier retirement is independent per tier, best-effort per sidecar."""
    chain = ["h1", "h2"]
    bfn = _write_pair(kv_dir, clean=True, chain=chain)
    mgr._last_restored_bin[_PORT] = bfn
    # NOTE: nothing written into persist_dir.

    lines = _fire(mgr, _slot(), _handle(), 0, caplog)

    assert len(lines) == 1, lines
    assert f"bin_retired={bfn}" in lines[0], lines[0]
    ram = _read_meta(kv_dir)
    assert ram.get("stale") is True
    assert ram.get("stale_reason") == "reuse_zero"
    from turbohaul.manager import _chain_fp
    assert mgr._poisoned_chain_fps[_PORT] == {_chain_fp(chain)}
