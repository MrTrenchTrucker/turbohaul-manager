"""Measurement-only instrument. The force-restore gate fires
because warm_chain is not a prefix of inc_chain, but the chain hash is
CUMULATIVE (H_i = SHA256(H_{i-1} || role_i || content_i)): a divergence at
turn 1 and a divergence at turn 80 are observationally IDENTICAL from
resolved_from alone -- both just say "not a prefix". Whether the force was a
correct call (the warm state genuinely diverged early) or an accidental one
(a late divergence where the short clean bin is a worse ancestor than the
long warm state looks) cannot be told apart without knowing WHERE the
divergence is. This adds warm_vs_inc_divergence_index / clean_vs_inc_
divergence_index (reusing the already-shipped, already-None-guarded
_chain_divergence_index) plus warm_turns/clean_turns/inc_turns to the SAME
WARM_CLASSIFY line the warm path already emits.

Zero decision impact: every assertion here about force/would_force/
forced_clean_restore/the actual restore POST matches this repo's own
pre-existing test_classifier.py assertions for the identical scenarios
-- this file adds fields to what is already logged, touches no predicate.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this test file> -v
"""
from __future__ import annotations

import json
import logging

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
from turbohaul.kv_policy import _prefix_hash_chain, kv_meta_fn
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot

_QWEN = "qwen3.6-27b"
_PORT = 60300


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
            default_port_base=_PORT,
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


@pytest.fixture(autouse=True)
def _warm_force_on(monkeypatch):
    monkeypatch.setenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", "1")


class _Resp:
    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _Client:
    def __init__(self, posts):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        return _Resp()


@pytest.fixture
def posts(monkeypatch):
    recorded = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _Client(recorded))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return recorded


def _write_clean_bin(kv_dir, model_tag, port, thread_id, sid, chain, prompt_len=40000):
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": 12345,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": True,
    }))
    return bin_fn


def _turn(i, marker):
    role = "user" if i % 2 == 0 else "assistant"
    return {"role": role, "content": f"turn-{i}-{marker}"}


def _turns(n, marker):
    return [_turn(i, marker) for i in range(n)]


def _warm_classify_lines(caplog):
    return [r.message for r in caplog.records if r.message.startswith("WARM_CLASSIFY")]


@pytest.mark.asyncio
class TestWarmVsCleanDivergence:
    async def test_discriminator_early_vs_late_warm_divergence_are_distinguishable(
        self, mgr, kv_dir, posts, caplog
    ):
        """The exact ambiguity this instrument exists to resolve: two
        scenarios that both force-restore (same event_type/resolved_from
        shape) but where warm_chain diverges from inc_chain at very
        different points. resolved_from alone cannot tell them apart --
        warm_vs_inc_divergence_index must."""
        # Scenario A: warm diverges almost immediately (index 1). clean_a is
        # a GENUINE prefix slice of the same underlying messages inc_a is
        # built from (same content -> same cumulative hashes), not a
        # separately-generated chain -- otherwise clean_valid is False and
        # the wrong branch (warm-clean-not-prefix) fires instead.
        msgs_a = _turns(9, "SHAREDA")
        inc_a = _prefix_hash_chain(msgs_a)
        clean_a = _prefix_hash_chain(msgs_a[:2])
        assert clean_a == inc_a[:2]  # sanity: genuine shared prefix
        _write_clean_bin(kv_dir, _QWEN, _PORT, "t-early", 0, clean_a)
        warm_a = [inc_a[0], "DIVERGED-AT-1"] + inc_a[2:9]
        slot_a = Slot.new(model_tag=_QWEN, thread_id="t-early", admission_hash_chain=inc_a)
        with caplog.at_level(logging.INFO):
            d_a = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot_a, warm_a)
        assert d_a["forced_clean_restore"] is True

        # Scenario B: warm diverges near the very end (index 7 of 9).
        msgs_b = _turns(9, "SHAREDB")
        inc_b = _prefix_hash_chain(msgs_b)
        clean_b = _prefix_hash_chain(msgs_b[:2])
        assert clean_b == inc_b[:2]
        _write_clean_bin(kv_dir, _QWEN, _PORT, "t-late", 0, clean_b)
        # The scan cache keys on directory mtime; a second bin written
        # within the same mtime tick as scenario A's own scan can be
        # invisible to _find_clean_bin without an explicit invalidate.
        mgr._kvcache_scan_cache_invalidate()
        warm_b = inc_b[:7] + ["DIVERGED-AT-7", inc_b[8]]
        slot_b = Slot.new(model_tag=_QWEN, thread_id="t-late", admission_hash_chain=inc_b)
        with caplog.at_level(logging.INFO):
            d_b = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot_b, warm_b)
        assert d_b["forced_clean_restore"] is True

        # Both fired via the identical resolved_from shape...
        assert d_a["resolved_from"] == d_b["resolved_from"] == "warm-force-clean-restore"
        # ...but the new field tells them apart, which resolved_from cannot.
        assert d_a["warm_vs_inc_divergence_index"] == 1
        assert d_b["warm_vs_inc_divergence_index"] == 7
        assert d_a["warm_vs_inc_divergence_index"] != d_b["warm_vs_inc_divergence_index"]

    async def test_discriminator_firing_scenario_clean_has_no_divergence_warm_does(
        self, mgr, kv_dir, posts, caplog
    ):
        """At the moment force actually fires, clean_chain is BY
        CONSTRUCTION a valid prefix of inc_chain (clean_valid is a
        precondition of would_force) -- so clean_vs_inc_divergence_index
        must be None every time force=True, while warm_vs_inc_divergence_
        index is a real number. This is the asymmetry that settles which
        candidate actually shares more with the incoming request."""
        msgs = _turns(2, "SHARED") + [{"role": "user", "content": "extra-1"},
                                       {"role": "user", "content": "extra-2"}]
        inc_chain = _prefix_hash_chain(msgs)
        clean_chain = _prefix_hash_chain(msgs[:2])
        assert clean_chain == inc_chain[:2]  # sanity: genuine shared prefix
        bin_fn = _write_clean_bin(kv_dir, _QWEN, _PORT, "t-asymmetry", 0, clean_chain)
        warm_chain = _prefix_hash_chain(
            msgs[:2] + [{"role": "assistant", "content": "<think>t</think>a"}])
        slot = Slot.new(model_tag=_QWEN, thread_id="t-asymmetry", admission_hash_chain=inc_chain)
        with caplog.at_level(logging.INFO):
            d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, warm_chain)
        assert d["forced_clean_restore"] is True
        assert d["clean_bin_id"] == bin_fn
        assert d["clean_vs_inc_divergence_index"] is None
        assert d["warm_vs_inc_divergence_index"] is not None

        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "clean_vs_inc_divergence_index=None" in line, line
        assert f"warm_vs_inc_divergence_index={d['warm_vs_inc_divergence_index']}" in line, line
        assert "shared_prefix_tokens=unavailable" in line, line
        assert f"warm_turns={len(warm_chain)}" in line, line
        assert f"clean_turns={len(clean_chain)}" in line, line
        assert f"inc_turns={len(inc_chain)}" in line, line

    async def test_discriminator_uncomputable_index_is_none_never_zero(
        self, mgr, kv_dir, posts, caplog
    ):
        """Mirror of the divergence-at-index-0 case (which pins the TRUE
        zero): here the index genuinely cannot be computed for EITHER
        chain (no clean bin found, warm_chain unknown/empty) -- the field
        must be the literal text 'None', never '0'. 0 is simultaneously a
        real 'diverges at the head' measurement AND the natural default for
        an int field that silently failed to compute -- conflating them
        would rebuild the exact common_prefix_turns:0 ambiguity this
        instrument exists to avoid, this time steering a live design
        decision instead of just mislabeling a log line."""
        inc_chain = _prefix_hash_chain(_turns(3, "UNCOMPUTABLE"))
        slot = Slot.new(model_tag=_QWEN, thread_id="t-uncomputable",
                         admission_hash_chain=inc_chain)
        with caplog.at_level(logging.INFO):
            d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, [])
        assert d["warm_vs_inc_divergence_index"] is None
        assert d["warm_vs_inc_divergence_index"] != 0
        assert d["clean_vs_inc_divergence_index"] is None
        assert d["clean_vs_inc_divergence_index"] != 0

        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "warm_vs_inc_divergence_index=None" in line, line
        assert "warm_vs_inc_divergence_index=0" not in line, line
        assert "clean_vs_inc_divergence_index=None" in line, line
        assert "clean_vs_inc_divergence_index=0" not in line, line
        # and the true-zero case (this same file, above) still renders as
        # the literal digit -- 0 keeps meaning exactly one thing.

    async def test_discriminator_clean_not_prefix_reports_its_own_divergence(
        self, mgr, kv_dir, posts, caplog
    ):
        """The 'warm-clean-not-prefix' branch (compression) is the ONE case
        where clean_chain itself diverges from inc_chain -- confirm
        clean_vs_inc_divergence_index is a real number there, not None."""
        clean_chain = _prefix_hash_chain(
            [{"role": "user", "content": "ORIGINAL-FIRST-TURN"}] + _turns(2, "CLEAN"))
        _write_clean_bin(kv_dir, _QWEN, _PORT, "t-notprefix", 0, clean_chain)
        inc_chain = _prefix_hash_chain(
            [{"role": "user", "content": "REWRITTEN-FIRST-TURN"}] + _turns(2, "CLEAN"))
        slot = Slot.new(model_tag=_QWEN, thread_id="t-notprefix", admission_hash_chain=inc_chain)
        with caplog.at_level(logging.INFO):
            d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, [])
        assert d["resolved_from"] == "warm-clean-not-prefix"
        assert d["clean_vs_inc_divergence_index"] == 0

    async def test_discriminator_no_clean_bin_site_reports_none_not_zero(
        self, mgr, kv_dir, posts, caplog
    ):
        """Site 1 (no disk clean bin at all): clean_vs_inc_divergence_index/
        clean_turns must be None -- there is no candidate to measure, which
        is a different fact from a candidate that measured to 0 turns."""
        inc_chain = _prefix_hash_chain(_turns(3, "NOBIN"))
        slot = Slot.new(model_tag=_QWEN, thread_id="t-nobin", admission_hash_chain=inc_chain)
        with caplog.at_level(logging.INFO):
            d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, [])
        assert d["resolved_from"] == "warm-no-clean-bin"
        assert d["clean_vs_inc_divergence_index"] is None
        assert d["clean_turns"] is None
        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        assert "clean_vs_inc_divergence_index=None" in lines[0], lines[0]
        assert "clean_turns=None" in lines[0], lines[0]

    async def test_good_shape_survives_force_predicates_unchanged(
        self, mgr, kv_dir, posts, caplog
    ):
        """Control: the exact scenario from this repo's own pre-existing
        test_classifier.py::test_force_on_think_strip_user_turn --
        forced_clean_restore/the actual POST/common_prefix_turns must be
        byte-identical to before this measurement was added; only new
        fields ride alongside."""
        _SYS = {"role": "system", "content": "system prompt long enough to matter"}
        _U1 = {"role": "user", "content": "first user turn"}
        _U2 = {"role": "user", "content": "second user turn"}
        clean_chain = _prefix_hash_chain([_SYS, _U1])
        bin_fn = _write_clean_bin(kv_dir, _QWEN, _PORT, "t", 0, clean_chain)
        inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
        warm = _prefix_hash_chain(
            [_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
        slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc)
        d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, warm)
        assert d["forced_clean_restore"] is True
        assert d["action"] == "restore" and d["event_type"] == "user-message"
        assert d["resolved_from"] == "warm-force-clean-restore"
        assert d["clean_bin_id"] == bin_fn and d["common_prefix_turns"] == 2
        assert len(posts) == 1
        url, body = posts[0]
        assert "action=restore" in url and body == {"filename": bin_fn}
        assert mgr._kv_classifier_forced == 1
        # And the new fields are simply present alongside, not instead of.
        assert "warm_vs_inc_divergence_index" in d
        assert "clean_vs_inc_divergence_index" in d
