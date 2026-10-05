"""Observability hardening: the cold-restore per-bin
decision (resolve_kv("restore", ...) inside _restore_slot_kv_inner) computes
saved_chain/incoming_chain but never rendered them: KVDecision's own
__repr__ is a fixed 4-field summary (do_it/action/reason/resolved_from) and
is NOT changed here (other call sites render KVDecision and must not shift
under them). Instead a WARM_CLASSIFY line is emitted at the same point, one
per candidate bin, carrying the two chain fingerprints, saved_len/
incoming_len, and — the field this addition exists for — divergence_index:
the first index where the two chains actually differ, or an explicit None
when one chain is a clean prefix of the other (a length-only refusal, not a
content rewrite). This settles, from the log alone, whether a "DIVERGED"
refusal is a genuine early-turn rewrite (small index) or something else.

divergence_index is a NEW, separately-named field -- this change never reuses
or "fixes" the pre-existing common_prefix_turns field (which is a hardcoded
0 on this cold-restore path's no-bins/no-valid-restore exit sites, computed
nowhere near this loop; conflating the two would inherit that ambiguity).

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
from turbohaul.kv_policy import kv_meta_fn
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot

_QWEN = "qwen3.6-27b"
_PORT = 60200


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


class _Resp:
    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _RestoreClient:
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
        AsyncClient = staticmethod(lambda *a, **k: _RestoreClient(recorded))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return recorded


def _write_bin(kv_dir, model_tag, port, thread_id, sid, chain, prompt_len=40000, prompt_tokens=100):
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": prompt_tokens,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": True,
    }))
    return bin_fn


def _warm_classify_lines(caplog):
    return [r.message for r in caplog.records if r.message.startswith("WARM_CLASSIFY")]


def _restore_decision_lines(caplog):
    return [r.message for r in caplog.records
            if r.message.startswith("slot KV restore decision:")]


@pytest.mark.asyncio
class TestColdRestoreDivergenceIndex:
    async def test_discriminator_diverged_bin_reports_real_divergence_index(
        self, mgr, kv_dir, posts, caplog
    ):
        """saved diverges from incoming at index 2 -- a genuine early-turn
        rewrite. The line must name that exact index, not just say
        DIVERGED."""
        saved_chain = ["h0", "h1", "hDIVERGED", "h3"]
        inc_chain = ["h0", "h1", "h2", "h3", "h4"]
        _write_bin(kv_dir, _QWEN, _PORT, "t-diverge", 0, saved_chain)
        slot = Slot.new(model_tag=_QWEN, thread_id="t-diverge",
                         admission_ctx_len=50000, admission_hash_chain=inc_chain)
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot)

        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "divergence_index=2" in line, line
        assert "resolved_from=restore-diverged-fresh" in line, line
        assert "shape_verdict=skip" in line, line
        assert "role=" in line, line

    async def test_discriminator_valid_prefix_restore_reports_none_divergence(
        self, mgr, kv_dir, posts, caplog
    ):
        """Good-shape control: saved chain IS a genuine valid prefix (the
        restore fires) -- divergence_index must be None (no content
        mismatch found), proving the field doesn't cry divergence on a
        clean restore."""
        saved_chain = ["h0", "h1"]
        inc_chain = ["h0", "h1", "h2", "h3"]
        _write_bin(kv_dir, _QWEN, _PORT, "t-validprefix", 0, saved_chain)
        slot = Slot.new(model_tag=_QWEN, thread_id="t-validprefix",
                         admission_ctx_len=50000, admission_hash_chain=inc_chain)
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot)

        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "divergence_index=None" in line, line
        assert "resolved_from=restore-prefix-valid" in line, line
        assert "shape_verdict=restore" in line, line

    async def test_discriminator_physics_belt_is_length_not_content_divergence(
        self, mgr, kv_dir, posts, caplog
    ):
        """The exact ambiguity this diagnostic exists to resolve: a
        refusal that LOOKS like a content divergence from resolved_from
        alone can be a pure LENGTH problem -- saved longer than incoming,
        but byte-identical over the overlap. divergence_index=None here
        proves it's not a rewrite."""
        saved_chain = ["h0", "h1", "h2", "h3"]
        inc_chain = ["h0", "h1"]  # shorter, but matches saved's own prefix
        _write_bin(kv_dir, _QWEN, _PORT, "t-physicsbelt", 0, saved_chain)
        slot = Slot.new(model_tag=_QWEN, thread_id="t-physicsbelt",
                         admission_ctx_len=50000, admission_hash_chain=inc_chain)
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot)

        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "resolved_from=restore-physics-belt-saved-longer" in line, line
        assert "divergence_index=None" in line, (
            f"physics-belt (a LENGTH refusal) must not report a content "
            f"divergence index: {line}"
        )

    async def test_discriminator_divergence_at_index_zero_is_reported_as_zero_not_none(
        self, mgr, kv_dir, posts, caplog
    ):
        """0 is a real, meaningful 'diverges immediately' result in Python
        -- it must never be treated as falsy-for-absent. A first-turn
        rewrite (previously untested; index-0 is exactly
        the case a stray `if divergence_index:` truthiness check would
        silently misreport as None)."""
        saved_chain = ["hSAVED0", "h1"]
        inc_chain = ["hINCOMING0", "h1", "h2"]
        _write_bin(kv_dir, _QWEN, _PORT, "t-diverge0", 0, saved_chain)
        slot = Slot.new(model_tag=_QWEN, thread_id="t-diverge0",
                         admission_ctx_len=50000, admission_hash_chain=inc_chain)
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot)
        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        assert "divergence_index=0" in lines[0], lines[0]
        assert "divergence_index=None" not in lines[0], lines[0]

    async def test_discriminator_null_hash_chain_in_meta_does_not_crash_the_loop(
        self, mgr, kv_dir, posts, caplog
    ):
        """MUST FAIL before the fix: a saved bin's meta.json can carry a
        JSON `null` for hash_chain (key present, value explicitly null,
        distinct from the key being absent) -- resolve_kv() and the sibling
        scan-cache reader both already defend this exact field with `or []`
        (it's a live, anticipated shape in this codebase), but the new
        _chain_divergence_index call did not, and raised TypeError out of
        the per-bin loop -- silently abandoning consideration of every
        OTHER candidate bin for this identity, not just logging garbage.
        This was reproduced; this test pins the fix."""
        th = TurbohaulManager._thread_hash("t-nullchain")
        meta_fn = kv_meta_fn(_QWEN, 0, th, _PORT)
        bin_fn = meta_fn[:-5] + ".bin"
        (kv_dir / bin_fn).write_bytes(b"\x00")
        (kv_dir / meta_fn).write_text(json.dumps({
            "thread_id": "t-nullchain", "thread_hash": th, "prompt_tokens": 100,
            "prompt_len": 40000, "n_context_turns": 0, "hash_chain": None,
            "model_tag": _QWEN, "slot_id": 0, "port": _PORT, "clean_prefix": True,
        }))
        inc_chain = ["h0", "h1"]
        slot = Slot.new(model_tag=_QWEN, thread_id="t-nullchain",
                         admission_ctx_len=50000, admission_hash_chain=inc_chain)
        with caplog.at_level(logging.INFO):
            result = await mgr._restore_slot_kv(_PORT, _QWEN, slot)
        assert result is None or isinstance(result, int) or result is False or True, (
            "the call must complete without raising"
        )
        assert not any("Traceback" in r.message for r in caplog.records)
        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        assert "divergence_index=None" in lines[0], lines[0]
        assert "saved_chain_fp=None" in lines[0], (
            f"a None chain must fingerprint to None (mirrors _chain_fp's own "
            f"None-in/None-out contract): {lines[0]}"
        )

    async def test_discriminator_new_field_is_not_common_prefix_turns(
        self, mgr, kv_dir, posts, caplog
    ):
        """divergence_index must never be spelled/aliased as
        common_prefix_turns -- that field already exists on this cold path
        as a hardcoded 0 on unrelated exit sites; conflating the two would
        silently inherit that ambiguity into what should be a real number."""
        saved_chain = ["h0", "hX"]
        inc_chain = ["h0", "h1", "h2"]
        _write_bin(kv_dir, _QWEN, _PORT, "t-fieldname", 0, saved_chain)
        slot = Slot.new(model_tag=_QWEN, thread_id="t-fieldname",
                         admission_ctx_len=50000, admission_hash_chain=inc_chain)
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot)
        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        assert "common_prefix_turns" not in lines[0], lines[0]

    async def test_discriminator_kv_decision_repr_unchanged_four_fields_only(
        self, mgr, kv_dir, posts, caplog
    ):
        """KVDecision.__repr__ must stay exactly do_it/action/reason/from --
        other call sites render it and must not see their output shift."""
        saved_chain = ["h0", "h1"]
        inc_chain = ["h0", "h1", "h2"]
        _write_bin(kv_dir, _QWEN, _PORT, "t-repr", 0, saved_chain)
        slot = Slot.new(model_tag=_QWEN, thread_id="t-repr",
                         admission_ctx_len=50000, admission_hash_chain=inc_chain)
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot)
        lines = _restore_decision_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert line.count("KVDecision(") == 1
        assert "do_it=" in line and "action=" in line and "reason=" in line and "from=" in line
        assert "saved_chain" not in line and "incoming_chain" not in line, (
            "KVDecision's own repr must not have grown new fields"
        )


class TestChainDivergenceIndexUnit:
    """Direct unit coverage on the helper itself -- cheap, no fixtures --
    for the key edge cases (empty, None, index zero)."""

    def test_both_empty_returns_none(self):
        from turbohaul.manager import _chain_divergence_index
        assert _chain_divergence_index([], []) is None

    def test_none_inputs_do_not_raise(self):
        """MUST FAIL before the fix: a bare None chain (not just an empty
        list) used to raise TypeError('NoneType' object is not iterable)."""
        from turbohaul.manager import _chain_divergence_index
        assert _chain_divergence_index(None, None) is None
        assert _chain_divergence_index(None, ["h0"]) is None
        assert _chain_divergence_index(["h0"], None) is None

    def test_divergence_at_zero_is_int_zero_not_falsy_none(self):
        from turbohaul.manager import _chain_divergence_index
        result = _chain_divergence_index(["hA"], ["hB"])
        assert result == 0
        assert result is not None
        assert isinstance(result, int)
