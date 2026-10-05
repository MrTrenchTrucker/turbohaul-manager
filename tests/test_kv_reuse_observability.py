"""Observability hardening: the KV-reuse metric
(reused vs. recomputed) is emitted by the component under test, on EVERY
restore decision, not just the ones that fire. Without this emission, reuse
was obtainable only from the engine's own log; the manager would not state it.

Design: KV_REUSE rides the shared _emit_classifier_decision chokepoint
unconditionally (see manager.py) -- incoming_turns/common_prefix_turns/
clean_bin_id are present on every decision dict built anywhere in the file
(both the warm and cold restore paths default them before any branch runs),
so this does not require touching every individual call site's own logic,
only stamping thread_hash (and, where cheaply real, restored_tokens) onto
the dicts. common_prefix_turn_pct is a TURN-based common-prefix ratio --
the manager never tokenizes the incoming request itself, so a token-based
version of it would be a mixed-unit guess; requested_tokens/recomputed_tokens
are honest explicit nulls for the same reason. restored_tokens is a REAL per-bin
engine-reported token count where the chokepoint has one (the cold path's
own saved-bin metadata), 0 where nothing restored, and an honest null on
the warm path (no equivalent field is available there without widening
_find_clean_bin's return shape, judged out of scope here).

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this file> -v
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

pytestmark = pytest.mark.asyncio

_QWEN = "qwen3.6-27b"
_PORT = 60100


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


_SYS = {"role": "system", "content": "system prompt long enough to matter"}
_U1 = {"role": "user", "content": "first user turn"}
_U2 = {"role": "user", "content": "second user turn"}
_PROMPT_TOKENS = 12345


def _write_clean_bin(kv_dir, model_tag, port, thread_id, sid, chain, prompt_len=40000):
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": _PROMPT_TOKENS,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": True,
    }))
    return bin_fn


def _kv_reuse_lines(caplog):
    """The reuse-observability decision line. The chokepoint fires one line
    per restore decision — KV_REUSE on the restore paths (carries the bin's
    token count), KV_CLASSIFY on the warm-path classify decisions
    (carries the classify path's turn-based estimate)."""
    return [
        r.message for r in caplog.records
        if r.message.startswith("KV_REUSE") or r.message.startswith("KV_CLASSIFY")
    ]


class TestKvReuseOnColdRestore:
    async def test_discriminator_successful_restore_carries_real_restored_tokens(
        self, mgr, kv_dir, posts, caplog
    ):
        clean_chain = _prefix_hash_chain([_SYS, _U1])
        bin_fn = _write_clean_bin(kv_dir, _QWEN, _PORT, "t-reuse-fire", 0, clean_chain)
        inc = _prefix_hash_chain(
            [_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
        slot = Slot.new(model_tag=_QWEN, thread_id="t-reuse-fire",
                         admission_ctx_len=50000, admission_hash_chain=inc)
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot)
        last = mgr._kv_classifier_last
        assert last["resolved_from"] == "wave-return-clean-restore"

        lines = _kv_reuse_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert f"restored_tokens={_PROMPT_TOKENS}" in line, line
        assert f"bin_id={bin_fn}" in line, line
        expected_pct = round(100.0 * len(clean_chain) / len(inc), 1)
        assert f"common_prefix_turn_pct={expected_pct}" in line, line
        assert "requested_tokens=unavailable" in line, line
        assert "recomputed_tokens=unavailable" in line, line

    async def test_discriminator_no_bins_carries_zero_restored_and_zero_reuse(
        self, mgr, kv_dir, posts, caplog
    ):
        inc = _prefix_hash_chain([_SYS, _U1])
        slot = Slot.new(model_tag=_QWEN, thread_id="t-reuse-nobins",
                         admission_ctx_len=50000, admission_hash_chain=inc)
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot)
        lines = _kv_reuse_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "restored_tokens=0" in line, line
        assert "common_prefix_turn_pct=0.0" in line, line
        assert "bin_id=None" in line, line

    async def test_discriminator_failed_restore_post_carries_zero_restored_tokens(
        self, mgr, kv_dir, caplog, monkeypatch
    ):
        """Non-vacuous pairing with the confirmed-restore test above: an
        engine-side POST failure must NOT still claim the bin's tokens as
        restored -- the metric stays truthful (mirrors the existing
        forced_clean_restore=False-on-failure discipline)."""
        clean_chain = _prefix_hash_chain([_SYS, _U1])
        _write_clean_bin(kv_dir, _QWEN, _PORT, "t-reuse-fail", 0, clean_chain)
        inc = _prefix_hash_chain(
            [_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
        slot = Slot.new(model_tag=_QWEN, thread_id="t-reuse-fail",
                         admission_ctx_len=50000, admission_hash_chain=inc)

        class _FailResp:
            def raise_for_status(self):
                raise RuntimeError("engine 500")

        class _FailClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, **kw):
                return _FailResp()

        class _FakeHttpxFail:
            AsyncClient = staticmethod(lambda *a, **k: _FailClient())
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpxFail)
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot)
        lines = _kv_reuse_lines(caplog)
        assert len(lines) == 1, lines
        assert "restored_tokens=0" in lines[0], lines[0]


class TestKvReuseOnWarmPath:
    @pytest.fixture(autouse=True)
    def _warm_force_on(self, monkeypatch):
        monkeypatch.setenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", "1")

    async def test_discriminator_warm_force_fires_carries_thread_hash_and_common_prefix_turn_pct(
        self, mgr, kv_dir, posts, caplog
    ):
        clean_chain = _prefix_hash_chain([_SYS, _U1])
        bin_fn = _write_clean_bin(kv_dir, _QWEN, _PORT, "t-warmreuse", 0, clean_chain)
        warm = _prefix_hash_chain(
            [_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
        inc = _prefix_hash_chain(
            [_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
        slot = Slot.new(model_tag=_QWEN, thread_id="t-warmreuse", admission_hash_chain=inc)
        with caplog.at_level(logging.INFO):
            d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, warm)
        assert d["forced_clean_restore"] is True

        lines = _kv_reuse_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        # The warm-path classify decision fires the KV_CLASSIFY line (the
        # classify path's turn-based estimate). The classify path
        # never carried bin_id/restored_tokens (those are restore-decision data
        # on the KV_REUSE line); the classify line carries the estimate
        # fields the test name claims: thread_hash + common_prefix_turn_pct.
        assert "thread_hash=" in line and "thread_hash=None" not in line, line
        expected_pct = round(100.0 * len(clean_chain) / len(inc), 1)
        assert f"common_prefix_turn_pct={expected_pct}" in line, line
        assert "reused_turns=" in line and "requested_turns=" in line, line

    async def test_good_shape_survives_guard_skip_still_emits_zero_reuse(
        self, mgr, kv_dir, caplog
    ):
        """Control: even the earliest guard-skip default decision (no
        incoming chain at all) gets a KV_REUSE line -- 'on every restore
        decision' means every decision, not just the ones with something
        interesting to report."""
        slot = Slot.new(model_tag=_QWEN, thread_id="", admission_hash_chain=[])
        with caplog.at_level(logging.INFO):
            await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, [])
        lines = _kv_reuse_lines(caplog)
        assert len(lines) == 1, lines
        assert "common_prefix_turn_pct=None" in lines[0], lines[0]
