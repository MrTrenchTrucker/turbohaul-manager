"""Observability hardening — the forced-clean-restore log lines
carry the fields that would have made the discarded-warm-slot regression
self-announcing, and the thread id on those two lines is no longer truncated.

The "warm-path forced clean restore" line says WHAT was
restored and WHAT it will reuse, and also what the live warm slot held or how
much of it was thrown away, and carries the full thread id, not a 12-char
truncation (which destroyed longer session ids entirely). Behavior is
byte-identical to before -- these are additive fields on the existing
decision dict, emitted through the pre-existing _emit_classifier_decision
chokepoint, plus more detail appended to the pre-existing log lines
themselves (same lines, no new ones).

Run:
    PYTHONPATH=<repo>/src python3 -m pytest tests/test_force_clean_observability.py -v
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
from turbohaul.kv_policy import _prefix_hash_chain, kv_meta_fn, kv_save_fn
from turbohaul.manager import TurbohaulManager, _kv_shadow_meta_fn, _kv_shadow_save_fn
from turbohaul.slot import Slot

pytestmark = pytest.mark.asyncio

_QWEN = "qwen3.6-27b"
_PORT = 59900
# Long enough that the OLD tid[:12] truncation would visibly cut it, and the
# NEW behavior (full tid) would visibly not.
_LONG_TID = "sess-0000000000-fullidentity-not-twelve-chars"


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


class _FakeResp:
    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _FakeClient:
    def __init__(self, posts):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        return _FakeResp()


@pytest.fixture
def posts(monkeypatch):
    recorded = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _FakeClient(recorded))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return recorded


_SYS = {"role": "system", "content": "system prompt long enough to matter"}
_U1 = {"role": "user", "content": "first user turn"}
_U2 = {"role": "user", "content": "second user turn"}


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


def _force_clean_scenario(kv_dir):
    """Exact shape of test_classifier.py::test_force_on_think_strip_user_turn
    -- the key case: warm KV holds the <think> turn, incoming is think-stripped,
    clean bin is a valid (shorter) prefix, warm diverges -> forces."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_clean_bin(kv_dir, _QWEN, _PORT, _LONG_TID, 0, clean_chain)
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    warm = _prefix_hash_chain(
        [_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
    slot = Slot.new(model_tag=_QWEN, thread_id=_LONG_TID, admission_hash_chain=inc)
    return slot, warm, bin_fn, clean_chain


class TestForcedCleanRestoreObservability:
    async def test_discriminator_clean_restore_line_carries_discarded_tokens_field(
        self, mgr, kv_dir, posts, caplog
    ):
        slot, warm, bin_fn, clean_chain = _force_clean_scenario(kv_dir)
        with caplog.at_level(logging.INFO):
            d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, warm)
        assert d["forced_clean_restore"] is True

        force_lines = [r.message for r in caplog.records
                       if "warm-path forced clean restore" in r.message]
        assert len(force_lines) == 1, force_lines
        line = force_lines[0]
        # ★ discarded_tokens must be on the SAME line as the restore verdict
        # (one line carries the whole verdict) -- honestly unavailable
        # (warm_chain carries turn hashes, not token counts) but present and
        # named, not silently absent.
        assert "discarded_tokens=unavailable" in line, line
        assert f"warm_slot_turns={len(warm)}" in line, line
        assert f"discarded_turns={len(warm)}" in line, line
        assert "rule=FORCE_CLEAN_WARM_DIVERGES_QWEN_FAMILY" in line, line

    async def test_discriminator_clean_restore_line_thread_id_not_truncated(
        self, mgr, kv_dir, posts, caplog
    ):
        slot, warm, bin_fn, clean_chain = _force_clean_scenario(kv_dir)
        with caplog.at_level(logging.INFO):
            await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, warm)
        force_lines = [r.message for r in caplog.records
                       if "warm-path forced clean restore" in r.message]
        assert len(force_lines) == 1
        line = force_lines[0]
        assert _LONG_TID in line, (
            f"full thread id not found in the log line -- still truncated: {line!r}"
        )
        # the specific truncated form this fix removes
        assert _LONG_TID[:12] + "," not in line or _LONG_TID in line

    async def test_discriminator_decision_dict_carries_all_force_clean_fields(
        self, mgr, kv_dir, posts
    ):
        """Non-vacuous per-field check on the object _emit_classifier_decision
        actually receives (mgr._kv_classifier_last), not just the log text."""
        slot, warm, bin_fn, clean_chain = _force_clean_scenario(kv_dir)
        d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, warm)
        last = mgr._kv_classifier_last
        assert last is d
        assert last["chosen_bin"] == bin_fn
        assert last["chosen_turns"] == len(clean_chain)
        assert last["warm_slot_turns"] == len(warm)
        assert last["warm_slot_tokens"] == "unavailable"
        assert last["discarded_turns"] == len(warm)
        assert last["discarded_tokens"] == "unavailable"
        assert last["rule_name"] == "FORCE_CLEAN_WARM_DIVERGES_QWEN_FAMILY"
        assert last["candidates"] == [{"bin": bin_fn, "turns": len(clean_chain)}]

    async def test_good_shape_survives_no_force_no_new_fields_required(
        self, mgr, kv_dir, posts
    ):
        """Control: a no-force decision (true continuation, warm covers) must
        not be expected to carry the force-only fields -- proves the above
        assertions are exercising the force path specifically, not something
        present on every decision unconditionally."""
        clean_chain = _prefix_hash_chain([_SYS, _U1])
        _write_clean_bin(kv_dir, "m", _PORT, "t-nocontrol", 0, clean_chain)
        a1 = {"role": "assistant", "content": "A"}
        warm = _prefix_hash_chain([_SYS, _U1, a1])
        inc = _prefix_hash_chain([_SYS, _U1, a1, _U2])
        slot = Slot.new(model_tag="m", thread_id="t-nocontrol", admission_hash_chain=inc)
        d = await mgr._maybe_force_clean_restore(_PORT, "m", slot, warm)
        assert d["forced_clean_restore"] is False
        assert "chosen_bin" not in d
        assert posts == []


class TestShadowBranchObservability:
    """Same fields, same same-line requirement, on the SHADOW restore branch
    (manager.py's r_kind == "shadow" arm) -- mirrors
    test_shadow_restore_prefer.py::test_flag_on_prefers_valid_shadow's shape."""

    _TID = "agent-ip-10.0.0.5-shadowbranch"
    _SID_CLEAN = 0
    _SID_SHADOW = 3

    @staticmethod
    def _msgs(k):
        return [{"role": ("user" if i % 2 == 0 else "assistant"),
                 "content": f"turn-{i}-content"} for i in range(k)]

    def _scenario(self, kv_dir):
        _THINK_FREE = {"role": "assistant", "content": "THE_FINAL_ANSWER"}
        _WITH_THINK = {"role": "assistant",
                       "content": "<think>reasoning</think>THE_FINAL_ANSWER"}
        _USER_NP1 = {"role": "user", "content": "the next user question"}
        clean_msgs = self._msgs(5)
        shadow_msgs = clean_msgs + [_THINK_FREE]
        inc_msgs = shadow_msgs + [_USER_NP1]
        warm_msgs = clean_msgs + [_WITH_THINK]
        clean_chain = _prefix_hash_chain(clean_msgs)
        shadow_chain = _prefix_hash_chain(shadow_msgs)
        inc_chain = _prefix_hash_chain(inc_msgs)
        warm_chain = _prefix_hash_chain(warm_msgs)

        th = TurbohaulManager._thread_hash(self._TID)
        clean_bin = kv_save_fn(_QWEN, self._SID_CLEAN, th, _PORT)
        (kv_dir / clean_bin).write_bytes(b"CLEAN_ANCHOR_BYTES")
        (kv_dir / kv_meta_fn(_QWEN, self._SID_CLEAN, th, _PORT)).write_text(json.dumps({
            "thread_id": self._TID, "thread_hash": th, "prompt_tokens": 500,
            "prompt_len": 40000, "n_context_turns": len(clean_chain),
            "hash_chain": clean_chain, "model_tag": _QWEN,
            "slot_id": self._SID_CLEAN, "port": _PORT, "clean_prefix": True,
        }))
        shadow_bin = _kv_shadow_save_fn(_QWEN, self._SID_SHADOW, th, _PORT)
        (kv_dir / shadow_bin).write_bytes(b"SHADOW_THINKFREE_BYTES")
        (kv_dir / _kv_shadow_meta_fn(_QWEN, self._SID_SHADOW, th, _PORT)).write_text(
            json.dumps({
                "thread_id": self._TID, "thread_hash": th, "prompt_tokens": 600,
                "prompt_len": 45000, "n_context_turns": len(shadow_chain),
                "hash_chain": shadow_chain, "model_tag": _QWEN,
                "slot_id": self._SID_SHADOW, "port": _PORT,
                "clean_prefix": False, "shadow": True,
            }))
        slot = Slot.new(model_tag=_QWEN, thread_id=self._TID, admission_hash_chain=inc_chain)
        return slot, warm_chain, clean_bin, clean_chain, shadow_bin, shadow_chain

    async def test_discriminator_shadow_restore_line_carries_all_fields(
        self, mgr, kv_dir, posts, monkeypatch, caplog
    ):
        monkeypatch.setenv("TURBOHAUL_SHADOW_RESTORE_PREFER", "1")
        (slot, warm_chain, clean_bin, clean_chain,
         shadow_bin, shadow_chain) = self._scenario(kv_dir)
        with caplog.at_level(logging.INFO):
            d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, warm_chain)
        assert d["resolved_from"] == "warm-force-shadow-restore"
        assert d["chosen_bin"] == shadow_bin
        assert d["chosen_turns"] == len(shadow_chain)
        assert d["warm_slot_turns"] == len(warm_chain)
        assert d["discarded_tokens"] == "unavailable"
        assert d["rule_name"] == "FORCE_CLEAN_WARM_DIVERGES_QWEN_FAMILY"
        # candidates lists BOTH bins actually looked at, not just the winner
        assert {"bin": clean_bin, "turns": len(clean_chain)} in d["candidates"]
        assert {"bin": shadow_bin, "turns": len(shadow_chain)} in d["candidates"]

        shadow_lines = [r.message for r in caplog.records
                        if "warm-path forced SHADOW restore" in r.message]
        assert len(shadow_lines) == 1, shadow_lines
        line = shadow_lines[0]
        assert "discarded_tokens=unavailable" in line, line
        assert self._TID in line, f"thread id truncated: {line!r}"
