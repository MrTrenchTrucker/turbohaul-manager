"""SHADOW-BIN RESTORE-PREFERENCE (manager delta).

Covers `_shadow_restore_prefer_enabled` + `_find_shadow_bin` + the flag-gated
preference wired into `_maybe_force_clean_restore` (the ONE warm forced-restore
seam where the reasoning-family divergence collapse fires). The gate that DECIDES
whether to force + the restore path (kv_policy.resolve_kv / _resolve_restore) are
NOT touched — this only changes WHICH already-valid bin gets restored.

Scenarios (all in the warm force-fires regime: clean anchor ⊑ incoming AND the
engine's warm KV holds the with-<think> turn that diverges from the think-stripped
resend -> the gate forces):

  1. FLAG OFF (default)  -> the clean anchor is restored (byte-identical to today);
     the shadow reader is never even consulted (counts stay empty).
  2. FLAG ON + valid, long-enough shadow bin -> the `.shadow` bin is PREFERRED
     (a longer valid prefix -> next decode strict-extends the think-free state).
  3. FLAG ON + STALE shadow bin (NOT a prefix of incoming) -> rejected by the SAME
     _is_prefix_match bar the clean path uses -> clean anchor restored.
  4. FLAG ON + NO shadow bin -> clean anchor restored.
  5. FLAG ON + valid but SHORTER shadow bin -> no-downgrade guard rejects it
     (never shrink the reused prefix) -> clean anchor restored.

Fixtures mirror test_shadow_reprefill.py.
"""
import json

from types import SimpleNamespace

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
from turbohaul.manager import (
    TurbohaulManager,
    _kv_shadow_meta_fn,
    _kv_shadow_save_fn,
)


# --- fixtures (mirror test_shadow_reprefill.py) ---------------------------------
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
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    """Isolated SLOT_SAVE_DIR (re-imported per call inside the manager methods)."""
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture
def prefer_on(monkeypatch):
    """Enable the RESTORE-side shadow-preference for a test."""
    monkeypatch.setenv("TURBOHAUL_SHADOW_RESTORE_PREFER", "1")


@pytest.fixture(autouse=True)
def _prefer_off_by_default(monkeypatch):
    """Guarantee a clean default even if the ambient env has the flag set."""
    monkeypatch.delenv("TURBOHAUL_SHADOW_RESTORE_PREFER", raising=False)
    # The WARM forced clean-bin restore is DEFAULT OFF; this module tests the
    # restore TARGET-preference INSIDE the force, so enable the force here.
    monkeypatch.setenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", "1")


# --- restore-POST capture (only /slots/{sid}?action=restore fires here) ---------
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
def capture_restore(monkeypatch):
    posts = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _RestoreClient(posts))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return posts


# --- shared conversation shape --------------------------------------------------
_QWEN = "qwen3-27b"          # tag matched by the reasoning-family force gate
_PORT = 59500
_TID = "agent-ip-10.0.0.5"
_SID_CLEAN = 0
_SID_SHADOW = 3                  # distinct slot so the POST url proves which won

_THINK_FREE = {"role": "assistant", "content": "THE_FINAL_ANSWER"}
_WITH_THINK = {"role": "assistant", "content": "<think>reasoning</think>THE_FINAL_ANSWER"}
_USER_NP1 = {"role": "user", "content": "the next user question"}


def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


# Turn N: the clean anchor is [1..N] (ends on the user turn just asked).
CLEAN_MSGS = _msgs(5)                              # user0,asst1,user2,asst3,user4
SHADOW_MSGS = CLEAN_MSGS + [_THINK_FREE]           # + think-free assistant-N (step c)
INC_MSGS = SHADOW_MSGS + [_USER_NP1]               # turn N+1 (harness resends stripped)
WARM_MSGS = CLEAN_MSGS + [_WITH_THINK]             # engine warm KV holds the with-think turn

CLEAN_CHAIN = _prefix_hash_chain(CLEAN_MSGS)       # len 5, ⊑ INC
SHADOW_CHAIN = _prefix_hash_chain(SHADOW_MSGS)     # len 6, ⊑ INC, longer than clean
INC_CHAIN = _prefix_hash_chain(INC_MSGS)           # len 7
WARM_CHAIN = _prefix_hash_chain(WARM_MSGS)         # len 6, DIVERGES from INC at idx 5

# stale: diverges at turn 0 -> NOT a prefix of INC (rejected by the validity bar)
STALE_SHADOW_CHAIN = _prefix_hash_chain(
    [{"role": "user", "content": "WRONG-turn-0"}] + _msgs(5)[1:] + [_THINK_FREE])
# valid prefix but SHORTER than the (here longer) clean anchor -> no-downgrade skip
SHORT_SHADOW_CHAIN = _prefix_hash_chain(_msgs(3))  # len 3, ⊑ INC
LONG_CLEAN_CHAIN = _prefix_hash_chain(CLEAN_MSGS + [_THINK_FREE])  # len 6, ⊑ INC


def _slot(inc_chain=INC_CHAIN, tid=_TID):
    return SimpleNamespace(thread_id=tid, admission_hash_chain=inc_chain)


def _write_clean(kv_dir, sid, chain, tid=_TID, port=_PORT, prompt_len=40000):
    th = TurbohaulManager._thread_hash(tid)
    bin_fn = kv_save_fn(_QWEN, sid, th, port)
    meta_fn = kv_meta_fn(_QWEN, sid, th, port)
    (kv_dir / bin_fn).write_bytes(b"CLEAN_ANCHOR_BYTES")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": tid, "thread_hash": th, "prompt_tokens": 500,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": _QWEN, "slot_id": sid, "port": port, "clean_prefix": True,
    }))
    return bin_fn


def _write_shadow(kv_dir, sid, chain, tid=_TID, port=_PORT):
    th = TurbohaulManager._thread_hash(tid)
    bin_fn = _kv_shadow_save_fn(_QWEN, sid, th, port)
    meta_fn = _kv_shadow_meta_fn(_QWEN, sid, th, port)
    (kv_dir / bin_fn).write_bytes(b"SHADOW_THINKFREE_BYTES")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": tid, "thread_hash": th, "prompt_tokens": 600,
        "prompt_len": 45000, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": _QWEN, "slot_id": sid, "port": port,
        "clean_prefix": False, "shadow": True,
    }))
    return bin_fn


def _the_restore(posts):
    """The single restore POST (url, body)."""
    restores = [(u, b) for (u, b) in posts if "action=restore" in u]
    assert len(restores) == 1, f"expected exactly one restore POST, got {posts}"
    return restores[0]


# ================================================================================
# Pre-flight: the scenario really does make the force gate fire (else the tests
# would pass vacuously with zero restores).
# ================================================================================
@pytest.mark.asyncio
async def test_scenario_forces_a_restore(mgr, kv_dir, capture_restore):
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    decision = await mgr._maybe_force_clean_restore(_PORT, _QWEN, _slot(), WARM_CHAIN)
    assert decision["forced_clean_restore"] is True     # the gate fired
    assert decision["action"] == "restore"


# ================================================================================
# 1. FLAG OFF (default) -> clean anchor restored, shadow reader never consulted
# ================================================================================
@pytest.mark.asyncio
async def test_flag_off_restores_clean_anchor(mgr, kv_dir, capture_restore):
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    _write_shadow(kv_dir, _SID_SHADOW, SHADOW_CHAIN)    # present, but flag OFF -> ignored

    decision = await mgr._maybe_force_clean_restore(_PORT, _QWEN, _slot(), WARM_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin                # today's clean anchor
    assert f"/slots/{_SID_CLEAN}?action=restore" in url
    assert decision["forced_clean_restore"] is True
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert decision["clean_bin_id"] == clean_bin
    assert mgr._kv_shadow_restore_counts == {}          # reader never ran (byte-identical)


# ================================================================================
# 2. FLAG ON + valid, long-enough shadow -> SHADOW preferred
# ================================================================================
@pytest.mark.asyncio
async def test_flag_on_prefers_valid_shadow(mgr, kv_dir, capture_restore, prefer_on):
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    shadow_bin = _write_shadow(kv_dir, _SID_SHADOW, SHADOW_CHAIN)

    decision = await mgr._maybe_force_clean_restore(_PORT, _QWEN, _slot(), WARM_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == shadow_bin               # the .shadow bin won
    assert f"/slots/{_SID_SHADOW}?action=restore" in url  # its distinct slot
    assert decision["forced_clean_restore"] is True
    assert decision["resolved_from"] == "warm-force-shadow-restore"
    assert decision["clean_bin_id"] == shadow_bin
    assert mgr._kv_shadow_restore_counts.get("preferred") == 1
    assert mgr._kv_shadow_restore_counts.get("clean_fallback") is None


# ================================================================================
# 3. FLAG ON + STALE shadow (not a prefix) -> rejected -> clean anchor restored
# ================================================================================
@pytest.mark.asyncio
async def test_flag_on_stale_shadow_rejected(mgr, kv_dir, capture_restore, prefer_on):
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    _write_shadow(kv_dir, _SID_SHADOW, STALE_SHADOW_CHAIN)   # NOT a prefix of INC

    decision = await mgr._maybe_force_clean_restore(_PORT, _QWEN, _slot(), WARM_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin                # fell back to the clean anchor
    assert f"/slots/{_SID_CLEAN}?action=restore" in url
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_shadow_restore_counts.get("preferred") is None
    assert mgr._kv_shadow_restore_counts.get("clean_fallback") == 1


# ================================================================================
# 4. FLAG ON + NO shadow bin -> clean anchor restored
# ================================================================================
@pytest.mark.asyncio
async def test_flag_on_no_shadow_uses_clean(mgr, kv_dir, capture_restore, prefer_on):
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)   # no shadow bin at all

    decision = await mgr._maybe_force_clean_restore(_PORT, _QWEN, _slot(), WARM_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert f"/slots/{_SID_CLEAN}?action=restore" in url
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_shadow_restore_counts.get("clean_fallback") == 1


# ================================================================================
# 5. FLAG ON + valid but SHORTER shadow -> no-downgrade guard -> clean restored
# ================================================================================
@pytest.mark.asyncio
async def test_flag_on_shorter_shadow_not_downgraded(mgr, kv_dir, capture_restore, prefer_on):
    # clean anchor is LONGER (6) than the (valid) shadow (3); preferring the shadow
    # would SHRINK the reused prefix -> the no-downgrade guard keeps the clean anchor.
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, LONG_CLEAN_CHAIN)
    _write_shadow(kv_dir, _SID_SHADOW, SHORT_SHADOW_CHAIN)

    decision = await mgr._maybe_force_clean_restore(_PORT, _QWEN, _slot(), WARM_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin                # longer clean anchor kept
    assert f"/slots/{_SID_CLEAN}?action=restore" in url
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_shadow_restore_counts.get("preferred") is None
    assert mgr._kv_shadow_restore_counts.get("clean_fallback") == 1


# ================================================================================
# 6. _find_shadow_bin unit: matches only `.shadow.json`/shadow:true, longest chain,
#    and never returns the clean anchor (nor vice-versa).
# ================================================================================
@pytest.mark.asyncio
async def test_find_shadow_bin_isolation(mgr, kv_dir):
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    shadow_bin = _write_shadow(kv_dir, _SID_SHADOW, SHADOW_CHAIN)

    sh = mgr._find_shadow_bin(_PORT, _QWEN, _TID)
    assert sh is not None
    s_bin, s_chain, s_sid = sh
    assert s_bin == shadow_bin and s_bin.endswith(".shadow.bin")
    assert s_chain == SHADOW_CHAIN and s_sid == _SID_SHADOW

    # the clean finder must NOT pick up the shadow bin, and returns the clean one
    cl = mgr._find_clean_bin(_PORT, _QWEN, _TID)
    assert cl is not None and cl[0] == clean_bin and not cl[0].endswith(".shadow.bin")

    # no shadow present -> None (caller falls through to clean)
    assert mgr._find_shadow_bin(_PORT, _QWEN, "different-thread") is None


# ================================================================================
# 7. flag reader parses truthy/falsey like its SAVE-side mirror
# ================================================================================
@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("YES", True), ("On", True),
    ("0", False), ("false", False), ("", False), ("nope", False),
])
def test_flag_reader(monkeypatch, val, expected):
    monkeypatch.setenv("TURBOHAUL_SHADOW_RESTORE_PREFER", val)
    assert manager_mod._shadow_restore_prefer_enabled() is expected


def test_flag_reader_unset(monkeypatch):
    monkeypatch.delenv("TURBOHAUL_SHADOW_RESTORE_PREFER", raising=False)
    assert manager_mod._shadow_restore_prefer_enabled() is False
