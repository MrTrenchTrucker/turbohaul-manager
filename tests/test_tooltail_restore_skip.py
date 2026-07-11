"""Tool-tail restore SKIP (Option A).

`_prefix_hash_chain` (kv_policy, BYTE-UNTOUCHED here) hashes role+content only, so an
assistant turn with truthy tool_calls (content null) and a role=="tool" result are
HASH-INVISIBLE. The upstream client re-serializes that tool region nondeterministically
every resend, so its TOKENS drift while the chain hash stays constant -> _is_prefix_match
is a FALSE POSITIVE -> the manager POSTs a token-STALE clean/shadow bin -> the engine hits
`stale > n_rs_seq` and CLEARs (full reprefill: a REGRESSION vs native reuse).

Fix (restore-DECISION path only): when the divergent tail a restore relies on (turns
BEYOND the common prefix) is tool-opaque, SKIP the force/POST and safe-degrade to the
engine's native get_common_prefix checkpoint reuse. A TEXT/think-strip tail (hash-
verifiable, no tool turn beyond common) STILL restores — the shadow restore-
preference is unaffected.

Covers BOTH gates:
  * WARM  — `_maybe_force_clean_restore` (force=False on a tool tail)
  * COLD  — `_restore_slot_kv` (skip the POST on a tool tail; gates the clean AND the
            cold-wire `.shadow` target — one POST, one guard)

Scenarios per path: (a) tool tail beyond common -> SKIP; (b) TEXT/think tail -> restore
STILL fires (shadow-preference intact); (c) tool turn BEFORE common (settled prefix) ->
does NOT skip; (d) flag OFF -> pre-fix behavior (restore fires). Plus pure-helper +
flag-reader unit tests. Fixtures mirror test_shadow_restore_prefer.py /
test_shadow_cold_restore.py.
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
    _divergent_tail_is_tool_opaque,
    _kv_shadow_meta_fn,
    _kv_shadow_save_fn,
    _turn_is_tool_opaque,
)


# --- fixtures (mirror test_shadow_restore_prefer.py) ---------------------------------
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
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture(autouse=True)
def _flags_clean(monkeypatch):
    """Default: tool-tail skip OFF (unset — post default-flip), shadow flags OFF — a clean
    ambient env. Unset means the tool-tail restore-skip guard is DISABLED by default
    (the byte-match fix covers the tool region, so the skip is redundant); tests that
    need the skip active must opt in via the `skip_on` fixture."""
    for f in ("TURBOHAUL_TOOLTAIL_RESTORE_SKIP", "TURBOHAUL_SHADOW_REPREFILL",
              "TURBOHAUL_SHADOW_RESTORE_PREFER", "TURBOHAUL_SHADOW_COLD_RESTORE"):
        monkeypatch.delenv(f, raising=False)
    # The WARM forced clean-bin restore is now DEFAULT OFF. The WARM
    # scenarios here assert the FORCE path (the tool-tail SKIP sits INSIDE it), so
    # enable it. No-op for the COLD (_restore_slot_kv) scenarios (that path is un-gated).
    monkeypatch.setenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", "1")


@pytest.fixture
def skip_off(monkeypatch):
    monkeypatch.setenv("TURBOHAUL_TOOLTAIL_RESTORE_SKIP", "0")


@pytest.fixture
def skip_on(monkeypatch):
    """Emergency A/B-rollback floor: explicitly enable the tool-tail restore-skip guard
    (default is now OFF). Symmetric to `skip_off`."""
    monkeypatch.setenv("TURBOHAUL_TOOLTAIL_RESTORE_SKIP", "1")


@pytest.fixture
def cold_on(monkeypatch):
    monkeypatch.setenv("TURBOHAUL_SHADOW_COLD_RESTORE", "1")


@pytest.fixture
def warm_prefer_on(monkeypatch):
    monkeypatch.setenv("TURBOHAUL_SHADOW_RESTORE_PREFER", "1")


# --- restore-POST capture (only /slots/{sid}?action=restore fires here) -------------
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


def _restores(posts):
    return [(u, b) for (u, b) in posts if "action=restore" in u]


def _the_restore(posts):
    r = _restores(posts)
    assert len(r) == 1, f"expected exactly one restore POST, got {posts}"
    return r[0]


# --- conversation shapes ------------------------------------------------------------
_QWEN = "qwen3-27b"          # matches _is_qwen_family() -> the force gate is armed
_PORT = 59500
_TID = "agent-ip-10.0.0.5"
_SID_CLEAN = 0
_SID_SHADOW = 3

_TOOL_CALL = {
    "role": "assistant", "content": None,
    "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "search", "arguments": "{\"q\":\"x\"}"}}],
}
_TOOL_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "tool result payload"}
_THINK_FREE = {"role": "assistant", "content": "THE_FINAL_ANSWER"}
_WITH_THINK = {"role": "assistant", "content": "<think>reasoning</think>THE_FINAL_ANSWER"}
_USER_NP1 = {"role": "user", "content": "the next user question"}


def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


# Base = 5 settled turns. The clean anchor covers [1..5].
CLEAN_MSGS = _msgs(5)

# (a) TOOL tail — the tail beyond common (=5) is tool-opaque -> SKIP
INC_TOOL_MSGS = CLEAN_MSGS + [_TOOL_CALL, _TOOL_RESULT]

# (b) TEXT/think tail — think-free assistant + user; NO tool beyond common -> RESTORE fires
SHADOW_MSGS = CLEAN_MSGS + [_THINK_FREE]              # len 6 (think-free asst-N)
INC_TEXT_MSGS = SHADOW_MSGS + [_USER_NP1]             # len 7, tail = text

# (c) tool turn BEFORE common — tool region is settled INSIDE the clean anchor [1..5];
#     the divergent tail is pure text -> does NOT skip
EARLY_TOOL_MSGS = [
    {"role": "user", "content": "turn-0-content"},
    _TOOL_CALL,                                       # idx 1 (opaque, but settled)
    _TOOL_RESULT,                                     # idx 2 (opaque, but settled)
    {"role": "user", "content": "turn-3-content"},
    {"role": "assistant", "content": "turn-4-content"},
]                                                    # len 5 = the clean anchor coverage
INC_EARLY_MSGS = EARLY_TOOL_MSGS + [_THINK_FREE, _USER_NP1]   # tail (idx>=5) = text

CLEAN_CHAIN = _prefix_hash_chain(CLEAN_MSGS)          # 5, ⊑ every INC above
SHADOW_CHAIN = _prefix_hash_chain(SHADOW_MSGS)        # 6, ⊑ INC_TEXT, longer than clean
EARLY_CLEAN_CHAIN = _prefix_hash_chain(EARLY_TOOL_MSGS)  # 5, ⊑ INC_EARLY

INC_TOOL_CHAIN = _prefix_hash_chain(INC_TOOL_MSGS)
INC_TEXT_CHAIN = _prefix_hash_chain(INC_TEXT_MSGS)
INC_EARLY_CHAIN = _prefix_hash_chain(INC_EARLY_MSGS)

# warm KV holds the with-<think> turn -> diverges from every (think-stripped) INC at idx 5
WARM_CHAIN = _prefix_hash_chain(CLEAN_MSGS + [_WITH_THINK])
WARM_EARLY_CHAIN = _prefix_hash_chain(EARLY_TOOL_MSGS + [_WITH_THINK])


def _warm_slot(inc_msgs, inc_chain, tid=_TID):
    """Warm-seam slot: force gate reads thread_id + admission_hash_chain; the tool-tail
    guard reads client_meta['messages'] (1:1 aligned with the chain)."""
    return SimpleNamespace(
        thread_id=tid,
        admission_hash_chain=inc_chain,
        client_meta={"messages": inc_msgs},
    )


def _cold_slot(inc_msgs, inc_chain, tid=_TID):
    return SimpleNamespace(
        thread_id=tid,
        admission_ctx_len=50000,
        admission_hash_chain=inc_chain,
        client_meta={"messages": inc_msgs},
    )


def _write_clean(kv_dir, sid, chain, tid=_TID, port=_PORT, prompt_len=40000,
                 clean_prefix=True):
    # clean_prefix=False -> a WITH-<think> polluted anchor (the cold length-guard is
    # dropped so a shorter valid shadow wins).
    th = TurbohaulManager._thread_hash(tid)
    bin_fn = kv_save_fn(_QWEN, sid, th, port)
    meta_fn = kv_meta_fn(_QWEN, sid, th, port)
    (kv_dir / bin_fn).write_bytes(b"CLEAN_ANCHOR_BYTES")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": tid, "thread_hash": th, "prompt_tokens": 500,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "prompt_hash": "", "model_tag": _QWEN, "slot_id": sid, "port": port,
        "clean_prefix": clean_prefix,
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
        "prompt_hash": "", "model_tag": _QWEN, "slot_id": sid, "port": port,
        "clean_prefix": False, "shadow": True,
    }))
    return bin_fn


# ====================================================================================
# PURE HELPER UNITS — the exact tool-opaque predicate (constraint #2)
# ====================================================================================
@pytest.mark.parametrize("turn,expected", [
    ({"role": "tool", "content": "result"}, True),                       # role==tool
    (_TOOL_CALL, True),                                                  # asst + tool_calls
    ({"role": "assistant", "content": None}, True),                     # null content
    ({"role": "assistant", "content": ""}, True),                      # empty str
    ({"role": "assistant", "content": "   "}, True),                   # whitespace-only
    ({"role": "assistant", "content": []}, True),                      # empty list content
    ({"role": "assistant", "content": {}}, True),                      # empty dict content
    ({"role": "assistant", "content": "real text"}, False),            # normal text
    ({"role": "user", "content": "hi"}, False),                        # normal user
    ({"role": "assistant", "content": "x", "tool_calls": []}, False),  # falsy tool_calls
    ({"role": "assistant", "content": [{"type": "text", "text": "hi"}]}, False),  # non-empty list
    ("not-a-dict", False),                                               # coerced by chain
    (12345, False),
])
def test_turn_is_tool_opaque_predicate(turn, expected):
    assert _turn_is_tool_opaque(turn) is expected


def test_divergent_tail_scan_before_vs_after_common():
    # tool turn at idx 1; common=3 -> NOT scanned -> False
    msgs = [_USER_NP1, _TOOL_CALL, _TOOL_RESULT, _THINK_FREE, _USER_NP1]
    assert _divergent_tail_is_tool_opaque(msgs, 3) is False
    # common=1 -> the tool turns at idx 1,2 ARE in the tail -> True
    assert _divergent_tail_is_tool_opaque(msgs, 1) is True
    # common at/after the last index and no tail -> False
    assert _divergent_tail_is_tool_opaque(msgs, len(msgs)) is False
    # empty / None messages -> safe-degrade False
    assert _divergent_tail_is_tool_opaque([], 0) is False
    assert _divergent_tail_is_tool_opaque(None, 0) is False
    # negative/oversized common is clamped, never raises
    assert _divergent_tail_is_tool_opaque(msgs, -5) is True   # scans whole list


@pytest.mark.parametrize("val,expected", [
    ("", False), ("1", True), ("true", True), ("YES", True), ("On", True),
    ("0", False), ("false", False), ("no", False), ("OFF", False),
])
def test_flag_default_off(monkeypatch, val, expected):
    # Default-flip: UNSET -> OFF; ONLY explicit truthy -> ON.
    if val == "":
        monkeypatch.delenv("TURBOHAUL_TOOLTAIL_RESTORE_SKIP", raising=False)
    else:
        monkeypatch.setenv("TURBOHAUL_TOOLTAIL_RESTORE_SKIP", val)
    assert manager_mod._tooltail_restore_skip_enabled() is expected


# ====================================================================================
# WARM path — _maybe_force_clean_restore
# ====================================================================================
@pytest.mark.asyncio
async def test_warm_a_tool_tail_skips_force(mgr, kv_dir, capture_restore, skip_on):
    """(a) tool tail beyond common + guard ON (emergency floor) -> force SKIPPED, no restore POST."""
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    decision = await mgr._maybe_force_clean_restore(
        _PORT, _QWEN, _warm_slot(INC_TOOL_MSGS, INC_TOOL_CHAIN), WARM_CHAIN)

    assert _restores(capture_restore) == []              # no bin POSTed
    assert decision["forced_clean_restore"] is False
    assert decision["resolved_from"] == "warm-tooltail-skip"
    assert mgr._kv_tooltail_skip_counts.get("warm") == 1


@pytest.mark.asyncio
async def test_warm_b_text_tail_still_forces(mgr, kv_dir, capture_restore):
    """(b) TEXT/think tail -> the clean anchor is STILL force-restored (no regression)."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    decision = await mgr._maybe_force_clean_restore(
        _PORT, _QWEN, _warm_slot(INC_TEXT_MSGS, INC_TEXT_CHAIN), WARM_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert decision["forced_clean_restore"] is True
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_tooltail_skip_counts == {}            # guard never tripped


@pytest.mark.asyncio
async def test_warm_b_text_tail_shadow_preference_intact(
        mgr, kv_dir, capture_restore, warm_prefer_on):
    """(b') the shadow restore-preference still wins on a TEXT tail under the tool-tail guard."""
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    shadow_bin = _write_shadow(kv_dir, _SID_SHADOW, SHADOW_CHAIN)   # len 6, valid, longer
    decision = await mgr._maybe_force_clean_restore(
        _PORT, _QWEN, _warm_slot(INC_TEXT_MSGS, INC_TEXT_CHAIN), WARM_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == shadow_bin                # shadow preferred, NOT skipped
    assert f"/slots/{_SID_SHADOW}?action=restore" in url
    assert decision["resolved_from"] == "warm-force-shadow-restore"
    assert mgr._kv_shadow_restore_counts.get("preferred") == 1
    assert mgr._kv_tooltail_skip_counts == {}


@pytest.mark.asyncio
async def test_warm_c_tool_before_common_does_not_skip(mgr, kv_dir, capture_restore):
    """(c) tool turn INSIDE the settled prefix (< common) -> does NOT trip the skip."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, EARLY_CLEAN_CHAIN)
    decision = await mgr._maybe_force_clean_restore(
        _PORT, _QWEN, _warm_slot(INC_EARLY_MSGS, INC_EARLY_CHAIN), WARM_EARLY_CHAIN)

    url, body = _the_restore(capture_restore)            # restore STILL fires
    assert body["filename"] == clean_bin
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_tooltail_skip_counts == {}


@pytest.mark.asyncio
async def test_warm_d_flag_off_restores_on_tool_tail(mgr, kv_dir, capture_restore, skip_off):
    """(d) flag OFF -> pre-fix behavior: the (buggy) restore fires even on a tool tail."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    decision = await mgr._maybe_force_clean_restore(
        _PORT, _QWEN, _warm_slot(INC_TOOL_MSGS, INC_TOOL_CHAIN), WARM_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert decision["forced_clean_restore"] is True
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_tooltail_skip_counts == {}


@pytest.mark.asyncio
async def test_warm_e_default_restores_on_tool_tail(mgr, kv_dir, capture_restore):
    """(e) DEFAULT env (no flag set) -> THE DURABLE FIX: the force restore REUSES on a tool
    tail (the byte-match fix covers the region, so the skip is redundant). Distinct from
    test_warm_d's EXPLICIT env=0 — this proves the post-flip DEFAULT gives reuse. Would have
    FAILED pre-flip (default was ON -> the tool tail tripped the skip). Mirrors the live
    env=0 reuse observed in production."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    decision = await mgr._maybe_force_clean_restore(
        _PORT, _QWEN, _warm_slot(INC_TOOL_MSGS, INC_TOOL_CHAIN), WARM_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert decision["forced_clean_restore"] is True
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_tooltail_skip_counts == {}


# ====================================================================================
# COLD path — _restore_slot_kv
# ====================================================================================
@pytest.mark.asyncio
async def test_cold_a_tool_tail_skips_post(mgr, kv_dir, capture_restore, skip_on):
    """(a) tool tail beyond common + guard ON (emergency floor) -> restore POST SKIPPED, truthful skip decision."""
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    await mgr._restore_slot_kv(_PORT, _QWEN, _cold_slot(INC_TOOL_MSGS, INC_TOOL_CHAIN))

    assert _restores(capture_restore) == []              # nothing restored
    assert mgr._kv_tooltail_skip_counts.get("cold") == 1
    assert mgr._kv_classifier_last["resolved_from"] == "cold-tooltail-skip"
    assert mgr._kv_classifier_last["action"] == "fresh"
    assert mgr._kv_classifier_last["forced_clean_restore"] is False


@pytest.mark.asyncio
async def test_cold_b_text_tail_still_restores(mgr, kv_dir, capture_restore):
    """(b) TEXT tail -> the clean anchor is STILL restored (no regression)."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    await mgr._restore_slot_kv(_PORT, _QWEN, _cold_slot(INC_TEXT_MSGS, INC_TEXT_CHAIN))

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert "cold" not in mgr._kv_tooltail_skip_counts


@pytest.mark.asyncio
async def test_cold_b_text_tail_coldwire_shadow_intact(mgr, kv_dir, capture_restore, cold_on):
    """(b') the cold-wire `.shadow` reuse still fires on a TEXT tail under the tool-tail guard."""
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    shadow_bin = _write_shadow(kv_dir, _SID_SHADOW, SHADOW_CHAIN)     # len 6, valid, longer
    await mgr._restore_slot_kv(_PORT, _QWEN, _cold_slot(INC_TEXT_MSGS, INC_TEXT_CHAIN))

    url, body = _the_restore(capture_restore)
    assert body["filename"] == shadow_bin                # cold-wire shadow preferred
    assert f"/slots/{_SID_SHADOW}?action=restore" in url
    assert mgr._kv_shadow_restore_counts.get("cold_preferred") == 1
    assert mgr._kv_classifier_last["resolved_from"] == "wave-return-shadow-restore"
    assert "cold" not in mgr._kv_tooltail_skip_counts


@pytest.mark.asyncio
async def test_cold_b2_coldwire_shadow_tool_tail_also_skipped(mgr, kv_dir, capture_restore, cold_on, skip_on):
    """(a/#3) with the guard ON (emergency floor) a tool tail skips the POST even when the cold-wire would prefer a shadow
    target — the shadow's tool turns are equally hash-invisible, ONE guard gates both."""
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    # shadow covers [1..6] (think-free asst-N) and is a valid prefix of INC_TOOL too
    _write_shadow(kv_dir, _SID_SHADOW, _prefix_hash_chain(CLEAN_MSGS + [_TOOL_CALL]))
    # incoming: tool region beyond the clean anchor (common=5) -> tool-opaque tail
    await mgr._restore_slot_kv(_PORT, _QWEN, _cold_slot(INC_TOOL_MSGS, INC_TOOL_CHAIN))

    assert _restores(capture_restore) == []              # neither clean NOR shadow POSTed
    assert mgr._kv_tooltail_skip_counts.get("cold") == 1
    assert mgr._kv_classifier_last["resolved_from"] == "cold-tooltail-skip"


@pytest.mark.asyncio
async def test_cold_c_tool_before_common_does_not_skip(mgr, kv_dir, capture_restore):
    """(c) tool turn INSIDE the settled prefix (< common) -> restore STILL fires."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, EARLY_CLEAN_CHAIN)
    await mgr._restore_slot_kv(_PORT, _QWEN, _cold_slot(INC_EARLY_MSGS, INC_EARLY_CHAIN))

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert "cold" not in mgr._kv_tooltail_skip_counts


@pytest.mark.asyncio
async def test_cold_d_flag_off_restores_on_tool_tail(mgr, kv_dir, capture_restore, skip_off):
    """(d) flag OFF -> pre-fix behavior: the (buggy) restore POST fires on a tool tail."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    await mgr._restore_slot_kv(_PORT, _QWEN, _cold_slot(INC_TOOL_MSGS, INC_TOOL_CHAIN))

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert mgr._kv_tooltail_skip_counts == {}


@pytest.mark.asyncio
async def test_cold_e_default_restores_on_tool_tail(mgr, kv_dir, capture_restore):
    """(e) DEFAULT env (no flag set) -> THE DURABLE FIX on the COLD wave-return path: the
    restore POST fires (reuses) on a tool tail. Complements test_cold_d (explicit env=0) by
    proving the post-flip DEFAULT reuses. Would have FAILED pre-flip (default ON -> skip)."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    await mgr._restore_slot_kv(_PORT, _QWEN, _cold_slot(INC_TOOL_MSGS, INC_TOOL_CHAIN))

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert mgr._kv_tooltail_skip_counts == {}


# ====================================================================================
# COLD-FRESHNESS (Option A) composes with the tool-tail guard: a TEXT tail
# passes the guard, THEN a valid-PREFIX shadow SHORTER than the clean anchor is
# PREFERRED (freshness > length). Distinct from the tool-tail fix (the shadow root).
# ====================================================================================
@pytest.mark.asyncio
async def test_cold_freshness_shorter_shadow_preferred_on_text_tail(
        mgr, kv_dir, capture_restore, cold_on):
    # shorter shadow wins only vs a POLLUTED (with-think) anchor.
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN, clean_prefix=False)  # polluted anchor len 5
    short_shadow_chain = _prefix_hash_chain(_msgs(3))            # valid prefix, len 3 (shorter)
    shadow_bin = _write_shadow(kv_dir, _SID_SHADOW, short_shadow_chain)

    await mgr._restore_slot_kv(_PORT, _QWEN, _cold_slot(INC_TEXT_MSGS, INC_TEXT_CHAIN))

    url, body = _the_restore(capture_restore)
    assert body["filename"] == shadow_bin                        # shorter valid shadow won (polluted anchor)
    assert f"/slots/{_SID_SHADOW}?action=restore" in url
    assert mgr._kv_shadow_restore_counts.get("cold_preferred") == 1
    assert mgr._kv_shadow_restore_counts.get("cold_clean_fallback") is None
    assert "cold" not in mgr._kv_tooltail_skip_counts            # guard passed (text tail)


# ====================================================================================
# Back-compat: a slot WITHOUT client_meta['messages'] (older submit path) -> guard
# inert (inc_messages -> []), restore behaves exactly as pre-fix. Run with skip_on so
# the guard is ENABLED and we actually exercise its missing-messages inertness path (the
# guard is now default-OFF, so without skip_on the flag check would short-circuit and the
# no-messages handling would never run).
# ====================================================================================
@pytest.mark.asyncio
async def test_warm_no_messages_guard_inert(mgr, kv_dir, capture_restore, skip_on):
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    slot = SimpleNamespace(thread_id=_TID, admission_hash_chain=INC_TOOL_CHAIN)  # no client_meta
    decision = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, WARM_CHAIN)
    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_tooltail_skip_counts == {}


@pytest.mark.asyncio
async def test_cold_no_messages_guard_inert(mgr, kv_dir, capture_restore, skip_on):
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    slot = SimpleNamespace(thread_id=_TID, admission_ctx_len=50000,
                           admission_hash_chain=INC_TOOL_CHAIN)  # no client_meta
    await mgr._restore_slot_kv(_PORT, _QWEN, slot)
    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert mgr._kv_tooltail_skip_counts == {}
