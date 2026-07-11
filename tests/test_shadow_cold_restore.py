"""COLD-path (`_restore_slot_kv`) shadow restore-preference.

The COLD path is the wave-return / swap-back seam (a freshly re-spawned sidecar, NO
warm KV). It selects the winning clean/normal bin via resolve_kv (UNCHANGED), then —
gated by the DISTINCT `TURBOHAUL_SHADOW_COLD_RESTORE` (default ON when SHADOW_REPREFILL
=1, independent of the WARM `TURBOHAUL_SHADOW_RESTORE_PREFER` held 0) — upgrades the
restore TARGET to the byte-matching think-free `.shadow` bin IFF it is a valid, long-
enough prefix (else clean; never worse than today).

Scenarios:
  0. preflight: clean bin alone -> the cold path restores it (do_it path reached).
  1. GATE OFF (explicit) -> clean restored, shadow reader never consulted, byte-identical.
     ALSO proves the `.isdigit()` numeric parse SKIPS the `.shadow.bin` (scoped bypass).
  2. GATE ON + valid long-enough shadow -> the `.shadow` bin restored (strict-extends).
  3. GATE ON + STALE shadow (not a prefix) -> rejected by the SAME bar -> clean.
  4. GATE ON + SHORTER valid shadow -> COLD-FRESHNESS: PREFERRED (freshness > length).
  5. GATE ON + NO shadow -> clean.
  6. cold gate INDEPENDENT of the warm flag (warm=0 has no effect on cold).
  7. gate DEFAULT follows SHADOW_REPREFILL; explicit COLD flag overrides either way.
  8. restore POST failure -> truthful (never worse than today; no crash).

Fixtures mirror test_shadow_restore_prefer.py.
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
def _flags_off_by_default(monkeypatch):
    for f in ("TURBOHAUL_SHADOW_REPREFILL", "TURBOHAUL_SHADOW_RESTORE_PREFER",
              "TURBOHAUL_SHADOW_COLD_RESTORE"):
        monkeypatch.delenv(f, raising=False)


@pytest.fixture
def cold_on(monkeypatch):
    monkeypatch.setenv("TURBOHAUL_SHADOW_COLD_RESTORE", "1")


class _Resp:
    def __init__(self, ok=True):
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("simulated 500")

    def json(self):
        return {}


class _RestoreClient:
    def __init__(self, posts, ok=True):
        self._posts = posts
        self._ok = ok

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        return _Resp(self._ok)


@pytest.fixture
def capture_restore(monkeypatch):
    posts = []

    def _install(ok=True):
        class _FakeHttpx:
            AsyncClient = staticmethod(lambda *a, **k: _RestoreClient(posts, ok))
            Timeout = staticmethod(lambda *a, **k: None)
        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return posts

    _install()      # default OK client
    return SimpleNamespace(posts=posts, install=_install)


_MODEL = "example-model-27b"
_PORT = 59500
_TID = "agent-ip-10.0.0.5"
_SID_CLEAN = 0
_SID_SHADOW = 3

_THINK_FREE = {"role": "assistant", "content": "THE_FINAL_ANSWER"}
_USER_NP1 = {"role": "user", "content": "the next user question"}


def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


CLEAN_MSGS = _msgs(5)
SHADOW_MSGS = CLEAN_MSGS + [_THINK_FREE]
INC_MSGS = SHADOW_MSGS + [_USER_NP1]

CLEAN_CHAIN = _prefix_hash_chain(CLEAN_MSGS)       # len 5, ⊑ INC
SHADOW_CHAIN = _prefix_hash_chain(SHADOW_MSGS)     # len 6, ⊑ INC, longer than clean
INC_CHAIN = _prefix_hash_chain(INC_MSGS)           # len 7

STALE_SHADOW_CHAIN = _prefix_hash_chain(
    [{"role": "user", "content": "WRONG-turn-0"}] + _msgs(5)[1:] + [_THINK_FREE])
SHORT_SHADOW_CHAIN = _prefix_hash_chain(_msgs(3))              # len 3, ⊑ INC
LONG_CLEAN_CHAIN = _prefix_hash_chain(CLEAN_MSGS + [_THINK_FREE])  # len 6, ⊑ INC


def _slot(inc_chain=INC_CHAIN, tid=_TID):
    # cold path reads thread_id + admission_ctx_len (>0) + admission_hash_chain
    return SimpleNamespace(
        thread_id=tid,
        admission_ctx_len=50000,
        admission_hash_chain=inc_chain,
    )


def _write_clean(kv_dir, sid, chain, tid=_TID, port=_PORT, prompt_len=40000,
                 clean_prefix=True):
    # clean_prefix=True -> a think-FREE clean_prefix anchor (MOD-B keeps the cold
    # length-guard); clean_prefix=False -> a WITH-<think> polluted anchor (MOD-B drops
    # the length-guard -> any valid-prefix shadow wins regardless of length).
    th = TurbohaulManager._thread_hash(tid)
    bin_fn = kv_save_fn(_MODEL, sid, th, port)
    meta_fn = kv_meta_fn(_MODEL, sid, th, port)
    (kv_dir / bin_fn).write_bytes(b"CLEAN_ANCHOR_BYTES")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": tid, "thread_hash": th, "prompt_tokens": 500,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "prompt_hash": "", "model_tag": _MODEL, "slot_id": sid, "port": port,
        "clean_prefix": clean_prefix,
    }))
    return bin_fn


def _write_shadow(kv_dir, sid, chain, tid=_TID, port=_PORT):
    th = TurbohaulManager._thread_hash(tid)
    bin_fn = _kv_shadow_save_fn(_MODEL, sid, th, port)
    meta_fn = _kv_shadow_meta_fn(_MODEL, sid, th, port)
    (kv_dir / bin_fn).write_bytes(b"SHADOW_THINKFREE_BYTES")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": tid, "thread_hash": th, "prompt_tokens": 600,
        "prompt_len": 45000, "n_context_turns": len(chain), "hash_chain": chain,
        "prompt_hash": "", "model_tag": _MODEL, "slot_id": sid, "port": port,
        "clean_prefix": False, "shadow": True,
    }))
    return bin_fn


def _the_restore(posts):
    restores = [(u, b) for (u, b) in posts if "action=restore" in u]
    assert len(restores) == 1, f"expected exactly one restore POST, got {posts}"
    return restores[0]


# ================================================================================
# 0. preflight: clean bin alone -> cold path restores it
# ================================================================================
@pytest.mark.asyncio
async def test_preflight_cold_restores_clean(mgr, kv_dir, capture_restore):
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    await mgr._restore_slot_kv(_PORT, _MODEL, _slot())
    url, body = _the_restore(capture_restore.posts)
    assert body["filename"] == clean_bin
    assert f"/slots/{_SID_CLEAN}?action=restore" in url


# ================================================================================
# 1. GATE OFF (explicit) -> clean restored; shadow reader never consulted; the
#    `.shadow.bin` is INVISIBLE to the numeric `.isdigit()` parse (scoped bypass).
# ================================================================================
@pytest.mark.asyncio
async def test_gate_off_restores_clean_and_shadow_isdigit_skipped(mgr, kv_dir, capture_restore, monkeypatch):
    monkeypatch.setenv("TURBOHAUL_SHADOW_COLD_RESTORE", "0")   # explicit OFF
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    _write_shadow(kv_dir, _SID_SHADOW, SHADOW_CHAIN)           # present but ignored

    await mgr._restore_slot_kv(_PORT, _MODEL, _slot())

    url, body = _the_restore(capture_restore.posts)
    assert body["filename"] == clean_bin                       # clean, NOT the shadow
    assert not body["filename"].endswith(".shadow.bin")        # numeric parse skipped it
    assert f"/slots/{_SID_CLEAN}?action=restore" in url
    assert mgr._kv_shadow_restore_counts == {}                 # reader never ran


# ================================================================================
# 2. GATE ON + valid long-enough shadow -> SHADOW restored
# ================================================================================
@pytest.mark.asyncio
async def test_gate_on_prefers_valid_shadow(mgr, kv_dir, capture_restore, cold_on):
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    shadow_bin = _write_shadow(kv_dir, _SID_SHADOW, SHADOW_CHAIN)

    await mgr._restore_slot_kv(_PORT, _MODEL, _slot())

    url, body = _the_restore(capture_restore.posts)
    assert body["filename"] == shadow_bin                      # the `.shadow` bin won
    assert f"/slots/{_SID_SHADOW}?action=restore" in url       # its distinct slot
    assert mgr._kv_shadow_restore_counts.get("cold_preferred") == 1
    assert mgr._kv_shadow_restore_counts.get("cold_clean_fallback") is None
    # wave-return counter bumps once (shadow cold-restore IS a wave-return)
    assert mgr._kv_classifier_wave_return == 1
    assert mgr._kv_classifier_last["resolved_from"] == "wave-return-shadow-restore"
    assert mgr._kv_classifier_last["clean_bin_id"] == shadow_bin


# ================================================================================
# 3. GATE ON + STALE shadow (not a prefix) -> clean
# ================================================================================
@pytest.mark.asyncio
async def test_gate_on_stale_shadow_falls_back(mgr, kv_dir, capture_restore, cold_on):
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    _write_shadow(kv_dir, _SID_SHADOW, STALE_SHADOW_CHAIN)     # NOT a prefix of INC

    await mgr._restore_slot_kv(_PORT, _MODEL, _slot())

    url, body = _the_restore(capture_restore.posts)
    assert body["filename"] == clean_bin
    assert f"/slots/{_SID_CLEAN}?action=restore" in url
    assert mgr._kv_shadow_restore_counts.get("cold_preferred") is None
    assert mgr._kv_shadow_restore_counts.get("cold_clean_fallback") == 1
    assert mgr._kv_classifier_last["resolved_from"] == "wave-return-clean-restore"


# ================================================================================
# 4. GATE ON + SHORTER valid shadow -> COLD-FRESHNESS: PREFERRED (freshness > length)
# ================================================================================
@pytest.mark.asyncio
async def test_gate_on_shorter_valid_shadow_now_preferred(mgr, kv_dir, capture_restore, cold_on):
    # COLD-FRESHNESS (Option A) + MOD-B: a valid-PREFIX think-free
    # shadow that is SHORTER than the anchor STILL WINS on cold WHEN the anchor is a
    # WITH-<think> POLLUTED bin (clean_prefix=False) — the polluted clean risks a full
    # ~81k CLEAR, so any valid shadow beats it. (No client_meta -> crit3 guard inert.)
    _write_clean(kv_dir, _SID_CLEAN, LONG_CLEAN_CHAIN, clean_prefix=False)  # len 6, POLLUTED
    shadow_bin = _write_shadow(kv_dir, _SID_SHADOW, SHORT_SHADOW_CHAIN)  # len 3, valid, SHORTER

    await mgr._restore_slot_kv(_PORT, _MODEL, _slot())

    url, body = _the_restore(capture_restore.posts)
    assert body["filename"] == shadow_bin                      # shorter valid shadow wins (polluted anchor)
    assert f"/slots/{_SID_SHADOW}?action=restore" in url
    assert mgr._kv_shadow_restore_counts.get("cold_preferred") == 1
    assert mgr._kv_shadow_restore_counts.get("cold_clean_fallback") is None


@pytest.mark.asyncio
async def test_modB_clean_prefix_longer_anchor_kept_over_shorter_shadow(mgr, kv_dir, capture_restore, cold_on):
    # MOD-B regression-prevention: when the winning anchor is a THINK-FREE
    # clean_prefix bin that is LONGER than the shadow, KEEP the clean (it strict-extends);
    # a shorter shadow would reprefill the gap for nothing. So the shadow is NOT preferred.
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, LONG_CLEAN_CHAIN, clean_prefix=True)  # len 6, think-free
    _write_shadow(kv_dir, _SID_SHADOW, SHORT_SHADOW_CHAIN)                             # len 3, valid, SHORTER

    await mgr._restore_slot_kv(_PORT, _MODEL, _slot())

    url, body = _the_restore(capture_restore.posts)
    assert body["filename"] == clean_bin                       # longer think-free clean kept
    assert mgr._kv_shadow_restore_counts.get("cold_preferred") is None


# ================================================================================
# 5. GATE ON + NO shadow -> clean
# ================================================================================
@pytest.mark.asyncio
async def test_gate_on_no_shadow_uses_clean(mgr, kv_dir, capture_restore, cold_on):
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)

    await mgr._restore_slot_kv(_PORT, _MODEL, _slot())

    url, body = _the_restore(capture_restore.posts)
    assert body["filename"] == clean_bin
    assert mgr._kv_shadow_restore_counts.get("cold_clean_fallback") == 1


# ================================================================================
# 6. cold gate INDEPENDENT of the WARM flag: warm=0 (its held value) has NO effect
#    on the cold path; the cold gate alone decides.
# ================================================================================
@pytest.mark.asyncio
async def test_cold_independent_of_warm_prefer_flag(mgr, kv_dir, capture_restore, monkeypatch):
    monkeypatch.setenv("TURBOHAUL_SHADOW_RESTORE_PREFER", "0")   # WARM held OFF
    monkeypatch.setenv("TURBOHAUL_SHADOW_COLD_RESTORE", "1")     # COLD ON
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    shadow_bin = _write_shadow(kv_dir, _SID_SHADOW, SHADOW_CHAIN)

    await mgr._restore_slot_kv(_PORT, _MODEL, _slot())

    _url, body = _the_restore(capture_restore.posts)
    assert body["filename"] == shadow_bin                        # cold prefers, warm=0 irrelevant
    # and the inverse: WARM ON but COLD OFF -> cold path does NOT prefer
    mgr._kv_shadow_restore_counts.clear()
    capture_restore.posts.clear()
    monkeypatch.setenv("TURBOHAUL_SHADOW_RESTORE_PREFER", "1")   # WARM ON
    monkeypatch.setenv("TURBOHAUL_SHADOW_COLD_RESTORE", "0")     # COLD OFF
    clean_bin = kv_save_fn(_MODEL, _SID_CLEAN, TurbohaulManager._thread_hash(_TID), _PORT)
    await mgr._restore_slot_kv(_PORT, _MODEL, _slot())
    _url2, body2 = _the_restore(capture_restore.posts)
    assert body2["filename"] == clean_bin                        # warm flag does NOT arm cold
    assert mgr._kv_shadow_restore_counts == {}


# ================================================================================
# 7. gate DEFAULT follows SHADOW_REPREFILL; explicit COLD overrides either way
# ================================================================================
def test_cold_gate_reader_default_and_override(monkeypatch):
    # default: unset -> follows SHADOW_REPREFILL
    monkeypatch.delenv("TURBOHAUL_SHADOW_COLD_RESTORE", raising=False)
    monkeypatch.delenv("TURBOHAUL_SHADOW_REPREFILL", raising=False)
    assert manager_mod._shadow_cold_restore_enabled() is False
    monkeypatch.setenv("TURBOHAUL_SHADOW_REPREFILL", "1")
    assert manager_mod._shadow_cold_restore_enabled() is True         # ON when save on
    # explicit OFF overrides SHADOW_REPREFILL=1
    monkeypatch.setenv("TURBOHAUL_SHADOW_COLD_RESTORE", "0")
    assert manager_mod._shadow_cold_restore_enabled() is False
    # explicit ON overrides SHADOW_REPREFILL unset
    monkeypatch.delenv("TURBOHAUL_SHADOW_REPREFILL", raising=False)
    monkeypatch.setenv("TURBOHAUL_SHADOW_COLD_RESTORE", "1")
    assert manager_mod._shadow_cold_restore_enabled() is True


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("YES", True), ("On", True),
    ("0", False), ("false", False), ("off", False),
])
def test_cold_gate_reader_truthy(monkeypatch, val, expected):
    monkeypatch.delenv("TURBOHAUL_SHADOW_REPREFILL", raising=False)
    monkeypatch.setenv("TURBOHAUL_SHADOW_COLD_RESTORE", val)
    assert manager_mod._shadow_cold_restore_enabled() is expected


# ================================================================================
# 8. never worse than today: a 500'd restore POST is truthful (no crash, no false
#    wave-return count) whether it targeted the shadow or the clean bin.
# ================================================================================
@pytest.mark.asyncio
async def test_restore_post_failure_is_truthful(mgr, kv_dir, capture_restore, cold_on):
    capture_restore.install(ok=False)          # engine 500s the restore
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    _write_shadow(kv_dir, _SID_SHADOW, SHADOW_CHAIN)

    await mgr._restore_slot_kv(_PORT, _MODEL, _slot())   # must not raise

    # the preference still ran (shadow chosen) but the POST failed -> truthful:
    assert mgr._kv_shadow_restore_counts.get("cold_preferred") == 1
    assert mgr._kv_classifier_wave_return == 0                     # NOT counted (post failed)
    assert mgr._kv_classifier_last["resolved_from"] == "restore-post-failed"
    assert mgr._kv_classifier_last["event_type"] == "guard-skip"


# ================================================================================
# 9. SWAP-BACK MEASUREMENT design (manager-side assertion of the RIGHT target).
#    The manager's job is to restore the byte-matching think-free [1..N] shadow; the
#    ACTUAL prompt_eval (SUCCESS = sub-tail only vs recurrent-limit = full CLEAR) is the
#    ENGINE's call on the next decode. This asserts the restored FILENAME is the shadow
#    (turns 1..N+1 here) — i.e. the main history is offered for strict-extension so only
#    the appended tail can need reprefill. See the report for the engine-log signatures
#    ("strict extension" vs "CLEAR") that distinguish (a) success from (b) recurrent-limit.
# ================================================================================
@pytest.mark.asyncio
async def test_swapback_restores_thinkfree_shadow_for_strict_extension(mgr, kv_dir, capture_restore, cold_on):
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)          # short clean anchor [1..N-1]
    shadow_bin = _write_shadow(kv_dir, _SID_SHADOW, SHADOW_CHAIN)   # full think-free [1..N]

    await mgr._restore_slot_kv(_PORT, _MODEL, _slot(inc_chain=INC_CHAIN))

    _url, body = _the_restore(capture_restore.posts)
    # the engine is handed the LONGEST valid think-free prefix -> the most main history
    # is eligible for strict-extension; only the incoming tail past turn-N can reprefill.
    assert body["filename"] == shadow_bin
    common = mgr._kv_classifier_last["common_prefix_turns"]
    assert common == len(SHADOW_CHAIN)                     # full [1..N] common with incoming
    assert common > len(CLEAN_CHAIN)                       # strictly more than the clean anchor
