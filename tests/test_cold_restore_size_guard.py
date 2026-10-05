"""Backstop to the root fix: the cold-restore SIZE decline
guard.

Two tiers, a pure decision table and a wiring test that drives the real call
site, so a table that is right but wired up wrongly cannot pass unnoticed
(two separate proofs):
  * test_classify_cold_restore_size_* — the pure decision table in isolation.
    Proves the table is right; proves nothing about whether the call site wires
    it up correctly (routing blind spot, a known pitfall of pure decision-table tests).
  * test_cold_restore_size_guard_* — drives the real production method
    (_restore_slot_kv / _restore_slot_kv_inner) end to end with a fake httpx
    client, so a reorder or a dropped call in the wiring shows up as a real
    failure, not a green pure-function suite hiding a broken caller.

Causal in both directions: an oversized-mismatch bin is
DECLINED, and a genuinely valid cold restore is still ACCEPTED (the one that
earns its keep — a guard that always declines would pass every decline test
while destroying the cold path). Also covers the polluted-bin case directly:
a WITH-THINK (polluted) bin, which legitimately carries more raw tokens than a
stripped incoming resend, must NOT be declined by this guard — that is the
ordinary healthy flow most likely to resemble the bad one, and it is proven
here, not just asserted in a comment.
"""
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
from turbohaul.kv_policy import _prefix_hash_chain, kv_meta_fn
from turbohaul.manager import (
    TurbohaulManager,
    _classify_cold_restore_size,
    _COLD_RESTORE_SIZE_MARGIN_TOKENS,
)
from turbohaul.slot import Slot


# --- fixtures (mirror test_wave_return.py) ----------------------------------------
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


_SYS = {"role": "system", "content": "system prompt long enough to matter"}
_U1 = {"role": "user", "content": "first user turn"}
_U2 = {"role": "user", "content": "second user turn"}
_A1 = {"role": "assistant", "content": "A"}
_QWEN = "qwen3.6-27b"


def _write_bin(kv_dir, model_tag, port, thread_id, sid, chain, *, clean,
               prompt_len=40000, prompt_tokens=12345):
    """Verbatim copy of test_wave_return.py's helper (same repo convention)."""
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": prompt_tokens,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": clean,
    }))
    return bin_fn


# --- fake httpx: apply-template + tokenize (this guard) + action=restore (existing) --
class _SizeGuardResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _SizeGuardClient:
    """Fakes the freshly-spawned engine for the cold-restore size guard:
    POST /apply-template -> a non-empty prompt string (content irrelevant, the
    guard only cares about its length via /tokenize); POST /tokenize -> a
    token list of exactly `incoming_tokens` length; POST action=restore ->
    recorded + ok. `render_fails`/`tokenize_fails` simulate a probe failure at
    each stage independently, for the fail-open tests."""

    def __init__(self, incoming_tokens, posts, *, render_fails=False, tokenize_fails=False):
        self._incoming_tokens = incoming_tokens
        self._posts = posts
        self._render_fails = render_fails
        self._tokenize_fails = tokenize_fails

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **kw):
        if "/apply-template" in url:
            if self._render_fails:
                raise ConnectionError("simulated apply-template failure")
            return _SizeGuardResp({"prompt": "x" * 64})
        if "/tokenize" in url:
            if self._tokenize_fails:
                raise ConnectionError("simulated tokenize failure")
            return _SizeGuardResp({"tokens": list(range(self._incoming_tokens))})
        # action=restore (or anything else) -- record + succeed
        self._posts.append((url, json))
        return _SizeGuardResp({"status": "ok"})


@pytest.fixture
def make_size_guard_httpx(monkeypatch):
    def _make(incoming_tokens, **kw):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(
                lambda *a, **k: _SizeGuardClient(incoming_tokens, posts, **kw))
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return posts

    return _make


def _restore_posts(posts):
    return [(u, b) for (u, b) in posts if "action=restore" in u]


# ==============================================================================
# TIER 1 — pure decision table (_classify_cold_restore_size)
# ==============================================================================
def test_classify_polluted_bin_always_proceeds():
    """THE POLLUTED-BIN CASE, proven directly: a WITH-THINK (polluted) bin is
    OUT OF SCOPE regardless of how lopsided the token counts are -- this must
    hold even when bin_tokens is grotesquely larger than incoming_tokens,
    because that excess is the EXPECTED reasoning-tag content, not an anomaly."""
    proceed, case = _classify_cold_restore_size("clean", False, 50000, 100, 32)
    assert proceed is True
    assert case == "polluted_out_of_scope"


def test_classify_unknown_bin_tokens_fails_open():
    proceed, case = _classify_cold_restore_size("clean", True, None, 500, 32)
    assert proceed is True
    assert case == "unknown_fail_open"


def test_classify_unknown_incoming_tokens_fails_open():
    proceed, case = _classify_cold_restore_size("shadow", True, 500, None, 32)
    assert proceed is True
    assert case == "unknown_fail_open"


def test_classify_oversized_clean_bin_declines():
    proceed, case = _classify_cold_restore_size("clean", True, 50000, 500, 32)
    assert proceed is False
    assert case == "size_decline"


def test_classify_oversized_shadow_bin_declines():
    """Shadow bins ARE in scope (r_kind == 'shadow' skips the polluted check
    entirely -- shadow bins are think-free by construction)."""
    proceed, case = _classify_cold_restore_size("shadow", True, 50000, 500, 32)
    assert proceed is False
    assert case == "size_decline"


def test_classify_valid_extension_proceeds():
    """THE ONE THAT EARNS ITS KEEP: incoming EXTENDS the bin (more tokens, not
    fewer) -- ordinary strict-extension continuation, must proceed."""
    proceed, case = _classify_cold_restore_size("clean", True, 500, 600, 32)
    assert proceed is True
    assert case == "size_ok"


def test_classify_margin_boundary_exact_proceeds():
    """bin_tokens == incoming_tokens + margin exactly is NOT '> margin' --
    must still proceed (off-by-one check on the comparison operator)."""
    proceed, case = _classify_cold_restore_size("clean", True, 532, 500, 32)
    assert proceed is True
    assert case == "size_ok"


def test_classify_margin_boundary_one_over_declines():
    proceed, case = _classify_cold_restore_size("clean", True, 533, 500, 32)
    assert proceed is False
    assert case == "size_decline"


# ==============================================================================
# TIER 2 — wiring: drives _restore_slot_kv_inner end to end
# ==============================================================================
@pytest.mark.asyncio
async def test_size_guard_declines_oversized_clean_bin(mgr, kv_dir, make_size_guard_httpx):
    """A clean bin with far more tokens than the incoming request renders to
    is DECLINED -- the restore POST never fires."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain, clean=True,
               prompt_tokens=50000)
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])
    slot = Slot.new(_QWEN, thread_id="t", admission_ctx_len=50000,
                    admission_hash_chain=inc, client_meta={"messages": [_SYS, _U1, _A1, _U2]})
    posts = make_size_guard_httpx(500)  # incoming renders to only 500 tokens

    await mgr._restore_slot_kv(59500, _QWEN, slot)

    assert _restore_posts(posts) == [], "restore POSTed despite the bin being oversized"
    assert mgr._kv_classifier_last["resolved_from"] == "cold-size-decline"
    assert mgr._kv_classifier_last["action"] == "fresh"
    assert mgr._kv_cold_size_decline_count == 1


@pytest.mark.asyncio
async def test_size_guard_accepts_valid_extension(mgr, kv_dir, make_size_guard_httpx):
    """THE ONE THAT EARNS ITS KEEP, driven end to end: a genuinely valid cold
    restore (bin is a real prefix, incoming EXTENDS it) still fires the
    restore POST -- a guard that always declines would pass the test above
    trivially while destroying every legitimate cold restore."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain, clean=True,
                        prompt_tokens=500)
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])
    slot = Slot.new(_QWEN, thread_id="t", admission_ctx_len=50000,
                    admission_hash_chain=inc, client_meta={"messages": [_SYS, _U1, _A1, _U2]})
    posts = make_size_guard_httpx(600)  # incoming renders to MORE than the bin -- extension

    await mgr._restore_slot_kv(59500, _QWEN, slot)

    assert len(_restore_posts(posts)) == 1
    url, body = _restore_posts(posts)[0]
    assert body == {"filename": bin_fn}
    assert mgr._kv_classifier_last["resolved_from"] == "wave-return-clean-restore"


@pytest.mark.asyncio
async def test_size_guard_does_not_reject_polluted_bin_ordinary_flow(
        mgr, kv_dir, make_size_guard_httpx):
    """THE POLLUTED-BIN CASE, driven end to end: a WITH-THINK (polluted) bin
    carrying far more tokens than a stripped incoming resend is the ORDINARY,
    designed-for fallback restore -- not the rare bad case. It must NOT be
    declined by this guard, exactly the shape of check that would break that flow."""
    saved_chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_bin(kv_dir, _QWEN, 59500, "t", 0, saved_chain, clean=False,
                        prompt_tokens=50000)  # polluted: huge <think> tail included
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])
    slot = Slot.new(_QWEN, thread_id="t", admission_ctx_len=50000,
                    admission_hash_chain=inc, client_meta={"messages": [_SYS, _U1, _A1, _U2]})
    # incoming's stripped resend renders to far FEWER tokens than the polluted
    # bin's raw count -- exactly the expected, harmless think-strip delta.
    posts = make_size_guard_httpx(500)

    await mgr._restore_slot_kv(59500, _QWEN, slot)

    assert len(_restore_posts(posts)) == 1, (
        "the ordinary with-think fallback restore was declined -- this is the "
        "polluted-bin mistake reproduced"
    )
    url, body = _restore_posts(posts)[0]
    assert body == {"filename": bin_fn}
    assert mgr._kv_classifier_last["resolved_from"] == "restore-prefix-valid"


@pytest.mark.asyncio
async def test_size_guard_fails_open_when_incoming_probe_fails(
        mgr, kv_dir, make_size_guard_httpx):
    """The apply-template probe fails (timeout/connection error) -> incoming
    token count is unknown -> FAIL OPEN, restore proceeds exactly as if this
    guard did not exist, even though the bin IS oversized."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain, clean=True,
                        prompt_tokens=50000)
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])
    slot = Slot.new(_QWEN, thread_id="t", admission_ctx_len=50000,
                    admission_hash_chain=inc, client_meta={"messages": [_SYS, _U1, _A1, _U2]})
    posts = make_size_guard_httpx(500, render_fails=True)

    await mgr._restore_slot_kv(59500, _QWEN, slot)

    assert len(_restore_posts(posts)) == 1, "an unknown incoming-token probe declined the restore"
    assert mgr._kv_classifier_last["resolved_from"] == "wave-return-clean-restore"


@pytest.mark.asyncio
async def test_size_guard_env_kill_switch_disables_it(
        mgr, kv_dir, make_size_guard_httpx, monkeypatch):
    """TURBOHAUL_COLD_RESTORE_SIZE_GUARD=0 returns the cold path to exactly
    its unguarded behavior -- an oversized bin restores unconditionally."""
    monkeypatch.setenv("TURBOHAUL_COLD_RESTORE_SIZE_GUARD", "0")
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain, clean=True,
                        prompt_tokens=50000)
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])
    slot = Slot.new(_QWEN, thread_id="t", admission_ctx_len=50000,
                    admission_hash_chain=inc, client_meta={"messages": [_SYS, _U1, _A1, _U2]})
    posts = make_size_guard_httpx(500)

    await mgr._restore_slot_kv(59500, _QWEN, slot)

    assert len(_restore_posts(posts)) == 1, "kill-switch did not disable the guard"
