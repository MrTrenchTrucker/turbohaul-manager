"""ERASE-ORDER regression -- the dirty-tip seam must run the cheap
save-guards BEFORE the destructive slot erase.

OBSERVED FAILURE SHAPE:

    dirty-tip ERASED (long chain)
      -> KV_SAVE_DECLINE NEVER_OVERWRITE_SMALLER, SAME MILLISECOND
    dirty-tip ERASED (long chain)
      -> same decline, SAME MILLISECOND
    dirty-tip ERASED (long chain)
      -> same decline, SAME MILLISECOND
    dirty-tip ERASED (long chain)
      -> same decline, SAME MILLISECOND

A long-chain reprefill takes MINUTES. Zero to one millisecond means the reprefill
NEVER RAN. The erase wiped main's sequence, then the never-overwrite-with-smaller
belt returned without rebuilding anything.

THE BUG: in _probe_and_save_clean_kv the dirty-tip ERASE block ran BEFORE the
three cheap guards that abort the operation:

    G1: inc_turns <= 0            -> return  (ZERO_TURN_CHAIN)
    G2: existing bin >= incoming  -> return  (NEVER_OVERWRITE_SMALLER)
    G3: zero-growth throttle      -> return  (ZERO_GROWTH_THROTTLE)

All three returns sat BETWEEN the erase and the strip-probe reprefill, so a
fired guard left main's sequence wiped with nothing rebuilding it. The guard
chain between the old erase position and the reprefill is exactly these three
returns; three MORE returns sit after the reprefill attempt (STRIP_PROBE_FAILED /
PROBE_POST_FAILED / SAVE_NOT_CONFIRMED) -- those are PRE-EXISTING and out of
scope here (the erase already ran on them, same as unmodified main).

THE FIX: run the cheap guards first; the erase fires ONLY when we are going to
ATTEMPT the reprefill and save. The erase still precedes the probe (bin == chain
by construction is untouched), and a guard-bail leaves the dirty flag SET so
every save routed through _save_slot_kv_inner is still refused on that port.

COVERAGE MATRIX -- every test asserts the EMITTED DECLINE REASON (never just
the absence of an erase POST), so an unrelated EARLY EXIT cannot satisfy the
"no erase" expectation for the wrong reason (e.g. _clean_prefix_save_enabled
off, a sub-min-ctx early return, or _prefix_hash_chain raising into the broad
except -- all of which would emit a different reason or UNVERIFIED_EXIT):

  * G2 negative: dirty tip + existing bin prompt_len >= incoming MUST NOT erase;
    asserts reason NEVER_OVERWRITE_SMALLER, UNVERIFIED_EXIT absent.
  * G3 negative: dirty tip + zero-growth throttle MUST NOT erase; asserts reason
    ZERO_GROWTH_THROTTLE, UNVERIFIED_EXIT absent.
  * G1 negative: dirty tip + degenerate 0-turn chain MUST NOT erase; asserts
    reason ZERO_TURN_CHAIN, UNVERIFIED_EXIT absent.
  * POSITIVE: dirty tip + ALL guards pass -> the erase DOES fire (guards must
    never over-suppress remediation). Without this, a mutation that deletes /
    reorders the ENTIRE 39-line erase block makes every negative pass vacuously.
  * FLAG PERSISTENCE: after a guard-bail the dirty flag STAYS SET; sub-min-ctx
    per-turn traffic (below the 40k min-ctx early return, where the per-turn clear
    lives) does NOT clear it, so the _save_slot_kv_inner chokepoint refuses saves
    (incl. an idle-teardown save) until a >= 40k main-class turn reaches the clear.
    Establishes by reproduction that the refusal is the
    chokepoint doing its job; the refusal window is bounded by any >= 40k main turn.

FAILS on unmodified main (an action=erase POST is recorded before the guard
fires). PASSES on the fix (the guard fires first; no erase call is made).
"""

import json
import time
import types

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


# Decay: the decline-reason stream.
KV_DECLINE_ZERO_TURN_CHAIN = manager_mod.KV_DECLINE_ZERO_TURN_CHAIN
KV_DECLINE_NEVER_OVERWRITE_SMALLER = manager_mod.KV_DECLINE_NEVER_OVERWRITE_SMALLER
KV_DECLINE_ZERO_GROWTH_THROTTLE = manager_mod.KV_DECLINE_ZERO_GROWTH_THROTTLE
KV_DECLINE_UNVERIFIED_EXIT = manager_mod.KV_DECLINE_UNVERIFIED_EXIT


# --- fixtures (mirror test_lagreducer.py / test_classifier.py) ------------
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
    """Isolated SLOT_SAVE_DIR (re-imported per call inside the manager methods)."""
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture(autouse=True)
def _covered_scaffold_strip_off(mgr):
    """Keep the clean-probe transport on the plain messages path (the guard
    ordering under test is transport-independent; the strip transport is
    covered by tests/test_covered_scaffold_strip.py)."""
    mgr.runtime.kv.covered_scaffold_strip = False


@pytest.fixture
def decline_reasons(monkeypatch):
    """Capture every KV_SAVE_DECLINE reason emitted through the one shared
    helper. The fixture wraps `_log_kv_save_decline` (module-global, called
    bare inside the manager methods), records the reason_name, and passes
    through. Tests assert the SPECIFIC reason -- so an unrelated early exit
    (which would emit a different reason, or nothing -> UNVERIFIED_EXIT via
    the finally backstop) can never satisfy a "no erase" expectation."""
    reasons = []
    _orig = manager_mod._log_kv_save_decline

    def _wrap(reason_name, **kw):
        reasons.append(reason_name)
        return _orig(reason_name, **kw)

    monkeypatch.setattr(manager_mod, "_log_kv_save_decline", _wrap)
    return reasons


class _SaveResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _ProbeSaveClient:
    """Fakes the sidecar for _probe_and_save_clean_kv: GET /slots returns the
    configured populated slots; every POST is recorded so the test can assert
    whether an action=erase ever fired.

    action=save materializes a real bin file under
    SLOT_SAVE_DIR (mirrors the idle-park tip-freshness test's
    convention), so _save_slot_kv_inner's os.replace() actually
    succeeds instead of raising FileNotFoundError on every call --
    otherwise these tests would silently exercise SAVE_NOT_CONFIRMED (the
    save is never genuinely confirmed) rather than the confirmed-save path
    their assertions/docstrings describe. Not part of the erase-order
    behaviour: it only keeps the fake sidecar faithful on a confirmed
    save."""

    def __init__(self, slots_payload, posts):
        self._slots_payload = slots_payload
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if "/slots" in url and "action=save" not in url:
            return _SaveResp(self._slots_payload)
        return _SaveResp({})

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        if "action=save" in url and json and "filename" in json:
            import os
            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            tmp_path = os.path.join(SLOT_SAVE_DIR, json["filename"])
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(b"dummy kv cache data")
        if "/v1/chat/completions" in url:
            # Plain-probe regime (covered_scaffold_strip off): the real
            # sidecar's /v1/chat/completions reply carries the tokenized
            # prompt's usage. The manager's strict clean stamp
            # keys off usage.prompt_tokens as its save evidence, so the fake
            # reports the slot's own engine-reported count — exactly what the
            # engine would return for the same render.
            _n = (self._slots_payload[0].get("n_prompt_tokens") or 1) if self._slots_payload else 1
            return _SaveResp({"status": "ok",
                               "usage": {"prompt_tokens": _n,
                                         "completion_tokens": 0,
                                         "total_tokens": _n}})
        return _SaveResp({"status": "ok"})


@pytest.fixture
def make_httpx(monkeypatch):
    """Patch manager_mod.httpx with a fake; returns the list recording POSTs."""
    def _make(payload):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(lambda *a, **k: _ProbeSaveClient(payload, posts))
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return posts

    return _make


# --- helpers (mirror test_lagreducer.py) --------------------------------------
# Generic model label only -- the dirty-tip seam path this test drives performs
# NO family check (family predicates live on the warm-force path, untested
# here), so any tag consistent between _write_clean_bin and the shim works.
_MODEL_TAG = "test-model"
_PORT = 59500
_PID = 4242


def _write_clean_bin(kv_dir, model_tag, port, thread_id, sid, chain, prompt_len=40000):
    """Write a pinned clean bin (.bin + clean_prefix .json meta) for a thread."""
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")  # non-empty bin so existence check passes
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": 12345,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": True,
    }))
    return meta_fn


def _msgs(k):
    """k structured turns -> _prefix_hash_chain(...) is length k (1 hash per turn)."""
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


def _handle(port=_PORT):
    return types.SimpleNamespace(parallel=1, port=port, pid=_PID)


def _dirty_flag(port=_PORT, tid="t"):
    """A curator-role dirty-tip record for this engine pid (passes the manager's validity checks)."""
    return {port: {"role": "curator", "pid": _PID, "thread": tid,
                   "ts": time.time()}}


def _shim(tid="t", messages=None, inc_len=50000):
    """SimpleNamespace shim mirroring _flush_clean_kv_at_unload's shim."""
    return types.SimpleNamespace(
        thread_id=tid,
        model_tag=_MODEL_TAG,
        admission_ctx_len=inc_len,
        client_meta={"messages": messages or _msgs(3)},
        context=None,
        prompt="",
        port=_PORT,
        pid=_PID,
    )


def _erase_posts(posts):
    return [u for (u, _j) in posts if "action=erase" in u]


# ==============================================================================
# G2 NEGATIVE -- never-overwrite-with-smaller MUST abort before the erase
# ==============================================================================
@pytest.mark.asyncio
async def test_dirty_tip_no_erase_when_never_overwrite_would_decline(
        mgr, kv_dir, make_httpx, decline_reasons):
    """Dirty tip + an existing clean bin with prompt_len >= incoming: the
    never-overwrite-with-smaller belt MUST abort BEFORE the slot erase, and
    MUST emit its specific decline reason (NEVER_OVERWRITE_SMALLER)."""
    chain = _prefix_hash_chain(_msgs(3))
    # existing clean bin with prompt_len >= incoming (50000) -> belt fires
    _write_clean_bin(kv_dir, _MODEL_TAG, _PORT, "t", 0, chain, prompt_len=60000)
    # engine reports ONE populated slot so an erase attempt would succeed
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

    mgr._kv_dirty_tail = _dirty_flag()
    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert _erase_posts(posts) == [], (
        "dirty-tip ERASE fired even though never-overwrite-with-smaller would "
        f"decline the save (erase POSTs: {_erase_posts(posts)}) -- sequence "
        "wiped, nothing rebuilt"
    )
    assert KV_DECLINE_NEVER_OVERWRITE_SMALLER in decline_reasons, (
        "G2 decline reason NEVER_OVERWRITE_SMALLER not emitted; an unrelated "
        f"early exit satisfied the no-erase expectation for the wrong reason "
        f"(emitted: {decline_reasons})"
    )
    assert KV_DECLINE_UNVERIFIED_EXIT not in decline_reasons, (
        f"UNVERIFIED_EXIT backstop fired ({decline_reasons}) -- the flow exited "
        "without a primary verdict; the no-erase assertion was satisfied for the "
        "wrong reason"
    )


# ==============================================================================
# G3 NEGATIVE -- zero-growth throttle MUST abort before the erase
# ==============================================================================
@pytest.mark.asyncio
async def test_dirty_tip_no_erase_when_zero_growth_throttle_would_decline(
        mgr, kv_dir, make_httpx, decline_reasons):
    """Dirty tip + an existing clean bin with prompt_len < incoming but equal
    turn count (growth 0 < min 1 at the seam): the zero-growth throttle MUST
    abort BEFORE the slot erase, emitting ZERO_GROWTH_THROTTLE."""
    chain = _prefix_hash_chain(_msgs(3))
    # prompt_len below incoming (belt passes) but n_context_turns == incoming
    # turns -> growth 0 -> throttle fires at save_to_disk (min_growth=1)
    _write_clean_bin(kv_dir, _MODEL_TAG, _PORT, "t", 0, chain, prompt_len=45000)
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

    mgr._kv_dirty_tail = _dirty_flag()
    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert _erase_posts(posts) == [], (
        "dirty-tip ERASE fired even though zero-growth throttle would decline "
        f"the save (erase POSTs: {_erase_posts(posts)}) -- sequence wiped, "
        "nothing rebuilt"
    )
    assert KV_DECLINE_ZERO_GROWTH_THROTTLE in decline_reasons, (
        "G3 decline reason ZERO_GROWTH_THROTTLE not emitted; an unrelated early "
        f"exit satisfied the no-erase expectation for the wrong reason "
        f"(emitted: {decline_reasons})"
    )
    assert KV_DECLINE_UNVERIFIED_EXIT not in decline_reasons, (
        f"UNVERIFIED_EXIT backstop fired ({decline_reasons}) -- the flow exited "
        "without a primary verdict; the no-erase assertion was satisfied for the "
        "wrong reason"
    )


# ==============================================================================
# G1 NEGATIVE -- zero-turn chain MUST abort before the erase
# ==============================================================================
@pytest.mark.asyncio
async def test_dirty_tip_no_erase_when_zero_turn_chain_declines(
        mgr, kv_dir, make_httpx, decline_reasons, monkeypatch):
    """Dirty tip + a degenerate 0-turn chain (a defensive belt for a truthy-but-
    empty messages edge): G1 MUST abort BEFORE the slot erase, emitting
    ZERO_TURN_CHAIN. The real _prefix_hash_chain never yields an empty chain for
    a non-empty messages list, so it is stubbed to [] to reach the belt."""
    monkeypatch.setattr(manager_mod, "_prefix_hash_chain", lambda ctx: [])
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

    mgr._kv_dirty_tail = _dirty_flag()
    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert _erase_posts(posts) == [], (
        "dirty-tip ERASE fired even though the zero-turn-chain belt would "
        f"decline the save (erase POSTs: {_erase_posts(posts)}) -- sequence "
        "wiped, nothing rebuilt"
    )
    assert KV_DECLINE_ZERO_TURN_CHAIN in decline_reasons, (
        "G1 decline reason ZERO_TURN_CHAIN not emitted; an unrelated early exit "
        f"satisfied the no-erase expectation for the wrong reason "
        f"(emitted: {decline_reasons})"
    )
    assert KV_DECLINE_UNVERIFIED_EXIT not in decline_reasons, (
        f"UNVERIFIED_EXIT backstop fired ({decline_reasons}) -- the flow exited "
        "without a primary verdict; the no-erase assertion was satisfied for the "
        "wrong reason"
    )


# ==============================================================================
# POSITIVE -- when ALL guards pass, the erase DOES fire
# ==============================================================================
@pytest.mark.asyncio
async def test_dirty_tip_erase_fires_when_all_guards_pass(
        mgr, kv_dir, make_httpx, decline_reasons):
    """Dirty tip + NO existing clean bin (G2 belt silent) + growth throttle NOT
    engaged (no saved bound) + a non-empty 3-turn chain (G1 silent): the erase
    MUST fire. Guards against the mirror-image regression the negative tests
    alone cannot see: a mutation that DELETES or reorders the whole 39-line
    erase block makes every negative pass vacuously; this positive pins the
    erase to actually run when remediation is warranted."""
    # no clean bin on disk -> G2 (never-overwrite) and G3 (throttle) both silent
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

    mgr._kv_dirty_tail = _dirty_flag()
    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert len(_erase_posts(posts)) == 1, (
        "dirty-tip ERASE did NOT fire while all three save-guards would pass "
        f"(erase POSTs: {_erase_posts(posts)}) -- the remediation is being "
        "over-suppressed (guards should never suppress a warranted erase)"
    )
    # the erase must STILL precede the reprefill probe (bin == chain by
    # construction: the probe re-prefills the canonical prefix, so erasing
    # AFTER it would wipe the just-prefilled prefix before the save)
    _erase_idx = next(i for i, (u, _j) in enumerate(posts) if "action=erase" in u)
    _probe_idxs = [i for i, (u, _j) in enumerate(posts) if "chat/completions" in u]
    assert _probe_idxs, (
        "reprefill probe POST never fired -- the erase ran but the probe "
        "reprefill that must follow it did not"
    )
    assert _erase_idx < _probe_idxs[0], (
        "erase fired AFTER the reprefill probe -- the probe just re-prefilled "
        f"the canonical prefix and the erase wiped it (erase idx {_erase_idx} "
        f">= probe idx {_probe_idxs[0]})"
    )
    # the erase's success clears the flag AND the anchor is stamped main --
    # flag is no longer live on this port
    assert _PORT not in (mgr._kv_dirty_tail or {}), (
        "dirty-tip flag not cleared after a successful erase -- the erase block "
        "must pop the per-port flag once it has remediated the tip"
    )


# ==============================================================================
# FLAG PERSISTENCE: the dirty flag persists through sub-min-ctx traffic and
#       the chokepoint refuses saves (incl. idle teardown) until a >=40k main
#       turn reaches the clear
# ==============================================================================
@pytest.mark.asyncio
async def test_dirty_flag_persists_and_blocks_saves_until_main_clear(
        mgr, kv_dir, make_httpx, decline_reasons, caplog):
    """Post-fix, a guard-bail leaves the dirty flag SET. The per-turn CLEAR at
    the top of _probe_and_save_clean_kv sits BEHIND the `inc_len < MIN_CTX_LEN`
    early return (40k chars), so a port whose subsequent traffic is all
    sub-min-ctx never reaches it -- and _save_slot_kv_inner then refuses every
    save on that port (including an idle-teardown save), emitting only a
    warning (a silent KV-save failure that reads as a performance regression).

    This test establishes by reproduction that:
      * guard-bail (G2) leaves the flag SET;
      * sub-min-ctx per-turn traffic does NOT clear it;
      * the chokepoint REFUSES the save while the flag is set;
      * a >= MIN_CTX_LEN main-class turn DOES reach the clear and pops it;
      * after the clear, a save is no longer refused on dirty-tip grounds.

    The refusal window is therefore BOUNDED: any >= 40k main-class turn (or engine
    teardown, which drops the flag) restores saves. The refusal is the
    chokepoint doing its job -- never persist a bin whose VRAM tip holds
    foreign tokens. It is SLOW (a stale bin -> delta reprefill on restore),
    never wrong."""

    # 1) guard-bail leaves the flag SET
    chain = _prefix_hash_chain(_msgs(3))
    _write_clean_bin(kv_dir, _MODEL_TAG, _PORT, "t", 0, chain, prompt_len=60000)
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    mgr._kv_dirty_tail = _dirty_flag()

    # G2 fires -> no erase -> flag survives
    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)
    assert _PORT in (mgr._kv_dirty_tail or {}), (
        "dirty flag must survive a guard-bail (the fix leaves it SET so the "
        "chokepoint keeps refusing saves on the port)"
    )

    # 2) sub-min-ctx per-turn traffic never reaches the per-turn CLEAR
    mgr._kv_dirty_tail = _dirty_flag()
    await mgr._probe_and_save_clean_kv(
        _handle(), _shim(inc_len=100), save_to_disk=False)
    assert _PORT in (mgr._kv_dirty_tail or {}), (
        "sub-min-ctx per-turn traffic must NOT clear the dirty flag: the "
        "per-turn CLEAR sits behind the MIN_CTX_LEN early return"
    )

    # 3) the chokepoint refuses a save while the flag is set (idle-teardown shape)
    caplog.clear()
    mgr._kv_dirty_tail = _dirty_flag()
    refused = await mgr._save_slot_kv(
        _PORT, _MODEL_TAG, _shim(), force_clean=True)
    assert refused is False, (
        "chokepoint must refuse the save while the dirty tip is set (foreign "
        "tokens would poison the bin)"
    )
    assert any("REFUSED (dirty tip)" in r.message for r in caplog.records), (
        "chokepoint must emit its dirty-tip refusal warning (the only signal "
        "an operator has that this port's saves are gated)"
    )

    # 4) a >= MIN_CTX_LEN main-class turn reaches the per-turn CLEAR
    mgr._kv_dirty_tail = _dirty_flag()
    await mgr._probe_and_save_clean_kv(
        _handle(), _shim(inc_len=50000), save_to_disk=False)
    assert _PORT not in (mgr._kv_dirty_tail or {}), (
        "a >= MIN_CTX_LEN main-class turn must clear the dirty flag (main "
        "serve reprocessed the tip)"
    )

    # 5) after the clear, a save is no longer refused on dirty-tip grounds
    #    (it may still decline for OTHER reasons -- the fake engine cannot
    #    produce a real bin -- but the dirty-tip refusal must be gone)
    caplog.clear()
    await mgr._save_slot_kv(_PORT, _MODEL_TAG, _shim(), force_clean=True)
    assert not any("REFUSED (dirty tip)" in r.message for r in caplog.records), (
        "post-clear save was still refused on dirty-tip grounds -- the clear "
        "did not unblock the chokepoint"
    )
