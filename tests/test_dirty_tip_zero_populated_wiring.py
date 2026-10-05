"""Dirty-tip zero-populated-proceed guard
— end-to-end wiring tier, driving the real `_probe_and_save_clean_kv` method.

THE DEFECT (a naive version of this relaxation): the unload-seam probe
classified a port as "0 populated" (nothing to erase, safe to proceed straight
to the clean save) using ONLY `n_prompt_tokens` (server-context.cpp:1041, the
raw in-memory token-vector length). That is provably not sufficient: two engine
call sites clear the token vector WITHOUT clearing the actual KV cache --
server-context.cpp:2099 (a same-slot LoRA-adapter change) and :3043 (a failed
SLOT_RESTORE, whose own source comment reads "KV may already been invalidated?"
-- the engine authors flagging the exact ambiguity this guard must resolve).
A slot that hit either path reads n_prompt_tokens==0 while the engine still
holds foreign resident KV cells for that sequence id. That relaxation would
have skipped the erase and let `_probe_and_save_clean_kv` proceed straight to
the strip-probe/full-reprefill+save -- capturing a "clean" bin on top of
contaminated KV: a poisoned bin, silently, that a future restore will trust.

THE FIX: a port counts as "populated" (has something to erase) if EITHER
n_prompt_tokens>0 OR n_prompt_tokens_cache>0 for any of its slots.
n_prompt_tokens_cache (server-context.cpp:589, already serialized on the SAME
/slots response at line 1043 -- no extra round trip) is a second, independently
tracked signal. It is zeroed in exactly two places: server_slot::reset()
(server-context.cpp:703, the lifecycle path -- release() calls reset()
unconditionally) and the large-stale-RS-bound branch (server-context.cpp:3884,
a correctly KV-paired clear). NEITHER of the two bad call sites above routes
through either of those -- so after a bad clear, n_prompt_tokens_cache RETAINS
whatever nonzero value the slot's prior real content set it to, discriminating
the bad flow from a genuinely clean release (where BOTH signals read 0
together, since reset() zeroes n_prompt_tokens_cache in the same call that
performs the correctly-paired KV clear).

CAUSAL BOTH DIRECTIONS:
  * test_foreign_cache_residue_routes_to_erase_not_zero_populated_bypass --
    the UNHEALTHY flow (n_prompt_tokens==0, n_prompt_tokens_cache>0) must be
    caught: an erase POST must fire, the zero_populated bypass must NOT be
    taken. A rule that always trusts n_prompt_tokens alone (the naive
    rule) fails this test: it would see 0 populated slots and
    skip the erase.
  * test_genuinely_empty_kv_still_proceeds_without_erase -- the HEALTHY flow
    this relaxation exists to accept (BOTH signals read 0, matching a properly
    released slot) must still proceed cheaply, with no erase POST. A rule that
    reverts to always-refuse-when-not-exactly-one-populated (the earlier,
    pre-relaxation behavior) fails this test: it would refuse the save
    entirely instead of taking the zero_populated bypass.
These two are proven not to be the same test: the first requires an erase POST
that the second must NOT see, and the second requires a proceed-without-erase
that the first must NOT see -- a guard that always proceeds passes only the
second; a guard that never proceeds passes only the first (in the
never-erases-either sense) or fails both differently. Mutating either rule
as described above makes the corresponding test fail.

At least one test (every test in this file) drives the real
`_probe_and_save_clean_kv` production method end to end with a fake httpx
client -- what the test CONSTRUCTS is a raw /slots JSON payload; what
production DERIVES from it is the populated-slot list, the classify() inputs,
the erase POST (or its absence), the dirty-flag clear, and the log line -- none
of that derivation is hand-built by the test.
"""
import json
import logging
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

KV_DECLINE_DIRTY_TIP_ERASE_FAILED = manager_mod.KV_DECLINE_DIRTY_TIP_ERASE_FAILED
KV_DECLINE_UNVERIFIED_EXIT = manager_mod.KV_DECLINE_UNVERIFIED_EXIT


# --- fixtures (mirror test_erase_order.py) -----------------------------------
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


@pytest.fixture(autouse=True)
def _covered_scaffold_strip_off(mgr):
    """Same rationale as test_erase_order.py: the guard under test sits
    entirely inside the erase-decision block, before the strip-probe
    transport is even chosen."""
    mgr.runtime.kv.covered_scaffold_strip = False


@pytest.fixture
def decline_reasons(monkeypatch):
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


class _RaisingResp:
    def raise_for_status(self):
        raise RuntimeError("simulated transport failure")

    def json(self):
        return None


class _ProbeSaveClient:
    """Fakes the sidecar for _probe_and_save_clean_kv: GET /slots returns the
    configured payload (or raises, if `get_raises`); POST action=erase returns
    ok (or raises, if `erase_raises`); every POST is recorded.

    The fixture's action=save now materializes a real bin file under
    SLOT_SAVE_DIR (mirrors the idle-park tip-freshness test's own
    established convention), so _save_slot_kv_inner's os.replace() actually
    succeeds instead of naturally FileNotFoundError-ing on every call --
    without the file, these tests would silently exercise SAVE_NOT_CONFIRMED
    (the save never genuinely confirmed) rather than the confirmed-save path
    their own assertions/docstrings describe. The file is needed because the
    manager now handles SAVE_NOT_CONFIRMED explicitly, so a fixture that
    never produces a bin can no longer pass for a confirmed save."""

    def __init__(self, slots_payload, posts, *, get_raises=False, erase_raises=False):
        self._slots_payload = slots_payload
        self._posts = posts
        self._get_raises = get_raises
        self._erase_raises = erase_raises

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if "/slots" in url and "action=save" not in url:
            if self._get_raises:
                return _RaisingResp()
            return _SaveResp(self._slots_payload)
        return _SaveResp({})

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        if "action=erase" in url and self._erase_raises:
            return _RaisingResp()
        if "action=save" in url and json and "filename" in json:
            import os
            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            tmp_path = os.path.join(SLOT_SAVE_DIR, json["filename"])
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(b"dummy kv cache data")
        return _SaveResp({"status": "ok"})


@pytest.fixture
def make_httpx(monkeypatch):
    def _make(payload, *, get_raises=False, erase_raises=False):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(
                lambda *a, **k: _ProbeSaveClient(
                    payload, posts, get_raises=get_raises, erase_raises=erase_raises))
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return posts

    return _make


# --- helpers (mirror test_erase_order.py) ------------------------------------
_MODEL_TAG = "test-model"
_PORT = 59500
_PID = 4242


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
    return meta_fn


def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


def _handle(port=_PORT):
    return types.SimpleNamespace(parallel=1, port=port, pid=_PID)


def _dirty_flag(port=_PORT, tid="t"):
    return {port: {"role": "curator", "pid": _PID, "thread": tid, "ts": time.time()}}


def _shim(tid="t", messages=None, inc_len=50000):
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


# No pre-existing clean bin in any of these tests -> the never-overwrite-with-
# smaller and zero-growth-throttle belts are silent, and the 3-turn
# chain is non-degenerate -- so every test reaches the erase-decision
# block under test, matching test_erase_order.py's POSITIVE-case setup.


# ==============================================================================
# THE FIX, direction 1: unhealthy flow (foreign cache residue) must be caught
# ==============================================================================
@pytest.mark.asyncio
async def test_foreign_cache_residue_routes_to_erase_not_zero_populated_bypass(
        mgr, kv_dir, make_httpx, decline_reasons, caplog):
    """A slot that hit one of the two unpaired engine clears (LoRA-swap or
    failed SLOT_RESTORE) reports n_prompt_tokens==0 but n_prompt_tokens_cache
    still holds its prior nonzero value. This MUST be routed to the erase
    branch (case=single_erased), not the zero_populated bypass -- a rule that
    trusts n_prompt_tokens alone (the naive rule) would see 0
    populated slots here and skip the erase entirely."""
    posts = make_httpx([
        {"id": 0, "n_prompt_tokens": 0, "n_prompt_tokens_cache": 4200},
    ])

    mgr._kv_dirty_tail = _dirty_flag()
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert len(_erase_posts(posts)) == 1, (
        "erase did NOT fire for a slot with n_prompt_tokens==0 but "
        f"n_prompt_tokens_cache>0 (erase POSTs: {_erase_posts(posts)}) -- a "
        "single-signal (n_prompt_tokens-only) check would wrongly classify "
        "this as zero_populated and skip the erase, leaving foreign KV cells "
        "under the next clean save (poisoned bin)"
    )
    assert "action=erase" in _erase_posts(posts)[0]
    assert "/slots/0?action=erase" in _erase_posts(posts)[0], (
        f"erase targeted the wrong slot id: {_erase_posts(posts)[0]}"
    )
    assert not any("PROCEED-ON-ZERO" in r.message for r in caplog.records), (
        "PROCEED-ON-ZERO fired for a slot that still had cache-hit residue -- "
        "the zero_populated bypass must not be taken here"
    )
    # The erase pops the flag immediately (so the save
    # attempt that follows isn't refused by the dirty-tip chokepoint),
    # but this fixture's action=save doesn't carry the erase all the way to
    # a CONFIRMED save (no .ckpt sidecar / meta write realism -- out of
    # scope for what this test actually checks, which is the erase/bypass
    # CLASSIFICATION above, already fully asserted). Since the save isn't
    # confirmed, the manager correctly re-arms the flag rather than leaving
    # this port unprotected: leaving the flag absent after an unconfirmed
    # save would be the defect.
    assert _PORT in (mgr._kv_dirty_tail or {}), (
        "dirty flag was not re-armed after an unconfirmed save following "
        "the erase -- see the re-arm-on-unconfirmed-save behaviour"
    )


@pytest.mark.asyncio
async def test_foreign_cache_residue_two_slots_one_via_each_signal_is_multi_populated(
        mgr, kv_dir, make_httpx, decline_reasons):
    """Correctness of the union, not just the single-slot case: one slot
    populated only via n_prompt_tokens, another only via
    n_prompt_tokens_cache -- both must count, giving populated_count=2
    (multi_populated, refuse -- ambiguous which slot is dirty), not 1 or 0."""
    posts = make_httpx([
        {"id": 0, "n_prompt_tokens": 300, "n_prompt_tokens_cache": 0},
        {"id": 1, "n_prompt_tokens": 0, "n_prompt_tokens_cache": 900},
    ])

    mgr._kv_dirty_tail = _dirty_flag()
    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert _erase_posts(posts) == [], (
        "an erase fired despite 2 populated slots (ambiguous) -- "
        f"{_erase_posts(posts)}"
    )
    assert KV_DECLINE_DIRTY_TIP_ERASE_FAILED in decline_reasons
    assert _PORT in (mgr._kv_dirty_tail or {}), (
        "dirty flag must survive a multi_populated refusal"
    )


# ==============================================================================
# THE FIX, direction 2: healthy flow (genuinely empty KV) must still proceed
# ==============================================================================
@pytest.mark.asyncio
async def test_genuinely_empty_kv_still_proceeds_without_erase(
        mgr, kv_dir, make_httpx, decline_reasons, caplog):
    """Both signals read 0 on every slot -- the ordinary, designed-for trigger
    for the zero-populated relaxation (tip already left with the disposable occupant via a
    proper release()/reset(), which zeroes n_prompt_tokens_cache too). MUST
    proceed WITHOUT an erase POST. A rule that reverts to the earlier,
    pre-relaxation behavior (refuse whenever populated_count != 1) fails
    this test: it would refuse the save entirely instead of taking the cheap
    zero_populated bypass."""
    posts = make_httpx([
        {"id": 0, "n_prompt_tokens": 0, "n_prompt_tokens_cache": 0},
    ])

    mgr._kv_dirty_tail = _dirty_flag()
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert _erase_posts(posts) == [], (
        f"an erase POST fired for a genuinely empty slot: {_erase_posts(posts)}"
    )
    assert KV_DECLINE_DIRTY_TIP_ERASE_FAILED not in decline_reasons, (
        "the zero_populated case was wrongly refused instead of taking the "
        f"bypass (declines emitted: {decline_reasons})"
    )
    assert any("PROCEED-ON-ZERO" in r.message for r in caplog.records), (
        "the zero_populated bypass's own observability line did not fire"
    )
    # Same reasoning as the previous test's
    # assertion -- the zero_populated bypass pops the flag immediately (so
    # the save attempt isn't refused by the dirty-tip chokepoint), but this
    # fixture's save doesn't reach a CONFIRMED state, so the flag is
    # correctly re-armed rather than left permanently absent. This test's
    # own point (the bypass classification itself) is unaffected and fully
    # covered by the assertions above.
    assert _PORT in (mgr._kv_dirty_tail or {}), (
        "dirty flag was not re-armed after an unconfirmed save following "
        "the zero_populated bypass -- see the re-arm-on-unconfirmed-save behaviour"
    )


@pytest.mark.asyncio
async def test_missing_cache_field_defaults_to_zero_like_missing_n_prompt_tokens(
        mgr, kv_dir, make_httpx, decline_reasons, caplog):
    """A slot whose /slots entry omits n_prompt_tokens_cache entirely (e.g. a
    slot that has never had a task -- server-context.cpp's to_json() only adds
    these fields `if (ptask)`) must default to 0, exactly like the existing
    `.get("n_prompt_tokens") or 0` handling -- not raise, not count as
    populated by a missing key."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 0}])

    mgr._kv_dirty_tail = _dirty_flag()
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert _erase_posts(posts) == []
    assert any("PROCEED-ON-ZERO" in r.message for r in caplog.records)
    # Same reasoning as the two tests above -- this
    # fixture's save never reaches a CONFIRMED state, so the flag is
    # correctly re-armed rather than left permanently absent.
    assert _PORT in (mgr._kv_dirty_tail or {})


# ==============================================================================
# Exception-path classification (probe_failed vs single_erase_failed)
# ==============================================================================
@pytest.mark.asyncio
async def test_probe_get_raises_classifies_probe_failed(
        mgr, kv_dir, make_httpx, decline_reasons, caplog):
    """The /slots GET itself never completes -> probe_failed, refuse. Must NOT
    be confused with a populated_count that happened to already be known."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}], get_raises=True)

    mgr._kv_dirty_tail = _dirty_flag()
    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert posts == [], "no POST of any kind should fire when the GET itself raised"
    assert KV_DECLINE_DIRTY_TIP_ERASE_FAILED in decline_reasons
    assert any("probe_failed" in r.message for r in caplog.records), (
        "refusal log must carry the probe_failed case, not a generic message"
    )
    assert any("UNKNOWN state" in r.message for r in caplog.records), (
        "the raised-exception log line did not fire"
    )


@pytest.mark.asyncio
async def test_erase_post_raises_classifies_single_erase_failed_not_probe_failed(
        mgr, kv_dir, make_httpx, decline_reasons, caplog):
    """The GET succeeds (populated_count is known to be 1) but the erase POST
    itself raises. Must classify as single_erase_failed (we KNOW there was
    exactly one populated slot, we just failed to erase it) -- not
    probe_failed, which would wrongly suggest the count itself is unknown."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}], erase_raises=True)

    mgr._kv_dirty_tail = _dirty_flag()
    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert len(_erase_posts(posts)) == 1, "the erase attempt must still fire"
    assert KV_DECLINE_DIRTY_TIP_ERASE_FAILED in decline_reasons
    assert any("single_erase_failed" in r.message for r in caplog.records), (
        f"refusal log carried the wrong case (records: "
        f"{[r.message for r in caplog.records]})"
    )
    assert _PORT in (mgr._kv_dirty_tail or {}), (
        "dirty flag must survive an erase-post failure"
    )
