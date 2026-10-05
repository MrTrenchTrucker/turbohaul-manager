"""LOAD_VERIFY final_status truthfulness.

The defect: LOAD_VERIFY kv_restore records could show
final_status='ok' while kv_restore_ok was NOT True -- every such restore
reported success having verified nothing. Root cause: the two
kv_restore call sites in manager.py passed a hardcoded literal final_status
="ok" in the same call that also carried the real, computed kv_restore_ok
verdict -- the true answer was in scope and was overridden. The fix has two
layers:

  BELT (manager.py, the two call sites): final_status is DERIVED from the
    verdict already in scope -- True -> "ok", None (never established) ->
    "unverified", anything else -> "failed". Mirrors the sibling model_load
    site's pre-existing "ok" if healthy else "failed" pattern.
  BRACES (log_load_verify itself): on a kv_restore record, kv_restore_ok
    reading anything other than True is STRUCTURALLY incompatible with a
    passing final_status -- a caller cannot emit that combination even if it
    tries. Coerced to "unverified", never silently dropped, never raised
    (this module's existing never-raise contract is unchanged).

This file covers both layers. The integration tests (through the real,
unedited-elsewhere manager.py) prove the belt; the direct unit tests prove
the braces catch a hardcoded-wrong caller even when the belt is bypassed
(simulating a call site that hardcodes the wrong status, and what a
future third caller might still get wrong).
"""
from __future__ import annotations

import asyncio
import json

import pytest

import turbohaul.manager as manager_mod
from turbohaul import load_verify_log
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
from unittest.mock import MagicMock
from turbohaul.subprocess_mgr import SidecarHandle

_QWEN = "qwen3.6-27b"
_SYS = {"role": "system", "content": "system prompt long enough to matter"}
_U1 = {"role": "user", "content": "first user turn"}


# --- harness (mirrors tests/test_kv_restore_ok_truthful.py's own local copies) ---
def _boot_runtime(tmp_path, *, max_parallel_sidecars=1):
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
    runtime = RuntimeConfig(
        queue=QueueConfig(
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=0,
            idle_hot_load_seconds=600,
            max_grace_extensions=5,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag, *, parallel=1):
    import yaml
    flags = {"split_mode": "none", "main_gpu": 0, "parallel": parallel}
    if parallel > 1:
        flags["kv_unified"] = True
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 0,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "llama_server_flags": flags,
    }))


def _fake_handle(model_tag, port, pid):
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _mocks():
    pid = [91000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete)


def _mk(boot, runtime):
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    mgr.runtime.queue.safety_enabled = False
    return mgr


async def _run_submit(mgr, model_tag, prompt, *, thread_id, admission_hash_chain=None,
                       admission_ctx_len=0):
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        return await asyncio.wait_for(
            mgr.submit_and_wait(
                model_tag, prompt, thread_id=thread_id,
                admission_hash_chain=admission_hash_chain,
                admission_ctx_len=admission_ctx_len,
            ),
            timeout=10,
        )
    finally:
        await mgr.shutdown()


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    import turbohaul.subprocess_mgr as subprocess_mgr
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


def _write_bin(kv_dir, model_tag, port, thread_id, sid, chain, *, clean=True,
               prompt_len=40000, prompt_tokens=15659):
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


class _RestoreResp:
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
        return _RestoreResp()


@pytest.fixture
def restore_posts(monkeypatch):
    recorded = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _RestoreClient(recorded))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return recorded


class _SlotsResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class _SlotsClient:
    def __init__(self, slots_payload, status=200):
        self._slots_payload = slots_payload
        self._status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        return _SlotsResp(self._slots_payload, self._status)


@pytest.fixture
def slots_get(monkeypatch):
    state = {"payload": [], "status": 200}

    class _FakeHttpx:
        AsyncClient = staticmethod(
            lambda *a, **k: _SlotsClient(state["payload"], state["status"]))

    monkeypatch.setattr(load_verify_log, "httpx", _FakeHttpx)

    def _set(payload, status=200):
        state["payload"] = payload
        state["status"] = status

    _set.state = state
    return _set


_PASSING = {"ok", "retried_ok"}


# =====================================================================
# Integration (BELT): through the real manager.py call sites, both triggers.
# =====================================================================
@pytest.mark.asyncio
async def test_integration_spawn_trigger_false_verdict_never_passes(tmp_path, kv_dir, restore_posts, slots_get):
    """★ BELT-BYPASS proof (cap<=1, trigger='spawn').

    IMPORTANT SHAPE, determined by checking the belt/braces overlap rather than
    trusting the reasoning: once the braces were made verdict-aware (so
    deleting the belt doesn't lose the False/None distinction -- see the braces
    docstrings below), the braces ALONE can also derive 'failed' from a False
    verdict. Asserting only `final_status == 'failed'` therefore no longer
    discriminates belt-working from belt-deleted-but-braces-catches-it --
    BOTH produce 'failed'. Reverting the belt
    while keeping the braces leaves every assertion on final_status alone
    green.

    The genuine belt-bypass signal is WHETHER coercion fired at all: the
    braces only auto-populate `reason` with a 'coerced to ...' message when
    THEY had to intervene. When the belt supplies the right value directly,
    the braces never trigger and `reason` is whatever verify_kv_restored's
    own (unrelated) reason was -- here, None, since a plain threshold-miss
    on real numbers doesn't set one. So: belt-working means final_status
    correct AND no 'coerced' marker in reason; belt-deleted-but-braces-
    catch-it means final_status ALSO correct but reason carries the tell.

    MUST FAIL on unmodified base (both assertions fail there -- no belt, no
    braces)."""
    boot, runtime = _boot_runtime(tmp_path)
    _seed_manifest(boot, _QWEN)
    mgr = _mk(boot, runtime)

    chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _QWEN, 59500, "t-spawn-false", 0, chain, prompt_tokens=15659)
    slots_get([{"id": 0, "n_prompt_tokens": 100}])  # genuinely short restore

    load_verify_log.clear_ring()
    await _run_submit(mgr, _QWEN, "hello", thread_id="t-spawn-false",
                       admission_hash_chain=chain, admission_ctx_len=50000)

    rec = [r for r in load_verify_log.get_recent() if r["event"] == "kv_restore"][-1]
    assert rec["kv_restore_ok"] is False
    assert rec["final_status"] == "failed", (
        f"kv_restore_ok=False must produce final_status='failed' specifically, "
        f"got {rec['final_status']!r}"
    )
    assert "coerced" not in (rec["reason"] or ""), (
        f"final_status read correctly, but the braces had to COERCE it -- the "
        f"belt (the manager.py call site's own derivation) did not supply the "
        f"right value directly. reason={rec['reason']!r}"
    )


@pytest.mark.asyncio
async def test_integration_session_return_trigger_false_verdict_never_passes(tmp_path, kv_dir, restore_posts, slots_get):
    """★ BELT-BYPASS proof (cap>=2, trigger='session-return'): same shape as
    the spawn-trigger test above (see its docstring for why BOTH the exact
    value AND the absence of a 'coerced' reason marker are required to
    genuinely discriminate belt-working from belt-deleted) -- the OTHER
    hardcoded call site. MUST FAIL on unmodified base for the same reason."""
    boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=2)
    _seed_manifest(boot, _QWEN, parallel=2)
    mgr = _mk(boot, runtime)

    chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _QWEN, 59500, "t-sr-false", 0, chain, prompt_tokens=15659)
    slots_get([{"id": 0, "n_prompt_tokens": 100}])

    load_verify_log.clear_ring()
    await _run_submit(mgr, _QWEN, "hello", thread_id="t-sr-false",
                       admission_hash_chain=chain, admission_ctx_len=50000)

    session_return = [r for r in load_verify_log.get_recent()
                       if r["event"] == "kv_restore" and r["trigger"] == "session-return"]
    assert session_return, "expected a session-return kv_restore record"
    rec = session_return[-1]
    assert rec["kv_restore_ok"] is False
    assert rec["final_status"] == "failed", (
        f"kv_restore_ok=False must produce final_status='failed' specifically, "
        f"got {rec['final_status']!r}"
    )
    assert "coerced" not in (rec["reason"] or ""), (
        f"final_status read correctly, but the braces had to COERCE it -- the "
        f"belt did not supply the right value directly. reason={rec['reason']!r}"
    )


@pytest.mark.asyncio
async def test_integration_control_genuinely_good_restore_still_passes(tmp_path, kv_dir, restore_posts, slots_get):
    """CONTROL: a genuinely successful restore still reports a passing
    status. A change that makes everything report failure would pass the two
    discriminators above perfectly and be just as broken."""
    boot, runtime = _boot_runtime(tmp_path)
    _seed_manifest(boot, _QWEN)
    mgr = _mk(boot, runtime)

    chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _QWEN, 59500, "t-spawn-true", 0, chain, prompt_tokens=15659)
    slots_get([{"id": 0, "n_prompt_tokens": 15659}])  # lands exactly

    load_verify_log.clear_ring()
    await _run_submit(mgr, _QWEN, "hello", thread_id="t-spawn-true",
                       admission_hash_chain=chain, admission_ctx_len=50000)

    rec = [r for r in load_verify_log.get_recent() if r["event"] == "kv_restore"][-1]
    assert rec["kv_restore_ok"] is True
    assert rec["final_status"] == "ok"


# =====================================================================
# Direct unit (BRACES): log_load_verify's own structural incompatibility.
# Bypasses the belt entirely -- simulates a caller hardcoding the wrong
# status, the shape the belt prevents at the two manager.py sites, and what a
# future third caller could still get wrong.
# =====================================================================
def test_coerces_false_verdict_hardcoded_ok():
    """A False verdict is coerced to 'failed', NOT 'unverified' -- the two
    are not interchangeable: 'failed' means the restore was checked and came
    up short; 'unverified' means it was never established at all. Flattening
    both to one value here would be real information loss the moment the
    belt (the manager.py call sites) is ever reverted or bypassed."""
    load_verify_log.clear_ring()
    rec = load_verify_log.log_load_verify(
        event="kv_restore", trigger="spawn", model_tag="m", port=1,
        kv_restore_ok=False, final_status="ok",
    )
    assert rec["final_status"] not in _PASSING
    assert rec["final_status"] == "failed"
    assert rec["reason"] is not None and "kv_restore_ok" in rec["reason"]


def test_coerces_none_verdict_hardcoded_retried_ok():
    """None (never established) is exactly as untrustworthy as False -- must
    also be coerced, not just literal False."""
    load_verify_log.clear_ring()
    rec = load_verify_log.log_load_verify(
        event="kv_restore", trigger="spawn", model_tag="m", port=1,
        kv_restore_ok=None, final_status="retried_ok",
    )
    assert rec["final_status"] not in _PASSING
    assert rec["final_status"] == "unverified"


def test_control_true_verdict_not_coerced():
    """CONTROL for the braces layer itself: a real True verdict passes
    through untouched -- a coercion that fires unconditionally would pass
    the two tests above perfectly and be just as broken as the bug."""
    load_verify_log.clear_ring()
    rec = load_verify_log.log_load_verify(
        event="kv_restore", trigger="spawn", model_tag="m", port=1,
        kv_restore_ok=True, final_status="ok",
    )
    assert rec["final_status"] == "ok"


def test_scoped_to_kv_restore_event_only():
    """A model_load record legitimately never sets kv_restore_ok -- the
    incompatibility check must not fire on it. Prevents an over-broad gate
    from downgrading every clean model-load-only record."""
    load_verify_log.clear_ring()
    rec = load_verify_log.log_load_verify(
        event="model_load", trigger="spawn", model_tag="m", port=1,
        final_status="ok",
    )
    assert rec["final_status"] == "ok"


def test_omitted_final_status_defaults_to_none_not_ok():
    load_verify_log.clear_ring()
    rec = load_verify_log.log_load_verify(
        event="kv_restore", trigger="spawn", model_tag="m", port=1,
        kv_restore_ok=True,
    )
    assert rec["final_status"] is None


def test_20_20_regression_shape_never_reads_as_pass():
    """The exact failing shape: kv_restore_ok is NOT True, n_past
    unavailable, and a caller (a hardcoded-wrong one, or a future one) still
    asserts a passing final_status. Must not read as a pass."""
    load_verify_log.clear_ring()
    rec = load_verify_log.log_load_verify(
        event="kv_restore", trigger="spawn", model_tag="m", port=1,
        kv_expected_tokens=None, kv_actual_n_past=None, kv_restore_ok=False,
        final_status="ok",
    )
    assert rec["kv_actual_n_past"] is None
    assert rec["final_status"] not in _PASSING


# =====================================================================
# verify_kv_restored does not leave a silent None+no-reason gap when a
# matching slot is found but carries no n_prompt_tokens.
# =====================================================================
class _Handle:
    def __init__(self, port=None, pid=None):
        self.port = port
        self.pid = pid


@pytest.mark.asyncio
async def test_matched_slot_missing_token_field_gets_a_named_reason(slots_get):
    """DISCRIMINATOR: MUST FAIL on unmodified base. expected_tokens=None here
    (the "nothing meaningfully expected" branch, e.g. a cold-fresh restore
    with no prior bin) is the branch that returns WITHOUT ever touching
    reason at all on base -- kv_actual_n_past=None, reason=None, completely
    silent, indistinguishable from "nothing was tried". A non-None
    expected_tokens would still get a generic (less specific) fallback
    reason from the arithmetic tail even on base, so this is the genuinely
    silent case, not just a less-specific-message case."""
    slots_get([{"id": 0}])  # slot present, no "n_prompt_tokens" key at all
    out = await load_verify_log.verify_kv_restored(_Handle(port=59500), 0, None)
    assert out["kv_actual_n_past"] is None
    assert out["reason"] is not None and "n_prompt_tokens" in out["reason"]


@pytest.mark.asyncio
async def test_matched_slot_with_token_field_unaffected(slots_get):
    """CONTROL: the ordinary successful-match path is untouched by this fix."""
    slots_get([{"id": 0, "n_prompt_tokens": 15659}])
    out = await load_verify_log.verify_kv_restored(_Handle(port=59500), 0, 15659)
    assert out["kv_actual_n_past"] == 15659
    assert out["reason"] is None


# =====================================================================
# Self-describing /slots read: n_past=None is ambiguous across FOUR states
# (engine down / slot absent / read failed / read never attempted) --
# structured fields let a reader (or an automated log-analysis pass)
# distinguish them without parsing `reason` prose.
# =====================================================================
@pytest.mark.asyncio
async def test_self_describing_success_sets_all_three_fields(slots_get):
    slots_get([{"id": 0, "n_prompt_tokens": 15659}])
    out = await load_verify_log.verify_kv_restored(_Handle(port=59500, pid=99999999), 0, 15659)
    assert out["slots_http_status"] == 200
    assert out["slot_id_found"] is True
    assert out["process_alive"] is False  # a pid this high should never be real


@pytest.mark.asyncio
async def test_self_describing_slot_absent_state(slots_get):
    """State 2/4: slot absent -- distinguishable from read failure or engine down."""
    slots_get([{"id": 7, "n_prompt_tokens": 100}])  # present, but not id 0
    out = await load_verify_log.verify_kv_restored(_Handle(port=59500), 0, 15659)
    assert out["slots_http_status"] == 200
    assert out["slot_id_found"] is False


@pytest.mark.asyncio
async def test_self_describing_read_failed_state(slots_get):
    """State 3/4: read failed (non-200) -- slot_id_found stays None (never
    got far enough to check), distinguishable from 'slot absent'."""
    slots_get([{"id": 0, "n_prompt_tokens": 15659}], status=500)
    out = await load_verify_log.verify_kv_restored(_Handle(port=59500), 0, 15659)
    assert out["slots_http_status"] == 500
    assert out["slot_id_found"] is None


@pytest.mark.asyncio
async def test_self_describing_never_attempted_state():
    """State 4/4: read never attempted (no port on the handle) -- the two
    /slots-dependent fields stay None ('not applicable', not 'checked and
    bad'). process_alive is a SEPARATE, independent signal (a pid liveness
    check, not a /slots read) -- it's still computed even without a port,
    consistent with verify_model_resident's own existing convention of
    always returning a bool for this field, never None."""
    out = await load_verify_log.verify_kv_restored(_Handle(port=None), 0, 15659)
    assert out["slots_http_status"] is None
    assert out["slot_id_found"] is None
    assert out["process_alive"] is False  # no pid on the handle -> not alive, not "unknown"


def test_self_describing_fields_thread_through_the_record():
    """The structured fields survive into the actual LOAD_VERIFY record, not
    just verify_kv_restored's own return dict -- log_load_verify must accept
    and store them."""
    load_verify_log.clear_ring()
    rec = load_verify_log.log_load_verify(
        event="kv_restore", trigger="spawn", model_tag="m", port=1,
        kv_restore_ok=False, slots_http_status=200, slot_id_found=True,
    )
    assert rec["slots_http_status"] == 200
    assert rec["slot_id_found"] is True


# =====================================================================
# ROOT FIX: no-expectation must never fabricate a verdict. "Is there a
# number" (restore_attempted / _num(actual)) and "did it succeed"
# (kv_restore_ok) are different questions -- answering the second with the
# first was the exact defect. A real case: kv_expected_tokens=None
# paired with kv_restore_ok=False -- those Falses were information-free.
# =====================================================================
@pytest.mark.asyncio
async def test_no_expectation_yields_kv_restore_ok_none_not_false(slots_get):
    """DISCRIMINATOR: no usable expected_tokens, but a real actual number IS
    present (a slot was found and reported a real n_prompt_tokens) -- MUST
    NOT read as False (a definitive-sounding failure verdict fabricated from
    "is there a number", not from any real comparison). MUST FAIL on
    unmodified base, which reads kv_restore_ok=True here via
    _num(actual) -- also wrong, just wrong in the other direction."""
    slots_get([{"id": 0, "n_prompt_tokens": 100}])
    out = await load_verify_log.verify_kv_restored(_Handle(port=59500), 0, None)
    assert out["kv_actual_n_past"] == 100  # a real number IS present
    assert out["kv_restore_ok"] is None    # but there was nothing to check it against
    assert out["restore_attempted"] is False
    assert out["reason"] is not None


@pytest.mark.asyncio
async def test_no_expectation_control_with_zero_or_negative_expected(slots_get):
    """CONTROL: expected_tokens=0 and a negative value hit the same branch as
    None (both are "no usable expectation") -- confirms the fix is keyed off
    validity, not specifically the None case."""
    slots_get([{"id": 0, "n_prompt_tokens": 100}])
    for bad_expected in (0, -5):
        out = await load_verify_log.verify_kv_restored(_Handle(port=59500), 0, bad_expected)
        assert out["kv_restore_ok"] is None
        assert out["restore_attempted"] is False


def test_no_expectation_record_does_not_render_as_failed():
    """The braces/belt mapping (kv_restore_ok is None -> 'unverified', never
    'failed') already handles this correctly with NO belt or braces code
    change -- this pins that a no-expectation record renders as
    'unverified', not the loud-wrong 'failed' that landing kv_restore_ok as
    a fabricated False would have produced through the existing belt logic."""
    load_verify_log.clear_ring()
    rec = load_verify_log.log_load_verify(
        event="kv_restore", trigger="spawn", model_tag="m", port=1,
        kv_expected_tokens=None, kv_actual_n_past=100, kv_restore_ok=None,
        restore_attempted=False, final_status="ok",  # a caller wrongly asserting "ok"
    )
    assert rec["final_status"] == "unverified"
    assert rec["final_status"] != "failed"


@pytest.mark.asyncio
async def test_integration_no_expectation_does_not_render_as_failed(tmp_path, kv_dir, restore_posts, slots_get):
    """Integration-level control through the real (unedited) manager.py
    call site: a cold-fresh thread with no saved bin (no expectation to
    verify against) must not produce final_status='failed' -- the
    shape this root fix exists to stop reproducing."""
    boot, runtime = _boot_runtime(tmp_path)
    _seed_manifest(boot, _QWEN)
    mgr = _mk(boot, runtime)

    # No _write_bin call -- no saved bin for this thread -> _restore_slot_kv
    # returns expected_tokens=None (nothing to restore).
    slots_get([])

    load_verify_log.clear_ring()
    await _run_submit(mgr, _QWEN, "hello", thread_id="t-no-expectation")

    rec = [r for r in load_verify_log.get_recent() if r["event"] == "kv_restore"][-1]
    assert rec["kv_expected_tokens"] is None
    assert rec["kv_restore_ok"] is None
    assert rec["final_status"] == "unverified"
    assert rec["final_status"] != "failed"


# =====================================================================
# CLASS-LEVEL invariant (not another instance fix): kv_restore_ok is False
# IFF a real number was obtained for BOTH actual and expected, AND actual
# fell short. Exhaustive on _kv_verdict, the SOLE permitted producer of
# kv_restore_ok -- a future branch that tries to set it any other way has
# to go through this function or bypass it visibly; this test pins the
# function's own boundary behavior so a regression inside it goes red too.
# =====================================================================
def test_invariant_kv_verdict_false_only_from_a_real_measurement():
    _kv_verdict = load_verify_log._kv_verdict
    # Unmeasurable inputs -> None, NEVER False. This is the exact set of
    # inputs that could (at various points) produce a
    # fabricated True or False: no expectation, zero/negative expectation,
    # non-numeric actual, non-numeric expected, bool actual (int subclass,
    # deliberately excluded).
    assert _kv_verdict(None, 1000, 0.98) is None
    assert _kv_verdict("512", 1000, 0.98) is None
    assert _kv_verdict(100, None, 0.98) is None
    assert _kv_verdict(100, 0, 0.98) is None
    assert _kv_verdict(100, -5, 0.98) is None
    assert _kv_verdict(100, "bad", 0.98) is None
    assert _kv_verdict(True, 1000, 0.98) is None
    # Measurable -> a real verdict, at the exact 0.98 threshold boundary.
    assert _kv_verdict(980, 1000, 0.98) is True
    assert _kv_verdict(979, 1000, 0.98) is False
    assert _kv_verdict(12839, 12000, 0.98) is True
    assert _kv_verdict(100, 999999, 0.98) is False


@pytest.mark.asyncio
async def test_attempted_but_unmeasurable_is_none_not_false(slots_get):
    """The scenario this fix exists for, and it is NOT hypothetical: a real
    expectation existed (a saved bin), the slot WAS found, but its /slots
    record carries no n_prompt_tokens -- as happens on a
    freshly-restored slot that has not yet processed a prompt. MUST FAIL on
    the pre-fix code, which read this as kv_restore_ok=False (a measured
    failure) via `if not _num(actual): kv_restore_ok = False` -- "could not
    measure" reported as "checked and it failed"."""
    slots_get([{"id": 0}])  # slot present, no n_prompt_tokens key at all
    out = await load_verify_log.verify_kv_restored(_Handle(port=59500), 0, 15659)
    assert out["kv_actual_n_past"] is None
    assert out["kv_restore_ok"] is None
    assert out["restore_attempted"] is True  # the expectation WAS real
    # reason is the specific message (set at the point the gap is
    # detected, upstream of the verdict) -- still present and still names
    # the real cause, just not literally the word "unmeasurable" (that
    # fallback wording is for cases with no earlier, more specific reason).
    assert out["reason"] is not None and "n_prompt_tokens" in out["reason"]


@pytest.mark.asyncio
async def test_integration_attempted_but_unmeasurable_renders_unverified(tmp_path, kv_dir, restore_posts, slots_get):
    """Integration-level control through the real manager.py call site: the
    same shape end to end -- a saved bin exists (real expectation), the
    engine's /slots omits n_prompt_tokens for the matching slot. Must render
    'unverified', not 'failed'."""
    boot, runtime = _boot_runtime(tmp_path)
    _seed_manifest(boot, _QWEN)
    mgr = _mk(boot, runtime)

    chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _QWEN, 59500, "t-unmeasurable", 0, chain, prompt_tokens=15659)
    slots_get([{"id": 0}])  # found, no n_prompt_tokens

    load_verify_log.clear_ring()
    await _run_submit(mgr, _QWEN, "hello", thread_id="t-unmeasurable",
                       admission_hash_chain=chain, admission_ctx_len=50000)

    rec = [r for r in load_verify_log.get_recent() if r["event"] == "kv_restore"][-1]
    assert rec["kv_expected_tokens"] == 15659
    assert rec["kv_actual_n_past"] is None
    assert rec["kv_restore_ok"] is None
    assert rec["final_status"] == "unverified"
    assert rec["final_status"] != "failed"


