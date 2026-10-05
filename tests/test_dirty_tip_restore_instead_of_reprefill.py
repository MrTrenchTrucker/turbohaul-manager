"""A curator/disposable-role request riding main's
thread id dirty-tips the shared VRAM tip; the manager erases it at
the next unload-seam flush of main's own identity (manager.py dirty-tip seam
remediation) -- and without a restore it ALWAYS pays for a full cold render+
prefill afterward, even when an already-existing, independently-validated
clean snapshot of main's own identity (saved at the moment the
disposable visitor was first admitted, before it ever touched the tip) could
serve via a much cheaper restore instead.

THE FIX (manager.py, inside the `if _dirty:` branch, after the existing
"dirty-tip ERASED/CLEARED ... FULL-reprefill" log): attempt
`_restore_slot_kv(port, model_tag, slot)`; INDEPENDENTLY VERIFY the result
against the engine's own actual post-restore token depth via
`load_verify_log.verify_kv_restored` (the same pattern already used at the
cold-spawn restore call site -- never trust a bare
POST 200); only skip the existing full reprefill when that verification
comes back `kv_restore_ok is True`. Any failure -- no snapshot, restore
exception, or a verification that comes back False/None/short -- falls
through to the EXISTING full-reprefill code, byte-unchanged.

CAUSAL, both the "restore serves" and "fallback" directions, plus the one
mutation that would silently defeat the whole point of verifying at all
(trusting `_restore_slot_kv`'s return alone without checking
`kv_restore_ok`):
  * test_verified_restore_skips_full_reprefill -- restore succeeds AND
    verifies -> the full-reprefill probe must NOT fire.
  * test_no_snapshot_falls_back_to_full_reprefill -- `_restore_slot_kv`
    returns None (nothing to restore) -> the EXISTING full-reprefill probe
    must still fire, unchanged.
  * test_unverified_restore_falls_back_to_full_reprefill -- `_restore_slot_kv`
    returns a token count but `verify_kv_restored` reports
    `kv_restore_ok=False` -> must NOT trust the bare restore return value;
    falls through to the full reprefill. A version of this fix that skipped
    verification (or checked only "was a number returned") would wrongly
    pass this test's SETUP but fail its ASSERTION -- this is the test that
    would have caught that exact mistake.
  * test_restore_exception_falls_back_to_full_reprefill -- `_restore_slot_kv`
    raises -> best-effort, falls through, no crash.

Plus the gating proof for the concurrency question (a second
disposable-role admission arriving while a seam remediation/teardown is
in flight must not land on the same handle mid-flush):
  * test_idle_handle_cleared_before_any_await_in_teardown -- a live
    concurrency test (not a read of the code) proving `self._idle_handle` is
    already `None` -- the exact field `worker_loop` consults to decide
    whether to warm-inherit -- at the instant `_teardown_idle_holder`'s own
    flush work is paused mid-flight, before it has finished.
"""
import logging
import time
import types

import pytest

import turbohaul.load_verify_log as load_verify_log
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
from turbohaul.manager import TurbohaulManager

_MODEL_TAG = "test-model"
_PORT = 59500
_PID = 4242


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


@pytest.fixture(autouse=True)
def _covered_scaffold_strip_off(mgr):
    mgr.runtime.kv.covered_scaffold_strip = False


@pytest.fixture(autouse=True)
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


class _SaveResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _ProbeSaveClient:
    """Fakes the sidecar for _probe_and_save_clean_kv: GET /slots reports
    exactly one populated slot (so the erase branch runs and succeeds);
    every POST is recorded so the full-reprefill probe's own POST to
    /v1/chat/completions can be asserted present or absent."""

    def __init__(self, posts):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if "/slots" in url:
            return _SaveResp([{"id": 0, "n_prompt_tokens": 100}])
        return _SaveResp({})

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        return _SaveResp({"status": "ok"})


@pytest.fixture
def make_httpx(monkeypatch):
    posts = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _ProbeSaveClient(posts))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return posts


def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


def _handle(port=_PORT, alive=True):
    class _Handle(types.SimpleNamespace):
        def is_alive(self_inner):
            return alive
    return _Handle(parallel=1, port=port, pid=_PID, model_tag=_MODEL_TAG)


def _shim(tid="main-thread", messages=None, inc_len=50000):
    return types.SimpleNamespace(
        thread_id=tid,
        model_tag=_MODEL_TAG,
        admission_ctx_len=inc_len,
        admission_hash_chain=[],
        client_meta={"messages": messages or _msgs(3)},
        context=None,
        prompt="",
        port=_PORT,
        pid=_PID,
        slot_id=None,
        engine_op="idle",
    )


def _dirty_flag(port=_PORT, tid="foreign-visitor"):
    return {port: {"role": "curator", "pid": _PID, "thread": tid, "ts": time.time()}}


def _reprefill_posts(posts):
    return [u for (u, _j) in posts if "/v1/chat/completions" in u]


# ==============================================================================
# Restore-branch causal tests
# ==============================================================================
@pytest.mark.asyncio
async def test_verified_restore_skips_full_reprefill(mgr, make_httpx, caplog, monkeypatch):
    posts = make_httpx
    mgr._kv_dirty_tail = _dirty_flag()

    async def _fake_restore(port, model_tag, slot):
        assert port == _PORT and model_tag == _MODEL_TAG
        return 12345

    async def _fake_verify(handle, slot_id, expected_tokens, **kw):
        assert expected_tokens == 12345
        return {"kv_restore_ok": True, "kv_actual_n_past": 12345}

    monkeypatch.setattr(mgr, "_restore_slot_kv", _fake_restore)
    monkeypatch.setattr(load_verify_log, "verify_kv_restored", _fake_verify)
    caplog.set_level(logging.INFO, logger="turbohaul.manager")

    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert _reprefill_posts(posts) == [], (
        "the full render+prefill POST fired even though the restore verified "
        f"successfully: {_reprefill_posts(posts)}"
    )
    assert any("RESTORED existing verified clean snapshot" in r.message
               for r in caplog.records), (
        "the restore-success log line did not fire"
    )
    assert _PORT not in (mgr._kv_dirty_tail or {}), (
        "dirty flag not cleared even though the erase succeeded"
    )


@pytest.mark.asyncio
async def test_no_snapshot_falls_back_to_full_reprefill(mgr, make_httpx, monkeypatch):
    posts = make_httpx
    mgr._kv_dirty_tail = _dirty_flag()

    async def _fake_restore(port, model_tag, slot):
        return None  # nothing to restore -- the ordinary, common case

    verify_called = []

    async def _fake_verify(*a, **kw):
        verify_called.append(True)
        return {"kv_restore_ok": True}

    monkeypatch.setattr(mgr, "_restore_slot_kv", _fake_restore)
    monkeypatch.setattr(load_verify_log, "verify_kv_restored", _fake_verify)

    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert len(_reprefill_posts(posts)) == 1, (
        "the existing full-reprefill probe did not fire when no snapshot was "
        f"available to restore: {posts}"
    )
    assert verify_called == [], (
        "verification must not even be attempted when restore returned nothing"
    )


@pytest.mark.asyncio
async def test_unverified_restore_falls_back_to_full_reprefill(mgr, make_httpx, caplog, monkeypatch):
    """The mutation this test exists to catch: a version of the fix that
    trusts `_restore_slot_kv`'s return value alone (skips the reprefill
    whenever a token count comes back, without checking kv_restore_ok) would
    pass this test's setup and then WRONGLY skip the reprefill -- this
    assertion is what tells the two apart."""
    posts = make_httpx
    mgr._kv_dirty_tail = _dirty_flag()

    async def _fake_restore(port, model_tag, slot):
        return 999  # a number came back, but...

    async def _fake_verify(handle, slot_id, expected_tokens, **kw):
        return {"kv_restore_ok": False, "kv_actual_n_past": 3}  # ...it didn't verify

    monkeypatch.setattr(mgr, "_restore_slot_kv", _fake_restore)
    monkeypatch.setattr(load_verify_log, "verify_kv_restored", _fake_verify)
    caplog.set_level(logging.INFO, logger="turbohaul.manager")

    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert len(_reprefill_posts(posts)) == 1, (
        "an UNVERIFIED restore (kv_restore_ok=False) was trusted anyway -- "
        f"the full reprefill must still fire as the safe fallback: {posts}"
    )
    assert any("UNVERIFIED" in r.message for r in caplog.records), (
        "the unverified-restore fallback log line did not fire"
    )


@pytest.mark.asyncio
async def test_restore_exception_falls_back_to_full_reprefill(mgr, make_httpx, monkeypatch):
    posts = make_httpx
    mgr._kv_dirty_tail = _dirty_flag()

    async def _raising_restore(port, model_tag, slot):
        raise RuntimeError("simulated restore transport failure")

    monkeypatch.setattr(mgr, "_restore_slot_kv", _raising_restore)

    await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

    assert len(_reprefill_posts(posts)) == 1, (
        "a raised exception from the restore attempt must not crash the seam "
        f"flush -- it must fall back to the full reprefill: {posts}"
    )


# ==============================================================================
# Gating proof: concurrent admission during teardown
# ==============================================================================
@pytest.mark.asyncio
async def test_idle_handle_cleared_before_any_await_in_teardown(mgr, monkeypatch):
    """worker_loop's own warm-reclaim decision reads
    self._idle_handle to decide whether a newly-arriving request can be
    routed onto the currently-idle-held engine. This test proves -- by
    actually pausing a real _teardown_idle_holder() call mid-flight and
    inspecting manager state from a concurrently-running coroutine, not by
    reading the code -- that self._idle_handle already reads None (so a
    concurrent arrival CANNOT warm-inherit this handle) well before the
    teardown's own flush/sigterm work has completed."""
    handle = _handle()
    mgr._set_idle_holder(
        handle, _MODEL_TAG, time.monotonic() + 30.0,
        thread_id="main-thread", admission_ctx_len=50000,
        client_meta={"messages": _msgs(3)},
    )
    assert mgr._idle_handle is handle  # sanity: really set before teardown starts

    reached_flush = __import__("asyncio").Event()
    release_flush = __import__("asyncio").Event()

    async def _paused_flush(*a, **kw):
        reached_flush.set()
        await release_flush.wait()

    async def _fake_sigterm(*a, **kw):
        return True, "ok"

    monkeypatch.setattr(mgr, "_flush_clean_kv_at_unload", _paused_flush)
    monkeypatch.setattr(mgr, "_sigterm", _fake_sigterm)
    monkeypatch.setattr(mgr, "_persist_clean_bin_to_ssd", lambda *a, **kw: None)
    monkeypatch.setattr(mgr, "_save_slot_kv", _fake_sigterm)  # any no-op awaitable
    monkeypatch.setattr(mgr, "_shadow_save_at_swap", _fake_sigterm)

    import asyncio
    teardown_task = asyncio.ensure_future(
        mgr._teardown_idle_holder("idle_expired"))

    await asyncio.wait_for(reached_flush.wait(), timeout=2.0)
    # THE PROOF: inspect live manager state from a DIFFERENT coroutine while
    # the teardown's own flush is genuinely still in flight (paused on
    # release_flush, not merely "about to run").
    assert mgr._idle_handle is None, (
        "a concurrently-arriving request would see self._idle_handle still "
        "set to the mid-teardown handle and could warm-inherit it -- exactly "
        "the race condition this test rules out, not argued away"
    )
    assert mgr._idle_thread_id is None
    assert mgr._idle_model_tag is None

    release_flush.set()
    await asyncio.wait_for(teardown_task, timeout=2.0)
