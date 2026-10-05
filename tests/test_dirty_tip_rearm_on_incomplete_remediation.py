"""The dirty-tip chokepoint must stay armed across
the three returns between a successful erase (which pops the flag) and a
CONFIRMED save: STRIP_PROBE_FAILED, PROBE_POST_FAILED, SAVE_NOT_CONFIRMED.

WHY THE POP CANNOT MOVE TO "after a confirmed save" (the obvious
alternative): verified directly against the dirty-tip chokepoint in
`_save_slot_kv_inner` (manager.py) -- its own comment says
the remediation "clears the flag before calling us", and its code refuses
ANY save while the port is still marked dirty. Moving the pop later would
make the very save this function is about to attempt refuse itself --
self-defeating for every erase+reprefill+save cycle, not just the three
named failure paths. So the fix keeps the early pop and instead RE-ARMS the
original dirty-tip record on exactly the three paths that fail to keep the
erase's implicit promise (a fresh, confirmed clean save).

* A RESTORE-SUCCESS RETURN IS NOT A FOURTH INSTANCE (verified directly):
its restore-SUCCESS path (`return` right after "RESTORED existing
verified clean snapshot") leaves the flag absent correctly -- the slot has
been REPOPULATED with genuinely clean, independently-verified content, so
"not dirty" is the true state, not a gap. Its restore-FAILURE paths
(unverified / no-snapshot) do not `return` at all -- they fall through to
the SAME pre-existing reprefill+save logic the three named returns already
cover. See the restore-success test below for the
direct proof of the first half.

Impact of the bug itself, as shown by tracing what an
empty, unprotected slot actually costs: SLOW, NEVER WRONG. An unprotected
port just means the NEXT warm request's per-turn probe finds nothing to
reuse and pays a full canonical reprefill from the engine's own native
prefix-matching against empty KV -- correct content, wasted time. This fix
is scoped accordingly: three tiny, additive re-arm calls, no restructuring
of the surrounding control flow (which keeps the change small and local
to this function).

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this test file> -v
"""
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
from turbohaul.manager import (
    TurbohaulManager,
    _rearm_dirty_tip_on_incomplete_remediation,
)

_MODEL_TAG = "test-model"
_PORT = 59970
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


class _SaveResp:
    status_code = 200

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
    """GET /slots -> one populated slot (so the erase fires and succeeds).
    POST action=erase -> ok. POST action=save -> deliberately does NOT write
    a real file (fail_save=True), so _save_slot_kv_inner's own os.replace()
    raises and it returns False for real -- not a hand-injected False, the
    same natural failure path this codebase's own save-confirm logic
    already produces. Any other
    POST (the plain reprefill probe) -> ok unless fail_probe_post is set."""

    def __init__(self, slots_payload, posts, *, fail_probe_post=False):
        self._slots_payload = slots_payload
        self._posts = posts
        self._fail_probe_post = fail_probe_post

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
        if "action=erase" in url:
            return _SaveResp({"status": "ok"})
        if "action=save" in url:
            # Deliberately never writes the bin file -- _save_slot_kv_inner's
            # own os.replace() will raise FileNotFoundError and it returns
            # False, exactly as happens naturally on a real failed save.
            return _SaveResp({"status": "ok"})
        if self._fail_probe_post and "/v1/chat/completions" in url:
            return _RaisingResp()
        return _SaveResp({"status": "ok"})


@pytest.fixture
def make_httpx(monkeypatch):
    def _make(payload, *, fail_probe_post=False):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(
                lambda *a, **k: _ProbeSaveClient(payload, posts, fail_probe_post=fail_probe_post))
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return posts

    return _make


def _handle(port=_PORT):
    return types.SimpleNamespace(parallel=1, port=port, pid=_PID, model_tag=_MODEL_TAG)


def _shim(tid="t-258", messages=None, inc_len=50000):
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


def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


def _dirty_flag(port=_PORT, tid="foreign-visitor"):
    return {port: {"role": "curator", "pid": _PID, "thread": tid, "ts": time.time()}}


class TestRearmHelperPure:
    """Unit-level pin on the pure function itself, independent of the
    wiring below."""

    def test_rearms_when_dirty_record_present(self):
        result = _rearm_dirty_tip_on_incomplete_remediation({}, 111, {"role": "curator"})
        assert result == {111: {"role": "curator"}}

    def test_noop_when_dirty_record_falsy(self):
        # The ordinary (never-dirty) seam flush shares these same return
        # sites -- nothing to re-arm, and nothing should be added.
        existing = {222: {"role": "main"}}
        result = _rearm_dirty_tip_on_incomplete_remediation(existing, 111, None)
        assert result == existing
        assert 111 not in result

    def test_preserves_other_ports_untouched(self):
        existing = {222: {"role": "main"}}
        result = _rearm_dirty_tip_on_incomplete_remediation(
            existing, 111, {"role": "curator"})
        assert result == {222: {"role": "main"}, 111: {"role": "curator"}}
        assert existing == {222: {"role": "main"}}, "must not mutate the input dict"


class TestRearmOnStripProbeFailed:
    @pytest.mark.asyncio
    async def test_flag_rearmed_when_strip_probe_fails(
            self, mgr, kv_dir, make_httpx, monkeypatch, caplog):
        mgr.runtime.kv.covered_scaffold_strip = True
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        monkeypatch.setattr(mgr, "_render_strip_prefill_probe",
                             lambda *a, **k: _false_coro())
        mgr._kv_dirty_tail = _dirty_flag()
        caplog.set_level(logging.INFO, logger="turbohaul.manager")

        await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

        assert _PORT in (mgr._kv_dirty_tail or {}), (
            "STRIP_PROBE_FAILED left the port unprotected -- the flag was "
            "not re-armed after the erase already ran"
        )
        assert mgr._kv_dirty_tail[_PORT]["role"] == "curator", (
            "re-armed record must be the ORIGINAL foreign visitor, not a "
            "fresh/different stamp"
        )


async def _false_coro():
    return False


class TestRearmOnProbePostFailed:
    @pytest.mark.asyncio
    async def test_flag_rearmed_when_plain_probe_post_fails(
            self, mgr, kv_dir, make_httpx, caplog):
        mgr.runtime.kv.covered_scaffold_strip = False
        make_httpx([{"id": 0, "n_prompt_tokens": 100}], fail_probe_post=True)
        mgr._kv_dirty_tail = _dirty_flag()
        caplog.set_level(logging.INFO, logger="turbohaul.manager")

        await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

        assert _PORT in (mgr._kv_dirty_tail or {}), (
            "PROBE_POST_FAILED left the port unprotected -- the flag was "
            "not re-armed after the erase already ran"
        )
        assert mgr._kv_dirty_tail[_PORT]["role"] == "curator"


class TestRearmOnSaveNotConfirmed:
    @pytest.mark.asyncio
    async def test_flag_rearmed_when_save_not_confirmed(
            self, mgr, kv_dir, make_httpx, caplog):
        mgr.runtime.kv.covered_scaffold_strip = False
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        mgr._kv_dirty_tail = _dirty_flag()
        caplog.set_level(logging.INFO, logger="turbohaul.manager")

        await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

        assert any("clean-prefix KV save NOT CONFIRMED" in r.message
                   for r in caplog.records), (
            "test setup did not actually reach SAVE_NOT_CONFIRMED -- the "
            "save must fail for real (no bin file materialized) for this "
            "to prove anything"
        )
        assert _PORT in (mgr._kv_dirty_tail or {}), (
            "SAVE_NOT_CONFIRMED left the port unprotected -- the flag was "
            "not re-armed after the erase already ran"
        )
        assert mgr._kv_dirty_tail[_PORT]["role"] == "curator"


class TestRestoreSuccessDoesNotNeedRearming:
    @pytest.mark.asyncio
    async def test_restore_success_correctly_leaves_flag_absent(
            self, mgr, kv_dir, make_httpx, monkeypatch, caplog):
        """The direct proof for the restore-success case: the restore-success return
        is NOT a fourth instance of the same bug. Force _restore_slot_kv to report a
        real count and verify_kv_restored to confirm it -- the flag must
        stay ABSENT (correctly; the tip is genuinely clean now), not
        re-armed, and this fix must not change that."""
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        mgr._kv_dirty_tail = _dirty_flag()

        async def _fake_restore(port, model_tag, slot):
            return 12345

        async def _fake_verify(handle, slot_id, expected_tokens, **kw):
            return {"kv_restore_ok": True, "kv_actual_n_past": 12345}

        monkeypatch.setattr(mgr, "_restore_slot_kv", _fake_restore)
        monkeypatch.setattr(
            manager_mod.load_verify_log, "verify_kv_restored", _fake_verify)
        caplog.set_level(logging.INFO, logger="turbohaul.manager")

        await mgr._probe_and_save_clean_kv(_handle(), _shim(), save_to_disk=True)

        assert any("RESTORED existing verified clean snapshot" in r.message
                   for r in caplog.records)
        assert _PORT not in (mgr._kv_dirty_tail or {}), (
            "the flag was re-armed after a genuinely successful, verified "
            "restore -- the tip IS clean now, re-arming here would be wrong, "
            "not just unnecessary (every future seam flush would pay an "
            "erase+reprefill this restore already made pointless)"
        )
