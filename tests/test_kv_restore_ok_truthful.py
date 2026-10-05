"""kv_restore_ok was a false negative by construction.

Three compounding causes are covered here (see load_verify_log.py / manager.py):

  Cause 1 — log_load_verify accepts `reason` and a `source` field too;
    the manager's kv_restore emit call threads BOTH from the
    verify_kv_restored result instead of dropping them on the floor.
  Cause 2 — a manager caller that hardcodes expected_tokens=None makes
    kv_restore_ok mean only "did I read a number", never a real
    comparison. _restore_slot_kv returns the winning bin's own saved
    token count (saved_n) for the caller to thread as a REAL expected_tokens.
  Cause 3 — the resident-driver restore path (_spawn_for_resident, the cap>=2
    "session-return" restore) emits a LOAD_VERIFY record too,
    trigger="session-return", distinct from the cap<=1 path's
    trigger="spawn".
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


# --- boot/runtime + manifest scaffolding (mirrors test_load_visibility_...) -----
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
    pid = [90000]

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
    """Drive a real request through worker_loop -> _process_slot (or
    _drive_resident on cap>=2), then shut the loop back down.

    admission_hash_chain/admission_ctx_len mirror what the real HTTP route
    stamps from the client's turn history -- needed for the restore path's
    prefix-match to pick a saved bin at all (an empty incoming chain, the
    submit_and_wait default, cannot prefix-match a non-empty saved chain).
    """
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


# --- manager_mod.httpx fake: the restore POST (action=restore) always succeeds ---
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


# --- load_verify_log.httpx fake: the engine's GET /slots read that
# verify_kv_restored performs to learn the ACTUAL post-restore n_past --------
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
    """Controls what verify_kv_restored's GET /slots sees. Call slots_get.set(...)."""
    state = {"payload": [], "status": 200}

    class _FakeHttpx:
        def __init__(self):
            pass
        AsyncClient = staticmethod(
            lambda *a, **k: _SlotsClient(state["payload"], state["status"]))

    fake = _FakeHttpx()
    monkeypatch.setattr(load_verify_log, "httpx", fake)

    def _set(payload, status=200):
        state["payload"] = payload
        state["status"] = status

    _set.state = state
    return _set


# ================================================================================
# Cause 1 — the diagnosis is kept, not thrown away: a real `reason` (and
# `source`) computed by verify_kv_restored now survives into the emitted
# LOAD_VERIFY record via the manager's kv_restore call site.
#
# Without this wiring the kv_restore emit call never passes
# reason= or source= at all, so the record's reason/source stay None no
# matter what verify_kv_restored computed internally.
# ================================================================================
@pytest.mark.asyncio
async def test_reason_and_source_survive_into_the_record(tmp_path, kv_dir, restore_posts, slots_get):
    boot, runtime = _boot_runtime(tmp_path)
    _seed_manifest(boot, _QWEN)
    mgr = _mk(boot, runtime)

    # No saved bin for this thread -> _restore_slot_kv_inner takes the
    # "no bins" early return (expected_tokens=None), and separately the
    # GET /slots read (driven by verify_kv_restored, independent of whether
    # a restore was attempted) returns an EMPTY slot list -> the read fails
    # to find slot 0 -> a real, named reason.
    slots_get([])  # empty /slots response: "slot 0 not found in /slots"

    load_verify_log.clear_ring()
    await _run_submit(mgr, _QWEN, "hello", thread_id="t-part1")

    kv_records = [r for r in load_verify_log.get_recent() if r["event"] == "kv_restore"]
    assert kv_records, "expected a kv_restore LOAD_VERIFY record from the cold-spawn path"
    rec = kv_records[-1]
    assert rec["reason"] is not None and "not found" in rec["reason"], (
        f"expected a named reason surfaced from verify_kv_restored, got: {rec['reason']!r}"
    )
    # source stays None on this specific branch (not-found early-return, before
    # "slots" is ever stamped) -- the cause-2 tests below cover the "slots" value
    # on the successful-match branch. The wiring itself (record["source"] is a
    # real field at all, not silently absent) is what this line proves.
    assert "source" in rec


# ================================================================================
# ★★ Cause 2 — the false arm. A real expected_tokens (the saved bin's OWN
# token count, 15659) beside a real, LOWER actual (100) must produce
# kv_restore_ok=False with BOTH raw numbers present — not decoration that can
# only ever read True.
#
# Without the real value, the manager hardcodes expected_tokens=None at the
# call site, so kv_restore_ok = "did I read a number" (True whenever actual
# is numeric) and can never read False here.
# ================================================================================
@pytest.mark.asyncio
async def test_kv_restore_ok_produces_a_real_false_arm(tmp_path, kv_dir, restore_posts, slots_get):
    boot, runtime = _boot_runtime(tmp_path)
    _seed_manifest(boot, _QWEN)
    mgr = _mk(boot, runtime)

    # A real saved bin for this thread with a KNOWN token count (15659) —
    # this is what _restore_slot_kv_inner will pick as the winning bin and
    # return as expected_tokens.
    chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _QWEN, 59500, "t-part3", 0, chain, prompt_tokens=15659)

    # The engine reports back a genuinely SHORT actual n_past for slot 0 —
    # the restore did not really land the expected depth.
    slots_get([{"id": 0, "n_prompt_tokens": 100}])

    load_verify_log.clear_ring()
    await _run_submit(mgr, _QWEN, "hello", thread_id="t-part3", admission_hash_chain=chain, admission_ctx_len=50000)

    kv_records = [r for r in load_verify_log.get_recent() if r["event"] == "kv_restore"]
    assert kv_records, "expected a kv_restore LOAD_VERIFY record"
    rec = kv_records[-1]
    assert rec["kv_expected_tokens"] == 15659, f"expected the saved bin's own token count, got {rec['kv_expected_tokens']!r}"
    assert rec["kv_actual_n_past"] == 100, f"expected the engine's real actual, got {rec['kv_actual_n_past']!r}"
    assert rec["source"] == "slots", f"expected source='slots', got {rec['source']!r}"
    assert rec["kv_restore_ok"] is False, (
        "kv_restore_ok must be able to read False when actual genuinely falls "
        f"short of expected -- got {rec['kv_restore_ok']!r} (decoration, not a metric)"
    )


@pytest.mark.asyncio
async def test_kv_restore_ok_reads_true_when_actual_meets_expected(tmp_path, kv_dir, restore_posts, slots_get):
    """Sanity control: the same wiring reads True when the restore genuinely lands."""
    boot, runtime = _boot_runtime(tmp_path)
    _seed_manifest(boot, _QWEN)
    mgr = _mk(boot, runtime)

    chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _QWEN, 59500, "t-part3b", 0, chain, prompt_tokens=15659)
    slots_get([{"id": 0, "n_prompt_tokens": 15659}])

    load_verify_log.clear_ring()
    await _run_submit(mgr, _QWEN, "hello", thread_id="t-part3b", admission_hash_chain=chain, admission_ctx_len=50000)

    rec = [r for r in load_verify_log.get_recent() if r["event"] == "kv_restore"][-1]
    assert rec["kv_restore_ok"] is True
    assert rec["kv_expected_tokens"] == 15659
    assert rec["kv_actual_n_past"] == 15659


# ================================================================================
# Cause 3 — the resident-driver ("session-return") restore path emits too,
# with its OWN trigger label, distinguishable in the ring from the cap<=1
# path's trigger="spawn".
#
# Without this wiring _spawn_for_resident's restore call has no
# verify/log_load_verify wiring at all -- zero kv_restore records
# with trigger="session-return" would ever appear.
# ================================================================================
@pytest.mark.asyncio
async def test_session_return_path_emits_its_own_trigger(tmp_path, kv_dir, restore_posts, slots_get):
    boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=2)
    _seed_manifest(boot, _QWEN, parallel=2)
    mgr = _mk(boot, runtime)

    chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _QWEN, 59500, "t-part4", 0, chain, prompt_tokens=15659)
    slots_get([{"id": 0, "n_prompt_tokens": 15659}])

    load_verify_log.clear_ring()
    await _run_submit(mgr, _QWEN, "hello", thread_id="t-part4", admission_hash_chain=chain, admission_ctx_len=50000)

    all_recs = load_verify_log.get_recent()
    session_return = [r for r in all_recs if r["event"] == "kv_restore" and r["trigger"] == "session-return"]
    spawn_triggered = [r for r in all_recs if r["event"] == "kv_restore" and r["trigger"] == "spawn"]
    assert session_return, (
        f"expected a kv_restore record with trigger='session-return' from the "
        f"cap>=2 resident-driver restore path; got triggers: {[r['trigger'] for r in all_recs]}"
    )
    assert not spawn_triggered, "cap>=2 path should not also emit a trigger='spawn' record"
    rec = session_return[-1]
    assert rec["kv_expected_tokens"] == 15659
    assert rec["kv_restore_ok"] is True


# ================================================================================
# Never-raises invariant (docstring contract on both functions): a malformed
# field must not raise into the spawn/restore path.
# ================================================================================
def test_log_load_verify_never_raises_with_new_source_field():
    load_verify_log.clear_ring()
    rec = load_verify_log.log_load_verify(
        event="kv_restore", trigger="session-return", model_tag="m", port=1,
        source=object(),  # deliberately unserializable-adjacent, must not raise
        reason="deliberately odd",
    )
    assert rec["source"] is not None
    assert load_verify_log.get_recent(1)[-1]["event"] == "kv_restore"
