"""The post-restore /slots verification race.

verify_kv_restored fires immediately after the restore POST, with no retry
(by its own design -- its docstring says the retry loop is the caller's
job). On a live engine, the engine's own async KV-state
settling can lose that race: GET /slots returning n_prompt_tokens=None right
after a restore that actually succeeded, with the same read returning a real
number much later. When this hits the dirty-tip remediation
(manager.py, the dirty-tip restore-instead-of-reprefill branch), the
strict `kv_restore_ok is True` check fails on the None, and it falls
through to the very full reprefill it exists to avoid -- safe (byte-
identical fallback) but INERT, and unmeasurably so (its own INFO line reads
"UNVERIFIED (actual=None)" instead of "RESTORED").

★ NOT THE SAME BUG as a known harness pitfall --
that was `load_verify_log` importing httpx at its own module level,
separate from manager.py's, so patching one binding didn't patch the other.
Identical UNVERIFIED/actual=None symptom, different cause. This test's
fixture patches BOTH bindings correctly so it
isolates the SECOND bug, the live race, on its own.

THE FIX: read the engine's
action=restore HTTP response for its own reported "n_restored" count
(confirmed by reading engine/llama-cpp-turboquant/tools/server/server-
task.cpp's server_task_result_slot_save_load::to_json() and server-
context.cpp -- the engine sets this synchronously, in the SAME
response, from llama_state_seq_load_file's own output param; it is not a
race-prone value at all) and pass it through verify_kv_restored's EXISTING
actual_n_past override, which skips the racy follow-up /slots GET entirely
when provided. De-races all three verify_kv_restored call sites at once
(manager.py's resident-driver restore, cold-spawn restore, and the
dirty-tip remediation) since all three already thread expected_tokens from
_restore_slot_kv the same way.

CAUSAL: this test's fake sidecar answers every GET /slots with
n_prompt_tokens=None (the exact documented race) while the restore POST's
OWN response carries a correct n_restored. Against manager.py BEFORE the
fix, the dirty-tip remediation site never reads the POST's own count, always races
the GET, and reports kv_restore_ok=None -> falls through to the reprefill
this test asserts must NOT happen -- so this test fails on unpatched main,
as required. Against the fix, the POST's own count is used, no GET is
raced, and the restore is correctly recognized as successful.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this test file> -v
"""
import json
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
from turbohaul.kv_policy import kv_meta_fn
from turbohaul.manager import TurbohaulManager

_MODEL_TAG = "test-model"
_PORT = 59960
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


def _write_clean_bin(kv_dir, model_tag, port, thread_id, sid, chain, prompt_tokens=15659):
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": prompt_tokens,
        "prompt_len": 60000, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": True,
    }))
    return meta_fn


class _RaceResp:
    """GET /slots -> the documented race shape: n_prompt_tokens=None
    (the engine's async KV-state settling has not caught up yet). POST
    action=restore -> a real 200 carrying the engine's own n_restored count,
    exactly as server_task_result_slot_save_load::to_json() actually returns
    it (per the engine source)."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _RacingClient:
    def __init__(self, restored_tokens, posts):
        self._restored_tokens = restored_tokens
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if "/slots" in url:
            return _RaceResp([{"id": 0, "n_prompt_tokens": None}])
        return _RaceResp({})

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        if "action=restore" in url:
            return _RaceResp({
                "id_slot": 0, "filename": (json or {}).get("filename"),
                "n_restored": self._restored_tokens, "n_read": self._restored_tokens * 4,
                "timings": {"restore_ms": 12.3},
            })
        return _RaceResp({"status": "ok"})


@pytest.fixture
def make_racing_httpx(monkeypatch):
    def _make(restored_tokens):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(lambda *a, **k: _RacingClient(restored_tokens, posts))
            Timeout = staticmethod(lambda *a, **k: None)

        # Both bindings, explicitly: load_verify_log imports httpx at its
        # OWN module level, separate from manager.py's -- missing this patch
        # is exactly the OTHER bug this test must not be mistaken for.
        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        monkeypatch.setattr(load_verify_log, "httpx", _FakeHttpx)
        return posts

    return _make


def _handle(port=_PORT):
    return types.SimpleNamespace(parallel=1, port=port, pid=_PID, model_tag=_MODEL_TAG)


def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


@pytest.mark.asyncio
async def test_remediation_recognizes_restore_despite_slots_race(
        mgr, kv_dir, make_racing_httpx, caplog):
    """THE LOAD-BEARING ASSERTION, and the causal proof: the dirty-tip
    remediation (manager.py, inside _probe_and_save_clean_kv's dirty-tip
    branch) must report RESTORED and must NOT fall through to the full
    reprefill, even though every /slots GET in this test answers with
    n_prompt_tokens=None -- the exact documented race. On manager.py
    BEFORE the fix, this test fails: the dirty-tip site never reads the restore
    POST's own n_restored, always races the GET, gets None back, and falls
    through to the reprefill this test asserts must be absent."""
    tid = "hermes-main-race-test"
    # The seeded bin must be a genuine, SHORTER prefix of the incoming chain
    # -- not just any pre-existing bin. Two reasons, both load-bearing:
    # (1) the never-overwrite-with-smaller / zero-growth-throttle belts sit
    # BEFORE the dirty-tip block in _probe_and_save_clean_kv -- a same-size
    # bin declines the save before the dirty-tip block is ever reached at all (an easy mistake
    # when building the restore-success arm). (2) the restore
    # path's own hash-chain prefix-validity check would correctly refuse a
    # same-length-but-unrelated bin as DIVERGED -- this must be a real
    # prefix for the restore to be accepted at all, not merely present.
    seed_chain = [f"h{i}" for i in range(2)]
    inc_chain = seed_chain + [f"h{i}" for i in range(2, 5)]
    _write_clean_bin(kv_dir, _MODEL_TAG, _PORT, tid, 0, seed_chain, prompt_tokens=15659)
    posts = make_racing_httpx(restored_tokens=15659)

    mgr._kv_dirty_tail = {_PORT: {"role": "curator", "pid": _PID, "thread": tid,
                                  "ts": time.time()}}
    slot = types.SimpleNamespace(
        thread_id=tid, model_tag=_MODEL_TAG, admission_ctx_len=90000,
        admission_hash_chain=inc_chain, client_meta={"messages": _msgs(5)},
        context=None, prompt="", port=_PORT, pid=_PID, slot_id=None, engine_op="idle",
    )
    caplog.set_level(logging.INFO, logger="turbohaul.manager")

    await mgr._probe_and_save_clean_kv(_handle(), slot, save_to_disk=True)

    assert any("RESTORED existing verified clean snapshot" in r.message
               for r in caplog.records), (
        "the restore-recognized log line never fired -- the dirty-tip remediation "
        "did not recognize a genuinely successful restore because it raced "
        "a /slots GET that (per the documented race) came back "
        "None, instead of trusting the restore POST's own reported count"
    )
    # NOTE: check the exact dirty-tip remediation phrase, not a bare "UNVERIFIED"
    # substring -- KV_DECLINE_UNVERIFIED_EXIT (an unrelated, pre-existing
    # generic exit-tracking constant elsewhere in this same function) also
    # contains "UNVERIFIED" and would false-positive a looser check.
    assert not any("restore attempted but UNVERIFIED" in r.message
                   for r in caplog.records), (
        "the restore was reported UNVERIFIED despite a real, correct "
        "n_restored count sitting in the restore POST's own response -- "
        "the race was not actually closed"
    )
    reprefill_posts = [u for (u, _j) in posts if "/v1/chat/completions" in u]
    assert reprefill_posts == [], (
        f"the full cold reprefill fired anyway -- the whole claim is an "
        f"avoided cost, and a genuinely successful restore must not fall "
        f"through to it just because a /slots GET raced: {reprefill_posts}"
    )


@pytest.mark.asyncio
async def test_verify_kv_restored_directly_skips_the_race_when_actual_n_past_given(
        mgr, kv_dir, make_racing_httpx):
    """Narrower unit-level pin on the mechanism itself, independent of the
    dirty-tip wiring above: verify_kv_restored's own actual_n_past override, when
    supplied, must produce kv_restore_ok=True even though every GET /slots
    in this fixture would answer None -- proving the override genuinely
    bypasses the race rather than merely being accepted as a parameter."""
    make_racing_httpx(restored_tokens=15659)
    result = await load_verify_log.verify_kv_restored(
        _handle(), 0, 15659, actual_n_past=15659)
    assert result["kv_restore_ok"] is True
    assert result["kv_actual_n_past"] == 15659


@pytest.mark.asyncio
async def test_verify_kv_restored_without_override_still_races_as_documented(
        mgr, kv_dir, make_racing_httpx):
    """Control: confirms this fixture genuinely reproduces the documented race (not
    a fixture that accidentally never exercises it) -- WITHOUT the
    actual_n_past override, the same fixture's /slots GET returns None and
    verify_kv_restored correctly (per its own contract) reports
    kv_restore_ok=None, not True. This is the exact "UNVERIFIED
    (actual=None)" production symptom, reproduced here on
    demand."""
    make_racing_httpx(restored_tokens=15659)
    result = await load_verify_log.verify_kv_restored(_handle(), 0, 15659)
    assert result["kv_restore_ok"] is None
    assert result["kv_actual_n_past"] is None
