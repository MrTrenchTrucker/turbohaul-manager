"""LOAD_VERIFY identity instrument, per-field reasons, engine
log health signal.

Background: without these checks Turbohaul's LOAD_VERIFY would report `final_status:"ok"` on a
model swap it never actually verified the SERVED identity of (only
`health_200 and n_ctx>0` -- proves *a* model is up, never which one), stay
silent when the engine's own log showed E-level CUDA OOM errors during that
same load, and hardcode `health_200=True, model_resident=True` on every
kv_restore record regardless of what was actually measured.

Three parts, each covered below:
  Part A -- verify_model_identity(): reads the engine's OWN answer (GET
    /props' model_path, which is present and unconditional-on-GET in this
    build of the vendored engine source; falling back to the
    engine's own logged "loading model '<path>'" line) and compares it to
    the model actually requested. Also: no more hardcoded True literals on
    kv_restore's health_200/model_resident (None -- genuinely not
    remeasured on that record).
  Part B -- verify_kv_restored() now carries kv_expected_reason (OUR side)
    and kv_actual_reason (the ENGINE's side) independently, instead of one
    shared first-write-wins `reason` slot that could silently drop
    whichever cause didn't get there first.
  Part C -- scan_engine_log_for_errors(): does the engine's own log for this
    spawn contain any E-level line, surfaced as its own field
    (engine_errors_detected) WITHOUT touching final_status's own health-only
    scoping: don't relax a check to make a record look better than reality,
    and the mirror image -- don't make a legitimately clean load read worse
    than reality either.

Two kinds of engine-log fixture are used below, both clearly labeled:

  SYNTHETIC (most tests) -- hand-written, format derived from reading the
    vendored engine source this build compiles from (tools/server/server-
    context.cpp's SRV_INF("loading model '%s'\\n", ...) call for the
    identity line; ggml/src/ggml-cuda/ggml-cuda.cu's real cudaMalloc-failed
    message text for the error line), cross-checked against
    Dockerfile.engine-src to confirm that source is what actually builds.
    Engine logs on a typical dev container (/var/lib/turbohaul/engine_logs/)
    are often empty (0 bytes) -- that is environment-specific,
    not global (see REAL below); it is why these fixtures exist at all.

  REAL (the .log files in this test's fixtures directory) -- captured
    llama-server logs from real engine spawns, the real spawns
    behind this test. Model tags and hardware identifiers (GPU/CPU
    model strings, per-device VRAM totals, GPU arch build tokens) are
    genericized to "model-a"/"model-b", matching an existing
    stub-file naming convention in this repo -- the ggml/CUDA
    error text and the content-addressed blob-hash identity line are
    unmodified, so every count and assertion below is exactly
    what the original captures showed. Copied into this tree
    (not read from a host path at test time -- that path won't
    exist in CI). These catch a defect a synthetic-only suite could not:
    a real line carries a
    timestamp FIELD before the level letter (`H.MM.SSS.uuu E ...`), which a
    column-0-anchored version of the matcher silently misses on
    every real error line while still passing every synthetic
    (un-timestamped) test.
"""
from __future__ import annotations

import asyncio
import os

import pytest

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

_MODEL_TAG = "model-ref"
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


def _expected_gguf_path(boot):
    """Mirrors manager.py's own gguf_path construction for _seed_manifest's
    fixed sha256, so the identity-wiring test can assert an EXACT value."""
    sha = "a" * 64
    return str(boot.storage.blob_store_path / "sha256" / sha[:2] / sha)


def _fake_handle(model_tag, port, pid, *, engine_log_path=None):
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag, engine_log_path=engine_log_path)


def _mocks(*, engine_log_path=None):
    pid = [91000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0], engine_log_path=engine_log_path)

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


def _mk(boot, runtime, *, engine_log_path=None):
    mgr = TurbohaulManager(boot, runtime, **_mocks(engine_log_path=engine_log_path))
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
    import json
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
    import turbohaul.manager as manager_mod
    recorded = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _RestoreClient(recorded))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return recorded


class _Handle:
    """Duck-typed stand-in for the manager's _active_handle -- direct
    (non-integration) tests only need .port/.pid, engine_log_path is passed
    explicitly to the functions under test rather than read off the handle."""

    def __init__(self, port=None, pid=None):
        self.port = port
        self.pid = pid


class _RouterResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class _RouterClient:
    """Fake httpx.AsyncClient that dispatches by URL SUFFIX, so /health,
    /slots and /props can each carry a different configured response in the
    same test -- the existing repo's `slots_get` fixture returns the same
    payload for every URL, which is not enough to test /props in isolation
    from /health or /slots."""

    def __init__(self, routes):
        self._routes = routes

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        for suffix, (payload, status) in self._routes.items():
            if url.endswith(suffix):
                return _RouterResp(payload, status)
        return _RouterResp({}, 404)


@pytest.fixture
def routed_get(monkeypatch):
    state = {"routes": {
        "/health": ({"status": "ok"}, 200),
        "/slots": ([{"id": 0, "n_ctx": 4096}], 200),
        "/props": ({}, 404),
    }}

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _RouterClient(state["routes"]))

    monkeypatch.setattr(load_verify_log, "httpx", _FakeHttpx)

    def _set(path_suffix, payload, status=200):
        state["routes"][path_suffix] = (payload, status)

    _set.state = state
    return _set


# =====================================================================
# Part A -- verify_model_identity: direct unit tests.
# =====================================================================
@pytest.mark.asyncio
async def test_identity_verified_true_when_props_reports_matching_model_path(routed_get):
    """CONTROL: a genuine match reads True, source='props'."""
    routed_get("/props", {"model_path": "/models/model-a-27b.gguf"}, 200)
    out = await load_verify_log.verify_model_identity(
        _Handle(port=59500), "/models/model-a-27b.gguf",
    )
    assert out["identity_verified"] is True
    assert out["identity_source"] == "props"
    assert out["served_model_path"] == "/models/model-a-27b.gguf"
    assert out["identity_reason"] is None


@pytest.mark.asyncio
async def test_identity_verified_false_when_props_reports_mismatched_model_path(routed_get):
    """DISCRIMINATOR -- the exact "resident-but-wrong-model" scenario this
    test covers (did a swap actually load the requested weights, or is
    the engine resident but serving something else?). Without the identity
    check, verify_model_identity does not exist at all, so
    this call raises AttributeError."""
    routed_get("/props", {"model_path": "/models/model-b-35b.gguf"}, 200)
    out = await load_verify_log.verify_model_identity(
        _Handle(port=59500), "/models/model-a-27b.gguf",
    )
    assert out["identity_verified"] is False
    assert out["identity_source"] == "props"
    assert out["served_model_path"] == "/models/model-b-35b.gguf"
    assert out["identity_reason"] is not None and "model-b-35b" in out["identity_reason"]


@pytest.mark.asyncio
async def test_identity_falls_back_to_engine_log_when_props_unavailable(routed_get, tmp_path):
    """/props refused (404) -> falls back to the engine's own log line."""
    routed_get("/props", {}, 404)
    log_path = str(tmp_path / "engine_x_p1_1.log")
    with open(log_path, "w") as f:
        f.write("I srv    load_model: loading model '/models/model-a-27b.gguf'\n")
    out = await load_verify_log.verify_model_identity(
        _Handle(port=59500), "/models/model-a-27b.gguf", engine_log_path=log_path,
    )
    assert out["identity_verified"] is True
    assert out["identity_source"] == "log"
    assert out["served_model_path"] == "/models/model-a-27b.gguf"


@pytest.mark.asyncio
async def test_identity_none_when_neither_props_nor_log_available(routed_get, tmp_path):
    """Degrade HONESTLY when nothing is available -- None, never a guessed
    True/False. Exactly the named-unknown requirement."""
    routed_get("/props", {}, 404)
    missing_path = str(tmp_path / "does_not_exist.log")
    out = await load_verify_log.verify_model_identity(
        _Handle(port=59500), "/models/model-a-27b.gguf", engine_log_path=missing_path,
    )
    assert out["identity_verified"] is None
    assert out["identity_source"] is None
    assert out["identity_reason"] is not None


# =====================================================================
# Part C -- scan_engine_log_for_errors: direct unit tests.
# =====================================================================
def test_engine_errors_detected_true_when_log_has_e_level_line(tmp_path):
    """DISCRIMINATOR: without the scan, scan_engine_log_for_errors
    does not exist (AttributeError)."""
    log_path = str(tmp_path / "engine_y_p2_2.log")
    with open(log_path, "w") as f:
        f.write("I srv    load_model: loading model '/models/model-b-35b.gguf'\n")
        f.write(
            "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8204.25 "
            "MiB on device 1: cudaMalloc failed: out of memory\n"
        )
    out = load_verify_log.scan_engine_log_for_errors(log_path)
    assert out["engine_errors_detected"] is True
    assert out["engine_error_lines"]
    assert any("cudaMalloc failed" in line for line in out["engine_error_lines"])


def test_engine_errors_detected_true_on_destructor_warn(tmp_path):
    """~ggml_backend_cuda_buffer_context's cudaFree failure does not
    abort the process -- it logs at W, not E, so this
    function's E-only match would
    otherwise let that line vanish from engine_errors_detected entirely,
    silently, for exactly the cases where it is genuinely worth surfacing
    (this destructor also fires mid-serving, not only at shutdown). Narrow
    match on this ONE destructor's own distinctive text -- see the control
    immediately below for the other required direction."""
    log_path = str(tmp_path / "engine_destructor_warn.log")
    with open(log_path, "w") as f:
        f.write("I srv    update_slots: all slots are idle\n")
        f.write(
            "W CUDA error freeing buffer in ~ggml_backend_cuda_buffer_context "
            "(destructor, not fatal): an illegal memory access was encountered\n"
        )
    out = load_verify_log.scan_engine_log_for_errors(log_path)
    assert out["engine_errors_detected"] is True
    assert any(
        "~ggml_backend_cuda_buffer_context" in line for line in out["engine_error_lines"]
    )


def test_engine_errors_detected_false_on_unrelated_w_level_line(tmp_path):
    """CONTROL, the other required direction: an ordinary, unrelated W
    line must NOT trip this function -- proves the match
    is narrow to this one destructor's own text, rather than broadening to
    catch every warning (which would flood this function and make it
    noisy again, the exact opposite mistake this test guards against)."""
    log_path = str(tmp_path / "engine_unrelated_warn.log")
    with open(log_path, "w") as f:
        f.write("I srv    load_model: loading model '/models/model-a-27b.gguf'\n")
        f.write("W common_params_parse: no chat template found, using default\n")
    out = load_verify_log.scan_engine_log_for_errors(log_path)
    assert out["engine_errors_detected"] is False
    assert out["engine_error_lines"] == []


def test_engine_errors_detected_false_when_log_is_clean(tmp_path):
    """CONTROL: a normal clean load is NOT flagged -- proves no over-correction."""
    log_path = str(tmp_path / "engine_z_p3_3.log")
    with open(log_path, "w") as f:
        f.write("I srv    load_model: loading model '/models/model-a-27b.gguf'\n")
        f.write("I srv       update_slots: all slots are idle\n")
    out = load_verify_log.scan_engine_log_for_errors(log_path)
    assert out["engine_errors_detected"] is False
    assert out["engine_error_lines"] == []


def test_engine_errors_detected_none_when_log_unreadable(tmp_path):
    """'Could not check' must never collapse into 'checked and clean' --
    same never-fabricate discipline as kv_restore_ok/model_resident
    elsewhere in this module."""
    missing_path = str(tmp_path / "does_not_exist.log")
    out = load_verify_log.scan_engine_log_for_errors(missing_path)
    assert out["engine_errors_detected"] is None
    assert out["engine_error_lines"] == []
    assert out["reason"] is not None


def test_engine_errors_detected_none_when_log_present_but_empty(tmp_path):
    """A log file that exists but has zero lines (a realistic shape for
    engine logs -- a crash before the
    first flush, or a rotated-but-not-yet-written file) must NOT read as
    'checked and clean' either -- it is indistinguishable from a log that
    never actually got written to."""
    log_path = str(tmp_path / "engine_empty_p4_4.log")
    open(log_path, "w").close()  # exists, 0 bytes
    out = load_verify_log.scan_engine_log_for_errors(log_path)
    assert out["engine_errors_detected"] is None
    assert out["engine_error_lines"] == []
    assert out["reason"] is not None


def test_engine_errors_detected_true_with_real_timestamp_prefix_format(tmp_path):
    """DISCRIMINATOR -- pins the timestamp defect on REAL
    engine logs: a real line is ``H.MM.SSS.uuu E <component>: ...``, a
    timestamp FIELD before the level letter, not the level letter as the
    very first character of the line. A matcher using
    ``line.lstrip().startswith("E ")`` is anchored to
    column 0 -- on a real, timestamped line that reads the timestamp itself
    (e.g. ``0.27...``), never ``E``, so ``engine_errors_detected`` would come back
    False on every real error line. MUST FAIL against that
    column-0 matcher; passes because the level letter is matched as its own
    whitespace-delimited field regardless of what (if anything) precedes it."""
    log_path = str(tmp_path / "engine_timestamped_p5_5.log")
    with open(log_path, "w") as f:
        f.write("0.00.431.972 I srv    load_model: loading model '/models/model-b-35b.gguf'\n")
        f.write(
            "0.27.406.615 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating "
            "8204.25 MiB on device 1: cudaMalloc failed: out of memory\n"
        )
    out = load_verify_log.scan_engine_log_for_errors(log_path)
    assert out["engine_errors_detected"] is True
    assert any("cudaMalloc failed" in line for line in out["engine_error_lines"])


# =====================================================================
# Part C, real-log pass -- REAL captured engine logs, copied into the
# test fixtures directory so CI does not
# depend on a host path outside this repo. NOT synthetic -- these are
# copies of real engine spawns: model tags and hardware
# identifiers are genericized
# ("model-a"/"model-b" matches the repo's
# own existing stub-file naming convention). The genericizing does not touch the
# ggml/CUDA error text or the content-addressed blob-hash identity line, so
# every count and assertion below matches the original captures. Ground truth
# (the E-line count of each fixture file, via `grep -cE '^[0-9.]+ E '`):
# the counts are exact, not merely
# assumed:
#   engine_model-b-35b_p11500_1785747744.log -> 3 E-lines (the CUDA OOM trio)
#   engine_model-a-27b_p11500_1785747014.log -> 0 E-lines
# =====================================================================
_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "load_verify_identity_and_engine_health")


def test_engine_errors_detected_true_on_real_35b_log_exact_count(tmp_path):
    """DISCRIMINATOR against REAL data -- the regression this test
    exists for. Ground truth: exactly 3 E-lines
    (the CUDA OOM trio), not merely "at least one"."""
    log_path = os.path.join(_FIXTURES_DIR, "engine_model-b-35b_p11500_1785747744.log")
    out = load_verify_log.scan_engine_log_for_errors(log_path)
    assert out["engine_errors_detected"] is True
    assert len(out["engine_error_lines"]) == 3
    assert all("cudaMalloc failed" in l or "failed to allocate" in l for l in out["engine_error_lines"])


def test_engine_errors_detected_false_on_real_27b_log_clean(tmp_path):
    """CONTROL against REAL data -- a genuinely clean real spawn is not
    flagged. Ground truth: 0 E-lines."""
    log_path = os.path.join(_FIXTURES_DIR, "engine_model-a-27b_p11500_1785747014.log")
    out = load_verify_log.scan_engine_log_for_errors(log_path)
    assert out["engine_errors_detected"] is False
    assert out["engine_error_lines"] == []


@pytest.mark.asyncio
async def test_identity_log_fallback_against_real_35b_log(routed_get):
    """Part A's log-fallback identity check against the SAME real fixture --
    confirms it still resolves correctly against real timestamped content
    (the fixture's own blob-hash identity line is the ground truth;
    pinned as a regression test)."""
    routed_get("/props", {}, 404)
    log_path = os.path.join(_FIXTURES_DIR, "engine_model-b-35b_p11500_1785747744.log")
    expected_35b = (
        "/var/lib/turbohaul/blobs/sha256/df/"
        "df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf"
    )
    expected_27b = (
        "/var/lib/turbohaul/blobs/sha256/40/"
        "4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095"
    )
    out_match = await load_verify_log.verify_model_identity(
        _Handle(port=59500), expected_35b, engine_log_path=log_path,
    )
    assert out_match["identity_verified"] is True
    assert out_match["identity_source"] == "log"

    out_mismatch = await load_verify_log.verify_model_identity(
        _Handle(port=59500), expected_27b, engine_log_path=log_path,
    )
    assert out_mismatch["identity_verified"] is False
    assert out_mismatch["served_model_path"] == expected_35b


# =====================================================================
# Part B -- verify_kv_restored: per-field reasons, direct unit test.
# =====================================================================
@pytest.mark.asyncio
async def test_kv_expected_and_actual_reasons_populated_independently(routed_get):
    """DISCRIMINATOR -- the exact bug: kv_expected_tokens
    is null (OUR side -- no expectation supplied) AND kv_actual_n_past is
    null (the ENGINE's side -- slot present, n_prompt_tokens missing) AT THE
    SAME TIME. With only one shared `reason` slot that is
    first-write-wins, whichever cause writes first silently hides the
    other. MUST FAIL against a single-slot design: kv_expected_reason/kv_actual_reason
    would never be set, so both read via .get() as None."""
    routed_get("/slots", [{"id": 0}], 200)  # present, no n_prompt_tokens key at all
    out = await load_verify_log.verify_kv_restored(_Handle(port=59500), 0, None)
    assert out["kv_expected_tokens"] is None
    assert out["kv_actual_n_past"] is None
    assert out.get("kv_expected_reason") is not None, "caller-side null must have its own reason"
    assert out.get("kv_actual_reason") is not None, "engine-side null must have its own reason"
    assert out["kv_expected_reason"] != out["kv_actual_reason"], (
        "the two causes are DIFFERENT facts and must not collapse into one message"
    )
    assert "n_prompt_tokens" in out["kv_actual_reason"]


# =====================================================================
# Integration: through the REAL manager.py call sites.
# =====================================================================
@pytest.mark.asyncio
async def test_integration_model_load_record_carries_identity_fields(tmp_path, kv_dir, restore_posts, routed_get):
    """Proves the REAL manager.py model_load call site (not just the helper
    function in isolation) threads gguf_path through to verify_model_identity
    and that the verdict lands on the record. Without the wiring --
    identity_verified/identity_source/served_model_path/expected_model_path
    are absent from the record; .get() reads None regardless of what
    /props reports, so the exact-mismatch assertions below cannot pass."""
    boot, runtime = _boot_runtime(tmp_path)
    _seed_manifest(boot, _MODEL_TAG)
    mgr = _mk(boot, runtime)
    # /props deliberately reports a DIFFERENT model than requested, proving
    # the wiring reaches a real verdict end to end, not just "non-None".
    routed_get("/props", {"model_path": "/some/other/model.gguf"}, 200)

    load_verify_log.clear_ring()
    await _run_submit(mgr, _MODEL_TAG, "hello", thread_id="t-identity-e2e")

    rec = [r for r in load_verify_log.get_recent() if r["event"] == "model_load"][-1]
    assert rec["identity_verified"] is False
    assert rec["identity_source"] == "props"
    assert rec["served_model_path"] == "/some/other/model.gguf"
    assert rec["expected_model_path"] == _expected_gguf_path(boot)


@pytest.mark.asyncio
async def test_integration_kv_restore_spawn_site_no_hardcoded_true(tmp_path, kv_dir, restore_posts, routed_get):
    """DISCRIMINATOR: fails if the call site hardcodes
    health_200=True, model_resident=True unconditionally on every kv_restore
    record from the cap<=1 spawn call site."""
    boot, runtime = _boot_runtime(tmp_path)
    _seed_manifest(boot, _MODEL_TAG)
    mgr = _mk(boot, runtime)
    chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _MODEL_TAG, 59500, "t-nohc-spawn", 0, chain, prompt_tokens=15659)
    routed_get("/slots", [{"id": 0, "n_prompt_tokens": 15659}], 200)

    load_verify_log.clear_ring()
    await _run_submit(mgr, _MODEL_TAG, "hello", thread_id="t-nohc-spawn",
                       admission_hash_chain=chain, admission_ctx_len=50000)

    rec = [r for r in load_verify_log.get_recent()
           if r["event"] == "kv_restore" and r["trigger"] == "spawn"][-1]
    assert rec["health_200"] is None
    assert rec["model_resident"] is None


@pytest.mark.asyncio
async def test_integration_kv_restore_session_return_site_no_hardcoded_true(tmp_path, kv_dir, restore_posts, routed_get):
    """DISCRIMINATOR: the THIRD call site
    (trigger='session-return' -- not one of the other two kv_restore
    sites). Fails for the identical
    reason as the spawn-site test above -- same hardcoded True literals."""
    boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=2)
    _seed_manifest(boot, _MODEL_TAG, parallel=2)
    mgr = _mk(boot, runtime)
    chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _MODEL_TAG, 59500, "t-nohc-sr", 0, chain, prompt_tokens=15659)
    routed_get("/slots", [{"id": 0, "n_prompt_tokens": 15659}], 200)

    load_verify_log.clear_ring()
    await _run_submit(mgr, _MODEL_TAG, "hello", thread_id="t-nohc-sr",
                       admission_hash_chain=chain, admission_ctx_len=50000)

    rec = [r for r in load_verify_log.get_recent()
           if r["event"] == "kv_restore" and r["trigger"] == "session-return"][-1]
    assert rec["health_200"] is None
    assert rec["model_resident"] is None


@pytest.mark.asyncio
async def test_final_status_not_downgraded_by_engine_errors_alone(tmp_path, kv_dir, restore_posts, routed_get):
    """The mirror-image half of Part C: engine_errors_detected=True must NOT make a healthy load's
    final_status read 'failed'. A rule that flips this the other way is the
    identical defect pointed backwards -- a legitimate cold start (or here, a
    load that logged an E-level line but still came up healthy) must not be
    stamped worse than reality. DISCRIMINATOR on engine_errors_detected
    (KeyError if the field is missing); CONTROL on final_status (proves
    the change doesn't over-correct into downgrading it)."""
    boot, runtime = _boot_runtime(tmp_path)
    _seed_manifest(boot, _MODEL_TAG)

    log_path = str(tmp_path / "engine_err_p59500_1.log")
    with open(log_path, "w") as f:
        f.write(
            "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8204.25 "
            "MiB on device 1: cudaMalloc failed: out of memory\n"
        )
    mgr = _mk(boot, runtime, engine_log_path=log_path)

    load_verify_log.clear_ring()
    await _run_submit(mgr, _MODEL_TAG, "hello", thread_id="t-engine-err-no-downgrade")

    rec = [r for r in load_verify_log.get_recent() if r["event"] == "model_load"][-1]
    assert rec["engine_errors_detected"] is True
    assert rec["engine_error_lines"]
    assert rec["final_status"] == "ok", (
        "a healthy load must not be downgraded to 'failed' just because the "
        "engine's own log also shows an E-level line -- engine_errors_detected "
        "is its OWN field, not a final_status input"
    )
