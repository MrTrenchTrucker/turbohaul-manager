"""Spec-downgrade visibility (manager + FE plumbing).

Part A: pure unit tests for spec_downgrade_log.py's parser/writer/ring,
mirroring the load-verify identity and engine-health test's
"Part C" style for scan_engine_log_for_errors (same non-fabrication
discipline: True/False/None, never coerced).

Part B: manager-level integration -- both spawn paths (the single-slot cold
path in _process_slot, and the resident-driver path in _spawn_for_resident,
which needs its own load-verify-style observability) actually
scan a real engine_log_path and produce a record reachable via
status_snapshot(). Also covers the manager_config tenant.
"""
from __future__ import annotations

import asyncio

import pytest
import yaml
from unittest.mock import MagicMock

from turbohaul import spec_downgrade_log
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
from turbohaul.subprocess_mgr import SidecarHandle


# =====================================================================
# Part A -- scan_engine_log_for_spec_downgrade: direct unit tests.
# =====================================================================

_REAL_LINE = (
    "13.24.501.221 W spec: SPEC_DOWNGRADED arch=qwen35 component=draft "
    "reason=draft_context_init_failed detail=nextn tensor missing for layer 12"
)


def test_spec_downgraded_true_full_parse(tmp_path):
    log_path = str(tmp_path / "engine.log")
    with open(log_path, "w") as f:
        f.write("13.00.000.000 I server: loading model\n")
        f.write(_REAL_LINE + "\n")
        f.write("13.25.000.000 I server: model loaded, serving\n")
    out = spec_downgrade_log.scan_engine_log_for_spec_downgrade(log_path)
    assert out["spec_downgraded"] is True
    assert out["arch"] == "qwen35"
    assert out["component"] == "draft"
    assert out["reason"] == "draft_context_init_failed"
    assert out["detail"] == "nextn tensor missing for layer 12"


def test_spec_downgraded_false_when_log_is_clean(tmp_path):
    """CONTROL: a normal load with no SPEC_DOWNGRADED line is NOT flagged."""
    log_path = str(tmp_path / "clean.log")
    with open(log_path, "w") as f:
        f.write("13.00.000.000 I server: loading model\n")
        f.write("13.01.000.000 I server: all slots are idle\n")
    out = spec_downgrade_log.scan_engine_log_for_spec_downgrade(log_path)
    assert out["spec_downgraded"] is False
    assert out["arch"] is None


def test_spec_downgraded_none_when_log_unreadable(tmp_path):
    """'Could not check' must never collapse into 'checked and clean' --
    same never-fabricate discipline scan_engine_log_for_errors already
    holds itself to."""
    missing = str(tmp_path / "does_not_exist.log")
    out = spec_downgrade_log.scan_engine_log_for_spec_downgrade(missing)
    assert out["spec_downgraded"] is None
    assert out["engine_log_reason"] is not None


def test_spec_downgraded_none_when_log_present_but_empty(tmp_path):
    log_path = str(tmp_path / "empty.log")
    open(log_path, "w").close()
    out = spec_downgrade_log.scan_engine_log_for_spec_downgrade(log_path)
    assert out["spec_downgraded"] is None
    assert out["engine_log_reason"] is not None


def test_spec_downgraded_none_when_path_is_none():
    out = spec_downgrade_log.scan_engine_log_for_spec_downgrade(None)
    assert out["spec_downgraded"] is None


def test_spec_downgraded_reads_stdio_sibling_when_primary_is_missing(tmp_path):
    """Same dual-sink discipline as scan_engine_log_for_errors -- the spec-downgrade
    line, like the E-level errors, is not guaranteed to reach the same one
    of .log / .stdio every time."""
    log_path = str(tmp_path / "engine.log")
    # .log deliberately NOT created -- only the .stdio sibling exists.
    with open(log_path + ".stdio", "w") as f:
        f.write(_REAL_LINE + "\n")
    out = spec_downgrade_log.scan_engine_log_for_spec_downgrade(log_path)
    assert out["spec_downgraded"] is True
    assert out["arch"] == "qwen35"


def test_spec_downgraded_matches_tag_regardless_of_prefix_shape(tmp_path):
    """Match the tag, not a status field, and don't assume the exact
    prefix format (matches scan_engine_log_for_errors's own 'never anchored
    to a specific timestamp shape' discipline)."""
    log_path = str(tmp_path / "no_timestamp.log")
    with open(log_path, "w") as f:
        f.write(
            "[spec] SPEC_DOWNGRADED arch=gemma4 component=mtp "
            "reason=draft_context_init_failed detail=some other message\n"
        )
    out = spec_downgrade_log.scan_engine_log_for_spec_downgrade(log_path)
    assert out["spec_downgraded"] is True
    assert out["arch"] == "gemma4"
    assert out["component"] == "mtp"


def test_spec_downgraded_none_status_and_reason_when_tag_found_but_fields_dont_parse(tmp_path):
    """Defensive: a future engine-side wording change to the FIELDS (not
    the tag, which never changes) must read as 'could not tell',
    never silently coerced to False (no downgrade) or fabricated True with
    missing data."""
    log_path = str(tmp_path / "malformed.log")
    with open(log_path, "w") as f:
        f.write("W spec: SPEC_DOWNGRADED something-not-matching-the-contract\n")
    out = spec_downgrade_log.scan_engine_log_for_spec_downgrade(log_path)
    assert out["spec_downgraded"] is None
    assert out["arch"] is None
    assert out["engine_log_reason"] is not None
    assert "arch/component/reason" in out["engine_log_reason"]


def test_spec_downgraded_detail_captures_free_text_to_end_of_line(tmp_path):
    """detail= is explicitly raw/unparsed free text and is always the LAST
    field -- must survive containing further '=' or extra whitespace."""
    log_path = str(tmp_path / "detail.log")
    with open(log_path, "w") as f:
        f.write(
            "W spec: SPEC_DOWNGRADED arch=qwen2 component=draft "
            "reason=draft_context_init_failed "
            "detail=tensor shape mismatch: expected=[1,4096] got=[1,2048]\n"
        )
    out = spec_downgrade_log.scan_engine_log_for_spec_downgrade(log_path)
    assert out["spec_downgraded"] is True
    assert out["detail"] == "tensor shape mismatch: expected=[1,4096] got=[1,2048]"


# =====================================================================
# Part A2 -- log_spec_downgrade / get_recent / clear_ring: direct unit tests.
# =====================================================================

def test_log_spec_downgrade_appends_to_ring_and_get_recent_reads_it_back():
    spec_downgrade_log.clear_ring()
    spec_downgrade_log.log_spec_downgrade(
        model_tag="m1", source="engine_log", arch="qwen35",
        component="draft", reason_code="draft_context_init_failed",
        detail="x",
    )
    recent = spec_downgrade_log.get_recent()
    assert len(recent) == 1
    assert recent[0]["model_tag"] == "m1"
    assert recent[0]["source"] == "engine_log"
    assert recent[0]["arch"] == "qwen35"


def test_log_spec_downgrade_allows_arch_and_component_to_be_none():
    """The manager_config tenant fires before any GGUF read --
    arch must be representable as genuinely absent, never fabricated."""
    spec_downgrade_log.clear_ring()
    rec = spec_downgrade_log.log_spec_downgrade(
        model_tag="m1", source="manager_config",
        reason_code="spec_type_mismatch", detail="x",
    )
    assert rec["arch"] is None
    assert rec["component"] is None
    assert spec_downgrade_log.get_recent()[0]["arch"] is None


def test_get_recent_n_zero_means_none_not_everything():
    spec_downgrade_log.clear_ring()
    spec_downgrade_log.log_spec_downgrade(model_tag="m1", source="engine_log")
    assert spec_downgrade_log.get_recent(0) == []


def test_get_recent_returns_newest_last():
    spec_downgrade_log.clear_ring()
    spec_downgrade_log.log_spec_downgrade(model_tag="m1", source="engine_log", reason_code="first")
    spec_downgrade_log.log_spec_downgrade(model_tag="m1", source="engine_log", reason_code="second")
    recent = spec_downgrade_log.get_recent()
    assert recent[-1]["reason_code"] == "second"


def test_clear_ring_empties_it():
    spec_downgrade_log.clear_ring()
    spec_downgrade_log.log_spec_downgrade(model_tag="m1", source="engine_log")
    spec_downgrade_log.clear_ring()
    assert spec_downgrade_log.get_recent() == []


def test_ring_never_raises_on_json_serialization_edge_cases():
    """Never-raise contract, same as log_load_verify -- an emit failure must
    not fail a spawn."""
    spec_downgrade_log.clear_ring()
    rec = spec_downgrade_log.log_spec_downgrade(
        model_tag="m1", source="engine_log", detail="unicode: ☃ \x00",
    )
    assert rec is not None  # did not raise


# =====================================================================
# Part B -- manager-level integration: both spawn paths, status_snapshot,
# and the manager_config tenant.
# =====================================================================

def _boot_runtime(tmp_path, **queue_kwargs):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
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
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_min_free_vram_mib=1000, **queue_kwargs),
        pull=PullConfig(),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag, *, spec_type="", spec_draft_hash=""):
    payload = {
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 3000 * 1024 * 1024,
        "context_size": 2048,
        "expected_vram_bytes": 3000 * 1024 * 1024,
        "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
    }
    if spec_type:
        payload["llama_server_flags"]["spec_type"] = spec_type
    if spec_draft_hash:
        payload["spec_draft_gguf_blob_sha256"] = spec_draft_hash
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump(payload))


def _write_downgrade_log(tmp_path, name="engine") -> str:
    log_path = str(tmp_path / f"{name}.log")
    with open(log_path, "w") as f:
        f.write("I server: loading model\n")
        f.write(_REAL_LINE + "\n")
        f.write("I server: model loaded, serving\n")
    return log_path


def _mocks_with_engine_log(spawn_calls, engine_log_path):
    pid = [90000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        proc = MagicMock()
        proc.pid = pid[0]
        proc.poll.return_value = None
        spawn_calls.append({"model_tag": model_tag, "argv": list(argv)})
        pid[0] += 1
        return SidecarHandle(
            proc=proc, port=port, model_tag=model_tag,
            engine_log_path=engine_log_path,
        )

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


class TestColdSpawnPathPicksUpEngineDowngrade:
    """The single-slot cold path (_process_slot) -- the ALREADY-instrumented
    call site (scan_engine_log_for_errors' one existing call site)."""

    async def test_spec_downgrade_reaches_status_snapshot(self, tmp_path):
        spec_downgrade_log.clear_ring()
        boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=1)
        _seed_manifest(boot, "m1")
        engine_log = _write_downgrade_log(tmp_path)
        spawn_calls = []
        mgr = TurbohaulManager(boot, runtime, **_mocks_with_engine_log(spawn_calls, engine_log))
        mgr.runtime.queue.safety_enabled = False
        try:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            await asyncio.wait_for(
                mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
            )
            snap = mgr.status_snapshot()
            recs = [r for r in snap["spec_downgrade"] if r["model_tag"] == "m1"]
            assert len(recs) == 1, snap["spec_downgrade"]
            assert recs[0]["source"] == "engine_log"
            assert recs[0]["arch"] == "qwen35"
            assert recs[0]["component"] == "draft"
            assert recs[0]["reason_code"] == "draft_context_init_failed"
        finally:
            await mgr.shutdown()

    async def test_no_record_when_engine_log_is_clean(self, tmp_path):
        spec_downgrade_log.clear_ring()
        boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=1)
        _seed_manifest(boot, "m1")
        log_path = str(tmp_path / "clean.log")
        with open(log_path, "w") as f:
            f.write("I server: loading model\nI server: all slots are idle\n")
        spawn_calls = []
        mgr = TurbohaulManager(boot, runtime, **_mocks_with_engine_log(spawn_calls, log_path))
        mgr.runtime.queue.safety_enabled = False
        try:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            await asyncio.wait_for(
                mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
            )
            snap = mgr.status_snapshot()
            recs = [r for r in snap["spec_downgrade"] if r["model_tag"] == "m1"]
            assert recs == [], snap["spec_downgrade"]
        finally:
            await mgr.shutdown()


class TestResidentDriverPathPicksUpEngineDowngrade:
    """The resident-driver path (_spawn_for_resident, cap>=2) needs its own
    load-verify-style engine-log observability, separate from the single-slot path --
    a spawn path with no engine-log scan would be invisible.
    Without this coverage, a cap>=2 deployment's UI would show nothing even
    though the feature is 'done'."""

    async def test_spec_downgrade_reaches_status_snapshot_via_cap_ge_2_path(self, tmp_path):
        spec_downgrade_log.clear_ring()
        boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=2)
        _seed_manifest(boot, "m1")
        engine_log = _write_downgrade_log(tmp_path)
        spawn_calls = []
        mgr = TurbohaulManager(boot, runtime, **_mocks_with_engine_log(spawn_calls, engine_log))
        mgr.runtime.queue.safety_enabled = False
        try:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            await asyncio.wait_for(
                mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
            )
            snap = mgr.status_snapshot()
            recs = [r for r in snap["spec_downgrade"] if r["model_tag"] == "m1"]
            assert len(recs) == 1, snap["spec_downgrade"]
            assert recs[0]["source"] == "engine_log"
            assert recs[0]["arch"] == "qwen35"
        finally:
            await mgr.shutdown()


class TestManagerConfigTenant:
    """A silent downgrade (spec_type doesn't qualify for a
    standalone drafter) -- the first REAL tenant of this channel, wired to
    prove the pipe end-to-end without needing the engine-side contract at all."""

    async def test_hash_set_but_spec_type_omitted_records_manager_config_downgrade(self, tmp_path):
        spec_downgrade_log.clear_ring()
        boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=1)
        _seed_manifest(boot, "m1", spec_draft_hash="b" * 64)  # spec_type omitted
        spawn_calls = []
        mgr = TurbohaulManager(boot, runtime, **_mocks_with_engine_log(spawn_calls, None))
        mgr.runtime.queue.safety_enabled = False
        try:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            await asyncio.wait_for(
                mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
            )
            snap = mgr.status_snapshot()
            recs = [r for r in snap["spec_downgrade"] if r["model_tag"] == "m1"]
            assert len(recs) == 1, snap["spec_downgrade"]
            assert recs[0]["source"] == "manager_config"
            assert recs[0]["component"] == "draft"
            assert recs[0]["reason_code"] == "spec_type_mismatch"
            assert recs[0]["arch"] is None
        finally:
            await mgr.shutdown()

    async def test_normal_manifest_produces_no_downgrade_record(self, tmp_path):
        """Back-compat: the overwhelming majority of models (no spec_type,
        no draft hash) must never produce a spec_downgrade record."""
        spec_downgrade_log.clear_ring()
        boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=1)
        _seed_manifest(boot, "m1")
        spawn_calls = []
        mgr = TurbohaulManager(boot, runtime, **_mocks_with_engine_log(spawn_calls, None))
        mgr.runtime.queue.safety_enabled = False
        try:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            await asyncio.wait_for(
                mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
            )
            snap = mgr.status_snapshot()
            recs = [r for r in snap["spec_downgrade"] if r["model_tag"] == "m1"]
            assert recs == [], snap["spec_downgrade"]
        finally:
            await mgr.shutdown()


class TestStatusSnapshotWiring:
    def test_spec_downgrade_key_present_and_matches_get_recent(self, tmp_path):
        spec_downgrade_log.clear_ring()
        spec_downgrade_log.log_spec_downgrade(
            model_tag="m1", source="engine_log", arch="qwen35",
            component="draft", reason_code="draft_context_init_failed",
        )
        boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=1)

        def fake_spawn(*a, **k):
            raise AssertionError("not expected to spawn in a snapshot-only test")

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(handle, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        snap = mgr.status_snapshot()
        assert "spec_downgrade" in snap
        assert snap["spec_downgrade"] == spec_downgrade_log.get_recent(20)
