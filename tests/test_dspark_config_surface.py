"""D-Spark/D-Flash manager+FE config surface.

Covers: the widened spec_type allowlist (draft-mtp/draft-dflash/draft-dspark,
draft-eagle3/draft-simple deliberately excluded), the new
spec_draft_gguf_blob_sha256 content-hash field + its argv-injection at spawn,
the closed forward-defense gap for spec_draft_model-shaped flags, and the
three manager.py sites keyed off SPEC_TYPES_NEEDING_RS_SEQ instead of a
bare "mtp" substring test.

Reuses the established per-file-fixture conventions: the synthetic-GGUF
helpers from the MTP draft-footprint wiring test (duplicated per
that file's own stated convention) and the _mocks_capturing_argv /
_argv_value / submit_and_wait dispatcher-drive pattern from
test_auto_placer.py.
"""
from __future__ import annotations

import asyncio
import struct

import pytest
import yaml
from pydantic import ValidationError
from unittest.mock import MagicMock

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
from turbohaul.manifest import (
    Manifest,
    ModelManifest,
    ManifestValidationError,
    SAFE_LLAMA_FLAG_STRING_ENUMS,
    SPEC_TYPES_NEEDING_RS_SEQ,
    _suffix_guard_check,
)
from turbohaul.safety import estimate_kv_cache_mib
from turbohaul.subprocess_mgr import SidecarHandle


# === Manifest-level validation ==============================================

class TestSpecTypeAllowlist:
    def test_draft_mtp_draft_dflash_draft_dspark_all_accepted(self):
        for st in ("draft-mtp", "draft-dflash", "draft-dspark"):
            m = ModelManifest(model_tag=f"t-{st}", gguf_blob_sha256="a" * 64,
                         llama_server_flags={"spec_type": st})
            assert m.llama_server_flags["spec_type"] == st

    def test_draft_eagle3_and_draft_simple_stay_rejected(self):
        """EAGLE3 and the classic external-draft-model mode are
        deliberately out of scope here -- the engine's real enum has 5
        members, this allowlist stays narrower on purpose."""
        for st in ("draft-eagle3", "draft-simple"):
            with pytest.raises(ValidationError):
                ModelManifest(model_tag="bad", gguf_blob_sha256="a" * 64,
                         llama_server_flags={"spec_type": st})

    def test_allowlist_is_exactly_the_three_values(self):
        assert SAFE_LLAMA_FLAG_STRING_ENUMS["spec_type"] == {
            "draft-mtp", "draft-dflash", "draft-dspark"
        }


class TestSpecDraftGgufBlobSha256Field:
    def test_default_empty(self):
        m = ModelManifest(model_tag="t", gguf_blob_sha256="a" * 64)
        assert m.spec_draft_gguf_blob_sha256 == ""

    def test_valid_64_hex_accepted(self):
        m = ModelManifest(model_tag="t", gguf_blob_sha256="a" * 64,
                     spec_draft_gguf_blob_sha256="b" * 64)
        assert m.spec_draft_gguf_blob_sha256 == "b" * 64

    def test_malformed_hash_rejected(self):
        for bad in ("not-a-hash", "b" * 63, "b" * 65, "g" * 64):
            with pytest.raises(ValidationError):
                ModelManifest(model_tag="t", gguf_blob_sha256="a" * 64,
                         spec_draft_gguf_blob_sha256=bad)

    def test_not_a_llama_server_flags_entry(self):
        """extra='forbid' on Manifest means a caller cannot smuggle this in
        under llama_server_flags -- it must be the dedicated top-level field,
        same shape as gguf_blob_sha256/mmproj_blob_sha256."""
        with pytest.raises(ValidationError):
            ModelManifest(model_tag="t", gguf_blob_sha256="a" * 64,
                     llama_server_flags={"spec_draft_gguf_blob_sha256": "b" * 64})


class TestForwardDefenseGapClosed:
    """spec_draft_model matched neither DENIED_FLAGS (exact-string
    only) nor any existing suffix pattern before this fix -- checked
    against the real allowlist, not assumed. This is the regression proof."""

    def test_spec_draft_model_now_caught(self):
        with pytest.raises(ManifestValidationError):
            _suffix_guard_check("spec_draft_model")

    def test_spec_draft_hf_repo_still_caught(self):
        with pytest.raises(ManifestValidationError):
            _suffix_guard_check("spec_draft_hf_repo")

    def test_neither_is_in_the_allowlist_either(self):
        """Belt-and-braces: even if the suffix guard were ever loosened,
        these must never be reachable via llama_server_flags."""
        for bad_flag in ("spec_draft_model", "spec_draft_hf_repo"):
            with pytest.raises(ValidationError):
                ModelManifest(model_tag="t", gguf_blob_sha256="a" * 64,
                         llama_server_flags={bad_flag: "x"})

    def test_no_currently_allowed_flag_ends_in__model(self):
        """Guards the guard: proves the new `.*_model$` pattern rejects
        nothing that works today (re-verified against the real allowlist, not just
        asserted once ad hoc)."""
        from turbohaul.manifest import SAFE_LLAMA_FLAGS
        offenders = [k for k in SAFE_LLAMA_FLAGS if k.endswith("_model")]
        assert offenders == []


# === manager.py mirror-site widening ========================================

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
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False, max_parallel_sidecars=1,
                          **queue_kwargs),
        pull=PullConfig(),
    )
    return boot, runtime


def _mocks():
    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True}

    def fake_spawn(*a, **k):
        raise AssertionError("not expected to spawn in a footprint-only test")

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


def _seed_manifest(boot, model_tag, *, spec_type, spec_draft_n_max=None,
                   arch="", gguf_size_bytes=0, ctx=2048):
    flags = {"spec_type": spec_type}
    if spec_draft_n_max is not None:
        flags["spec_draft_n_max"] = spec_draft_n_max
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": gguf_size_bytes,
        "context_size": ctx,
        "arch": arch,
        "llama_server_flags": flags,
    }))


class TestEngineFingerprintNRsSeqWidened:
    """manager.py (_engine_fingerprint). Real read_manifest path, not
    a mocked one -- mirrors the existing MTP coverage's own rigor."""

    async def test_draft_dspark_gets_nonzero_n_rs_seq(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        _seed_manifest(boot, "dspark-model", spec_type="draft-dspark",
                       spec_draft_n_max=5)
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        fp = mgr._engine_fingerprint("dspark-model")
        assert fp["n_rs_seq"] == 5
        await mgr.shutdown()

    async def test_draft_dflash_gets_nonzero_n_rs_seq(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        _seed_manifest(boot, "dflash-model", spec_type="draft-dflash",
                       spec_draft_n_max=2)
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        fp = mgr._engine_fingerprint("dflash-model")
        assert fp["n_rs_seq"] == 2
        await mgr.shutdown()

    async def test_draft_dspark_omitted_n_max_mirrors_engine_default_of_3(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        _seed_manifest(boot, "dspark-default", spec_type="draft-dspark")
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        fp = mgr._engine_fingerprint("dspark-default")
        assert fp["n_rs_seq"] == 3
        await mgr.shutdown()

    async def test_plain_model_still_gets_zero(self, tmp_path):
        """Absence must still resolve safely -- most models have no spec_type
        at all, and the enum validator never touches an absent key."""
        boot, runtime = _boot_runtime(tmp_path)
        p = boot.storage.manifests_path / "plain-model.yaml"
        p.write_text(yaml.safe_dump({
            "model_tag": "plain-model", "gguf_blob_sha256": "a" * 64,
            "llama_server_flags": {},
        }))
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        fp = mgr._engine_fingerprint("plain-model")
        assert fp["n_rs_seq"] == 0
        await mgr.shutdown()


class TestReadModelFootprintDsparkWidening:
    """manager.py (_read_model_footprint). Proves the widened m_mtp is
    real and reachable, not dead code, using the SAME synthetic-GGUF
    construction as the MTP draft-footprint wiring test -- once for
    a D-Spark model with no bundled-head GGUF field (today's real shape,
    term must stay INERT) and once with nextn_predict_layers hypothetically
    present (proves the widened predicate would correctly pick it up if a
    future GGUF parser ever populates it for a D-Spark model)."""

    @staticmethod
    def _gguf_string(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    @classmethod
    def _kv_str(cls, k: str, v: str) -> bytes:
        return cls._gguf_string(k) + struct.pack("<I", 8) + cls._gguf_string(v)

    @classmethod
    def _kv_u32(cls, k: str, v: int) -> bytes:
        return cls._gguf_string(k) + struct.pack("<I", 4) + struct.pack("<I", v)

    @classmethod
    def _build_gguf(cls, kvs: list) -> bytes:
        body = b"GGUF" + struct.pack("<I", 3)
        body += struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
        return body + b"".join(kvs)

    _GGUF_BODY_BYTES = 19 * 1024 * 1024 * 1024
    _CTX = 250_000

    def _seed_blob(self, boot, gguf_bytes):
        blob_path = boot.storage.blob_store_path / "sha256" / "aa"
        blob_path.mkdir(parents=True, exist_ok=True)
        (blob_path / ("a" * 64)).write_bytes(gguf_bytes)

    async def test_no_nextn_field_stays_inert_for_dspark(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        gguf_bytes = self._build_gguf([
            self._kv_str("general.architecture", "qwen35moe"),
            self._kv_u32("qwen35moe.block_count", 1),
            self._kv_u32("qwen35moe.full_attention_interval", 0),
            self._kv_u32("qwen35moe.attention.head_count_kv", 4),
            self._kv_u32("qwen35moe.attention.key_length", 256),
            self._kv_u32("qwen35moe.attention.value_length", 256),
            # deliberately NO nextn_predict_layers -- today's real D-Spark
            # shape (standalone draft model, not a bundled head).
        ])
        self._seed_blob(boot, gguf_bytes)
        _seed_manifest(boot, "dspark-model", spec_type="draft-dspark",
                       arch="qwen35moe", gguf_size_bytes=self._GGUF_BODY_BYTES,
                       ctx=self._CTX)
        p = boot.storage.manifests_path / "plain-model.yaml"
        p.write_text(yaml.safe_dump({
            "model_tag": "plain-model", "gguf_blob_sha256": "a" * 64,
            "gguf_size_bytes": self._GGUF_BODY_BYTES, "context_size": self._CTX,
            "arch": "qwen35moe", "llama_server_flags": {},
        }))
        mgr = TurbohaulManager(boot, runtime, **_mocks())

        need_dspark, *_ = mgr._read_model_footprint("dspark-model")
        need_plain, *_ = mgr._read_model_footprint("plain-model")
        assert need_dspark == need_plain, (
            "with no nextn_predict_layers in the GGUF, widening m_mtp to "
            "include draft-dspark must NOT fabricate a KV term -- the inner "
            "attn_dims.nextn_predict_layers > 0 gate is what protects this"
        )
        await mgr.shutdown()

    async def test_nextn_field_present_activates_for_dspark(self, tmp_path):
        """Hypothetical: if a future GGUF parser update ever populates
        nextn_predict_layers for a D-Spark model, this proves the widened
        predicate is live wiring, not a comment-only claim."""
        boot, runtime = _boot_runtime(tmp_path)
        gguf_bytes = self._build_gguf([
            self._kv_str("general.architecture", "qwen35moe"),
            self._kv_u32("qwen35moe.block_count", 1),
            self._kv_u32("qwen35moe.full_attention_interval", 0),
            self._kv_u32("qwen35moe.attention.head_count_kv", 4),
            self._kv_u32("qwen35moe.attention.key_length", 256),
            self._kv_u32("qwen35moe.attention.value_length", 256),
            self._kv_u32("qwen35moe.nextn_predict_layers", 1),
        ])
        self._seed_blob(boot, gguf_bytes)
        _seed_manifest(boot, "dspark-model", spec_type="draft-dspark",
                       arch="qwen35moe", gguf_size_bytes=self._GGUF_BODY_BYTES,
                       ctx=self._CTX)
        p = boot.storage.manifests_path / "plain-model.yaml"
        p.write_text(yaml.safe_dump({
            "model_tag": "plain-model", "gguf_blob_sha256": "a" * 64,
            "gguf_size_bytes": self._GGUF_BODY_BYTES, "context_size": self._CTX,
            "arch": "qwen35moe", "llama_server_flags": {},
        }))
        mgr = TurbohaulManager(boot, runtime, **_mocks())

        need_dspark, *_ = mgr._read_model_footprint("dspark-model")
        need_plain, *_ = mgr._read_model_footprint("plain-model")
        assert need_dspark > need_plain, (
            "with nextn_predict_layers present, a widened m_mtp for "
            "draft-dspark MUST add the draft-context term -- if this fails, "
            "the predicate widening isn't actually reachable"
        )
        await mgr.shutdown()


# === spawn-argv injection for spec_draft_gguf_blob_sha256 ==================

def _boot_runtime_dispatch(tmp_path):
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
            default_port_base=59800,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(max_parallel_sidecars=1, safety_min_free_vram_mib=1000),
        pull=PullConfig(),
    )
    return boot, runtime


def _seed_dispatch_manifest(boot, model_tag, *, spec_draft_hash="", spec_type=""):
    """spec_draft_hash and spec_type are INDEPENDENT kwargs -- the
    whole point of the fix is that the two can be set independently in a
    real manifest, so the fixture must be able to construct every combination,
    not just the coupled "both or neither" shape that a simpler version of
    this helper would allow."""
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    payload = {
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 3000 * 1024 * 1024,
        "context_size": 2048,
        "expected_vram_bytes": 3000 * 1024 * 1024,
        "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
    }
    if spec_draft_hash:
        payload["spec_draft_gguf_blob_sha256"] = spec_draft_hash
    if spec_type:
        payload["llama_server_flags"]["spec_type"] = spec_type
    p.write_text(yaml.safe_dump(payload))


def _fake_handle(model_tag: str, port: int, pid: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _mocks_capturing_argv(spawn_calls):
    pid = [90000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        spawn_calls.append({"model_tag": model_tag, "argv": list(argv)})
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
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


def _argv_value(argv, flag):
    key = "--" + flag.replace("_", "-")
    if key not in argv:
        return None
    return argv[argv.index(key) + 1]


class TestSpecDraftModelArgvInjection:
    async def test_spec_draft_model_injected_when_hash_and_spec_type_set(self, tmp_path):
        boot, runtime = _boot_runtime_dispatch(tmp_path)
        _seed_dispatch_manifest(boot, "m1", spec_draft_hash="b" * 64,
                                spec_type="draft-dspark")
        spawn_calls = []
        mgr = TurbohaulManager(boot, runtime, **_mocks_capturing_argv(spawn_calls))
        mgr.runtime.queue.safety_enabled = False
        try:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            await asyncio.wait_for(
                mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
            )
            assert len(spawn_calls) == 1
            argv = spawn_calls[0]["argv"]
            resolved = _argv_value(argv, "spec-draft-model")
            expected = str(
                boot.storage.blob_store_path / "sha256" / ("b" * 64)[:2] / ("b" * 64)
            )
            assert resolved == expected, (argv, expected)
        finally:
            await mgr.shutdown()

    async def test_spec_draft_model_absent_when_both_unset(self, tmp_path):
        """Back-compat: a model with neither field set (every existing
        manifest) must not get a --spec-draft-model flag at all."""
        boot, runtime = _boot_runtime_dispatch(tmp_path)
        _seed_dispatch_manifest(boot, "m1")
        spawn_calls = []
        mgr = TurbohaulManager(boot, runtime, **_mocks_capturing_argv(spawn_calls))
        mgr.runtime.queue.safety_enabled = False
        try:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            await asyncio.wait_for(
                mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
            )
            assert len(spawn_calls) == 1
            argv = spawn_calls[0]["argv"]
            assert "--spec-draft-model" not in argv, argv
        finally:
            await mgr.shutdown()

    async def test_spec_draft_model_absent_when_spec_type_set_but_hash_unset(self, tmp_path):
        """The other decoupled direction: spec_type: draft-dspark with no
        hash. Documents, doesn't just assert, what happens: no
        --spec-draft-model is injected (there's nothing to resolve), so the
        engine gets --spec-type draft-dspark with no draft model named --
        REASONED, not directly observed here (no live engine in this test),
        matching the documented expectation that an omitted
        hash fails at the engine, not at manifest validation or at argv
        construction."""
        boot, runtime = _boot_runtime_dispatch(tmp_path)
        _seed_dispatch_manifest(boot, "m1", spec_type="draft-dspark")
        spawn_calls = []
        mgr = TurbohaulManager(boot, runtime, **_mocks_capturing_argv(spawn_calls))
        mgr.runtime.queue.safety_enabled = False
        try:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            await asyncio.wait_for(
                mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
            )
            assert len(spawn_calls) == 1
            argv = spawn_calls[0]["argv"]
            assert "--spec-draft-model" not in argv, argv
            assert _argv_value(argv, "spec-type") == "draft-dspark"
        finally:
            await mgr.shutdown()

    async def test_spec_draft_model_not_injected_when_hash_set_but_spec_type_omitted(
        self, tmp_path, caplog,
    ):
        """The core regression test for the argv-injection
        concern. A hash-only manifest (spec_type entirely omitted) must NOT
        get a second, unbudgeted model loaded."""
        boot, runtime = _boot_runtime_dispatch(tmp_path)
        _seed_dispatch_manifest(boot, "m1", spec_draft_hash="b" * 64)
        spawn_calls = []
        mgr = TurbohaulManager(boot, runtime, **_mocks_capturing_argv(spawn_calls))
        mgr.runtime.queue.safety_enabled = False
        try:
            with caplog.at_level("WARNING"):
                mgr._worker_task = asyncio.create_task(mgr.worker_loop())
                await asyncio.wait_for(
                    mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
                )
            assert len(spawn_calls) == 1
            argv = spawn_calls[0]["argv"]
            assert "--spec-draft-model" not in argv, argv
            assert any(
                "spec_draft_gguf_blob_sha256 set" in r.message
                and "does not use a standalone draft model" in r.message
                for r in caplog.records
            ), [r.message for r in caplog.records]
        finally:
            await mgr.shutdown()

    async def test_spec_draft_model_not_injected_for_draft_mtp_plus_hash_and_logs(
        self, tmp_path, caplog,
    ):
        """The residual case that the
        narrower fix must handle: spec_type: draft-mtp (bundled head, needs no standalone
        drafter) PLUS a hash set anyway -- a nonsensical but unvalidated
        combination. Must not inject, and must log (silent-and-correct is
        worse than loud-and-correct by design)."""
        boot, runtime = _boot_runtime_dispatch(tmp_path)
        _seed_dispatch_manifest(boot, "m1", spec_draft_hash="b" * 64,
                                spec_type="draft-mtp")
        spawn_calls = []
        mgr = TurbohaulManager(boot, runtime, **_mocks_capturing_argv(spawn_calls))
        mgr.runtime.queue.safety_enabled = False
        try:
            with caplog.at_level("WARNING"):
                mgr._worker_task = asyncio.create_task(mgr.worker_loop())
                await asyncio.wait_for(
                    mgr.submit_and_wait("m1", "p", thread_id="t1"), timeout=5,
                )
            assert len(spawn_calls) == 1
            argv = spawn_calls[0]["argv"]
            assert "--spec-draft-model" not in argv, argv
            assert _argv_value(argv, "spec-type") == "draft-mtp"
            assert any(
                "spec_draft_gguf_blob_sha256 set" in r.message
                and "draft-mtp" in r.message
                for r in caplog.records
            ), [r.message for r in caplog.records]
        finally:
            await mgr.shutdown()


class TestStandaloneDraftModelSetIsSubsetOfRsSeqSet:
    """Pins the relationship between the two constants so they
    cannot silently drift into contradiction -- if a future standalone-draft
    type is added to SPEC_TYPES_WITH_STANDALONE_DRAFT_MODEL without also
    being added to SPEC_TYPES_NEEDING_RS_SEQ, this fails and says so,
    instead of shipping a type that loads a draft model the engine never
    reserves recurrent-state sequences for."""

    def test_standalone_set_is_a_strict_subset(self):
        from turbohaul.manifest import (
            SPEC_TYPES_NEEDING_RS_SEQ,
            SPEC_TYPES_WITH_STANDALONE_DRAFT_MODEL,
        )
        assert SPEC_TYPES_WITH_STANDALONE_DRAFT_MODEL < SPEC_TYPES_NEEDING_RS_SEQ, (
            SPEC_TYPES_WITH_STANDALONE_DRAFT_MODEL, SPEC_TYPES_NEEDING_RS_SEQ,
        )

    def test_draft_mtp_is_the_one_member_that_differs(self):
        """Names the exact difference, not just its existence -- draft-mtp
        needs rs_seq but is never a standalone drafter."""
        from turbohaul.manifest import (
            SPEC_TYPES_NEEDING_RS_SEQ,
            SPEC_TYPES_WITH_STANDALONE_DRAFT_MODEL,
        )
        assert SPEC_TYPES_NEEDING_RS_SEQ - SPEC_TYPES_WITH_STANDALONE_DRAFT_MODEL == {
            "draft-mtp"
        }
