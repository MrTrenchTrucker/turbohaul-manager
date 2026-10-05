"""Manager-level wiring for the MTP draft-context KV term: proves
spec_type flows from a REAL manifest
YAML through _attn_kv_dims_for's real GGUF parse into
_read_model_footprint's need, not just through the safety.py function
signatures in isolation (see the safety-level MTP draft-KV test for that).

Mirrors tests/test_multislot_concurrency.py's exact
test_footprint_cpu_moe_trusts_measured_expected_vram pattern (real
TurbohaulManager, a hand-written manifest YAML, calling
mgr._read_model_footprint(tag) directly) and its _boot_runtime_multislot /
_mocks helpers, duplicated locally per this repo's per-file-fixture
convention.
"""
import struct

import yaml

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
from turbohaul.safety import estimate_kv_cache_mib


def _boot_runtime(tmp_path):
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
            default_port_base=59600,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False, max_parallel_sidecars=1),
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


# --- Synthetic MTP GGUF bytes (same helpers as test_dimension_aware_kv.py) ----

def _gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(k: str, v: str) -> bytes:
    return _gguf_string(k) + struct.pack("<I", 8) + _gguf_string(v)


def _kv_u32(k: str, v: int) -> bytes:
    return _gguf_string(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def _build_gguf(kvs: list) -> bytes:
    body = b"GGUF" + struct.pack("<I", 3)
    body += struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    return body + b"".join(kvs)


_MTP_GGUF_BYTES = _build_gguf([
    _kv_str("general.architecture", "qwen35moe"),
    _kv_u32("qwen35moe.block_count", 1),
    _kv_u32("qwen35moe.full_attention_interval", 0),
    _kv_u32("qwen35moe.attention.head_count_kv", 4),
    _kv_u32("qwen35moe.attention.key_length", 256),
    _kv_u32("qwen35moe.attention.value_length", 256),
    _kv_u32("qwen35moe.nextn_predict_layers", 1),
])
_SHA = "b" * 64
_GGUF_BODY_BYTES = 19 * 1024 * 1024 * 1024
_CTX = 250_000


def _seed_blob(boot):
    blob_path = boot.storage.blob_store_path / "sha256" / _SHA[:2]
    blob_path.mkdir(parents=True, exist_ok=True)
    (blob_path / _SHA).write_bytes(_MTP_GGUF_BYTES)


def _seed_manifest(boot, model_tag: str, *, spec_type: "str | None"):
    flags = {
        "split_mode": "none", "main_gpu": 0, "parallel": 1,
        "ctx_size": _CTX, "cache_type_k": "f16",
        # Explicit, NOT the manifest-level default (cache_type_v defaults
        # to "turbo3" when unset, per manifest.py's per-model flag
        # defaults) -- pinned to f16 so this test's independently-computed
        # "expected" delta isn't silently comparing against a different
        # quant than the real pipeline resolves.
        "cache_type_v": "f16",
    }
    if spec_type is not None:
        flags["spec_type"] = spec_type
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": _SHA,
        "gguf_size_bytes": _GGUF_BODY_BYTES,
        "context_size": _CTX,
        "arch": "qwen35moe",
        "llama_server_flags": flags,
    }))


class TestReadModelFootprintMtpDraftWiring:
    async def test_spec_type_draft_mtp_adds_the_draft_term_to_need(self, tmp_path):
        """The REAL end-to-end path: a manifest YAML with spec_type:
        draft-mtp, backed by a real (synthetic) GGUF blob carrying
        nextn_predict_layers=1, must produce a strictly larger `need` than
        the identical manifest without spec_type set -- and the exact
        delta must match calling estimate_kv_cache_mib directly with the
        same dims/ctx, proving _read_model_footprint's own
        mtp_draft_active derivation (the "mtp" in spec_type.lower() check)
        is wired correctly, not just present in the source."""
        boot, runtime = _boot_runtime(tmp_path)
        _seed_blob(boot)
        _seed_manifest(boot, "mtp-model", spec_type="draft-mtp")
        _seed_manifest(boot, "plain-model", spec_type=None)
        mgr = TurbohaulManager(boot, runtime, **_mocks())

        need_mtp, *_ = mgr._read_model_footprint("mtp-model")
        need_plain, *_ = mgr._read_model_footprint("plain-model")

        assert need_mtp > need_plain, (need_mtp, need_plain)

        from turbohaul._gguf_meta import KVDims
        dims = KVDims(arch="qwen35moe", block_count=1, full_attention_interval=0,
                      n_head_kv=4, key_length=256, value_length=256,
                      nextn_predict_layers=1)
        expected_off = estimate_kv_cache_mib(_CTX, _GGUF_BODY_BYTES, "f16", "f16",
                                             attn_dims=dims, mtp_draft_active=False)
        expected_on = estimate_kv_cache_mib(_CTX, _GGUF_BODY_BYTES, "f16", "f16",
                                            attn_dims=dims, mtp_draft_active=True)
        assert need_mtp - need_plain == expected_on - expected_off

        await mgr.shutdown()

    # NOTE: a case-insensitivity test for the "mtp" in spec_type.lower()
    # check (mirroring _engine_fingerprint's convention) is NOT constructible
    # through a real manifest: manifest.py's validator enforces spec_type
    # as a strict enum of exactly {"draft-mtp"} (case-sensitive), so a
    # mixed-case value is rejected by Manifest(**data) before
    # _read_model_footprint ever runs. The .lower() call is defensive
    # consistency with the existing convention, not something exercisable
    # via today's validated inputs -- see
    # the MTP draft-KV test for the safety.py-level coverage of
    # the "mtp" in spec_type" check pattern used at the derivation site.
