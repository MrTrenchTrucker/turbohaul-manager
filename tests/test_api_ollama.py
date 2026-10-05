"""Tests for Ollama-compat read-only routes."""
import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import turbohaul.api.ollama as ollama_module
from turbohaul import blob_store
from turbohaul.api.main import create_app
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
from turbohaul.manifest import Manifest, ModelManifest, write_manifest_atomic

from tests.test_gguf_modelcard import _build_gguf, _kv_str, _kv_u32, _kv_u64


SAMPLE_SHA = "f" * 64
SECOND_SHA = "e" * 64


def _write_blob(blobs_root, sha256: str, content: bytes) -> None:
    path = blob_store.blob_path(blobs_root, sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture(autouse=True)
def _clear_model_card_cache():
    """The model-card cache is keyed by gguf_blob_sha256 and is
    meant to be safe indefinitely in production (content-addressed = immutable
    blobs), but tests reuse the same fake sha strings across different
    tmp_path blob roots -- clear between tests so one test's fixture can't
    leak a cached ModelCard into another's assertions."""
    ollama_module._MODEL_CARD_CACHE.clear()
    yield
    ollama_module._MODEL_CARD_CACHE.clear()


@pytest.fixture
def app_with_manifests(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    manifests_path = storage_root / "manifests"
    manifests_path.mkdir()
    (storage_root / "import-staging").mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=manifests_path,
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

    # Pre-populate with two manifests
    write_manifest_atomic(
        manifests_path,
        ModelManifest(
            model_tag="qwen3.6-35b-moe",
            display_name="Qwen 3.6 35B-A3B MoE",
            description="MoE Q4",
            gguf_blob_sha256=SAMPLE_SHA,
            gguf_size_bytes=22_000_000_000,
            context_size=131072,
            expected_vram_bytes=22_500_000_000,
            llama_server_flags={"ctx_size": 131072, "n_gpu_layers": 999},
        ),
    )
    write_manifest_atomic(
        manifests_path,
        ModelManifest(
            model_tag="qwen-coder",
            display_name="Qwen Coder",
            gguf_blob_sha256=SECOND_SHA,
            gguf_size_bytes=15_000_000_000,
            context_size=32768,
            expected_vram_bytes=16_000_000_000,
        ),
    )

    # Real GGUF-header bytes for the MoE model, so the enrichment
    # path is exercised end-to-end (not hand-fed). qwen-coder's blob is
    # deliberately ABSENT -- exercises the missing-blob -> null-fields-but-200
    # path (HARD requirement: one model's bad blob never breaks the list).
    _write_blob(storage_root / "blobs", SAMPLE_SHA, _build_gguf([
        _kv_str("general.architecture", "qwen2moe"),
        _kv_u64("general.parameter_count", 35_000_000_000),
        _kv_str("general.size_label", "35B"),
        _kv_u32("qwen2moe.expert_count", 8),
    ]))

    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


class TestApiTags:
    def test_tags_returns_installed_models(self, app_with_manifests):
        app, client = app_with_manifests
        r = client.get("/api/tags")
        assert r.status_code == 200
        body = r.json()
        names = {m["name"] for m in body["models"]}
        assert "qwen3.6-35b-moe" in names
        assert "qwen-coder" in names
        # Verify the response shape
        moe = next(m for m in body["models"] if m["name"] == "qwen3.6-35b-moe")
        assert moe["digest"].startswith("sha256:")
        assert moe["size"] == 22_000_000_000
        assert moe["details"]["format"] == "gguf"
        assert moe["details"]["context_length"] == 131072
        assert moe["revision"] == 1
        # New additive fields, read from a real GGUF-header fixture
        # written in the blob store (not hand-fed).
        assert moe["modified_at"] is not None
        assert moe["details"]["parameter_count"] == 35_000_000_000
        assert moe["details"]["size_label"] == "35B"
        assert moe["details"]["architecture"] == "qwen2moe"
        assert moe["details"]["is_moe"] is True
        assert moe["details"]["expert_count"] == 8
        assert moe["details"]["modality"] == "text"  # no mmproj, no vision KV

    def test_tags_null_new_fields_on_missing_blob(self, app_with_manifests):
        """Hard requirement: a missing blob nulls that model's new
        fields but never breaks the list -- still 200, existing fields on
        THIS model and every other model stay exactly as before."""
        app, client = app_with_manifests
        r = client.get("/api/tags")
        assert r.status_code == 200
        body = r.json()
        coder = next(m for m in body["models"] if m["name"] == "qwen-coder")
        # Existing fields byte-identical
        assert coder["size"] == 15_000_000_000
        assert coder["digest"] == "sha256:" + SECOND_SHA
        assert coder["details"]["context_length"] == 32768
        # New fields degrade to null; list still 200
        assert coder["details"]["parameter_count"] is None
        assert coder["details"]["size_label"] is None
        assert coder["details"]["architecture"] is None
        assert coder["details"]["is_moe"] is None
        assert coder["details"]["expert_count"] is None
        assert coder["details"]["modality"] == "text"
        # modified_at still populated -- the manifest .yaml itself DOES exist;
        # only the GGUF blob is missing.
        assert coder["modified_at"] is not None
        # The moe model (which DOES have a real blob) is unaffected by
        # qwen-coder's missing blob.
        moe = next(m for m in body["models"] if m["name"] == "qwen3.6-35b-moe")
        assert moe["details"]["is_moe"] is True

    def test_tags_missing_blob_none_not_cached_repull_reenriches(self, app_with_manifests):
        """A miss (None) is NOT
        cached, so once the previously-absent blob is (re)staged at the SAME
        content-addressed sha, the next /api/tags re-reads it and the model's new
        fields populate -- no stale null-until-restart after a delete + re-pull."""
        app, client = app_with_manifests
        # 1) qwen-coder's blob is absent -> new fields null
        coder = next(m for m in client.get("/api/tags").json()["models"] if m["name"] == "qwen-coder")
        assert coder["details"]["parameter_count"] is None
        assert coder["details"]["is_moe"] is None
        # 2) stage the blob at qwen-coder's exact sha (SECOND_SHA) with a real GGUF header
        blobs_root = app.state.manager.boot.storage.blob_store_path
        _write_blob(blobs_root, SECOND_SHA, _build_gguf([
            _kv_str("general.architecture", "qwen2"),
            _kv_u64("general.parameter_count", 7_000_000_000),
            _kv_str("general.size_label", "7B"),
        ]))
        # 3) next call re-reads (None was never cached) -> enriched, not stale-null
        coder2 = next(m for m in client.get("/api/tags").json()["models"] if m["name"] == "qwen-coder")
        assert coder2["details"]["parameter_count"] == 7_000_000_000
        assert coder2["details"]["size_label"] == "7B"
        assert coder2["details"]["architecture"] == "qwen2"

    async def test_tags_parse_offloaded_does_not_block_event_loop(self, app_with_manifests, monkeypatch):
        """The per-model GGUF-header parse (which
        on a cold cache walks each model's 100K-300K-entry tokenizer arrays) must run
        OFF the event loop via asyncio.to_thread, so a cold /api/tags does NOT stall
        other in-flight work on the same loop -- the no-regression
        guarantee for concurrent streaming completions.

        Discriminator: a background ticker (an ordinary awaiting coroutine) must keep
        advancing WHILE /api/tags is parsing. Offloaded -> the loop is free and the
        ticker advances many times; run inline -> the blocking parse freezes the loop
        and the ticker cannot tick. (Verified vacuity-free: reverting the to_thread to
        an inline call drops ticks_during to ~0 and fails this assert.)"""
        app, _ = app_with_manifests

        def _slow_blocking(m, blobs_root):
            time.sleep(0.4)  # stand-in for the real tokenizer-array walk (blocking, sync)
            return {
                "parameter_count": None, "size_label": None, "architecture": None,
                "is_moe": None, "expert_count": None, "modality": "text",
            }
        monkeypatch.setattr(ollama_module, "_model_card_details", _slow_blocking)

        ticks = 0

        async def _ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://smoke") as client:
            tk = asyncio.create_task(_ticker())
            await asyncio.sleep(0.03)  # let the ticker get going
            before = ticks
            resp = await client.get("/api/tags")  # ~0.4s x N models of (offloaded) parse
            during = ticks - before
            tk.cancel()
            assert resp.status_code == 200
            # Offloaded: loop free through the whole parse -> dozens of ticks. Inline:
            # loop frozen in time.sleep -> ~0 ticks. Bound is generous vs 0.4s/model.
            assert during > 8, f"event loop froze during /api/tags parse (only {during} ticks) -> parse not offloaded"

    def test_tags_empty_when_no_manifests(self, tmp_path):
        boot = BootConfig(
            server=ServerConfig(),
            storage=StorageConfig(
                blob_store_path=tmp_path / "b",
                manifests_path=tmp_path / "m",
                import_allowed_root=tmp_path / "i",
                state_db_path=tmp_path / "s.sqlite",
            ),
            runtime=RuntimePathsConfig(
                llama_server_binary=tmp_path / "fake",
                default_port_base=59500,
            ),
            ui=UIConfig(static_path=tmp_path / "ui"),
        )
        runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
        (tmp_path / "m").mkdir()
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as c:
            r = c.get("/api/tags")
            assert r.status_code == 200
            assert r.json() == {"models": []}


class TestApiShow:
    def test_show_returns_manifest_details(self, app_with_manifests):
        app, client = app_with_manifests
        r = client.get("/api/show?name=qwen3.6-35b-moe")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "qwen3.6-35b-moe"
        assert body["display_name"] == "Qwen 3.6 35B-A3B MoE"
        assert body["context_length"] == 131072
        assert body["expected_vram_bytes"] == 22_500_000_000
        assert body["llama_server_flags"]["ctx_size"] == 131072

    def test_show_includes_new_additive_fields(self, app_with_manifests):
        """get_show mirrors get_tags' new fields under a NEW
        "details" key (get_show was previously fully flat -- the new fields
        deliberately mirror get_tags' field-paths rather than flatten)."""
        app, client = app_with_manifests
        r = client.get("/api/show?name=qwen3.6-35b-moe")
        assert r.status_code == 200
        body = r.json()
        # Existing flat fields untouched
        assert body["context_length"] == 131072
        assert body["expected_vram_bytes"] == 22_500_000_000
        assert body["llama_server_flags"]["ctx_size"] == 131072
        # New fields
        assert body["modified_at"] is not None
        assert body["details"]["parameter_count"] == 35_000_000_000
        assert body["details"]["size_label"] == "35B"
        assert body["details"]["is_moe"] is True
        assert body["details"]["expert_count"] == 8
        assert body["details"]["modality"] == "text"

    def test_show_null_new_fields_on_missing_blob(self, app_with_manifests):
        app, client = app_with_manifests
        r = client.get("/api/show?name=qwen-coder")
        assert r.status_code == 200
        body = r.json()
        assert body["context_length"] == 32768  # existing field untouched
        assert body["details"]["parameter_count"] is None
        assert body["details"]["is_moe"] is None
        assert body["details"]["modality"] == "text"

    def test_show_404_for_unknown_model(self, app_with_manifests):
        app, client = app_with_manifests
        r = client.get("/api/show?name=nonexistent-model")
        assert r.status_code == 404

    def test_show_400_for_invalid_tag(self, app_with_manifests):
        app, client = app_with_manifests
        r = client.get("/api/show?name=../etc/passwd")
        assert r.status_code in (400, 422)  # 422 = FastAPI validation; 400 = our reject


BROKEN_YAML_CONTENT = "model_tag: broken-yaml\n  bad indent: [unclosed\n"
STALE_SCHEMA_CONTENT = (
    "model_tag: stale-schema\n"
    "display_name: Old Model\n"
    "leaked_marker_value_xyz789: present\n"
)  # missing required gguf_blob_sha256 -> pydantic ValidationError; the
# leaked_marker_value_xyz789 field exists specifically so a test can assert
# it does NOT leak into any response/log (str(e) would echo it verbatim).
NOT_A_MAPPING_CONTENT = "- just\n- a\n- list\n"


class TestManifestErrorHardening:
    """read_manifest fails 4 ways (FileNotFoundError,
    ManifestValidationError, yaml.YAMLError, pydantic ValidationError) but
    /api/tags and /api/show only caught the first 2, so a corrupt-YAML or
    stale-schema manifest produced an UNHANDLED 500. LIST (/api/tags) must
    keep skipping+200 (now also logging a WARNING so the skip isn't silent);
    SINGLE (/api/show) must surface a handled 500 with a fixed message,
    never str(e)."""

    def _boot_runtime(self, tmp_path):
        storage_root = tmp_path / "state"
        storage_root.mkdir()
        (storage_root / "blobs").mkdir()
        manifests_path = storage_root / "manifests"
        manifests_path.mkdir()
        (storage_root / "import-staging").mkdir()
        boot = BootConfig(
            server=ServerConfig(),
            storage=StorageConfig(
                blob_store_path=storage_root / "blobs",
                manifests_path=manifests_path,
                import_allowed_root=storage_root / "import-staging",
                state_db_path=storage_root / "state.sqlite",
            ),
            runtime=RuntimePathsConfig(
                llama_server_binary=tmp_path / "fake", default_port_base=59500,
            ),
            ui=UIConfig(static_path=tmp_path / "ui"),
        )
        runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
        return manifests_path, boot, runtime

    def _app_with_corrupt_manifest(self, tmp_path, tag, content):
        manifests_path, boot, runtime = self._boot_runtime(tmp_path)
        (manifests_path / f"{tag}.yaml").write_text(content)
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        return app

    # --- /api/tags: skip + 200, now with a WARNING log ---

    def test_tags_skips_broken_yaml_200(self, tmp_path, caplog):
        app = self._app_with_corrupt_manifest(tmp_path, "broken-yaml", BROKEN_YAML_CONTENT)
        with caplog.at_level("WARNING"), TestClient(app) as client:
            r = client.get("/api/tags")
        assert r.status_code == 200
        assert r.json() == {"models": []}

    def test_tags_skips_stale_schema_200(self, tmp_path, caplog):
        app = self._app_with_corrupt_manifest(tmp_path, "stale-schema", STALE_SCHEMA_CONTENT)
        with caplog.at_level("WARNING"), TestClient(app) as client:
            r = client.get("/api/tags")
        assert r.status_code == 200
        assert r.json() == {"models": []}

    def test_tags_skip_is_not_silent_logs_warning_with_type_not_message(self, tmp_path, caplog):
        """Required addition: the skip must emit a WARNING naming
        the tag and exception TYPE, never str(exc) (leak rule)."""
        app = self._app_with_corrupt_manifest(tmp_path, "stale-schema", STALE_SCHEMA_CONTENT)
        with caplog.at_level("WARNING"), TestClient(app) as client:
            client.get("/api/tags")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "stale-schema" in msg
        assert "ValidationError" in msg
        # Leak guard: the message must name the TYPE only, never the raw
        # manifest content str(exc) would include.
        assert "leaked_marker_value_xyz789" not in msg

    def test_tags_skips_non_mapping_200_unchanged(self, tmp_path):
        """Regression: the shape already caught before the hardening still works."""
        app = self._app_with_corrupt_manifest(tmp_path, "not-a-mapping", NOT_A_MAPPING_CONTENT)
        with TestClient(app) as client:
            r = client.get("/api/tags")
        assert r.status_code == 200
        assert r.json() == {"models": []}

    # --- /api/show: handled 500, fixed message, never str(e) ---

    def test_show_500_broken_yaml_fixed_message(self, tmp_path):
        app = self._app_with_corrupt_manifest(tmp_path, "broken-yaml", BROKEN_YAML_CONTENT)
        with TestClient(app) as client:
            r = client.get("/api/show?name=broken-yaml")
        assert r.status_code == 500
        assert r.json()["detail"] == "manifest for 'broken-yaml' is present but unreadable"

    def test_show_500_stale_schema_fixed_message_no_leak(self, tmp_path):
        app = self._app_with_corrupt_manifest(tmp_path, "stale-schema", STALE_SCHEMA_CONTENT)
        with TestClient(app) as client:
            r = client.get("/api/show?name=stale-schema")
        assert r.status_code == 500
        assert r.json()["detail"] == "manifest for 'stale-schema' is present but unreadable"
        # Leak guard: neither pydantic's raw error text nor the manifest's
        # own field values may reach the client.
        assert "leaked_marker_value_xyz789" not in r.text
        assert "input_value" not in r.text

    def test_show_400_non_mapping_unchanged(self, tmp_path):
        """Regression: the existing 400 branch (str(e), no paths) untouched."""
        app = self._app_with_corrupt_manifest(tmp_path, "not-a-mapping", NOT_A_MAPPING_CONTENT)
        with TestClient(app) as client:
            r = client.get("/api/show?name=not-a-mapping")
        assert r.status_code == 400


class TestModelCardModalityComposition:
    """Modality = "vision" iff
    manifest.mmproj_blob_sha256 is set. An earlier version also ORed in a bare
    GGUF vision-marker key too, but --mmproj only ever reaches llama-server
    from the manifest's own projector blob (manager.py, in two
    places), and the raw `mmproj` flag is denied outright
    (manifest.py) -- so a model advertised as vision via the GGUF key
    alone could not actually be served with vision. That OR clause is
    deleted; the vision-key-alone case below now asserts "text", not
    "vision". Each case still runs in its own minimal app/tmp_path (not the
    shared app_with_manifests fixture) to keep each case isolated."""

    def _make_boot_runtime(self, tmp_path):
        storage_root = tmp_path / "state"
        storage_root.mkdir()
        (storage_root / "blobs").mkdir()
        manifests_path = storage_root / "manifests"
        manifests_path.mkdir()
        (storage_root / "import-staging").mkdir()
        boot = BootConfig(
            server=ServerConfig(),
            storage=StorageConfig(
                blob_store_path=storage_root / "blobs",
                manifests_path=manifests_path,
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
        return storage_root, manifests_path, boot, runtime

    def test_vision_via_manifest_mmproj_field(self, tmp_path):
        storage_root, manifests_path, boot, runtime = self._make_boot_runtime(tmp_path)
        sha = "1" * 64
        write_manifest_atomic(
            manifests_path,
            ModelManifest(
                model_tag="gemma4-vl",
                gguf_blob_sha256=sha,
                mmproj_blob_sha256="2" * 64,
                gguf_size_bytes=1000,
            ),
        )
        # GGUF itself has NO vision marker -- modality must come from mmproj alone.
        _write_blob(storage_root / "blobs", sha, _build_gguf([
            _kv_str("general.architecture", "gemma4"),
        ]))
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as client:
            r = client.get("/api/tags")
        m = r.json()["models"][0]
        assert m["details"]["modality"] == "vision"

    def test_vision_via_gguf_clip_key_no_mmproj(self, tmp_path):
        """Re-pointed (was asserting "vision" -- that assertion
        WAS the OR clause, and this fix deletes it on purpose; see
        the class docstring for why). No mmproj_blob_sha256 set, GGUF alone
        carries a clip.* vision-marker key: --mmproj can never be launched
        for this model (manager.py only ever derives it from
        mmproj_blob_sha256), so it must not be advertised as vision-capable
        even though the GGUF header itself looks like a vision model."""
        storage_root, manifests_path, boot, runtime = self._make_boot_runtime(tmp_path)
        sha = "3" * 64
        write_manifest_atomic(
            manifests_path,
            ModelManifest(
                model_tag="fused-vision",
                gguf_blob_sha256=sha,
                gguf_size_bytes=1000,
            ),
        )
        # No mmproj_blob_sha256 set -- the GGUF's own clip.* key alone must
        # NOT be enough to advertise vision.
        _write_blob(storage_root / "blobs", sha, _build_gguf([
            _kv_str("general.architecture", "llava"),
            _kv_u32("clip.vision_feature_layer", 1),
        ]))
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as client:
            r = client.get("/api/tags")
        m = r.json()["models"][0]
        assert m["details"]["modality"] == "text"

    def test_text_only_when_neither_marker_present(self, tmp_path):
        storage_root, manifests_path, boot, runtime = self._make_boot_runtime(tmp_path)
        sha = "4" * 64
        write_manifest_atomic(
            manifests_path,
            ModelManifest(
                model_tag="plain-dense",
                gguf_blob_sha256=sha,
                gguf_size_bytes=1000,
            ),
        )
        _write_blob(storage_root / "blobs", sha, _build_gguf([
            _kv_str("general.architecture", "llama"),
        ]))
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as client:
            r = client.get("/api/tags")
        m = r.json()["models"][0]
        assert m["details"]["modality"] == "text"
