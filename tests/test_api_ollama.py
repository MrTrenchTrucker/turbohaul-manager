"""Tests for Ollama-compat read-only routes ()."""
import pytest
from fastapi.testclient import TestClient

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
from turbohaul.manifest import Manifest, write_manifest_atomic


SAMPLE_SHA = "f" * 64
SECOND_SHA = "e" * 64


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
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())

    # Pre-populate with two manifests
    write_manifest_atomic(
        manifests_path,
        Manifest(
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
        Manifest(
            model_tag="qwen-coder",
            display_name="Qwen Coder",
            gguf_blob_sha256=SECOND_SHA,
            gguf_size_bytes=15_000_000_000,
            context_size=32768,
            expected_vram_bytes=16_000_000_000,
        ),
    )

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
        # Verify shape per v0.2 §9
        moe = next(m for m in body["models"] if m["name"] == "qwen3.6-35b-moe")
        assert moe["digest"].startswith("sha256:")
        assert moe["size"] == 22_000_000_000
        assert moe["details"]["format"] == "gguf"
        assert moe["details"]["context_length"] == 131072
        assert moe["revision"] == 1

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
        runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
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

    def test_show_404_for_unknown_model(self, app_with_manifests):
        app, client = app_with_manifests
        r = client.get("/api/show?name=nonexistent-model")
        assert r.status_code == 404

    def test_show_400_for_invalid_tag(self, app_with_manifests):
        app, client = app_with_manifests
        r = client.get("/api/show?name=../etc/passwd")
        assert r.status_code in (400, 422)  # 422 = FastAPI validation; 400 = our reject
