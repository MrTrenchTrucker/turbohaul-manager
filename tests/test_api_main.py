"""Tests for FastAPI app skeleton (v0.2 §9)."""
import pytest
from fastapi.testclient import TestClient

from turbohaul import __version__
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


@pytest.fixture
def app_and_client(tmp_path):
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
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    # In tests, skip auto-spawning the worker so we don't run the skeleton's
    # mark-cold-loop and contaminate state.sqlite mid-test.
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


class TestHealth:
    def test_health_returns_ok(self, app_and_client):
        app, client = app_and_client
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__


class TestStatus:
    def test_status_initial_empty(self, app_and_client):
        app, client = app_and_client
        r = client.get("/status")
        assert r.status_code == 200
        body = r.json()
        assert body["queue"]["acceptance_buffer_depth"] == 0
        assert body["queue"]["staging_queue_depth"] == 0
        assert body["active"] is None
        assert body["grace"] is None
        assert body["idle_hot"] is None
        assert body["parallel_slots"]["used"] == 0


class TestApiVersion:
    def test_api_version_payload(self, app_and_client):
        app, client = app_and_client
        r = client.get("/api/version")
        assert r.status_code == 200
        body = r.json()
        assert body["version"] == __version__
        assert body["api_compat"] == "ollama-superset"
        assert "Ollama-compatible" in body["user_agent"]
        # backend_sha_pinned should be False when empty in test config
        assert body["backend_sha_pinned"] is False


class TestApiConfig:
    def test_api_config_returns_split_view(self, app_and_client):
        app, client = app_and_client
        r = client.get("/api/config")
        assert r.status_code == 200
        body = r.json()
        # Boot sections
        assert "server" in body
        assert body["server"]["host"] == "127.0.0.1"  # never 0.0.0.0 default
        assert body["server"]["port"] == 11401
        assert "storage" in body
        assert "runtime" in body
        assert "ui" in body
        # Runtime sections
        assert "queue" in body
        assert body["queue"]["grace_seconds"] == 30  # v0.2 conservative default
        assert body["queue"]["idle_hot_load_seconds"] == 120
        assert "pull" in body
        assert body["pull"]["pull_url_https_only"] is True


class TestAppCreation:
    def test_app_has_manager_state(self, app_and_client):
        app, client = app_and_client
        assert hasattr(app.state, "manager")
        from turbohaul.manager import TurbohaulManager
        assert isinstance(app.state.manager, TurbohaulManager)

    def test_unknown_endpoint_404(self, app_and_client):
        app, client = app_and_client
        r = client.get("/nonexistent")
        assert r.status_code == 404
