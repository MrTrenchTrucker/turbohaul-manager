"""Tests for PUT /api/config split ( - v0.2 §7.1)."""
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


@pytest.fixture
def app_test(tmp_path):
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
            llama_server_binary=tmp_path / "fake",
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


class TestPutConfigRuntime:
    def test_put_queue_grace_seconds(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"queue": {"grace_seconds": 60}})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["current"]["queue"]["grace_seconds"] == 60
        # Verify manager's grace timer config was refreshed
        mgr = app.state.manager
        assert mgr.grace.grace_seconds == 60

    def test_put_queue_idle_hot(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"queue": {"idle_hot_load_seconds": 240}})
        assert r.status_code == 200
        mgr = app.state.manager
        assert mgr.idle.idle_seconds == 240

    def test_put_pull_concurrency(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"pull": {"pull_concurrency": 4}})
        assert r.status_code == 200
        mgr = app.state.manager
        assert mgr.runtime.pull.pull_concurrency == 4

    def test_get_after_put_reflects_change(self, app_test):
        app, client = app_test
        client.put("/api/config", json={"queue": {"grace_seconds": 45}})
        r = client.get("/api/config")
        body = r.json()
        assert body["queue"]["grace_seconds"] == 45


class TestPutConfigBootForbidden:
    def test_put_server_403(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"server": {"port": 11500}})
        assert r.status_code == 403
        assert "BOOT-ONLY" in r.text or "boot-only" in r.text.lower()

    def test_put_storage_403(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"storage": {"blob_store_path": "/tmp/evil"}})
        assert r.status_code == 403

    def test_put_runtime_paths_403(self, app_test):
        """The killer attack: changing runtime.llama_server_binary → RCE primitive (Security F7)."""
        app, client = app_test
        r = client.put(
            "/api/config",
            json={"runtime": {"llama_server_binary": "/tmp/evil.sh"}},
        )
        assert r.status_code == 403

    def test_put_ui_403(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"ui": {"static_path": "/etc"}})
        assert r.status_code == 403


class TestPutConfigBadInput:
    def test_unknown_section_400(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"evil_section": {"x": 1}})
        assert r.status_code == 400

    def test_non_object_section_400(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"queue": "not-an-object"})
        assert r.status_code == 400

    def test_invalid_value_400(self, app_test):
        app, client = app_test
        # grace_seconds has bound 0..3600
        r = client.put("/api/config", json={"queue": {"grace_seconds": -1}})
        assert r.status_code == 400


class TestPutConfigUnknownFieldInSection:
    def test_unknown_field_in_queue_rejected(self, app_test):
        app, client = app_test
        # QueueConfig has extra="forbid" so unknown subfield → 400
        r = client.put("/api/config", json={"queue": {"evil_field": 1}})
        assert r.status_code == 400
