"""Tests for /api/manifests CRUD routes ( - v0.2 §8.1 + §8.2)."""
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


SAMPLE_SHA = "abcdef01" + "0" * 56


@pytest.fixture
def app_blank(tmp_path):
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


def _valid_payload(tag="my-model"):
    return {
        "model_tag": tag,
        "display_name": "Test Model",
        "gguf_blob_sha256": SAMPLE_SHA,
        "gguf_size_bytes": 10_000_000_000,
        "context_size": 4096,
        "expected_vram_bytes": 11_000_000_000,
        "llama_server_flags": {"ctx_size": 4096, "n_gpu_layers": 999},
    }


class TestPutManifest:
    def test_create_new_manifest_no_if_match(self, app_blank):
        app, client = app_blank
        r = client.put("/api/manifests/my-model", json=_valid_payload("my-model"))
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["revision"] == 1
        assert r.headers["ETag"] == '"1"'

    def test_update_with_correct_if_match_increments(self, app_blank):
        app, client = app_blank
        client.put("/api/manifests/my-model", json=_valid_payload("my-model"))
        r2 = client.put(
            "/api/manifests/my-model",
            json={**_valid_payload("my-model"), "display_name": "Updated"},
            headers={"If-Match": '"1"'},
        )
        assert r2.status_code == 200
        assert r2.json()["revision"] == 2

    def test_update_with_wrong_if_match_412(self, app_blank):
        app, client = app_blank
        client.put("/api/manifests/my-model", json=_valid_payload("my-model"))
        r2 = client.put(
            "/api/manifests/my-model",
            json={**_valid_payload("my-model"), "display_name": "Stale"},
            headers={"If-Match": '"99"'},
        )
        assert r2.status_code == 412

    def test_update_without_if_match_412(self, app_blank):
        """Note: PUT without If-Match on existing manifest -> 412."""
        app, client = app_blank
        r1 = client.put("/api/manifests/my-model", json=_valid_payload("my-model"))
        assert r1.status_code == 200
        r2 = client.put(
            "/api/manifests/my-model",
            json={**_valid_payload("my-model"), "display_name": "NoEtag"},
        )
        assert r2.status_code == 412

    def test_reject_denied_flag_400(self, app_blank):
        app, client = app_blank
        payload = _valid_payload("my-model")
        payload["llama_server_flags"]["mmproj"] = "/etc/passwd"
        r = client.put("/api/manifests/my-model", json=payload)
        assert r.status_code == 400
        assert "denied" in r.text.lower() or "mmproj" in r.text.lower()

    def test_reject_unknown_flag_400(self, app_blank):
        app, client = app_blank
        payload = _valid_payload("my-model")
        payload["llama_server_flags"]["evil_unknown"] = "x"
        r = client.put("/api/manifests/my-model", json=payload)
        assert r.status_code == 400

    def test_reject_path_traversal_in_url_tag(self, app_blank):
        app, client = app_blank
        payload = _valid_payload("../etc/passwd")
        r = client.put("/api/manifests/..%2Fetc%2Fpasswd", json=payload)
        assert r.status_code in (400, 404)

    def test_tag_in_url_overrides_payload(self, app_blank):
        """URL tag is authoritative; payload model_tag is overridden."""
        app, client = app_blank
        payload = _valid_payload("payload-name")
        r = client.put("/api/manifests/url-name", json=payload)
        assert r.status_code == 200
        assert r.json()["model_tag"] == "url-name"


class TestGetManifest:
    def test_get_returns_manifest_with_etag(self, app_blank):
        app, client = app_blank
        client.put("/api/manifests/my-model", json=_valid_payload("my-model"))
        r = client.get("/api/manifests/my-model")
        assert r.status_code == 200
        assert r.headers["ETag"] == '"1"'
        body = r.json()
        assert body["model_tag"] == "my-model"

    def test_get_404_unknown(self, app_blank):
        app, client = app_blank
        r = client.get("/api/manifests/nonexistent")
        assert r.status_code == 404


class TestDeleteManifest:
    def test_delete_existing(self, app_blank):
        app, client = app_blank
        client.put("/api/manifests/my-model", json=_valid_payload("my-model"))
        r = client.delete("/api/manifests/my-model")
        assert r.status_code == 200
        # Subsequent get → 404
        r2 = client.get("/api/manifests/my-model")
        assert r2.status_code == 404

    def test_delete_404_unknown(self, app_blank):
        app, client = app_blank
        r = client.delete("/api/manifests/nonexistent")
        assert r.status_code == 404
