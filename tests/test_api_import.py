"""Tests for /api/import + DELETE /api/delete (v0.2 §9.2 + §12.1)."""
import hashlib
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.import_ import GGUF_MAGIC
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
        yield app, client, storage_root


def _make_gguf_file(path: Path, body: bytes = b"") -> bytes:
    """Write a file starting with GGUF magic + body. Returns full contents."""
    contents = GGUF_MAGIC + body
    path.write_bytes(contents)
    return contents


class TestImportValidation:
    def test_import_400_missing_path(self, app_test):
        app, client, _ = app_test
        r = client.post("/api/import", json={})
        assert r.status_code == 400

    def test_import_400_non_absolute(self, app_test):
        app, client, _ = app_test
        r = client.post("/api/import", json={"path": "relative/path"})
        assert r.status_code == 400
        assert "absolute" in r.text

    def test_import_400_etc_denied(self, app_test):
        app, client, _ = app_test
        r = client.post("/api/import", json={"path": "/etc/passwd"})
        assert r.status_code == 400
        assert "denied prefix" in r.text

    def test_import_400_proc_denied(self, app_test):
        app, client, _ = app_test
        r = client.post("/api/import", json={"path": "/proc/self/environ"})
        assert r.status_code == 400

    def test_import_400_root_denied(self, app_test):
        app, client, _ = app_test
        r = client.post("/api/import", json={"path": "/root/.config/nc_claude.env"})
        assert r.status_code == 400

    def test_import_400_escape_via_traversal(self, app_test):
        app, client, storage = app_test
        # Path under import_allowed_root but contains ..
        bad_path = str(storage / "import-staging" / ".." / ".." / "etc-shadow")
        r = client.post("/api/import", json={"path": bad_path})
        assert r.status_code == 400

    def test_import_400_nonexistent(self, app_test):
        app, client, storage = app_test
        r = client.post(
            "/api/import",
            json={"path": str(storage / "import-staging" / "missing.gguf")},
        )
        assert r.status_code == 400

    def test_import_400_symlink_rejected(self, app_test):
        app, client, storage = app_test
        target = storage / "import-staging" / "real.gguf"
        _make_gguf_file(target)
        symlink = storage / "import-staging" / "link.gguf"
        os.symlink(target, symlink)
        r = client.post("/api/import", json={"path": str(symlink)})
        assert r.status_code == 400
        assert "symlink" in r.text.lower()

    def test_import_400_no_gguf_magic(self, app_test):
        app, client, storage = app_test
        f = storage / "import-staging" / "fake.gguf"
        f.write_bytes(b"NOT A GGUF FILE")
        r = client.post("/api/import", json={"path": str(f)})
        assert r.status_code == 400
        assert "GGUF" in r.text


class TestImportHappyPath:
    def test_import_succeeds(self, app_test):
        app, client, storage = app_test
        body = b"weights" * 1000
        f = storage / "import-staging" / "model.gguf"
        contents = _make_gguf_file(f, body)
        expected_sha = hashlib.sha256(contents).hexdigest()
        r = client.post("/api/import", json={"path": str(f)})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["sha256"] == expected_sha
        assert out["bytes_written"] == len(contents)
        assert out["status"] == "complete"

    def test_import_with_expected_sha_pass(self, app_test):
        app, client, storage = app_test
        f = storage / "import-staging" / "m.gguf"
        contents = _make_gguf_file(f, b"data")
        expected = hashlib.sha256(contents).hexdigest()
        r = client.post("/api/import", json={"path": str(f), "expected_sha256": expected})
        assert r.status_code == 200

    def test_import_with_wrong_sha_fails(self, app_test):
        app, client, storage = app_test
        f = storage / "import-staging" / "m.gguf"
        _make_gguf_file(f, b"data")
        r = client.post(
            "/api/import", json={"path": str(f), "expected_sha256": "f" * 64}
        )
        assert r.status_code == 400


class TestDeleteRoute:
    def test_delete_existing_blob(self, app_test):
        app, client, storage = app_test
        # Import first to get a blob
        f = storage / "import-staging" / "m.gguf"
        contents = _make_gguf_file(f, b"to-delete")
        r1 = client.post("/api/import", json={"path": str(f)})
        sha = r1.json()["sha256"]
        # Delete by sha256
        r2 = client.request("DELETE", "/api/delete", json={"sha256": sha})
        assert r2.status_code == 200
        assert r2.json()["status"] == "deleted"
        # 2nd delete → 404
        r3 = client.request("DELETE", "/api/delete", json={"sha256": sha})
        assert r3.status_code == 404

    def test_delete_with_digest_format(self, app_test):
        app, client, storage = app_test
        f = storage / "import-staging" / "m.gguf"
        contents = _make_gguf_file(f, b"y")
        r1 = client.post("/api/import", json={"path": str(f)})
        sha = r1.json()["sha256"]
        r2 = client.request(
            "DELETE", "/api/delete", json={"digest": "sha256:" + sha}
        )
        assert r2.status_code == 200

    def test_delete_400_missing_sha(self, app_test):
        app, client, _ = app_test
        r = client.request("DELETE", "/api/delete", json={})
        assert r.status_code == 400

    def test_delete_404_unknown(self, app_test):
        app, client, _ = app_test
        r = client.request("DELETE", "/api/delete", json={"sha256": "f" * 64})
        assert r.status_code == 404
