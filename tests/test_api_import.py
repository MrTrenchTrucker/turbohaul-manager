"""Tests for /api/import + DELETE /api/delete (spec §9.2 + §12.1)."""
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
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
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
        r = client.post("/api/import", json={"path": "/root/private/secrets.env"})
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


class TestListBlobsRoute:
    """GET /api/blobs (blob listing).

    Pure enumeration of the blob store on disk -- no manifest join. Required
    coverage: (a) an existing blob is listed, (b) total == len(blobs),
    (c) an unreadable/vanished entry doesn't 500 the whole listing.
    """

    def test_lists_existing_blob(self, app_test):
        app, client, storage = app_test
        f = storage / "import-staging" / "m.gguf"
        contents = _make_gguf_file(f, b"weights-for-listing")
        expected_sha = hashlib.sha256(contents).hexdigest()
        r1 = client.post("/api/import", json={"path": str(f)})
        assert r1.status_code == 200, r1.text

        r = client.get("/api/blobs")
        assert r.status_code == 200, r.text
        out = r.json()
        digests = {b["digest"] for b in out["blobs"]}
        assert expected_sha in digests
        entry = next(b for b in out["blobs"] if b["digest"] == expected_sha)
        assert entry["size_bytes"] == len(contents)
        assert "sha256:" not in entry["digest"]
        assert entry["description"] is None

    def test_empty_store_returns_empty_list(self, app_test):
        app, client, _ = app_test
        r = client.get("/api/blobs")
        assert r.status_code == 200
        assert r.json() == {"blobs": [], "total": 0}

    def test_total_matches_blobs_length(self, app_test):
        app, client, storage = app_test
        for i in range(3):
            f = storage / "import-staging" / f"m{i}.gguf"
            _make_gguf_file(f, f"weights-{i}".encode())
            r = client.post("/api/import", json={"path": str(f)})
            assert r.status_code == 200, r.text

        r = client.get("/api/blobs")
        out = r.json()
        assert out["total"] == len(out["blobs"])
        assert out["total"] == 3

    def test_vanished_entry_does_not_500_the_listing(self, app_test, monkeypatch):
        app, client, storage = app_test
        f = storage / "import-staging" / "m.gguf"
        contents = _make_gguf_file(f, b"stays-visible")
        expected_sha = hashlib.sha256(contents).hexdigest()
        r1 = client.post("/api/import", json={"path": str(f)})
        assert r1.status_code == 200

        # A second blob whose on-disk file we delete out from under the
        # route AFTER list_blobs() has already returned its digest, to
        # simulate a stat() raising ENOENT mid-listing (raced with a
        # concurrent delete) without needing real concurrency.
        from turbohaul.api import import_ as import_module

        real_list_blobs = import_module.list_blobs

        def fake_list_blobs(root):
            digests = real_list_blobs(root)
            return [*digests, "f" * 64]  # a digest with no backing file

        monkeypatch.setattr(import_module, "list_blobs", fake_list_blobs)

        r = client.get("/api/blobs")
        assert r.status_code == 200, r.text
        out = r.json()
        digests = {b["digest"] for b in out["blobs"]}
        assert expected_sha in digests
        assert "f" * 64 not in digests
        assert out["total"] == len(out["blobs"])


class TestBlobDescriptionRoute:
    """PUT /api/blobs/{digest}/description (blob description).

    Required coverage: round-trip PUT->GET, unset reads as null, empty
    string clears the key, unknown digest 404s, malformed digest 400s,
    over-length 400s, a missing/corrupt metadata file doesn't 500 the
    listing, and delete PRUNES the entry.
    """

    def _import_one(self, client, storage, body=b"weights-desc"):
        f = storage / "import-staging" / "d.gguf"
        contents = _make_gguf_file(f, body)
        sha = hashlib.sha256(contents).hexdigest()
        r = client.post("/api/import", json={"path": str(f)})
        assert r.status_code == 200, r.text
        return sha

    def test_round_trip_put_then_get(self, app_test):
        app, client, storage = app_test
        sha = self._import_one(client, storage)

        r1 = client.put(f"/api/blobs/{sha}/description", json={"description": "A fine model."})
        assert r1.status_code == 200, r1.text
        assert r1.json() == {"digest": sha, "description": "A fine model."}

        r2 = client.get("/api/blobs")
        entry = next(b for b in r2.json()["blobs"] if b["digest"] == sha)
        assert entry["description"] == "A fine model."

    def test_unset_description_reads_as_null(self, app_test):
        app, client, storage = app_test
        self._import_one(client, storage)
        r = client.get("/api/blobs")
        assert r.json()["blobs"][0]["description"] is None

    def test_empty_string_clears_the_key(self, app_test):
        app, client, storage = app_test
        sha = self._import_one(client, storage)
        client.put(f"/api/blobs/{sha}/description", json={"description": "temporary"})
        r = client.put(f"/api/blobs/{sha}/description", json={"description": ""})
        assert r.status_code == 200, r.text
        assert r.json()["description"] is None

        from turbohaul.model_meta import read_model_meta
        meta = read_model_meta(app.state.manager.boot.storage.manifests_path)
        assert sha not in meta  # key actually gone, not stored as ""

    def test_null_description_clears_the_key(self, app_test):
        app, client, storage = app_test
        sha = self._import_one(client, storage)
        client.put(f"/api/blobs/{sha}/description", json={"description": "temporary"})
        r = client.put(f"/api/blobs/{sha}/description", json={"description": None})
        assert r.status_code == 200, r.text
        assert r.json()["description"] is None

    def test_unknown_digest_404(self, app_test):
        app, client, _ = app_test
        r = client.put(f"/api/blobs/{'a' * 64}/description", json={"description": "x"})
        assert r.status_code == 404

    def test_malformed_digest_400(self, app_test):
        app, client, _ = app_test
        r = client.put("/api/blobs/not-a-digest/description", json={"description": "x"})
        assert r.status_code == 400

    def test_missing_description_field_400(self, app_test):
        app, client, storage = app_test
        sha = self._import_one(client, storage)
        r = client.put(f"/api/blobs/{sha}/description", json={})
        assert r.status_code == 400

    def test_wrong_type_description_400(self, app_test):
        app, client, storage = app_test
        sha = self._import_one(client, storage)
        r = client.put(f"/api/blobs/{sha}/description", json={"description": 12345})
        assert r.status_code == 400

    def test_over_length_description_400(self, app_test):
        app, client, storage = app_test
        sha = self._import_one(client, storage)
        r = client.put(
            f"/api/blobs/{sha}/description", json={"description": "x" * 2001}
        )
        assert r.status_code == 400

    def test_max_length_description_accepted(self, app_test):
        app, client, storage = app_test
        sha = self._import_one(client, storage)
        r = client.put(
            f"/api/blobs/{sha}/description", json={"description": "x" * 2000}
        )
        assert r.status_code == 200, r.text

    def test_missing_metadata_file_does_not_500_the_listing(self, app_test):
        app, client, storage = app_test
        self._import_one(client, storage)
        r = client.get("/api/blobs")
        assert r.status_code == 200
        assert r.json()["blobs"][0]["description"] is None

    def test_corrupt_metadata_file_does_not_500_the_listing(self, app_test):
        app, client, storage = app_test
        sha = self._import_one(client, storage)
        meta_path = storage / "model_meta.json"
        meta_path.write_text("{not valid json")
        r = client.get("/api/blobs")
        assert r.status_code == 200, r.text
        entry = next(b for b in r.json()["blobs"] if b["digest"] == sha)
        assert entry["description"] is None

    def test_delete_prunes_the_metadata_entry(self, app_test):
        app, client, storage = app_test
        sha = self._import_one(client, storage)
        r1 = client.put(f"/api/blobs/{sha}/description", json={"description": "gone soon"})
        assert r1.status_code == 200

        from turbohaul.model_meta import read_model_meta
        meta_before = read_model_meta(app.state.manager.boot.storage.manifests_path)
        assert sha in meta_before

        r2 = client.request("DELETE", "/api/delete", json={"sha256": sha})
        assert r2.status_code == 200, r2.text

        meta_after = read_model_meta(app.state.manager.boot.storage.manifests_path)
        assert sha not in meta_after

    def test_metadata_store_lives_beside_manifests_not_inside_blobs(self, app_test):
        app, client, storage = app_test
        sha = self._import_one(client, storage)
        client.put(f"/api/blobs/{sha}/description", json={"description": "location check"})

        from turbohaul.model_meta import meta_path
        path = meta_path(app.state.manager.boot.storage.manifests_path)
        assert path == storage / "model_meta.json"
        assert path.exists()
        # And definitely not inside the content-addressed blob tree.
        blobs_root = app.state.manager.boot.storage.blob_store_path
        assert not str(path).startswith(str(blobs_root))
