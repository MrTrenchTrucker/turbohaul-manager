"""Regression tests: one invalid manifest must not
blank or 500 the /api/manifests surface.

Root cause:
`read_manifest` surfaces a pydantic `ValidationError` when a stored manifest
violates the cross-field rule (reasoning_budget >= n_predict), and the
manifests API read paths only caught `ManifestValidationError` -- a sibling
ValueError, NOT a pydantic class. The result:

- GET /api/manifests (management list) -> 500 for the WHOLE list (the
  Models page fails to load and no model can be selected);
- GET/PATCH/restore-defaults /api/manifests/{tag} -> 500 on an invalid
  existing manifest instead of a 400.

The list endpoint's own comment states the intent -- 'One unreadable manifest
must not blank the whole management view' -- and the except tuple missed the
error class the validator actually raises.

Fix under test: pydantic ValidationError is caught at every read_manifest
site (list -> 'unreadable' row; per-tag reads -> 400 with detail).
"""
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
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


def _valid_payload(tag="good-model"):
    return {
        "model_tag": tag,
        "display_name": "Good Model",
        "gguf_blob_sha256": SAMPLE_SHA,
        "gguf_size_bytes": 10_000_000_000,
        "context_size": 4096,
        "expected_vram_bytes": 11_000_000_000,
        "llama_server_flags": {"ctx_size": 4096, "n_gpu_layers": 999},
    }


def _write_invalid_yaml(app, tag="bad-model"):
    """Hand-write a manifest file that parses as YAML but violates the
    cross-field rule (reasoning_budget >= n_predict) -- the exact failure
    class of a real stored manifest."""
    manifests_dir = app.state.manager.boot.storage.manifests_path
    text = (
        "model_tag: bad-model\n"
        "display_name: 'Bad Model'\n"
        f"gguf_blob_sha256: {SAMPLE_SHA}\n"
        "gguf_size_bytes: 10000000000\n"
        "context_size: 4096\n"
        "expected_vram_bytes: 11000000000\n"
        "llama_server_flags:\n"
        "  ctx_size: 4096\n"
        "  n_gpu_layers: 999\n"
        "  n_predict: 100\n"
        "  reasoning_budget: 200\n"
    )
    (manifests_dir / f"{tag}.yaml").write_text(text)


class TestListSurvivesInvalidManifest:
    def test_list_returns_200_with_unreadable_row(self, app_blank):
        app, client = app_blank
        client.put("/api/manifests/good-model", json=_valid_payload())
        _write_invalid_yaml(app)

        r = client.get("/api/manifests")
        assert r.status_code == 200, (
            f"one invalid manifest must not 500 the whole list: {r.status_code}"
        )
        body = r.json()
        assert body["total"] == 2
        rows = {row["model_tag"]: row for row in body["manifests"]}
        # The invalid file degrades to the designed 'unreadable' row...
        assert rows["bad-model"]["error"] == "unreadable"
        # The unreadable row has no manifest object to
        # read display_name/gguf_blob_sha256 off of, so both keys must still
        # be PRESENT, as None -- not omitted. A consumer doing row.display_name
        # on a shape that sometimes lacks the key would break, which is the
        # whole point of keeping the two row shapes uniform.
        assert rows["bad-model"]["display_name"] is None
        assert rows["bad-model"]["gguf_blob_sha256"] is None
        # ...while every other manifest stays fully listed.
        assert rows["good-model"]["hidden"] is False
        assert rows["good-model"]["revision"] == 1
        assert rows["good-model"]["etag"] == '"1"'
        assert rows["good-model"]["display_name"] == "Good Model"
        assert rows["good-model"]["gguf_blob_sha256"] == SAMPLE_SHA

    def test_empty_list_still_200(self, app_blank):
        app, client = app_blank
        r = client.get("/api/manifests")
        assert r.status_code == 200
        assert r.json()["total"] == 0


class TestPerTagReadsReturn400Not500:
    @pytest.fixture
    def seeded(self, app_blank):
        app, client = app_blank
        _write_invalid_yaml(app)
        return app, client

    def test_get_invalid_manifest_400(self, seeded):
        app, client = seeded
        r = client.get("/api/manifests/bad-model")
        assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"
        assert r.json()["detail"]

    def test_patch_invalid_manifest_400(self, seeded):
        app, client = seeded
        r = client.patch("/api/manifests/bad-model", json={"hidden": True})
        assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"

    def test_restore_defaults_invalid_manifest_400(self, seeded):
        app, client = seeded
        r = client.post("/api/manifests/bad-model/restore-defaults")
        assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"

    def test_valid_manifest_still_readable(self, seeded):
        app, client = seeded
        client.put("/api/manifests/good-model", json=_valid_payload())
        r = client.get("/api/manifests/good-model")
        assert r.status_code == 200
        assert r.json()["model_tag"] == "good-model"
