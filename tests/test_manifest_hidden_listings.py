"""Hidden manifests: omitted from the discovery listings, still served by name.

`hidden: true` on a manifest removes the model from BOTH discovery endpoints
(/v1/models and /api/tags) while leaving it fully reachable by its exact tag.
The serve path reads the manifest directly and never enumerates the listing, so
hiding is a listings-only concern; these tests pin that boundary from both sides.
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
from turbohaul.manifest import (
    Manifest,
    ModelManifest,
    PluginManifest,
    read_manifest,
    write_manifest_atomic,
)

VISIBLE_SHA = "a" * 64
HIDDEN_SHA = "b" * 64
DEFAULT_SHA = "c" * 64


def _make_boot_runtime(tmp_path):
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
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return manifests_path, boot, runtime


@pytest.fixture
def app_mixed_visibility(tmp_path):
    """One explicitly visible model, one hidden, one that never sets the field."""
    manifests_path, boot, runtime = _make_boot_runtime(tmp_path)
    write_manifest_atomic(
        manifests_path,
        ModelManifest(
            model_tag="visible-model",
            gguf_blob_sha256=VISIBLE_SHA,
            gguf_size_bytes=1_000,
            hidden=False,
        ),
    )
    write_manifest_atomic(
        manifests_path,
        ModelManifest(
            model_tag="hidden-model",
            gguf_blob_sha256=HIDDEN_SHA,
            gguf_size_bytes=2_000,
            hidden=True,
        ),
    )
    write_manifest_atomic(
        manifests_path,
        ModelManifest(
            model_tag="unset-model",
            gguf_blob_sha256=DEFAULT_SHA,
            gguf_size_bytes=3_000,
        ),
    )
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client, manifests_path


class TestHiddenExcludedFromListings:
    def test_v1_models_omits_hidden(self, app_mixed_visibility):
        _, client, _ = app_mixed_visibility
        r = client.get("/v1/models")
        assert r.status_code == 200
        ids = {m["id"] for m in r.json()["data"]}
        assert "hidden-model" not in ids
        # The other two must survive: a filter that empties the listing would
        # pass a "hidden is absent" assertion on its own.
        assert ids == {"visible-model", "unset-model"}

    def test_api_tags_omits_hidden(self, app_mixed_visibility):
        _, client, _ = app_mixed_visibility
        r = client.get("/api/tags")
        assert r.status_code == 200
        names = {m["name"] for m in r.json()["models"]}
        assert "hidden-model" not in names
        assert names == {"visible-model", "unset-model"}

    def test_both_endpoints_agree(self, app_mixed_visibility):
        """The two listings must curate identically or consumers disagree."""
        _, client, _ = app_mixed_visibility
        openai_ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
        ollama_names = {m["name"] for m in client.get("/api/tags").json()["models"]}
        assert openai_ids == ollama_names


class TestHiddenStillReachableByName:
    def test_single_model_route_still_returns_hidden(self, app_mixed_visibility):
        """Hiding is listings-only: an exact-tag request still resolves."""
        _, client, _ = app_mixed_visibility
        r = client.get("/v1/models/hidden-model")
        assert r.status_code == 200
        assert r.json()["id"] == "hidden-model"

    def test_manifest_read_unaffected_by_hidden(self, app_mixed_visibility):
        """The serve path reads the manifest directly; hiding must not alter it."""
        _, _, manifests_path = app_mixed_visibility
        m = read_manifest(manifests_path, "hidden-model")
        assert m.model_tag == "hidden-model"
        assert m.gguf_blob_sha256 == HIDDEN_SHA
        assert m.hidden is True


class TestDefaultVisibility:
    def test_field_defaults_to_false(self):
        """Absent field means visible, so existing manifests need no edit."""
        m = ModelManifest(model_tag="no-flag", gguf_blob_sha256=DEFAULT_SHA)
        assert m.hidden is False

    def test_hidden_survives_write_read_roundtrip(self, tmp_path):
        manifests_path, _, _ = _make_boot_runtime(tmp_path)
        write_manifest_atomic(
            manifests_path,
            ModelManifest(model_tag="rt", gguf_blob_sha256=DEFAULT_SHA, hidden=True),
        )
        assert read_manifest(manifests_path, "rt").hidden is True

    def test_hidden_is_a_declared_field_not_an_extra(self):
        """The schema forbids unknown keys, so this must be declared to be legal.

        Note: Manifest is a discriminated-union type alias, not a
        class -- .model_fields lives on the concrete variants. hidden is
        declared on HardenedManifestBase and inherited by both, so checking
        it on ModelManifest (equally true of PluginManifest) still proves
        the same thing the docstring claims.
        """
        assert "hidden" in ModelManifest.model_fields
        assert "hidden" in PluginManifest.model_fields
