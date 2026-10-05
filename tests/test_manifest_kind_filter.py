"""Regression tests: manifest consumers assuming every manifest
is a MODEL. The Manifest split fixed the CONSTRUCTION chokepoint
(parse_manifest) everywhere but missed the CONSUMERS that iterate/read
manifests generically. A visible PluginManifest:

  - crashed /api/tags (ollama.py get_tags reads model-only fields unconditionally)
  - crashed /api/show for the plugin's own tag (same file, same shape)
  - was silently listed as a chat model by /v1/models and /v1/models/{model}

Fix: `isinstance(m, ModelManifest)` at each MODEL-listing/lookup consumer,
matching the pattern api/plugins.py's list_plugins already ships
(`if not isinstance(m, PluginManifest): continue`) -- no new shared helper.
api/manifests.py's general listing stays inclusive of both kinds by design,
but now reports `kind` per entry so a consumer can tell them apart.

chat_completion.py and embeddings.py have the identical crash shape
(read_manifest then unconditional access to model-only fields) but are
DELIBERATELY NOT touched here -- left for a separate change to the hot
inference path.
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
from turbohaul.manifest import ModelManifest, PluginManifest, write_manifest_atomic

SAMPLE_SHA = "a" * 64


def _mk_plugin(**overrides) -> PluginManifest:
    base = dict(
        model_tag="whisperx-transcribe",
        kind="plugin",
        lane="cpu",
        resource_key="whisperx-main",
        capabilities=["transcribe", "diarize"],
    )
    base.update(overrides)
    return PluginManifest(**base)


def _mk_model(**overrides) -> ModelManifest:
    base = dict(
        model_tag="qwen3.6-35b-moe",
        display_name="Qwen 3.6 35B-A3B MoE",
        gguf_blob_sha256=SAMPLE_SHA,
        gguf_size_bytes=22_000_000_000,
        context_size=131072,
        expected_vram_bytes=22_500_000_000,
        llama_server_flags={"ctx_size": 131072},
    )
    base.update(overrides)
    return ModelManifest(**base)


@pytest.fixture
def app_with_model_and_plugin(tmp_path):
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
    write_manifest_atomic(manifests_path, _mk_model())
    write_manifest_atomic(manifests_path, _mk_plugin())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client, manifests_path


class TestApiTagsExcludesPlugins:
    """/api/tags (ollama.py get_tags) -- would be a hard 500 with a plugin manifest
    present and not excluded."""

    def test_does_not_500_with_a_plugin_manifest_present(self, app_with_model_and_plugin):
        _, client, _ = app_with_model_and_plugin
        r = client.get("/api/tags")
        assert r.status_code == 200

    def test_plugin_tag_is_excluded(self, app_with_model_and_plugin):
        _, client, _ = app_with_model_and_plugin
        r = client.get("/api/tags")
        names = {m["name"] for m in r.json()["models"]}
        assert "whisperx-transcribe" not in names

    def test_real_model_still_lists_with_every_field_intact(self, app_with_model_and_plugin):
        """★ Positive control: a fix that hides plugins by hiding everything
        passes a naive test and breaks the product."""
        _, client, _ = app_with_model_and_plugin
        r = client.get("/api/tags")
        assert r.status_code == 200
        models = {m["name"]: m for m in r.json()["models"]}
        assert set(models) == {"qwen3.6-35b-moe"}
        m = models["qwen3.6-35b-moe"]
        assert m["model"] == "qwen3.6-35b-moe"
        assert m["size"] == 22_000_000_000
        assert m["digest"] == "sha256:" + SAMPLE_SHA
        assert m["details"]["context_length"] == 131072
        assert m["details"]["expected_vram_bytes"] == 22_500_000_000
        assert m["details"]["display_name"] == "Qwen 3.6 35B-A3B MoE"


class TestApiShowExcludesPlugins:
    """/api/show (ollama.py get_show) -- same file, same shape, same crash:
    reads six model-only fields unconditionally. Folded in because
    fixing the listing while the detail view still
    crashes on the same tag would be half a fix."""

    def test_plugin_tag_is_404_not_500(self, app_with_model_and_plugin):
        _, client, _ = app_with_model_and_plugin
        r = client.get("/api/show", params={"name": "whisperx-transcribe"})
        assert r.status_code == 404
        assert r.json()["detail"] == "model not found: whisperx-transcribe"

    def test_real_model_still_shows_with_every_field_intact(self, app_with_model_and_plugin):
        _, client, _ = app_with_model_and_plugin
        r = client.get("/api/show", params={"name": "qwen3.6-35b-moe"})
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "qwen3.6-35b-moe"
        assert body["size"] == 22_000_000_000
        assert body["digest"] == "sha256:" + SAMPLE_SHA
        assert body["context_length"] == 131072
        assert body["expected_vram_bytes"] == 22_500_000_000
        assert body["display_name"] == "Qwen 3.6 35B-A3B MoE"
        # Not exact-dict-equality: manifest read injects additional default
        # flags unrelated to this change (turbohaul's own default-flag
        # machinery) -- only assert the field survived, not its full shape.
        assert body["llama_server_flags"]["ctx_size"] == 131072


class TestV1ModelsExcludesPlugins:
    """/v1/models (models.py list_models) -- did NOT crash, which is the
    worse failure mode: it silently advertised the plugin as a selectable
    chat model."""

    def test_plugin_tag_is_excluded_from_data(self, app_with_model_and_plugin):
        _, client, _ = app_with_model_and_plugin
        r = client.get("/v1/models")
        assert r.status_code == 200
        ids = {m["id"] for m in r.json()["data"]}
        assert "whisperx-transcribe" not in ids

    def test_real_model_still_lists_with_every_field_intact(self, app_with_model_and_plugin):
        _, client, _ = app_with_model_and_plugin
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()["data"]
        assert len(data) == 1
        assert data[0]["id"] == "qwen3.6-35b-moe"
        assert data[0]["object"] == "model"
        assert data[0]["owned_by"] == "turbohaul"
        assert isinstance(data[0]["created"], int)


class TestV1ModelSingleExcludesPlugins:
    """/v1/models/{model} (models.py get_model) -- NOT in the original scope,
    but the same file, same low-risk class as list_models (a metadata lookup,
    not the hot inference path): a client could GET
    /v1/models/whisperx-transcribe and get back {"object": "model", ...} for
    a plugin. Covered explicitly, not left implicit."""

    def test_plugin_tag_is_404_not_a_model(self, app_with_model_and_plugin):
        _, client, _ = app_with_model_and_plugin
        r = client.get("/v1/models/whisperx-transcribe")
        assert r.status_code == 404

    def test_real_model_still_returns_200(self, app_with_model_and_plugin):
        _, client, _ = app_with_model_and_plugin
        r = client.get("/v1/models/qwen3.6-35b-moe")
        assert r.status_code == 200
        assert r.json()["id"] == "qwen3.6-35b-moe"


class TestApiManifestsStaysInclusiveWithKind:
    """api/manifests.py's general listing is deliberately NOT
    filtered -- it IS the manifest listing, both kinds belong there (by
    design). It now reports `kind` per row so a consumer that
    mixes both kinds together can actually tell them apart."""

    def test_both_kinds_still_listed(self, app_with_model_and_plugin):
        _, client, _ = app_with_model_and_plugin
        r = client.get("/api/manifests")
        assert r.status_code == 200
        tags = {row["model_tag"] for row in r.json()["manifests"]}
        assert tags == {"qwen3.6-35b-moe", "whisperx-transcribe"}

    def test_kind_field_present_and_correct(self, app_with_model_and_plugin):
        _, client, _ = app_with_model_and_plugin
        r = client.get("/api/manifests")
        by_tag = {row["model_tag"]: row for row in r.json()["manifests"]}
        assert by_tag["qwen3.6-35b-moe"]["kind"] == "model"
        assert by_tag["whisperx-transcribe"]["kind"] == "plugin"


class TestApiManifestsDisplayNameAndBlobSha:
    """Amendment: the model-tile grid (label and grouping)
    needs display_name + gguf_blob_sha256 per row so the FE doesn't have to
    refetch every manifest individually. Both come off the SAME `m` object
    list_manifests already loads -- zero extra I/O."""

    def test_model_row_carries_both_fields(self, app_with_model_and_plugin):
        _, client, _ = app_with_model_and_plugin
        r = client.get("/api/manifests")
        by_tag = {row["model_tag"]: row for row in r.json()["manifests"]}
        row = by_tag["qwen3.6-35b-moe"]
        assert row["display_name"] == "Qwen 3.6 35B-A3B MoE"
        assert row["gguf_blob_sha256"] == SAMPLE_SHA

    def test_plugin_row_does_not_crash_and_reports_none_for_model_only_field(
        self, app_with_model_and_plugin
    ):
        """The regression this test exists to catch: PluginManifest declares
        NO gguf_blob_sha256 field at all (manifest.py's own docstring: 'no
        llama_server_flags, no gguf_blob_sha256, no context_size'), and this
        listing is deliberately NOT kind-filtered. A verbatim `m.gguf_blob_sha256`
        AttributeErrors on this row and 500s the WHOLE list -- the same failure
        shape from a new angle. display_name IS shared (HardenedManifestBase,
        default ''), so the plugin's own default should come through unmangled."""
        _, client, _ = app_with_model_and_plugin
        r = client.get("/api/manifests")
        assert r.status_code == 200, r.text
        by_tag = {row["model_tag"]: row for row in r.json()["manifests"]}
        plugin_row = by_tag["whisperx-transcribe"]
        assert plugin_row["gguf_blob_sha256"] is None
        assert plugin_row["display_name"] == ""
