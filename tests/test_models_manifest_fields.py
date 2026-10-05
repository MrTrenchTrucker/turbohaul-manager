"""GET /v1/models (+ bare /models) publishes
manifest-derived fields it already holds instead of the four useless ones.

Four new per-entry fields, asymmetric hybrid shape, specified as
follows:
  context_length            (flat)   -- int(llama_server_flags.ctx_size or
                                         manifest.context_size): the SAME
                                         resolution expression manager.py
                                         uses at its four ctx_size call
                                         sites --
                                         never bare manifest.context_size,
                                         which never reaches llama-server.
  context_length_source     (flat)   -- "flag"|"manifest". Named _source,
                                         not
                                         context_length_enforced -- nothing
                                         at this layer rejects a longer
                                         request, so "enforced" would be the
                                         wrong word for "where this number
                                         came from". Both context_length and
                                         context_length_source derive
                                         from the SAME ctx_size lookup
                                         (_ctx_flag in _to_openai_model), so
                                         they cannot disagree by
                                         construction.
  turbohaul.display_name    (nested) -- falls back to model_tag when empty.
  turbohaul.multimodal      (nested) -- bool(mmproj_blob_sha256); the digest
                                         itself must never be published.

Flat vs nested is deliberate, not tidied: context_length/_source are one
fact in two parts (a caveat nested away from the number it qualifies is
invisible to the naive client that is mis-detecting today); display_name/
multimodal are turbohaul-owned concepts namespaced under one vendor key.

Same bare-route handler as /v1/models (stacked @router.get, not a second
function) for BOTH the list and the single-model fetch
-- /models/{model} exists alongside /v1/models/{model}.
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

GGUF_SHA = "a" * 64
MMPROJ_SHA = "d1" * 32


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
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return manifests_path, boot, runtime


def _app(tmp_path, *models, plugins=()):
    manifests_path, boot, runtime = _make_boot_runtime(tmp_path)
    for m in models:
        write_manifest_atomic(manifests_path, m)
    for p in plugins:
        write_manifest_atomic(manifests_path, p)
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    return app


def _model(tag, **overrides):
    kwargs = dict(model_tag=tag, gguf_blob_sha256=GGUF_SHA, gguf_size_bytes=1_000)
    kwargs.update(overrides)
    return ModelManifest(**kwargs)


def _plugin(tag="a-plugin"):
    return PluginManifest(
        model_tag=tag, kind="plugin", lane="cpu",
        resource_key="whisperx-main", capabilities=["transcribe"],
    )


def _entry(body, tag):
    return next(m for m in body["data"] if m["id"] == tag)


class TestTheDecidingAssertion:
    def test_context_length_prefers_llama_server_flags_over_manifest_context_size(
        self, tmp_path
    ):
        """The one assertion that distinguishes the correct fix from the
        naive one: manifest.context_size is a number the engine NEVER
        receives when llama_server_flags.ctx_size is set (manager.py's own
        four call sites resolve it flags-first). Publishing bare
        manifest.context_size here would republish the exact defect users
        complained about, one layer inward."""
        app = _app(
            tmp_path,
            _model(
                "divergent",
                context_size=250_000,
                llama_server_flags={"ctx_size": 500_000},
            ),
        )
        with TestClient(app) as client:
            body = client.get("/v1/models").json()
        entry = _entry(body, "divergent")
        assert entry["context_length"] == 500_000
        assert isinstance(entry["context_length"], int)


class TestContextLengthSource:
    def test_flag_when_ctx_size_flag_present(self, tmp_path):
        app = _app(
            tmp_path,
            _model(
                "with-flag", context_size=4096,
                llama_server_flags={"ctx_size": 8192},
            ),
        )
        with TestClient(app) as client:
            body = client.get("/v1/models").json()
        entry = _entry(body, "with-flag")
        assert entry["context_length_source"] == "flag"
        assert entry["context_length"] == 8192

    def test_manifest_when_ctx_size_flag_absent_and_context_length_still_independent(
        self, tmp_path
    ):
        """The manifest arm must ALSO prove context_length is still populated
        from the manifest fallback -- i.e. the two fields are each derived
        from the same ctx_size lookup, not one computed from the other's
        result."""
        app = _app(tmp_path, _model("no-flag", context_size=4096, llama_server_flags={}))
        with TestClient(app) as client:
            body = client.get("/v1/models").json()
        entry = _entry(body, "no-flag")
        assert entry["context_length_source"] == "manifest"
        assert entry["context_length"] == 4096


class TestTurbohaulNamespace:
    def test_display_name_present_is_used_verbatim(self, tmp_path):
        app = _app(tmp_path, _model("named", display_name="My Cool Model"))
        with TestClient(app) as client:
            body = client.get("/v1/models").json()
        entry = _entry(body, "named")
        assert entry["turbohaul"]["display_name"] == "My Cool Model"

    def test_display_name_falls_back_to_model_tag_when_empty(self, tmp_path):
        app = _app(tmp_path, _model("unnamed"))
        with TestClient(app) as client:
            body = client.get("/v1/models").json()
        entry = _entry(body, "unnamed")
        assert entry["turbohaul"]["display_name"] == "unnamed"

    def test_multimodal_true_with_mmproj(self, tmp_path):
        app = _app(tmp_path, _model("vision", mmproj_blob_sha256=MMPROJ_SHA))
        with TestClient(app) as client:
            body = client.get("/v1/models").json()
        entry = _entry(body, "vision")
        assert entry["turbohaul"]["multimodal"] is True

    def test_multimodal_false_without_mmproj(self, tmp_path):
        app = _app(tmp_path, _model("textonly"))
        with TestClient(app) as client:
            body = client.get("/v1/models").json()
        entry = _entry(body, "textonly")
        assert entry["turbohaul"]["multimodal"] is False

    def test_multimodal_is_derived_the_digest_itself_never_leaks(self, tmp_path):
        """DERIVED -- never publish the digest. multimodal is a boolean,
        never the mmproj_blob_sha256 value itself."""
        app = _app(tmp_path, _model("vision2", mmproj_blob_sha256=MMPROJ_SHA))
        with TestClient(app) as client:
            r = client.get("/v1/models")
        assert MMPROJ_SHA not in r.text

    def test_nested_fields_are_not_also_flattened_to_top_level(self, tmp_path):
        app = _app(
            tmp_path,
            _model("nestcheck", display_name="X", mmproj_blob_sha256=MMPROJ_SHA),
        )
        with TestClient(app) as client:
            body = client.get("/v1/models").json()
        entry = _entry(body, "nestcheck")
        assert "display_name" not in entry
        assert "multimodal" not in entry
        assert set(entry["turbohaul"]) == {"display_name", "multimodal"}


class TestContextFieldsAreFlatNotNested:
    def test_context_length_and_enforced_are_top_level(self, tmp_path):
        app = _app(tmp_path, _model("flatcheck", context_size=4096))
        with TestClient(app) as client:
            body = client.get("/v1/models").json()
        entry = _entry(body, "flatcheck")
        assert "context_length" in entry
        assert "context_length_source" in entry
        assert "context_length" not in entry.get("turbohaul", {})


class TestLegacyKeysUnchanged:
    def test_four_legacy_keys_unchanged_in_name_and_value(self, tmp_path):
        app = _app(tmp_path, _model("legacy", display_name="Legacy Model"))
        with TestClient(app) as client:
            body = client.get("/v1/models").json()
        entry = _entry(body, "legacy")
        assert entry["id"] == "legacy"
        assert entry["object"] == "model"
        assert entry["owned_by"] == "turbohaul"
        assert isinstance(entry["created"], int)
        assert entry["created"] > 0
        # No new TOP-LEVEL envelope key -- only entry-level fields changed.
        assert set(body.keys()) == {"object", "data"}


class TestNegativeArmBothHalves:
    def test_hidden_model_absent_from_both_routes(self, tmp_path):
        app = _app(
            tmp_path,
            _model("visible"),
            _model("hidden", hidden=True),
        )
        with TestClient(app) as client:
            v1_ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
            bare_ids = {m["id"] for m in client.get("/models").json()["data"]}
        assert "hidden" not in v1_ids
        assert "hidden" not in bare_ids
        assert v1_ids == bare_ids == {"visible"}

    def test_plugin_manifest_absent_from_both_routes(self, tmp_path):
        app = _app(tmp_path, _model("visible"), plugins=[_plugin("a-plugin")])
        with TestClient(app) as client:
            v1_ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
            bare_ids = {m["id"] for m in client.get("/models").json()["data"]}
        assert "a-plugin" not in v1_ids
        assert "a-plugin" not in bare_ids
        assert v1_ids == bare_ids == {"visible"}

    def test_hidden_and_plugin_filters_compose_on_both_routes(self, tmp_path):
        """Both filters must apply TOGETHER, not just independently -- proves
        neither one silently swallows or shadows the other's job (the
        arithmetic this preserves: hidden models and plugin manifests are
        each excluded, and the exclusions stack; that gap is deliberate)."""
        app = _app(
            tmp_path,
            _model("keep-1"),
            _model("keep-2"),
            _model("hide-me", hidden=True),
            plugins=[_plugin("plugin-1")],
        )
        with TestClient(app) as client:
            v1_ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
            bare_ids = {m["id"] for m in client.get("/models").json()["data"]}
        assert v1_ids == bare_ids == {"keep-1", "keep-2"}


class TestBareModelsRoute:
    def test_bare_models_payload_identical_to_v1_models(self, tmp_path):
        app = _app(
            tmp_path,
            _model("m1", display_name="One", context_size=4096,
                   llama_server_flags={"ctx_size": 8192}),
            _model("m2", mmproj_blob_sha256=MMPROJ_SHA),
        )
        with TestClient(app) as client:
            r1 = client.get("/v1/models")
            r2 = client.get("/models")
        assert r1.status_code == r2.status_code == 200
        assert r1.json() == r2.json()

    def test_bare_models_is_not_just_200_but_the_real_shape(self, tmp_path):
        app = _app(tmp_path, _model("shapecheck", context_size=4096))
        with TestClient(app) as client:
            body = client.get("/models").json()
        entry = _entry(body, "shapecheck")
        assert entry["context_length"] == 4096
        assert entry["context_length_source"] == "manifest"
        assert entry["turbohaul"]["display_name"] == "shapecheck"
        assert entry["turbohaul"]["multimodal"] is False


class TestBareSingleModelRoute:
    """Single-model route: /v1/models/{model} stacked a bare
    /models/{model}, mirroring the list routes' pattern -- one
    decorator, no new logic."""

    def test_bare_single_model_route_matches_v1(self, tmp_path):
        app = _app(
            tmp_path,
            _model("solo", context_size=4096, llama_server_flags={"ctx_size": 8192}),
        )
        with TestClient(app) as client:
            r1 = client.get("/v1/models/solo")
            r2 = client.get("/models/solo")
        assert r1.status_code == r2.status_code == 200
        assert r1.json() == r2.json()

    def test_bare_single_model_404_is_the_real_handler_not_a_routing_miss(self, tmp_path):
        """A route that simply doesn't exist also 404s -- FastAPI's own
        generic 'Not Found' body. This must fail for the RIGHT reason: the
        bare route dispatching into get_model's own FileNotFoundError arm,
        not a routing miss that happens to share a status code."""
        app = _app(tmp_path)
        with TestClient(app) as client:
            r = client.get("/models/does-not-exist")
        assert r.status_code == 404
        assert r.json()["detail"] == "model not found: does-not-exist"


class TestSingleModelRouteGetsTheSameFields:
    """Not strictly required, but /v1/models/{model} shares
    _to_openai_model with the list routes -- verified, not assumed."""

    def test_get_model_also_carries_the_new_fields(self, tmp_path):
        app = _app(
            tmp_path,
            _model("single", context_size=4096, llama_server_flags={"ctx_size": 8192}),
        )
        with TestClient(app) as client:
            body = client.get("/v1/models/single").json()
        assert body["context_length"] == 8192
        assert body["context_length_source"] == "flag"
        assert body["turbohaul"]["multimodal"] is False
