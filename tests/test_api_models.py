"""Tests for OpenAI-compat discovery: GET /v1/models + /v1/models/{model}.

TurboHaul model discovery. Same list_manifests/read_manifest source
/api/tags already uses; no GGUF model-card parse (discovery-only).
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
from turbohaul.manifest import Manifest, ModelManifest, write_manifest_atomic

SAMPLE_SHA = "f" * 64
SECOND_SHA = "e" * 64


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
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return manifests_path, boot, runtime


@pytest.fixture
def app_with_manifests(tmp_path):
    manifests_path, boot, runtime = _make_boot_runtime(tmp_path)
    write_manifest_atomic(
        manifests_path,
        ModelManifest(
            model_tag="qwen3.6-35b-moe",
            display_name="Qwen 3.6 35B-A3B MoE",
            gguf_blob_sha256=SAMPLE_SHA,
            gguf_size_bytes=22_000_000_000,
        ),
    )
    write_manifest_atomic(
        manifests_path,
        ModelManifest(
            model_tag="qwen-coder",
            gguf_blob_sha256=SECOND_SHA,
            gguf_size_bytes=15_000_000_000,
        ),
    )
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client, manifests_path


class TestListModels:
    def test_lists_all_models_openai_shape(self, app_with_manifests):
        _, client, _ = app_with_manifests
        r = client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        ids = {m["id"] for m in body["data"]}
        assert ids == {"qwen3.6-35b-moe", "qwen-coder"}
        for m in body["data"]:
            assert m["object"] == "model"
            assert m["owned_by"] == "turbohaul"
            assert isinstance(m["created"], int)
            assert m["created"] > 0

    def test_empty_when_no_manifests(self, tmp_path):
        _, boot, runtime = _make_boot_runtime(tmp_path)
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as client:
            r = client.get("/v1/models")
        assert r.status_code == 200
        assert r.json() == {"object": "list", "data": []}

    def test_corrupt_manifest_skipped_not_fatal(self, app_with_manifests):
        """A .yaml whose root is not a mapping makes read_manifest raise
        ManifestValidationError directly ("manifest root must be mapping",
        manifest.py). list_manifests' TAG_RE still admits the filename (it's
        a valid tag shape), so this exercises the real skip-mid-list path,
        not a hand-fed exception."""
        _, client, manifests_path = app_with_manifests
        (manifests_path / "corrupt-tag.yaml").write_text("- just\n- a\n- list\n")
        r = client.get("/v1/models")
        assert r.status_code == 200
        ids = {m["id"] for m in r.json()["data"]}
        assert ids == {"qwen3.6-35b-moe", "qwen-coder"}
        assert "corrupt-tag" not in ids

    def test_unparseable_yaml_skipped_not_fatal(self, app_with_manifests):
        """A .yaml with broken SYNTAX raises yaml.YAMLError out of
        yaml.safe_load — NOT ManifestValidationError. Catching only
        (FileNotFoundError, ManifestValidationError) lets it escape and 500s
        the whole listing. /api/tags can hit
        that same gap and return 500 on this input;
        the list path here must stay hardened."""
        _, client, manifests_path = app_with_manifests
        (manifests_path / "badsyntax.yaml").write_text("this: [is not\n  a valid manifest")
        r = client.get("/v1/models")
        assert r.status_code == 200, "one unparseable .yaml must not 500 the listing"
        ids = {m["id"] for m in r.json()["data"]}
        assert ids == {"qwen3.6-35b-moe", "qwen-coder"}
        assert "badsyntax" not in ids

    def test_stale_schema_manifest_skipped_not_fatal(self, app_with_manifests):
        """A well-formed mapping that no longer satisfies the Manifest schema
        raises pydantic ValidationError, which is likewise NOT
        ManifestValidationError. This is the REALISTIC case: any manifest
        written by an older turbohaul, predating a newly-required field,
        raises it — so after any schema change every stale manifest would
        otherwise take the entire discovery surface down with it."""
        _, client, manifests_path = app_with_manifests
        (manifests_path / "staleschema.yaml").write_text("model_tag: staleschema\n")
        r = client.get("/v1/models")
        assert r.status_code == 200, "a stale-schema manifest must not 500 the listing"
        ids = {m["id"] for m in r.json()["data"]}
        assert ids == {"qwen3.6-35b-moe", "qwen-coder"}
        assert "staleschema" not in ids

    def test_list_order_deterministic(self, app_with_manifests):
        _, client, _ = app_with_manifests
        first = [m["id"] for m in client.get("/v1/models").json()["data"]]
        second = [m["id"] for m in client.get("/v1/models").json()["data"]]
        assert first == second == sorted(first)

    def test_skip_is_not_silent_logs_warning_with_type_not_message(self, app_with_manifests, caplog):
        """Required behavior: a skip must never be silent -- an
        operator debugging "why did my model vanish" needs a trace. WARNING
        names the tag + exception TYPE, never str(exc) (leak rule: pydantic
        ValidationError echoes manifest field VALUES)."""
        _, client, manifests_path = app_with_manifests
        (manifests_path / "staleschema.yaml").write_text(
            "model_tag: staleschema\nleaked_marker_value_xyz789: present\n"
        )
        with caplog.at_level("WARNING"):
            client.get("/v1/models")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "staleschema" in msg
        assert "ValidationError" in msg
        assert "leaked_marker_value_xyz789" not in msg


class TestGetModel:
    def test_200_known_model(self, app_with_manifests):
        _, client, _ = app_with_manifests
        r = client.get("/v1/models/qwen3.6-35b-moe")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "qwen3.6-35b-moe"
        assert body["object"] == "model"
        assert body["owned_by"] == "turbohaul"
        assert isinstance(body["created"], int)

    def test_404_unknown_model(self, app_with_manifests):
        _, client, _ = app_with_manifests
        r = client.get("/v1/models/nonexistent-model")
        assert r.status_code == 404

    def test_400_encoded_traversal(self, app_with_manifests):
        """Security-load-bearing: read_manifest's
        internal validate_tag call raises ManifestValidationError for a
        traversal-shaped tag; the narrow `except ManifestValidationError`
        in get_model is what turns that into a clean 400 instead of an
        unhandled 500 with a stack trace.

        Note which encoded payload actually
        reaches this branch over HTTP: a %2F-bearing path (e.g.
        ..%2F..%2Fetc%2Fpasswd) never reaches get_model at all -- the ASGI
        layer decodes %2F to a literal '/' in scope['path'] BEFORE Starlette
        routes it, so it doesn't match the single-segment {model} converter
        and 404s at the router, before any handler code runs. A slash-free
        encoded segment (%2e%2e -> "..") DOES reach {model} as one segment
        and is the real, HTTP-reachable proof of the ManifestValidationError
        -> 400 branch."""
        _, client, _ = app_with_manifests
        r = client.get("/v1/models/%2e%2e")
        assert r.status_code == 400
        assert "regex" in r.json()["detail"]

    def test_404_not_500_for_slash_bearing_traversal_blocked_at_router(self, app_with_manifests):
        """Companion to the above: a %2F-bearing traversal payload never
        reaches the handler's exception branches at all -- Starlette's router
        itself refuses to match it (404), which is stronger than the 400
        branch, not weaker. Documents the actual blast radius rather than
        leaving it unverified."""
        _, client, _ = app_with_manifests
        r = client.get("/v1/models/..%2F..%2Fetc%2Fpasswd")
        assert r.status_code == 404

    def test_500_broken_yaml_fixed_message(self, app_with_manifests):
        """read_manifest raises yaml.YAMLError (broken syntax),
        which would otherwise be unhandled here -> 500 with a stack trace. The
        handler returns a handled 500 with a fixed, tag-only message."""
        _, client, manifests_path = app_with_manifests
        (manifests_path / "brokenyaml.yaml").write_text("model_tag: brokenyaml\n  bad: [unclosed\n")
        r = client.get("/v1/models/brokenyaml")
        assert r.status_code == 500
        assert r.json()["detail"] == "manifest for 'brokenyaml' is present but unreadable"

    def test_500_stale_schema_fixed_message_no_leak(self, app_with_manifests):
        """pydantic ValidationError (stale schema),
        which would otherwise be unhandled here -> 500 with a stack trace that echoes manifest field
        VALUES. The handler returns a handled 500 with a fixed message; no leak."""
        _, client, manifests_path = app_with_manifests
        (manifests_path / "staleschema.yaml").write_text(
            "model_tag: staleschema\nleaked_marker_value_xyz789: present\n"
        )
        r = client.get("/v1/models/staleschema")
        assert r.status_code == 500
        assert r.json()["detail"] == "manifest for 'staleschema' is present but unreadable"
        assert "leaked_marker_value_xyz789" not in r.text
        assert "input_value" not in r.text
