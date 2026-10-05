"""Tests for /api/manifests CRUD routes."""
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
        """PUT without If-Match on an existing manifest -> 412."""
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


class TestModelTagImmutabilityGuard:
    """Rename guard: the API needs a guard so
    model renames cannot happen. model_tag IS a manifest's identity -- the
    tag names the file, the URL, and every downstream
    reference to that model. Changing it on an EXISTING manifest is the
    rename that must not happen.

    This pins behaviour the code ALREADY has (verified by reading
    manifests.py and manifest.py directly, not
    assumed) across all three write-capable routes, through the REAL HTTP
    routes, not by calling Manifest(...)/write_manifest_atomic directly --
    the whole point is proving the ROUTE enforces it, matching
    test_tag_in_url_overrides_payload's own existing convention above but
    for an EXISTING manifest, not a fresh one (the create case already
    covered above does not exercise the "identity already established"
    scenario this guard is about).

    Causal both directions: a mutant letting PUT accept a mismatched
    model_tag on update must fail test_put_cannot_rename_an_existing_manifest;
    a mutant letting PATCH accept a model_tag key must fail
    test_patch_rejects_model_tag_key -- a DIFFERENT test, since they guard
    different routes and a regression in one must not be masked by the
    other still passing.
    """

    def test_put_cannot_rename_an_existing_manifest(self, app_blank):
        app, client = app_blank
        create = client.put("/api/manifests/bart-d1", json=_valid_payload("bart-d1"))
        assert create.status_code == 200
        etag = create.headers["ETag"]

        rename_attempt = _valid_payload("totally-different-name")
        r = client.put(
            "/api/manifests/bart-d1",
            json=rename_attempt,
            headers={"If-Match": etag},
        )
        assert r.status_code == 200
        # The response itself never claims the rename took.
        assert r.json()["model_tag"] == "bart-d1"

        # The persisted identity is unchanged -- read it back independently
        # of the write response, through the real GET route.
        after = client.get("/api/manifests/bart-d1")
        assert after.status_code == 200
        assert after.json()["model_tag"] == "bart-d1"

        # And no second manifest was created under the payload's claimed
        # name -- a rename-guard that silently CLONED under the new tag
        # instead of rejecting it would still be a way to end up with an
        # orphaned old tag plus a confusingly-named new one.
        clone_check = client.get("/api/manifests/totally-different-name")
        assert clone_check.status_code == 404

    def test_patch_rejects_model_tag_key(self, app_blank):
        app, client = app_blank
        client.put("/api/manifests/bart-d1", json=_valid_payload("bart-d1"))

        r = client.patch(
            "/api/manifests/bart-d1",
            json={"model_tag": "renamed-via-patch", "hidden": True},
        )
        assert r.status_code == 400

        # Persisted identity unchanged regardless of the response code --
        # the invariant that actually matters, checked independently of
        # whether the rejection mechanism is a 400 or something else.
        after = client.get("/api/manifests/bart-d1")
        assert after.status_code == 200
        assert after.json()["model_tag"] == "bart-d1"

    def test_patch_hidden_only_still_works(self, app_blank):
        """Control for the test above: PATCH accepting `hidden` alone must
        still succeed -- proves the 400 above is specifically about
        model_tag, not a broken endpoint that rejects everything."""
        app, client = app_blank
        create = client.put("/api/manifests/bart-d1", json=_valid_payload("bart-d1"))
        r = client.patch(
            "/api/manifests/bart-d1",
            json={"hidden": True},
            headers={"If-Match": create.headers["ETag"]},
        )
        assert r.status_code == 200
        assert r.json()["hidden"] is True

    def test_restore_defaults_preserves_model_tag(self, app_blank):
        """restore-defaults takes no caller payload at all -- there is no
        direct attack surface here, but the invariant is pinned as a
        regression guard: this route rebuilds the manifest from the
        already-loaded object, and must keep doing so if the route is ever
        changed to accept a body."""
        app, client = app_blank
        create = client.put("/api/manifests/bart-d1", json=_valid_payload("bart-d1"))
        r = client.post(
            "/api/manifests/bart-d1/restore-defaults",
            headers={"If-Match": create.headers["ETag"]},
        )
        assert r.status_code == 200
        assert r.json()["model_tag"] == "bart-d1"
        after = client.get("/api/manifests/bart-d1")
        assert after.json()["model_tag"] == "bart-d1"


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

    def test_get_cache_reuse_inert_marker_true_with_mmproj(self, app_blank):
        # positive control, end-to-end through the real route: a
        # manifest matching a real vision model's actual combination.
        app, client = app_blank
        payload = _valid_payload("vision-model")
        payload["llama_server_flags"]["cache_reuse"] = 256
        payload["mmproj_blob_sha256"] = "5a6b7c8d" + "0" * 56
        client.put("/api/manifests/vision-model", json=payload)
        r = client.get("/api/manifests/vision-model")
        assert r.status_code == 200
        assert r.json()["cache_reuse_inert_mmproj"] is True

    def test_get_cache_reuse_inert_marker_false_without_mmproj(self, app_blank):
        # negative control, end-to-end: cache_reuse alone must not
        # be flagged inert -- a signal that always fires is not a signal.
        app, client = app_blank
        payload = _valid_payload("text-model")
        payload["llama_server_flags"]["cache_reuse"] = 256
        client.put("/api/manifests/text-model", json=payload)
        r = client.get("/api/manifests/text-model")
        assert r.status_code == 200
        assert r.json()["cache_reuse_inert_mmproj"] is False


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
