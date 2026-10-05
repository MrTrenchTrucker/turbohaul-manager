"""Ollama-compat resolved context + vision-projector-only modality.

Context resolution: ollama.py's `/api/tags` and `/api/show` read raw `m.context_size` directly instead of
resolving it the way `/v1/models` does -- so a manifest whose `llama_server_flags.ctx_size`
diverges from `context_size` gets a WRONG number here. `_resolve_context(m) -> (value, source)` is
file-local to ollama.py on purpose (no shared helper with models.py -- ollama.py says so itself, and
the manifest-decode blast-radius test's own doctrine: a shared ASSERTION, never shared
CODE). `TestFifthDriftTest` below is that shared assertion for this pair of surfaces.

Vision modality: `is_vision` in ollama.py additionally treats a bare GGUF vision-marker key as vision
support even with no `mmproj_blob_sha256` on the manifest. But `--mmproj` only ever reaches
llama-server from the manifest's own projector blob (manager.py, at both launch sites -- each
gated on `manifest.mmproj_blob_sha256`), and the raw `mmproj` flag is in DENIED_FLAGS
(manifest.py) so it cannot arrive any other way. A model advertised as vision-capable on the
GGUF key alone therefore cannot actually be served with vision, whatever the GGUF carries -- the
projector clause in `is_vision` promises something the launcher structurally cannot deliver.

Both are latent, not live: in real manifests the `ctx_size` flag already
agrees with `context_size`, and the advertised-vision set already equals the mmproj set exactly.
Neither divergence occurs in real manifests; both are fixture-only in this file.
"""
import pytest
from fastapi.testclient import TestClient

import turbohaul.api.ollama as ollama_module
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
from turbohaul.manifest import ModelManifest, write_manifest_atomic

from tests.test_gguf_modelcard import _build_gguf, _kv_str, _kv_u32
from turbohaul._gguf_meta import read_model_card


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
    return storage_root, manifests_path, boot, runtime


def _write_blob(blobs_root, sha256, content):
    from turbohaul import blob_store
    path = blob_store.blob_path(blobs_root, sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture(autouse=True)
def _clear_model_card_cache():
    ollama_module._MODEL_CARD_CACHE.clear()
    yield
    ollama_module._MODEL_CARD_CACHE.clear()


class TestDivergentArmIsTheRealRed:
    """(a): flag and context_size DIVERGE. Both endpoints must report the FLAG value -- pre-fix
    both read raw m.context_size, which is the WRONG number here. This is a real RED on the VALUE,
    not just on the new key."""

    def test_tags_reports_the_flag_not_the_manifest_size(self, tmp_path):
        storage_root, manifests_path, boot, runtime = _make_boot_runtime(tmp_path)
        sha = "1" * 64
        write_manifest_atomic(
            manifests_path,
            ModelManifest(
                model_tag="divergent-tags",
                gguf_blob_sha256=sha,
                gguf_size_bytes=1000,
                context_size=8192,
                llama_server_flags={"ctx_size": 250000},
            ),
        )
        _write_blob(storage_root / "blobs", sha, _build_gguf([_kv_str("general.architecture", "llama")]))
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as client:
            r = client.get("/api/tags")
        details = r.json()["models"][0]["details"]
        assert details["context_length"] == 250000
        assert details["context_length_source"] == "flag"

    def test_show_reports_the_flag_not_the_manifest_size(self, tmp_path):
        storage_root, manifests_path, boot, runtime = _make_boot_runtime(tmp_path)
        sha = "2" * 64
        write_manifest_atomic(
            manifests_path,
            ModelManifest(
                model_tag="divergent-show",
                gguf_blob_sha256=sha,
                gguf_size_bytes=1000,
                context_size=8192,
                llama_server_flags={"ctx_size": 250000},
            ),
        )
        _write_blob(storage_root / "blobs", sha, _build_gguf([_kv_str("general.architecture", "llama")]))
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as client:
            r = client.get("/api/show", params={"name": "divergent-show"})
        body = r.json()
        assert body["context_length"] == 250000
        assert body["context_length_source"] == "flag"


class TestManifestSourceArmIsHonestAboutNotBeingARealRed:
    """(b): NO ctx_size flag. value == context_size (pre-fix already emits this -- NOT a real RED
    on the value), source == "manifest" (pre-fix has no such key at all -- IS a real RED, but only
    on the missing key, not on any wrong value). Neither arm occurs in practice: all real
    manifests carry a ctx_size flag, so this arm is fixtures-only, same as (a)."""

    def test_tags_manifest_source_value_and_source(self, tmp_path):
        storage_root, manifests_path, boot, runtime = _make_boot_runtime(tmp_path)
        sha = "3" * 64
        write_manifest_atomic(
            manifests_path,
            ModelManifest(
                model_tag="no-flag-tags",
                gguf_blob_sha256=sha,
                gguf_size_bytes=1000,
                context_size=32768,
            ),
        )
        _write_blob(storage_root / "blobs", sha, _build_gguf([_kv_str("general.architecture", "llama")]))
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as client:
            r = client.get("/api/tags")
        details = r.json()["models"][0]["details"]
        # value alone cannot distinguish pre-/post-fix -- pre-fix already emits context_size here.
        assert details["context_length"] == 32768
        # source is the only part of this arm that can go red pre-fix (key absent entirely).
        assert details["context_length_source"] == "manifest"

    def test_show_manifest_source_value_and_source(self, tmp_path):
        storage_root, manifests_path, boot, runtime = _make_boot_runtime(tmp_path)
        sha = "4" * 64
        write_manifest_atomic(
            manifests_path,
            ModelManifest(
                model_tag="no-flag-show",
                gguf_blob_sha256=sha,
                gguf_size_bytes=1000,
                context_size=32768,
            ),
        )
        _write_blob(storage_root / "blobs", sha, _build_gguf([_kv_str("general.architecture", "llama")]))
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as client:
            r = client.get("/api/show", params={"name": "no-flag-show"})
        body = r.json()
        assert body["context_length"] == 32768
        assert body["context_length_source"] == "manifest"


class TestVisionLookalikeTrap:
    """(c): no mmproj, but a real GGUF blob carrying a clip.* key -- has_vision_kv=True. The
    lookalike: if the fixture's card read silently failed, card would be None and
    `bool(card and ...)` would be False, landing on "text" for the WRONG reason, passing
    identically before and after the fix. Guarded by an explicit precondition that the card
    actually observed has_vision_kv=True, straight off the built GGUF bytes -- not through the
    endpoint -- before trusting any endpoint assertion built on top of it."""

    def test_precondition_the_fixture_gguf_really_has_the_vision_marker(self, tmp_path):
        blob_path = tmp_path / "check.gguf"
        blob_path.write_bytes(_build_gguf([
            _kv_str("general.architecture", "llava"),
            _kv_u32("clip.vision_feature_layer", 1),
        ]))
        card = read_model_card(blob_path)
        assert card is not None
        assert card.has_vision_kv is True

    def test_no_mmproj_gguf_vision_key_alone_is_not_advertised_as_vision(self, tmp_path):
        storage_root, manifests_path, boot, runtime = _make_boot_runtime(tmp_path)
        sha = "5" * 64
        write_manifest_atomic(
            manifests_path,
            ModelManifest(
                model_tag="fused-no-mmproj",
                gguf_blob_sha256=sha,
                gguf_size_bytes=1000,
                # mmproj_blob_sha256 deliberately absent.
            ),
        )
        _write_blob(storage_root / "blobs", sha, _build_gguf([
            _kv_str("general.architecture", "llava"),
            _kv_u32("clip.vision_feature_layer", 1),
        ]))
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as client:
            r = client.get("/api/tags")
        details = r.json()["models"][0]["details"]
        # This is the post-fix contract. Pre-fix, this manifest's GGUF-only
        # clip key still trips the OR clause and reports "vision" -- that failure is the RED:
        # the assertion below fails with actual="vision", proving the trap fixture really built
        # the condition (has_vision_kv=True observed above) rather than silently landing on
        # "text" for the wrong reason.
        assert details["modality"] == "text"

    def test_green_control_with_mmproj_still_reports_vision_after_the_fix(self, tmp_path):
        """Required control: without it, is_vision=False passes this class's tests trivially."""
        storage_root, manifests_path, boot, runtime = _make_boot_runtime(tmp_path)
        sha = "6" * 64
        write_manifest_atomic(
            manifests_path,
            ModelManifest(
                model_tag="real-vision",
                gguf_blob_sha256=sha,
                mmproj_blob_sha256="7" * 64,
                gguf_size_bytes=1000,
            ),
        )
        # GGUF itself carries NO vision marker -- modality must come from mmproj alone, both
        # before and after the fix.
        _write_blob(storage_root / "blobs", sha, _build_gguf([_kv_str("general.architecture", "llava")]))
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as client:
            r = client.get("/api/tags")
        details = r.json()["models"][0]["details"]
        assert details["modality"] == "vision"


class TestFifthDriftTest:
    """(d): the existing `test_no_two_consumers_disagree` pins SET-EQUALITY of a CONSTANT
    (_UNREADABLE_MANIFEST across four file-local tuples) -- a structural pin, unrelated to any
    request. This one pins BEHAVIOUR on real inputs: the same manifest, read through /v1/models
    (+/v1/models/{tag}) and through /api/tags+/api/show, must agree on both context_length AND
    context_length_source, despite the two ollama.py endpoints publishing that pair in a
    different JSON position (nested vs top-level) than models.py does (always top-level).
    Neither guarantee covers the other: the constant-set pin never runs a request, and this one
    never touches _UNREADABLE_MANIFEST."""

    def test_all_four_endpoints_agree_on_value_and_source(self, tmp_path):
        storage_root, manifests_path, boot, runtime = _make_boot_runtime(tmp_path)
        sha = "8" * 64
        tag = "cross-surface-divergent"
        write_manifest_atomic(
            manifests_path,
            ModelManifest(
                model_tag=tag,
                gguf_blob_sha256=sha,
                gguf_size_bytes=1000,
                context_size=8192,
                llama_server_flags={"ctx_size": 250000},
            ),
        )
        _write_blob(storage_root / "blobs", sha, _build_gguf([_kv_str("general.architecture", "llama")]))
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        with TestClient(app) as client:
            v1_list = client.get("/v1/models").json()["data"][0]
            v1_single = client.get(f"/v1/models/{tag}").json()
            tags_body = client.get("/api/tags").json()["models"][0]["details"]
            show_body = client.get("/api/show", params={"name": tag}).json()

        values = {
            "/v1/models": v1_list["context_length"],
            "/v1/models/{tag}": v1_single["context_length"],
            "/api/tags": tags_body["context_length"],
            "/api/show": show_body["context_length"],
        }
        sources = {
            "/v1/models": v1_list["context_length_source"],
            "/v1/models/{tag}": v1_single["context_length_source"],
            "/api/tags": tags_body["context_length_source"],
            "/api/show": show_body["context_length_source"],
        }
        assert len(set(values.values())) == 1, values
        assert len(set(sources.values())) == 1, sources
        assert values["/v1/models"] == 250000
        assert sources["/v1/models"] == "flag"
