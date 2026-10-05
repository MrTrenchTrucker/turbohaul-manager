"""Pin the hidden-manifest contract through the REAL completion
path, not just the listing endpoints.

`hidden: true` on a manifest already has automated coverage for the
listings-exclusion half (test_manifest_hidden_listings.py) and the single
GET route (/v1/models/{model}). What has never been automated is the other
half of the same contract: a hidden model still SERVES -- answers a real
completion by exact name -- because the completion routes read the manifest
directly and never enumerate the (hidden-filtering) listing. Before this
file, that half rested on a one-shot manual live proof.

This matters more now that the UI offers hide/show: hiding
a model is now a routine user action, so a regression here means a user
hides a model and silently loses the ability to call it -- not a cosmetic
listing bug.

Both directions, each on its own test so a real product regression fails
exactly one of them, not both at once:
- test_hidden_model_still_completes_by_exact_name: if hiding started ALSO
  blocking completions (e.g. a hidden check added to the completion routes'
  manifest lookup), THIS test goes red.
- test_hidden_model_absent_from_listings: if hiding stopped hiding (the
  listing filter removed/broken), THIS test goes red.

What the test CONSTRUCTS vs what production DERIVES (a test must not
hand-build what production derives): the manifest is written to disk via the real
write_manifest_atomic + ModelManifest(hidden=True) and read back through the
real read_manifest/listing/routing code -- nothing here hand-builds a
"hidden" flag inline or bypasses the manifest store. The completion itself
is driven through the real FastAPI route, the real manager worker_loop, and
the real per-request manifest lookup; only the actual subprocess spawn and
the final engine HTTP call are stubbed (the established pattern in
test_api_chat_completion.py's app_completion_autostart fixture -- there is
no real GPU/model in this test environment, and that seam is the same one
every other completion-path test in this suite already relies on).

safety_enabled=False: the real safety-gate check compares against the
CONTAINER'S actual ambient CPU load, which is unrelated to anything this
test is about and can make this test flake under real,
unrelated machine load (a busy shared host
is enough) -- disabled here for the same reason drained_sigterm windows
and grace_seconds are already shortened to 0 in every fixture in this file:
determinism, not because production runs with safety off.
"""
from unittest.mock import MagicMock

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
from turbohaul.subprocess_mgr import SidecarHandle

HIDDEN_SHA = "b" * 64
VISIBLE_SHA = "a" * 64


def _make_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 12345
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


@pytest.fixture
def app_hidden_and_visible(tmp_path):
    """One hidden, one visible -- exclusion assertions always have a live
    control (a filter that empties the whole listing would pass a bare
    "hidden is absent" assertion on its own; this fixture's tests check the
    visible model survives too, matching test_manifest_hidden_listings.py's
    own established discipline)."""
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    manifests_path = storage_root / "manifests"
    manifests_path.mkdir()
    (storage_root / "import-staging").mkdir()

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
            model_tag="visible-model",
            gguf_blob_sha256=VISIBLE_SHA,
            gguf_size_bytes=1_000,
            hidden=False,
        ),
    )

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
            default_port_base=59800,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            grace_seconds=0,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            safety_enabled=False,
        ),
        pull=PullConfig(),
    )
    # auto_start_worker=True: the worker_loop must actually run for a POSTed
    # completion to be picked up and processed, matching
    # test_api_chat_completion.py's app_completion_autostart fixture.
    app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
    mgr = app.state.manager

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        return _make_handle(model_tag, port)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def fake_sigterm(handle, **kwargs):
        return True, "sigterm-clean"

    async def fake_vram(**kwargs):
        return True, 100

    async def fake_complete(slot, handle):
        # Real OpenAI-shape completion echoing which model actually served
        # the request -- lets a test assert the RIGHT manifest resolved,
        # not just that SOME 200 came back.
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": slot.model_tag,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"served by {slot.model_tag}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
        }

    mgr._spawn = fake_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = fake_sigterm
    mgr._vram_verify = fake_vram
    mgr._complete_fn = fake_complete

    with TestClient(app) as client:
        yield app, client


class TestHiddenManifestStillServes:
    """Fails RED if hiding starts ALSO blocking completions."""

    def test_hidden_model_still_completes_by_exact_name(self, app_hidden_and_visible):
        _, client = app_hidden_and_visible
        r = client.post(
            "/v1/chat/completions",
            json={"model": "hidden-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "hidden-model"
        assert body["choices"][0]["message"]["content"] == "served by hidden-model"

    def test_hidden_model_still_completes_via_ollama_chat_route(self, app_hidden_and_visible):
        """The second real completion route (/api/chat, Ollama-compat) shares
        the identical read_manifest-by-tag, no-hidden-check pattern as
        /v1/chat/completions -- covered separately since the hide/show
        UI didn't specify which client protocol a caller uses."""
        _, client = app_hidden_and_visible
        r = client.post(
            "/api/chat",
            json={"model": "hidden-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text

    def test_visible_model_still_completes_too(self, app_hidden_and_visible):
        """Control: the visible model's completion path must be unaffected
        by the hidden model's presence -- rules out a fixture-level accident
        where only ONE model in the store can ever complete."""
        _, client = app_hidden_and_visible
        r = client.post(
            "/v1/chat/completions",
            json={"model": "visible-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "visible-model"


class TestHiddenManifestAbsentFromListings:
    """Fails RED if hiding stops hiding (a DIFFERENT test from the class
    above, so that each direction is covered separately)."""

    def test_hidden_model_absent_from_listings(self, app_hidden_and_visible):
        _, client = app_hidden_and_visible
        tags = client.get("/api/tags").json()
        models = client.get("/v1/models").json()
        tag_names = {m["name"] for m in tags["models"]}
        model_ids = {m["id"] for m in models["data"]}
        assert "hidden-model" not in tag_names
        assert "hidden-model" not in model_ids
        # Control, same request: a filter that empties the WHOLE listing
        # would also pass "hidden is absent" on its own.
        assert "visible-model" in tag_names
        assert "visible-model" in model_ids


class TestHiddenManifestCombinedContract:
    """The full contract in one scenario: absent from listings AND still
    completable, together -- the two existing test files each cover one
    half in isolation; this is the first place both are asserted in the
    SAME manifest store at once, closest to the real gap."""

    def test_hidden_absent_from_listings_and_still_completes(self, app_hidden_and_visible):
        _, client = app_hidden_and_visible
        tags = client.get("/api/tags").json()
        assert "hidden-model" not in {m["name"] for m in tags["models"]}

        r = client.post(
            "/v1/chat/completions",
            json={"model": "hidden-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "hidden-model"
