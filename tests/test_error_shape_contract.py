"""Error-shape contract defects, a follow-up to an earlier error-envelope
change. Four live defects (a fifth, envelope nesting, is OUT OF
SCOPE and not changed here):

SHARED SHAPE -- `error` was a bare STRING (not a dict) on the SidecarUnavailable paths
     at three sites in chat_completion.py, plus embeddings.py's own (extended
     scope, same raise statement as the Retry-After fix). Fixed via ONE shared constructor
     (sidecar_unavailable_error_body), mirroring capacity_unavailable_error_body's
     own reasoning.
SSE TRUNCATION -- the SSE error frame truncated `message` to 500 chars; the non-stream
     body never did. RESOLVED: match the non-stream side (no truncation) --
     the cap was not a systemic wire-safety policy (1 truncating site vs 16
     unbounded ones in the same file), just an inconsistency.
UNSAFE LADDERS -- two RuntimeError ladders (chat_completion.py's SSE route, embeddings.py's
     route) had no `except VramOverCommitError` ahead of their RuntimeError
     arm, so it would be swallowed flat as a generic 500/503. LATENT, not
     live -- neither ladder's wrapped call (submit_for_streaming) can raise it
     synchronously; all 3 real construct sites (in manager.py)
     hand it to _fail_completion_future for async delivery instead. Pinned
     here via a SYNTHETIC raise (monkeypatched submit_for_streaming), which
     is the honest test for a latent path -- it does NOT claim the path is
     reachable today.
RETRY-AFTER -- embeddings.py hardcoded Retry-After: "5" instead of the exception's
     own retry_after_s. Fixed alongside the shared-shape extension to embeddings.py
     (same raise statement).

⭐ MUTATION, not a green tick: every test below fails (a real RED) when its production-code
fix is reverted -- each
test was shown to fail on the unfixed code.
"""
import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from turbohaul.api.chat_completion import (
    SidecarUnavailableError,
    _stream_error_frame,
    capacity_unavailable_error_body,
    sidecar_unavailable_error_body,
)
from turbohaul.api.main import create_app
from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig,
    RuntimePathsConfig, ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.slot import VramOverCommitError
from turbohaul.subprocess_mgr import SidecarHandle


def _make_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 12345
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _write_manifest_yaml(manifests_root, tag: str, embeddings_enabled: bool = False):
    flags_block = ""
    if embeddings_enabled:
        flags_block = "llama_server_flags:\n  embeddings: true\n"
    (manifests_root / f"{tag}.yaml").write_text(
        f"""model_tag: {tag}
gguf_blob_sha256: "{'a' * 64}"
gguf_size_bytes: 1000
{flags_block}"""
    )


@pytest.fixture
def app_and_client(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    _write_manifest_yaml(storage_root / "manifests", "m", embeddings_enabled=True)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake", default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


# ============================================================================
# Shared constructor, typed dict shape (not a bare string)
# ============================================================================


class TestSidecarUnavailableSharedShape:
    def test_shared_constructor_produces_typed_dict_not_string(self):
        """Direct unit pin on the shared constructor itself -- the contract
        this whole defect is about: `error` must be a dict with a `type`
        key, not a bare string."""
        exc = SidecarUnavailableError(
            "sidecar crashed", cause="sidecar_disconnected_or_crashed",
            retry_after_s=30,
        )
        body = sidecar_unavailable_error_body(exc)
        assert isinstance(body, dict)
        assert body["type"] == "sidecar_unavailable"
        assert body["cause"] == "sidecar_disconnected_or_crashed"
        assert body["message"] == "sidecar crashed"

    def test_openai_non_stream_uses_the_dict_shape_end_to_end(self, app_and_client):
        app, client = app_and_client
        mgr = app.state.manager

        async def fake_submit_and_wait(*a, **k):
            raise SidecarUnavailableError(
                "sidecar crashed", cause="sidecar_disconnected_or_crashed",
                retry_after_s=30,
            )

        mgr.submit_and_wait = fake_submit_and_wait
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503, r.text
        err = r.json()["detail"]["error"]
        assert isinstance(err, dict), (
            f"error must be a dict, not {type(err).__name__} -- got {err!r}"
        )
        assert err["type"] == "sidecar_unavailable"
        assert err["cause"] == "sidecar_disconnected_or_crashed"

    def test_embeddings_uses_the_same_shared_shape_end_to_end(self, app_and_client):
        """Extended scope: embeddings.py's ORIGINAL
        raise statement -- same statement the Retry-After fix touches --
        now goes through the identical shared constructor as
        chat_completion.py's 3 sites."""
        app, client = app_and_client
        mgr = app.state.manager

        async def fake_submit_for_streaming(*a, **k):
            raise SidecarUnavailableError(
                "sidecar crashed", cause="sidecar_disconnected_or_crashed",
                retry_after_s=42,
            )

        mgr.submit_for_streaming = fake_submit_for_streaming
        r = client.post(
            "/v1/embeddings", json={"model": "m", "input": "hi"},
        )
        assert r.status_code == 503, r.text
        err = r.json()["detail"]["error"]
        assert isinstance(err, dict), (
            f"error must be a dict, not {type(err).__name__} -- got {err!r}"
        )
        assert err["type"] == "sidecar_unavailable"
        assert err["cause"] == "sidecar_disconnected_or_crashed"


# ============================================================================
# Retry-After -- embeddings.py must use the exception's OWN retry_after_s, not the
# hardcoded "5"
# ============================================================================


class TestRealRetryAfter:
    def test_embeddings_advertises_the_real_retry_after_not_hardcoded_5(
        self, app_and_client,
    ):
        app, client = app_and_client
        mgr = app.state.manager

        async def fake_submit_for_streaming(*a, **k):
            raise SidecarUnavailableError(
                "sidecar crashed", cause="sidecar_disconnected_or_crashed",
                retry_after_s=77,  # deliberately NOT 5, to prove it's not hardcoded
            )

        mgr.submit_for_streaming = fake_submit_for_streaming
        r = client.post(
            "/v1/embeddings", json={"model": "m", "input": "hi"},
        )
        assert r.status_code == 503, r.text
        assert r.headers.get("Retry-After") == "77", (
            f"expected the exception's own retry_after_s (77), not the "
            f"hardcoded 5; got {r.headers.get('Retry-After')!r}"
        )


# ============================================================================
# Unsafe ladders -- both previously-unsafe ladders now catch VramOverCommitError ahead of
# RuntimeError. SYNTHETIC raise (the honest test for a LATENT path)
# -- monkeypatches submit_for_streaming to raise it directly.
# ============================================================================


class TestSyntheticRaiseInUnsafeLadders:
    def test_chat_sse_ladder_catches_vram_over_commit_not_flat_500(
        self, app_and_client,
    ):
        """chat_completion.py's SSE route (its try block) had NO
        VramOverCommitError arm ahead of its bare RuntimeError catch, which
        raised a flat HTTP 500 f"sidecar failed: {e}" string. Now it must
        surface the SAME typed 503 capacity shape the two SAFE ladders
        already produce."""
        app, client = app_and_client
        mgr = app.state.manager

        async def fake_submit_for_streaming(*a, **k):
            raise VramOverCommitError("no room", retry_after_s=13)

        mgr.submit_for_streaming = fake_submit_for_streaming
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert r.status_code == 503, r.text
        assert r.headers.get("Retry-After") == "13"
        err = r.json()["detail"]["error"]
        assert err["type"] == "capacity_unavailable", (
            f"expected the typed capacity shape, not a flat 500 string; "
            f"got status={r.status_code} detail={r.text!r}"
        )

    def test_embeddings_ladder_catches_vram_over_commit_not_flat_503(
        self, app_and_client,
    ):
        """embeddings.py's route (its try block) had NO
        VramOverCommitError arm ahead of its bare RuntimeError catch, which
        raised a flat 503 with a HARDCODED Retry-After and a string detail.
        Now it must route through _raise_capacity_or_routing_failure, the
        SAME shared site the discrimination branch below it already uses."""
        app, client = app_and_client
        mgr = app.state.manager

        async def fake_submit_for_streaming(*a, **k):
            raise VramOverCommitError("no room", retry_after_s=13)

        mgr.submit_for_streaming = fake_submit_for_streaming
        r = client.post(
            "/v1/embeddings", json={"model": "m", "input": "hi"},
        )
        assert r.status_code == 503, r.text
        assert r.headers.get("Retry-After") == "13"
        err = r.json()["detail"]["error"]
        assert err["type"] == "capacity_unavailable"


# ============================================================================
# SSE frame no longer truncates `message` to 500 chars
# ============================================================================


class TestNoTruncation:
    def test_stream_error_frame_does_not_truncate_a_long_message(self):
        long_message = "x" * 5000
        frame = _stream_error_frame("sidecar_unavailable", long_message)
        body = json.loads(frame.decode().split("data: ", 1)[1].strip())
        assert body["error"]["message"] == long_message, (
            f"expected the full {len(long_message)}-char message untruncated; "
            f"got {len(body['error']['message'])} chars"
        )

    def test_stream_and_non_stream_bodies_carry_the_same_untruncated_message(
        self,
    ):
        """The actual truncation defect: same keys, different values for a long
        message. Build both halves from the SAME production functions (not
        hardcoded literals), mirroring the unified capacity-error-shape test's
        own established pattern."""
        long_message = "y" * 2000
        exc = VramOverCommitError(long_message, retry_after_s=10)
        nonstream_body = capacity_unavailable_error_body(exc)
        frame = _stream_error_frame(
            nonstream_body["type"], nonstream_body["message"],
            cause=nonstream_body["cause"], retry_after=nonstream_body["retry_after"],
        )
        stream_body = json.loads(frame.decode().split("data: ", 1)[1].strip())
        assert stream_body["error"]["message"] == nonstream_body["message"]
        assert len(stream_body["error"]["message"]) == 2000
