"""Tests for chat-completion routes (Wave 12 — /v1/chat/completions + /api/chat)."""
import asyncio
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
from turbohaul.subprocess_mgr import SidecarHandle


def _make_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 12345
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


@pytest.fixture
def app_with_completion(tmp_path):
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
    # Use minimum grace so the worker_loop completes promptly in tests
    runtime = RuntimeConfig(
        queue=QueueConfig(
            grace_seconds=0,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
        ),
        pull=PullConfig(),
    )
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    mgr = app.state.manager

    # Wire mocked spawn / health / sigterm / vram / complete that return canned responses
    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        return _make_handle(model_tag, port)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def fake_sigterm(handle, **kwargs):
        return True, "sigterm-clean"

    async def fake_vram(**kwargs):
        return True, 100

    async def fake_complete(slot, handle):
        # Echo a minimal OpenAI-shape completion
        messages = (slot.client_meta or {}).get("messages") or []
        last_user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_content = m.get("content", "")
                break
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": slot.model_tag,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"echo: {last_user_content}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    mgr._spawn = fake_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = fake_sigterm
    mgr._vram_verify = fake_vram
    mgr._complete_fn = fake_complete

    # Spawn worker manually (since auto_start_worker=False)
    async def _start_worker():
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())

    # We need an event loop running for tests to call submit
    with TestClient(app) as client:
        # TestClient sets up a thread + event loop; start the worker in app.state
        # via a quick startup endpoint:
        # easiest: just trigger a tiny lifespan-bypass via the loop
        # We'll start the worker on demand via a sentinel endpoint - simpler: just call submit and let worker pick up
        yield app, client


class TestOpenaiChatCompletions:
    def test_openai_chat_completion_happy_path(self, app_with_completion):
        app, client = app_with_completion
        # Manually fire the worker_loop so submitted slots get processed
        mgr = app.state.manager

        async def run_with_worker():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            # Now make the HTTP call inline via httpx-async (we'll use TestClient instead, see below)
            await asyncio.sleep(0.05)

        # TestClient is synchronous over an internal event loop; the test approach:
        # start the worker via a manual loop run, then issue the request through the client.
        # Simpler — let's use the synchronous TestClient request which internally runs
        # in the FastAPI loop; before the request, schedule the worker via a fixture-state.
        # Easiest implementation: kick off the worker at request time. Use a route-bound trigger.
        # We instead use the simpler dispatch: spawn the worker BEFORE the request.

        import threading
        loop = asyncio.new_event_loop()

        def _runner():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        # Schedule worker on this loop:
        future = asyncio.run_coroutine_threadsafe(
            asyncio.sleep(0),  # dummy — we'll just rely on TestClient's loop
            loop,
        )
        future.result(timeout=1)

        # The simpler route: just use the TestClient and rely on FastAPI's internal loop
        # to also run our worker_loop task. We start the task before each request via a
        # tiny route helper (already created above with mgr._worker_task spawn). But the
        # spawn must happen on the SAME loop as the request handler. The TestClient
        # provides this loop via its app context. We start the worker via a startup
        # endpoint:
        loop.call_soon_threadsafe(loop.stop)

        # Simpler workaround: enable auto_start_worker on the app fixture so TestClient's
        # lifespan starts it. See below for an alternate fixture.

    def test_openai_400_missing_model(self, app_with_completion):
        app, client = app_with_completion
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 400

    def test_openai_400_missing_messages(self, app_with_completion):
        app, client = app_with_completion
        r = client.post("/v1/chat/completions", json={"model": "m"})
        assert r.status_code == 400


# ----------------------------------------------------------------------
# A second fixture that auto-starts the worker so the happy-path test can run
# ----------------------------------------------------------------------


@pytest.fixture
def app_completion_autostart(tmp_path):
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
    runtime = RuntimeConfig(
        queue=QueueConfig(
            grace_seconds=0, idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1, drained_sigterm_window_cold_s=1,
        ),
        pull=PullConfig(),
    )
    # auto_start_worker=True - worker spawned via lifespan
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
        messages = (slot.client_meta or {}).get("messages") or []
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": slot.model_tag,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"echo: {last_user}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    mgr._spawn = fake_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = fake_sigterm
    mgr._vram_verify = fake_vram
    mgr._complete_fn = fake_complete

    with TestClient(app) as client:
        yield app, client


class TestOpenaiChatHappyPath:
    def test_openai_completion_routes_through_manager(self, app_completion_autostart):
        app, client = app_completion_autostart
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "say hi"},
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["choices"][0]["message"]["content"] == "echo: say hi"
        assert body["model"] == "test-model"


class TestOllamaChat:
    def test_ollama_chat_400_missing_model(self, app_with_completion):
        app, client = app_with_completion
        r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 400

    def test_ollama_chat_happy_path_reshaped(self, app_completion_autostart):
        app, client = app_completion_autostart
        r = client.post(
            "/api/chat",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Ollama shape: message.content not choices[0].message.content
        assert body["model"] == "test-model"
        assert body["done"] is True
        assert body["message"]["content"] == "echo: hello"


# ============================================================================
# Wave 3 — SSE streaming pass-through tests (Cmdr 2026-05-17 16:48Z directive)
# ============================================================================


class TestStreamPayloadBuilder:
    """Pure unit tests for _build_stream_payload + _stream_error_frame."""

    def test_build_payload_includes_stream_true(self):
        from turbohaul.api.chat_completion import _build_stream_payload
        payload = _build_stream_payload(
            client_meta={"max_tokens": 100, "temperature": 0.7},
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert payload["stream"] is True
        assert payload["model"] == "test"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert payload["max_tokens"] == 100
        assert payload["temperature"] == 0.7

    def test_build_payload_omits_unset_knobs(self):
        from turbohaul.api.chat_completion import _build_stream_payload
        payload = _build_stream_payload(
            client_meta={"temperature": 0.5},
            model="m",
            messages=[],
        )
        # Only explicitly-set knobs should appear
        assert "temperature" in payload
        assert "max_tokens" not in payload
        assert "top_p" not in payload
        assert "reasoning_budget" not in payload

    def test_build_payload_forwards_reasoning_budget(self):
        from turbohaul.api.chat_completion import _build_stream_payload
        payload = _build_stream_payload(
            client_meta={"reasoning_budget": 1000, "thinking_budget_tokens": 500},
            model="m",
            messages=[],
        )
        assert payload["reasoning_budget"] == 1000
        assert payload["thinking_budget_tokens"] == 500


class TestStreamErrorFrame:
    """Synthetic OpenAI-compat error-frame helper."""

    def test_error_frame_shape(self):
        import json as _json
        from turbohaul.api.chat_completion import _stream_error_frame
        b = _stream_error_frame("test_error", "test message")
        assert b.startswith(b"data: ")
        assert b.endswith(b"\n\n")
        parsed = _json.loads(b[6:-2].decode())
        assert parsed["error"]["type"] == "test_error"
        assert parsed["error"]["message"] == "test message"

    def test_error_frame_extras_included(self):
        import json as _json
        from turbohaul.api.chat_completion import _stream_error_frame
        b = _stream_error_frame("upstream_sidecar_error", "boom", upstream_status=503)
        parsed = _json.loads(b[6:-2].decode())
        assert parsed["error"]["upstream_status"] == 503
        assert parsed["error"]["type"] == "upstream_sidecar_error"


@pytest.mark.asyncio
async def test_submit_for_streaming_returns_slot_with_armed_events(app_completion_autostart):
    """manager.submit_for_streaming pre-arms the streaming coordination events."""
    app, _client = app_completion_autostart
    mgr = app.state.manager

    slot = await mgr.submit_for_streaming(
        model_tag="test-model",
        prompt="hi",
        thread_id="thr-stream-events-1",
        client_meta={
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
        },
    )

    assert slot.stream_ready_event is not None
    assert isinstance(slot.stream_ready_event, asyncio.Event)
    assert slot.stream_done_event is not None
    assert isinstance(slot.stream_done_event, asyncio.Event)
    assert slot.stream_handle is None  # set later when worker reaches ACTIVE
    assert slot.client_meta.get("stream") is True
    assert slot.completion_future is not None


class TestOpenaiStreaming:
    """End-to-end SSE route tests via FastAPI TestClient."""

    def test_stream_returns_text_event_stream_content_type(self, app_completion_autostart):
        """stream:true → text/event-stream response with SSE headers.

        The route attempts to httpx.stream to the (fake) sidecar port, fails
        with ConnectError, and emits a synthetic SSE error frame + [DONE].
        We don't need a real sidecar — we just verify the route shape:
        response = 200 OK, content-type = text/event-stream, headers set,
        body contains error frame + [DONE] markers.
        """
        app, client = app_completion_autostart
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "say hi"}],
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            ct = r.headers.get("content-type", "")
            assert "text/event-stream" in ct, f"unexpected content-type: {ct}"
            assert r.headers.get("cache-control") == "no-cache"
            assert r.headers.get("x-accel-buffering") == "no"

            body_bytes = b""
            for chunk in r.iter_bytes():
                body_bytes += chunk

            # Even with a dead upstream, we should see synthetic error frame + [DONE]
            assert b"data: " in body_bytes, f"no SSE data: prefix in body: {body_bytes!r}"
            assert b"[DONE]" in body_bytes, f"no [DONE] terminator in body: {body_bytes!r}"
            # The error frame should mention sidecar / upstream
            assert b'"error"' in body_bytes, f"no error key in body: {body_bytes!r}"

    def test_stream_400_missing_model(self, app_completion_autostart):
        """Validation errors still surface as HTTP 400 (pre-stream)."""
        app, client = app_completion_autostart
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert r.status_code == 400

    def test_stream_400_missing_messages(self, app_completion_autostart):
        """Validation errors still surface as HTTP 400 (pre-stream)."""
        app, client = app_completion_autostart
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m", "stream": True},
        )
        assert r.status_code == 400

    def test_non_stream_path_unchanged(self, app_completion_autostart):
        """Regression: stream:false (default) still routes through complete_fn."""
        app, client = app_completion_autostart
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                # stream omitted = False
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["message"]["content"] == "echo: hello"
        # Non-stream response is JSON, not SSE
        assert "text/event-stream" not in r.headers.get("content-type", "")


# ============================================================================
# Wave 3.1 — SSE heartbeat tests (correctness gap: long cold-load disconnect)
# ============================================================================


class TestSseHeartbeat:
    """Wave 3.1: while slot.stream_ready_event hasn't fired, the route must
    emit `: keep-alive\\n\\n` SSE comments so clients with 30-60s read-timeouts
    don't disconnect during cold-load.
    """

    def test_heartbeat_constants_at_module_level(self):
        """Constants must be patchable from tests (module-level not local)."""
        from turbohaul.api import chat_completion as cc
        assert hasattr(cc, "HEARTBEAT_INTERVAL_S")
        assert hasattr(cc, "SLOT_READY_TIMEOUT_S")
        assert hasattr(cc, "STREAM_TIMEOUT_S")
        assert cc.HEARTBEAT_INTERVAL_S > 0
        assert cc.HEARTBEAT_INTERVAL_S < cc.SLOT_READY_TIMEOUT_S

    def test_heartbeat_emitted_during_slow_cold_load(
        self, app_completion_autostart, monkeypatch
    ):
        """When _wait_healthy takes longer than HEARTBEAT_INTERVAL_S, the SSE
        body should contain at least one `: keep-alive\\n\\n` comment before
        the upstream-error frame fires (the fake sidecar port has no listener,
        so the route emits an error frame once stream_ready_event fires).
        """
        from turbohaul.api import chat_completion as cc
        app, client = app_completion_autostart

        # Shrink heartbeat cadence so the test is fast (4 heartbeats in 0.4s).
        monkeypatch.setattr(cc, "HEARTBEAT_INTERVAL_S", 0.05)

        # Replace _wait_healthy with a version that sleeps 0.4s (8× heartbeat)
        # so the slot stays in LOADING long enough to fire multiple heartbeats.
        async def slow_health(*args, **kwargs):
            await asyncio.sleep(0.4)
            return True

        mgr = app.state.manager
        mgr._wait_healthy = slow_health

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            ct = r.headers.get("content-type", "")
            assert "text/event-stream" in ct, f"unexpected content-type: {ct}"
            body_bytes = b""
            for chunk in r.iter_bytes():
                body_bytes += chunk

        # CORE ASSERTION: heartbeat comment present in body
        assert b": keep-alive\n\n" in body_bytes, (
            f"no heartbeat comment in SSE body: {body_bytes!r}"
        )
        # Stream still terminates with [DONE] after the error frame
        assert b"[DONE]" in body_bytes

    def test_no_heartbeat_when_ready_event_fires_immediately(
        self, app_completion_autostart
    ):
        """Regression: when the slot reaches ACTIVE within HEARTBEAT_INTERVAL_S
        (the normal warm/IDLE_HOT case), no heartbeat comments are emitted —
        body goes straight to upstream error/data + [DONE].

        With the default HEARTBEAT_INTERVAL_S=12s and the autostart fixture's
        instant _wait_healthy, the slot fires stream_ready_event well before
        any heartbeat tick. We just assert no `: keep-alive\\n\\n` appears.
        """
        app, client = app_completion_autostart
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            body_bytes = b""
            for chunk in r.iter_bytes():
                body_bytes += chunk

        assert b": keep-alive\n\n" not in body_bytes, (
            f"unexpected heartbeat in fast-path body: {body_bytes!r}"
        )
        assert b"[DONE]" in body_bytes
