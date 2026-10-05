"""Tests for chat-completion routes (/v1/chat/completions + /api/chat)."""
import asyncio
import re
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from turbohaul import load_verify_log
from turbohaul import manager as manager_module
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


# The same OOM engine-log signature the resident-driver/single-slot
# suites use, so a real (unstubbed) classify_load_failure call reads it as
# "oom" and the manager's own machinery -- not a test-injected shortcut --
# sets slot.oom_requeue_pending.
_OOM_SCAN = {
    "engine_errors_detected": True,
    "engine_error_lines": [
        "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 884.62 MiB "
        "on device 0: cudaMalloc failed: out of memory"
    ],
    "reason": None,
}


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
        queue=QueueConfig(safety_enabled=False,
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
    # Chat/completion routes now 404 on an unknown model tag, so the
    # tags this fixture's tests dispatch against ("m" / "test-model") need a
    # manifest present. Individual tests may still overwrite "m" via
    # _write_manifest_yaml(..., reasoning_budget=...) for their own scenario.
    _write_manifest_yaml(storage_root / "manifests", "m")
    _write_manifest_yaml(storage_root / "manifests", "test-model")
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
        queue=QueueConfig(safety_enabled=False,
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
# SSE streaming pass-through tests
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

    def test_build_payload_forwards_tool_call_fields(self):
        """The tools / tool_choice / parallel_tool_calls / function_call /
        functions must be passed through to llama-server when present. Structured
        values (list of tool defs, dict tool_choice, etc.) are forwarded as-is.
        """
        from turbohaul.api.chat_completion import _build_stream_payload
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        tool_choice_obj = {"type": "function", "function": {"name": "get_weather"}}
        payload = _build_stream_payload(
            client_meta={
                "tools": tools,
                "tool_choice": tool_choice_obj,
                "parallel_tool_calls": False,
            },
            model="qwen3.6-27b-dense",
            messages=[{"role": "user", "content": "what's the weather?"}],
        )
        assert payload["tools"] == tools
        assert payload["tool_choice"] == tool_choice_obj
        assert payload["parallel_tool_calls"] is False
        # Enum form of tool_choice also passes through.
        payload2 = _build_stream_payload(
            client_meta={"tool_choice": "auto"}, model="m", messages=[]
        )
        assert payload2["tool_choice"] == "auto"


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
# SSE heartbeat tests (correctness gap: long cold-load disconnect)
# ============================================================================


class TestSseHeartbeat:
    """While slot.stream_ready_event hasn't fired, the route must
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


# ============================================================================
# stream_gen's SLOT_READY_TIMEOUT_S deadline must not
# count down while slot.oom_requeue_pending is True (see the tests
# below).
# ============================================================================


class TestPausableReadyDeadline:
    def test_deadline_paused_while_oom_requeue_parked(
        self, app_completion_autostart, monkeypatch
    ):
        """The ready-deadline must not fire slot_ready_timeout while the slot
        is OOM-requeue-parked, even after several multiples of the (tiny,
        for test speed) SLOT_READY_TIMEOUT_S have elapsed. The slot is parked
        via the REAL mechanism (an unhealthy first attempt + a real OOM
        engine-log signature -> _requeue_on_oom_load_failure), not a
        test-injected flag -- _spawn_for_resident unconditionally clears
        oom_requeue_pending at the top of every admission attempt, so setting
        it from outside that machinery would just get wiped out. Nothing in
        this test ever calls _make_room_signal.notify_all(), so (per the
        companion's own documented backstock-timeout behaviour) the health
        check is never invoked a second time -- always-unhealthy, exactly
        like the resident-driver suite's own fake_health, never a hang.

        NOTE on ending the stream: starlette's TestClient runs the whole ASGI
        response to completion before handing any of it back to a
        synchronous ``client.stream()`` reader (proven empirically while
        writing this test -- breaking out of ``r.iter_bytes()`` early does
        NOT stop the server-side generator, so a generator that heartbeats
        forever while genuinely parked hangs the whole test). So this test
        force-completes ``stream_ready_event`` from a background task
        scheduled on the SAME loop after several heartbeat intervals --
        enough to prove the deadline didn't fire while parked -- then lets
        the route's normal "handle is None" fallback end the response.
        """
        from turbohaul.api import chat_completion as cc

        app, client = app_completion_autostart
        monkeypatch.setattr(cc, "SLOT_READY_TIMEOUT_S", 0.1)
        monkeypatch.setattr(cc, "HEARTBEAT_INTERVAL_S", 0.03)
        monkeypatch.setattr(manager_module, "_OOM_REQUEUE_POLL_S", 0.02)
        monkeypatch.setattr(manager_module, "_OOM_REQUEUE_SETTLE_S", 0.0)
        monkeypatch.setattr(
            load_verify_log, "scan_engine_log_for_errors", lambda *a, **k: _OOM_SCAN
        )

        health_calls = [0]

        async def always_unhealthy(*args, **kwargs):
            health_calls[0] += 1
            return False

        mgr = app.state.manager
        mgr._wait_healthy = always_unhealthy

        orig_submit = mgr.submit_for_streaming
        captured = {}

        async def wrapped_submit(*a, **k):
            slot = await orig_submit(*a, **k)
            captured["slot"] = slot

            async def _force_ready_after_delay():
                # ~8 heartbeats at HEARTBEAT_INTERVAL_S=0.03s (0.24s), well
                # past SLOT_READY_TIMEOUT_S=0.1s -- would already have fired
                # slot_ready_timeout if the deadline weren't exempted.
                await asyncio.sleep(0.26)
                if not slot.stream_ready_event.is_set():
                    slot.stream_ready_event.set()

            asyncio.create_task(_force_ready_after_delay())
            return slot

        mgr.submit_for_streaming = wrapped_submit

        body_bytes = b""
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
            for chunk in r.iter_bytes():
                body_bytes += chunk

        heartbeats_seen = body_bytes.count(b": keep-alive\n\n")
        assert health_calls[0] >= 1, (
            "precondition: the health check must have run to actually park "
            "the slot via the real OOM-requeue mechanism"
        )
        assert heartbeats_seen >= 6, (
            f"only {heartbeats_seen} heartbeats in the body -- not enough "
            "elapsed time to prove the tiny deadline would have fired "
            "unpaused"
        )
        assert b"slot_ready_timeout" not in body_bytes, (
            "the deadline fired WHILE the slot was still oom_requeue_pending "
            "-- it must not count down while parked"
        )
        assert b"[DONE]" in body_bytes

    def test_control_ordinary_slot_still_times_out_normally(
        self, app_completion_autostart, monkeypatch
    ):
        """Control: an ORDINARY (never OOM-requeued) slot's ready deadline is
        unaffected by the pausable-budget change -- it must still expire at
        (approximately) SLOT_READY_TIMEOUT_S. Kills a mutant that widens or
        removes the exemption check so every slot becomes exempt."""
        from turbohaul.api import chat_completion as cc

        app, client = app_completion_autostart
        monkeypatch.setattr(cc, "SLOT_READY_TIMEOUT_S", 0.1)
        monkeypatch.setattr(cc, "HEARTBEAT_INTERVAL_S", 0.03)

        async def never_healthy(*args, **kwargs):
            await asyncio.sleep(999)
            return True

        mgr = app.state.manager
        mgr._wait_healthy = never_healthy

        started = time.monotonic()
        body_bytes = b""
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
            for chunk in r.iter_bytes():
                body_bytes += chunk
        elapsed = time.monotonic() - started

        assert b"slot_ready_timeout" in body_bytes
        assert elapsed < 2.0, (
            f"an ordinary (never-parked) slot took {elapsed:.2f}s to time "
            "out on a 0.1s deadline -- the pausable-budget change must not "
            "widen the deadline for a slot that's never OOM-requeued"
        )


# ============================================================================
# stream_gen's finally sets disconnect_event ONLY when
# oom_requeue_pending. Dedicated tests.
# ============================================================================


class TestScopedFinallyDisconnectSet:
    def test_finally_sets_disconnect_event_when_oom_requeue_pending(
        self, app_completion_autostart, monkeypatch
    ):
        """When the stream ends while the slot is still OOM-requeue-parked,
        stream_gen's finally must set disconnect_event so the parked
        _oom_requeue_wait_then_readmit companion notices and finishes
        instead of waiting forever for a watch_disconnect that was
        just cancelled and will never observe anything again. Parked via the
        REAL mechanism (see the parked-deadline test above for why a test-injected
        flag doesn't survive _spawn_for_resident's own admission-start
        reset), and ended the same way the parked-deadline test ends it (a forced
        stream_ready_event from a background task on the same loop -- see
        that test's NOTE for why breaking out of the client-side read
        doesn't stop a still-running server-side generator under this
        TestClient). watch_disconnect is neutralized so disconnect_event can
        only end up set by stream_gen's own finally, not by the pre-existing
        watcher racing to the same answer."""
        from turbohaul.api import chat_completion as cc

        app, client = app_completion_autostart
        monkeypatch.setattr(cc, "SLOT_READY_TIMEOUT_S", 0.1)
        monkeypatch.setattr(cc, "HEARTBEAT_INTERVAL_S", 0.03)
        monkeypatch.setattr(manager_module, "_OOM_REQUEUE_POLL_S", 0.02)
        monkeypatch.setattr(manager_module, "_OOM_REQUEUE_SETTLE_S", 0.0)
        monkeypatch.setattr(
            load_verify_log, "scan_engine_log_for_errors", lambda *a, **k: _OOM_SCAN
        )

        async def inert_watch_disconnect(request, disconnect_event):
            await asyncio.sleep(999)

        monkeypatch.setattr(cc, "watch_disconnect", inert_watch_disconnect)

        health_calls = [0]

        async def always_unhealthy(*args, **kwargs):
            health_calls[0] += 1
            return False

        mgr = app.state.manager
        mgr._wait_healthy = always_unhealthy

        orig_submit = mgr.submit_for_streaming
        captured = {}

        async def wrapped_submit(*a, **k):
            slot = await orig_submit(*a, **k)
            captured["slot"] = slot

            async def _force_ready_after_delay():
                await asyncio.sleep(0.15)
                if not slot.stream_ready_event.is_set():
                    slot.stream_ready_event.set()

            asyncio.create_task(_force_ready_after_delay())
            return slot

        mgr.submit_for_streaming = wrapped_submit

        body_bytes = b""
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
            for chunk in r.iter_bytes():
                body_bytes += chunk

        slot = captured["slot"]
        assert health_calls[0] >= 1, (
            "precondition: the health check must have run to actually park "
            "the slot via the real OOM-requeue mechanism"
        )
        assert slot.disconnect_event.is_set(), (
            "an OOM-requeue-parked stream that ends must set "
            "disconnect_event in the finally -- watch_disconnect was "
            "neutralized above, so ONLY stream_gen's own finally could have "
            "set it here"
        )

    def test_control_ordinary_stream_finally_leaves_disconnect_event_unset(
        self, app_completion_autostart
    ):
        """Control (scope-control): an ORDINARY stream
        that ends normally (never OOM-requeue-parked) must NOT have
        disconnect_event set by stream_gen's finally -- only watch_disconnect
        (a real client disconnect) may set it for that population. Kills a
        mutant that sets disconnect_event unconditionally instead of only
        when oom_requeue_pending."""
        app, client = app_completion_autostart

        mgr = app.state.manager
        orig_submit = mgr.submit_for_streaming
        captured = {}

        async def wrapped_submit(*a, **k):
            slot = await orig_submit(*a, **k)
            captured["slot"] = slot
            return slot

        mgr.submit_for_streaming = wrapped_submit

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
            for _chunk in r.iter_bytes():
                pass

        slot = captured["slot"]
        assert not slot.disconnect_event.is_set(), (
            "an ordinary, never-parked stream's finally must leave "
            "disconnect_event untouched -- the real client never "
            "disconnected here (TestClient read the body to completion)"
        )


# ============================================================================
# Revive the typed capacity_unavailable SSE frame
# ============================================================================


class TestSseCapacityFrame:
    """The SSE ready-wait no longer times
    out on a capacity failure (stream_ready_event is now set on failure too), so
    the `except asyncio.TimeoutError` capacity check never fires for
    an exception-driven wakeup and control falls to the `handle is None` branch
    instead -- which, before this change, only knew how to emit a message-only
    `placement_failed` frame (no cause, no retry_after). This silently retired
    the typed `capacity_unavailable` frame on the streaming transport while the
    non-stream 503 kept it. The contract (both transports carry
    the SAME nested error shape) requires it back.
    """

    def test_capacity_failure_emits_typed_frame_not_placement_failed(
        self, app_completion_autostart, monkeypatch
    ):
        from turbohaul.api import chat_completion as cc
        from turbohaul.slot import VramOverCommitError

        app, client = app_completion_autostart
        mgr = app.state.manager

        # `watch_disconnect`'s cancellation
        # deadlocks in this exact scheduling shape (a capacity failure resolves
        # the ready-wait on effectively the FIRST scheduling round, starving the
        # disconnect watcher of its first tick before `finally` cancels it).
        # This was also reproduced on a REAL uvicorn socket and
        # confirmed to wedge for real -- it is NOT a TestClient-only artifact.
        # Out of scope here. Stub it here only
        # to keep THIS test's own runtime bounded.
        #
        # Scope of that wedge, measured on a real socket:
        # the wedge is EMBEDDINGS-ROUTE-SPECIFIC. On the identical
        # already-failed-at-entry timing the SSE path returns this typed frame
        # immediately and does NOT wedge, while /v1/embeddings never returns.
        # So stubbing here is belt-and-braces for THIS path, not load-bearing.
        async def _stub_watch_disconnect(request, disconnect_event):
            await disconnect_event.wait()

        monkeypatch.setattr(cc, "watch_disconnect", _stub_watch_disconnect)

        async def fake_submit_capacity_fail(model_tag, prompt="", thread_id="",
                                            client_meta=None, **kwargs):
            slot = MagicMock()
            slot.stream_ready_event = asyncio.Event()
            exc = VramOverCommitError("no VRAM capacity", retry_after_s=7)
            fut = asyncio.get_running_loop().create_future()
            fut.set_exception(exc)
            slot.completion_future = fut
            # PRECONDITION, constructed directly -- NOT coverage of the manager.
            # This fake replaces mgr.submit_for_streaming wholesale, so
            # manager.py::_fail_completion_future never runs in this test. The
            # two lines below are this fixture standing in for what the manager
            # would have done on a capacity failure, so
            # that what actually gets tested is the ROUTE's behaviour: 503
            # capacity_unavailable with retry_after. Same framing as the matching
            # comment in test_api_embeddings.py.
            # ⛔ Do not read this as gating the manager. It cannot -- it asserts
            # what it just performed. The manager side is gated by
            # the fail-completion-future wake-up test;
            # before that test existed, the wake-up code could be deleted
            # outright with this whole suite still green.
            slot.stream_ready_failed_reason = str(exc)
            slot.stream_ready_event.set()
            slot.stream_handle = None
            slot.stream_done_event = asyncio.Event()
            slot.slot_id = "cap-fail"
            slot.thread_id = thread_id
            slot.model_tag = model_tag
            slot.client_meta = client_meta or {}
            return slot

        mgr.submit_for_streaming = fake_submit_capacity_fail

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

        import json as _json
        data_lines = [
            line[len(b"data: "):] for line in body_bytes.split(b"\n\n")
            if line.startswith(b"data: ") and line != b"data: [DONE]"
        ]
        assert data_lines, f"no error frame found in SSE body: {body_bytes!r}"
        err = _json.loads(data_lines[0])["error"]
        assert err["type"] == "capacity_unavailable", (
            f"expected the typed capacity frame, got {err!r} "
            f"(placement_failed means the typed frame is still silently retired)"
        )
        assert err["cause"] == "vram_over_commit"
        assert err["retry_after"] == 7
        assert "no VRAM capacity" in err["message"]


# ============================================================================
# keep_alive parser + client_meta plumbing
# ============================================================================


class TestParseKeepAlive:
    """parse_keep_alive returns Turbohaul's capped Ollama-style retention value."""

    def test_none_returns_none(self):
        from turbohaul.api.chat_completion import parse_keep_alive
        assert parse_keep_alive(None) is None

    def test_int_passthrough(self):
        from turbohaul.api.chat_completion import parse_keep_alive
        assert parse_keep_alive(60) == 60
        assert parse_keep_alive(0) == 0
        assert parse_keep_alive(-1) == 1800  # negative means capped pin-warm
        assert parse_keep_alive(10000) == 10000  # caller clamps

    def test_float_truncates_to_int(self):
        from turbohaul.api.chat_completion import parse_keep_alive
        assert parse_keep_alive(60.7) == 60
        assert parse_keep_alive(-1.0) == 1800

    def test_string_int_forms(self):
        from turbohaul.api.chat_completion import parse_keep_alive
        assert parse_keep_alive("0") == 0
        assert parse_keep_alive("-1") == 1800
        assert parse_keep_alive("60") == 60
        assert parse_keep_alive(" 60 ") == 60  # strip
        assert parse_keep_alive("") is None

    def test_string_ollama_suffix_forms(self):
        from turbohaul.api.chat_completion import parse_keep_alive
        assert parse_keep_alive("30s") == 30
        assert parse_keep_alive("5m") == 300
        assert parse_keep_alive("2h") == 7200
        # Case-insensitive
        assert parse_keep_alive("5M") == 300

    def test_string_invalid_returns_none(self):
        from turbohaul.api.chat_completion import parse_keep_alive
        assert parse_keep_alive("abc") is None
        assert parse_keep_alive("1x") is None  # unknown unit
        assert parse_keep_alive("m") is None  # no digit
        # Day/week suffixes intentionally NOT supported (over-engineering
        # -- Ollama itself documents s/m/h only).
        assert parse_keep_alive("1d") is None

    def test_bool_false_means_zero(self):
        """Ollama keep_alive: false → unload immediately (matches `0`).

        Regression guard — without this, real Ollama
        clients sending {"keep_alive": false} would silently fall through
        to default 300s instead of the immediate teardown they asked for.
        """
        from turbohaul.api.chat_completion import parse_keep_alive
        assert parse_keep_alive(False) == 0

    def test_bool_true_means_use_default(self):
        """Ollama keep_alive: true → "on", let server pick default."""
        from turbohaul.api.chat_completion import parse_keep_alive
        assert parse_keep_alive(True) is None


class TestKeepAliveClientMetaPlumbing:
    """Regression: streaming path must propagate
    keep_alive_s into client_meta (it's Turbohaul-internal, NOT forwarded to
    llama-server). Agent clients run streaming-only — this is THE fix.
    """

    def test_keep_alive_NOT_in_stream_payload(self):
        """keep_alive_s is Turbohaul-internal — it must NOT leak to llama-server.

        The streaming payload helper consumes _STREAM_FORWARDED_KNOBS, which
        does NOT include keep_alive_s (or keep_alive). Verifying this prevents
        accidental future regressions where someone adds it to the knob tuple.
        """
        from turbohaul.api.chat_completion import (
            _STREAM_FORWARDED_KNOBS,
            _build_stream_payload,
        )
        assert "keep_alive" not in _STREAM_FORWARDED_KNOBS
        assert "keep_alive_s" not in _STREAM_FORWARDED_KNOBS
        payload = _build_stream_payload(
            client_meta={"keep_alive_s": 600, "temperature": 0.5},
            model="m",
            messages=[],
        )
        assert "keep_alive" not in payload
        assert "keep_alive_s" not in payload
        # Sanity: forwarded knobs still flow.
        assert payload["temperature"] == 0.5


# ============================================================================
# response_format json_object MVP contract tests
# ============================================================================


class TestResponseFormatContract:
    """Contract tests — accept json_object, reject json_schema, no regression.

    Pattern: override mgr._complete_fn (or _build_stream_payload monkeypatch)
    to capture the slot.client_meta / stream-payload that crosses the
    route-to-backend boundary. Assert response_format presence/absence on the
    captured value — NOT echo-handler tautologies.
    """

    @staticmethod
    def _install_client_meta_capture(app):
        """Swap mgr._complete_fn with a capturer that records slot.client_meta."""
        captured: list[dict] = []

        async def capturing_complete(slot, handle):
            captured.append(dict(slot.client_meta or {}))
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1700000000,
                "model": slot.model_tag,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        app.state.manager._complete_fn = capturing_complete
        return captured

    @staticmethod
    def _install_stream_payload_capture(monkeypatch):
        """Wrap _build_stream_payload to record its client_meta input."""
        from turbohaul.api import chat_completion as cc
        captured: list[dict] = []
        original = cc._build_stream_payload

        def capturing(client_meta, model, messages):
            captured.append(dict(client_meta or {}))
            return original(client_meta, model, messages)

        monkeypatch.setattr(cc, "_build_stream_payload", capturing)
        return captured

    # (a) openai stream + json_object → reaches stream payload
    def test_a_openai_stream_json_object_in_stream_payload(
        self, app_completion_autostart, monkeypatch,
    ):
        app, client = app_completion_autostart
        captured = self._install_stream_payload_capture(monkeypatch)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "response_format": {"type": "json_object"},
            },
        ) as r:
            assert r.status_code == 200
            for _ in r.iter_bytes():
                pass  # drain — upstream connect-error → synthetic SSE error frame
        assert captured, "_build_stream_payload was never invoked"
        assert captured[0].get("response_format") == {"type": "json_object"}

    # (b) openai NON-stream + json_object → present in captured client_meta
    def test_b_openai_nonstream_json_object_in_client_meta(
        self, app_completion_autostart,
    ):
        app, client = app_completion_autostart
        captured = self._install_client_meta_capture(app)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
            },
        )
        assert r.status_code == 200, r.text
        assert captured, "_complete_fn was never invoked"
        assert captured[0].get("response_format") == {"type": "json_object"}

    # (c) openai json_schema body, validated as a schema.
    # json_schema is validated, not rejected as unsupported (no HTTP 400 "not yet supported").
    # A bad schema gives 422 schema_validation_failed.
    # This test guards the 422 contract for the bad-schema path (object
    # type missing additionalProperties: false).
    # See the strict json_schema test class for the comprehensive json_schema set.
    def test_c_openai_json_schema_validation_failed_422(self, app_completion_autostart):
        app, client = app_completion_autostart
        # Use the canonical OpenAI structured-outputs envelope (json_schema:{schema:...})
        # with an object type missing additionalProperties: false → 422 schema_validation_failed.
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}, "name": "T"},
            },
        }
        # Non-stream variant
        r1 = client.post("/v1/chat/completions", json=body)
        assert r1.status_code == 422, r1.text
        d1 = r1.json()["detail"]
        assert d1["error"] == "schema_validation_failed"
        assert "additionalProperties" in d1["message"]
        assert not re.search(r"\b(?:RC|SUB|WO)-\w+", str(d1))  # no internal ticket-id leak
        # Stream variant — must also reject at entry, BEFORE any SSE
        body_stream = {**body, "stream": True}
        r2 = client.post("/v1/chat/completions", json=body_stream)
        assert r2.status_code == 422, r2.text
        assert r2.json()["detail"]["error"] == "schema_validation_failed"

    # (d) openai no response_format → not in captured client_meta (None pass-through)
    def test_d_openai_no_response_format_is_none(self, app_completion_autostart):
        app, client = app_completion_autostart
        captured = self._install_client_meta_capture(app)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert captured[0].get("response_format") is None

    # (e) ollama stream=True + json_object → response_format carried in client_meta.
    # Regression guard (previously skipped): ollama_chat
    # + stream=True must not hang (manager._STREAM_TIMEOUT_S, 3600s) inside
    # manager.submit()'s stream-event arming: submit() would otherwise arm
    # slot.stream_ready_event/stream_done_event off client_meta["stream"] ALONE,
    # regardless of which wrapper called it, so ollama_chat's plain
    # submit_and_wait() (it never calls submit_for_streaming and never drives
    # its own httpx stream, unlike openai_chat_completions) would be routed into the
    # streaming wait branch in _serve_on_resident anyway. Nothing on the
    # ollama_chat path ever sets stream_done_event, so the worker would block on
    # asyncio.wait_for(..., timeout=_STREAM_TIMEOUT_S) for the full 3600s before
    # falling through with a placeholder {"_streamed": True} result -- and
    # because that path never reaches _complete_fn at all, `captured` would stay
    # empty and `captured[0]` would raise IndexError, which is the real pre-fix
    # failure mode this test proves causally (see the timeout-patch below).
    #
    # THE FIX: manager.submit() requires an explicit, wrapper-controlled
    # `arm_stream_events=True` (only submit_for_streaming passes it) in
    # addition to client_meta["stream"] before it ever creates the events --
    # client-influenced data alone can no longer arm an internal protocol
    # handshake. ollama_chat's own client_meta["stream"] is also
    # unconditionally False (it has no streaming implementation regardless of
    # what the client asked for), with the client's original request recorded
    # separately as client_meta["stream_requested"] for observability.
    #
    # Background:
    # NOTE: ollama_chat does NOT branch to a streaming helper
    # (unlike openai_chat_completions); the stream flag flows into client_meta
    # but the upstream POST goes through _complete_fn non-streaming. This test
    # guards the "stream flag does NOT strip response_format" regression on the
    # ollama path. The validator still fires (covered by test_g stream variant).
    def test_e_ollama_stream_flag_does_not_strip_response_format(
        self, app_completion_autostart, monkeypatch,
    ):
        # Patch the real 3600s ceiling down to a tiny value BEFORE
        # this request fires. Safe either way this test runs: post-fix the
        # streaming branch is never entered at all (arm_stream_events is False
        # for submit_and_wait), so this timeout is never consulted; pre-fix
        # (verified by reverting the fix) it is
        # exactly what turns an unsafe 3600s real wait into a fast, deterministic
        # reproduction of the same TimeoutError code path -- never a live engine,
        # never an unbounded hang, so the run is bounded and
        # deterministic.
        import turbohaul.manager as manager_mod
        monkeypatch.setattr(manager_mod, "_STREAM_TIMEOUT_S", 0.05)

        app, client = app_completion_autostart
        captured = self._install_client_meta_capture(app)
        r = client.post(
            "/api/chat",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "response_format": {"type": "json_object"},
            },
        )
        assert r.status_code == 200, r.text
        assert len(captured) == 1, (
            f"expected exactly one _complete_fn call, got {captured} -- if this "
            "is empty, ollama_chat's request never reached the non-streaming "
            "completion path at all (the pre-fix hang shape)"
        )
        assert captured[0].get("response_format") == {"type": "json_object"}
        # The client's request IS recorded, just no longer under the
        # key that arms the manager's streaming handshake (see comment above).
        assert captured[0].get("stream") is False
        assert captured[0].get("stream_requested") is True

    # (f) ollama NON-stream + json_object → present in captured client_meta
    def test_f_ollama_nonstream_json_object_in_client_meta(
        self, app_completion_autostart,
    ):
        app, client = app_completion_autostart
        captured = self._install_client_meta_capture(app)
        r = client.post(
            "/api/chat",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
            },
        )
        assert r.status_code == 200, r.text
        assert captured[0].get("response_format") == {"type": "json_object"}

    # (g) ollama json_schema body, validated as a schema.
    # A bad schema gives 422 schema_validation_failed. See test_c_openai
    # docstring + the strict json_schema test class for the full json_schema set.
    def test_g_ollama_json_schema_validation_failed_422(self, app_completion_autostart):
        app, client = app_completion_autostart
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}, "name": "T"},
            },
        }
        r1 = client.post("/api/chat", json=body)
        assert r1.status_code == 422
        d1 = r1.json()["detail"]
        assert d1["error"] == "schema_validation_failed"
        assert "additionalProperties" in d1["message"]
        assert not re.search(r"\b(?:RC|SUB|WO)-\w+", str(d1))
        r2 = client.post("/api/chat", json={**body, "stream": True})
        assert r2.status_code == 422

    # (h) ollama no response_format → not in captured client_meta
    def test_h_ollama_no_response_format_is_none(self, app_completion_autostart):
        app, client = app_completion_autostart
        captured = self._install_client_meta_capture(app)
        r = client.post(
            "/api/chat",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert captured[0].get("response_format") is None

    # (NEG sentinel) merger skip-branch fires for json_object — no <think> wrap
    def test_neg_merge_reasoning_skipped_for_json_object(self):
        from turbohaul.api.chat_completion import _merge_reasoning_into_content

        result = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"a": 1}',
                    "reasoning_content": "internal chain-of-thought blob",
                },
                "finish_reason": "stop",
            }],
        }
        _merge_reasoning_into_content(
            result, response_format={"type": "json_object"},
        )
        content = result["choices"][0]["message"]["content"]
        assert content == '{"a": 1}', (
            "skip-branch failed — json_object content was modified by merger; "
            "thinking-mode JSON corruption regression"
        )
        assert "<think>" not in content

        # Control: same input with response_format=None DOES merge
        result2 = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"a": 1}',
                    "reasoning_content": "internal chain-of-thought blob",
                },
                "finish_reason": "stop",
            }],
        }
        _merge_reasoning_into_content(result2, response_format=None)
        assert "<think>" in result2["choices"][0]["message"]["content"]

    # (UNIT smoke) _build_stream_payload carries response_format through
    def test_unit_build_stream_payload_carries_response_format(self):
        from turbohaul.api.chat_completion import _build_stream_payload

        payload = _build_stream_payload(
            client_meta={"response_format": {"type": "json_object"}},
            model="m",
            messages=[],
        )
        assert payload.get("response_format") == {"type": "json_object"}


# ============================================================================
# response_format json_schema FULL contract tests
# ============================================================================


def _write_manifest_yaml(manifests_root, tag: str, reasoning_budget: int = 0):
    """Helper: write a minimal valid manifest with optional reasoning_budget."""
    flags_block = ""
    if reasoning_budget:
        flags_block = f"llama_server_flags:\n  reasoning_budget: {reasoning_budget}\n"
    (manifests_root / f"{tag}.yaml").write_text(
        f"""model_tag: {tag}
gguf_blob_sha256: "{'a' * 64}"
gguf_size_bytes: 1000
{flags_block}"""
    )


class TestResponseFormatJsonSchemaFull:
    """json_schema FULL contract tests — validate+retry+thinking-strip.

    Coverage:
      - Valid schema (with additionalProperties: false) → 200 + propagation
      - 5 DoS-variant schemas → 422 with specific reason
      - Missing json_schema field → 422
      - is_thinking_payload helper unit
      - _strip_thinking_wrapper helper unit (incl rsplit malformed-multi-tag)
      - thinking_manifest stashed in client_meta on json_schema request
      - jsonschema lazy-import fail-soft on missing dep (sim)
      - _complete retry happy path (validation passes after retry overlay)
      - _complete retry exhaustion → SidecarUpstreamError 502
      - streaming + thinking + json_schema carve-out (validator at entry, no retry)
    """

    @staticmethod
    def _install_client_meta_capture(app):
        """Reuse the json_object capture pattern — record slot.client_meta passed to _complete_fn."""
        captured: list[dict] = []

        async def capturing_complete(slot, handle):
            captured.append(dict(slot.client_meta or {}))
            return {
                "id": "chatcmpl-test", "object": "chat.completion",
                "created": 1700000000, "model": slot.model_tag,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"x": 1}'},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        app.state.manager._complete_fn = capturing_complete
        return captured

    # (a) Valid json_schema with additionalProperties: false → 200 + propagation
    def test_a_valid_schema_with_addprops_false_returns_200(
        self, app_completion_autostart, tmp_path,
    ):
        app, client = app_completion_autostart
        captured = self._install_client_meta_capture(app)
        manifests_root = app.state.manager.boot.storage.manifests_path
        _write_manifest_yaml(manifests_root, "m", reasoning_budget=0)
        valid_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "integer"}},
        }
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": valid_schema, "name": "T"},
                },
            },
        )
        assert r.status_code == 200, r.text
        assert captured[0].get("response_format", {}).get("type") == "json_schema"

    # (b) Schema size exceeded → 422
    def test_b_schema_size_exceeded_returns_422(self, app_completion_autostart, tmp_path):
        app, client = app_completion_autostart
        _write_manifest_yaml(app.state.manager.boot.storage.manifests_path, "m")
        # Build a schema bigger than _SCHEMA_MAX_BYTES (65536)
        huge_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {f"k{i}": {"type": "string"} for i in range(8000)},
        }
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": huge_schema, "name": "T"},
                },
            },
        )
        assert r.status_code == 422
        d = r.json()["detail"]
        assert d["error"] == "schema_validation_failed"
        assert "schema_size_exceeded" in d["message"]

    # (c) Schema depth exceeded → 422
    def test_c_schema_depth_exceeded_returns_422(self, app_completion_autostart):
        app, client = app_completion_autostart
        _write_manifest_yaml(app.state.manager.boot.storage.manifests_path, "m")
        # Nest 20 levels deep (cap = 16)
        deep_schema: dict = {
            "type": "object",
            "additionalProperties": False,
        }
        cursor = deep_schema
        for _ in range(20):
            cursor["properties"] = {"n": {"type": "object", "additionalProperties": False}}
            cursor = cursor["properties"]["n"]
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": deep_schema, "name": "T"},
                },
            },
        )
        assert r.status_code == 422
        assert "schema_depth_exceeded" in r.json()["detail"]["message"]

    # (d) Schema property count exceeded → 422
    def test_d_schema_property_count_exceeded_returns_422(self, app_completion_autostart):
        app, client = app_completion_autostart
        _write_manifest_yaml(app.state.manager.boot.storage.manifests_path, "m")
        many_props = {
            "type": "object",
            "additionalProperties": False,
            "properties": {f"p{i}": {"type": "string"} for i in range(100)},
        }
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": many_props, "name": "T"},
                },
            },
        )
        assert r.status_code == 422
        assert "schema_property_count_exceeded" in r.json()["detail"]["message"]

    # (e) Schema contains $ref → 422
    def test_e_schema_ref_unsupported_returns_422(self, app_completion_autostart):
        app, client = app_completion_autostart
        _write_manifest_yaml(app.state.manager.boot.storage.manifests_path, "m")
        ref_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"$ref": "#/definitions/Foo"}},
            "definitions": {"Foo": {"type": "integer"}},
        }
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": ref_schema, "name": "T"},
                },
            },
        )
        assert r.status_code == 422
        assert "schema_contains_ref_unsupported" in r.json()["detail"]["message"]

    # (f) Missing json_schema envelope → 422
    def test_f_missing_json_schema_field_returns_422(self, app_completion_autostart):
        app, client = app_completion_autostart
        _write_manifest_yaml(app.state.manager.boot.storage.manifests_path, "m")
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_schema"},  # no json_schema envelope
            },
        )
        assert r.status_code == 422
        assert "missing_or_malformed_json_schema_field" in r.json()["detail"]["message"]

    # (g) ollama mirror: valid schema → 200
    def test_g_ollama_valid_schema_200(self, app_completion_autostart):
        app, client = app_completion_autostart
        _write_manifest_yaml(app.state.manager.boot.storage.manifests_path, "m")
        captured = self._install_client_meta_capture(app)
        valid_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"y": {"type": "string"}},
        }
        r = client.post(
            "/api/chat",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": valid_schema, "name": "T"},
                },
            },
        )
        assert r.status_code == 200, r.text
        assert captured[0].get("response_format", {}).get("type") == "json_schema"

    # (h) thinking_manifest stashed on client_meta when json_schema + reasoning_budget>0
    def test_h_thinking_manifest_stashed_when_thinking_model(
        self, app_completion_autostart,
    ):
        app, client = app_completion_autostart
        _write_manifest_yaml(
            app.state.manager.boot.storage.manifests_path, "m", reasoning_budget=4096,
        )
        captured = self._install_client_meta_capture(app)
        valid_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "integer"}},
        }
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": valid_schema, "name": "T"},
                },
            },
        )
        assert r.status_code == 200, r.text
        assert captured[0].get("thinking_manifest", {}).get("reasoning_budget") == 4096

    # (i) Non-thinking model: thinking_manifest reasoning_budget=0 → retry path disabled
    def test_i_non_thinking_model_thinking_manifest_zero(self, app_completion_autostart):
        app, client = app_completion_autostart
        _write_manifest_yaml(
            app.state.manager.boot.storage.manifests_path, "m", reasoning_budget=0,
        )
        captured = self._install_client_meta_capture(app)
        valid_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "integer"}},
        }
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": valid_schema, "name": "T"},
                },
            },
        )
        assert r.status_code == 200, r.text
        # Manifest read happens but reasoning_budget defaults to 0
        assert captured[0].get("thinking_manifest", {}).get("reasoning_budget", 0) == 0

    # (j) unit: is_thinking_payload helper
    def test_j_unit_is_thinking_payload(self):
        from turbohaul.api.chat_completion import is_thinking_payload

        assert is_thinking_payload({}, {"reasoning_budget": 4096}) is True
        assert is_thinking_payload({}, {"reasoning_budget": 0}) is False
        assert is_thinking_payload({}, {}) is False
        assert is_thinking_payload({}, {"reasoning_budget": "not-int"}) is False
        # Payload arg is interface-parity only — must be ignored per FP N2
        assert is_thinking_payload(
            {"chat_template_kwargs": {"enable_thinking": True}}, {"reasoning_budget": 0}
        ) is False

    # (k) unit: _strip_thinking_wrapper rsplit handles malformed multi-tag
    def test_k_unit_strip_thinking_wrapper(self):
        from turbohaul.api.chat_completion import _strip_thinking_wrapper

        # Normal single-tag
        assert _strip_thinking_wrapper("<think>x</think>final") == "final"
        # No tag → unchanged
        assert _strip_thinking_wrapper("plain content") == "plain content"
        # Malformed multi-tag — surfaces the LAST post-think payload
        assert (
            _strip_thinking_wrapper("<think>a</think>partial<think>b</think>real")
            == "real"
        )
        # Leading whitespace stripped after wrapper
        assert _strip_thinking_wrapper("<think>x</think>\n  json") == "json"

    # (l) unit: _validate_json_schema lazy-import fail-soft simulation
    def test_l_unit_validate_json_schema_lib_unavailable(self, monkeypatch):
        """Simulate jsonschema missing → returns (False, 'jsonschema_lib_unavailable')."""
        import builtins
        from turbohaul.api import chat_completion as cc

        real_import = builtins.__import__

        def shim(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "jsonschema":
                raise ImportError("simulated absence")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", shim)
        ok, reason = cc._validate_json_schema(
            {"json_schema": {"schema": {"type": "object", "additionalProperties": False}}}
        )
        assert ok is False
        assert reason == "jsonschema_lib_unavailable"

    # (m) Streaming + json_schema invalid → still 422 at validator entry (carve-out
    # doesn't bypass validation; only retry is disabled downstream for streams).
    def test_m_streaming_json_schema_invalid_still_422(
        self, app_completion_autostart,
    ):
        app, client = app_completion_autostart
        _write_manifest_yaml(app.state.manager.boot.storage.manifests_path, "m")
        bad_schema = {"type": "object"}  # missing additionalProperties
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": bad_schema, "name": "T"},
                },
            },
        )
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "schema_validation_failed"
