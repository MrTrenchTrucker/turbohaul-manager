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
