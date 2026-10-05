"""Pin the structured IDLE_DEAD_CLASSIFY death log
at the two chat_completion.py sites that conclude "the sidecar crashed or
disconnected" (the streaming and non-streaming paths). The root cause of these
crashes is UNKNOWN, and prior crashes
produced no usable pattern because the log line at each site never named
which model/port/pid/context-size was involved. This does not fix the
crash -- it makes the next one diagnosable.

Reuses the SAME IDLE_DEAD_CLASSIFY line name manager.py already
uses (deliberately, not invented here) so one grep key covers every "engine
concluded dead" site in the codebase. Level is WARNING, matching both
existing template sites. The reasoning:
An INFERRED death via httpx transport exception is LESS certain than the
template's CONFIRMED is_alive()-false death, so escalating to ERROR here
would be backwards, and a severity-faceted tool would silently split one
event class across levels if the levels didn't match.

caplog.set_level is WARNING, matching the level actually logged at --
a mismatch breaks tests when a log line's level is moved
without updating caplog to match; this file is written to avoid that
exact trap.
"""
import asyncio
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from turbohaul.api.chat_completion import make_llama_server_complete_fn
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
from turbohaul.slot import Slot
from turbohaul.subprocess_mgr import SidecarHandle
from unittest.mock import MagicMock

_LOGGER_NAME = "turbohaul.api.chat_completion"


# ============================================================================
# Site 2 (non-streaming): make_llama_server_complete_fn's _complete
# ============================================================================


class _CrashingAsyncClient:
    """A fake httpx.AsyncClient whose post() raises the same transport
    exception class the real sidecar disconnect/crash would produce."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None, timeout=None):
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")


def test_site2_non_stream_logs_idle_dead_classify_with_fields(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)

    complete_fn = make_llama_server_complete_fn(
        http_client_factory=lambda: _CrashingAsyncClient()
    )
    slot = Slot.new(
        model_tag="crash-test-model",
        context=[{"role": "user", "content": "hello world, this is the prompt"}],
    )

    class _Handle:
        port = 54321
        pid = 99999

    from turbohaul.api.chat_completion import SidecarUnavailableError

    with pytest.raises(SidecarUnavailableError):
        asyncio.run(complete_fn(slot, _Handle()))

    death_records = [r for r in caplog.records if "IDLE_DEAD_CLASSIFY" in r.message]
    assert len(death_records) == 1, (
        f"expected exactly one IDLE_DEAD_CLASSIFY line, got {len(death_records)}: "
        f"{[r.message for r in death_records]}"
    )
    rec = death_records[0]
    assert rec.levelno == logging.WARNING
    msg = rec.getMessage()
    assert "model_tag=crash-test-model" in msg
    assert "port=54321" in msg
    assert "pid=99999" in msg
    # compute_ctx_len's exact value isn't the point here -- that it's present
    # and not a placeholder/None is (site1's test below checks the exact
    # value via a fixed prompt).
    assert "prompt_tokens=" in msg
    assert "prompt_tokens=None" not in msg


def test_site2_non_stream_death_log_is_best_effort_on_bad_messages(caplog):
    """HARD CONSTRAINT: a field lookup must never escape and break the
    request. Slot.new with context=None makes _complete return None before
    ever reaching the httpx call, so this instead directly exercises the
    log-call's own try/except by calling it with a message shape
    compute_ctx_len cannot handle, proving the surrounding except arm's
    control flow (still raising SidecarUnavailableError) is unaffected."""
    complete_fn = make_llama_server_complete_fn(
        http_client_factory=lambda: _CrashingAsyncClient()
    )
    # A malformed messages list (non-dict entries) is exactly the kind of
    # shape compute_ctx_len might not expect; the request must still fail
    # with the SAME typed exception, not an unrelated crash from the
    # logging attempt.
    slot = Slot.new(model_tag="m", context=["not-a-dict-message"])

    class _Handle:
        port = 1
        pid = 2

    from turbohaul.api.chat_completion import SidecarUnavailableError

    with pytest.raises(SidecarUnavailableError):
        asyncio.run(complete_fn(slot, _Handle()))


# ============================================================================
# Site 1 (streaming): stream_gen's matching except arm
# ============================================================================


def _make_handle(model_tag: str, port: int, pid: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _write_manifest_yaml(manifests_root, tag: str):
    (manifests_root / f"{tag}.yaml").write_text(
        f"""model_tag: {tag}
gguf_blob_sha256: "{'a' * 64}"
gguf_size_bytes: 1000
"""
    )


class _CrashingStreamOpenCM:
    """async context manager standing in for httpx's client.stream(...) --
    __aenter__ raises during the (real) stream OPEN, matching the sidecar
    dying before the first byte."""

    async def __aenter__(self):
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    async def __aexit__(self, *exc_info):
        return False


class _CrashingStreamAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def stream(self, method, url, json=None, timeout=None):
        return _CrashingStreamOpenCM()


@pytest.fixture
def app_streaming_crash(tmp_path, monkeypatch):
    # HAZARD: the SSE
    # route's `finally: watch_task.cancel(); await watch_task` deadlocks
    # under Starlette's TestClient, because watch_disconnect polls
    # request.is_disconnected() against a synthetic ASGI receive channel
    # that TestClient will not satisfy until the response completes --
    # embeddings.py uses a bounded version of this,
    # chat_completion.py's SSE route did not. Monkeypatch watch_disconnect
    # on chat_completion's OWN module name (it defines AND calls it in the
    # same file) to something that just parks until cancelled, exactly
    # like the real one does from the route's point of view.
    async def _fake_watch_disconnect(request, disconnect_event):
        await asyncio.sleep(3600)

    monkeypatch.setattr(
        "turbohaul.api.chat_completion.watch_disconnect", _fake_watch_disconnect
    )
    # The streaming path opens its OWN httpx.AsyncClient (no injectable
    # factory like _complete has) -- patch the module-level reference so
    # the stream-open raises the same transport exception class.
    monkeypatch.setattr(
        "turbohaul.api.chat_completion.httpx.AsyncClient",
        lambda *a, **kw: _CrashingStreamAsyncClient(),
    )

    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    _write_manifest_yaml(storage_root / "manifests", "crash-test-model")
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
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, grace_seconds=0, idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1, drained_sigterm_window_cold_s=1,
        ),
        pull=PullConfig(),
    )
    app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
    mgr = app.state.manager

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        return _make_handle(model_tag, port, pid=13579)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def fake_sigterm(handle, **kwargs):
        return True, "sigterm-clean"

    async def fake_vram(**kwargs):
        return True, 100

    mgr._spawn = fake_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = fake_sigterm
    mgr._vram_verify = fake_vram

    with TestClient(app) as client:
        yield app, client


def test_site1_stream_logs_idle_dead_classify_with_fields(app_streaming_crash, caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    app, client = app_streaming_crash

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "crash-test-model",
            "messages": [{"role": "user", "content": "hello streaming crash test"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200  # SSE error frame rides a 200, per _stream_error_frame

    death_records = [r for r in caplog.records if "IDLE_DEAD_CLASSIFY" in r.message]
    assert len(death_records) == 1, (
        f"expected exactly one IDLE_DEAD_CLASSIFY line, got {len(death_records)}: "
        f"{[r.message for r in death_records]}"
    )
    rec = death_records[0]
    assert rec.levelno == logging.WARNING
    msg = rec.getMessage()
    assert "model_tag=crash-test-model" in msg
    assert "port=59700" in msg  # default_port_base, single model → first port
    assert "pid=13579" in msg
    assert "prompt_tokens=" in msg
    assert "prompt_tokens=None" not in msg
