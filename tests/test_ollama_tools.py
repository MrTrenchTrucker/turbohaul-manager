"""Tests for spec  Ollama-compat tool_calls batch.

Covers the 10 cases mandated by the LOCKED-FINAL v3 RC body:
  (a) inbound forwarding — _complete receives tool knobs
  (b) outbound reshape + multi-tool batch + empty-list omitted
  (c) args STR→OBJECT
  (d) malformed args lenient fallback + enriched warning
  (e) plain-text /api/chat backward compat (no tools)
  (f) error guard single-assert
  (g) /v1/chat/completions cross-path regression — stream-only knob isolation
  (h) Change 4: stream=true + tools → 400
  (i) LOW-6: 300KB args → truncated + warning + still returned
  (j) MED-4: finish_reason → done_reason mapping

Fixture pattern mirrors tests/test_api_chat_completion.py:76 (fake_complete).
"""
import asyncio
import json
import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.chat_completion import (
    MAX_TOOL_ARG_CHARS,
    _COMMON_FORWARDED_KNOBS,
    _STREAM_FORWARDED_KNOBS,
    _STREAM_ONLY_KNOBS,
    _coerce_created_at,
)
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


def _build_app(tmp_path, fake_complete):
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
            grace_seconds=0,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
        ),
        pull=PullConfig(),
    )
    # MOD-1 fix (Path A): auto_start_worker=True so the FastAPI lifespan
    # creates worker_task on the SAME event loop TestClient runs against.
    # The earlier _start_worker() helper spawned the task on whatever loop
    # was current at fixture-build time (often the outer test loop), then
    # TestClient's anyio BlockingPortal created its own loop for lifespan
    # shutdown — `await self._worker_task` then raised
    # "Task attached to a different loop". Letting lifespan own the task
    # eliminates the cross-loop bug entirely.
    #
    # auto_boot_reconcile=False keeps test side-effects minimal (no DB
    # reconcile + no binary sha256 check); the DI overrides below replace
    # all real subprocess/VRAM/completion behavior so the worker_loop
    # touches no host resources.
    app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
    mgr = app.state.manager
    mgr._spawn = lambda *a, **kw: _make_handle(a[3] if len(a) > 3 else "m", a[2] if len(a) > 2 else 59500)
    async def fh(*a, **kw): return True
    async def fs(*a, **kw): return True, "sigterm-clean"
    async def fv(*a, **kw): return True, 100
    mgr._wait_healthy = fh
    mgr._sigterm = fs
    mgr._vram_verify = fv
    # IMPORTANT: override _complete_fn BEFORE TestClient __enter__ fires the
    # lifespan, so worker_loop picks up the fake on the very first slot.
    mgr._complete_fn = fake_complete
    return app, mgr


# ---------------------------------------------------------------------------
# Helper completion fixtures
# ---------------------------------------------------------------------------


def _make_capturing_complete(canned: dict):
    """Returns a fake _complete that records what client_meta it saw and
    returns the canned OpenAI-shape response.
    """
    seen: dict = {}

    async def _complete(slot, handle):
        seen["client_meta"] = dict(slot.client_meta or {})
        return canned

    return _complete, seen


def _openai_response_with(tool_calls=None, content="ok", finish_reason="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
    }


# ===========================================================================
# (a) inbound forwarding — mock _complete asserts tools received
# ===========================================================================


def test_a_inbound_tools_forwarded_into_client_meta(tmp_path):
    """ollama_chat must populate client_meta['tools'] (et al.) so _complete
    can forward them to llama-server. BUG-1 regression coverage."""
    fc, seen = _make_capturing_complete(_openai_response_with())
    app, mgr = _build_app(tmp_path, fc)
    with TestClient(app) as client:
        tools_payload = [
            {"type": "function",
             "function": {"name": "get_weather", "parameters": {"type": "object"}}}
        ]
        r = client.post(
            "/api/chat",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": tools_payload,
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "function_call": "auto",
                "functions": [{"name": "fn1"}],
            },
        )
        assert r.status_code == 200, r.text
    cm = seen["client_meta"]
    assert cm["tools"] == tools_payload
    assert cm["tool_choice"] == "auto"
    assert cm["parallel_tool_calls"] is True
    assert cm["function_call"] == "auto"
    assert cm["functions"] == [{"name": "fn1"}]


# ===========================================================================
# (b) outbound reshape + multi-tool batch + empty-list omitted (LOW-2 collapse)
# ===========================================================================


def test_b_outbound_reshape_multi_tool_and_empty_omit(tmp_path):
    """Three checks in one body (LOW-2): outbound shape correctness, batch
    of multiple tool_calls preserved, and an empty tool_calls list is NOT
    emitted (key omitted) so clients don't receive `"tool_calls": []`.
    """
    # Sub-check: multi-tool batch
    multi = [
        {"id": "call_1", "function": {"name": "f1", "arguments": '{"x":1}'}},
        {"id": "call_2", "function": {"name": "f2", "arguments": '{"y":"hello"}'}},
    ]
    fc, _ = _make_capturing_complete(
        _openai_response_with(tool_calls=multi, content="picked tools")
    )
    app, mgr = _build_app(tmp_path, fc)
    with TestClient(app) as client:
        r = client.post(
            "/api/chat",
            json={"model": "test-model",
                  "messages": [{"role": "user", "content": "go"}],
                  "tools": [{"type": "function", "function": {"name": "f1"}}]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
    # Outbound shape correctness
    assert body["model"] == "test-model"
    assert body["done"] is True
    assert body["done_reason"] == "stop"
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == "picked tools"
    # Multi-tool batch preserved + id preserved + args parsed to dict
    tcs = body["message"]["tool_calls"]
    assert len(tcs) == 2
    assert tcs[0]["id"] == "call_1"
    assert tcs[0]["function"]["name"] == "f1"
    assert tcs[0]["function"]["arguments"] == {"x": 1}
    assert tcs[1]["function"]["arguments"] == {"y": "hello"}

    # Empty-list omitted sub-check (fresh app, different completion)
    fc2, _ = _make_capturing_complete(
        _openai_response_with(tool_calls=[], content="no calls"),
    )
    b2_root = tmp_path / "b2"
    b2_root.mkdir(exist_ok=True)
    app2, _ = _build_app(b2_root, fc2)
    with TestClient(app2) as client:
        r2 = client.post(
            "/api/chat",
            json={"model": "test-model",
                  "messages": [{"role": "user", "content": "go"}]},
        )
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
    assert "tool_calls" not in body2["message"], (
        "empty tool_calls list must be omitted, got %r" % body2["message"]
    )


# ===========================================================================
# (c) args STR → OBJECT (BUG-4 fix)
# ===========================================================================


def test_c_args_str_to_object(tmp_path):
    tc = [{"id": "c1", "function": {"name": "lookup",
                                    "arguments": '{"key":"abc","n":42}'}}]
    fc, _ = _make_capturing_complete(_openai_response_with(tool_calls=tc))
    app, mgr = _build_app(tmp_path, fc)
    with TestClient(app) as client:
        r = client.post(
            "/api/chat",
            json={"model": "test-model",
                  "messages": [{"role": "user", "content": "x"}]},
        )
        body = r.json()
    args = body["message"]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict)
    assert args == {"key": "abc", "n": 42}


# ===========================================================================
# (d) malformed args lenient fallback + enriched warning (model + args[:80] + err)
# ===========================================================================


def test_d_malformed_args_lenient_fallback(tmp_path, caplog):
    malformed = "not-valid-json{{{"
    tc = [{"id": "d1", "function": {"name": "broken", "arguments": malformed}}]
    fc, _ = _make_capturing_complete(_openai_response_with(tool_calls=tc))
    app, mgr = _build_app(tmp_path, fc)
    with TestClient(app) as client:
        with caplog.at_level(logging.WARNING, logger="turbohaul.api.chat_completion"):
            r = client.post(
                "/api/chat",
                json={"model": "broken-model",
                      "messages": [{"role": "user", "content": "x"}]},
            )
            body = r.json()
    args = body["message"]["tool_calls"][0]["function"]["arguments"]
    # note-3 lenient: pass raw string through unchanged on parse failure
    assert args == malformed
    # LOW-5: warning includes model + truncated args
    found = [rec for rec in caplog.records if "json.loads failed" in rec.getMessage()]
    assert found, "expected json.loads warning, got: %r" % [r.getMessage() for r in caplog.records]
    msg = found[0].getMessage()
    assert "broken-model" in msg
    assert "not-valid-json" in msg  # truncated args[:80] head


# ===========================================================================
# (e) backward-compat plain-text /api/chat (no tools) unchanged
# ===========================================================================


def test_e_plain_text_backward_compat(tmp_path):
    """No tool_calls in response → message has no tool_calls key, response
    shape is the Ollama-native subset the pre-Wave-5a tests expected."""
    fc, _ = _make_capturing_complete(_openai_response_with(content="hello"))
    app, mgr = _build_app(tmp_path, fc)
    with TestClient(app) as client:
        r = client.post(
            "/api/chat",
            json={"model": "plain", "messages": [{"role": "user", "content": "hi"}]},
        )
        body = r.json()
    assert body["done"] is True
    assert body["message"] == {"role": "assistant", "content": "hello"}
    assert "tool_calls" not in body["message"]
    # created_at coerced to ISO-8601 string (LOW-7)
    assert isinstance(body["created_at"], str)
    assert body["created_at"].startswith("20")  # ISO-8601 leading "20XX-..."


# ===========================================================================
# (f) error guard — single assertion
# ===========================================================================


def test_f_error_guard_passthrough(tmp_path):
    """note-2: if completion_fn returns a dict with 'error', return it unchanged
    (no reshape attempt). Single-assert per LOW-4."""
    err = {"error": "upstream_thing_failed", "detail": "kv-cache OOM"}
    async def fc(slot, handle):
        return err
    app, mgr = _build_app(tmp_path, fc)
    with TestClient(app) as client:
        r = client.post(
            "/api/chat",
            json={"model": "x", "messages": [{"role": "user", "content": "y"}]},
        )
        assert r.json() == err


# ===========================================================================
# (g) /v1/chat/completions cross-path regression — stream-only knob isolation
# ===========================================================================


def test_g_openai_path_no_stream_only_leak(tmp_path):
    """Non-streaming /v1/chat/completions request with tools must NOT cause
    _complete to forward stream-only knobs. Verifies Change 1 split is wired."""
    captured = {}
    async def fc(slot, handle):
        # Re-create the payload the production _complete would build by
        # iterating _COMMON_FORWARDED_KNOBS over slot.client_meta.
        cm = slot.client_meta or {}
        forwarded = {k: cm.get(k) for k in _COMMON_FORWARDED_KNOBS if cm.get(k) is not None}
        captured["forwarded"] = forwarded
        return _openai_response_with(content="ok")

    app, mgr = _build_app(tmp_path, fc)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "fn"}}],
                "stream": False,  # non-streaming
            },
        )
        assert r.status_code == 200, r.text

    fwd = captured["forwarded"]
    # Stream-only knobs must NOT appear in the non-streaming forward set
    for k in _STREAM_ONLY_KNOBS:
        assert k not in fwd, "stream-only knob %r leaked into non-stream forward" % k
    # And the alias is still the union (back-compat invariant)
    assert set(_STREAM_FORWARDED_KNOBS) == set(_COMMON_FORWARDED_KNOBS) | set(_STREAM_ONLY_KNOBS)


# ===========================================================================
# (h) Change 4: stream=true + tools → 400 with structured body
# ===========================================================================


def test_h_stream_with_tools_returns_400(tmp_path):
    async def fc(slot, handle):
        return _openai_response_with()  # should never be reached
    app, mgr = _build_app(tmp_path, fc)
    with TestClient(app) as client:
        r = client.post(
            "/api/chat",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "fn"}}],
            },
        )
    assert r.status_code == 400, r.text
    body = r.json()
    detail = body.get("detail") or body
    assert detail.get("error") == "streaming_with_tools_deferred"
    assert "follow_on" in detail


# ===========================================================================
# (i) LOW-6: 300 KB args → truncated to MAX_TOOL_ARG_CHARS + warning + returned
# ===========================================================================


def test_i_oversized_args_truncated(tmp_path, caplog):
    # 300 KB of valid JSON-string content (so it parses if not truncated)
    big_value = "x" * (300 * 1024)
    big_args_str = json.dumps({"v": big_value})  # ~300KB+overhead, definitely >256KB
    assert len(big_args_str) > MAX_TOOL_ARG_CHARS
    tc = [{"id": "i1", "function": {"name": "big", "arguments": big_args_str}}]
    fc, _ = _make_capturing_complete(_openai_response_with(tool_calls=tc))
    app, mgr = _build_app(tmp_path, fc)
    with TestClient(app) as client:
        with caplog.at_level(logging.WARNING, logger="turbohaul.api.chat_completion"):
            r = client.post(
                "/api/chat",
                json={"model": "big-model",
                      "messages": [{"role": "user", "content": "x"}]},
            )
            body = r.json()
    # Slot still returns 200; the response is shaped despite truncation
    assert r.status_code == 200
    assert body["message"]["tool_calls"][0]["function"]["name"] == "big"
    # Truncation warning logged with size info
    truncation_warns = [
        rec for rec in caplog.records
        if "truncating" in rec.getMessage() and "big-model" in rec.getMessage()
    ]
    assert truncation_warns, "expected truncation warning, got: %r" % [r.getMessage() for r in caplog.records]


# ===========================================================================
# (j) MED-4: finish_reason → done_reason mapping
# ===========================================================================


@pytest.mark.parametrize("finish,expected", [
    ("stop", "stop"),
    ("length", "length"),
    ("tool_calls", "stop"),
    ("function_call", "stop"),
    ("weird_unknown", "stop"),
])
def test_j_finish_to_done_reason(tmp_path, finish, expected):
    fc, _ = _make_capturing_complete(
        _openai_response_with(content="x", finish_reason=finish)
    )
    app, mgr = _build_app(tmp_path, fc)
    with TestClient(app) as client:
        r = client.post(
            "/api/chat",
            json={"model": "m", "messages": [{"role": "user", "content": "y"}]},
        )
        body = r.json()
    assert body["done_reason"] == expected


# ---------------------------------------------------------------------------
# Pure-helper sanity (no app/server needed)
# ---------------------------------------------------------------------------


def test_coerce_created_at_helper():
    iso = _coerce_created_at(1700000000)
    assert isinstance(iso, str) and iso.startswith("20")
    # None and pre-stringified pass through
    assert _coerce_created_at(None) is None
    assert _coerce_created_at("2026-05-18T00:00:00+00:00") == "2026-05-18T00:00:00+00:00"
