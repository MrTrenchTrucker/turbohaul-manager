"""Tests for POST /v1/embeddings.

6 cases (~1:1 ratio src:test):
1. happy_single        — single string, manifest:embeddings=true, OpenAI shape
2. happy_batch_2       — list[str] of 2 items, both embeddings + index
3. capability_refuse   — manifest embeddings=false → 400 plain-string
4. 413_oversize        — list >64 items → 413 batch-cap
5. 400_base64          — encoding_format='base64' → 400
6. 400_dimensions      — dimensions param present → 400
"""
import asyncio
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

import turbohaul.api.embeddings as embeddings_mod
from turbohaul.api.main import create_app
from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig,
    RuntimePathsConfig, ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.subprocess_mgr import SidecarHandle
from turbohaul.slot import VramOverCommitError  # used by the capacity-failure test


def _make_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 12345
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _write_manifest(manifests_root, tag: str, embeddings_enabled: bool) -> None:
    """Write a minimal valid manifest YAML for the test."""
    (manifests_root / f"{tag}.yaml").write_text(
        f"""model_tag: {tag}
gguf_blob_sha256: "{'a' * 64}"
gguf_size_bytes: 1000
llama_server_flags:
  embeddings: {str(embeddings_enabled).lower()}
"""
    )


@pytest.fixture
def app_test(tmp_path, monkeypatch):
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
            llama_server_binary=tmp_path / "fake", default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    mgr = app.state.manager

    # Fake submit_for_streaming: pre-arm a slot with stream_ready_event set.
    async def fake_submit_for_streaming(model_tag, prompt="", thread_id="", client_meta=None, **kwargs):
        slot = MagicMock()
        slot.stream_ready_event = asyncio.Event()
        slot.stream_ready_event.set()  # pre-fired so route doesn't wait
        slot.stream_ready_failed_reason = None  # success path, not a failure wakeup
        slot.stream_done_event = asyncio.Event()
        slot.stream_handle = _make_handle(model_tag, 11500)
        slot.slot_id = "test-slot"
        slot.thread_id = thread_id
        slot.model_tag = model_tag
        slot.client_meta = client_meta or {}
        return slot

    mgr.submit_for_streaming = fake_submit_for_streaming

    # Mock httpx upstream to return canned OpenAI-compat embeddings response.
    def _mock_handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        import json as _json
        payload = _json.loads(body)
        inp = payload.get("input")
        items = [inp] if isinstance(inp, str) else list(inp)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": i}
                    for i, _ in enumerate(items)
                ],
                "model": payload.get("model"),
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler))
    monkeypatch.setattr(embeddings_mod, "_HTTPX_CLIENT", mock_client)

    with TestClient(app) as client:
        yield app, client, storage_root / "manifests"

    asyncio.run(mock_client.aclose())


# Case 1
def test_happy_single(app_test):
    app, client, manifests_root = app_test
    _write_manifest(manifests_root, "qwen-emb", embeddings_enabled=True)
    r = client.post("/v1/embeddings", json={"model": "qwen-emb", "input": "hello"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert body["model"] == "qwen-emb"


# Case 2
def test_happy_batch_2(app_test):
    app, client, manifests_root = app_test
    _write_manifest(manifests_root, "qwen-emb", embeddings_enabled=True)
    r = client.post(
        "/v1/embeddings", json={"model": "qwen-emb", "input": ["a", "b"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 2
    assert [d["index"] for d in body["data"]] == [0, 1]


# Case 3
def test_capability_refuse(app_test):
    app, client, manifests_root = app_test
    _write_manifest(manifests_root, "chat-only", embeddings_enabled=False)
    r = client.post("/v1/embeddings", json={"model": "chat-only", "input": "x"})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert isinstance(detail, str)
    assert "does not expose embeddings" in detail
    assert "llama_server_flags.embeddings" in detail


# Case 4
def test_413_oversize_batch(app_test):
    app, client, manifests_root = app_test
    _write_manifest(manifests_root, "qwen-emb", embeddings_enabled=True)
    r = client.post(
        "/v1/embeddings",
        json={"model": "qwen-emb", "input": ["x"] * 65},
    )
    assert r.status_code == 413
    detail = r.json()["detail"]
    assert "exceeds" in detail and "64" in detail


# Case 5
def test_400_base64(app_test):
    app, client, manifests_root = app_test
    _write_manifest(manifests_root, "qwen-emb", embeddings_enabled=True)
    r = client.post(
        "/v1/embeddings",
        json={"model": "qwen-emb", "input": "x", "encoding_format": "base64"},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert isinstance(detail, str)
    assert "base64" in detail
    assert "use 'float'" in detail


# Case 6
def test_400_dimensions(app_test):
    app, client, manifests_root = app_test
    _write_manifest(manifests_root, "qwen-emb", embeddings_enabled=True)
    r = client.post(
        "/v1/embeddings",
        json={"model": "qwen-emb", "input": "x", "dimensions": 512},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert isinstance(detail, str)
    assert "dimensions param not supported" in detail


# Capacity failure must not hang the route
def test_capacity_failure_503_not_2h_hang(app_test, monkeypatch):
    """A routing failure (VRAM over-commit) fails the completion_future WITHOUT ever
    setting stream_ready_event. A route waiting on it alone would block _SLOT_READY_TIMEOUT_S (2h)
    then 504; this route races the two and returns a retryable 503 immediately."""
    app, client, manifests_root = app_test
    _write_manifest(manifests_root, "qwen-emb", embeddings_enabled=True)
    mgr = app.state.manager

    # The ready-wait resolves on the VERY FIRST
    # asyncio.wait() tick (no extra cancel+await cycle), which starves the
    # disconnect-watcher task of its first scheduling slot before this route's
    # `finally` cancels+awaits it. watch_disconnect is then cancelled having
    # never been scheduled, and the `await watch_task` in `finally` never
    # returns. Stubbing it here is what lets THIS test assert a status code at
    # all.
    #
    # ⛔ DO NOT READ THIS STUB AS "TEST-ONLY PLUMBING". The deadlock it hides is
    # NOT a TestClient artifact. The same wedge was reproduced under a REAL
    # uvicorn server on a REAL socket with the REAL watch_disconnect (no stubs,
    # no TestClient):
    #   * old shape (event never set) ............... 503 in 0.005s  [control]
    #   * current, failure fires 0.25s AFTER entry ..... 503 in 0.255s [this fix]
    #   * current, failure ALREADY fired at entry ...... NO RESPONSE, >55s,
    #                                                    never returns
    # The third row is a live defect, concealed by this stub -- the wedge is in
    # the `finally` block, not in the capacity discrimination below, so it
    # survives the capacity fix and this stub conceals it. It is covered by the
    # already-failed-at-entry test below. Reachability of that third timing in production (whether the
    # queue worker can fail a slot inside submit_for_streaming's own await
    # window) is unproven.
    async def _stub_watch_disconnect(request, disconnect_event):
        await disconnect_event.wait()

    monkeypatch.setattr(embeddings_mod, "watch_disconnect", _stub_watch_disconnect)

    async def fake_submit_capacity_fail(model_tag, prompt="", thread_id="",
                                        client_meta=None, **kwargs):
        slot = MagicMock()
        slot.stream_ready_event = asyncio.Event()
        exc = VramOverCommitError("no VRAM capacity", retry_after_s=7)
        fut = asyncio.get_running_loop().create_future()
        fut.set_exception(exc)
        slot.completion_future = fut
        # PRECONDITION, constructed directly -- NOT coverage of the
        # manager. This fake replaces mgr.submit_for_streaming wholesale, so
        # manager.py::_fail_completion_future never runs in this test. The two
        # lines below are this fixture standing in for what the manager would
        # have done on a capacity failure (it does not
        # hangs the wait, it wakes it with a reason), so that what actually gets
        # tested is the ROUTE's behaviour: 503 capacity_unavailable with
        # retry_after. Same framing as the row-3 comment further down this file.
        # ⛔ Do not read this as gating the manager. It cannot -- it asserts what
        # it just performed. The manager side is gated by
        # the fail-completion-future wakeup test; without
        # it, the manager's wakeup lines could be deleted outright with
        # this whole suite still green.
        slot.stream_ready_failed_reason = str(exc)
        slot.stream_ready_event.set()
        slot.stream_handle = None
        slot.slot_id = "cap-fail"
        slot.thread_id = thread_id
        slot.model_tag = model_tag
        slot.client_meta = client_meta or {}
        return slot

    mgr.submit_for_streaming = fake_submit_capacity_fail
    r = client.post("/v1/embeddings", json={"model": "qwen-emb", "input": "hi"})
    assert r.status_code == 503, r.text
    err = r.json()["detail"]["error"]
    assert err["type"] == "capacity_unavailable"
    assert err["retry_after"] == 7
    assert r.headers.get("retry-after") == "7"


# Already-failed-at-entry must resolve within a bound
@pytest.mark.timeout(8, method="thread")
def test_row3_already_fired_at_entry_resolves_bounded_not_a_55s_hang(app_test):
    """The wedge the capacity-failure 503 test's own stub conceals
    (see that test's comment): stream_ready_event ALREADY set + a failure
    reason already stamped by the time the route reaches its ready/
    completion_future race -- the real-uvicorn row 3, >55s, never
    returns. Reachable in production via manager.py's own narrow
    _reserve_and_start_locked "LAST-RESORT refuse" race combined with
    submit()'s real async work (audit db write) between enqueue and return.

    Uses the REAL watch_disconnect (no stub) -- the actual mechanism this change
    fixes -- and drives it through TestClient's real, unstubbed request
    lifecycle. @pytest.mark.timeout(8, method="thread") is the enforcement:
    on UNAMENDED code this test hangs past 8s and pytest-timeout kills it
    with a genuine TIMEOUT failure, not a slow pass -- a test that merely
    happened to return quickly would prove nothing. On the fix, the request
    resolves within
    _WATCH_TASK_TEARDOWN_TIMEOUT_S (1.0s) of teardown, nowhere near either
    bound.
    """
    app, client, manifests_root = app_test
    _write_manifest(manifests_root, "qwen-emb", embeddings_enabled=True)
    mgr = app.state.manager

    async def fake_submit_already_failed_at_entry(
        model_tag, prompt="", thread_id="", client_meta=None, **kwargs
    ):
        # Mirrors _fail_completion_future having ALREADY run to completion
        # (row 3) -- stream_ready_event is set and stream_ready_failed_reason
        # is stamped BEFORE this function even returns to the route, unlike
        # the row-2 test above where the route's own race is what discovers
        # the failure. This is the row-3 precondition itself, constructed
        # directly rather than raced against real dispatcher timing.
        slot = MagicMock()
        slot.stream_ready_event = asyncio.Event()
        exc = VramOverCommitError("no VRAM capacity", retry_after_s=7)
        fut = asyncio.get_running_loop().create_future()
        fut.set_exception(exc)
        slot.completion_future = fut
        slot.stream_ready_failed_reason = str(exc)
        slot.stream_ready_event.set()
        slot.stream_handle = None
        slot.slot_id = "row3-already-failed"
        slot.thread_id = thread_id
        slot.model_tag = model_tag
        slot.client_meta = client_meta or {}
        return slot

    mgr.submit_for_streaming = fake_submit_already_failed_at_entry
    r = client.post("/v1/embeddings", json={"model": "qwen-emb", "input": "hi"})
    assert r.status_code == 503, r.text
    err = r.json()["detail"]["error"]
    assert err["type"] == "capacity_unavailable"
    assert err["retry_after"] == 7
