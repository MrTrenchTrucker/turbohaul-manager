"""Tests for POST /v1/embeddings (spec LOCKED-FINAL).

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
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    mgr = app.state.manager

    # Fake submit_for_streaming: pre-arm a slot with stream_ready_event set.
    async def fake_submit_for_streaming(model_tag, prompt="", thread_id="", client_meta=None, **kwargs):
        slot = MagicMock()
        slot.stream_ready_event = asyncio.Event()
        slot.stream_ready_event.set()  # pre-fired so route doesn't wait
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
