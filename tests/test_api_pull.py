"""Tests for pull endpoints ( - v0.2 §9.1)."""
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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


@pytest.fixture
def app_test(tmp_path):
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
        queue=QueueConfig(),
        pull=PullConfig(per_stream_max_bytes=1024 * 1024 * 100),  # 100MB cap for tests
    )
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


def _make_mock_httpx_factory(chunks_to_yield: list[bytes], status_code: int = 200):
    """Return a factory that produces a mock AsyncClient whose .stream() yields chunks."""

    class _MockResponse:
        def __init__(self, chunks, status):
            self._chunks = chunks
            self.status_code = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{self.status_code}", request=MagicMock(), response=MagicMock(status_code=self.status_code)
                )

        async def aiter_bytes(self, chunk_size=64 * 1024):
            for c in self._chunks:
                yield c

    class _MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        def stream(self, method, url, headers=None):
            return _MockResponse(chunks_to_yield, status_code)

    def factory():
        return _MockClient()

    return factory


class TestPullUrl:
    def test_pull_url_400_missing_url(self, app_test):
        app, client = app_test
        r = client.post("/api/pull-url", json={})
        assert r.status_code == 400

    def test_pull_url_400_http_scheme(self, app_test):
        app, client = app_test
        r = client.post("/api/pull-url", json={"url": "http://example.com/x"})
        assert r.status_code == 400
        assert "scheme" in r.text

    def test_pull_url_400_private_ip(self, app_test):
        app, client = app_test
        r = client.post("/api/pull-url", json={"url": "https://10.0.0.1/x"})
        assert r.status_code == 400
        assert "denied" in r.text.lower()

    def test_pull_url_400_imds(self, app_test):
        app, client = app_test
        r = client.post("/api/pull-url", json={"url": "https://169.254.169.254/x"})
        assert r.status_code == 400

    def test_pull_url_happy_path(self, app_test):
        app, client = app_test
        data = b"some-model-data" * 1000
        expected_sha = hashlib.sha256(data).hexdigest()
        # Mock DNS to resolve to public IP
        with patch("turbohaul.ssrf_guard.socket.getaddrinfo") as ga:
            ga.return_value = [(2, 1, 0, "", ("1.1.1.1", 0))]
            # Inject mock httpx factory via app.state
            app.state.http_client_factory = _make_mock_httpx_factory([data])
            r = client.post(
                "/api/pull-url",
                json={"url": "https://example.com/model.gguf"},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["sha256"] == expected_sha
            assert body["bytes_written"] == len(data)
            assert body["status"] == "complete"

    def test_pull_url_hash_mismatch(self, app_test):
        app, client = app_test
        data = b"actual data"
        with patch("turbohaul.ssrf_guard.socket.getaddrinfo") as ga:
            ga.return_value = [(2, 1, 0, "", ("1.1.1.1", 0))]
            app.state.http_client_factory = _make_mock_httpx_factory([data])
            r = client.post(
                "/api/pull-url",
                json={
                    "url": "https://example.com/model.gguf",
                    "expected_sha256": "f" * 64,
                },
            )
            assert r.status_code == 400
            assert "computed" in r.text

    def test_pull_url_size_exceeded(self, app_test, tmp_path):
        """Cap is 100MB; stream 200MB → 413."""
        app, client = app_test
        # Lower the cap dynamically via PUT /api/config
        client.put("/api/config", json={"pull": {"per_stream_max_bytes": 1024}})
        data = b"x" * 4096
        with patch("turbohaul.ssrf_guard.socket.getaddrinfo") as ga:
            ga.return_value = [(2, 1, 0, "", ("1.1.1.1", 0))]
            app.state.http_client_factory = _make_mock_httpx_factory([data])
            r = client.post(
                "/api/pull-url", json={"url": "https://example.com/big.gguf"}
            )
            assert r.status_code == 413


class TestPullHf:
    def test_pull_hf_400_missing_fields(self, app_test):
        app, client = app_test
        r = client.post("/api/pull-hf", json={"repo_id": "x"})
        assert r.status_code == 400
        r2 = client.post("/api/pull-hf", json={"filename": "y"})
        assert r2.status_code == 400

    def test_pull_hf_happy_path(self, app_test):
        app, client = app_test
        data = b"weights" * 500
        expected_sha = hashlib.sha256(data).hexdigest()
        with patch("turbohaul.ssrf_guard.socket.getaddrinfo") as ga:
            ga.return_value = [(2, 1, 0, "", ("3.5.20.1", 0))]  # HF CDN public range
            app.state.http_client_factory = _make_mock_httpx_factory([data])
            r = client.post(
                "/api/pull-hf",
                json={"repo_id": "Qwen/Qwen2.5-0.5B", "filename": "model.gguf"},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["sha256"] == expected_sha
            assert body["bytes_written"] == len(data)
            assert "huggingface.co" in body["host"]


class TestPullOllamaRegistry:
    def test_pull_returns_501(self, app_test):
        app, client = app_test
        r = client.post("/api/pull", json={"name": "llama2:7b"})
        assert r.status_code == 501


class TestProgressEvents:
    def test_pull_url_emits_progress_events(self, app_test):
        app, client = app_test
        mgr = app.state.manager
        # Subscribe a queue to event_bus
        import asyncio
        q = asyncio.Queue()
        mgr.event_bus.subscribe(q)
        data = b"data"
        with patch("turbohaul.ssrf_guard.socket.getaddrinfo") as ga:
            ga.return_value = [(2, 1, 0, "", ("1.1.1.1", 0))]
            app.state.http_client_factory = _make_mock_httpx_factory([data])
            r = client.post(
                "/api/pull-url",
                json={"url": "https://example.com/x.gguf"},
            )
        assert r.status_code == 200
        # Drain events
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        event_types = [e.get("event") for e in events]
        assert "pull_url_started" in event_types
        assert "pull_url_complete" in event_types
