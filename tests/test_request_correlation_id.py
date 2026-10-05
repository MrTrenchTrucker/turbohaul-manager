"""Observability hardening, GAP 4 — the two layers that each compute their
own view of a request's identity (chat_completion.py's identity_recompose,
manager.py's R2B_REQ_IDENTITY) now carry a shared request correlation id, so
they can be joined by grep under load instead of being two disconnected
lines a human has to guess belong to the same request. identity_recompose
also now states WHICH source it read role/session from (payload vs
idle-hot-recovery vs absent) instead of only the yes/no outcome.

Design: a per-request uuid4, generated
in chat_completion.py right where identity_recompose already runs, carried
through the EXISTING client_meta dict (already flows unmodified from the API
layer into slot.client_meta and hence into _emit_request_identity) -- no
request-path SIGNATURE changes anywhere, the least invasive option that
survives concurrency (uuid4 has no shared mutable state).

Run:
    PYTHONPATH=<repo>/src python3 -m pytest tests/test_request_correlation_id.py -v
"""
from __future__ import annotations

import asyncio
import logging
import re
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


def _write_manifest_yaml(manifests_root, tag: str):
    (manifests_root / f"{tag}.yaml").write_text(
        f"""model_tag: {tag}
gguf_blob_sha256: "{'a' * 64}"
gguf_size_bytes: 1000
"""
    )


@pytest.fixture
def app_completion_autostart(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
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


_REQ_ID_RE = re.compile(r'req_id[=":\s]+"?([A-Za-z0-9_-]{4,})')


def _extract_req_ids(caplog, substring):
    ids = set()
    for r in caplog.records:
        if substring in r.message:
            m = _REQ_ID_RE.search(r.message)
            if m:
                ids.add(m.group(1))
    return ids


class TestRequestCorrelationId:
    def test_discriminator_req_id_joins_identity_recompose_to_r2b_req_identity(
        self, app_completion_autostart, caplog
    ):
        """End-to-end through the real route + manager: the SAME req_id must
        appear on both identity_recompose and R2B_REQ_IDENTITY for one
        request -- ids present on both sides but never matching would still
        be a FAIL (the verify script checks the join, not just presence)."""
        app, client = app_completion_autostart
        with caplog.at_level(logging.INFO):
            r = client.post(
                "/v1/chat/completions",
                json={"model": "test-model",
                      "messages": [{"role": "user", "content": "say hi"}]},
            )
        assert r.status_code == 200, r.text

        recompose_ids = _extract_req_ids(caplog, "identity_recompose")
        r2b_ids = _extract_req_ids(caplog, "R2B_REQ_IDENTITY")
        assert recompose_ids, "no req_id found on any identity_recompose line"
        assert r2b_ids, "no req_id found on any R2B_REQ_IDENTITY line"
        joined = recompose_ids & r2b_ids
        assert joined, (
            f"identity_recompose ids {recompose_ids} and R2B_REQ_IDENTITY ids "
            f"{r2b_ids} share no value -- the two layers cannot be joined"
        )

    def test_discriminator_identity_recompose_states_role_and_session_source(
        self, app_completion_autostart, caplog
    ):
        """The request carries neither role nor session_id -- the line must
        say WHY (source=absent for both), not just session_present=False."""
        app, client = app_completion_autostart
        with caplog.at_level(logging.INFO):
            r = client.post(
                "/v1/chat/completions",
                json={"model": "test-model",
                      "messages": [{"role": "user", "content": "say hi"}]},
            )
        assert r.status_code == 200, r.text
        lines = [rec.message for rec in caplog.records
                 if rec.message.startswith("identity_recompose")]
        assert len(lines) == 1, lines
        line = lines[0]
        assert "role_source=absent" in line, line
        assert "session_source=absent" in line, line

    def test_discriminator_identity_recompose_states_payload_source(
        self, app_completion_autostart, caplog
    ):
        """role/session_id ARE present on the payload -- source must say so."""
        app, client = app_completion_autostart
        with caplog.at_level(logging.INFO):
            r = client.post(
                "/v1/chat/completions",
                json={"model": "test-model",
                      "messages": [{"role": "user", "content": "say hi"}],
                      "role": "main", "session_id": "sess-abc123"},
            )
        assert r.status_code == 200, r.text
        lines = [rec.message for rec in caplog.records
                 if rec.message.startswith("identity_recompose")]
        assert len(lines) == 1, lines
        line = lines[0]
        assert "role_source=payload" in line, line
        assert "session_source=payload" in line, line

    def test_discriminator_two_requests_get_two_different_req_ids(
        self, app_completion_autostart, caplog
    ):
        """Non-vacuous: the id genuinely correlates a SINGLE request, not a
        constant stamped on every line regardless of which request it is."""
        app, client = app_completion_autostart
        with caplog.at_level(logging.INFO):
            client.post("/v1/chat/completions",
                        json={"model": "test-model",
                              "messages": [{"role": "user", "content": "first"}]})
            client.post("/v1/chat/completions",
                        json={"model": "test-model",
                              "messages": [{"role": "user", "content": "second"}]})
        ids = _extract_req_ids(caplog, "identity_recompose")
        assert len(ids) == 2, f"expected 2 distinct req_ids across 2 requests, got {ids}"

    def test_discriminator_streaming_route_uses_the_same_req_id_not_a_new_one(
        self, app_completion_autostart, caplog
    ):
        """MUST FAIL before the fix: _openai_chat_completions_stream built its
        own client_meta referencing the OUTER function's _req_id local, which
        does not exist in its own scope -- a NameError -> 500 on every
        streaming request (no earlier test here exercised stream=true, so
        nothing caught it before this one). The streaming leg must carry the
        SAME req_id identity_recompose already stamped, passed in explicitly,
        not a second independently-generated one -- a different id would
        equally defeat the join even without crashing."""
        app, client = app_completion_autostart
        with caplog.at_level(logging.INFO):
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "test-model",
                      "messages": [{"role": "user", "content": "say hi"}],
                      "stream": True},
            ) as r:
                assert r.status_code == 200, "streaming route raised (500) instead of streaming"
                for _ in r.iter_bytes():
                    pass

        recompose_ids = _extract_req_ids(caplog, "identity_recompose")
        r2b_ids = _extract_req_ids(caplog, "R2B_REQ_IDENTITY")
        assert recompose_ids, "no req_id found on any identity_recompose line"
        assert r2b_ids, "no req_id found on any R2B_REQ_IDENTITY line"
        assert recompose_ids & r2b_ids, (
            f"streaming route: identity_recompose ids {recompose_ids} and "
            f"R2B_REQ_IDENTITY ids {r2b_ids} share no value -- the streaming "
            f"leg minted its own id instead of reusing the caller's"
        )

    def test_discriminator_ollama_route_joins_identity_recompose_to_r2b_req_identity(
        self, app_completion_autostart, caplog
    ):
        """Third route: assert every observability line on every route it
        can be reached from, not just the one route a test happened to
        exercise first -- ollama_chat has its own
        identity_recompose block and client_meta literal, structurally
        independent of the openai routes' two. Confirmed by direct read:
        _req_id (assigned once inside ollama_chat) and both its uses sit in
        the SAME function body, no nested/separate function the way the
        openai streaming route had -- but that's exactly the kind of claim
        this test verifies empirically instead of trusting the reading."""
        app, client = app_completion_autostart
        with caplog.at_level(logging.INFO):
            r = client.post(
                "/api/chat",
                json={"model": "test-model",
                      "messages": [{"role": "user", "content": "say hi"}]},
            )
        assert r.status_code == 200, r.text

        recompose_ids = _extract_req_ids(caplog, "identity_recompose")
        r2b_ids = _extract_req_ids(caplog, "R2B_REQ_IDENTITY")
        assert recompose_ids, "no req_id found on any identity_recompose line"
        assert r2b_ids, "no req_id found on any R2B_REQ_IDENTITY line"
        assert recompose_ids & r2b_ids, (
            f"ollama route: identity_recompose ids {recompose_ids} and "
            f"R2B_REQ_IDENTITY ids {r2b_ids} share no value"
        )
