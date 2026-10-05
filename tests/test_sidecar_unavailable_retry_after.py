"""Graceful-recovery assessment -- the Retry-After fix.

Assessment found the recovery mechanism itself (idle-dead sweep, respawn) already
works and detection is fast; the real gap was that the 503 sidecar_unavailable
response advertised a hardcoded Retry-After (30s) with zero relationship to the
system's real respawn times (typically minutes, not seconds) -- a
client following the server's own advice would very likely retry into a still-dead
service.

Design: derive Retry-After from queue.loading_health_timeout_s (the SAME
deadline the manager already gives itself for a model load) instead of inventing a
second arbitrary number, per the safety principle NEVER ADVERTISE SHORTER THAN WE
WOULD WAIT -- err long, never short, since an optimistic value can turn one crash
into a retry storm and a pessimistic one only costs a small delay.

Also fixes the free-text sidecar_unavailable message so it no longer LEADS with the
raw httpx transport exception name (misread as "a networking bug") -- the
structured `cause` field was already correct and stays untouched.

Three controls, as follows:
  NEGATIVE/FLOOR -- loading_health_timeout_s smaller than the exception's own
    default: the default (floor) wins, Retry-After is never advertised LOWER
    than the exception's own baseline.
  POSITIVE/DERIVED -- loading_health_timeout_s larger than the default: the
    derived (config-driven) value wins, proving the header actually reflects
    what the system would really wait.
  LIVE-READ -- the value is read fresh at call time, not captured once: mutating
    mgr.runtime between two calls changes the second call's result, proving this
    doesn't go stale the way a construction-time capture would after a
    PUT /api/config swap (config_put.py replaces mgr.runtime wholesale).
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.chat_completion import (
    SidecarUnavailableError,
    _sidecar_unavailable_retry_after_s,
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


# ============================================================================
# A. _sidecar_unavailable_retry_after_s -- pure function
# ============================================================================


def _mgr_with_timeout(timeout_s):
    """Minimal stand-in exposing exactly the attribute path the function reads."""
    return SimpleNamespace(runtime=SimpleNamespace(queue=SimpleNamespace(
        loading_health_timeout_s=timeout_s,
    )))


class TestSidecarUnavailableRetryAfter:
    def test_negative_control_floor_wins_when_timeout_is_smaller(self):
        # loading_health_timeout_s=10 (below the exception's own 30 default) --
        # the advertised value must never drop below the exception's floor.
        exc = SidecarUnavailableError("x", retry_after_s=30)
        mgr = _mgr_with_timeout(10)
        assert _sidecar_unavailable_retry_after_s(mgr, exc) == 30

    def test_positive_control_derived_value_wins_when_larger(self):
        # The shipped production default (120s, the model-load timeout) --
        # must actually be advertised, not silently capped at 30.
        exc = SidecarUnavailableError("x", retry_after_s=30)
        mgr = _mgr_with_timeout(120)
        assert _sidecar_unavailable_retry_after_s(mgr, exc) == 120

    def test_exact_equal_boundary(self):
        exc = SidecarUnavailableError("x", retry_after_s=30)
        mgr = _mgr_with_timeout(30)
        assert _sidecar_unavailable_retry_after_s(mgr, exc) == 30

    def test_live_read_not_captured_at_construction(self):
        # Proves this reads mgr.runtime FRESH each call, not a snapshot --
        # the property PUT /api/config's wholesale mgr.runtime replacement
        # depends on (config_put.py: `mgr.runtime = new_runtime`).
        exc = SidecarUnavailableError("x", retry_after_s=30)
        mgr = _mgr_with_timeout(60)
        first = _sidecar_unavailable_retry_after_s(mgr, exc)
        # Simulate an operator PUT /api/config retuning the timeout --
        # config_put.py's real mechanism replaces mgr.runtime wholesale.
        mgr.runtime = _mgr_with_timeout(300).runtime
        second = _sidecar_unavailable_retry_after_s(mgr, exc)
        assert first == 60
        assert second == 300, "must reflect the retuned config, not a stale capture"

    def test_never_returns_below_exception_floor_even_on_read_failure(self):
        # mgr missing the expected attribute path entirely -- fail SAFE to the
        # exception's own floor, never crash, never return something smaller.
        exc = SidecarUnavailableError("x", retry_after_s=30)
        broken_mgr = SimpleNamespace()  # no .runtime at all
        assert _sidecar_unavailable_retry_after_s(broken_mgr, exc) == 30

    def test_default_exception_floor_is_30(self):
        # Documents the floor this whole design leans on; if this constant
        # ever changes, this test should be the one that notices.
        exc = SidecarUnavailableError("x")
        assert exc.retry_after_s == 30


# ============================================================================
# B. Free-text message no longer leads with the raw transport exception name
# ============================================================================


class TestMessageDoesNotLeadWithTransportExceptionName:
    def test_message_leads_with_human_cause_not_exception_class_name(self):
        exc = SidecarUnavailableError(
            "sidecar crashed or disconnected "
            "(transport error: RemoteProtocolError: Server disconnected)",
            cause="sidecar_disconnected_or_crashed",
        )
        msg = str(exc)
        assert msg.startswith("sidecar crashed or disconnected"), msg
        assert not msg.startswith("RemoteProtocolError"), msg
        # The exception detail is not thrown away -- just demoted, not deleted.
        assert "RemoteProtocolError" in msg
        # cause field: unchanged by this fix, still the structured, correct signal.
        assert exc.cause == "sidecar_disconnected_or_crashed"


# ============================================================================
# C. End-to-end through the real route
# ============================================================================


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


def _app_with_crashing_complete(tmp_path, loading_health_timeout_s):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    # Chat/completion routes 404 on an unknown model tag -- needs a
    # real manifest present, same requirement test_api_chat_completion.py's
    # own app_completion_autostart fixture documents.
    _write_manifest_yaml(storage_root / "manifests", "m")
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
            default_port_base=59800,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, grace_seconds=0, idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1, drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=loading_health_timeout_s,
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

    async def fake_complete_crashes(slot, handle):
        # Same shape make_llama_server_complete_fn would raise on a real
        # RemoteProtocolError -- exercises the actual route-level Retry-After
        # computation this fix changed, end to end.
        raise SidecarUnavailableError(
            "sidecar crashed or disconnected (transport error: "
            "RemoteProtocolError: Server disconnected without sending a "
            "response.)",
            cause="sidecar_disconnected_or_crashed",
            retry_after_s=30,
        )

    mgr._spawn = fake_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = fake_sigterm
    mgr._vram_verify = fake_vram
    mgr._complete_fn = fake_complete_crashes
    return app, mgr


class TestEndToEndRetryAfter:
    def test_shipped_default_120s_advertised_end_to_end(self, tmp_path):
        # The model-load timeout's shipped production value.
        app, mgr = _app_with_crashing_complete(tmp_path, loading_health_timeout_s=120)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 503
        assert r.headers["Retry-After"] == "120"
        # Note: detail["error"] is now a typed dict, not a bare
        # string -- {"error": {"type": ..., "cause": ..., "message": ...}},
        # matching every other error family's shape in this API.
        err = r.json()["detail"]["error"]
        assert err["type"] == "sidecar_unavailable"
        assert err["cause"] == "sidecar_disconnected_or_crashed"
        assert not err["message"].startswith("RemoteProtocolError")

    def test_low_timeout_floors_at_the_exception_default(self, tmp_path):
        # loading_health_timeout_s below the range floor (10, the config's own
        # minimum, ge=10) -- must never advertise less than 30.
        app, mgr = _app_with_crashing_complete(tmp_path, loading_health_timeout_s=10)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 503
        assert r.headers["Retry-After"] == "30"

    def test_ollama_compat_route_also_advertises_derived_value(self, tmp_path):
        # The fix touches THREE call sites (openai non-stream, openai stream,
        # ollama-compat) -- prove the ollama route independently, not just
        # assume symmetry with the openai one.
        app, mgr = _app_with_crashing_complete(tmp_path, loading_health_timeout_s=120)
        with TestClient(app) as client:
            r = client.post(
                "/api/chat",
                json={
                    "model": "m", "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
        assert r.status_code == 503
        assert r.headers["Retry-After"] == "120"
