"""Fast Lane label on the request-identity surface.

When a Fast Lane-registered IP with an operator-saved label is the client of
the admitted request, the label rides on the request-identity dict (stashed on
``manager._last_request_identity`` and served on ``/status.request_identity``)
so the FE can render it between the IP and the role tag.

Contract: an IP with NO saved label shows NO label —
``label`` is ``null`` when fastlane is disabled, no rule matches, or the
matching rule's label is empty. No hardcoded labels anywhere: the value comes
from the operator's Fast Lane config (``runtime.fastlane.rules[].label``),
stamped at admission onto ``slot.fastlane`` and read through here.

The TestClient's ASGI scope reports the pseudo-host 'testclient', which
``compile_fastlane`` correctly refuses (fastlane.py: normalize_address ->
ipaddress.ip_address; unparseable addresses are skipped, not stored). So these
tests wrap the app in a tiny ASGI override that pins the connection to a real
documented-test IP (192.0.2.55, TEST-NET-1) — the same shape of input the
production path receives (request.client.host over a real socket).

Without the label support, the identity dict has no ``label`` key at all.
Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this test file> -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.main import create_app
from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.subprocess_mgr import SidecarHandle

CLIENT_IP = "192.0.2.55"  # TEST-NET-1 (RFC 5737) — documented for tests
OTHER_IP = "192.0.2.99"   # a different documented-test IP: no rule matches it


class _ClientOverride:
    """ASGI wrapper pinning scope['client'] to a real IP (the TestClient's
    pseudo-host 'testclient' is not a parseable address and would be
    correctly skipped by compile_fastlane)."""

    def __init__(self, app, host: str):
        self._app = app
        self._host = host

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope["client"] = (self._host, 50000)
        await self._app(scope, receive, send)


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


def _make_app(tmp_path: Path, fastlane: FastLaneConfig | None) -> "object":
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
        queue=QueueConfig(
            safety_enabled=False,
            grace_seconds=0, idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1, drained_sigterm_window_cold_s=1,
        ),
        pull=PullConfig(),
        # fastlane=None -> explicit disabled default (RuntimeConfig.fastlane is
        # non-optional; FastLaneConfig()'s defaults are enabled=False, rules=[])
        fastlane=fastlane if fastlane is not None else FastLaneConfig(),
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
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": slot.model_tag,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
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
    return app


def _post(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "test-model",
              "messages": [{"role": "user", "content": "say hi"}]},
    )
    assert r.status_code == 200, r.text
    return r


class TestFastLaneLabelOnRequestIdentity:

    def test_labeled_rule_ip_carries_label_on_status_and_last_identity(self, tmp_path):
        fastlane = FastLaneConfig(
            enabled=True,
            rules=[FastLaneRule(address=CLIENT_IP, label="TestRouter")],
        )
        app = _make_app(tmp_path, fastlane)
        with TestClient(_ClientOverride(app, CLIENT_IP)) as client:
            _post(client)
            mgr = app.state.manager
            ident = mgr._last_request_identity
            assert ident is not None, "request identity was never emitted"
            assert "label" in ident, (
                "identity dict must carry the 'label' key (request-identity label feature); "
                f"got keys: {sorted(ident)}"
            )
            assert ident["ip"] == CLIENT_IP, (
                f"identity ip should be the pinned client {CLIENT_IP}, got {ident['ip']!r}"
            )
            assert ident["label"] == "TestRouter"
            # Live-path proof: the /status payload the FE polls at 1 Hz carries it.
            status = client.get("/status")
            assert status.status_code == 200, status.text
            assert status.json()["request_identity"]["label"] == "TestRouter"

    def test_unmatched_ip_gets_null_label(self, tmp_path):
        fastlane = FastLaneConfig(
            enabled=True,
            rules=[FastLaneRule(address=OTHER_IP, label="SomeoneElse")],
        )
        app = _make_app(tmp_path, fastlane)
        with TestClient(_ClientOverride(app, CLIENT_IP)) as client:
            _post(client)
            ident = app.state.manager._last_request_identity
            assert ident is not None
            assert ident["label"] is None, (
                "no matching rule -> label must be null (no placeholder, no guess)"
            )

    def test_empty_rule_label_gets_null_label(self, tmp_path):
        fastlane = FastLaneConfig(
            enabled=True,
            rules=[FastLaneRule(address=CLIENT_IP, label="")],
        )
        app = _make_app(tmp_path, fastlane)
        with TestClient(_ClientOverride(app, CLIENT_IP)) as client:
            _post(client)
            ident = app.state.manager._last_request_identity
            assert ident is not None
            assert ident["label"] is None, (
                "rule with an empty saved label == no label saved -> null"
            )

    def test_fastlane_disabled_gets_null_label(self, tmp_path):
        app = _make_app(tmp_path, None)  # FastLaneConfig() default: disabled
        with TestClient(_ClientOverride(app, CLIENT_IP)) as client:
            _post(client)
            ident = app.state.manager._last_request_identity
            assert ident is not None
            assert ident["label"] is None

    def test_identity_keys_otherwise_unchanged(self, tmp_path):
        """The feature ADDS 'label'; it must not drop or rename any existing
        identity field the FE or grep tooling already joins on."""
        fastlane = FastLaneConfig(
            enabled=True,
            rules=[FastLaneRule(address=CLIENT_IP, label="TestRouter")],
        )
        app = _make_app(tmp_path, fastlane)
        with TestClient(_ClientOverride(app, CLIENT_IP)) as client:
            _post(client)
            ident = app.state.manager._last_request_identity
            assert ident is not None
            for key in ("ip", "model_tag", "session_id", "is_main",
                        "is_sub_agent", "is_curator", "is_compression",
                        "resolved_class", "thread_id"):
                assert key in ident, f"existing identity key {key!r} missing"
