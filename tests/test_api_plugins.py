"""Tests for /api/plugins routes.

Mounts turbohaul.api.plugins.router directly on a bare FastAPI app with a
lightweight fake manager (only .boot / .runtime are touched by these routes) --
not the full create_app() lifecycle, which boots a real TurbohaulManager and is
unnecessary for routes that only read manifests + boot config. Fixture manifests
are written directly via write_manifest_atomic (there is no POST /api/plugins
route to create one -- manifest creation stays PUT /api/manifests, out of this
file's scope).

plugin_invoke.invoke_plugin / resolve_endpoint are in a separate module, mocked
throughout -- this file proves the api/ layer's own logic (routing, redaction,
error-reason-to-HTTP mapping), not that module's implementation.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from turbohaul.api.plugins import router as plugins_router
from turbohaul.config import PluginEndpoint, PluginRuntimeConfig, PluginsConfig
from turbohaul.manifest import ModelManifest, PluginManifest, write_manifest_atomic
from turbohaul.plugin_invoke import PluginInvokeError


SAMPLE_SHA = "a" * 64
RESOURCE_KEY = "whisperx-main"
NO_PROGRESS_TIMEOUT = 123.0


def _mk_plugin(**overrides) -> PluginManifest:
    base = dict(
        model_tag="whisperx-transcribe",
        kind="plugin",
        lane="gpu",
        resource_key=RESOURCE_KEY,
        capabilities=["transcribe", "diarize"],
    )
    base.update(overrides)
    return PluginManifest(**base)


def _mk_model(**overrides) -> ModelManifest:
    base = dict(model_tag="a-real-model", gguf_blob_sha256=SAMPLE_SHA)
    base.update(overrides)
    return ModelManifest(**base)


@pytest.fixture
def app_client(tmp_path):
    manifests_path = tmp_path / "manifests"
    manifests_path.mkdir()
    registry = {RESOURCE_KEY: PluginEndpoint(host="whisperx.internal", port=9000)}
    fake_mgr = SimpleNamespace(
        boot=SimpleNamespace(
            storage=SimpleNamespace(manifests_path=manifests_path),
            plugins=PluginsConfig(registry=registry),
        ),
        runtime=SimpleNamespace(
            plugin_runtime=PluginRuntimeConfig(no_progress_timeout_s=NO_PROGRESS_TIMEOUT),
        ),
    )
    app = FastAPI()
    app.include_router(plugins_router)
    app.state.manager = fake_mgr
    with TestClient(app) as client:
        yield app, client, manifests_path


class TestListPlugins:
    def test_lists_plugin_manifests_with_configured_status(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin())
        with patch("turbohaul.api.plugins.resolve_endpoint") as mock_resolve:
            mock_resolve.return_value = object()
            r = client.get("/api/plugins")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        # Deliberately EXHAUSTIVE (==, not a subset check). The response includes
        # provides_routes / provides_executables / invoke, and this assertion
        # forces any addition to be reviewed key-by-key rather than
        # slipped in -- on an endpoint whose contract is "never host or port",
        # an exhaustive assertion on the response shape is a feature. Keep it.
        assert body["plugins"][0] == {
            "model_tag": "whisperx-transcribe",
            "lane": "gpu",
            "capabilities": ["transcribe", "diarize"],
            # Named "configured", not "resolves": the value is a
            # registry-lookup result, and a reachability name would assert
            # reachability that nothing has established.
            "configured": True,
            "provides_routes": [],
            "provides_executables": [],
            "invoke": {
                "method": "POST",
                "url": "/api/plugins/whisperx-transcribe/invoke",
                "body": {"path": "<one of provides_routes>", "payload": {}},
            },
        }
        mock_resolve.assert_called_once()

    def test_unregistered_resource_key_reports_false_not_an_error(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin(model_tag="orphan-plugin"))
        with patch("turbohaul.api.plugins.resolve_endpoint") as mock_resolve:
            mock_resolve.side_effect = PluginInvokeError("unknown_resource", "not in registry")
            r = client.get("/api/plugins")
        assert r.status_code == 200
        assert r.json()["plugins"][0]["configured"] is False

    def test_hidden_plugin_omitted_from_list(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin(model_tag="hidden-one", hidden=True))
        write_manifest_atomic(manifests_path, _mk_plugin(model_tag="visible-one"))
        with patch("turbohaul.api.plugins.resolve_endpoint"):
            r = client.get("/api/plugins")
        tags = [p["model_tag"] for p in r.json()["plugins"]]
        assert tags == ["visible-one"]

    def test_model_manifests_excluded_from_plugin_list(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_model())
        write_manifest_atomic(manifests_path, _mk_plugin())
        with patch("turbohaul.api.plugins.resolve_endpoint"):
            r = client.get("/api/plugins")
        assert r.json()["total"] == 1
        assert r.json()["plugins"][0]["model_tag"] == "whisperx-transcribe"

    def test_never_returns_host_or_port(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin())
        with patch("turbohaul.api.plugins.resolve_endpoint"):
            r = client.get("/api/plugins")
        raw = r.text
        assert "whisperx.internal" not in raw  # the configured host
        assert "9000" not in raw  # the configured port
        for plugin in r.json()["plugins"]:
            assert "host" not in plugin
            assert "port" not in plugin

    def test_unreadable_manifest_skipped_not_fatal(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin())
        (manifests_path / "corrupt.yaml").write_text("not: valid: yaml: [[[")
        with patch("turbohaul.api.plugins.resolve_endpoint"):
            r = client.get("/api/plugins")
        assert r.status_code == 200
        assert r.json()["total"] == 1  # corrupt.yaml skipped, not fatal


class TestConfiguredIsNotAReachabilityClaim:
    """The CONTROL for the bug these tests exist to kill.

    GET /api/plugins must not report `resolves: true` as a reachability claim:
    a capability can be configured and resolvable while every request to it
    fails, because resolve_endpoint is a registry lookup + address-policy
    check that never opens a socket. Whether a live endpoint accepts
    connections is
    NOT checked here and is
    not checkable from a test. These are the hermetic half: they pin the
    contradiction in-tree so it cannot return under any name.

    resolve_endpoint is deliberately NOT mocked in this class -- unlike the
    rest of this file, the point here is the REAL function on the real path
    that produced the bug.
    """

    @staticmethod
    def _client(tmp_path, registry):
        manifests_path = tmp_path / "manifests"
        manifests_path.mkdir()
        fake_mgr = SimpleNamespace(
            boot=SimpleNamespace(
                storage=SimpleNamespace(manifests_path=manifests_path),
                plugins=PluginsConfig(registry=registry),
            ),
            runtime=SimpleNamespace(
                plugin_runtime=PluginRuntimeConfig(no_progress_timeout_s=NO_PROGRESS_TIMEOUT),
            ),
        )
        app = FastAPI()
        app.include_router(plugins_router)
        app.state.manager = fake_mgr
        return TestClient(app), manifests_path

    def test_dead_rfc1918_address_reports_configured_true_and_that_is_now_honest(self, tmp_path):
        """The exact failing shape: a private address, nothing behind it.

        RFC1918 is DELIBERATELY absent from plugin_invoke's kept denyset
        (real plugin containers live there), so
        resolve_endpoint returns cleanly for an address with nothing listening.
        The VALUE stays true, and that is correct: the entry genuinely IS
        configured. What must never come back is the field CLAIMING the
        container is up.
        """
        registry = {RESOURCE_KEY: PluginEndpoint(host="172.16.0.8", port=8080)}
        client, manifests_path = self._client(tmp_path, registry)
        write_manifest_atomic(manifests_path, _mk_plugin())
        r = client.get("/api/plugins")
        assert r.status_code == 200
        entry = r.json()["plugins"][0]
        assert entry["configured"] is True
        assert "resolves" not in entry, (
            "the reachability-named field must be GONE from the wire -- nothing "
            "in this request path opened a socket, so nothing here may assert "
            "reachability under any name"
        )

    def test_registered_at_a_refused_address_reports_configured_false(self, tmp_path):
        """Why the field is NOT called `registered`.

        This entry IS registered and still reports false: 169.254.0.0/16 is in
        plugin_invoke's KEPT denyset (the IMDS range), so resolve_endpoint
        raises blocked_target. A field named `registered` would mislabel
        exactly this row -- a subtler instance of the same defect these tests are
        about. Executable record of that naming decision.
        """
        registry = {RESOURCE_KEY: PluginEndpoint(host="169.254.169.254", port=80)}
        client, manifests_path = self._client(tmp_path, registry)
        write_manifest_atomic(manifests_path, _mk_plugin())
        r = client.get("/api/plugins")
        assert r.status_code == 200
        entry = r.json()["plugins"][0]
        assert entry["configured"] is False
        assert "resolves" not in entry

    def test_listing_opens_no_socket_so_it_cannot_be_asserting_reachability(self, tmp_path):
        """Structural proof, not a naming one: the value CANNOT be a
        reachability claim, because the request path never builds an HTTP
        client. Also pins a design constraint -- one live probe per plugin, with
        timeouts, on an endpoint the UI polls would turn a silent wrong answer
        into a slow one.
        """
        registry = {RESOURCE_KEY: PluginEndpoint(host="172.16.0.8", port=8080)}
        client, manifests_path = self._client(tmp_path, registry)
        write_manifest_atomic(manifests_path, _mk_plugin())

        def _explode(*a, **kw):
            raise AssertionError(
                "GET /api/plugins built an HTTP client. Probing here is what "
                "is forbidden; reachability belongs to invoke_plugin."
            )

        with patch("turbohaul.plugin_invoke.httpx.AsyncClient", _explode), \
             patch("turbohaul.api.plugins.invoke_plugin") as mock_invoke:
            r = client.get("/api/plugins")
        assert r.status_code == 200
        assert r.json()["plugins"][0]["configured"] is True
        mock_invoke.assert_not_called()


class TestSelfDescription:
    """The listing must tell an agent enough to CALL the plugin,
    while the host/port property survives intact."""

    def test_routes_surfaced_in_listing(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin(
            provides_routes=["/analyze-audio", "/video-frames", "/describe"],
            provides_executables=["ffmpeg", "ffprobe"],
        ))
        with patch("turbohaul.api.plugins.resolve_endpoint"):
            r = client.get("/api/plugins")
        entry = r.json()["plugins"][0]
        assert entry["provides_routes"] == ["/analyze-audio", "/video-frames", "/describe"]
        assert entry["provides_executables"] == ["ffmpeg", "ffprobe"]

    def test_invoke_shape_is_discoverable_and_relative(self, app_client):
        """The missing half of discoverability: knowing /analyze-audio exists
        is useless without knowing it is reached by POSTing {"path": ...} to
        the invoke route. The url must be RELATIVE -- an absolute one would be
        a host leak wearing a helpful hat."""
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin(provides_routes=["/describe"]))
        with patch("turbohaul.api.plugins.resolve_endpoint"):
            r = client.get("/api/plugins")
        invoke = r.json()["plugins"][0]["invoke"]
        assert invoke["method"] == "POST"
        assert invoke["url"] == "/api/plugins/whisperx-transcribe/invoke"
        assert invoke["url"].startswith("/")
        assert "://" not in invoke["url"]
        assert set(invoke["body"]) == {"path", "payload"}

    def test_new_fields_do_not_become_a_host_or_port_leak_channel(self, app_client):
        """The no-host/no-port property, re-asserted with the new fields POPULATED --
        the original test only ever saw them empty. Note this passes for a
        structural reason: the host/port-bearing shapes cannot be STORED in
        provides_routes (manifest validation rejects ':' and '//'), so there is
        nothing here to redact."""
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin(
            provides_routes=["/analyze-audio", "/video-frames", "/describe"],
            provides_executables=["ffmpeg"],
        ))
        with patch("turbohaul.api.plugins.resolve_endpoint"):
            r = client.get("/api/plugins")
        raw = r.text
        assert "whisperx.internal" not in raw
        assert "9000" not in raw
        for plugin in r.json()["plugins"]:
            assert "host" not in plugin
            assert "port" not in plugin

    def test_a_manifest_cannot_declare_a_host_bearing_route_in_the_first_place(self):
        """Belt and braces on the line above: prove the storage-side rejection
        rather than inferring it from a clean response body."""
        with pytest.raises(ValidationError):
            _mk_plugin(provides_routes=["//whisperx.internal:9000/x"])

    def test_absent_fields_serialize_as_empty_not_missing(self, app_client):
        """An agent parsing the listing should never have to branch on a key
        being absent for older manifests."""
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin())
        with patch("turbohaul.api.plugins.resolve_endpoint"):
            r = client.get("/api/plugins")
        entry = r.json()["plugins"][0]
        assert entry["provides_routes"] == []
        assert entry["provides_executables"] == []


class TestInvokePlugin:
    def test_happy_path(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin())
        with patch("turbohaul.api.plugins.invoke_plugin", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = {"text": "hello world"}
            r = client.post(
                "/api/plugins/whisperx-transcribe/invoke",
                json={"path": "/transcribe", "payload": {"audio_b64": "..."}},
            )
        assert r.status_code == 200
        assert r.json() == {"text": "hello world"}

    def test_invoke_wires_resource_key_registry_path_payload_and_timeout(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin())
        with patch("turbohaul.api.plugins.invoke_plugin", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = {}
            client.post(
                "/api/plugins/whisperx-transcribe/invoke",
                json={"path": "/transcribe", "payload": {"k": "v"}},
            )
        mock_invoke.assert_called_once_with(
            resource_key=RESOURCE_KEY,
            registry={RESOURCE_KEY: PluginEndpoint(host="whisperx.internal", port=9000)},
            path="/transcribe",
            payload={"k": "v"},
            no_progress_timeout_s=NO_PROGRESS_TIMEOUT,
        )

    @pytest.mark.parametrize("reason,expected_status", [
        ("unknown_resource", 400),
        ("blocked_target", 400),
        ("unreachable", 502),
        ("plugin_error", 502),
        ("no_progress", 504),
    ])
    def test_plugin_invoke_error_reason_maps_to_specific_status(self, app_client, reason, expected_status):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin())
        with patch("turbohaul.api.plugins.invoke_plugin", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = PluginInvokeError(reason, "boom detail")
            r = client.post(
                "/api/plugins/whisperx-transcribe/invoke",
                json={"path": "/transcribe", "payload": {}},
            )
        assert r.status_code == expected_status
        detail = r.json()["detail"]
        assert detail["reason"] == reason
        assert detail["detail"] == "boom detail"

    def test_unrecognized_reason_falls_back_to_502_not_crash(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin())
        with patch("turbohaul.api.plugins.invoke_plugin", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = PluginInvokeError("a_future_reason_not_in_the_table", "x")
            r = client.post(
                "/api/plugins/whisperx-transcribe/invoke",
                json={"path": "/transcribe", "payload": {}},
            )
        assert r.status_code == 502
        assert r.json()["detail"]["reason"] == "a_future_reason_not_in_the_table"

    def test_invoke_against_model_manifest_400(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_model())
        r = client.post(
            "/api/plugins/a-real-model/invoke",
            json={"path": "/x", "payload": {}},
        )
        assert r.status_code == 400
        assert "not a plugin manifest" in r.json()["detail"]

    def test_invoke_unknown_tag_404(self, app_client):
        app, client, manifests_path = app_client
        r = client.post(
            "/api/plugins/does-not-exist/invoke",
            json={"path": "/x", "payload": {}},
        )
        assert r.status_code == 404

    def test_invoke_hidden_plugin_still_reachable_by_exact_tag(self, app_client):
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin(model_tag="hidden-plugin", hidden=True))
        with patch("turbohaul.api.plugins.invoke_plugin", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = {"ok": True}
            r = client.post(
                "/api/plugins/hidden-plugin/invoke",
                json={"path": "/x", "payload": {}},
            )
        assert r.status_code == 200

    def test_never_reimplements_invoke_plugin_always_calls_the_plugin_invoke_function(self, app_client):
        """Structural guard: proves the route calls turbohaul.plugin_invoke.invoke_plugin
        (patchable at that seam), not a locally re-derived HTTP call. If a future edit
        started making its own httpx/requests call instead of using invoke_plugin, this
        mock would go uncalled and the happy-path assertion below would fail on a
        NotImplementedError from the stub instead of returning the mocked value."""
        app, client, manifests_path = app_client
        write_manifest_atomic(manifests_path, _mk_plugin())
        with patch("turbohaul.api.plugins.invoke_plugin", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = {"proof": True}
            r = client.post(
                "/api/plugins/whisperx-transcribe/invoke",
                json={"path": "/x", "payload": {}},
            )
        assert mock_invoke.await_count == 1
        assert r.json() == {"proof": True}
