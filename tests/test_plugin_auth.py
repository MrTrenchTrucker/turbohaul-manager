"""Tests for optional bearer auth on plugin endpoints.

Bug: whisperx returns 401 through the manager because PluginEndpoint had
nowhere to carry a credential. Fix: PluginEndpoint.auth_token_file (a PATH,
never the secret itself, by design), read fresh at
invoke time by invoke_plugin, sent as `Authorization: Bearer <token>`.

Same conventions as the existing plugin-invoke and
plugin-invoke error-redaction test files: httpx.MockTransport via
_mocked_client(), caplog.at_level(logging.ERROR, logger="turbohaul.plugin_invoke")
for log capture, every negative case asserts the SPECIFIC reason.
"""
import logging

import httpx
import pytest
from pydantic import ValidationError
from unittest.mock import patch

from turbohaul.config import PluginEndpoint
from turbohaul.plugin_invoke import PluginInvokeError, invoke_plugin

_RealAsyncClient = httpx.AsyncClient

# A realistic-looking but fake token -- never a real credential. Used
# throughout as the LITERAL string every leak assertion checks for.
TOKEN = "sk-testtok1-do-not-leak-9f8e7d6c5b4a3210"


def _mocked_client(handler):
    """Same pattern as the existing plugin_invoke test files -- invoke_plugin's
    own code is untouched, only what it talks to."""

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)

    return patch("turbohaul.plugin_invoke.httpx.AsyncClient", side_effect=_factory)


def _token_file(tmp_path, content=TOKEN):
    p = tmp_path / "whisperx.token"
    p.write_text(content + "\n" if content else content)
    return str(p)


def _auth_endpoint(tmp_path, host="172.16.0.8", port=8080):
    return PluginEndpoint(host=host, port=port, auth_token_file=_token_file(tmp_path))


def _no_auth_endpoint(host="172.16.8.5", port=8080):
    return PluginEndpoint(host=host, port=port)


# === Group 1: must-fail control -- red arm first (the bug was reproduced
# === against a live plugin endpoint; this is the automated,
# === repeatable form of the same proof, run against a disposable pre-fix
# === copy). ===================================================================


class TestBearerAuthEndToEnd:
    @pytest.mark.asyncio
    async def test_configured_endpoint_sends_correct_bearer_header_and_succeeds(self, tmp_path):
        def handler(request):
            if request.headers.get("authorization") != f"Bearer {TOKEN}":
                return httpx.Response(401, json={"detail": "Unauthorized"})
            return httpx.Response(200, json={"transcript": "hello world"})

        registry = {"whisperx": _auth_endpoint(tmp_path)}
        with _mocked_client(handler):
            result = await invoke_plugin(
                resource_key="whisperx", registry=registry, path="/transcribe-diarize", payload={},
            )
        assert result == {"transcript": "hello world"}

    @pytest.mark.asyncio
    async def test_unconfigured_endpoint_against_an_auth_requiring_plugin_gets_401(self):
        """The bug, automated: with NO auth_token_file set -- what every
        registry entry looked like before this fix -- a plugin that demands
        bearer auth returns 401, mapped to plugin_error, matching the
        observed result (502, "returned HTTP 401: '{"detail":
        "Unauthorized"}'")."""
        def handler(request):
            if request.headers.get("authorization") is None:
                return httpx.Response(401, json={"detail": "Unauthorized"})
            return httpx.Response(200, json={"transcript": "hello world"})

        registry = {"whisperx": _no_auth_endpoint(host="172.16.0.8")}
        with _mocked_client(handler):
            with pytest.raises(PluginInvokeError) as exc_info:
                await invoke_plugin(
                    resource_key="whisperx", registry=registry, path="/transcribe-diarize", payload={},
                )
        assert exc_info.value.reason == "plugin_error"
        assert "401" in exc_info.value.detail


# === Group 2: THE leak test -- assert the literal token is absent from
# === BOTH the response/error detail AND the captured logs, on every distinct
# === failure mode this module maps to a different reason. ====================


class TestAuthTokenNeverLeaks:
    @pytest.mark.asyncio
    async def test_token_absent_on_non_2xx_plugin_error(self, tmp_path, caplog):
        def handler(request):
            return httpx.Response(401, json={"detail": "Unauthorized"})

        registry = {"whisperx": _auth_endpoint(tmp_path)}
        with caplog.at_level(logging.ERROR, logger="turbohaul.plugin_invoke"):
            with _mocked_client(handler):
                with pytest.raises(PluginInvokeError) as exc_info:
                    await invoke_plugin(
                        resource_key="whisperx", registry=registry, path="/x", payload={},
                    )
        assert exc_info.value.reason == "plugin_error"
        assert TOKEN not in exc_info.value.detail
        assert TOKEN not in caplog.text

    @pytest.mark.asyncio
    async def test_token_absent_on_connect_error(self, tmp_path, caplog):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        registry = {"whisperx": _auth_endpoint(tmp_path)}
        with caplog.at_level(logging.ERROR, logger="turbohaul.plugin_invoke"):
            with _mocked_client(handler):
                with pytest.raises(PluginInvokeError) as exc_info:
                    await invoke_plugin(
                        resource_key="whisperx", registry=registry, path="/x", payload={},
                    )
        assert exc_info.value.reason == "unreachable"
        assert TOKEN not in exc_info.value.detail
        assert TOKEN not in caplog.text

    @pytest.mark.asyncio
    async def test_token_absent_on_no_progress_timeout(self, tmp_path, caplog):
        def handler(request):
            raise httpx.ReadTimeout("stall", request=request)

        registry = {"whisperx": _auth_endpoint(tmp_path)}
        with caplog.at_level(logging.ERROR, logger="turbohaul.plugin_invoke"):
            with _mocked_client(handler):
                with pytest.raises(PluginInvokeError) as exc_info:
                    await invoke_plugin(
                        resource_key="whisperx", registry=registry, path="/x", payload={},
                        no_progress_timeout_s=5.0,
                    )
        assert exc_info.value.reason == "no_progress"
        assert TOKEN not in exc_info.value.detail
        assert TOKEN not in caplog.text

    @pytest.mark.asyncio
    async def test_token_absent_on_transport_error(self, tmp_path, caplog):
        def handler(request):
            raise httpx.ReadError("reset", request=request)

        registry = {"whisperx": _auth_endpoint(tmp_path)}
        with caplog.at_level(logging.ERROR, logger="turbohaul.plugin_invoke"):
            with _mocked_client(handler):
                with pytest.raises(PluginInvokeError) as exc_info:
                    await invoke_plugin(
                        resource_key="whisperx", registry=registry, path="/x", payload={},
                    )
        assert exc_info.value.reason == "unreachable"
        assert TOKEN not in exc_info.value.detail
        assert TOKEN not in caplog.text

    @pytest.mark.asyncio
    async def test_token_absent_when_the_response_body_itself_is_verbose(self, tmp_path, caplog):
        """A verbose/misbehaving plugin error response (echoing back headers,
        as a buggy or hostile container might) must still not put the token in
        OUR OWN error detail beyond whatever the response text itself
        contained -- this specific scenario is inherently server-controlled
        (invoke_plugin only ever forwards response.text[:500] verbatim, it
        cannot scrub a server's own response), so this test documents that
        boundary rather than asserting something invoke_plugin cannot
        possibly guarantee: it deliberately does NOT configure a request that
        would make the mock echo the header, only confirms invoke_plugin adds
        nothing of its own beyond the response text."""
        def handler(request):
            return httpx.Response(500, text="internal plugin crash: OOM, no request data echoed")

        registry = {"whisperx": _auth_endpoint(tmp_path)}
        with caplog.at_level(logging.ERROR, logger="turbohaul.plugin_invoke"):
            with _mocked_client(handler):
                with pytest.raises(PluginInvokeError) as exc_info:
                    await invoke_plugin(
                        resource_key="whisperx", registry=registry, path="/x", payload={},
                    )
        assert TOKEN not in exc_info.value.detail
        assert TOKEN not in caplog.text

    def test_token_absent_from_httpx_exception_str_and_repr_directly(self):
        """The subtlest vector: does httpx's OWN exception stringification
        include the outgoing request (and therefore a real Authorization
        header) when a transport error hits a request that genuinely carries
        one? Bypasses this module entirely -- constructs a real httpx.Request
        with the real header and inspects str(exc)/repr(exc) directly, so this
        proves the UPSTREAM library doesn't hand invoke_plugin a
        pre-poisoned string it would have no way to know to scrub."""
        request = httpx.Request(
            "POST", "http://172.16.0.8:8080/x",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        exc = httpx.ConnectError("refused", request=request)
        assert TOKEN not in str(exc)
        assert TOKEN not in repr(exc)

    def test_token_absent_from_the_constructed_endpoint_repr(self, tmp_path):
        """A different leak vector: PluginEndpoint stores a PATH, never
        content -- so its own auto-generated Pydantic repr/str must never
        contain the token even though the file it points at does."""
        endpoint = _auth_endpoint(tmp_path)
        assert TOKEN not in repr(endpoint)
        assert TOKEN not in str(endpoint)


# === Group 3: the no-auth path must be untouched. ==========================


class TestNoAuthPathUnchanged:
    @pytest.mark.asyncio
    async def test_marker_style_entry_sends_no_authorization_header(self, tmp_path):
        captured = {}

        def handler(request):
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json={"ok": True})

        registry = {"marker": _no_auth_endpoint()}
        with _mocked_client(handler):
            result = await invoke_plugin(
                resource_key="marker", registry=registry, path="/convert", payload={},
            )
        assert result == {"ok": True}
        assert "authorization" not in captured["headers"]

    def test_auth_token_file_defaults_to_none_and_is_optional(self):
        # extra="forbid" already proves nothing NEW is required -- construction
        # with only the pre-existing fields must still succeed unchanged.
        endpoint = PluginEndpoint(host="172.16.8.5", port=8080)
        assert endpoint.auth_token_file is None


# === Group 4: fail LOUDLY at boot, not silently at request time. ============


class TestAuthTokenFileBootValidation:
    def test_missing_file_raises_at_construction_not_invoke(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginEndpoint(host="172.16.0.8", port=8080, auth_token_file="/nonexistent/whisperx.token")
        assert "/nonexistent/whisperx.token" in str(exc_info.value)

    def test_empty_file_raises_at_construction(self, tmp_path):
        empty = tmp_path / "empty.token"
        empty.write_text("")
        with pytest.raises(ValidationError) as exc_info:
            PluginEndpoint(host="172.16.0.8", port=8080, auth_token_file=str(empty))
        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_only_file_raises_at_construction(self, tmp_path):
        ws = tmp_path / "whitespace.token"
        ws.write_text("   \n\t\n")
        with pytest.raises(ValidationError):
            PluginEndpoint(host="172.16.0.8", port=8080, auth_token_file=str(ws))

    def test_unreadable_file_raises_at_construction(self, tmp_path):
        unreadable = tmp_path / "noperm.token"
        unreadable.write_text(TOKEN)
        unreadable.chmod(0o000)
        try:
            with pytest.raises(ValidationError):
                PluginEndpoint(host="172.16.0.8", port=8080, auth_token_file=str(unreadable))
        finally:
            unreadable.chmod(0o644)  # restore so tmp_path cleanup can remove it

    def test_valid_file_constructs_successfully(self, tmp_path):
        endpoint = _auth_endpoint(tmp_path)
        assert endpoint.auth_token_file is not None


# === Must-fail control for the must-fail-control tests: prove they can
# === actually distinguish a broken fix from a working one. ===================


class TestMustFailControlIsReal:
    """★ Break the fix, prove the specific tests above go red, restore. Not a
    disposable mutant (module-level monkeypatch is simpler and
    exactly proves the same thing: this test file discriminates)."""

    @pytest.mark.asyncio
    async def test_a_neutered_header_build_is_caught_by_the_e2e_test(self, tmp_path, monkeypatch):
        """Simulates the exact regression class this whole feature's tests exist to catch:
        invoke_plugin silently NOT sending the header even though
        auth_token_file is configured (e.g. a future refactor that drops the
        `headers=headers` kwarg from the client.post call)."""
        import turbohaul.plugin_invoke as pi

        real_post = httpx.AsyncClient.post

        async def post_dropping_headers(self, url, *, json=None, headers=None, **kw):
            # MUTANT: headers silently discarded, exactly the regression class
            return await real_post(self, url, json=json, **kw)

        monkeypatch.setattr(httpx.AsyncClient, "post", post_dropping_headers)

        def handler(request):
            if request.headers.get("authorization") != f"Bearer {TOKEN}":
                return httpx.Response(401, json={"detail": "Unauthorized"})
            return httpx.Response(200, json={"ok": True})

        registry = {"whisperx": _auth_endpoint(tmp_path)}
        with _mocked_client(handler):
            with pytest.raises(PluginInvokeError) as exc_info:
                await invoke_plugin(
                    resource_key="whisperx", registry=registry, path="/x", payload={},
                )
        assert exc_info.value.reason == "plugin_error"
        assert "401" in exc_info.value.detail
        # Restored automatically by monkeypatch fixture teardown; the
        # unmutated TestBearerAuthEndToEnd test above re-proves GREEN.
