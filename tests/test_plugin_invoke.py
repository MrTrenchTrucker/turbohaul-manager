"""Tests for turbohaul.plugin_invoke.

Every negative case asserts the SPECIFIC PluginInvokeError.reason, not a bare
raises. The HTTP layer is always mocked via httpx.MockTransport -- no real
container is ever contacted.
"""
from unittest.mock import patch

import httpx
import pytest

from turbohaul.config import PluginEndpoint
from turbohaul.plugin_invoke import (
    PluginInvokeError,
    _is_never_valid_ip,
    _parse_ip_literal,
    _validate_path,
    invoke_plugin,
    resolve_endpoint,
)

_RealAsyncClient = httpx.AsyncClient


def _mocked_client(handler):
    """Patch httpx.AsyncClient so invoke_plugin's internal client construction
    is transparently backed by a MockTransport -- invoke_plugin's own code is
    never touched, only what it talks to."""

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)

    return patch("turbohaul.plugin_invoke.httpx.AsyncClient", side_effect=_factory)


def _endpoint(host="172.16.0.8", port=8080, health_path="/health"):
    return PluginEndpoint(host=host, port=port, health_path=health_path)


# === resolve_endpoint ========================================================


class TestResolveEndpointUnknownResource:
    def test_missing_key_raises_unknown_resource(self):
        with pytest.raises(PluginInvokeError) as exc_info:
            resolve_endpoint("nope", {})
        assert exc_info.value.reason == "unknown_resource"

    def test_empty_registry(self):
        with pytest.raises(PluginInvokeError) as exc_info:
            resolve_endpoint("whisperx", {})
        assert exc_info.value.reason == "unknown_resource"


class TestResolveEndpointBlockedTarget:
    """Only the curated never-valid subset is blocked -- NOT the RFC1918/
    CGNAT portion of ssrf_guard.py's denyset. Loopback IS blocked in both
    families (an explicit tightening): no plugin container legitimately
    serves on loopback, and both families reach the manager's own API."""

    @pytest.mark.parametrize(
        "host",
        [
            "224.0.0.5",  # multicast
            "240.0.0.1",  # reserved
            "0.0.0.5",  # "this network"
            "127.0.0.1",  # loopback -- BLOCKED since the loopback tightening (reaches the manager's own API)
            "169.254.169.254",  # link-local / cloud IMDS -- an explicit exception
            "::ffff:0:0",  # IPv4-mapped IPv6 (in ::ffff:0:0/96)
            "::1",  # IPv6 loopback -- falls in the kept ::/96 range
        ],
    )
    def test_never_valid_ip_is_blocked(self, host):
        registry = {"bad": _endpoint(host=host)}
        with pytest.raises(PluginInvokeError) as exc_info:
            resolve_endpoint("bad", registry)
        assert exc_info.value.reason == "blocked_target"

    @pytest.mark.parametrize(
        "host",
        [
            "172.16.0.8",  # whisperx -- an example plugin address in a private range
            "172.16.8.5",  # marker -- an example plugin address in a private range
            "10.9.1.9",  # RFC1918 private range
            "192.168.1.20",  # RFC1918
            "100.64.0.5",  # CGNAT
        ],
    )
    def test_operator_internal_ip_is_allowed(self, host):
        """Positive control: addresses that collided with the denyset must NOT be blocked."""
        registry = {"ok": _endpoint(host=host)}
        endpoint = resolve_endpoint("ok", registry)
        assert endpoint.host == host

    def test_hostname_is_passed_through_without_dns_resolution(self):
        registry = {"ok": _endpoint(host="whisperx.internal")}
        endpoint = resolve_endpoint("ok", registry)
        assert endpoint.host == "whisperx.internal"


class TestResolveEndpointBracketedIpv6:
    """A bracket-wrapped IPv6 literal ("[::1]") must be checked identically to its
    unbracketed form -- ipaddress.ip_address() rejects the brackets outright, so
    without _parse_ip_literal's unwrap step this would silently be treated as an
    opaque hostname and skip the SSRF check entirely."""

    def test_bracketed_never_valid_ip_is_still_blocked(self):
        registry = {"bad": _endpoint(host="[64:ff9b::a9fe:a9fe]")}  # NAT64-encoded IMDS
        with pytest.raises(PluginInvokeError) as exc_info:
            resolve_endpoint("bad", registry)
        assert exc_info.value.reason == "blocked_target"

    def test_bracketed_legitimate_ip_is_still_allowed(self):
        registry = {"ok": _endpoint(host="[fc00::1]")}  # IPv6 unique-local -- ULA, kept allowed
        endpoint = resolve_endpoint("ok", registry)
        assert endpoint.host == "[fc00::1]"


class TestResolveEndpointIpv4MappedUnwrap:
    """::ffff:a.b.c.d is IPv4 a.b.c.d in IPv6 notation -- must be evaluated against
    the IPv4 policy, not blanket-blocked (that would reject the exact 172.16.0.8
    address, a real and legitimate plugin address, just spelled differently)."""

    def test_ipv4_mapped_legitimate_address_is_allowed(self):
        registry = {"ok": _endpoint(host="::ffff:172.16.0.8")}
        endpoint = resolve_endpoint("ok", registry)
        assert endpoint.host == "::ffff:172.16.0.8"

    def test_ipv4_mapped_never_valid_address_is_still_blocked(self):
        registry = {"bad": _endpoint(host="::ffff:169.254.169.254")}  # IMDS via IPv4-mapped
        with pytest.raises(PluginInvokeError) as exc_info:
            resolve_endpoint("bad", registry)
        assert exc_info.value.reason == "blocked_target"


class TestResolveEndpointIpv6LinkLocal:
    """fe80::/10 is the IPv6 analog of the 169.254.0.0/16 IMDS exception
    named explicitly -- same rationale, same asymmetry would be a real gap if
    only the IPv4 side were covered."""

    def test_ipv6_link_local_is_blocked(self):
        registry = {"bad": _endpoint(host="fe80::1")}
        with pytest.raises(PluginInvokeError) as exc_info:
            resolve_endpoint("bad", registry)
        assert exc_info.value.reason == "blocked_target"


class TestParseIpLiteral:
    def test_strips_matching_brackets(self):
        assert str(_parse_ip_literal("[::1]")) == "::1"

    def test_passes_through_unbracketed(self):
        assert str(_parse_ip_literal("172.16.0.8")) == "172.16.0.8"

    def test_hostname_returns_none(self):
        assert _parse_ip_literal("whisperx.internal") is None


class TestValidatePathNonPrintable:
    """These pass the leading-'/' and no-'..' checks but would otherwise reach
    URL construction and crash with an uncaught httpx.InvalidURL."""

    @pytest.mark.parametrize("path", ["/foo\tbar", "/foo\nbar", "/foo\x00bar"])
    def test_control_character_path_is_blocked(self, path):
        with pytest.raises(PluginInvokeError) as exc_info:
            _validate_path(path)
        assert exc_info.value.reason == "blocked_target"


class TestIsNeverValidIpMustFailControl:
    """★ Must-fail control: break the check, prove exactly the blocked_target
    tests go red, then restore."""

    def test_never_valid_helper_actually_discriminates(self, monkeypatch):
        import ipaddress

        real = _is_never_valid_ip
        try:
            monkeypatch.setattr(
                "turbohaul.plugin_invoke._is_never_valid_ip", lambda ip: False
            )
            # With the check neutered, a multicast address must now be let through.
            registry = {"bad": _endpoint(host="224.0.0.5")}
            from turbohaul.plugin_invoke import resolve_endpoint as re_after

            endpoint = re_after("bad", registry)
            assert endpoint.host == "224.0.0.5"
        finally:
            pass
        # Restored automatically by monkeypatch fixture teardown.
        assert _is_never_valid_ip(ipaddress.ip_address("224.0.0.5")) is True


# === invoke_plugin: path validation =========================================


class TestInvokePluginPathValidation:
    @pytest.mark.asyncio
    async def test_path_without_leading_slash_is_blocked(self):
        registry = {"whisperx": _endpoint()}
        with pytest.raises(PluginInvokeError) as exc_info:
            await invoke_plugin(
                resource_key="whisperx", registry=registry, path="transcribe", payload={}
            )
        assert exc_info.value.reason == "blocked_target"

    @pytest.mark.asyncio
    async def test_path_with_traversal_is_blocked(self):
        registry = {"whisperx": _endpoint()}
        with pytest.raises(PluginInvokeError) as exc_info:
            await invoke_plugin(
                resource_key="whisperx",
                registry=registry,
                path="/../etc/passwd",
                payload={},
            )
        assert exc_info.value.reason == "blocked_target"

    @pytest.mark.asyncio
    async def test_unknown_resource_checked_before_network(self):
        with pytest.raises(PluginInvokeError) as exc_info:
            await invoke_plugin(
                resource_key="nope", registry={}, path="/transcribe", payload={}
            )
        assert exc_info.value.reason == "unknown_resource"


# === invoke_plugin: network layer (mocked) ==================================


class TestInvokePluginUnreachable:
    @pytest.mark.asyncio
    async def test_connection_refused(self):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        registry = {"whisperx": _endpoint()}
        with _mocked_client(handler):
            with pytest.raises(PluginInvokeError) as exc_info:
                await invoke_plugin(
                    resource_key="whisperx",
                    registry=registry,
                    path="/transcribe",
                    payload={"audio": "..."},
                )
        assert exc_info.value.reason == "unreachable"

    @pytest.mark.asyncio
    async def test_dns_failure(self):
        def handler(request):
            raise httpx.ConnectError(
                "[Errno -2] Name or service not known", request=request
            )

        registry = {"whisperx": _endpoint(host="does-not-exist.invalid")}
        with _mocked_client(handler):
            with pytest.raises(PluginInvokeError) as exc_info:
                await invoke_plugin(
                    resource_key="whisperx", registry=registry, path="/transcribe", payload={}
                )
        assert exc_info.value.reason == "unreachable"

    @pytest.mark.asyncio
    async def test_transport_error_mid_flight(self):
        def handler(request):
            raise httpx.ReadError("connection reset", request=request)

        registry = {"whisperx": _endpoint()}
        with _mocked_client(handler):
            with pytest.raises(PluginInvokeError) as exc_info:
                await invoke_plugin(
                    resource_key="whisperx", registry=registry, path="/transcribe", payload={}
                )
        assert exc_info.value.reason == "unreachable"


class TestInvokePluginNoProgress:
    @pytest.mark.asyncio
    async def test_read_stall_maps_to_no_progress(self):
        def handler(request):
            raise httpx.ReadTimeout("no bytes within window", request=request)

        registry = {"whisperx": _endpoint()}
        with _mocked_client(handler):
            with pytest.raises(PluginInvokeError) as exc_info:
                await invoke_plugin(
                    resource_key="whisperx",
                    registry=registry,
                    path="/transcribe",
                    payload={},
                    no_progress_timeout_s=5.0,
                )
        assert exc_info.value.reason == "no_progress"

    @pytest.mark.asyncio
    async def test_write_stall_maps_to_no_progress(self):
        def handler(request):
            raise httpx.WriteTimeout("stalled sending body", request=request)

        registry = {"whisperx": _endpoint()}
        with _mocked_client(handler):
            with pytest.raises(PluginInvokeError) as exc_info:
                await invoke_plugin(
                    resource_key="whisperx",
                    registry=registry,
                    path="/transcribe",
                    payload={"big": "x" * 1000},
                    no_progress_timeout_s=5.0,
                )
        assert exc_info.value.reason == "no_progress"


class TestInvokePluginPluginError:
    @pytest.mark.asyncio
    async def test_non_2xx_carries_status_and_body_snippet(self):
        def handler(request):
            return httpx.Response(500, text="internal plugin crash: OOM")

        registry = {"whisperx": _endpoint()}
        with _mocked_client(handler):
            with pytest.raises(PluginInvokeError) as exc_info:
                await invoke_plugin(
                    resource_key="whisperx", registry=registry, path="/transcribe", payload={}
                )
        assert exc_info.value.reason == "plugin_error"
        assert "500" in exc_info.value.detail
        assert "OOM" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_404_is_plugin_error(self):
        def handler(request):
            return httpx.Response(404, text="not found")

        registry = {"whisperx": _endpoint()}
        with _mocked_client(handler):
            with pytest.raises(PluginInvokeError) as exc_info:
                await invoke_plugin(
                    resource_key="whisperx", registry=registry, path="/transcribe", payload={}
                )
        assert exc_info.value.reason == "plugin_error"

    @pytest.mark.asyncio
    async def test_invalid_json_body_is_plugin_error(self):
        def handler(request):
            return httpx.Response(200, text="not json at all {{{")

        registry = {"whisperx": _endpoint()}
        with _mocked_client(handler):
            with pytest.raises(PluginInvokeError) as exc_info:
                await invoke_plugin(
                    resource_key="whisperx", registry=registry, path="/transcribe", payload={}
                )
        assert exc_info.value.reason == "plugin_error"


class TestInvokePluginIpv6HostBracketing:
    @pytest.mark.asyncio
    async def test_bare_ipv6_host_is_bracketed_in_the_built_url(self):
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"ok": True})

        registry = {"whisperx": _endpoint(host="fc00::1", port=8080)}
        with _mocked_client(handler):
            result = await invoke_plugin(
                resource_key="whisperx", registry=registry, path="/transcribe", payload={}
            )
        assert result == {"ok": True}
        assert captured["url"].startswith("http://[fc00::1]:8080/")

    @pytest.mark.asyncio
    async def test_already_bracketed_ipv6_host_is_not_double_bracketed(self):
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"ok": True})

        registry = {"whisperx": _endpoint(host="[fc00::1]", port=8080)}
        with _mocked_client(handler):
            await invoke_plugin(
                resource_key="whisperx", registry=registry, path="/transcribe", payload={}
            )
        assert captured["url"].startswith("http://[fc00::1]:8080/")
        assert "[[" not in captured["url"]


class TestInvokePluginInvalidUrlSafetyNet:
    """The host comes from PluginEndpoint, whose own validator only rejects
    "://"/"/"/"?"/"@" -- it does not guarantee every other character is
    URL-safe. A control character in the host is a genuine gap in that
    validator (not something _validate_path can catch, since path is fine
    here) -- invoke_plugin must still fail with PluginInvokeError, not an
    uncaught httpx.InvalidURL."""

    @pytest.mark.asyncio
    async def test_control_character_in_host_is_caught_not_raised_raw(self):
        registry = {"whisperx": _endpoint(host="exa\tmple.com", port=8080)}
        with pytest.raises(PluginInvokeError) as exc_info:
            await invoke_plugin(
                resource_key="whisperx", registry=registry, path="/transcribe", payload={}
            )
        assert exc_info.value.reason == "blocked_target"


class TestInvokePluginHappyPath:
    @pytest.mark.asyncio
    async def test_returns_parsed_json_body_inline(self):
        expected = {"transcript": "hello world", "confidence": 0.97}

        def handler(request):
            assert request.method == "POST"
            assert request.url.path == "/transcribe"
            return httpx.Response(200, json=expected)

        registry = {"whisperx": _endpoint(host="172.16.0.8", port=8080)}
        with _mocked_client(handler):
            result = await invoke_plugin(
                resource_key="whisperx",
                registry=registry,
                path="/transcribe",
                payload={"audio_id": "abc123"},
            )
        assert result == expected

    @pytest.mark.asyncio
    async def test_payload_forwarded_as_json_body(self):
        sent_payload = {"audio_id": "xyz", "language": "en"}
        captured = {}

        def handler(request):
            import json as _json

            captured["body"] = _json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        registry = {"whisperx": _endpoint()}
        with _mocked_client(handler):
            await invoke_plugin(
                resource_key="whisperx",
                registry=registry,
                path="/transcribe",
                payload=sent_payload,
            )
        assert captured["body"] == sent_payload

    @pytest.mark.asyncio
    async def test_never_returns_none_on_success(self):
        def handler(request):
            return httpx.Response(200, json={})

        registry = {"whisperx": _endpoint()}
        with _mocked_client(handler):
            result = await invoke_plugin(
                resource_key="whisperx", registry=registry, path="/transcribe", payload={}
            )
        assert result is not None
        assert result == {}
