"""Regression tests: client-facing PluginInvokeError.detail must NEVER contain
the registry endpoint's internal host:port.

The contract (see the api/plugins.py module docstring) is that callers never see the registry's
host:port -- the invoke route forwards e.detail to the client VERBATIM, so the
detail itself must be host:port-free. The full diagnostic (address + exception)
goes to the server log instead.
"""
import logging
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from turbohaul.config import PluginEndpoint
from turbohaul.plugin_invoke import (
    PluginInvokeError,
    invoke_plugin,
    resolve_endpoint,
)

_RealAsyncClient = httpx.AsyncClient


def _mocked_client(handler):
    """Patch httpx.AsyncClient so invoke_plugin's internal client is backed by a
    MockTransport -- invoke_plugin's own code is untouched, only what it talks to."""

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)

    return patch("turbohaul.plugin_invoke.httpx.AsyncClient", side_effect=_factory)


def _endpoint(host="172.16.0.8", port=8080):
    return PluginEndpoint(host=host, port=port, health_path="/health")


class TestErrorDetailOmitsEndpointAddress:
    @pytest.mark.asyncio
    async def test_unreachable_detail_has_no_host_port(self, caplog):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        with _mocked_client(handler):
            with caplog.at_level(logging.ERROR, logger="turbohaul.plugin_invoke"):
                with pytest.raises(PluginInvokeError) as exc_info:
                    await invoke_plugin(
                        resource_key="whisperx",
                        registry={"whisperx": _endpoint()},
                        path="/transcribe",
                        payload={},
                    )
        assert exc_info.value.reason == "unreachable"
        assert "172.16.0.8" not in exc_info.value.detail
        assert "8080" not in exc_info.value.detail
        # the full diagnostic went to the server log instead
        assert "172.16.0.8:8080" in caplog.text

    @pytest.mark.asyncio
    async def test_no_progress_detail_has_no_host_port(self, caplog):
        def handler(request):
            raise httpx.ReadTimeout("stall", request=request)

        with _mocked_client(handler):
            with caplog.at_level(logging.ERROR, logger="turbohaul.plugin_invoke"):
                with pytest.raises(PluginInvokeError) as exc_info:
                    await invoke_plugin(
                        resource_key="whisperx",
                        registry={"whisperx": _endpoint()},
                        path="/transcribe",
                        payload={},
                        no_progress_timeout_s=5.0,
                    )
        assert exc_info.value.reason == "no_progress"
        assert "172.16.0.8" not in exc_info.value.detail
        assert "172.16.0.8:8080" in caplog.text

    @pytest.mark.asyncio
    async def test_transport_error_detail_has_no_host_port(self):
        def handler(request):
            raise httpx.ReadError("reset", request=request)

        with _mocked_client(handler):
            with pytest.raises(PluginInvokeError) as exc_info:
                await invoke_plugin(
                    resource_key="whisperx",
                    registry={"whisperx": _endpoint()},
                    path="/transcribe",
                    payload={},
                )
        assert exc_info.value.reason == "unreachable"
        assert "172.16.0.8" not in exc_info.value.detail

    def test_blocked_target_detail_has_no_registered_host(self, caplog):
        # a mis-registered IMDS address: the blocked host must not reach the client
        registry = {"misregistered": _endpoint(host="169.254.169.254", port=80)}
        with caplog.at_level(logging.ERROR, logger="turbohaul.plugin_invoke"):
            with pytest.raises(PluginInvokeError) as exc_info:
                resolve_endpoint("misregistered", registry)
        assert exc_info.value.reason == "blocked_target"
        assert "169.254.169.254" not in exc_info.value.detail
        assert "169.254.169.254:80" in caplog.text


class TestLoopbackTightening:
    """Gate tightening: loopback must be BLOCKED in both
    families -- 127/8 via _KEPT_IPV4, ::1 via the kept ::/96 range. Both reach
    the manager's own API (HTTP 200 observed on 127.0.0.1:11401 and
    [::1]:11401 from inside the container). The IPv4 case is the RED case:
    it FAILS on the pre-tightening tree, passes only after 127.0.0.0/8 is kept."""

    def test_ipv4_loopback_registration_rejected(self):
        registry = {"local": _endpoint(host="127.0.0.1", port=11401)}
        with pytest.raises(PluginInvokeError) as exc_info:
            resolve_endpoint("local", registry)
        assert exc_info.value.reason == "blocked_target"

    def test_ipv6_loopback_registration_rejected(self):
        registry = {"local6": _endpoint(host="::1", port=11401)}
        with pytest.raises(PluginInvokeError) as exc_info:
            resolve_endpoint("local6", registry)
        assert exc_info.value.reason == "blocked_target"


class TestShorthandIpHostRejected:
    """Shorthand IP spellings are NEITHER bare
    IPs nor hostnames -- they must be rejected at REGISTRATION (the
    PluginEndpoint contract), because ipaddress rejects them (so
    resolve_endpoint's guard never sees them) while the OS inet_aton still
    expands them to their real target: a live 127.1 registration HIT
    127.0.0.1 with status 200. The bracketed forms are the same hole class
    (bracket-strip in _parse_ip_literal)."""

    @pytest.mark.parametrize(
        "host",
        [
            "127.1",  # short form -- observed live 200
            "2130706433",  # decimal 127.0.0.1
            "0177.0.0.1",  # octal 127.0.0.1
            "0x7f.0.0.1",  # hex 127.0.0.1
            "[127.1]",  # bracketed short form
            "[2130706433]",  # bracketed decimal
        ],
    )
    def test_shorthand_loopback_host_rejected_at_registration(self, host):
        with pytest.raises(ValidationError):
            _endpoint(host=host)

    def test_real_hostnames_still_construct(self):
        # names that must keep working
        assert _endpoint(host="whisperx.internal").host == "whisperx.internal"
        assert _endpoint(host="localhost").host == "localhost"

    def test_real_ips_still_resolve_allowed(self):
        # a tightening that kills the real containers breaks the
        # feature -- the real container addresses must still resolve ALLOWED
        registry = {
            "whisperx": _endpoint(host="172.16.0.8", port=8000),
            "marker": _endpoint(host="172.16.8.5", port=8000),
        }
        assert resolve_endpoint("whisperx", registry).host == "172.16.0.8"
        assert resolve_endpoint("marker", registry).host == "172.16.8.5"
