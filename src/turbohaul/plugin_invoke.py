"""Plugin invocation core + SSRF guard for the media hook.

Turbohaul ships no media tooling itself -- this module is the ONLY way any code in
this project calls out to an operator-declared plugin container. A plugin is
addressed exclusively by `resource_key` into the boot-time `PluginsConfig.registry`
(turbohaul.config) -- never a URL or path from a caller, a manifest, or a request
body. api/plugins.py carries the route contract (callers never see the
registry's host:port) that this file implements.

The api/ layer calls invoke_plugin() / resolve_endpoint(); it must never
reimplement either. This file owns no FastAPI route -- pure library code + tests.
"""
import ipaddress
import json
import logging
from typing import TYPE_CHECKING

import httpx

from turbohaul.ssrf_guard import DENY_IPV4_NETWORKS, DENY_IPV6_NETWORKS

if TYPE_CHECKING:
    from turbohaul.config import PluginEndpoint


log = logging.getLogger(__name__)


# === Which of ssrf_guard.py's denyset genuinely applies here -- the argument in
# === short: ssrf_guard.py exists to stop an arbitrary
# === user-supplied pull URL from reaching this deployment's OWN internal infrastructure.
# === Here the host is the OPERATOR's own boot-time PluginsConfig.registry entry --
# === the entire point of that registry is to let the operator point at a container
# === on the same internal network. For example, one plugin container may
# === live at 172.16.0.10 and another at 172.16.8.20, both inside ssrf_guard.py's
# === 172.16.0.0/12 deny range -- blanket-applying that denyset would refuse to
# === call every plugin this feature exists for.
#
# Kept: ranges that are never a legitimate service destination for ANYONE,
# operator-trusted or not (multicast / 0.0.0.0 "this network" / reserved / the
# NAT64 bypass class / the IPv6 documentation range), PLUS two deliberate
# link-local exceptions: 169.254.0.0/16 (named explicitly) and its
# IPv6 analog fe80::/10 -- no operator legitimately registers a plugin on the
# metadata endpoint in either address family, and there is zero legitimate-use
# cost to keeping either blocked.
# Dropped: the RFC1918 (10/8, 172.16/12, 192.168/16) + CGNAT (100.64/10)
# portion -- that is exactly the address space real plugin containers live in.
# Loopback is BLOCKED in BOTH families (by design): IPv4 via
# 127.0.0.0/8 kept below; IPv6 via ::1, which falls inside the kept ::/96
# range. No plugin container legitimately serves on loopback, and both
# families reach the manager's own API (it answers on both 127.0.0.1
# and [::1]).
# NOT listed here: ::ffff:0:0/96 (IPv4-mapped IPv6). Blocking that whole range
# would incorrectly reject e.g. ::ffff:172.16.0.10 -- a legitimate operator
# address (the 172.16.0.10 example above) just spelled in IPv4-mapped
# notation. _is_never_valid_ip below unwraps an IPv4-mapped address to its
# embedded IPv4 form and re-checks it against the IPv4 policy instead.
_KEPT_IPV4 = {"0.0.0.0/8", "127.0.0.0/8", "224.0.0.0/4", "240.0.0.0/4", "169.254.0.0/16"}
_KEPT_IPV6 = {"ff00::/8", "64:ff9b::/96", "::/96", "2001:db8::/32", "fe80::/10"}
_NEVER_VALID_IPV4 = [n for n in DENY_IPV4_NETWORKS if str(n) in _KEPT_IPV4]
_NEVER_VALID_IPV6 = [n for n in DENY_IPV6_NETWORKS if str(n) in _KEPT_IPV6]
assert len(_NEVER_VALID_IPV4) == len(_KEPT_IPV4), "a kept IPv4 range name drifted from ssrf_guard.py"
assert len(_NEVER_VALID_IPV6) == len(_KEPT_IPV6), "a kept IPv6 range name drifted from ssrf_guard.py"

# TCP connect phase only -- deliberately independent of caller-supplied
# no_progress_timeout_s. An internal-network connect either succeeds fast or the
# target is genuinely unreachable; an unset no_progress_timeout_s (the "no
# duration timeout" default) must not be read as "wait forever to even open the
# socket."
_CONNECT_TIMEOUT_S = 10.0


class PluginInvokeError(Exception):
    """Raised for any failure reaching or using a plugin. Carries a machine-readable reason."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def _is_never_valid_ip(ip: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped
        if mapped is not None:
            # ::ffff:a.b.c.d is IPv4 a.b.c.d spelled in IPv6 notation -- check the
            # embedded address against the IPv4 policy, not a blanket IPv6 block,
            # so e.g. ::ffff:172.16.0.10 is treated identically to bare 172.16.0.10.
            return any(mapped in net for net in _NEVER_VALID_IPV4)
        return any(ip in net for net in _NEVER_VALID_IPV6)
    return any(ip in net for net in _NEVER_VALID_IPV4)


def _parse_ip_literal(
    host: str,
) -> "ipaddress.IPv4Address | ipaddress.IPv6Address | None":
    """ipaddress.ip_address() rejects bracket-wrapped IPv6 ("[::1]") even though
    that is a valid literal in URL/host-header contexts -- strip one matching pair
    before parsing so a bracketed literal is checked exactly like its unbracketed
    form, not silently treated as an opaque hostname that skips the SSRF check."""
    unwrapped = host
    if len(host) >= 2 and host.startswith("[") and host.endswith("]"):
        unwrapped = host[1:-1]
    try:
        return ipaddress.ip_address(unwrapped)
    except ValueError:
        return None


def resolve_endpoint(resource_key: str, registry: dict) -> "PluginEndpoint":
    """Pure lookup + SSRF validation. Raises PluginInvokeError('unknown_resource'|'blocked_target').

    Deliberately does NOT resolve hostnames. It only rejects an IP-LITERAL host that
    falls in the never-a-valid-destination subset above. A hostname's own
    reachability -- including DNS failure -- is invoke_plugin's concern
    ('unreachable'): this function's own contract permits only
    unknown_resource/blocked_target, and a DNS failure has no honest fit in either,
    so resolving anything here would be reaching past what this function is
    documented to do.
    """
    endpoint = registry.get(resource_key)
    if endpoint is None:
        raise PluginInvokeError(
            "unknown_resource", f"no registry entry for resource_key {resource_key!r}"
        )
    ip = _parse_ip_literal(endpoint.host)
    if ip is None:
        return endpoint  # hostname, not an IP literal -- see docstring
    if _is_never_valid_ip(ip):
        # The blocked address itself goes to the server log only: this detail is
        # forwarded to callers verbatim (api/plugins.py's invoke route), and the
        # contract (api/plugins.py's module docstring) is that callers never see the registry's
        # host:port -- even a mis-registered one.
        log.error(
            "plugin endpoint blocked at resolve: resource_key=%s endpoint=%s:%s",
            resource_key, endpoint.host, endpoint.port,
        )
        raise PluginInvokeError(
            "blocked_target",
            f"resource_key {resource_key!r} is registered at an address that is "
            "never a valid plugin destination (multicast / reserved / link-local-IMDS / "
            "NAT64 / IPv4-mapped range; see server log for the address)",
        )
    return endpoint


def _validate_path(path: str) -> None:
    if not path.startswith("/"):
        raise PluginInvokeError("blocked_target", f"path {path!r} must start with '/'")
    if ".." in path:
        raise PluginInvokeError("blocked_target", f"path {path!r} must not contain '..'")
    if not path.isprintable():
        # Control/non-printable characters (tab, newline, NUL, ...) pass the two
        # checks above but still produce an httpx.InvalidURL when the request is
        # actually built -- reject them here, at the layer whose whole job is
        # deciding what's an acceptable path, instead of leaking an uncaught
        # transport-layer exception past invoke_plugin's documented contract.
        raise PluginInvokeError(
            "blocked_target", f"path {path!r} contains a non-printable character"
        )


async def invoke_plugin(
    *,
    resource_key: str,
    registry: "dict[str, PluginEndpoint]",
    path: str,
    payload: dict,
    no_progress_timeout_s: float | None = None,
) -> dict:
    """Resolve resource_key -> endpoint, POST payload, return the parsed JSON body INLINE.
    Raises PluginInvokeError on every failure path. NEVER returns a partial or a None."""
    endpoint = resolve_endpoint(resource_key, registry)
    _validate_path(path)

    # Optional bearer auth. Read fresh on every call -- never cached
    # from boot, never imported from turbohaul.config (that would force a real
    # runtime import cycle the two modules deliberately avoid; see the
    # TYPE_CHECKING-only PluginEndpoint reference above). `token` is scoped to
    # this block and is NEVER passed to log.error/log.warning/an f-string that
    # reaches a log call, and NEVER included in a PluginInvokeError.detail --
    # both of those are forwarded to the caller (api/plugins.py) or written to
    # the server log, either of which would reintroduce the same leak class that
    # resolve_endpoint avoids for a host:port disclosure on this same forwarding
    # path. Only the PATH (never content) may appear in an error/log.
    headers: dict[str, str] = {}
    if endpoint.auth_token_file is not None:
        try:
            token = endpoint.auth_token_file.read_text().strip()
        except OSError as e:
            log.error(
                "plugin invoke auth token file unreadable: resource_key=%s "
                "auth_token_file=%s error=%s",
                resource_key, endpoint.auth_token_file, e,
            )
            raise PluginInvokeError(
                "plugin_error",
                f"could not read the auth token file for plugin {resource_key!r} "
                "(see server log for the path)",
            ) from e
        if not token:
            log.error(
                "plugin invoke auth token file empty: resource_key=%s auth_token_file=%s",
                resource_key, endpoint.auth_token_file,
            )
            raise PluginInvokeError(
                "plugin_error",
                f"the auth token file for plugin {resource_key!r} is empty "
                "(see server log for the path)",
            )
        headers["Authorization"] = f"Bearer {token}"
        del token  # not needed past header construction; narrows its live range

    # Bracket an IPv6 literal host for URL construction -- httpx.Request requires
    # "[::1]", not the bare "::1" a PluginEndpoint may legitimately carry (its own
    # validator only rejects "://"/"/"/"?"/"@", not bracket-less IPv6 shape).
    # _parse_ip_literal also strips a PRE-EXISTING bracket pair first, so a host
    # already written bracketed is never double-wrapped.
    host_ip = _parse_ip_literal(endpoint.host)
    host_for_url = (
        f"[{host_ip}]" if isinstance(host_ip, ipaddress.IPv6Address) else endpoint.host
    )
    url = f"http://{host_for_url}:{endpoint.port}{path}"
    # read/write ARE per-operation-inactivity timeouts in httpx (time since the
    # last byte sent/received), not a total-duration cap -- this is genuinely the
    # no-progress mechanism required here, not a repurposed duration
    # timeout. connect/pool stay on the fixed, parameter-independent floor above.
    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT_S,
        read=no_progress_timeout_s,
        write=no_progress_timeout_s,
        pool=_CONNECT_TIMEOUT_S,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
        # Full diagnostic (endpoint address + exception) goes to the server log
        # only -- this detail is forwarded to the caller verbatim, and the
        # contract (api/plugins.py's module docstring) is that callers never see the endpoint's
        # internal host:port.
        log.error(
            "plugin invoke unreachable: resource_key=%s endpoint=%s:%s error=%s",
            resource_key, endpoint.host, endpoint.port, e,
        )
        raise PluginInvokeError(
            "unreachable",
            f"could not reach plugin {resource_key!r} (see server log for the address)",
        ) from e
    except (httpx.ReadTimeout, httpx.WriteTimeout) as e:
        log.error(
            "plugin invoke no progress: resource_key=%s endpoint=%s:%s error=%s",
            resource_key, endpoint.host, endpoint.port, e,
        )
        raise PluginInvokeError(
            "no_progress",
            f"plugin {resource_key!r} produced no activity within {no_progress_timeout_s}s",
        ) from e
    except httpx.InvalidURL as e:
        # Safety net: _validate_path rejects the known-bad path shapes, but the
        # HOST also comes from PluginEndpoint, whose own validator only rejects
        # "://"/"/"/"?"/"@" -- it does not guarantee every other character is
        # URL-safe. Anything that still slips through and breaks URL construction
        # must not leak past invoke_plugin's own "raises PluginInvokeError on
        # every failure path" guarantee.
        log.error(
            "plugin invoke invalid URL: resource_key=%s endpoint=%s:%s error=%s",
            resource_key, endpoint.host, endpoint.port, e,
        )
        raise PluginInvokeError(
            "blocked_target",
            f"could not build a request URL for plugin {resource_key!r} "
            "(see server log for details)",
        ) from e
    except httpx.TransportError as e:
        # ReadError/WriteError/CloseError etc -- the connection dropped mid-flight,
        # which is a disruption, not a silence. Closer to "could not reach it" than
        # to "reached it but nothing came back," so unreachable, not no_progress.
        log.error(
            "plugin invoke transport error: resource_key=%s endpoint=%s:%s error=%s",
            resource_key, endpoint.host, endpoint.port, e,
        )
        raise PluginInvokeError(
            "unreachable",
            f"transport error talking to plugin {resource_key!r} (see server log)",
        ) from e

    if not (200 <= response.status_code < 300):
        raise PluginInvokeError(
            "plugin_error",
            f"{resource_key!r} returned HTTP {response.status_code}: {response.text[:500]!r}",
        )

    try:
        return response.json()
    except json.JSONDecodeError as e:
        raise PluginInvokeError(
            "plugin_error", f"{resource_key!r} returned invalid JSON: {e}"
        ) from e
