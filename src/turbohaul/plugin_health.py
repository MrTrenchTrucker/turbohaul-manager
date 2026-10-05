"""Boot + periodic health probe over the plugin registry.

`health_path` is consumed only here: its only other code references are the
field itself and its own validator in config.py, so without this module
nothing would probe it.
A registry entry in turbohaul.yaml could stay dead for days and that
silence would never produce a single log line, because nothing ever asked. That is
the gap this module closes. It does not correct a wrong address in the config, which
is a config edit, and it does not add `resolve_endpoint`
hostname support, which already exists (see plugin_invoke.py).

Contract with any future status endpoint (not built yet): this module
publishes NOTHING agent-facing. `api/plugins.py`'s `configured` field is a
registry-lookup + address-policy check that deliberately never opens a socket
(the field is named `configured`, not `resolves`, for
exactly this reason). The snapshot below is an
in-process accessor that nothing publishes today; a future status endpoint can build an
operator-facing status enum on top of it later without this module changing.

Same redaction contract `resolve_endpoint`/`invoke_plugin` observe: a
registry host:port is a server-log-only detail and must never appear in a
WARNING or anywhere else a client could ever see it (see how plugin_invoke.py
redacts it). `auth_token_file` is read fresh on every probe -- never cached,
never logged -- exactly like `invoke_plugin`.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

# A HEALTHY (or never-yet-probed) entry is re-checked at this cadence.
_BASE_INTERVAL_S = 60.0
# A repeatedly-failing entry backs off so a permanently dead endpoint does not
# turn into a probe every _BASE_INTERVAL_S forever -- capped so it can never
# drift past _MAX_INTERVAL_S even after days of failure.
_BACKOFF_FACTOR = 2.0
_MAX_INTERVAL_S = 600.0
# Policy: log on every
# transition, PLUS restate any STILL-DEAD entry at this slow cadence even
# when nothing changed. A steady-dead entry that goes permanently quiet after
# one WARNING is invisible to anyone who was not watching at the moment it
# flipped -- a fresh log window, a rotated file, an operator who joined an
# hour later, a container that booted already-broken. That silence is the
# exact bug this module exists to fix. Do NOT "optimise" this restatement away as
# redundant with the transition log -- it answers a different question
# ("what is broken right now" vs "what broke while you were watching").
_RESTATE_INTERVAL_S = 1800.0

# In practice, connect failures against a dead endpoint come back
# well under a second; this is a generous ceiling, not a tuned
# value -- a probe hanging this long already IS the "no_route" classification.
_DEFAULT_CONNECT_TIMEOUT_S = 3.0
# Outer scheduling granularity. Deliberately much finer than _BASE_INTERVAL_S
# so a newly-added registry entry (or one whose backoff just expired) is
# picked up promptly rather than waiting for the next multiple of the base
# interval.
_TICK_S = 15.0

HEALTHY = "healthy"
BAD_STATUS = "bad_status"
CONNECTION_REFUSED = "connection_refused"
NO_ROUTE = "no_route"
DNS_FAILURE = "dns_failure"
UNKNOWN = "unknown"


async def _resolve(host: str, port: int) -> None:
    """Raises socket.gaierror if `host` (a name or an IP literal) does not
    resolve. Split into its own function so a test can patch exactly this
    call site instead of DNS itself or the whole event loop."""
    await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)


async def _tcp_connect(host: str, port: int, timeout_s: float):
    """Raises ConnectionRefusedError (host answered, port did not) or
    asyncio.TimeoutError (nothing answered at all -- no route). Split into
    its own function for the same reason as `_resolve`."""
    return await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout_s)


def _host_for_url(host: str) -> str:
    """Bracket a bare IPv6 literal for URL construction, same shape check
    `invoke_plugin` uses (see plugin_invoke.py) -- kept file-local rather
    than importing that module's private `_parse_ip_literal`, matching the
    file-local convention already used by plugin_invoke.py's
    siblings (ollama.py, plugins.py)."""
    if host.startswith("[") and host.endswith("]"):
        return host
    try:
        ip = socket.inet_pton(socket.AF_INET6, host)
        del ip
    except OSError:
        return host
    return f"[{host}]"


@dataclass
class _EntryHealth:
    status: str = UNKNOWN
    detail: str = ""
    consecutive_failures: int = 0
    last_checked_monotonic: float = 0.0
    last_logged_monotonic: float = 0.0


class PluginHealthMonitor:
    """Owns per-resource_key probe state for one TurbohaulManager. Boot calls
    `probe_once()` directly (best-effort, wrapped by the caller); the
    lifespan then hands `run_periodic()` to `asyncio.create_task` the same
    way the existing live-inference monitor tasks are (see api/main.py)."""

    def __init__(
        self,
        registry_fn,
        *,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
    ) -> None:
        # registry_fn: zero-arg callable returning the CURRENT
        # dict[str, PluginEndpoint] -- asked fresh every probe rather than
        # captured once, even though PluginsConfig is frozen per boot today.
        self._registry_fn = registry_fn
        self._connect_timeout_s = connect_timeout_s
        self._state: dict[str, _EntryHealth] = {}

    def snapshot(self) -> dict[str, dict]:
        """In-process accessor. Nothing publishes this today (no
        status endpoint yet) -- it exists so a future consumer has real data to build on."""
        return {
            key: {
                "status": e.status,
                "detail": e.detail,
                "consecutive_failures": e.consecutive_failures,
            }
            for key, e in self._state.items()
        }

    async def probe_once(self) -> None:
        """Probes every registry entry whose own backoff says it is due.
        One entry's failure never stops the others in the same pass."""
        registry = self._registry_fn()
        for resource_key, endpoint in registry.items():
            try:
                await self._probe_entry(resource_key, endpoint)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "plugin health probe crashed for resource_key=%s (best-effort, "
                    "other entries are unaffected)",
                    resource_key,
                )

    async def run_periodic(self, tick_s: float = _TICK_S) -> None:
        """Runs until cancelled. Never raises out of the loop -- a probe
        exception must not cost every future tick too (mirrors the
        best-effort shape in api/main.py)."""
        while True:
            await asyncio.sleep(tick_s)
            try:
                await self.probe_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("plugin health probe tick failed (best-effort)")

    def _interval_for(self, entry: _EntryHealth) -> float:
        if entry.consecutive_failures == 0:
            return _BASE_INTERVAL_S
        interval = _BASE_INTERVAL_S * (_BACKOFF_FACTOR ** (entry.consecutive_failures - 1))
        return min(interval, _MAX_INTERVAL_S)

    async def _probe_entry(self, resource_key: str, endpoint) -> None:
        now = time.monotonic()
        prior = self._state.get(resource_key)
        if prior is not None and prior.status != UNKNOWN:
            if now - prior.last_checked_monotonic < self._interval_for(prior):
                return  # not due yet -- this entry's own backoff/base interval

        status, detail = await self._classify(endpoint)
        entry = self._state.setdefault(resource_key, _EntryHealth())
        was_status = entry.status
        entry.status = status
        entry.detail = detail
        entry.last_checked_monotonic = now
        entry.consecutive_failures = 0 if status == HEALTHY else entry.consecutive_failures + 1

        transitioned = was_status != status
        if transitioned:
            entry.last_logged_monotonic = now
            if status == HEALTHY:
                log.info(
                    "plugin endpoint recovered: resource_key=%s detail=%s",
                    resource_key, detail,
                )
            else:
                # The address itself NEVER appears here -- only resource_key
                # and the failure class, per the redaction contract this
                # module's docstring states above.
                log.warning(
                    "plugin endpoint unhealthy: resource_key=%s status=%s detail=%s",
                    resource_key, status, detail,
                )
        elif status != HEALTHY and now - entry.last_logged_monotonic >= _RESTATE_INTERVAL_S:
            # Restate a STILL-dead entry on a slow cadence
            # even though nothing changed -- see _RESTATE_INTERVAL_S above.
            entry.last_logged_monotonic = now
            log.warning(
                "plugin endpoint still unhealthy: resource_key=%s status=%s detail=%s "
                "consecutive_failures=%d",
                resource_key, status, detail, entry.consecutive_failures,
            )

    async def _classify(self, endpoint) -> tuple[str, str]:
        host, port = endpoint.host, endpoint.port

        # Pre-flight, in two real-exception-typed steps, BEFORE the real GET.
        # httpx.AsyncClient collapses both a DNS failure and a refused
        # connection into the identical httpx.ConnectError, distinguishable
        # only by a locale/libc-dependent message string ("Name or service
        # not known" vs "All connection attempts failed" -- verified
        # empirically) -- too fragile to classify three operator-actionable
        # outcomes on. socket.gaierror / ConnectionRefusedError / TimeoutError
        # are real, stable exception types instead.
        try:
            await _resolve(host, port)
        except socket.gaierror as e:
            return DNS_FAILURE, f"name did not resolve ({e})"

        try:
            _reader, writer = await _tcp_connect(host, port, self._connect_timeout_s)
        except ConnectionRefusedError as e:
            return CONNECTION_REFUSED, f"host answered, port refused ({e})"
        except (asyncio.TimeoutError, OSError) as e:
            return NO_ROUTE, f"no response within {self._connect_timeout_s}s ({e})"
        else:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

        headers: dict[str, str] = {}
        if endpoint.auth_token_file is not None:
            # Read fresh at call time, never cached, never logged -- same
            # contract as invoke_plugin (see plugin_invoke.py). A read
            # failure here degrades to an unauthenticated probe rather than
            # skipping the health check entirely: the boot-time validator
            # (in config.py) already guarantees this file was readable and
            # non-empty at startup, so a failure here is transient, and the
            # probe's job is to observe the endpoint, not the credential.
            try:
                token = endpoint.auth_token_file.read_text().strip()
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                del token
            except OSError:
                pass

        url = f"http://{_host_for_url(host)}:{port}{endpoint.health_path}"
        timeout = httpx.Timeout(
            connect=self._connect_timeout_s,
            read=self._connect_timeout_s,
            write=self._connect_timeout_s,
            pool=self._connect_timeout_s,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=headers)
        except httpx.HTTPError as e:
            # Reachable at TCP but the HTTP layer still failed (e.g. the
            # pre-flight connect raced a container that dropped the
            # connection before the request completed) -- report it as the
            # same operator-actionable class as a pre-flight no-route.
            return NO_ROUTE, f"connected but request failed ({type(e).__name__})"

        if 200 <= response.status_code < 300:
            return HEALTHY, f"HTTP {response.status_code}"
        return BAD_STATUS, f"HTTP {response.status_code}"
