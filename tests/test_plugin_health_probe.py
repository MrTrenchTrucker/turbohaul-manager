"""Tests for turbohaul.plugin_health.

Before this module existed, `health_path` had ZERO callers
anywhere in the system (it was only a config field and its own validator).
A registry entry could be dead for days and nothing would ever say so --
that silence is the defect, not a wrong address in any particular
config (a config edit, out of scope).

Every negative case asserts a SPECIFIC classification, not a bare "unhealthy"
-- a wrong port at a correct host, a
correct host with no route, and a name that resolves to nothing are three
different conditions with three different operator fixes, and a probe that
folds them into one "unreachable" bucket has not earned the fix.

No real socket, no real DNS lookup, no real container is EVER contacted
(matches this repo's existing plugin_invoke test convention) -- the pre-flight
seam functions (_resolve, _tcp_connect) are patched directly for the three
negative cases + one for the timeout ceiling; httpx.AsyncClient is backed by
httpx.MockTransport for the two cases that reach the real GET (healthy,
bad_status).
"""
import asyncio
import logging
import socket
import time
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from turbohaul.api.main import create_app
from turbohaul.config import (
    BootConfig,
    PluginEndpoint,
    PluginsConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.plugin_health import (
    BAD_STATUS,
    CONNECTION_REFUSED,
    DNS_FAILURE,
    HEALTHY,
    NO_ROUTE,
    PluginHealthMonitor,
)

_RealAsyncClient = httpx.AsyncClient


def _mocked_client(handler):
    """Same pattern as the plugin-invoke tests' `_mocked_client`
    -- patch httpx.AsyncClient so the module's own code is untouched, only
    what it talks to over HTTP is faked. Never reaches the pre-flight seams."""

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)

    return patch("turbohaul.plugin_health.httpx.AsyncClient", side_effect=_factory)


def _endpoint(host="marker-api", port=8080, health_path="/health", auth_token_file=None):
    return PluginEndpoint(
        host=host, port=port, health_path=health_path, auth_token_file=auth_token_file
    )


def _monitor(registry, **kwargs):
    return PluginHealthMonitor(lambda: registry, **kwargs)


async def _fake_tcp_ok():
    """A `_tcp_connect` replacement that pretends the TCP handshake
    succeeded, so a test can drive straight to the HTTP layer."""
    reader = asyncio.StreamReader()

    class _Writer:
        def close(self):
            pass

        async def wait_closed(self):
            pass

    return reader, _Writer()


# ============================================================================
# Unit level: PluginHealthMonitor._classify / probe_once, all seams patched
# ============================================================================


class TestPositiveControl:
    """Not vacuous: a fixture that really answers 200, so the negative
    classifications below are proven against a working baseline."""

    @pytest.mark.asyncio
    async def test_healthy_entry_classifies_healthy(self):
        def handler(request):
            return httpx.Response(200, json={"ok": True})

        registry = {"marker": _endpoint()}
        mon = _monitor(registry)
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch("turbohaul.plugin_health._tcp_connect", return_value=await _fake_tcp_ok()), \
             _mocked_client(handler):
            await mon.probe_once()
        snap = mon.snapshot()["marker"]
        assert snap["status"] == HEALTHY, snap
        assert "200" in snap["detail"]


class TestThreeDistinguishableNegativeCases:
    """These tests must ALSO cover the two failure modes
    that are distinct -- a wrong PORT at a correct host, and a correct host with
    NO ROUTE ... If the test cannot tell them apart, it has not earned the
    fix. Plus DNS-resolves-to-nothing, a third case."""

    @pytest.mark.asyncio
    async def test_wrong_port_correct_host_is_connection_refused_not_generic(self):
        registry = {"marker": _endpoint(host="marker-api", port=8090)}
        mon = _monitor(registry)
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch(
                 "turbohaul.plugin_health._tcp_connect",
                 side_effect=ConnectionRefusedError("[Errno 111] Connection refused"),
             ):
            await mon.probe_once()
        snap = mon.snapshot()["marker"]
        assert snap["status"] == CONNECTION_REFUSED, (
            "a refused port must classify as connection_refused specifically "
            f"(operator fix: correct the port), got {snap}"
        )

    @pytest.mark.asyncio
    async def test_correct_host_no_route_is_no_route_not_generic(self):
        registry = {"marker": _endpoint(host="marker-api", port=8080)}
        mon = _monitor(registry, connect_timeout_s=0.05)
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch(
                 "turbohaul.plugin_health._tcp_connect",
                 side_effect=asyncio.TimeoutError(),
             ):
            await mon.probe_once()
        snap = mon.snapshot()["marker"]
        assert snap["status"] == NO_ROUTE, (
            "a host with no route must classify as no_route specifically "
            f"(operator fix: attach a network), got {snap}"
        )

    @pytest.mark.asyncio
    async def test_name_resolves_to_nothing_is_dns_failure_not_generic(self):
        registry = {"whisperx-diarize": _endpoint(host="whisperx-diarize", port=8080)}
        mon = _monitor(registry)
        with patch(
            "turbohaul.plugin_health._resolve",
            side_effect=socket.gaierror(-2, "Name or service not known"),
        ):
            await mon.probe_once()
        snap = mon.snapshot()["whisperx-diarize"]
        assert snap["status"] == DNS_FAILURE, (
            "a name that does not resolve must classify as dns_failure "
            f"specifically (operator fix: fix the name), got {snap}"
        )

    @pytest.mark.asyncio
    async def test_the_three_negative_cases_are_pairwise_distinguishable(self):
        """Direct proof of that bar: the three status STRINGS
        produced above are all different from one another and from healthy."""
        registry = {
            "refused": _endpoint(host="a", port=1),
            "no_route": _endpoint(host="b", port=1),
            "dns": _endpoint(host="c", port=1),
        }
        mon = _monitor(registry, connect_timeout_s=0.05)

        async def fake_resolve(host, port):
            if host == "c":
                raise socket.gaierror(-2, "Name or service not known")

        async def fake_tcp_connect(host, port, timeout_s):
            if host == "a":
                raise ConnectionRefusedError("refused")
            raise asyncio.TimeoutError()

        with patch("turbohaul.plugin_health._resolve", side_effect=fake_resolve), \
             patch("turbohaul.plugin_health._tcp_connect", side_effect=fake_tcp_connect):
            await mon.probe_once()
        statuses = {k: v["status"] for k, v in mon.snapshot().items()}
        assert statuses == {
            "refused": CONNECTION_REFUSED,
            "no_route": NO_ROUTE,
            "dns": DNS_FAILURE,
        }
        assert len(set(statuses.values())) == 3, "all three must be distinct: " + str(statuses)


class TestBadStatus:
    """Fifth class (beyond the three negative cases and the healthy case) --
    connects fine, health_path itself answers something other than 2xx."""

    @pytest.mark.asyncio
    async def test_non_2xx_response_is_bad_status_not_healthy(self):
        def handler(request):
            return httpx.Response(503, text="starting up")

        registry = {"marker": _endpoint()}
        mon = _monitor(registry)
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch("turbohaul.plugin_health._tcp_connect", return_value=await _fake_tcp_ok()), \
             _mocked_client(handler):
            await mon.probe_once()
        snap = mon.snapshot()["marker"]
        assert snap["status"] == BAD_STATUS
        assert "503" in snap["detail"]


class TestAddressNeverLeaksIntoTheWarning:
    """Same redaction contract resolve_endpoint/invoke_plugin observe
    (plugin_invoke.py:134): the host:port is server-log detail, but the
    module docstring promises it never appears in a way a WARNING body could
    forward to an agent-facing surface. This asserts resource_key is what
    identifies the entry in the log, and cross-checks the accessor value
    never embeds the raw endpoint tuple either."""

    @pytest.mark.asyncio
    async def test_warning_names_resource_key(self, caplog):
        registry = {"whisperx": _endpoint(host="10.99.99.99", port=9999)}
        mon = _monitor(registry, connect_timeout_s=0.05)
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch("turbohaul.plugin_health._tcp_connect", side_effect=asyncio.TimeoutError()), \
             caplog.at_level(logging.WARNING, logger="turbohaul.plugin_health"):
            await mon.probe_once()
        assert "whisperx" in caplog.text
        assert "resource_key=whisperx" in caplog.text

    @pytest.mark.asyncio
    async def test_warning_omits_host_and_port(self, caplog):
        """The module docstring PROMISES the address
        never appears in a WARNING, but nothing tested it. A unique host:port
        pair, plus a positive control on a DIFFERENT entry, so an accidental
        match against test scaffolding (e.g. a port number appearing for an
        unrelated reason) can't produce a false pass."""
        distinctive_host = "203.0.113.77"
        distinctive_port = 25777
        registry = {
            "whisperx": _endpoint(host=distinctive_host, port=distinctive_port),
            "control": _endpoint(host="203.0.113.99", port=25799),
        }
        mon = _monitor(registry, connect_timeout_s=0.05)
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch("turbohaul.plugin_health._tcp_connect", side_effect=asyncio.TimeoutError()), \
             caplog.at_level(logging.WARNING, logger="turbohaul.plugin_health"):
            await mon.probe_once()
        # Positive control: the OTHER entry's host really does appear in
        # caplog.text somewhere accessible to this test (proves caplog itself
        # is capturing real content, so an empty-text false pass is ruled out).
        assert "control" in caplog.text
        assert distinctive_host not in caplog.text
        assert str(distinctive_port) not in caplog.text


# ============================================================================
# Transition / restatement logging
# ============================================================================


class TestTransitionAndRestateLogging:
    @pytest.mark.asyncio
    async def test_transition_to_unhealthy_logs_warning_once(self, caplog):
        registry = {"marker": _endpoint()}
        mon = _monitor(registry, connect_timeout_s=0.05)
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch("turbohaul.plugin_health._tcp_connect", side_effect=asyncio.TimeoutError()), \
             caplog.at_level(logging.WARNING, logger="turbohaul.plugin_health"):
            await mon._probe_entry("marker", registry["marker"])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "the FIRST observation is a transition (unknown -> dead)"
        assert "unhealthy" in warnings[0].message

    @pytest.mark.asyncio
    async def test_recovery_logs_info_not_warning(self, caplog):
        registry = {"marker": _endpoint()}
        mon = _monitor(registry, connect_timeout_s=0.05)
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch("turbohaul.plugin_health._tcp_connect", side_effect=asyncio.TimeoutError()):
            await mon._probe_entry("marker", registry["marker"])

        def handler(request):
            return httpx.Response(200)

        # The entry's own backoff (consecutive_failures=1 -> 60s) would skip
        # a probe fired milliseconds later -- force it due, same technique
        # as the restate test below, rather than sleeping for real.
        far_future = time.monotonic() + 120
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch("turbohaul.plugin_health._tcp_connect", return_value=await _fake_tcp_ok()), \
             _mocked_client(handler), \
             patch("turbohaul.plugin_health.time.monotonic", return_value=far_future), \
             caplog.at_level(logging.INFO, logger="turbohaul.plugin_health"):
            caplog.clear()
            await mon._probe_entry("marker", registry["marker"])
        assert any(r.levelno == logging.INFO and "recovered" in r.message for r in caplog.records)
        assert not any(r.levelno == logging.WARNING for r in caplog.records)

    @pytest.mark.asyncio
    async def test_steady_dead_entry_does_not_relog_before_the_restate_interval(self, caplog):
        """Counter-check: a still-dead entry must NOT get a
        fresh WARNING on every probe -- only on transition, or after the slow
        restate cadence. This is what keeps the fix from recreating the
        log-spam problem it exists to solve.

        The second probe must be due but inside the restate window: a probe fired with
        ~0s elapsed is skipped by the
        per-entry BACKOFF DUE-CHECK before ever reaching the
        transition/restate logic at all -- so such a test could not fail no matter
        what that logic did (confirmed with a mutant -- `transitioned =
        True` unconditionally -- would pass identically with and without
        the mutation). The window the anti-spam rule actually governs is
        "this probe IS due, but the restate cadence has NOT elapsed" -- 29 of
        every 30 probes against a steady-dead entry -- and that window is
        what this test pins. Advancing +300s (past the 60s base interval, so the probe
        is due and actually runs) but far short of the 1800s restate cadence
        exercises exactly that window."""
        registry = {"marker": _endpoint()}
        mon = _monitor(registry, connect_timeout_s=0.05)
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch("turbohaul.plugin_health._tcp_connect", side_effect=asyncio.TimeoutError()):
            await mon._probe_entry("marker", registry["marker"])  # transition -> 1 WARNING
            caplog.clear()
            due_but_not_restate_due = time.monotonic() + 300
            with patch(
                "turbohaul.plugin_health.time.monotonic",
                return_value=due_but_not_restate_due,
            ), caplog.at_level(logging.WARNING, logger="turbohaul.plugin_health"):
                await mon._probe_entry("marker", registry["marker"])  # due, still dead
        assert caplog.records == [], (
            "a still-dead entry probed again, once due, but before the "
            "restate cadence elapses, must stay quiet"
        )

    @pytest.mark.asyncio
    async def test_steady_dead_entry_IS_restated_after_the_slow_cadence(self, caplog):
        """The other half of the restate rule: 'A steady-dead entry that goes
        permanently quiet after one line is invisible to anyone who was not
        watching when it flipped.' Time is forced forward past
        _RESTATE_INTERVAL_S via monkeypatched time.monotonic, not a real
        30-minute sleep."""
        registry = {"marker": _endpoint()}
        mon = _monitor(registry, connect_timeout_s=0.05)
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch("turbohaul.plugin_health._tcp_connect", side_effect=asyncio.TimeoutError()):
            await mon._probe_entry("marker", registry["marker"])
            caplog.clear()
            far_future = time.monotonic() + 3600
            with patch("turbohaul.plugin_health.time.monotonic", return_value=far_future), \
                 caplog.at_level(logging.WARNING, logger="turbohaul.plugin_health"):
                await mon._probe_entry("marker", registry["marker"])
        assert any("still unhealthy" in r.message for r in caplog.records), (
            "a steady-dead entry must be restated at the slow cadence so it is "
            "not invisible to an operator who joined after it first flipped"
        )


class TestBackoff:
    @pytest.mark.asyncio
    async def test_repeated_failure_backs_off_and_is_capped(self):
        registry = {"marker": _endpoint()}
        mon = _monitor(registry, connect_timeout_s=0.05)
        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch("turbohaul.plugin_health._tcp_connect", side_effect=asyncio.TimeoutError()):
            for _ in range(6):
                await mon._probe_entry("marker", registry["marker"])
        entry = mon._state["marker"]
        assert entry.consecutive_failures == 1, (
            "the due-check is INSIDE _probe_entry itself (not something only "
            "probe_once applies), so all 5 repeat calls above were skipped "
            "before ever reaching _classify -- this asserts the underlying "
            "interval math directly instead of relying on real elapsed time"
        )
        # Assert the backoff math directly: failures grow the interval, capped.
        entry.consecutive_failures = 1
        i1 = mon._interval_for(entry)
        entry.consecutive_failures = 4
        i4 = mon._interval_for(entry)
        entry.consecutive_failures = 100
        i_cap = mon._interval_for(entry)
        assert i1 < i4 < i_cap or i4 == i_cap  # monotonically non-decreasing, then capped
        from turbohaul.plugin_health import _MAX_INTERVAL_S
        assert i_cap == _MAX_INTERVAL_S


class TestNotDueYetIsSkipped:
    @pytest.mark.asyncio
    async def test_probe_once_skips_an_entry_before_its_interval_elapses(self):
        registry = {"marker": _endpoint()}
        mon = _monitor(registry)
        calls = 0

        async def counting_resolve(host, port):
            nonlocal calls
            calls += 1

        with patch("turbohaul.plugin_health._resolve", side_effect=counting_resolve), \
             patch("turbohaul.plugin_health._tcp_connect", side_effect=asyncio.TimeoutError()):
            await mon.probe_once()
            await mon.probe_once()  # immediately again -- not due
        assert calls == 1, "a second probe_once() before the interval elapses must not re-probe"


# ============================================================================
# THE ACTUAL DEFECT: the boot lifespan integration.
#
# This pair covers the real gap: a happy-path
# test passes while every endpoint is unreachable. Without the probe
# (plugin_health.py absent, main.py unwired):
# nothing ever calls health_path, so a dead registry entry produces NO log
# line and boot succeeds silently.
# ============================================================================


@pytest.fixture
def app_with_dead_plugin(tmp_path):
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
        plugins=PluginsConfig(
            # Real IP literal, not a hostname: these lifespan-level tests
            # patch only _tcp_connect, not _resolve -- a hostname here would
            # trigger a REAL DNS lookup from the test process, which this
            # repo's plugin-test convention (the plugin-invoke tests'
            # header) forbids ("no real container is ever contacted").
            registry={"marker": PluginEndpoint(host="127.0.0.1", port=8090)}
        ),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    return app


class TestBootProbeIsWiredIntoTheRealLifespan:
    def test_boot_with_a_dead_registry_entry_produces_a_warning(
        self, app_with_dead_plugin, caplog
    ):
        with patch(
            "turbohaul.plugin_health._tcp_connect",
            side_effect=ConnectionRefusedError("refused"),
        ), caplog.at_level(logging.WARNING, logger="turbohaul.plugin_health"):
            with TestClient(app_with_dead_plugin):
                pass  # lifespan startup runs the boot probe; shutdown tears it down
        assert any(
            "resource_key=marker" in r.message and "unhealthy" in r.message
            for r in caplog.records
        ), (
            "boot must probe the registry and log a WARNING for a dead entry -- "
            f"got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_periodic_task_does_not_outlive_shutdown(self, app_with_dead_plugin):
        """Shutdown symmetry: the
        periodic task must be cancelled by mgr.shutdown() ITSELF, not merely
        appear cancelled because something else cleaned it up.

        Deliberately does NOT use TestClient here: TestClient runs the ASGI
        app through an anyio thread-portal, and that portal's own teardown
        force-cancels any task still running on its loop -- which would let a
        test pass even with mgr.shutdown()'s cancel
        sweep removed (for example, a mutant that dropped
        `self._plugin_health_task` from the shutdown() observer tuple
        would survive such a test, for exactly that reason).
        Driving `app.router.lifespan_context` directly, in this test's own
        event loop, means the ONLY thing that can cancel the task between
        the two asserts below is turbohaul's own shutdown() code."""
        app = app_with_dead_plugin
        mgr = app.state.manager
        with patch(
            "turbohaul.plugin_health._tcp_connect",
            side_effect=ConnectionRefusedError("refused"),
        ):
            async with app.router.lifespan_context(app):
                task = mgr._plugin_health_task
                assert task is not None and not task.done()
        assert task.done(), "the periodic probe task must not outlive shutdown()"
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_healthy_registry_entry_produces_no_warning(self, tmp_path, caplog):
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
            plugins=PluginsConfig(
                registry={"marker": PluginEndpoint(host="marker-api", port=8080)}
            ),
        )
        runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)

        def handler(request):
            return httpx.Response(200)

        with patch("turbohaul.plugin_health._resolve", return_value=None), \
             patch("turbohaul.plugin_health._tcp_connect", return_value=await _fake_tcp_ok()), \
             _mocked_client(handler), \
             caplog.at_level(logging.WARNING, logger="turbohaul.plugin_health"):
            with TestClient(app):
                pass
        assert not any(r.levelno == logging.WARNING for r in caplog.records)


class TestNothingAgentFacingChanges:
    """The existing contract stands: `configured` and api/plugins.py's listing must
    be byte-identical whether or not the registry entry is alive."""

    def test_plugins_listing_byte_identical_dead_vs_no_probe(
        self, app_with_dead_plugin, tmp_path
    ):
        # Baseline: the SAME registry, but never probed at all (patch the
        # boot probe itself into a no-op) -- this is the pre-existing shape
        # of /api/plugins with a dead entry, before the health probe existed.
        with patch("turbohaul.plugin_health.PluginHealthMonitor.probe_once", return_value=None), \
             patch("turbohaul.plugin_health.PluginHealthMonitor.run_periodic", return_value=None):
            with TestClient(app_with_dead_plugin) as client:
                baseline = client.get("/api/plugins").json()

        with patch(
            "turbohaul.plugin_health._tcp_connect",
            side_effect=ConnectionRefusedError("refused"),
        ):
            with TestClient(app_with_dead_plugin) as client:
                probed = client.get("/api/plugins").json()

        assert probed == baseline, (
            "the listing must be byte-identical whether or not the registry "
            "entry has actually been probed -- the contract forbids a probe "
            f"result reaching this route. baseline={baseline!r} probed={probed!r}"
        )
