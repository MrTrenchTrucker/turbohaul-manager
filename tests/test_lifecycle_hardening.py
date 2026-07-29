"""Controller lifecycle hardening tests (PR 1 — lifecycle-only).

Covers the orphan/stale-slot/stuck-slot recovery seam from the P12 incident
with ownership-aware reconciliation that ONLY reaps PROVEN Turbohaul-owned
stale engine processes. Ownership is proven by parent-chain proof (candidate
is reparented to init/subreaper = dead-manager orphan, OR parent chain
contains a recognisable Turbohaul manager process), NOT by cmdline alone —
``llama-server --port N`` in the managed range is necessary but not
sufficient. A foreign or independently managed ``llama-server`` in the managed
range is NEVER reaped (reported only). A foreign process listening in the
managed range (nginx, python http.server) is NEVER reaped.

  1. reap_orphan kills the whole process group (not just the leader PID).
  2. boot_reconcile reaps a no-listener orphan (cmdline-identified
     llama-server, no PPid match, no listener socket) — the real P12 seam.
     END-TO-END: a LIVE recorded P12 stale engine (pid_is_alive_fn=True,
     recorded triple in state.sqlite) IS reaped — the boot-time known_pids
     skip-list is EMPTY (flock singleton ownership), so an alive recorded
     stale engine is NOT skipped.
  3. boot_reconcile does NOT reap a foreign process listening in the managed
     range (e.g. nginx / python http.server on the same port).
  4. boot_reconcile reports stale listeners count (diagnostics, not reaping).
  5. A health-load timeout releases the slot + fails the request in bounded
     time (no 600s silent wait).
  6. Adversarial ownership proof: a FOREIGN ``llama-server --port N`` in the
     managed range whose parent chain has no Turbohaul manager is NOT reaped
     (cmdline alone does not prove ownership). A genuine Turbohaul-owned
     no-listener orphan (parent chain -> turbohaul-manager, or reparented to
     init) IS reaped — the P12 ownership route survives the hardening.
  7. Conservative ownership at boot: a foreign/unrecorded llama-server
     (including PPid=1) that is ALIVE is NOT reaped (no recorded identity ->
     report-only). The crash window between spawn and identity persist is
     fail-safe report-only/manual recovery, not a false-positive kill.
"""
import asyncio
import signal
from unittest.mock import MagicMock

import pytest

from turbohaul.singleton import reap_orphan


# ---------------------------------------------------------------------------
# shared fixture (mirrors test_manager.py's boot_and_runtime)
# ---------------------------------------------------------------------------

@pytest.fixture
def boot_and_runtime(tmp_path):
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
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59500,  # nothing on this range
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    return boot, runtime


# ---------------------------------------------------------------------------
# 1. reap_orphan process-group safety
# ---------------------------------------------------------------------------

class TestReapOrphanProcessGroup:
    def test_reap_orphan_kills_process_group_not_just_pid(self):
        """reap_orphan must killpg the whole process group, not os.kill the
        single leader PID. spawn_sidecar uses start_new_session=True so the
        engine may have grandchildren in its group; a PID-only kill leaks them.
        """
        calls = []

        def killpg_fn(pgid, sig):
            calls.append(("killpg", pgid, sig))

        def getpgid_fn(pid):
            return 4242  # the orphan's process group id

        def kill0_fn(pid, sig):
            # pid 0 existence probe; report gone once SIGKILL sent
            if sig == 0 and ("killpg", 4242, signal.SIGKILL) not in calls:
                return None  # still alive during sigterm window
            raise ProcessLookupError

        def starttime_fn(pid):
            return 100

        ok, status = reap_orphan(
            7777,
            sigterm_wait_s=0.05,
            kill_fn=kill0_fn,
            killpg_fn=killpg_fn,
            getpgid_fn=getpgid_fn,
            starttime_fn=starttime_fn,
        )
        assert ("killpg", 4242, signal.SIGTERM) in calls
        assert ok is True

    def test_reap_orphan_escalates_killpg_sigkill(self):
        """On SIGTERM timeout, escalate to killpg(SIGKILL) on the group."""
        calls = []
        alive = {"v": True}

        def killpg_fn(pgid, sig):
            calls.append(("killpg", pgid, sig))
            if sig == signal.SIGKILL:
                alive["v"] = False

        def getpgid_fn(pid):
            return 4242

        def kill0_fn(pid, sig):
            if sig == 0:
                if not alive["v"]:
                    raise ProcessLookupError
                return None
            calls.append(("kill", pid, sig))

        def starttime_fn(pid):
            return 100

        ok, status = reap_orphan(
            7777,
            sigterm_wait_s=0.05,
            kill_fn=kill0_fn,
            killpg_fn=killpg_fn,
            getpgid_fn=getpgid_fn,
            starttime_fn=starttime_fn,
        )
        assert ("killpg", 4242, signal.SIGTERM) in calls
        assert ("killpg", 4242, signal.SIGKILL) in calls
        assert ok is True

    def test_reap_orphan_pid_already_gone(self):
        """If the pid is already gone (getpgid raises ProcessLookupError),
        returns already-gone without any killpg.
        """
        calls = []

        def killpg_fn(pgid, sig):
            calls.append(("killpg", pgid, sig))

        def getpgid_fn(pid):
            raise ProcessLookupError

        def kill_fn(pid, sig):
            raise ProcessLookupError

        def starttime_fn(pid):
            return 100

        ok, status = reap_orphan(
            7777,
            sigterm_wait_s=0.05,
            kill_fn=kill_fn,
            killpg_fn=killpg_fn,
            getpgid_fn=getpgid_fn,
            starttime_fn=starttime_fn,
        )
        assert ok is True
        assert "already-gone" in status
        assert calls == []  # no killpg on an already-gone pid

    def test_reap_orphan_sigterm_clean(self):
        """If the group dies during the SIGTERM window, returns sigterm-clean."""
        calls = []

        def killpg_fn(pgid, sig):
            calls.append(("killpg", pgid, sig))

        def getpgid_fn(pid):
            return 4242

        probes = {"n": 0}

        def kill0_fn(pid, sig):
            if sig == 0:
                probes["n"] += 1
                if probes["n"] >= 2:
                    raise ProcessLookupError  # gone after one alive probe
                return None
            calls.append(("kill", pid, sig))

        def starttime_fn(pid):
            return 100

        ok, status = reap_orphan(
            7777,
            sigterm_wait_s=1.0,
            kill_fn=kill0_fn,
            killpg_fn=killpg_fn,
            getpgid_fn=getpgid_fn,
            starttime_fn=starttime_fn,
        )
        assert ok is True
        assert status == "sigterm-clean"
        assert ("killpg", 4242, signal.SIGTERM) in calls
        # No SIGKILL escalation needed
        assert all(s != signal.SIGKILL for _, _, s in calls)


# ---------------------------------------------------------------------------
# 2. boot_reconcile reaps a no-listener orphan (cmdline-identified, no PPid
#    match, no listener socket) — the real P12 seam.
# ---------------------------------------------------------------------------

class TestNoListenerOrphanReaped:
    def test_boot_reconcile_reaps_no_listener_orphan(self, boot_and_runtime, monkeypatch):
        """boot_reconcile must detect + reap a Turbohaul-owned stale engine
        process that has NO listener socket (the P12 finding: "orphaned
        llama-server ... without a listener"). Detection is ownership-aware:
        the process cmdline contains ``llama-server`` + ``--port`` in the
        managed range, regardless of PPid or listener state.

        This is a NO-LISTENER scenario: find_orphan_llama_servers (PPid-based)
        finds nothing (orphan still parented to a dying manager, PPid not in
        reapers), port_listeners_in_range (socket scan) finds nothing (no
        listener), but find_llama_servers_in_port_range (cmdline scan) finds
        the stale engine and reaps it.
        """
        boot, runtime = boot_and_runtime
        port = boot.runtime.default_port_base  # 59500
        from turbohaul.manager import TurbohaulManager
        mgr = TurbohaulManager(boot, runtime)

        # PPid-based scan finds nothing (orphan still parented to a dying
        # manager, PPid not in reapers).
        monkeypatch.setattr(
            "turbohaul.singleton.find_orphan_llama_servers",
            lambda port_base, port_range_size=100, **kw: [],
        )
        # Listener socket scan finds nothing — the orphan has NO listener.
        monkeypatch.setattr(
            "turbohaul.singleton.port_listeners_in_range",
            lambda port_base, port_range_size=100: [],
        )
        # Seed a RECORDED engine identity (pid=5555, port, starttime=7000) in
        # state.sqlite so the durable proof matches. The stale-engine scan
        # nominates pid 5555 (cmdline + port in range) AND the live starttime
        # matches the record → reaped (the P12 no-listener route via durable
        # identity, NOT parent-chain heuristic).
        from turbohaul.state import record_engine_identity, upsert_slot, open_state_db
        _c = open_state_db(boot.storage.state_db_path)
        upsert_slot(_c, {"slot_id": "s-stale", "model_tag": "m",
                         "state": "ACTIVE", "pid": 5555, "port": port})
        record_engine_identity(_c, "s-stale", 5555, port, 7000)
        _c.close()
        # Inject /proc starttime reader so the durable proof reads 7000.
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_starttime", lambda pid: 7000
        )
        # Stale-engine scan: let the REAL find_llama_servers_in_port_range run
        # with injected /proc readers — it nominates pid 5555 (cmdline +
        # --port 59500 in range) and the durable identity match retains it.
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_cmdline",
            lambda pid: (
                "llama-server --port 59500 --model /x/y.gguf" if pid == 5555 else ""
            ),
        )
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_ppid", lambda pid: 7777
        )
        monkeypatch.setattr(
            "turbohaul.singleton._list_proc_pids", lambda: [5555]
        )
        reaped = {"pids": []}

        def fake_reap_orphan(pid, **kw):
            reaped["pids"].append(pid)
            return True, "sigterm-clean"

        result = mgr.boot_reconcile(reap_orphan_fn=fake_reap_orphan)
        assert 5555 in reaped["pids"], (
            "no-listener orphan (cmdline-identified llama-server) must be reaped"
        )
        assert result["orphans_reaped"] >= 1
        # stale_listeners is a diagnostics-only count from the socket scan
        # (no listener here → 0).
        assert result["stale_listeners"] == 0


# ---------------------------------------------------------------------------
# 3. boot_reconcile does NOT reap a foreign process listening in the managed
#    range (e.g. nginx / python http.server on the same port).
# ---------------------------------------------------------------------------

class TestForeignListenerNotKilled:
    def test_foreign_listener_in_range_not_reaped(self, boot_and_runtime, monkeypatch):
        """A foreign process (e.g. nginx, python http.server) listening on a
        managed port must NOT be reaped. The ownership-aware reconciliation only
        reaps processes whose cmdline contains ``llama-server`` + ``--port`` in
        the managed range. port_listeners_in_range detects the foreign listener
        for diagnostics, but boot_orphan_reaper never reaps from it.
        """
        boot, runtime = boot_and_runtime
        port = boot.runtime.default_port_base  # 59500
        from turbohaul.manager import TurbohaulManager
        mgr = TurbohaulManager(boot, runtime)

        # PPid-based scan finds nothing.
        monkeypatch.setattr(
            "turbohaul.singleton.find_orphan_llama_servers",
            lambda port_base, port_range_size=100, **kw: [],
        )
        # Listener socket scan finds a FOREIGN process (nginx) on the managed
        # port. This is a diagnostics-only signal; it must NOT trigger reaping.
        monkeypatch.setattr(
            "turbohaul.singleton.port_listeners_in_range",
            lambda port_base, port_range_size=100: [
                {
                    "pid": 9999,
                    "port": port,
                    "cmdline": "nginx: worker process",
                },
            ],
        )
        # Stale-engine scan finds NO llama-server — the foreign process is
        # not a llama-server, so it is not nominated. No recorded identity
        # anyway → report-only.
        monkeypatch.setattr(
            "turbohaul.singleton.find_llama_servers_in_port_range",
            lambda port_base, port_range_size=100, **kw: [],
        )
        reaped = {"pids": []}

        def fake_reap_orphan(pid, **kw):
            reaped["pids"].append(pid)
            return True, "sigterm-clean"

        result = mgr.boot_reconcile(reap_orphan_fn=fake_reap_orphan)
        assert 9999 not in reaped["pids"], (
            "foreign process listening in the managed range must NOT be reaped"
        )
        assert result["orphans_reaped"] == 0
        # The foreign listener is reported in diagnostics but not reaped.
        assert result["stale_listeners"] == 1


# ---------------------------------------------------------------------------
# 4. boot_reconcile reports stale listeners count (diagnostics, not reaping)
# ---------------------------------------------------------------------------

class TestPortFreeAfterReconcile:
    def test_boot_reconcile_reports_zero_stale_listeners_on_clean_range(
        self, boot_and_runtime, monkeypatch
    ):
        """After reconciliation on a clean port range, boot_reconcile reports
        zero stale listeners so an operator can confirm the managed ports are
        free before allocation.
        """
        boot, runtime = boot_and_runtime
        from turbohaul.manager import TurbohaulManager
        mgr = TurbohaulManager(boot, runtime)

        monkeypatch.setattr(
            "turbohaul.singleton.find_orphan_llama_servers",
            lambda port_base, port_range_size=100, **kw: [],
        )
        monkeypatch.setattr(
            "turbohaul.singleton.port_listeners_in_range",
            lambda port_base, port_range_size=100: [],
        )
        monkeypatch.setattr(
            "turbohaul.singleton.find_llama_servers_in_port_range",
            lambda port_base, port_range_size=100, **kw: [],
        )
        result = mgr.boot_reconcile()
        assert result["stale_listeners"] == 0
        assert result["orphans_reaped"] == 0


# ---------------------------------------------------------------------------
# 5. health-load timeout releases the slot + fails the request in bounded time
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestHealthTimeoutBounded:
    async def test_health_timeout_releases_slot_and_fails_request_fast(self, boot_and_runtime):
        """A health-load timeout must fail the request in bounded time
        (not a 600s silent wait), release the slot, and reap the engine.

        Drives the worker_loop with an injected _wait_healthy that returns
        False quickly, a mock sidecar handle that is 'alive', and a mock
        _sigterm/_vram_verify. Asserts the completion_future is failed with
        the timeout error and _active_handle is cleared within a small budget.
        """
        from turbohaul.config import QueueConfig, RuntimeConfig, PullConfig
        from turbohaul.manager import TurbohaulManager
        from turbohaul.subprocess_mgr import SidecarHandle

        boot, runtime = boot_and_runtime
        runtime = RuntimeConfig(
            queue=QueueConfig(loading_health_timeout_s=10, max_parallel_sidecars=1),
            pull=PullConfig(),
        )
        mgr = TurbohaulManager(boot, runtime)
        # Injected health: immediately unhealthy (returns False instantly)
        async def fake_wait_healthy(port, timeout_s, **kw):
            return False
        mgr._wait_healthy = fake_wait_healthy
        # Mock handle: alive
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None  # alive
        fake_handle = SidecarHandle(proc=proc, port=boot.runtime.default_port_base, model_tag="m")
        mgr._spawn = lambda *a, **k: fake_handle
        # Mock sigterm: succeed fast
        async def fake_sigterm(handle, **kw):
            return True, "sigterm-clean"
        mgr._sigterm = fake_sigterm
        async def fake_vram(**kw):
            return True, 0
        mgr._vram_verify = fake_vram

        slot = await mgr.submit(model_tag="m", prompt="x", wait_for_completion=True)
        # Run the worker_loop briefly to process the slot
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            assert slot.completion_future is not None
            # The future is FAILED with RuntimeError("loading-fail-health-timeout"),
            # so awaiting it raises — this IS the bounded-failure contract (the
            # request fails in bounded time instead of a 600s silent wait).
            with pytest.raises(RuntimeError, match="loading-fail"):
                await asyncio.wait_for(slot.completion_future, timeout=5.0)
            assert slot.completion_future.done()
            exc = slot.completion_future.exception()
            assert exc is not None
            assert "loading-fail" in str(exc).lower() or "health" in str(exc).lower()
        finally:
            await mgr.shutdown()
        # Active handle must be cleared after the failed teardown
        assert mgr._active_handle is None


# ---------------------------------------------------------------------------
# 6. Adversarial: a FOREIGN llama-server --port in the managed range whose
#    parent is NOT a Turbohaul manager must NOT be reaped. Cmdline alone
#    (llama-server + --port) does NOT prove Turbohaul ownership; the parent
#    chain must contain a recognisable Turbohaul manager process (or the
#    process must be reparented to init/subreaper — orphan of a dead manager).
# ---------------------------------------------------------------------------

class TestForeignLlamaServerOwnershipProof:
    """DURABLE IDENTITY ownership proof (replaces the parent-chain heuristic):
    a candidate is Turbohaul-owned ONLY if its live (pid, port, starttime)
    matches a recorded identity. Cmdline alone does NOT prove ownership; a
    foreign llama-server in the managed range has no recorded identity and is
    report-only, never reaped. The parent chain (turbohaul-marker / PPid=1)
    is NOT consulted as ownership proof.
    """

    # Foreign llama-server: parent is a user shell (NOT Turbohaul). No
    # recorded identity.
    _FOREIGN_PROCS = {
        5001: {"cmdline": "llama-server --port 59500 -m /x.gguf", "ppid": 8888,
               "starttime": 5001},
        8888: {"cmdline": "bash -l", "ppid": 1, "starttime": 5000},
    }

    # Turbohaul-owned stale engine: recorded identity (pid=5002, port=59501,
    # starttime=7002). Parent chain is irrelevant to the durable proof; the
    # starttime match is the authority.
    _OWNED_STALE_PROCS = {
        5002: {"cmdline": "llama-server --port 59501 -m /y.gguf", "ppid": 7777,
               "starttime": 7002},
        7777: {"cmdline": "turbohaul-manager", "ppid": 1, "starttime": 7000},
    }

    # Turbohaul-owned orphan: recorded identity (pid=5003, port=59502,
    # starttime=7003), reparented to init (PPid=1). The record + starttime
    # match is the proof, NOT the PPid.
    _OWNED_REPARENTED_PROCS = {
        5003: {"cmdline": "llama-server --port 59502 -m /z.gguf", "ppid": 1,
               "starttime": 7003},
    }

    @staticmethod
    def _make_fakes(procs):
        def cmdline_fn(pid):
            return procs.get(pid, {}).get("cmdline", "")

        def ppid_fn(pid):
            return procs.get(pid, {}).get("ppid", None)

        def starttime_fn(pid):
            return procs.get(pid, {}).get("starttime", None)

        return cmdline_fn, ppid_fn, starttime_fn

    def test_foreign_llama_server_not_returned(self, monkeypatch):
        """find_llama_servers_in_port_range must NOT return a foreign
        llama-server — it has no recorded engine identity, so the durable
        proof fails and it is dropped (report-only). Cmdline + port is only a
        nominator.
        """
        from turbohaul.singleton import find_llama_servers_in_port_range

        monkeypatch.setattr("turbohaul.singleton._SUBREAPER_PID", None)
        cmdline_fn, ppid_fn, starttime_fn = self._make_fakes(self._FOREIGN_PROCS)
        result = find_llama_servers_in_port_range(
            59500,
            port_range_size=100,
            cmdline_fn=cmdline_fn,
            ppid_fn=ppid_fn,
            proc_pids_fn=lambda: list(self._FOREIGN_PROCS.keys()),
            starttime_fn=starttime_fn,
            known_engine_identities=set(),  # no recorded identity for foreign
        )
        assert result == [], (
            "foreign llama-server (no recorded identity) must NOT be returned "
            "by find_llama_servers_in_port_range — cmdline alone does not "
            "prove Turbohaul ownership"
        )

    def test_turbohaul_owned_stale_engine_returned(self, monkeypatch):
        """A Turbohaul-owned stale engine with a RECORDED identity IS
        returned — the live (pid, port, starttime) matches the record. The
        foreign one in the same process table has no record and is NOT
        returned, proving the durable filter is selective.
        """
        from turbohaul.singleton import find_llama_servers_in_port_range

        monkeypatch.setattr("turbohaul.singleton._SUBREAPER_PID", None)
        procs = {**self._OWNED_STALE_PROCS, **self._FOREIGN_PROCS}
        cmdline_fn, ppid_fn, starttime_fn = self._make_fakes(procs)
        result = find_llama_servers_in_port_range(
            59500,
            port_range_size=100,
            cmdline_fn=cmdline_fn,
            ppid_fn=ppid_fn,
            proc_pids_fn=lambda: list(procs.keys()),
            starttime_fn=starttime_fn,
            known_engine_identities={(5002, 59501, 7002)},
        )
        pids = {r["pid"] for r in result}
        assert 5002 in pids, (
            "Turbohaul-owned stale engine (recorded identity matches live "
            "starttime) must be returned for reaping — the durable proof "
            "preserves the P12 no-listener route"
        )
        assert 5001 not in pids, (
            "foreign llama-server (no recorded identity) must NOT be returned "
            "even when mixed with a recorded Turbohaul-owned one"
        )

    def test_turbohaul_owned_reparented_orphan_returned(self, monkeypatch):
        """A Turbohaul-owned orphan reparented to init IS returned when its
        recorded identity matches the live starttime. PPid=1 is NOT the
        proof — the recorded (pid, port, starttime) match is.
        """
        from turbohaul.singleton import find_llama_servers_in_port_range

        monkeypatch.setattr("turbohaul.singleton._SUBREAPER_PID", None)
        cmdline_fn, ppid_fn, starttime_fn = self._make_fakes(self._OWNED_REPARENTED_PROCS)
        result = find_llama_servers_in_port_range(
            59500,
            port_range_size=100,
            cmdline_fn=cmdline_fn,
            ppid_fn=ppid_fn,
            proc_pids_fn=lambda: list(self._OWNED_REPARENTED_PROCS.keys()),
            starttime_fn=starttime_fn,
            known_engine_identities={(5003, 59502, 7003)},
        )
        pids = {r["pid"] for r in result}
        assert 5003 in pids, (
            "Turbohaul-owned orphan (recorded identity matches, PPid=1) must "
            "be returned — the durable proof, not the PPid, is the authority"
        )

    def test_foreign_llama_server_not_reaped_by_boot_orphan_reaper(
        self, monkeypatch
    ):
        """Integration: boot_orphan_reaper must NOT reap a foreign
        llama-server — it has no recorded identity, so the durable proof
        filters it out before reaping.
        """
        from turbohaul.singleton import boot_orphan_reaper

        monkeypatch.setattr("turbohaul.singleton._SUBREAPER_PID", None)
        monkeypatch.setattr(
            "turbohaul.singleton.find_orphan_llama_servers",
            lambda port_base, port_range_size=100, **kw: [],
        )
        monkeypatch.setattr(
            "turbohaul.singleton.port_listeners_in_range",
            lambda port_base, port_range_size=100: [],
        )
        procs = dict(self._FOREIGN_PROCS)
        cmdline_fn, ppid_fn, starttime_fn = self._make_fakes(procs)
        monkeypatch.setattr("turbohaul.singleton._read_proc_cmdline", cmdline_fn)
        monkeypatch.setattr("turbohaul.singleton._read_proc_ppid", ppid_fn)
        monkeypatch.setattr("turbohaul.singleton._read_proc_starttime", starttime_fn)
        monkeypatch.setattr(
            "turbohaul.singleton._list_proc_pids", lambda: list(procs.keys())
        )
        reaped: list[int] = []

        def fake_reap(pid, **kw):
            reaped.append(pid)
            return True, "sigterm-clean"

        result = boot_orphan_reaper(59500, reap_fn=fake_reap)
        assert 5001 not in reaped, (
            "foreign llama-server must NOT be reaped by boot_orphan_reaper — "
            "no recorded identity, durable proof filters it out"
        )
        assert result["reaped"] == 0

    def test_turbohaul_owned_stale_engine_reaped_by_boot_orphan_reaper(
        self, monkeypatch
    ):
        """Integration: boot_orphan_reaper DOES reap a genuine Turbohaul-owned
        no-listener orphan whose recorded identity matches the live starttime,
        proving the P12 route survives the durable-identity hardening.
        """
        from turbohaul.singleton import boot_orphan_reaper

        monkeypatch.setattr("turbohaul.singleton._SUBREAPER_PID", None)
        monkeypatch.setattr(
            "turbohaul.singleton.find_orphan_llama_servers",
            lambda port_base, port_range_size=100, **kw: [],
        )
        monkeypatch.setattr(
            "turbohaul.singleton.port_listeners_in_range",
            lambda port_base, port_range_size=100: [],
        )
        procs = dict(self._OWNED_STALE_PROCS)
        cmdline_fn, ppid_fn, starttime_fn = self._make_fakes(procs)
        monkeypatch.setattr("turbohaul.singleton._read_proc_cmdline", cmdline_fn)
        monkeypatch.setattr("turbohaul.singleton._read_proc_ppid", ppid_fn)
        monkeypatch.setattr("turbohaul.singleton._read_proc_starttime", starttime_fn)
        monkeypatch.setattr(
            "turbohaul.singleton._list_proc_pids", lambda: list(procs.keys())
        )
        reaped: list[int] = []

        def fake_reap(pid, **kw):
            reaped.append(pid)
            return True, "sigterm-clean"

        result = boot_orphan_reaper(
            59500, reap_fn=fake_reap,
            known_engine_identities={(5002, 59501, 7002)},
        )
        assert 5002 in reaped, (
            "Turbohaul-owned no-listener orphan (recorded identity matches "
            "live starttime) must be reaped"
        )
        assert result["reaped"] >= 1

# ---------------------------------------------------------------------------
# 7. Durable engine-identity proof (replaces heuristic ownership proof).
#    A candidate is Turbohaul-owned ONLY if state.sqlite carries a recorded
#    engine identity (pid + starttime + port) for it, and the LIVE
#    /proc/<pid> starttime + --port match that record. Reparenting to init
#    (PPid=1) does NOT prove ownership — a foreign/independent llama-server
#    that is itself orphaned (PPid=1) must NOT be reaped. PID reuse is
#    prevented by the starttime cross-check: a PID-reused replacement has a
#    different starttime, so it can never match a recorded identity.
# ---------------------------------------------------------------------------

class TestDurableEngineIdentityProof:
    """Durable ownership proof: a recorded (pid, starttime, port) identity
    in state.sqlite is the ONLY ownership signal. PPid=1 / parent-chain
    heuristics are NOT ownership proof — a foreign orphaned llama-server
    must be report-only, never reaped.
    """

    # Foreign llama-server that is ITSELF orphaned (PPid=1). The OLD
    # heuristic wrongly treated PPid=1 as ownership proof and would reap
    # this; the durable proof must NOT — there is no recorded identity for
    # it in state.sqlite.
    _FOREIGN_ORPHAN_PROCS = {
        6001: {"cmdline": "llama-server --port 59500 -m /foreign.gguf",
               "ppid": 1, "starttime": 9001},
    }

    # Recorded Turbohaul-owned P12 no-listener stale engine: the manager
    # died, engine reparented to init. state.sqlite recorded its identity
    # at spawn (pid=6002, port=59501, starttime=9002). The live /proc entry
    # STILL has starttime=9002 (same process), so the durable proof MATCHES
    # and the engine is reaped — the P12 no-listener route survives.
    _OWNED_RECORDED_PROCS = {
        6002: {"cmdline": "llama-server --port 59501 -m /owned.gguf",
               "ppid": 1, "starttime": 9002},
    }

    @staticmethod
    def _make_fakes(procs):
        def cmdline_fn(pid):
            return procs.get(pid, {}).get("cmdline", "")

        def ppid_fn(pid):
            return procs.get(pid, {}).get("ppid", None)

        def starttime_fn(pid):
            return procs.get(pid, {}).get("starttime", None)

        return cmdline_fn, ppid_fn, starttime_fn

    def test_foreign_ppid1_llama_server_not_reaped(self, monkeypatch):
        """RED: a FOREIGN llama-server --port in the managed range that is
        ITSELF orphaned (PPid=1) must NOT be reaped. PPid=1 is NOT ownership
        proof — reparenting to init only means SOME parent died, not that
        the parent was Turbohaul. The durable proof requires a recorded
        (pid, starttime, port) identity in state.sqlite; a foreign orphan
        has none, so it is report-only.
        """
        from turbohaul.singleton import boot_orphan_reaper

        monkeypatch.setattr("turbohaul.singleton._SUBREAPER_PID", None)
        # Let the REAL find_orphan_llama_servers run with injected /proc
        # readers. It NOMINATES pid 6001 (PPid=1, port in range) but the
        # durable identity check inside it must DROP the candidate (no
        # recorded identity) — so boot_orphan_reaper never sees it.
        monkeypatch.setattr(
            "turbohaul.singleton.port_listeners_in_range",
            lambda port_base, port_range_size=100: [],
        )
        procs = dict(self._FOREIGN_ORPHAN_PROCS)
        cmdline_fn, ppid_fn, starttime_fn = self._make_fakes(procs)
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_cmdline", cmdline_fn
        )
        monkeypatch.setattr("turbohaul.singleton._read_proc_ppid", ppid_fn)
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_starttime", starttime_fn
        )
        monkeypatch.setattr(
            "turbohaul.singleton._list_proc_pids",
            lambda: list(procs.keys()),
        )
        reaped: list[int] = []

        def fake_reap(pid, **kw):
            reaped.append(pid)
            return True, "sigterm-clean"

        # NO recorded engine identities → foreign orphan must NOT be reaped.
        result = boot_orphan_reaper(
            59500,
            reap_fn=fake_reap,
            known_engine_identities=set(),  # no recorded proof
        )
        assert 6001 not in reaped, (
            "foreign orphaned llama-server (PPid=1) must NOT be reaped — "
            "PPid=1 is NOT ownership proof; only a recorded (pid, starttime, "
            "port) identity in state.sqlite proves Turbohaul ownership"
        )
        assert result["reaped"] == 0

    def test_recorded_p12_no_listener_stale_engine_reaped(self, monkeypatch):
        """GREEN target: a RECORDED Turbohaul-owned P12 no-listener stale
        engine (state.sqlite carries pid=6002, starttime=9002, port=59501)
        IS reaped, because the LIVE /proc starttime (9002) + port (59501)
        match the recorded identity. This is the P12 route preserved by the
        durable proof: the manager crashed after recording the identity,
        the engine has no listener socket, but the recorded identity +
        live starttime match proves ownership.
        """
        from turbohaul.singleton import boot_orphan_reaper

        monkeypatch.setattr("turbohaul.singleton._SUBREAPER_PID", None)
        monkeypatch.setattr(
            "turbohaul.singleton.find_orphan_llama_servers",
            lambda port_base, port_range_size=100, **kw: [],
        )
        monkeypatch.setattr(
            "turbohaul.singleton.port_listeners_in_range",
            lambda port_base, port_range_size=100: [],
        )
        procs = dict(self._OWNED_RECORDED_PROCS)
        cmdline_fn, ppid_fn, starttime_fn = self._make_fakes(procs)
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_cmdline", cmdline_fn
        )
        monkeypatch.setattr("turbohaul.singleton._read_proc_ppid", ppid_fn)
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_starttime", starttime_fn
        )
        monkeypatch.setattr(
            "turbohaul.singleton._list_proc_pids",
            lambda: list(procs.keys()),
        )
        reaped: list[int] = []

        def fake_reap(pid, **kw):
            reaped.append(pid)
            return True, "sigterm-clean"

        # Recorded identity matches the live process → reaped.
        result = boot_orphan_reaper(
            59500,
            reap_fn=fake_reap,
            known_engine_identities={(6002, 59501, 9002)},
        )
        assert 6002 in reaped, (
            "recorded Turbohaul-owned P12 no-listener stale engine "
            "(state.sqlite pid+starttime+port matches live /proc) must be "
            "reaped — the durable identity proof preserves the P12 route"
        )
        assert result["reaped"] >= 1

    def test_pid_reuse_not_reaped(self, monkeypatch):
        """PID reuse safety: a recorded identity (pid=6002, starttime=9002)
        must NOT match a live process that re-used pid 6002 but has a
        DIFFERENT starttime (the original exited and the kernel recycled
        the pid). The starttime cross-check prevents reaping a PID-reused
        replacement.
        """
        from turbohaul.singleton import boot_orphan_reaper

        monkeypatch.setattr("turbohaul.singleton._SUBREAPER_PID", None)
        monkeypatch.setattr(
            "turbohaul.singleton.find_orphan_llama_servers",
            lambda port_base, port_range_size=100, **kw: [],
        )
        monkeypatch.setattr(
            "turbohaul.singleton.port_listeners_in_range",
            lambda port_base, port_range_size=100: [],
        )
        # Same pid + port, DIFFERENT starttime (PID was reused).
        procs = {
            6002: {"cmdline": "llama-server --port 59501 -m /reuse.gguf",
                   "ppid": 1, "starttime": 9999},
        }
        cmdline_fn, ppid_fn, starttime_fn = self._make_fakes(procs)
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_cmdline", cmdline_fn
        )
        monkeypatch.setattr("turbohaul.singleton._read_proc_ppid", ppid_fn)
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_starttime", starttime_fn
        )
        monkeypatch.setattr(
            "turbohaul.singleton._list_proc_pids",
            lambda: list(procs.keys()),
        )
        reaped: list[int] = []

        def fake_reap(pid, **kw):
            reaped.append(pid)
            return True, "sigterm-clean"

        result = boot_orphan_reaper(
            59500,
            reap_fn=fake_reap,
            known_engine_identities={(6002, 59501, 9002)},  # original
        )
        assert 6002 not in reaped, (
            "PID-reused replacement (same pid, different starttime) must "
            "NOT be reaped — starttime cross-check prevents it"
        )
        assert result["reaped"] == 0


# ---------------------------------------------------------------------------
# 8. END-TO-END boot_reconcile: a LIVE recorded P12 no-listener stale engine
#    must be reaped even when pid_is_alive_fn reports it ALIVE. This is the
#    core correctness bug in the durable-identity wiring: boot_reconcile
#    computes live_pids (PIDs alive per state.sqlite) and passes it as
#    known_pids to boot_orphan_reaper. A recorded P12 stale engine is alive
#    by definition, so the reaper's known_pids skip-list contains it and
#    SKIPS it — defeating boot cleanup. At boot there is no current-manager
#    active sidecar to preserve (flock guarantees singleton ownership), so
#    the skip-list must NOT include recorded-but-stale engines.
# ---------------------------------------------------------------------------

class TestBootReconcileLiveRecordedStaleEngineReaped:
    """End-to-end TurbohaulManager.boot_reconcile: a recorded P12 no-listener
    stale engine that is ALIVE (pid_is_alive_fn=True) must be reaped, not
    skipped. The known_pids skip-list passed to boot_orphan_reaper must be
    EMPTY at boot (no current-manager active sidecars survive — flock
    singleton ownership), so the durable-identity match in the stale-engine
    scan is the sole reaping authority."""

    def test_live_recorded_p12_stale_engine_reaped_end_to_end(
        self, boot_and_runtime, monkeypatch
    ):
        """RED->GREEN: boot_reconcile with a recorded triple (pid=5555,
        port=59500, starttime=7000) in state.sqlite AND pid_is_alive_fn that
        reports 5555 ALIVE must still reap 5555. The bug: live_pids contained
        5555 (alive), passed as known_pids, so the reaper skipped it."""
        boot, runtime = boot_and_runtime
        port = boot.runtime.default_port_base  # 59500
        from turbohaul.manager import TurbohaulManager
        from turbohaul.state import (
            record_engine_identity, upsert_slot, open_state_db,
        )
        mgr = TurbohaulManager(boot, runtime)

        # Seed a RECORDED engine identity (pid=5555, port, starttime=7000).
        _c = open_state_db(boot.storage.state_db_path)
        upsert_slot(_c, {"slot_id": "s-stale", "model_tag": "m",
                         "state": "ACTIVE", "pid": 5555, "port": port})
        record_engine_identity(_c, "s-stale", 5555, port, 7000)
        _c.close()

        # PPid-based scan finds nothing (no-listener, PPid not in reapers).
        monkeypatch.setattr(
            "turbohaul.singleton.find_orphan_llama_servers",
            lambda port_base, port_range_size=100, **kw: [],
        )
        # Listener socket scan finds nothing — the orphan has NO listener.
        monkeypatch.setattr(
            "turbohaul.singleton.port_listeners_in_range",
            lambda port_base, port_range_size=100: [],
        )
        # /proc starttime reader so the durable proof reads 7000.
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_starttime", lambda pid: 7000
        )
        # Stale-engine scan nominates pid 5555 (cmdline + --port in range).
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_cmdline",
            lambda pid: (
                "llama-server --port 59500 --model /x/y.gguf"
                if pid == 5555 else ""
            ),
        )
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_ppid", lambda pid: 7777
        )
        monkeypatch.setattr(
            "turbohaul.singleton._list_proc_pids", lambda: [5555]
        )
        reaped = {"pids": []}

        def fake_reap_orphan(pid, **kw):
            reaped["pids"].append(pid)
            return True, "sigterm-clean"

        # KEY: the stale engine is ALIVE. The bug made boot_reconcile pass
        # live_pids (={5555}) as known_pids, so the reaper skipped it.
        result = mgr.boot_reconcile(
            reap_orphan_fn=fake_reap_orphan,
            pid_is_alive_fn=lambda pid: pid == 5555,  # stale engine ALIVE
        )
        assert 5555 in reaped["pids"], (
            "LIVE recorded P12 no-listener stale engine must be reaped at "
            "boot — the durable-identity match is the sole reaping authority; "
            "an alive recorded stale engine is NOT a current-manager sidecar "
            "to preserve (flock singleton ownership), so it must NOT be in "
            "the reaper's known_pids skip-list"
        )
        assert result["orphans_reaped"] >= 1

    def test_foreign_unrecorded_alive_process_not_reaped_end_to_end(
        self, boot_and_runtime, monkeypatch
    ):
        """Conservative ownership: a foreign/unrecorded llama-server in the
        managed range (including PPid=1) that is ALIVE must NOT be reaped,
        even when pid_is_alive_fn reports it alive. It has no recorded
        identity, so the durable-identity filter drops it (report-only)."""
        boot, runtime = boot_and_runtime
        port = boot.runtime.default_port_base  # 59500
        from turbohaul.manager import TurbohaulManager
        from turbohaul.state import open_state_db
        mgr = TurbohaulManager(boot, runtime)

        # NO recorded engine identity for the foreign process.
        monkeypatch.setattr(
            "turbohaul.singleton.find_orphan_llama_servers",
            lambda port_base, port_range_size=100, **kw: [],
        )
        monkeypatch.setattr(
            "turbohaul.singleton.port_listeners_in_range",
            lambda port_base, port_range_size=100: [],
        )
        # Foreign llama-server on the managed port, PPid=1 (orphaned foreign).
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_cmdline",
            lambda pid: (
                "llama-server --port 59500 --model /foreign.gguf"
                if pid == 8888 else ""
            ),
        )
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_ppid", lambda pid: 1
        )
        monkeypatch.setattr(
            "turbohaul.singleton._read_proc_starttime", lambda pid: 9000
        )
        monkeypatch.setattr(
            "turbohaul.singleton._list_proc_pids", lambda: [8888]
        )
        reaped = {"pids": []}

        def fake_reap_orphan(pid, **kw):
            reaped["pids"].append(pid)
            return True, "sigterm-clean"

        result = mgr.boot_reconcile(
            reap_orphan_fn=fake_reap_orphan,
            pid_is_alive_fn=lambda pid: pid == 8888,  # foreign ALIVE
        )
        assert 8888 not in reaped["pids"], (
            "foreign/unrecorded llama-server (PPid=1, no recorded identity) "
            "must NOT be reaped even when alive — conservative ownership"
        )
        assert result["orphans_reaped"] == 0
