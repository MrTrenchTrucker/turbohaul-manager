"""The cap>=2 post-teardown orphan reap.

The cap<=1 path reaps stray llama-server processes after every teardown
(`_teardown` and `_teardown_idle_holder` in manager.py).
The cap>=2 teardown paths (`_unload_teardown` + `_drive_resident`'s finally)
must call BOTH reapers: the kill chain (drained_sigterm -> killpg on the
sidecar's own session) cannot reach setsid-detached helpers
(see manager.py and subprocess_mgr.py), and nothing else on the
cap>=2 path walks /proc during the manager's lifetime. Cap=2 is a common
production setting, so without the reap grandchild orphans would stay alive.

These tests prove:
  1. The reap RUNS on the cap>=2 path — at BOTH teardown claim points
     (the exactly-once `r.torn_down` mutual exclusives).
  2. NEGATIVE CONTROL: the reap does NOT kill a co-resident's LIVE sidecar —
     the protection is `_live_handle_pids()` (the cap>=2-aware union of every
     resident's live handle + idle handle + booting pid, in manager.py),
     and this test exercises the REAL reaper functions against a controlled
     fake /proc rather than restating the argument.

RED-FIRST: on the unpatched base these tests FAIL because
`_unload_teardown` / the driver finally never call the reapers at all. The
failing assertion is the `reap_spy.calls` / `intra_spy.calls` membership
check (or, for the negative control, the "orphan was never killed" check).
On the unpatched base the tests fail with an assertion error, not an exception.

Fixtures adapted from tests/test_multislot_concurrency.py.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.state import (
    record_engine_identity,
    state_db_session,
    upsert_slot,
)
from turbohaul.subprocess_mgr import SidecarHandle

PORT_BASE = 59500

# Fixed pids for the negative control's fake /proc world.
VICTIM_PID = 49900
CO_PID = 49901
ORPHAN_PID = 49902
FOREIGN_PID = 49903
ORPHAN_STARTTIME = 123456789
MANAGER_LIKE_PPID = 42  # not 1, not a subreaper: a normal parent


def _boot_runtime_multislot(tmp_path, *, max_parallel_sidecars=2,
                            grace_seconds=0, idle_hot_load_seconds=0):
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
            default_port_base=PORT_BASE,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=idle_hot_load_seconds,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag, *, expected_vram_mib=0,
                   split_mode="none", main_gpu=0):
    """Minimal manifest; split_mode/main_gpu drive the co-residence gate
    (co-residence admitted ONLY for split_mode='none' on DISTINCT main_gpu)."""
    import yaml
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": expected_vram_mib * 1024 * 1024,
        "context_size": 2048,
        "expected_vram_bytes": expected_vram_mib * 1024 * 1024,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


def _fake_handle(model_tag: str, port: int, pid: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None  # alive
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _high_vram():
    # TWO cards, each ample (co-residence needs DISTINCT main_gpu cards).
    return patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000])


def _mocks(spawn_calls, sigterm_calls=None):
    pid = [90000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        spawn_calls.append({"model_tag": model_tag, "port": port})
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        if sigterm_calls is not None:
            sigterm_calls.append(handle.model_tag)
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


def _mk(boot, runtime, **mocks):
    mgr = TurbohaulManager(boot, runtime, **mocks)
    mgr.runtime.queue.safety_enabled = False
    return mgr


class _ReapSpy:
    """Records the manager-level reaper calls (patches the names the manager
    module resolves to)."""

    def __init__(self):
        self.boot_calls: list[dict] = []
        self.intra_calls: list[dict] = []

    def _boot(self, *a, **k):
        self.boot_calls.append(k)
        return {"scanned": 0, "orphans_found": 0, "reaped": 0, "failed": 0,
                "details": [], "stale_listeners": 0}

    def _intra(self, *a, **k):
        self.intra_calls.append(k)
        return {"scanned": 0, "matched": 0, "reaped": 0, "errors": 0}

    def patch(self, monkeypatch):
        monkeypatch.setattr("turbohaul.manager.boot_orphan_reaper", self._boot)
        monkeypatch.setattr("turbohaul.manager.intra_lifetime_orphan_scan", self._intra)


async def _wait_until(pred, timeout_s: float = 6.0):
    """Poll an awaitable predicate; True when it holds, False on timeout."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


class TestTwoResidentCapOrphanReapWiring:
    """The reap must run on the cap>=2 path — at BOTH teardown claim points.

    RED on the unpatched base: _unload_teardown / the driver finally never call
    either reaper, so the spy lists stay empty and the first assert fires.
    """

    async def test_evict_teardown_reaps_orphans_two_resident_cap(self, tmp_path, monkeypatch):
        """External eviction (the _unload_teardown claim point)."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120,
        )
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)
        spawn_calls, sigterm_calls = [], []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        spy = _ReapSpy()
        spy.patch(monkeypatch)
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                # Two live co-residents.
                await asyncio.wait_for(
                    mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.wait_for(
                    mgr.submit_and_wait("m2", "b", thread_id="t2"), timeout=5)
                await asyncio.sleep(0.15)
                r1 = mgr._residents.get("m1")
                r2 = mgr._residents.get("m2")
                assert r1 is not None and r2 is not None, "both residents must be live"
                assert r2.handle is not None, "co-resident handle must be set"
                m2_pid = r2.handle.pid
                assert m2_pid is not None

                # Evict m1 from OUTSIDE the driver (the usual eviction idiom).
                async with mgr._registry_lock:
                    mgr._begin_unload_locked(r1)
                ok = await _wait_until(lambda: mgr._residents.get("m1") is None)
                assert ok, "m1 never left the registry"
                # The teardown task runs after the claim; give the reap a turn.
                await _wait_until(
                    lambda: (spy.boot_calls or spy.intra_calls), timeout_s=5)

                assert "m1" in sigterm_calls, "teardown kill chain never ran"
                assert spy.boot_calls, (
                    "boot_orphan_reaper was NEVER called on the "
                    "cap>=2 _unload_teardown path — the orphan reap is missing"
                )
                assert spy.intra_calls, (
                    "intra_lifetime_orphan_scan was NEVER called on "
                    "the cap>=2 _unload_teardown path — the orphan reap is missing"
                )
                for kw in spy.boot_calls:
                    assert m2_pid in kw.get("known_pids", set()), (
                        "co-resident m2 is NOT protected in known_pids — the "
                        "reaper could kill a live sibling"
                    )
                for kw in spy.intra_calls:
                    assert m2_pid in kw.get("known_handle_pids", set()), (
                        "co-resident m2 is NOT protected in known_handle_pids"
                    )
                assert all(
                    kw.get("port_base") == PORT_BASE for kw in spy.boot_calls
                ), "port_base must come from the runtime config"
            finally:
                await mgr.shutdown()

    async def test_drive_resident_finally_reaps_orphans_two_resident_cap(self, tmp_path, monkeypatch):
        """Natural idle timeout INSIDE the driver loop — the driver's own
        finally is (or races to be) the claim point. Either way, the reap must
        happen on the cap>=2 path."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=1,
        )
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        spawn_calls, sigterm_calls = [], []
        mgr = _mk(boot, runtime, **_mocks(spawn_calls, sigterm_calls=sigterm_calls))
        spy = _ReapSpy()
        spy.patch(monkeypatch)
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(
                    mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.sleep(0.1)
                r1 = mgr._residents.get("m1")
                assert r1 is not None and r1.state is ResidentState.IDLE_EVICTABLE
                # Let the real 1s idle window elapse inside the driver's own
                # wait_for (no direct manager call — same idiom).
                ok = await _wait_until(lambda: mgr._residents.get("m1") is None,
                                       timeout_s=8)
                assert ok, "m1 never evicted on idle timeout"
                await _wait_until(
                    lambda: (spy.boot_calls or spy.intra_calls), timeout_s=5)

                assert "m1" in sigterm_calls, "teardown kill chain never ran"
                assert spy.boot_calls, (
                    "boot_orphan_reaper was NEVER called on the "
                    "cap>=2 driver-finally/idle-timeout teardown path"
                )
                assert spy.intra_calls, (
                    "intra_lifetime_orphan_scan was NEVER called on "
                    "the cap>=2 driver-finally/idle-timeout teardown path"
                )
            finally:
                await mgr.shutdown()


class TestTwoResidentCapOrphanReapNegativeControl:
    """THE control: the real reaper functions run against a controlled fake
    /proc with a live co-resident, an identity-recorded orphan, and a foreign
    non-llama process. After tearing down the victim, EXACTLY the orphan must
    be killed — never the co-resident's live sidecar, never the foreign one.
    """

    def _seed_orphan_identity(self, boot):
        """Record a durable engine identity for the orphan (as a real spawn
        would have) so the identity-gated reaper has proof of ownership."""
        with state_db_session(boot.storage.state_db_path) as conn:
            upsert_slot(conn, {
                "slot_id": "reap-orphan",
                "model_tag": "orphan",
                "thread_id": "t-orph",
                "state": "ACTIVE",
                "port": PORT_BASE + 2,
                "pid": ORPHAN_PID,
                "extension_count": 0,
                "client_meta": None,
            })
            record_engine_identity(
                conn, "reap-orphan", ORPHAN_PID, PORT_BASE + 2,
                ORPHAN_STARTTIME,
            )

    async def test_reap_kills_orphan_but_not_live_co_resident(self, tmp_path,
                                                               monkeypatch):
        import turbohaul.singleton as single

        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120,
        )
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        _seed_manifest(boot, "m2", split_mode="none", main_gpu=1)
        self._seed_orphan_identity(boot)

        # --- fake /proc world -------------------------------------------------
        # The victim (49900) is DEAD at reap time (fake sigterm flips it), so a
        # faithful fake /proc does NOT list it.
        cmdlines = {
            CO_PID: f"llama-server --port {PORT_BASE + 1} --model m2",
            ORPHAN_PID: f"llama-server --port {PORT_BASE + 2} --model orphan",
            FOREIGN_PID: "python3 -m http.server",
        }
        ppid = {CO_PID: MANAGER_LIKE_PPID, ORPHAN_PID: 1,
                FOREIGN_PID: MANAGER_LIKE_PPID}
        starttimes = {CO_PID: 111, ORPHAN_PID: ORPHAN_STARTTIME,
                      FOREIGN_PID: 333}

        monkeypatch.setattr(single, "_list_proc_pids",
                            lambda: [CO_PID, ORPHAN_PID, FOREIGN_PID])
        monkeypatch.setattr(single, "_read_proc_cmdline",
                            lambda pid: cmdlines.get(pid, ""))
        monkeypatch.setattr(single, "_read_proc_ppid", lambda pid: ppid.get(pid))
        monkeypatch.setattr(single, "_read_proc_starttime",
                            lambda pid: starttimes.get(pid))

        reaped_pids: list[int] = []
        monkeypatch.setattr(single, "reap_orphan",
                            lambda pid: reaped_pids.append(pid) or (True, "test-reaped"))
        monkeypatch.setattr(single, "port_listeners_in_range", lambda *a, **k: [])

        killed_pids: list[int] = []

        def fake_kill(pid, sig):
            killed_pids.append(pid)

        fake_os = SimpleNamespace(kill=fake_kill)
        monkeypatch.setattr(single, "os", fake_os)

        # --- manager world ----------------------------------------------------
        pids_by_model = {"m1": VICTIM_PID, "m2": CO_PID}

        def fixed_spawn(binary, gguf, port, model_tag, argv, **_kw):
            proc = MagicMock()
            proc.pid = pids_by_model[model_tag]
            proc.poll.return_value = None
            return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

        sigterm_calls: list[str] = []

        async def fake_sigterm(handle, **k):
            sigterm_calls.append(handle.model_tag)
            if handle.model_tag == "m1":
                handle.proc.poll.return_value = -15  # dead after SIGTERM
            return True, "sigterm-clean"

        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fixed_spawn, health_fn=_fake_health,
            sigterm_fn=fake_sigterm, vram_fn=_fake_vram, complete_fn=_fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(
                    mgr.submit_and_wait("m1", "a", thread_id="t1"), timeout=5)
                await asyncio.wait_for(
                    mgr.submit_and_wait("m2", "b", thread_id="t2"), timeout=5)
                await asyncio.sleep(0.15)
                r1 = mgr._residents.get("m1")
                assert r1 is not None, "victim resident missing"
                # sanity: both handles live before teardown
                r2 = mgr._residents.get("m2")
                assert (
                    r2 is not None
                    and r2.handle is not None
                    and r2.handle.pid == CO_PID
                ), "co-resident m2 must be live with its fixed pid"

                async with mgr._registry_lock:
                    mgr._begin_unload_locked(r1)
                ok = await _wait_until(lambda: mgr._residents.get("m1") is None)
                assert ok, "victim never left the registry"
                # Wait until the orphan shows up as killed (bounded).
                ok = await _wait_until(lambda: ORPHAN_PID in (set(reaped_pids) | set(killed_pids)),
                                       timeout_s=5)
                assert ok, (
                    "the identity-recorded orphan was NEVER killed on "
                    "the cap>=2 path — the reap is missing (or not wired)"
                )

                # THE negative control: exactly the orphan was killed.
                killed = set(reaped_pids) | set(killed_pids)
                assert killed == {ORPHAN_PID}, (
                    f"reap killed the wrong process set: {sorted(killed)} — "
                    f"expected ONLY the orphan {ORPHAN_PID}. "
                    f"reap_orphan={reaped_pids} intra_kill={killed_pids}"
                )
                assert CO_PID not in killed, (
                    "LIVE CO-RESIDENT SIDEKAR WAS KILLED — protection failure"
                )
                assert FOREIGN_PID not in killed, "foreign process was killed"
                assert "m1" in sigterm_calls, "victim kill chain never ran"
            finally:
                await mgr.shutdown()


async def _fake_health(*a, **k):
    return True


async def _fake_vram(**k):
    return True, 100


async def _fake_complete(slot, handle):
    return {"ok": True, "model": handle.model_tag}
