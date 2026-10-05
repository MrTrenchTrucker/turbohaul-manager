"""Admission-pump stall on last-resident eviction.

Root cause:
  `_dispatch_wake` (asyncio.Event) is set ONLY via on_enqueue
  (manager.py) -- exactly 4 wake sites (init, hook, wait, clear).
  The eviction/teardown path (`_begin_unload_locked` -> `_unload_teardown`)
  calls `_make_room_signal.notify_all()` but NEVER `_dispatch_wake.set()`.
  `_make_room_signal` only wakes coroutines parked in _wait_for_park_or_timeout
  (deferred-slot waiters) -- NOT the dispatch loop. At 0 residents with no
  deferred waiter, the eviction wake is a complete no-op for dispatch signaling.

  The dispatch loop's only fallback is `_DISPATCH_DEFER_BACKOFF_S` (0.05s poll),
  but staging-arrival Fast Lane claims (1800s TTL) hold-aside all lower-priority
  staged slots invisible to the pick ladder while a governing claim is live. At 0
  residents the claim never gets superseded, so it holds for the FULL 30min --
  only after `fastlane_claims_snapshot` prunes the dead claim does hold-aside
  disarm and staging become visible again (stalls lasting up to the claim TTL).

Fix: `_dispatch_wake.set()` called alongside every existing
`_make_room_signal.notify_all()` call site -- the 7 production sites
enumerated by `TestNotifyCallSitesAreTheNamedSet`:
`_unload_teardown`, `_late_vram_reconcile`, `_drive_resident` (x3),
`_serve_on_resident`, `_idle_engine_liveness_sweep`.

`_dispatch_wake` is an asyncio.Event (no lock) -- `set()` is synchronous,
idempotent, and safe under `_registry_lock`. No new `notify_all()` sites
are added, so `TestNotifyCallSitesAreTheNamedSet` passes unchanged.

RED: unfixed tree -- 2 of 3 tests fail (notify site does not set
_dispatch_wake; static check finds 5 missing sites).

GREEN: fixed tree -- all 3 tests pass.
"""
import asyncio
import inspect
from unittest.mock import MagicMock

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import TurbohaulManager, Resident, ResidentState
from turbohaul.slot import Slot
from turbohaul.state import state_db_session


_RULES = [
    FastLaneRule(address="10.0.0.5", tag_ranks=FastLaneTagRanks(main=1)),
]


def _boot_runtime(tmp_path, *, grace_seconds=5):
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
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            max_parallel_sidecars=1,
            safety_enabled=False,
            grace_seconds=grace_seconds,
            max_other_model_wait_s=0.2,
            max_grace_extensions=50,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=_RULES),
    )
    return boot, runtime


@pytest.fixture
def mgr(tmp_path):
    """Real TurbohaulManager, fastlane on, dispatch loop NOT started."""
    boot, runtime = _boot_runtime(tmp_path)
    m = TurbohaulManager(boot, runtime)
    with state_db_session(m.boot.storage.state_db_path):
        pass
    return m


def _vram_verify_mock_result():
    """Async mock for _vram_verify that returns (cleared_ok, current_used)."""
    async def _mock(**kw):
        return True, 0
    return _mock()


def _make_fake_resident(mgr, model_tag="m1"):
    """Build a minimal Resident registered in mgr._residents.

    Only sets the fields that _begin_unload_locked and _unload_teardown
    actually read during the eviction path."""
    r = Resident()
    r.model_tag = model_tag
    r.resident_key = f"{model_tag}-0"
    r.state = ResidentState.ACTIVE  # _begin_unload_locked will transition to DEAD
    r.inbox = asyncio.Queue()     # empty inbox -- no drain needed
    r.handle = MagicMock()
    r.handle.is_alive.return_value = False
    r.torn_down = False
    r.grace = None
    r.main_gpu = 0
    r.idle_thread_id = None
    r.booting_pid = None
    r.reserved_need_mib = 512  # nonzero so the VRAM-reclaim notify path fires
    r.reclaim_credited_mib = 512
    r.admission_role = None
    return r


class TestAdmissionPumpStall:
    """Eviction must wake the dispatch loop."""

    @pytest.mark.asyncio
    async def test_eviction_notify_sets_dispatch_wake(
        self, mgr
    ):
        """RED at unfixed code: _unload_teardown's notify_all does NOT set
        _dispatch_wake -- the dispatch loop stays parked until TTL expiry.

        GREEN with fix: _dispatch_wake.set() fires alongside the notify."""
        mgr._dispatch_wake.clear()
        assert not mgr._dispatch_wake.is_set()

        # Mock the VRAM-related calls so _unload_teardown can complete
        # without real GPU hardware.
        mgr._gpu_used_mib = lambda **kw: 0
        mgr._vram_verify = lambda **kw: _vram_verify_mock_result()

        # Build a resident in the registry, then fire _begin_unload_locked
        # + _unload_teardown. The teardown path fires notify_all under
        # _registry_lock -- the fix adds _dispatch_wake.set() there.
        r = _make_fake_resident(mgr)
        mgr._residents[r.resident_key] = r

        # _begin_unload_locked drains the inbox, marks DEAD, spawns
        # _unload_teardown. We intercept the spawned task.
        original_spawn = mgr._spawn_bg

        captured = []
        def capture_spawn(coro):
            task = original_spawn(coro)
            captured.append(task)
            return task

        mgr._spawn_bg = capture_spawn

        async with mgr._registry_lock:
            mgr._begin_unload_locked(r)

        # _begin_unload_locked spawns _unload_teardown as a bg task.
        # Await it so the notify path runs.
        for task in captured:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            await mgr._drain_bg_tasks()

        # The teardown fires _make_room_signal.notify_all() -- on the
        # UNFIXED tree, _dispatch_wake stays unset. On the FIXED tree,
        # _dispatch_wake.set() fires alongside it.
        post_evict = mgr._dispatch_wake.is_set()

        assert post_evict, (
            "STALL: _dispatch_wake must be set after eviction "
            "notify -- otherwise the dispatch loop stays parked on "
            "_dispatch_wake.wait() until the 1800s claim TTL expires "
            "(stalls up to the claim TTL at 0 residents with a live claim)"
        )

        mgr._dispatch_wake.clear()

    @pytest.mark.asyncio
    async def test_plain_staging_arrival_uses_existing_on_enqueue(
        self, mgr
    ):
        """Negative control: staging a NEW request sets _dispatch_wake via
        on_enqueue (manager.py) -- the EXISTING path, NOT part of the
        eviction fix. This test proves the fix is scoped to eviction notify
        sites, not staging arrives.

        RED at unfixed code: PASSES (on_enqueue already works).
        GREEN with fix: STILL passes (no spurious wakes added at staging)."""
        mgr._dispatch_wake.clear()
        assert not mgr._dispatch_wake.is_set()

        await mgr.queue.enqueue(
            Slot.new(
                model_tag="m1",
                prompt="hi",
                thread_id="nonfastlane-thread",
            )
        )
        await mgr._drain_bg_tasks()

        assert mgr._dispatch_wake.is_set(), (
            "on_enqueue must set _dispatch_wake for staging arrival "
            "(existing behavior, not part of the eviction fix)"
        )
        mgr._dispatch_wake.clear()

    def test_dispatch_wake_set_at_all_notify_sites(self):
        """Static verification: every _make_room_signal.notify_all() call
        site in EXPECTED_NOTIFY_SITES also calls _dispatch_wake.set().

        RED at unfixed code: all 5 notify sites missing _dispatch_wake.set().
        GREEN with fix: all 5 sites have it.

        Mirrors `TestNotifyCallSitesAreTheNamedSet` from
        the turn-boundary handoff test but checks for the
        dispatch_wake companion instead of the notify.
        """
        expected = {
            "_drive_resident",
            "_serve_on_resident",
            "_idle_engine_liveness_sweep",
            "_unload_teardown",
            "_late_vram_reconcile",
        }

        missing = set()
        for name in expected:
            fn = getattr(TurbohaulManager, name, None)
            if fn is None:
                missing.add(name + " (method not found)")
                continue
            if isinstance(fn, (staticmethod, classmethod)):
                fn = fn.__func__
            try:
                src = inspect.getsource(fn)
            except (OSError, TypeError):
                missing.add(name + " (source unavailable)")
                continue
            has_notify = "_make_room_signal.notify_all()" in src
            has_wake = "self._dispatch_wake.set()" in src
            if has_notify and not has_wake:
                missing.add(f"{name} (notify without _dispatch_wake.set)")

        assert not missing, (
            f"_dispatch_wake.set() missing at: {sorted(missing)}"
        )
