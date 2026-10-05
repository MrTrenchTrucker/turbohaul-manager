"""Fast Lane make-room gate: a decision-moment wake, not a timer, drives the advance.

The make-room eviction GATE must advance a parked Fast Lane request on the
DECISION-MOMENT WAKE (the registry-lock Condition `_make_room_signal`), with the
backoff timeout as a LOST-WAKE BACKSTOP ONLY -- never the routine mechanism.

A poll-only gate (a decoration on a 0.05s poll) would fail this contract:
`_wait_for_park_or_timeout` would return None and not distinguish a wake from a
timeout, and `_requeue_after_backoff` would re-enqueue unconditionally. If
the timeout is what routinely advances the request, the gate is not
implemented -- so the wake-vs-timeout distinction that makes the wake the
load-bearing signal would be MISSING. That is exactly what these tests pin down.

Fails before the change (the wait returns None): the `woken is True/False`
distinction is missing -- the wait resolves to None, so the wake-driven
assertions fail.
Passes after the change: `_wait_for_park_or_timeout` returns a bool (`woken`) distinguishing
the decision-moment WAKE (True, the routine advance) from the LOST-WAKE
BACKSTOP (False), and `_requeue_after_backoff` re-enqueues on the
wake as the routine advance (still on the backstop so it can never hang).

The wake is real, not decoration on an un-paired notify: every
`_make_room_signal.notify_all()` site in manager.py is paired with
`_dispatch_wake.set()` (a source-scan, extending the existing scan in
the admission-pump stall test).
"""
import asyncio
import ast
import inspect
import time

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
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot, SlotState
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


async def _await_parked_on_make_room(mgr, *, timeout=2.0) -> bool:
    """Return True once a waiter is registered on `_make_room_signal`.

    Polls the Condition's pending-waiters list (CPython: `Condition._waiters`)
    so the test fires `notify_all()` ONLY after the waiter has actually entered
    `Condition.wait()` -- eliminating the classic notify-before-wait lost-wake
    race. Falls back to a plain yield if the private attribute is absent.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waiters = getattr(mgr._make_room_signal, "_waiters", None)
        if waiters:  # non-empty => a coroutine is parked in Condition.wait()
            return True
        await asyncio.sleep(0)
    return False


def _fire_make_room_wake(mgr):
    """The THIRD decision moment: capacity released by a completed
    eviction / a resident parking to IDLE_EVICTABLE / a grace lapsing. Modeled
    with the EXACT primitive the production notify sites use -- notify_all()
    under `_registry_lock`, paired with `_dispatch_wake.set()`."""

    async def _do():
        async with mgr._registry_lock:
            mgr._dispatch_wake.set()
            mgr._make_room_signal.notify_all()

    return _do()


class TestFastLaneGateWakeDrivesAdvance:
    """The wake is the routine advance, the timeout a backstop."""

    @pytest.mark.asyncio
    async def test_wait_woken_on_decision_moment_wake_not_timer(self, mgr):
        """THE key assertion: a parked request advances on the
        WAKE, not the backoff. With a very LARGE backoff (30s -- far beyond any
        reasonable wall-clock bound), firing the make-room WAKE resolves the wait
        promptly and reports `woken=True`: the timeout did NOT drive it.

        Fails before the change: `_wait_for_park_or_timeout` returns None (no distinction) ->
        `woken is True` fails. Passes after: returns True.
        """
        t0 = time.monotonic()
        waiter = asyncio.create_task(mgr._wait_for_park_or_timeout(30.0))
        assert await _await_parked_on_make_room(mgr), (
            "waiter never parked on _make_room_signal -- test harness failure"
        )

        await _fire_make_room_wake(mgr)
        woken = await asyncio.wait_for(waiter, timeout=5.0)
        elapsed = time.monotonic() - t0

        assert woken is True, (
            "the wait must be advanced by the decision-moment "
            "WAKE (woken is True); the backoff is a backstop, not the "
            "mechanism. A None return means the wake is decoration. got {!r}".format(
                woken
            )
        )
        # Promptness vs the 30s backoff is the behavioral proof that the WAKE,
        # not the timer, advanced the request (the timer must not be the
        # routine mechanism).
        assert elapsed < 1.0, (
            f"VIOLATION: request advanced via the 30s timer, not the "
            f"wake (elapsed={elapsed:.3f}s)"
        )
        mgr._dispatch_wake.clear()

    @pytest.mark.asyncio
    async def test_wait_woken_false_on_lost_wake_backstock(self, mgr):
        """A LOST WAKE degrades to the backstop and does NOT hang.
        No wake fires -> the backstop resolves the wait and reports
        `woken=False` (the backstop, never the routine advance).

        Fails before the change (returns None -> `woken is False` fails); passes after (False).
        """
        t0 = time.monotonic()
        woken = await asyncio.wait_for(
            mgr._wait_for_park_or_timeout(0.2), timeout=5.0
        )
        elapsed = time.monotonic() - t0

        assert woken is False, (
            "with no wake the backstop must resolve the wait and "
            "report woken=False (lost-wake degrades, never hangs); got {!r}"
            .format(woken)
        )
        assert elapsed >= 0.15, (
            f"wait returned early ({elapsed:.3f}s) without a wake -- the "
            f"backstop never elapsed; lost-wake degradation is broken"
        )
        assert elapsed < 1.0, f"VIOLATION: lost-wake backstop hung ({elapsed:.3f}s)"

    @pytest.mark.asyncio
    async def test_busy_path_wake_still_distinguished_from_backstock(self, mgr):
        """The production busy-path backoff is _DISPATCH_DEFER_BACKOFF_S (0.05s).
        Even there the WAKE must be distinguished from the 0.05s backstop:
        firing the wake reports `woken=True` before the 0.05s elapses; NOT firing
        anything reports `woken=False` at the backstop.

        Fails before the change (returns None on both): both assertions fail. Passes after.
        """
        from turbohaul.manager import _DISPATCH_DEFER_BACKOFF_S

        # (a) wake fires promptly -> woken True (routine advance).
        waiter = asyncio.create_task(
            mgr._wait_for_park_or_timeout(_DISPATCH_DEFER_BACKOFF_S)
        )
        assert await _await_parked_on_make_room(mgr)
        await _fire_make_room_wake(mgr)
        woken_a = await asyncio.wait_for(waiter, timeout=5.0)
        assert woken_a is True, (
            "busy-path wake must be the routine advance "
            "(woken=True); None means the wake is not distinguished. got {!r}".format(woken_a)
        )
        mgr._dispatch_wake.clear()

        # (b) no wake -> backstop (0.05s) elapses -> woken False (not hang).
        woken_b = await asyncio.wait_for(
            mgr._wait_for_park_or_timeout(_DISPATCH_DEFER_BACKOFF_S), timeout=5.0
        )
        assert woken_b is False, (
            "busy-path with no wake degrades to the backstop "
            "(woken=False); None means the wake is not distinguished. got {!r}".format(woken_b)
        )

    @pytest.mark.asyncio
    async def test_requeue_advances_on_wake_without_timer_elapsing(self, mgr):
        """Behavioral safety net: the full
        `_requeue_after_backoff` re-enqueues the slot at the staging HEAD PROMPTLY
        when the wake fires -- even with a 30s backoff -- proving the wake, not
        the timer, performed the re-enqueue.

        (This is already true on the wake without the change; the load-bearing distinction
        itself is pinned by the `woken` assertions above. This guards the
        end-to-end re-enqueue contract so the change never drops a slot.)"""
        slot = Slot.new(model_tag="m1", prompt="hi", thread_id="t1")
        slot.state = SlotState.STAGED  # re-queueing an already-staged slot
        t0 = time.monotonic()
        driver = asyncio.create_task(
            mgr._requeue_after_backoff(slot, backoff_s=30.0)
        )
        assert await _await_parked_on_make_room(mgr)
        await _fire_make_room_wake(mgr)
        await asyncio.wait_for(driver, timeout=5.0)
        elapsed = time.monotonic() - t0

        assert mgr.queue._staging and mgr.queue._staging[0] is slot, (
            "the parked request must be re-enqueued at the head by "
            "the wake (30s backoff must NOT have elapsed)"
        )
        assert elapsed < 1.0, (
            f"VIOLATION: re-enqueue driven by the 30s timer, not the wake "
            f"(elapsed={elapsed:.3f}s)"
        )
        mgr._dispatch_wake.clear()

    @pytest.mark.asyncio
    async def test_requeue_lost_wake_backstock_never_hangs(self, mgr):
        """Behavioral safety: with NO wake, the
        backstop still re-enqueues the slot within the backstop interval --
        the request can never hang."""
        slot = Slot.new(model_tag="m1", prompt="hi", thread_id="t1")
        slot.state = SlotState.STAGED
        t0 = time.monotonic()
        await asyncio.wait_for(
            mgr._requeue_after_backoff(slot, backoff_s=0.2), timeout=5.0
        )
        elapsed = time.monotonic() - t0

        assert mgr.queue._staging and mgr.queue._staging[0] is slot, (
            "a lost wake must still re-enqueue (backstop) so it never hangs"
        )
        assert elapsed < 1.0, (
            f"VIOLATION: lost-wake backstop stalled ({elapsed:.3f}s)"
        )

    def test_notify_sites_paired_with_dispatch_wake(self):
        """Source scan (checkable claim): every `_make_room_signal.notify_all()` call
        site in TurbohaulManager lives in a method that ALSO calls
        `_dispatch_wake.set()` -- so the make-room WAKE is not decoration on an
        un-paired notify.

        Extends `TestNotifyCallSitesAreTheNamedSet` /
        `test_dispatch_wake_set_at_all_notify_sites` (in the
        admission-pump stall test), which checks a fixed named set of
        5 methods. This scan walks EVERY method and flags any notify_all()
        without a paired dispatch_wake.set().
        """
        offenders = []
        for name, fn in inspect.getmembers(TurbohaulManager, inspect.isfunction):
            if not fn.__qualname__.startswith("TurbohaulManager."):
                continue  # inherited (e.g. object.*)
            if "<locals>" in fn.__qualname__:
                continue  # nested helper, not a method
            try:
                src = inspect.getsource(fn)
            except (OSError, TypeError):
                continue
            if "_make_room_signal.notify_all()" not in src:
                continue
            if "_dispatch_wake.set()" not in src:
                offenders.append(name)
        assert not offenders, (
            "_make_room_signal.notify_all() WITHOUT a paired "
            "_dispatch_wake.set() -- the wake would be decoration on an "
            "un-paired notify: { offenders}".format(offenders=offenders)
        )
