"""Turn-boundary handoff for make-room starvation.

The root cause is visible at the turn boundary: nearly all starvation
events show BOTH residents mid-turn at the exact instant
make-room sampled -- a resident is only observably IDLE_EVICTABLE for a window governed entirely by
its own inbox state, and a plain fixed-backoff retry
(`asyncio.sleep(backoff_s)`, `_requeue_after_backoff`) polls blindly against that window instead of
being woken at the boundary.

The fix: one `asyncio.Condition` (`self._make_room_signal`)
sharing the existing `_registry_lock` (not wrapping/replacing it), one `notify_all()` at the single
place a resident newly becomes a make-room candidate -- the IDLE_EVICTABLE park in `_drive_resident`
-- and `_requeue_after_backoff`'s wait swapped from a plain sleep to
`asyncio.wait_for(self._make_room_signal.wait(), timeout=backoff_s)`, same budget as before. Touches
NOTHING inside `_lru_idle_unloadable`, the three admission-deferral functions, or the staleness functions -- a woken
(or timed-out) claimant only re-enqueues; the real eligibility/priority/staleness decision is always
re-made fresh, under lock, by the pre-existing, unmodified `_route_or_reserve` ->
`_lru_idle_unloadable` sequence.

Test arms (ten tests):
  1. TestParkFiresNotify            -- notify actually fires at a REAL park, driven end-to-end.
  2. TestDeferredRetryWakesEarly    -- a deferred slot re-enqueues well before its full backoff_s
                                        once a resident parks, driven end-to-end.
  3. TestLostNotificationTimeoutFallback -- the positive control for the timeout fallback: no park ever happens,
                                        `_wait_for_park_or_timeout` still resolves at (not before,
                                        not never) backoff_s, wrapped in a pytest-level outer
                                        timeout so a regression that hangs FAILS LOUDLY.
  4. TestNotifyBeforeWaitRace       -- the classic missed-wakeup condition-variable race: a notify
                                        fired before anyone is waiting must NOT be remembered.
  5. TestTwoClaimantsNoDoubleEvict  -- two deferred slots, one resident parks: exactly one claimant
                                        wins, the other correctly re-samples to eligible==0 (via
                                        the attribution instrument) and re-arms, no double-evict.
  6. TestStalenessProtectedNotEvictedViaNotify -- the staleness grant is re-checked fresh on every
                                        wake; a notify can never hand out a protected resident.
  7. TestNotifyCallSitesAreTheNamedSet -- structural: the SET of methods holding a
                                        notify_all() is exactly the named set, asserted BOTH
                                        directions. NOT a count -- a count cannot see a MOVED
                                        site. To add a wake site, EXTEND EXPECTED_NOTIFY_SITES;
                                        do not rename this class and do not adjust a number,
                                        because there is no longer one. Currently
                                        `_serve_on_resident` only, nowhere in the three admission-deferral functions or
                                        `_begin_unload_locked` (mirrors the attribution instrument's own wiring-count
                                        precedent).
  8. TestPriorityAdmitFromInboxStillAwaitFree -- regression guard: this design must not add a
                                        suspension point to a function whose whole contract is zero.
  9. TestRegistryLockReleasedDuringWait -- the deadlock-avoidance constraint: a claimant parked in
                                        the wait does not hold `_registry_lock`.
  10. TestLockOrderPreserved        -- `queue._lock` (touched by `enqueue_head`) is never acquired
                                        while `_registry_lock` is still held by the same coroutine.

Plus, TestTraverseLatency --
wake->win latency measured directly, HARD regime (continuously-fed incumbent --
inbox non-empty at park) vs EASY regime (inbox empty at park), 1 waiter and N waiters, every number
labelled with its regime. This class is the
executable evidence for any quoted latency numbers.

Caveat this suite CANNOT resolve, disclosed rather than implied: this measures the mechanism's own
latency floor inside a test process (fake spawn/health/sigterm/vram/complete, no real subprocess or
GPU I/O) -- it is a representative-shape measurement of the code path's own overhead, not a
guarantee of real production wall-clock. The subtraction cross-check
(full_cycle - backoff - attribution/log overhead, taken from logs) is the complementary
real-traffic reconciliation this suite cannot substitute for.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import statistics
import textwrap
import time
from unittest.mock import MagicMock, patch

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import (
    Resident,
    ResidentState,
    TurbohaulManager,
    _STALENESS_GRANT_THRESHOLD_S,
)
from turbohaul.slot import Slot
from turbohaul.subprocess_mgr import SidecarHandle


# ============================================================================
# Shared helpers -- mirror the precedent patterns in the busy-defer-budget test
# and the make-room eviction observability test rather than reinventing.
# ============================================================================


def _boot_runtime(
    tmp_path,
    *,
    max_parallel_sidecars=2,
    grace_seconds=0,
    max_grace_extensions=0,
    idle_hot_load_seconds=0,
    fastlane_rules=(),
    fastlane_enabled=None,
):
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
            safety_enabled=False,
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=grace_seconds,
            max_grace_extensions=max_grace_extensions,
            idle_hot_load_seconds=idle_hot_load_seconds,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(
            # Staleness protection is gated on this flag, so a
            # test that exercises it must switch the feature ON explicitly. When
            # the caller says nothing, `enabled` is still derived from the rule
            # list exactly as before.
            enabled=bool(fastlane_rules) if fastlane_enabled is None else fastlane_enabled,
            rules=[FastLaneRule(address=addr) for addr in fastlane_rules],
        ),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag, *, split_mode="none", main_gpu=0):
    """Co-residence needs split_mode='none' + a DISTINCT main_gpu per model
    (mirrors test_multislot_concurrency.py / the busy-defer-budget test's own helper)."""
    import yaml
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 0,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


def _fake_handle(model_tag: str, port: int, pid: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _high_vram():
    return patch(
        "turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000],
    )


def _make_fakes(pid_start: int, gates: dict):
    """gates: model_tag -> asyncio.Event the fake completion for that tag
    blocks on. A model_tag not in `gates` completes instantly."""
    pid = [pid_start]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        gate = gates.get(handle.model_tag)
        if gate is not None:
            await gate.wait()
        return {"ok": True, "model": handle.model_tag}

    return fake_spawn, fake_health, fake_sigterm, fake_vram, fake_complete


async def _wait_until(predicate, *, timeout=5.0, interval=0.005):
    """Poll predicate() until truthy or timeout; returns elapsed time on
    success, raises AssertionError on timeout. interval is deliberately much
    finer than any of the windows under test (~30ms HARD case) so it never
    becomes the bottleneck being measured."""
    t0 = time.monotonic()
    while True:
        if predicate():
            return time.monotonic() - t0
        if time.monotonic() - t0 > timeout:
            raise AssertionError(f"predicate never became true within {timeout}s")
        await asyncio.sleep(interval)


# ============================================================================
# 1 + 2 -- real end-to-end: notify fires at a genuine park, and the deferred
# retry wakes early rather than riding out a large artificial backoff.
# ============================================================================


@pytest.mark.asyncio
class TestParkFiresNotifyAndDeferredRetryWakesEarly:
    async def test_park_notifies_and_deferred_slot_wins_well_before_full_backoff(
        self, tmp_path,
    ):
        """Drives the REAL _drive_resident/_route_or_reserve/_requeue_after_backoff
        path: incumbent + other resident occupy the cap=2 count, a third distinct
        model is deferred (make-room-starved), the incumbent's turn is then let
        complete (a genuine park, notify fires), and the deferred slot must win
        (evict the incumbent) in well under the artificially large backoff_s used
        here -- proving the wake is real, not a coincidence of a short backoff.

        Mutant this kills: reverting `_wait_for_park_or_timeout` to plain
        `asyncio.sleep(backoff_s)` (undoing the whole feature) -- the deferred
        slot would then only win at or after the full (large) backoff_s, and this
        test's tight bound would fail.
        """
        boot, runtime = _boot_runtime(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=30,
        )
        _seed_manifest(boot, "incumbent", main_gpu=0)
        _seed_manifest(boot, "other", main_gpu=1)
        _seed_manifest(boot, "waiter", main_gpu=0)

        other_gate = asyncio.Event()  # "other" never completes -- never evictable
        incumbent_gate = asyncio.Event()
        fakes = _make_fakes(90000, {"other": other_gate, "incumbent": incumbent_gate})
        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fakes[0], health_fn=fakes[1], sigterm_fn=fakes[2],
            vram_fn=fakes[3], complete_fn=fakes[4],
        )
        mgr.runtime.queue.safety_enabled = False

        # Deliberately absurd backoff so a win can ONLY be explained by the
        # wake, never by the timeout fallback landing early by chance.
        HUGE_BACKOFF_S = 30.0
        with patch("turbohaul.manager._DISPATCH_DEFER_BACKOFF_S", HUGE_BACKOFF_S):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                t_inc = asyncio.create_task(
                    mgr.submit_and_wait("incumbent", "a", thread_id="t-inc")
                )
                t_other = asyncio.create_task(
                    mgr.submit_and_wait("other", "a", thread_id="t-other")
                )
                await _wait_until(
                    lambda: len(mgr._model_residents()) == 2
                    and all(r.state is ResidentState.ACTIVE for r in mgr._model_residents()),
                    timeout=5.0,
                )

                slot_w = await mgr.submit(
                    model_tag="waiter", prompt="a", thread_id="t-waiter",
                    wait_for_completion=True,
                )
                # Let it genuinely enter the deferred retry (busy regime: count-cap
                # full, no idle candidate) before releasing the incumbent.
                await _wait_until(
                    lambda: getattr(slot_w, "_dispatch_defer_count", 0) >= 1,
                    timeout=5.0,
                )

                t_release = time.monotonic()
                incumbent_gate.set()  # incumbent's turn completes -> parks -> notify
                won_after = await _wait_until(
                    lambda: "waiter" in {r.model_tag for r in mgr._model_residents()},
                    timeout=5.0,
                )

                assert won_after < 2.0, (
                    f"waiter won {won_after:.3f}s after release -- with a "
                    f"{HUGE_BACKOFF_S}s backoff and no wake mechanism this "
                    f"could only have happened by riding out {HUGE_BACKOFF_S}s; "
                    f"a fast win proves the notify-driven wake fired"
                )
                assert "incumbent" not in {
                    r.model_tag for r in mgr._model_residents()
                }, "incumbent must have been evicted to free room for waiter"

                other_gate.set()
                incumbent_gate.set()
                for t in (t_inc, t_other):
                    t.cancel()
            finally:
                other_gate.set()
                incumbent_gate.set()
                await mgr.shutdown()


# ============================================================================
# 3 + 4 -- unit-level, direct on _wait_for_park_or_timeout: the timeout
# fallback (the timeout-fallback positive control) and the missed-wakeup race.
# ============================================================================


@pytest.fixture
def boot_and_runtime(tmp_path):
    boot, runtime = _boot_runtime(tmp_path)
    return boot, runtime


class TestLostNotificationTimeoutFallback:
    @pytest.mark.asyncio
    async def test_no_park_ever_happens_resolves_at_backoff_not_before_not_never(
        self, boot_and_runtime,
    ):
        """The timeout fallback is the condition that matters most. Wrapped in an
        outer pytest-level timeout so a regression that removes the inner
        timeout entirely FAILS LOUDLY (an AssertionError/TimeoutError) instead
        of hanging the test process/CI.

        Mutant this kills: changing `timeout=backoff_s` to `timeout=None` in
        `_wait_for_park_or_timeout` -- the outer `asyncio.wait_for` below would
        then itself raise TimeoutError, which this test does not catch, so the
        test errors out instead of the graceful assertions below running.
        """
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        backoff_s = 0.15
        t0 = time.monotonic()
        await asyncio.wait_for(
            mgr._wait_for_park_or_timeout(backoff_s), timeout=backoff_s * 8,
        )
        elapsed = time.monotonic() - t0
        assert elapsed >= backoff_s * 0.8, (
            f"resolved too early ({elapsed:.3f}s) with no notify -- should "
            f"wait out ~{backoff_s}s"
        )
        assert elapsed < backoff_s * 4, (
            f"resolved suspiciously late ({elapsed:.3f}s) -- should be "
            f"bounded near {backoff_s}s, not drifting toward the outer bound"
        )


class TestNotifyBeforeWaitRace:
    @pytest.mark.asyncio
    async def test_notify_fired_before_anyone_waiting_is_not_remembered(
        self, boot_and_runtime,
    ):
        """The classic condition-variable missed-wakeup race, forced
        deterministically: notify BEFORE any waiter exists, then start
        waiting -- must fall back to the full backoff_s, not resolve
        instantly. This is exactly the semantics that
        a sticky asyncio.Event alternative would have broken.

        Mutant this kills: swapping the Condition-based implementation for a
        sticky asyncio.Event (`.set()`, never `.clear()`d) -- every
        subsequent wait would then return instantly regardless of whether
        real news happened, so `elapsed` here would collapse to ~0.
        """
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        async with mgr._make_room_signal:
            mgr._make_room_signal.notify_all()  # nobody is waiting yet

        backoff_s = 0.15
        t0 = time.monotonic()
        await asyncio.wait_for(
            mgr._wait_for_park_or_timeout(backoff_s), timeout=backoff_s * 8,
        )
        elapsed = time.monotonic() - t0
        assert elapsed >= backoff_s * 0.8, (
            f"a notify fired before anyone was waiting must NOT wake the "
            f"next wait early -- got {elapsed:.3f}s, which reads as the "
            f"stale notify being (wrongly) remembered"
        )


# ============================================================================
# 9 + 10 -- lock discipline: released during the wait; never held nested
# under queue._lock.
# ============================================================================


class TestRegistryLockReleasedDuringWait:
    @pytest.mark.asyncio
    async def test_registry_lock_not_held_while_genuinely_waiting(
        self, boot_and_runtime,
    ):
        """Deadlock-avoidance constraint: a claimant parked
        in the wait must never hold `_registry_lock`, or nothing else
        (e.g. a concurrent `_route_or_reserve`) could ever acquire it.

        Mutant this kills: replacing `Condition.wait()`'s call with a
        hand-rolled loop that keeps the lock held throughout (e.g. a bare
        `await asyncio.sleep(backoff_s)` INSIDE `async with
        self._make_room_signal:` without ever calling `.wait()`) -- the lock
        would then read locked() == True during the wait, failing the
        assertion below.
        """
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        waiter = asyncio.create_task(mgr._wait_for_park_or_timeout(2.0))
        await asyncio.sleep(0.05)  # let it genuinely enter .wait()
        assert not waiter.done()
        assert not mgr._registry_lock.locked(), (
            "_registry_lock must be released while a claimant is parked in "
            "the wait -- a held lock here would deadlock any concurrent "
            "acquirer (e.g. a real _route_or_reserve pass)"
        )
        # A concurrent acquirer must actually succeed while the waiter waits
        # -- not just read locked()==False, but genuinely get in.
        acquired = asyncio.Event()
        async def concurrent_acquirer():
            async with mgr._registry_lock:
                acquired.set()
                await asyncio.sleep(0.01)
        acq_task = asyncio.create_task(concurrent_acquirer())
        await asyncio.wait_for(acquired.wait(), timeout=1.0)
        await acq_task
        async with mgr._make_room_signal:
            mgr._make_room_signal.notify_all()
        await asyncio.wait_for(waiter, timeout=1.0)


class TestLockOrderPreserved:
    @pytest.mark.asyncio
    async def test_registry_lock_released_before_queue_lock_is_touched(
        self, boot_and_runtime, monkeypatch,
    ):
        """queue.py's contract: `queue._lock -> _registry_lock`,
        never reversed, never nested the other way. `enqueue_head` (called
        right after the wait, in `_requeue_after_backoff`) touches
        `queue._lock` -- this proves `_registry_lock` is NOT still held at
        that point, i.e. the two acquisitions are sequential, not nested.

        Mutant this kills: moving the `enqueue_head` call to inside
        `_wait_for_park_or_timeout`'s `async with self._make_room_signal:`
        block -- `_registry_lock` would then still be held (locked() ->
        True) when the spied `enqueue_head` fires, failing the assertion.
        """
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        slot = Slot.new("m")
        slot.completion_future = asyncio.get_event_loop().create_future()
        observed = {}
        real_enqueue_head = mgr.queue.enqueue_head

        async def spy_enqueue_head(s):
            observed["registry_lock_locked_at_enqueue"] = mgr._registry_lock.locked()
            return await real_enqueue_head(s)

        monkeypatch.setattr(mgr.queue, "enqueue_head", spy_enqueue_head)
        await mgr._requeue_after_backoff(slot, backoff_s=0.01)
        assert observed.get("registry_lock_locked_at_enqueue") is False, (
            "_registry_lock must be fully released before enqueue_head "
            "touches queue._lock -- holding both at once (in this order) "
            "approaches the queue._lock -> _registry_lock order from the "
            "forbidden direction"
        )


# ============================================================================
# 7 + 8 -- structural / source-inspection regression guards.
# ============================================================================


class TestNotifyCallSitesAreTheNamedSet:
    """A SECOND, INTENTIONAL notify call site
    exists in ``_serve_on_resident`` (grace-lapse -- natural deadline,
    ``grace_fastlane_breakout``, or ``grace_starvation_breakout``), on top
    of the park notify in ``_drive_resident``. This class replaces a
    count-based check (an exact number of notify call sites) -- the invariant
    that check protected
    ("no *accidental* extra call site") still matters, the exact count it
    hard-coded does not.

    ⚠ OBSERVABILITY IS ASYMMETRIC, AND THIS TEST IS LOAD-BEARING FOR THE GRACE-LAPSE SITE ALONE.
    It would be wrong to claim "both call sites are audited as
    SEPARATE events ... so a regression in either is independently observable
    in logs": only one of them is.
    In manager.py (the grace-lapse site), the ``_serve_on_resident`` wake does
    ``async with self._registry_lock: self._make_room_signal.notify_all()``
    followed by ``transition(slot, SlotState.POPPED)`` -- NO audit emission, no
    distinct event. The park site in ``_drive_resident`` has its own
    bookkeeping; the grace-lapse site has none.
    ⇒ A regression at the park site would surface in logs. A regression at the
    grace-lapse site would surface NOWHERE ELSE -- this structural assertion is the only
    guard on it. Claiming otherwise would OVERCLAIM the observability
    and UNDERSELL this test.

    ⛔ THIS ASSERTION IS A NAMED SET, NEVER A COUNT.
    A count-based check hard-codes a number (``== 1``, then ``== 2``, ...)
    and each new call site forces a RENAME of the test.
    Worse, a count is blind to a SWAPPED MEMBER: moving the notify out of
    ``_drive_resident`` and into some third method leaves the count at 2 and
    the test GREEN, while the park-side wake it exists to protect is gone.
    A count-based check is exposed to exactly that shape of failure,
    and a rename of the test also confuses tooling that diffs failure
    sets between runs (a rename reads as one test fixed plus one test
    regressed).

    So: assert WHICH methods hold a notify, both directions -- every named
    site present, and NO unnamed site anywhere on the class. Adding a third,
    redundant ``notify_all()`` (e.g. in the idle-timeout-direct-evict branch,
    which consumes a candidate rather than creating one) fails on the
    "unexpected" arm and NAMES the offending method. Moving one fails on the
    "missing" arm. Neither can hide inside a total.

    To add a wake site (park, grace-lapse, teardown-complete, ...),
    EXTEND ``EXPECTED_NOTIFY_SITES`` -- do not rename
    the test and do not adjust a number, because there is no longer one.

    A THIRD named site,
    ``_idle_engine_liveness_sweep`` -- a standalone cap>=2 background
    task (NOT ``_drive_resident``, NOT ``_serve_on_resident``) that detects
    an idle-hot resident's engine dying between requests and calls
    ``_begin_unload_locked`` + this same notify. Same "capacity
    released by a completed eviction, no immediate claimant" third-moment
    pattern already shipped inside ``_drive_resident`` -- this
    one just fires from a death it newly detects instead of an
    idle-timeout eviction, so it could not land inside an already-named
    method. ``_begin_unload_locked`` itself is still untouched.
    """

    #: The complete set of methods permitted to hold a
    #: ``_make_room_signal.notify_all()``. Extend deliberately; every entry
    #: is a design decision, not an observation of current code.
    #:
    #: ⚠ KNOWN LIMITATION -- MEASURED EMPTY TODAY, NOT STRUCTURALLY IMPOSSIBLE.
    #: The finder walks ``vars()`` and accepts only functions (unwrapping
    #: staticmethod/classmethod), which is a NARROWER domain than the
    #: whole-class source read this replaced: a notify inside a ``property``
    #: getter, or inside a non-function class attribute, would be invisible to
    #: the "unexpected site" arm.
    #: MEASURED on TurbohaulManager at the time of writing: 190 functions in ``vars()``,
    #: **0 property-decorated attributes**, non-function attribute types only
    #: {getset_descriptor, int, staticmethod, str}, and no property holds a
    #: notify. ⇒ The gap is EMPTY today.
    #: ⛔ But that is a property of the CURRENT CLASS SHAPE, not of the probe.
    #: If a ``property`` is ever added to this class, re-derive this note --
    #: "there is nothing to do here" and "we are choosing not to do it here"
    #: produce the same diff and completely different documentation.
    EXPECTED_NOTIFY_SITES = {
        "_drive_resident",      # park
        "_serve_on_resident",   # grace-lapse
        "_idle_engine_liveness_sweep",  # cap>=2 idle-engine-death
                                              # eviction -- same
                                              # third-moment pattern as
                                              # _drive_resident's two
                                              # existing sites
        "_unload_teardown",      # VRAM-CONFIRMED credit release.
                                 # The "teardown-complete" wake site
                                 # (see the docstring above) -- the real
                                 # moment-3 wake for a VRAM-over-commit eviction
                                 # and driver-death-reap, which
                                 # both spawn _unload_teardown from the SAME
                                 # single call site inside _begin_unload_locked.
                                 # Deliberately fires for EVERY
                                 # VRAM-credited teardown regardless of eviction
                                 # reason --
                                 # including count-cap and the three sites
                                 # above, redundantly but inertly.
        "_late_vram_reconcile",  # same moment, slow/timed-out
                                  # sub-case -- the initial 30s verify inside
                                  # _unload_teardown can miss the window; this
                                  # one-shot grace-delayed re-check is the
                                  # SAME "capacity confirmed released" moment
                                  # for that remainder. Without this site the
                                  # slow path silently degrades back to pure
                                  # timeout -- the exact violation of the capacity-release wake rule
                                  # that the other sites fix, just for a narrower case.
    }

    @staticmethod
    def _methods_holding_a_notify():
        """Every attribute on the class whose own source contains the call.

        Iterates ``vars()`` rather than ``inspect.getmembers`` so inherited
        and dynamically-attached attributes cannot silently widen the set,
        and skips non-functions explicitly instead of swallowing errors --
        a bare ``except`` here would turn "could not read the source" into
        "no notify found", which is the empty-result-as-evidence trap.
        """
        found = set()
        for name, attr in vars(TurbohaulManager).items():
            fn = attr.__func__ if isinstance(attr, (staticmethod, classmethod)) else attr
            if not (inspect.isfunction(fn) or inspect.iscoroutinefunction(fn)):
                continue
            if "_make_room_signal.notify_all()" in inspect.getsource(fn):
                found.add(name)
        return found

    def test_notify_all_call_sites_are_exactly_the_named_set(self):
        found = self._methods_holding_a_notify()
        unexpected = found - self.EXPECTED_NOTIFY_SITES
        missing = self.EXPECTED_NOTIFY_SITES - found
        assert not unexpected, (
            f"UNNAMED _make_room_signal.notify_all() call site(s): "
            f"{sorted(unexpected)} -- every wake site is a design decision "
            f"(see the class docstring). Add it to EXPECTED_NOTIFY_SITES with a justification, or "
            f"remove the call."
        )
        assert not missing, (
            f"MISSING _make_room_signal.notify_all() in {sorted(missing)} -- "
            f"a named wake site lost its notify. A count-based assertion "
            f"would not have seen this if another site gained one."
        )

    def test_the_probe_itself_can_see_a_notify(self):
        """Green control: the finder is proven capable of a NON-EMPTY answer.

        Without this, a probe that silently found nothing (a changed literal,
        an unreadable source) would satisfy the "no unexpected sites" arm
        vacuously and report a clean pass over a class with no wakes at all.
        """
        assert self._methods_holding_a_notify(), (
            "the finder returned an EMPTY set -- it is not measuring "
            "anything, so both arms above are vacuous"
        )

    def test_notify_all_is_inside_drive_resident_and_serve_on_resident_only(self):
        drive_resident_src = inspect.getsource(TurbohaulManager._drive_resident)
        serve_on_resident_src = inspect.getsource(TurbohaulManager._serve_on_resident)
        assert "_make_room_signal.notify_all()" in drive_resident_src
        assert "_make_room_signal.notify_all()" in serve_on_resident_src
        for name in (
            "_priority_admit_from_inbox",
            "_begin_unload_locked",
            "_should_defer_admission",
            "_defer_admission_window",
            "_maybe_defer_admission",
            "_lru_idle_unloadable",
        ):
            fn_src = inspect.getsource(getattr(TurbohaulManager, name))
            assert "_make_room_signal.notify_all()" not in fn_src, (
                f"notify_all() must not appear inside {name} -- it would be "
                f"a rejected design alternative / a touch to protected logic"
            )


class TestPriorityAdmitFromInboxStillAwaitFree:
    def test_no_await_expression_anywhere_in_the_function_body(self):
        """AST-based, not substring-based: the function's own docstring
        contains the literal word 'await' in prose, so a naive text search
        would false-positive on the comment, not the code. Walking the
        parsed AST for real `ast.Await` nodes checks the actual code.

        Mutant this kills: inserting a spurious `await asyncio.sleep(0)`
        anywhere in the function body -- `awaits` would become non-empty.
        """
        src = textwrap.dedent(inspect.getsource(TurbohaulManager._priority_admit_from_inbox))
        tree = ast.parse(src)
        awaits = [n for n in ast.walk(tree) if isinstance(n, ast.Await)]
        assert awaits == [], (
            f"_priority_admit_from_inbox must have zero suspension points "
            f"-- found {len(awaits)}"
        )


# ============================================================================
# 5 -- two claimants, one victim: exactly one wins, the loser re-samples
# correctly (via the attribution instrument) and re-arms, no double-evict.
# ============================================================================


@pytest.mark.asyncio
class TestTwoClaimantsNoDoubleEvict:
    async def test_two_deferred_slots_one_park_exactly_one_wins_loser_resamples(
        self, tmp_path,
    ):
        """Mutant this kills: having the LOSING claimant act on a
        remembered/stale victim reference instead of re-running
        `_lru_idle_unloadable` fresh on its own next attempt -- it would then
        attempt to evict an already-gone resident (crash / wrong state)
        instead of correctly finding `eligible == 0` and re-deferring.
        """
        boot, runtime = _boot_runtime(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=30,
        )
        _seed_manifest(boot, "incumbent", main_gpu=0)
        _seed_manifest(boot, "other", main_gpu=1)
        _seed_manifest(boot, "waiter-a", main_gpu=0)
        _seed_manifest(boot, "waiter-b", main_gpu=0)

        other_gate = asyncio.Event()
        incumbent_gate = asyncio.Event()
        fakes = _make_fakes(91000, {"other": other_gate, "incumbent": incumbent_gate})
        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fakes[0], health_fn=fakes[1], sigterm_fn=fakes[2],
            vram_fn=fakes[3], complete_fn=fakes[4],
        )
        mgr.runtime.queue.safety_enabled = False

        with patch("turbohaul.manager._DISPATCH_DEFER_BACKOFF_S", 20.0):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                t_inc = asyncio.create_task(
                    mgr.submit_and_wait("incumbent", "a", thread_id="t-inc")
                )
                t_other = asyncio.create_task(
                    mgr.submit_and_wait("other", "a", thread_id="t-other")
                )
                await _wait_until(
                    lambda: len(mgr._model_residents()) == 2
                    and all(r.state is ResidentState.ACTIVE for r in mgr._model_residents()),
                    timeout=5.0,
                )

                slot_a = await mgr.submit(
                    model_tag="waiter-a", prompt="a", thread_id="t-wa",
                    wait_for_completion=True,
                )
                slot_b = await mgr.submit(
                    model_tag="waiter-b", prompt="a", thread_id="t-wb",
                    wait_for_completion=True,
                )
                await _wait_until(
                    lambda: getattr(slot_a, "_dispatch_defer_count", 0) >= 1
                    and getattr(slot_b, "_dispatch_defer_count", 0) >= 1,
                    timeout=5.0,
                )

                incumbent_gate.set()
                await _wait_until(
                    lambda: "waiter-a" in {r.model_tag for r in mgr._model_residents()}
                    or "waiter-b" in {r.model_tag for r in mgr._model_residents()},
                    timeout=5.0,
                )

                live_tags = {r.model_tag for r in mgr._model_residents()}
                winners = live_tags & {"waiter-a", "waiter-b"}
                assert len(winners) == 1, (
                    f"exactly one of waiter-a/waiter-b should have won the "
                    f"single freed resident; live tags: {live_tags}"
                )
                assert "incumbent" not in live_tags
                assert "other" in live_tags, "other was never evictable (busy_gate held)"

                loser_slot = slot_b if "waiter-a" in winners else slot_a
                # The loser must still be alive and correctly re-deferring
                # (proof it re-ran _lru_idle_unloadable fresh and found
                # nothing, not that it crashed or silently vanished).
                assert not loser_slot.completion_future.done() or (
                    loser_slot.completion_future.done()
                    and loser_slot.completion_future.exception() is None
                ), "loser must not have failed/crashed -- it should simply re-defer"

                other_gate.set()
                for t in (t_inc, t_other):
                    t.cancel()
            finally:
                other_gate.set()
                incumbent_gate.set()
                await mgr.shutdown()


# ============================================================================
# 6 -- staleness interaction: a staleness-protected resident can never be handed
# out via the notify path, because the notify never bypasses
# _lru_idle_unloadable's own fresh re-check.
# ============================================================================


class TestStalenessProtectedNotEvictedViaNotify:
    @pytest.mark.asyncio
    async def test_park_and_notify_of_a_staleness_protected_resident_yields_no_victim(
        self, tmp_path,
    ):
        """Direct, deterministic proof at the level the design's decision
        actually depends on: a resident inside its own staleness-
        protection window is EXCLUDED by `_lru_idle_unloadable` regardless of
        whether the re-check was triggered by a notify or a timeout -- the
        notify only changes WHEN the check runs, never WHAT it decides.

        Mutant this kills: implementing the REJECTED alternative
        (targeted hand-off) -- i.e. having the wake path evict the resident
        that fired the notify directly, bypassing `_lru_idle_unloadable` --
        would evict `r` here despite its live grant, failing the assertion
        that it survives.
        """
        # Not the shared `boot_and_runtime` fixture, which builds Fast
        # Lane OFF. Staleness protection is gated on the feature flag, so this test must turn
        # the feature ON to keep testing the staleness clause it is named for -- with
        # the feature off it would assert staleness protection on a deployment that never
        # enabled Fast Lane, which would test nothing meaningful.
        boot, runtime = _boot_runtime(tmp_path, fastlane_enabled=True)
        mgr = TurbohaulManager(boot, runtime)
        r = Resident(
            model_tag="protected-model",
            resident_key="protected-model",
            state=ResidentState.IDLE_EVICTABLE,
            idle_client_meta={"ip": "5.5.5.5"},
        )
        mgr._residents["protected-model"] = r
        # Grant recorded just past threshold -- currently protected (mirrors
        # the staleness-grant test's own TestGrantIsGranted pattern).
        from turbohaul.manager import _fastlane_census_key
        mgr._fastlane_staleness_grants[_fastlane_census_key("5.5.5.5")] = (
            time.monotonic() - (_STALENESS_GRANT_THRESHOLD_S + 1.0)
        )

        async with mgr._registry_lock:
            # Simulate the park's own notify firing (the real call site is
            # manager.py's _drive_resident; this proves the DOWNSTREAM
            # decision -- what a woken _route_or_reserve pass would find --
            # is unaffected by who fired the notify).
            mgr._make_room_signal.notify_all()

        victim = mgr._lru_idle_unloadable()
        assert victim is None, (
            "the ONLY resident in the pool is staleness-protected -- "
            "_lru_idle_unloadable must still return None after a notify, "
            "exactly as it would on a timeout-triggered re-check"
        )
        assert "protected-model" in mgr._residents, (
            "the protected resident must still be alive -- a notify must "
            "never bypass the staleness check to evict it directly"
        )

    @pytest.mark.asyncio
    async def test_end_to_end_real_park_wakes_waiter_who_still_skips_a_protected_sibling(
        self, tmp_path,
    ):
        """The end-to-end sibling of the test above -- that one proves the
        DECISION is right in isolation; this one drives the REAL park site
        (`_drive_resident`, manager.py) for real and confirms the resulting
        wake still respects a DIFFERENT resident's live staleness grant. Needed
        because the unit test above never executes `_drive_resident` at all,
        so a mutant that adds an unconditional evict AT THE PARK SITE ITSELF
        (rather than going through `_lru_idle_unloadable`) would not be
        caught by it.

        Design note on why `protected` is a SEPARATE resident from the one
        that actually parks, not the same one: the staleness rule
        (`_spend_staleness_grant_if_completed`) spends a grant the INSTANT
        the identity holding it completes a turn -- so a resident cannot
        simultaneously be "the one whose turn just completed" (about to
        park) and "still holding a live grant from that same turn". Testing
        staleness protection against the resident that fires the real notify would
        therefore test a self-contradictory state. `protected` is instead
        constructed directly (bypassing `_drive_resident` entirely, exactly
        like the unit test above) so nothing ever spends its grant, while
        `incumbent` is the one driven for real and is the legitimate,
        unprotected eviction target once it parks.

        Mutant this kills: the park site calling `_begin_unload_locked`
        unconditionally on EVERY currently-idle-evictable resident (or on
        `protected` specifically) instead of going through
        `_lru_idle_unloadable`'s real filter -- `protected` would then be
        evicted here despite its live grant.
        """
        boot, runtime = _boot_runtime(
            # fastlane_enabled=True -- staleness protection is gated on the
            # feature flag, and this test's whole subject is a live staleness grant.
            tmp_path, max_parallel_sidecars=3, idle_hot_load_seconds=30,
            fastlane_enabled=True,
        )
        _seed_manifest(boot, "protected", main_gpu=0)
        _seed_manifest(boot, "incumbent", main_gpu=1)
        _seed_manifest(boot, "other", main_gpu=0)
        _seed_manifest(boot, "waiter", main_gpu=0)

        other_gate = asyncio.Event()
        incumbent_gate = asyncio.Event()
        fakes = _make_fakes(93000, {"other": other_gate, "incumbent": incumbent_gate})
        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fakes[0], health_fn=fakes[1], sigterm_fn=fakes[2],
            vram_fn=fakes[3], complete_fn=fakes[4],
        )
        mgr.runtime.queue.safety_enabled = False

        # `protected`: inserted directly, IDLE_EVICTABLE, with a live grant
        # -- never driven, so nothing ever spends it during this test.
        protected = Resident(
            model_tag="protected", resident_key="protected",
            state=ResidentState.IDLE_EVICTABLE, idle_client_meta={"ip": "7.7.7.7"},
            main_gpu=0, split_mode="none",
        )
        mgr._residents["protected"] = protected
        from turbohaul.manager import _fastlane_census_key
        mgr._fastlane_staleness_grants[_fastlane_census_key("7.7.7.7")] = (
            time.monotonic() - (_STALENESS_GRANT_THRESHOLD_S + 1.0)
        )

        try:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            t_inc = asyncio.create_task(
                mgr.submit_and_wait("incumbent", "a", thread_id="t-inc")
            )
            t_other = asyncio.create_task(
                mgr.submit_and_wait("other", "a", thread_id="t-other")
            )
            await _wait_until(
                lambda: len(mgr._model_residents()) == 3
                and mgr._residents.get("incumbent") is not None
                and mgr._residents["incumbent"].state is ResidentState.ACTIVE
                and mgr._residents.get("other") is not None
                and mgr._residents["other"].state is ResidentState.ACTIVE,
                timeout=5.0,
            )
            assert mgr._fastlane_staleness_protects(protected) is True, (
                "test setup problem, not proof of anything -- the grant "
                "must read as live before the scenario proceeds"
            )

            slot_w = await mgr.submit(
                model_tag="waiter", prompt="a", thread_id="t-waiter",
                wait_for_completion=True,
            )
            await _wait_until(
                lambda: getattr(slot_w, "_dispatch_defer_count", 0) >= 1, timeout=5.0,
            )

            incumbent_gate.set()  # incumbent's turn completes -> REAL park -> REAL notify
            await _wait_until(
                lambda: "incumbent" not in {r.model_tag for r in mgr._model_residents()},
                timeout=5.0,
            )

            live_tags = {r.model_tag for r in mgr._model_residents()}
            assert "protected" in live_tags, (
                "the staleness-protected sibling must still be alive -- a "
                "real park+notify elsewhere must never sweep it up"
            )
            assert "waiter" in live_tags, (
                "the waiter should legitimately have won -- incumbent, "
                "once parked, was a genuine unprotected candidate"
            )

            for t in (t_inc, t_other):
                t.cancel()
        finally:
            other_gate.set()
            incumbent_gate.set()
            await mgr.shutdown()


# ============================================================================
# Traverse latency, HARD vs EASY regime, 1 waiter and N,
# distribution reported.
# ============================================================================


@pytest.mark.asyncio
class TestTraverseLatency:
    """Not a pass/fail correctness gate in the usual sense -- these tests
    assert only sane bounds (the mechanism completes at all, within a
    generous ceiling) so they stay GREEN regardless of the actual numbers;
    the numbers themselves are printed to stdout, to be collected by a
    separate run rather than
    hard-asserted here, per the "report, don't gate on a specific
    ms figure picked arbitrarily" spirit -- the verdict rule (p50 in/out of 30ms)
    is applied by a human reading the output, not by this test suite.
    """

    async def _one_trial(self, tmp_path_factory, *, hard_regime: bool, n_waiters: int):
        tmp_path = tmp_path_factory.mktemp("traverse")
        boot, runtime = _boot_runtime(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=30,
        )
        _seed_manifest(boot, "incumbent", main_gpu=0)
        _seed_manifest(boot, "other", main_gpu=1)
        for i in range(n_waiters):
            _seed_manifest(boot, f"waiter-{i}", main_gpu=0)

        other_gate = asyncio.Event()
        incumbent_gate = asyncio.Event()
        fakes = _make_fakes(
            92000 + n_waiters, {"other": other_gate, "incumbent": incumbent_gate}
        )
        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fakes[0], health_fn=fakes[1], sigterm_fn=fakes[2],
            vram_fn=fakes[3], complete_fn=fakes[4],
        )
        mgr.runtime.queue.safety_enabled = False
        try:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            t_inc = asyncio.create_task(
                mgr.submit_and_wait("incumbent", "a", thread_id="ti")
            )
            t_other = asyncio.create_task(
                mgr.submit_and_wait("other", "a", thread_id="to")
            )
            await _wait_until(
                lambda: len(mgr._model_residents()) == 2
                and all(r.state is ResidentState.ACTIVE for r in mgr._model_residents()),
                timeout=5.0,
            )

            waiter_slots = []
            for i in range(n_waiters):
                s = await mgr.submit(
                    model_tag=f"waiter-{i}", prompt="a", thread_id=f"tw{i}",
                    wait_for_completion=True,
                )
                waiter_slots.append(s)
            await _wait_until(
                lambda: all(
                    getattr(s, "_dispatch_defer_count", 0) >= 1 for s in waiter_slots
                ),
                timeout=5.0,
            )

            if hard_regime:
                # HARD (continuously-fed incumbent): incumbent's
                # inbox is NON-EMPTY the instant it parks -- queue a same-
                # model follow-up now, before releasing the gate, so the
                # follow-up is already sitting in r.inbox at park time.
                t_followup = asyncio.create_task(
                    mgr.submit_and_wait("incumbent", "b", thread_id="ti-followup")
                )
                await _wait_until(
                    lambda: (
                        (r := mgr._residents.get("incumbent")) is not None
                        and r.inbox is not None
                        and not r.inbox.empty()
                    ),
                    timeout=5.0,
                )
            else:
                t_followup = None  # EASY: nothing queued behind it

            t_release = time.monotonic()
            incumbent_gate.set()
            # ⭐ THE TRAVERSE ENDPOINT, by definition
            # ("wake -> reacquire _registry_lock -> enqueue_head -> dispatcher
            # pop -> _route_or_reserve -> _lru_idle_unloadable") is the MAKE-ROOM
            # EVICTION DECISION -- i.e. the instant `_begin_unload_locked` removes
            # `incumbent` from `mgr._residents`, which happens SYNCHRONOUSLY
            # inside the same _registry_lock block as `_lru_idle_unloadable`'s
            # own call. It is NOT the moment the winning model finishes being
            # spawned + health-checked (`_reserve_and_start_locked`) -- that is
            # a separate, later pipeline this test's fake spawn/health chain
            # has its own overhead in, and measuring THAT instead would silently
            # inflate every number here with test-harness spawn latency that
            # has nothing to do with the wake->win traverse under test. Measuring
            # the winner's appearance instead gives
            # nearly identical numbers in BOTH regimes -- a sign
            # the real traverse is
            # being swamped by a shared, unrelated cost.
            elapsed = await _wait_until(
                lambda: "incumbent" not in {r.model_tag for r in mgr._model_residents()},
                timeout=10.0,
                interval=0.001,
            )
            # Separately, confirm exactly one waiter EVENTUALLY gets served
            # (full spawn included) -- a correctness check, not the timed
            # quantity.
            await _wait_until(
                lambda: any(
                    f"waiter-{i}" in {r.model_tag for r in mgr._model_residents()}
                    for i in range(n_waiters)
                ),
                timeout=10.0,
            )
            live_tags = {r.model_tag for r in mgr._model_residents()}
            won = [i for i in range(n_waiters) if f"waiter-{i}" in live_tags]

            if t_followup is not None:
                incumbent_gate.set()  # let the follow-up (2nd incumbent turn) drain too
                await asyncio.sleep(0)
                try:
                    await asyncio.wait_for(t_followup, timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    t_followup.cancel()

            return {
                "traverse_s": elapsed,
                "n_won": len(won),
                "n_waiters": n_waiters,
                "hard_regime": hard_regime,
            }
        finally:
            other_gate.set()
            incumbent_gate.set()
            for t in (t_inc, t_other):
                t.cancel()
            await mgr.shutdown()

    async def test_hard_regime_single_waiter_distribution(self, tmp_path_factory):
        """HARD regime (continuously-fed incumbent), 1 waiter, N_TRIALS
        repetitions -- prints p50/p95/max to stdout (captured by the
        harness run) and asserts only a
        generous sanity ceiling so this stays green as a regression guard
        regardless of the exact numbers the verdict rule will read."""
        N_TRIALS = 20
        samples = []
        for _ in range(N_TRIALS):
            r = await self._one_trial(tmp_path_factory, hard_regime=True, n_waiters=1)
            assert r["n_won"] == 1
            samples.append(r["traverse_s"])
        samples.sort()
        p50 = statistics.median(samples)
        p95 = samples[int(0.95 * (len(samples) - 1))]
        print(
            f"\n[TRAVERSE][HARD][n_waiters=1][n_trials={N_TRIALS}] "
            f"p50={p50*1000:.2f}ms p95={p95*1000:.2f}ms "
            f"max={max(samples)*1000:.2f}ms min={min(samples)*1000:.2f}ms"
        )
        assert max(samples) < 5.0, (
            "sanity ceiling only -- a multi-second traverse would indicate "
            "the mechanism is broken outright, not merely slow"
        )

    async def test_easy_regime_single_waiter_distribution(self, tmp_path_factory):
        """EASY regime (intermittent incumbent, inbox empty at park) --
        contrast case. This regime's window is SECONDS
        long, so a traverse measured only here would pass trivially and
        prove nothing about the HARD deadline; kept as a labelled contrast,
        never substituted for the HARD-regime numbers above."""
        N_TRIALS = 10
        samples = []
        for _ in range(N_TRIALS):
            r = await self._one_trial(tmp_path_factory, hard_regime=False, n_waiters=1)
            assert r["n_won"] == 1
            samples.append(r["traverse_s"])
        samples.sort()
        p50 = statistics.median(samples)
        print(
            f"\n[TRAVERSE][EASY][n_waiters=1][n_trials={N_TRIALS}] "
            f"p50={p50*1000:.2f}ms max={max(samples)*1000:.2f}ms"
        )
        assert max(samples) < 5.0

    async def test_hard_regime_n_waiters_distribution(self, tmp_path_factory):
        """HARD regime, N=3 simultaneous claimants -- the Nth serially-woken
        claimant may structurally miss a window the 1st hits (a known
        caveat); this reports the WINNER's traverse distribution under
        contention, labelled by n_waiters so it is never conflated with the
        n_waiters=1 numbers above."""
        N_TRIALS = 12
        N_WAITERS = 3
        samples = []
        for _ in range(N_TRIALS):
            r = await self._one_trial(
                tmp_path_factory, hard_regime=True, n_waiters=N_WAITERS,
            )
            assert r["n_won"] == 1, "exactly one of the N should win the single victim"
            samples.append(r["traverse_s"])
        samples.sort()
        p50 = statistics.median(samples)
        p95 = samples[int(0.95 * (len(samples) - 1))]
        print(
            f"\n[TRAVERSE][HARD][n_waiters={N_WAITERS}][n_trials={N_TRIALS}] "
            f"p50={p50*1000:.2f}ms p95={p95*1000:.2f}ms "
            f"max={max(samples)*1000:.2f}ms min={min(samples)*1000:.2f}ms"
        )
        assert max(samples) < 5.0
