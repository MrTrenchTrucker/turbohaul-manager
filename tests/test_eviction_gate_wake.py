"""The eviction decision is a GATE,
not a polling loop.

"Capacity being released by a completed eviction" (a third decision
moment) must issue a wake -- each of `_drive_resident`'s two
`_begin_unload_locked(r)` call sites (manager.py, designated-victim
immediate-evict and self-idle-unload) has a `_make_room_signal.notify_all()`
after it, as do the park and grace-exit
(in `_serve_on_resident`) sites. `_begin_unload_locked` itself
stays untouched by design -- a notify inside it is explicitly
rejected (see
TestNotifyCallSitesAreTheNamedSet). A coroutine parked in
`_wait_for_park_or_timeout` waiting specifically for THAT moment would always fall
through to its `backoff_s` timeout without it -- exactly the anti-pattern ("If the
timeout is what routinely advances the request, the gate is not implemented
-- it is decoration on a polling loop").

THE KILLING TEST (Test A below) does not assert "the claimant was eventually
admitted" -- a retry-on-a-timer would pass that too. It asserts the wait
resolves in a fraction of a second while configured with an
absurdly-large `backoff_s` (3600s) -- the ONLY way that can happen is a real
notify, never a coincidental timeout. Test B is the CONTROL: it exercises
the PRE-EXISTING park notify site (unmodified by this change) with the
SAME measurement technique, proving the technique itself correctly detects a
real wake (methodology sanity) on a code path this patch never touches.
Test C proves the fallback rule (a lost/absent wake can never hang -- it degrades to
the plain timeout, not a stall). Test D is a real end-to-end admission proof
through the actual dispatcher/claim-registration path (no timing assertion --
because the busy-regime's 50ms backoff makes end-to-end latency
an unreliable discriminator; Test A is the discriminator, Test D is the
functional/regression safety net).

Fixture shape for A/B/C follows the idle-unload timer test
(drives `_drive_resident` directly, no full dispatcher -- registers the Fast
Lane claim directly, the same construction three independent existing tests
already use). Test D uses the shared fast-lane fixture helpers
(`ranked_rules`/`make_fakes`/`high_vram`/`drive_to_active` -- real
`worker_loop`/`_dispatch_loop`/`submit_and_wait`).
"""
import asyncio
import time
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
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle

from _fastlane_fixture import (
    assert_resolves,
    boot_ranked_runtime,
    drive_to_active,
    high_vram,
    make_fakes,
    ranked_rules,
    resident_for,
    seed_manifest,
    wait_until,
)

# A wait configured with a backoff this large can ONLY resolve quickly via a
# real notify -- at 3600s, "resolved in under a second" and "resolved via the
# timeout" are mutually exclusive by six orders of magnitude. This is the
# entire point of Test A: it is a discriminator, not a latency estimate.
_HUGE_BACKOFF_S = 3600.0


def _boot_runtime(tmp_path, *, grace_seconds=5, idle_hot_load_seconds=120, fastlane_rules=None):
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
            default_port_base=59960,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, max_parallel_sidecars=2,
            grace_seconds=grace_seconds, max_grace_extensions=50,
            idle_hot_load_seconds=idle_hot_load_seconds,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=fastlane_rules or []),
    )
    return boot, runtime


def _mocks():
    pid = [91_000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **kw):
        pid[0] += 1
        proc = MagicMock()
        proc.pid = pid[0]
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True}

    return dict(
        spawn_fn=fake_spawn, health_fn=fake_health, sigterm_fn=fake_sigterm,
        vram_fn=fake_vram, complete_fn=fake_complete,
    )


def _resident(mgr, model_tag, port, *, rank_client_meta, grace_seconds, idle_seconds=0):
    r = Resident(
        model_tag=model_tag, resident_key=model_tag, port=port,
        grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=idle_seconds),
        rank_client_meta=rank_client_meta,
        last_active_monotonic=time.monotonic(),
    )
    r.inbox = asyncio.Queue()
    mgr._residents[model_tag] = r
    return r


async def _drive_one_turn(mgr, r, model_tag, *, timeout_s):
    """Puts one anchor slot in r.inbox, runs _drive_resident directly, polls
    for the FIRST of: resident deregistered (evicted) or parked
    IDLE_EVICTABLE. Cancels the driver task either way."""
    anchor = Slot.new(model_tag, prompt="hi", thread_id="t1")
    anchor.state = SlotState.STAGED
    await r.inbox.put(anchor)

    drive_task = asyncio.create_task(mgr._drive_resident(r))
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            cur = mgr._residents.get(model_tag)
            if cur is None:
                return "evicted"
            if cur.state is ResidentState.IDLE_EVICTABLE:
                return "idle_evictable"
            await asyncio.sleep(0.02)
        return "timeout"
    finally:
        drive_task.cancel()
        try:
            await drive_task
        except asyncio.CancelledError:
            pass


async def _drive_through_park(mgr, r, model_tag, *, timeout_s):
    """Like `_drive_one_turn`, but does NOT cancel the
    driver task once the resident parks (IDLE_EVICTABLE) -- Test E needs the
    task to keep running PAST park, into the self-idle-unload wait
    (in manager.py), which `_drive_one_turn`'s cancel-on-park would
    cut off before it ever fires. Returns (outcome, drive_task); the caller
    owns the task's lifecycle (cancel it in their own finally)."""
    anchor = Slot.new(model_tag, prompt="hi", thread_id="t1")
    anchor.state = SlotState.STAGED
    await r.inbox.put(anchor)

    drive_task = asyncio.create_task(mgr._drive_resident(r))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cur = mgr._residents.get(model_tag)
        if cur is None:
            return "evicted", drive_task
        if cur.state is ResidentState.IDLE_EVICTABLE:
            return "idle_evictable", drive_task
        await asyncio.sleep(0.02)
    return "timeout", drive_task


@pytest.mark.asyncio
class TestThirdMomentWake:
    async def test_A_SUBJECT_notify_wakes_wait_when_designated_victim_evicted(self, tmp_path):
        """★ KILLING TEST for the completed-eviction wake. MUST FAIL on unmodified code: with no
        notify at the completed-eviction moment, a coroutine parked in
        `_wait_for_park_or_timeout(backoff_s=3600.0)` has nothing to wake it
        when the designated victim is evicted -- it can only resolve by
        actually waiting out 3600s, so the bounded `asyncio.wait_for(...,
        timeout=2.0)` around it times out first. After the fix: the new
        notify right after `_drive_resident`'s `_begin_unload_locked(r)` call
        wakes it in a fraction of a second."""
        grace_seconds = 5
        rules = [
            FastLaneRule(address="9.9.9.9", tag_ranks=FastLaneTagRanks(main=1)),
            FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1)),
        ]
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            idle_hot_load_seconds=120, fastlane_rules=rules,
        )
        mgr = TurbohaulManager(boot, runtime, **_mocks())

        victim = _resident(
            mgr, "victim-model", 59961,
            rank_client_meta={"ip": "1.2.3.4"}, grace_seconds=grace_seconds,
        )
        victim.state = ResidentState.ACTIVE
        victim.last_active_monotonic = 1.0

        claimant = Slot.new("claimant-model", prompt="hi", thread_id="claim")
        claimant.fastlane = FastLaneMatch(
            rule_index=0, raw_address="9.9.9.9", label="",
            effective_tag="main", rank=1,
        )
        mgr._register_fastlane_claim_locked(claimant, "make_room_starved_count_cap")
        assert mgr._is_designated_unload_target_locked(victim) is True

        try:
            wait_task = asyncio.create_task(
                mgr._wait_for_park_or_timeout(backoff_s=_HUGE_BACKOFF_S)
            )
            # Let wait_task actually enter Condition.wait() before the
            # eviction fires -- otherwise the notify could race ahead of the
            # wait registration (the classic notify-before-wait CV race) and
            # this test would wrongly blame the fix for a sequencing gap.
            await asyncio.sleep(0.05)
            assert not wait_task.done(), (
                "wait_task resolved before the eviction even started -- "
                "sequencing bug in the test, not a fix defect"
            )

            t0 = time.monotonic()
            outcome = await _drive_one_turn(mgr, victim, "victim-model", timeout_s=6.0)
            assert outcome == "evicted"

            await asyncio.wait_for(wait_task, timeout=2.0)
            elapsed = time.monotonic() - t0
            assert elapsed < 1.0, (
                f"wait_for_park_or_timeout took {elapsed:.3f}s to resolve "
                f"after the designated victim's eviction completed -- with "
                f"backoff_s={_HUGE_BACKOFF_S}, anything this fast can only "
                f"be a real notify, never the timeout"
            )
        finally:
            await mgr.shutdown()

    async def test_B_GREEN_CONTROL_existing_park_notify_still_wakes(self, tmp_path):
        """★ CONTROL -- shares NO code path with Test A's subject. Exercises
        the PRE-EXISTING park notify (unmodified by
        this patch) with the identical measurement technique, proving
        the technique correctly detects a real wake and that this patch did
        not disturb the already-working park site. Must pass before AND
        after the fix."""
        grace_seconds = 1
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            idle_hot_load_seconds=120, fastlane_rules=[],
        )
        mgr = TurbohaulManager(boot, runtime, **_mocks())

        solo = _resident(
            mgr, "solo-model", 59962,
            rank_client_meta=None, grace_seconds=grace_seconds, idle_seconds=0,
        )
        solo.state = ResidentState.ACTIVE
        assert mgr._is_designated_unload_target_locked(solo) is False

        try:
            wait_task = asyncio.create_task(
                mgr._wait_for_park_or_timeout(backoff_s=_HUGE_BACKOFF_S)
            )
            await asyncio.sleep(0.05)
            assert not wait_task.done()

            # Unlike Test A's designated victim (which skips grace entirely,
            # by design), a non-victim waits out the full grace_seconds FIRST -- so
            # the clock for the wake-latency assertion starts once park
            # actually happens (when _drive_one_turn observes
            # IDLE_EVICTABLE, i.e. right after the park notify already fired),
            # not before the grace wait that precedes it.
            outcome = await _drive_one_turn(
                mgr, solo, "solo-model", timeout_s=grace_seconds + 3.0,
            )
            assert outcome == "idle_evictable"
            t_notify = time.monotonic()

            await asyncio.wait_for(wait_task, timeout=2.0)
            elapsed = time.monotonic() - t_notify
            assert elapsed < 1.0, (
                f"the PRE-EXISTING park notify took {elapsed:.3f}s -- if "
                f"this fails, the measurement technique itself is broken, "
                f"not this patch (this path is unmodified)"
            )
        finally:
            await mgr.shutdown()

    async def test_C_lost_wake_degrades_to_timeout_never_hangs(self, tmp_path):
        """Fallback rule: a lost/absent wake degrades to the previous (timeout)
        behavior and can never hang. No park, no grace-exit, no eviction --
        nothing ever notifies `_make_room_signal` -- so the wait must resolve
        via its own backoff_s timeout, neither early nor never."""
        boot, runtime = _boot_runtime(tmp_path, fastlane_rules=[])
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        try:
            backoff_s = 0.3
            t0 = time.monotonic()
            await asyncio.wait_for(
                mgr._wait_for_park_or_timeout(backoff_s=backoff_s), timeout=2.0,
            )
            elapsed = time.monotonic() - t0
            assert 0.25 <= elapsed <= 1.0, (
                f"expected the wait to resolve at ~{backoff_s}s via its own "
                f"timeout backstop (no notify was ever sent), got {elapsed:.3f}s"
            )
        finally:
            await mgr.shutdown()

    async def test_E_SUBJECT_notify_wakes_wait_on_self_idle_unload(self, tmp_path):
        """Self-idle-unload wake test --
        REBUILT from a written description of the check, not ported: the original
        throwaway probe is not available in this environment, so
        it could not be ported. Rebuilding
        from the description and saying so is the honest option; a claimed
        port that could not be verified byte-for-byte would be worse.

        ★ KILLING TEST for `_drive_resident`'s SECOND `_begin_unload_locked`
        call site -- self-idle-unload (manager.py, "if r.state is
        ResidentState.IDLE_EVICTABLE: self._begin_unload_locked(r) ...
        notify_all()"). Test A above already covers the FIRST site
        (designated-victim immediate-evict) -- by the principle
        "a mutant proves the site you mutated, not the patch": that
        coverage says nothing about THIS site; the four existing tests
        exercise only the park notify (Test B's subject), not this one, which
        is exactly the gap this test closes.

        SEQUENCING, matching the written gate description exactly: the resident
        parks FIRST (IDLE_EVICTABLE, firing the PRE-EXISTING park notify
        to ZERO registered waiters -- our wait_task does not exist
        yet), and ONLY THEN do we register `_wait_for_park_or_timeout`. This
        is deliberate, not incidental: if the wait were registered BEFORE
        park, a pass could mean either the park notify (unmodified) OR
        the self-idle-unload notify (this test's subject) woke it, and
        the test would not discriminate between them. Registering after park
        means the ONLY notify that can still be pending when we start waiting
        is the self-idle-unload one -- forcing the test to depend on
        precisely the site it claims to test, not a neighbour.

        Same discriminator technique as Test A: `backoff_s=3600.0` means
        resolving in under a second can only be a real notify. 3s idle
        window (the description's own figure) keeps the test's own wall-clock bounded
        while still being long enough that a coincidental race with the
        (already-fired) park notify cannot explain a pass.

        BASE (unpatched) RED: TimeoutError from the outer `asyncio.wait_for`
        -- nothing wakes the wait, so it would have to survive the full
        3600s backoff, which the 2.0s outer bound never allows. PATCHED
        GREEN: wakes in a fraction of a second via the notify the patch
        added."""
        idle_seconds = 3
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=1, idle_hot_load_seconds=idle_seconds,
            fastlane_rules=[],
        )
        mgr = TurbohaulManager(boot, runtime, **_mocks())

        solo = _resident(
            mgr, "solo-model-e", 59963,
            rank_client_meta=None, grace_seconds=1, idle_seconds=idle_seconds,
        )
        assert mgr._is_designated_unload_target_locked(solo) is False

        drive_task = None
        try:
            outcome, drive_task = await _drive_through_park(
                mgr, solo, "solo-model-e", timeout_s=6.0,
            )
            assert outcome == "idle_evictable", (
                f"expected the resident to park before self-idle-unload can "
                f"even become relevant, got {outcome!r} -- fixture problem, "
                f"not a code defect"
            )

            # Registered AFTER park -- see the sequencing note above.
            wait_task = asyncio.create_task(
                mgr._wait_for_park_or_timeout(backoff_s=_HUGE_BACKOFF_S)
            )
            await asyncio.sleep(0.05)
            assert not wait_task.done(), (
                "wait_task resolved immediately on registration -- sequencing "
                "bug in the test (a stale notify?), not a fix defect"
            )

            deadline = time.monotonic() + idle_seconds + 3.0
            while time.monotonic() < deadline and mgr._residents.get("solo-model-e") is not None:
                await asyncio.sleep(0.02)
            assert mgr._residents.get("solo-model-e") is None, (
                f"self-idle-unload never evicted the resident within "
                f"{idle_seconds + 3.0}s -- fixture problem (idle window "
                f"wiring), not this test's actual subject"
            )
            t_notify = time.monotonic()

            await asyncio.wait_for(wait_task, timeout=2.0)
            elapsed = time.monotonic() - t_notify
            assert elapsed < 1.0, (
                f"wait_for_park_or_timeout took {elapsed:.3f}s to resolve "
                f"after self-idle-unload evicted the resident -- with "
                f"backoff_s={_HUGE_BACKOFF_S}, anything this fast can only "
                f"be a real notify, never the timeout"
            )
        finally:
            if drive_task is not None:
                drive_task.cancel()
                try:
                    await drive_task
                except BaseException:
                    pass
            await mgr.shutdown()


@pytest.mark.asyncio
class TestEndToEndAdmission:
    async def test_D_claimant_admitted_after_victim_turn_completes(self, tmp_path):
        """Functional/regression check through the REAL dispatcher: a
        higher-priority claimant deferred at count-cap is genuinely ADMITTED
        (its submit_and_wait resolves, not a VramOverCommitError) once the
        designated victim's turn completes and the resident is evicted --
        proving the fix doesn't just resolve an internal wait, it results in
        real admission, with no change to `_lru_idle_unloadable` or the
        victim-selection predicates (untouched by this patch). No
        timing assertion here -- see the module docstring for why the
        busy-regime's 50ms backoff makes end-to-end latency an unreliable
        wake-vs-poll discriminator; Test A owns that job.

        BOTH loaded residents are held mid-turn (a gate each) until released.
        A single gate on only the victim measurably (confirmed while
        building this test) lets the OTHER resident finish its turn first,
        enter GRACE, and get reclaimed by the pre-existing
        `grace_fastlane_breakout` mechanism instead (a different, unmodified
        code path this patch does not touch) -- which admits the claimant
        without ever exercising `_is_designated_unload_target_locked` or the completed-eviction wake at all.
        Holding both mid-turn forces the scenario through the one this patch
        actually changed: the victim's OWN turn completing."""
        boot, runtime = boot_ranked_runtime(
            tmp_path,
            rules=ranked_rules(
                ("10.0.0.1", 1),  # claimant -- best rank
                ("10.0.0.2", 2),  # other-model -- survives
                ("10.0.0.3", 3),  # victim-model -- worst rank
            ),
            max_parallel_sidecars=2,
            grace_seconds=120, max_grace_extensions=5,
            idle_hot_load_seconds=120,
        )
        seed_manifest(boot, "victim-model", main_gpu=0)
        seed_manifest(boot, "other-model", main_gpu=1)
        seed_manifest(boot, "claimant-model", main_gpu=0)

        gates = {"victim-model": asyncio.Event(), "other-model": asyncio.Event()}
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes(gates)

        with high_vram():
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            victim_turn = other_turn = None
            try:
                victim_turn = await drive_to_active(
                    mgr, "victim-model", thread_id="v1",
                    client_meta={"ip": "10.0.0.3"},
                )
                other_turn = await drive_to_active(
                    mgr, "other-model", thread_id="o1",
                    client_meta={"ip": "10.0.0.2"},
                )
                assert_resolves(mgr, "victim-model")
                assert_resolves(mgr, "other-model")

                claim_task = asyncio.create_task(
                    mgr.submit_and_wait(
                        "claimant-model", "p", thread_id="c1",
                        client_meta={"ip": "10.0.0.1"},
                    )
                )
                await wait_until(
                    lambda: mgr._is_designated_unload_target_locked(
                        resident_for(mgr, "victim-model")
                    ),
                    timeout=3.0,
                )

                gates["victim-model"].set()
                slot, result = await asyncio.wait_for(claim_task, timeout=5.0)
                assert slot.model_tag == "claimant-model"
                assert result.get("model") == "claimant-model", result
            finally:
                gates["other-model"].set()
                for t in (victim_turn, other_turn):
                    if t is not None:
                        t.cancel()
                mgr._worker_task.cancel()
                try:
                    await mgr._worker_task
                except asyncio.CancelledError:
                    pass
                await mgr.shutdown()
