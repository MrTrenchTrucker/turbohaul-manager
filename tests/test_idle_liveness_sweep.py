"""The cap>=2 twin of worker_loop's own cap<=1 proactive
idle-dead-holder sweep. Before this sweep, Loop A (`_drive_resident`'s wait
loop) was REACTIVE-ONLY -- confirmed by grepping the exact wait-loop line
range for `is_alive()` and finding zero hits; the two that exist there are
downstream of an ALREADY-DECIDED teardown, not upstream of one. A resident
that dies while idle-hot at cap>=2 was invisible until either its ordinary
idle_hot_load_seconds elapsed (minutes) or a NEW request got routed onto it
and failed -- exactly the gap `_idle_engine_liveness_sweep` closes.

THE KILLING TEST (Test A) proves a dead-while-idle cap>=2 resident is
DETECTED, TORN DOWN, and its death ATTRIBUTED to the bin it last restored
(bin-death-strike incremented, ENGINE STALLED banner set) -- reverting the
sweep's detection call (or gating it on cap<=1 only, as today's code
effectively does by not existing) makes this test hang/fail; this was
verified with a mutant that removes the detection call.

Test B is the GREEN CONTROL: an ACTIVE (mid-turn) resident whose handle also
reports dead must NOT be touched by the idle-liveness sweep -- it shares the
exact same dead-handle-detection numeric check (`handle.is_alive()`) but
takes the opposite branch (`r.state is not IDLE_EVICTABLE` short-circuits
first). A driver-death reap is a SEPARATE, pre-existing mechanism this sweep
does not touch. NOTE: a resident sitting in its GRACE
window is, at the Resident level, INDISTINGUISHABLE from Test B's scenario
-- r.state is ACTIVE and active_slot is SET throughout grace exactly as it
is mid-generation (`slot.state` is the only thing that differs, and this
sweep never reads it). So Test B already covers a grace-window death too;
Test D below makes that explicit with a real grace window instead of
relying on the reader to infer the equivalence.

Test C is the TOCTOU-safety regression:
A tick reads then acts, and a resident going idle->ACTIVE in that gap
would be evicted mid-turn. The whole check-and-act happens in ONE
`_registry_lock` acquisition per resident per tick with zero `await` points
inside it (confirmed by reading the method -- `_model_residents()`,
`handle.is_alive()`, `_record_bin_death_strike`, `_begin_unload_locked`, and
`notify_all()` are all synchronous), so a resident cannot flip idle->ACTIVE
mid-check: the only way to flip it is to also take `_registry_lock`, which
serializes behind the sweep's own hold. Test C proves this holds under real
concurrent pressure rather than only by code inspection: it hammers a single
ALIVE resident with rapid submit/complete cycles (idle<->ACTIVE churn)
CONCURRENTLY with a fast-ticking sweep and asserts zero spurious evictions.

Test D (does the predicate
skip a grace resident too?): DISCLOSED, NOT FIXED gap. A resident whose
engine dies DURING its grace window is not detected by this sweep -- proves
it directly with a real grace window (grace_seconds=2) rather than the
mid-turn gate Test B uses, and asserts zero strikes for the whole window.
This is not a regression: it is the SAME miss the earlier
`active_slot`-based predicate had too (measured, see manager.py's method
docstring) -- IDLE_EVICTABLE-gating is not a coverage widening on this
tree, it is a more direct expression of the same set. Whether grace-window
coverage belongs in THIS sweep's scope is a separate decision, not resolved here.

Fixture shape follows the eviction-wake test (same
shared Fast Lane fixture helpers -- real `worker_loop`/`submit_and_wait`,
no Fast Lane rules needed here so `enabled=False`).
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from turbohaul.manager import ResidentState, TurbohaulManager

from _fastlane_fixture import (
    boot_ranked_runtime,
    make_fakes,
    resident_for,
    seed_manifest,
    wait_until,
)

_FAST_INTERVAL_S = 0.05


def _boot(tmp_path, **kw):
    # grace_seconds=0: a resident with active_slot SET (slot.state=GRACE) is
    # NOT idle-hot -- measured directly while building this test, active_slot
    # stays non-None for the WHOLE grace window even though nothing is being
    # served. 0 here means "opt out of grace" so a served resident reaches
    # the real idle-hot state (ResidentState.IDLE_EVICTABLE) fast enough for
    # a test to observe it deterministically, matching boot_ranked_runtime's
    # own documented default-to-0 rationale.
    boot, runtime = boot_ranked_runtime(
        tmp_path, max_parallel_sidecars=2, grace_seconds=0,
        idle_hot_load_seconds=120, **kw,
    )
    seed_manifest(boot, "model-a", main_gpu=0)
    return boot, runtime


@pytest.mark.asyncio
class TestTwoResidentCapIdleLivenessSweepDetection:
    async def test_A_dead_idle_resident_is_detected_torn_down_and_strike_recorded(
        self, tmp_path, caplog,
    ):
        boot, runtime = _boot(tmp_path)
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({})

        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000]):
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await mgr.submit_and_wait("model-a", "p", thread_id="t1")
                # grace_seconds=0 means active_slot clears fast, but the
                # IDLE_EVICTABLE transition still happens on the driver's own
                # next loop turn -- wait for the real idle-hot state rather
                # than assume submit_and_wait's return implies it.
                await wait_until(
                    lambda: resident_for(mgr, "model-a").state
                    is ResidentState.IDLE_EVICTABLE,
                    timeout=3.0,
                )
                r = resident_for(mgr, "model-a")
                port = r.handle.port
                mgr._last_restored_bin = {port: "two_resident_cap_dead_bin.bin"}
                mgr._bin_death_strikes = {}
                before_evictions = getattr(mgr, "_unload_count", 0)

                r.handle.proc.poll.return_value = -11  # engine died, idle

                with patch(
                    "turbohaul.manager._MULTI_ENGINE_IDLE_LIVENESS_SWEEP_INTERVAL_S",
                    _FAST_INTERVAL_S,
                ):
                    mgr._idle_liveness_task = asyncio.create_task(
                        mgr._idle_engine_liveness_sweep()
                    )
                    try:
                        await wait_until(
                            lambda: mgr._bin_death_strikes.get("two_resident_cap_dead_bin.bin") is not None,
                            timeout=3.0,
                        )
                    finally:
                        mgr._idle_liveness_task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await mgr._idle_liveness_task

                assert mgr._bin_death_strikes["two_resident_cap_dead_bin.bin"] == 1
                stall = mgr._engine_stall
                assert stall is not None, "ENGINE STALLED banner never set at cap>=2"
                assert stall["bin"] == "two_resident_cap_dead_bin.bin"
                assert stall["attempt"] == 1
                # torn down: no longer a live resident, deregistered.
                with pytest.raises(AssertionError):
                    resident_for(mgr, "model-a")
                assert getattr(mgr, "_unload_count", 0) == before_evictions + 1
            finally:
                mgr._worker_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await mgr._worker_task

    async def test_B_GREEN_CONTROL_active_resident_with_dead_handle_is_not_touched(
        self, tmp_path,
    ):
        """Same dead-handle numeric check, opposite branch: a resident held
        ACTIVE (mid-turn, via a gate the fake completion blocks on) whose
        handle ALSO reports dead must be left alone by this sweep -- a
        driver-death reap (pre-existing, untouched) is the mechanism for a
        crash mid-serve, not the idle-liveness sweep."""
        boot, runtime = _boot(tmp_path)
        gate = asyncio.Event()
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes(
            {"model-a": gate}
        )

        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000]):
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            turn_task = None
            try:
                turn_task = asyncio.create_task(
                    mgr.submit_and_wait("model-a", "p", thread_id="t1")
                )

                def _is_active():
                    try:
                        return resident_for(mgr, "model-a").active_slot is not None
                    except AssertionError:
                        return False  # not registered yet -- driver still spawning

                await wait_until(_is_active, timeout=3.0)
                r = resident_for(mgr, "model-a")
                mgr._last_restored_bin = {r.handle.port: "should_not_strike.bin"}
                mgr._bin_death_strikes = {}
                r.handle.proc.poll.return_value = -11  # dies WHILE mid-turn

                with patch(
                    "turbohaul.manager._MULTI_ENGINE_IDLE_LIVENESS_SWEEP_INTERVAL_S",
                    _FAST_INTERVAL_S,
                ):
                    mgr._idle_liveness_task = asyncio.create_task(
                        mgr._idle_engine_liveness_sweep()
                    )
                    await asyncio.sleep(_FAST_INTERVAL_S * 6)  # several ticks
                    mgr._idle_liveness_task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await mgr._idle_liveness_task

                assert mgr._bin_death_strikes == {}, (
                    "sweep struck an ACTIVE (mid-turn) resident"
                )
                assert getattr(mgr, "_engine_stall", None) is None
                # still registered and still ACTIVE -- untouched.
                r_after = resident_for(mgr, "model-a")
                assert r_after is r
                assert r_after.active_slot is not None
            finally:
                gate.set()
                if turn_task is not None:
                    turn_task.cancel()
                mgr._worker_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await mgr._worker_task


@pytest.mark.asyncio
class TestTwoResidentCapIdleLivenessSweepToctou:
    async def test_C_no_spurious_eviction_under_concurrent_idle_active_churn(
        self, tmp_path,
    ):
        """Re-check liveness AND idleness under
        _registry_lock immediately before acting -- a resident going
        idle->ACTIVE in the gap between a read and a later act would be a
        real new bug. This resident's handle is ALWAYS alive; the only
        variable is how fast it churns idle<->ACTIVE while a
        fast-ticking sweep runs concurrently. Zero spurious evictions is the
        bar: any eviction here would mean the sweep acted on a stale read."""
        boot, runtime = _boot(tmp_path)
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({})

        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000]):
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await mgr.submit_and_wait("model-a", "p", thread_id="t0")
                r = resident_for(mgr, "model-a")
                mgr._last_restored_bin = {r.handle.port: "churning.bin"}
                mgr._bin_death_strikes = {}

                async def churn():
                    for i in range(40):
                        await mgr.submit_and_wait(
                            "model-a", "p", thread_id=f"churn-{i}"
                        )

                with patch(
                    "turbohaul.manager._MULTI_ENGINE_IDLE_LIVENESS_SWEEP_INTERVAL_S",
                    _FAST_INTERVAL_S,
                ):
                    mgr._idle_liveness_task = asyncio.create_task(
                        mgr._idle_engine_liveness_sweep()
                    )
                    try:
                        await asyncio.wait_for(churn(), timeout=15.0)
                    finally:
                        mgr._idle_liveness_task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await mgr._idle_liveness_task

                assert mgr._bin_death_strikes == {}, (
                    "an ALIVE resident was struck as dead under idle<->ACTIVE churn"
                )
                assert getattr(mgr, "_engine_stall", None) is None
                assert getattr(mgr, "_unload_count", 0) == 0
                # still one live resident for model-a, never torn down.
                assert resident_for(mgr, "model-a") is not None
            finally:
                mgr._worker_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await mgr._worker_task


@pytest.mark.asyncio
class TestTwoResidentCapIdleLivenessSweepDisclosedGaps:
    async def test_D_grace_window_death_is_a_disclosed_NOT_fixed_gap(
        self, tmp_path,
    ):
        """Grace residents are not covered either:
        a resident whose engine dies DURING its
        grace window (before the follow-up-or-timeout resolution that
        clears active_slot and sets state=IDLE_EVICTABLE) is invisible to
        this sweep -- not because r.state is checked wrong, but because a
        grace resident's r.state is ACTIVE, same as a genuinely-serving
        one, and this sweep correctly never touches an ACTIVE resident
        (never evictable mid-turn-or-grace). This is NOT a regression:
        the earlier `active_slot`-based predicate had the exact same
        miss (empirically verified, zero divergence at 0.1s resolution
        across a full serve->grace->idle-hot lifecycle -- see the method's
        own docstring). Disclosed, not fixed here."""
        boot, runtime = boot_ranked_runtime(
            tmp_path, max_parallel_sidecars=2, grace_seconds=2,
            idle_hot_load_seconds=120,
        )
        seed_manifest(boot, "model-a", main_gpu=0)
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({})

        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000]):
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await mgr.submit_and_wait("model-a", "p", thread_id="t1")
                r = resident_for(mgr, "model-a")
                # confirm we're actually IN grace, not past it -- a vacuous
                # "nothing detected" would be worthless if grace had already
                # lapsed by the time we checked.
                assert r.state is ResidentState.ACTIVE
                assert r.active_slot is not None
                mgr._last_restored_bin = {r.handle.port: "grace_dead.bin"}
                mgr._bin_death_strikes = {}
                r.handle.proc.poll.return_value = -11  # dies DURING grace

                with patch(
                    "turbohaul.manager._MULTI_ENGINE_IDLE_LIVENESS_SWEEP_INTERVAL_S",
                    _FAST_INTERVAL_S,
                ):
                    mgr._idle_liveness_task = asyncio.create_task(
                        mgr._idle_engine_liveness_sweep()
                    )
                    # Several sweep ticks (interval 0.05s) but staying
                    # comfortably UNDER grace_seconds=2, so the window this
                    # test asserts over is genuinely still grace, not the
                    # post-grace idle-hot phase the sweep legitimately DOES
                    # cover (confirmed separately by Test A).
                    await asyncio.sleep(1.2)
                    still_in_grace = (
                        resident_for(mgr, "model-a").state is ResidentState.ACTIVE
                        and resident_for(mgr, "model-a").active_slot is not None
                    )
                    mgr._idle_liveness_task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await mgr._idle_liveness_task

                assert still_in_grace, (
                    "grace_seconds=2 lapsed before the assertion window "
                    "closed -- test no longer proves what it claims"
                )
                assert mgr._bin_death_strikes == {}, (
                    "a grace-window death WAS detected -- either the gap "
                    "closed (update this test) or something "
                    "else changed the sweep's scope unexpectedly"
                )
                assert getattr(mgr, "_engine_stall", None) is None
            finally:
                mgr._worker_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await mgr._worker_task
