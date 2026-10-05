"""Grace timer countdown free-runs during active serving.

Root cause (diagnosed and confirmed):
  GraceTimer (queue.py) has NO pause/stop mechanism. remaining_s() always
  counts wall-clock: `elapsed = time.monotonic() - self._started_at`.
  `_resident_phase` (manager.py) reports GRACE + countdown whenever
  grace hasn't expired -- REGARDLESS of whether the resident is actively
  serving a follow-up (prefill or token-generation phases).

Fix:
  1. GraceTimer.pause() / resume() / is_paused() -- freeze/unfreeze via a
     separate _paused_remaining field (NEVER modifies _started_at).
  2. Resident.in_grace_loop flag -- True when grace window is LIVE (started,
     not yet exited). Consumed ONLY by _resident_phase for countdown display.
     _grace_active_exclusions KEEPS using grace.expired() for the grace-window victim
     protection (the countdown display must not affect it).
  3. _resident_phase reports GRACE countdown ONLY when idle in grace
     (in_grace_loop and NOT is_paused). During active serving, reports
     ACTIVE (countdown is paused/invisible).
  4. _serve_on_resident calls grace.pause() at ACTIVE_MATCH entry,
     grace.resume() after restart_for_followup(), sets in_grace_loop
     at start/lapse.

RED: unfixed tree -- grace countdown decreases during simulated serving.
GREEN: fixed tree -- countdown frozen during serving, resumes when idle.

Grace-protection regression guard: test_exclusion_not_regressed -- proves
_grace_active_exclusions still uses grace.expired() (NOT in_grace_loop),
so grace-window victim protection is NOT regressed.
"""
import asyncio
import time

import pytest

from turbohaul.queue import GraceTimer
from turbohaul.manager import Resident, ResidentState


class TestGraceTimerPauseResume:
    """Unit tests for GraceTimer.pause()/resume()/is_paused() -- RED at
    unfixed code (no pause method), GREEN with fix."""

    def test_pause_freezes_remaining_s(self):
        """RED: AttributeError (no pause method). GREEN: remaining_s() frozen."""
        gt = GraceTimer(grace_seconds=10.0)
        gt.start("t1", "m1")
        assert not gt.is_paused()

        time.sleep(0.1)
        remaining_before = gt.remaining_s()

        gt.pause()
        time.sleep(0.2)
        remaining_after = gt.remaining_s()

        assert gt.is_paused(), "GraceTimer should be paused after pause()"
        assert remaining_after == pytest.approx(remaining_before, abs=0.01), (
            "Countdown must FREEZE during pause: "
            f"before={remaining_before:.3f}, after={remaining_after:.3f}"
        )

    def test_resume_anchors_from_pause_point(self):
        """After resume(), countdown continues from where it was paused."""
        gt = GraceTimer(grace_seconds=10.0)
        gt.start("t1", "m1")
        time.sleep(0.1)
        gt.pause()
        remaining_paused = gt.remaining_s()

        gt.resume()
        assert not gt.is_paused()
        time.sleep(0.05)
        remaining_after = gt.remaining_s()

        assert remaining_after < remaining_paused, (
            "Countdown must RESUME from pause point after resume(): "
            f"at_pause={remaining_paused:.3f}, after={remaining_after:.3f}"
        )

    def test_pause_is_idempotent(self):
        """Double pause does not change the freeze point."""
        gt = GraceTimer(grace_seconds=10.0)
        gt.start("t1", "m1")
        time.sleep(0.05)
        gt.pause()
        first_freeze = gt.remaining_s()

        time.sleep(0.1)
        gt.pause()
        second_freeze = gt.remaining_s()

        assert first_freeze == pytest.approx(second_freeze, abs=0.01)

    def test_expired_still_works_with_pause(self):
        """expired() returns True when grace lapses (even if paused)."""
        gt = GraceTimer(grace_seconds=0.01)
        gt.start("t1", "m1")
        time.sleep(0.05)  # let the 0.01s grace lapse first
        gt.pause()
        assert gt.expired(), (
            "Expired grace should report expired even after pause "
            "(pause freezes remaining at 0)"
        )


class TestResidentPhaseGraceDuringActiveServing:
    """Integration tests for _resident_phase with pause/resume lifecycle.
    RED at unfixed code (no in_grace_loop / pause checks), GREEN with fix."""

    def test_resident_phase_reports_active_during_serving(self):
        """RED: _resident_phase reports GRACE during active serving.
        GREEN: reports ACTIVE when grace is paused (actively serving)."""
        from turbohaul.manager import _resident_phase

        r = Resident()
        r.grace = GraceTimer(grace_seconds=10.0)
        r.in_grace_loop = True
        r.grace.start("t1", "m1")
        r.state = ResidentState.ACTIVE

        # Phase 1: idle in grace -- should report GRACE with countdown
        phase = _resident_phase(r, time.monotonic())
        assert phase.phase == ResidentState.GRACE.value, (
            f"Idle grace should report GRACE, got {phase.phase}"
        )
        assert phase.remaining_s is not None and phase.remaining_s > 0

        # Phase 2: actively serving -- pause the countdown
        r.grace.pause()
        phase = _resident_phase(r, time.monotonic())
        assert phase.phase == ResidentState.ACTIVE.value, (
            f"Countdown is PAUSED (actively serving) -- must report ACTIVE, "
            f"not GRACE. Got {phase.phase}"
        )

    def test_resident_phase_reports_grace_when_idle_after_resume(self):
        """After resume (follow-up complete, back to idle grace), countdown
        resumes and _resident_phase reports GRACE again."""
        from turbohaul.manager import _resident_phase

        r = Resident()
        r.grace = GraceTimer(grace_seconds=10.0)
        r.in_grace_loop = True
        r.grace.start("t1", "m1")
        r.state = ResidentState.ACTIVE

        r.grace.pause()
        phase = _resident_phase(r, time.monotonic())
        assert phase.phase == ResidentState.ACTIVE.value  # serving

        r.grace.resume()
        phase = _resident_phase(r, time.monotonic())
        assert phase.phase == ResidentState.GRACE.value, (
            f"After resume (back to idle grace), should report GRACE, got {phase.phase}"
        )

    def test_resident_phase_reports_state_when_no_grace(self):
        """When grace is None or in_grace_loop=False, falls through to r.state."""
        from turbohaul.manager import _resident_phase

        r = Resident()
        r.grace = None
        r.state = ResidentState.ACTIVE
        phase = _resident_phase(r, time.monotonic())
        assert phase.phase == ResidentState.ACTIVE.value
        assert phase.resolved_from == "r.state"

        # in_grace_loop=False with live grace timer -> report ACTIVE (not GRACE)
        r2 = Resident()
        r2.grace = GraceTimer(grace_seconds=10.0)
        r2.in_grace_loop = False
        r2.grace.start("t1", "m1")
        r2.state = ResidentState.ACTIVE
        phase = _resident_phase(r2, time.monotonic())
        assert phase.phase == ResidentState.ACTIVE.value, (
            "in_grace_loop=False (not in grace loop) must NOT report GRACE"
        )


class TestVictimProtectionNotRegressed:
    """Grace-window victim protection must NOT be regressed by the
    in_grace_loop flag change.

    KEY POINT: _grace_active_exclusions KEEPS
    using grace.expired() -- NOT in_grace_loop. The in_grace_loop flag is
    consumed ONLY by _resident_phase for countdown display. This test proves
    that: _grace_active_exclusions still references grace.expired()
    directly, not in_grace_loop.

    These tests use the shared fixture of the paired grace victim-protection tests
    to drive a REAL resident through a real admission -> grace lifecycle,
    then assert that victim protection holds."""

    @pytest.mark.asyncio
    async def test_exclusion_not_regressed_nonvictim_protected(self, tmp_path):
        """A non-victim's grace-held pair is still protected by
        _grace_active_exclusions. This is the regression guard.

        RED at unfixed code: _grace_active_exclusions reads grace.expired()
        directly (no in_grace_loop check) -- passes because grace IS not expired.
        GREEN with fix: _grace_active_exclusions STILL uses grace.expired()
        (NOT in_grace_loop) -- still passes.
        """
        from tests._fastlane_fixture import (
            boot_ranked_runtime, seed_manifest, make_fakes, high_vram,
            drive_to_active, install_victim_predicate, resident_for,
        )
        from turbohaul.manager import TurbohaulManager

        boot, runtime = boot_ranked_runtime(tmp_path, grace_seconds=5)
        seed_manifest(boot, "m1", main_gpu=0)
        gate = asyncio.Event()
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate})
        with high_vram():
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            task = await drive_to_active(mgr, "m1", thread_id="t1", client_meta={})
            gate.set()
            await asyncio.wait_for(task, timeout=5.0)

        try:
            r = resident_for(mgr, "m1")
            assert r is not None, "resident must exist after drive_to_active"
            assert r.grace is not None and not r.grace.expired(), (
                "sanity: resident must have a live grace timer"
            )
            # in_grace_loop must be True for an idle grace resident
            assert r.in_grace_loop is True, (
                "in_grace_loop must be True when resident is parked in idle grace"
            )

            install_victim_predicate(mgr, ["some-other-tag"])  # m1 is NOT victim
            excluded = await mgr._grace_active_exclusions()
            assert ("t1", "m1") in excluded, (
                "a non-victim grace-held resident MUST still be "
                "excluded from pop_next's pick ladder. The in_grace_loop "
                "change (used ONLY in _resident_phase, NOT in "
                "_grace_active_exclusions) must NOT weaken victim protection."
            )
        finally:
            await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_victim_still_not_excluded(self, tmp_path):
        """A designated victim is still NOT protected."""
        from tests._fastlane_fixture import (
            boot_ranked_runtime, seed_manifest, make_fakes, high_vram,
            drive_to_active, install_victim_predicate, resident_for,
        )
        from turbohaul.manager import TurbohaulManager

        boot, runtime = boot_ranked_runtime(tmp_path, grace_seconds=5)
        seed_manifest(boot, "m1", main_gpu=0)
        gate = asyncio.Event()
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate})
        with high_vram():
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            task = await drive_to_active(mgr, "m1", thread_id="t1", client_meta={})
            gate.set()
            await asyncio.wait_for(task, timeout=5.0)

        try:
            r = resident_for(mgr, "m1")
            assert r is not None
            assert r.grace is not None and not r.grace.expired()

            install_victim_predicate(mgr, ["m1"])  # m1 IS the victim
            excluded = await mgr._grace_active_exclusions()
            assert ("t1", "m1") not in excluded, (
                "Designated-victim rule: a designated victim gets NO grace protection. "
                "The victim predicate still overrides to un-protect."
            )
        finally:
            await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_exclusion_uses_grace_expired_not_in_grace_loop(self, tmp_path):
        """Regression guard: proves that
        _grace_active_exclusions uses grace.expired(), NOT in_grace_loop.

        We explicitly set in_grace_loop=False on a resident with a LIVE
        (not expired) grace timer. If _grace_active_exclusions were
        changed to use in_grace_loop,
        this resident would NOT be excluded -- silently breaking the grace-window
        victim protection. As implemented, it IS still excluded
        because _grace_active_exclusions uses grace.expired().
        """
        from tests._fastlane_fixture import (
            boot_ranked_runtime, seed_manifest, make_fakes, high_vram,
            drive_to_active, install_victim_predicate, resident_for,
        )
        from turbohaul.manager import TurbohaulManager

        boot, runtime = boot_ranked_runtime(tmp_path, grace_seconds=5)
        seed_manifest(boot, "m1", main_gpu=0)
        gate = asyncio.Event()
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate})
        with high_vram():
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            task = await drive_to_active(mgr, "m1", thread_id="t1", client_meta={})
            gate.set()
            await asyncio.wait_for(task, timeout=5.0)

        try:
            r = resident_for(mgr, "m1")
            assert r is not None
            assert r.grace is not None and not r.grace.expired()

            # NOTE: in_grace_loop does NOT gate _grace_active_exclusions.
            # Even if in_grace_loop=False (which shouldn't happen in normal flow
            # for an idle grace resident, but proves the point), the grace timer
            # is still NOT expired, so grace-window protection MUST still apply.
            r.in_grace_loop = False  # deliberately break the flag
            r.grace.pause()          # also pause the countdown

            install_victim_predicate(mgr, ["some-other-tag"])  # non-victim
            excluded = await mgr._grace_active_exclusions()
            assert ("t1", "m1") in excluded, (
                "regression guard: _grace_active_exclusions MUST use "
                "grace.expired(), NOT in_grace_loop. Even with in_grace_loop=False "
                "and grace paused, a non-expired grace timer still protects the "
                "resident's (thread_id, model_tag) pair from pop_next."
            )
        finally:
            await mgr.shutdown()
