"""Dead-engine detection must work at the
SHIPPING DEFAULT (max_parallel_sidecars=1): strike, ENGINE STALLED, and
3-strike auto-quarantine. This is not a cap-safety nuance: the gate at api/main.py
(`_idle_liveness_task`, which must be `>= 1`, not `>= 2`) gates the ONLY idle-dead-
holder detector that exists at ANY cap, and must not switch it off at exactly the
cap where nothing else covers the gap.

THE ACTUAL DEFECT IS IN THE WIRING, NOT THE SWEEP. `_idle_engine_liveness_sweep`
itself needed zero logic changes: `_model_residents()` was already
correct at cap=1 (it excludes `_SINGLETON_RESIDENT_KEY` on purpose, and that
sentinel resident is never the live one -- `_route_or_reserve` is
model-tag-keyed with a real `_drive_resident` task at every cap, no branch).
Only the wiring condition decides whether the sweep runs at cap=1; the sweep
logic itself is the same at every cap.

Test A drives the REAL FastAPI lifespan (`app.router.lifespan_context`, not
TestClient's anyio thread-portal -- see the plugin health-probe test's
own test_periodic_task_does_not_outlive_shutdown for why a portal's teardown
can force-cancel a task and make a mutated wiring pass by accident) and
asserts the liveness task actually gets created at cap=1 -- the DIRECT proof
of the wiring defect and its fix.

Tests B/C then drive the same real lifespan end-to-end with a REAL, killed OS
subprocess as the engine, because a test that simulates a dead engine by
setting a flag proves nothing (the flag-faking shape); a synthetic death
cannot show that the sweep notices a real one. A real process is possible
because `_spawn` is a
pure DI seam (every test in this suite already injects a fake through it);
nothing stops that fake from wrapping a REAL `subprocess.Popen` instead of a
MagicMock -- `SidecarHandle.is_alive()` is `self.proc.poll() is None`, which
means exactly the same thing against a real Popen as it does against a mock.
Test B SIGKILLs it and waits for `.poll()` to genuinely return non-None (an
actual reap, not a stand-in) BEFORE asserting anything, so "the engine is
dead" and "the sweep noticed" stay two separate, un-conflated facts.

Test C reaches the 3rd-strike / auto-quarantine branch by seeding
`_bin_death_strikes` to 2 before one real kill -- the exact technique
the shared bin-death strike helper test's own
test_third_strike_triggers_auto_quarantine already uses and documents as
"a legitimate, standard way to reach the 3rd-strike branch without three full
realistic restore/die cycles." This is not the flag-faking shape: that approach
hand-set the STATE THE DETECTOR ITSELF READS to fake being idle-hot (an
unreachable precondition); seeding a plain integer counter the detector still
increments for itself, off a real death it still has to notice for itself, is
priming an input, not faking the mechanism under test.

Test D is the cap=2 control: the gate
admits MORE caps (`>= 1` rather than `>= 2`), so cap=2 must still create and
run the task exactly as before. Full behavioral coverage at cap=2 (detection,
TOCTOU safety, the disclosed grace-window gap) stays
the cap=2 idle-liveness sweep tests' own job, unmodified and
unaffected by this file (the two sets are disjoint) -- Test D here is only
the narrow "still wired" check.

WHY A REAL SUBPROCESS: a mocked `proc.poll.return_value` (the convention
of the cap=2 idle-liveness sweep tests) would only be an analogy to a
real death. The real-vs-simulated distinction is the point of Tests B/C,
so driving an actual OS-level death removes any doubt rather than
resting on an analogy to a sibling test's
convention.
The tests fail (RED) without the api/main.py fix and pass (GREEN) with it,
on this exact file, rather than being carried over from any other test
file.
"""
import asyncio
import subprocess
from unittest.mock import patch

import pytest

from turbohaul.api.main import create_app
from turbohaul.manager import ResidentState
from turbohaul.subprocess_mgr import SidecarHandle

from _fastlane_fixture import boot_ranked_runtime, high_vram, seed_manifest, wait_until

_FAST_INTERVAL_S = 0.05


def _real_engine_fakes():
    """spawn_fn wraps a REAL, killable OS subprocess (not a MagicMock) so
    Tests B/C drive a genuine process death, not a scripted return value."""
    procs = []

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        proc = subprocess.Popen(["sleep", "300"])
        procs.append(proc)
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return fake_spawn, fake_health, fake_sigterm, fake_vram, fake_complete, procs


def _mock_engine_fakes():
    """Test A/D only assert task creation -- never admit a request, so a
    plain MagicMock spawn (no real process, no cleanup burden) is enough."""
    from unittest.mock import MagicMock

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        proc = MagicMock()
        proc.pid = 99999
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return fake_spawn, fake_health, fake_sigterm, fake_vram, fake_complete


def _boot_app(tmp_path, *, max_parallel_sidecars, real_engine):
    boot, runtime = boot_ranked_runtime(
        tmp_path, max_parallel_sidecars=max_parallel_sidecars,
        grace_seconds=0, idle_hot_load_seconds=120,
    )
    seed_manifest(boot, "model-a", main_gpu=0)
    app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
    mgr = app.state.manager
    if real_engine:
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn, procs = _real_engine_fakes()
    else:
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = _mock_engine_fakes()
        procs = []
    mgr._spawn = spawn_fn
    mgr._wait_healthy = health_fn
    mgr._sigterm = sigterm_fn
    mgr._vram_verify = vram_fn
    mgr._complete_fn = complete_fn
    return app, mgr, procs


def _live_resident(mgr, model_tag):
    for r in mgr._model_residents():
        if r.model_tag == model_tag:
            return r
    raise AssertionError(f"no live model-tag-keyed resident for {model_tag!r}")


async def _kill_for_real(proc, timeout=3.0):
    """SIGKILL a real subprocess and wait for poll() to genuinely reflect
    it -- an actual reap, not a mocked return value."""
    proc.kill()
    await asyncio.to_thread(proc.wait, timeout)
    assert proc.poll() is not None, "process still not reaped after wait()"


def _reap_all(procs):
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=3.0)


@pytest.mark.asyncio
class TestOneResidentCapLivenessTaskWiring:
    async def test_A_liveness_task_is_created_at_one_resident_cap(self, tmp_path):
        """Direct proof of the wiring defect/fix. Pre-fix (gate `>= 2`): this
        attribute is never assigned at cap=1 -- getattr(...) is None. Post-fix
        (gate `>= 1`): a real, running asyncio.Task."""
        app, mgr, _procs = _boot_app(tmp_path, max_parallel_sidecars=1, real_engine=False)
        with high_vram():
            async with app.router.lifespan_context(app):
                task = getattr(mgr, "_idle_liveness_task", None)
                assert task is not None, (
                    "no idle-dead-holder liveness task was created at cap=1 -- "
                    "the shipping default has zero dead-engine detection"
                )
                assert not task.done()


@pytest.mark.asyncio
class TestOneResidentCapDeadEngineDetection:
    async def test_B_a_really_killed_one_resident_cap_engine_is_detected_and_struck(
        self, tmp_path,
    ):
        app, mgr, procs = _boot_app(tmp_path, max_parallel_sidecars=1, real_engine=True)
        try:
            with high_vram():
                with patch(
                    "turbohaul.manager._MULTI_ENGINE_IDLE_LIVENESS_SWEEP_INTERVAL_S",
                    _FAST_INTERVAL_S,
                ):
                    async with app.router.lifespan_context(app):
                        assert mgr.runtime.queue.max_parallel_sidecars == 1
                        await mgr.submit_and_wait("model-a", "p", thread_id="t1")
                        await wait_until(
                            lambda: _live_resident(mgr, "model-a").state
                            is ResidentState.IDLE_EVICTABLE,
                            timeout=3.0,
                        )
                        r = _live_resident(mgr, "model-a")
                        port = r.handle.port
                        real_proc = r.handle.proc
                        mgr._last_restored_bin = {port: "one_resident_cap_dead_bin.bin"}
                        mgr._bin_death_strikes = {}
                        before_evictions = getattr(mgr, "_unload_count", 0)

                        await _kill_for_real(real_proc)  # a genuine OS-level death

                        await wait_until(
                            lambda: mgr._bin_death_strikes.get("one_resident_cap_dead_bin.bin")
                            is not None,
                            timeout=3.0,
                        )

                        assert mgr._bin_death_strikes["one_resident_cap_dead_bin.bin"] == 1
                        stall = mgr._engine_stall
                        assert stall is not None, "ENGINE STALLED banner never set at cap=1"
                        assert stall["bin"] == "one_resident_cap_dead_bin.bin"
                        assert stall["attempt"] == 1
                        # torn down: no longer a live resident, deregistered.
                        with pytest.raises(AssertionError):
                            _live_resident(mgr, "model-a")
                        assert getattr(mgr, "_unload_count", 0) == before_evictions + 1
        finally:
            _reap_all(procs)

    async def test_C_third_real_death_on_the_same_bin_triggers_quarantine(
        self, tmp_path,
    ):
        app, mgr, procs = _boot_app(tmp_path, max_parallel_sidecars=1, real_engine=True)
        try:
            with high_vram():
                with patch(
                    "turbohaul.manager._MULTI_ENGINE_IDLE_LIVENESS_SWEEP_INTERVAL_S",
                    _FAST_INTERVAL_S,
                ):
                    async with app.router.lifespan_context(app):
                        await mgr.submit_and_wait("model-a", "p", thread_id="t1")
                        await wait_until(
                            lambda: _live_resident(mgr, "model-a").state
                            is ResidentState.IDLE_EVICTABLE,
                            timeout=3.0,
                        )
                        r = _live_resident(mgr, "model-a")
                        port = r.handle.port
                        real_proc = r.handle.proc
                        mgr._last_restored_bin = {port: "repeat_offender_one_resident_cap.bin"}
                        # Seed two PRIOR strikes -- the same standard technique
                        # the shared bin-death strike helper test's own
                        # test_third_strike_triggers_auto_quarantine uses to reach
                        # the 3rd-strike branch without three full kill cycles.
                        # The detector still has to notice THIS death and
                        # increment for itself; only the prior count is primed.
                        mgr._bin_death_strikes = {"repeat_offender_one_resident_cap.bin": 2}

                        with patch.object(mgr, "_quarantine_bin_triplet") as mock_q:
                            await _kill_for_real(real_proc)

                            await wait_until(
                                lambda: mgr._bin_death_strikes.get(
                                    "repeat_offender_one_resident_cap.bin"
                                )
                                == 3,
                                timeout=3.0,
                            )
                            await wait_until(lambda: mock_q.called, timeout=2.0)
                            mock_q.assert_called_once_with("repeat_offender_one_resident_cap.bin")
                            assert mgr._engine_stall["attempt"] == 3
        finally:
            _reap_all(procs)


@pytest.mark.asyncio
class TestTwoResidentCapControlUnaffected:
    async def test_D_liveness_task_still_created_at_two_resident_cap(self, tmp_path):
        """The gate admits MORE caps (`>= 1` rather than `>= 2`); cap=2 must
        still spawn the task exactly as before. Full behavioral coverage at
        cap=2 stays the cap=2 idle-liveness sweep tests' own job,
        unmodified and unaffected by this file -- this is the narrow
        "still wired" check."""
        app, mgr, _procs = _boot_app(tmp_path, max_parallel_sidecars=2, real_engine=False)
        with high_vram():
            async with app.router.lifespan_context(app):
                task = getattr(mgr, "_idle_liveness_task", None)
                assert task is not None
                assert not task.done()
