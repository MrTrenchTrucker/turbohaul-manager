"""The stranded manager-level grace/idle
surface behind `status_snapshot()`.

Anchors below are cited by SYMBOL, not by line number -- the code may
move, so line numbers would go stale. The analysis was done against the
then-current manager.py and config.py; re-check the cited symbols if either
has changed since.

ONE bug, TWO user-visible symptoms, both rooted in the same fact: the dispatch
unification makes `worker_loop` route EVERY request through
`_dispatch_loop` whenever `max_parallel_sidecars >= 1` -- and config.py's
`max_parallel_sidecars` field is `Field(default=1, ge=1, le=32)`, so that
condition is unconditionally true. `_process_slot` -- the legacy body that used
to drive requests at cap<=1 -- is therefore dead code at EVERY valid cap, not
only cap>=2. This is corroborated independently three ways, not just read off
the source: (1) `worker_loop`'s own comment calls the `>= 1` check "provably
unconditional"; (2) the cap=1 unified-dispatch test suite is an entire
acceptance suite built to prove cap=1 now runs the SAME parallel dispatcher and
creates a REAL model-keyed Resident; (3)
the per-resident-identity test (the arm3 cap<=1 status_snapshot case)
is stale against the unified dispatch because it still asserts the pre-unification
world ("residents empty at cap<=1").

SCOPE: this file corrects a tempting framing of the problem. One could ask
whether the "residents non-empty
implies not all status scalars None" invariant holds at every cap, or only at
cap>=2, given "at cap<=1 the retired path IS the legitimate driver, so the
scalars may be populated there by design"? Reading the code (not reasoning from
the framing) answers this: that premise is FALSE for the current code.
`_process_slot` cannot be the driver at ANY valid cap once the dispatch unification is
present, so there is no cap at which the retired path legitimately populates
anything. The mechanism this file's tests guard against is cap-general. Both
tests nonetheless BOOT THE FIXTURE AT max_parallel_sidecars=2 EXPLICITLY (the
hard constraint -- the env lies, TURBOHAUL_MAX_PARALLEL=2 may be set by an
operator) so what is measured matches real configuration; the docstrings below say so,
so a future reader does not mistake "tested at cap=2" for "true only at cap=2".

THE MECHANISM, PRECISELY (manager.py):
  self.grace (a module-level GraceTimer, __init__) is armed ONLY by
  self.grace.start() and re-armed ONLY by self.grace.restart_for_followup() --
  every call site of both lives inside submit()'s grace-shortcut branch or
  _process_slot. status_snapshot()'s grace block reads self.grace exclusively.
  Since _process_slot is dead (above), only submit()'s call sites are live --
  and submit() calls self.grace.restart_for_followup() ONLY inside
  `if self.grace.matches(...):`, and GraceTimer.matches() (queue.py) requires
  `self._started_at is not None`, which only start() (dead) or
  restart_for_followup() (gated behind matches() itself) can set. Circular:
  self.grace can never be armed by anything reachable, on any tree with
  the dispatch unification. self.grace.expired() is therefore always True and
  status_snapshot()["grace"] is always None -- REGARDLESS of whether a real
  resident has a genuinely open grace window right now.

  The identical shape affects the OTHER manager-level scalar status_snapshot
  reads for "idle_hot": self._idle_handle is written ONLY by _set_idle_holder,
  whose one real call site is also inside _process_slot; self.idle.start() (the
  IdleHotTimer fallback status_snapshot also checks) has its one call site
  there too. Both are therefore also permanently unreachable on the live path.

  Meanwhile the REAL per-resident lifecycle is alive and correct:
  _serve_on_resident arms `r.grace` (a separate, per-Resident GraceTimer) the
  instant an ordinary (non-designated-victim) turn completes, and
  `active`/`loading` in status_snapshot already read the real
  per-resident state cap-awarely (via `_resolve_top_level_active_slot`
  / `_resolve_top_level_active_handle`). `grace` and `idle_hot` do not: they
  are the two fields still reading a stranded manager-level scalar nothing
  reachable ever writes.

REGRESSION EVIDENCE: these tests cover a behaviour-changing bug fix. Both tests below
fail without the fix -- see the two "MUST FAIL WITHOUT THE FIX"
paragraphs -- and each carries a non-vacuity / ground-truth
control that must independently confirm the state the test claims to have
reached, so a fixture failure cannot be mistaken for the defect. With the
fix in place, re-run this file: both must go GREEN with
no other change to this file needed, since the same
assertions describe the fixed behaviour.

TREE REQUIREMENT: like the shared fast-lane test fixture module itself, this file
assumes the shared priority-key resolution (`_resident_priority_key`) and the
dispatch unification are both present. If your checkout predates
either, update it rather than working around it -- see the fixture module's own
"TREE REQUIREMENT" docstring section for why.

WIDENING (why Test 2 runs at more than one cap):
Test 2 (the invariant) is widened to run
PARAMETRISED over cap in {1, 2}, not cap=2 alone. Two independent reasons:
(1) cap=1 is the SHIPPING
DEFAULT (config.py `Field(default=1, ...)`) -- a test that only runs at
cap=2 cannot see the configuration most installs actually get; (2) the failure is
cap-general, as this file's module docstring argues above,
and it produces real regressions at cap=1
rather than dead-path noise --
one of them being
`self._idle_handle` never populating at cap=1, i.e. Test 2's own `idle_hot`
scalar. The invariant this file guards is therefore not hypothetical at cap=1;
it is violated in the shipping default without the fix. The cap=2 arm is KEPT and
explicitly labelled deployed-reality (TURBOHAUL_MAX_PARALLEL=2, an operator setting) so
it is not later "simplified away" -- both arms earn their place for different
reasons. Test 1 is NOT widened: it is not framed as an
invariant, and the widening is scoped to "the
invariant" alone.
"""

import asyncio

import pytest

from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.state import open_state_db

from _fastlane_fixture import (
    boot_ranked_runtime,
    seed_manifest,
    make_fakes,
    high_vram,
    wait_until,
    resident_for,
)


def _audit_events(boot, slot_id):
    conn = open_state_db(boot.storage.state_db_path)
    cur = conn.execute(
        "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
        (slot_id,),
    )
    events = [row["event_type"] for row in cur.fetchall()]
    conn.close()
    return events


@pytest.mark.asyncio
class TestStatusSnapshotGraceStrandedSurface:
    async def test_grace_surfaces_during_a_live_per_resident_grace_window(self, tmp_path):
        """TEST 1 -- grace surfaces.

        WHY THIS TEST EXISTS: `/status`'s grace block is what the FE Dashboard
        renders as "this model is warm, N seconds left before it can unload."
        It is built exclusively from the manager-level `self.grace` singleton,
        which -- per this file's module docstring -- can never be armed by
        anything reachable on this tree: `_process_slot` (its only unconditional
        start() site) is dead at every cap, and `submit()`'s own re-arm is
        gated behind a `matches()` check that only a successful start() could
        ever satisfy. `self.grace.expired()` is therefore permanently True.

        WHAT BREAKS IN THE PRODUCT IF THIS GOES RED: the FE Dashboard's grace
        countdown pill goes dark for every model, on every turn, forever --
        even though the resident genuinely has a live warm-hold window a
        same-thread follow-up could still reuse. An operator watching the
        Dashboard sees "nothing is warm" and may manually re-warm, evict, or
        route traffic around a resident that is, in fact, still hot; a
        same-thread follow-up that WOULD be served warm looks, from the
        Dashboard alone, indistinguishable from a cold path. This test exists
        so a fix that re-arms `self.grace` from the new dispatch path, or that
        repoints `status_snapshot` at the real per-resident state, cannot ship
        without keeping this contract green -- and so a future change that
        re-strands the read (e.g. by re-gating it behind a legacy branch)
        breaks immediately and visibly, rather than silently going dark again.

        MUST FAIL WITHOUT THE FIX: the assertions below drive a REAL
        turn to completion with no follow-up queued, confirm -- against
        `r.grace` itself, the ground truth, NOT status_snapshot -- that the
        resident's own grace window is genuinely open, and only THEN check
        `status_snapshot()["grace"]`. Without the fix that is `None` regardless,
        because `self.grace` (what status_snapshot actually reads) was never
        touched by this request at all.

        Non-vacuity: the `r.grace` ground-truth assertion is the control. If
        it fails, the fixture never reached the state under test and the
        failure is a fixture/timing problem, not a defect in
        status_snapshot.
        """
        boot, runtime = boot_ranked_runtime(
            tmp_path, max_parallel_sidecars=2, grace_seconds=10,
        )
        assert runtime.queue.max_parallel_sidecars == 2, (
            "explicit cap -- the env lies (TURBOHAUL_MAX_PARALLEL=2 set by an "
            "operator); a test that inherits a default risks silently "
            "measuring the wrong dispatch path"
        )
        seed_manifest(boot, "m1", main_gpu=0)
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({})
        with high_vram():
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                anchor = await mgr.submit(
                    model_tag="m1", prompt="hi", thread_id="t1", client_meta={},
                )
                # Real driven traffic reaching its own grace_enter audit event
                # -- the same signal the grace-tip test uses --
                # rather than a fixed sleep, which would race grace_seconds.
                await wait_until(
                    lambda: "grace_enter" in _audit_events(boot, anchor.slot_id),
                    timeout=5.0,
                )

                r = resident_for(mgr, "m1")
                # GROUND TRUTH / non-vacuity control.
                assert r.grace is not None and not r.grace.expired(), (
                    "precondition failed: the resident's own grace window is "
                    "not open -- fixture/timing problem, not a status_snapshot "
                    "issue"
                )
                ground_truth_remaining = r.grace.remaining_s()
                assert ground_truth_remaining > 0

                snap = mgr.status_snapshot()
                assert snap["grace"] is not None, (
                    "status_snapshot()['grace'] is None while resident 'm1' "
                    f"has a genuinely open grace window "
                    f"({ground_truth_remaining:.1f}s remaining per r.grace, "
                    "the ground truth) -- the FE Dashboard's grace countdown "
                    "will show nothing for a model that really is warm. This "
                    "is the self.grace-vs-r.grace stranding this test exists "
                    "to catch."
                )
                assert 0 < snap["grace"]["remaining_s"] <= 10, (
                    "status_snapshot()['grace']['remaining_s'] = "
                    f"{snap['grace']['remaining_s']!r}, expected a sane value "
                    "in (0, 10]"
                )
            finally:
                await mgr.shutdown()

    @pytest.mark.parametrize(
        "cap",
        [
            pytest.param(1, id="one-resident-cap-shipping-default"),
            pytest.param(2, id="two-resident-cap-deployed-reality"),
        ],
    )
    async def test_residents_nonempty_implies_not_all_status_scalars_none(self, tmp_path, cap):
        """TEST 2 -- THE INVARIANT, the important one.

        WHY THIS TEST EXISTS: `residents non-empty` MUST IMPLY `NOT all of
        (active, loading, grace, idle_hot) are None` -- because
        `synthesizeResident.ts` (FE) builds its entire rendered model from
        exactly those four scalars, while `api.ts` independently holds the
        live `residents[]` list. When the implication fails the two halves of
        the UI disagree: the resident LIST is full (api.ts) while the
        SYNTHESISED MODEL is empty (all four scalars None) -- which is the FE
        flicker an operator observes in the live dashboard.
        This is stated as an invariant, not a single
        scenario, because it is what catches this whole class of
        defect -- any future manager-level scalar added to
        this same four-field block inherits the same stranding risk the
        instant its one write site stops being reachable, and a scenario test
        pinned to today's specific mechanism would not catch that; this one
        would, because it asserts the OUTPUT CONTRACT, not the internal cause.

        WHY PARAMETRISED OVER cap IN {1, 2} (see this file's module
        docstring "WIDENING" paragraph for the full reasoning; the
        default is declared in config.py): cap=1 is the
        SHIPPING DEFAULT (config.py), and the stranded read produces
        real regressions
        living exactly at cap=1 -- including `self._idle_handle` never
        populating there, this test's own `idle_hot` scalar. A cap=2-only test
        cannot see the configuration most installs actually run. The cap=2 arm
        is KEPT and labelled deployed-reality (a hard constraint of this test,
        TURBOHAUL_MAX_PARALLEL=2, an operator setting) -- do not simplify it away in a
        later edit; it earns its place for a different reason than cap=1 does.

        WHAT BREAKS IN THE PRODUCT IF THIS GOES RED: exactly the reported
        symptom -- residents are genuinely loaded and doing nothing wrong, and
        the Dashboard renders as if nothing is loaded at all. An operator
        cannot trust the Dashboard's "empty" state to mean "actually empty."
        At cap=1 (the shipping default) this is not a hypothetical: the
        scalars really are None there.

        MUST FAIL WITHOUT THE FIX AT BOTH CAPS: at cap=2, two residents
        are driven, via real traffic, through a full turn each with no
        follow-up, to `ResidentState.IDLE_EVICTABLE` -- reproducing the
        symptom frame (`n=2 states=['IDLE_EVICTABLE','IDLE_EVICTABLE']`,
        all four scalars None) rather than a synthetic minimal case. At cap=1,
        only ONE resident is driven: by the cap=1 design (see the
        cap=1 unified-dispatch test suite: only one sidecar is
        spawned at a time), a second model cannot co-reside at cap=1 --
        it would queue behind the first without ever creating a second
        Resident, so asking for two there would be asking the fixture for a
        state production cannot reach at this cap (an unreachable-state fixture
        is a known failure mode to avoid). The invariant itself does not require
        two residents -- "non-empty" holds at len==1 -- so one resident is
        both the honest and the only reachable population at cap=1. Either
        way: `active`/`loading` are correctly None (nothing is ACTIVE or
        LOADING -- not a bug); `grace` is None because `self.grace` is
        stranded (Test 1's subject, cap-general); `idle_hot` is ALSO None
        because `self._idle_handle` / `self.idle.start()` have exactly one
        real call site each, both inside the same dead `_process_slot` (see
        this file's module docstring) -- and this is the SAME call site
        regardless of cap, which is exactly why the invariant is expected RED
        at both.

        Non-vacuity: the residents-count and `ResidentState.IDLE_EVICTABLE`
        checks below are the ground-truth controls -- if either fails, the
        fixture did not reach the expected population for this cap and the
        failure is a fixture/timing problem, not an invariant violation.
        """
        boot, runtime = boot_ranked_runtime(
            tmp_path, max_parallel_sidecars=cap, grace_seconds=0,
            idle_hot_load_seconds=60,
        )
        assert runtime.queue.max_parallel_sidecars == cap, (
            f"explicit cap={cap} -- a test that inherits a default risks "
            "silently measuring the wrong dispatch path"
        )
        if cap >= 2:
            # Distinct main_gpu per model: required for genuine co-residence
            # (the shared fast-lane fixture module's own documented trap) -- two
            # models on one card silently give one resident and a queue, not
            # the two-resident frame.
            seed_manifest(boot, "m1", main_gpu=0)
            seed_manifest(boot, "m2", main_gpu=1)
            expected_tags = {"m1", "m2"}
        else:
            # cap=1: only one resident can ever exist at a time (see the
            # docstring above) -- seeding a second model here would ask the
            # fixture for a state this cap cannot reach.
            seed_manifest(boot, "m1", main_gpu=0)
            expected_tags = {"m1"}

        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({})
        with high_vram():
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                for tag in sorted(expected_tags):
                    await mgr.submit(
                        model_tag=tag, prompt="hi", thread_id=f"t-{tag}",
                        client_meta={},
                    )

                def _idle_tags():
                    return {
                        r.model_tag for r in mgr._model_residents()
                        if r.state is ResidentState.IDLE_EVICTABLE
                    }

                await wait_until(lambda: _idle_tags() == expected_tags, timeout=6.0)

                residents = mgr._model_residents()
                # GROUND TRUTH / non-vacuity control: the expected population
                # for this cap really was reached.
                assert {r.model_tag for r in residents} == expected_tags, (
                    f"cap={cap}: expected residents {expected_tags}, got "
                    f"{[r.model_tag for r in residents]} -- fixture failed to "
                    "reach the expected precondition for this cap"
                )
                assert all(r.state is ResidentState.IDLE_EVICTABLE for r in residents), (
                    f"cap={cap}: expected all residents IDLE_EVICTABLE, got "
                    f"{[(r.model_tag, r.state) for r in residents]}"
                )

                snap = mgr.status_snapshot()
                scalars = {
                    "active": snap["active"],
                    "loading": snap["loading"],
                    "grace": snap["grace"],
                    "idle_hot": snap["idle_hot"],
                }
                assert not all(v is None for v in scalars.values()), (
                    f"INVARIANT VIOLATED at cap={cap}: residents non-empty "
                    f"({sorted(r.model_tag for r in residents)}) but "
                    f"status_snapshot's active/loading/grace/idle_hot are ALL "
                    f"None ({scalars}). This is the captured symptom: "
                    "synthesizeResident.ts builds its FE model exclusively "
                    "from these four scalars while api.ts's independently-"
                    "populated residents[] list stays full -- the two halves "
                    "of the UI disagree, which is the flicker."
                )
            finally:
                await mgr.shutdown()
