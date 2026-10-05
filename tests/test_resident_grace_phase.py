"""The grace countdown is absent from resident cards at cap>=2.

SYMPTOM: the front end shows no grace timer on the resident card; it
just says 'busy'. This is a code unification defect: the per-resident view
lacks what the top-level view already has.

MECHANISM:
  - `r.grace` is the REAL, live grace timer, started by production code in
    `_serve_on_resident` on EVERY turn completion regardless of cap.
  - `_resolve_top_level_grace_resident()` already projects it into
    `status_snapshot()["grace"]` -- so the TOP-LEVEL surface knows the seconds.
  - `_residents_snapshot()` -- the cap>=2 PER-RESIDENT view the card actually reads --
    emits NO grace field at all. Its only time field is `idle_expires_in_s`, gated on
    `r.state is ResidentState.IDLE_EVICTABLE`.
  - `ResidentState.GRACE` has ZERO assignments in the tree (7 executable `.state =`
    assignments -- 4 ACTIVE, 2 DEAD, 1 IDLE_EVICTABLE -- plus the constructor,
    which sets RESERVED_LOADING).
  => During grace the resident payload carries nothing, so the front-end resident card has no
     input and renders no countdown. The card is not failing to show grace; it is
     confidently showing the one other thing that looks identical from the outside.

WHY THE BOOT CONFIG IS PART OF THE TEST: the configuration matters,
because at cap<=1 `_model_residents()` EXCLUDES the legacy
singleton, so `residents[]` is empty, the FE `synthesizeResident` bridge runs instead, and
the timer DOES render. **A single-resident-at-cap-1 test passes and proves nothing** --
that is exactly how a regression can slip through unnoticed. These tests boot
`max_parallel_sidecars=2` EXPLICITLY and assert `residents[]` NON-EMPTY before asserting
anything else, so a configuration regression fails loudly instead of passing vacuously.

If these go red: the grace countdown is absent from every resident card at cap>=2.
"""
import asyncio

import pytest

from tests._fastlane_fixture import (
    boot_ranked_runtime,
    drive_to_active,
    high_vram,
    make_fakes,
    resident_for,
    seed_manifest,
)
from turbohaul.manager import ResidentState, TurbohaulManager

GRACE_SECONDS = 5.0
IDLE_SECONDS = 10.0


async def _drive_real_turn_to_grace(tmp_path):
    """Drive ONE resident admission -> ACTIVE -> turn-complete through the SHARED
    fixture, so its GraceTimer is populated by production code (`_serve_on_resident`'s
    `r.grace.start(...)`) and never hand-constructed.

    Hand-building `r.grace` would reproduce the PAYLOAD without the producer's other
    SIDE EFFECTS, and a fixture that never enters the production state can hold every
    assertion green while the real path is broken.
    """
    boot, runtime = boot_ranked_runtime(
        tmp_path,
        max_parallel_sidecars=2,          # EXPLICIT: the bug's precondition is cap>=2
        grace_seconds=GRACE_SECONDS,
        idle_hot_load_seconds=IDLE_SECONDS,
    )
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
    return mgr, resident_for(mgr, "m1")


def _row_for(mgr, model_tag):
    """The `residents[]` row the FE card is built from, with the non-vacuity guard
    that makes a cap-1 regression fail loudly instead of passing empty."""
    rows = mgr._residents_snapshot()
    assert rows, (
        "NON-VACUITY: _residents_snapshot() is EMPTY. At cap<=1 it is empty BY DESIGN "
        "and the FE bridge renders the timer instead -- so an empty snapshot means this "
        "test is measuring the configuration the bug does not live in. Boot config "
        "regressed away from max_parallel_sidecars=2."
    )
    for row in rows:
        if row["model_tag"] == model_tag:
            return row
    raise AssertionError(f"no residents[] row for {model_tag!r}; rows={rows!r}")


@pytest.mark.asyncio
async def test_resident_row_reports_grace_phase_after_a_real_turn(tmp_path):
    """PROPERTY 1, ALONE: during grace the resident row's `phase` says GRACE.

    `phase` carries the ResidentState value names VERBATIM (a design decision)
    so the fix introduces no fourth vocabulary: it means exactly "what r.state
    would say if GRACE were assignable".
    """
    mgr, r = await _drive_real_turn_to_grace(tmp_path)
    try:
        # FIRING CONTROL: the production state this test claims to measure was actually
        # reached. Without this, a fixture that never starts a timer passes vacuously.
        assert r.grace is not None and not r.grace.expired(), (
            "fixture never entered the production state: a real turn completion must "
            "leave a LIVE GraceTimer on the resident, else this test measures nothing"
        )
        assert r.state is not ResidentState.GRACE, (
            "premise check: r.state is NOT GRACE today (zero assignments) -- if this "
            "ever fails, the FSM changed and this test's reason for existing changed too"
        )
        row = _row_for(mgr, "m1")
        assert row.get("phase") == ResidentState.GRACE.value, (
            "during a live grace window the resident row must report "
            f"phase=='GRACE'; got {row.get('phase')!r}. The card has no other input."
        )
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_resident_row_reports_grace_remaining_seconds(tmp_path):
    """PROPERTY 2, ALONE and independently red-capable: the row carries the SECONDS.

    Kept separate from the phase test on purpose (when a control
    asserts more than one property, prove each is independently red-capable -- ordering
    hides the dead half). A phase label with no number still renders no countdown, which
    is precisely why assigning ResidentState.GRACE would not have fixed this.
    """
    mgr, r = await _drive_real_turn_to_grace(tmp_path)
    try:
        assert r.grace is not None and not r.grace.expired(), (
            "fixture never entered the production state -- see property 1"
        )
        row = _row_for(mgr, "m1")
        remaining = row.get("remaining_s")
        assert remaining is not None, (
            "the resident row must carry remaining_s during grace; "
            "the front-end resident card renders a countdown from a NUMBER, not from a state label"
        )
        assert 0 < remaining <= GRACE_SECONDS, (
            f"remaining_s must be a live countdown in (0, {GRACE_SECONDS}], "
            f"got {remaining!r}"
        )
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_a_resident_serving_a_followup_reports_GRACE_BY_DECISION(tmp_path):
    """⭐ PINS A DELIBERATE DECISION, NOT A BUG. (Design decision.)

    A resident serving a follow-up INSIDE its grace window reports phase=='GRACE'. If you
    are here because that looks wrong and you want to add a "serving beats grace" guard:
    read `_resident_phase`'s docstring first. The short version:

      · Both facts are true at once -- the grace window really is live during a follow-up,
        which is what makes warm-slot reuse legal. One `phase` is a lossy projection.
      · The TOP-LEVEL surface reports grace in this same window too (its own gate is
        `active_info is None`, and active_info IS None here). A per-resident precedence
        would make the two surfaces DISAGREE -- the exact defect this change removes.
      · No signal exists to implement it: r.inflight is always 0 even mid-turn,
        r.active_slot is never cleared, _resolve_top_level_active_slot() is None throughout
        a model-keyed turn, and r.inbox.qsize() flickers within one in-flight window.

    A guard of the form `r.state is ACTIVE and r.inflight` would be
    UNREACHABLE: mutating it away would leave every test passing. This test exists so
    the next reader meets the decision as an executable fact rather than as prose.
    """
    boot, runtime = boot_ranked_runtime(
        tmp_path,
        max_parallel_sidecars=2,
        grace_seconds=GRACE_SECONDS,
        idle_hot_load_seconds=IDLE_SECONDS,
    )
    seed_manifest(boot, "m1", main_gpu=0)
    gate = asyncio.Event()
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate})
    with high_vram():
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
            sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        t2 = None
        try:
            t1 = await drive_to_active(mgr, "m1", thread_id="t1", client_meta={})
            gate.set()
            await asyncio.wait_for(t1, timeout=5.0)
            r = resident_for(mgr, "m1")
            assert r.grace is not None and not r.grace.expired(), (
                "premise 1: turn 1 must leave a LIVE GraceTimer"
            )
            grace_thread = r.grace.thread_id

            # A SECOND turn, held open by the gate, while turn 1's window is still live.
            gate.clear()
            t2 = asyncio.create_task(
                mgr.submit_and_wait("m1", "p2", thread_id="t2", client_meta={})
            )
            observed = []
            for _ in range(6):
                await asyncio.sleep(0.05)
                if t2.done():
                    break
                observed.append(_row_for(mgr, "m1")["phase"])

            # PREMISES, asserted before the property: the window this pins must exist.
            assert len(observed) >= 4, (
                f"premise 2: needed samples taken while turn 2 was genuinely in flight; "
                f"got {len(observed)} -- the fixture no longer holds the turn open and "
                f"this test is pinning nothing"
            )
            r = resident_for(mgr, "m1")
            assert r.grace is not None and not r.grace.expired(), (
                "premise 3: turn 1's grace window must still be live during turn 2"
            )
            assert r.grace.thread_id == grace_thread, (
                "premise 4: the grace window must still belong to TURN 1 -- if it has been "
                "restarted for turn 2 then the two facts are no longer simultaneous and "
                "this test is measuring something else"
            )

            assert set(observed) == {ResidentState.GRACE.value}, (
                "deliberate decision: a resident serving a follow-up inside "
                f"its grace window reports GRACE, consistently. Observed {observed!r}. A "
                "MIXED result is the real failure here -- it would mean the card strobes "
                "between phases while the engine works."
            )
        finally:
            gate.set()
            if t2 is not None:
                try:
                    await asyncio.wait_for(t2, timeout=5.0)
                except Exception:
                    pass
            await mgr.shutdown()


@pytest.mark.asyncio
async def test_idle_expires_in_s_alias_is_unchanged_during_grace(tmp_path):
    """GREEN CONTROL, NOT A RED: the additive alias must not change behaviour.

    A design condition is that `idle_expires_in_s` be DERIVED FROM THE
    SAME RESOLVER rather than computed alongside it -- otherwise the two-sources-of-truth
    defect this change exists to remove is re-created in a new place under the word "alias".

    This test pins the alias's OBSERVABLE contract: a resident in grace is not
    IDLE_EVICTABLE, so `idle_expires_in_s` stays None. It PASSES BEFORE the fix and must still
    pass after the fix. It is stated as a green control so nobody later reads it as a
    gate it never was.
    """
    mgr, r = await _drive_real_turn_to_grace(tmp_path)
    try:
        assert r.grace is not None and not r.grace.expired(), (
            "fixture never entered the production state -- see property 1"
        )
        row = _row_for(mgr, "m1")
        assert row.get("idle_expires_in_s") is None, (
            "alias regression: a resident in GRACE is not IDLE_EVICTABLE, so the "
            f"idle alias must stay None; got {row.get('idle_expires_in_s')!r}"
        )
    finally:
        await mgr.shutdown()
