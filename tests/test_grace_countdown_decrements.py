"""The grace countdown must actually COUNT DOWN.

Contract: the grace timer must show up in the FE
only when it is actually counting down, and it must count down on both the FE and
the BE.

⛔ WHY THIS FILE EXISTS — IT IS A **TEST** GAP, NOT A CODE GAP.
The per-resident and fast-lane semantics are already met STRUCTURALLY at
the current baseline. What nothing asserted is that the number MOVES. Seven
python and seven FE files touch ``remaining_s`` and **not one asserts it decreases**:

  the resident-grace-phase test                 -> RANGE only: ``0 < remaining <= GRACE``
  the grace-idle FE-reporting test              -> RANGE only
  the grace-timer FE-reporting test             -> RANGE only
  the FE grace-countdown render test            -> STATIC render from a fixed value

A bounds check cannot distinguish a live countdown from a FROZEN one: a constant 4 sits
inside ``(0, GRACE_SECONDS]`` forever and every one of those files stays green. That is
a familiar shape — a surface that is confidently rendering something, while
the property that matters is untested.

=== WHY THE WINDOW IS DRIVEN BY REAL TRAFFIC AND NOT HAND-SET ===
``r.grace`` is started by production code in ``_serve_on_resident`` on turn completion.
Hand-building it would reproduce the PAYLOAD without the producer's other SIDE EFFECTS,
and a fixture that never enters the production state can hold every assertion green while
the real path is broken. (The existing grace-reporting tests make this point too; that exact
defect is why the shared ``_drive_real_turn_to_grace`` pattern exists.)

=== WHY THIS IS DETERMINISTIC AND NOT A FLAKY TIMING TEST ===
The resolver truncates: ``_resident_phase`` -> ``max(0, int(grace.remaining_s()))``, and
``_residents_snapshot`` passes a FRESH ``time.monotonic()`` on every call. So
``remaining(t) == int(E - t)``, and for two samples separated by ``dt > 1.0`` s,
``int(x) - int(x - dt) >= 1`` **always**. A strict decrease is arithmetic, not a race.
``SAMPLE_GAP_S`` is 6.0 s against a 30 s window — 5x margin before the window closes.
The gap is 6.0 rather than the minimum viable 1.2 so that the RATE test below also
discriminates a TOO-SLOW countdown by construction; see the comment on the constant.

⚠ The elapsed time is MEASURED and reported, never assumed: under load ``asyncio.sleep``
may overshoot, which only widens the gap, but if the loop were ever starved so badly that
it UNDERSHOT, the failure message says so instead of blaming the countdown.

=== THE THREE FAILURE MODES ARE KEPT SEPARATE, ON PURPOSE ===
A setup miss must never masquerade as a decrement failure (a test must be watched RED for the
RIGHT REASON). So the preconditions carry their OWN distinct messages:
  * fixture never reached grace            -> "FIXTURE" message, not a decrement verdict
  * window closed between the two samples  -> "INCONCLUSIVE" message, not a decrement verdict
  * both samples valid, second not smaller -> **the decrement assertion** — the one the
    FREEZE MUTANT must trip.
"""
import asyncio
import time

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

# PRODUCTION DEFAULTS, deliberately not shortened for convenience (never touch
# timing config to make a test easy). config.py: grace_seconds=30, idle_hot_load_seconds=600.
GRACE_SECONDS = 30.0
IDLE_SECONDS = 600.0

# ⛔ 6.0 IS LOAD-BEARING. DO NOT "OPTIMISE" IT BACK DOWN TO ~1.2 s.
#
# 1.2 s is the MINIMUM that makes the DECREMENT test sound: any gap > 1.0 s forces
# int() truncation to drop by >= 1, so a FROZEN countdown cannot hide. That much is
# arithmetic and 1.2 s would be enough -- if frozen were the only defect worth catching.
#
# It is not, and 1.2 s is NOT enough for the RATE test (the too-slow property):
#   * at a 1.2 s gap the rate window is `1..2`, so the +/-1 truncation tolerance IS THE
#     WHOLE SIGNAL -- there is no room left to tell "correct" from "wrong but close";
#   * a HALF-rate clock advances only 0.6 s across a 1.2 s gap, which is BELOW the 1.0 s
#     floor the determinism argument above requires. So whether the truncated value drops
#     by 0 or by 1 depends on WHERE IN THE SECOND sample 1 lands -- i.e. on fixture phase,
#     not on the property under test. A half-rate clock could fail on some runs, but
#     with only ~0.7 s of drift between fixture and boundary it would PASS instead.
#     A check that discriminates only by phase is a flaky gate, and a flaky gate launders a pass.
#
# At 6.0 s the tolerance becomes small relative to elapsed and the discrimination is
# arithmetic again: a half-rate clock yields drop 3 against a required 5..7 -- outside by
# a margin of 2, deterministically. (Measured. At 10.0 s the margin is 4; 6.0 s buys the
# property for 9.5 s of runtime, against a 30 s grace window that still leaves 5x headroom.)
#
# ⇒ Shrinking this constant does not make the suite faster, it silently deletes the
#   too-slow half of what these tests claim -- which is the same "asserts less than its
#   name implies" defect this whole file exists to close.
SAMPLE_GAP_S = 6.0


async def _drive_real_turn_to_grace(tmp_path):
    """ONE admission -> ACTIVE -> turn-complete, so the GraceTimer is populated by
    production code (``_serve_on_resident``) and never hand-constructed."""
    boot, runtime = boot_ranked_runtime(
        tmp_path,
        max_parallel_sidecars=2,  # EXPLICIT: the per-resident view is empty at cap<=1
        grace_seconds=GRACE_SECONDS,
        idle_hot_load_seconds=IDLE_SECONDS,
    )
    seed_manifest(boot, "m1", main_gpu=0)
    gate = asyncio.Event()
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate})
    with high_vram():
        mgr = TurbohaulManager(
            boot,
            runtime,
            spawn_fn=spawn_fn,
            health_fn=health_fn,
            sigterm_fn=sigterm_fn,
            vram_fn=vram_fn,
            complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        task = await drive_to_active(mgr, "m1", thread_id="t1", client_meta={})
        gate.set()
        await asyncio.wait_for(task, timeout=5.0)
    return mgr, resident_for(mgr, "m1")


def _row_for(mgr, model_tag):
    """The ``residents[]`` row the FE card is built from, with the non-vacuity guard that
    makes a cap-1 regression fail loudly instead of passing empty."""
    rows = mgr._residents_snapshot()
    assert rows, (
        "NON-VACUITY: _residents_snapshot() is EMPTY. At cap<=1 it is empty BY DESIGN, so "
        "an empty snapshot means this test is measuring a configuration the countdown "
        "does not live in. Boot config regressed away from max_parallel_sidecars=2."
    )
    for row in rows:
        if row["model_tag"] == model_tag:
            return row
    raise AssertionError(f"no residents[] row for {model_tag!r}; rows={rows!r}")


def _grace_sample(mgr, model_tag, *, which):
    """Read ``remaining_s`` off the per-resident row, asserting the PRECONDITION with its
    own distinct message so a setup/window problem can never be read as a frozen counter."""
    row = _row_for(mgr, model_tag)
    phase = row.get("phase")
    remaining = row.get("remaining_s")
    assert phase == ResidentState.GRACE.value and remaining is not None, (
        f"PRECONDITION FAILED AT SAMPLE {which} -- this is NOT a decrement verdict. "
        f"The resident row must still be in a live grace window to be sampled; got "
        f"phase={phase!r} remaining_s={remaining!r}. If this fires at sample 2 the window "
        f"closed between reads (grace_seconds={GRACE_SECONDS}, gap={SAMPLE_GAP_S}s) and "
        f"the run is INCONCLUSIVE, not a failure of the countdown."
    )
    return remaining


@pytest.mark.asyncio
async def test_grace_remaining_s_strictly_decreases_across_real_elapsed_time(tmp_path):
    """THE REQUIRED PROPERTY: the number goes DOWN.

    This is the assertion a FREEZE MUTANT (``_resident_phase`` returning a constant
    ``remaining_s``) must trip. A range check cannot: a frozen 29 satisfies
    ``0 < x <= 30`` forever.
    """
    mgr, r = await _drive_real_turn_to_grace(tmp_path)
    try:
        # FIRING CONTROL: the production state this test claims to measure was reached.
        assert r.grace is not None and not r.grace.expired(), (
            "FIXTURE never entered the production state -- a real turn completion must "
            "leave a LIVE GraceTimer on the resident. This is NOT a decrement verdict."
        )

        t0 = time.monotonic()
        first = _grace_sample(mgr, "m1", which=1)
        await asyncio.sleep(SAMPLE_GAP_S)
        second = _grace_sample(mgr, "m1", which=2)
        elapsed = time.monotonic() - t0

        # Guard the instrument itself: if the loop were starved so badly that less than a
        # whole second actually passed, truncation no longer guarantees a decrease and the
        # right answer is "inconclusive", not "the countdown is frozen".
        assert elapsed > 1.0, (
            f"INCONCLUSIVE, not a decrement verdict: only {elapsed:.3f}s of real time "
            f"elapsed between samples (needed > 1.0s for int() truncation to guarantee a "
            f"strict decrease). The event loop was starved; re-run."
        )

        assert second < first, (
            f"The grace countdown is NOT COUNTING DOWN. remaining_s was "
            f"{first} and is still {second} after {elapsed:.3f}s of real elapsed time "
            f"inside a {GRACE_SECONDS}s window. The card renders this number directly, so "
            f"a frozen value is a frozen timer on the user's screen. "
            f"(int() truncation guarantees a drop of >=1 for any gap > 1.0s.)"
        )
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_grace_countdown_decrease_tracks_the_elapsed_time_not_merely_any_change(tmp_path):
    """STRONGER THAN 'it changed': the drop must be CONSISTENT WITH THE CLOCK.

    Kept separate from the test above (rule: one property per test, each
    independently red-capable). A mutant that merely jitters the value -- or one that
    decrements by a fixed step per CALL rather than per SECOND -- passes "it went down"
    while still not being a clock. This pins the magnitude to measured elapsed time.
    """
    mgr, r = await _drive_real_turn_to_grace(tmp_path)
    try:
        assert r.grace is not None and not r.grace.expired(), (
            "FIXTURE never entered the production state -- see the decrement test."
        )
        t0 = time.monotonic()
        first = _grace_sample(mgr, "m1", which=1)
        await asyncio.sleep(SAMPLE_GAP_S)
        second = _grace_sample(mgr, "m1", which=2)
        elapsed = time.monotonic() - t0

        drop = first - second
        # int() truncation puts the drop within one second of the elapsed time in either
        # direction; anything outside that is not a wall-clock countdown.
        #
        # ⛔ THE `max(1, ...)` IS LOAD-BEARING: A FROZEN COUNTER MUST FAIL HERE.
        # A bare `int(elapsed) - 1` floor is wrong. At the minimum viable gap
        # (1.2s) that lower bound is ZERO, so a FROZEN counter (drop == 0) would sit inside
        # the tolerance and this test would PASS for a frozen counter while the test
        # above correctly goes red. A rate check whose window admits zero is not a rate
        # check. Clamping the floor to 1 makes "it never moved" fail here too, without
        # weakening the upper bound that catches a per-CALL decrementer.
        lo = max(1, int(elapsed) - 1)
        hi = int(elapsed) + 1
        assert lo <= drop <= hi, (
            f"The countdown moved, but NOT AT THE RATE OF THE CLOCK. "
            f"remaining_s dropped by {drop} ({first} -> {second}) across {elapsed:.3f}s of "
            f"real time; a wall-clock countdown truncated by int() must drop by "
            f"{lo}..{hi}. A value that changes per CALL rather than per SECOND satisfies "
            f"'it decreased' and is still not a timer."
        )
    finally:
        await mgr.shutdown()
