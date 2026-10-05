"""A resident designated MID-GRACE must surrender the window at once.

THE RULE UNDER TEST. A designated victim surrenders both the grace timer and
the idle-unload countdown immediately on designation, not when its turn ends.
The entry-time `skip_grace` check in manager.py
is evaluated ONCE. A resident that enters grace as a NON-victim and is
designated five seconds later then sits the FULL window, because the loop never
re-asks.

SYMPTOM: a registered higher-priority claim was admitted only
when grace expired: the eviction took about as long as the configured
grace window -- i.e. served by
grace EXPIRY, not by designation.

WHY THE SEAM IS A MONKEYPATCHED PREDICATE. The defect is precisely "the loop
does not re-ask". Reproducing the full designation machinery mid-window would
test the machinery; flipping the predicate tests the ONLY thing that changed --
whether the loop asks again. The fixture asserts BOTH the False-at-entry and
True-later halves actually happened, so a patch that never calls the predicate
a second time cannot pass by accident.

THE NON-VICTIM RULE IS NOT WEAKENED, AND THIS FILE PROVES IT. `test_a_non_victim_still_runs_its
_full_window` is the negative control: a resident that is NEVER designated must
still count its grace down fully. Without that control, "breaks out early" would
pass just as well for code that broke out for everyone -- which is the
forbidden behaviour and a violation of that rule.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from turbohaul.slot import Slot, SlotState

from test_victim_enters_no_grace import (  # noqa: E402
    _audit_events,
    _boot_runtime,
    _claimant,
    _mgr,
    _resident,
    _settle_bg,
)

_SHORT_GRACE = 8.0
# The break must be decisively faster than the window, not marginally: a flaky
# scheduler must not be able to make "broke out" and "waited it out" overlap.
_BREAK_CEILING = 3.0


async def _survivor_in_grace(tmp_path, *, designate_after: int | None):
    """A resident that is NOT the designated target at grace entry.

    designate_after=None  -> never designated (the non-victim negative control)
    designate_after=1     -> False on the entry check, True on every loop check
    """
    boot, runtime = _boot_runtime(tmp_path, grace_seconds=_SHORT_GRACE)
    mgr = _mgr(boot, runtime)
    survivor, s_handle = _resident(mgr, "survivor-model", 59902, last_active=1000.0)
    _resident(mgr, "victim-model", 59901, last_active=1.0)
    mgr._register_fastlane_claim_locked(_claimant(), "make_room_starved_count_cap")
    await _settle_bg(mgr)

    assert mgr._is_designated_unload_target_locked(survivor) is False, (
        "fixture precondition: the resident under test must NOT be the "
        "designated target at entry, or it takes the skip_grace branch and "
        "never reaches the loop this change affects -- every assertion would be vacuous"
    )

    calls = {"n": 0}
    real = mgr._is_designated_unload_target_locked

    def flipping(r):
        if r is not survivor:
            return real(r)
        calls["n"] += 1
        if designate_after is None:
            return False
        return calls["n"] > designate_after

    mgr._is_designated_unload_target_locked = flipping

    completions: list[str] = []
    real_oc = mgr._telemetry.on_completion
    mgr._telemetry.on_completion = lambda slot, reason: (
        completions.append(reason), real_oc(slot, reason))[1]

    return boot, mgr, survivor, s_handle, calls, completions


@pytest.mark.asyncio
class TestMidGraceDesignationSurrendersTheWindow:
    """RED on the pre-fix code: the loop never re-asks, so every test here times out at
    the full grace window or misses the new event."""

    async def test_designation_mid_window_breaks_out_early(self, tmp_path):
        boot, mgr, survivor, handle, calls, _ = await _survivor_in_grace(
            tmp_path, designate_after=1)
        anchor = Slot.new("survivor-model", prompt="hi", thread_id="t-mid")
        anchor.state = SlotState.STAGED

        t0 = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(survivor, anchor, handle),
            timeout=_SHORT_GRACE + 4.0,
        )
        elapsed = time.monotonic() - t0

        assert calls["n"] >= 2, (
            "the predicate was asked "
            f"{calls['n']} time(s). ONE call means the loop never re-checked -- "
            "that IS the defect, and no timing assertion below could tell you so"
        )
        assert elapsed < _BREAK_CEILING, (
            f"designated mid-grace but still sat {elapsed:.2f}s of a "
            f"{_SHORT_GRACE}s window -- grace was NOT surrendered the instant "
            "it was designated"
        )
        await mgr.shutdown()

    async def test_it_emits_a_DISTINCT_event_not_the_never_entered_grace_one(
            self, tmp_path):
        """`grace_designated_victim_skip` means, per the manager.py comment, "no
        GRACE state, no grace_started_at, no GraceTimer window and no
        grace_enter event -- the ONLY grace-family event on its slot". This slot
        has all four. Reusing that key would put a provable falsehood on the
        audit surface and emit two states that have always been exclusive."""
        boot, mgr, survivor, handle, _, _ = await _survivor_in_grace(
            tmp_path, designate_after=1)
        anchor = Slot.new("survivor-model", prompt="hi", thread_id="t-evt")
        anchor.state = SlotState.STAGED
        await asyncio.wait_for(
            mgr._serve_on_resident(survivor, anchor, handle),
            timeout=_SHORT_GRACE + 4.0)
        await _settle_bg(mgr)

        events = _audit_events(boot, anchor.slot_id)
        assert "grace_enter" in events, (
            f"fixture sanity: this resident DID enter grace, events={events!r}")
        assert "grace_designated_unload_target_break" in events, (
            f"the mid-window surrender must be visible on the wire, events={events!r}")
        assert "grace_designated_victim_skip" not in events, (
            "REUSING THE NEVER-ENTERED-GRACE KEY FOR A SLOT THAT ENTERED GRACE. "
            f"events={events!r} -- grace_enter and grace_designated_victim_skip "
            "have always been mutually exclusive; this makes them co-occur")

    async def test_exactly_one_on_completion_per_anchor_turn(self, tmp_path):
        """manager.py states the invariant in the file itself:
        "still exactly one on_completion per anchor turn, either way". The
        `else` branch already fired one before the loop was entered,
        so a second call here double-counts completion telemetry."""
        _boot, mgr, survivor, handle, _, completions = await _survivor_in_grace(
            tmp_path, designate_after=1)
        anchor = Slot.new("survivor-model", prompt="hi", thread_id="t-oc")
        anchor.state = SlotState.STAGED
        await asyncio.wait_for(
            mgr._serve_on_resident(survivor, anchor, handle),
            timeout=_SHORT_GRACE + 4.0)

        assert completions == ["grace_enter"], (
            "exactly one on_completion per anchor turn -- got "
            f"{completions!r}. A second call emits MORE telemetry than the "
            "cap<=1 path ever does, which the manager.py comment names as 'new "
            "behaviour, not parity'")
        await mgr.shutdown()

    async def test_a_non_victim_still_runs_its_full_window(self, tmp_path):
        """★ THE NON-VICTIM NEGATIVE CONTROL, and the reason the other three mean
        anything. The rule: a client that is NOT the designated victim is not
        evicted for the waiting request -- its grace counts down fully.
        A breakout that fires on "someone higher-priority is
        waiting" would hit non-victims. If this test ever goes GREEN alongside
        an early break, the fix has re-introduced that violation."""
        _boot, mgr, survivor, handle, calls, _ = await _survivor_in_grace(
            tmp_path, designate_after=None)
        anchor = Slot.new("survivor-model", prompt="hi", thread_id="t-ctl")
        anchor.state = SlotState.STAGED

        t0 = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(survivor, anchor, handle),
            timeout=_SHORT_GRACE + 6.0)
        elapsed = time.monotonic() - t0

        assert calls["n"] >= 2, (
            "control is vacuous unless the loop actually asked repeatedly and "
            f"kept getting False -- asked {calls['n']} time(s)")
        assert elapsed >= _SHORT_GRACE - 1.0, (
            f"a NON-designated resident left grace after {elapsed:.2f}s of a "
            f"{_SHORT_GRACE}s window. The rule says its grace counts down FULLY. "
            "The breakout is firing for non-victims -- that is the forbidden "
            "behaviour")
        await mgr.shutdown()

