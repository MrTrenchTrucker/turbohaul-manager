"""Which engine the instance picker chooses, pinned against the picker's behaviour in v0.8.0.

The picker ``TurbohaulManager._pick_instance_for`` hands a new session to the least loaded live engine
of a model: ready engines before booting ones, then the engine with the least pending work, then the
engine that has been idle the longest. The only deliberate rule is how an engine that sits in its
grace window after a finished turn is counted: the turn it has already finished is not pending work.

Every case here is a small set of engines built by hand (no server, no real model) and handed to the
real picker. For each case the test reads which engine was chosen and which engine the new session's
affinity record now points to, with the engines listed in the given order and in the reverse order.

Two kinds of case, marked on each one in plain words:

* ``unchanged from v0.8.0``: no engine is in its grace window with a finished turn counted against it,
  or counting it makes no difference to the choice. The expected values are literals captured from the
  v0.8.0 picker.
* ``changed since v0.8.0``: an engine in its grace window with a finished turn competes with an
  engine of equal or lower other work. The v0.8.0 picker counted the finished turn as one unit of
  load and chose differently; the expected values here are the intended choice.
"""

from __future__ import annotations

import asyncio

import _resident_identity_harness as H
import pytest
import pytest_asyncio

from turbohaul.manager import Resident, ResidentState
from turbohaul.slot import Slot, SlotState

TAG = H.MODEL
FIRST = TAG
SECOND = f"{TAG}#1"
THIRD = f"{TAG}#2"
UNCHANGED = "unchanged from v0.8.0"
CHANGED = "changed since v0.8.0"

IDLE = ResidentState.IDLE_EVICTABLE
BOOTING = ResidentState.RESERVED_LOADING
DEAD = ResidentState.DEAD


def eng(key, last, *, state=IDLE, turn=False, riders=0, inbox=0, grace=False, slot_state=None):
    """A description of one engine: ``turn`` sets a slot on it, ``riders`` is the count in flight,
    ``inbox`` the queue depth, ``grace`` the grace-window flag, ``last`` its last-active time.

    ``slot_state`` is the state of the slot set on the engine. By default an engine in its grace
    window carries the slot of the turn it has finished (state GRACE) and any other engine carries
    a slot that is running (state ACTIVE)."""
    if slot_state is None:
        slot_state = SlotState.GRACE if grace else SlotState.ACTIVE
    return {"key": key, "last": last, "state": state, "turn": turn, "riders": riders,
            "inbox": inbox, "grace": grace, "slot_state": slot_state}


# name -> (kind, plain-words description, engines in listed order)
CASES = {
    "idle_vs_idle_the_longer_idle_wins": (
        UNCHANGED, "two idle engines: the one idle for longer is chosen",
        [eng(FIRST, 1.0), eng(SECOND, 2.0)]),
    "idle_vs_idle_the_longer_idle_wins_second_engine": (
        UNCHANGED, "the same with the second engine idle for longer",
        [eng(FIRST, 2.0), eng(SECOND, 1.0)]),
    "mid_turn_vs_idle": (
        UNCHANGED, "an engine in the middle of a turn loses to an idle one, even when it was idle for longer",
        [eng(FIRST, 1.0, turn=True), eng(SECOND, 2.0)]),
    "riders_two_vs_one": (
        UNCHANGED, "two riders in flight lose to one",
        [eng(FIRST, 1.0, riders=2), eng(SECOND, 2.0, riders=1)]),
    "inbox_two_vs_empty": (
        UNCHANGED, "an inbox of two loses to an empty inbox",
        [eng(FIRST, 1.0, inbox=2), eng(SECOND, 2.0)]),
    "turn_plus_rider_vs_one_rider": (
        UNCHANGED, "a running turn with a rider is more load than a single rider",
        [eng(FIRST, 1.0, turn=True, riders=1), eng(SECOND, 2.0, riders=1)]),
    "ready_beats_booting": (
        UNCHANGED, "a booting engine with no load loses to a ready engine that is mid-turn",
        [eng(FIRST, 1.0, state=BOOTING), eng(SECOND, 2.0, turn=True)]),
    "booting_vs_booting_by_inbox": (
        UNCHANGED, "among booting engines the one with the shorter inbox is chosen",
        [eng(FIRST, 1.0, state=BOOTING, inbox=1), eng(SECOND, 2.0, state=BOOTING)]),
    "equal_load_the_longer_idle_wins": (
        UNCHANGED, "two engines mid-turn: the tie goes to the engine idle for longer before its turn",
        [eng(FIRST, 5.0, turn=True), eng(SECOND, 3.0, turn=True)]),
    "complete_tie_first_in_the_list": (
        UNCHANGED, "everything equal: the first engine in the list is chosen",
        [eng(FIRST, 1.0), eng(SECOND, 1.0)]),
    "single_engine": (
        UNCHANGED, "one engine: it is chosen even when busy",
        [eng(FIRST, 1.0, turn=True, riders=2, inbox=3)]),
    "one_dead_engine_is_skipped": (
        UNCHANGED, "a dead engine with no load is never chosen",
        [eng(FIRST, 1.0, state=DEAD), eng(SECOND, 2.0, turn=True)]),
    "only_dead_engines": (
        UNCHANGED, "only dead engines: nothing is chosen",
        [eng(FIRST, 1.0, state=DEAD), eng(SECOND, 2.0, state=DEAD)]),
    "three_engines_least_loaded": (
        UNCHANGED, "three engines: the idle one that has waited longest is chosen",
        [eng(FIRST, 1.0, turn=True), eng(SECOND, 3.0), eng(THIRD, 2.0)]),
    "grace_engine_with_a_waiting_request_vs_idle": (
        UNCHANGED, "an engine in its grace window with a request already waiting in its inbox loses to an idle engine",
        [eng(FIRST, 1.0, turn=True, grace=True, inbox=1), eng(SECOND, 2.0)]),
    "idle_engine_idle_longer_than_a_grace_engine": (
        UNCHANGED, "an idle engine that has been idle for longer than an engine in its grace window is chosen "
                   "both before and after",
        [eng(FIRST, 1.0), eng(SECOND, 2.0, turn=True, grace=True)]),
    "grace_engine_vs_running_turn_and_a_waiting_request": (
        UNCHANGED, "a grace engine against an engine with a running turn and a waiting request: "
                   "the grace engine wins either way",
        [eng(FIRST, 1.0, turn=True, inbox=1), eng(SECOND, 2.0, turn=True, grace=True)]),
    "follow_up_running_in_grace_window_vs_idle": (
        UNCHANGED, "an engine in its grace window that is running a follow-up turn (slot ACTIVE) is loaded: "
                   "the idle engine is chosen, as before",
        [eng(FIRST, 1.0, turn=True, grace=True, slot_state=SlotState.ACTIVE), eng(SECOND, 2.0)]),
    "follow_up_just_promoted_in_grace_window_vs_idle": (
        UNCHANGED, "the same with the follow-up slot in state ACTIVE_MATCH: still loaded, the idle engine is chosen",
        [eng(FIRST, 1.0, turn=True, grace=True, slot_state=SlotState.ACTIVE_MATCH), eng(SECOND, 2.0)]),
    "finished_slot_state_outside_the_grace_loop_counts": (
        UNCHANGED, "a slot in state GRACE on an engine that is not in its grace loop is counted as a live turn: "
                   "the idle engine is chosen, as before",
        [eng(FIRST, 1.0, turn=True, grace=False, slot_state=SlotState.GRACE), eng(SECOND, 2.0)]),
    # ---- the finished turn of an engine in its grace window is not load -------------------------
    "grace_engine_vs_running_turn": (
        CHANGED, "an engine in its grace window with a finished turn against an engine in the middle of a turn: "
                 "the engine in grace is chosen (it used to tie on load and lose to the longer idle one)",
        [eng(FIRST, 1.0, turn=True), eng(SECOND, 2.0, turn=True, grace=True)]),
    "grace_engine_vs_one_rider": (
        CHANGED, "an engine in its grace window against an engine with one rider in flight: "
                 "the engine in grace is chosen",
        [eng(FIRST, 1.0, riders=1), eng(SECOND, 2.0, turn=True, grace=True)]),
    "grace_engine_vs_one_waiting_request": (
        CHANGED, "an engine in its grace window against an engine with one request waiting: "
                 "the engine in grace is chosen",
        [eng(FIRST, 1.0, inbox=1), eng(SECOND, 2.0, turn=True, grace=True)]),
    "finished_grace_engine_vs_follow_up_in_grace_window": (
        CHANGED, "an engine that finished its turn in its grace window against an engine running a follow-up "
                 "in its own grace window: only the finished one stops counting, so it is chosen (the two used "
                 "to tie and the longer idle follow-up engine won)",
        [eng(FIRST, 1.0, turn=True, grace=True, slot_state=SlotState.ACTIVE), eng(SECOND, 2.0, turn=True, grace=True)]),
    "grace_engine_longer_idle_vs_idle_engine": (
        CHANGED, "an engine in its grace window that has been quiet for longer than an idle engine: both carry no "
                 "load now, so the longer idle one, the engine in grace, is chosen (the idle engine used to win)",
        [eng(FIRST, 1.0, turn=True, grace=True), eng(SECOND, 2.0)]),
}

# Literals captured from the v0.8.0 picker, for the cases marked unchanged:
# name -> (engine chosen, engine the affinity record points to) with the engines in the listed order,
# then the same with the list reversed.
FROZEN = {'booting_vs_booting_by_inbox': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'complete_tie_first_in_the_list': [['duo', 'duo'], ['duo#1', 'duo#1']],
   'equal_load_the_longer_idle_wins': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'finished_slot_state_outside_the_grace_loop_counts': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'follow_up_just_promoted_in_grace_window_vs_idle': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'follow_up_running_in_grace_window_vs_idle': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'grace_engine_vs_running_turn_and_a_waiting_request': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'grace_engine_with_a_waiting_request_vs_idle': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'idle_engine_idle_longer_than_a_grace_engine': [['duo', 'duo'], ['duo', 'duo']],
   'idle_vs_idle_the_longer_idle_wins': [['duo', 'duo'], ['duo', 'duo']],
   'idle_vs_idle_the_longer_idle_wins_second_engine': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'inbox_two_vs_empty': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'mid_turn_vs_idle': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'one_dead_engine_is_skipped': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'only_dead_engines': [[None, None], [None, None]],
   'ready_beats_booting': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'riders_two_vs_one': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']],
   'single_engine': [['duo', 'duo'], ['duo', 'duo']],
   'three_engines_least_loaded': [['duo#2', 'duo#2'], ['duo#2', 'duo#2']],
   'turn_plus_rider_vs_one_rider': [['duo#1', 'duo#1'], ['duo#1', 'duo#1']]}

# The intended choice for the cases marked changed (both list orders).
INTENDED = {
    "grace_engine_vs_running_turn": SECOND,
    "grace_engine_vs_one_rider": SECOND,
    "grace_engine_vs_one_waiting_request": SECOND,
    "finished_grace_engine_vs_follow_up_in_grace_window": SECOND,
    "grace_engine_longer_idle_vs_idle_engine": FIRST,
}


def build(spec):
    r = Resident(
        model_tag=TAG, resident_key=spec["key"], state=spec["state"],
        last_active_monotonic=spec["last"], inbox=asyncio.Queue(),
    )
    if spec["turn"]:
        r.active_slot = Slot.new(TAG)
        r.active_slot.state = spec["slot_state"]
    r.inflight.extend(Slot.new(TAG) for _ in range(spec["riders"]))
    for _ in range(spec["inbox"]):
        r.inbox.put_nowait(Slot.new(TAG))
    r.in_grace_loop = spec["grace"]
    return r


def observe(mgr, engines, reverse):
    """Call the real picker with the engines (as built from the descriptions) and read what it did."""
    built = [build(e) for e in engines]
    if reverse:
        built.reverse()
    mgr._session_affinity.clear()
    slot = Slot.new(TAG, thread_id="t-new", client_meta={"ip": "10.0.0.7", "session_id": "s-new"})
    chosen = mgr._pick_instance_for(slot, TAG, built)
    recorded = list(mgr._session_affinity.values())
    return (None if chosen is None else chosen.resident_key), (recorded[0] if recorded else None)


@pytest_asyncio.fixture
async def mgr(tmp_path):
    manager = H.new_manager(tmp_path)
    try:
        yield manager
    finally:
        await manager.shutdown()


UNCHANGED_NAMES = sorted(n for n, c in CASES.items() if c[0] == UNCHANGED)
CHANGED_NAMES = sorted(n for n, c in CASES.items() if c[0] == CHANGED)
ORDERS = [False, True]
ORDER_IDS = ["listed-order", "reversed-order"]


def test_every_case_has_an_expected_value():
    assert sorted(FROZEN) == UNCHANGED_NAMES, (
        f"captured values missing or extra: {sorted(set(FROZEN) ^ set(UNCHANGED_NAMES))}"
    )
    assert sorted(INTENDED) == CHANGED_NAMES, (
        f"intended values missing or extra: {sorted(set(INTENDED) ^ set(CHANGED_NAMES))}"
    )


@pytest.mark.parametrize("reverse", ORDERS, ids=ORDER_IDS)
@pytest.mark.parametrize("name", UNCHANGED_NAMES)
async def test_choice_unchanged_from_v080(mgr, name, reverse):
    kind, why, engines = CASES[name]
    got = observe(mgr, engines, reverse)
    expected = tuple(FROZEN[name][1 if reverse else 0])
    assert got == expected, (
        f"{name} ({kind}; {why}; {'reversed' if reverse else 'listed'} order): "
        f"chose {got[0]!r} with affinity {got[1]!r}, captured from v0.8.0: {expected!r}"
    )


@pytest.mark.parametrize("reverse", ORDERS, ids=ORDER_IDS)
@pytest.mark.parametrize("name", CHANGED_NAMES)
async def test_choice_changed_since_v080(mgr, name, reverse):
    kind, why, engines = CASES[name]
    got = observe(mgr, engines, reverse)
    expected = (INTENDED[name], INTENDED[name])
    assert got == expected, (
        f"{name} ({kind}; {why}; {'reversed' if reverse else 'listed'} order): "
        f"chose {got[0]!r} with affinity {got[1]!r}, intended: {expected!r}"
    )
