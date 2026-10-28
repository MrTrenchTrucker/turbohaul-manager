"""resident_identity.engine_load: the engine load number used to pick an engine for a new request."""

import enum
import queue

from turbohaul.resident_identity import pending_load


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"resident_identity: {text} (the contract is in src/turbohaul/resident_identity/AGENTS.md, Invariants)"


class _S(str, enum.Enum):  # noqa: UP042 -- the same base classes as the real slot state
    """A str enum like the real slot state: it equals its plain string value."""

    GRACE = "GRACE"
    ACTIVE_MATCH = "ACTIVE_MATCH"
    ACTIVE = "ACTIVE"


class Slot:
    """A request slot. ``state`` is left off when None is not given, like a slot of another shape."""

    def __init__(self, *state) -> None:
        if state:
            self.state = state[0]


class Bare:
    """A resident with only what the function must always be able to read; the rest is set per test."""

    def __init__(self, inflight=(), active_slot=None) -> None:
        self.inflight = list(inflight)
        self.active_slot = active_slot


class DepthOnly:
    """An inbox that only knows its depth, like the one qsize call the function makes."""

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.calls = 0

    def qsize(self) -> int:
        self.calls += 1
        return self.depth


def res(inflight=0, slot=None, grace=None, inbox=None):
    """Build a resident. ``grace`` None leaves the attribute off; ``inbox`` None leaves the attribute off."""
    r = Bare(inflight=["rider"] * inflight, active_slot=slot)
    if grace is not None:
        r.in_grace_loop = grace
    if inbox is not None:
        r.inbox = inbox
    return r


def inbox_of(n: int) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    for i in range(n):
        q.put(i)
    return q


# -- the three terms on their own ------------------------------------------------------------------

def test_an_empty_resident_has_no_load():
    assert pending_load(res()) == 0, why("nothing in flight, no slot, no inbox is load 0")


def test_each_rider_in_flight_counts_one():
    assert pending_load(res(inflight=1)) == 1, why("one rider")
    assert pending_load(res(inflight=3)) == 3, why("three riders")


def test_a_live_turn_outside_the_grace_loop_counts_one():
    assert pending_load(res(slot=Slot("ACTIVE"), grace=False)) == 1, why("a running turn is load")


def test_a_live_turn_counts_one_when_the_grace_flag_is_not_set_at_all():
    assert pending_load(res(slot=Slot("ACTIVE"))) == 1, why("a resident without the grace attribute is not in grace")


def test_a_finished_slot_in_the_grace_loop_is_not_load():
    assert pending_load(res(slot=Slot("GRACE"), grace=True)) == 0, why("the finished slot of a grace engine is no load")


def test_the_grace_loop_with_no_slot_is_not_load():
    assert pending_load(res(slot=None, grace=True)) == 0, why("grace loop and no slot is load 0")


def test_a_follow_up_turn_inside_the_grace_loop_counts_one():
    assert pending_load(res(slot=Slot("ACTIVE_MATCH"), grace=True)) == 1, why("a follow-up turn is a running turn")


def test_a_finished_state_outside_the_grace_loop_counts_one():
    assert pending_load(res(slot=Slot("GRACE"), grace=False)) == 1, why("both conditions are needed to drop the slot")
    assert pending_load(res(slot=Slot("GRACE"))) == 1, why("no grace attribute: the slot counts")


def test_a_slot_with_no_state_attribute_counts_one_even_in_the_grace_loop():
    slot = Slot()
    assert not hasattr(slot, "state"), "the fake slot must have no state attribute for this test to mean anything"
    assert pending_load(res(slot=slot, grace=True)) == 1, why("a slot of unknown state is not treated as finished")


def test_a_slot_whose_state_is_none_counts_one_in_the_grace_loop():
    assert pending_load(res(slot=Slot(None), grace=True)) == 1, why("a None state is not the finished state")


def test_only_the_finished_state_drops_the_slot():
    for state in ("GRACE_BUSY", "ACTIVE", "ACTIVE_MATCH", "STAGED", "POPPED"):
        assert pending_load(res(slot=Slot(state), grace=True)) == 1, why(f"state {state} in the grace loop counts")


def test_a_str_enum_state_behaves_like_the_plain_string():
    assert pending_load(res(slot=Slot(_S.GRACE), grace=True)) == 0, why("an enum finished state is dropped")
    assert pending_load(res(slot=Slot(_S.ACTIVE_MATCH), grace=True)) == 1, why("an enum follow-up state counts")
    assert pending_load(res(slot=Slot(_S.ACTIVE), grace=False)) == 1, why("an enum live state counts")
    assert pending_load(res(slot=Slot(_S.GRACE), grace=False)) == 1, why("an enum finished state outside grace counts")


def test_inbox_depth_counts():
    assert pending_load(res(inbox=inbox_of(1))) == 1, why("one queued request")
    assert pending_load(res(inbox=inbox_of(4))) == 4, why("four queued requests")


def test_an_inbox_that_only_has_a_depth_counts_that_depth():
    box = DepthOnly(2)
    assert pending_load(res(inbox=box)) == 2, why("the depth comes from qsize")
    assert box.calls == 1, why("the depth is read once")


def test_an_empty_inbox_counts_zero():
    assert pending_load(res(inbox=inbox_of(0))) == 0, why("an empty queue is no load")


def test_an_inbox_set_to_none_counts_zero():
    r = res()
    r.inbox = None
    assert pending_load(r) == 0, why("a resident whose inbox is None has no queued requests")


def test_a_resident_without_an_inbox_attribute_counts_zero():
    r = res(inflight=1)
    assert not hasattr(r, "inbox"), "the fake must have no inbox attribute for this test to mean anything"
    assert pending_load(r) == 1, why("a missing inbox attribute reads as no queue")


# -- the terms together ------------------------------------------------------------------------------

def test_a_grace_engine_with_a_finished_slot_still_counts_riders_and_inbox_depth():
    assert pending_load(res(inflight=2, slot=Slot("GRACE"), grace=True, inbox=inbox_of(3))) == 5, \
        why("inflight and inbox always count, only the finished slot is dropped")


def test_a_grace_engine_serving_a_follow_up_counts_all_three_terms():
    assert pending_load(res(inflight=2, slot=Slot("ACTIVE_MATCH"), grace=True, inbox=inbox_of(3))) == 6, \
        why("inflight + follow-up turn + inbox depth")


def test_the_three_terms_add_for_a_live_turn():
    assert pending_load(res(inflight=2, slot=Slot("ACTIVE"), grace=False, inbox=inbox_of(3))) == 6, \
        why("inflight + running turn + inbox depth")


def test_a_busy_engine_ranks_above_a_finished_grace_engine():
    busy = res(slot=Slot("ACTIVE"), grace=False)
    finished = res(slot=Slot("GRACE"), grace=True)
    assert pending_load(finished) < pending_load(busy), why("the picker must prefer the engine that is not running")


def test_an_engine_in_a_follow_up_turn_ranks_above_an_idle_engine():
    follow_up = res(slot=Slot("ACTIVE_MATCH"), grace=True)
    idle = res(grace=False)
    assert pending_load(idle) < pending_load(follow_up), why("a follow-up turn must not look idle: no piling on")


def test_two_idle_engines_have_equal_load():
    assert pending_load(res(grace=False)) == pending_load(res()) == 0, why("idle engines tie, so new work spreads")


# -- shape of the answer ---------------------------------------------------------------------------

def test_the_answer_is_a_plain_int():
    for r in (res(), res(slot=Slot("ACTIVE")), res(inflight=2, inbox=inbox_of(1))):
        assert type(pending_load(r)) is int, why("the load is an int, usable as a sort key")


def test_the_resident_is_not_changed():
    slot = Slot("GRACE")
    r = res(inflight=2, slot=slot, grace=True, inbox=inbox_of(2))
    before = (list(r.inflight), r.active_slot, r.in_grace_loop, r.inbox.qsize(), sorted(vars(r)), slot.state)
    pending_load(r)
    after = (list(r.inflight), r.active_slot, r.in_grace_loop, r.inbox.qsize(), sorted(vars(r)), slot.state)
    assert before == after, why("pending_load is read-only")
    assert list(r.inbox.queue) == [0, 1], why("the queued items are left in place")


def test_the_grace_flag_is_read_as_a_truth_value():
    r = res(slot=Slot("GRACE"))
    r.in_grace_loop = 1
    assert pending_load(r) == 0, why("a truthy grace flag with a finished slot: not counted")
    r.in_grace_loop = 0
    assert pending_load(r) == 1, why("a falsy grace flag: the slot counts")
