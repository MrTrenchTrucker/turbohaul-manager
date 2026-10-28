"""co_residence.scan: inbox helpers, candidate gathering and the pick of one candidate."""

import asyncio

from _co_residence_fakes import FakeHost, FakeResident, FakeSlot, FakeState

from turbohaul.co_residence import (
    Candidate,
    RecheckState,
    gather_candidates,
    inbox_slots,
    pick_candidate,
    take_from_inbox,
)

NOW = 1000.0


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"co_residence: {text} (the contract is in src/turbohaul/co_residence/AGENTS.md, section Invariants)"


def slot(sid="s", tag="m", age=10.0, **kw) -> FakeSlot:
    return FakeSlot(slot_id=sid, model_tag=tag, created_at=NOW - age, **kw)


def ids(cands) -> list[str]:
    return [c.slot.slot_id for c in cands]


# -- inbox_slots / take_from_inbox -----------------------------------------------

def test_inbox_slots_lists_the_queue_oldest_first_as_a_tuple():
    a, b = slot("a"), slot("b")
    r = FakeResident().put(a, b)
    got = inbox_slots(r)
    assert isinstance(got, tuple) and got == (a, b), why(f"inbox_slots gave {got!r}")
    assert r.inbox.qsize() == 2, why("reading the inbox must not remove anything")


def test_inbox_slots_of_a_resident_without_an_inbox_is_empty():
    assert inbox_slots(FakeResident(inbox=None)) == (), why("no inbox means no slots")
    assert inbox_slots(object()) == (), why("an object with no inbox attribute means no slots")


def test_take_from_inbox_takes_one_slot_and_keeps_the_others_in_order():
    a, b, c = slot("a"), slot("b"), slot("c")
    r = FakeResident().put(a, b, c)
    assert take_from_inbox(r, b) is b, why("the named slot must be returned")
    assert inbox_slots(r) == (a, c), why("the others must stay, in their original order")


def test_take_from_inbox_matches_by_identity_not_equality():
    twin_a, twin_b = slot("same"), slot("same")
    assert twin_a == twin_b and twin_a is not twin_b
    r = FakeResident().put(twin_a)
    assert take_from_inbox(r, twin_b) is None, why("an equal but different slot must not be taken")
    assert inbox_slots(r) == (twin_a,), why("nothing may be removed on a miss")


def test_take_from_inbox_returns_none_for_an_absent_slot_and_changes_nothing():
    a, b = slot("a"), slot("b")
    r = FakeResident().put(a, b)
    assert take_from_inbox(r, slot("gone")) is None, why("absent slot gives None")
    assert inbox_slots(r) == (a, b), why("order must survive a miss")


def test_take_from_inbox_with_no_inbox_or_an_empty_one_is_none():
    assert take_from_inbox(FakeResident(inbox=None), slot()) is None
    assert take_from_inbox(FakeResident(), slot()) is None


def test_take_from_inbox_takes_only_the_first_of_the_same_object_listed_twice():
    a = slot("a")
    r = FakeResident().put(a, a)
    assert take_from_inbox(r, a) is a
    assert inbox_slots(r) == (a,), why("one copy is taken, the other stays")


def test_take_from_inbox_is_synchronous_and_leaves_no_slot_lost():
    # Draining and refilling must keep every other slot even for a larger queue.
    items = [slot(f"s{i}") for i in range(7)]
    r = FakeResident().put(*items)
    assert take_from_inbox(r, items[3]) is items[3]
    assert inbox_slots(r) == tuple(items[:3] + items[4:])
    assert not asyncio.iscoroutinefunction(take_from_inbox)


# -- gather_candidates ------------------------------------------------------------

def test_a_staged_slot_for_a_tag_with_a_live_engine_is_a_candidate():
    s = slot("st", age=30.0)
    host = FakeHost(residents=[FakeResident()], staged=[s])
    c = gather_candidates(host, NOW, frozenset())
    assert c == [Candidate(30.0, 0, "staged", s, None)], why(f"staged candidate wrong: {c!r}")


def test_a_staged_slot_for_a_tag_without_a_live_engine_is_skipped():
    host = FakeHost(residents=[FakeResident(model_tag="other")], staged=[slot("st", age=30.0)])
    assert gather_candidates(host, NOW, frozenset()) == [], why("a cold tag is not a co-residence question")


def test_an_evicted_staged_slot_is_skipped():
    host = FakeHost(residents=[FakeResident()], staged=[slot("st", age=30.0, is_evicted=True)])
    assert gather_candidates(host, NOW, frozenset()) == []


def test_staged_slot_age_boundary_is_five_seconds():
    host = FakeHost(residents=[FakeResident()], staged=[slot("young", age=4.99), slot("edge", age=5.0)])
    assert ids(gather_candidates(host, NOW, frozenset())) == ["edge"], why("5.0 s counts, 4.99 s does not")


def test_a_protected_staged_slot_waits_one_hundred_twenty_seconds():
    pair = ("t1", "m")
    host = FakeHost(
        residents=[FakeResident()],
        staged=[slot("p_young", age=119.0, thread_id="t1"), slot("p_old", age=120.0, thread_id="t1"),
                slot("other_thread", age=10.0, thread_id="t2")],
    )
    got = ids(gather_candidates(host, NOW, frozenset({pair})))
    assert got == ["p_old", "other_thread"], why(f"protected ages wrong: {got!r}")


def test_an_inbox_slot_of_an_active_resident_is_a_candidate_with_its_owner():
    s = slot("ib", age=20.0)
    r = FakeResident().put(s)
    c = gather_candidates(FakeHost(residents=[r]), NOW, frozenset())
    assert c == [Candidate(20.0, 1, "inbox", s, r)], why(f"inbox candidate wrong: {c!r}")


def test_inbox_slots_of_a_resident_that_is_not_active_are_skipped():
    for name in ("IDLE_EVICTABLE", "GRACE", "RESERVED_LOADING", "DEAD"):
        r = FakeResident(state=FakeState(name)).put(slot("ib", age=20.0))
        assert gather_candidates(FakeHost(residents=[r]), NOW, frozenset()) == [], why(f"state {name} must be skipped")


def test_inbox_slot_for_another_tag_or_evicted_is_skipped():
    r = FakeResident().put(slot("wrong_tag", tag="x", age=20.0), slot("gone", age=20.0, is_evicted=True))
    assert gather_candidates(FakeHost(residents=[r]), NOW, frozenset()) == []


def test_an_inbox_slot_of_the_engines_idle_thread_is_protected():
    r = FakeResident(idle_thread_id="th").put(slot("mine", age=60.0, thread_id="th"), slot("none", age=60.0))
    got = ids(gather_candidates(FakeHost(residents=[r]), NOW, frozenset()))
    assert got == ["none"], why("the idle thread's own follow-up waits 120 s")
    r2 = FakeResident(idle_thread_id="th").put(slot("mine", age=121.0, thread_id="th"))
    assert ids(gather_candidates(FakeHost(residents=[r2]), NOW, frozenset())) == ["mine"]


def test_an_empty_idle_thread_id_protects_nothing():
    r = FakeResident(idle_thread_id="").put(slot("a", age=10.0, thread_id=""))
    assert ids(gather_candidates(FakeHost(residents=[r]), NOW, frozenset())) == ["a"]


def test_an_inbox_slot_in_a_grace_pair_is_protected():
    r = FakeResident().put(slot("g", age=60.0, thread_id="t9"))
    pairs = frozenset({("t9", "m")})
    assert gather_candidates(FakeHost(residents=[r]), NOW, pairs) == []


def test_candidates_come_oldest_first_and_ties_put_staged_before_inbox():
    r = FakeResident().put(slot("ib_tie", age=40.0), slot("ib_old", age=90.0))
    host = FakeHost(residents=[r], staged=[slot("st_tie", age=40.0), slot("st_new", age=10.0)])
    got = ids(gather_candidates(host, NOW, frozenset()))
    assert got == ["ib_old", "st_tie", "ib_tie", "st_new"], why(f"ordering wrong: {got!r}")


def test_a_resident_without_an_inbox_adds_no_candidates():
    host = FakeHost(residents=[FakeResident(inbox=None)])
    assert gather_candidates(host, NOW, frozenset()) == []


# -- pick_candidate -----------------------------------------------------------------

def cand(tag="m", age=30.0, sid="c", source="staged") -> Candidate:
    return Candidate(age, 0 if source == "staged" else 1, source, slot(sid, tag=tag, age=age), None)


def test_pick_returns_none_for_no_candidates():
    assert pick_candidate(FakeHost(), RecheckState(), [], NOW) is None


def test_pick_returns_the_oldest_candidate_of_the_first_tag_that_may_run():
    host = FakeHost(residents=[FakeResident(model_tag="a"), FakeResident(model_tag="b")])
    older, newer = cand("a", 90.0, "a_old"), cand("a", 30.0, "a_new")
    other = cand("b", 50.0, "b1")
    got = pick_candidate(host, RecheckState(), [older, other, newer], NOW)
    assert got is older, why(f"expected the oldest candidate, got {got!r}")


def test_pick_skips_a_tag_whose_no_room_memo_is_live():
    host = FakeHost(residents=[FakeResident(model_tag="a"), FakeResident(model_tag="b")])
    st = RecheckState()
    st.norom.note("a", NOW)
    a1, b1 = cand("a", 90.0, "a1"), cand("b", 50.0, "b1")
    assert pick_candidate(host, st, [a1, b1], NOW) is b1, why("the memo must hold tag a back")
    assert pick_candidate(host, st, [a1, b1], NOW + 59.0) is b1
    assert pick_candidate(host, st, [a1, b1], NOW + 60.0) is a1, why("the memo expires after 60 s")


def test_pick_returns_none_when_every_tag_is_memoed():
    host = FakeHost(residents=[FakeResident(model_tag="a")])
    st = RecheckState()
    st.norom.note("a", NOW)
    assert pick_candidate(host, st, [cand("a")], NOW) is None


def test_pick_ignores_a_tag_with_no_live_engine():
    host = FakeHost(residents=[FakeResident(model_tag="b")])
    assert pick_candidate(host, RecheckState(), [cand("a")], NOW) is None, why("no live engine, no re-check")


def test_pick_asks_the_host_for_the_live_engines_of_each_tag_it_considers():
    host = FakeHost(residents=[FakeResident(model_tag="a")])
    pick_candidate(host, RecheckState(), [cand("a")], NOW)
    assert ("a",) in host.args.get("_instances_for", []), why("pick must count the live engines of the tag")
