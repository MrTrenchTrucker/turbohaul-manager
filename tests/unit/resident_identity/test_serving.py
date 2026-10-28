"""resident_identity.serving: find the resident that serves a request handle, and its spawn counter."""

import _resident_identity_fakes as fakes

from turbohaul.resident_identity import serving_resident, spawn_seq_for_handle


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"resident_identity: {text} (the contract is in src/turbohaul/resident_identity/AGENTS.md, Invariants)"


def res(key, handle, seq=0, tag="m"):
    return fakes.FakeResident(resident_key=key, model_tag=tag, handle=handle, spawn_seq=seq)


# -- serving_resident ------------------------------------------------------------------------------

def test_the_resident_that_owns_the_handle_is_returned():
    h1, h2 = object(), object()
    a, b = res("m#1", h1), res("m#2", h2)
    host = fakes.FakeHost(residents=[a, b])
    assert serving_resident(host, h1) is a, why("first handle")
    assert serving_resident(host, h2) is b, why("second handle: two residents of one model keep their own handles")


def test_a_none_handle_gives_none_even_when_a_resident_has_no_handle():
    idle = res("m#1", None)
    host = fakes.FakeHost(residents=[idle])
    assert serving_resident(host, None) is None, why("a None handle never matches a resident with no handle")
    assert host.called("_model_residents") == [], why("a None handle does not even ask for the residents")


def test_a_stray_handle_gives_none():
    host = fakes.FakeHost(residents=[res("m#1", object()), res("m#2", None)])
    assert serving_resident(host, object()) is None, why("a handle nobody owns")


def test_no_residents_gives_none():
    assert serving_resident(fakes.FakeHost(residents=[]), object()) is None, why("empty registry")


def test_the_match_is_by_identity_not_equality():
    mine, theirs = fakes.AlwaysEqual(), fakes.AlwaysEqual()
    assert mine == theirs, "the fake handles must compare equal for this test to mean anything"
    a, b = res("m#1", theirs), res("m#2", mine)
    host = fakes.FakeHost(residents=[a, b])
    assert serving_resident(host, mine) is b, why("an equal but different handle must not match")
    assert serving_resident(host, fakes.AlwaysEqual()) is None, why("an equal stranger matches nobody")


def test_equal_lists_are_not_the_same_handle():
    a = res("m#1", [1])
    assert serving_resident(fakes.FakeHost(residents=[a]), [1]) is None, why("equal value, different object")


def test_the_first_match_wins_when_two_residents_share_a_handle():
    shared = object()
    a, b = res("m#1", shared), res("m#2", shared)
    assert serving_resident(fakes.FakeHost(residents=[a, b]), shared) is a, why("first in order")
    assert serving_resident(fakes.FakeHost(residents=[b, a]), shared) is b, why("order of the snapshot decides")


def test_a_resident_missing_from_the_snapshot_is_never_found():
    gone, live = object(), object()
    host = fakes.FakeHost(residents=[res("m#2", live)])
    assert serving_resident(host, gone) is None, why("a dead resident is simply absent from the snapshot")
    assert serving_resident(host, live).resident_key == "m#2", why("the live one is still found")


def test_the_snapshot_is_asked_for_once_and_nothing_is_changed():
    h = object()
    residents = [res("m#1", object(), seq=3), res("m#2", h, seq=4)]
    host = fakes.FakeHost(residents=residents)
    before = [(r.resident_key, r.handle, r.spawn_seq) for r in residents]
    serving_resident(host, h)
    assert host.called("_model_residents") == [("_model_residents",)], why("one snapshot per lookup")
    assert [c[0] for c in host.calls] == ["_model_residents"], why("nothing else is asked of the host")
    assert [(r.resident_key, r.handle, r.spawn_seq) for r in residents] == before, why("read-only")
    assert host._last_request_identity_by_model == {}, why("the identity store is not touched")


# -- spawn_seq_for_handle --------------------------------------------------------------------------

def test_a_serving_resident_gives_its_own_spawn_seq_and_the_model_counter_is_not_asked():
    h = object()
    host = fakes.FakeHost(residents=[res("m#1", h, seq=5)], model_spawn_seq=99)
    assert spawn_seq_for_handle(host, h, "m") == 5, why("the serving resident's counter")
    assert host.called("_spawn_seq_for_model") == [], why("no fallback when a resident owns the handle")


def test_residents_of_one_model_give_each_their_own_counter():
    h1, h2 = object(), object()
    host = fakes.FakeHost(residents=[res("m#1", h1, seq=3), res("m#2", h2, seq=8)], model_spawn_seq=99)
    assert spawn_seq_for_handle(host, h1, "m") == 3, why("first resident")
    assert spawn_seq_for_handle(host, h2, "m") == 8, why("second resident")


def test_a_serving_resident_with_a_zero_counter_is_not_replaced_by_the_fallback():
    h = object()
    host = fakes.FakeHost(residents=[res("m#1", h, seq=0)], model_spawn_seq=7)
    assert spawn_seq_for_handle(host, h, "m") == 0, why("zero is a real counter, not a missing one")


def test_no_owner_falls_back_to_the_model_counter_with_the_model_tag():
    host = fakes.FakeHost(residents=[res("m#1", object(), seq=3)], model_spawn_seq=42)
    assert spawn_seq_for_handle(host, object(), "the-model") == 42, why("fallback value")
    assert host.called("_spawn_seq_for_model") == [("_spawn_seq_for_model", "the-model")], \
        why("the fallback is asked once, with the model tag given")


def test_a_none_handle_falls_back_to_the_model_counter():
    host = fakes.FakeHost(residents=[res("m#1", None, seq=3)], model_spawn_seq=42)
    assert spawn_seq_for_handle(host, None, "m") == 42, why("a None handle never matches the idle resident")


def test_a_none_model_tag_is_passed_on_unchanged():
    host = fakes.FakeHost(residents=[], model_spawn_seq=1)
    assert spawn_seq_for_handle(host, object(), None) == 1, why("fallback value")
    assert host.called("_spawn_seq_for_model") == [("_spawn_seq_for_model", None)], why("None passed through")


def test_a_dead_resident_absent_from_the_snapshot_falls_back():
    gone = object()
    host = fakes.FakeHost(residents=[], model_spawn_seq=11)
    assert spawn_seq_for_handle(host, gone, "m") == 11, why("the lookup follows the snapshot")
