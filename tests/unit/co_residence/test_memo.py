"""co_residence.memo: a bounded no-room memo whose entries expire. The class never reads a clock."""

import pytest

from turbohaul.co_residence import MEMO_MAX_ENTRIES, NO_ROOM_MEMO_S, NoRoomMemo


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"co_residence: {text} (the contract is in src/turbohaul/co_residence/AGENTS.md, section Invariants)"


# -- defaults ----------------------------------------------------------------------

def test_default_ttl_is_the_no_room_constant():
    m = NoRoomMemo()
    m.note("t", 100.0)
    assert m.blocked("t", 100.0 + NO_ROOM_MEMO_S - 0.001) is True, why("blocked just before the default ttl")
    assert m.blocked("t", 100.0 + NO_ROOM_MEMO_S) is False, why("free at the default ttl")


def test_default_bound_is_the_memo_max_entries_constant():
    m = NoRoomMemo()
    for i in range(MEMO_MAX_ENTRIES + 5):
        m.note(f"k{i:03d}", float(i))
    assert len(m) == MEMO_MAX_ENTRIES, why(f"default bound is {MEMO_MAX_ENTRIES}, held {len(m)}")
    assert m.keys()[0] == "k005" and m.keys()[-1] == f"k{MEMO_MAX_ENTRIES + 4:03d}", why("the 5 earliest were dropped")


# -- note / blocked / expiry --------------------------------------------------------

def test_unknown_key_is_not_blocked():
    assert NoRoomMemo().blocked("never-seen", 0.0) is False, why("unknown key")


def test_noted_key_is_blocked_until_ttl_exclusive():
    m = NoRoomMemo(ttl_s=60.0)
    m.note("t", 100.0)
    assert m.blocked("t", 100.0) is True, why("blocked at the instant it is noted")
    assert m.blocked("t", 159.999) is True, why("blocked one millisecond before the ttl")
    assert m.blocked("t", 160.0) is False, why("not blocked at the ttl exactly (strict <)")


def test_key_past_expiry_is_not_blocked_and_is_removed():
    m = NoRoomMemo(ttl_s=60.0)
    m.note("t", 100.0)
    assert len(m) == 1
    assert m.blocked("t", 161.0) is False, why("after the ttl")
    assert len(m) == 0 and m.keys() == [], why("an expired key is removed when it is looked up")


def test_expired_lookup_removes_only_that_key():
    m = NoRoomMemo(ttl_s=60.0)
    m.note("old", 0.0)
    m.note("new", 100.0)
    assert m.blocked("old", 100.0) is False
    assert m.keys() == ["new"], why("the live key survives an expired key's removal")
    assert m.blocked("new", 100.0) is True


def test_blocked_on_a_live_key_does_not_extend_it():
    m = NoRoomMemo(ttl_s=60.0)
    m.note("t", 0.0)
    assert m.blocked("t", 59.0) is True
    assert m.blocked("t", 60.0) is False, why("looking a key up must not refresh its expiry")


def test_custom_ttl_is_used():
    m = NoRoomMemo(ttl_s=5.0)
    m.note("t", 10.0)
    assert m.blocked("t", 14.9) is True and m.blocked("t", 15.0) is False


def test_renoting_refreshes_the_expiry_from_the_new_now():
    m = NoRoomMemo(ttl_s=60.0)
    m.note("t", 0.0)
    m.note("t", 50.0)
    assert m.blocked("t", 100.0) is True, why("second note moves the expiry to 110")
    assert len(m) == 1, why("a re-note keeps one entry for the key")
    assert m.blocked("t", 110.0) is False
    assert len(m) == 0


def test_note_with_an_earlier_time_may_shorten_the_expiry():
    # The expiry is always now + ttl of the LAST note: no max() with the old value.
    m = NoRoomMemo(ttl_s=60.0)
    m.note("t", 100.0)
    m.note("t", 10.0)
    assert m.blocked("t", 70.0) is False, why("the last note wins, even if it is earlier")


def test_keys_are_independent():
    m = NoRoomMemo(ttl_s=60.0)
    m.note("a", 0.0)
    assert m.blocked("b", 1.0) is False, why("noting a does not block b")
    m.note("b", 30.0)
    assert m.blocked("a", 61.0) is False and m.blocked("b", 61.0) is True


def test_instances_do_not_share_state():
    one, two = NoRoomMemo(), NoRoomMemo()
    one.note("t", 0.0)
    assert two.blocked("t", 1.0) is False and len(two) == 0, why("each memo has its own table")


# -- clear ----------------------------------------------------------------------------

def test_clear_forgets_a_key():
    m = NoRoomMemo()
    m.note("t", 0.0)
    m.clear("t")
    assert m.blocked("t", 1.0) is False and len(m) == 0, why("cleared key")


def test_clear_of_an_absent_key_is_silent_and_leaves_others():
    m = NoRoomMemo()
    m.note("keep", 0.0)
    m.clear("absent")
    assert m.keys() == ["keep"], why("clearing an absent key changes nothing")


def test_note_after_clear_blocks_again():
    m = NoRoomMemo(ttl_s=10.0)
    m.note("t", 0.0)
    m.clear("t")
    m.note("t", 100.0)
    assert m.blocked("t", 105.0) is True


# -- len and keys ---------------------------------------------------------------------

def test_len_counts_stored_keys_and_keys_are_sorted():
    m = NoRoomMemo()
    for k in ("b", "c", "a"):
        m.note(k, 0.0)
    assert len(m) == 3 and m.keys() == ["a", "b", "c"], why("len and sorted keys")
    assert isinstance(m.keys(), list)


# -- the bound ---------------------------------------------------------------------------

def test_exactly_max_entries_drops_nothing():
    m = NoRoomMemo(ttl_s=10.0, max_entries=3)
    for i, k in enumerate("abc"):
        m.note(k, float(i))
    assert m.keys() == ["a", "b", "c"], why("at the bound, nothing is dropped")


def test_one_over_the_bound_drops_exactly_the_earliest_expiring_key():
    m = NoRoomMemo(ttl_s=10.0, max_entries=3)
    for i, k in enumerate("abcd"):
        m.note(k, float(i))
    assert m.keys() == ["b", "c", "d"], why("the earliest expiry, a, goes")


def test_five_notes_with_a_bound_of_three_keep_the_three_latest_expiring():
    m = NoRoomMemo(ttl_s=10.0, max_entries=3)
    for i, k in enumerate("abcde"):
        m.note(k, float(i))
    assert m.keys() == ["c", "d", "e"], why(f"kept {m.keys()}")
    assert len(m) == 3


def test_eviction_goes_by_expiry_not_by_insertion_order():
    m = NoRoomMemo(ttl_s=10.0, max_entries=3)
    m.note("late", 50.0)
    m.note("mid", 20.0)
    m.note("early", 10.0)
    m.note("newest", 60.0)
    assert m.keys() == ["late", "mid", "newest"], why("'early' expires first, whatever its insertion position")


def test_the_key_just_noted_is_never_dropped_even_when_it_expires_earliest():
    m = NoRoomMemo(ttl_s=10.0, max_entries=3)
    for i, k in enumerate("xyz"):
        m.note(k, 100.0 + i)
    m.note("old", 0.0)
    assert "old" in m.keys(), why("the key just noted survives its own eviction pass")
    assert m.keys() == ["old", "y", "z"], why("the earliest-expiring OTHER key, x, went instead")
    assert m.blocked("old", 5.0) is True


def test_renoting_an_existing_key_at_the_bound_evicts_nobody():
    m = NoRoomMemo(ttl_s=10.0, max_entries=3)
    for i, k in enumerate("abc"):
        m.note(k, float(i))
    m.note("a", 50.0)
    assert m.keys() == ["a", "b", "c"], why("a re-note does not grow the table")


def test_the_bound_holds_over_a_long_stream_of_distinct_keys():
    m = NoRoomMemo(ttl_s=10.0, max_entries=4)
    for i in range(200):
        m.note(f"k{i}", float(i))
        assert len(m) <= 4, why(f"the table grew to {len(m)} after {i + 1} notes")
    assert len(m) == 4 and m.keys() == ["k196", "k197", "k198", "k199"]


def test_bound_of_one_keeps_only_the_latest_noted_key():
    m = NoRoomMemo(ttl_s=10.0, max_entries=1)
    m.note("a", 100.0)
    m.note("b", 0.0)
    assert m.keys() == ["b"], why("with a bound of 1 the key just noted is the one kept")


def test_equal_expiries_are_dropped_by_key_not_by_insertion_order():
    # Two keys noted at the same instant expire together; which one goes must not depend on
    # which was noted first.
    for order in (("b", "a"), ("a", "b")):
        m = NoRoomMemo(ttl_s=10.0, max_entries=2)
        for k in order:
            m.note(k, 0.0)
        m.note("c", 5.0)
        assert m.keys() == ["b", "c"], why(f"tie dropped the wrong key for insertion order {order}")


def test_bound_of_zero_still_keeps_the_key_just_noted():
    m = NoRoomMemo(ttl_s=10.0, max_entries=0)
    m.note("a", 0.0)
    m.note("b", 1.0)
    assert m.keys() == ["b"], why("the fresh key is never dropped, even when the bound is below one")


# -- never raises ------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["", " ", "m:q4_K_M", "tag|reason", "caf\u00e9", "x" * 5000])
def test_odd_string_keys_are_accepted(key):
    m = NoRoomMemo(ttl_s=10.0, max_entries=2)
    m.note(key, 0.0)
    assert m.blocked(key, 1.0) is True, why(f"key {key[:20]!r}")
    m.clear(key)
    assert m.blocked(key, 1.0) is False


@pytest.mark.parametrize("now", [0.0, -5.0, 1e12, 3])
def test_odd_times_are_accepted(now):
    m = NoRoomMemo(ttl_s=10.0)
    m.note("t", now)
    assert m.blocked("t", now) is True
    assert m.blocked("t", now + 10.0) is False
