"""co_residence constants: the tuning numbers are part of the contract, so their values are pinned."""

import turbohaul.co_residence as co_residence
from turbohaul.co_residence import (
    DEMAND_MIN_AGE_S,
    MEMO_MAX_ENTRIES,
    NO_ROOM_MEMO_S,
    PROTECTED_DEMAND_MIN_AGE_S,
    RECHECK_MIN_INTERVAL_S,
)


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"co_residence: {text} (the contract is in src/turbohaul/co_residence/AGENTS.md, section Invariants)"


def test_recheck_interval_is_ten_seconds():
    assert RECHECK_MIN_INTERVAL_S == 10.0, why("one re-check attempt per 10.0 s, box-wide")
    assert type(RECHECK_MIN_INTERVAL_S) is float


def test_no_room_memo_is_sixty_seconds():
    assert NO_ROOM_MEMO_S == 60.0, why("a no-room verdict holds for 60.0 s")
    assert type(NO_ROOM_MEMO_S) is float


def test_unprotected_demand_age_is_five_seconds():
    assert DEMAND_MIN_AGE_S == 5.0, why("unprotected demand counts after 5.0 s")
    assert type(DEMAND_MIN_AGE_S) is float


def test_protected_demand_age_is_one_hundred_twenty_seconds():
    assert PROTECTED_DEMAND_MIN_AGE_S == 120.0, why("protected demand counts after 120.0 s")
    assert type(PROTECTED_DEMAND_MIN_AGE_S) is float


def test_memo_bound_is_sixty_four_entries():
    assert MEMO_MAX_ENTRIES == 64, why("the memo remembers at most 64 keys")
    assert type(MEMO_MAX_ENTRIES) is int


def test_protected_waits_longer_than_unprotected_and_memo_outlasts_interval():
    # The ordering is the design: a protected request must never count sooner than an
    # unprotected one, and the memo must cover several re-check intervals.
    assert PROTECTED_DEMAND_MIN_AGE_S > DEMAND_MIN_AGE_S, why("protected age exceeds unprotected age")
    assert NO_ROOM_MEMO_S > RECHECK_MIN_INTERVAL_S, why("the memo outlasts one re-check interval")


def test_package_entry_point_reexports_the_same_values():
    assert co_residence.RECHECK_MIN_INTERVAL_S == 10.0
    assert co_residence.NO_ROOM_MEMO_S == 60.0
    assert co_residence.DEMAND_MIN_AGE_S == 5.0
    assert co_residence.PROTECTED_DEMAND_MIN_AGE_S == 120.0
    assert co_residence.MEMO_MAX_ENTRIES == 64
