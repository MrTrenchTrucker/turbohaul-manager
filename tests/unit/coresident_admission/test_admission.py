"""Interface tests for coresident_admission: fits, reserved_loading_mib and admit.

The rule is one inequality, free - reserve + credit >= need. These tests use plain
numbers and SiblingLoad values, so no GPU, no file and no patch is needed.
"""

import sys
from pathlib import Path

import pytest

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.coresident_admission import (  # noqa: E402
    SiblingLoad,
    admit,
    fits,
    reserved_loading_mib,
)


# -- fits: the one inequality ------------------------------------------------

@pytest.mark.parametrize(
    "free, reserve, credit, need, expected",
    [
        (100, 0, 0, 100, True),
        (100, 0, 0, 101, False),
        (100, 30, 0, 70, True),
        (100, 30, 0, 71, False),
        (100, 0, 25, 125, True),
        (100, 0, 25, 126, False),
        (10, 50, 0, 0, False),   # negative effective free still refuses a zero need
        (0, 0, 0, 0, True),
    ],
)
def test_fits_is_free_minus_reserve_plus_credit_at_least_need(free, reserve, credit, need, expected):
    got = fits(free, reserve, credit, need)
    assert got is expected, (
        f"coresident_admission.fits: free - reserve + credit >= need broke for "
        f"free={free} reserve={reserve} credit={credit} need={need}; "
        f"expected {expected}, got {got}"
    )


# -- reserved_loading_mib: the reserve sum -----------------------------------

def test_reserved_loading_mib_sums_only_loading_siblings_on_that_card():
    siblings = [
        SiblingLoad(0, 10000, True),
        SiblingLoad(0, 4000, False),
        SiblingLoad(1, 7000, True),
    ]
    assert reserved_loading_mib(siblings, 0) == 10000, (
        "coresident_admission.reserved_loading_mib: card 0 must count its booting "
        "sibling only, not the loaded one and not the one on card 1"
    )
    assert reserved_loading_mib(siblings, 1) == 7000, (
        "coresident_admission.reserved_loading_mib: card 1 must count its own booting "
        "sibling only"
    )


def test_not_loading_sibling_on_the_same_card_adds_zero():
    got = reserved_loading_mib([SiblingLoad(0, 18000, False)], 0)
    assert got == 0, (
        f"coresident_admission.reserved_loading_mib: a sibling that is already loaded "
        f"is inside the live reading and must add 0, got {got}"
    )


def test_loading_sibling_on_another_card_adds_zero():
    got = reserved_loading_mib([SiblingLoad(1, 18000, True)], 0)
    assert got == 0, (
        f"coresident_admission.reserved_loading_mib: a booting sibling on another card "
        f"must add 0 to card 0, got {got}"
    )


def test_several_loading_siblings_on_the_card_add_up():
    siblings = [
        SiblingLoad(0, 15000, True),
        SiblingLoad(0, 15000, True),
        SiblingLoad(0, 4000, False),
    ]
    got = reserved_loading_mib(siblings, 0)
    assert got == 30000, (
        f"coresident_admission.reserved_loading_mib: two booting siblings on card 0 "
        f"must add up to 30000 and the loaded one must add 0, got {got}"
    )


def test_no_siblings_gives_zero_reserve():
    got = reserved_loading_mib([], 0)
    assert got == 0, (
        f"coresident_admission.reserved_loading_mib: no siblings must give a reserve "
        f"of 0, got {got}"
    )


# -- admit: empty reading refuses --------------------------------------------

@pytest.mark.parametrize("vals", [[], None], ids=["empty-list", "none"])
def test_admit_refuses_an_empty_or_missing_reading_even_when_credit_covers_need(vals):
    # Credit alone (50000) covers the need (1000). The caller checks credit before it
    # calls admit, so admit itself must still refuse: it never admits blind.
    got = admit(vals, 0, [], 50000, 1000)
    assert got is False, (
        f"coresident_admission.admit: an empty or missing reading must refuse, never "
        f"admit blind, got {got} for vals={vals!r}"
    )


# -- admit: card choice -------------------------------------------------------

@pytest.mark.parametrize(
    "vals, main_gpu, expected",
    [
        ([80000, 1000], 0, True),    # card 0 has room, card 1 does not
        ([1000, 80000], 1, True),    # card 1 has room, card 0 does not
        ([1000, 80000], 0, False),   # the same reading refuses on card 0
        ([80000, 1000], 1, False),   # and refuses on card 1 in the other order
    ],
)
def test_admit_decides_on_the_card_named_by_main_gpu(vals, main_gpu, expected):
    got = admit(vals, main_gpu, [], 0, 5000)
    assert got is expected, (
        f"coresident_admission.admit: with main_gpu={main_gpu} in range the decision "
        f"must read that card, vals={vals}; expected {expected}, got {got}"
    )


# -- admit: out-of-range main_gpu reads card 0 --------------------------------

@pytest.mark.parametrize(
    "vals, main_gpu, expected",
    [
        ([80000, 1000], -1, True),
        ([80000, 1000], 2, True),    # equal to the card count
        ([80000, 1000], 7, True),
        ([1000, 80000], -1, False),
        ([1000, 80000], 2, False),
        ([1000, 80000], 7, False),
    ],
)
def test_admit_with_main_gpu_out_of_range_reads_card_zero(vals, main_gpu, expected):
    got = admit(vals, main_gpu, [], 0, 5000)
    assert got is expected, (
        f"coresident_admission.admit: main_gpu={main_gpu} is out of range for "
        f"vals={vals}, so card 0 must be read; expected {expected}, got {got}"
    )


# -- admit: reserve filters on the target card only --------------------------

@pytest.mark.parametrize(
    "siblings, expected",
    [
        ([SiblingLoad(0, 15000, True)], False),   # booting on the target card: subtracted
        ([SiblingLoad(0, 15000, False)], True),   # already loaded: not subtracted again
        ([SiblingLoad(1, 15000, True)], True),    # booting on another card: not subtracted
        ([], True),
    ],
    ids=["booting-on-target", "loaded-on-target", "booting-on-other-card", "no-siblings"],
)
def test_admit_subtracts_only_a_booting_sibling_on_the_target_card(siblings, expected):
    # 30000 free on card 0, need 20000: admits only when no reserve is taken.
    got = admit([30000, 30000], 0, siblings, 0, 20000)
    assert got is expected, (
        f"coresident_admission.admit: the reserve must count only booting siblings on "
        f"the target card; siblings={siblings}; expected {expected}, got {got}"
    )


def test_admit_adds_the_credit_after_the_reserve_is_taken():
    # 30000 free - 10000 booting reserve + 5000 credit = 25000, which covers 25000.
    booting = [SiblingLoad(0, 10000, True)]
    got = admit([30000], 0, booting, 5000, 25000)
    assert got is True, (
        f"coresident_admission.admit: free - reserve + credit must be 25000 here, "
        f"which covers need 25000; got {got}"
    )


# -- admit: out-of-range main_gpu reserves by the number given ---------------

def test_admit_out_of_range_main_gpu_has_no_reserve_when_no_sibling_is_pinned_to_it():
    got = admit([30000, 1000], 5, [], 0, 20000)
    assert got is True, (
        f"coresident_admission.admit: with no siblings, 30000 free on card 0 must "
        f"admit need 20000 even when main_gpu=5 is out of range; got {got}"
    )


def test_admit_out_of_range_main_gpu_reserves_for_siblings_pinned_to_that_number():
    # main_gpu=5 is out of range, so card 0 is read (30000 free). The reserve is
    # still taken for main_gpu=5 itself, so the booting sibling pinned to 5 counts.
    booting_on_five = [SiblingLoad(5, 15000, True)]
    got = admit([30000, 1000], 5, booting_on_five, 0, 20000)
    assert got is False, (
        f"coresident_admission.admit: a booting sibling pinned to the out-of-range "
        f"main_gpu=5 must still be reserved; 30000 - 15000 < 20000 must refuse, got {got}"
    )


def test_admit_out_of_range_main_gpu_ignores_booting_sibling_on_the_card_it_read():
    # The sibling sits on card 0, the card that was read, but the reserve filters on
    # main_gpu=5. So it is not subtracted and the same reading admits.
    booting_on_zero = [SiblingLoad(0, 15000, True)]
    got = admit([30000, 1000], 5, booting_on_zero, 0, 20000)
    assert got is True, (
        f"coresident_admission.admit: the reserve filters on main_gpu itself, so a "
        f"booting sibling on card 0 must not reduce the read for main_gpu=5; got {got}"
    )


# -- admit: credit boundary ---------------------------------------------------

@pytest.mark.parametrize(
    "credit, need, expected",
    [
        (4000, 5000, True),    # 1000 free + 4000 credit == need admits
        (4000, 5001, False),   # one more refuses
        (0, 1000, True),
        (0, 1001, False),
    ],
)
def test_admit_adds_the_credit_with_an_exact_boundary(credit, need, expected):
    got = admit([1000], 0, [], credit, need)
    assert got is expected, (
        f"coresident_admission.admit: free + credit >= need must be exact at the "
        f"boundary; credit={credit} need={need} expected {expected}, got {got}"
    )
