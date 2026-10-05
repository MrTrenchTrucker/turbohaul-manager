"""Dirty-tip zero-populated-proceed (two-signal derivation) — pure
decision-table tier for `_classify_dirty_tip_probe`.

Two tiers, same discipline as the test_cold_restore_size_guard.py split
(itself a pure-table file paired with a wiring file for the same two-tier
coverage, test_dirty_tip_zero_populated_wiring.py, used as the model):
  * this file — the pure decision table in isolation. Proves the table is
    right; proves nothing about whether the call site derives its inputs
    correctly (that's test_dirty_tip_zero_populated_wiring.py).
  * test_dirty_tip_zero_populated_wiring.py — drives the real production
    method (_probe_and_save_clean_kv) end to end, including the
    two-signal "populated" derivation that feeds this table.

`_classify_dirty_tip_probe` itself is a pure function with a fixed SHAPE
(same four/five-way case split, same case-label names) — the call site alone
decides what feeds it (see the wiring file). These tests exist to pin the pure
function's contract independent of how its inputs are derived.
"""
import pytest

from turbohaul.manager import _classify_dirty_tip_probe


def test_probe_failed_when_probe_not_ok():
    proceed, case = _classify_dirty_tip_probe(False, 0, False)
    assert (proceed, case) == (False, "probe_failed")


def test_probe_failed_when_populated_count_none_even_if_probe_ok():
    """Defensive: should not happen when probe_ok is True in production (the
    call site always derives a count once probe_ok is set), but the pure
    function must not silently treat an absent count as zero."""
    proceed, case = _classify_dirty_tip_probe(True, None, False)
    assert (proceed, case) == (False, "probe_failed")


def test_zero_populated_proceeds_without_erase():
    proceed, case = _classify_dirty_tip_probe(True, 0, False)
    assert (proceed, case) == (True, "zero_populated")


def test_single_populated_erased_proceeds():
    proceed, case = _classify_dirty_tip_probe(True, 1, True)
    assert (proceed, case) == (True, "single_erased")


def test_single_populated_erase_failed_refuses():
    """populated_count==1 but the erase POST itself failed/raised — refuse,
    do not fall back to treating it as zero_populated."""
    proceed, case = _classify_dirty_tip_probe(True, 1, False)
    assert (proceed, case) == (False, "single_erase_failed")


@pytest.mark.parametrize("count", [2, 3, 8])
def test_multi_populated_refuses_regardless_of_erased(count):
    """Ambiguous (can't identify which slot is dirty) -- refuse regardless of
    what `erased` happens to hold (it's meaningless once count != 1)."""
    for erased in (True, False):
        proceed, case = _classify_dirty_tip_probe(True, count, erased)
        assert (proceed, case) == (False, "multi_populated")
