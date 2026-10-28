"""Interface tests for coresident_admission.retry_read_free_vram.

The free-VRAM probe can flake and return None or an empty list. The retry reads
again, a bounded number of times, and stops at the first usable reading. These
tests script the reader and count its calls, so no GPU is needed.
"""

import sys
from pathlib import Path

import pytest

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.coresident_admission import retry_read_free_vram  # noqa: E402


class _ScriptedReader:
    """Return the scripted answers in order and count the calls.

    Once the script runs out the reader answers None, so a reader that is called
    too often fails on the call count with a plain assertion.
    """

    def __init__(self, *answers):
        self._answers = list(answers)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self._answers:
            return self._answers.pop(0)
        return None


def test_healthy_first_read_is_returned_after_exactly_one_call():
    reader = _ScriptedReader([80000, 1000])
    got = retry_read_free_vram(reader)
    assert got == [80000, 1000], (
        f"coresident_admission.retry_read_free_vram: a healthy first read must be "
        f"returned as it is, got {got}"
    )
    assert reader.calls == 1, (
        f"coresident_admission.retry_read_free_vram: a healthy first read must cost "
        f"exactly 1 call, got {reader.calls}"
    )


@pytest.mark.parametrize("flake", [None, []], ids=["none-flake", "empty-list-flake"])
def test_one_flake_then_a_reading_returns_that_reading_after_two_calls(flake):
    reader = _ScriptedReader(flake, [5000, 6000])
    got = retry_read_free_vram(reader)
    assert got == [5000, 6000], (
        f"coresident_admission.retry_read_free_vram: after one flake the second read "
        f"must be returned, got {got}"
    )
    assert reader.calls == 2, (
        f"coresident_admission.retry_read_free_vram: one flake then a reading must "
        f"take exactly 2 calls, got {reader.calls}"
    )


@pytest.mark.parametrize(
    "first, second",
    [(None, None), ([], []), (None, []), ([], None)],
    ids=["none-none", "empty-empty", "none-empty", "empty-none"],
)
def test_two_flakes_return_empty_list_after_exactly_two_calls(first, second):
    # A usable third answer is scripted on purpose: a third read would return it,
    # so the call count and the empty result together pin the bound.
    reader = _ScriptedReader(first, second, [9000])
    got = retry_read_free_vram(reader)
    assert got == [], (
        f"coresident_admission.retry_read_free_vram: when every read flakes the "
        f"result must be an empty list, never a guess, got {got}"
    )
    assert reader.calls == 2, (
        f"coresident_admission.retry_read_free_vram: the default is 2 attempts, so "
        f"two flakes must cost exactly 2 calls, got {reader.calls}"
    )


def test_attempts_three_recovers_on_the_third_read():
    reader = _ScriptedReader(None, None, [4000, 4001])
    got = retry_read_free_vram(reader, attempts=3)
    assert got == [4000, 4001], (
        f"coresident_admission.retry_read_free_vram: with attempts=3 the third read "
        f"must be returned, got {got}"
    )
    assert reader.calls == 3, (
        f"coresident_admission.retry_read_free_vram: with attempts=3 and two flakes "
        f"the reader must be called 3 times, got {reader.calls}"
    )


def test_attempts_one_with_a_flake_returns_empty_list_after_one_call():
    reader = _ScriptedReader(None, [7000])
    got = retry_read_free_vram(reader, attempts=1)
    assert got == [], (
        f"coresident_admission.retry_read_free_vram: with attempts=1 a flake must "
        f"give an empty list, got {got}"
    )
    assert reader.calls == 1, (
        f"coresident_admission.retry_read_free_vram: with attempts=1 the reader must "
        f"be called once, got {reader.calls}"
    )


@pytest.mark.parametrize("attempts", [0, -1])
def test_attempts_below_one_raises_value_error_before_any_read(attempts):
    reader = _ScriptedReader([1000])
    with pytest.raises(ValueError, match="attempts must be at least 1"):
        retry_read_free_vram(reader, attempts=attempts)
    assert reader.calls == 0, (
        f"coresident_admission.retry_read_free_vram: attempts={attempts} must be "
        f"refused before the reader is called, got {reader.calls} calls"
    )


@pytest.mark.parametrize(
    "reading",
    [[3000, 1000, 2000], [5, 5, 1], [0], [0, 0]],
    ids=["three-cards-unsorted", "duplicates", "one-card-zero-free", "two-cards-zero-free"],
)
def test_returned_list_is_the_reading_unchanged_in_values_and_order(reading):
    # A card with 0 MiB free is a real reading, not a flake: [0] must come back.
    reader = _ScriptedReader(list(reading))
    got = retry_read_free_vram(reader)
    assert got == reading, (
        f"coresident_admission.retry_read_free_vram: the reading must come back "
        f"unchanged in values and order, expected {reading}, got {got}"
    )
    assert reader.calls == 1, (
        f"coresident_admission.retry_read_free_vram: a usable reading must cost one "
        f"call, got {reader.calls}"
    )
