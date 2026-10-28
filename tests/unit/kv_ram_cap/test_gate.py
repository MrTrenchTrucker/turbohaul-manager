"""Module-interface tests for kv_ram_cap.cap_exceeded and enforce_save_cap (the save-time gate).

The probes count every call, so each test pins how many times measure and make_room ran,
as well as the SaveGate the caller receives.
"""

import sys
from pathlib import Path

import pytest

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.kv_ram_cap import SaveGate, cap_exceeded, enforce_save_cap  # noqa: E402

CARD = "see src/turbohaul/kv_ram_cap/AGENTS.md"


class _Probe:
    """measure returns the scripted sizes in turn, then repeats the last one."""

    def __init__(self, sizes: list[int], room_error: Exception | None = None):
        self._sizes = list(sizes)
        self._room_error = room_error
        self.measure_calls = 0
        self.room_calls = 0

    def measure(self) -> int:
        self.measure_calls += 1
        if len(self._sizes) > 1:
            return self._sizes.pop(0)
        return self._sizes[0]

    async def make_room(self) -> None:
        self.room_calls += 1
        if self._room_error is not None:
            raise self._room_error


def test_cap_exceeded_is_false_just_below_the_cap():
    assert cap_exceeded(99, 100) is False, f"cap_exceeded: 99 bytes under a 100 cap is not exceeded ({CARD})"


def test_cap_exceeded_is_true_exactly_at_the_cap():
    assert cap_exceeded(100, 100) is True, f"cap_exceeded: uses >=, so exactly at the cap is exceeded ({CARD})"


def test_cap_exceeded_is_true_above_the_cap():
    assert cap_exceeded(101, 100) is True, f"cap_exceeded: 101 bytes over a 100 cap is exceeded ({CARD})"


def test_cap_of_zero_means_no_cap_even_at_zero_bytes():
    assert cap_exceeded(10 ** 12, 0) is False, f"cap_exceeded: a cap of 0 is never exceeded ({CARD})"
    assert cap_exceeded(0, 0) is False, f"cap_exceeded: a cap of 0 is never exceeded, even at 0 bytes ({CARD})"


def test_negative_cap_means_no_cap():
    assert cap_exceeded(5, -1) is False, f"cap_exceeded: a negative cap is never exceeded ({CARD})"


@pytest.mark.asyncio
async def test_zero_cap_never_measures_and_never_makes_room():
    p = _Probe([10 ** 9])
    gate = await enforce_save_cap(0, p.measure, p.make_room)
    assert gate == SaveGate(True, 0, 0, False), f"enforce_save_cap: a zero cap must proceed with zeros ({CARD})"
    assert (p.measure_calls, p.room_calls) == (0, 0), f"enforce_save_cap: a zero cap must not measure ({CARD})"


@pytest.mark.asyncio
async def test_negative_cap_never_measures_and_never_makes_room():
    p = _Probe([10 ** 9])
    gate = await enforce_save_cap(-5, p.measure, p.make_room)
    assert gate == SaveGate(True, 0, -5, False), f"enforce_save_cap: a negative cap must proceed ({CARD})"
    assert (p.measure_calls, p.room_calls) == (0, 0), f"enforce_save_cap: a negative cap must not measure ({CARD})"


@pytest.mark.asyncio
async def test_under_the_cap_measures_once_and_skips_make_room():
    p = _Probe([99])
    gate = await enforce_save_cap(100, p.measure, p.make_room)
    assert gate == SaveGate(True, 99, 100, False), f"enforce_save_cap: 99 under 100 must proceed unchanged ({CARD})"
    assert (p.measure_calls, p.room_calls) == (1, 0), f"enforce_save_cap: under the cap, GC must not run ({CARD})"


@pytest.mark.asyncio
async def test_exactly_at_the_cap_counts_as_over_and_runs_make_room_once():
    p = _Probe([100, 40])
    gate = await enforce_save_cap(100, p.measure, p.make_room)
    assert gate == SaveGate(True, 40, 100, True), f"enforce_save_cap: GC made room, so the save proceeds ({CARD})"
    assert (p.measure_calls, p.room_calls) == (2, 1), (
        f"enforce_save_cap: at the cap, measure twice and GC once ({CARD})")


@pytest.mark.asyncio
async def test_over_the_cap_then_made_room_proceeds():
    p = _Probe([150, 80])
    gate = await enforce_save_cap(100, p.measure, p.make_room)
    assert gate == SaveGate(True, 80, 100, True), (
        f"enforce_save_cap: room made, so proceed with made_room True ({CARD})")
    assert (p.measure_calls, p.room_calls) == (2, 1), f"enforce_save_cap: over the cap, make_room runs once ({CARD})"


@pytest.mark.asyncio
async def test_still_over_the_cap_after_make_room_declines_and_reports_the_last_measurement():
    p = _Probe([150, 130])
    gate = await enforce_save_cap(100, p.measure, p.make_room)
    assert gate == SaveGate(False, 130, 100, True), (
        f"enforce_save_cap: still over after GC must decline, with the second total ({CARD})")
    assert (p.measure_calls, p.room_calls) == (2, 1), f"enforce_save_cap: make_room must run at most once ({CARD})"


@pytest.mark.asyncio
async def test_make_room_failure_propagates_and_no_second_measurement_is_taken():
    p = _Probe([150], room_error=RuntimeError("gc boom"))
    with pytest.raises(RuntimeError, match="gc boom"):
        await enforce_save_cap(100, p.measure, p.make_room)
    assert (p.measure_calls, p.room_calls) == (1, 1), f"enforce_save_cap: a make_room failure must propagate ({CARD})"


@pytest.mark.asyncio
async def test_measure_failure_propagates_before_any_gc_runs():
    room_calls = []

    def measure() -> int:
        raise OSError("measure boom")

    async def make_room() -> None:
        room_calls.append(1)

    with pytest.raises(OSError, match="measure boom"):
        await enforce_save_cap(100, measure, make_room)
    assert room_calls == [], f"enforce_save_cap: a failing measure must not run the GC ({CARD})"
