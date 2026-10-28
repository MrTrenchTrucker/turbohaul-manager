"""resolve_exit_reason: the stamped reason wins, otherwise the idle flag picks the default.

resident_exit_wiring, see src/turbohaul/resident_exit_wiring/AGENTS.md
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.resident_exit import ExitReason  # noqa: E402
from turbohaul.resident_exit_wiring import resolve_exit_reason  # noqa: E402

CARD = "resident_exit_wiring, see src/turbohaul/resident_exit_wiring/AGENTS.md"


@pytest.mark.parametrize("idle", [True, False])
def test_a_stamped_reason_is_returned_whatever_the_idle_flag(idle):
    r = SimpleNamespace(exit_reason=ExitReason.DRIVER_REAP)
    got = resolve_exit_reason(r, idle)
    assert got is ExitReason.DRIVER_REAP, f"{CARD}: stamped reason came back as {got!r} (idle={idle})"


def test_a_stamped_plain_string_is_returned_unchanged():
    r = SimpleNamespace(exit_reason="make_room")
    got = resolve_exit_reason(r, True)
    assert got == "make_room", f"{CARD}: got {got!r}"
    assert type(got) is str, f"{CARD}: the stamp must not be converted, got {type(got)!r}"


def test_an_unstamped_idle_resident_resolves_to_self_idle_timeout():
    r = SimpleNamespace(exit_reason=None)
    got = resolve_exit_reason(r, True)
    assert got is ExitReason.SELF_IDLE_TIMEOUT, f"{CARD}: got {got!r}"


def test_an_unstamped_resident_that_is_not_idle_resolves_to_unspecified():
    r = SimpleNamespace(exit_reason=None)
    got = resolve_exit_reason(r, False)
    assert got is ExitReason.UNSPECIFIED, f"{CARD}: got {got!r}"


@pytest.mark.parametrize(
    ("idle", "expected"),
    [(True, ExitReason.SELF_IDLE_TIMEOUT), (False, ExitReason.UNSPECIFIED)],
    ids=["idle", "not-idle"],
)
def test_a_resident_with_no_exit_reason_attribute_at_all_is_treated_as_unstamped(idle, expected):
    got = resolve_exit_reason(SimpleNamespace(), idle)
    assert got is expected, f"{CARD}: got {got!r} for idle={idle}"
