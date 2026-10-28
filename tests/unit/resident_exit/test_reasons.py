"""ExitReason: the eight causes, each with its exact text (resident_exit, see src/turbohaul/resident_exit/AGENTS.md)."""

import sys
from pathlib import Path

import pytest

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.resident_exit import ExitReason  # noqa: E402

CARD = "resident_exit, see src/turbohaul/resident_exit/AGENTS.md"

EXPECTED = {
    "IMMEDIATE_EVICT": "immediate_evict_keep_alive_0",
    "DESIGNATED_VICTIM": "designated_victim_idle_break",
    "SELF_IDLE_TIMEOUT": "self_idle_timeout",
    "DRIVER_EXIT": "driver_exit",
    "DRIVER_REAP": "driver_reap",
    "IDLE_ENGINE_DEAD": "idle_engine_dead",
    "MAKE_ROOM": "make_room",
    "UNSPECIFIED": "unspecified",
}


def test_members_are_exactly_the_spec_names_and_values():
    got = {m.name: m.value for m in ExitReason}
    assert got == EXPECTED, f"{CARD}: ExitReason names and values differ: {got!r}"


def test_member_count_is_exactly_eight():
    count = len(list(ExitReason))
    assert count == 8, f"{CARD}: expected 8 ExitReason members, got {count}"


def test_values_are_all_distinct():
    values = [m.value for m in ExitReason]
    assert len(set(values)) == 8, f"{CARD}: ExitReason values are not distinct: {values!r}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_member_by_name_has_its_value(name):
    member = ExitReason[name]
    assert member.value == EXPECTED[name], f"{CARD}: {name} value is {member.value!r}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_member_is_a_str_and_renders_its_value(name):
    member = ExitReason[name]
    assert isinstance(member, str), f"{CARD}: {name} is not a str"
    rendered_fstring = f"{member}"
    assert rendered_fstring == EXPECTED[name], f"{CARD}: f-string of {name} is {rendered_fstring!r}"
    assert str(member) == EXPECTED[name], f"{CARD}: str() of {name} is {str(member)!r}"
