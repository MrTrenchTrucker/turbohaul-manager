"""needs_exit_line and format_resident_exit: the exit line's text and which deaths get one.

resident_exit, see src/turbohaul/resident_exit/AGENTS.md
"""

import sys
from pathlib import Path

import pytest

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.resident_exit import ExitReason, format_resident_exit, needs_exit_line  # noqa: E402

CARD = "resident_exit, see src/turbohaul/resident_exit/AGENTS.md"

ORDER = ["reason", "resident_key", "model_tag", "thread", "port", "state_before", "state_after"]
BASE_KW = dict(
    reason=ExitReason.DRIVER_REAP,
    resident_key="resident-key-A",
    model_tag="model-tag-B",
    thread="thread-C",
    port=9001,
    state_before="active",
    state_after="dead",
)
BASE_VALUES = dict(
    reason="driver_reap",
    resident_key="resident-key-A",
    model_tag="model-tag-B",
    thread="thread-C",
    port="9001",
    state_before="active",
    state_after="dead",
)
BLANKS = [None, "", "  \n "]
BLANK_FIELDS = ["resident_key", "model_tag", "thread", "port", "state_before", "state_after"]
NON_MAKE_ROOM = [m for m in ExitReason if m is not ExitReason.MAKE_ROOM]


class _Odd:
    """An object whose only useful property is its str(), with a space and a tab inside."""

    def __str__(self):
        return "w e\tird"


def _render(**over):
    kw = dict(BASE_KW)
    kw.update(over)
    return format_resident_exit(**kw)


def _expect(**over):
    vals = dict(BASE_VALUES)
    vals.update(over)
    return "RESIDENT_EXIT " + " ".join(f"{k}={vals[k]}" for k in ORDER)


# -- (a) which deaths get a line ---------------------------------------------

def test_make_room_gets_no_line_as_member_and_as_plain_string():
    assert needs_exit_line(ExitReason.MAKE_ROOM) is False, f"{CARD}: MAKE_ROOM member gets a line"
    assert needs_exit_line("make_room") is False, f"{CARD}: plain 'make_room' gets a line"


def test_there_are_exactly_seven_other_members():
    assert len(NON_MAKE_ROOM) == 7, f"{CARD}: expected 7 non-MAKE_ROOM members, got {len(NON_MAKE_ROOM)}"


@pytest.mark.parametrize("reason", NON_MAKE_ROOM, ids=[m.name for m in NON_MAKE_ROOM])
def test_every_other_member_gets_a_line(reason):
    assert needs_exit_line(reason) is True, f"{CARD}: {reason.name} gets no exit line"


@pytest.mark.parametrize("reason", ["some_unknown_reason", ""])
def test_unknown_and_empty_strings_get_a_line(reason):
    assert needs_exit_line(reason) is True, f"{CARD}: {reason!r} gets no exit line"


# -- (b) the exact line -------------------------------------------------------

def test_normal_input_renders_the_exact_spec_line():
    line = _render()
    assert line == (
        "RESIDENT_EXIT reason=driver_reap resident_key=resident-key-A model_tag=model-tag-B "
        "thread=thread-C port=9001 state_before=active state_after=dead"
    ), f"{CARD}: exit line text differs: {line!r}"


# -- (c) one line, whatever the values contain ---------------------------------

def test_line_stays_one_line_when_values_contain_newlines_tabs_and_spaces():
    line = _render(thread="alpha\nbeta\tgamma  delta\r\nomega", model_tag="m\nx")
    assert "\n" not in line and "\r" not in line and "\t" not in line, (
        f"{CARD}: a control character leaked into the line: {line!r}"
    )
    assert line == _expect(thread="alpha_beta_gamma_delta_omega", model_tag="m_x"), (
        f"{CARD}: whitespace runs did not collapse to single underscores: {line!r}"
    )


# -- (d) blank values read '-', one field at a time -----------------------------

@pytest.mark.parametrize("blank", BLANKS, ids=["none", "empty", "whitespace_only"])
@pytest.mark.parametrize("field", BLANK_FIELDS)
def test_blank_field_renders_a_dash(field, blank):
    line = _render(**{field: blank})
    assert line == _expect(**{field: "-"}), (
        f"{CARD}: blank {blank!r} in {field} did not render as '-': {line!r}"
    )


def test_inner_and_edge_whitespace_become_underscores():
    line = _render(resident_key=" a  b ")
    assert line == _expect(resident_key="_a_b_"), f"{CARD}: edge/inner whitespace rendered as {line!r}"


# -- (e) port ------------------------------------------------------------------

@pytest.mark.parametrize("port,text", [(8123, "8123"), (0, "0")])
def test_port_as_an_int_renders_as_digits(port, text):
    line = _render(port=port)
    assert line == _expect(port=text), f"{CARD}: port {port!r} rendered as {line!r}"


# -- (f) the thread cut, and only the thread ------------------------------------

def test_thread_is_cut_at_exactly_sixty_characters():
    assert _render(thread="t" * 61) == _expect(thread="t" * 60), f"{CARD}: 61-char thread not cut to 60"
    assert _render(thread="t" * 60) == _expect(thread="t" * 60), f"{CARD}: 60-char thread was changed"


def test_cut_counts_characters_after_whitespace_collapses():
    # 59 letters, a space and an x collapse to 61 characters, so the cut takes the last one.
    assert _render(thread="a" * 59 + " x") == _expect(thread="a" * 59 + "_"), (
        f"{CARD}: the cut did not apply after the whitespace collapse"
    )


def test_model_tag_is_not_cut():
    tag = "m" * 100
    assert _render(model_tag=tag) == _expect(model_tag=tag), f"{CARD}: a 100-char model_tag was cut"


# -- (g) member and plain string render the same ---------------------------------

def test_reason_member_and_plain_string_render_identically():
    member = _render(reason=ExitReason.IMMEDIATE_EVICT)
    plain = _render(reason="immediate_evict_keep_alive_0")
    assert member == plain, f"{CARD}: member and string differ: {member!r} vs {plain!r}"
    assert member.split(" ")[1] == "reason=immediate_evict_keep_alive_0", (
        f"{CARD}: reason field is {member.split(' ')[1]!r}"
    )


# -- (h) prefix and field order ---------------------------------------------------

def test_line_starts_with_the_prefix_and_fields_come_in_spec_order():
    line = _render()
    assert line.startswith("RESIDENT_EXIT reason="), f"{CARD}: wrong prefix: {line!r}"
    positions = [line.index("reason=")] + [line.index(f" {name}=") for name in ORDER[1:]]
    assert all(a < b for a, b in zip(positions, positions[1:])), (
        f"{CARD}: field tokens are out of order, positions {positions}"
    )


# -- (i) never raises ----------------------------------------------------------------

def test_never_raises_for_an_object_whose_str_works():
    line = format_resident_exit(
        reason="custom",
        resident_key=_Odd(),
        model_tag=_Odd(),
        thread=_Odd(),
        port=_Odd(),
        state_before=_Odd(),
        state_after=_Odd(),
    )
    assert line == _expect(
        reason="custom", resident_key="w_e_ird", model_tag="w_e_ird", thread="w_e_ird",
        port="w_e_ird", state_before="w_e_ird", state_after="w_e_ird",
    ), f"{CARD}: object values rendered as {line!r}"
