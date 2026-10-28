"""co_residence.lines: the decline reasons and the two single-line log formats."""

import enum

import pytest

from turbohaul.co_residence import DeclineReason, format_recheck_action, format_step2_decline


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"co_residence: {text} (the contract is in src/turbohaul/co_residence/AGENTS.md, section Invariants)"


DECLINE_KEYS = [
    "via", "tag", "reason", "eff_cap", "instances", "residents", "max_sidecars", "need_mib", "card", "split",
]
RECHECK_KEYS = ["tag", "source", "slot", "waited_s", "instances", "residents"]


def _decline(**kw) -> str:
    args = dict(
        via="arrival", tag="model-a:q4", reason=DeclineReason.NO_CARD, eff_cap=2, instances=1, residents=1,
        max_sidecars=2, need_mib=18000, card=0, split="none",
    )
    args.update(kw)
    return format_step2_decline(**args)


def _action(**kw) -> str:
    args = dict(tag="model-a:q4", source="staged", slot_id="slot-1", waited_s=12.0, instances=1, residents=1)
    args.update(kw)
    return format_recheck_action(**args)


def _fields(line: str, prefix: str) -> list[tuple[str, str]]:
    head, *rest = line.split(" ")
    assert head == prefix, why(f"line starts with {head!r}, not {prefix!r}")
    return [tuple(p.split("=", 1)) for p in rest]  # type: ignore[misc]


# -- DeclineReason -------------------------------------------------------------------

def test_decline_reason_is_a_str_enum_with_exactly_the_four_values_in_order():
    assert issubclass(DeclineReason, enum.StrEnum), why("DeclineReason is a StrEnum")
    assert [(m.name, m.value) for m in DeclineReason] == [
        ("CAP", "cap"), ("BOX_FULL", "box_full"), ("NO_CARD", "no_card"), ("VRAM_REFUSED", "vram_refused"),
    ], why("DeclineReason members")


@pytest.mark.parametrize("member, text", [
    (DeclineReason.CAP, "cap"), (DeclineReason.BOX_FULL, "box_full"),
    (DeclineReason.NO_CARD, "no_card"), (DeclineReason.VRAM_REFUSED, "vram_refused"),
])
def test_decline_reason_str_is_its_value_and_equals_the_plain_string(member, text):
    assert str(member) == text and member == text and isinstance(member, str), why(f"{member!r}")


# -- format_step2_decline: shape ------------------------------------------------------

def test_decline_line_is_exactly_this_text():
    assert _decline() == (
        "CO_RESIDENCE_DECLINED via=arrival tag=model-a:q4 reason=no_card eff_cap=2 instances=1 residents=1 "
        "max_sidecars=2 need_mib=18000 card=0 split=none"
    ), why("full decline line")


def test_decline_fields_come_in_the_fixed_order_with_their_own_values():
    line = format_step2_decline(
        via="V", tag="T", reason="R", eff_cap="E", instances="I", residents="S", max_sidecars="M",
        need_mib="N", card="C", split="P",
    )
    assert _fields(line, "CO_RESIDENCE_DECLINED") == list(zip(
        DECLINE_KEYS, ["V", "T", "R", "E", "I", "S", "M", "N", "C", "P"], strict=True,
    )), why("each keyword lands under its own name, in order")


def test_decline_line_has_no_trailing_newline_and_is_one_line():
    line = _decline()
    assert not line.endswith("\n") and "\n" not in line and "\r" not in line, why("one line, no newline")


def test_decline_reason_enum_member_renders_its_value():
    for member in DeclineReason:
        assert f" reason={member.value} " in _decline(reason=member), why(f"{member!r}")


def test_decline_reason_as_plain_string_renders_the_same():
    assert _decline(reason="no_card") == _decline(reason=DeclineReason.NO_CARD)


def test_decline_zero_and_false_render_as_themselves_not_a_dash():
    # A falsy value is a real value; only None and blank text mean "absent".
    line = _decline(eff_cap=0, instances=0, residents=0, max_sidecars=0, need_mib=0, card=0, split=False)
    assert dict(_fields(line, "CO_RESIDENCE_DECLINED")) == {
        "via": "arrival", "tag": "model-a:q4", "reason": "no_card", "eff_cap": "0", "instances": "0",
        "residents": "0", "max_sidecars": "0", "need_mib": "0", "card": "0", "split": "False",
    }, why("0 and False are real values")


def test_decline_absent_values_render_as_a_dash():
    line = _decline(need_mib=None, card=None, split=None)
    f = dict(_fields(line, "CO_RESIDENCE_DECLINED"))
    assert (f["need_mib"], f["card"], f["split"]) == ("-", "-", "-"), why("None renders '-'")
    assert len(line.split(" ")) == 11, why("the field count is unchanged when values are absent")


@pytest.mark.parametrize("blank", ["", " ", "   ", "\t", "\n", " \r\n\t ", "\u00a0", "\u2003 \u2003"])
def test_decline_blank_text_renders_as_a_dash(blank):
    line = _decline(tag=blank, via=blank, split=blank)
    f = dict(_fields(line, "CO_RESIDENCE_DECLINED"))
    assert (f["via"], f["tag"], f["split"]) == ("-", "-", "-"), why(f"{blank!r} is blank")


def test_decline_embedded_whitespace_becomes_one_underscore_per_run():
    f = dict(_fields(_decline(tag="a b", split="a  \t\n b", card="x\r\ny"), "CO_RESIDENCE_DECLINED"))
    assert f["tag"] == "a_b", why("a space becomes _")
    assert f["split"] == "a_b", why("a whole run of mixed whitespace becomes ONE _")
    assert f["card"] == "x_y", why("CRLF is one run")


def test_decline_leading_and_trailing_whitespace_become_underscores():
    f = dict(_fields(_decline(tag=" a ", split="\nb\n"), "CO_RESIDENCE_DECLINED"))
    assert f["tag"] == "_a_" and f["split"] == "_b_", why("edge whitespace is collapsed, not stripped")


def test_decline_unicode_whitespace_is_collapsed_like_str_isspace():
    f = dict(_fields(_decline(tag="a\u00a0b\u2003\u2003c"), "CO_RESIDENCE_DECLINED"))
    assert f["tag"] == "a_b_c", why("any character with str.isspace counts")


def test_decline_embedded_newlines_in_every_field_keep_one_line():
    nasty = "x\ny\r\nz"
    line = format_step2_decline(
        via=nasty, tag=nasty, reason=nasty, eff_cap=nasty, instances=nasty, residents=nasty,
        max_sidecars=nasty, need_mib=nasty, card=nasty, split=nasty,
    )
    assert "\n" not in line and "\r" not in line, why("no field may break the line")
    assert [v for _k, v in _fields(line, "CO_RESIDENCE_DECLINED")] == ["x_y_z"] * 10


def test_decline_has_no_length_limit_and_keeps_non_whitespace_characters():
    tag = "m" * 3000 + ":q4|x=y"
    assert dict(_fields(_decline(tag=tag), "CO_RESIDENCE_DECLINED"))["tag"] == tag, why("long tag is untouched")


def test_decline_is_keyword_only_and_every_field_is_required():
    with pytest.raises(TypeError):
        format_step2_decline("arrival", "t", "cap", 1, 1, 1, 2, None, None, None)  # type: ignore[misc]
    kw = dict(via="a", tag="t", reason="cap", eff_cap=1, instances=1, residents=1, max_sidecars=2,
              need_mib=None, card=None, split=None)
    for name in kw:
        short = {k: v for k, v in kw.items() if k != name}
        with pytest.raises(TypeError):
            format_step2_decline(**short)


# -- format_recheck_action ------------------------------------------------------------

def test_action_line_is_exactly_this_text():
    assert _action() == (
        "CO_RESIDENCE_RECHECK tag=model-a:q4 source=staged slot=slot-1 waited_s=12.0 instances=1 residents=1"
    ), why("full re-check line")


def test_action_fields_come_in_the_fixed_order_with_their_own_values():
    line = format_recheck_action(tag="T", source="S", slot_id="L", waited_s=1.5, instances="I", residents="R")
    assert _fields(line, "CO_RESIDENCE_RECHECK") == list(zip(
        RECHECK_KEYS, ["T", "S", "L", "1.5", "I", "R"], strict=True,
    )), why("each keyword lands under its own name, in order")


@pytest.mark.parametrize("waited, text", [
    (0, "0.0"), (5, "5.0"), (12.0, "12.0"), (12.34, "12.3"), (12.26, "12.3"), (1234.56, "1234.6"),
    (7.5, "7.5"), ("7.5", "7.5"), (" 7 ", "7.0"), (True, "1.0"), (-3.0, "-3.0"),
])
def test_action_waited_has_exactly_one_decimal(waited, text):
    got = dict(_fields(_action(waited_s=waited), "CO_RESIDENCE_RECHECK"))["waited_s"]
    assert got == text, why(f"waited_s={waited!r} rendered {got!r}")


@pytest.mark.parametrize("junk", [None, "", "abc", "1,5", [], {}, object(), (1, 2)])
def test_action_waited_that_cannot_be_a_float_is_a_dash(junk):
    got = dict(_fields(_action(waited_s=junk), "CO_RESIDENCE_RECHECK"))["waited_s"]
    assert got == "-", why(f"waited_s={junk!r} rendered {got!r}")


def test_action_absent_and_blank_fields_are_dashes_and_zero_is_a_value():
    f = dict(_fields(_action(tag=None, source="", slot_id="  ", instances=0, residents=0), "CO_RESIDENCE_RECHECK"))
    assert (f["tag"], f["source"], f["slot"], f["instances"], f["residents"]) == ("-", "-", "-", "0", "0")


def test_action_whitespace_collapses_and_the_line_stays_single():
    line = _action(tag="a b\nc", slot_id="s\r\n 1", source="in box")
    f = dict(_fields(line, "CO_RESIDENCE_RECHECK"))
    assert (f["tag"], f["slot"], f["source"]) == ("a_b_c", "s_1", "in_box"), why("whitespace runs")
    assert "\n" not in line and not line.endswith("\n")


def test_action_is_keyword_only():
    with pytest.raises(TypeError):
        format_recheck_action("t", "staged", "s", 1.0, 1, 1)  # type: ignore[misc]


# -- totality ----------------------------------------------------------------------------

ODD = [None, 0, -1, 1.5, float("nan"), float("inf"), True, b"bytes", [1, "a b"], {"k": "v\n"}, (1,), object(), "", "\n"]


@pytest.mark.parametrize("value", ODD, ids=repr)
def test_formatters_never_raise_and_always_return_one_line(value):
    a = format_step2_decline(
        via=value, tag=value, reason=value, eff_cap=value, instances=value, residents=value,
        max_sidecars=value, need_mib=value, card=value, split=value,
    )
    b = format_recheck_action(tag=value, source=value, slot_id=value, waited_s=value, instances=value, residents=value)
    for line in (a, b):
        assert isinstance(line, str) and "\n" not in line and "\r" not in line, why(f"{value!r} broke a line")
    assert len(a.split(" ")) == 11 and len(b.split(" ")) == 7, why(f"{value!r} changed the field count")
