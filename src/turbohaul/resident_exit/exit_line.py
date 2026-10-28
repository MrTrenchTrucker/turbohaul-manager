"""The single exit-class line each resident death produces, and which deaths get one.

A resident that reaches DEAD leaves exactly one RESIDENT_EXIT line, with the same
fields in the same order every time, so a reader can join a death to its cause without
guessing. The make-room path already writes its own MAKE_ROOM_EVICTION line, so MAKE_ROOM
gets no second line here. This module only shapes the text; the caller logs it.
"""

from __future__ import annotations

from .reasons import ExitReason


def needs_exit_line(reason: "ExitReason | str") -> bool:
    """False only for MAKE_ROOM, whose death is already logged by the make-room path.

    Compares the text, so a plain string "make_room" is treated the same as the member.
    """
    return str(reason) != ExitReason.MAKE_ROOM.value


def _collapse_whitespace(text: str) -> str:
    """Each run of whitespace becomes one underscore.

    A log reader splits fields on spaces, so a value with a space or a newline in it
    would push the fields after it out of place.
    """
    out: list[str] = []
    in_run = False
    for ch in text:
        if ch.isspace():
            if not in_run:
                out.append("_")
            in_run = True
        else:
            out.append(ch)
            in_run = False
    return "".join(out)


def _clean(value: object, limit: int | None = None) -> str:
    """One field's text: "-" when there is nothing, whitespace collapsed, optionally cut.

    A missing value, an empty one, or one that is only whitespace is "-": a blank field
    is nothing, and it must not read as a stray underscore. The cut is applied after
    collapsing, so the limit counts characters the reader sees.
    """
    text = "" if value is None else str(value)
    if not text.strip():
        return "-"
    text = _collapse_whitespace(text)
    if limit is not None:
        text = text[:limit]
    return text


def format_resident_exit(
    *,
    reason: "ExitReason | str",
    resident_key: object,
    model_tag: object,
    thread: object,
    port: object,
    state_before: object,
    state_after: object,
) -> str:
    """The one RESIDENT_EXIT line for a death, with no trailing newline.

    The field order starts with reason= to match IDLE_TEARDOWN and MAKE_ROOM_EVICTION,
    so the three lines line up when read together. The thread id is cut at 60 characters
    so one long id cannot push the other fields off a readable line.
    """
    return (
        f"RESIDENT_EXIT reason={_clean(reason)} resident_key={_clean(resident_key)} "
        f"model_tag={_clean(model_tag)} thread={_clean(thread, limit=60)} "
        f"port={_clean(port)} state_before={_clean(state_before)} "
        f"state_after={_clean(state_after)}"
    )
