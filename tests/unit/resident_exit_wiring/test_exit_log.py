"""log_resident_exit: what is written, at what level, from which fields, and what is swallowed.

resident_exit_wiring, see src/turbohaul/resident_exit_wiring/AGENTS.md
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.resident_exit import ExitReason, format_resident_exit  # noqa: E402
from turbohaul.resident_exit_wiring import log_resident_exit  # noqa: E402

CARD = "resident_exit_wiring, see src/turbohaul/resident_exit_wiring/AGENTS.md"
LOGGER = "turbohaul.manager"


def _resident(*, tip=None, slot=None, idle=None, handle_port=None, key="key-1", tag="model-1"):
    return SimpleNamespace(
        resident_key=key,
        model_tag=tag,
        grace_tip=None if tip is None else SimpleNamespace(thread_id=tip),
        active_slot=None if slot is None else SimpleNamespace(thread_id=slot),
        idle_thread_id=idle,
        handle=None if handle_port is None else SimpleNamespace(port=handle_port),
    )


def _records(caplog, level):
    return [rec for rec in caplog.records if rec.name == LOGGER and rec.levelno == level]


def _write(r, reason="driver_exit", state_before="ACTIVE", port=None, **kw):
    kw.setdefault("format_line", format_resident_exit)
    kw.setdefault("state_after", "DEAD")
    log_resident_exit(r, reason, state_before, port, **kw)


def test_the_exact_line_is_written_once_at_info_on_the_manager_logger(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    _write(_resident(tip="thread-abc", handle_port=5111), reason=ExitReason.DRIVER_EXIT)
    infos = _records(caplog, logging.INFO)
    assert [rec.getMessage() for rec in infos] == [
        "RESIDENT_EXIT reason=driver_exit resident_key=key-1 model_tag=model-1 "
        "thread=thread-abc port=5111 state_before=ACTIVE state_after=DEAD"
    ], f"{CARD}: got {[rec.getMessage() for rec in caplog.records]!r}"
    assert len(caplog.records) == 1, f"{CARD}: extra records {[rec.getMessage() for rec in caplog.records]!r}"


def test_the_formatter_is_called_with_these_keyword_arguments_only():
    seen = []

    def fake_format(*args, **kwargs):
        seen.append((args, kwargs))
        return "LINE"

    r = _resident(tip="t-1", handle_port=7001)
    log_resident_exit(r, "driver_reap", "IDLE", None, format_line=fake_format, state_after="AFTER")
    assert seen == [((), {
        "reason": "driver_reap", "resident_key": "key-1", "model_tag": "model-1",
        "thread": "t-1", "port": 7001, "state_before": "IDLE", "state_after": "AFTER",
    })], f"{CARD}: the formatter saw {seen!r}"


def test_the_text_the_formatter_returns_is_what_gets_logged(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    _write(_resident(), format_line=lambda **kw: "exactly this text")
    assert [rec.getMessage() for rec in caplog.records] == ["exactly this text"]


@pytest.mark.parametrize("reason", [ExitReason.MAKE_ROOM, "make_room"], ids=["member", "string"])
def test_a_make_room_death_writes_nothing_and_never_calls_the_formatter(caplog, reason):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    calls = []

    def fake_format(**kw):
        calls.append(kw)
        return "LINE"

    _write(_resident(), reason=reason, format_line=fake_format)
    assert calls == [], f"{CARD}: the formatter ran for make_room: {calls!r}"
    assert caplog.records == [], f"{CARD}: wrote {[rec.getMessage() for rec in caplog.records]!r}"


@pytest.mark.parametrize(
    ("tip", "slot", "idle", "expected"),
    [
        ("tip-thread", "slot-thread", "idle-thread", "tip-thread"),
        (None, "slot-thread", "idle-thread", "slot-thread"),
        (None, None, "idle-thread", "idle-thread"),
        (None, None, None, None),
    ],
    ids=["grace-tip", "active-slot", "idle-thread", "none"],
)
def test_the_thread_falls_back_in_order(tip, slot, idle, expected):
    seen = []
    _write(_resident(tip=tip, slot=slot, idle=idle), format_line=lambda **kw: seen.append(kw) or "LINE")
    assert [kw["thread"] for kw in seen] == [expected], f"{CARD}: got {seen!r}"


def test_an_empty_grace_tip_thread_falls_through_to_the_next_source():
    seen = []
    _write(_resident(tip="", slot="slot-thread"), format_line=lambda **kw: seen.append(kw) or "LINE")
    assert seen[0]["thread"] == "slot-thread", f"{CARD}: got {seen!r}"


@pytest.mark.parametrize(
    ("given", "handle_port", "expected"),
    [
        (9000, 5111, 9000),
        (0, 5111, 0),
        (None, 5111, 5111),
        (None, None, None),
    ],
    ids=["given-wins", "zero-is-given", "handle-fallback", "nothing"],
)
def test_the_port_is_the_one_passed_in_else_the_handle_port(given, handle_port, expected):
    seen = []
    _write(_resident(handle_port=handle_port), port=given, format_line=lambda **kw: seen.append(kw) or "LINE")
    assert [kw["port"] for kw in seen] == [expected], f"{CARD}: got {seen!r}"


def test_a_resident_with_none_of_the_fields_still_logs_a_line_of_dashes(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    _write(SimpleNamespace())
    assert [rec.getMessage() for rec in caplog.records] == [
        "RESIDENT_EXIT reason=driver_exit resident_key=- model_tag=- thread=- port=- "
        "state_before=ACTIVE state_after=DEAD"
    ]


def test_a_failing_formatter_is_swallowed_with_one_debug_record_and_no_line(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)

    def boom(**kw):
        raise ValueError("boom")

    _write(_resident(), format_line=boom)
    debugs = _records(caplog, logging.DEBUG)
    assert [rec.getMessage() for rec in debugs] == ["resident exit line failed (best-effort)"], (
        f"{CARD}: got {[(rec.levelname, rec.getMessage()) for rec in caplog.records]!r}"
    )
    assert isinstance(debugs[0].exc_info, tuple) and debugs[0].exc_info[0] is ValueError, (
        f"{CARD}: the debug record must carry the traceback, got {debugs[0].exc_info!r}"
    )
    assert _records(caplog, logging.INFO) == [], f"{CARD}: an info line was written after the failure"


def test_a_failing_field_read_is_swallowed_too(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)

    class Exploding:
        @property
        def grace_tip(self):
            raise RuntimeError("field read failed")

    _write(Exploding())
    assert [rec.getMessage() for rec in _records(caplog, logging.DEBUG)] == [
        "resident exit line failed (best-effort)"
    ]
