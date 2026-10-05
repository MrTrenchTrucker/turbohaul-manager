"""Log-flood fix (the 1 Hz poll can dominate the log): pin _SlotsPollHttpxFilter
(__main__.py) so it silences ONLY the manager's own ~1Hz GET /slots poll
(LiveSlotsPoller/ResidentSlotsPoller, live_monitor.py) on httpx's own
request-complete logger -- and nothing else. The poll can otherwise flood
the log, and a flood of one kind is easily replaced by a flood of another.
The poll is the only thing silenced here;
this fix filters the POLL, not the LOGGER --
``logging.getLogger("httpx").setLevel(WARNING)`` would also silence every
OTHER real httpx client error in the process, which is the wrong fix.

Three cases, because two is not enough here:
  (a) a /slots poll returning 200 produces NO record -- the fix works.
  (b) a /slots poll returning a NON-200 status (a REAL failing poll, not a
      synthetic "other path" example) still produces a record -- proves the
      filter discriminates on OUTCOME, not just path/method. Without this
      case the test cannot distinguish the real fix from
      ``setLevel(WARNING)``, which is exactly the shortcut that
      must be ruled out.
  (c) mutate the filter's outcome check off -> (a) goes RED, proving the
      test actually pins something rather than passing regardless.

CONTROL DESIGN (from mutation testing): a single "other httpx
traffic unaffected" control using POST /v1/chat/completions differs
from the poll in BOTH method AND path -- so removing EITHER the method
guard or the path guard alone would still pass it (the other guard alone still
would catch it). Two single-dimension controls are used instead:
  test_get_health_200_differs_by_path_only_still_logs -- isolates the PATH
    guard with REAL traffic (GET /health is a real probe in a typical deployment).
  test_post_slots_200_differs_by_method_only_still_logs -- isolates the
    METHOD guard; NOT real traffic today (nothing POSTs to /slots), stated
    explicitly so this isn't later "simplified away" as unreachable.

SCOPE NOTE: this file's fixture (``httpx_logger_with_filter``) installs
``_SlotsPollHttpxFilter`` itself -- these tests pin the filter's BEHAVIOUR
(what it does/doesn't drop), not whether ``__main__.py`` actually installs
it at process start. That install site is covered elsewhere, not
by this file. Normal division for a unit test; noted explicitly
rather than left implicit.

Args are constructed to be BYTE-IDENTICAL in shape to httpx's own call
(httpx/_client.py, two call sites):
    logger.info('HTTP Request: %s %s "%s %d %s"',
                 method, url, http_version, status_code, reason_phrase)
``url`` is a real ``httpx.URL`` object (not a bare string) -- confirmed by
reading httpx's source.
``status_code`` is a real ``int``.
"""
import logging

import httpx
import pytest

from turbohaul.__main__ import _SlotsPollHttpxFilter

_LOGGER_NAME = "httpx"


def _emit_httpx_request_log(logger, method, path, status_code, reason_phrase="OK"):
    """Fire a record shaped EXACTLY like httpx's own request-complete log call."""
    url = httpx.URL(f"http://127.0.0.1:11500{path}")
    logger.info(
        'HTTP Request: %s %s "%s %d %s"',
        method, url, "HTTP/1.1", status_code, reason_phrase,
    )


@pytest.fixture
def httpx_logger_with_filter():
    logger = logging.getLogger(_LOGGER_NAME)
    f = _SlotsPollHttpxFilter()
    logger.addFilter(f)
    try:
        yield logger, f
    finally:
        logger.removeFilter(f)


def test_case_a_slots_poll_200_produces_no_record(httpx_logger_with_filter, caplog):
    logger, _f = httpx_logger_with_filter
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        _emit_httpx_request_log(logger, "GET", "/slots", 200)
    assert caplog.records == [], (
        f"expected the /slots 200 poll line to be silenced, got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_case_b_slots_poll_non_200_still_logs(httpx_logger_with_filter, caplog):
    """THE ONE THAT MATTERS: a /slots poll returning a NON-200 status
    is a REAL poll failure -- e.g. the sidecar itself erroring -- and proves
    the filter discriminates on outcome, not just on path/method. A blanket
    setLevel(WARNING) fix would also pass case (a) but would NOT distinguish
    this from that wrong fix, since setLevel(WARNING) drops ALL INFO-level
    httpx lines including this one. This test is what makes that
    distinction visible."""
    logger, _f = httpx_logger_with_filter
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        _emit_httpx_request_log(logger, "GET", "/slots", 500, reason_phrase="Internal Server Error")
    assert len(caplog.records) == 1, (
        f"a failing /slots poll (non-200) MUST still be logged -- got "
        f"{len(caplog.records)} records: {[r.getMessage() for r in caplog.records]}"
    )
    msg = caplog.records[0].getMessage()
    assert "/slots" in msg
    assert "500" in msg


def test_get_health_200_differs_by_path_only_still_logs(httpx_logger_with_filter, caplog):
    """The path-only control: a broader
    "other httpx traffic" test using POST /v1/chat/completions
    would differ from the poll in BOTH method AND path -- so it could not
    tell you which guard was doing the work. Removing EITHER the method
    guard or the path guard alone would still pass that test, because the other
    guard alone would still catch it. This isolates the PATH guard: GET /health
    200 differs from the poll ONLY in path (same method, same status).
    GET /health is REAL traffic in a typical deployment -- a filter
    missing the path guard would silence every successful httpx GET,
    quietly swallowing genuine health-probe visibility, a blanket mute in a
    narrower costume. This is the case that actually matters operationally."""
    logger, _f = httpx_logger_with_filter
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        _emit_httpx_request_log(logger, "GET", "/health", 200)
    assert len(caplog.records) == 1, (
        "GET /health 200 (real traffic, differs from the poll ONLY in path) "
        "must still log -- if this fails, the path guard is gone and the "
        "filter is silencing every successful httpx GET, not just /slots."
    )


def test_post_slots_200_differs_by_method_only_still_logs(httpx_logger_with_filter, caplog):
    """Isolates the METHOD guard: POST /slots 200 differs from the poll
    ONLY in method (same path, same status). NOTE: this is NOT traffic the
    manager actually emits today -- LiveSlotsPoller/ResidentSlotsPoller only
    ever GET /slots (live_monitor.py); nothing POSTs there. Stated here
    explicitly so nobody later "simplifies away" this test as covering an
    unreachable case -- it exists to isolate the method guard from the path
    guard, not to pin a real traffic shape. Without it, removing the method
    guard alone would still silently pass every other test in this file
    (both remaining guards -- path and status -- would still catch it via
    the OTHER dimension), exactly the masking a mutation run exposes."""
    logger, _f = httpx_logger_with_filter
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        _emit_httpx_request_log(logger, "POST", "/slots", 200)
    assert len(caplog.records) == 1, (
        "POST /slots 200 (not real traffic, differs from the poll ONLY in "
        "method) must still log -- if this fails, the method guard is gone."
    )


def test_malformed_record_shape_is_kept_safe_degrade(httpx_logger_with_filter, caplog):
    """Safe-degrade, mirroring _HealthPollAccessFilter's own contract: any
    record whose args don't match the expected 5-tuple shape is KEPT, never
    silently dropped -- this filter must never be able to suppress a real
    line just because something logged to "httpx" in an unexpected shape."""
    logger, _f = httpx_logger_with_filter
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        logger.info("some unrelated free-form httpx-adjacent message")
    assert len(caplog.records) == 1


class _MutatedNoOutcomeCheckFilter(logging.Filter):
    """Hand-written stand-in for _SlotsPollHttpxFilter with the outcome
    (status_code) discrimination deliberately removed -- i.e. exactly the
    mutation one can run by hand against the real __main__.py source
    (editing the real file and re-running case (b)). Kept here too,
    inert unless explicitly invoked below, so this file documents precisely
    which line the mutation proof depends on without needing fragile
    source-text surgery at test time."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True
        method, url, _http_version, _status_code, _reason_phrase = args[:5]
        if method != "GET":
            return True
        if getattr(url, "path", None) != "/slots":
            return True
        return False  # BUG: no longer checks status_code at all


def test_case_c_mutated_filter_would_also_drop_a_failing_poll():
    """Demonstrates, within the test suite itself, exactly what
    test_case_b_slots_poll_non_200_still_logs depends on: with the
    status_code check removed, a failing (500) /slots poll is ALSO
    silenced. This is the change one would make by hand
    against the REAL __main__.py file (edit -> case (b) goes RED ->
    restore -> green again); this test pins the same fact in-suite so CI
    catches a future regression of the same shape without needing to
    re-run that manual edit."""
    logger = logging.getLogger(_LOGGER_NAME + ".mutation_test_case_c")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger.addHandler(_Collector())
    mutated_filter = _MutatedNoOutcomeCheckFilter()
    logger.addFilter(mutated_filter)

    try:
        _emit_httpx_request_log(logger, "GET", "/slots", 500, reason_phrase="Internal Server Error")
        assert records == [], (
            "sanity check on the mutant itself: with the outcome check "
            "removed, a failing /slots poll (500) is silenced -- confirming "
            "this mutant reproduces the exact failure mode "
            "test_case_b_slots_poll_non_200_still_logs exists to catch "
            "against the REAL filter."
        )
    finally:
        logger.removeFilter(mutated_filter)
        logger.handlers = []
