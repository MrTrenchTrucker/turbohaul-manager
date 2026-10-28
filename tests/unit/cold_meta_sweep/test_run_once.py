"""run_once: one bounded run of the retention pass, against a real sqlite file and a fake strip."""
import contextlib
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from turbohaul.cold_meta_sweep import run_once

NOW = datetime.now(UTC)
OLD = (NOW - timedelta(days=3)).isoformat(timespec="seconds")
TWO_HOURS = (NOW - timedelta(hours=2)).isoformat(timespec="seconds")
RECENT = (NOW - timedelta(minutes=5)).isoformat(timespec="seconds")
KEYS = {"stripped", "unchanged", "invalid", "missing", "error", "rows", "pass_done"}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "slots.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE slots (slot_id TEXT PRIMARY KEY, state TEXT, updated_at TEXT, client_meta_json TEXT)")
    conn.commit()
    conn.close()
    return path


def _insert(path, slot_id, *, state="COLD", updated_at=OLD):
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO slots (slot_id, state, updated_at, client_meta_json) VALUES (?, ?, ?, '{}')",
                 (slot_id, state, updated_at))
    conn.commit()
    conn.close()


class Session:
    """A stand-in for the manager's database session: records the path it was opened with."""

    def __init__(self):
        self.opened = []

    @contextlib.contextmanager
    def __call__(self, path):
        self.opened.append(path)
        conn = sqlite3.connect(path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


class Strip:
    """A stand-in for the strip of one row: records each call and answers from a table."""

    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.calls = []

    def __call__(self, conn, slot_id):
        assert isinstance(conn, sqlite3.Connection)
        self.calls.append(slot_id)
        return self.outcomes.get(slot_id, "stripped")


def test_the_session_is_opened_on_the_given_path_and_every_row_goes_to_the_strip_in_row_order(db):
    for sid in ("c", "a", "b"):
        _insert(db, sid)
    session, strip = Session(), Strip()
    run_once(session, strip, db, 0, 1.0, 10)
    assert session.opened == [db]
    assert strip.calls == ["c", "a", "b"], "rows must be handled in rowid (insert) order"


def test_the_counts_have_exactly_the_documented_keys_and_tally_each_outcome(db):
    for sid in ("s", "u", "i", "m", "e", "s2"):
        _insert(db, sid)
    strip = Strip({"u": "unchanged", "i": "invalid", "m": "missing", "e": "error"})
    counts, _ = run_once(Session(), strip, db, 0, 1.0, 10)
    assert counts == {"stripped": 2, "unchanged": 1, "invalid": 1, "missing": 1, "error": 1,
                      "rows": 6, "pass_done": True}
    assert set(counts) == KEYS


def test_only_old_ended_rows_are_picked(db):
    _insert(db, "old-cold")
    _insert(db, "recent-cold", updated_at=RECENT)
    for state in ("ACTIVE", "STAGED", "IDLE_HOT", "LOADING", "POPPED"):
        _insert(db, f"old-{state}", state=state)
    strip = Strip()
    counts, _ = run_once(Session(), strip, db, 0, 1.0, 10)
    assert strip.calls == ["old-cold"]
    assert counts["rows"] == 1


def test_the_cutoff_follows_the_age_argument(db):
    _insert(db, "two-hours-old", updated_at=TWO_HOURS)
    young_run = Strip()
    counts, _ = run_once(Session(), young_run, db, 0, 24.0, 10)
    assert young_run.calls == [] and counts["rows"] == 0
    old_run = Strip()
    counts, _ = run_once(Session(), old_run, db, 0, 1.0, 10)
    assert old_run.calls == ["two-hours-old"] and counts["rows"] == 1


def test_a_partial_batch_finishes_the_pass_and_resets_the_watermark(db):
    for i in range(3):
        _insert(db, f"s{i}")
    counts, watermark = run_once(Session(), Strip(), db, 0, 1.0, 5)
    assert counts["rows"] == 3 and counts["pass_done"] is True
    assert watermark == 0


def test_a_full_batch_returns_the_last_rowid_and_the_next_run_resumes_after_it(db):
    for i in range(5):
        _insert(db, f"s{i}")
    first_strip = Strip()
    counts, watermark = run_once(Session(), first_strip, db, 0, 1.0, 2)
    assert (counts["rows"], counts["pass_done"], watermark) == (2, False, 2)
    assert first_strip.calls == ["s0", "s1"]
    second_strip = Strip()
    counts, watermark = run_once(Session(), second_strip, db, watermark, 1.0, 2)
    assert (counts["rows"], counts["pass_done"], watermark) == (2, False, 4)
    assert second_strip.calls == ["s2", "s3"]
    third_strip = Strip()
    counts, watermark = run_once(Session(), third_strip, db, watermark, 1.0, 2)
    assert (counts["rows"], counts["pass_done"], watermark) == (1, True, 0)
    assert third_strip.calls == ["s4"]


def test_a_batch_that_exactly_fits_is_not_yet_the_end_of_the_pass(db):
    for i in range(2):
        _insert(db, f"s{i}")
    counts, watermark = run_once(Session(), Strip(), db, 0, 1.0, 2)
    assert (counts["rows"], counts["pass_done"], watermark) == (2, False, 2)


def test_an_empty_run_returns_zero_rows_a_finished_pass_and_a_zero_watermark(db):
    counts, watermark = run_once(Session(), Strip(), db, 7, 1.0, 5)
    assert counts == {"stripped": 0, "unchanged": 0, "invalid": 0, "missing": 0, "error": 0,
                      "rows": 0, "pass_done": True}
    assert watermark == 0


def test_a_failure_inside_the_run_propagates_so_the_caller_gets_no_new_watermark(db):
    for i in range(3):
        _insert(db, f"s{i}")
    seen = []

    def boom(conn, slot_id):
        seen.append(slot_id)
        if slot_id == "s1":
            raise RuntimeError("strip blew up")
        return "stripped"

    with pytest.raises(RuntimeError, match="strip blew up"):
        run_once(Session(), boom, db, 0, 1.0, 5)
    assert seen == ["s0", "s1"], "the run stops at the failing row"
