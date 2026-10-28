"""The rename of the verified file over the database happens while the run still holds the exclusive lock."""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import os
import sqlite3

from state_db_cleanup.helpers import (
    assert_stop,
    internals,
    make_fixture,
    quiet_env,
    real_run,
    report_of,
)

measure = internals("measure")
sequence = internals("sequence")
swap = internals("swap")


def _try_to_write(db):
    """What a second process would meet right now: 'locked', or 'written' if the file was free."""
    other = sqlite3.connect(db, timeout=0, isolation_level=None)
    try:
        other.execute("BEGIN IMMEDIATE")
        other.execute("ROLLBACK")
        return "written"
    except sqlite3.OperationalError as exc:
        return str(exc)
    finally:
        other.close()


def _watch_the_rename(monkeypatch, fx, outcomes, refuse=False):
    real_replace = os.replace

    def replace(src, dst):
        if str(dst) == str(fx.db):
            outcomes.append(_try_to_write(fx.db))
            if refuse:
                raise OSError(16, "Device or resource busy")
        return real_replace(src, dst)

    monkeypatch.setattr(swap.os, "replace", replace)


class _Recording:
    """Stands in for the exclusive connection: forwards everything, notes when it is closed."""

    def __init__(self, conn, events):
        self._conn = conn
        self._events = events

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        self._events.append("close")
        return self._conn.close()


class TestTheRenameHappensUnderTheLock:
    def test_a_second_writer_is_refused_at_the_moment_of_the_rename(self, tmp_path, monkeypatch):
        fx = make_fixture(tmp_path)
        outcomes = []
        _watch_the_rename(monkeypatch, fx, outcomes)
        report = real_run(fx)
        assert report["status"] == "completed", report.get("stop")
        assert len(outcomes) == 1 and "database is locked" in outcomes[0], outcomes

    def test_the_before_swap_hook_runs_with_the_lock_held(self, tmp_path):
        fx = make_fixture(tmp_path)
        outcomes = []
        report = real_run(fx, quiet_env(before_swap=lambda db: outcomes.append(_try_to_write(db))))
        assert report["status"] == "completed", report.get("stop")
        assert len(outcomes) == 1 and "database is locked" in outcomes[0], outcomes

    def test_the_order_is_rename_then_the_check_of_the_swapped_file_then_the_close(self, tmp_path, monkeypatch):
        fx = make_fixture(tmp_path)
        events = []
        real_open, real_integrity, real_replace = sequence._open_exclusive, measure._integrity, os.replace

        def open_recording(*args, **kwargs):
            return _Recording(real_open(*args, **kwargs), events)

        def integrity(conn):
            events.append("integrity")
            return real_integrity(conn)

        def replace(src, dst):
            if str(dst) == str(fx.db):
                events.append("replace")
            return real_replace(src, dst)

        monkeypatch.setattr(sequence, "_open_exclusive", open_recording)
        monkeypatch.setattr(measure, "_integrity", integrity)
        monkeypatch.setattr(swap.os, "replace", replace)
        report = real_run(fx)
        assert report["status"] == "completed", report.get("stop")
        assert events.count("close") == 1 and events.count("replace") == 1
        assert events[-3:] == ["replace", "integrity", "close"], events

    def test_the_run_still_swaps_and_leaves_only_the_database_in_its_directory(self, tmp_path):
        fx = make_fixture(tmp_path)
        inode = fx.db.stat().st_ino
        report = real_run(fx)
        assert report["status"] == "completed", report.get("stop")
        assert fx.db.stat().st_ino != inode and os.listdir(fx.data) == ["state.sqlite"]
        check = sqlite3.connect(fx.db)
        try:
            assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        finally:
            check.close()

    def test_a_refused_rename_stops_cleanly_and_lets_go_of_the_lock(self, tmp_path, monkeypatch):
        fx = make_fixture(tmp_path)
        inode = fx.db.stat().st_ino
        outcomes = []
        _watch_the_rename(monkeypatch, fx, outcomes, refuse=True)
        report = report_of(lambda: real_run(fx))
        assert_stop(report, "swap", "could not replace the database file")
        assert "database is locked" in outcomes[0] and fx.db.stat().st_ino == inode
        assert _try_to_write(fx.db) == "written", "the lock must be released when the run has ended"
