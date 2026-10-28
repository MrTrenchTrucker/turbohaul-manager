"""The backup is compared with the database by the content of the stored values, not only by their total length."""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import sqlite3
from pathlib import Path

from state_db_cleanup.helpers import (
    assert_stop,
    cfg_for,
    cleanup,
    internals,
    make_fixture,
    md5,
    quiet_env,
    real_run,
    stopped_cleanly,
    tracing_env,
)

checks = internals("checks")
measure = internals("measure")
world = internals("world")

SAME_LENGTH_PROBLEM = "stored request values (client_meta_json) differ by content hash"


def _sum_length(path):
    conn = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    try:
        return conn.execute("SELECT sum(length(client_meta_json)) FROM slots").fetchone()[0]
    finally:
        conn.close()


def _change_one_value(copy, new_value):
    """Replace one stored value in a copy; ``new_value`` gets the old value as its argument."""
    conn = sqlite3.connect(copy)
    try:
        old = conn.execute("SELECT client_meta_json FROM slots WHERE slot_id='live-ACTIVE'").fetchone()[0]
        conn.execute("UPDATE slots SET client_meta_json=? WHERE slot_id='live-ACTIVE'", (new_value(old),))
        conn.commit()
    finally:
        conn.close()


def _same_length(old):
    changed = old.replace('"ip": "192.0.2.99"', '"ip": "192.0.2.98"')
    assert changed != old and len(changed) == len(old)
    return changed


def _all_bodies(path):
    conn = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    try:
        return sorted(map(str, conn.execute("SELECT slot_id, client_meta_json FROM slots").fetchall()))
    finally:
        conn.close()


def _spy_on_fingerprints(monkeypatch):
    """Record (file the connection is open on, where clause) for every call of the content hash."""
    real = measure._body_fingerprint
    calls = []

    def spy(conn, where="1=1", params=()):
        calls.append((conn.execute("PRAGMA database_list").fetchone()[2], where))
        return real(conn, where, params)

    monkeypatch.setattr(measure, "_body_fingerprint", spy)
    return calls


class TestTheBackupIsComparedByContent:
    def _run(self, tmp_path, change, monkeypatch=None):
        fx = make_fixture(tmp_path)
        before, stmts = md5(fx.db), []
        source_sum = _sum_length(fx.db)
        seen = {}

        def backup(conn, dest):
            world._default_backup(conn, dest)
            _change_one_value(dest, change)
            seen["sum_copy"] = _sum_length(dest)

        report = real_run(fx, tracing_env(stmts, backup_fn=backup))
        return fx, report, before, stmts, source_sum, seen

    def test_a_backup_with_one_value_changed_but_the_same_length_stops_the_run(self, tmp_path):
        fx, report, before, stmts, source_sum, seen = self._run(tmp_path, _same_length)
        assert seen["sum_copy"] == source_sum, "the fixture must build the defect: equal total length"
        stopped_cleanly(fx, report, "verify-backup", before, stmts)
        message = assert_stop(report, "verify-backup", SAME_LENGTH_PROBLEM)
        assert "sum(length(client_meta_json)) differs" not in message and ".partial" in message

    def test_two_values_swapped_between_slots_are_caught_because_the_hash_includes_the_slot_id(self, tmp_path):
        fx = make_fixture(tmp_path)
        before, stmts = md5(fx.db), []
        source_sum = _sum_length(fx.db)
        a, b = "cold-000", "cold-001"
        seen = {}

        def swap_two_values(conn, dest):
            world._default_backup(conn, dest)
            copy = sqlite3.connect(dest)
            try:
                get = "SELECT client_meta_json FROM slots WHERE slot_id=?"
                va, vb = (copy.execute(get, (x,)).fetchone()[0] for x in (a, b))
                assert len(va) == len(vb) and va != vb, "the two values must have equal length and different text"
                bodies = sorted((r[0] for r in copy.execute("SELECT client_meta_json FROM slots")), key=str)
                put = "UPDATE slots SET client_meta_json=? WHERE slot_id=?"
                copy.execute(put, (vb, a))
                copy.execute(put, (va, b))
                copy.commit()
                seen["same_multiset"] = bodies == sorted(
                    (r[0] for r in copy.execute("SELECT client_meta_json FROM slots")), key=str)
                seen["swapped"] = copy.execute(get, (a,)).fetchone()[0] == vb
            finally:
                copy.close()
            seen["sum_copy"] = _sum_length(dest)

        report = real_run(fx, tracing_env(stmts, backup_fn=swap_two_values))
        assert report["status"] != "failed", report.get("failure")
        assert seen == {"same_multiset": True, "swapped": True, "sum_copy": source_sum}, seen
        stopped_cleanly(fx, report, "verify-backup", before, stmts)
        assert_stop(report, "verify-backup", SAME_LENGTH_PROBLEM)

    def test_a_value_of_a_different_length_is_caught_by_the_length_sum_and_the_hash_is_not_read(
            self, tmp_path, monkeypatch):
        calls = _spy_on_fingerprints(monkeypatch)
        fx, report, before, stmts, source_sum, seen = self._run(tmp_path, lambda old: old[:100])
        assert seen["sum_copy"] != source_sum
        stopped_cleanly(fx, report, "verify-backup", before, stmts)
        message = assert_stop(report, "verify-backup", "sum(length(client_meta_json)) differs")
        assert SAME_LENGTH_PROBLEM not in message
        assert [c for c in calls if c[0].endswith(".partial")] == [], "the cheap check failed, so no hash read"

    def test_a_faithful_backup_is_hashed_once_and_the_run_completes(self, tmp_path, monkeypatch):
        calls = _spy_on_fingerprints(monkeypatch)
        fx = make_fixture(tmp_path)
        report = real_run(fx)
        assert report["status"] == "completed", report.get("stop")
        assert len([c for c in calls if c[0].endswith(".partial")]) == 1

    def test_the_rebuilt_file_is_not_compared_with_the_source_by_content(self, tmp_path, monkeypatch):
        calls = _spy_on_fingerprints(monkeypatch)
        fx = make_fixture(tmp_path)
        source_bodies = _all_bodies(fx.db)
        report = real_run(fx)
        assert report["status"] == "completed", report.get("stop")
        assert _all_bodies(fx.db) != source_bodies, "every COLD body differs on purpose"
        rebuilt = [c for c in calls if ".rebuilt-" in c[0]]
        assert len(rebuilt) == 1, "only the existing check against the database just before the rebuild reads it"

    def test_the_source_fingerprint_is_taken_once_before_the_backup_and_only_on_a_real_run(self, tmp_path,
                                                                                            monkeypatch):
        calls = _spy_on_fingerprints(monkeypatch)
        fx = make_fixture(tmp_path)
        conn = sqlite3.connect(fx.db)
        expected = measure._body_fingerprint(conn)
        conn.close()
        calls.clear()
        dry = cleanup.run(cfg_for(fx), quiet_env())
        assert calls == [] and "body_fingerprint" not in dry["before"]
        report = real_run(fx)
        on_the_copy = [i for i, c in enumerate(calls) if c[0].endswith(".partial")]
        assert on_the_copy, "the backup was never hashed"
        first_backup_read = on_the_copy[0]
        whole_table_before = [c for c in calls[:first_backup_read] if c[1] == "1=1"]
        assert len(whole_table_before) == 1
        assert report["before"]["body_fingerprint"] == expected

    def test_the_hash_is_read_only_when_the_expectation_carries_one(self, tmp_path):
        fx = make_fixture(tmp_path)
        conn = sqlite3.connect(fx.db)
        try:
            expect = measure._snapshot(conn)
            full = {**expect, "body_fingerprint": measure._body_fingerprint(conn)}
        finally:
            conn.close()
        copy = Path(tmp_path) / "copy.sqlite"
        src = sqlite3.connect(fx.db)
        world._default_backup(src, copy)
        src.close()
        _change_one_value(copy, _same_length)
        assert checks._verify_copy(copy, full, quiet_env(), "copy")["problems"] == [SAME_LENGTH_PROBLEM]
        assert checks._verify_copy(copy, expect, quiet_env(), "copy")["ok"] is True
