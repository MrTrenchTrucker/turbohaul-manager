"""The backup must verify before anything is written to the database."""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import os
import sqlite3
from pathlib import Path

from state_db_cleanup.helpers import (
    assert_stop,
    internals,
    make_fixture,
    md5,
    quiet_env,
    real_run,
    stopped_cleanly,
    tracing_env,
)

world = internals("world")


class TestBackupMustVerifyBeforeAnyWrite:
    def _run_with_backup(self, tmp_path, make_backup):
        fx = make_fixture(tmp_path)
        before = md5(fx.db)
        stmts = []

        def backup(conn, dest):
            world._default_backup(conn, dest)
            make_backup(Path(dest), fx)

        report = real_run(fx, tracing_env(stmts, backup_fn=backup))
        return fx, report, before, stmts

    def test_a_zero_byte_backup_stops_the_run(self, tmp_path):
        fx, report, before, stmts = self._run_with_backup(tmp_path, lambda p, fx: p.write_bytes(b""))
        stopped_cleanly(fx, report, "verify-backup", before, stmts)

    def test_a_truncated_backup_stops_the_run(self, tmp_path):
        def truncate(p, fx):
            with open(p, "r+b") as f:
                f.truncate(p.stat().st_size // 2)
        fx, report, before, stmts = self._run_with_backup(tmp_path, truncate)
        stopped_cleanly(fx, report, "verify-backup", before, stmts)

    def test_a_backup_with_a_missing_row_stops_the_run(self, tmp_path):
        def drop_row(p, fx):
            c = sqlite3.connect(p)
            c.execute("DELETE FROM audit_events WHERE event_id = (SELECT max(event_id) FROM audit_events)")
            c.commit()
            c.close()
        fx, report, before, stmts = self._run_with_backup(tmp_path, drop_row)
        stopped_cleanly(fx, report, "verify-backup", before, stmts)
        assert_stop(report, "verify-backup", "row_counts differs")

    def test_a_backup_whose_stored_values_differ_in_length_stops_the_run(self, tmp_path):
        def shorten(p, fx):
            c = sqlite3.connect(p)
            c.execute("UPDATE slots SET client_meta_json = substr(client_meta_json, 1, 100) "
                      "WHERE slot_id='live-ACTIVE'")
            c.commit()
            c.close()
        fx, report, before, stmts = self._run_with_backup(tmp_path, shorten)
        stopped_cleanly(fx, report, "verify-backup", before, stmts)
        assert_stop(report, "verify-backup", "sum(length(client_meta_json)) differs")

    def test_a_backup_whose_other_columns_differ_stops_the_run(self, tmp_path):
        def alter(p, fx):
            c = sqlite3.connect(p)
            c.execute("UPDATE slots SET port = 1 WHERE slot_id = 'cold-000'")
            c.commit()
            c.close()
        fx, report, before, stmts = self._run_with_backup(tmp_path, alter)
        stopped_cleanly(fx, report, "verify-backup", before, stmts)
        assert_stop(report, "verify-backup", "fingerprints differs")

    def test_a_corrupt_backup_stops_the_run(self, tmp_path):
        def corrupt(p, fx):
            with open(p, "r+b") as f:
                f.seek(p.stat().st_size // 2)
                f.write(os.urandom(8192))
        fx, report, before, stmts = self._run_with_backup(tmp_path, corrupt)
        stopped_cleanly(fx, report, "verify-backup", before, stmts)

    def test_a_faithful_backup_lets_the_run_proceed(self, tmp_path):
        fx, report, before, stmts = self._run_with_backup(tmp_path, lambda p, fx: None)
        assert report["status"] == "completed", report.get("stop")

    def test_the_backup_gets_its_final_name_only_after_it_verified(self, tmp_path):
        seen = []

        def spy(conn, dest):
            world._default_backup(conn, dest)
            seen.append(sorted(p.name for p in Path(dest).parent.iterdir()))

        fx = make_fixture(tmp_path)
        real_run(fx, quiet_env(backup_fn=spy))
        assert all(n.endswith(".partial") for n in seen[0]), "while it is unverified the backup must carry .partial"
        assert not any(n.endswith(".partial") for n in os.listdir(fx.backups))

    def test_an_existing_backup_is_never_overwritten(self, tmp_path):
        fx = make_fixture(tmp_path)
        existing = fx.backups / "state.sqlite.pre-cleanup-20261024T120000Z"
        existing.write_bytes(b"precious")
        before = md5(fx.db)
        stmts = []
        report = real_run(fx, tracing_env(stmts))
        stopped_cleanly(fx, report, "backup", before, stmts, backups_ok=True)
        assert existing.read_bytes() == b"precious"
