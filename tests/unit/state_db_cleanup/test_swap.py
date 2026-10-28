"""The swap of the verified file over the database."""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import os
import sqlite3
from pathlib import Path

from state_db_cleanup.helpers import (
    assert_stop,
    internals,
    make_fixture,
    quiet_env,
    real_run,
    report_of,
)

measure = internals("measure")
swap = internals("swap")


class TestSwap:
    def test_a_non_empty_write_ahead_log_blocks_the_swap(self, tmp_path):
        fx = make_fixture(tmp_path)
        inode = fx.db.stat().st_ino
        report = real_run(fx, quiet_env(before_swap=lambda db: Path(str(db) + "-wal").write_bytes(b"junk" * 100)))
        assert_stop(report, "swap")
        assert fx.db.stat().st_ino == inode
        assert Path(report["new_file"]).exists()

    def test_empty_leftover_wal_and_shm_files_are_removed_before_the_swap(self, tmp_path):
        fx = make_fixture(tmp_path)

        def leave_files(db):
            Path(str(db) + "-wal").write_bytes(b"")
            Path(str(db) + "-shm").write_bytes(b"\0" * 32768)

        report = real_run(fx, quiet_env(before_swap=leave_files))
        assert report["status"] == "completed", report.get("stop")
        assert sorted(os.listdir(fx.data)) == ["state.sqlite"], "stale -wal/-shm must not survive next to the new file"

    def test_a_new_file_that_cannot_take_the_databases_owner_blocks_the_swap(self, tmp_path, monkeypatch):
        fx = make_fixture(tmp_path)
        inode = fx.db.stat().st_ino
        monkeypatch.setattr(swap.os, "chown", lambda *a, **k: (_ for _ in ()).throw(PermissionError()))
        monkeypatch.setattr(swap, "_owner", lambda p: (4242, 4242) if "rebuilt" in str(p) else (1, 1))
        report = real_run(fx)
        assert_stop(report, "swap", "owner")
        assert fx.db.stat().st_ino == inode

    def test_a_swapped_database_that_fails_its_check_is_reported_as_a_stop(self, tmp_path, monkeypatch):
        fx = make_fixture(tmp_path)
        real = measure._integrity
        calls = []

        def third_fails(conn):
            calls.append(1)
            return (False, ["damaged"]) if len(calls) == 3 else real(conn)

        monkeypatch.setattr(measure, "_integrity", third_fails)
        report = real_run(fx)
        assert len(calls) == 3, "the backup, the new file and the swapped database must each be checked"
        assert_stop(report, "swap", "restore from the backup")

    def test_a_database_file_that_cannot_be_replaced_is_a_stop_naming_the_verified_file(self, tmp_path, monkeypatch):
        fx = make_fixture(tmp_path)
        inode = fx.db.stat().st_ino

        real_replace = os.replace

        def busy_for_the_database_only(src, dst):
            # the backup step also renames a file; only the swap onto the database is refused
            if str(dst) == str(fx.db):
                raise OSError(16, "Device or resource busy")
            return real_replace(src, dst)

        monkeypatch.setattr(swap.os, "replace", busy_for_the_database_only)
        report = report_of(lambda: real_run(fx))
        message = assert_stop(report, "swap", "could not replace the database file")
        assert report["new_file"] in message and Path(report["new_file"]).exists()
        assert fx.db.stat().st_ino == inode
        kept = [p for p in fx.backups.iterdir()
                if ".pre-cleanup-" in p.name and not p.name.endswith((".partial", ".json"))]
        assert kept, "the verified backup must still be there"

    def test_the_swap_replaces_the_database_with_the_verified_file(self, tmp_path):
        fx = make_fixture(tmp_path)
        inode = fx.db.stat().st_ino
        report = real_run(fx)
        assert report["status"] == "completed"
        assert fx.db.stat().st_ino != inode
        conn = sqlite3.connect(fx.db)
        try:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("INSERT INTO audit_events (slot_id, event_type, occurred_at) VALUES ('x','e','now')")
            conn.commit()
        finally:
            conn.close()
