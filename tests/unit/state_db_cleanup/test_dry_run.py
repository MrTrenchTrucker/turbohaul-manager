"""The dry run: it reports what the run would do and writes nothing."""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import os
import sqlite3
from types import SimpleNamespace

import pytest

from state_db_cleanup.helpers import (
    IDENTITY,
    cfg_for,
    cleanup,
    internals,
    make_fixture,
    md5,
    quiet_env,
)

connect = internals("connect")


class TestDryRun:
    def test_it_writes_nothing(self, tmp_path):
        fx = make_fixture(tmp_path)
        before_md5 = md5(fx.db)
        before_files = set(os.listdir(fx.data))
        report = cleanup.run(cfg_for(fx), quiet_env())
        assert report["status"] == "dry-run complete", report.get("stop")
        assert md5(fx.db) == before_md5, "a dry run must leave the database file byte-identical"
        extra = set(os.listdir(fx.data)) - before_files
        assert extra <= {"state.sqlite-wal", "state.sqlite-shm"}, f"dry run left unexpected files: {extra}"
        wal = fx.data / "state.sqlite-wal"
        assert not wal.exists() or wal.stat().st_size == 0
        assert os.listdir(fx.backups) == [], "a dry run must not create a backup"
        assert report["db_modified"] is False

    def test_its_connection_cannot_write(self, tmp_path):
        fx = make_fixture(tmp_path)
        conn = connect._open_readonly(fx.db)
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("UPDATE slots SET client_meta_json = '{}'")
        finally:
            conn.close()

    def test_without_the_execute_flag_the_command_line_writes_nothing(self, tmp_path):
        fx = make_fixture(tmp_path)
        before_md5 = md5(fx.db)
        code = cleanup.main(["--db", str(fx.db), "--backup-dir", str(fx.backups), "--margin-gib", "0"],
                            quiet_env())
        assert code == 0
        assert md5(fx.db) == before_md5
        assert os.listdir(fx.backups) == []

    def test_its_counts_match_an_independent_baseline(self, tmp_path):
        fx = make_fixture(tmp_path)
        conn = sqlite3.connect(f"file:{fx.db}?immutable=1", uri=True)
        allow = ",".join(f"'{k}'" for k in sorted(IDENTITY))
        base = {
            "rows": conn.execute("SELECT count(*) FROM slots WHERE state='COLD'").fetchone()[0],
            "with_extra_keys": conn.execute(
                "SELECT count(*) FROM slots WHERE state='COLD' AND json_valid(client_meta_json) "
                "AND json_type(client_meta_json)='object' AND EXISTS (SELECT 1 FROM json_each(client_meta_json) "
                f"WHERE key NOT IN ({allow}))").fetchone()[0],
            "not_object": conn.execute(
                "SELECT count(*) FROM slots WHERE state='COLD' AND client_meta_json IS NOT NULL AND "
                "(NOT json_valid(client_meta_json) OR json_type(client_meta_json) != 'object')").fetchone()[0],
            "nulls": conn.execute(
                "SELECT count(*) FROM slots WHERE state='COLD' AND client_meta_json IS NULL").fetchone()[0],
            "chars": conn.execute(
                "SELECT sum(length(client_meta_json)) FROM slots WHERE state='COLD'").fetchone()[0],
        }
        conn.close()
        plan = cleanup.run(cfg_for(fx), quiet_env())["plan"]
        assert plan["rows"] == base["rows"]
        assert plan["stripped"] == base["with_extra_keys"]
        assert plan["invalid"] == base["not_object"]
        assert plan["missing"] == base["nulls"]
        assert plan["unchanged"] == base["rows"] - base["with_extra_keys"] - base["not_object"] - base["nulls"]
        assert plan["chars_before"] == base["chars"], "the character total must agree with SQLite's length()"

    def test_it_reports_the_space_figures_it_measured(self, tmp_path):
        fx = make_fixture(tmp_path)
        space = cleanup.run(cfg_for(fx), quiet_env())["space"]
        assert space["ok"] is True
        vol = space["volumes"][0]
        assert vol["items"]["backup"] > 0 and vol["items"]["new_file"] > 0
        assert vol["required"] == sum(vol["items"].values()) + vol["margin"]
        assert vol["free"] > 0

    def test_it_stops_with_status_two_when_the_space_gate_would_fail(self, tmp_path):
        fx = make_fixture(tmp_path)
        env = quiet_env(statvfs=lambda p: SimpleNamespace(f_bavail=1, f_frsize=1))
        code = cleanup.main(["--db", str(fx.db), "--backup-dir", str(fx.backups), "--margin-gib", "0"], env)
        assert code == 2
