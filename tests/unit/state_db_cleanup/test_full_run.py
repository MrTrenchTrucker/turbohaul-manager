"""A full run: the result, the report, and that only the stored request value is written."""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import json
import os
import re
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from state_db_cleanup.helpers import (
    IDENTITY,
    NOW,
    RECENT,
    cleanup,
    make_fixture,
    projection,
    quiet_env,
    read_all,
    real_run,
    sub,
    tracing_env,
)


class TestFullRun:
    @pytest.fixture
    def done(self, tmp_path):
        fx = make_fixture(tmp_path)
        before = read_all(fx.db)
        size_before = fx.db.stat().st_size
        report = real_run(fx)
        return fx, before, size_before, report

    def test_the_run_completes(self, done):
        fx, before, size_before, report = done
        assert report["status"] == "completed", report.get("stop")
        assert report["allowlist"] == list(projection.IDENTITY_KEYS), "the report must name the keys that were kept"
        assert [s["step"] for s in report["steps"]] == [
            "preflight", "lock", "measure", "plan", "space", "backup", "verify-backup", "strip",
            "check-strip", "rebuild", "verify-new", "swap"], "the steps must run in this gated order"

    def test_cold_values_are_reduced_to_identity_keys_and_keep_their_identity_values(self, done):
        fx, before, size_before, report = done
        after = read_all(fx.db)
        for slot_id in fx.cold_ids:
            old = json.loads(before["bodies"][slot_id])
            new = json.loads(after["bodies"][slot_id])
            assert set(new) <= IDENTITY, f"{slot_id} still holds {set(new) - IDENTITY}"
            assert new == {k: v for k, v in old.items() if k in IDENTITY}

    def test_values_that_are_not_json_objects_are_left_byte_identical(self, done):
        fx, before, size_before, report = done
        after = read_all(fx.db)
        for slot_id, value in fx.invalid.items():
            assert after["bodies"][slot_id] == value
        assert report["strip"]["invalid"] == 4 and report["strip"]["missing"] == 1

    def test_rows_that_are_not_cold_are_byte_identical(self, done):
        fx, before, size_before, report = done
        after = read_all(fx.db)
        for slot_id, value in fx.other.items():
            assert after["bodies"][slot_id] == value
        assert report["strip"]["unchanged"] == 1, "the identity-only COLD row counts as unchanged"

    def test_no_row_is_lost_and_no_other_column_or_table_changes(self, done):
        fx, before, size_before, report = done
        after = read_all(fx.db)
        assert after["slots_other"] == before["slots_other"]
        assert after["audit"] == before["audit"]
        assert after["pulls"] == before["pulls"]
        assert after["seq"] == before["seq"], "autoincrement counters must survive the rebuild"
        assert after["schema"] == before["schema"]
        assert set(after["bodies"]) == set(before["bodies"])

    def test_the_report_carries_the_before_and_after_figures_and_they_are_true(self, done):
        fx, before, size_before, report = done
        conn = sqlite3.connect(f"file:{fx.db}?immutable=1", uri=True)
        real_sum = conn.execute("SELECT sum(length(client_meta_json)) FROM slots").fetchone()[0]
        real_counts = {name: conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
                       for name in report["after"]["row_counts"]}
        conn.close()
        assert report["after"]["sum_body_length"] == real_sum < report["before"]["sum_body_length"]
        assert report["after"]["row_counts"] == real_counts == report["before"]["row_counts"]
        assert report["sizes_before"]["db"] == size_before
        assert report["strip"]["rows"] == report["plan"]["rows"]
        assert report["strip"]["stripped"] == report["plan"]["stripped"]
        sample = report["verify_new"]["sample"]
        assert sample and all(set(r["keys"]) <= IDENTITY for r in sample)

    def test_the_file_shrinks_and_passes_the_integrity_check(self, done):
        fx, before, size_before, report = done
        assert fx.db.stat().st_size < size_before / 3
        assert read_all(fx.db)["integrity"] == [("ok",)]
        assert report["sizes_after"]["db"] == fx.db.stat().st_size
        assert not (fx.data / "state.sqlite-wal").exists() or (fx.data / "state.sqlite-wal").stat().st_size == 0

    def test_the_backup_is_a_complete_copy_of_the_original(self, done):
        fx, before, size_before, report = done
        backups = [p for p in fx.backups.iterdir() if ".pre-cleanup-" in p.name and p.suffix != ".json"]
        assert len(backups) == 1 and not backups[0].name.endswith(".partial")
        assert report["backup"]["path"] == str(backups[0]) and report["backup"]["ok"] is True
        copy = read_all(backups[0])
        assert copy["bodies"] == before["bodies"], "the backup must hold every original value"
        assert copy["audit"] == before["audit"] and copy["slots_other"] == before["slots_other"]

    def test_the_report_is_written_beside_the_backup_not_in_the_database_directory(self, tmp_path):
        fx = make_fixture(tmp_path)
        code = cleanup.main(["--db", str(fx.db), "--backup-dir", str(fx.backups), "--margin-gib", "0",
                             "--sleep-ms", "0", "--execute"], quiet_env())
        assert code == 0
        reports = [p for p in fx.backups.iterdir() if p.name.endswith(".json")]
        assert len(reports) == 1
        data = json.loads(reports[0].read_text())
        assert data["status"] == "completed" and "sizes_before" in data and "sizes_after" in data
        assert data["before"]["row_counts"]["slots"] == len(fx.cold_ids) + len(fx.invalid) + len(fx.other)
        assert data["after"]["row_counts"] == data["before"]["row_counts"]
        assert data["after"]["sum_body_length"] < data["before"]["sum_body_length"]
        assert not any(p.name.endswith(".json") for p in fx.data.iterdir())

    def test_a_second_run_changes_nothing(self, tmp_path):
        fx = make_fixture(tmp_path)
        real_run(fx)
        once = read_all(fx.db)
        def later():
            return NOW + timedelta(seconds=5)

        report = real_run(fx, quiet_env(now=later))
        assert report["status"] == "completed"
        assert report["strip"]["stripped"] == 0
        assert read_all(fx.db) == once

    def test_min_age_hours_leaves_recent_rows_alone(self, tmp_path):
        fx = make_fixture(tmp_path, n_cold=4)
        conn = sqlite3.connect(fx.db)
        conn.execute("UPDATE slots SET updated_at=? WHERE slot_id IN ('cold-000','cold-001')", (RECENT,))
        conn.commit()
        conn.close()
        before = read_all(fx.db)
        report = real_run(fx, min_age_hours=1.0)
        assert report["status"] == "completed", report.get("stop")
        after = read_all(fx.db)
        for young in ("cold-000", "cold-001"):
            assert after["bodies"][young] == before["bodies"][young]
        for old in ("cold-002", "cold-003"):
            assert set(json.loads(after["bodies"][old])) <= IDENTITY
        assert report["strip"]["rows"] == report["plan"]["rows"] < 10

    def test_no_swap_keeps_the_database_in_place_and_leaves_the_verified_new_file(self, tmp_path):
        fx = make_fixture(tmp_path)
        inode = fx.db.stat().st_ino
        report = real_run(fx, swap=False)
        assert report["status"].startswith("completed without swap")
        assert fx.db.stat().st_ino == inode
        rebuilt = Path(report["new_file"])
        assert rebuilt.exists() and rebuilt.stat().st_size < fx.db.stat().st_size
        assert read_all(rebuilt)["bodies"] == read_all(fx.db)["bodies"]

    def test_the_new_file_keeps_the_databases_permissions(self, tmp_path):
        fx = make_fixture(tmp_path)
        os.chmod(fx.db, 0o640)
        real_run(fx)
        assert (fx.db.stat().st_mode & 0o777) == 0o640

    def test_work_files_are_cleaned_up_and_the_temp_directory_variable_is_restored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_TMPDIR", str(tmp_path))
        fx = make_fixture(tmp_path)
        real_run(fx)
        assert sorted(os.listdir(fx.data)) == ["state.sqlite"]
        assert os.environ["SQLITE_TMPDIR"] == str(tmp_path)
        monkeypatch.delenv("SQLITE_TMPDIR")
        fx2 = make_fixture(sub(tmp_path, "second"))
        real_run(fx2)
        assert "SQLITE_TMPDIR" not in os.environ


class TestOnlyTheStoredValueIsWritten:
    def test_every_statement_is_a_read_a_pragma_a_rebuild_or_a_single_column_update(self, tmp_path):
        fx = make_fixture(tmp_path, n_cold=6, pad=2_000)
        stmts = []
        real_run(fx, tracing_env(stmts))
        assert stmts, "statement capture saw nothing"
        updates = [s for s in stmts if s.lstrip().upper().startswith("UPDATE")]
        assert updates, "the run never updated anything"
        for s in updates:
            assert re.match(r"^UPDATE slots SET client_meta_json = '.*' WHERE rowid = \d+$", s, re.S), s[:200]
        forbidden = re.compile(r"^\s*(DELETE|DROP|ALTER|INSERT|CREATE|REPLACE|ATTACH)\b", re.I)
        offenders = [s[:120] for s in stmts if forbidden.match(s)]
        assert not offenders, f"the tool must never delete rows or change the schema: {offenders}"
        assert any(s.lstrip().upper().startswith("VACUUM INTO") for s in stmts)
        assert not any(re.match(r"^\s*VACUUM\s*;?\s*$", s, re.I) for s in stmts), "never an in-place VACUUM"

    def test_updates_reach_only_cold_rows(self, tmp_path):
        fx = make_fixture(tmp_path, n_cold=5, pad=2_000)
        conn = sqlite3.connect(fx.db)
        non_cold = {r[0] for r in conn.execute("SELECT rowid FROM slots WHERE state != 'COLD'")}
        conn.close()
        stmts = []
        real_run(fx, tracing_env(stmts))
        touched = {int(re.search(r"rowid = (\d+)$", s).group(1)) for s in stmts if s.startswith("UPDATE slots")}
        assert touched and not (touched & non_cold)
