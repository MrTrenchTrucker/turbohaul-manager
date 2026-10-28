"""A run that does not finish leaves a trail: a report on every real-run path, and a marker that blocks a re-run."""
import json
import os
import sqlite3
import stat
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from state_db_cleanup.helpers import (
    NOW,
    assert_stop,
    cfg_for,
    cleanup,
    internals,
    make_fixture,
    md5,
    quiet_env,
    real_run,
    report_of,
    sub,
)

sequence = internals("sequence")
STAMP = NOW.strftime("%Y%m%dT%H%M%SZ")


def _mode(path):
    assert Path(path).exists(), f"{path} was not written"
    return stat.S_IMODE(Path(path).stat().st_mode)


def _marker(fx):
    return fx.data / "state.sqlite.cleanup-unfinished"


def _report_file(fx):
    return fx.backups / f"state.sqlite.cleanup-report-{STAMP}.json"


def _read(path):
    assert Path(path).exists(), f"{path} was not written"
    return json.loads(Path(path).read_text())


def _main(fx, env, *extra, backups=None):
    return cleanup.main(["--db", str(fx.db), "--backup-dir", str(backups or fx.backups), "--margin-gib", "0",
                         "--sleep-ms", "0", "--execute", *extra], env)


def _disk_full_on_third_batch():
    calls = []

    def sleep(_):
        calls.append(1)
        if len(calls) == 3:
            raise OSError(28, "No space left on device")

    return sleep


def _failed_run(tmp_path, capsys):
    """A real run that dies with a full disk part way through the strip; returns the fixture and the exit code."""
    fx = make_fixture(tmp_path)
    code = _main(fx, quiet_env(sleep=_disk_full_on_third_batch()), "--batch", "4")
    capsys.readouterr()
    return fx, code


class TestAFailedRunLeavesAReport:
    def test_a_full_disk_in_the_strip_ends_with_a_failed_report_and_exit_one(self, tmp_path, capsys):
        fx = make_fixture(tmp_path)
        code = _main(fx, quiet_env(sleep=_disk_full_on_third_batch()), "--batch", "4")
        assert code == 1
        printed = json.loads(capsys.readouterr().out)
        written = _read(_report_file(fx))
        assert printed == written
        assert written["status"] == "failed" and written["db_modified"] is True
        assert written["failure"] == {"step": "strip", "type": "OSError",
                                      "message": "[Errno 28] No space left on device"}
        assert written["steps"][-1] == {"step": "strip", "ok": False, "seconds": written["steps"][-1]["seconds"]}
        assert _mode(_report_file(fx)) == 0o600
        backup = Path(written["backup"]["path"])
        assert backup.exists() and backup.parent == fx.backups and written["backup_planned"] == str(backup)

    def test_run_returns_the_failed_report_instead_of_raising(self, tmp_path):
        fx = make_fixture(tmp_path)
        report = report_of(lambda: real_run(fx, quiet_env(sleep=_disk_full_on_third_batch()), batch=4))
        assert report["status"] == "failed" and report["failure"]["type"] == "OSError"

    def test_a_sqlite_error_in_the_rebuild_is_recorded_at_the_rebuild(self, tmp_path):
        fx = make_fixture(tmp_path)

        def vacuum(conn, dest):
            raise sqlite3.OperationalError("disk I/O error")

        report = report_of(lambda: real_run(fx, quiet_env(vacuum_fn=vacuum)))
        assert report["status"] == "failed"
        assert report["failure"]["step"] == "rebuild" and report["failure"]["type"] == "OperationalError"

    def test_any_other_error_in_a_gate_is_recorded_too(self, tmp_path, monkeypatch):
        fx = make_fixture(tmp_path)
        monkeypatch.setattr(sequence, "_space_gate", lambda *a, **k: {}["missing"])
        report = report_of(lambda: real_run(fx))
        assert report["status"] == "failed"
        assert report["failure"]["step"] == "space" and report["failure"]["type"] == "KeyError"

    def test_a_completed_run_writes_a_private_report_naming_the_backup(self, tmp_path, capsys):
        fx = make_fixture(tmp_path)
        assert _main(fx, quiet_env()) == 0
        written = _read(_report_file(fx))
        assert written["status"] == "completed" and _mode(_report_file(fx)) == 0o600
        assert written["backup"]["path"] == written["backup_planned"] and Path(written["backup"]["path"]).exists()

    def test_a_run_stopped_after_the_strip_began_writes_a_report_naming_the_backup(self, tmp_path):
        fx = make_fixture(tmp_path)
        junk = quiet_env(before_swap=lambda db: Path(str(db) + "-wal").write_bytes(b"junk" * 100))
        assert _main(fx, junk) == 2
        written = _read(_report_file(fx))
        assert_stop(written, "swap")
        assert Path(written["backup"]["path"]).exists() and _mode(_report_file(fx)) == 0o600

    def test_a_run_stopped_by_a_backup_that_does_not_verify_names_the_planned_backup(self, tmp_path):
        fx = make_fixture(tmp_path)
        broken = quiet_env(backup_fn=lambda conn, dest: Path(dest).write_bytes(b""))
        assert _main(fx, broken) == 2
        written = _read(_report_file(fx))
        assert_stop(written, "verify-backup")
        assert str(written.get("backup_planned")).startswith(str(fx.backups)) and "backup" not in written

    def test_a_dry_run_writes_no_report_and_no_marker(self, tmp_path):
        fx = make_fixture(tmp_path)
        code = cleanup.main(["--db", str(fx.db), "--backup-dir", str(fx.backups), "--margin-gib", "0"], quiet_env())
        assert code == 0 and os.listdir(fx.backups) == [] and not _marker(fx).exists()


class TestTheUnfinishedRunMarker:
    def test_a_failed_run_leaves_a_private_marker_naming_the_report_and_the_backup(self, tmp_path, capsys):
        fx, code = _failed_run(tmp_path, capsys)
        assert code == 1 and _mode(_marker(fx)) == 0o600
        data = _read(_marker(fx))
        written = _read(_report_file(fx))
        assert data["report_path"] == str(_report_file(fx)) == written["report_path"]
        assert data["backup_planned"] == written["backup"]["path"] and data["db"] == str(fx.db)
        assert data["stamp"] == STAMP and data["pid"] == os.getpid() and data["created"]
        assert written["marker"] == str(_marker(fx))

    def test_a_second_real_run_is_refused_even_with_another_backup_directory(self, tmp_path, capsys):
        fx, _ = _failed_run(tmp_path, capsys)
        other = sub(tmp_path, "other_backups")
        data_md5, backups_before = md5(fx.db), sorted(os.listdir(fx.backups))
        report = real_run(fx, quiet_env(now=lambda: NOW + timedelta(seconds=30)), backup_dir=other)
        message = assert_stop(report, "preflight", str(_marker(fx)))
        assert str(_report_file(fx)) in message and "state.sqlite.pre-cleanup-" in message
        assert md5(fx.db) == data_md5 and os.listdir(other) == []
        assert sorted(os.listdir(fx.backups)) == backups_before

    def test_deleting_the_marker_lets_the_run_go_on(self, tmp_path, capsys):
        fx, _ = _failed_run(tmp_path, capsys)
        assert _marker(fx).exists(), "the failed run left no marker to delete"
        _marker(fx).unlink()
        report = real_run(fx, quiet_env(now=lambda: NOW + timedelta(seconds=30)))
        assert report["status"] == "completed", report.get("stop")
        assert not _marker(fx).exists()

    def test_a_marker_left_by_a_killed_process_refuses_a_real_run_and_not_a_dry_run(self, tmp_path):
        fx = make_fixture(tmp_path)
        _marker(fx).write_text(json.dumps({"report_path": "R.json", "backup_planned": "B.pre"}))
        before = md5(fx.db)
        report = real_run(fx)
        message = assert_stop(report, "preflight", str(_marker(fx)))
        assert "R.json" in message and "B.pre" in message and md5(fx.db) == before
        assert os.listdir(fx.backups) == []
        assert cleanup.run(cfg_for(fx), quiet_env())["status"] == "dry-run complete"

    def test_a_marker_that_cannot_be_read_still_refuses_and_is_named(self, tmp_path):
        fx = make_fixture(tmp_path)
        _marker(fx).write_text("{ not json")
        assert_stop(real_run(fx), "preflight", str(_marker(fx)))

    def test_a_completed_run_leaves_no_marker(self, tmp_path):
        fx = make_fixture(tmp_path)
        assert real_run(fx)["status"] == "completed" and not _marker(fx).exists()

    def test_a_run_that_completes_without_a_swap_leaves_no_marker(self, tmp_path):
        fx = make_fixture(tmp_path)
        assert real_run(fx, swap=False)["status"].startswith("completed") and not _marker(fx).exists()

    def test_a_run_stopped_before_any_write_leaves_no_marker_and_may_be_repeated(self, tmp_path):
        fx = make_fixture(tmp_path)
        broken = quiet_env(backup_fn=lambda conn, dest: Path(dest).write_bytes(b""))
        assert_stop(real_run(fx, broken), "verify-backup")
        assert not _marker(fx).exists()
        again = real_run(fx, quiet_env(now=lambda: NOW + timedelta(seconds=30)))
        assert again["status"] == "completed", again.get("stop")

    def test_a_failed_space_gate_leaves_no_marker(self, tmp_path):
        fx = make_fixture(tmp_path)
        report = real_run(fx, quiet_env(statvfs=lambda p: SimpleNamespace(f_bavail=10, f_frsize=1)))
        assert_stop(report, "space")
        assert not _marker(fx).exists() and os.listdir(fx.backups) == []

    def test_a_marker_that_cannot_be_created_stops_before_the_backup(self, tmp_path, monkeypatch):
        fx = make_fixture(tmp_path)
        real_open = os.open

        def open_refusing_the_marker(path, *args, **kwargs):
            if str(path).endswith(".cleanup-unfinished"):
                raise OSError(28, "No space left on device")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", open_refusing_the_marker)
        before = md5(fx.db)
        report = report_of(lambda: real_run(fx))
        message = assert_stop(report, "marker", "nothing was written")
        assert str(_marker(fx)) in message and "marker" not in report
        assert os.listdir(fx.backups) == [] and report["db_modified"] is False and md5(fx.db) == before


class TestAnInterruptLeavesTheSameTrail:
    def test_a_keyboard_interrupt_writes_the_report_keeps_the_marker_and_goes_on_raising(self, tmp_path):
        fx = make_fixture(tmp_path)

        def interrupt(_):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            real_run(fx, quiet_env(sleep=interrupt), batch=4)
        written = _read(_report_file(fx))
        assert written["status"] == "failed" and written["failure"]["type"] == "KeyboardInterrupt"
        assert written["failure"]["step"] == "strip" and _mode(_report_file(fx)) == 0o600
        assert _marker(fx).exists() and Path(written["backup"]["path"]).exists()


class TestWhenTheReportCannotBeWritten:
    def test_the_report_is_on_stdout_and_the_backup_is_named_on_stderr(self, tmp_path, capsys):
        fx = make_fixture(tmp_path)
        code = report_of(lambda: _main(fx, quiet_env(), "--report", str(tmp_path / "no_such_dir" / "r.json")))
        out, err = capsys.readouterr()
        assert code == 0 and json.loads(out)["status"] == "completed"
        backup = json.loads(out)["backup"]["path"]
        assert err.startswith("REPORT NOT WRITTEN (") and f"backup: {backup}" in err
        assert err.rstrip().endswith("unfinished-run marker: none")

    def test_after_a_failure_the_stderr_line_names_the_marker_too(self, tmp_path, capsys):
        fx = make_fixture(tmp_path)
        code = report_of(lambda: _main(fx, quiet_env(sleep=_disk_full_on_third_batch()), "--batch", "4",
                                       "--report", str(tmp_path / "no_such_dir" / "r.json")))
        out, err = capsys.readouterr()
        assert code == 1 and json.loads(out)["status"] == "failed"
        assert "REPORT NOT WRITTEN (" in err and "state.sqlite.pre-cleanup-" in err
        assert f"unfinished-run marker: {_marker(fx)}" in err
