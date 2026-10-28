"""The backup is a copy of the whole database, so only its owner may read it; and the backup directory must be safe."""
import os
import sqlite3
import stat
from pathlib import Path

import pytest

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
    sub,
    tracing_env,
)

backup_module = internals("backup")
world = internals("world")
_GROUP_WRITABLE = 0o775


def _mode(path):
    return stat.S_IMODE(Path(path).stat().st_mode)


class _Umask:
    """Set the process umask for one test and restore it afterwards."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = os.umask(self.value)

    def __exit__(self, *exc):
        os.umask(self.old)


def _backup_file(fx):
    found = [p for p in fx.backups.iterdir() if ".pre-cleanup-" in p.name and not p.name.endswith(".json")]
    assert len(found) == 1, found
    return found[0]


class TestTheBackupIsOwnerOnly:
    @pytest.mark.parametrize("umask", [0o022, 0o000])
    def test_the_final_backup_is_mode_0600_whatever_the_umask(self, tmp_path, umask):
        fx = make_fixture(tmp_path)
        with _Umask(umask):
            report = real_run(fx)
        assert report["status"] == "completed", report.get("stop")
        assert _mode(_backup_file(fx)) == 0o600
        assert report["backup"]["mode"] == "0600"

    def test_the_partial_file_is_already_0600_before_sqlite_writes_into_it(self, tmp_path):
        fx = make_fixture(tmp_path)
        seen = {}

        def backup(conn, dest):
            seen["before"] = (_mode(dest), Path(dest).stat().st_size) if Path(dest).exists() else "no file yet"
            world._default_backup(conn, dest)
            seen["after"] = _mode(dest)

        with _Umask(0o000):
            report = real_run(fx, quiet_env(backup_fn=backup))
        assert report["status"] == "completed", report.get("stop")
        assert seen["before"] == (0o600, 0), "the file must exist, empty and owner-only, before the copy starts"
        assert seen["after"] == 0o600

    def test_the_default_backup_function_works_on_an_empty_file_and_on_a_missing_one(self, tmp_path):
        fx = make_fixture(tmp_path)
        src = sqlite3.connect(fx.db)
        try:
            empty = tmp_path / "empty.copy"
            empty.write_bytes(b"")
            world._default_backup(src, empty)
            missing = tmp_path / "missing.copy"
            world._default_backup(src, missing)
        finally:
            src.close()
        for copy in (empty, missing):
            c = sqlite3.connect(f"file:{copy}?immutable=1", uri=True)
            try:
                assert c.execute("SELECT count(*) FROM slots").fetchone()[0] > 0
            finally:
                c.close()

    def test_a_backup_that_comes_back_wider_than_owner_only_stops_before_any_write(self, tmp_path):
        fx = make_fixture(tmp_path)
        before = md5(fx.db)
        stmts = []

        def widen(conn, dest):
            world._default_backup(conn, dest)
            os.chmod(dest, 0o644)

        report = real_run(fx, tracing_env(stmts, backup_fn=widen))
        stopped_cleanly(fx, report, "backup", before, stmts)
        message = report["stop"]["message"]
        assert "0644" in message and ".partial" in message, message
        assert [s["step"] for s in report["steps"]][-1] == "backup", "the run must not reach the verification"

    def test_a_final_backup_that_is_wider_after_the_rename_stops_before_any_write(self, tmp_path, monkeypatch):
        fx = make_fixture(tmp_path)
        before = md5(fx.db)
        stmts = []
        real_replace = os.replace
        widened = []

        def replace_then_widen(src, dst):
            real_replace(src, dst)
            if ".pre-cleanup-" in str(dst) and not str(dst).endswith(".partial"):
                os.chmod(dst, 0o644)
                widened.append(str(dst))

        monkeypatch.setattr(backup_module.os, "replace", replace_then_widen)
        seen = {}

        def backup(conn, dest):
            world._default_backup(conn, dest)
            seen["partial_mode"] = _mode(dest)

        report = real_run(fx, tracing_env(stmts, backup_fn=backup))
        assert seen["partial_mode"] == 0o600 and len(widened) == 1, "the partial must pass its own check first"
        stopped_cleanly(fx, report, "backup", before, stmts, backups_ok=True)
        message = report["stop"]["message"]
        assert "0644" in message and widened[0] in message, message
        assert [s["step"] for s in report["steps"]][-1] == "verify-backup"

    def test_a_group_readable_backup_is_also_refused(self, tmp_path):
        fx = make_fixture(tmp_path)
        before = md5(fx.db)
        stmts = []

        def group_readable(conn, dest):
            world._default_backup(conn, dest)
            os.chmod(dest, 0o640)

        report = real_run(fx, tracing_env(stmts, backup_fn=group_readable))
        stopped_cleanly(fx, report, "backup", before, stmts)
        assert "0640" in report["stop"]["message"]

    def test_nothing_in_the_backup_directory_has_a_group_or_other_bit_after_a_run(self, tmp_path):
        fx = make_fixture(tmp_path)
        with _Umask(0o000):
            code = cleanup.main(["--db", str(fx.db), "--backup-dir", str(fx.backups), "--margin-gib", "0",
                                 "--sleep-ms", "0", "--execute"], quiet_env())
        assert code == 0
        names = sorted(os.listdir(fx.backups))
        assert len(names) == 2 and any(n.endswith(".json") for n in names), names
        assert [n for n in names if _mode(fx.backups / n) & 0o077] == []


class TestTheBackupDirectoryMustNotBeWritableByOthers:
    @pytest.mark.parametrize("mode", [0o775, 0o757, 0o777])
    def test_a_group_or_world_writable_backup_directory_is_refused_for_a_real_run(self, tmp_path, mode):
        fx = make_fixture(tmp_path)
        os.chmod(fx.backups, mode)
        before = md5(fx.db)
        stmts = []
        report = cleanup.run(cfg_for(fx, execute=True), tracing_env(stmts))
        stopped_cleanly(fx, report, "preflight", before, stmts)
        assert_stop(report, "preflight", "writable by group or others")
        assert os.listdir(fx.backups) == []

    def test_a_dry_run_is_not_refused_for_it(self, tmp_path):
        fx = make_fixture(tmp_path)
        os.chmod(fx.backups, _GROUP_WRITABLE)
        assert cleanup.run(cfg_for(fx), quiet_env())["status"] == "dry-run complete"

    def test_an_owner_only_or_read_only_for_others_directory_is_accepted(self, tmp_path):
        for mode in (0o700, 0o755):
            fx = make_fixture(sub(tmp_path, f"m{mode:o}"))
            os.chmod(fx.backups, mode)
            report = real_run(fx)
            assert report["status"] == "completed", (oct(mode), report.get("stop"))
