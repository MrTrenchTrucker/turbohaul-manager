"""The preflight gates and the free-space gate."""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import os
import sqlite3
from types import SimpleNamespace

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
    report_of,
    stopped_cleanly,
    sub,
    tracing_env,
)

config = internals("config")
connect = internals("connect")


class TestPreflightGates:
    def _stop(self, tmp_path, cfg_kw=None, env_kw=None, execute=True):
        fx = make_fixture(tmp_path)
        before = md5(fx.db)
        stmts = []
        cfg = cfg_for(fx, execute=execute, **(cfg_kw or {}))
        report = cleanup.run(cfg, tracing_env(stmts, **(env_kw or {})))
        return fx, report, before, stmts

    def test_a_backup_directory_inside_the_database_directory_is_refused(self, tmp_path):
        fx = make_fixture(tmp_path)
        before = md5(fx.db)
        report = cleanup.run(cfg_for(fx, execute=True, backup_dir=fx.data), quiet_env())
        assert_stop(report, "preflight")
        assert md5(fx.db) == before

    def test_a_missing_backup_directory_is_refused(self, tmp_path):
        fx = make_fixture(tmp_path)
        report = cleanup.run(cfg_for(fx, execute=True, backup_dir=tmp_path / "nowhere"), quiet_env())
        assert_stop(report, "preflight", "backup directory not found")

    def test_a_backup_directory_that_cannot_be_written_is_refused_for_a_real_run_only(self, tmp_path):
        if os.geteuid() == 0:
            pytest.skip("permissions do not bind root")
        fx = make_fixture(tmp_path)
        os.chmod(fx.backups, 0o500)
        try:
            assert cleanup.run(cfg_for(fx), quiet_env())["status"] == "dry-run complete"
            before = md5(fx.db)
            report = report_of(lambda: cleanup.run(cfg_for(fx, execute=True), quiet_env()))
            assert_stop(report, "preflight", "not writable")
            assert md5(fx.db) == before
        finally:
            os.chmod(fx.backups, 0o700)

    def test_opening_for_exclusive_access_never_creates_a_missing_file(self, tmp_path):
        with pytest.raises(cleanup.Stop) as exc:
            connect._open_exclusive(tmp_path / "nothing.sqlite", 0.1)
        assert exc.value.step == "lock"
        assert not (tmp_path / "nothing.sqlite").exists()

    def test_a_missing_database_is_refused_and_not_created(self, tmp_path):
        fx = make_fixture(tmp_path)
        report = report_of(lambda: cleanup.run(cfg_for(fx, execute=True, db=tmp_path / "nothing.sqlite"), quiet_env()))
        assert_stop(report, "preflight", "database not found")
        assert not (tmp_path / "nothing.sqlite").exists()

    def test_a_work_directory_on_another_filesystem_is_refused(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        fx = make_fixture(sub(tmp_path, "fx"))
        env = quiet_env(dev_of=lambda p: 2 if str(p).startswith(str(work)) else 1)
        before = md5(fx.db)
        report = cleanup.run(cfg_for(fx, execute=True, work_dir=work), env)
        assert_stop(report, "preflight", "same filesystem")
        assert md5(fx.db) == before

    def test_a_user_who_is_not_root_or_the_databases_owner_is_refused_before_any_write(self, tmp_path):
        fx = make_fixture(tmp_path)
        stranger = os.stat(fx.db).st_uid + 4242
        before = md5(fx.db)
        stmts = []
        report = cleanup.run(cfg_for(fx, execute=True), tracing_env(stmts, geteuid=lambda: stranger))
        stopped_cleanly(fx, report, "preflight", before, stmts)
        assert "owner" in report["stop"]["message"]

    def test_root_and_the_owner_are_accepted(self, tmp_path):
        for label, euid in (("root", lambda fx: 0), ("owner", lambda fx: os.stat(fx.db).st_uid)):
            fx = make_fixture(sub(tmp_path, label))
            report = cleanup.run(cfg_for(fx, execute=True), quiet_env(geteuid=lambda fx=fx, f=euid: f(fx)))
            assert report["status"] == "completed", (label, report.get("stop"))

    def test_the_owner_check_does_not_apply_to_a_dry_run(self, tmp_path):
        fx = make_fixture(tmp_path)
        stranger = os.stat(fx.db).st_uid + 4242
        report = cleanup.run(cfg_for(fx), quiet_env(geteuid=lambda: stranger))
        assert report["status"] == "dry-run complete", report.get("stop")

    def test_a_sqlite_too_old_for_vacuum_into_is_refused(self, tmp_path):
        fx, report, before, stmts = self._stop(tmp_path, env_kw={"sqlite_version": (3, 26, 0)})
        stopped_cleanly(fx, report, "preflight", before, stmts)

    def test_the_run_refuses_while_another_connection_has_the_database_open(self, tmp_path):
        fx = make_fixture(tmp_path)
        other = sqlite3.connect(fx.db)
        other.execute("SELECT count(*) FROM slots").fetchone()
        before = md5(fx.db)
        stmts = []
        try:
            report = real_run(fx, tracing_env(stmts))
        finally:
            other.close()
        assert_stop(report, "lock")
        assert md5(fx.db) == before and os.listdir(fx.backups) == []
        assert not [n for n in os.listdir(fx.data) if n.startswith(config._TMP_PREFIX)]

    def test_the_run_refuses_while_another_connection_is_writing(self, tmp_path):
        fx = make_fixture(tmp_path)
        other = sqlite3.connect(fx.db, isolation_level=None)
        other.execute("BEGIN IMMEDIATE")
        try:
            report = real_run(fx)
        finally:
            other.execute("ROLLBACK")
            other.close()
        assert_stop(report, "lock")

    def test_while_the_run_holds_the_database_nothing_else_can_open_it(self, tmp_path):
        fx = make_fixture(tmp_path)
        attempts = []

        def probe(conn):
            other = sqlite3.connect(fx.db, timeout=0.2)
            try:
                other.execute("SELECT count(*) FROM slots").fetchone()
                attempts.append("opened")
            except sqlite3.OperationalError as exc:
                attempts.append(str(exc))
            finally:
                other.close()

        real_run(fx, quiet_env(sleep=lambda s: probe(None)), batch=4)
        assert attempts and all("locked" in a for a in attempts), attempts


class TestSpaceGate:
    def _free(self, n):
        return lambda p: SimpleNamespace(f_bavail=n, f_frsize=1)

    def test_one_volume_must_hold_the_backup_the_new_file_and_the_margin(self, tmp_path):
        fx = make_fixture(tmp_path)
        probe = cleanup.run(cfg_for(fx, margin_bytes=1000), quiet_env())["space"]
        required = probe["volumes"][0]["required"]
        assert required == probe["backup_bytes"] + probe["new_file_estimate"] + 1000
        for free, expect_ok in ((required, True), (required - 1, False)):
            report = cleanup.run(cfg_for(fx, margin_bytes=1000), quiet_env(statvfs=self._free(free)))
            assert (report["status"] != "stopped") is expect_ok, (free, required)
            if not expect_ok:
                assert_stop(report, "space")

    def test_a_failed_space_gate_stops_before_any_write(self, tmp_path):
        fx = make_fixture(tmp_path)
        before = md5(fx.db)
        stmts = []
        report = real_run(fx, tracing_env(stmts, statvfs=self._free(10)))
        stopped_cleanly(fx, report, "space", before, stmts)
        assert os.listdir(fx.backups) == []

    def test_the_margin_is_part_of_the_requirement(self, tmp_path):
        fx = make_fixture(tmp_path)
        need = cleanup.run(cfg_for(fx), quiet_env())["space"]["volumes"][0]["required"]
        report = cleanup.run(cfg_for(fx, margin_bytes=5000), quiet_env(statvfs=self._free(need + 4999)))
        assert_stop(report, "space")

    def test_separate_volumes_are_each_checked_against_their_own_share(self, tmp_path):
        backups = tmp_path / "other_volume"
        backups.mkdir()
        fx = make_fixture(sub(tmp_path, "fx"))
        def dev(p):
            return 2 if str(p).startswith(str(backups)) else 1

        probe = cleanup.run(cfg_for(fx, backup_dir=backups), quiet_env(dev_of=dev))["space"]
        need_backup, need_new = probe["backup_bytes"], probe["new_file_estimate"]

        def free_for(backup_free, work_free):
            return lambda p: SimpleNamespace(f_bavail=backup_free if str(p).startswith(str(backups)) else work_free,
                                             f_frsize=1)

        ok = cleanup.run(cfg_for(fx, backup_dir=backups),
                         quiet_env(dev_of=dev, statvfs=free_for(need_backup, need_new)))
        assert ok["status"] == "dry-run complete", ok.get("stop")
        low_backup = cleanup.run(cfg_for(fx, backup_dir=backups),
                                 quiet_env(dev_of=dev, statvfs=free_for(need_backup - 1, need_new)))
        assert low_backup["stop"]["step"] == "space"
        low_work = cleanup.run(cfg_for(fx, backup_dir=backups),
                               quiet_env(dev_of=dev, statvfs=free_for(need_backup, need_new - 1)))
        assert low_work["stop"]["step"] == "space"

    def test_the_new_file_estimate_is_never_below_the_real_size_and_not_wildly_above(self, tmp_path):
        fx = make_fixture(tmp_path, pad=200_000)
        report = real_run(fx)
        estimate, actual = report["space"]["new_file_estimate"], report["sizes_after"]["db"]
        assert estimate >= actual, "an estimate below the real size would let the gate pass when it should not"
        assert estimate <= 2 * actual, "the estimate must follow what the strip removes, not the old size"

    def test_the_new_file_estimate_stays_high_when_there_is_little_to_remove(self, tmp_path):
        fx = make_fixture(tmp_path, pad=10)
        space = cleanup.run(cfg_for(fx), quiet_env())["space"]
        assert space["new_file_estimate"] >= 0.5 * space["backup_bytes"]
