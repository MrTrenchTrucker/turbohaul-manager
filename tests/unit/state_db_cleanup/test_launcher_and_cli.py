"""The module is built on the projection module; the launcher; defaults and exit codes."""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import json
import os
import subprocess
import sys

import pytest

from state_db_cleanup.helpers import (
    TOOL,
    cleanup,
    internals,
    make_fixture,
    projection,
    quiet_env,
    sub,
)

cli = internals("cli")
sequence = internals("sequence")
strip = internals("strip")


class TestBuiltOnTheProjectionModule:
    def test_the_allowlist_and_the_reduction_come_from_the_projection_module(self):
        assert strip.strip_stored_meta is projection.strip_stored_meta
        assert sequence.IDENTITY_KEYS is projection.IDENTITY_KEYS

    def test_the_command_line_runs_from_any_directory(self, tmp_path):
        fx = make_fixture(sub(tmp_path, "fx"))
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        done = subprocess.run(  # noqa: S603 - runs this repository's own script with fixed arguments
            [sys.executable, "-B", str(TOOL), "--db", str(fx.db), "--backup-dir", str(fx.backups), "--margin-gib", "0"],
            cwd=elsewhere, env=env, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr[-500:]
        assert json.loads(done.stdout)["status"] == "dry-run complete"

    def test_the_database_and_the_backup_directory_are_required(self):
        with pytest.raises(SystemExit) as exc:
            cleanup.parse_args(["--db", "x"])
        assert exc.value.code == 2


class TestDefaultsAndExitCodes:
    def test_the_defaults_are_the_ruled_ones(self):
        a = cleanup.parse_args(["--db", "x", "--backup-dir", "y"])
        assert (a.batch, a.sleep_ms, a.checkpoint_every, a.min_age_hours) == (500, 50, 20, 0.0)
        assert a.margin_gib == 5.0 and a.execute is False and a.no_swap is False

    def test_an_unexpected_failure_exits_with_one_and_says_so(self, tmp_path, monkeypatch, capsys):
        fx = make_fixture(tmp_path)
        monkeypatch.setattr(cli, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        code = cleanup.main(["--db", str(fx.db), "--backup-dir", str(fx.backups)], quiet_env())
        assert code == 1 and "boom" in capsys.readouterr().out

    def test_a_gate_stop_exits_with_two(self, tmp_path):
        fx = make_fixture(tmp_path)
        code = cleanup.main(["--db", str(fx.db), "--backup-dir", str(fx.db.parent)], quiet_env())
        assert code == 2
