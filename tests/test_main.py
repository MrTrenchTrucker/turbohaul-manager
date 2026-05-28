"""Tests for the CLI entry point (src/turbohaul/__main__.py)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from turbohaul.__main__ import build_parser, main


@pytest.fixture
def write_minimal_yaml(tmp_path: Path):
    """Yield a yaml path containing a valid TurbohaulConfig."""
    blob = tmp_path / "blobs"
    blob.mkdir()
    man = tmp_path / "manifests"
    man.mkdir()
    imp = tmp_path / "import-staging"
    imp.mkdir()
    ui = tmp_path / "ui_dist"
    ui.mkdir()
    binary = tmp_path / "fake_llama_server"
    binary.write_bytes(b"")

    yaml_text = f"""
server:
  host: 127.0.0.1
  port: 11401
  allow_public_bind: false
storage:
  blob_store_path: {blob}
  manifests_path: {man}
  import_allowed_root: {imp}
  state_db_path: {tmp_path / "state.sqlite"}
runtime:
  llama_server_binary: {binary}
  llama_server_binary_sha256: ""
  default_port_base: 11500
ui:
  enabled: true
  static_path: {ui}
queue: {{}}
pull: {{}}
"""
    path = tmp_path / "turbohaul.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


class TestParser:
    def test_default_config_path(self):
        args = build_parser().parse_args([])
        # Either the test env's TURBOHAUL_CONFIG_PATH OR the hardcoded default — both fine.
        assert args.config.suffix == ".yaml"

    def test_allow_public_bind_off_by_default(self, monkeypatch):
        monkeypatch.delenv("TURBOHAUL_ALLOW_PUBLIC_BIND", raising=False)
        args = build_parser().parse_args([])
        assert args.allow_public_bind is False

    def test_allow_public_bind_via_flag(self):
        args = build_parser().parse_args(["--allow-public-bind"])
        assert args.allow_public_bind is True

    def test_allow_public_bind_via_env(self, monkeypatch):
        monkeypatch.setenv("TURBOHAUL_ALLOW_PUBLIC_BIND", "1")
        args = build_parser().parse_args([])
        assert args.allow_public_bind is True


class TestMain:
    def test_main_missing_config_returns_2(self, tmp_path, capsys):
        rc = main(["--config", str(tmp_path / "nope.yaml")])
        assert rc == 2

    def test_main_loads_config_and_invokes_uvicorn(
        self, write_minimal_yaml, monkeypatch
    ):
        """main() must reach uvicorn.run with host from BootConfig (no public bind)."""
        called = {}

        def fake_run(app, **kwargs):
            called["app"] = app
            called["host"] = kwargs.get("host")
            called["port"] = kwargs.get("port")

        with patch("turbohaul.__main__.uvicorn.run", side_effect=fake_run):
            rc = main(["--config", str(write_minimal_yaml)])
        assert rc == 0
        assert called["host"] == "127.0.0.1"
        assert called["port"] == 11401

    def test_main_allow_public_bind_overrides_host(
        self, write_minimal_yaml
    ):
        called = {}

        def fake_run(app, **kwargs):
            called["host"] = kwargs.get("host")

        with patch("turbohaul.__main__.uvicorn.run", side_effect=fake_run):
            rc = main([
                "--config", str(write_minimal_yaml),
                "--allow-public-bind",
            ])
        assert rc == 0
        assert called["host"] == "0.0.0.0"

    def test_main_log_level_passed_to_uvicorn(self, write_minimal_yaml):
        called = {}

        def fake_run(app, **kwargs):
            called["log_level"] = kwargs.get("log_level")

        with patch("turbohaul.__main__.uvicorn.run", side_effect=fake_run):
            main([
                "--config", str(write_minimal_yaml),
                "--log-level", "debug",
            ])
        assert called["log_level"] == "debug"
