"""Tests for BootConfig + RuntimeConfig schema (v0.2 §7 + §7.1)."""
import pytest
from pathlib import Path
from pydantic import ValidationError

from turbohaul.config import (
    KEEP_ALIVE_MAX_S,
    BootConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    TurbohaulConfig,
    UIConfig,
    apply_env_overrides,
    load_config_yaml,
)


class TestServerConfig:
    def test_default_host_loopback(self):
        s = ServerConfig()
        assert s.host == "127.0.0.1"
        assert s.port == 11401
        assert s.allow_public_bind is False

    def test_reject_zero_zero_zero_zero_host(self):
        with pytest.raises(ValidationError, match="0.0.0.0"):
            ServerConfig(host="0.0.0.0")

    def test_frozen_after_construction(self):
        s = ServerConfig()
        with pytest.raises(ValidationError):
            s.host = "1.2.3.4"  # type: ignore[misc]

    def test_port_bounds(self):
        with pytest.raises(ValidationError):
            ServerConfig(port=0)
        with pytest.raises(ValidationError):
            ServerConfig(port=70000)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            ServerConfig(host="127.0.0.1", evil_flag=True)  # type: ignore[call-arg]


class TestQueueConfig:
    def test_v0_2_conservative_defaults(self):
        q = QueueConfig()
        assert q.grace_seconds == 30
        # bumped 120 → 300 default (advisor MSG 23 Option E).
        # bumped 300 → 600 (a test surface — Qwen3
        # reasoning_budget=1000 produces 5-7min inter-turn gaps on complex
        # prompts, eating the 300s coverage). Covers OpenAI-SDK clients that
        # can't send keep_alive natively (Ollama Issue #11458).
        assert q.idle_hot_load_seconds == 600
        assert q.max_grace_extensions == 5
        assert q.drained_sigterm_window_active_s == 15
        assert q.drained_sigterm_window_cold_s == 5

    def test_wave_4b5_keep_alive_max_constant(self):
        # module-level constant (not a Field — operational policy
        # per review note verdict, not per-deployment knob).
        assert KEEP_ALIVE_MAX_S == 1800

    def test_reject_unknown_field(self):
        with pytest.raises(ValidationError):
            QueueConfig(unknown_field=1)  # type: ignore[call-arg]

    def test_grace_seconds_bounds(self):
        with pytest.raises(ValidationError):
            QueueConfig(grace_seconds=-1)
        with pytest.raises(ValidationError):
            QueueConfig(grace_seconds=3601)


class TestPullConfig:
    def test_default_https_only(self):
        p = PullConfig()
        assert p.pull_url_https_only is True
        assert "huggingface.co" in p.hf_host_allowlist
        assert "hf.co" in p.hf_host_allowlist

    def test_reject_unknown_field(self):
        with pytest.raises(ValidationError):
            PullConfig(evil_flag=True)  # type: ignore[call-arg]


class TestLoadConfigYaml:
    def test_load_full_yaml(self, temp_etc_config):
        cfg = load_config_yaml(temp_etc_config)
        assert isinstance(cfg, TurbohaulConfig)
        assert cfg.server.host == "127.0.0.1"
        assert cfg.queue.grace_seconds == 30

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config_yaml(tmp_path / "missing.yaml")

    def test_split_returns_boot_and_runtime(self, temp_etc_config):
        cfg = load_config_yaml(temp_etc_config)
        boot, runtime = cfg.split()
        assert isinstance(boot, BootConfig)
        assert isinstance(runtime, RuntimeConfig)
        assert boot.server.host == "127.0.0.1"
        assert runtime.queue.grace_seconds == 30


class TestEnvOverrides:
    def test_env_beats_yaml(self, temp_etc_config, monkeypatch):
        cfg = load_config_yaml(temp_etc_config)
        assert cfg.queue.grace_seconds == 30
        monkeypatch.setenv("TURBOHAUL_GRACE_S", "45")
        cfg2 = apply_env_overrides(cfg)
        assert cfg2.queue.grace_seconds == 45

    def test_port_override(self, temp_etc_config, monkeypatch):
        cfg = load_config_yaml(temp_etc_config)
        monkeypatch.setenv("TURBOHAUL_PORT", "11402")
        cfg2 = apply_env_overrides(cfg)
        assert cfg2.server.port == 11402

    def test_idle_hot_s_override(self, temp_etc_config, monkeypatch):
        cfg = load_config_yaml(temp_etc_config)
        monkeypatch.setenv("TURBOHAUL_IDLE_HOT_S", "240")
        cfg2 = apply_env_overrides(cfg)
        assert cfg2.queue.idle_hot_load_seconds == 240

    def test_no_env_means_yaml_preserved(self, temp_etc_config, monkeypatch):
        # Ensure no env var set
        monkeypatch.delenv("TURBOHAUL_GRACE_S", raising=False)
        cfg = load_config_yaml(temp_etc_config)
        cfg2 = apply_env_overrides(cfg)
        assert cfg2.queue.grace_seconds == cfg.queue.grace_seconds
