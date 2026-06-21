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
        # Default raised in steps from 120 → 300 → 600s. Models with a large
        # reasoning_budget (e.g. Qwen3 reasoning_budget=1000) can produce
        # 5-7min inter-turn gaps on complex prompts, which ate the earlier
        # 300s coverage. Covers OpenAI-SDK clients that can't send keep_alive
        # natively (Ollama Issue #11458).
        assert q.idle_hot_load_seconds == 600
        assert q.max_grace_extensions == 5
        assert q.drained_sigterm_window_active_s == 15
        assert q.drained_sigterm_window_cold_s == 5

    def test_keep_alive_max_constant(self):
        # Module-level constant (not a Field) — this is an operational policy
        # ceiling, not a per-deployment knob.
        assert KEEP_ALIVE_MAX_S == 1800

    def test_reject_unknown_field(self):
        with pytest.raises(ValidationError):
            QueueConfig(unknown_field=1)  # type: ignore[call-arg]

    def test_grace_seconds_bounds(self):
        with pytest.raises(ValidationError):
            QueueConfig(grace_seconds=-1)
        with pytest.raises(ValidationError):
            QueueConfig(grace_seconds=3601)

    def test_model_affinity_defaults(self):
        # Single-mutator-safe parallelism support: model-affinity pop tuning.
        q = QueueConfig()
        assert q.max_consecutive_same_model == 3
        assert q.max_other_model_wait_s == 20.0

    def test_model_affinity_fields_parse(self):
        q = QueueConfig(max_consecutive_same_model=7, max_other_model_wait_s=2.5)
        assert q.max_consecutive_same_model == 7
        assert q.max_other_model_wait_s == 2.5

    def test_max_consecutive_same_model_bounds(self):
        with pytest.raises(ValidationError):
            QueueConfig(max_consecutive_same_model=0)  # ge=1
        with pytest.raises(ValidationError):
            QueueConfig(max_consecutive_same_model=1001)  # le=1000

    def test_max_other_model_wait_s_bounds(self):
        with pytest.raises(ValidationError):
            QueueConfig(max_other_model_wait_s=-0.1)  # ge=0.0
        with pytest.raises(ValidationError):
            QueueConfig(max_other_model_wait_s=3600.1)  # le=3600.0


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

    def test_max_consecutive_same_model_override(self, temp_etc_config, monkeypatch):
        cfg = load_config_yaml(temp_etc_config)
        assert cfg.queue.max_consecutive_same_model == 3
        monkeypatch.setenv("TURBOHAUL_MAX_CONSECUTIVE_SAME_MODEL", "8")
        cfg2 = apply_env_overrides(cfg)
        assert cfg2.queue.max_consecutive_same_model == 8

    def test_max_other_model_wait_s_override(self, temp_etc_config, monkeypatch):
        cfg = load_config_yaml(temp_etc_config)
        assert cfg.queue.max_other_model_wait_s == 20.0
        monkeypatch.setenv("TURBOHAUL_MAX_OTHER_MODEL_WAIT_S", "4.5")
        cfg2 = apply_env_overrides(cfg)
        assert cfg2.queue.max_other_model_wait_s == 4.5
