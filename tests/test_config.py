"""Tests for BootConfig + RuntimeConfig schema."""
import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from turbohaul.config import (
    KEEP_ALIVE_MAX_S,
    BootConfig,
    MonitorConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    TurbohaulConfig,
    UIConfig,
    apply_env_overrides,
    compute_shipped_baseline,
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
        # Default raised over time: 120 -> 300 -> 600.
        # A reasoning model with reasoning_budget=1000 can produce 5-7min
        # inter-turn gaps on complex prompts, which would eat a 300s coverage
        # window. Covers OpenAI-SDK clients that can't send keep_alive
        # natively (Ollama Issue #11458).
        assert q.idle_hot_load_seconds == 600
        assert q.max_grace_extensions == 5
        assert q.drained_sigterm_window_active_s == 15
        assert q.drained_sigterm_window_cold_s == 5

    def test_keep_alive_max_constant(self):
        # Module-level constant (not a Field -- this is a fixed operational
        # policy, deliberately not a per-deployment knob).
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


class TestComputeShippedBaseline:
    """The 'what a fresh deployment gets' baseline the
    startup divergence log compares the real effective config against --
    code default, overridden by the shipped yaml if present, nothing else."""

    def test_yaml_present_value_wins_over_code_default(self, temp_etc_config):
        # temp_etc_config's queue.idle_hot_load_seconds: 120, vs QueueConfig's
        # own code default of 600 -- a real divergence, not a coincidence,
        # so a baseline that ignored the yaml would be caught here.
        raw = yaml.safe_load(temp_etc_config.read_text())
        baseline = compute_shipped_baseline(raw)
        assert baseline["queue"]["idle_hot_load_seconds"] == 120

    def test_yaml_absent_field_falls_back_to_code_default(self, temp_etc_config):
        # queue.max_consecutive_same_model is NOT set in temp_etc_config's yaml
        # at all -- the baseline for it must be QueueConfig's own code default
        # (3), not some other value, and must not KeyError.
        raw = yaml.safe_load(temp_etc_config.read_text())
        baseline = compute_shipped_baseline(raw)
        assert baseline["queue"]["max_consecutive_same_model"] == QueueConfig().max_consecutive_same_model
        assert QueueConfig().max_consecutive_same_model == 3  # pin the assumption

    def test_missing_section_entirely_falls_back_to_all_defaults(self, temp_etc_config):
        # temp_etc_config's yaml has no "monitor" section at all (legal --
        # MonitorConfig is Field(default_factory=MonitorConfig) at the
        # TurbohaulConfig level) -- must not KeyError; every field falls back
        # to MonitorConfig()'s own default. Boot sections (which DO have
        # required fields with no default) are present in this fixture's
        # yaml, so this isolates "a runtime section is entirely absent" from
        # "a required boot field is absent" -- a different failure mode.
        raw = yaml.safe_load(temp_etc_config.read_text())
        assert "monitor" not in raw  # pin the fixture's actual shape
        baseline = compute_shipped_baseline(raw)
        assert baseline["monitor"] == MonitorConfig().model_dump(mode="json")

    def test_baseline_is_independent_of_env_and_persisted_layers(
        self, temp_etc_config, monkeypatch
    ):
        # A causal control: setting an env var must NOT move the baseline --
        # the baseline is deliberately the pre-env, pre-persisted value. If
        # compute_shipped_baseline accidentally read os.environ, this fails.
        monkeypatch.setenv("TURBOHAUL_GRACE_S", "999")
        raw = yaml.safe_load(temp_etc_config.read_text())
        baseline = compute_shipped_baseline(raw)
        assert baseline["queue"]["grace_seconds"] == 30  # yaml's value, not 999


def test_persist_max_bytes_env_override(temp_etc_config, monkeypatch):
    """Regression: the env var must land on the field the
    persist GC + FE snapshot actually read (persist.max_bytes) — it must not
    map to a dead queue field, which would make the operator env override a
    silent no-op."""
    cfg = load_config_yaml(temp_etc_config)
    monkeypatch.setenv("TURBOHAUL_KVCACHE_PERSIST_MAX_BYTES", "1073741824")
    cfg2 = apply_env_overrides(cfg)
    assert cfg2.persist.max_bytes == 1073741824
