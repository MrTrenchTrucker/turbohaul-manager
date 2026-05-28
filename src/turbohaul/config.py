"""Top-level configuration: BootConfig (read-only at runtime) vs RuntimeConfig (PUT-mutable).

Per v0.2 ARCHITECTURE.md §7 + §7.1 - security finding F7 must-fix.

BootConfig fields require restart to change (server bind, storage paths, binary path).
RuntimeConfig fields are mutable via PUT /api/config (queue timings, pull params).
"""
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


# maximum honored client `keep_alive` value (advisor MSG 23 cap).
# Module constant, not a QueueConfig field — operational policy, not per-deployment
# knob. Bump here if hardware changes.
KEEP_ALIVE_MAX_S = 1800


class ServerConfig(BaseModel):
    """Boot-only: server bind config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=11401, ge=1, le=65535)
    allow_public_bind: bool = False

    @field_validator("host")
    @classmethod
    def host_safe_default(cls, v: str) -> str:
        if v == "0.0.0.0":
            raise ValueError(
                "server.host cannot be 0.0.0.0 from yaml; set allow_public_bind: true "
                "AND pass --allow-public-bind CLI flag explicitly to bind public"
            )
        return v


class StorageConfig(BaseModel):
    """Boot-only: storage paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blob_store_path: Path
    manifests_path: Path
    import_allowed_root: Path
    state_db_path: Path


class RuntimePathsConfig(BaseModel):
    """Boot-only: binary path + sha256 pin + child port base."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    llama_server_binary: Path
    llama_server_binary_sha256: str = ""  # empty = skip verify (dev only)
    default_port_base: int = Field(default=11500, ge=1024, le=65000)


class UIConfig(BaseModel):
    """Boot-only: UI static path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    static_path: Path


class QueueConfig(BaseModel):
    """Runtime-mutable: queue + timing constants."""

    model_config = ConfigDict(extra="forbid")

    max_parallel_sidecars: int = Field(default=1, ge=1, le=32)
    staging_queue_depth: int = Field(default=100, ge=1, le=10000)
    acceptance_buffer_max: int = Field(default=10000, ge=1)
    grace_seconds: int = Field(default=30, ge=0, le=3600)
    #  (advisor MSG 23 Option E): default bumped 120 → 300 so multi-turn
    # agents (an agent / OpenAI-SDK class) with PC-side tool-exec / reflection gaps
    # in the 2-5min range keep their slot warm without needing to send keep_alive.
    # OpenAI-SDK clients can't send keep_alive natively (Ollama Issue #11458).
    # Tuning: bumped 300 → 600 because
    # Qwen3 reasoning_budget=1000 on complex compare prompts produces 5-7min
    # client-side inter-turn gaps. The 300s window was eaten by an agent' reasoning
    # chain on the FIRST tool-result reflection, not by Turbohaul itself.
    idle_hot_load_seconds: int = Field(default=600, ge=0, le=86400)
    # safety guardrails -- mirror Ollama pre-spawn safety posture
    safety_enabled: bool = True
    safety_min_free_ram_mib: int = Field(default=1024, ge=0)
    safety_min_free_vram_mib: int = Field(default=512, ge=0)
    safety_max_load_per_core: float = Field(default=0.9, ge=0.0)
    safety_max_iowait_percent: float = Field(default=30.0, ge=0.0, le=100.0)
    safety_iowait_sample_window_s: float = Field(default=0.4, ge=0.05, le=5.0)
    max_grace_extensions: int = Field(default=5, ge=0, le=1000)
    loading_health_timeout_s: int = Field(default=600, ge=10, le=7200)
    drained_sigterm_window_active_s: int = Field(default=15, ge=1, le=300)
    drained_sigterm_window_cold_s: int = Field(default=5, ge=1, le=300)
    # Background sweeper cadence — finalizes state-row for
    # client-disconnect evictions that landed audit-only via the _audit_event_only_async pool path.
    # 60s aligns with the audit pool rhythm. Sweeper requires staleness ≥ 24h
    # (background_sweep_min_age_s) so in-flight slots are never reaped.
    background_sweep_interval_s: int = Field(default=60, ge=1, le=86400)
    background_sweep_min_age_s: int = Field(default=86400, ge=60, le=2592000)  # Design note: floor stays at 60s — actual SQL gate is `state=STAGED` (NOT grace-rematch states), so operator misconfig cannot reap in-flight grace-rematch slots; gate-filter is sufficient defense; 60s floor preserved for synthetic-age test boundary


class PullConfig(BaseModel):
    """Runtime-mutable: pull endpoints + safety constraints."""

    model_config = ConfigDict(extra="forbid")

    hf_api_key_env: str = "HF_API_KEY"
    hf_host_allowlist: list[str] = Field(default_factory=lambda: ["huggingface.co", "hf.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co"])
    pull_url_https_only: bool = True
    pull_concurrency: int = Field(default=2, ge=1, le=16)
    pull_chunk_size_mb: int = Field(default=64, ge=1, le=1024)
    per_stream_max_bytes: int = Field(default=107_374_182_400, ge=1)


class BootConfig(BaseModel):
    """Top-level boot-only configuration (frozen after load)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    server: ServerConfig
    storage: StorageConfig
    runtime: RuntimePathsConfig
    ui: UIConfig


class RuntimeConfig(BaseModel):
    """Top-level runtime-mutable configuration (PUT-able)."""

    model_config = ConfigDict(extra="forbid")

    queue: QueueConfig
    pull: PullConfig


class TurbohaulConfig(BaseModel):
    """Full config = boot + runtime, used for yaml load/save."""

    model_config = ConfigDict(extra="forbid")

    server: ServerConfig
    storage: StorageConfig
    runtime: RuntimePathsConfig
    ui: UIConfig
    queue: QueueConfig
    pull: PullConfig

    def split(self) -> tuple[BootConfig, RuntimeConfig]:
        boot = BootConfig(
            server=self.server,
            storage=self.storage,
            runtime=self.runtime,
            ui=self.ui,
        )
        runtime = RuntimeConfig(queue=self.queue, pull=self.pull)
        return boot, runtime


def load_config_yaml(path: Path) -> TurbohaulConfig:
    """Load + validate turbohaul.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"config root must be mapping, got {type(data).__name__}")
    return TurbohaulConfig(**data)


_ENV_MAP: dict[str, tuple[str, str, type]] = {
    "TURBOHAUL_HOST": ("server", "host", str),
    "TURBOHAUL_PORT": ("server", "port", int),
    "TURBOHAUL_MAX_PARALLEL": ("queue", "max_parallel_sidecars", int),
    "TURBOHAUL_STAGING_DEPTH": ("queue", "staging_queue_depth", int),
    "TURBOHAUL_ACCEPT_MAX": ("queue", "acceptance_buffer_max", int),
    "TURBOHAUL_GRACE_S": ("queue", "grace_seconds", int),
    "TURBOHAUL_IDLE_HOT_S": ("queue", "idle_hot_load_seconds", int),
    "TURBOHAUL_MAX_GRACE_EXT": ("queue", "max_grace_extensions", int),
}


def apply_env_overrides(cfg: TurbohaulConfig) -> TurbohaulConfig:
    """Apply TURBOHAUL_* env var overrides. Env beats yaml."""
    data: dict[str, Any] = cfg.model_dump()
    for env_key, (section, field, cast) in _ENV_MAP.items():
        v = os.environ.get(env_key)
        if v is not None:
            data[section][field] = cast(v)
    return TurbohaulConfig(**data)
