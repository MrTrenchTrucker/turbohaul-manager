"""Top-level configuration: BootConfig (read-only at runtime) vs RuntimeConfig (PUT-mutable).

The boot-vs-runtime split keeps restart-only settings isolated from hot-mutable ones.

BootConfig fields require restart to change (server bind, storage paths, binary path).
RuntimeConfig fields are mutable via PUT /api/config (queue timings, pull params).
"""
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Maximum honored client `keep_alive` value.
# Module constant, not a QueueConfig field — operational policy, not per-deployment
# knob. Bump here if your hardware changes.
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
    gpu_backend: str = "auto"  # "auto" | "nvidia" | "rocm"

    @field_validator("gpu_backend")
    @classmethod
    def _validate_gpu_backend(cls, v: str) -> str:
        if v not in ("auto", "nvidia", "rocm"):
            raise ValueError(f"gpu_backend must be 'auto', 'nvidia', or 'rocm'; got '{v}'")
        return v


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
    # Model-affinity pop tuning (single-mutator-safe parallelism support).
    # Both default to behavior that is a strict-FIFO no-op unless the
    # worker_loop opts in by passing warm_model_tag to pop_next:
    #   - max_consecutive_same_model bounds the run-length of one model the
    #     affinity path will cluster before forcing the FIFO head (fairness).
    #     A value of 1 disables batching entirely (every pop honors FIFO head).
    #   - max_other_model_wait_s is the age (via Slot.created_at monotonic
    #     clock) past which a starved other-model head request forces a swap,
    #     overriding affinity. 0.0 means "starve immediately" => strict FIFO.
    max_consecutive_same_model: int = Field(default=3, ge=1, le=1000)
    max_other_model_wait_s: float = Field(default=20.0, ge=0.0, le=3600.0)
    grace_seconds: int = Field(default=30, ge=0, le=3600)
    # Default bumped 120 → 300 so multi-turn
    # agents (Hermes / OpenAI-SDK class) with client-side tool-exec / reflection gaps
    # in the 2-5min range keep their slot warm without needing to send keep_alive.
    # OpenAI-SDK clients can't send keep_alive natively (Ollama Issue #11458).
    # Bumped 300 → 600 because
    # Qwen3 reasoning_budget=1000 on complex compare prompts produces 5-7min
    # client-side inter-turn gaps. The 300s window was eaten by the client's reasoning
    # chain on the FIRST tool-result reflection, not by Turbohaul itself.
    idle_hot_load_seconds: int = Field(default=600, ge=0, le=86400)
    # Safety guardrails -- mirror Ollama pre-spawn safety posture
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
    # evictions that landed audit-only via _audit_event_only_async pool path.
    # 60s aligns with the audit pool rhythm. Sweeper requires staleness ≥ 24h
    # (background_sweep_min_age_s) so in-flight slots are never reaped.
    background_sweep_interval_s: int = Field(default=60, ge=1, le=86400)
    background_sweep_min_age_s: int = Field(default=86400, ge=60, le=2592000)  # floor stays at 60s — actual SQL gate is `state=STAGED` (NOT grace-rematch states), so operator misconfig cannot reap in-flight grace-rematch slots; gate-filter is sufficient defense; 60s floor preserved for synthetic-age test boundary


class PullConfig(BaseModel):
    """Runtime-mutable: pull endpoints + safety constraints."""

    model_config = ConfigDict(extra="forbid")

    hf_api_key_env: str = "HF_API_KEY"
    hf_host_allowlist: list[str] = Field(default_factory=lambda: ["huggingface.co", "hf.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co"])
    pull_url_https_only: bool = True
    pull_concurrency: int = Field(default=2, ge=1, le=16)
    pull_chunk_size_mb: int = Field(default=64, ge=1, le=1024)
    per_stream_max_bytes: int = Field(default=107_374_182_400, ge=1)


class MonitorConfig(BaseModel):
    """Runtime-mutable: live inference monitor (tok/s + progress + live output).

    enabled is an ops kill-switch; poll_interval_s is the single-poller /slots
    cadence (one reader regardless of FE client count). The remaining tuning
    (smoothing, stall thresholds, text-tail size) are module-level constants in
    live_monitor.py — not speculative config surface.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    poll_interval_s: float = Field(default=1.0, gt=0.0, le=60.0)


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
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)


class TurbohaulConfig(BaseModel):
    """Full config = boot + runtime, used for yaml load/save."""

    model_config = ConfigDict(extra="forbid")

    server: ServerConfig
    storage: StorageConfig
    runtime: RuntimePathsConfig
    ui: UIConfig
    queue: QueueConfig
    pull: PullConfig
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)

    def split(self) -> tuple[BootConfig, RuntimeConfig]:
        boot = BootConfig(
            server=self.server,
            storage=self.storage,
            runtime=self.runtime,
            ui=self.ui,
        )
        runtime = RuntimeConfig(queue=self.queue, pull=self.pull, monitor=self.monitor)
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
    "TURBOHAUL_MAX_CONSECUTIVE_SAME_MODEL": ("queue", "max_consecutive_same_model", int),
    "TURBOHAUL_MAX_OTHER_MODEL_WAIT_S": ("queue", "max_other_model_wait_s", float),
    "TURBOHAUL_GPU_BACKEND": ("runtime", "gpu_backend", str),
}


def apply_env_overrides(cfg: TurbohaulConfig) -> TurbohaulConfig:
    """Apply TURBOHAUL_* env var overrides. Env beats yaml."""
    data: dict[str, Any] = cfg.model_dump()
    for env_key, (section, field, cast) in _ENV_MAP.items():
        v = os.environ.get(env_key)
        if v is not None:
            data[section][field] = cast(v)
    return TurbohaulConfig(**data)
