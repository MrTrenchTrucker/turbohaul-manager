"""Base interface for inference backends.

Every backend (llama.cpp, MLX, etc.) implements these methods so the
manager's worker_loop can drive any engine through a uniform contract.

SidecarHandle is shared across backends — it wraps the subprocess + port +
model_tag identity. The manager uses this for port assignment, PID tracking,
and idle-hot-holder logic without knowing which backend is active.
"""
import abc
import dataclasses
import subprocess
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class SidecarHandle:
    """Wraps a running inference subprocess + its identity.

    Shared across all backends. The manager uses this for:
    - Port-based routing (handle.port → HTTP target)
    - PID tracking (state.sqlite bookkeeping)
    - Process-lifetime management (alive check, teardown)
    """

    proc: subprocess.Popen
    port: int
    model_tag: str
    spawned_at: float = dataclasses.field(default_factory=time.monotonic)
    activated_at: float | None = None
    # The model identifier the sidecar process was started with.
    # For llama.cpp this is the same as model_tag (ignored by llama-server).
    # For MLX this is the local path or HF repo ID — must match what
    # mlx_lm.server was given via --model, since mlx_lm uses it as a
    # lookup key and will try to fetch from HuggingFace if it doesn't match.
    sidecar_model_id: str = ""

    @property
    def pid(self) -> int:
        return self.proc.pid

    def is_alive(self) -> bool:
        return self.proc.poll() is None


@dataclasses.dataclass
class SpawnRequest:
    """Parameters needed to spawn a backend subprocess.

    ``binary`` / ``gguf_path`` / ``argv_flags`` are llama.cpp-specific.
    ``model_path`` is MLX-specific. Each backend uses only the fields
    relevant to its engine.
    """

    # === llama.cpp (llama-server) ===
    binary: Path | None = None
    gguf_path: Path | None = None
    argv_flags: list[str] = dataclasses.field(default_factory=list)
    binary_fd: int | None = None

    # === MLX (mlx-lm) ===
    model_path: Path | None = None
    model_repo: str = ""
    python_binary: Path | None = None
    mlx_flags: list[str] = dataclasses.field(default_factory=list)

    # === Shared ===
    port: int = 11500
    model_tag: str = ""


class BackendInterface(abc.ABC):
    """Abstract interface every inference backend must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g. 'llama.cpp', 'mlx')."""

    @abc.abstractmethod
    def spawn(self, req: SpawnRequest) -> SidecarHandle:
        """Spawn the inference subprocess.

        Returns a SidecarHandle wrapping the live process.
        """

    @abc.abstractmethod
    async def wait_healthy(
        self,
        port: int,
        timeout_s: float,
        poll_interval_s: float = 2.0,
    ) -> bool:
        """Poll until the server becomes healthy, or timeout.

        Returns True on healthy, False on timeout.
        """

    @abc.abstractmethod
    async def teardown(
        self,
        handle: SidecarHandle,
        drained_window_s: float,
        is_active: bool,
        cold_window_s: float = 5.0,
    ) -> tuple[bool, str]:
        """SIGTERM the process group → wait → SIGKILL on timeout.

        Returns (success, status_str).
        """

    @abc.abstractmethod
    def make_completion_fn(
        self,
        base_url: str,
    ) -> Callable[..., Awaitable[dict | None]]:
        """Build an async completion function that POSTs to the backend.

        ``base_url`` is ``http://127.0.0.1:<port>``.

        Returns a callable with signature:
            async def complete(slot, handle) -> dict | None
        """

    @abc.abstractmethod
    def health_endpoint(self) -> str:
        """HTTP path for health checks (e.g. '/health', '/v1/models')."""
