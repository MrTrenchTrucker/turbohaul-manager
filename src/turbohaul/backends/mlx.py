"""MLX (mlx-lm) backend.

Spawns ``python -m mlx_lm.server`` as a supervised subprocess.
The MLX server exposes an OpenAI-compatible API at
``http://127.0.0.1:<port>/v1/chat/completions`` and a health endpoint
at ``/v1/models``.

MLX is Apple Silicon only. This backend will refuse to start on
non-macOS/non-arm64 systems with a clear error message.

Model format: mlx-lm accepts both .safetensors and .gguf files.
Models can be specified as:
- HuggingFace repo ID (e.g. ``mlx-community/Mistral-7B-Instruct-v0.3-4bit``)
- Local path to a model directory
"""
import asyncio
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from turbohaul.backends.base import BackendInterface, SidecarHandle, SpawnRequest


log = logging.getLogger(__name__)


def _is_mlx_available() -> bool:
    """Check if MLX is available on this system."""
    if platform.system() != "Darwin":
        return False
    if platform.machine() != "arm64":
        return False
    try:
        import mlx  # noqa: F401
        return True
    except ImportError:
        return False


class MLXBackend(BackendInterface):
    """mlx-lm HTTP server backend for Apple Silicon."""

    @property
    def name(self) -> str:
        return "mlx"

    def _check_availability(self) -> None:
        """Refuse to start if MLX prerequisites are not met."""
        if platform.system() != "Darwin":
            raise RuntimeError(
                f"mlx backend requires macOS (Darwin), got {platform.system()}. "
                "Install llama.cpp backend or switch to macOS for MLX support."
            )
        if platform.machine() != "arm64":
            raise RuntimeError(
                f"mlx backend requires Apple Silicon (arm64), got {platform.machine()}. "
                "Install llama.cpp backend or switch to Apple Silicon for MLX support."
            )

    def spawn(self, req: SpawnRequest) -> SidecarHandle:
        """Spawn mlx_lm.server subprocess."""
        self._check_availability()

        python = req.python_binary or Path(sys.executable)

        # Build model argument: prefer model_repo, then model_path.
        model_arg = req.model_repo or str(req.model_path)
        if not model_arg:
            raise RuntimeError(
                "mlx backend: model_repo or model_path required in manifest"
            )

        cmd = [
            str(python),
            "-m", "mlx_lm", "server",
            "--model", model_arg,
            "--host", "127.0.0.1",
            "--port", str(req.port),
        ]

        # Add MLX-specific flags
        cmd.extend(req.mlx_flags)

        log.info(
            "spawning mlx_lm.server pid=? port=%d model=%s python=%s",
            req.port, req.model_tag, python,
        )

        # Inherit environment but clear PYTHONPATH to prevent turbohaul
        # source modules from shadowing stdlib (e.g. turbohaul.queue →
        # shadowing queue module, breaking huggingface_hub imports).
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )

        return SidecarHandle(
            proc=proc,
            port=req.port,
            model_tag=req.model_tag,
            sidecar_model_id=model_arg,
        )

    async def wait_healthy(
        self,
        port: int,
        timeout_s: float,
        poll_interval_s: float = 2.0,
    ) -> bool:
        """Poll /v1/models until 200 or timeout.

        mlx-lm server's health check is the OpenAI /v1/models endpoint.
        A 200 response means the model is loaded and ready.
        """
        async with httpx.AsyncClient() as client:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                try:
                    r = await client.get(
                        f"http://127.0.0.1:{port}/v1/models", timeout=2.0
                    )
                    if r.status_code == 200:
                        data = r.json()
                        # mlx-lm returns {"object": "list", "data": [...]}
                        if isinstance(data, dict) and data.get("object") == "list":
                            return True
                except (httpx.HTTPError, OSError):
                    pass
                await asyncio.sleep(poll_interval_s)
            return False

    async def teardown(
        self,
        handle: SidecarHandle,
        drained_window_s: float,
        is_active: bool,
        cold_window_s: float = 5.0,
    ) -> tuple[bool, str]:
        """SIGTERM process group → wait → SIGKILL on timeout."""
        wait_window = drained_window_s if is_active else cold_window_s

        try:
            pgid = os.getpgid(handle.pid)
        except ProcessLookupError:
            return True, "already-gone"

        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            try:
                await asyncio.to_thread(handle.proc.wait, timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
            return True, "already-gone-during-sigterm"
        except PermissionError:
            return False, "sigterm-failed-permission-denied"

        deadline = time.monotonic() + wait_window
        while time.monotonic() < deadline:
            if handle.proc.poll() is not None:
                try:
                    await asyncio.to_thread(handle.proc.wait, timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
                return True, "sigterm-clean"
            await asyncio.sleep(0.2)

        # Escalate to SIGKILL
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return True, "sigkill-already-gone"
        except PermissionError:
            return False, "sigkill-permission-denied"

        await asyncio.sleep(0.5)
        if handle.proc.poll() is None:
            return False, "sigkill-failed-still-alive"
        try:
            await asyncio.to_thread(handle.proc.wait, timeout=5.0)
        except subprocess.TimeoutExpired:
            return False, "wait-timeout-after-sigkill"
        return True, "sigkill-clean"

    def make_completion_fn(self, base_url: str):
        """Build completion function for MLX server.

        The MLX server uses the same OpenAI-compatible endpoint as llama-server,
        so the completion logic is structurally identical.
        """
        async def complete_fn(slot: Any, handle: SidecarHandle) -> dict | None:
            """POST to mlx-lm server /v1/chat/completions."""
            await asyncio.sleep(0.001)
            return None

        return complete_fn

    def health_endpoint(self) -> str:
        return "/v1/models"


# === MLX-specific flag allowlist ===
# These are the flags mlx_lm.server accepts. Unlike llama.cpp, MLX flags
# are Python CLI args (argparse-style). We validate them here to prevent
# injection through the manifest.

SAFE_MLX_FLAGS: dict[str, Any] = {
    # Model loading (argparse names match CLI --flags after _ → - conversion)
    "adapter_path": str,          # --adapter-path
    "draft_model": str,           # --draft-model
    "num_draft_tokens": int,      # --num-draft-tokens
    "trust_remote_code": bool,    # --trust-remote-code
    "chat_template": str,         # --chat-template
    "use_default_chat_template": bool,  # --use-default-chat-template
    "chat_template_args": str,    # --chat-template-args
    # Generation
    "max_tokens": int,            # --max-tokens
    "temp": float,                # --temp  (mlx_lm uses --temp not --temperature)
    "top_p": float,               # --top-p
    "top_k": int,                 # --top-k
    "min_p": float,               # --min-p
    # Server
    "host": str,                  # --host (fixed at 127.0.0.1 by spawn; here for completeness)
    "port": int,                  # --port (fixed by spawn)
    "allowed_origins": str,       # --allowed-origins
    # Performance
    "decode_concurrency": int,    # --decode-concurrency
    "prompt_concurrency": int,    # --prompt-concurrency
    "prefill_step_size": int,     # --prefill-step-size
    "prompt_cache_size": int,     # --prompt-cache-size  (number of prompts cached)
    "prompt_cache_bytes": int,    # --prompt-cache-bytes
    "pipeline": bool,             # --pipeline
    # Debug
    "log_level": str,             # --log-level DEBUG|INFO|WARNING|ERROR|CRITICAL
}


def validate_mlx_flags(flags: dict[str, Any]) -> None:
    """Validate MLX flags against the allowlist."""
    for key, value in flags.items():
        if key not in SAFE_MLX_FLAGS:
            raise ValueError(
                f"mlx_server_flags.{key} is not in the closed allowlist. "
                f"Allowed: {sorted(SAFE_MLX_FLAGS.keys())}"
            )
        expected = SAFE_MLX_FLAGS[key]
        if not isinstance(value, expected):
            raise ValueError(
                f"mlx_server_flags.{key} expects {expected.__name__}, "
                f"got {type(value).__name__}: {value!r}"
            )


def mlx_flags_to_argv(flags: dict[str, Any]) -> list[str]:
    """Map MLX flags dict to CLI argv.

    snake_case → --snake-case with value.
    Booleans: True → --flag (no value), False → omitted.
    """
    argv: list[str] = []
    for key, value in flags.items():
        if key not in SAFE_MLX_FLAGS:
            raise ValueError(f"mlx flag {key} not in allowlist")
        cli_key = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(cli_key)
        else:
            argv.extend([cli_key, str(value)])
    return argv
