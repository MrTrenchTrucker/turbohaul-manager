"""MLX backend spawner (Apple Silicon only).

Spawns ``python -m mlx_lm.server`` as a supervised subprocess and returns a
``subprocess_mgr.SidecarHandle`` -- the SAME shape the llama.cpp path uses, so
the rest of the manager (health-wait, teardown, completion) stays backend-
agnostic.

MLX is Apple Silicon only. The spawn refuses to start on non-Darwin / non-arm64
hosts with a clear error. The completion path is NOT handled here: mlx-lm serves
the same OpenAI-compatible ``/v1/chat/completions`` endpoint as llama-server, so
the manager's existing httpx completion proxy is reused as-is.

Security: mlx_server_flags are mapped through a CLOSED allowlist (SAFE_MLX_FLAGS)
before becoming CLI argv, preventing flag-injection through a manifest. The child
process environment has PYTHONPATH popped (a stale PYTHONPATH can shadow stdlib
``queue`` and hang mlx_lm.server -- see pythonpath-shadow-bug).
"""
import logging
import os
import platform
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from turbohaul.subprocess_mgr import SidecarHandle


log = logging.getLogger(__name__)


# Closed allowlist of mlx_lm.server CLI flags. mlx flags are argparse-style
# (--snake-case). We validate type AND membership to block injection.
SAFE_MLX_FLAGS: dict[str, Any] = {
    # Model loading
    "adapter_path": str,
    "draft_model": str,
    "num_draft_tokens": int,
    "trust_remote_code": bool,
    "chat_template": str,
    "use_default_chat_template": bool,
    "chat_template_args": str,
    # Generation
    "max_tokens": int,
    "temp": float,
    "top_p": float,
    "top_k": int,
    "min_p": float,
    # Server
    "host": str,
    "port": int,
    "allowed_origins": str,
    # Performance
    "decode_concurrency": int,
    "prompt_concurrency": int,
    "prefill_step_size": int,
    "prompt_cache_size": int,
    "prompt_cache_bytes": int,
    "pipeline": bool,
    # Debug
    "log_level": str,
}


def validate_mlx_flags(flags: dict[str, Any]) -> None:
    """Validate MLX server flags against the closed allowlist + type check."""
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

    snake_case -> --snake-case with value. Booleans: True -> --flag (no value),
    False -> omitted.
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


def is_mlx_available() -> bool:
    """True only on Apple Silicon macOS with the mlx package importable."""
    if platform.system() != "Darwin":
        return False
    if platform.machine() != "arm64":
        return False
    try:
        import mlx  # noqa: F401
    except ImportError:
        return False
    return True


def _check_mlx_preconditions(python: Path) -> None:
    """Refuse to spawn if MLX prerequisites are not met. Raises RuntimeError."""
    if platform.system() != "Darwin":
        raise RuntimeError(
            f"mlx backend requires macOS (Darwin), got {platform.system()}. "
            f"Use backend: llama.cpp on this host."
        )
    if platform.machine() != "arm64":
        raise RuntimeError(
            f"mlx backend requires Apple Silicon (arm64), got "
            f"{platform.machine()}. Use backend: llama.cpp on this host."
        )
    # Preflight: the target python must be able to import mlx_lm.
    probe = subprocess.run(
        [str(python), "-c", "import mlx_lm"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"mlx backend: 'mlx_lm' not importable in {python}. "
            f"Install with: conda install -c conda-forge mlx-lm  (or pip install mlx-lm)"
        )


def mlx_spawn(
    port: int,
    model_tag: str,
    model_repo: str,
    model_path: str,
    mlx_flags: dict[str, Any],
    python_binary: Path | None = None,
    popen_factory: Callable[..., subprocess.Popen] | None = None,
) -> SidecarHandle:
    """Spawn an mlx_lm.server subprocess.

    Returns a ``SidecarHandle`` compatible with the llama.cpp path. The manager
    uses ``handle.port`` / ``handle.pid`` / ``handle.is_alive()`` and points its
    httpx completion proxy at ``http://127.0.0.1:<port>/v1/chat/completions``.

    Raises RuntimeError if preconditions fail (non-Apple-Silicon, mlx_lm missing).
    """
    _check_mlx_preconditions(python_binary or Path(sys.executable))

    python = python_binary or Path(sys.executable)

    # No hardcoded default model: the model MUST come from the manifest, either
    # as a HuggingFace repo id (model_repo) or a local dir (model_path). If the
    # caller supplies neither we refuse to spawn rather than silently picking one.
    if model_path:
        model_arg = model_path
    elif model_repo:
        model_arg = model_repo
    else:
        raise RuntimeError(
            "mlx backend: manifest needs model_repo (HF repo id) or "
            "model_path (local dir) -- no model is built in"
        )

    # Validate + map flags through the closed allowlist BEFORE they become argv.
    validate_mlx_flags(mlx_flags)
    extra_argv = mlx_flags_to_argv(mlx_flags)

    cmd = [
        str(python),
        "-m", "mlx_lm", "server",
        "--model", model_arg,
        "--host", "127.0.0.1",
        "--port", str(port),
        *extra_argv,
    ]

    log.info(
        "spawning mlx_lm.server pid=? port=%d model=%s python=%s",
        port, model_tag, python,
    )

    # pythonpath-shadow-bug: a stale PYTHONPATH in the inherited env can shadow
    # stdlib `queue` and hang mlx_lm.server. Pop it for the child.
    child_env = dict(os.environ)
    child_env.pop("PYTHONPATH", None)

    factory = popen_factory or subprocess.Popen
    proc = factory(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # setsid - own process group -> killpg works
        env=child_env,
    )

    return SidecarHandle(
        proc=proc,
        port=port,
        model_tag=model_tag,
        parallel=1,  # mlx-lm is single-slot per process
        model_id=model_arg,  # mlx_lm server routes by request `model`; match --model
    )
