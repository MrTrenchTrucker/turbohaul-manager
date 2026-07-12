"""End-to-end smoke test for the MLX backend.

Boots a *real* ``mlx_lm server`` through Turbohaul's ``mlx_spawn`` path and
verifies it serves the OpenAI-compatible ``/v1/chat/completions`` endpoint (the
same endpoint ``manager.py`` proxies to for completion). This is a REAL-engine
test: it loads an MLX model and spawns a process, so it is opt-in and skipped
unless all of:

  * running on macOS / Apple Silicon (``mlx_lm`` is Metal-only), and
  * ``MLX_E2E=1`` is set in the environment (explicit opt-in), and
  * the test is selected (e.g. ``pytest -m e2e``).

The model + python are taken from the environment so the test runs on ANY
Apple Silicon Mac, not just the author's machine:

  MLX_E2E_MODEL_REPO  HF repo id  (default: mlx-community/Llama-3.2-1B-Instruct-4bit)
  MLX_E2E_MODEL_PATH  local MLX dir (takes precedence over MODEL_REPO)
  MLX_E2E_PYTHON      python with mlx_lm importable (default: sys.executable)

The spawned server is ALWAYS terminated after the test (no leaked process).
The model is cached by mlx_lm under ``~/.cache/huggingface`` (conventional for
HF tools) and is NOT deleted by default; set ``MLX_E2E_CLEAN_CACHE=1`` to wipe
the downloaded repo after the run.

Run:
  MLX_E2E=1 pytest -m e2e tests/e2e_mlx_smoke.py
  MLX_E2E=1 MLX_E2E_MODEL_PATH=/path/to/local/mlx/dir pytest -m e2e tests/e2e_mlx_smoke.py
Or manually:
  MLX_E2E=1 python tests/e2e_mlx_smoke.py
"""
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import httpx
import pytest

from turbohaul.mlx_spawn import mlx_spawn

DEFAULT_MODEL_REPO = "mlx-community/Llama-3.2-1B-Instruct-4bit"
PORT = 11577


def _mlx_platform_ok() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# Skip on non-Apple-Silicon, and unless explicitly opted in via MLX_E2E=1.
# (String conditions are evaluated by pytest against this module's globals.)
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        "'MLX_E2E' not in os.environ",
        reason="real-engine MLX test is opt-in: set MLX_E2E=1",
    ),
    pytest.mark.skipif(
        "not _mlx_platform_ok()",
        reason="mlx_lm is Apple Silicon / macOS only",
    ),
]


def _resolve_model() -> tuple[str, str]:
    """Return (model_repo, model_path) from the environment."""
    model_path = os.environ.get("MLX_E2E_MODEL_PATH", "").strip()
    model_repo = os.environ.get("MLX_E2E_MODEL_REPO", DEFAULT_MODEL_REPO).strip()
    return model_repo, model_path


def _resolve_python() -> Path:
    py = os.environ.get("MLX_E2E_PYTHON", "").strip()
    return Path(py) if py else Path(sys.executable)


def _hf_cache_dir_for(repo_id: str) -> Path:
    # ~/.cache/huggingface/hub/models--<owner>--<name>
    dashed = "models--" + repo_id.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / dashed


def _run() -> None:
    model_repo, model_path = _resolve_model()
    python = _resolve_python()
    print(f"[e2e] python={python} model_repo={model_repo!r} model_path={model_path!r}")

    handle = mlx_spawn(
        port=PORT,
        model_tag="mlx-e2e-smoke",
        model_repo=model_repo,
        model_path=model_path,
        mlx_flags={"use_default_chat_template": True, "max_tokens": 256},
        python_binary=python,
    )
    print(f"[e2e] spawned pid={handle.pid} port={handle.port}")

    try:
        # Use the SAME readiness endpoint the manager uses (wait_until_healthy
        # polls /health, which mlx_lm serves as {"status":"ok"}).
        base = f"http://127.0.0.1:{PORT}"
        healthy = False
        for _ in range(90):
            try:
                r = httpx.get(f"{base}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") in (
                    "ok",
                    "ready",
                    "healthy",
                    "loaded",
                ):
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(1)
        if not healthy:
            raise SystemExit("[e2e] FAILED health wait (/health)")

        # Completion via the SAME endpoint manager.py proxies to. mlx_lm server
        # routes by the request `model` field (re-resolves it as a HF repo), so
        # the manager rewrites `model` to the backend identity (handle.model_id).
        # Replicate that here: model = the --model arg.
        model_id = model_path or model_repo
        resp = httpx.post(
            f"{base}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say hi in 5 words."}],
                "max_tokens": 32,
            },
            timeout=120,
        )
        print(f"[e2e] /v1/chat/completions -> {resp.status_code}")
        if resp.status_code != 200:
            raise SystemExit(f"[e2e] FAILED completion: {resp.text[:500]}")
        content = resp.json()["choices"][0]["message"]["content"]
        print(f"[e2e] model replied: {content!r}")
    finally:
        # Always tear down the spawned server (killpg via start_new_session).
        try:
            handle.proc.terminate()
            handle.proc.wait(timeout=10)
        except Exception:
            try:
                handle.proc.kill()
            except Exception:
                pass

    # Optional model-cache cleanup (off by default — HF caches are meant to
    # persist; re-downloading ~400MB every run is wasteful).
    cache = _hf_cache_dir_for(model_repo)
    if os.environ.get("MLX_E2E_CLEAN_CACHE") and not model_path:
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
            print(f"[e2e] removed model cache: {cache}")
    elif not model_path:
        print(
            f"[e2e] NOTE: model cached at {cache} (HF convention — kept between runs).\n"
            f"        To free it: re-run with MLX_E2E_CLEAN_CACHE=1, or `rm -rf {cache}`"
        )

    print("[e2e] PASS — MLX model served through Turbohaul's spawn path")


def test_mlx_e2e_smoke() -> None:
    """Boot a real MLX model via mlx_spawn and verify chat completions."""
    _run()


if __name__ == "__main__":
    _run()
