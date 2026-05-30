"""Inference backend abstraction layer.

Supports multiple inference engines behind a unified interface:
- ``llama.cpp`` — llama-server subprocess (CUDA/Metal/Vulkan via GGUF)
- ``mlx`` — mlx-lm HTTP server (Apple Silicon MLX framework)

Each backend implements:
- ``spawn()`` — start the inference subprocess
- ``wait_healthy()`` — poll until the server responds
- ``teardown()`` — SIGTERM the process group
- ``completion_fn()`` — HTTP function for chat completions

The SidecarHandle is shared across backends so the manager layer
does not need backend-specific logic for port/PID tracking.
"""

from turbohaul.backends.base import BackendInterface, SidecarHandle, SpawnRequest
from turbohaul.backends.llamacpp import LlamaCppBackend
from turbohaul.backends.mlx import MLXBackend

__all__ = [
    "BackendInterface",
    "SidecarHandle",
    "SpawnRequest",
    "LlamaCppBackend",
    "MLXBackend",
]
