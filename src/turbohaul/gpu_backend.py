"""Pluggable GPU backend abstraction.

Replaces hardcoded nvidia-smi coupling with a Protocol-based backend
that supports both NVIDIA (nvidia-smi) and AMD ROCm (rocm-smi) GPUs.

Usage:
    from turbohaul.gpu_backend import get_backend, set_backend, reset_backend

    backend = get_backend()          # auto-detect on first call
    set_backend("rocm")             # force ROCm
    reset_backend()                 # clear singleton cache

Backward-compat: existing callers using nvidia_smi_runner callable injection
still work -- the default runner delegates to the active backend.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Protocol

log = logging.getLogger(__name__)


class GPUBackend(Protocol):
    """Structural type for GPU monitoring backends."""

    def get_gpu_free_mib(self) -> list[int] | None:
        """Free MiB per GPU device. None if probe unavailable."""
        ...

    def get_gpu_total_mib(self) -> list[int] | None:
        """Total MiB per GPU device. None if probe unavailable."""
        ...

    def get_gpu_used_mib(self) -> int | None:
        """Used MiB on GPU 0. None if probe unavailable."""
        ...

    def scan_compute_apps(self) -> list[dict]:
        """Return [{pid, used_memory_mib}, ...] of GPU compute processes."""
        ...


class NvidiaBackend:
    """NVIDIA GPU backend via nvidia-smi (CSV query format)."""

    def __init__(self) -> None:
        self._path = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"

    def _query(self, query: str) -> str | None:
        try:
            return subprocess.check_output(
                [self._path, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return None

    def _parse_csv_ints(self, raw: str | None) -> list[int] | None:
        if not raw:
            return None
        vals: list[int] = []
        for line in raw.strip().splitlines():
            try:
                vals.append(int(line.strip().split(",")[0].strip()))
            except (ValueError, IndexError):
                continue
        return vals or None

    def get_gpu_free_mib(self) -> list[int] | None:
        return self._parse_csv_ints(self._query("memory.free"))

    def get_gpu_total_mib(self) -> list[int] | None:
        return self._parse_csv_ints(self._query("memory.total"))

    def get_gpu_used_mib(self) -> int | None:
        vals = self._parse_csv_ints(self._query("memory.used"))
        return vals[0] if vals else None

    def scan_compute_apps(self) -> list[dict]:
        try:
            out = subprocess.check_output(
                [
                    self._path,
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return []
        apps: list[dict] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    apps.append({"pid": int(parts[0]), "used_memory_mib": int(parts[1])})
                except ValueError:
                    continue
        return apps


class RocmBackend:
    """AMD ROCm GPU backend via rocm-smi (JSON query format).

    Uses --showmeminfo vram for VRAM stats and --showcomputeapps for processes.
    """

    def __init__(self) -> None:
        self._path = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"

    def _run(self, *args: str) -> str | None:
        try:
            result = subprocess.run(
                [self._path, *args],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            return result.stdout
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return None

    def _parse_vram_json(self, raw: str | None) -> list[dict[str, int]] | None:
        """Parse rocm-smi --showmeminfo vram JSON output."""
        if not raw:
            return None
        try:
            import json
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        results: list[dict[str, int]] = []
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, dict) and "vram" in val:
                    vram = val["vram"]
                    if isinstance(vram, dict):
                        total = vram.get("total", 0)
                        used = vram.get("used", 0)
                        try:
                            results.append({
                                "total": int(total) // (1024 * 1024),
                                "used": int(used) // (1024 * 1024),
                                "free": (int(total) - int(used)) // (1024 * 1024),
                            })
                        except (ValueError, TypeError):
                            continue
        return results or None

    def get_gpu_free_mib(self) -> list[int] | None:
        raw = self._run("--showmeminfo", "vram", "--json")
        devs = self._parse_vram_json(raw)
        if devs is None:
            return None
        return [d["free"] for d in devs]

    def get_gpu_total_mib(self) -> list[int] | None:
        raw = self._run("--showmeminfo", "vram", "--json")
        devs = self._parse_vram_json(raw)
        if devs is None:
            return None
        return [d["total"] for d in devs]

    def get_gpu_used_mib(self) -> int | None:
        free = self.get_gpu_free_mib()
        total = self.get_gpu_total_mib()
        if free and total:
            return total[0] - free[0]
        return None

    def scan_compute_apps(self) -> list[dict]:
        raw = self._run("--showpids")
        if not raw or "error" in raw.lower() or "usage" in raw.lower() or "ambiguous" in raw.lower():
            return []
        apps: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("/") or "PID" in line or "---" in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    apps.append({"pid": int(parts[0]), "used_memory_mib": 0})
                except ValueError:
                    continue
        return apps


def detect_backend(preferred: str = "auto") -> NvidiaBackend | RocmBackend | None:
    """Detect available GPU backend.

    preferred: "auto" | "nvidia" | "rocm"
      - auto: probe nvidia-smi first, then rocm-smi
      - nvidia: force NVIDIA, return None if unavailable
      - rocm: force AMD, return None if unavailable
    """
    if preferred == "nvidia":
        b = NvidiaBackend()
        if b.get_gpu_free_mib() is not None or b.scan_compute_apps():
            return b
        return None
    if preferred == "rocm":
        b = RocmBackend()
        if b.get_gpu_free_mib() is not None or b.scan_compute_apps():
            return b
        return None
    # auto
    for cls in (NvidiaBackend, RocmBackend):
        b = cls()
        if b.get_gpu_free_mib() is not None or b.scan_compute_apps():
            return b
    return None


_backend: GPUBackend | None = None


def get_backend() -> GPUBackend | None:
    """Return the singleton GPU backend. Auto-initializes on first call."""
    global _backend
    if _backend is None:
        _backend = detect_backend()
    return _backend


def set_backend(backend_or_name: GPUBackend | str | None) -> None:
    """Override the singleton GPU backend. None disables GPU probing."""
    global _backend
    if isinstance(backend_or_name, str):
        _backend = detect_backend(backend_or_name)
    else:
        _backend = backend_or_name


def reset_backend() -> None:
    """Clear the cached backend (forces re-detection on next get_backend call)."""
    global _backend
    _backend = None
