"""Safety guardrails: VRAM / RAM / IO-wait / CPU-load pre-spawn checks.

Design intent: mirror Ollama's safety posture so
Turbohaul-Manager refuses to spawn a sidecar when the host cannot safely
run it. Each gate is tunable via RuntimeConfig.queue.safety_*; the
all_safety_gates aggregator returns the list of failures so the manager
can surface them on the loading_fail audit + completion_future error.

All gates degrade gracefully: if the underlying probe is unavailable
(nvidia-smi missing in dev / /proc unreadable in some containers) the
gate returns "passed-no-probe" rather than blocking the spawn. The operator can
disable the whole subsystem via runtime.queue.safety_enabled = False.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


log = logging.getLogger(__name__)

_NVIDIA_SMI_PATH = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"


@dataclass(frozen=True)
class GateResult:
    name: str
    ok: bool
    detail: str  # human-readable; included in audit + error surfaced to caller


def _read_meminfo_kib() -> dict[str, int]:
    """Parse /proc/meminfo into a dict keyed by field name (values in KiB)."""
    try:
        text = Path("/proc/meminfo").read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return {}
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        name = parts[0].strip()
        val = parts[1].strip().split()
        if val and val[0].isdigit():
            out[name] = int(val[0])
    return out


def check_free_ram(min_free_mib: int) -> GateResult:
    """Refuse spawn if /proc/meminfo MemAvailable < min_free_mib."""
    info = _read_meminfo_kib()
    avail_kib = info.get("MemAvailable")
    if avail_kib is None:
        return GateResult("ram", True, "passed-no-probe")
    avail_mib = avail_kib // 1024
    if avail_mib < min_free_mib:
        return GateResult(
            "ram", False,
            f"only {avail_mib} MiB free; need >= {min_free_mib} MiB",
        )
    return GateResult("ram", True, f"{avail_mib} MiB free")


def check_load_avg(max_per_core: float) -> GateResult:
    """Refuse spawn if 1-min load avg per logical core > max_per_core."""
    try:
        load1 = os.getloadavg()[0]
    except (OSError, AttributeError):
        return GateResult("cpu_load", True, "passed-no-probe")
    cpus = os.cpu_count() or 1
    per_core = load1 / cpus
    if per_core > max_per_core:
        return GateResult(
            "cpu_load", False,
            f"1min-load-per-core={per_core:.2f} > max {max_per_core:.2f}",
        )
    return GateResult(
        "cpu_load", True, f"1min-load-per-core={per_core:.2f}",
    )


def _read_stat_iowait_jiffies() -> tuple[int, int] | None:
    """Return (total_jiffies, iowait_jiffies) from /proc/stat first cpu line.

    Returns None if /proc/stat is unavailable / malformed.
    """
    try:
        text = Path("/proc/stat").read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    first = text.splitlines()[0] if text else ""
    parts = first.split()
    if len(parts) < 6 or parts[0] != "cpu":
        return None
    try:
        # cpu  user nice system idle iowait irq softirq steal guest guest_nice
        nums = [int(x) for x in parts[1:]]
    except ValueError:
        return None
    total = sum(nums)
    iowait = nums[4] if len(nums) > 4 else 0
    return total, iowait


def check_iowait(max_percent: float, sample_window_s: float = 0.4) -> GateResult:
    """Sample /proc/stat over sample_window_s; refuse if iowait% > max_percent."""
    sample_a = _read_stat_iowait_jiffies()
    if sample_a is None:
        return GateResult("iowait", True, "passed-no-probe")
    time.sleep(sample_window_s)
    sample_b = _read_stat_iowait_jiffies()
    if sample_b is None:
        return GateResult("iowait", True, "passed-no-probe-second")
    d_total = sample_b[0] - sample_a[0]
    d_iowait = sample_b[1] - sample_a[1]
    if d_total <= 0:
        return GateResult("iowait", True, "passed-zero-delta")
    pct = 100.0 * d_iowait / d_total
    if pct > max_percent:
        return GateResult(
            "iowait", False,
            f"iowait {pct:.1f}% > max {max_percent:.1f}%",
        )
    return GateResult("iowait", True, f"iowait {pct:.1f}%")


def _read_free_vram_mib() -> int | None:
    """Query nvidia-smi for GPU 0 free memory in MiB. None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                _NVIDIA_SMI_PATH,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i", "0",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    try:
        return int(line.strip().split(",")[0].strip())
    except (ValueError, IndexError):
        return None


def check_free_vram(min_free_mib: int, manifest_expected_bytes: int = 0) -> GateResult:
    """Refuse spawn if free VRAM < max(min_free_mib, manifest_expected/1024).

    manifest_expected_bytes is from manifest.expected_vram_bytes (if known).
    The gate requires headroom = max(min_free_mib, expected_mib) so we don't
    OOM mid-load.
    """
    free_mib = _read_free_vram_mib()
    if free_mib is None:
        return GateResult("vram", True, "passed-no-probe")
    expected_mib = manifest_expected_bytes // (1024 * 1024)
    threshold = max(min_free_mib, expected_mib)
    if free_mib < threshold:
        return GateResult(
            "vram", False,
            f"only {free_mib} MiB free; need >= {threshold} MiB "
            f"(min_floor={min_free_mib}, manifest={expected_mib})",
        )
    return GateResult(
        "vram", True,
        f"{free_mib} MiB free (threshold {threshold})",
    )


# --- KV-cache fit estimator ( LEAD fix per review synthesis) ---
# Closed-form pre-spawn check that refuses when (model body + KV cache + overhead)
# would not fit free VRAM. Independent of (and complementary to) check_free_vram's
# manifest-driven expected_vram_bytes check — this one is computed from ctx_size
# directly, so user can bump ctx_size in the manifest WITHOUT manually re-tuning
# expected_vram_bytes and the gate still catches over-commit.
#
# Empirical calibration (Qwen3.6-27B Q4_K_XL):
#   17 GiB GGUF body + ~150 KB/token f16 KV → ~9.5 GiB KV at 64K ctx.
# Generalized: ~9 KB/token per GiB of model body at f16. Quant halves/quarters
# proportionally. Overhead floor = 1 GiB for activations + scratch.

_KV_QUANT_SCALE: dict[str, float] = {
    "f32": 2.0,
    "f16": 1.0,
    "bf16": 1.0,
    "q8_0": 0.5,
    "q4_0": 0.25,
    "q4_1": 0.25,
    "iq4_nl": 0.25,
    "q5_0": 0.32,
    "q5_1": 0.32,
    "turbo2": 0.125,
    "turbo3": 0.1875,
    "turbo4": 0.25,
}


def estimate_kv_cache_mib(
    ctx_size: int,
    gguf_size_bytes: int,
    kv_cache_quant: str = "f16",
) -> int:
    """Closed-form KV-cache size estimate in MiB.

    Scales linearly with ctx_size and with model body size (gguf bytes), then
    scaled by quant factor for cache_type_k/cache_type_v.
    """
    if ctx_size <= 0 or gguf_size_bytes <= 0:
        return 0
    gguf_mib = gguf_size_bytes // (1024 * 1024)
    # f16 baseline: ~9 KB/token per GiB of model body. Per-token in KB:
    bytes_per_token_kb_f16 = (9 * gguf_mib) // 1024
    scale = _KV_QUANT_SCALE.get(kv_cache_quant.lower(), 1.0)
    bytes_per_token_kb = int(bytes_per_token_kb_f16 * scale)
    total_kib = bytes_per_token_kb * ctx_size  # KB total
    return total_kib // 1024  # MiB


def check_kv_cache_fit(
    ctx_size: int,
    gguf_size_bytes: int,
    overhead_mib: int = 1024,
    kv_cache_quant: str = "f16",
) -> GateResult:
    """Refuse spawn if (body + KV-cache + overhead) > free VRAM.

    Closed-form: doesn't trust the manifest's hand-tuned expected_vram_bytes;
    derives the prediction from ctx_size + gguf_size_bytes + quant. This is
    the load-bearing change for user-programmable ctx_size — when a user
    bumps ctx_size from 4096 to 65536 in the manifest, this gate refuses
    the spawn if the resulting KV cache won't fit on local hardware
    (regardless of whether expected_vram_bytes was hand-tuned to match).
    """
    if ctx_size <= 0 or gguf_size_bytes <= 0:
        # Insufficient info to predict — pass through to other gates.
        return GateResult("kv_cache_fit", True, "passed-insufficient-input")
    free_mib = _read_free_vram_mib()
    if free_mib is None:
        return GateResult("kv_cache_fit", True, "passed-no-probe")
    gguf_mib = gguf_size_bytes // (1024 * 1024)
    kv_mib = estimate_kv_cache_mib(ctx_size, gguf_size_bytes, kv_cache_quant)
    total_mib = gguf_mib + kv_mib + overhead_mib
    if total_mib > free_mib:
        return GateResult(
            "kv_cache_fit", False,
            f"need ~{total_mib} MiB "
            f"(body={gguf_mib} + KV@ctx{ctx_size}={kv_mib} "
            f"[{kv_cache_quant}] + overhead={overhead_mib}); "
            f"only {free_mib} MiB free",
        )
    return GateResult(
        "kv_cache_fit", True,
        f"need ~{total_mib} MiB / {free_mib} free "
        f"(body={gguf_mib} KV={kv_mib} overhead={overhead_mib} quant={kv_cache_quant})",
    )


def all_safety_gates(
    *,
    min_free_ram_mib: int,
    min_free_vram_mib: int,
    max_load_per_core: float,
    max_iowait_percent: float,
    manifest_expected_vram_bytes: int = 0,
    iowait_sample_window_s: float = 0.4,
    ctx_size: int = 0,
    gguf_size_bytes: int = 0,
    kv_cache_overhead_mib: int = 1024,
    kv_cache_quant: str = "f16",
) -> list[GateResult]:
    """Run all gates; return their results in order. Caller decides on failures.

    A "fail" in any GateResult.ok = False entry is a refusal signal. The
    aggregator does not short-circuit -- collecting all gates' status gives
    the audit + completion_future error a complete picture.

    The new kv_cache_fit gate () refuses spawn when the predicted
    KV cache + model body + overhead exceeds free VRAM. When ctx_size or
    gguf_size_bytes is unknown (0), the gate passes (caller still has the
    other VRAM gate via manifest_expected_vram_bytes).
    """
    return [
        check_free_ram(min_free_ram_mib),
        check_free_vram(min_free_vram_mib, manifest_expected_vram_bytes),
        check_kv_cache_fit(
            ctx_size, gguf_size_bytes,
            overhead_mib=kv_cache_overhead_mib,
            kv_cache_quant=kv_cache_quant,
        ),
        check_load_avg(max_load_per_core),
        check_iowait(max_iowait_percent, sample_window_s=iowait_sample_window_s),
    ]
