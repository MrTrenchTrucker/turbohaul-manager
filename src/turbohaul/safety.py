"""Safety guardrails: VRAM / RAM / IO-wait / CPU-load pre-spawn checks.

Mirror Ollama's safety posture so Turbohaul-Manager refuses to spawn a
sidecar when the host cannot safely run it. Each gate is tunable via
RuntimeConfig.queue.safety_*; the
all_safety_gates aggregator returns the list of failures so the manager
can surface them on the loading_fail audit + completion_future error.

All gates degrade gracefully: if the underlying probe is unavailable
(nvidia-smi missing in dev / /proc unreadable in some containers) the
gate returns "passed-no-probe" rather than blocking the spawn. You can
disable the whole subsystem via runtime.queue.safety_enabled = False.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from turbohaul.gpu_backend import get_backend


log = logging.getLogger(__name__)


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


def _read_free_vram_all_mib() -> list[int] | None:
    """Free MiB for EVERY GPU device (one entry per GPU, index order).

    None if GPU backend is unavailable. Querying all rows makes the
    VRAM gates GPU-count agnostic: 1 card -> a 1-element list; N cards -> N elements.
    """
    backend = get_backend()
    if backend is None:
        return None
    return backend.get_gpu_free_mib()


def _read_free_vram_mib() -> int | None:
    """Back-compat: free MiB on GPU 0 (first device). None if unavailable."""
    vals = _read_free_vram_all_mib()
    return vals[0] if vals else None


def _read_total_vram_all_mib() -> list[int] | None:
    """Total VRAM MiB for EVERY GPU device (one entry per GPU, index order).

    Boot-time read -- total VRAM doesn't change at runtime.
    """
    backend = get_backend()
    if backend is None:
        return None
    return backend.get_gpu_total_mib()


def _vram_budget(split_mode: str = "layer", main_gpu: int = 0):
    """Return (free_for_fit, min_per_card_free, n_cards) for the spawn budget.

    llama.cpp's default with multiple visible GPUs is a LAYER split across all
    of them, so a multi-GPU spawn must be budgeted against the AGGREGATE free
    VRAM:
      split_mode in {layer,row,tensor} or absent -> spans ALL cards ->
        free_for_fit = sum(all), min_per_card_free = min(all).
      split_mode == 'none' -> single-GPU pin on main_gpu ->
        free_for_fit = min_per_card_free = free[main_gpu].
    Probe unavailable -> (None, None, 0); callers keep the historical
    degrade-open (parallel:1) / refuse-blind (parallel>1) doctrine. With a
    single physical GPU every branch collapses to that one card, so behaviour
    is identical to the legacy GPU0-only gate.
    """
    vals = _read_free_vram_all_mib()
    if not vals:
        return None, None, 0
    if (split_mode or "layer").lower() == "none":
        if 0 <= main_gpu < len(vals):
            idx = main_gpu
        else:
            log.warning(
                "vram_budget: split_mode=none main_gpu=%s out of range "
                "(%d GPU(s) visible) -- budgeting GPU0 instead",
                main_gpu, len(vals),
            )
            idx = 0
        return vals[idx], vals[idx], 1
    return sum(vals), min(vals), len(vals)


def check_free_vram(min_free_mib: int, manifest_expected_bytes: int = 0,
                    split_mode: str = "layer", main_gpu: int = 0) -> GateResult:
    """Refuse spawn if free VRAM < max(min_free_mib, manifest_expected/1024).

    GPU-count agnostic: a layer-split model (split_mode != 'none') is budgeted
    against the AGGREGATE free VRAM across all cards; a single-GPU pin
    (split_mode == 'none') against the main_gpu card only. With one physical
    GPU this collapses to the legacy GPU0-only behaviour.
    """
    free_fit, min_card, n = _vram_budget(split_mode, main_gpu)
    if free_fit is None:
        return GateResult("vram", True, "passed-no-probe")
    expected_mib = manifest_expected_bytes // (1024 * 1024)
    threshold = max(min_free_mib, expected_mib)
    if free_fit < threshold:
        return GateResult(
            "vram", False,
            f"only {free_fit} MiB free across {n} card(s); need >= {threshold} MiB "
            f"(min_floor={min_free_mib}, manifest={expected_mib})",
        )
    return GateResult(
        "vram", True,
        f"{free_fit} MiB free across {n} card(s) (min/card {min_card}, threshold {threshold})",
    )


# --- KV-cache fit estimator ---
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

# Marginal VRAM (MiB) per ADDITIONAL llama.cpp --parallel slot, on top of the
# single-slot baseline. ctx_size is the AGGREGATE KV window llama.cpp splits
# across N slots, so the ctx-linear scratch term already counts all N slots;
# this FLAT floor covers only the per-slot compute/attention buffers
# llama-server allocates per extra concurrent slot. CONSERVATIVE default (errs
# toward refusing) until MEASURED on the real 35b-moe via nvidia-smi at
# parallel:2 (cold-load, max of two samples 5s apart after first decode). Do
# NOT raise to ship parallel:3 without that live measurement.
PER_SLOT_COMPUTE_FLOOR_MIB = 256


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
    no_kv_offload: bool = False,
    parallel: int = 1,
    split_mode: str = "layer",
    main_gpu: int = 0,
    expected_vram_mib: int = 0,
    cpu_moe_offload: bool = False,
) -> GateResult:
    """Refuse spawn if (body + KV-cache + overhead) > free VRAM.

    Closed-form: doesn't trust the manifest's hand-tuned expected_vram_bytes;
    derives the prediction from ctx_size + gguf_size_bytes + quant. This is
    the load-bearing change for user-programmable ctx_size — when a user
    bumps ctx_size from 4096 to 65536 in the manifest, this gate refuses
    the spawn if the resulting KV cache won't fit on local hardware
    (regardless of whether expected_vram_bytes was hand-tuned to match).

    no_kv_offload (llama-server --no-kv-offload): when set, the KV cache lives
    in HOST RAM, not VRAM. The VRAM prediction then DROPS the KV term (only
    body + overhead must fit VRAM), and a complementary host-RAM-fit check
    ensures the KV cache fits free system RAM. Without this branch the gate
    over-counts the RAM-resident KV against VRAM and refuses high-ctx all-RAM
    configs that actually fit (e.g. 256K: ~17 GiB VRAM real vs ~23.7 GiB
    over-estimate). This is NOT a safety relaxation — the VRAM requirement is
    accurately lower and the freed requirement is re-checked against host RAM.

    M3: ``no_kv_offload: true`` is the CANONICAL manifest flag for this — it is
    what ``flags_to_argv`` emits as ``--no-kv-offload``. Note ``kv_offload:
    false`` is NOT equivalent here: false bools are omitted by ``flags_to_argv``
    (a no-op leaving KV in VRAM), so the gate must key only on ``no_kv_offload``.
    """
    if ctx_size <= 0 or gguf_size_bytes <= 0:
        # Insufficient info to predict — pass through to other gates.
        return GateResult("kv_cache_fit", True, "passed-insufficient-input")
    p = max(1, parallel)
    free_mib, _min_card, _n_cards = _vram_budget(split_mode, main_gpu)
    if free_mib is None:
        if p > 1:
            # A parallel:N config blind-spawned with no VRAM probe = guaranteed
            # OOM risk; refuse rather than degrade-open. parallel:1 keeps the
            # historical degrade-open doctrine (passed-no-probe).
            return GateResult(
                "kv_cache_fit", False,
                f"parallel={p} requires a VRAM probe; nvidia-smi unreadable, "
                f"refusing blind spawn",
            )
        return GateResult("kv_cache_fit", True, "passed-no-probe")
    gguf_mib = gguf_size_bytes // (1024 * 1024)
    kv_mib = estimate_kv_cache_mib(ctx_size, gguf_size_bytes, kv_cache_quant)
    # Marginal VRAM for the extra concurrent llama.cpp slots: a FLAT per-slot
    # floor. ctx_size//128 scratch is AGGREGATE (already counts all N slots) so
    # it is NOT multiplied; host-RAM KV (kv_unified) is shared so NOT multiplied.
    par_extra_mib = (p - 1) * PER_SLOT_COMPUTE_FLOOR_MIB
    par_note = (
        f"; parallel={p} (+{par_extra_mib} MiB per-slot scratch)" if p > 1 else ""
    )
    if cpu_moe_offload and expected_vram_mib > 0:
        # cpu_moe / n_cpu_moe offloads expert weights to HOST RAM. The closed-form
        # body=gguf term over-counts (it assumes every weight is GPU-resident), so for
        # expert-offload configs the operator's MEASURED expected_vram_bytes (already
        # the reduced on-GPU body PLUS its KV) is the authoritative VRAM footprint.
        # (live-E2E 2026-06-25: 35b n-cpu-moe @500K closed-form ~30.9GiB vs a proven
        # 19.4GiB.) ctx-bump safety for these configs is the operator's to maintain via
        # expected_vram_bytes; normal (non-offload) models keep the closed-form below.
        vram_need = expected_vram_mib + overhead_mib + par_extra_mib
        if vram_need > free_mib:
            return GateResult(
                "kv_cache_fit", False,
                f"need ~{vram_need} MiB VRAM (expected_vram={expected_vram_mib} "
                f"[cpu-moe measured] + overhead={overhead_mib}); only {free_mib} MiB "
                f"free VRAM" + par_note,
            )
        return GateResult(
            "kv_cache_fit", True,
            f"fits ~{vram_need} MiB (expected_vram={expected_vram_mib} [cpu-moe "
            f"measured]); {free_mib} MiB free" + par_note,
        )
    if no_kv_offload:
        # KV cache is in host RAM (--no-kv-offload). VRAM holds the model body
        # plus a ctx-scaled compute/attention scratch (NOT the KV cache).
        # Edge case: the flat overhead floor does NOT scale with
        # ctx, but the VRAM-side attention scratch grows with ctx even when the
        # KV is offloaded -- so add a conservative ctx-linear scratch term on
        # top of the floor. (qwopus @256K observed ~2.9 GiB VRAM over body;
        # this estimates ~3.1 GiB = 1024 floor + 262144//128.)
        vram_scratch_mib = overhead_mib + ctx_size // 128 + par_extra_mib
        vram_need = gguf_mib + vram_scratch_mib
        if vram_need > free_mib:
            return GateResult(
                "kv_cache_fit", False,
                f"need ~{vram_need} MiB VRAM "
                f"(body={gguf_mib} + scratch={vram_scratch_mib}; "
                f"KV@ctx{ctx_size}={kv_mib} [{kv_cache_quant}] in host RAM); "
                f"only {free_mib} MiB free VRAM" + par_note,
            )
        # Complementary host-RAM-fit: the KV cache must fit free system RAM.
        # M2: if MemAvailable is unreadable this sub-check is skipped, but the
        # independent check_free_ram() gate in all_safety_gates() reads the same
        # MemAvailable, so host-RAM exhaustion is still caught there.
        ram_avail_kib = _read_meminfo_kib().get("MemAvailable")
        if ram_avail_kib is not None:
            ram_avail_mib = ram_avail_kib // 1024
            if kv_mib > ram_avail_mib:
                return GateResult(
                    "kv_cache_fit", False,
                    f"KV@ctx{ctx_size}={kv_mib} MiB [{kv_cache_quant}] needs host "
                    f"RAM (--no-kv-offload) but only {ram_avail_mib} MiB free RAM",
                )
            ram_detail = f"ram_free={ram_avail_mib}"
        else:
            ram_detail = "ram_free=n/a"
        return GateResult(
            "kv_cache_fit", True,
            f"need ~{vram_need} MiB VRAM / {free_mib} free "
            f"(body={gguf_mib} scratch={vram_scratch_mib}; "
            f"KV={kv_mib} in host RAM, {ram_detail})" + par_note,
        )
    total_mib = gguf_mib + kv_mib + overhead_mib + par_extra_mib
    if total_mib > free_mib:
        return GateResult(
            "kv_cache_fit", False,
            f"need ~{total_mib} MiB "
            f"(body={gguf_mib} + KV@ctx{ctx_size}={kv_mib} "
            f"[{kv_cache_quant}] + overhead={overhead_mib}); "
            f"only {free_mib} MiB free" + par_note,
        )
    return GateResult(
        "kv_cache_fit", True,
        f"need ~{total_mib} MiB / {free_mib} free "
        f"(body={gguf_mib} KV={kv_mib} overhead={overhead_mib} quant={kv_cache_quant})"
        + par_note,
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
    no_kv_offload: bool = False,
    parallel: int = 1,
    split_mode: str = "layer",
    main_gpu: int = 0,
    cpu_moe_offload: bool = False,
) -> list[GateResult]:
    """Run all gates; return their results in order. Caller decides on failures.

    A "fail" in any GateResult.ok = False entry is a refusal signal. The
    aggregator does not short-circuit -- collecting all gates' status gives
    the audit + completion_future error a complete picture.

    The kv_cache_fit gate refuses spawn when the predicted
    KV cache + model body + overhead exceeds free VRAM. When ctx_size or
    gguf_size_bytes is unknown (0), the gate passes (caller still has the
    other VRAM gate via manifest_expected_vram_bytes).
    """
    return [
        check_free_ram(min_free_ram_mib),
        check_free_vram(min_free_vram_mib, manifest_expected_vram_bytes,
                        split_mode=split_mode, main_gpu=main_gpu),
        check_kv_cache_fit(
            ctx_size, gguf_size_bytes,
            overhead_mib=kv_cache_overhead_mib,
            kv_cache_quant=kv_cache_quant,
            no_kv_offload=no_kv_offload,
            parallel=parallel,
            split_mode=split_mode,
            main_gpu=main_gpu,
            expected_vram_mib=int(manifest_expected_vram_bytes // (1024 * 1024)),
            cpu_moe_offload=cpu_moe_offload,
        ),
        check_load_avg(max_load_per_core),
        check_iowait(max_iowait_percent, sample_window_s=iowait_sample_window_s),
    ]
