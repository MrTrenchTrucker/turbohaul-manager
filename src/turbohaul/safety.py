"""Safety guardrails: VRAM / RAM / IO-wait / CPU-load pre-spawn checks.

Mirror Ollama's safety posture so Turbohaul-Manager refuses to spawn a
sidecar when the host cannot safely run it. Each gate is tunable via
RuntimeConfig.queue.safety_*; the all_safety_gates aggregator returns the
list of failures so the manager can surface them on the loading_fail
audit + completion_future error.

All gates degrade gracefully: if the underlying probe is unavailable
(nvidia-smi missing in dev / /proc unreadable in some containers) the
gate returns "passed-no-probe" rather than blocking the spawn. You can
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
    # Spawn reclaim barrier: structural classification of a refusal,
    # separate from `name` -- two gates can share a name but mean opposite things
    # (kv_cache_fit's VRAM branch vs its host-RAM branch below), and a future gate
    # must not be trusted merely by NOT being named "vram". All three fields are
    # defaulted so every existing positional GateResult(name, ok, detail)
    # construction in this file stays valid unchanged.
    #   "vram"   -- a live release task on the gated card(s) could change this
    #               reading; eligible for the reclaim barrier.
    #   "host" / "config" / "probe" -- waiting cannot help; terminal immediately.
    #   None (the default, and every gate that is not tagged) -- unclassified;
    #               fails CLOSED (never treated as "vram", per _run's own
    #               `all(g.blocked_on == "vram" ...)` entry predicate).
    blocked_on: str | None = None
    required_mib: int | None = None  # VRAM gates only; diagnostic, never a control input
    available_mib: int | None = None  # VRAM gates only; diagnostic, never a control input


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
            blocked_on="host",
        )
    return GateResult("ram", True, f"{avail_mib} MiB free")


def check_moe_ram_fit(
    expert_offloaded_mib: int = 0,
    kv_ram_mib: int = 0,
) -> GateResult:
    """Additive RAM-fit check. ``--n-cpu-moe`` offloaded expert
    weights, plus the KV cache when ``no_kv_offload`` is set, both land in
    host RAM -- refuse spawn if they will not actually fit.

    Purely additive and independent of the other RAM/VRAM gates: does NOT
    touch ``check_free_ram``'s flat floor (that gate still runs separately
    in ``all_safety_gates``) or ``check_kv_cache_fit``'s VRAM-side
    ``cpu_moe_offload`` branch (which still trusts ``expected_vram_bytes``,
    unchanged). This specifically closes the gap where a config sets BOTH
    ``cpu_moe`` offload AND ``no_kv_offload``: ``check_kv_cache_fit``'s
    ``cpu_moe_offload`` branch returns before ever reaching its own
    ``no_kv_offload`` RAM sub-check, so today neither the offloaded experts
    nor the offloaded KV are checked against RAM for that combination. This
    gate checks both, together, regardless of which combination triggered
    them.

    A model using neither MoE CPU-offload nor ``no_kv_offload`` calls this
    with both args at their 0 default -- a pure no-op (always passes),
    byte-identical to not having this gate at all."""
    need_mib = expert_offloaded_mib + kv_ram_mib
    if need_mib <= 0:
        return GateResult("moe_ram_fit", True, "no offloaded weights/KV in RAM")
    info = _read_meminfo_kib()
    avail_kib = info.get("MemAvailable")
    if avail_kib is None:
        return GateResult("moe_ram_fit", True, "passed-no-probe")
    avail_mib = avail_kib // 1024
    if need_mib > avail_mib:
        return GateResult(
            "moe_ram_fit", False,
            f"need ~{need_mib} MiB RAM (offloaded experts={expert_offloaded_mib} "
            f"MiB + KV-in-RAM={kv_ram_mib} MiB); only {avail_mib} MiB free RAM",
            blocked_on="host",
        )
    return GateResult(
        "moe_ram_fit", True,
        f"need ~{need_mib} MiB / {avail_mib} MiB free RAM "
        f"(offloaded experts={expert_offloaded_mib} KV-in-RAM={kv_ram_mib})",
    )


def check_load_avg(max_per_core: float) -> GateResult:
    """Refuse spawn if 1-min load avg per logical core > max_per_core.

    Not called by all_safety_gates. Load average
    counts tasks that are runnable OR blocked (e.g. on disk I/O), not cores
    actually doing work -- it can read high while real CPU is mostly idle
    (a high 1-min load average per core does not by itself mean
    the CPU is busy). Kept
    defined (still correct, still unit-tested) rather than deleted, since
    nothing else in this file depends on removing it and a future caller may
    still want raw load-average as a DIFFERENT signal from busy%. See
    check_cpu_util for the metric used in the gate list instead.
    """
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
            blocked_on="host",
        )
    return GateResult(
        "cpu_load", True, f"1min-load-per-core={per_core:.2f}",
    )


def _read_stat_cpu_jiffies() -> tuple[int, int] | None:
    """Return (total_jiffies, idle_jiffies) from /proc/stat first cpu line.

    Same shape as _read_stat_iowait_jiffies (independent read, not shared --
    sharing one /proc/stat sample across both
    gates would restructure all_safety_gates's call shape for a small saving
    on a long spawn, widening the blast radius of a spawn-safety change for a marginal
    win; the reads are deliberately kept independent).

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
    idle = nums[3] if len(nums) > 3 else 0
    return total, idle


def check_cpu_util(max_percent: float, sample_window_s: float = 0.4) -> GateResult:
    """Sample /proc/stat over sample_window_s; refuse if real CPU busy% > max_percent.

    Used in the gate list instead of check_load_avg. Host-wide
    by construction -- /proc/stat is not PID-namespaced (identical
    cumulative counters are visible from the host and from inside a container),
    so this sees the whole machine exactly like
    the load-average check does, deliberately -- spawning a large
    model loads the whole box, so the gate should see the whole box. Only the
    METRIC differs: real CPU utilisation (cores actually doing work) instead
    of load average (tasks runnable OR blocked, which reads high under
    ambient disk/queueing pressure with no relationship to spare CPU).

    Shape mirrors check_iowait exactly: two independent /proc/stat reads
    separated by sample_window_s, busy% = 100 * (1 - idle_delta/total_delta).
    A short blocking sample is established architecture in this file already
    (check_iowait) and all_safety_gates already runs inside
    asyncio.to_thread, so this doesn't introduce a new class of
    risk.
    """
    sample_a = _read_stat_cpu_jiffies()
    if sample_a is None:
        return GateResult("cpu_util", True, "passed-no-probe")
    time.sleep(sample_window_s)
    sample_b = _read_stat_cpu_jiffies()
    if sample_b is None:
        return GateResult("cpu_util", True, "passed-no-probe-second")
    d_total = sample_b[0] - sample_a[0]
    d_idle = sample_b[1] - sample_a[1]
    if d_total <= 0:
        return GateResult("cpu_util", True, "passed-zero-delta")
    busy_pct = 100.0 * (d_total - d_idle) / d_total
    if busy_pct > max_percent:
        return GateResult(
            "cpu_util", False,
            f"cpu-busy={busy_pct:.1f}% > max {max_percent:.1f}%",
            blocked_on="host",
        )
    return GateResult(
        "cpu_util", True, f"cpu-busy={busy_pct:.1f}%",
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
            blocked_on="host",
        )
    return GateResult("iowait", True, f"iowait {pct:.1f}%")


def _read_free_vram_all_mib() -> list[int] | None:
    """Free MiB for EVERY CUDA device (one entry per GPU, index order).

    None if nvidia-smi is unavailable. Querying all rows (no ``-i 0``) makes the
    VRAM gates GPU-count agnostic: 1 card -> a 1-element list (identical to the
    legacy GPU0-only probe); N cards -> N elements so a layer-split model can be
    budgeted against the AGGREGATE free VRAM across all cards.
    """
    try:
        out = subprocess.check_output(
            [
                _NVIDIA_SMI_PATH,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    vals: list[int] = []
    for line in out.strip().splitlines():
        try:
            vals.append(int(line.strip().split(",")[0].strip()))
        except (ValueError, IndexError):
            continue
    return vals or None


def _read_free_vram_mib() -> int | None:
    """Back-compat: free MiB on GPU 0 (first device). None if unavailable."""
    vals = _read_free_vram_all_mib()
    return vals[0] if vals else None


def _read_total_vram_all_mib() -> list[int] | None:
    """Total VRAM MiB for EVERY CUDA device (one entry per GPU, index order).

    Boot-time read — total VRAM doesn't change at runtime.
    """
    try:
        out = subprocess.check_output(
            [
                _NVIDIA_SMI_PATH,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    vals: list[int] = []
    for line in out.strip().splitlines():
        try:
            vals.append(int(line.strip().split(",")[0].strip()))
        except (ValueError, IndexError):
            continue
    return vals or None


def _read_gpu_util_all_percent() -> list[int] | None:
    """Compute (SM) utilization percent for EVERY CUDA device (one entry per
    GPU, index order). None if nvidia-smi is unavailable.

    Mirrors ``_read_free_vram_all_mib``'s shape exactly (same
    query/format flags, same 5s timeout, same degrade-to-None on any probe
    failure) so it can be cached by the SAME off-loop ~1Hz pollers that
    already cache VRAM (``LiveSlotsPoller``/``LiveResidentsSupervisor``
    ``_refresh_vram``, live_monitor.py) — never called from a locked path.
    """
    try:
        out = subprocess.check_output(
            [
                _NVIDIA_SMI_PATH,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    vals: list[int] = []
    for line in out.strip().splitlines():
        try:
            vals.append(int(line.strip().split(",")[0].strip()))
        except (ValueError, IndexError):
            continue
    return vals or None


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
                    split_mode: str = "layer", main_gpu: int = 0,
                    occupied_vram_mib: int = 0) -> GateResult:
    """Refuse spawn if free VRAM < max(min_free_mib, manifest_expected/1024).

    GPU-count agnostic: a layer-split model (split_mode != 'none') is budgeted
    against the AGGREGATE free VRAM across all cards; a single-GPU pin
    (split_mode == 'none') against the main_gpu card only. With one physical
    GPU this collapses to the legacy GPU0-only behaviour.

    occupied_vram_mib: VRAM reserved by a same-card sibling that
    is still loading (RESERVED_LOADING) -- see
    TurbohaulManager._occupied_vram_mib. The gate budgets against EFFECTIVE
    free = nvidia-smi free minus this hold, so it refuses a spawn that would
    over-commit against a concurrent load on the same card. An
    ACTIVE/GRACE/IDLE_EVICTABLE sibling is NOT what this counts -- that VRAM
    is already absent from the live nvidia-smi reading, so it must not be
    subtracted a second time. Zero (the
    default, and every pre-existing caller) keeps historical
    behaviour byte-identical.
    """
    free_fit, min_card, n = _vram_budget(split_mode, main_gpu)
    if free_fit is None:
        return GateResult("vram", True, "passed-no-probe")
    expected_mib = manifest_expected_bytes // (1024 * 1024)
    threshold = max(min_free_mib, expected_mib)
    effective_free = free_fit - occupied_vram_mib
    if effective_free < 0:
        return GateResult(
            "vram", False,
            f"nvidia-smi reports {free_fit} MiB free but a same-card "
            f"still-loading sibling is reserving ~{occupied_vram_mib} MiB — "
            f"effective free negative; refusing spawn (need >= {threshold} "
            f"MiB).",
            blocked_on="vram", required_mib=threshold, available_mib=effective_free,
        )
    if effective_free < threshold:
        return GateResult(
            "vram", False,
            f"only {effective_free} MiB effective free across {n} card(s) "
            f"({free_fit} MiB free minus {occupied_vram_mib} MiB reserved by "
            f"a still-loading sibling); need >= {threshold} MiB "
            f"(min_floor={min_free_mib}, manifest={expected_mib})",
            blocked_on="vram", required_mib=threshold, available_mib=effective_free,
        )
    return GateResult(
        "vram", True,
        f"{effective_free} MiB effective free across {n} card(s) "
        f"(min/card {min_card}, threshold {threshold})"
        + (f" [{free_fit} raw minus {occupied_vram_mib} reserved]" if occupied_vram_mib else ""),
    )


# --- KV-cache fit estimator ---
# Closed-form pre-spawn check that refuses when (model body + KV cache + overhead)
# would not fit free VRAM. Independent of (and complementary to) check_free_vram's
# manifest-driven expected_vram_bytes check — this one is computed from ctx_size
# directly, so user can bump ctx_size in the manifest WITHOUT manually re-tuning
# expected_vram_bytes and the gate still catches over-commit.
#
# Empirical calibration (a 27B model at Q4_K_XL):
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

# Single source of truth for every engine-honoured KV cache quant type name.
# Anything that needs the full type set (manifest enum bounds, FE parity
# tests) keys off this rather than re-listing the 12 names.
KV_CACHE_TYPES: frozenset[str] = frozenset(_KV_QUANT_SCALE)

# Marginal VRAM (MiB) per ADDITIONAL llama.cpp --parallel slot, on top of the
# single-slot baseline. ctx_size is the AGGREGATE KV window llama.cpp splits
# across N slots, so the ctx-linear scratch term already counts all N slots;
# this FLAT floor covers only the per-slot compute/attention buffers
# llama-server allocates per extra concurrent slot. CONSERVATIVE default (errs
# toward refusing) until MEASURED on a real 35B MoE model via nvidia-smi at
# parallel:2 (cold-load, max of two samples 5s apart after first decode). Do
# NOT raise to ship parallel:3 without that live measurement.
PER_SLOT_COMPUTE_FLOOR_MIB = 256

# MTP draft-context KV margin.
# The geometry-derived draft_bytes_per_token formula (see
# estimate_kv_cache_mib's Tier-2 branch below) can come out slightly below
# the real draft-context KV growth, leaving a small residual, so a
# margin is added on top of it.
# Without the margin the estimate would under-count the draft context.
# The residual attribution is unknown (a single-model, small-sample
# slope fit), and reproducing it needs live GPU and blob-store access.
# The margin is a proportional factor (MTP_DRAFT_MARGIN below).
# THE TRADE: over-estimating here costs a
# refused spawn that would have fit; under-estimating costs an OOM — the
# exact failure this margin exists to prevent. A proportional margin (not a
# fixed byte constant) so it scales with whichever model's geometry it is
# applied to, since only one model's geometry was ever measured.
# An order of magnitude of headroom over the observed residual. On a typical model this is
# a small absolute margin — a trivial price for erring in the safe direction. Do NOT
# "optimize" this away without a live measurement on more than one model.
MTP_DRAFT_MARGIN = 1.05


def estimate_kv_cache_mib(
    ctx_size: int,
    gguf_size_bytes: int,
    kv_cache_quant: str = "f16",
    kv_cache_quant_v: str | None = None,
    hybrid_kv_ratio: float = 1.0,
    attn_dims: "object | None" = None,
    kv_bytes_per_token: "float | None" = None,
    mtp_draft_active: bool = False,
) -> int:
    """Closed-form KV-cache size estimate in MiB.

    Scales linearly with ctx_size and with model body size (gguf bytes), then
    by the KV quant factor. The KV cache is ~half K, half V, so the
    K half is scaled by cache_type_k and the V half by cache_type_v. When
    kv_cache_quant_v is None the V half falls back to the K quant (legacy
    single-quant behavior). This stops the gate over-counting a lightly-quantized
    K + heavily-quantized V (e.g. K=f16 + V=turbo3) as full-f16 for the whole KV
    and wrongly refusing a spawn that fits.

    hybrid_kv_ratio scales the per-token KV estimate for
    SSM+attention hybrid models (arch == "qwen35"). SSM layers store a fixed
    recurrent state (constant per-token, not a growing per-token KV cache), so
    only the attention-layer fraction contributes to per-token KV growth.
    hybrid_kv_ratio == 1.0 (default) → byte-identical to today for all existing
    models. hybrid_kv_ratio < 1.0 → the attention-fraction of per-token KV.

    Dimension-aware fix: the file-size heuristic below derives KV
    bytes/token from the model FILE size — calibrated on Q4 models, it badly
    UNDER-counts ultra-low-bit models (e.g. a 7 GB file for a 27B model whose
    true attention dims are much larger), so the gate can admit an over-commit
    that OOMs at high ctx. Two additive, higher-precedence paths fix this:
      1. ``kv_bytes_per_token`` — an operator-MEASURED effective KV bytes/token
         (e.g. from nvidia-smi). Authoritative when set.
      2. ``attn_dims`` — real GGUF attention dims (n_attn_layers, n_head_kv,
         key/value_length). Computes per-token KV from first principles.
    Precedence: measured override > parsed dims > file-size heuristic. BOTH new
    paths deliberately IGNORE hybrid_kv_ratio (n_attn_layers already counts only
    the growing attention layers; multiplying again would 4x under-count — see
    the note on the measured-override path below). hybrid_kv_ratio applies ONLY to the
    legacy file-size path. When attn_dims and kv_bytes_per_token are both None
    (every existing model), behaviour is byte-identical to the legacy path.

    Assumes quantized KV is paired with flash_attn (llama.cpp requires
    --flash-attn for quantized K/V); a manifest with a quantized cache_type
    but no flash_attn would be over-credited here -- but llama-server refuses
    that config at spawn, so the mis-estimate never reaches a live OOM.

    mtp_draft_active: when True AND
    attn_dims carries a nextn_predict_layers > 0 (an MTP draft head is
    present in the GGUF), adds the draft-context KV term to the dims-based
    (Tier-2) estimate: nextn_predict_layers * n_head_kv * (key_length +
    value_length) * 2, scaled by MTP_DRAFT_MARGIN, multiplied by ctx_size
    (the draft context is created at the SAME n_ctx_seq as the target --
    NOT spec_draft_n_max). The trailing 2 is f16 bytes/element and is
    CORRECT UNCONDITIONALLY: the draft cache runs cache_k=f16 cache_v=f16
    regardless of the manifest's cache_type_k/v, so draft quant scale is
    ALWAYS 1.0 -- never scale_k/scale_v. Confined to the Tier-2 branch: the
    term needs real GGUF geometry to be derived at all, so it never applies
    to the measured-override or legacy file-size paths. False (the default,
    every existing caller) keeps this byte-identical to before the MTP term.
    """
    if ctx_size <= 0 or gguf_size_bytes <= 0:
        return 0
    scale_k = _KV_QUANT_SCALE.get(kv_cache_quant.lower(), 1.0)
    scale_v = _KV_QUANT_SCALE.get((kv_cache_quant_v or kv_cache_quant).lower(), 1.0)

    # === Measured-override and dimension-aware paths ================
    # Precedence: measured override > parsed dims > legacy file-size heuristic.
    # NEITHER path applies hybrid_kv_ratio (they already reflect only the growing
    # attention-layer KV; re-applying it would double-discount). hybrid_kv_ratio
    # is confined to the legacy path below.
    if kv_bytes_per_token is not None and kv_bytes_per_token > 0:
        # Operator-measured EFFECTIVE KV bytes/token (already post quant + hybrid).
        # Used verbatim — no quant scale, no hybrid multiply.
        total_bytes = int(kv_bytes_per_token * ctx_size)
        return total_bytes // (1024 * 1024)
    if attn_dims is not None and getattr(attn_dims, "n_attn_layers", 0) > 0:
        # f16 KV bytes/token from real dims, attention layers only; K and V
        # halves each scaled by their own cache_type. No hybrid multiply.
        n_attn = attn_dims.n_attn_layers
        k_bytes_f16 = n_attn * attn_dims.n_head_kv * attn_dims.key_length * 2
        v_bytes_f16 = n_attn * attn_dims.n_head_kv * attn_dims.value_length * 2
        eff_bytes_per_token = k_bytes_f16 * scale_k + v_bytes_f16 * scale_v
        total_bytes = int(eff_bytes_per_token * ctx_size)
        # SWA-aware term. Sliding-window layers are CAPPED at
        # min(ctx, window) tokens per layer, NOT zero -- an SSM layer (fixed
        # recurrent state) is free, a sliding-window layer merely stops
        # growing past the window. Conflating the two under-estimates KV for
        # any sliding model. ⚠ Guard the min() call itself, not just the
        # multiplier: on a non-sliding KVDims (n_swa_layers=0, window=None)
        # `0 * min(ctx, None)` does NOT short-circuit in Python -- min(ctx,
        # None) raises TypeError before the multiply ever runs. This guard
        # is REQUIRED for correctness, not defensive style; without it every
        # non-sliding model's call would crash.
        n_swa = getattr(attn_dims, "n_swa_layers", 0)
        window = getattr(attn_dims, "sliding_window", None)
        swa_capped_tokens = min(ctx_size, window) if (n_swa > 0 and window) else 0
        if swa_capped_tokens > 0:
            swa_k_bytes_f16 = n_swa * attn_dims.n_head_kv * attn_dims.key_length * 2
            swa_v_bytes_f16 = n_swa * attn_dims.n_head_kv * attn_dims.value_length * 2
            swa_eff_bytes_per_token = swa_k_bytes_f16 * scale_k + swa_v_bytes_f16 * scale_v
            total_bytes += int(swa_eff_bytes_per_token * swa_capped_tokens)
        # MTP draft-context term. Gated
        # on BOTH mtp_draft_active (manifest opts into draft-mtp) AND a
        # real, GGUF-parsed nextn_predict_layers > 0 -- never fabricated.
        # Unconditional f16 (scale=1.0, never scale_k/scale_v -- the draft
        # cache ignores cache_type_k/v entirely), times
        # ctx_size (NOT spec_draft_n_max -- the draft context shares the
        # target's n_ctx_seq). MTP_DRAFT_MARGIN documented above.
        nextn = getattr(attn_dims, "nextn_predict_layers", 0)
        if mtp_draft_active and nextn > 0:
            draft_bytes_per_token = (
                nextn * attn_dims.n_head_kv
                * (attn_dims.key_length + attn_dims.value_length) * 2
            ) * MTP_DRAFT_MARGIN
            total_bytes += int(draft_bytes_per_token * ctx_size)
        return total_bytes // (1024 * 1024)

    # === Legacy file-size heuristic (BYTE-IDENTICAL to the legacy behaviour) ================
    gguf_mib = gguf_size_bytes // (1024 * 1024)
    # f16 baseline: ~9 KB/token per GiB of model body. Per-token in KB:
    bytes_per_token_kb_f16 = (9 * gguf_mib) // 1024
    # KV is ~50% K + 50% V; scale each half by its own cache_type.
    scale = (scale_k + scale_v) / 2.0
    bytes_per_token_kb = int(bytes_per_token_kb_f16 * scale)
    # Apply hybrid_kv_ratio to per-token KV (SSM layers don't grow).
    # Clamped to [0.0, 1.0] at the manifest level (Field ge=0.0, le=1.0).
    # When hybrid_kv_ratio == 1.0 (default), this is a no-op multiply.
    total_kib = int(bytes_per_token_kb * ctx_size * hybrid_kv_ratio)  # KB total
    return total_kib // 1024  # MiB


def check_kv_cache_fit(
    ctx_size: int,
    gguf_size_bytes: int,
    overhead_mib: int = 1024,
    kv_cache_quant: str = "f16",
    kv_cache_quant_v: str | None = None,
    no_kv_offload: bool = False,
    parallel: int = 1,
    split_mode: str = "layer",
    main_gpu: int = 0,
    expected_vram_mib: int = 0,
    cpu_moe_offload: bool = False,
    hybrid_kv_ratio: float = 1.0,
    attn_dims: "object | None" = None,
    kv_bytes_per_token: "float | None" = None,
    occupied_vram_mib: int = 0,
    mtp_draft_active: bool = False,
) -> GateResult:
    """Refuse spawn if (body + KV-cache + overhead) > free VRAM.

    Closed-form: doesn't trust the manifest's hand-tuned expected_vram_bytes;
    derives the prediction from ctx_size + gguf_size_bytes + quant. This is
    the load-bearing change for user-programmable ctx_size — when a user
    bumps ctx_size from 4096 to 65536 in the manifest, this gate refuses
    the spawn if the resulting KV cache won't fit on local hardware
    (regardless of whether expected_vram_bytes was hand-tuned to match).

    hybrid_kv_ratio (default 1.0) is passed through to
    estimate_kv_cache_mib(). When the model is a qwen35 hybrid (SSM+attn),
    the manifest sets kv_hybrid_ratio < 1.0 and arch == "qwen35", which
    scales the per-token KV estimate down proportionally.

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

    occupied_vram_mib: VRAM reserved by a same-card sibling that
    is still loading -- see TurbohaulManager._occupied_vram_mib and
    check_free_vram's identical parameter (this gate derives free_mib from
    the SAME _vram_budget(...) probe check_free_vram uses, so subtracting a
    measured occupancy here is the identical subtraction, not a new
    formula). Zero (the default) keeps historical behaviour byte-identical.
    Irrelevant when the probe itself is unreadable (free_mib is None) —
    there is no free figure to subtract against.

    mtp_draft_active: passed straight
    through to estimate_kv_cache_mib -- see that function for the formula.
    False (the default) keeps this byte-identical to before the MTP term.
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
                blocked_on="probe",
            )
        return GateResult("kv_cache_fit", True, "passed-no-probe")
    # Budget against EFFECTIVE free, same subtraction as
    # check_free_vram (both read the identical _vram_budget(...) probe).
    free_mib -= occupied_vram_mib
    gguf_mib = gguf_size_bytes // (1024 * 1024)
    kv_mib = estimate_kv_cache_mib(
        ctx_size, gguf_size_bytes, kv_cache_quant, kv_cache_quant_v, hybrid_kv_ratio,
        attn_dims=attn_dims, kv_bytes_per_token=kv_bytes_per_token,
        mtp_draft_active=mtp_draft_active)
    q_label = (
        kv_cache_quant
        if not kv_cache_quant_v or kv_cache_quant_v == kv_cache_quant
        else f"K={kv_cache_quant}/V={kv_cache_quant_v}"
    )
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
        # (for example, a large MoE n-cpu-moe config at a very long ctx has a closed-form
        # estimate well above its real footprint.) ctx-bump safety for these configs is the operator's to maintain via
        # expected_vram_bytes; normal (non-offload) models keep the closed-form below.
        vram_need = expected_vram_mib + overhead_mib + par_extra_mib
        if vram_need > free_mib:
            return GateResult(
                "kv_cache_fit", False,
                f"need ~{vram_need} MiB VRAM (expected_vram={expected_vram_mib} "
                f"[cpu-moe measured] + overhead={overhead_mib}); only {free_mib} MiB "
                f"free VRAM" + par_note,
                blocked_on="vram", required_mib=vram_need, available_mib=free_mib,
            )
        return GateResult(
            "kv_cache_fit", True,
            f"fits ~{vram_need} MiB (expected_vram={expected_vram_mib} [cpu-moe "
            f"measured]); {free_mib} MiB free" + par_note,
        )
    if no_kv_offload:
        # KV cache is in host RAM (--no-kv-offload). VRAM holds the model body
        # plus a ctx-scaled compute/attention scratch (NOT the KV cache).
        # The flat overhead floor does NOT scale with
        # ctx, but the VRAM-side attention scratch grows with ctx even when the
        # KV is offloaded -- so add a conservative ctx-linear scratch term on
        # top of the floor. (an example 27B model at 256K was observed to need less VRAM over its body than the estimate;
        # this estimates ~3.1 GiB = 1024 floor + 262144//128.)
        vram_scratch_mib = overhead_mib + ctx_size // 128 + par_extra_mib
        vram_need = gguf_mib + vram_scratch_mib
        if vram_need > free_mib:
            return GateResult(
                "kv_cache_fit", False,
                f"need ~{vram_need} MiB VRAM "
                f"(body={gguf_mib} + scratch={vram_scratch_mib}; "
                f"KV@ctx{ctx_size}={kv_mib} [{q_label}] in host RAM); "
                f"only {free_mib} MiB free VRAM" + par_note,
                blocked_on="vram", required_mib=vram_need, available_mib=free_mib,
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
                    f"KV@ctx{ctx_size}={kv_mib} MiB [{q_label}] needs host "
                    f"RAM (--no-kv-offload) but only {ram_avail_mib} MiB free RAM",
                    blocked_on="host",
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
            f"[{q_label}] + overhead={overhead_mib}); "
            f"only {free_mib} MiB free" + par_note,
            blocked_on="vram", required_mib=total_mib, available_mib=free_mib,
        )
    return GateResult(
        "kv_cache_fit", True,
        f"need ~{total_mib} MiB / {free_mib} free "
        f"(body={gguf_mib} KV={kv_mib} overhead={overhead_mib} quant={q_label})"
        + par_note,
    )


def check_tensor_split_devices(tensor_split: str | None) -> GateResult:
    """Refuse spawn if tensor_split's element count != visible device count.

    Device count is a runtime property (the same manifest may be read on a
    different box), so this can't be enforced in the static manifest-time
    validator (turbohaul.manifest.parse_strict_csv_numeric) -- it runs here,
    at spawn, against the live nvidia-smi probe. We refuse
    rather than risk OOM from wrong placement. The engine has its own
    backstop ("got %zu input configs, but system only has %zu devices" in
    libllama-common), but the manager refuses first with a clear error.

    No tensor_split set, or the nvidia-smi probe unavailable -> pass
    (same degrade-open doctrine as every other gate in this module).
    """
    if not tensor_split:
        return GateResult("tensor_split_devices", True, "no tensor_split set")
    vals = _read_free_vram_all_mib()
    if not vals:
        return GateResult("tensor_split_devices", True, "passed-no-probe")
    device_count = len(vals)
    # Safe to count commas directly rather than re-running
    # parse_strict_csv_numeric: tensor_split is manifest-validated (strict
    # [0-9.,] whitelist) before it can reach a spawn, so the field count
    # here cannot disagree with the validated parse.
    split_count = tensor_split.count(",") + 1
    if split_count != device_count:
        return GateResult(
            "tensor_split_devices", False,
            f"tensor_split has {split_count} element(s) but {device_count} "
            "device(s) are visible; refusing rather than risk OOM from "
            "wrong placement",
            blocked_on="config",
        )
    return GateResult(
        "tensor_split_devices", True,
        f"{split_count} element(s) matches {device_count} visible device(s)",
    )


def all_safety_gates(
    *,
    min_free_ram_mib: int,
    min_free_vram_mib: int,
    max_cpu_busy_percent: float,
    max_iowait_percent: float,
    manifest_expected_vram_bytes: int = 0,
    cpu_util_sample_window_s: float = 0.4,
    iowait_sample_window_s: float = 0.4,
    ctx_size: int = 0,
    gguf_size_bytes: int = 0,
    kv_cache_overhead_mib: int = 1024,
    kv_cache_quant: str = "f16",
    kv_cache_quant_v: str | None = None,
    no_kv_offload: bool = False,
    parallel: int = 1,
    split_mode: str = "layer",
    main_gpu: int = 0,
    cpu_moe_offload: bool = False,
    hybrid_kv_ratio: float = 1.0,
    attn_dims: "object | None" = None,
    kv_bytes_per_token: "float | None" = None,
    expert_offloaded_mib: int = 0,
    tensor_split: str | None = None,
    occupied_vram_mib: int = 0,
    mtp_draft_active: bool = False,
) -> list[GateResult]:
    """Run all gates; return their results in order. Caller decides on failures.

    A "fail" in any GateResult.ok = False entry is a refusal signal. The
    aggregator does not short-circuit -- collecting all gates' status gives
    the audit + completion_future error a complete picture.

    The kv_cache_fit gate refuses spawn when the predicted
    KV cache + model body + overhead exceeds free VRAM. When ctx_size or
    gguf_size_bytes is unknown (0), the gate passes (caller still has the
    other VRAM gate via manifest_expected_vram_bytes).

    hybrid_kv_ratio (default 1.0) is passed through to
    check_kv_cache_fit → estimate_kv_cache_mib(). For qwen35 hybrid models
    the manifest sets kv_hybrid_ratio < 1.0, which scales the per-token
    KV estimate down proportionally.

    expert_offloaded_mib (default 0, from the caller's exact
    tensor-offset derivation) feeds the additive moe_ram_fit gate, together
    with an independently-computed KV-in-RAM figure when no_kv_offload is
    set. Both default to a no-op for a model using neither MoE CPU-offload
    nor no_kv_offload.

    tensor_split (default None): the manifest's raw
    tensor_split CSV string, if set. check_tensor_split_devices refuses
    when its element count doesn't match the visible device count.

    occupied_vram_mib (default 0): VRAM reserved by a same-card
    still-loading sibling -- see TurbohaulManager._occupied_vram_mib.
    Threaded into BOTH check_free_vram (the coarse min-free check) and
    check_kv_cache_fit (the precise body+KV+overhead check) so neither one
    is left blind to a concurrent load on the same card. Default 0 keeps
    every existing caller byte-identical.

    mtp_draft_active: threaded into
    BOTH the no_kv_offload host-RAM pre-check below AND check_kv_cache_fit
    -- see the comment at the pre-check call for why the former is an
    explicit, UNVERIFIED placement decision, not a confirmed one. False
    (the default) keeps every existing caller byte-identical.
    """
    kv_ram_mib = 0
    if no_kv_offload and ctx_size > 0 and gguf_size_bytes > 0:
        # mtp_draft_active is threaded into this
        # host-RAM pre-check too, so a draft-mtp model under
        # --no-kv-offload gets the same draft-context budget as the
        # VRAM-resident default path below. THIS PLACEMENT IS UNVERIFIED.
        # The combination spec_type=draft-mtp with no_kv_offload=true
        # is not covered by any observation, so nothing shows
        # where the draft KV cache actually lands when the
        # target's KV is offloaded to host RAM. What IS known: for models
        # observed so far, the draft context is created gpu_layers=-1
        # (GPU-resident by layer placement) -- suggestive, but layer
        # placement and KV offload are different mechanisms, so this is
        # not proof either way. This code deliberately follows
        # kv_mib's existing monolithic placement (the pre-existing code
        # already treats "the KV cache" as living entirely in ONE place,
        # VRAM or host RAM, never split) rather than inventing MTP-specific
        # placement logic with zero evidence -- that would be exactly the
        # "pick a number not derived from GGUF geometry" failure this
        # estimate is meant to avoid. If the combination is used, this
        # placement is a candidate to
        # instrument.
        kv_ram_mib = estimate_kv_cache_mib(
            ctx_size, gguf_size_bytes, kv_cache_quant, kv_cache_quant_v,
            hybrid_kv_ratio, attn_dims=attn_dims,
            kv_bytes_per_token=kv_bytes_per_token,
            mtp_draft_active=mtp_draft_active,
        )
    return [
        check_free_ram(min_free_ram_mib),
        check_free_vram(min_free_vram_mib, manifest_expected_vram_bytes,
                        split_mode=split_mode, main_gpu=main_gpu,
                        occupied_vram_mib=occupied_vram_mib),
        check_kv_cache_fit(
            ctx_size, gguf_size_bytes,
            overhead_mib=kv_cache_overhead_mib,
            kv_cache_quant=kv_cache_quant,
            kv_cache_quant_v=kv_cache_quant_v,
            no_kv_offload=no_kv_offload,
            parallel=parallel,
            split_mode=split_mode,
            main_gpu=main_gpu,
            expected_vram_mib=int(manifest_expected_vram_bytes // (1024 * 1024)),
            cpu_moe_offload=cpu_moe_offload,
            hybrid_kv_ratio=hybrid_kv_ratio,
            attn_dims=attn_dims,
            kv_bytes_per_token=kv_bytes_per_token,
            occupied_vram_mib=occupied_vram_mib,
            mtp_draft_active=mtp_draft_active,
        ),
        check_moe_ram_fit(
            expert_offloaded_mib=expert_offloaded_mib,
            kv_ram_mib=kv_ram_mib,
        ),
        check_cpu_util(max_cpu_busy_percent, sample_window_s=cpu_util_sample_window_s),
        check_iowait(max_iowait_percent, sample_window_s=iowait_sample_window_s),
        check_tensor_split_devices(tensor_split),
    ]
