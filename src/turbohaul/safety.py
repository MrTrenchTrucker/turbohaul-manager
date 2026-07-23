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

# Marginal VRAM (MiB) per ADDITIONAL llama.cpp --parallel slot, on top of the
# single-slot baseline. ctx_size is the AGGREGATE KV window llama.cpp splits
# across N slots, so the ctx-linear scratch term already counts all N slots;
# this FLAT floor covers only the per-slot compute/attention buffers
# llama-server allocates per extra concurrent slot. CONSERVATIVE default (errs
# toward refusing) until MEASURED on a real 35B MoE model via nvidia-smi at
# parallel:2 (cold-load, max of two samples 5s apart after first decode). Do
# NOT raise to ship parallel:3 without that live measurement.
PER_SLOT_COMPUTE_FLOOR_MIB = 256


def estimate_kv_cache_mib(
    ctx_size: int,
    gguf_size_bytes: int,
    kv_cache_quant: str = "f16",
    kv_cache_quant_v: str | None = None,
    hybrid_kv_ratio: float = 1.0,
    attn_dims: "object | None" = None,
    kv_bytes_per_token: "float | None" = None,
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
    the growing attention layers; multiplying again would 4x under-count).
    hybrid_kv_ratio applies ONLY to the legacy file-size path. When attn_dims
    and kv_bytes_per_token are both None (every existing model), behaviour is
    byte-identical to the prior file-size path.

    Assumes quantized KV is paired with flash_attn (llama.cpp requires
    --flash-attn for quantized K/V); a manifest with a quantized cache_type
    but no flash_attn would be over-credited here -- but llama-server refuses
    that config at spawn, so the mis-estimate never reaches a live OOM.
    """
    if ctx_size <= 0 or gguf_size_bytes <= 0:
        return 0
    scale_k = _KV_QUANT_SCALE.get(kv_cache_quant.lower(), 1.0)
    scale_v = _KV_QUANT_SCALE.get((kv_cache_quant_v or kv_cache_quant).lower(), 1.0)

    # === Measured-override and dimension-aware paths ===========================
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
        return total_bytes // (1024 * 1024)

    # === Legacy file-size heuristic (BYTE-IDENTICAL to prior behaviour) ========
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
    kv_mib = estimate_kv_cache_mib(
        ctx_size, gguf_size_bytes, kv_cache_quant, kv_cache_quant_v, hybrid_kv_ratio,
        attn_dims=attn_dims, kv_bytes_per_token=kv_bytes_per_token)
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
        # (measured example: a 35B MoE n-cpu-moe config at 500K ctx has a closed-form
        # ~30.9GiB estimate vs a proven 19.4GiB footprint.) ctx-bump safety for these
        # configs is the operator's to maintain via expected_vram_bytes; normal
        # (non-offload) models keep the closed-form below.
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
        # The flat overhead floor does NOT scale with ctx, but the VRAM-side
        # attention scratch grows with ctx even when the KV is offloaded -- so add
        # a conservative ctx-linear scratch term on top of the floor. (an example
        # 27B model at 256K observed ~2.9 GiB VRAM over body; this estimates
        # ~3.1 GiB = 1024 floor + 262144//128.)
        vram_scratch_mib = overhead_mib + ctx_size // 128 + par_extra_mib
        vram_need = gguf_mib + vram_scratch_mib
        if vram_need > free_mib:
            return GateResult(
                "kv_cache_fit", False,
                f"need ~{vram_need} MiB VRAM "
                f"(body={gguf_mib} + scratch={vram_scratch_mib}; "
                f"KV@ctx{ctx_size}={kv_mib} [{q_label}] in host RAM); "
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
                    f"KV@ctx{ctx_size}={kv_mib} MiB [{q_label}] needs host "
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
            f"[{q_label}] + overhead={overhead_mib}); "
            f"only {free_mib} MiB free" + par_note,
        )
    return GateResult(
        "kv_cache_fit", True,
        f"need ~{total_mib} MiB / {free_mib} free "
        f"(body={gguf_mib} KV={kv_mib} overhead={overhead_mib} quant={q_label})"
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
    kv_cache_quant_v: str | None = None,
    no_kv_offload: bool = False,
    parallel: int = 1,
    split_mode: str = "layer",
    main_gpu: int = 0,
    cpu_moe_offload: bool = False,
    hybrid_kv_ratio: float = 1.0,
    attn_dims: "object | None" = None,
    kv_bytes_per_token: "float | None" = None,
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
    """
    return [
        check_free_ram(min_free_ram_mib),
        check_free_vram(min_free_vram_mib, manifest_expected_vram_bytes,
                        split_mode=split_mode, main_gpu=main_gpu),
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
        ),
        check_load_avg(max_load_per_core),
        check_iowait(max_iowait_percent, sample_window_s=iowait_sample_window_s),
    ]
