"""Per-model manifest: closed flag allowlist + atomic writes + ETag/If-Match concurrency.

See ARCHITECTURE.md for the manifest design.
Addresses the flag-injection RCE class (CRIT), tag path-traversal (CRIT),
the lost-update race, and the atomic-write requirement.

Hardening summary:
- SAFE_LLAMA_FLAGS expanded from 30 → ~80 (Ollama parity + reasoning_budget
  + fit-target + RoPE/YaRN + sampling completeness + server toggles
  + KV cache controls + debug knobs).
- DENIED_FLAGS expanded +22 (path-bearing/RCE: model_url, hf_repo*,
  api_key_file, ssl_*, path, media_path, tools, control_vector*, lookup_cache_*).
- Suffix-pattern forward-defense guard rejects future path-bearing pulls.
- Numeric bounds via SAFE_LLAMA_FLAG_BOUNDS prevent DoS-by-extreme.
- flash_attn type fixed: int → bool|str-enum (on/off/auto).
- chat_template hardened: must match built-in enum OR be plain non-Jinja string
  (closes the SSTI gap).
"""
import contextlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


# === Closed allowlist of safe llama-server flags (flag-injection guard) ===
# Each entry is (key, expected_python_type | tuple of types). To add a new
# flag here requires a code change + review; yaml cannot smuggle it in.
# Special-cased types: "flash_attn" accepts bool OR str-enum (handled below).
SAFE_LLAMA_FLAGS: dict[str, Any] = {
    # === Performance + memory layout ===
    "ctx_size": int,
    "n_gpu_layers": (int, str),  # accept "all" / "auto"
    "threads": int,
    "threads_batch": int,
    "threads_http": int,
    "parallel": int,
    "batch_size": int,
    "ubatch_size": int,
    "n_predict": int,
    "keep": int,
    "flash_attn": (bool, str),  # tri-state on/off/auto (post llama.cpp PR ~17000)
    "mlock": bool,
    "no_mmap": bool,
    "numa": str,  # enum: none/distribute/isolate/numactl
    "swa_full": bool,
    "no_perf": bool,
    "sleep_idle_seconds": int,
    "cache_reuse": int,
    "no_context_shift": bool,
    "slot_prompt_similarity": float,
    "warmup": bool,
    "check_tensors": bool,
    "repack": bool,
    "op_offload": bool,
    "no_host": bool,
    "direct_io": bool,
    "cont_batching": bool,
    # === KV-cache ===
    "cache_type_k": str,  # enum: f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1
    "cache_type_v": str,
    "kv_offload": bool,
    "no_kv_offload": bool,  # --no-kv-offload: KV cache in host RAM, not VRAM
    "kv_unified": bool,
    "cache_idle_slots": bool,
    "cache_prompt": bool,
    "cache_ram": int,
    "ctx_checkpoints": int,
    "checkpoint_every_n_tokens": int,
    # === Context / RoPE / YaRN ===
    "rope_scaling": str,  # enum: none/linear/yarn
    "rope_scale": float,
    "rope_freq_base": float,
    "rope_freq_scale": float,
    "yarn_orig_ctx": int,
    "yarn_ext_factor": float,
    "yarn_attn_factor": float,
    "yarn_beta_slow": float,
    "yarn_beta_fast": float,
    # === MoE / multi-GPU ===
    "cpu_moe": bool,            # -cmoe (all MoE on CPU)
    "n_cpu_moe": int,           # -ncmoe N (count of MoE layers on CPU)
    "split_mode": str,          # enum: none/layer/row/tensor
    "main_gpu": int,
    # NOTE: tensor_split intentionally NOT added — CSV-string with shell-meta risk;
    # defer until a validator can parse "N,N,N" safely.
    "fit": str,                 # enum on/off (Tom's Fork auto-mem-fit)
    "fit_ctx": int,
    # NOTE: fit_target also CSV-string; defer.
    # === Sampling — full set ===
    "temp": float,
    "top_k": int,
    "top_p": float,
    "min_p": float,
    "typical_p": float,         # Ollama parity
    "top_n_sigma": float,
    "repeat_penalty": float,
    "repeat_last_n": int,
    "presence_penalty": float,  # Ollama parity
    "frequency_penalty": float, # Ollama parity
    "seed": int,
    "mirostat": int,            # Ollama parity — 0/1/2
    "mirostat_lr": float,
    "mirostat_ent": float,
    "xtc_probability": float,
    "xtc_threshold": float,
    "dynatemp_range": float,
    "dynatemp_exp": float,
    "dry_multiplier": float,
    "dry_base": float,
    "dry_allowed_length": int,
    "dry_penalty_last_n": int,
    "adaptive_target": float,
    "adaptive_decay": float,
    "ignore_eos": bool,
    # === Chat / template (value names + bounded strings only) ===
    "chat_template": str,       # gated: built-in enum OR plain non-Jinja string
    "jinja": bool,
    "skip_chat_parsing": bool,
    "special": bool,
    "spm_infill": bool,
    # === Reasoning — Hermes preserved-thinking ===
    "reasoning_format": str,    # enum: none/deepseek/deepseek-legacy/auto
    "reasoning": str,           # enum: on/off/auto
    "reasoning_budget": int,    # -1/0/N — Hermes preserved-thinking knob
    # === Speculative decoding / MTP (PR #22673 = build b9180; needs a GGUF nextn head) ===
    # Composes with the turbo cache_type_k/v (turbo2/3/4) + flash_attn. Spawn-level argv (cold-spawn to apply).
    "spec_type": str,                    # enum: "draft-mtp" (multi-token-prediction speculative decode)
    "spec_draft_n_max": int,             # max draft tokens per step (MTP default 3)
    "spec_draft_n_min": int,             # min draft tokens per step
    "spec_draft_p_min": float,           # min prob to continue drafting
    "spec_draft_p_split": float,         # draft split probability threshold
    "spec_draft_ngl": int,               # draft GPU layers (bundled MTP head; usually same device)
    "spec_draft_backend_sampling": bool, # backend-side sampling for the draft path
    # === Server toggles ===
    "metrics": bool,
    "slots": bool,
    "props": bool,
    "embeddings": bool,
    "reranking": bool,
    "pooling": str,             # enum: none/mean/cls/last/rank
    "offline": bool,
    # === Debug ===
    "verbose": bool,
    "log_disable": bool,
    "log_colors": str,          # enum: on/off/auto
    "log_prefix": bool,
    "log_timestamps": bool,
    "log_verbosity": int,
}


# === Numeric bounds (DoS prevention) ===
# (min, max) inclusive; None = unbounded on that side.
SAFE_LLAMA_FLAG_BOUNDS: dict[str, tuple[Any, Any]] = {
    "ctx_size": (1, 2_000_000),
    "n_gpu_layers": (-1, 999),
    "n_predict": (-1, 1_000_000),
    "threads": (-1, 256),
    "threads_batch": (-1, 256),
    "threads_http": (-1, 256),
    "parallel": (1, 256),
    "batch_size": (1, 65536),
    "ubatch_size": (1, 65536),
    "keep": (-1, 65536),
    "sleep_idle_seconds": (-1, 86400),
    "cache_reuse": (0, 65536),
    "n_cpu_moe": (0, 256),
    "main_gpu": (0, 16),
    "fit_ctx": (1, 2_000_000),
    "cache_ram": (0, 256 * 1024),  # MiB
    "ctx_checkpoints": (0, 1024),
    "checkpoint_every_n_tokens": (1, 1_000_000),
    "yarn_orig_ctx": (0, 2_000_000),
    "temp": (0.0, 10.0),
    "top_k": (0, 10000),
    "top_p": (0.0, 1.0),
    "min_p": (0.0, 1.0),
    "typical_p": (0.0, 1.0),
    "top_n_sigma": (-1.0, 100.0),
    "repeat_penalty": (0.0, 10.0),
    "repeat_last_n": (-1, 65536),
    "presence_penalty": (-10.0, 10.0),
    "frequency_penalty": (-10.0, 10.0),
    "seed": (-1, 2**63 - 1),
    "mirostat": (0, 2),
    "mirostat_lr": (0.0, 1.0),
    "mirostat_ent": (0.0, 100.0),
    "xtc_probability": (0.0, 1.0),
    "xtc_threshold": (0.0, 1.0),
    "dynatemp_range": (0.0, 10.0),
    "dynatemp_exp": (0.0, 10.0),
    "dry_multiplier": (0.0, 10.0),
    "dry_base": (1.0, 10.0),
    "dry_allowed_length": (0, 65536),
    "dry_penalty_last_n": (-1, 65536),
    "adaptive_target": (-1.0, 100.0),
    "adaptive_decay": (0.0, 1.0),
    "slot_prompt_similarity": (0.0, 1.0),
    "rope_scale": (0.0, 1000.0),
    "rope_freq_base": (0.0, 10_000_000.0),
    "rope_freq_scale": (0.0, 100.0),
    "yarn_ext_factor": (-1.0, 100.0),
    "yarn_attn_factor": (-1.0, 100.0),
    "yarn_beta_slow": (-1.0, 100.0),
    "yarn_beta_fast": (-1.0, 100.0),
    "reasoning_budget": (-1, 1_000_000),
    "log_verbosity": (0, 4),
    # Speculative / MTP bounds (DoS prevention)
    "spec_draft_n_max": (0, 64),
    "spec_draft_n_min": (0, 64),
    "spec_draft_p_min": (0.0, 1.0),
    "spec_draft_p_split": (0.0, 1.0),
    "spec_draft_ngl": (-1, 999),
}


# === String enum bounds (closes the chat_template Jinja injection gap) ===
# Only fixed enum values allowed for these string flags. chat_template special-cased
# below (accepts enum OR plain non-Jinja string).
SAFE_LLAMA_FLAG_STRING_ENUMS: dict[str, set[str]] = {
    "numa": {"none", "distribute", "isolate", "numactl"},
    "cache_type_k": {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "turbo2", "turbo3", "turbo4"},
    "cache_type_v": {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "turbo2", "turbo3", "turbo4"},
    "rope_scaling": {"none", "linear", "yarn"},
    "split_mode": {"none", "layer", "row", "tensor"},
    "fit": {"on", "off"},
    "reasoning_format": {"none", "deepseek", "deepseek-legacy", "auto"},
    "reasoning": {"on", "off", "auto"},
    "pooling": {"none", "mean", "cls", "last", "rank"},
    "log_colors": {"on", "off", "auto"},
    # spec_type: confirm full accepted token set from `llama-server --help` (PR#22673 enum:
    # draft / draft-eagle3 / draft-mtp). We only enable draft-mtp.
    "spec_type": {"draft-mtp"},
}

# flash_attn — special-case tri-state. Accepts bool (legacy) OR str enum.
_FLASH_ATTN_STR_VALUES: set[str] = {"on", "off", "auto", "enabled", "disabled"}

# n_gpu_layers — accept int OR str "all"/"auto"
_N_GPU_LAYERS_STR_VALUES: set[str] = {"all", "auto"}

# Built-in chat_template names (subset of llama.cpp's bundled templates;
# extracted from `llama-server --help` and tools/server/CHAT_TEMPLATES.md).
# Accept these OR plain non-Jinja string. Reject anything containing
# `{%` or `{{` (Jinja constructs that could SSTI-inject via filesystem
# reads in non-sandboxed Jinja envs).
SAFE_CHAT_TEMPLATE_NAMES: set[str] = {
    "chatml", "llama2", "llama3", "llama3.1", "llama3.2", "llama3.3",
    "gemma", "gemma2", "gemma3", "gemma4",
    "mistral", "mistral-v1", "mistral-v3", "mistral-v3-tekken", "mistral-v7",
    "phi3", "phi4",
    "deepseek", "deepseek2", "deepseek-r1",
    "qwen", "qwen2", "qwen2.5", "qwen3", "qwen3.5", "qwen3.6",
    "command-r", "command-r-plus",
    "vicuna", "alpaca", "zephyr", "chatglm3", "chatglm4",
    "openchat", "orion", "yi", "monarch", "smollm", "minicpm",
    "exaone3", "rwkv-world", "granite", "qwen3-thinking", "qwq",
    "default",
}

# Suffix-pattern forward-defense.
# Any flag whose name matches one of these regexes is REJECTED unless
# explicitly listed below in SUFFIX_GUARD_ALLOWLIST_EXCEPTIONS. Catches
# future Tom's Fork pulls that ship path-bearing or credential flags
# we haven't yet seen.
_SUFFIX_GUARD_PATTERNS: list[re.Pattern] = [
    re.compile(r".*_file$"),
    re.compile(r".*_path$"),
    re.compile(r".*_dir$"),
    re.compile(r".*_url$"),
    re.compile(r".*_repo$"),
    re.compile(r".*_key$"),
    re.compile(r"^hf_"),
    re.compile(r"^lora"),
    re.compile(r"^control_vector"),
    re.compile(r"^lookup_cache_"),
    re.compile(r"^ssl_"),
    re.compile(r"^api_key"),
    re.compile(r"^slot_save_"),
    re.compile(r"^webui_"),
    re.compile(r"^docker_"),
]

# Exceptions to suffix-guard — flags that LOOK path-bearing by name but
# are actually safe value-only (none right now, but reserved for future).
_SUFFIX_GUARD_EXCEPTIONS: set[str] = set()


def _suffix_guard_check(key: str) -> None:
    """Forward-defense: reject any key matching path/cred/URL suffix patterns.

    This catches NEW flags that slip into SAFE_LLAMA_FLAGS via a code-review
    miss. Raises ManifestValidationError on match.
    """
    if key in _SUFFIX_GUARD_EXCEPTIONS:
        return
    for p in _SUFFIX_GUARD_PATTERNS:
        if p.match(key):
            raise ManifestValidationError(
                f"llama_server_flags.{key} is rejected by suffix-pattern "
                f"forward-defense guard (matches {p.pattern!r}). If this "
                "flag is genuinely safe value-only, add to "
                "_SUFFIX_GUARD_EXCEPTIONS with an audit trail."
            )


# === Explicit denylist of path-bearing flags (CRITICAL) ===
# Any of these in llama_server_flags would allow file read/write injection,
# credential exfil, SSRF, or direct RCE via llama-server's tool-call interface.
DENIED_FLAGS: set[str] = {
    # Original 20
    "mmproj",
    "lora",
    "lora_base",
    "lora_scaled",
    "grammar_file",
    "json_schema_file",
    "log_file",
    "slot_save_path",
    "chat_template_file",
    "in_prefix_file",
    "in_suffix_file",
    "hf_token",
    "override_kv",
    "cache_prompt_file",
    "binary_override",
    "model",
    "alias",
    "rpc",
    "host",
    "port",
    # +22 more path-bearing/credential flags that ship upstream but were unguarded
    "model_draft",            # -md — arbitrary GGUF path
    "model_url",              # SSRF + RCE — network fetch by attacker URL
    "model_url_draft",
    "hf_repo",                # SSRF + arbitrary download via HF
    "hf_repo_draft",
    "hf_file",
    "hf_repo_v",              # vocoder variant
    "hf_file_v",
    "docker_repo",            # docker-hub fetch primitive
    "api_key",                # credential injection
    "api_key_file",           # path read
    "ssl_key_file",           # path read (PEM exfil)
    "ssl_cert_file",
    "lookup_cache_static",    # -lcs — arbitrary read/write
    "lookup_cache_dynamic",   # -lcd
    "model_vocoder",          # -mv — arbitrary file read
    "webui_config_file",      # arbitrary JSON read
    "webui_mcp_proxy",        # CORS bypass / SSRF (per Tom's Fork README)
    "path",                   # CRITICAL — sets static-files dir for HTTP serve, /etc exfil
    "media_path",             # CRITICAL — same exfil class
    "models_dir",             # path read + arbitrary model load
    "models_preset",          # arbitrary INI read
    "control_vector",         # path read
    "control_vector_scaled",
    "tools",                  # DIRECT RCE — enables exec_shell_command / write_file / edit_file via server API
    "grammar",                # inline BNF — deferred (needs grammar-parser pre-validator)
    "tensor_split",           # CSV-string with shell-meta risk — deferred
    "samplers",               # semi-colon list — deferred (validator needed)
    "dry_sequence_breaker",   # str list — deferred
    "chat_template_kwargs",   # JSON-str — deferred (recursive scalar validator needed)
    "reasoning_budget_message", # str injected mid-stream — deferred (length-cap + ctrl-char strip needed)
    "fit_target",             # CSV "MiB,MiB" — deferred
}


# === Tag validation regex (tag path-traversal guard, CRITICAL) ===
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

# Minimum per-slot context window when llama.cpp --parallel > 1 splits the
# aggregate ctx_size across N concurrent slots. Below this, each slot's KV
# window is too small to be useful (and an indivisible ctx_size silently
# truncates per-slot context in llama-server). Module const so the parallel
# validator and tests share one source of truth.
PER_SLOT_CTX_FLOOR = 8192


class ManifestValidationError(ValueError):
    """Schema, allowlist, or path-safety violation."""


class ConcurrencyError(RuntimeError):
    """ETag/If-Match mismatch - caller returns HTTP 412 Precondition Failed."""


def validate_tag(tag: str) -> None:
    """Validate model_tag against regex. Raises ManifestValidationError on fail."""
    if not isinstance(tag, str):
        raise ManifestValidationError(f"tag must be string, got {type(tag).__name__}")
    if not TAG_RE.match(tag):
        raise ManifestValidationError(
            f"tag {tag!r} fails regex ^[a-z0-9][a-z0-9._-]{{0,63}}$ - "
            "ASCII lowercase only, no path separators, no traversal, max 64 chars"
        )


class PromptTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_default: str = ""
    stop_tokens: list[str] = Field(default_factory=list)


def _check_jinja_injection(value: str) -> None:
    """Reject Jinja2 constructs in chat_template body (SSTI guard)."""
    if "{%" in value or "{{" in value:
        raise ManifestValidationError(
            "chat_template contains Jinja constructs ({% or {{). Reject — "
            "non-sandboxed Jinja could SSTI. Use a built-in template name "
            f"from SAFE_CHAT_TEMPLATE_NAMES ({len(SAFE_CHAT_TEMPLATE_NAMES)} "
            "options) or DENIED_FLAGS.chat_template_file for custom Jinja."
        )


def _validate_flag_value(key: str, value: Any) -> None:
    """Validate a single flag value against allowlist + bounds + enum constraints."""
    expected = SAFE_LLAMA_FLAGS[key]

    # Special-case: flash_attn (bool OR str enum)
    if key == "flash_attn":
        if isinstance(value, bool):
            return
        if isinstance(value, str) and value.lower() in _FLASH_ATTN_STR_VALUES:
            return
        raise ManifestValidationError(
            f"llama_server_flags.flash_attn expects bool or str in "
            f"{sorted(_FLASH_ATTN_STR_VALUES)}, got {type(value).__name__}: {value!r}"
        )

    # Special-case: n_gpu_layers (int OR "all"/"auto")
    if key == "n_gpu_layers":
        if isinstance(value, bool):
            raise ManifestValidationError(
                f"llama_server_flags.n_gpu_layers expects int or str, got bool"
            )
        if isinstance(value, int):
            lo, hi = SAFE_LLAMA_FLAG_BOUNDS.get(key, (None, None))
            if lo is not None and value < lo:
                raise ManifestValidationError(f"n_gpu_layers {value} < min {lo}")
            if hi is not None and value > hi:
                raise ManifestValidationError(f"n_gpu_layers {value} > max {hi}")
            return
        if isinstance(value, str) and value.lower() in _N_GPU_LAYERS_STR_VALUES:
            return
        raise ManifestValidationError(
            f"n_gpu_layers expects int or str in {sorted(_N_GPU_LAYERS_STR_VALUES)}, got {value!r}"
        )

    # Special-case: chat_template (enum OR plain non-Jinja string)
    if key == "chat_template":
        if not isinstance(value, str):
            raise ManifestValidationError(
                f"chat_template expects str, got {type(value).__name__}"
            )
        _check_jinja_injection(value)
        # Accept if in built-in enum, OR plain string short enough not to be a template body
        if value in SAFE_CHAT_TEMPLATE_NAMES:
            return
        if len(value) > 256:
            raise ManifestValidationError(
                f"chat_template value too long ({len(value)} chars; max 256 for "
                "non-built-in names). Use a built-in name or chat_template_file."
            )
        # Plain identifier-shaped string — accept (loose to allow custom names
        # that aren't yet in SAFE_CHAT_TEMPLATE_NAMES but are clearly not Jinja)
        if not re.match(r"^[A-Za-z0-9_.\-]+$", value):
            raise ManifestValidationError(
                f"chat_template value {value!r} has invalid chars; must be "
                "alphanumeric + . _ - only (or use built-in enum name)"
            )
        return

    # General string-enum validation
    if key in SAFE_LLAMA_FLAG_STRING_ENUMS:
        if not isinstance(value, str):
            raise ManifestValidationError(
                f"llama_server_flags.{key} expects str enum, got {type(value).__name__}"
            )
        if value not in SAFE_LLAMA_FLAG_STRING_ENUMS[key]:
            raise ManifestValidationError(
                f"llama_server_flags.{key}={value!r} not in allowed enum "
                f"{sorted(SAFE_LLAMA_FLAG_STRING_ENUMS[key])}"
            )
        return

    # Tuple-type spec (e.g., (int, str))
    if isinstance(expected, tuple):
        if not isinstance(value, expected):
            raise ManifestValidationError(
                f"llama_server_flags.{key} expects one of "
                f"{[t.__name__ for t in expected]}, got {type(value).__name__}"
            )
    else:
        # bool is a subclass of int; reject int→bool coercion explicitly
        if expected is bool:
            if not isinstance(value, bool):
                raise ManifestValidationError(
                    f"llama_server_flags.{key} expects bool, got {type(value).__name__}"
                )
        elif expected is int and isinstance(value, bool):
            # Reject bool-for-int coerce
            raise ManifestValidationError(
                f"llama_server_flags.{key} expects int, got bool (Python "
                "bool-is-int coerce explicitly rejected)"
            )
        elif expected is float and isinstance(value, int) and not isinstance(value, bool):
            pass  # int → float promotion OK
        elif not isinstance(value, expected):
            raise ManifestValidationError(
                f"llama_server_flags.{key} expects {expected.__name__}, "
                f"got {type(value).__name__}"
            )

    # Numeric bounds (DoS prevention)
    if key in SAFE_LLAMA_FLAG_BOUNDS:
        lo, hi = SAFE_LLAMA_FLAG_BOUNDS[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if lo is not None and value < lo:
                raise ManifestValidationError(
                    f"llama_server_flags.{key}={value} below min {lo}"
                )
            if hi is not None and value > hi:
                raise ManifestValidationError(
                    f"llama_server_flags.{key}={value} above max {hi}"
                )


def _validate_parallel_ctx(flags: dict[str, Any]) -> None:
    """Cross-field guard: a parallel>1 config must split ctx_size cleanly into
    per-slot windows that each meet PER_SLOT_CTX_FLOOR.

    llama.cpp's --parallel N divides the AGGREGATE ctx_size across N concurrent
    slots. If ctx_size is not divisible by parallel, the per-slot window is
    silently truncated; if ctx_size // parallel is below the floor, each slot's
    usable context is too small. Rejected at manifest-validation time (matches
    the file's ManifestValidationError style) so an over-subscribed parallel
    config never reaches spawn. No-op when parallel <= 1 (back-compat).
    """
    parallel = flags.get("parallel", 1)
    # parallel itself is allowlist/bounds-validated by _validate_flag_value;
    # guard against a non-int sneaking through (it would already have raised).
    if not isinstance(parallel, int) or isinstance(parallel, bool) or parallel <= 1:
        return
    # A parallel>1 config MUST set kv_unified:true. Without a unified
    # KV pool, --parallel N's per-slot KV accounting diverges from the single
    # count the VRAM gate uses (that count is only accidentally correct because
    # --parallel divides ctx). The unified pool keeps the cache exact and flat
    # across concurrent slots (verified: a 35B model at parallel:2 + kv_unified adds ~0 VRAM).
    if not bool(flags.get("kv_unified", False)):
        raise ManifestValidationError(
            f"llama_server_flags: parallel={parallel} requires kv_unified: true "
            "(a unified KV pool keeps the cache accounting exact and flat across "
            "concurrent slots). Add 'kv_unified: true'."
        )
    ctx_size = flags.get("ctx_size")
    if not isinstance(ctx_size, int) or isinstance(ctx_size, bool):
        # No ctx_size to validate against; ctx-fit is enforced elsewhere
        # (safety.check_kv_cache_fit). Per-slot split check needs a concrete int.
        return
    if ctx_size % parallel != 0:
        raise ManifestValidationError(
            f"llama_server_flags: ctx_size={ctx_size} is not divisible by "
            f"parallel={parallel}; the per-slot KV window would be silently "
            f"truncated. Choose a ctx_size that divides evenly across {parallel} "
            "slots."
        )
    per_slot = ctx_size // parallel
    if per_slot < PER_SLOT_CTX_FLOOR:
        raise ManifestValidationError(
            f"llama_server_flags: ctx_size={ctx_size} split across "
            f"parallel={parallel} gives {per_slot} tokens/slot, below the "
            f"PER_SLOT_CTX_FLOOR={PER_SLOT_CTX_FLOOR}. Raise ctx_size or lower "
            "parallel."
        )


class Manifest(BaseModel):
    """A single model manifest from /var/lib/turbohaul/manifests/<tag>.yaml."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_tag: str
    display_name: str = ""
    description: str = ""
    gguf_blob_sha256: str
    # OPTIONAL vision projector as a CONTENT-ADDRESSED blob sha (NOT a path). The
    # manager resolves it to the blob store and injects --mmproj at spawn; the raw
    # path-bearing `mmproj` CLI flag stays in DENIED_FLAGS. Empty = text-only.
    mmproj_blob_sha256: str = ""
    gguf_size_bytes: int = Field(default=0, ge=0)
    context_size: int = Field(default=2048, ge=1)
    expected_vram_bytes: int = Field(default=0, ge=0)  # mandatory for VRAM-fit pre-check
    # Opt-in smart GPU auto-placer. False (default) = today's behavior,
    # honor llama_server_flags.main_gpu verbatim. True + split_mode:none = the
    # manager picks the least-loaded card at admit time (manager._auto_pick_gpu).
    # A plain top-level field (NOT a llama_server_flag) -- model_config
    # extra="forbid" above means it must be declared here to be legal.
    auto_place: bool = False
    # Additive hybrid (qwen35 SSM+attn) support.
    # arch: model architecture identifier. Empty string = unknown/legacy (existing
    #   models ship with no arch field → default ""). The hybrid KV-fit branch in
    #   safety.py activates ONLY when arch == "qwen35".
    arch: str = ""
    # hybrid_kv_ratio: fraction of layers that contribute to per-token KV growth
    #   (1.0 = pure attention, <1.0 for SSM+attn hybrids). SSM layers store a
    #   fixed recurrent state (constant per-token), NOT a growing per-token KV
    #   cache, so only the attention-layers' fraction contributes to per-token KV.
    #   Default 1.0 = byte-identical to today for any manifest without this field.
    #   Note: name matches the safety.py function param for direct passthrough.
    hybrid_kv_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    # OPTIONAL operator-measured effective KV cache cost, in BYTES
    #   per token (already post-quant + post-hybrid — e.g. derived from an
    #   nvidia-smi measurement: marginal_KV_MiB * 1048576 / ctx_size). When set,
    #   it OVERRIDES both the dimension-aware estimate and the file-size heuristic
    #   for this model's KV-fit gate — authoritative because a live measurement
    #   beats a first-principles estimate (same rationale as expected_vram_bytes
    #   winning for cpu_moe configs). Default None = disabled = byte-identical to
    #   the hybrid-support behavior. Units are BYTES/token (NOT KB/token): e.g. a
    #   measured 13.5 KiB/token is 13824.0 here.
    kv_bytes_per_token: float | None = Field(default=None, ge=1024.0)  # floor rejects a KiB-vs-bytes typo (a value <1 KiB/token would silently under-count -> gate always passes)
    revision: int = Field(default=1, ge=1)  # ETag value
    llama_server_flags: dict[str, Any] = Field(default_factory=dict)
    prompt_template: PromptTemplate = Field(default_factory=PromptTemplate)

    @field_validator("model_tag")
    @classmethod
    def _tag_safe(cls, v: str) -> str:
        validate_tag(v)
        return v

    @field_validator("gguf_blob_sha256")
    @classmethod
    def _sha256_format(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ManifestValidationError(
                f"gguf_blob_sha256 must be 64 hex chars; got {v[:32]}... (len={len(v)})"
            )
        return v

    @field_validator("mmproj_blob_sha256")
    @classmethod
    def _mmproj_sha256_format(cls, v: str) -> str:
        # Empty = text-only. Else must be a 64-hex blob sha (NOT a path), so the
        # manager can only ever resolve it inside the content-addressed blob store.
        if v and not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ManifestValidationError(
                f"mmproj_blob_sha256 must be empty or 64 hex chars; got {v[:32]}... (len={len(v)})"
            )
        return v

    @field_validator("llama_server_flags")
    @classmethod
    def _flags_allowlist(cls, v: dict[str, Any]) -> dict[str, Any]:
        for key, value in v.items():
            if key in DENIED_FLAGS:
                raise ManifestValidationError(
                    f"llama_server_flags.{key} is explicitly denied "
                    f"(path-traversal/RCE class)."
                )
            # Suffix-pattern forward-defense: catches future upstream pulls
            # that ship path-bearing flags before they reach DENIED_FLAGS.
            _suffix_guard_check(key)
            if key not in SAFE_LLAMA_FLAGS:
                raise ManifestValidationError(
                    f"llama_server_flags.{key} is not in the closed allowlist. "
                    f"unknown flags REJECTED. "
                    f"Allowlist currently has {len(SAFE_LLAMA_FLAGS)} entries."
                )
            _validate_flag_value(key, value)
        # Cross-field: parallel>1 must split ctx_size into per-slot windows that
        # meet PER_SLOT_CTX_FLOOR (runs in the same validation pass).
        _validate_parallel_ctx(v)
        return v


def _safe_manifest_path(manifests_root: Path, tag: str) -> Path:
    """Resolve manifest path with realpath check (tag path-traversal guard)."""
    validate_tag(tag)
    manifests_root = Path(manifests_root)
    target_unresolved = manifests_root / f"{tag}.yaml"
    target = target_unresolved.resolve()
    root_real = manifests_root.resolve()
    try:
        target.relative_to(root_real)
    except ValueError as e:
        raise ManifestValidationError(
            f"manifest path {target} escapes manifests root {root_real}"
        ) from e
    if target_unresolved.is_symlink() or target.is_symlink():
        raise ManifestValidationError(
            f"manifest path is a symlink - refusing (safety)"
        )
    return target


def read_manifest(manifests_root: Path, tag: str) -> Manifest:
    """Load and validate a manifest by tag."""
    path = _safe_manifest_path(manifests_root, tag)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {tag}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ManifestValidationError(
            f"manifest root must be mapping, got {type(data).__name__}"
        )
    return Manifest(**data)


def manifest_etag(manifests_root: Path, tag: str) -> str:
    m = read_manifest(manifests_root, tag)
    return f'"{m.revision}"'


def write_manifest_atomic(
    manifests_root: Path, manifest: Manifest, if_match: str | None = None
) -> Manifest:
    """Atomic write with ETag/If-Match concurrency check.

    - First write (no existing manifest): writes as-is, revision preserved;
      if_match must be None on create (else 412).
    - Subsequent writes: if_match REQUIRED. Mismatch -> ConcurrencyError.
      Missing -> ConcurrencyError too (guards the lost-update class: previously
      this silently overwrote a concurrent write).
    - POSIX-atomic: tempfile-in-same-dir + fsync(file) + rename + fsync(dir).
    """
    target = _safe_manifest_path(manifests_root, manifest.model_tag)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        existing = read_manifest(manifests_root, manifest.model_tag)
        if if_match is None:
            # Refuse update without If-Match. Previously a caller could omit
            # the header and silently overwrite the concurrent write of
            # another caller. Lost-update class.
            raise ConcurrencyError(
                "If-Match header required for manifest update "
                f"(current ETag is \"{existing.revision}\")"
            )
        actual = f'"{existing.revision}"'
        if if_match != actual:
            raise ConcurrencyError(
                f"If-Match {if_match!r} does not match current ETag {actual!r}"
            )
        # Increment revision on update
        manifest = manifest.model_copy(update={"revision": existing.revision + 1})

    # Serialize
    payload = manifest.model_dump(mode="json")
    yaml_text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)

    # Atomic write
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".yaml", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(yaml_text)
            f.flush()
            os.fsync(f.fileno())
        # chmod the tempfile BEFORE rename so the final inode never has a
        # window with tempfile's default mode (mkstemp is 0o600 on Linux
        # already, this is paranoia-grade defense in depth).
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
        # fsync parent dir (POSIX durability)
        dir_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise

    return manifest


def list_manifests(manifests_root: Path) -> list[str]:
    """Return sorted list of valid manifest tag names."""
    manifests_root = Path(manifests_root)
    if not manifests_root.exists():
        return []
    tags: list[str] = []
    for p in manifests_root.iterdir():
        if p.suffix == ".yaml" and not p.name.startswith("."):
            tag = p.stem
            if TAG_RE.match(tag):
                tags.append(tag)
    return sorted(tags)


def delete_manifest(manifests_root: Path, tag: str) -> bool:
    """Delete a manifest. Returns True if existed and removed."""
    target = _safe_manifest_path(manifests_root, tag)
    try:
        target.unlink()
        return True
    except FileNotFoundError:
        return False


# === llama-server CLI flag mapping ===
def flags_to_argv(flags: dict[str, Any]) -> list[str]:
    """Map snake_case flags dict to llama-server CLI argv.

    Validates against SAFE_LLAMA_FLAGS allowlist (defense-in-depth; manifest
    validator already enforces this on parse).

    Boolean True → `--<flag>` (no value).
    Boolean False → flag OMITTED (not `--<flag> false`).
    Other types → `--<flag> <value>`.

    flash_attn tri-state:
    - flash_attn bool True → "--flash-attn on"
    - flash_attn bool False → "--flash-attn off"
    - flash_attn str → "--flash-attn <value>"
    """
    argv: list[str] = []
    for key, value in flags.items():
        if key not in SAFE_LLAMA_FLAGS or key in DENIED_FLAGS:
            raise ManifestValidationError(
                f"flag {key} blocked at argv-build (allowlist enforcement)"
            )
        cli_key = "--" + key.replace("_", "-")
        # Special-case: flash_attn tri-state CLI
        if key == "flash_attn":
            if isinstance(value, bool):
                argv.extend([cli_key, "on" if value else "off"])
            else:
                argv.extend([cli_key, str(value).lower()])
            continue
        if isinstance(value, bool):
            if value:
                argv.append(cli_key)
            # else omit
        else:
            argv.extend([cli_key, str(value)])
    return argv
