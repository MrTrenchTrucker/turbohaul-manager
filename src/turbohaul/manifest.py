"""Per-model manifest: closed flag allowlist + atomic writes + ETag/If-Match concurrency.

See ARCHITECTURE.md for the manifest design.
Addresses the flag-injection RCE class, tag path traversal,
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
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from turbohaul.safety import KV_CACHE_TYPES

log = logging.getLogger(__name__)


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
    "cache_type_k": str,  # enum: see safety.KV_CACHE_TYPES for the full list
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
    # tensor_split: CSV "N,N[,...]" per-GPU split ratio. Special-cased in
    # _validate_flag_value (parse_strict_csv_numeric) below; a spawn-time
    # gate (safety.check_tensor_split_devices) additionally refuses when the
    # element count doesn't match the visible device count, since device
    # count is a runtime property a static manifest validator can't see.
    #
    # Threat model: the "CSV-string with shell-meta risk" concern does not apply
    # to this codebase:
    # subprocess_mgr.py never sets shell=True, and flags_to_argv below builds
    # argv as a list, so the whole value is one opaque argv element no shell
    # ever parses. The flag is allowed because a strict validator exists.
    # Threat model, priority order:
    #   (1) malformed/non-numeric -> undefined engine behaviour
    #   (2) element count != visible device count -> wrong placement -> OOM
    #       (the real blast radius; checked at spawn, not here)
    #   (3) negative/degenerate (all-zero) splits -> undefined placement
    #   (4) shell metacharacters -> defense-in-depth only, kept because the
    #       spawn path could change
    "tensor_split": str,
    "fit": str,                 # enum on/off (Tom's Fork auto-mem-fit)
    "fit_ctx": int,
    # NOTE: fit_target is also a CSV string; it could reuse
    # parse_strict_csv_numeric (only tensor_split is wired here -- each
    # allowlist entry is validated independently).
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
    # === Reasoning — preserved-thinking ===
    "reasoning_format": str,    # enum: none/deepseek/deepseek-legacy/auto
    "reasoning": str,           # enum: on/off/auto
    "reasoning_budget": int,    # -1/0/N — preserved-thinking knob
    # === Speculative decoding: MTP / D-Flash / D-Spark (PR #22673 = build b9180; D-Flash and
    # D-Spark are also supported) ===
    # Composes with TurboQuant cache_type_k/v (turbo2/3/4) + flash_attn. Spawn-level argv (cold-spawn to apply).
    "spec_type": str,                    # enum: draft-mtp / draft-dflash / draft-dspark (single-select by construction)
    "spec_draft_n_max": int,             # max draft tokens per step (MTP default 3)
    "spec_draft_n_min": int,             # min draft tokens per step
    "spec_draft_p_min": float,           # min prob to continue drafting
    "spec_draft_p_split": float,         # draft split probability threshold
    "spec_draft_ngl": int,               # draft GPU layers (bundled MTP head; usually same device)
    "spec_draft_backend_sampling": bool, # backend-side sampling for the draft path
    "spec_draft_type_k": str,            # draft-cache K quant; enum: see safety.KV_CACHE_TYPES
    "spec_draft_type_v": str,            # draft-cache V quant; enum: see safety.KV_CACHE_TYPES
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
    "log_verbosity": (0, 5),
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
    "cache_type_k": KV_CACHE_TYPES,
    "cache_type_v": KV_CACHE_TYPES,
    "rope_scaling": {"none", "linear", "yarn"},
    "split_mode": {"none", "layer", "row", "tensor"},
    "fit": {"on", "off"},
    "reasoning_format": {"none", "deepseek", "deepseek-legacy", "auto"},
    "reasoning": {"on", "off", "auto"},
    "pooling": {"none", "mean", "cls", "last", "rank"},
    "log_colors": {"on", "off", "auto"},
    # spec_type: engine's real CLI name-map (common/speculative.cpp) registers
    # draft-simple / draft-eagle3 / draft-mtp / draft-dflash / draft-dspark
    # (verified against mainline speculative.cpp:33-37). We enable
    # draft-mtp (existing), plus draft-dflash and draft-dspark (both are ported here;
    # D-Spark = D-Flash + a Markov head, not an alternative -- exposing both because
    # different HF-published draft checkpoints may ship one without the other).
    # draft-simple (separate external draft model) and draft-eagle3 (deliberately
    # out of scope) stay excluded -- narrower than the engine's real enum on purpose.
    "spec_type": {"draft-mtp", "draft-dflash", "draft-dspark"},
    "spec_draft_type_k": KV_CACHE_TYPES,
    "spec_draft_type_v": KV_CACHE_TYPES,
}

# Speculative types whose engine dispatch requires reserved recurrent-state
# sequences. Mirrors the engine's own
# need_n_rs_seq(): returns nonzero iff
# `types` contains MTP, DFLASH, or DSPARK. EAGLE3 is deliberately excluded --
# not handled here. manager.py's three mirror
# sites (spawn-gate KV estimate x2, engine fingerprint's n_rs_seq stamp) all
# key off this ONE set (the intent being
# "one manifest field, read the same way everywhere") so they cannot
# silently diverge from each other or from the engine.
SPEC_TYPES_NEEDING_RS_SEQ: frozenset[str] = frozenset(
    {"draft-mtp", "draft-dflash", "draft-dspark"}
)

# Speculative types that load a STANDALONE draft model (their own separate
# GGUF, resolved via spec_draft_gguf_blob_sha256 -- see manager.py's
# --spec-draft-model injection sites) rather than a bundled head merged into
# the target's own GGUF.
# A DIFFERENT question from SPEC_TYPES_NEEDING_RS_SEQ above -- draft-mtp
# needs rs_seq reservation but does NOT load a standalone drafter (its head
# lives inside the target GGUF), so the two sets overlap on draft-dflash/
# draft-dspark without being the same set. This is deliberately a SECOND,
# NAMED constant rather than reusing SPEC_TYPES_NEEDING_RS_SEQ -- gating the
# standalone-draft-model injection on the 3-member rs_seq set would still
# admit a nonsensical draft-mtp + spec_draft_gguf_blob_sha256 manifest,
# loading an unbudgeted second model exactly like the bug this set prevents,
# just reached via "also picking draft-mtp" instead of "omitting spec_type."
# The D-Spark config-surface test asserts this set is a strict
# subset of SPEC_TYPES_NEEDING_RS_SEQ so the two cannot silently drift into
# contradiction (e.g. a future standalone type added here without also
# being added there).
SPEC_TYPES_WITH_STANDALONE_DRAFT_MODEL: frozenset[str] = frozenset(
    {"draft-dflash", "draft-dspark"}
)

# === Per-model flag defaults (applied when a manifest does not set them) ===
# These are DEFAULTS, never overrides: a value present in a manifest's own
# llama_server_flags always wins, so hand-tuned models are unaffected. They exist
# because an unset flag otherwise silently inherits llama-server's own default,
# which is not the value we want by default:
#
#   ctx_checkpoints -- llama-server defaults to 32. The checkpoint ladder is what
#     lets a slot restore PARTIALLY when the incoming conversation is a near-miss
#     against the saved one, instead of recomputing the whole prompt. A ladder of
#     1 discards the engine's own fallback rung, turning every near-miss into a
#     full recompute; 2 keeps it. A ladder of 2 lets a warm restore rebuild a
#     very long context by processing only a small fraction of its tokens; cold
#     restore at this depth has not been characterised.
#
#   cache_type_k / cache_type_v -- llama-server defaults to f16. The turbo cache
#     types are purpose-built for KV data rather than a general-purpose
#     quantisation applied to it, so they hold accuracy well below the width at
#     which q8_0 would. turbo3 (3-bit, 0.1875x f16) is the balance point: ~2.7x
#     smaller than q8_0, and deliberately NOT the smallest available -- turbo2
#     trades more fidelity than is wanted as a general default.
#
# Adding an entry here changes behaviour for every model that has not set it
# explicitly, so keep this table small and each entry justified above.
MANIFEST_FLAG_DEFAULTS: dict[str, Any] = {
    "ctx_checkpoints": 2,
    "cache_type_k": "turbo3",
    "cache_type_v": "turbo3",
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

# Suffix-pattern forward-defense (rejects unknown path-bearing flag names).
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
    # Closes a gap for draft-model manifest fields:
    # "spec_draft_model" (same arbitrary-
    # local-path-read risk class as the already-denied model_draft/model)
    # matched NEITHER DENIED_FLAGS (exact-string only) nor any pattern
    # above. No existing SAFE_LLAMA_FLAGS key ends
    # in "_model", so this rejects
    # nothing that works today
    # (names ending in "_model" are denied).
    re.compile(r".*_model$"),
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
    "samplers",               # semi-colon list — deferred (validator needed)
    "dry_sequence_breaker",   # str list — deferred
    "chat_template_kwargs",   # JSON-str — deferred (recursive scalar validator needed)
    "reasoning_budget_message", # str injected mid-stream — deferred (length-cap + ctrl-char strip needed)
    "fit_target",             # CSV "MiB,MiB" — deferred to a later pass
}


# === Tag validation regex (tag path-traversal guard, CRITICAL) ===
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

# === Plugin resource_key validation regex ===
# An operator-declared key into the boot-config plugin registry (config.py) --
# NEVER a URL, NEVER a path. Deliberately narrower than TAG_RE (no dots): a
# resource_key is a registry lookup key, not a filename stem, so there is no
# reason to admit the characters TAG_RE allows for that purpose.
RESOURCE_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# === Plugin provides_routes validation regex ===
# A DOWNSTREAM HTTP ROUTE this plugin identity offers, e.g. "/analyze-audio" --
# the `path` an agent puts in POST /api/plugins/{tag}/invoke's body. It is
# ATTACKER-INFLUENCED: anything that can write a manifest can write this list,
# so it is validated as hostile input, not as documentation.
#
# The character class is a CLOSED allowlist, which is what makes the rules
# below hold rather than merely being checked: no ":" (no "http:" scheme, no
# "host:port"), no "%" (so "%2e%2e" and every other encoded traversal is
# unspellable, not merely unmatched), no "\\", no "@" (no URL userinfo), no
# "?" or "#" (query and fragment belong in `payload`, not in the route), no
# whitespace or control characters. Leading "/" is mandatory and "//" is
# rejected separately, because "//evil.example" is protocol-relative and is
# the one shape that reads like a path but resolves OFF-HOST.
#
# NAMED "routes", NOT "paths", by design: in this codebase
# "_path" means FILESYSTEM (manifests_path, health_path, auth_token_file's
# sibling naming, and the .*_path$ suffix forward-defense above). These are
# HTTP routes and must not be named one careless edit away from a guard that
# would be RIGHT to reject a filesystem path here.
PROVIDES_ROUTE_RE = re.compile(r"^/[A-Za-z0-9._~/-]{0,127}$")

# === Plugin provides_executables validation regex ===
# A BARE BINARY NAME this plugin serves, e.g. "ffmpeg" -- never a path, never
# an argument. The generic exec bridge that CONSUMES this is a separate component and
# is deliberately not built here; this is the declaration and its guard only.
# No "/" and no "." leading, so neither "/usr/bin/x", "../../bin/x" nor "./x"
# is spellable. The first character may not be "-": a name like "--upload-file"
# would be argv, not a binary, and an exec bridge that interpolates a declared
# name into a command line should never be handed one that can pose as a flag.
PROVIDES_EXECUTABLE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,63}$")

# Bounds on both lists. A manifest declaring hundreds of routes is not a
# feature request, it is either a mistake or an attempt to bloat every
# /api/plugins response; fail it loudly at load rather than serve it.
MAX_PROVIDES_ENTRIES = 64

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


# tensor_split (and fit_target, which is not yet wired) share this shape:
# a CSV list of N non-negative decimal numbers, bounded by the count and length limits below.
_TENSOR_SPLIT_MIN_COUNT = 2
_TENSOR_SPLIT_MAX_COUNT = 16
_CSV_NUMERIC_MAX_LEN = 64


def parse_strict_csv_numeric(
    value: str, *, min_count: int, max_count: int, max_len: int = _CSV_NUMERIC_MAX_LEN,
) -> list[float]:
    """Strict whitelist parser for a "N,N[,...]" CSV of non-negative decimals.

    Shared helper: wired to tensor_split here; fit_target
    is not yet wired to it and could reuse this unchanged.

    Whitelist-only by design: the value must fullmatch ``[0-9.,]+`` before
    any per-field parse runs, so scientific notation (1e3), inf, nan,
    signs (+/-), and whitespace are excluded by the character class itself
    — they never reach a `float()` call. This sidesteps the fact that
    Python's `float()` silently ACCEPTS `"1e3"`, `"inf"`, and `"nan"`; a
    naive `try: float(x)` validator would admit all three.

    Raises ManifestValidationError on any violation. Returns the parsed
    values (order preserved) when the whole string validates.
    """
    if not isinstance(value, str):
        raise ManifestValidationError(
            f"expected str, got {type(value).__name__}: {value!r}"
        )
    if not value or len(value) > max_len:
        raise ManifestValidationError(
            f"{value!r} length {len(value)} outside 1..{max_len} chars"
        )
    if not re.fullmatch(r"[0-9.,]+", value):
        raise ManifestValidationError(
            f"{value!r} contains characters outside [0-9.,] — no whitespace, "
            "signs, scientific notation, or shell metacharacters"
        )
    fields = value.split(",")
    if len(fields) < min_count or len(fields) > max_count:
        raise ManifestValidationError(
            f"{value!r} has {len(fields)} comma-separated field(s); need "
            f"{min_count}..{max_count}"
        )
    nums: list[float] = []
    for f in fields:
        if not re.fullmatch(r"\d+(\.\d+)?", f):
            raise ManifestValidationError(
                f"{value!r} has a malformed field {f!r} — each field must "
                "be a plain non-negative decimal, e.g. '0.5' or '3'"
            )
        nums.append(float(f))  # safe: f is already whitelisted to [0-9.]+
    if not any(n > 0 for n in nums):
        raise ManifestValidationError(
            f"{value!r} is degenerate (all fields are zero); at least one "
            "must be > 0"
        )
    return nums


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

    # Special-case: tensor_split (strict CSV numeric list)
    if key == "tensor_split":
        if not isinstance(value, str):
            raise ManifestValidationError(
                f"llama_server_flags.tensor_split expects str, got "
                f"{type(value).__name__}"
            )
        parse_strict_csv_numeric(
            value,
            min_count=_TENSOR_SPLIT_MIN_COUNT,
            max_count=_TENSOR_SPLIT_MAX_COUNT,
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


# Import-time self-check: every default must itself be a flag the allowlist
# accepts. The defaults are injected by a model_validator, which runs AFTER the
# llama_server_flags field_validator -- so an invalid entry in the table above
# would otherwise bypass validation entirely and reach llama-server's argv. Fail
# at import instead, where it is impossible to miss.
for _flag, _default in MANIFEST_FLAG_DEFAULTS.items():
    if _flag not in SAFE_LLAMA_FLAGS:
        raise ManifestValidationError(
            f"MANIFEST_FLAG_DEFAULTS[{_flag!r}] is not in SAFE_LLAMA_FLAGS"
        )
    _validate_flag_value(_flag, _default)
del _flag, _default


def _validate_parallel_split(parallel: int, ctx_size: Any) -> None:
    """The per-slot split check itself, extracted so it can be exercised
    directly (bypassing the kv_unified requirement in _validate_parallel_ctx
    below, which currently makes this unreachable from the public Manifest
    API). Applies WITHOUT a unified KV pool: llama.cpp's --parallel N then
    divides ctx_size across N concurrent slots (llama-context.cpp:230-233,
    the `else` branch). If ctx_size is not divisible by parallel, the
    per-slot window is silently truncated; if ctx_size // parallel is below
    PER_SLOT_CTX_FLOOR, each slot's usable context is too small.
    """
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


def _validate_parallel_ctx(flags: dict[str, Any]) -> None:
    """Cross-field guard: a parallel>1 config WITHOUT a unified KV pool must
    split ctx_size cleanly into per-slot windows that each meet
    PER_SLOT_CTX_FLOOR.

    llama.cpp's --parallel N divides the AGGREGATE ctx_size across N concurrent
    slots ONLY when kv_unified is false (vendored engine,
    llama-context.cpp:230-233: kv_unified true -> n_ctx_seq = n_ctx, the FULL
    context per slot; false -> n_ctx_seq = n_ctx / n_seq_max, the split this
    function was written to guard). If ctx_size is not divisible by parallel,
    the per-slot window is silently truncated; if ctx_size // parallel is below
    the floor, each slot's usable context is too small. Rejected at
    manifest-validation time (matches the file's ManifestValidationError style)
    so an over-subscribed parallel config never reaches spawn. No-op when
    parallel <= 1 (back-compat) or when kv_unified is true, since a unified
    pool gives every slot the full ctx_size and neither constraint applies.
    """
    parallel = flags.get("parallel", 1)
    # parallel itself is allowlist/bounds-validated by _validate_flag_value;
    # guard against a non-int sneaking through (it would already have raised).
    if not isinstance(parallel, int) or isinstance(parallel, bool) or parallel <= 1:
        return
    # Design rule: a parallel>1 config MUST set kv_unified:true. Without a unified
    # KV pool, --parallel N's per-slot KV accounting diverges from the single
    # count the VRAM gate uses (that count is only accidentally correct because
    # --parallel divides ctx). The unified pool keeps the cache exact and flat
    # across concurrent slots (a unified pool, kv_unified, adds almost no VRAM per extra slot).
    kv_unified = bool(flags.get("kv_unified", False))
    if not kv_unified:
        raise ManifestValidationError(
            f"llama_server_flags: parallel={parallel} requires kv_unified: true "
            "(a unified KV pool keeps the cache accounting exact and flat across "
            "concurrent slots). Add 'kv_unified: true'."
        )
    # Past this point kv_unified is guaranteed true (the raise above), so per
    # llama-context.cpp each slot already gets the FULL ctx_size, not a
    # 1/parallel split -- _validate_parallel_split only applies to the split
    # that happens WITHOUT a unified pool. Gating the call on `not kv_unified`
    # here means it is CURRENTLY UNREACHABLE as long as the requirement above
    # stays mandatory -- intentional, not an oversight: it keeps the
    # split-validation logic correctly scoped to its real precondition, so if
    # that requirement is ever relaxed for some other reason, this check
    # reactivates automatically instead of silently staying wrong (which is
    # the exact failure mode this gating avoids). _validate_parallel_split is
    # directly unit-tested on its own (bypassing this gate, not weakening it)
    # to prove the reactivation claim is real, not just documented.
    if not kv_unified:
        _validate_parallel_split(parallel, flags.get("ctx_size"))


def _validate_reasoning_budget(flags: dict[str, Any], model_tag: str) -> None:
    """Save-time cross-field guard: a locked reasoning_budget that
    meets or exceeds a bounded n_predict lets the model burn its whole thinking
    allowance inside <think> and return no answer at all -- both values are
    individually legal (each is its own bounds-checked flag above) and nothing
    compares them against each other. Mirrors _validate_parallel_ctx's error
    shape (field + actual values -> why -> fix instruction). n_predict <= 0
    (i.e. -1, unbounded) is exempt: there is no ceiling for reasoning_budget to
    violate.
    """
    n_predict = flags.get("n_predict")
    reasoning_budget = flags.get("reasoning_budget")
    if not isinstance(n_predict, int) or isinstance(n_predict, bool):
        return
    if not isinstance(reasoning_budget, int) or isinstance(reasoning_budget, bool):
        return
    if n_predict > 0 and reasoning_budget >= n_predict:
        raise ManifestValidationError(
            f"llama_server_flags: reasoning_budget={reasoning_budget} >= "
            f"n_predict={n_predict} for model {model_tag!r} -- "
            "the model would burn its whole reasoning_budget inside <think> "
            "and return no answer. Lower reasoning_budget below n_predict, "
            "or set n_predict: -1 for unbounded."
        )


def cache_reuse_inert_on_mmproj(flags: dict[str, Any], mmproj_blob_sha256: str) -> bool:
    """True when this manifest's ``cache_reuse`` is a no-op on the engine.

    This project's inference engine is llama.cpp, vendored here as a fork at
    ``engine/llama-cpp-turboquant``. On some models llama.cpp silently ignores
    the ``cache_reuse`` setting. This function predicts when that happens, so
    the setting is not simply accepted and then quietly discarded.

    Nothing is broken when it does. Cache reuse on such a model produces no
    wrong output and no error -- it is only inert. So this is not a bug in
    turbohaul to fix, and not a combination to reject: refusing the save would
    refuse a manifest that works today.

    This function is the single source of truth for both the save-time notice
    emitted by ``_validate_cache_reuse_mmproj`` and the marker the
    manifest-read API exposes -- one computation, two surfaces, so they can
    never disagree.

    IMPORTANT: this is a PROXY, not the test llama.cpp actually performs.

    A manifest has no field describing a model's architecture or its rope type,
    so this function keys on the only related signal it does have: whether an
    mmproj blob is set. That identifies multimodal models, which covers the
    common case -- but it is not what the inference engine checks.

    llama.cpp checks this in TWO PLACES, and it is worth keeping them apart.

    At MODEL-LOAD time it decides whether the setting is usable at all, and
    logs why when it is not. There are two separate reasons it can refuse:

      1. An mmproj (the multimodal projector file) is loaded. It logs:
         "cache_reuse is not supported by multimodal"
      2. The context cannot be shifted. It logs:
         "cache_reuse is not supported by this context"

    "Shifting" is llama.cpp's technique for sliding a conversation window
    forward without re-processing the prompt from scratch. A context cannot be
    shifted when the model uses the STEP35 architecture, or when it uses M-ROPE
    or I-M-ROPE position encodings -- rotary encodings that reserve dimensions
    for image and video position, which is why they show up on multimodal
    models. On a typical multimodal model BOTH reasons apply and llama.cpp logs
    both lines on a single load.

    At REQUEST time it re-checks per slot::

        can_cache_reuse = llama_memory_can_shift(...)
                          && !slot.prompt.tokens.has_mtmd

    so a request carrying multimodal tokens skips reuse even on a model where
    the setting survived load. Both checks live in llama.cpp's
    server-context.cpp, and llama.cpp logs its own SRV_WRN / SLT_WRN warnings.

    The KV cache type is NOT one of the reasons. turbo2, turbo3 and turbo4
    never appear anywhere in llama.cpp's shiftability check, so a text-only
    model using turbo KV reuses its prefix cache normally. This is worth
    stating plainly because turbo3 is the default in these manifests, which means
    it is present on nearly every affected model -- and that made it
    look like the cause when it is only a bystander.

    Known blind spot: a model that uses M-ROPE but ships no mmproj really is
    inert, yet has no mmproj for this function to detect, so this function will
    wrongly report it as fine. There is no way to catch that from a manifest
    alone; llama.cpp's own load-time log line is the reliable signal there.
    """
    cache_reuse = flags.get("cache_reuse")
    if not isinstance(cache_reuse, int) or isinstance(cache_reuse, bool):
        return False
    return cache_reuse > 0 and bool(mmproj_blob_sha256)


def _validate_cache_reuse_mmproj(flags: dict[str, Any], mmproj_blob_sha256: str, model_tag: str) -> None:
    """Make an inert cache_reuse visible without reading engine logs.

    Deliberately never raises -- unlike _validate_reasoning_budget above, this
    combination is not wrong (see cache_reuse_inert_on_mmproj's docstring), so
    a save-time reject would refuse to save a manifest that works
    today. Mirrors that function's
    field+values -> why shape for the message only; the verdict is a log line,
    not a ManifestValidationError.

    Logged at INFO, not WARNING: this validator runs on every ModelManifest
    construction (save AND load-from-YAML, per HardenedManifestBase), so a
    manifest listing that reads every manifest on disk would otherwise emit a
    WARNING for every affected model on every single page load -- log-level should
    reflect "worth knowing", not "something is broken and needs fixing now".
    """
    if cache_reuse_inert_on_mmproj(flags, mmproj_blob_sha256):
        log.info(
            "cache_reuse=%s is inert for model %r: an mmproj is set "
            "(mmproj_blob_sha256=%s...) and the engine unconditionally "
            "disables prefix cache reuse for multimodal requests. "
            "The flag is harmless to leave -- it does not error or produce "
            "wrong output -- but it currently does nothing for this model.",
            flags.get("cache_reuse"), model_tag, mmproj_blob_sha256[:12],
        )


# Top-level keys that older manifests may still carry but that no longer mean
# anything. They are dropped on load (with one warning) instead of being rejected
# by extra="forbid", and a one-time sweep removes them from stored files.
RETIRED_MODEL_KEYS: tuple[str, ...] = ("max_instances",)
_MISSING = object()


def _drop_retired_keys(payload: dict) -> list[str]:
    """Remove retired top-level keys from `payload` IN PLACE; return their names.

    Only the keys in RETIRED_MODEL_KEYS are touched: any other key, known or
    unknown, is left alone (an unknown key must still fail extra="forbid"), and
    nested dicts are never inspected or changed. Returns [] when nothing was
    removed.
    """
    return [key for key in RETIRED_MODEL_KEYS if payload.pop(key, _MISSING) is not _MISSING]


class HardenedManifestBase(BaseModel):
    """Fields + validation shared by EVERY manifest variant.

    Split into ModelManifest/PluginManifest because their
    REQUIRED fields and validation shape diverge completely (no overlap on
    gguf_blob_sha256 / llama_server_flags / context_size), but hoist here
    whatever genuinely IS common -- extra="forbid", the model_tag identity +
    traversal guard, and the bookkeeping fields that read_manifest /
    write_manifest_atomic / manifest_etag and the existing API routes already
    read generically off ANY loaded manifest with no kind branch (revision,
    hidden -- as the existing callers show).
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_tag: str
    display_name: str = ""
    description: str = ""
    revision: int = Field(default=1, ge=1)  # ETag value
    # Listings-only visibility. True = omit this manifest from the discovery
    # endpoints (/v1/models and /api/tags) while leaving it FULLY loadable and
    # servable by its exact tag -- nothing in the serve or scheduler path reads
    # this field. Lets a registry carry fixtures/variants without every consumer
    # inheriting them. Absent/false = visible, so existing manifests are
    # unaffected and need no edit. A plain top-level field (NOT a
    # llama_server_flag) -- model_config extra="forbid" above means it must be
    # declared here to be legal. Shared, not ModelManifest-only: api/models.py
    # and api/ollama.py's discovery listings read `.hidden` off every manifest
    # they enumerate with no kind branch.
    hidden: bool = False

    @field_validator("model_tag")
    @classmethod
    def _tag_safe(cls, v: str) -> str:
        validate_tag(v)
        return v


class ModelManifest(HardenedManifestBase):
    """A single model manifest from /var/lib/turbohaul/manifests/<tag>.yaml."""

    kind: Literal["model"] = "model"

    backend: str = "llama.cpp"  # "llama.cpp" (default) or "mlx"
    # MLX backends carry no GGUF blob, so this may be "" when backend == "mlx".
    # llama.cpp backends still require a 64-hex sha256 (enforced by the validator).
    gguf_blob_sha256: str = ""
    # OPTIONAL vision projector as a CONTENT-ADDRESSED blob sha (NOT a path). The
    # manager resolves it to the blob store and injects --mmproj at spawn; the raw
    # path-bearing `mmproj` CLI flag stays in DENIED_FLAGS. Empty = text-only.
    mmproj_blob_sha256: str = ""
    # OPTIONAL standalone draft model (D-Flash / D-Spark)
    # as a CONTENT-ADDRESSED blob sha, same pattern as mmproj_blob_sha256 above --
    # NOT a path, NOT a repo, NOT a URL. The manager resolves it to the blob store
    # and injects the draft-model argv at spawn; spec_draft_model/spec_draft_hf_repo
    # are deliberately NOT allowlisted flags (the raw path/URL forms are exactly
    # the SSRF/arbitrary-file-read class DENIED_FLAGS + the suffix guard exist to
    # block -- see _SUFFIX_GUARD_PATTERNS' `.*_model$` entry). Empty = spec_type
    # doesn't need a standalone draft model (e.g. draft-mtp's head is bundled in
    # the target's own GGUF). Required in practice for draft-dflash/draft-dspark;
    # not enforced as a hard cross-field rule here since a config missing it simply
    # fails to spawn with a clear error, which is an acceptable failure mode for a
    # v1 (matches this file's own degrade-open precedent elsewhere for optional
    # companion fields).
    spec_draft_gguf_blob_sha256: str = ""
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
    #   the prior behaviour. Units are BYTES/token (NOT KB/token): e.g. a value of 13.5
    #   KiB/token is 13824.0 here.
    kv_bytes_per_token: float | None = Field(default=None, ge=1024.0)  # floor rejects a KiB-vs-bytes typo (a value <1 KiB/token would silently under-count -> gate always passes)
    llama_server_flags: dict[str, Any] = Field(default_factory=dict)
    # MLX backend fields (Apple Silicon only). model_repo = HF repo id;
    # model_path = local dir; mlx_server_flags = closed allowlist of
    # mlx_lm.server CLI args.
    model_repo: str = ""
    model_path: str = ""
    mlx_server_flags: dict[str, Any] = Field(default_factory=dict)
    prompt_template: PromptTemplate = Field(default_factory=PromptTemplate)

    # Which flags came from MANIFEST_FLAG_DEFAULTS rather than from the file.
    # A PrivateAttr so it never appears in model_dump() / the YAML. The write
    # path uses it to strip injected defaults back out, so a default is never
    # frozen into a manifest as though an operator had chosen it.
    _defaulted_flags: set[str] = PrivateAttr(default_factory=set)

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_keys_with_warning(cls, data: Any) -> Any:
        """Accept an old-style body: drop a retired key, warn once, never reject.

        extra="forbid" would otherwise turn every manifest file, PUT body or
        import that still carries a retired key into a validation error. This is
        the one place every construction path (read_manifest, parse_manifest, the
        API routes, ModelManifest(**d), TypeAdapter) passes through, so it is the
        one place that drops the key. It works on a copy: the caller's dict is
        never changed. The warning names the tag and the key only, never a value
        or any other manifest content. Unknown keys are untouched and still fail.
        """
        if not isinstance(data, dict) or not any(k in data for k in RETIRED_MODEL_KEYS):
            return data
        data = dict(data)
        tag = data.get("model_tag")
        tag_text = repr(tag[:64]) if isinstance(tag, str) else "<unknown>"
        dropped = _drop_retired_keys(data)
        log.warning(
            "manifest %s: dropped retired key(s) %s (no longer supported)",
            tag_text, ", ".join(dropped),
        )
        return data

    @field_validator("backend")
    @classmethod
    def _backend_valid(cls, v: str) -> str:
        if v not in ("llama.cpp", "mlx"):
            raise ManifestValidationError(
                f"backend must be 'llama.cpp' or 'mlx', got {v!r}"
            )
        return v

    @field_validator("gguf_blob_sha256")
    @classmethod
    def _sha256_format(cls, v: str, info: "ValidationInfo") -> str:
        # MLX models have no GGUF blob, so the empty string is valid for them.
        # llama.cpp models still require a 64-hex sha256.
        if v == "":
            if info.data.get("backend") == "mlx":
                return v
            raise ManifestValidationError(
                "gguf_blob_sha256 must be 64 hex chars for llama.cpp backends; "
                "MLX backends may use an empty string"
            )
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

    @field_validator("spec_draft_gguf_blob_sha256")
    @classmethod
    def _spec_draft_sha256_format(cls, v: str) -> str:
        # Same shape as _mmproj_sha256_format: empty = no standalone draft model.
        # Else must be a 64-hex blob sha (NOT a path/repo/URL), so the manager can
        # only ever resolve it inside the content-addressed blob store.
        if v and not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ManifestValidationError(
                f"spec_draft_gguf_blob_sha256 must be empty or 64 hex chars; "
                f"got {v[:32]}... (len={len(v)})"
            )
        return v

    def is_mlx(self) -> bool:
        """True if this manifest uses the MLX backend."""
        return self.backend == "mlx"

    def is_llama_cpp(self) -> bool:
        """True if this manifest uses the llama.cpp backend."""
        return self.backend == "llama.cpp"

    @field_validator("llama_server_flags")
    @classmethod
    def _flags_allowlist(cls, v: dict[str, Any]) -> dict[str, Any]:
        for key, value in v.items():
            if key in DENIED_FLAGS:
                raise ManifestValidationError(
                    f"llama_server_flags.{key} is explicitly denied "
                    f"(path-traversal/RCE class). See ARCHITECTURE.md."
                )
            # Suffix-pattern forward-defense: catches future Tom's Fork
            # pulls that ship path-bearing flags before they reach DENIED_FLAGS.
            _suffix_guard_check(key)
            if key not in SAFE_LLAMA_FLAGS:
                raise ManifestValidationError(
                    f"llama_server_flags.{key} is not in the closed allowlist. "
                    f"See ARCHITECTURE.md - unknown flags REJECTED. "
                    f"Allowlist currently has {len(SAFE_LLAMA_FLAGS)} entries."
                )
            _validate_flag_value(key, value)
        # Cross-field: parallel>1 must split ctx_size into per-slot windows that
        # meet PER_SLOT_CTX_FLOOR (runs in the same validation pass).
        _validate_parallel_ctx(v)
        return v

    @model_validator(mode="after")
    def _apply_flag_defaults(self) -> "ModelManifest":
        """Fill in MANIFEST_FLAG_DEFAULTS for flags this manifest did not set.

        setdefault, never assignment: an explicit manifest value ALWAYS wins,
        including a deliberate 0 or f16. This runs as a model_validator rather
        than inside the llama_server_flags field_validator because a field
        validator does not fire when the field is absent entirely -- and a newly
        added model whose manifest omits llama_server_flags is exactly the case
        these defaults exist to cover.
        """
        for flag, default in MANIFEST_FLAG_DEFAULTS.items():
            if flag not in self.llama_server_flags:
                self.llama_server_flags[flag] = default
                self._defaulted_flags.add(flag)
        return self

    @model_validator(mode="after")
    def _check_reasoning_budget(self) -> "ModelManifest":
        """Needs llama_server_flags AND model_tag together (the
        error message names the model) -- same reason _apply_flag_defaults
        above is a model_validator rather than living in the llama_server_flags
        field_validator alone, which never sees model_tag.
        """
        _validate_reasoning_budget(self.llama_server_flags, self.model_tag)
        return self

    @model_validator(mode="after")
    def _check_cache_reuse_mmproj(self) -> "ModelManifest":
        """Needs llama_server_flags AND mmproj_blob_sha256 AND
        model_tag together, same reason _check_reasoning_budget above is a
        model_validator rather than living in a single field_validator.
        """
        _validate_cache_reuse_mmproj(self.llama_server_flags, self.mmproj_blob_sha256, self.model_tag)
        return self

    @field_validator("mlx_server_flags")
    @classmethod
    def _mlx_flags_allowlist(cls, v: dict[str, Any]) -> dict[str, Any]:
        # Closed allowlist + type check against the MLX backend's SAFE_MLX_FLAGS.
        # Re-checked at spawn time in mlx_spawn, this is defense-in-depth at parse.
        from turbohaul.mlx_spawn import validate_mlx_flags

        try:
            validate_mlx_flags(v)
        except ValueError as e:
            raise ManifestValidationError(str(e)) from e
        return v


class PluginManifest(HardenedManifestBase):
    """A resource-plugin manifest (media hook).

    Deliberately declares NONE of ModelManifest's fields -- no
    llama_server_flags, no gguf_blob_sha256, no context_size.
    Under a flat optional-fields design, "plugin carries
    no llama_server_flags" would be an invariant some validator has to
    remember to enforce regardless of kind; here it is not an invariant at
    all, it is a fact about the schema -- the field does not exist to violate.
    """

    kind: Literal["plugin"]

    lane: Literal["cpu", "gpu"]
    # Operator-declared key into config.py's boot-time plugin registry --
    # NEVER a URL, NEVER a path (the operator writes
    # boot config; a manifest author cannot, so resolving resource_key against
    # that registry never re-introduces the thing DENIED_FLAGS/the suffix
    # guard exist to block). Shape-only here: regex-constrained the same way
    # model_tag is TAG_RE-constrained. Resolving the key against the actual
    # registry contents is a downstream/runtime concern (manager.py / the api
    # layer, where the loaded config lives) -- by design, not
    # built here; this file has zero config.py import.
    resource_key: str
    capabilities: list[str] = Field(default_factory=list)
    # SELF-DESCRIPTION. `capabilities` says what this plugin can do
    # in human-ish nouns; these two say what an agent can actually CALL. Both
    # default EMPTY, so every manifest on disk today stays valid with zero
    # migration -- the same "absent = byte-identical to before" property
    # `hidden` and `kind` were given.
    #
    # Declared on PluginManifest ONLY. A ModelManifest carrying either FAILS,
    # and it fails STRUCTURALLY rather than by a rule somebody has to remember:
    # HardenedManifestBase sets extra="forbid", so on a ModelManifest these
    # fields do not exist to be violated. Same mechanism as "a plugin carries
    # no llama_server_flags" (see this class's docstring).
    #
    # NOT AN ALLOWLIST -- READ THIS BEFORE RELYING ON IT. provides_routes is a
    # DECLARATION consumed by GET /api/plugins for discovery. The invoke route
    # does NOT check that a caller's `path` appears in this list, and must not
    # be assumed to: enforcement at call time is plugin_invoke._validate_path's
    # job, independently. Adding a route here grants nothing and removing one
    # revokes nothing; this list widens what an agent can DISCOVER, never what
    # it can REACH. Anything that ever wants to treat it as a gate has to make
    # that a deliberate, separately-reviewed change.
    provides_routes: list[str] = Field(default_factory=list)
    # Binaries this plugin serves, e.g. ["ffmpeg", "ffprobe"].
    # Declared here, CONSUMED ELSEWHERE: the generic exec bridge is a separate
    # component and no exec routing exists in this file or in api/plugins.py. It
    # lives here so that a single module owns the schema of both fields and
    # the declaration cannot silently drift from its guard.
    provides_executables: list[str] = Field(default_factory=list)

    @field_validator("provides_routes")
    @classmethod
    def _provides_routes_safe(cls, v: list[str]) -> list[str]:
        """Validate every declared route as hostile input.

        An entry must NEVER be able to redirect a call off-host, escape the
        container's route space, or smuggle a host/port into a listing that is
        contractually host-free.
        """
        if len(v) > MAX_PROVIDES_ENTRIES:
            raise ManifestValidationError(
                f"provides_routes declares {len(v)} entries, max is "
                f"{MAX_PROVIDES_ENTRIES}"
            )
        seen: set[str] = set()
        for entry in v:
            if not isinstance(entry, str):  # pragma: no cover - pydantic types it
                raise ManifestValidationError(
                    f"provides_routes entry {entry!r} is not a string"
                )
            # MESSAGE LAYER, NOT THE GUARD. Deleting this branch changes no
            # verdict, because
            # PROVIDES_ROUTE_RE below already requires a leading "/" and so
            # rejects "analyze-audio" and "" on its own. It is kept because
            # "must start with '/'" is a far better error than a regex dump --
            # but do NOT loosen the regex on the belief that this branch is
            # independently enforcing the rule. It is not.
            if not entry.startswith("/"):
                raise ManifestValidationError(
                    f"provides_routes entry {entry!r} must start with '/' - it is a "
                    "route on the already-resolved plugin container, not a URL"
                )
            # INDEPENDENTLY LOAD-BEARING: "//evil.example/x" is
            # spelled entirely from the closed character class, so the regex
            # passes it. This branch is the whole off-host defense.
            if entry.startswith("//"):
                raise ManifestValidationError(
                    f"provides_routes entry {entry!r} must not start with '//' - a "
                    "protocol-relative reference resolves OFF-HOST"
                )
            # INDEPENDENTLY LOAD-BEARING: the regex ACCEPTS
            # "/../../etc/passwd" -- every character in it is in the closed
            # class -- so this branch is the only thing rejecting rooted
            # traversal. Note that a bare "../etc/passwd" is caught earlier by
            # the leading-slash/regex rule instead, so that shape does NOT
            # exercise this line; the rooted form is the one with distinguishing
            # power, and both are tested.
            if ".." in entry:
                raise ManifestValidationError(
                    f"provides_routes entry {entry!r} must not contain '..'"
                )
            if not PROVIDES_ROUTE_RE.match(entry):
                raise ManifestValidationError(
                    f"provides_routes entry {entry!r} fails regex "
                    f"{PROVIDES_ROUTE_RE.pattern!r} - no scheme (':'), no percent-"
                    "encoding ('%'), no userinfo ('@'), no query or fragment "
                    "('?', '#'), no backslash, no whitespace, max 128 chars"
                )
            if entry in seen:
                raise ManifestValidationError(
                    f"provides_routes declares {entry!r} more than once"
                )
            seen.add(entry)
        return v

    @field_validator("provides_executables")
    @classmethod
    def _provides_executables_safe(cls, v: list[str]) -> list[str]:
        """Validate every declared executable as a BARE NAME.

        Never a path, never a flag. The consumer is an exec bridge that has not
        been built yet, which is exactly why the declaration is constrained now
        rather than when someone is mid-way through interpolating it into argv.
        """
        if len(v) > MAX_PROVIDES_ENTRIES:
            raise ManifestValidationError(
                f"provides_executables declares {len(v)} entries, max is "
                f"{MAX_PROVIDES_ENTRIES}"
            )
        seen: set[str] = set()
        for entry in v:
            if not isinstance(entry, str):  # pragma: no cover - pydantic types it
                raise ManifestValidationError(
                    f"provides_executables entry {entry!r} is not a string"
                )
            # MESSAGE LAYER, NOT THE GUARD: deleting this branch
            # alone changes no verdict -- PROVIDES_EXECUTABLE_RE already rejects "/"
            # and "\\". Dropping BOTH lets "/usr/bin/ffmpeg",
            # "./ffmpeg" and "ffmpeg; rm -rf /" through, which is what proves
            # the class is guarded. Kept for the specific error message only.
            if "/" in entry or "\\" in entry:
                raise ManifestValidationError(
                    f"provides_executables entry {entry!r} must be a bare binary "
                    "name, not a path - the exec bridge resolves it, the manifest "
                    "does not get to say WHERE it lives"
                )
            if ".." in entry:
                raise ManifestValidationError(
                    f"provides_executables entry {entry!r} must not contain '..'"
                )
            if not PROVIDES_EXECUTABLE_RE.match(entry):
                raise ManifestValidationError(
                    f"provides_executables entry {entry!r} fails regex "
                    f"{PROVIDES_EXECUTABLE_RE.pattern!r} - must not start with '-' "
                    "(it would pose as an argv flag) or '.', max 64 chars"
                )
            if entry in seen:
                raise ManifestValidationError(
                    f"provides_executables declares {entry!r} more than once"
                )
            seen.add(entry)
        return v

    @field_validator("resource_key")
    @classmethod
    def _resource_key_shape(cls, v: str) -> str:
        if not RESOURCE_KEY_RE.match(v):
            raise ManifestValidationError(
                f"resource_key {v!r} fails regex ^[a-z0-9][a-z0-9_-]{{0,63}}$ - "
                "ASCII lowercase only, no path separators, no traversal, max 64 "
                "chars, no dots (it is a registry lookup key, not a filename)"
            )
        return v


# A type alias, not a class -- Manifest(**payload) no longer works. Use
# parse_manifest(payload) (below) or TypeAdapter(Manifest).validate_python(payload)
# directly. read_manifest (below) is the one in-file chokepoint that does this;
# api/manifests.py's two other construction sites (PUT + restore-defaults) route
# through parse_manifest as well (see api/manifests.py).
Manifest = Annotated[Union[ModelManifest, PluginManifest], Field(discriminator="kind")]

# Built once at import time (schema-building has real cost; every call site
# reuses this instead of constructing a fresh TypeAdapter per call).
_MANIFEST_ADAPTER: TypeAdapter = TypeAdapter(Manifest)


def parse_manifest(payload: dict) -> "ModelManifest | PluginManifest":
    """Validate a raw manifest payload against the Manifest union. THE chokepoint.

    Pydantic's discriminated-union resolution inspects the raw `kind` key in
    the input mapping to pick a variant BEFORE that variant's own field
    defaults ever run -- verified empirically (pydantic 2.9): a payload with
    no `kind` key at all raises "Unable to extract tag using discriminator
    'kind'" even though ModelManifest.kind defaults to "model". A Python-level
    Literal default is not consulted during tag extraction. So the "every
    existing on-disk manifest (no kind key) still loads with ZERO migration"
    requirement needs an explicit setdefault here, not just the default on
    ModelManifest.kind alone.

    Operates on a copy: the caller's dict is never mutated.
    """
    payload = dict(payload)
    payload.setdefault("kind", "model")
    return _MANIFEST_ADAPTER.validate_python(payload)



def _safe_manifest_path(manifests_root: Path, tag: str) -> Path:
    """Resolve manifest path with realpath check (path-traversal guard)."""
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
            f"manifest path is a symlink - refusing (symlink safety)"
        )
    return target


def read_manifest(manifests_root: Path, tag: str) -> Manifest:
    """Load and validate a manifest by tag.

    THIS FUNCTION'S FAILURE SURFACE IS THE POINT. It raises SIX classes
    from three libraries plus the stdlib, and every caller that wants to
    degrade rather than 500 must enumerate all of them:

      FileNotFoundError      the manifest is absent (raised explicitly below)
      OSError                unreadable for any other reason -- permissions, I/O
      UnicodeDecodeError     the bytes are not valid UTF-8 (from read_text)
      yaml.YAMLError         valid UTF-8, invalid YAML -- an ordinary typo
      ManifestValidationError  bad tag/symlink/escape, or a non-mapping root
      ValidationError        pydantic: parses, but violates the Manifest schema

    ⛔ TWO SUBCLASS RELATIONS ARE EASY TO GET WRONG, and both change what a
    caller must catch:
      · FileNotFoundError IS an OSError -- so `except OSError` swallows
        "absent" too, and a caller that must tell absent from corrupt (a 404
        vs a 500) has to catch FileNotFoundError FIRST.
      · UnicodeDecodeError is a **ValueError, NOT an OSError** -- so
        `except OSError` around a read_text() reads as "unreadable file" and
        does not catch undecodable bytes. Catch UnicodeDecodeError (or
        ValueError) explicitly.

    This function deliberately does NOT wrap these into one class. Doing so
    was considered and rejected: api/models.py renders
    ManifestValidationError into a client-visible `str(e)`, and the
    comment there records that yaml.YAMLError leaks parser position and
    content while pydantic's ValidationError echoes manifest field VALUES.
    Wrapping would pipe both through that 400. The enumeration is the
    contract; the drift test in tests/ pins every consumer against it.
    """
    path = _safe_manifest_path(manifests_root, tag)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {tag}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ManifestValidationError(
            f"manifest root must be mapping, got {type(data).__name__}"
        )
    return parse_manifest(data)


def manifest_etag(manifests_root: Path, tag: str) -> str:
    m = read_manifest(manifests_root, tag)
    return f'"{m.revision}"'


# mtime-keyed cache for HOT, high-frequency callers
# (e.g. manager.py's _effective_cap, which runs on EVERY route including the
# default single-instance HIT path) where the
# manifest is re-read far more often than it actually changes. Keyed by
# resolved path, not tag, so distinct manifests_root trees never collide.
_manifest_cache: "dict[Path, tuple[int, Manifest]]" = {}


def read_manifest_cached(manifests_root: Path, tag: str) -> Manifest:
    """Cached wrapper around ``read_manifest`` for hot, high-frequency call
    sites. NOT a replacement for ``read_manifest`` -- callers on a
    read-modify-write path that must see a just-written manifest immediately
    (the manifest API's own PUT/import routes) must keep calling
    ``read_manifest`` directly.

    Caches the LAST SUCCESSFUL parse only, keyed by (resolved path -> (mtime_ns,
    parsed)). A read that raises (missing/corrupt manifest) is NOT cached and
    re-raises on every call, same as an uncached read — any existing
    degrade-on-exception behavior at the caller (e.g. _effective_cap's
    degrade-to-1) is unaffected. Re-parses only when mtime changes, so an
    edited manifest is picked up on its next read.

    Bounded by ``st_mtime_ns`` granularity (nanosecond on ext4 and most
    modern filesystems, vs whole-second-rounded ``st_mtime`` as a float --
    a live manifest edit landing in the SAME
    second as a preceding read would otherwise be invisible to this cache
    under the coarser float comparison). A manifest edited and re-read
    within the SAME mtime_ns tick (sub-microsecond in practice) can still
    serve the prior cached parse. Manifest edits during live operation are
    an operator action outside the request-serving hot path this cache
    targets, so that residual window is an accepted, documented edge — not
    a correctness gap this cache is meant to close.
    """
    path = _safe_manifest_path(manifests_root, tag)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        # Path vanished/unreadable between resolution and stat — let
        # read_manifest raise its own (more specific) error, same as an
        # uncached call would; don't serve a now-untrustworthy cache entry.
        _manifest_cache.pop(path, None)
        return read_manifest(manifests_root, tag)
    cached = _manifest_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    parsed = read_manifest(manifests_root, tag)
    _manifest_cache[path] = (mtime, parsed)
    return parsed


def _strip_injected_defaults(payload: dict, defaulted: "set[str]") -> None:
    """Remove flags that MANIFEST_FLAG_DEFAULTS supplied, in place.

    Only strips a key whose value still EQUALS the default -- if a caller
    changed it after construction, that is a real choice and is preserved.
    """
    if not defaulted:
        return
    flags = payload.get("llama_server_flags")
    if not isinstance(flags, dict):
        return
    for key in defaulted:
        if key in flags and flags[key] == MANIFEST_FLAG_DEFAULTS.get(key):
            del flags[key]


def write_manifest_atomic(
    manifests_root: Path, manifest: Manifest, if_match: str | None = None
) -> Manifest:
    """Atomic write with ETag/If-Match concurrency check.

    - First write (no existing manifest): writes as-is, revision preserved;
      if_match must be None on create (else 412).
    - Subsequent writes: if_match REQUIRED. Mismatch -> ConcurrencyError.
      Missing -> ConcurrencyError too (a silent overwrite would
      open a lost-update class).
    - POSIX-atomic: tempfile-in-same-dir + fsync(file) + rename + fsync(dir).
    """
    # Capture BEFORE any model_copy below: model_copy() need not carry
    # private attributes, and losing this set would re-persist the defaults.
    defaulted = set(getattr(manifest, "_defaulted_flags", set()) or set())
    target = _safe_manifest_path(manifests_root, manifest.model_tag)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        existing = read_manifest(manifests_root, manifest.model_tag)
        if if_match is None:
            # Refuse update without If-Match. Otherwise a
            # caller could omit the header and silently overwrite the
            # concurrent write of another caller. Lost-update class.
            raise ConcurrencyError(
                "If-Match header required for manifest update "
                f"(current ETag is \"{existing.revision}\")"
            )
        actual = f'"{existing.revision}"'
        if if_match != actual:
            raise ConcurrencyError(
                f"If-Match {if_match!r} does not match current ETag {actual!r}"
            )
        # Increment revision on update. Capture the injected-default set FIRST:
        # model_copy() is not guaranteed to carry private attributes, and losing
        # it here would silently re-persist the defaults this strip exists to
        # prevent.
        manifest = manifest.model_copy(update={"revision": existing.revision + 1})

    # Serialize
    payload = manifest.model_dump(mode="json")
    # A default must never be PERSISTED. Injecting it at load is what makes an
    # unset flag pick up the default value; writing it into the file would freeze
    # today's default as an explicit per-model choice, so a later change to
    # MANIFEST_FLAG_DEFAULTS would silently skip every manifest that had been
    # saved in the meantime -- splitting the installed models between those that follow the
    # default and those that only look like they do.
    _strip_injected_defaults(payload, defaulted)
    yaml_text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)

    # Atomic write
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".yaml", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(yaml_text)
            f.flush()
            os.fsync(f.fileno())
        # chmod the tempfile BEFORE rename so the final inode
        # never has a window with tempfile's default mode (mkstemp is
        # 0o600 on Linux already, this is paranoia-grade defense in depth).
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


def _write_text_atomic(target: Path, text: str) -> None:
    """Replace `target` with `text` the way write_manifest_atomic does.

    Tempfile in the same directory, mode 0o600, fsync the file, os.replace, fsync
    the directory. Any failure removes the tempfile and leaves `target` as it was.
    """
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".yaml", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
        dir_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def migrate_retired_keys(manifests_root: Path) -> list[str]:
    """One-time sweep: remove retired top-level keys from the stored manifests.

    Acts only on the files the store itself loads as manifests (the
    list_manifests selection) that are regular, non-symlink files; backups and
    every other file in the directory are never opened for writing. A file with
    no retired key is not touched at all (same bytes, same mtime). A file with
    one is rewritten atomically with the key removed and every other field
    unchanged, INCLUDING `revision`: this is a pure key removal, not an edit, so
    a client holding the old ETag can still save. A file that cannot be read or
    parsed, or whose root is not a mapping, is left byte-for-byte as it is with
    one warning. A problem with one file never stops the sweep of the others and
    never raises out of this function. Returns the tags rewritten, sorted;
    a second run returns [].
    """
    rewritten: list[str] = []
    for tag in list_manifests(manifests_root):
        try:
            path = _safe_manifest_path(manifests_root, tag)
        except ManifestValidationError:
            log.debug("retired-key sweep: skipping %r (symlink or unsafe path)", tag)
            continue
        except OSError as e:
            log.warning("retired-key sweep: cannot resolve %r (%s)", tag, type(e).__name__)
            continue
        try:
            if not path.is_file():
                continue
            data = yaml.safe_load(path.read_text())
        except Exception as e:  # noqa: BLE001 - one file must not stop the sweep
            log.warning(
                "retired-key sweep: leaving %r untouched, cannot read it (%s)",
                tag, type(e).__name__,
            )
            continue
        if not isinstance(data, dict):
            log.warning(
                "retired-key sweep: leaving %r untouched, its root is not a mapping", tag
            )
            continue
        dropped = _drop_retired_keys(data)
        if not dropped:
            continue
        try:
            _write_text_atomic(
                path, yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
            )
        except Exception as e:  # noqa: BLE001 - one file must not stop the sweep
            log.warning(
                "retired-key sweep: could not rewrite %r, left as it was (%s)",
                tag, type(e).__name__,
            )
            continue
        log.warning(
            "manifest %r: removed retired key(s) %s from the stored file",
            tag, ", ".join(dropped),
        )
        rewritten.append(tag)
    return sorted(rewritten)


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

    Tri-state flash_attn handling:
    - flash_attn bool True → "--flash-attn on" (Tom's Fork tri-state)
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
