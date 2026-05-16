"""Per-model manifest: closed flag allowlist + atomic writes + ETag/If-Match concurrency.

Per v0.2 ARCHITECTURE.md §8 + §8.1 + §8.2.
Addresses Security #58 F1 (CRIT - flag injection RCE), F2 (CRIT - tag path traversal),
Brainstormer F4 (lost-update), Failure Predictor M4 (atomic-write).
"""
import contextlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


# === Closed allowlist of safe llama-server flags (v0.2 §8.1, Security F1) ===
# Each entry is (key, expected_python_type). To add a new flag here requires a code
# change + review; yaml cannot smuggle it in. This is the explicit gate.
SAFE_LLAMA_FLAGS: dict[str, type] = {
    # Performance + memory layout
    "ctx_size": int,
    "n_gpu_layers": int,
    "cache_type_k": str,
    "cache_type_v": str,
    "flash_attn": int,
    "threads": int,
    "parallel": int,
    "mlock": bool,
    "no_context_shift": bool,
    "cache_reuse": int,
    "slot_prompt_similarity": float,
    "no_perf": bool,
    "sleep_idle_seconds": int,
    "batch_size": int,
    "ubatch_size": int,
    "n_predict": int,
    # Sampling
    "temp": float,
    "top_k": int,
    "top_p": float,
    "min_p": float,
    "repeat_penalty": float,
    "repeat_last_n": int,
    "seed": int,
    # Chat template (value names, NOT paths)
    "chat_template": str,
    "jinja": bool,
    "reasoning_format": str,
    # MoE
    "n_cpu_moe": bool,
    # Misc safe flags
    "verbose": bool,
    "log_disable": bool,
}


# === Explicit denylist of path-bearing flags (Security F1 CRITICAL) ===
# Any of these in llama_server_flags would allow file read/write injection via
# llama-server. We REJECT them at schema validation, not warn.
DENIED_FLAGS: set[str] = {
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
}


# === Tag validation regex (v0.2 §8.1, Security F2 CRITICAL) ===
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


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


class Manifest(BaseModel):
    """A single model manifest from /var/lib/turbohaul/manifests/<tag>.yaml."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_tag: str
    display_name: str = ""
    description: str = ""
    gguf_blob_sha256: str
    gguf_size_bytes: int = Field(default=0, ge=0)
    context_size: int = Field(default=2048, ge=1)
    expected_vram_bytes: int = Field(default=0, ge=0)  # mandatory for VRAM-fit pre-check (v0.2 §10 + §15)
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

    @field_validator("llama_server_flags")
    @classmethod
    def _flags_allowlist(cls, v: dict[str, Any]) -> dict[str, Any]:
        for key, value in v.items():
            if key in DENIED_FLAGS:
                raise ManifestValidationError(
                    f"llama_server_flags.{key} is explicitly denied "
                    f"(Security #58 F1 path-traversal class). See v0.2 §8.1."
                )
            if key not in SAFE_LLAMA_FLAGS:
                raise ManifestValidationError(
                    f"llama_server_flags.{key} is not in the closed allowlist. "
                    "See v0.2 §8.1 - unknown flags REJECTED (not silently warned)."
                )
            expected = SAFE_LLAMA_FLAGS[key]
            # bool is a subclass of int; reject int→bool coercion explicitly
            if expected is bool:
                if not isinstance(value, bool):
                    raise ManifestValidationError(
                        f"llama_server_flags.{key} expects bool, got {type(value).__name__}"
                    )
            elif expected is float and isinstance(value, int) and not isinstance(value, bool):
                # int → float promotion is fine
                continue
            elif not isinstance(value, expected):
                raise ManifestValidationError(
                    f"llama_server_flags.{key} expects {expected.__name__}, "
                    f"got {type(value).__name__}"
                )
        return v


def _safe_manifest_path(manifests_root: Path, tag: str) -> Path:
    """Resolve manifest path with realpath check (Security #58 F2)."""
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
            f"manifest path is a symlink - refusing (v0.2 §8.1 safety)"
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
    """Atomic write with ETag/If-Match concurrency check (v0.2 §8.2 + HAUL M-1).

    - First write (no existing manifest): writes as-is, revision preserved;
      if_match must be None on create (else 412).
    - Subsequent writes: if_match REQUIRED. Mismatch -> ConcurrencyError.
      Missing -> ConcurrencyError too (HAUL M-1 fix: previously this
      silently overwrote, opening a lost-update class).
    - POSIX-atomic: tempfile-in-same-dir + fsync(file) + rename + fsync(dir).
    """
    target = _safe_manifest_path(manifests_root, manifest.model_tag)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        existing = read_manifest(manifests_root, manifest.model_tag)
        if if_match is None:
            # HAUL M-1 fix: refuse update without If-Match. Previously a
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
        os.replace(tmp_path, target)
        # fsync parent dir (POSIX durability)
        dir_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        os.chmod(target, 0o600)
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
    if target.exists():
        target.unlink()
        return True
    return False


# === llama-server CLI flag mapping (v0.2 §8 + §10) ===
def flags_to_argv(flags: dict[str, Any]) -> list[str]:
    """Map snake_case flags dict to llama-server CLI argv.

    Validates against SAFE_LLAMA_FLAGS allowlist (defense-in-depth; manifest
    validator already enforces this on parse).

    Boolean True → `--<flag>` (no value).
    Boolean False → flag OMITTED (not `--<flag> false`).
    Other types → `--<flag> <value>`.
    """
    argv: list[str] = []
    for key, value in flags.items():
        if key not in SAFE_LLAMA_FLAGS or key in DENIED_FLAGS:
            raise ManifestValidationError(
                f"flag {key} blocked at argv-build (allowlist enforcement)"
            )
        cli_key = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(cli_key)
            # else omit
        else:
            argv.extend([cli_key, str(value)])
    return argv
