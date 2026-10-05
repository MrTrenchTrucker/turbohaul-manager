"""Ollama-compatible API routes.

Provides the read-only endpoints (/api/tags, /api/show; /api/version is still
served from main.py). The streaming completion routes (/api/generate, /api/chat)
are handled separately via httpx-proxy completion forwarding.

Trademark hygiene: 'Ollama-compatible' (nominative fair use) only.
"""
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from turbohaul import blob_store
from turbohaul._gguf_meta import ModelCard, read_model_card
from turbohaul.manifest import (
    Manifest,
    ManifestValidationError,
    ModelManifest,
    list_manifests,
    read_manifest,
)


log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ollama-compat"])

# read_manifest can fail SIX ways: FileNotFoundError,
# ManifestValidationError (root-not-mapping, or traversal/invalid tag via
# validate_tag), yaml.YAMLError (broken YAML syntax), and pydantic
# ValidationError (a well-formed mapping that no longer satisfies the
# Manifest schema -- e.g. a manifest written by an older turbohaul that
# predates a newly-required field; after any schema change every stale
# manifest raises this), plus OSError and UnicodeDecodeError. Catching only
# the first two would let one such file 500 the entire /api/tags listing. Local to this
# file on purpose -- NOT imported from models.py's identical-in-spirit
# _UNREADABLE_MANIFEST, so that each module owns its own tuple (the real
# consolidation candidate runs through chat_completion.py, the
# hottest path, and is a separate refactor).
# OSError and UnicodeDecodeError complete the enumeration of
# read_manifest's six. The file-local
# choice above stands; only the
# COVERAGE is aligned, not the ownership. A drift test pins all four tuples
# to the same set so the next divergence cannot ship silently.
_UNREADABLE_MANIFEST = (
    FileNotFoundError,
    ManifestValidationError,
    yaml.YAMLError,
    ValidationError,
    OSError,
    UnicodeDecodeError,
)

# Best-effort GGUF model-card cache, keyed by gguf_blob_sha256.
# Blob CONTENT is content-addressed (immutable), so a SUCCESSFUL read never goes
# stale. Blob PRESENCE, however, is mutable (delete + re-pull re-stages the same
# sha), so we deliberately do NOT cache a None/miss — otherwise a model whose blob
# was transiently absent/corrupt at first read would show null metadata forever
# until process restart even after the blob is repaired.
# Only positive reads are memoized.
_MODEL_CARD_CACHE: dict[str, "ModelCard"] = {}


def _get_model_card(blobs_root: Path, sha256: str) -> "ModelCard | None":
    """Cached, best-effort GGUF model-card read. Never raises. A miss (None) is
    NOT cached, so a later-staged/repaired blob at the same sha is re-read."""
    if sha256 in _MODEL_CARD_CACHE:
        return _MODEL_CARD_CACHE[sha256]
    try:
        card = read_model_card(blob_store.blob_path(blobs_root, sha256))
    except Exception:
        card = None
    if card is not None:
        _MODEL_CARD_CACHE[sha256] = card
    return card


def _manifest_modified_at_iso(manifests_root: Path, tag: str) -> str | None:
    """Best-effort ISO-8601 mtime of a model's manifest .yaml file. Never raises.

    ``tag`` has already been TAG_RE-validated by ``list_manifests``/``read_manifest``
    by the time this is called, so a direct read-only stat here is safe.
    """
    try:
        mtime = (Path(manifests_root) / f"{tag}.yaml").stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _model_card_details(m: Manifest, blobs_root: Path) -> dict:
    """Additive, nullable sort/display fields for one model. Never raises —
    any read failure (bad/missing blob, corrupt GGUF) degrades to an
    all-null dict for this model only."""
    card = None
    try:
        card = _get_model_card(blobs_root, m.gguf_blob_sha256)
    except Exception:
        card = None
    architecture = m.arch or (card.architecture if card else None)
    # Vision support is keyed on the manifest's declared
    # projector alone, not on a bare GGUF vision-marker key
    # (card.has_vision_kv): --mmproj only ever reaches
    # llama-server from manifest.mmproj_blob_sha256 (manager.py gates
    # on that field) and the raw `mmproj` flag is
    # denied outright (see manifest.py), so a model with a vision-looking
    # GGUF but no projector blob would be advertised as vision-capable while
    # structurally unable to ever be launched with one. OR-ing in the marker
    # key would therefore be wrong, even where the advertised-vision
    # set happens to match the mmproj set for the models in use.
    is_vision = bool(m.mmproj_blob_sha256)
    return {
        "parameter_count": card.parameter_count if card else None,
        "size_label": card.size_label if card else None,
        "architecture": architecture or None,
        "is_moe": card.is_moe if card else None,
        "expert_count": card.expert_count if card else None,
        "modality": "vision" if is_vision else "text",
    }


def _resolve_context(m: ModelManifest) -> tuple[int, str]:
    """Resolve the context a model is CONFIGURED for, and where that number came
    from: ``llama_server_flags.ctx_size`` when that flag is set, otherwise the
    manifest's own ``context_size``. Returned TOGETHER so a caller physically
    cannot take the value without its provenance.

    ``/api/tags`` and ``/api/show`` must not read raw
    ``m.context_size`` directly here -- wrong whenever ``ctx_size`` (what
    actually reaches llama-server) diverges from the manifest's declared
    ``context_size``. File-local on purpose, independently written rather
    than imported from ``api/models.py``'s identical-in-spirit resolution for
    ``/v1/models`` -- see the comment above _UNREADABLE_MANIFEST for
    the file-local choice. Held to one behaviour by a shared
    ASSERTION instead of shared code: the context/vision cross-surface
    drift test, alongside the manifest-decode blast-radius test's
    existing drift test for ``_UNREADABLE_MANIFEST`` (the same mechanism
    applied to that tuple).
    """
    ctx_flag = m.llama_server_flags.get("ctx_size")
    return int(ctx_flag or m.context_size), ("flag" if ctx_flag else "manifest")


@router.get("/tags")
async def get_tags(request: Request) -> dict:
    """Ollama-compat: list installed models from blob store.

    Response shape mirrors Ollama: {"models": [{"name": ..., "size": ..., ...}]}.
    """
    mgr = request.app.state.manager
    manifests_root = mgr.boot.storage.manifests_path
    blobs_root = mgr.boot.storage.blob_store_path
    tags = list_manifests(manifests_root)
    models = []
    for tag in tags:
        try:
            m = read_manifest(manifests_root, tag)
        except _UNREADABLE_MANIFEST as exc:
            # A skip must never be silent -- an operator debugging
            # "why did my model vanish from /api/tags" needs a trace. Type
            # name only, never str(exc): yaml.YAMLError leaks parser
            # position/content, pydantic ValidationError echoes manifest
            # field VALUES.
            log.warning("skipping unreadable manifest %r: %s", tag, type(exc).__name__)
            continue
        if not isinstance(m, ModelManifest):
            # list_manifests() returns every manifest tag regardless of
            # kind. This endpoint is a MODEL listing -- a plugin manifest has
            # none of the model-only fields read below (matches
            # the isinstance(m, PluginManifest) filter api/plugins.py already
            # ships, inverse direction).
            continue
        if m.hidden:
            # Listings-only, same contract as /v1/models: hidden models are
            # omitted from discovery but still serve normally by exact name.
            continue
        try:
            modified_at = _manifest_modified_at_iso(manifests_root, tag)
        except Exception:
            modified_at = None
        try:
            # Offload the sync GGUF-header parse OFF the event
            # loop. On a cold cache it walks the model's 100K-300K-entry tokenizer
            # arrays per uncached model; run inline it would block the loop for
            # multi-seconds during model discovery and stall in-flight streaming
            # completions. asyncio.to_thread keeps the loop free (self-healing once
            # the positive read is cached).
            card_details = await asyncio.to_thread(_model_card_details, m, blobs_root)
        except Exception:
            card_details = {
                "parameter_count": None,
                "size_label": None,
                "architecture": None,
                "is_moe": None,
                "expert_count": None,
                "modality": "text",
            }
        # Resolved value+source, same position as its sibling
        # context_length already has here -- nested under details.
        ctx_value, ctx_source = _resolve_context(m)
        models.append(
            {
                "name": m.model_tag,
                "model": m.model_tag,
                "size": m.gguf_size_bytes,
                "digest": "sha256:" + m.gguf_blob_sha256,
                "modified_at": modified_at,
                "details": {
                    "format": "gguf",
                    "context_length": ctx_value,
                    "context_length_source": ctx_source,
                    "expected_vram_bytes": m.expected_vram_bytes,
                    "display_name": m.display_name,
                    "description": m.description,
                    **card_details,
                },
                "revision": m.revision,
            }
        )
    return {"models": models}


@router.get("/show")
async def get_show(name: str, request: Request) -> dict:
    """Ollama-compat: show details for a single model by name.

    Note: response strings (display_name, description, chat_template) are returned
    as plain text - the FE renders them via text-only React text node as part of
    its XSS-defense policy.
    """
    mgr = request.app.state.manager
    manifests_root = mgr.boot.storage.manifests_path
    blobs_root = mgr.boot.storage.blob_store_path
    try:
        m = read_manifest(manifests_root, name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"model not found: {name}") from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (yaml.YAMLError, ValidationError) as e:
        # The tag is valid and the model isn't absent -- the file
        # is present but broken (corrupt YAML syntax, or a stale manifest
        # that predates a schema change). Not a 4xx (client did nothing
        # wrong) and not a 404 (conflating "you typo'd it" with "your
        # manifest is corrupt" destroys an operator's ability to debug a
        # vanished model). A handled 5xx: the server failed to fulfil a
        # valid request because of bad server-side data. Fixed message only
        # -- never str(e): yaml.YAMLError leaks parser position/content,
        # pydantic ValidationError echoes manifest field VALUES.
        # Full detail is logged server-side instead.
        log.exception("manifest for %r is present but unreadable", name)
        raise HTTPException(
            status_code=500, detail=f"manifest for '{name}' is present but unreadable"
        ) from e
    if not isinstance(m, ModelManifest):
        # Same shape as get_tags above -- a plugin manifest
        # has none of the model-only fields this endpoint reads below. From
        # this endpoint's own contract, a plugin tag simply isn't a model:
        # same 404 an unknown tag gets, not a new error shape.
        raise HTTPException(status_code=404, detail=f"model not found: {name}")
    try:
        modified_at = _manifest_modified_at_iso(manifests_root, name)
    except Exception:
        modified_at = None
    try:
        # Same off-loop offload as get_tags (see there).
        card_details = await asyncio.to_thread(_model_card_details, m, blobs_root)
    except Exception:
        card_details = {
            "parameter_count": None,
            "size_label": None,
            "architecture": None,
            "is_moe": None,
            "expert_count": None,
            "modality": "text",
        }
    # Resolved value+source, same position as its sibling
    # context_length already has here -- top-level.
    ctx_value, ctx_source = _resolve_context(m)
    return {
        "name": m.model_tag,
        "model": m.model_tag,
        "size": m.gguf_size_bytes,
        "digest": "sha256:" + m.gguf_blob_sha256,
        "context_length": ctx_value,
        "context_length_source": ctx_source,
        "expected_vram_bytes": m.expected_vram_bytes,
        "display_name": m.display_name,
        "description": m.description,
        "revision": m.revision,
        "llama_server_flags": m.llama_server_flags,
        "prompt_template": m.prompt_template.model_dump(mode="json"),
        "modified_at": modified_at,
        "details": card_details,
    }
