"""Ollama-compatible API routes per v0.2 ARCHITECTURE.md §9.

This module hosts the read-only endpoints (/api/tags, /api/show; /api/version
is also served from main.py). The streaming completion routes
(/api/generate, /api/chat) use the httpx-proxy completion forwarding.

Trademark hygiene per v0.2 §14: 'Ollama-compatible' (nominative fair use) only.
"""
from fastapi import APIRouter, HTTPException, Request

from turbohaul.manifest import (
    ManifestValidationError,
    list_manifests,
    read_manifest,
)


router = APIRouter(prefix="/api", tags=["ollama-compat"])


@router.get("/tags")
async def get_tags(request: Request) -> dict:
    """Ollama-compat: list installed models from blob store.

    Response shape mirrors Ollama: {"models": [{"name": ..., "size": ..., ...}]}.
    """
    mgr = request.app.state.manager
    manifests_root = mgr.boot.storage.manifests_path
    tags = list_manifests(manifests_root)
    models = []
    for tag in tags:
        try:
            m = read_manifest(manifests_root, tag)
        except (FileNotFoundError, ManifestValidationError):
            continue
        models.append(
            {
                "name": m.model_tag,
                "model": m.model_tag,
                "size": m.gguf_size_bytes,
                "digest": "sha256:" + m.gguf_blob_sha256,
                "details": {
                    "format": "gguf",
                    "context_length": m.context_size,
                    "expected_vram_bytes": m.expected_vram_bytes,
                    "display_name": m.display_name,
                    "description": m.description,
                },
                "revision": m.revision,
            }
        )
    return {"models": models}


@router.get("/show")
async def get_show(name: str, request: Request) -> dict:
    """Ollama-compat: show details for a single model by name.

    Note: response strings (display_name, description, chat_template) are returned
    as plain text - the FE renders them via text-only React text node per v0.2 §11.2
    XSS-defense policy.
    """
    mgr = request.app.state.manager
    manifests_root = mgr.boot.storage.manifests_path
    try:
        m = read_manifest(manifests_root, name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"model not found: {name}") from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "name": m.model_tag,
        "model": m.model_tag,
        "size": m.gguf_size_bytes,
        "digest": "sha256:" + m.gguf_blob_sha256,
        "context_length": m.context_size,
        "expected_vram_bytes": m.expected_vram_bytes,
        "display_name": m.display_name,
        "description": m.description,
        "revision": m.revision,
        "llama_server_flags": m.llama_server_flags,
        "prompt_template": m.prompt_template.model_dump(mode="json"),
    }
