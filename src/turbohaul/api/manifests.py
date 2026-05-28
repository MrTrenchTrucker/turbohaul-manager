"""Manifests CRUD routes per v0.2 ARCHITECTURE.md §8.1 + §8.2.

GET/PUT /api/manifests/{tag} with ETag/If-Match concurrency control + closed flag
allowlist enforcement (security finding F1 + F2 must-fixes).
"""
from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import ValidationError

from turbohaul.manifest import (
    ConcurrencyError,
    Manifest,
    ManifestValidationError,
    delete_manifest,
    manifest_etag,
    read_manifest,
    validate_tag,
    write_manifest_atomic,
)


router = APIRouter(prefix="/api/manifests", tags=["manifests"])


@router.get("/{tag}")
async def get_manifest(tag: str, request: Request, response: Response) -> dict:
    """Read a manifest by tag. Returns ETag header for subsequent PUT."""
    try:
        validate_tag(tag)
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    mgr = request.app.state.manager
    try:
        m = read_manifest(mgr.boot.storage.manifests_path, tag)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"manifest not found: {tag}") from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    response.headers["ETag"] = f'"{m.revision}"'
    return m.model_dump(mode="json")


@router.put("/{tag}")
async def put_manifest(
    tag: str,
    payload: dict,
    request: Request,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> dict:
    """Write a manifest. ETag/If-Match required for updates (v0.2 §8.2).

    First write (no existing manifest) succeeds without If-Match.
    Subsequent updates require If-Match: "<current-revision>"; mismatch → 412.
    """
    try:
        validate_tag(tag)
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Force model_tag in payload to match URL to prevent tag-mismatch confusion
    payload = dict(payload)  # copy
    payload["model_tag"] = tag

    try:
        manifest = Manifest(**payload)
    except (ValidationError, ManifestValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    mgr = request.app.state.manager
    try:
        written = write_manifest_atomic(
            mgr.boot.storage.manifests_path, manifest, if_match=if_match
        )
    except ConcurrencyError as e:
        raise HTTPException(status_code=412, detail=str(e)) from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    response.headers["ETag"] = f'"{written.revision}"'
    return {
        "status": "ok",
        "model_tag": written.model_tag,
        "revision": written.revision,
        "restart_required": False,  # per-model yaml hot-reloads on next stage
    }


@router.delete("/{tag}")
async def delete_manifest_route(tag: str, request: Request) -> dict:
    """Remove a manifest."""
    try:
        validate_tag(tag)
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    mgr = request.app.state.manager
    removed = delete_manifest(mgr.boot.storage.manifests_path, tag)
    if not removed:
        raise HTTPException(status_code=404, detail=f"manifest not found: {tag}")
    return {"status": "deleted", "model_tag": tag}
