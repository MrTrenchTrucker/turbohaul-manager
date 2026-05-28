"""Import + delete endpoints per v0.2 §9.2 + §12.1.

POST /api/import — local file → blob store. Path MUST be under import_allowed_root.
DELETE /api/delete — remove blob by sha256 (Ollama-compat shape).

security finding F4 must-fix: import_allowed_root sandbox + O_NOFOLLOW + GGUF magic
check + denylist of system paths. Path-traversal + symlink escape REJECTED.
"""
import logging
import os
import secrets
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from turbohaul.blob_store import (
    BlobError,
    BlobHashMismatch,
    BlobSizeExceeded,
    delete_blob,
    write_stream_atomic,
)


log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["import-delete"])


# Always-denied path prefixes (defense-in-depth even if import_allowed_root is
# misconfigured)
DENIED_PATH_PREFIXES: tuple[str, ...] = (
    "/proc/",
    "/sys/",
    "/dev/",
    "/etc/",
    "/root/",
    "/var/run/",
    "/var/lib/dpkg/",
    "/boot/",
)


GGUF_MAGIC = b"GGUF"


class ImportSafetyError(ValueError):
    pass


def _validate_import_path(import_allowed_root: Path, candidate: str) -> Path:
    """Resolve candidate path safely under import_allowed_root.

    Raises ImportSafetyError on:
      - non-absolute paths
      - denylist hit (/proc /sys /dev /etc /root)
      - escape from import_allowed_root via .. / symlinks (realpath check)
      - file is a symlink itself
    """
    if not candidate or not isinstance(candidate, str):
        raise ImportSafetyError("`path` must be non-empty string")
    if not candidate.startswith("/"):
        raise ImportSafetyError("path must be absolute")
    for denied in DENIED_PATH_PREFIXES:
        if candidate.startswith(denied):
            raise ImportSafetyError(
                f"path {candidate!r} starts with denied prefix {denied}"
            )

    target = Path(candidate)
    if target.is_symlink():
        raise ImportSafetyError(
            f"path {candidate} is a symlink (rejected per v0.2 §9.2)"
        )

    resolved = target.resolve(strict=False)
    root_resolved = import_allowed_root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as e:
        raise ImportSafetyError(
            f"path {candidate} escapes import_allowed_root {root_resolved}"
        ) from e
    if not resolved.exists():
        raise ImportSafetyError(f"path {candidate} does not exist")
    if not resolved.is_file():
        raise ImportSafetyError(f"path {candidate} is not a regular file")
    return resolved


def _stream_local_file(path: Path, chunk_size: int = 64 * 1024):
    """Read local file via O_NOFOLLOW + yield chunks. GGUF magic check on first chunk."""
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        # Magic check
        head = os.read(fd, 4)
        if len(head) < 4 or head != GGUF_MAGIC:
            raise ImportSafetyError(
                f"file does not start with GGUF magic (saw {head!r})"
            )
        yield head
        while True:
            chunk = os.read(fd, chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        os.close(fd)


@router.post("/import")
async def import_local(payload: dict, request: Request) -> dict:
    """Import a local GGUF file into the blob store.

    Path must be under storage.import_allowed_root + pass safety checks.
    First 4 bytes verified as `GGUF` (magic). O_NOFOLLOW used to defeat
    symlink-after-validation races.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be JSON object")
    raw_path = payload.get("path")
    expected_sha256 = payload.get("expected_sha256")

    mgr = request.app.state.manager
    try:
        safe_path = _validate_import_path(
            mgr.boot.storage.import_allowed_root, raw_path or ""
        )
    except ImportSafetyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    pull_id = "import-" + secrets.token_hex(8)
    mgr.event_bus.publish_nowait(
        {"event": "import_started", "pull_id": pull_id, "path_basename": safe_path.name}
    )

    try:
        sha, bytes_written = write_stream_atomic(
            mgr.boot.storage.blob_store_path,
            _stream_local_file(safe_path),
            expected_sha256=expected_sha256,
            per_stream_max_bytes=mgr.runtime.pull.per_stream_max_bytes,
        )
    except ImportSafetyError as e:
        mgr.event_bus.publish_nowait(
            {"event": "import_failed", "pull_id": pull_id, "reason": "magic-check"}
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except BlobSizeExceeded as e:
        mgr.event_bus.publish_nowait(
            {"event": "import_failed", "pull_id": pull_id, "reason": "size-exceeded"}
        )
        raise HTTPException(status_code=413, detail=str(e)) from e
    except BlobHashMismatch as e:
        mgr.event_bus.publish_nowait(
            {"event": "import_failed", "pull_id": pull_id, "reason": "hash-mismatch"}
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except BlobError as e:
        mgr.event_bus.publish_nowait(
            {"event": "import_failed", "pull_id": pull_id, "reason": "blob-error"}
        )
        raise HTTPException(status_code=500, detail=str(e)) from e

    mgr.event_bus.publish_nowait(
        {
            "event": "import_complete",
            "pull_id": pull_id,
            "sha256": sha,
            "bytes_written": bytes_written,
        }
    )
    return {
        "pull_id": pull_id,
        "status": "complete",
        "sha256": sha,
        "bytes_written": bytes_written,
        "source": "local-import",
    }


@router.delete("/delete")
async def delete_blob_route(payload: dict, request: Request) -> dict:
    """Ollama-compat blob delete by sha256."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be JSON object")
    sha = payload.get("sha256") or payload.get("digest", "").removeprefix("sha256:")
    if not sha:
        raise HTTPException(
            status_code=400, detail="`sha256` (or `digest: sha256:...`) required"
        )
    mgr = request.app.state.manager
    try:
        removed = delete_blob(mgr.boot.storage.blob_store_path, sha)
    except BlobError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not removed:
        raise HTTPException(status_code=404, detail=f"blob not found: sha256:{sha}")
    return {"status": "deleted", "sha256": sha}
