"""Import + delete endpoints.

POST /api/import — local file → blob store. Path MUST be under import_allowed_root.
DELETE /api/delete — remove blob by sha256 (Ollama-compat shape).
GET /api/blobs — enumerate the blob store on disk.
PUT /api/blobs/{digest}/description — set/clear a model-level description
  Blob-level metadata, kept separate from manifests.

Security hardening: import_allowed_root sandbox + O_NOFOLLOW + GGUF magic
check + denylist of system paths. Path-traversal + symlink escape REJECTED.
"""
import asyncio
import logging
import os
import re
import secrets
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request

from turbohaul.blob_store import (
    BlobError,
    BlobHashMismatch,
    BlobSizeExceeded,
    blob_exists,
    blob_path,
    delete_blob,
    list_blobs,
    write_stream_atomic,
)
from turbohaul.model_meta import (
    MAX_DESCRIPTION_LEN,
    description_for,
    is_valid_digest,
    prune_digest,
    read_model_meta,
    set_description,
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
            f"path {candidate} is a symlink (rejected)"
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
        # write_stream_atomic drives the sync _stream_local_file generator
        # (blocking os.read) + does the atomic write + os.fsync — run it in a thread
        # so it doesn't block the event loop. The helper
        # stays a sync generator; only this call site is wrapped.
        sha, bytes_written = await asyncio.to_thread(
            write_stream_atomic,
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


# The three manifest fields that can name a blob. A blob is IN USE if any
# manifest names it in ANY of them: gguf_blob_sha256 is the model itself,
# mmproj_blob_sha256 its vision projector, spec_draft_gguf_blob_sha256 the
# small model it speculates against. Checking only the first would let a
# blob referenced solely as a projector or a drafter count zero manifests,
# render as an empty model offering Delete, and be deleted by this route.
# Re-acquiring one is a multi-GB download, not a rollback.
BLOB_REFERENCE_FIELDS: tuple[str, ...] = (
    "gguf_blob_sha256",
    "mmproj_blob_sha256",
    "spec_draft_gguf_blob_sha256",
)


def _manifests_referencing(manifests_root: Path, sha: str) -> list[str]:
    """Tags of every manifest naming `sha`, each labelled with how it matched.

    Reads the STORED YAML rather than a validated Manifest, deliberately. A
    manifest pydantic rejects -- a malformed one -- is exactly the row a
    read_manifest()-based scan skips, and skipping a row in the guard whose
    whole job is preventing data loss is a hole in the guard. The manifest is
    a text file an operator repairs in a minute; the blob is a multi-GB
    download. So a broken manifest's reference still counts.

    Two truthiness traps, both live, both guarded below. On the SHA side: a
    short or empty digest is a substring of everything and a prefix of
    everything, so the unparseable-file fallback would match half the
    registry -- hence the is_valid_digest() gate, at this boundary rather
    than relying on a caller. On the FIELD side: mmproj_blob_sha256 and
    spec_draft_gguf_blob_sha256 are the EMPTY STRING on most manifests, not
    null (the manifest schema explicitly allows ""), so every value is
    checked for truthiness and type before it is compared.
    """
    if not is_valid_digest(sha):
        # Manifest digest fields are validated 64-hex, so nothing can
        # legitimately reference a malformed digest -- and the text fallback
        # below would match indiscriminately on a short one.
        return []
    hits: list[str] = []
    needle = re.compile(re.escape(sha), re.IGNORECASE)
    # Same enumeration as the manifests listing and
    # deliberately the wider one -- not manifest.list_manifests()'s
    # TAG_RE-filtered view. A safety scan wants the superset of anything that
    # could be a manifest.
    for path in sorted(Path(manifests_root).glob("*.yaml")):
        try:
            text = path.read_text()
        except FileNotFoundError:
            # Raced with a manifest delete. It is gone; it references nothing.
            continue
        except (OSError, UnicodeDecodeError):
            # Unreadable for some other reason. Cannot rule it out, so refuse
            # rather than guess -- named, so the operator can act on it.
            #
            # UnicodeDecodeError is a ValueError,
            # NOT an OSError, so a manifest whose bytes are not valid UTF-8 would
            # escape BOTH handlers, propagate out of delete_blob_route (which
            # does not guard this call) and 500 the route -- and because this
            # scan walks the whole directory, ONE such file would brick delete for
            # EVERY blob. It belongs to this arm and not to the text fallback
            # below: a file we cannot DECODE is a file we cannot rule out, which
            # is exactly what this arm is for.
            # ⛔ Do NOT "fix" this by decoding with errors="replace" to reach the
            # fallback. That fallback is sound only because every byte decoded
            # intact -- U+FFFD substitutions can land ON the 64-hex digest, the
            # match then misses, and a still-referenced blob is unlinked
            # SILENTLY. That is the "silently open a hole" outcome the comment
            # below says this design avoids, and it is the harm this guard exists to prevent.
            hits.append(f"{path.stem} (unreadable file)")
            continue
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            data = None
        if isinstance(data, dict):
            # Structure is available, so structure is authoritative: a text
            # scan here would match a digest quoted in `description` and
            # refuse a delete nobody asked to block.
            for field in BLOB_REFERENCE_FIELDS:
                value = data.get(field)
                # `and value` is REDUNDANT while the is_valid_digest() gate
                # above stands (it never changes the outcome today). Kept
                # deliberately: the FE displays
                # 12-char digest prefixes, so "accept a prefix" is a plausible
                # future relaxation of that gate, and this is then the only
                # thing standing between '' and every text-only manifest.
                if isinstance(value, str) and value and value.lower() == sha.lower():
                    hits.append(f"{path.stem} ({field})")
                    break
            continue
        # Not YAML at all, or a root that is not a mapping: no structure to
        # read. Fall back to a case-insensitive text match on the raw file --
        # 64 hex characters do not occur by accident. This refuses precisely
        # instead of letting one corrupt file either brick every delete or
        # silently open a hole.
        if needle.search(text):
            hits.append(f"{path.stem} (unparseable file)")
    return hits


@router.delete("/delete")
async def delete_blob_route(payload: dict, request: Request) -> dict:
    """Ollama-compat blob delete by sha256.

    Refuses with 409 when any manifest still names the digest. Existence
    is checked FIRST so an absent blob keeps its 404 -- deleting a file that is
    not there is a no-op with nothing to protect, and telling a caller to go
    remove references to a file that does not exist is a worse instruction
    than "not found".
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be JSON object")
    sha = payload.get("sha256") or payload.get("digest", "").removeprefix("sha256:")
    if not sha:
        raise HTTPException(
            status_code=400, detail="`sha256` (or `digest: sha256:...`) required"
        )
    mgr = request.app.state.manager
    # blob_exists() raises BlobError on a malformed digest by the same route
    # delete_blob() does (both go through blob_path/_final_path), so malformed
    # input gets the same 400 from either call.
    try:
        present = blob_exists(mgr.boot.storage.blob_store_path, sha)
    except BlobError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not present:
        raise HTTPException(status_code=404, detail=f"blob not found: sha256:{sha}")
    referencing = _manifests_referencing(mgr.boot.storage.manifests_path, sha)
    if referencing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"blob sha256:{sha} is still referenced by {len(referencing)} "
                f"manifest(s): {', '.join(referencing)}. Delete or re-point "
                "them first (DELETE /api/manifests/<tag>), then retry."
            ),
        )
    try:
        removed = delete_blob(mgr.boot.storage.blob_store_path, sha)
    except BlobError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not removed:
        raise HTTPException(status_code=404, detail=f"blob not found: sha256:{sha}")
    # Prune the model_meta.json entry too, else the
    # store quietly accumulates descriptions of files that no longer exist
    # -- the same accumulation problem seen from another direction.
    #
    # BEST-EFFORT, and deliberately unable to change the outcome.
    # delete_blob above has ALREADY unlinked the file -- the destructive act
    # succeeded -- so an exception raised from here reports failure for
    # something that definitively happened: a caller retrying on the 500 gets
    # a 404, a script treating 500 as "did not happen" is wrong about a
    # DESTRUCTIVE operation, and blob_changed below never fires so no other
    # tab is told. That is the failure shape this guards against.
    #
    # ⛔ Catching broadly is the point, not laziness. Handling UnicodeDecodeError in
    # read_model_meta alone would close the decode INSTANCE; catching broadly closes the CLASS.
    # prune_digest -> set_description -> _write_model_meta_atomic also WRITES,
    # and a write fails on its own terms (disk full, permissions, a read-only
    # mount) with the read perfectly healthy. Handling one instance and leaving
    # the class open is the mistake this guard exists to avoid.
    # The traceback is not lost -- it goes to the server log.
    try:
        prune_digest(mgr.boot.storage.manifests_path, sha)
    except Exception:
        log.exception(
            "prune_digest failed for sha256:%s after the blob was already "
            "unlinked; reporting the delete as the success it was",
            sha[:16],
        )
    # Identifier only. Deliberately on the SUCCESS tail --
    # the existence check and the manifest-reference guard sit at this
    # route's HEAD, and a 409-refused delete must never reach this line:
    # nothing changed, so nothing is announced.
    mgr.event_bus.publish_nowait({"event": "blob_changed", "sha256": sha})
    return {"status": "deleted", "sha256": sha}


@router.get("/blobs")
async def list_blobs_route(request: Request) -> dict:
    """Enumerate the blob store on disk. Pure, read-only, no manifest join.

    Deliberately does not consult manifests -- this is what makes a
    zero-manifest blob visible at all. The FE
    does its own join against GET /api/manifests. Fixed response contract,
    the FE builds against it: {"blobs": [{"digest", "size_bytes",
    "description"}], "total"}. `description` is the one piece of BLOB-level
    metadata that exists -- still not a manifest
    join, the purity rule holds.
    """
    mgr = request.app.state.manager
    root = mgr.boot.storage.blob_store_path
    # Read once, not once per blob -- description_for() just indexes into it.
    meta = read_model_meta(mgr.boot.storage.manifests_path)
    blobs = []
    for digest in list_blobs(root):
        try:
            size_bytes = blob_path(root, digest).stat().st_size
        except OSError:
            # Raced with a delete, or otherwise gone/unreadable between the
            # directory walk and the stat -- must not 500 the whole listing.
            # Same defensive shape as the per-entry try/except in the manifests
            # listing (one bad entry must not blank the
            # whole management view).
            continue
        blobs.append({
            "digest": digest,
            "size_bytes": size_bytes,
            "description": description_for(meta, digest),
        })
    return {"blobs": blobs, "total": len(blobs)}


@router.put("/blobs/{digest}/description")
async def set_blob_description_route(digest: str, payload: dict, request: Request) -> dict:
    """Set or clear a blob's human-written, model-level description.

    Body: {"description": "<str>"}. Empty string or null CLEARS the entry
    (does not store an empty one) -- keeps model_meta.json from
    accumulating dead rows. Still blob-level metadata, not a manifest field.
    """
    if not is_valid_digest(digest):
        raise HTTPException(status_code=400, detail=f"invalid sha256 hex: {digest[:32]}...")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be JSON object")
    if "description" not in payload:
        raise HTTPException(
            status_code=400, detail="`description` field required (string or null)"
        )
    description = payload["description"]
    if description is not None and not isinstance(description, str):
        raise HTTPException(status_code=400, detail="`description` must be a string or null")
    if description and len(description) > MAX_DESCRIPTION_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"description exceeds {MAX_DESCRIPTION_LEN} chars "
            f"(got {len(description)})",
        )
    mgr = request.app.state.manager
    if not blob_exists(mgr.boot.storage.blob_store_path, digest):
        raise HTTPException(status_code=404, detail=f"blob not found: sha256:{digest}")
    set_description(mgr.boot.storage.manifests_path, digest, description)
    # The digest and NOTHING else. ⛔ `description` is
    # operator-written prose sitting right there in scope -- it is precisely
    # the class of content ws_state.py forbids on this channel ("NEVER
    # broadcasts: prompt text, response text, ..."). Subscribers learn THAT
    # it changed and re-GET the value through the authenticated route.
    mgr.event_bus.publish_nowait({"event": "blob_changed", "sha256": digest})
    return {"digest": digest, "description": description or None}
