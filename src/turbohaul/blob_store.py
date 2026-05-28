"""Content-addressed blob storage per v0.2 ARCHITECTURE.md §12 + §12.1.

Per security finding F5 must-fix (blob TOCTOU + disk-fill DoS):
  1. Stream to incoming/<random>.tmp with per-stream byte ceiling enforced
  2. On stream completion: compute sha256, verify (or set as canonical)
  3. Atomic rename to blobs/sha256/<ab>/<full-hash>
  4. chmod 0o400 (read-only after rename - tamper-evident)
  5. re-verify hash on stage (defense-in-depth against blob swap)

Layout:
    /var/lib/turbohaul/blobs/
    └── sha256/
        ├── incoming/
        │   └── <random>.tmp     (during pull/import)
        └── <ab>/
            └── <full-64-char>   (final, chmod 0o400)
"""
import contextlib
import hashlib
import logging
import os
import re
import secrets
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path


log = logging.getLogger(__name__)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BlobError(RuntimeError):
    """Base for blob lifecycle errors."""


class BlobSizeExceeded(BlobError):
    """Stream exceeded per_stream_max_bytes."""


class BlobHashMismatch(BlobError):
    """Computed hash didn't match expected (post-stream verification)."""


class BlobIntegrityFailure(BlobError):
    """Re-verify on stage detected hash drift (possible TOCTOU swap)."""


def _blob_root(blobs_root: Path) -> Path:
    return blobs_root / "sha256"


def _incoming_dir(blobs_root: Path) -> Path:
    return _blob_root(blobs_root) / "incoming"


def _final_path(blobs_root: Path, sha256: str) -> Path:
    """Return blobs/sha256/<ab>/<full-hash>. Caller must have validated sha256."""
    if not SHA256_RE.match(sha256):
        raise BlobError(f"invalid sha256 hex: {sha256[:32]}...")
    return _blob_root(blobs_root) / sha256[:2] / sha256


def ensure_blob_layout(blobs_root: Path) -> None:
    """Create blobs/sha256/{incoming/} directory structure."""
    _incoming_dir(blobs_root).mkdir(parents=True, exist_ok=True)


def make_incoming_path(blobs_root: Path) -> Path:
    """Generate a fresh incoming/<random>.tmp path for streaming."""
    ensure_blob_layout(blobs_root)
    name = f"pull_{secrets.token_hex(16)}.tmp"
    return _incoming_dir(blobs_root) / name


def write_stream_atomic(
    blobs_root: Path,
    chunks: Iterator[bytes],
    expected_sha256: str | None = None,
    per_stream_max_bytes: int = 100 * 1024**3,  # 100 GB default
) -> tuple[str, int]:
    """Stream chunks to a tempfile under incoming/, verify, atomic-rename to final.

    Returns (sha256_hex, bytes_written).

    Raises:
        BlobSizeExceeded — stream exceeded per_stream_max_bytes (tempfile deleted)
        BlobHashMismatch — expected_sha256 provided and didn't match computed
        BlobError — IO / rename / hash format errors
    """
    ensure_blob_layout(blobs_root)
    tmp = make_incoming_path(blobs_root)
    h = hashlib.sha256()
    bytes_written = 0

    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                for chunk in chunks:
                    if not chunk:
                        continue
                    if bytes_written + len(chunk) > per_stream_max_bytes:
                        raise BlobSizeExceeded(
                            f"stream exceeded per_stream_max_bytes "
                            f"({per_stream_max_bytes}); have {bytes_written + len(chunk)} bytes"
                        )
                    f.write(chunk)
                    h.update(chunk)
                    bytes_written += len(chunk)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

        computed = h.hexdigest()
        if expected_sha256 is not None:
            if not SHA256_RE.match(expected_sha256):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp)
                raise BlobError(f"expected_sha256 not valid hex: {expected_sha256[:32]}...")
            if computed != expected_sha256:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp)
                raise BlobHashMismatch(
                    f"computed {computed[:16]}... != expected {expected_sha256[:16]}..."
                )

        # Atomic rename → final path
        final = _final_path(blobs_root, computed)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(tmp), str(final))

        # fsync the parent dir (POSIX durability)
        dir_fd = os.open(str(final.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

        # Read-only post-rename (defense against in-place tamper)
        os.chmod(str(final), 0o400)

        return computed, bytes_written
    finally:
        # Belt-and-suspenders cleanup of orphan tempfile if rename never happened
        if tmp.exists():
            with contextlib.suppress(FileNotFoundError, PermissionError):
                os.unlink(tmp)


async def write_stream_atomic_async(
    blobs_root: Path,
    chunks: AsyncIterator[bytes],
    expected_sha256: str | None = None,
    per_stream_max_bytes: int = 100 * 1024**3,
) -> tuple[str, int]:
    """Async variant for httpx.stream / aiter_bytes sources.

    Same lifecycle as write_stream_atomic: incoming/<tmp> → fsync → hash verify →
    atomic-rename → chmod 0o400 → fsync(parent dir).
    """
    ensure_blob_layout(blobs_root)
    tmp = make_incoming_path(blobs_root)
    h = hashlib.sha256()
    bytes_written = 0

    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    if bytes_written + len(chunk) > per_stream_max_bytes:
                        raise BlobSizeExceeded(
                            f"stream exceeded per_stream_max_bytes "
                            f"({per_stream_max_bytes}); have {bytes_written + len(chunk)} bytes"
                        )
                    f.write(chunk)
                    h.update(chunk)
                    bytes_written += len(chunk)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

        computed = h.hexdigest()
        if expected_sha256 is not None:
            if not SHA256_RE.match(expected_sha256):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp)
                raise BlobError(
                    f"expected_sha256 not valid hex: {expected_sha256[:32]}..."
                )
            if computed != expected_sha256:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp)
                raise BlobHashMismatch(
                    f"computed {computed[:16]}... != expected {expected_sha256[:16]}..."
                )

        final = _final_path(blobs_root, computed)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(tmp), str(final))

        dir_fd = os.open(str(final.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

        os.chmod(str(final), 0o400)

        return computed, bytes_written
    finally:
        if tmp.exists():
            with contextlib.suppress(FileNotFoundError, PermissionError):
                os.unlink(tmp)


def blob_path(blobs_root: Path, sha256: str) -> Path:
    """Return the final on-disk path for a content-addressed blob."""
    return _final_path(blobs_root, sha256)


def blob_exists(blobs_root: Path, sha256: str) -> bool:
    return blob_path(blobs_root, sha256).exists()


def verify_blob_on_stage(blobs_root: Path, expected_sha256: str) -> bool:
    """Re-verify blob hash at stage time. Defends against TOCTOU swap (v0.2 §12.1).

    Returns True on success. Raises BlobIntegrityFailure if hash differs.
    """
    path = blob_path(blobs_root, expected_sha256)
    if not path.exists():
        raise BlobIntegrityFailure(f"blob missing at stage time: {expected_sha256}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_sha256:
        raise BlobIntegrityFailure(
            f"blob hash drift: on-disk {actual[:16]}... != expected {expected_sha256[:16]}..."
        )
    return True


def delete_blob(blobs_root: Path, sha256: str) -> bool:
    """Delete a blob. Returns True if existed + removed.

    Note: unlink does not require write permission on the file (POSIX
    only checks write on the parent dir); the prior chmod 0o400 -> 0o600
    created a needless tamper window.
    """
    path = blob_path(blobs_root, sha256)
    if not path.exists():
        return False
    os.unlink(str(path))
    return True


def list_blobs(blobs_root: Path) -> list[str]:
    """Return list of sha256 hashes of stored blobs."""
    root = _blob_root(blobs_root)
    if not root.exists():
        return []
    hashes: list[str] = []
    for shard in root.iterdir():
        if shard.name == "incoming" or not shard.is_dir():
            continue
        if len(shard.name) != 2:
            continue
        for blob in shard.iterdir():
            if SHA256_RE.match(blob.name):
                hashes.append(blob.name)
    return sorted(hashes)


def gc_stale_incoming(blobs_root: Path, max_age_s: float = 3600.0) -> int:
    """Remove stale incoming/<>.tmp files (e.g., after manager crash).

    Returns count of files removed.
    """
    import time

    inc = _incoming_dir(blobs_root)
    if not inc.exists():
        return 0
    now = time.time()
    n = 0
    for p in inc.iterdir():
        try:
            if now - p.stat().st_mtime > max_age_s:
                p.unlink()
                n += 1
        except (FileNotFoundError, PermissionError):
            continue
    return n
