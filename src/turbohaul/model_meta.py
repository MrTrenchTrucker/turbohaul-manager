"""Per-blob (per-model) metadata: currently just a human-written description
(a free-text note per model).

A model has no record of its own -- it is a blob on disk plus whatever
manifests happen to point at it, and manifest-level `description` describes
the VARIANT, not the model (in practice, variants of one model rarely have
descriptions that agree). This is where model-level facts that don't belong
to any one manifest variant live. Keyed by blob DIGEST, not by manifest tag,
so a description survives a rename/duplicate of every manifest pointing at
the same weights.

Deliberately NOT inside the blob store directory: that tree is
content-addressed and `blob_store.list_blobs()` walks it expecting only
2-char shard dirs and 64-hex filenames -- dropping a JSON file in there
risks confusing that walk or a future GC pass. Lives beside the manifests
directory instead (one level up from `manifests_path`).
"""
import contextlib
import json
import os
import re
import tempfile
from pathlib import Path


MAX_DESCRIPTION_LEN = 2000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def is_valid_digest(digest: str) -> bool:
    return bool(_SHA256_RE.match(digest))


def meta_path(manifests_root: Path) -> Path:
    return Path(manifests_root).parent / "model_meta.json"


def read_model_meta(manifests_root: Path) -> dict:
    """Load the metadata file. Missing or corrupt -> empty dict, NEVER raises.

    Same defensive posture as manifests.py's per-entry catch:
    a metadata file must never be able to 500 the blob listing.

    "NEVER raises" depends on catching UnicodeDecodeError below. It is a
    ValueError, not an OSError, so a model_meta.json that is not
    valid UTF-8 would walk straight past the read guard -- while the class that
    would catch it sits one line lower, in the json try. Left uncaught, it would
    500 GET /api/blobs, PUT /api/blobs/{digest}/description, and
    DELETE /api/delete (the last AFTER the blob had already been unlinked).
    The read guard below therefore catches UnicodeDecodeError as well.
    """
    path = meta_path(manifests_root)
    try:
        text = path.read_text()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return {}
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def description_for(meta: dict, digest: str) -> str | None:
    """Pull one digest's description out of an already-loaded meta dict.

    Split out from get_description() so a caller enumerating many digests
    (GET /api/blobs) reads the file ONCE, not once per blob.
    """
    entry = meta.get(digest)
    if not isinstance(entry, dict):
        return None
    desc = entry.get("description")
    return desc if isinstance(desc, str) else None


def get_description(manifests_root: Path, digest: str) -> str | None:
    return description_for(read_model_meta(manifests_root), digest)


def _write_model_meta_atomic(manifests_root: Path, data: dict) -> None:
    """Tempfile-in-same-dir + fsync(file) + rename + fsync(dir) -- same
    discipline as manifest.py's write_manifest_atomic. A half-written
    metadata file must never be readable."""
    path = meta_path(manifests_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def set_description(manifests_root: Path, digest: str, description: str | None) -> None:
    """Set or clear one digest's description.

    None or empty string DELETES the key entirely rather than storing an
    empty entry -- keeps the store clean.
    """
    data = read_model_meta(manifests_root)
    if not description:
        data.pop(digest, None)
    else:
        data[digest] = {"description": description}
    _write_model_meta_atomic(manifests_root, data)


def prune_digest(manifests_root: Path, digest: str) -> None:
    """Drop a digest's metadata entry when its blob is deleted.

    Otherwise the store quietly accumulates descriptions of files that no
    longer exist.
    """
    set_description(manifests_root, digest, None)
