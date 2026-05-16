"""Tests for blob_store - content-addressed storage + lifecycle (v0.2 §12.1)."""
import hashlib
import os

import pytest

from turbohaul.blob_store import (
    BlobError,
    BlobHashMismatch,
    BlobIntegrityFailure,
    BlobSizeExceeded,
    blob_exists,
    blob_path,
    delete_blob,
    ensure_blob_layout,
    gc_stale_incoming,
    list_blobs,
    make_incoming_path,
    verify_blob_on_stage,
    write_stream_atomic,
)


def _chunks_of(data: bytes, chunk_size: int = 1024):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


class TestLayout:
    def test_ensure_blob_layout(self, tmp_path):
        ensure_blob_layout(tmp_path)
        assert (tmp_path / "sha256" / "incoming").is_dir()

    def test_make_incoming_path_unique(self, tmp_path):
        p1 = make_incoming_path(tmp_path)
        p2 = make_incoming_path(tmp_path)
        assert p1 != p2
        assert p1.suffix == ".tmp"


class TestWriteStream:
    def test_write_basic(self, tmp_path):
        data = b"hello world" * 1000  # 11KB
        expected = hashlib.sha256(data).hexdigest()
        sha, n = write_stream_atomic(tmp_path, _chunks_of(data))
        assert sha == expected
        assert n == len(data)
        # Blob at final path
        final = blob_path(tmp_path, sha)
        assert final.exists()
        # 0o400 read-only post-rename
        mode = final.stat().st_mode & 0o777
        assert mode == 0o400

    def test_write_with_expected_hash_pass(self, tmp_path):
        data = b"data" * 50
        expected = hashlib.sha256(data).hexdigest()
        sha, n = write_stream_atomic(
            tmp_path, _chunks_of(data), expected_sha256=expected
        )
        assert sha == expected

    def test_write_with_wrong_expected_hash_raises(self, tmp_path):
        data = b"data" * 50
        with pytest.raises(BlobHashMismatch, match="computed"):
            write_stream_atomic(
                tmp_path, _chunks_of(data), expected_sha256="f" * 64
            )
        # Tempfile should be cleaned up
        assert not list((tmp_path / "sha256" / "incoming").iterdir())

    def test_write_exceeds_size_cap(self, tmp_path):
        data = b"x" * 1000
        with pytest.raises(BlobSizeExceeded, match="per_stream_max_bytes"):
            write_stream_atomic(
                tmp_path, _chunks_of(data, 100), per_stream_max_bytes=500
            )
        # Tempfile cleanup
        assert not list((tmp_path / "sha256" / "incoming").iterdir())

    def test_write_bad_expected_hash_format(self, tmp_path):
        with pytest.raises(BlobError, match="expected_sha256 not valid hex"):
            write_stream_atomic(tmp_path, _chunks_of(b"x"), expected_sha256="bad")


class TestBlobOps:
    def test_blob_exists(self, tmp_path):
        data = b"x" * 100
        sha, _ = write_stream_atomic(tmp_path, _chunks_of(data))
        assert blob_exists(tmp_path, sha) is True
        assert blob_exists(tmp_path, "f" * 64) is False

    def test_list_blobs(self, tmp_path):
        d1 = b"first"
        d2 = b"second"
        s1, _ = write_stream_atomic(tmp_path, _chunks_of(d1))
        s2, _ = write_stream_atomic(tmp_path, _chunks_of(d2))
        listed = list_blobs(tmp_path)
        assert s1 in listed
        assert s2 in listed

    def test_delete_blob(self, tmp_path):
        data = b"to-delete"
        sha, _ = write_stream_atomic(tmp_path, _chunks_of(data))
        assert delete_blob(tmp_path, sha) is True
        assert blob_exists(tmp_path, sha) is False
        assert delete_blob(tmp_path, sha) is False  # idempotent


class TestVerifyBlobOnStage:
    def test_verify_passes_for_clean_blob(self, tmp_path):
        data = b"clean blob" * 100
        sha, _ = write_stream_atomic(tmp_path, _chunks_of(data))
        assert verify_blob_on_stage(tmp_path, sha) is True

    def test_verify_raises_on_missing(self, tmp_path):
        with pytest.raises(BlobIntegrityFailure, match="missing at stage"):
            verify_blob_on_stage(tmp_path, "f" * 64)

    def test_verify_detects_tamper(self, tmp_path):
        """Simulate TOCTOU blob swap — overwrite on-disk content."""
        data = b"original"
        sha, _ = write_stream_atomic(tmp_path, _chunks_of(data))
        path = blob_path(tmp_path, sha)
        # Bypass 0o400 to overwrite
        os.chmod(str(path), 0o600)
        path.write_bytes(b"TAMPERED CONTENT")
        with pytest.raises(BlobIntegrityFailure, match="hash drift"):
            verify_blob_on_stage(tmp_path, sha)


class TestGcStaleIncoming:
    def test_no_files_to_gc(self, tmp_path):
        ensure_blob_layout(tmp_path)
        assert gc_stale_incoming(tmp_path) == 0

    def test_gc_removes_old_tempfiles(self, tmp_path):
        ensure_blob_layout(tmp_path)
        old_tmp = tmp_path / "sha256" / "incoming" / "stale.tmp"
        old_tmp.write_bytes(b"stale")
        # backdate the file
        import time
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(str(old_tmp), (old_time, old_time))
        n = gc_stale_incoming(tmp_path, max_age_s=3600.0)
        assert n == 1
        assert not old_tmp.exists()

    def test_gc_keeps_fresh_tempfiles(self, tmp_path):
        ensure_blob_layout(tmp_path)
        fresh = tmp_path / "sha256" / "incoming" / "fresh.tmp"
        fresh.write_bytes(b"fresh")
        n = gc_stale_incoming(tmp_path, max_age_s=3600.0)
        assert n == 0
        assert fresh.exists()


class TestPathFormat:
    def test_path_uses_2char_shard(self, tmp_path):
        data = b"x"
        sha, _ = write_stream_atomic(tmp_path, _chunks_of(data))
        # blob lives at sha256/<first-2>/<full>
        expected = tmp_path / "sha256" / sha[:2] / sha
        assert expected.exists()

    def test_path_rejects_bad_hex(self, tmp_path):
        with pytest.raises(BlobError, match="invalid sha256 hex"):
            blob_path(tmp_path, "notvalid")
