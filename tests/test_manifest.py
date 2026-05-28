"""Tests for manifest schema + atomic write + ETag/If-Match concurrency (v0.2 §8.1+§8.2)."""
import pytest
from pydantic import ValidationError as PydanticValidationError

from turbohaul.manifest import (
    ConcurrencyError,
    Manifest,
    ManifestValidationError,
    delete_manifest,
    flags_to_argv,
    list_manifests,
    manifest_etag,
    read_manifest,
    validate_tag,
    write_manifest_atomic,
)

SAMPLE_TAG = "qwen3.6-35b-moe"
SAMPLE_SHA = "1a2b3c4d" + "0" * 56  # 64 hex chars


def make_manifest(**overrides) -> Manifest:
    base = dict(
        model_tag=SAMPLE_TAG,
        display_name="Qwen 3.6 35B-A3B MoE Q4",
        description="test",
        gguf_blob_sha256=SAMPLE_SHA,
        gguf_size_bytes=22_000_000_000,
        context_size=131072,
        expected_vram_bytes=22_500_000_000,
        revision=1,
        llama_server_flags={"ctx_size": 131072, "n_gpu_layers": 999, "mlock": True},
    )
    base.update(overrides)
    return Manifest(**base)


class TestTagValidation:
    def test_valid_tags(self):
        for t in ["qwen3.6-35b-moe", "abc", "a", "tag-name_v1.0", "model123"]:
            validate_tag(t)

    def test_invalid_tags(self):
        for bad in [
            "",
            "../etc/passwd",
            "tag/path",
            "tag\\back",
            "Tag-Upper",
            ".dotted",
            "a" * 65,
            "tag with space",
            "tag\x00null",
        ]:
            with pytest.raises(ManifestValidationError):
                validate_tag(bad)


class TestManifestSchema:
    def test_valid_manifest_parses(self):
        m = make_manifest()
        assert m.model_tag == SAMPLE_TAG
        assert m.llama_server_flags["ctx_size"] == 131072

    def test_reject_unknown_flag(self):
        with pytest.raises(PydanticValidationError, match="not in the closed allowlist"):
            make_manifest(llama_server_flags={"evil_unknown": "x"})

    def test_reject_path_bearing_flag_mmproj(self):
        with pytest.raises(PydanticValidationError, match="explicitly denied"):
            make_manifest(llama_server_flags={"mmproj": "/etc/passwd"})

    def test_reject_path_bearing_flag_lora(self):
        with pytest.raises(PydanticValidationError, match="explicitly denied"):
            make_manifest(llama_server_flags={"lora": "/root/.config/nc_claude.env"})

    def test_reject_path_bearing_flag_log_file(self):
        with pytest.raises(PydanticValidationError, match="explicitly denied"):
            make_manifest(llama_server_flags={"log_file": "/etc/cron.d/pwn"})

    def test_reject_hf_token_override(self):
        with pytest.raises(PydanticValidationError, match="explicitly denied"):
            make_manifest(llama_server_flags={"hf_token": "attacker-token"})

    def test_reject_model_override(self):
        with pytest.raises(PydanticValidationError, match="explicitly denied"):
            make_manifest(llama_server_flags={"model": "/tmp/evil.gguf"})

    def test_reject_bad_sha256(self):
        with pytest.raises(PydanticValidationError, match="64 hex"):
            make_manifest(gguf_blob_sha256="notahash")

    def test_reject_short_sha256(self):
        with pytest.raises(PydanticValidationError):
            make_manifest(gguf_blob_sha256="abc123")

    def test_reject_tag_with_traversal(self):
        with pytest.raises(PydanticValidationError):
            make_manifest(model_tag="../../etc/passwd")

    def test_reject_top_level_unknown_field(self):
        with pytest.raises(PydanticValidationError):
            Manifest(
                model_tag=SAMPLE_TAG,
                gguf_blob_sha256=SAMPLE_SHA,
                evil_field=1,  # type: ignore[call-arg]
            )

    def test_bool_type_strict(self):
        # int 1 must NOT be accepted as bool for mlock
        with pytest.raises(PydanticValidationError, match="expects bool"):
            make_manifest(llama_server_flags={"mlock": 1})


class TestAtomicWriteAndRead:
    def test_write_then_read(self, tmp_path):
        m = make_manifest()
        out = write_manifest_atomic(tmp_path, m)
        assert out.revision == 1
        read = read_manifest(tmp_path, SAMPLE_TAG)
        assert read.model_tag == SAMPLE_TAG
        assert read.gguf_blob_sha256 == SAMPLE_SHA
        assert read.revision == 1

    def test_etag_after_write(self, tmp_path):
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        etag = manifest_etag(tmp_path, SAMPLE_TAG)
        assert etag == '"1"'

    def test_second_write_correct_if_match_increments(self, tmp_path):
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        out2 = write_manifest_atomic(tmp_path, make_manifest(display_name="Updated"), if_match='"1"')
        assert out2.revision == 2
        # Verify on-disk
        read = read_manifest(tmp_path, SAMPLE_TAG)
        assert read.revision == 2
        assert read.display_name == "Updated"

    def test_second_write_wrong_if_match_raises(self, tmp_path):
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        with pytest.raises(ConcurrencyError, match="If-Match"):
            write_manifest_atomic(tmp_path, make_manifest(display_name="Stale"), if_match='"99"')

    def test_no_if_match_on_update_raises(self, tmp_path):
        """Note: PUT without If-Match on existing manifest must raise."""
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        with pytest.raises(ConcurrencyError):
            write_manifest_atomic(tmp_path, make_manifest(display_name="No-ETag"))

    def test_file_mode_is_0o600(self, tmp_path):
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        path = tmp_path / f"{SAMPLE_TAG}.yaml"
        mode = path.stat().st_mode & 0o777
        assert mode & 0o600 == 0o600
        assert mode & 0o077 == 0  # no group/other access


class TestListAndDelete:
    def test_list_manifests(self, tmp_path):
        write_manifest_atomic(tmp_path, make_manifest())
        write_manifest_atomic(tmp_path, make_manifest(model_tag="second"))
        tags = list_manifests(tmp_path)
        assert SAMPLE_TAG in tags
        assert "second" in tags
        assert len(tags) == 2

    def test_list_empty_dir(self, tmp_path):
        assert list_manifests(tmp_path) == []

    def test_list_ignores_hidden(self, tmp_path):
        write_manifest_atomic(tmp_path, make_manifest())
        # Write a hidden file (simulating partial-write leftover)
        (tmp_path / ".tmp_garbage.yaml").write_text("oops")
        tags = list_manifests(tmp_path)
        assert ".tmp_garbage" not in tags
        assert SAMPLE_TAG in tags

    def test_delete(self, tmp_path):
        write_manifest_atomic(tmp_path, make_manifest())
        assert delete_manifest(tmp_path, SAMPLE_TAG) is True
        assert delete_manifest(tmp_path, SAMPLE_TAG) is False  # idempotent

    def test_read_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path, "nope")


class TestFlagsToArgv:
    def test_basic_mapping(self):
        argv = flags_to_argv({"ctx_size": 4096, "n_gpu_layers": 100})
        assert "--ctx-size" in argv
        assert "4096" in argv
        assert "--n-gpu-layers" in argv
        assert "100" in argv

    def test_bool_true_no_value(self):
        argv = flags_to_argv({"mlock": True})
        assert argv == ["--mlock"]

    def test_bool_false_omitted(self):
        argv = flags_to_argv({"mlock": False})
        assert argv == []

    def test_denied_flag_rejected_at_argv(self):
        with pytest.raises(ManifestValidationError, match="blocked at argv-build"):
            flags_to_argv({"mmproj": "/etc/passwd"})

    def test_unknown_flag_rejected_at_argv(self):
        with pytest.raises(ManifestValidationError, match="blocked at argv-build"):
            flags_to_argv({"random_unknown_thing": "foo"})

    def test_snake_to_kebab(self):
        argv = flags_to_argv({"n_cpu_moe": True})
        assert "--n-cpu-moe" in argv
