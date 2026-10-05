"""Tests for model_meta -- per-blob (per-model) description storage
(one description per model blob)."""
import json

import pytest

from turbohaul.model_meta import (
    MAX_DESCRIPTION_LEN,
    description_for,
    get_description,
    is_valid_digest,
    meta_path,
    prune_digest,
    read_model_meta,
    set_description,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


class TestIsValidDigest:
    def test_valid_64_hex(self):
        assert is_valid_digest("a" * 64)

    def test_rejects_wrong_length(self):
        assert not is_valid_digest("a" * 63)
        assert not is_valid_digest("a" * 65)

    def test_rejects_non_hex(self):
        assert not is_valid_digest("g" * 64)

    def test_rejects_sha256_prefix(self):
        assert not is_valid_digest("sha256:" + "a" * 64)


class TestMetaPath:
    def test_lives_beside_manifests_dir_not_inside_it(self, tmp_path):
        manifests_root = tmp_path / "state" / "manifests"
        manifests_root.mkdir(parents=True)
        path = meta_path(manifests_root)
        assert path == tmp_path / "state" / "model_meta.json"
        assert path.parent == manifests_root.parent


class TestReadModelMeta:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert read_model_meta(tmp_path / "manifests") == {}

    def test_corrupt_json_returns_empty_dict_not_raise(self, tmp_path):
        manifests_root = tmp_path / "manifests"
        manifests_root.mkdir()
        meta_path(manifests_root).write_text("{not json at all")
        assert read_model_meta(manifests_root) == {}

    def test_non_dict_json_returns_empty_dict(self, tmp_path):
        manifests_root = tmp_path / "manifests"
        manifests_root.mkdir()
        meta_path(manifests_root).write_text("[1, 2, 3]")
        assert read_model_meta(manifests_root) == {}

    def test_reads_back_what_was_written(self, tmp_path):
        manifests_root = tmp_path / "manifests"
        manifests_root.mkdir()
        set_description(manifests_root, SHA_A, "hello")
        assert read_model_meta(manifests_root) == {SHA_A: {"description": "hello"}}


class TestDescriptionFor:
    def test_present_entry(self):
        meta = {SHA_A: {"description": "hi"}}
        assert description_for(meta, SHA_A) == "hi"

    def test_absent_digest_is_none(self):
        assert description_for({}, SHA_A) is None

    def test_malformed_entry_is_none_not_raise(self):
        assert description_for({SHA_A: "not-a-dict"}, SHA_A) is None

    def test_non_string_description_is_none(self):
        assert description_for({SHA_A: {"description": 123}}, SHA_A) is None


class TestSetDescription:
    def test_set_then_get(self, tmp_path):
        root = tmp_path / "manifests"
        root.mkdir()
        set_description(root, SHA_A, "A fine model")
        assert get_description(root, SHA_A) == "A fine model"

    def test_empty_string_clears_key(self, tmp_path):
        root = tmp_path / "manifests"
        root.mkdir()
        set_description(root, SHA_A, "temp")
        set_description(root, SHA_A, "")
        assert SHA_A not in read_model_meta(root)

    def test_none_clears_key(self, tmp_path):
        root = tmp_path / "manifests"
        root.mkdir()
        set_description(root, SHA_A, "temp")
        set_description(root, SHA_A, None)
        assert SHA_A not in read_model_meta(root)

    def test_clearing_absent_key_is_a_noop_not_an_error(self, tmp_path):
        root = tmp_path / "manifests"
        root.mkdir()
        set_description(root, SHA_A, None)  # never set -- must not raise
        assert read_model_meta(root) == {}

    def test_does_not_disturb_other_digests(self, tmp_path):
        root = tmp_path / "manifests"
        root.mkdir()
        set_description(root, SHA_A, "for A")
        set_description(root, SHA_B, "for B")
        set_description(root, SHA_A, None)
        assert read_model_meta(root) == {SHA_B: {"description": "for B"}}

    def test_write_is_atomic_no_tempfile_left_behind(self, tmp_path):
        root = tmp_path / "manifests"
        root.mkdir()
        set_description(root, SHA_A, "hello")
        leftovers = list((tmp_path).glob("*.tmp_*"))
        assert leftovers == []


class TestPruneDigest:
    def test_prunes_existing_entry(self, tmp_path):
        root = tmp_path / "manifests"
        root.mkdir()
        set_description(root, SHA_A, "goes away")
        prune_digest(root, SHA_A)
        assert SHA_A not in read_model_meta(root)

    def test_prune_absent_digest_is_a_noop(self, tmp_path):
        root = tmp_path / "manifests"
        root.mkdir()
        prune_digest(root, SHA_A)  # never existed -- must not raise
        assert read_model_meta(root) == {}

    def test_prune_leaves_other_digests_intact(self, tmp_path):
        root = tmp_path / "manifests"
        root.mkdir()
        set_description(root, SHA_A, "for A")
        set_description(root, SHA_B, "for B")
        prune_digest(root, SHA_A)
        assert read_model_meta(root) == {SHA_B: {"description": "for B"}}
