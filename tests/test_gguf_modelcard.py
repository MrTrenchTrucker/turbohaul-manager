"""Tests for ModelCard / read_model_card.

Independent of read_kv_dims's own test suite (test_dimension_aware_kv.py) --
this exercises a different GGUF key set (general.* + {arch}.expert_count +
vision markers) via its own compact byte-builder, kept self-contained so the
two test files aren't coupled to each other's internals. The builder here is
also imported by test_api_ollama.py to construct real blob fixtures.
"""
import struct

from turbohaul._gguf_meta import read_kv_dims, read_model_card


def _gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(k: str, v: str) -> bytes:
    return _gguf_string(k) + struct.pack("<I", 8) + _gguf_string(v)


def _kv_u32(k: str, v: int) -> bytes:
    return _gguf_string(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def _kv_u64(k: str, v: int) -> bytes:
    return _gguf_string(k) + struct.pack("<I", 10) + struct.pack("<Q", v)


def _build_gguf(kvs: list, tensor_count: int = 0) -> bytes:
    body = b"GGUF" + struct.pack("<I", 3)
    body += struct.pack("<Q", tensor_count) + struct.pack("<Q", len(kvs))
    return body + b"".join(kvs)


class TestReadModelCard:
    def test_moe_vision_fixture(self, tmp_path):
        p = tmp_path / "moe_vision.gguf"
        p.write_bytes(_build_gguf([
            _kv_str("general.architecture", "qwen2moe"),
            _kv_u64("general.parameter_count", 27_000_000_000),
            _kv_str("general.size_label", "27B"),
            _kv_u32("qwen2moe.expert_count", 8),
            _kv_str("clip.vision.some_key", "x"),
        ]))
        card = read_model_card(p)
        assert card is not None
        assert card.architecture == "qwen2moe"
        assert card.parameter_count == 27_000_000_000
        assert card.size_label == "27B"
        assert card.expert_count == 8
        assert card.is_moe is True
        assert card.has_vision_kv is True

    def test_dense_text_only_fixture(self, tmp_path):
        p = tmp_path / "dense.gguf"
        p.write_bytes(_build_gguf([
            _kv_str("general.architecture", "llama"),
            _kv_u64("general.parameter_count", 8_000_000_000),
        ]))
        card = read_model_card(p)
        assert card is not None
        assert card.architecture == "llama"
        assert card.parameter_count == 8_000_000_000
        assert card.size_label is None
        assert card.expert_count is None
        assert card.is_moe is None  # key absent -> unknown, NOT False
        assert card.has_vision_kv is False

    def test_expert_count_zero_is_dense_not_unknown(self, tmp_path):
        """expert_count key PRESENT but 0 -> is_moe False (known-dense),
        distinct from the key being absent entirely (unknown -> None)."""
        p = tmp_path / "zero_experts.gguf"
        p.write_bytes(_build_gguf([
            _kv_str("general.architecture", "llama"),
            _kv_u32("llama.expert_count", 0),
        ]))
        card = read_model_card(p)
        assert card.expert_count == 0
        assert card.is_moe is False

    def test_vision_key_via_clip_prefix_no_dot_vision(self, tmp_path):
        p = tmp_path / "clip.gguf"
        p.write_bytes(_build_gguf([
            _kv_str("general.architecture", "gemma4"),
            _kv_u32("clip.vision_feature_layer", 1),
        ]))
        card = read_model_card(p)
        assert card.has_vision_kv is True

    def test_malformed_returns_none(self, tmp_path):
        p = tmp_path / "bad.gguf"
        p.write_bytes(b"NOPEyadda")
        assert read_model_card(p) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert read_model_card(tmp_path / "does_not_exist.gguf") is None

    def test_non_path_arg_never_raises(self):
        assert read_model_card(None) is None
        assert read_model_card([1, 2]) is None
        assert read_model_card(object()) is None

    def test_no_architecture_key_leaves_expert_and_moe_none(self, tmp_path):
        """No general.architecture at all -> can't namespace {arch}.expert_count,
        so expert_count/is_moe are None; parameter_count is still read (it isn't
        arch-namespaced)."""
        p = tmp_path / "noarch.gguf"
        p.write_bytes(_build_gguf([
            _kv_u64("general.parameter_count", 1_000_000),
        ]))
        card = read_model_card(p)
        assert card is not None
        assert card.architecture is None
        assert card.parameter_count == 1_000_000
        assert card.expert_count is None
        assert card.is_moe is None

    def test_read_kv_dims_unaffected_by_new_reader(self, tmp_path):
        """Sanity: read_model_card and read_kv_dims are independent parsers
        over the same file -- exercising one must not perturb the other
        (read_kv_dims stays byte-identical: the new reader must not alter it)."""
        p = tmp_path / "both.gguf"
        p.write_bytes(_build_gguf([
            _kv_str("general.architecture", "qwen35"),
            _kv_u32("qwen35.block_count", 64),
            _kv_u32("qwen35.full_attention_interval", 4),
            _kv_u32("qwen35.attention.head_count_kv", 4),
            _kv_u32("qwen35.attention.key_length", 256),
            _kv_u32("qwen35.attention.value_length", 256),
        ]))
        dims = read_kv_dims(p)
        card = read_model_card(p)
        assert dims is not None and dims.arch == "qwen35" and dims.block_count == 64
        assert card is not None and card.architecture == "qwen35"
