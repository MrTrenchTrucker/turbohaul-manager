"""tensor_split CSV-numeric validator: the documented accept/reject table
IS the test matrix -- one test per accept case, one per reject case."""
import pytest
from pydantic import ValidationError as PydanticValidationError

from turbohaul.manifest import (
    Manifest,
    ModelManifest,
    ManifestValidationError,
    flags_to_argv,
    parse_strict_csv_numeric,
)

SAMPLE_TAG = "qwen3.6-35b-moe"
SAMPLE_SHA = "1a2b3c4d" + "0" * 56  # 64 hex chars


def make_manifest(**overrides) -> Manifest:
    base = dict(
        model_tag=SAMPLE_TAG,
        display_name="test",
        description="test",
        gguf_blob_sha256=SAMPLE_SHA,
        gguf_size_bytes=22_000_000_000,
        context_size=131072,
        expected_vram_bytes=22_500_000_000,
        revision=1,
        llama_server_flags={"ctx_size": 131072},
    )
    base.update(overrides)
    return ModelManifest(**base)


def _with_tensor_split(value) -> dict:
    return {"ctx_size": 131072, "tensor_split": value}


class TestAccept:
    """Accept list: 0.72,0.28 / 0.5,0.5 / 1,0 / 0,1 / 1,1,1 / 3,1 / 0.75,0.25."""

    @pytest.mark.parametrize(
        "value",
        ["0.72,0.28", "0.5,0.5", "1,0", "0,1", "1,1,1", "3,1", "0.75,0.25"],
    )
    def test_accepted(self, value):
        m = make_manifest(llama_server_flags=_with_tensor_split(value))
        assert m.llama_server_flags["tensor_split"] == value

    def test_one_zero_is_not_degenerate(self):
        # Explicit in the accept table: placing everything on one card is legitimate;
        # only ALL-zero is degenerate.
        make_manifest(llama_server_flags=_with_tensor_split("1,0"))
        make_manifest(llama_server_flags=_with_tensor_split("0,1"))

    def test_unnormalised_ratio_is_legal(self):
        # llama.cpp accepts unnormalised ratios (3,1), not just 0..1 fractions.
        make_manifest(llama_server_flags=_with_tensor_split("3,1"))

    def test_argv_emits_single_opaque_element(self):
        # Confirms the value reaches argv as
        # ONE list element, never shell-interpreted.
        argv = flags_to_argv({"tensor_split": "0.72,0.28"})
        assert argv == ["--tensor-split", "0.72,0.28"]


class TestReject:
    """Reject table, row by row."""

    # Row 1: non-str types (bool must be checked before int, since bool is an int subclass)
    @pytest.mark.parametrize("value", [0.5, [0.5, 0.5], True, None])
    def test_row1_non_str_types(self, value):
        with pytest.raises(PydanticValidationError):
            make_manifest(llama_server_flags=_with_tensor_split(value))

    # Row 2: empty / whitespace-only / internal whitespace
    @pytest.mark.parametrize("value", ["", " ", "0.5, 0.5"])
    def test_row2_whitespace(self, value):
        with pytest.raises(PydanticValidationError):
            make_manifest(llama_server_flags=_with_tensor_split(value))

    # Row 3: any character outside [0-9.,]
    @pytest.mark.parametrize(
        "value",
        [
            "0.5;0.5", "0.5|0.5", "0.5&0.5", "0.5$0.5", "0.5`0.5",
            "0.5(0.5)", "0.5<0.5>", "0.5\n0.5", "0.5\\0.5", "0.5'0.5",
            '0.5"0.5', "0.5*0.5", "0.5?0.5", "0.5~0.5", "0.5!0.5",
            "0.5#0.5", "abc,def",
        ],
    )
    def test_row3_invalid_chars(self, value):
        with pytest.raises(PydanticValidationError):
            make_manifest(llama_server_flags=_with_tensor_split(value))

    # Row 4: scientific / special numerics -- explicitly asserted given the
    # known hazard so a future pattern loosening cannot silently re-admit
    # them (float() alone accepts all of these).
    @pytest.mark.parametrize("value", ["1e3,1", "1E3,1", "inf,1", "nan,1", "Infinity,1"])
    def test_row4_scientific_and_special(self, value):
        with pytest.raises(PydanticValidationError):
            make_manifest(llama_server_flags=_with_tensor_split(value))

    # Row 5: signs
    @pytest.mark.parametrize("value", ["-1,2", "+1,2"])
    def test_row5_signs(self, value):
        with pytest.raises(PydanticValidationError):
            make_manifest(llama_server_flags=_with_tensor_split(value))

    # Row 6: malformed fields
    @pytest.mark.parametrize(
        "value", ["1.2.3,1", ".", ",0.5", "0.5,", "0.5,,0.5"],
    )
    def test_row6_malformed_fields(self, value):
        with pytest.raises(PydanticValidationError):
            make_manifest(llama_server_flags=_with_tensor_split(value))

    # Row 7: count < 2
    def test_row7_count_below_min(self):
        with pytest.raises(PydanticValidationError):
            make_manifest(llama_server_flags=_with_tensor_split("1"))

    # Row 8: count > 16 (static hard cap). Single-digit fields keep the
    # string well under the row-10 length cap, so this isolates the count
    # check specifically.
    def test_row8_count_above_max(self):
        value = ",".join(["1"] * 17)
        with pytest.raises(PydanticValidationError):
            make_manifest(llama_server_flags=_with_tensor_split(value))

    def test_row8_count_16_is_the_boundary_accept(self):
        value = ",".join(["1"] * 16)
        make_manifest(llama_server_flags=_with_tensor_split(value))

    # Row 9: all-zero (degenerate)
    @pytest.mark.parametrize("value", ["0,0", "0,0,0"])
    def test_row9_all_zero(self, value):
        with pytest.raises(PydanticValidationError):
            make_manifest(llama_server_flags=_with_tensor_split(value))

    # Row 10: length > 64 chars
    def test_row10_too_long(self):
        value = ",".join(["0.123456"] * 9)  # 9 fields * 8 chars + 8 commas = 80 > 64
        assert len(value) > 64
        with pytest.raises(PydanticValidationError):
            make_manifest(llama_server_flags=_with_tensor_split(value))

    # Row 11: leading '-' (explicit argv-injection assertion, same
    # mechanism as row 5 but called out separately in the reject table).
    def test_row11_leading_dash(self):
        with pytest.raises(PydanticValidationError):
            make_manifest(llama_server_flags=_with_tensor_split("-0.5,0.5"))


class TestParseStrictCsvNumericDirect:
    """Direct unit coverage of the shared helper (independent of Manifest/pydantic)."""

    def test_returns_parsed_floats_in_order(self):
        assert parse_strict_csv_numeric(
            "0.72,0.28", min_count=2, max_count=16,
        ) == [0.72, 0.28]

    def test_non_str_input_rejected(self):
        with pytest.raises(ManifestValidationError):
            parse_strict_csv_numeric(0.5, min_count=2, max_count=16)

    def test_max_len_override(self):
        # A shorter max_len (as a caller such as fit_target might use) rejects
        # a value that would otherwise be valid under the default 64.
        with pytest.raises(ManifestValidationError):
            parse_strict_csv_numeric("0.1,0.2", min_count=2, max_count=16, max_len=5)
