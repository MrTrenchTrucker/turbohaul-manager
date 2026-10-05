"""The per-model `max_instances` setting is retired.

It used to cap how many engine processes one model could run and silently
overrode the host-wide parallel-engine budget. Now the only limits are that
budget and the cards the model fits on. An old manifest, PUT body or import that
still carries the key must keep loading: the key is dropped, one warning names
the tag and the key (never a value), and nothing is rejected. Any OTHER unknown
key must still fail, because the models stay `extra="forbid"`.
"""
from __future__ import annotations

import copy
import logging

import pytest
from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from turbohaul.manifest import (
    RETIRED_MODEL_KEYS,
    Manifest,
    ModelManifest,
    _drop_retired_keys,
    parse_manifest,
)

MANIFEST_LOGGER = "turbohaul.manifest"


def _base(**over):
    data = {
        "model_tag": "borage-sub",
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 1024,
        "context_size": 2048,
    }
    data.update(over)
    return data


def _retired_warnings(caplog):
    return [
        r for r in caplog.records
        if r.name == MANIFEST_LOGGER and r.levelno == logging.WARNING
        and "max_instances" in r.getMessage()
    ]


def test_retired_key_set_is_exactly_max_instances():
    assert RETIRED_MODEL_KEYS == ("max_instances",)


def test_key_with_value_one_parses_and_is_dropped():
    m = parse_manifest(_base(max_instances=1))
    assert "max_instances" not in m.model_dump()
    assert "max_instances" not in m.model_dump(mode="json")
    assert "max_instances" not in ModelManifest.model_fields
    assert not hasattr(m, "max_instances")


def test_default_manifest_has_no_such_field():
    m = parse_manifest(_base(model_tag="legacy"))
    assert "max_instances" not in m.model_dump()


def test_old_validator_is_gone_multi_without_auto_place_loads():
    # The old rule rejected max_instances > 1 unless auto_place was true and
    # split_mode was explicitly "none". Neither is needed any more.
    m = parse_manifest(_base(max_instances=2, auto_place=False))
    assert m.auto_place is False
    assert "max_instances" not in m.model_dump()


def test_old_validator_is_gone_layer_split_loads():
    m = parse_manifest(
        _base(max_instances=2, auto_place=True, llama_server_flags={"split_mode": "layer"})
    )
    assert m.llama_server_flags["split_mode"] == "layer"
    assert "max_instances" not in m.model_dump()


def test_old_validator_is_gone_absent_split_mode_loads():
    m = parse_manifest(_base(max_instances=2, auto_place=True))
    assert "max_instances" not in m.model_dump()


@pytest.mark.parametrize("value", [0, 99, 9, -3, "many", None])
def test_out_of_old_range_or_odd_values_are_dropped_not_rejected(value):
    m = parse_manifest(_base(max_instances=value))
    assert "max_instances" not in m.model_dump()


def test_exactly_one_warning_naming_tag_and_key_and_no_value(caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    parse_manifest(_base(model_tag="warn-tag", max_instances=7))
    warnings = _retired_warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "manifest 'warn-tag': dropped retired key(s) max_instances (no longer supported)"
    )
    # The value must not appear anywhere in the record.
    assert "7" not in warnings[0].getMessage()
    assert 7 not in (warnings[0].args or ())


def test_warning_does_not_echo_other_manifest_content(caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    parse_manifest(_base(
        model_tag="quiet-tag", max_instances=3,
        display_name="SECRET-DISPLAY-NAME", description="SECRET-DESCRIPTION",
    ))
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "SECRET-DISPLAY-NAME" not in text
    assert "SECRET-DESCRIPTION" not in text


def test_payload_without_the_key_logs_no_retired_key_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    parse_manifest(_base())
    assert _retired_warnings(caplog) == []
    assert [r for r in caplog.records if r.name == MANIFEST_LOGGER
            and r.levelno >= logging.WARNING] == []


def test_other_unknown_key_still_fails_extra_forbid():
    with pytest.raises(PydanticValidationError) as exc:
        parse_manifest(_base(not_a_real_key=1))
    assert "not_a_real_key" in str(exc.value)


def test_other_unknown_key_still_fails_even_with_the_retired_key_alongside():
    with pytest.raises(PydanticValidationError) as exc:
        parse_manifest(_base(max_instances=2, not_a_real_key=1))
    text = str(exc.value)
    assert "not_a_real_key" in text
    # The retired key is not what gets reported: it was dropped first.
    assert "max_instances" not in text


@pytest.mark.parametrize(
    "validate",
    [
        pytest.param(parse_manifest, id="parse_manifest"),
        pytest.param(ModelManifest.model_validate, id="model_validate"),
        pytest.param(TypeAdapter(Manifest).validate_python, id="type_adapter"),
    ],
)
def test_callers_dict_is_not_mutated(validate):
    # model_validate and the TypeAdapter hand the before-validator the caller's
    # OWN dict object (parse_manifest makes its own copy first), so those two
    # are the paths that prove the validator copies before it drops the key.
    payload = _base(max_instances=4, kind="model", llama_server_flags={"ctx_size": 2048})
    before = copy.deepcopy(payload)
    m = validate(payload)
    assert "max_instances" not in m.model_dump()
    assert payload == before
    assert payload["max_instances"] == 4


def test_direct_construction_accepts_the_key_and_does_not_mutate():
    payload = _base(max_instances=2)
    before = copy.deepcopy(payload)
    m = ModelManifest(**payload)
    assert "max_instances" not in m.model_dump()
    assert payload == before


def test_direct_construction_logs_exactly_one_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    ModelManifest(**_base(max_instances=2))
    assert len(_retired_warnings(caplog)) == 1


def test_type_adapter_over_the_union_accepts_the_key():
    payload = _base(max_instances=2, kind="model")
    before = copy.deepcopy(payload)
    m = TypeAdapter(Manifest).validate_python(payload)
    assert isinstance(m, ModelManifest)
    assert "max_instances" not in m.model_dump()
    assert payload == before


def test_discriminated_union_still_picks_model_manifest_when_key_present():
    # No explicit kind: parse_manifest defaults it, then the union picks
    # ModelManifest and ITS before-validator drops the key.
    m = parse_manifest(_base(max_instances=2))
    assert type(m) is ModelManifest
    assert m.kind == "model"


def test_plugin_manifest_with_the_key_still_fails():
    plugin = {
        "kind": "plugin",
        "model_tag": "a-plugin",
        "lane": "cpu",
        "resource_key": "some-key",
    }
    parse_manifest(plugin)  # control: the plugin body is valid without the key
    with pytest.raises(PydanticValidationError) as exc:
        parse_manifest({**plugin, "max_instances": 2})
    assert "max_instances" in str(exc.value)


# --- _drop_retired_keys unit tests ---------------------------------------------


def test_drop_retired_keys_removes_in_place_and_returns_names():
    payload = {"model_tag": "x", "max_instances": 3, "context_size": 2048}
    returned = _drop_retired_keys(payload)
    assert returned == ["max_instances"]
    assert payload == {"model_tag": "x", "context_size": 2048}


def test_drop_retired_keys_returns_empty_list_when_none_present():
    payload = {"model_tag": "x", "context_size": 2048}
    assert _drop_retired_keys(payload) == []
    assert payload == {"model_tag": "x", "context_size": 2048}


def test_drop_retired_keys_leaves_unknown_keys_alone():
    payload = {"max_instances": 1, "mystery": "keep-me", "another": None}
    assert _drop_retired_keys(payload) == ["max_instances"]
    assert payload == {"mystery": "keep-me", "another": None}


def test_drop_retired_keys_removes_a_key_whose_value_is_none():
    payload = {"max_instances": None, "model_tag": "x"}
    assert _drop_retired_keys(payload) == ["max_instances"]
    assert payload == {"model_tag": "x"}


def test_drop_retired_keys_never_touches_nested_dicts():
    nested = {"max_instances": 5, "n_gpu_layers": 9}
    payload = {"model_tag": "x", "llama_server_flags": nested}
    assert _drop_retired_keys(payload) == []
    assert payload["llama_server_flags"] is nested
    assert nested == {"max_instances": 5, "n_gpu_layers": 9}


def test_drop_retired_keys_second_call_is_a_no_op():
    payload = {"max_instances": 2}
    assert _drop_retired_keys(payload) == ["max_instances"]
    assert _drop_retired_keys(payload) == []


# --- the before-validator's edge cases ----------------------------------------


@pytest.mark.parametrize("bad", [5, None, 1.5, True, "text", ["model_tag"]])
def test_non_mapping_input_is_a_validation_error_not_a_crash(bad):
    # The validator only looks inside a dict. Anything else must reach pydantic
    # untouched and come back as its usual validation error, not as a TypeError
    # from `key in <non-container>`.
    with pytest.raises(PydanticValidationError) as exc:
        ModelManifest.model_validate(bad)
    assert exc.value.errors()[0]["type"] == "model_type"


def test_long_tag_is_cut_to_64_characters_in_the_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    with pytest.raises(PydanticValidationError):   # 300 chars is not a valid tag
        ModelManifest(**_base(model_tag="a" * 300, max_instances=1))
    messages = [r.getMessage() for r in caplog.records if r.name == MANIFEST_LOGGER]
    assert messages == [
        "manifest '" + "a" * 64 + "': dropped retired key(s) max_instances (no longer supported)"
    ]


@pytest.mark.parametrize("tag", [123, None, ["x" * 10], 1.5])
def test_non_string_tag_is_reported_as_unknown_in_the_warning(caplog, tag):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    with pytest.raises(PydanticValidationError):
        ModelManifest(**_base(model_tag=tag, max_instances=1))
    messages = [r.getMessage() for r in caplog.records if r.name == MANIFEST_LOGGER]
    assert messages == [
        "manifest <unknown>: dropped retired key(s) max_instances (no longer supported)"
    ]


def test_missing_tag_is_reported_as_unknown_in_the_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    payload = _base(max_instances=1)
    del payload["model_tag"]
    with pytest.raises(PydanticValidationError):
        ModelManifest(**payload)
    messages = [r.getMessage() for r in caplog.records if r.name == MANIFEST_LOGGER]
    assert messages == [
        "manifest <unknown>: dropped retired key(s) max_instances (no longer supported)"
    ]
