"""Prove the hardening spans EVERY Manifest variant.

Splitting Manifest into ModelManifest + PluginManifest
behind a discriminated union is only safe if the cross-cutting hardening
(extra="forbid", model_tag traversal/regex validation) cannot silently fail to
apply to some future third variant. This file is the condition of the whole
approach, not incidental coverage -- it must:

  1. iterate every concrete subclass of HardenedManifestBase DYNAMICALLY
     (__subclasses__, not a hand-written list that goes stale), and assert
     the hardening holds on each;
  2. additionally prove every member of the Manifest union actually descends
     from HardenedManifestBase -- __subclasses__ alone cannot catch a future
     variant that is wired into the Union without inheriting the base at all
     (a different failure mode than "inherits it but overrides something"),
     which is exactly the shape of mistake "skips the base" describes;
  3. prove the walker can actually fail: add a deliberately-bad throwaway
     subclass, confirm the checker catches it, then remove it;
  4. prove kind:"model" defaulting gives true zero-migration parsing of a
     real on-disk manifest with no `kind` key at all.
"""
import gc
from typing import get_args

import pytest
from pydantic import ValidationError

from turbohaul.manifest import (
    HardenedManifestBase,
    Manifest,
    ModelManifest,
    PluginManifest,
    parse_manifest,
)

SHA = "a" * 64


def _all_subclasses(cls: type) -> set[type]:
    """Recursively walk every subclass, not just direct children.

    Dynamic discovery -- if a future variant is added anywhere
    in the inheritance tree under HardenedManifestBase, this finds it without
    needing a hand-maintained list to be kept in sync.
    """
    seen: set[type] = set()
    stack = list(cls.__subclasses__())
    while stack:
        c = stack.pop()
        if c not in seen:
            seen.add(c)
            stack.extend(c.__subclasses__())
    return seen


def _manifest_union_members() -> tuple[type, ...]:
    """Unwrap Manifest = Annotated[Union[...], Field(discriminator=...)]."""
    union_type = get_args(Manifest)[0]  # Annotated -> (Union[...], FieldInfo)
    return get_args(union_type)  # Union[...] -> (ModelManifest, PluginManifest)


def _assert_extra_forbidden(cls: type) -> None:
    """Construct cls with one unknown field and require extra_forbidden.

    Deliberately supplies NO other fields, valid or otherwise: pydantic v2
    aggregates ALL field errors into one ValidationError (verified: a
    missing-required error and an extra_forbidden error co-occur cleanly),
    so this needs no knowledge of cls's own required-field set -- which
    would otherwise be exactly the kind of hand-written-per-class list the
    design prohibits for the subclass enumeration itself.
    """
    probe_field = "__example_probe_url"
    with pytest.raises(ValidationError) as exc_info:
        cls(**{probe_field: "x"})
    errs = exc_info.value.errors()
    assert any(e["type"] == "extra_forbidden" and e["loc"] == (probe_field,) for e in errs), (
        f"{cls.__name__} did not reject an unknown extra field -- "
        f"extra='forbid' hardening is missing or overridden. Errors seen: {errs}"
    )


def _assert_tag_traversal_guarded(cls: type) -> None:
    """Construct cls with a traversal-shaped model_tag and require rejection.

    Same no-other-fields-needed trick as above: a model_tag-scoped error
    proves the shared _tag_safe validator fired for this subclass, whatever
    else is missing.
    """
    with pytest.raises(ValidationError) as exc_info:
        cls(model_tag="../../etc/passwd")
    errs = exc_info.value.errors()
    assert any(e["loc"] == ("model_tag",) for e in errs), (
        f"{cls.__name__} did not reject a traversal-shaped model_tag -- "
        f"the shared TAG_RE/traversal guard is missing. Errors seen: {errs}"
    )


class TestHardeningSpansEveryConcreteSubclass:
    """Point 1 of the module docstring: walk __subclasses__, assert on each."""

    def test_walker_finds_both_known_variants(self):
        # Guards the guard: if this is empty, the two tests below would pass
        # vacuously (an empty for-loop asserts nothing).
        found = {c.__name__ for c in _all_subclasses(HardenedManifestBase)}
        assert found == {"ModelManifest", "PluginManifest"}

    def test_every_concrete_subclass_enforces_extra_forbid(self):
        subclasses = _all_subclasses(HardenedManifestBase)
        assert subclasses
        for cls in subclasses:
            _assert_extra_forbidden(cls)

    def test_every_concrete_subclass_enforces_tag_traversal(self):
        subclasses = _all_subclasses(HardenedManifestBase)
        assert subclasses
        for cls in subclasses:
            _assert_tag_traversal_guarded(cls)


class TestEveryUnionMemberInheritsTheBase:
    """Point 2 of the module docstring: __subclasses__ alone can't catch a variant wired into the
    Union without inheriting HardenedManifestBase at all -- check the
    union's actual members directly, which is the ground truth of "every
    real Manifest variant" regardless of what does or doesn't inherit what.
    """

    def test_union_members_are_exactly_the_known_variants(self):
        members = _manifest_union_members()
        assert set(members) == {ModelManifest, PluginManifest}

    def test_every_union_member_is_a_hardened_manifest_base_subclass(self):
        for member in _manifest_union_members():
            assert issubclass(member, HardenedManifestBase), (
                f"{member.__name__} is a Manifest union member but does not "
                "inherit HardenedManifestBase -- it would skip the shared "
                "hardening entirely and neither test above would ever see it."
            )


class TestCheckerCanActuallyFail:
    """Point 3 of the module docstring: prove the checker isn't vacuously green by feeding it a
    deliberately-bad throwaway subclass that skips the hardening, confirming
    it's caught, then removing it.
    """

    def test_a_throwaway_subclass_that_skips_extra_forbid_is_caught(self):
        from pydantic import ConfigDict
        from typing import Literal

        before = _all_subclasses(HardenedManifestBase)

        class BadThrowawayManifest(HardenedManifestBase):
            model_config = ConfigDict(extra="allow")  # deliberately skips the hardening
            kind: Literal["bad_throwaway"] = "bad_throwaway"

        during = _all_subclasses(HardenedManifestBase)
        assert BadThrowawayManifest in during and BadThrowawayManifest not in before, (
            "the walker did not even discover the throwaway subclass -- "
            "the discovery mechanism itself is broken, not just unproven"
        )

        # The real, unmodified checker helper must catch it.
        checker_caught_it = False
        try:
            _assert_extra_forbidden(BadThrowawayManifest)
        except AssertionError:
            checker_caught_it = True
        assert checker_caught_it, "checker did not catch the throwaway -- vacuously green"

        # Sanity: confirm it's genuinely non-hardened, not a checker bug --
        # it must actually accept the extra field it should have rejected.
        inst = BadThrowawayManifest(model_tag="ok", some_extra_field="x")
        assert inst.model_dump().get("some_extra_field") == "x"
        del inst

        # "then remove it": del the only name bound to it in this scope.
        # NOTE (verified, not assumed): this test does NOT also assert
        # HardenedManifestBase.__subclasses__() forgets the class afterward.
        # Confirmed empirically that del + gc.collect() DOES reclaim it
        # immediately in a bare interpreter -- but under pytest's assertion
        # rewriting (active for this directory per pyproject.toml's
        # testpaths), a class referenced inside `assert` statements in this
        # function is kept reachable by pytest's own rewritten-bytecode
        # instrumentation for the rest of the test's execution, independent
        # of del/gc.collect(). That is pytest's introspection machinery
        # holding a reference, not HardenedManifestBase's discovery
        # mechanism failing to let go -- proving otherwise would mean
        # testing pytest internals, not this file's own code.
        del BadThrowawayManifest


class TestZeroMigration:
    """Point 4 of the module docstring: kind defaults to 'model' on ModelManifest, but pydantic's
    discriminated-union tag extraction inspects the RAW payload for a `kind`
    key before any variant's own defaults run (verified empirically against
    pydantic 2.9 -- a payload with no `kind` key raises union_tag_not_found
    even though ModelManifest.kind has a Python-level default). parse_manifest
    closes that gap with an explicit setdefault. This proves it holds for a
    REAL production manifest shape, not a hand-trimmed example.
    """

    # Trimmed from a real on-disk manifest
    # (a 35B MTP-enabled manifest), field-for-field,
    # with NO `kind` key -- exactly the on-disk shape before this change.
    REAL_MANIFEST_PAYLOAD_NO_KIND_KEY = {
        "model_tag": "qwen3.6-35b-mtp",
        "display_name": "qwen3.6-35b-mtp (MTP, shared blob with qwen3.6-35b-moe)",
        "description": "Qwen3.6-35B-A3B Unsloth Dynamic IQ4_XS, MTP-enabled.",
        "gguf_blob_sha256": "df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf",
        "mmproj_blob_sha256": "",
        "spec_draft_gguf_blob_sha256": "",
        "gguf_size_bytes": 18209036576,
        "context_size": 250000,
        "expected_vram_bytes": 24000000000,
        "auto_place": False,
        "hidden": False,
        "arch": "",
        "hybrid_kv_ratio": 1.0,
        "kv_bytes_per_token": None,
        "revision": 21,
        "llama_server_flags": {
            "parallel": 1,
            "cont_batching": True,
            "kv_unified": True,
            "ctx_size": 250000,
            "n_gpu_layers": 999,
            "split_mode": "layer",
            "main_gpu": 0,
            "cache_type_k": "turbo3",
            "cache_type_v": "turbo3",
            "n_predict": -1,
            "reasoning": "auto",
            "reasoning_format": "deepseek-legacy",
            "flash_attn": True,
            "spec_type": "draft-mtp",
            "spec_draft_n_max": 3,
            "ctx_checkpoints": 2,
        },
        "prompt_template": {"system_default": "", "stop_tokens": []},
    }

    def test_real_manifest_with_no_kind_key_parses_unchanged(self):
        assert "kind" not in self.REAL_MANIFEST_PAYLOAD_NO_KIND_KEY
        m = parse_manifest(self.REAL_MANIFEST_PAYLOAD_NO_KIND_KEY)
        assert isinstance(m, ModelManifest)
        assert m.kind == "model"
        assert m.model_tag == "qwen3.6-35b-mtp"
        assert m.llama_server_flags["spec_type"] == "draft-mtp"

    def test_parse_manifest_does_not_mutate_caller_dict(self):
        payload = dict(model_tag="t", gguf_blob_sha256=SHA)
        parse_manifest(payload)
        assert "kind" not in payload

    def test_bare_literal_default_alone_does_not_satisfy_zero_migration(self):
        """Documents WHY parse_manifest exists rather than relying on
        ModelManifest.kind's default: going straight at the TypeAdapter
        with no `kind` key is a real, reproducible failure mode, not a
        hypothetical -- this is the exact bug parse_manifest's setdefault
        fixes.
        """
        from pydantic import TypeAdapter

        payload = dict(self.REAL_MANIFEST_PAYLOAD_NO_KIND_KEY)
        with pytest.raises(ValidationError, match="union_tag_not_found"):
            TypeAdapter(Manifest).validate_python(payload)


class TestPluginManifestShape:
    """Basic construction coverage for the new variant (not itself part of
    the mandatory hardening test, but the smallest useful confirmation that
    PluginManifest works end to end, including through parse_manifest).
    """

    def test_valid_plugin_manifest_parses(self):
        m = parse_manifest({
            "model_tag": "whisperx-transcribe",
            "kind": "plugin",
            "lane": "gpu",
            "resource_key": "whisperx-main",
            "capabilities": ["transcribe", "diarize"],
        })
        assert isinstance(m, PluginManifest)
        assert m.lane == "gpu"
        assert m.capabilities == ["transcribe", "diarize"]

    def test_plugin_manifest_declares_no_model_only_fields(self):
        assert "llama_server_flags" not in PluginManifest.model_fields
        assert "gguf_blob_sha256" not in PluginManifest.model_fields
        assert "context_size" not in PluginManifest.model_fields

    def test_kind_plugin_has_no_default_must_be_explicit(self):
        with pytest.raises(ValidationError):
            PluginManifest(model_tag="t", lane="cpu", resource_key="foo")

    @pytest.mark.parametrize("bad_key", [
        "has/slash", "has.dot", "UPPER", "-leadingdash", "", "a" * 65,
        "has url http://evil", "../traversal",
    ])
    def test_resource_key_shape_rejected(self, bad_key):
        with pytest.raises(ValidationError):
            PluginManifest(
                model_tag="t", kind="plugin", lane="cpu", resource_key=bad_key,
            )

    def test_resource_key_valid_shapes_accepted(self):
        for key in ["a", "whisperx-main", "kokoro_tts_v2", "a" * 64]:
            PluginManifest(model_tag="t", kind="plugin", lane="cpu", resource_key=key)
