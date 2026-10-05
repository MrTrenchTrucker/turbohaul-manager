"""Per-model llama_server_flags defaults.

These defaults exist because an unset flag silently inherits llama-server's own
default, which is not the intended value. The property that matters most
is the one these tests exist to protect: they are DEFAULTS, never overrides. A
manifest that sets a value explicitly must keep it -- including a falsy one.
"""
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from turbohaul import manifest as manifest_mod
from turbohaul.manifest import (
    MANIFEST_FLAG_DEFAULTS,
    SAFE_LLAMA_FLAGS,
    Manifest,
    ModelManifest,
    ManifestValidationError,
)

SHA = "a" * 64


def _mk(**kwargs) -> Manifest:
    return ModelManifest(model_tag="test-model", gguf_blob_sha256=SHA, **kwargs)


def _seen(m: Manifest) -> dict:
    return {k: m.llama_server_flags.get(k) for k in MANIFEST_FLAG_DEFAULTS}


class TestDefaultsApplied:
    def test_empty_flags_dict_gets_every_default(self):
        assert _seen(_mk(llama_server_flags={})) == MANIFEST_FLAG_DEFAULTS

    def test_flags_field_omitted_entirely_still_gets_defaults(self):
        """The case a field_validator would MISS.

        Pydantic does not run field validators on a field's default value, so
        injecting from the llama_server_flags validator would silently skip any
        manifest that omits the key -- exactly the newly-added-model case these
        defaults exist for. This test fails if the injection is ever moved back
        into a field_validator.
        """
        assert _seen(_mk()) == MANIFEST_FLAG_DEFAULTS

    def test_unrelated_flags_are_preserved_alongside_defaults(self):
        m = _mk(llama_server_flags={"ctx_size": 4096})
        assert m.llama_server_flags["ctx_size"] == 4096
        assert _seen(m) == MANIFEST_FLAG_DEFAULTS


class TestExplicitValuesWin:
    """A default must never overwrite an operator's own choice."""

    def test_explicit_value_beats_default(self):
        m = _mk(llama_server_flags={"cache_type_k": "f16"})
        assert m.llama_server_flags["cache_type_k"] == "f16"
        # ...and the flags the manifest did NOT set are still defaulted.
        assert m.llama_server_flags["cache_type_v"] == MANIFEST_FLAG_DEFAULTS["cache_type_v"]

    def test_explicit_ZERO_survives(self):
        """The falsy-value trap.

        `flags.get(k) or default` would silently replace a deliberate 0 with the
        default. ctx_checkpoints=0 is a real configuration -- a model that opts
        out of the checkpoint ladder entirely -- so 0 must survive untouched.
        """
        m = _mk(llama_server_flags={"ctx_checkpoints": 0})
        assert m.llama_server_flags["ctx_checkpoints"] == 0

    @pytest.mark.parametrize("flag,override", [
        ("ctx_checkpoints", 1),
        ("cache_type_k", "q8_0"),
        ("cache_type_v", "f16"),
    ])
    def test_every_default_is_individually_overridable(self, flag, override):
        m = _mk(llama_server_flags={flag: override})
        assert m.llama_server_flags[flag] == override


class TestDefaultsTableItself:
    def test_every_default_is_an_allowlisted_flag(self):
        for flag in MANIFEST_FLAG_DEFAULTS:
            assert flag in SAFE_LLAMA_FLAGS, f"{flag} not allowlisted"

    def test_every_default_value_passes_flag_validation(self):
        """Guards the bypass: defaults are injected AFTER field validation.

        An invalid entry in the table would otherwise reach llama-server's argv
        unchecked. The module asserts this at import; this test states it as a
        contract so the guard cannot be quietly deleted.
        """
        m = _mk(llama_server_flags=dict(MANIFEST_FLAG_DEFAULTS))
        assert _seen(m) == MANIFEST_FLAG_DEFAULTS

    def test_invalid_manifest_value_is_still_rejected(self):
        """Adding defaults must not weaken ordinary flag validation.

        Pydantic wraps the ManifestValidationError raised inside the field
        validator, so the observable type here is pydantic's ValidationError --
        asserting the inner type would pass only by accident.
        """
        with pytest.raises(PydanticValidationError):
            ModelManifest(
                model_tag="test-model",
                gguf_blob_sha256=SHA,
                llama_server_flags={"cache_type_k": "not_a_real_cache_type"},
            )

    def test_import_time_guard_rejects_a_bad_default(self):
        """The guard fires at IMPORT, so it cannot be reached by constructing a
        Manifest -- the only honest test is to re-execute the module with the
        table mutated. If this passes with the guard deleted, it is vacuous, so
        it asserts on the mutant rather than on the real module.
        """
        src = Path(manifest_mod.__file__).read_text(encoding="utf-8")
        mutant = src.replace('"cache_type_k": "turbo3",',
                             '"cache_type_k": "not_a_real_cache_type",', 1)
        assert mutant != src, "mutation did not apply -- test is vacuous"
        # exec() rebuilds the module, so its ManifestValidationError is a NEW
        # class object -- not the one imported here. Match on name + message
        # rather than identity, which would never hold.
        with pytest.raises(Exception) as exc:
            exec(compile(mutant, "<mutant manifest>", "exec"), {"__name__": "_mutant"})
        assert type(exc.value).__name__ == "ManifestValidationError"
        assert "cache_type_k" in str(exc.value)

    def test_pinned_values(self):
        """Pinned deliberately: changing a default changes behaviour for every
        model that has not opted out, so it should require editing this test."""
        assert MANIFEST_FLAG_DEFAULTS == {
            "ctx_checkpoints": 2,
            "cache_type_k": "turbo3",
            "cache_type_v": "turbo3",
        }


class TestDefaultsAreNeverPersisted:
    """A default must apply on READ and never be written into the file.

    Writing it in would freeze today's value as an explicit per-model choice, so
    a later change to MANIFEST_FLAG_DEFAULTS would silently skip every manifest
    saved in the meantime -- splitting the models between those that follow the
    default and models that only look like they do.
    """

    def _write(self, tmp_path, **flags):
        from turbohaul.manifest import write_manifest_atomic
        m = ModelManifest(model_tag="probe", gguf_blob_sha256=SHA, llama_server_flags=flags)
        write_manifest_atomic(tmp_path, m)
        import yaml
        return yaml.safe_load((tmp_path / "probe.yaml").read_text())["llama_server_flags"]

    def test_injected_defaults_are_stripped_before_write(self, tmp_path):
        on_disk = self._write(tmp_path, ctx_size=8192)
        for flag in MANIFEST_FLAG_DEFAULTS:
            assert flag not in on_disk, f"{flag} was frozen into the manifest"
        assert on_disk["ctx_size"] == 8192

    def test_explicit_values_survive_the_write(self, tmp_path):
        on_disk = self._write(tmp_path, ctx_size=8192, cache_type_k="turbo2",
                              ctx_checkpoints=0)
        assert on_disk["cache_type_k"] == "turbo2"
        # the falsy trap again -- 0 is a real choice, not an absence
        assert on_disk["ctx_checkpoints"] == 0
        # ...while the one NOT set is still stripped
        assert "cache_type_v" not in on_disk

    def test_defaults_still_apply_when_read_back(self, tmp_path):
        from turbohaul.manifest import read_manifest
        self._write(tmp_path, ctx_size=8192)
        eff = read_manifest(tmp_path, "probe").llama_server_flags
        assert {k: eff[k] for k in MANIFEST_FLAG_DEFAULTS} == MANIFEST_FLAG_DEFAULTS

    def test_a_round_trip_does_not_accumulate_flags(self, tmp_path):
        """Save-load-save must be stable, or repeated FE edits grow the file."""
        from turbohaul.manifest import read_manifest, write_manifest_atomic
        import yaml
        self._write(tmp_path, ctx_size=8192)
        for _ in range(3):
            m = read_manifest(tmp_path, "probe")
            write_manifest_atomic(tmp_path, m, if_match=f'"{m.revision}"')
        on_disk = yaml.safe_load((tmp_path / "probe.yaml").read_text())["llama_server_flags"]
        assert on_disk == {"ctx_size": 8192}, f"flags accumulated: {on_disk}"


class TestRestoreDefaultsClearsOnlyDefaultedFlags:
    def test_overrides_cleared_but_unrelated_tuning_untouched(self, tmp_path):
        """The dangerous failure mode: clearing ctx_size or tensor_split would
        break the very models most likely to be reset."""
        from turbohaul.manifest import read_manifest, write_manifest_atomic
        import yaml
        m = ModelManifest(model_tag="tuned", gguf_blob_sha256=SHA, llama_server_flags={
            "ctx_size": 250000, "tensor_split": "0.55,0.45",
            "cache_type_k": "turbo2", "ctx_checkpoints": 0})
        write_manifest_atomic(tmp_path, m)

        existing = read_manifest(tmp_path, "tuned")
        stored = dict(existing.llama_server_flags)
        injected = set(getattr(existing, "_defaulted_flags", set()) or set())
        cleared = [k for k in stored if k in MANIFEST_FLAG_DEFAULTS and k not in injected]
        assert "cache_type_k" in cleared and "ctx_checkpoints" in cleared
        # remove ALL defaulted flags, not just the overrides -- an injected one
        # left here would be re-persisted as an explicit value
        for k in MANIFEST_FLAG_DEFAULTS:
            stored.pop(k, None)
        payload = existing.model_dump(mode="json")
        payload["llama_server_flags"] = stored
        write_manifest_atomic(tmp_path, ModelManifest(**payload),
                              if_match=f'"{existing.revision}"')

        on_disk = yaml.safe_load((tmp_path / "tuned.yaml").read_text())["llama_server_flags"]
        assert on_disk["ctx_size"] == 250000
        assert on_disk["tensor_split"] == "0.55,0.45"
        assert not any(k in on_disk for k in MANIFEST_FLAG_DEFAULTS)
        eff = read_manifest(tmp_path, "tuned").llama_server_flags
        assert {k: eff[k] for k in MANIFEST_FLAG_DEFAULTS} == MANIFEST_FLAG_DEFAULTS
