"""Contract test for the slot_meta_projection module's public interface.

Asserts that slot_meta_projection exports exactly the registry's `public` list
(modules.toml [modules.slot_meta_projection]) and that each name is usable.
"""
import ast
import pathlib

import turbohaul.slot_meta_projection as slot_meta_projection

EXPECTED_PUBLIC = {"IDENTITY_KEYS", "CENSUS_KEYS", "CENSUS_SLOTS_SQL", "identity_client_meta", "parse_census_meta",
                   "strip_stored_meta"}


def test_exports_exactly_the_registry_public_names():
    assert set(slot_meta_projection.__all__) == EXPECTED_PUBLIC


def test_every_public_name_is_present():
    for name in EXPECTED_PUBLIC:
        assert hasattr(slot_meta_projection, name), f"slot_meta_projection has no attribute {name!r}"


def test_the_two_functions_are_callable_and_the_rest_are_plain_values():
    assert callable(slot_meta_projection.identity_client_meta)
    assert callable(slot_meta_projection.parse_census_meta)
    assert callable(slot_meta_projection.strip_stored_meta)
    assert isinstance(slot_meta_projection.IDENTITY_KEYS, tuple)
    assert isinstance(slot_meta_projection.CENSUS_KEYS, tuple)
    assert isinstance(slot_meta_projection.CENSUS_SLOTS_SQL, str)


def test_the_module_imports_nothing_from_turbohaul():
    root = pathlib.Path(slot_meta_projection.__file__).parent
    for f in root.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mods = [node.module or ""]
            for m in mods:
                assert not m.startswith("turbohaul"), f"{f.name} imports {m}; the card says nothing inside TurboHaul"
