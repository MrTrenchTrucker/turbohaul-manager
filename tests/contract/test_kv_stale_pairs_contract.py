"""Contract test for the kv_stale_pairs module's public interface.

Asserts that kv_stale_pairs exports exactly the registry's `public` list (modules.toml
[modules.kv_stale_pairs]), that each name is usable, that the registry says depends_on = [], and
that no file in the module imports anything from turbohaul.
"""

import ast
import dataclasses
import inspect
import sys
import tomllib
from pathlib import Path

_TREE = Path(__file__).resolve().parents[2]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

import turbohaul.kv_stale_pairs as kv_stale_pairs  # noqa: E402

EXPECTED_PUBLIC = {"drop_stale_pairs", "DropResult", "StalePair"}
FUNCTIONS = {"drop_stale_pairs"}
CLASSES = {"DropResult", "StalePair"}
MODULE_DIR = _TREE / "src" / "turbohaul" / "kv_stale_pairs"
REGISTRY = _TREE / "modules.toml"
CARD = "see src/turbohaul/kv_stale_pairs/AGENTS.md"


def _registry_entry() -> dict:
    with REGISTRY.open("rb") as fh:
        return tomllib.load(fh)["modules"]["kv_stale_pairs"]


def test_all_is_exactly_the_three_public_names():
    assert set(kv_stale_pairs.__all__) == EXPECTED_PUBLIC and len(kv_stale_pairs.__all__) == 3, (
        f"kv_stale_pairs.__all__ must be exactly the 3 public names ({CARD})")


def test_registry_public_list_equals_all():
    public = _registry_entry()["public"]
    assert public == list(kv_stale_pairs.__all__), (
        f"modules.toml [modules.kv_stale_pairs] public must equal kv_stale_pairs.__all__ in order ({CARD})")
    assert set(public) == EXPECTED_PUBLIC, f"the registry public list must be the 3 names ({CARD})"


def test_every_public_name_is_present_with_the_right_kind():
    assert FUNCTIONS | CLASSES == EXPECTED_PUBLIC, "the test's own name split must cover all 3 names"
    for name in sorted(EXPECTED_PUBLIC):
        assert hasattr(kv_stale_pairs, name), f"kv_stale_pairs has no attribute {name!r} ({CARD})"
        obj = getattr(kv_stale_pairs, name)
        if name in FUNCTIONS:
            assert inspect.isfunction(obj), f"kv_stale_pairs.{name} must be a function ({CARD})"
        else:
            assert inspect.isclass(obj) and dataclasses.is_dataclass(obj), (
                f"kv_stale_pairs.{name} must be a dataclass ({CARD})")


def test_the_result_types_are_frozen_with_the_documented_fields():
    assert [f.name for f in dataclasses.fields(kv_stale_pairs.StalePair)] == [
        "name", "port", "slot_id", "pair_bytes"], f"StalePair fields ({CARD})"
    assert [f.name for f in dataclasses.fields(kv_stale_pairs.DropResult)] == [
        "dropped", "skipped", "landed_bytes", "n_written"], f"DropResult fields ({CARD})"
    assert kv_stale_pairs.StalePair.__dataclass_params__.frozen is True
    assert kv_stale_pairs.DropResult.__dataclass_params__.frozen is True


def test_drop_stale_pairs_takes_the_documented_arguments():
    params = list(inspect.signature(kv_stale_pairs.drop_stale_pairs).parameters)
    assert params == ["save_dir", "model_tag", "thread_hash", "new_port", "live_turns", "landed_bytes",
                      "n_written", "remove"], f"drop_stale_pairs signature ({CARD})"


def test_the_public_function_is_usable_through_the_entry_point(tmp_path):
    res = kv_stale_pairs.drop_stale_pairs(str(tmp_path), "m", "a1b2c3d4e5f6", 11501, [], 10, 10)
    assert isinstance(res, kv_stale_pairs.DropResult) and res.skipped is None and res.dropped == (), (
        f"an empty directory runs and drops nothing ({CARD})")


def test_registry_depends_on_is_empty():
    assert _registry_entry()["depends_on"] == [], (
        f"modules.toml [modules.kv_stale_pairs] depends_on must be [] (the module is standard library only) ({CARD})")


def test_no_module_file_imports_from_turbohaul():
    offenders = []
    for path in sorted(MODULE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [f"{path.name}: import {a.name}" for a in node.names
                              if a.name == "turbohaul" or a.name.startswith("turbohaul.")]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == "turbohaul" or node.module.startswith("turbohaul."):
                    offenders.append(f"{path.name}: from {node.module} import ...")
    assert offenders == [], f"kv_stale_pairs must import nothing from turbohaul, found: {offenders}"
