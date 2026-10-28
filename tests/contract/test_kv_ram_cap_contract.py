"""Contract test for the kv_ram_cap module's public interface.

Asserts that kv_ram_cap exports exactly the registry's `public` list (modules.toml
[modules.kv_ram_cap]), in the same order, that each name is usable, that the registry says
depends_on = [], and that no file in the module imports anything from turbohaul.
"""

import ast
import inspect
import sys
import tomllib
from pathlib import Path

_TREE = Path(__file__).resolve().parents[2]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

import turbohaul.kv_ram_cap as kv_ram_cap  # noqa: E402

EXPECTED_PUBLIC = [
    "dir_accounting",
    "ProtectionSplit",
    "split_protection",
    "cap_exceeded",
    "SaveGate",
    "enforce_save_cap",
    "soft_age_targets",
    "soft_ceiling_order",
]
FUNCTIONS = {"dir_accounting", "split_protection", "cap_exceeded", "enforce_save_cap",
             "soft_age_targets", "soft_ceiling_order"}
CLASSES = {"ProtectionSplit", "SaveGate"}
MODULE_DIR = _TREE / "src" / "turbohaul" / "kv_ram_cap"
REGISTRY = _TREE / "modules.toml"


def _registry_entry() -> dict:
    with REGISTRY.open("rb") as fh:
        return tomllib.load(fh)["modules"]["kv_ram_cap"]


def test_exports_exactly_the_eight_names_in_registry_order():
    assert list(kv_ram_cap.__all__) == EXPECTED_PUBLIC, (
        "kv_ram_cap.__all__ must be the 8 names in order (see src/turbohaul/kv_ram_cap/AGENTS.md)")


def test_every_public_name_is_present_with_the_right_kind():
    for name in EXPECTED_PUBLIC:
        assert hasattr(kv_ram_cap, name), f"kv_ram_cap has no attribute {name!r} (see its AGENTS.md)"
        obj = getattr(kv_ram_cap, name)
        if name in FUNCTIONS:
            assert inspect.isfunction(obj), f"kv_ram_cap.{name} must be a function (see its AGENTS.md)"
        else:
            assert inspect.isclass(obj), f"kv_ram_cap.{name} must be a class (see its AGENTS.md)"
    assert FUNCTIONS | CLASSES == set(EXPECTED_PUBLIC), "the test's own name split must cover all 8 names"


def test_registry_public_list_equals_all_in_the_same_order():
    public = _registry_entry()["public"]
    assert public == list(kv_ram_cap.__all__), (
        "modules.toml [modules.kv_ram_cap] public must equal kv_ram_cap.__all__ in order")
    assert public == EXPECTED_PUBLIC, "modules.toml [modules.kv_ram_cap] public must be the 8 names in order"


def test_registry_depends_on_is_empty():
    assert _registry_entry()["depends_on"] == [], (
        "modules.toml [modules.kv_ram_cap] depends_on must be [] (the module is pure policy)")


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
    assert offenders == [], f"kv_ram_cap must import nothing from turbohaul, found: {offenders}"
