"""Contract test for the kv_stack_wiring module's public interface.

Asserts that kv_stack_wiring exports exactly the registry's `public` list (modules.toml
[modules.kv_stack_wiring]), in the same order, that each name is usable, that the registry says
depends_on = ["kv_ram_cap", "kv_stale_pairs"], and that the module reaches into turbohaul only through
the package entry points of those two modules.
"""

import ast
import inspect
import sys
import tomllib
from pathlib import Path

_TREE = Path(__file__).resolve().parents[2]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

import turbohaul.kv_stack_wiring as kv_stack_wiring  # noqa: E402

EXPECTED_PUBLIC = [
    "dir_accounting_logged",
    "live_turn_groups",
    "live_kv_protection",
    "save_declined_by_cap",
    "drop_stale_pairs_after_save",
    "evict_soft_floor",
]
ASYNC_FUNCTIONS = {"save_declined_by_cap"}
SYNC_FUNCTIONS = set(EXPECTED_PUBLIC) - ASYNC_FUNCTIONS
MODULE_DIR = _TREE / "src" / "turbohaul" / "kv_stack_wiring"
REGISTRY = _TREE / "modules.toml"
ALLOWED_TURBOHAUL_IMPORTS = {"turbohaul.kv_ram_cap", "turbohaul.kv_stale_pairs"}
SIBLING_FILES = {p.stem for p in MODULE_DIR.glob("*.py")} - {"__init__"}


def _registry_entry() -> dict:
    with REGISTRY.open("rb") as fh:
        return tomllib.load(fh)["modules"]["kv_stack_wiring"]


def test_exports_exactly_the_six_names_in_registry_order():
    assert list(kv_stack_wiring.__all__) == EXPECTED_PUBLIC, (
        "kv_stack_wiring.__all__ must be the 6 names in order (see src/turbohaul/kv_stack_wiring/AGENTS.md)")


def test_every_public_name_is_present_and_a_function_of_the_right_kind():
    for name in EXPECTED_PUBLIC:
        assert hasattr(kv_stack_wiring, name), f"kv_stack_wiring has no attribute {name!r} (see its AGENTS.md)"
        obj = getattr(kv_stack_wiring, name)
        assert inspect.isfunction(obj), f"kv_stack_wiring.{name} must be a function (see its AGENTS.md)"
        assert inspect.iscoroutinefunction(obj) == (name in ASYNC_FUNCTIONS), (
            f"kv_stack_wiring.{name}: async only for {sorted(ASYNC_FUNCTIONS)} (see its AGENTS.md)")
    assert SYNC_FUNCTIONS | ASYNC_FUNCTIONS == set(EXPECTED_PUBLIC), "the test's own split must cover all 6 names"


def test_registry_public_list_equals_all_in_the_same_order():
    public = _registry_entry()["public"]
    assert public == list(kv_stack_wiring.__all__), (
        "modules.toml [modules.kv_stack_wiring] public must equal kv_stack_wiring.__all__ in order")
    assert public == EXPECTED_PUBLIC, "modules.toml [modules.kv_stack_wiring] public must be the 6 names in order"


def test_registry_depends_on_is_the_two_decision_modules():
    assert _registry_entry()["depends_on"] == ["kv_ram_cap", "kv_stale_pairs"], (
        "modules.toml [modules.kv_stack_wiring] depends_on must be exactly kv_ram_cap and kv_stale_pairs")


def test_turbohaul_is_reached_only_through_the_two_package_entry_points():
    offenders = []
    for path in sorted(MODULE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [f"{path.name}: import {a.name}" for a in node.names
                              if a.name == "turbohaul" or a.name.startswith("turbohaul.")]
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module and (node.module == "turbohaul"
                                                        or node.module.startswith("turbohaul.")):
                    if node.module not in ALLOWED_TURBOHAUL_IMPORTS:
                        offenders.append(f"{path.name}: from {node.module} import ...")
                elif node.level == 1 and node.module not in SIBLING_FILES:
                    offenders.append(f"{path.name}: from .{node.module} import ...")
                elif node.level > 1:
                    offenders.append(f"{path.name}: relative import above the package")
    assert offenders == [], (
        f"kv_stack_wiring may import turbohaul only through kv_ram_cap and kv_stale_pairs, found: {offenders}")
