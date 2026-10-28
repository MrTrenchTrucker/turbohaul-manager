"""Contract test for the cold_meta_sweep module's public interface.

Asserts that cold_meta_sweep exports exactly the registry's `public` list
(modules.toml [modules.cold_meta_sweep]), in order, that each name is usable, that the registry entry
agrees, and that the module imports nothing from turbohaul.
"""
import ast
import inspect
import pathlib
import tomllib

import turbohaul.cold_meta_sweep as cold_meta_sweep

EXPECTED_PUBLIC = ["run_once", "maybe_sweep"]
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_exports_exactly_the_registry_public_names_in_order():
    assert list(cold_meta_sweep.__all__) == EXPECTED_PUBLIC


def test_every_public_name_is_present():
    for name in EXPECTED_PUBLIC:
        assert hasattr(cold_meta_sweep, name), f"cold_meta_sweep has no attribute {name!r}"


def test_run_once_is_a_plain_function_and_maybe_sweep_is_a_coroutine_function():
    assert callable(cold_meta_sweep.run_once)
    assert not inspect.iscoroutinefunction(cold_meta_sweep.run_once)
    assert inspect.iscoroutinefunction(cold_meta_sweep.maybe_sweep)


def test_the_registry_entry_agrees_with_the_package():
    reg = tomllib.loads((REPO_ROOT / "modules.toml").read_text())
    entry = reg["modules"]["cold_meta_sweep"]
    assert entry["public"] == EXPECTED_PUBLIC
    assert entry["depends_on"] == []
    assert entry["path"] == "src/turbohaul/cold_meta_sweep"
    assert pathlib.Path(cold_meta_sweep.__file__).parent == REPO_ROOT / entry["path"]
    assert (REPO_ROOT / entry["card"]).is_file()


def test_the_module_imports_nothing_from_turbohaul():
    root = pathlib.Path(cold_meta_sweep.__file__).parent
    for f in root.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mods = [node.module or ""]
            for m in mods:
                assert not m.startswith("turbohaul"), f"{f.name} imports {m}; the card says nothing inside TurboHaul"
