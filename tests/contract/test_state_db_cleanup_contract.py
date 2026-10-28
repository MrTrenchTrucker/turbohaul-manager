"""Contract test for the state_db_cleanup module's public interface.

Asserts that state_db_cleanup exports exactly the registry's `public` list
(modules.toml [modules.state_db_cleanup]), that each name is usable, and that the module imports from
TurboHaul only what the registry says it depends on.
"""
import ast
import pathlib
import tomllib

import turbohaul.state_db_cleanup as state_db_cleanup

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPECTED_PUBLIC = {"run", "main", "Config", "Env", "Stop", "parse_args"}


def _entry():
    return tomllib.loads((ROOT / "modules.toml").read_text())["modules"]["state_db_cleanup"]


def test_exports_exactly_the_registry_public_names():
    assert set(state_db_cleanup.__all__) == EXPECTED_PUBLIC
    assert set(_entry()["public"]) == EXPECTED_PUBLIC


def test_every_public_name_is_present():
    for name in EXPECTED_PUBLIC:
        assert hasattr(state_db_cleanup, name), f"state_db_cleanup has no attribute {name!r}"


def test_the_functions_are_callable_and_stop_is_an_exception():
    for name in ("run", "main", "parse_args"):
        assert callable(getattr(state_db_cleanup, name))
    assert issubclass(state_db_cleanup.Stop, Exception)


def test_the_module_imports_only_what_the_registry_declares():
    root = pathlib.Path(state_db_cleanup.__file__).parent
    declared = set(_entry()["depends_on"])
    assert declared == {"slot_meta_projection"}
    seen = set()
    for f in root.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mods = [node.module or ""]
            for m in mods:
                parts = m.split(".")
                if parts[0] == "turbohaul":
                    seen.add(parts[1] if len(parts) > 1 else "")
    assert seen - {"state_db_cleanup"} == declared, f"imports {sorted(seen)}; the card declares {sorted(declared)}"
