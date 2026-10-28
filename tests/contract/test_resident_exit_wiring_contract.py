"""Contract test for the resident_exit_wiring public interface (see src/turbohaul/resident_exit_wiring/AGENTS.md).

Asserts that resident_exit_wiring exports exactly the three names of the spec in order, that each is usable,
that the module imports nothing from turbohaul except resident_exit through its package entry point, and that the
registry entry [modules.resident_exit_wiring] in modules.toml agrees with the package.
"""

import ast
import sys
import tomllib
from pathlib import Path

_TREE = Path(__file__).resolve().parents[2]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

import turbohaul.resident_exit_wiring as wiring  # noqa: E402

CARD = "resident_exit_wiring, see src/turbohaul/resident_exit_wiring/AGENTS.md"
EXPECTED_ALL = [
    "hold_warm_window",
    "log_resident_exit",
    "resolve_exit_reason",
]
PKG_DIR = _TREE / "src" / "turbohaul" / "resident_exit_wiring"
ALLOWED_PROJECT_IMPORT = "turbohaul.resident_exit"


def test_all_is_exactly_the_three_spec_names_in_order():
    assert list(wiring.__all__) == EXPECTED_ALL, f"{CARD}: __all__ is {list(wiring.__all__)!r}"


def test_every_public_name_is_present_and_callable():
    for name in EXPECTED_ALL:
        assert hasattr(wiring, name), f"{CARD}: resident_exit_wiring has no attribute {name!r}"
        assert callable(getattr(wiring, name)), f"{CARD}: resident_exit_wiring.{name} is not callable"


def _imports_of(path: Path):
    """(absolute module names, relative module names) for every import statement in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    absolute, relative = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            absolute.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level >= 1:
                relative.append(node.module or "")
            else:
                absolute.append(node.module or "")
    return absolute, relative


def test_module_imports_nothing_from_turbohaul_except_the_resident_exit_entry_point():
    files = ["__init__.py", "exit_log.py", "hold.py"]
    seen_relative = set()
    seen_allowed = set()
    bad = []
    for name in files:
        path = PKG_DIR / name
        assert path.is_file(), f"{CARD}: missing source file {name}"
        absolute, relative = _imports_of(path)
        seen_relative.update(relative)
        for mod in absolute:
            if mod.split(".")[0] != "turbohaul":
                continue
            if mod == ALLOWED_PROJECT_IMPORT:
                seen_allowed.add(name)
            else:
                bad.append((name, mod))
    assert not bad, f"{CARD}: forbidden project imports: {bad!r}"
    # The scan must have seen the package's own relative imports and the allowed import, or it proved nothing.
    assert {"exit_log", "hold"} <= seen_relative, (
        f"{CARD}: the import scan saw relative imports {sorted(seen_relative)!r}"
    )
    assert seen_allowed == {"exit_log.py", "hold.py"}, (
        f"{CARD}: the files importing {ALLOWED_PROJECT_IMPORT} are {sorted(seen_allowed)!r}"
    )


def test_registry_entry_agrees_with_the_package():
    with (_TREE / "modules.toml").open("rb") as fh:
        data = tomllib.load(fh)
    entry = data["modules"]["resident_exit_wiring"]
    assert entry["depends_on"] == ["resident_exit"], f"{CARD}: registry depends_on is {entry['depends_on']!r}"
    assert entry["public"] == list(wiring.__all__), f"{CARD}: registry public is {entry['public']!r}"
