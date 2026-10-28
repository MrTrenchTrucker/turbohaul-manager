"""Contract test for the resident_exit module's public interface (see src/turbohaul/resident_exit/AGENTS.md).

Asserts that resident_exit exports exactly the six names of the spec in order, that each is usable,
that the module imports nothing from turbohaul, logging, threading or asyncio, and that the registry
entry [modules.resident_exit] in modules.toml agrees with the package.
"""

import ast
import sys
import tomllib
from pathlib import Path

_TREE = Path(__file__).resolve().parents[2]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

import turbohaul.resident_exit as resident_exit  # noqa: E402

CARD = "resident_exit, see src/turbohaul/resident_exit/AGENTS.md"
EXPECTED_ALL = [
    "ExitReason",
    "needs_exit_line",
    "format_resident_exit",
    "RetentionInputs",
    "RetentionDecision",
    "decide_retention",
]
PKG_DIR = _TREE / "src" / "turbohaul" / "resident_exit"
FORBIDDEN_TOP = {"logging", "threading", "asyncio"}


def test_all_is_exactly_the_six_spec_names_in_order():
    assert list(resident_exit.__all__) == EXPECTED_ALL, (
        f"{CARD}: __all__ is {list(resident_exit.__all__)!r}"
    )


def test_every_public_name_is_present_and_callable():
    for name in EXPECTED_ALL:
        assert hasattr(resident_exit, name), f"{CARD}: resident_exit has no attribute {name!r}"
        assert callable(getattr(resident_exit, name)), f"{CARD}: resident_exit.{name} is not callable"


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


def test_module_imports_nothing_from_turbohaul_logging_threading_or_asyncio():
    files = ["__init__.py", "reasons.py", "exit_line.py", "retention.py"]
    seen_relative = set()
    bad = []
    for name in files:
        path = PKG_DIR / name
        assert path.is_file(), f"{CARD}: missing source file {name}"
        absolute, relative = _imports_of(path)
        seen_relative.update(relative)
        for mod in absolute:
            top = mod.split(".")[0]
            if top == "turbohaul" or top in FORBIDDEN_TOP:
                bad.append((name, mod))
    assert not bad, f"{CARD}: forbidden imports: {bad!r}"
    # The scan must have seen the package's own relative imports, or it proved nothing.
    assert {"exit_line", "reasons", "retention"} <= seen_relative, (
        f"{CARD}: the import scan saw relative imports {sorted(seen_relative)!r}"
    )


def test_registry_entry_agrees_with_the_package():
    with (_TREE / "modules.toml").open("rb") as fh:
        data = tomllib.load(fh)
    entry = data["modules"]["resident_exit"]
    assert entry["depends_on"] == [], f"{CARD}: registry depends_on is {entry['depends_on']!r}"
    assert entry["public"] == list(resident_exit.__all__), (
        f"{CARD}: registry public is {entry['public']!r}"
    )
