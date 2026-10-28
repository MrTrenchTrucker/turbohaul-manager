"""Contract test for the coresident_admission module's public interface.

Asserts that the module exports exactly the registry's public list
(modules.toml [modules.coresident_admission]), that each name is usable, that the
module card names every public name, that the module stays pure, and that the
signatures match the documented shape.
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

import turbohaul.coresident_admission as coresident_admission  # noqa: E402
from turbohaul.coresident_admission import (  # noqa: E402
    SiblingLoad,
    admit,
    fits,
    reserved_loading_mib,
    retry_read_free_vram,
)

EXPECTED_PUBLIC = {"SiblingLoad", "retry_read_free_vram", "reserved_loading_mib", "fits", "admit"}

CARD_HEADINGS = [
    "Purpose",
    "Owns",
    "Does Not Own",
    "Public Interface",
    "Depends On",
    "Invariants",
    "Test Locations",
    "Known Gotchas",
]

# Modules this one must never import, directly or through a name it pulls in.
# turbohaul is the package itself: the module depends on nothing inside it.
FORBIDDEN_TOP_LEVEL_IMPORTS = {"turbohaul", "os", "subprocess", "logging", "socket", "time", "threading"}

_MODULE_DIR = _TREE / "src" / "turbohaul" / "coresident_admission"
_REGISTRY = _TREE / "modules.toml"


def _registry_entry():
    data = tomllib.loads(_REGISTRY.read_text(encoding="utf-8"))
    return data["modules"]["coresident_admission"]


def _card_text():
    return (_MODULE_DIR / "AGENTS.md").read_text(encoding="utf-8")


def _public_interface_section():
    """Return the card text under '## Public Interface', up to the next '## ' heading."""
    lines = _card_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "## Public Interface"]
    assert len(starts) == 1, (
        "coresident_admission AGENTS.md must have exactly one '## Public Interface' heading, "
        f"found {len(starts)}"
    )
    section = []
    for line in lines[starts[0] + 1:]:
        if line.startswith("## "):
            break
        section.append(line)
    return "\n".join(section)


# Names the module file imports and that are not part of its interface.
IMPORTED_HELPER_NAMES = {"annotations", "Callable", "Iterable", "Sequence", "dataclass"}


def _assert_every_export_is_defined_here_and_nothing_else_is_public():
    """Each __all__ name is defined by this module, and no other public name exists."""
    module_vars = vars(coresident_admission)
    for name in sorted(coresident_admission.__all__):
        assert name in module_vars, (
            f"coresident_admission.__all__ lists {name!r} but the module does not define it"
        )
        obj = module_vars[name]
        defined_in = getattr(obj, "__module__", None)
        assert defined_in == coresident_admission.__name__, (
            f"coresident_admission.{name} must be defined in the module itself, "
            f"but its __module__ is {defined_in!r}"
        )
    public = {name for name in module_vars if not name.startswith("_")}
    extras = sorted(public - set(coresident_admission.__all__) - IMPORTED_HELPER_NAMES)
    assert extras == [], (
        "coresident_admission has public module-level names outside __all__ and the "
        f"known import helpers: {extras}; put them behind an underscore or add them to __all__"
    )


def test_exports_exactly_the_literal_public_set():
    assert set(coresident_admission.__all__) == EXPECTED_PUBLIC, (
        "coresident_admission.__all__ must export exactly the five public names: "
        f"expected {sorted(EXPECTED_PUBLIC)}, got {sorted(coresident_admission.__all__)}"
    )
    _assert_every_export_is_defined_here_and_nothing_else_is_public()


def test_exports_exactly_the_registry_public_list():
    registry_public = _registry_entry()["public"]
    assert set(coresident_admission.__all__) == set(registry_public), (
        "coresident_admission.__all__ must equal the modules.toml public list: "
        f"registry {sorted(registry_public)}, __all__ {sorted(coresident_admission.__all__)}"
    )
    _assert_every_export_is_defined_here_and_nothing_else_is_public()


def test_registry_public_list_is_the_literal_set_without_duplicates():
    registry_public = _registry_entry()["public"]
    assert len(registry_public) == len(set(registry_public)), (
        f"modules.toml [modules.coresident_admission] public list repeats a name: {registry_public}"
    )
    assert set(registry_public) == EXPECTED_PUBLIC, (
        "modules.toml [modules.coresident_admission] public list must be the five "
        f"documented names, got {sorted(registry_public)}"
    )


def test_registry_declares_no_dependencies():
    depends_on = _registry_entry()["depends_on"]
    assert depends_on == [], (
        f"coresident_admission must depend on nothing (pure module), registry says {depends_on}"
    )


def test_registry_paths_point_at_the_module_folder_and_its_card():
    entry = _registry_entry()
    assert (_TREE / entry["path"]).resolve() == _MODULE_DIR.resolve(), (
        f"modules.toml path must be the coresident_admission folder, got {entry['path']}"
    )
    assert (_TREE / entry["card"]).is_file(), (
        f"modules.toml card must name an existing file, got {entry['card']}"
    )
    card = (_TREE / entry["card"]).resolve()
    assert card.name == "AGENTS.md", (
        f"modules.toml card must be the module's AGENTS.md, got {entry['card']}"
    )
    assert card.is_relative_to(_MODULE_DIR.resolve()), (
        f"modules.toml card must sit inside the coresident_admission folder, got {entry['card']}"
    )


def test_every_public_name_is_present_and_callable():
    for name in sorted(EXPECTED_PUBLIC):
        assert hasattr(coresident_admission, name), (
            f"coresident_admission has no attribute {name!r}"
        )
        assert callable(getattr(coresident_admission, name)), (
            f"coresident_admission.{name} is not callable"
        )


def test_card_has_the_eight_headings_in_order():
    headings = [
        line[3:].strip()
        for line in _card_text().splitlines()
        if line.startswith("## ")
    ]
    assert headings == CARD_HEADINGS, (
        "coresident_admission AGENTS.md must have the eight card headings in order: "
        f"expected {CARD_HEADINGS}, got {headings}"
    )


def test_card_names_every_public_name():
    section = _public_interface_section()
    assert section.strip(), (
        "coresident_admission AGENTS.md has an empty '## Public Interface' section"
    )
    for name in sorted(EXPECTED_PUBLIC):
        assert name in section, (
            f"coresident_admission AGENTS.md '## Public Interface' section does not name "
            f"the public interface item {name!r}"
        )


def _imported_top_level_names(tree):
    names = set()
    relative = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = True
            if node.module:
                names.add(node.module.split(".")[0])
    return names, relative


def test_module_source_imports_nothing_from_turbohaul_or_the_io_modules():
    module_files = sorted(_MODULE_DIR.glob("*.py"))
    assert module_files, "coresident_admission has no Python files to check"
    for path in module_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names, relative = _imported_top_level_names(tree)
        forbidden = sorted(names & FORBIDDEN_TOP_LEVEL_IMPORTS)
        assert forbidden == [], (
            f"coresident_admission must be pure and depend on nothing in turbohaul; "
            f"{path.name} imports {forbidden}"
        )
        assert not relative, (
            f"coresident_admission {path.name} must not use relative imports; "
            "the module depends on nothing inside the package"
        )


def test_admit_signature_is_the_five_argument_entry():
    params = list(inspect.signature(admit).parameters)
    assert params == ["vals", "main_gpu", "siblings", "credit", "need"], (
        f"coresident_admission.admit must take (vals, main_gpu, siblings, credit, need) "
        f"in that order, got {params}"
    )


def test_retry_read_free_vram_signature_has_attempts_defaulting_to_two():
    sig = inspect.signature(retry_read_free_vram)
    assert list(sig.parameters) == ["read", "attempts"], (
        f"coresident_admission.retry_read_free_vram must take (read, attempts), got {list(sig.parameters)}"
    )
    assert sig.parameters["attempts"].default == 2, (
        "coresident_admission.retry_read_free_vram must default attempts to 2 so the "
        f"retry matches the old two-read idiom, got {sig.parameters['attempts'].default!r}"
    )


def test_fits_signature_is_the_four_argument_inequality():
    params = list(inspect.signature(fits).parameters)
    assert params == ["free_mib", "reserve_mib", "credit_mib", "need_mib"], (
        f"coresident_admission.fits must take (free_mib, reserve_mib, credit_mib, need_mib), got {params}"
    )


def test_reserved_loading_mib_signature_is_siblings_then_gpu():
    params = list(inspect.signature(reserved_loading_mib).parameters)
    assert params == ["siblings", "gpu"], (
        f"coresident_admission.reserved_loading_mib must take (siblings, gpu), got {params}"
    )


def test_sibling_load_is_a_frozen_dataclass_with_three_fields():
    assert inspect.isclass(SiblingLoad), "coresident_admission.SiblingLoad must be a class"
    assert dataclasses.is_dataclass(SiblingLoad), "coresident_admission.SiblingLoad must be a dataclass"
    assert SiblingLoad.__dataclass_params__.frozen is True, (
        "coresident_admission.SiblingLoad must be frozen"
    )
    fields = [f.name for f in dataclasses.fields(SiblingLoad)]
    assert fields == ["main_gpu", "reserved_need_mib", "loading"], (
        f"coresident_admission.SiblingLoad must have fields (main_gpu, reserved_need_mib, "
        f"loading), got {fields}"
    )
