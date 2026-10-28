"""Contract test for the resident_identity module: the registry's public list equals the real exports, the
card agrees with them, the module touches exactly the host members it declares, imports nothing from
the rest of the project, and every file is under 300 lines."""
import ast
import inspect
import logging
import tomllib
from pathlib import Path

import _resident_identity_fakes as fakes

import turbohaul.resident_identity as resident_identity

_ROOT = Path(__file__).resolve().parents[2]
_PKG = Path(resident_identity.__file__).resolve().parent
_CARD = "src/turbohaul/resident_identity/AGENTS.md"

PUBLIC = ["ResidentIdentityHost", "ResidentLike", "SlotLike", "stash_identity", "identity_for", "forget_identity",
          "serving_resident", "spawn_seq_for_handle", "LoadLike", "pending_load"]
EXPECTED_FILES = ["__init__.py", "engine_load.py", "identity.py", "identity_host.py", "serving.py"]
CARD_HEADINGS = ["Purpose", "Owns", "Does Not Own", "Public Interface", "Depends On", "Invariants",
                 "Test Locations", "Known Gotchas"]
SINGLETON = "__legacy_singleton__"


def _registry():
    with open(_ROOT / "modules.toml", "rb") as f:
        reg = tomllib.load(f)["modules"].get("resident_identity")
    assert reg is not None, f"resident_identity is not registered in modules.toml; see {_CARD}"
    return reg


def _py_files():
    return sorted(_PKG.glob("*.py"))


def _card() -> str:
    return (_PKG / "AGENTS.md").read_text()


# -- the public surface ---------------------------------------------------------------------------

def test_all_is_the_pinned_public_list_in_order():
    assert resident_identity.__all__ == PUBLIC, f"resident_identity: __all__ differs from the pinned list; see {_CARD}"


def test_registry_public_list_equals_all_in_order():
    assert _registry()["public"] == resident_identity.__all__, (
        f"resident_identity: modules.toml public list and __all__ differ; see {_CARD}"
    )


def test_registry_entry_fields():
    reg = _registry()
    assert reg["path"] == "src/turbohaul/resident_identity"
    assert reg["card"] == _CARD
    assert reg["depends_on"] == [], "resident_identity: depends_on must stay empty (the module takes a host)"


def test_every_public_name_is_unique_present_and_callable():
    names = resident_identity.__all__
    assert len(names) == len(set(names)), "resident_identity: __all__ has a duplicate"
    for name in resident_identity.__all__:
        assert hasattr(resident_identity, name), f"resident_identity has no attribute {name!r}"
        assert callable(getattr(resident_identity, name)), f"resident_identity.{name} is not callable"


def _params(fn):
    return list(inspect.signature(fn).parameters)


def test_function_signatures_match_the_card():
    assert _params(resident_identity.stash_identity) == ["host", "r", "slot", "singleton_key"]
    assert _params(resident_identity.identity_for) == ["host", "r"]
    assert _params(resident_identity.forget_identity) == ["host", "r"]
    assert _params(resident_identity.serving_resident) == ["host", "handle"]
    assert _params(resident_identity.spawn_seq_for_handle) == ["host", "handle", "model_tag"]
    for name in ("stash_identity", "identity_for", "forget_identity", "serving_resident", "spawn_seq_for_handle"):
        assert not inspect.iscoroutinefunction(getattr(resident_identity, name)), \
            f"resident_identity.{name} must stay await-free"


# -- the host contract: members touched == members declared ----------------------------------------

def _protocol_members(proto) -> set[str]:
    names = set(getattr(proto, "__annotations__", {}))
    names |= {n for n, v in vars(proto).items() if inspect.isfunction(v) and not n.startswith("__")}
    return names


DECLARED_HOST = _protocol_members(resident_identity.ResidentIdentityHost)
DECLARED_RESIDENT = _protocol_members(resident_identity.ResidentLike)
DECLARED_SLOT = _protocol_members(resident_identity.SlotLike)


def test_the_declared_members_are_the_pinned_ones():
    assert DECLARED_HOST == {"_last_request_identity_by_model", "_identity_dict", "_model_residents",
                             "_spawn_seq_for_model"}, f"resident_identity: host members changed; see {_CARD}"
    assert DECLARED_RESIDENT == {"resident_key", "model_tag", "handle", "spawn_seq"}
    assert DECLARED_SLOT == {"client_meta", "model_tag", "thread_id", "fastlane"}


class Recording:
    """Forwards every attribute read to the wrapped object and remembers the name."""

    def __init__(self, inner, touched: set) -> None:
        self.__dict__["_rec_inner"] = inner
        self.__dict__["_rec_touched"] = touched

    def __getattr__(self, name):
        self.__dict__["_rec_touched"].add(name)
        return getattr(self.__dict__["_rec_inner"], name)

    def __setattr__(self, name, value) -> None:
        self.__dict__["_rec_touched"].add(name)
        setattr(self.__dict__["_rec_inner"], name, value)


def _scenarios():
    """Run every public function every way; return the members touched on the host, a resident and a slot."""
    host_t, res_t, slot_t = set(), set(), set()
    handle = object()
    a = fakes.FakeResident(resident_key="m#1", model_tag="m", handle=handle, spawn_seq=4)
    b = fakes.FakeResident(resident_key="", model_tag="m", handle=object(), spawn_seq=5)
    s = fakes.FakeResident(resident_key=SINGLETON, model_tag="m", handle=None, spawn_seq=0)
    slot = fakes.FakeSlot()
    inner = fakes.FakeHost(residents=[a, b, s])
    host = Recording(inner, host_t)
    wrap = lambda r: Recording(r, res_t)  # noqa: E731
    inner.residents = [wrap(a), wrap(b), wrap(s)]

    for r in (a, b, s):
        resident_identity.stash_identity(host, wrap(r), Recording(slot, slot_t), SINGLETON)
        resident_identity.identity_for(host, wrap(r))
    resident_identity.forget_identity(host, wrap(a))
    resident_identity.serving_resident(host, handle)
    resident_identity.serving_resident(host, None)
    resident_identity.spawn_seq_for_handle(host, handle, "m")
    resident_identity.spawn_seq_for_handle(host, object(), "m")
    return host_t, res_t, slot_t


def test_the_module_touches_only_declared_members_and_uses_every_one():
    logging.disable(logging.CRITICAL)
    try:
        host_t, res_t, slot_t = _scenarios()
    finally:
        logging.disable(logging.NOTSET)
    for kind, touched, declared in (("host", host_t, DECLARED_HOST), ("resident", res_t, DECLARED_RESIDENT),
                                    ("slot", slot_t, DECLARED_SLOT)):
        extra, unused = sorted(touched - declared), sorted(declared - touched)
        assert not extra, f"resident_identity uses undeclared {kind} members {extra}; see {_CARD}"
        assert not unused, f"resident_identity: declared {kind} members never used: {unused}; see {_CARD}"


def test_the_recording_host_can_catch_an_undeclared_member():
    touched: set = set()
    rec = Recording(fakes.FakeHost(), touched)
    _ = rec.calls
    assert "calls" in touched and not touched <= DECLARED_HOST, "the recorder must notice an undeclared member"


def test_the_host_protocol_has_no_any():
    tree = ast.parse((_PKG / "identity_host.py").read_text())
    names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    names += [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]
    assert "Any" not in names, "resident_identity/identity_host.py: no Any in the host contract"
    host_file = Path(inspect.getfile(resident_identity.ResidentIdentityHost)).resolve()
    assert host_file == _PKG / "identity_host.py", "the host contract must live in resident_identity/identity_host.py"


# -- imports: nothing from the rest of the project -----------------------------------------------

def _project_imports(source: str) -> list[str]:
    """Names of project imports in ``source``: absolute turbohaul imports, relative imports that
    climb above the package, and dynamic imports."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] == "turbohaul"]
        elif isinstance(node, ast.ImportFrom):
            if node.level >= 2:
                bad.append("." * node.level + (node.module or ""))
            elif node.level == 0 and (node.module or "").split(".")[0] == "turbohaul":
                bad.append(node.module)
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in {"__import__", "import_module"}:
                bad.append(name + "()")
    return bad


def test_the_import_check_can_fail():
    for src in ("import turbohaul", "import turbohaul.config as c", "from turbohaul import config",
                "from turbohaul.manager import M", "from .. import manager", "from ..queue import Q",
                "from ... import x", "__import__('turbohaul')", "import importlib\nimportlib.import_module('x')"):
        assert _project_imports(src), f"the import check missed: {src!r}"
    for src in ("import logging", "from . import identity_host", "from .identity_host import ResidentIdentityHost",
                "from typing import Protocol"):
        assert not _project_imports(src), f"the import check flagged a legal import: {src!r}"


def test_no_module_file_imports_the_rest_of_the_project():
    for p in _py_files():
        assert _project_imports(p.read_text()) == [], f"resident_identity/{p.name} imports the project; see {_CARD}"


# -- size and layout ---------------------------------------------------------------------------------

def test_the_package_is_exactly_the_stated_files():
    assert [p.name for p in _py_files()] == EXPECTED_FILES


def test_every_file_is_under_300_lines_and_ascii():
    for p in _py_files():
        assert len(p.read_text().splitlines()) < 300, f"resident_identity/{p.name} is not under 300 lines"
    for p in _py_files() + [_PKG / "AGENTS.md", _PKG / "README.md"]:
        assert p.read_text().isascii(), f"resident_identity/{p.name} must be ASCII"


# -- the card and the README -------------------------------------------------------------------------

def test_card_has_the_title_and_the_eight_sections_in_order():
    lines = _card().split("\n")
    assert lines[0] == "# Module: resident_identity"
    assert [ln[3:] for ln in lines if ln.startswith("## ")] == CARD_HEADINGS, f"{_CARD}: sections out of order"


def test_card_public_interface_names_every_public_name():
    iface = _card().split("\n## Public Interface\n")[1].split("\n## Depends On\n")[0]
    for name in resident_identity.__all__:
        assert f"`{name}" in iface, f"resident_identity: Public Interface of {_CARD} does not list {name}"


def test_card_names_every_declared_host_member_under_depends_on():
    deps = _card().split("\n## Depends On\n")[1].split("\n## Invariants\n")[0]
    for name in sorted(DECLARED_HOST | DECLARED_RESIDENT | DECLARED_SLOT):
        assert name in deps, f"resident_identity: Depends On of {_CARD} does not document member {name}"


def test_card_depends_on_nothing_inside_the_project_and_readme_exists():
    deps = _card().split("\n## Depends On\n")[1].split("\n## Invariants\n")[0]
    assert "Nothing inside TurboHaul" in deps
    assert (_PKG / "README.md").read_text().strip(), "resident_identity: README.md is missing or empty"


def test_the_readme_names_every_source_file():
    readme = (_PKG / "README.md").read_text()
    for name in EXPECTED_FILES:
        assert f"`{name}`" in readme, f"resident_identity: README.md does not name {name}"
