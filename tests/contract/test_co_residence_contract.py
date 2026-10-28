"""Contract test for the co_residence module: the registry's public list equals the real exports, the
card agrees with them, the module touches exactly the host members it declares, imports only
turbohaul.queue_liveness from the rest of the project, and every file is under 300 lines."""
import ast
import dataclasses
import inspect
import logging
import re
import tomllib
import typing
from pathlib import Path

import _co_residence_fakes as fakes

import turbohaul.co_residence as co_residence

_ROOT = Path(__file__).resolve().parents[2]
_PKG = Path(co_residence.__file__).resolve().parent
_CARD = "src/turbohaul/co_residence/AGENTS.md"

EXPECTED_FILES = ["__init__.py", "constants.py", "decision.py", "host.py", "lines.py", "memo.py",
                  "recheck.py", "scan.py", "state.py", "verdict.py", "window.py"]
CARD_HEADINGS = ["Purpose", "Owns", "Does Not Own", "Public Interface", "Depends On", "Invariants",
                 "Test Locations", "Known Gotchas"]
DISAMBIGUATION = (
    "Not coresident_admission: that module owns the probe-retry admission rule; co_residence owns "
    "the fit-based periodic re-check and its CO_RESIDENCE_* lines."
)


def _registry():
    with open(_ROOT / "modules.toml", "rb") as f:
        reg = tomllib.load(f)["modules"].get("co_residence")
    assert reg is not None, f"co_residence is not registered in modules.toml; see {_CARD}"
    return reg


def _py_files():
    return sorted(_PKG.glob("*.py"))


def _card() -> str:
    return (_PKG / "AGENTS.md").read_text()


# -- the public surface ---------------------------------------------------------------------------

def test_registry_public_list_equals_all_in_order():
    assert _registry()["public"] == co_residence.__all__, (
        f"co_residence: modules.toml public list and __all__ differ; see {_CARD}"
    )


def test_registry_entry_fields():
    reg = _registry()
    assert reg["path"] == "src/turbohaul/co_residence"
    assert reg["card"] == _CARD
    assert reg["depends_on"] == ["queue_liveness"], (
        "co_residence: depends_on must be exactly ['queue_liveness'] "
        "(the module takes a host, and imports only that one module)"
    )


def test_every_public_name_is_unique_present_and_the_right_kind():
    assert len(co_residence.__all__) == len(set(co_residence.__all__)), "co_residence: __all__ has a duplicate"
    typed = {"RECHECK_MIN_INTERVAL_S": float, "NO_ROOM_MEMO_S": float, "DEMAND_MIN_AGE_S": float,
             "PROTECTED_DEMAND_MIN_AGE_S": float, "MEMO_MAX_ENTRIES": int}
    for name in co_residence.__all__:
        assert hasattr(co_residence, name), f"co_residence has no attribute {name!r}"
        obj = getattr(co_residence, name)
        if name in typed:
            assert type(obj) is typed[name], f"co_residence.{name} must be a {typed[name].__name__}"
        else:
            assert callable(obj), f"co_residence.{name} is not callable"


def _params(fn):
    return list(inspect.signature(fn).parameters)


def test_dataclass_and_tuple_shapes():
    assert [f.name for f in dataclasses.fields(co_residence.RecheckGate)] == ["open", "reason"]
    assert [f.name for f in dataclasses.fields(co_residence.RecheckDecision)] == ["run", "reason"]
    assert [f.name for f in dataclasses.fields(co_residence.RecheckInputs)] == [
        "tag_live_instances", "eligible_demand", "memo_blocked"]
    for cls in (co_residence.RecheckGate, co_residence.RecheckDecision, co_residence.RecheckInputs):
        assert cls.__dataclass_params__.frozen, f"co_residence.{cls.__name__} must be frozen"
    assert co_residence.Verdict._fields == ("admit", "card", "split", "reason", "need", "eff_cap")
    assert co_residence.Candidate._fields == ("age", "rank", "source", "slot", "owner")
    assert [m.value for m in co_residence.DeclineReason] == ["cap", "box_full", "no_card", "vram_refused"]


def test_function_signatures_match_the_card():
    assert _params(co_residence.recheck_window) == ["now", "last_recheck_at", "min_interval_s"]
    assert _params(co_residence.demand_is_eligible) == ["age_s", "protected"]
    assert _params(co_residence.decide_recheck) == ["inputs"]
    assert _params(co_residence.format_step2_decline) == [
        "via", "tag", "reason", "eff_cap", "instances", "residents", "max_sidecars", "need_mib", "card", "split"]
    assert _params(co_residence.format_recheck_action) == [
        "tag", "source", "slot_id", "waited_s", "instances", "residents"]
    assert _params(co_residence.NoRoomMemo.__init__) == ["self", "ttl_s", "max_entries"]
    assert _params(co_residence.RecheckState.__init__) == ["self"]
    assert _params(co_residence.inbox_slots) == ["r"]
    assert _params(co_residence.take_from_inbox) == ["r", "slot"]
    assert _params(co_residence.gather_candidates) == ["host", "now", "grace_pairs"]
    assert _params(co_residence.pick_candidate) == ["host", "state", "cands", "now"]
    assert _params(co_residence.fit_verdict) == ["host", "state", "tag", "insts", "now"]
    assert _params(co_residence.arrival_verdict) == ["host", "tag", "insts", "eff_cap", "slot"]
    assert _params(co_residence.run_recheck) == ["host"]
    for fn in (co_residence.fit_verdict, co_residence.arrival_verdict, co_residence.run_recheck):
        assert inspect.iscoroutinefunction(fn), f"co_residence.{fn.__name__} must be async"
    assert not inspect.iscoroutinefunction(co_residence.take_from_inbox), "take_from_inbox must stay await-free"


# -- the host contract: members touched == members declared ----------------------------------------

def _protocol_members(proto) -> set[str]:
    names = set(getattr(proto, "__annotations__", {}))
    names |= {n for n, v in vars(proto).items() if inspect.isfunction(v) and not n.startswith("__")}
    return names


# The view types are private to the host contract: reach them through the public host type's hints.
_HOST_HINTS = typing.get_type_hints(co_residence.CoResidenceHost)
_QUEUE_VIEW = _HOST_HINTS["queue"]
_RUNTIME_SETTINGS = typing.get_type_hints(_HOST_HINTS["runtime"])["queue"]
DECLARED_TOP = _protocol_members(co_residence.CoResidenceHost)
DECLARED_NESTED = (
    {f"queue.{m}" for m in _protocol_members(_QUEUE_VIEW)}
    | {f"runtime.queue.{m}" for m in _protocol_members(_RUNTIME_SETTINGS)}
)


class RecordingHost:
    """Forwards every attribute read on the host to a fake and remembers the name."""

    def __init__(self, inner) -> None:
        self.__dict__["_rec_inner"] = inner
        self.__dict__["_rec_touched"] = set()

    def __getattr__(self, name):
        self.__dict__["_rec_touched"].add(name)
        return getattr(self.__dict__["_rec_inner"], name)


def _staged():
    host = fakes.FakeHost(residents=[fakes.FakeResident()], staged=[fakes.FakeSlot(created_at=970.0)], now=1000.0)
    host.cap = 3
    return host


async def _scenarios():
    """Run every way through the module; return the (top-level, nested) host members touched."""
    touched, nested = set(), set()

    async def run(host, coro_fn):
        rec = RecordingHost(host)
        await coro_fn(rec)
        touched.update(rec.__dict__["_rec_touched"])
        nested.update(c for c in host.calls if c.startswith(("queue.", "runtime.")))

    admitted = _staged()
    admitted.reserve_result = fakes.FakeResident(resident_key="k")
    await run(admitted, co_residence.run_recheck)

    declined = _staged()
    declined.vram_ok = False
    await run(declined, co_residence.run_recheck)

    r = fakes.FakeResident().put(fakes.FakeSlot(slot_id="ib", created_at=970.0))
    inbox = fakes.FakeHost(residents=[r], now=1000.0)
    await run(inbox, co_residence.run_recheck)

    no_card = _staged()
    no_card.pick_result = (None, "none")
    await run(no_card, co_residence.run_recheck)

    evicted = _staged()
    original = evicted.queue.remove

    async def remove(slot_id):
        taken = await original(slot_id)
        taken.is_evicted = True
        return taken

    evicted.queue.remove = remove
    await run(evicted, co_residence.run_recheck)

    failing = _staged()
    failing.reserve_exc = RuntimeError("boom")
    await run(failing, co_residence.run_recheck)

    arrival = fakes.FakeHost(residents=[fakes.FakeResident()], now=1000.0)

    async def call_arrival(rec):
        await co_residence.arrival_verdict(rec, "m", rec._instances_for("m"), 2, fakes.FakeSlot())

    await run(arrival, call_arrival)
    return touched, nested


async def test_the_module_touches_only_declared_host_members_and_uses_every_one():
    logging.disable(logging.CRITICAL)
    try:
        touched, nested = await _scenarios()
    finally:
        logging.disable(logging.NOTSET)
    extra = sorted(touched - DECLARED_TOP)
    assert touched <= DECLARED_TOP, f"co_residence uses undeclared host members {extra}; see {_CARD}"
    assert DECLARED_TOP <= touched, f"declared but never used: {sorted(DECLARED_TOP - touched)}; see {_CARD}"
    assert nested == DECLARED_NESTED, f"nested members used {sorted(nested)} != declared {sorted(DECLARED_NESTED)}"


def test_the_card_names_every_declared_host_member_under_depends_on():
    deps = _card().split("\n## Depends On\n")[1].split("\n## Invariants\n")[0]
    for name in sorted(DECLARED_TOP | {"staging_snapshot", "remove", "max_parallel_sidecars"}):
        assert name in deps, f"co_residence: Depends On of {_CARD} does not document host member {name}"


def test_the_host_protocol_has_no_any_and_no_bare_object_annotation():
    names = [n.id for n in ast.walk(ast.parse((_PKG / "host.py").read_text())) if isinstance(n, ast.Name)]
    names += [n.attr for n in ast.walk(ast.parse((_PKG / "host.py").read_text())) if isinstance(n, ast.Attribute)]
    assert "Any" not in names and "object" not in names, "host.py: no Any and no object reference in the host contract"


# -- imports: only the queue_liveness package entry point from the rest of the project -----------

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
    for src in ("import asyncio", "from . import state", "from .state import RecheckState", "from typing import Any"):
        assert not _project_imports(src), f"the import check flagged a legal import: {src!r}"


_ALLOWED_PROJECT_IMPORT = "turbohaul.queue_liveness"


def _imported_project_names(source: str) -> list[str]:
    """Every project name ``source`` imports, resolved to the dotted name it reaches.

    ``from turbohaul import x`` reaches ``turbohaul.x``; ``from turbohaul.a import b`` reaches
    ``turbohaul.a``; ``import turbohaul.a.b`` reaches ``turbohaul.a.b``. A relative import that
    climbs above the package and a dynamic import are kept as written, so they never match.
    """
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names if a.name.split(".")[0] == "turbohaul"]
        elif isinstance(node, ast.ImportFrom):
            if node.level >= 2:
                names.append("." * node.level + (node.module or ""))
            elif node.level == 0 and (node.module or "").split(".")[0] == "turbohaul":
                if node.module == "turbohaul":
                    names += [f"turbohaul.{a.name}" for a in node.names]
                else:
                    names.append(node.module)
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in {"__import__", "import_module"}:
                names.append(name + "()")
    return names


def test_the_only_project_import_of_a_module_file_is_the_queue_liveness_entry_point():
    for p in _py_files():
        wrong = [n for n in _imported_project_names(p.read_text()) if n != _ALLOWED_PROJECT_IMPORT]
        assert wrong == [], (
            f"co_residence/{p.name} imports {wrong} from the project; "
            f"only {_ALLOWED_PROJECT_IMPORT} is allowed; see {_CARD}"
        )


# -- size and layout ---------------------------------------------------------------------------------

def test_the_package_is_exactly_the_stated_files():
    assert [p.name for p in _py_files()] == EXPECTED_FILES


def test_every_file_is_under_300_lines_and_ascii():
    for p in _py_files():
        assert len(p.read_text().splitlines()) < 300, f"co_residence/{p.name} is not under 300 lines"
    for p in _py_files() + [_PKG / "AGENTS.md", _PKG / "README.md"]:
        assert p.read_text().isascii(), f"co_residence/{p.name} must be ASCII"


# -- the card and the README -------------------------------------------------------------------------

def test_card_has_the_title_and_the_eight_sections_in_order():
    lines = _card().split("\n")
    assert lines[0] == "# Module: co_residence"
    assert [ln[3:] for ln in lines if ln.startswith("## ")] == CARD_HEADINGS, f"{_CARD}: sections out of order"


def test_card_carries_the_disambiguation_line_whole():
    assert DISAMBIGUATION in _card().split("\n"), f"{_CARD} must carry the coresident_admission line whole"


def test_card_public_interface_names_every_public_name():
    iface = _card().split("\n## Public Interface\n")[1].split("\n## Depends On\n")[0]
    for name in co_residence.__all__:
        assert f"`{name}" in iface, f"co_residence: Public Interface of {_CARD} does not list {name}"


def test_card_depends_on_only_queue_liveness_inside_the_project_and_readme_exists():
    deps = _card().split("\n## Depends On\n")[1].split("\n## Invariants\n")[0]
    assert "`queue_liveness`" in deps, f"co_residence: Depends On of {_CARD} must name queue_liveness"
    assert "otherwise nothing inside turbohaul" in deps.lower(), (
        f"co_residence: Depends On of {_CARD} must say nothing else inside the project is used"
    )
    with open(_ROOT / "modules.toml", "rb") as f:
        others = sorted(set(tomllib.load(f)["modules"]) - {"co_residence", "queue_liveness"})
    named = [m for m in others if re.search(rf"\b{re.escape(m)}\b", deps)]
    assert named == [], f"co_residence: Depends On of {_CARD} names project modules {named} besides queue_liveness"
    assert (_PKG / "README.md").read_text().strip(), "co_residence: README.md is missing or empty"
