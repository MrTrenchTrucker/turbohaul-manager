"""Contract test for the queue_liveness module: the registry's public list equals the real exports, the
card agrees with them, the module touches exactly the host members it declares, imports nothing from
the rest of the project, and every file is under 300 lines."""
import ast
import asyncio
import inspect
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import turbohaul.queue_liveness as queue_liveness

_ROOT = Path(__file__).resolve().parents[2]
_PKG = Path(queue_liveness.__file__).resolve().parent
_CARD = "src/turbohaul/queue_liveness/AGENTS.md"

PUBLIC = ["QueueLivenessHost", "ResidentLike", "SlotLike", "client_gone", "stamp_if_disconnected", "drop_if_gone",
          "evict_unstarted", "notify_client_gone"]
FUNCTIONS = ["client_gone", "stamp_if_disconnected", "drop_if_gone", "evict_unstarted", "notify_client_gone"]
EXPECTED_FILES = ["__init__.py", "eviction.py", "liveness.py", "liveness_host.py"]
CARD_HEADINGS = ["Purpose", "Owns", "Does Not Own", "Public Interface", "Depends On", "Invariants",
                 "Test Locations", "Known Gotchas"]


def _registry():
    with open(_ROOT / "modules.toml", "rb") as f:
        reg = tomllib.load(f)["modules"].get("queue_liveness")
    assert reg is not None, f"queue_liveness is not registered in modules.toml; see {_CARD}"
    return reg


def _py_files():
    return sorted(_PKG.glob("*.py"))


def _card() -> str:
    return (_PKG / "AGENTS.md").read_text()


# -- the public surface ---------------------------------------------------------------------------

def test_all_is_the_pinned_public_list_in_order():
    assert queue_liveness.__all__ == PUBLIC, f"queue_liveness: __all__ differs from the pinned list; see {_CARD}"


def test_registry_public_list_equals_all_in_order():
    assert _registry()["public"] == queue_liveness.__all__, (
        f"queue_liveness: modules.toml public list and __all__ differ; see {_CARD}"
    )


def test_registry_entry_fields():
    reg = _registry()
    assert reg["path"] == "src/turbohaul/queue_liveness"
    assert reg["card"] == _CARD
    assert reg["depends_on"] == [], "queue_liveness: depends_on must stay empty (the module takes a host)"


def test_every_public_name_is_unique_and_present():
    names = queue_liveness.__all__
    assert len(names) == len(set(names)), "queue_liveness: __all__ has a duplicate"
    for name in names:
        assert hasattr(queue_liveness, name), f"queue_liveness has no attribute {name!r}"
    for name in FUNCTIONS:
        assert callable(getattr(queue_liveness, name)), f"queue_liveness.{name} is not callable"


def _params(fn):
    return list(inspect.signature(fn).parameters)


def test_function_signatures_match_the_card():
    assert _params(queue_liveness.client_gone) == ["slot"]
    assert _params(queue_liveness.stamp_if_disconnected) == ["slot"]
    assert _params(queue_liveness.drop_if_gone) == ["host", "slot"]
    assert _params(queue_liveness.evict_unstarted) == ["host", "slot"]
    assert _params(queue_liveness.notify_client_gone) == ["host", "event"]


def test_only_evict_unstarted_is_a_coroutine_function():
    for name in FUNCTIONS:
        is_async = inspect.iscoroutinefunction(getattr(queue_liveness, name))
        assert is_async == (name == "evict_unstarted"), (
            f"queue_liveness.{name}: await-free functions must stay await-free (and only evict_unstarted awaits)"
        )


# -- the host contract: members touched == members declared ----------------------------------------

def _protocol_members(proto) -> set[str]:
    names = set(getattr(proto, "__annotations__", {}))
    names |= {n for n, v in vars(proto).items() if inspect.isfunction(v) and not n.startswith("__")}
    return names


DECLARED_HOST = _protocol_members(queue_liveness.QueueLivenessHost)
DECLARED_RESIDENT = _protocol_members(queue_liveness.ResidentLike)
DECLARED_SLOT = _protocol_members(queue_liveness.SlotLike)
DECLARED_QUEUE = _protocol_members(typing.get_type_hints(queue_liveness.QueueLivenessHost)["queue"])


def test_the_declared_members_are_the_pinned_ones():
    assert DECLARED_HOST == {"queue", "_registry_lock", "_model_residents", "_release_fastlane_claim_locked",
                             "_handle_unloaded_slot", "_spawn_bg"}, f"queue_liveness: host members changed; see {_CARD}"
    assert DECLARED_RESIDENT == {"state", "active_slot", "inbox"}
    assert DECLARED_SLOT == {"slot_id", "state", "disconnect_event", "is_evicted", "completion_future"}
    assert DECLARED_QUEUE == {"waiting_snapshot", "remove"}


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


@dataclass(eq=False)
class _Slot:
    slot_id: str
    state: object
    disconnect_event: object
    is_evicted: bool = False
    completion_future: object = None


@dataclass(eq=False)
class _Resident:
    state: object
    active_slot: object
    inbox: object = field(default_factory=asyncio.Queue)


class _Queue:
    def __init__(self) -> None:
        self.slots: list = []

    def waiting_snapshot(self):
        return list(self.slots)

    async def remove(self, slot_id):
        self.slots = [s for s in self.slots if s.slot_id != slot_id]


class _Host:
    def __init__(self) -> None:
        self.queue = _Queue()
        self._registry_lock = asyncio.Lock()
        self.residents: list = []
        self.spawned: list = []
        self.released: list = []
        self.failed: list = []

    def _model_residents(self):
        return tuple(self.residents)

    def _release_fastlane_claim_locked(self, slot, reason) -> None:
        self.released.append((slot.slot_id, reason))

    def _handle_unloaded_slot(self, slot) -> None:
        self.failed.append(slot.slot_id)
        slot.completion_future.set_result(None)

    def _spawn_bg(self, coro):
        task = asyncio.get_running_loop().create_task(coro)
        self.spawned.append(task)
        return task


async def _scenarios():
    """Run every public function every way; return the members touched on the host, queue, a resident and a slot."""
    host_t, queue_t, res_t, slot_t = set(), set(), set(), set()
    loop = asyncio.get_running_loop()
    inner = _Host()
    raw_queue = inner.queue
    host = Recording(inner, host_t)
    inner.queue = Recording(raw_queue, queue_t)

    def new_slot(slot_id, state="STAGED"):
        s = _Slot(slot_id, SimpleNamespace(name=state), asyncio.Event(), completion_future=loop.create_future())
        return Recording(s, slot_t)

    busy = Recording(_Resident(SimpleNamespace(name="ACTIVE"), SimpleNamespace()), res_t)
    parked = Recording(_Resident(SimpleNamespace(name="IDLE_EVICTABLE"), None), res_t)
    inner.residents = [busy, parked]

    live = new_slot("live")
    queue_liveness.client_gone(live)
    queue_liveness.stamp_if_disconnected(live)
    queue_liveness.drop_if_gone(host, live)

    gone = new_slot("gone")
    gone.disconnect_event.set()
    queue_liveness.drop_if_gone(host, gone)

    queued, in_busy, in_parked, started = (new_slot("q"), new_slot("b"), new_slot("p"), new_slot("x", "ACTIVE"))
    raw_queue.slots.append(queued)
    busy.inbox.put_nowait(in_busy)
    parked.inbox.put_nowait(in_parked)
    for s in (queued, in_busy, in_parked, started):
        await queue_liveness.evict_unstarted(host, s)

    notified = new_slot("n")
    raw_queue.slots.append(notified)
    queue_liveness.notify_client_gone(host, notified.disconnect_event)
    await asyncio.gather(*inner.spawned)
    return host_t, queue_t, res_t, slot_t


async def test_the_module_touches_only_declared_members_and_uses_every_one():
    host_t, queue_t, res_t, slot_t = await _scenarios()
    for kind, touched, declared in (("host", host_t, DECLARED_HOST), ("queue", queue_t, DECLARED_QUEUE),
                                    ("resident", res_t, DECLARED_RESIDENT), ("slot", slot_t, DECLARED_SLOT)):
        extra, unused = sorted(touched - declared), sorted(declared - touched)
        assert not extra, f"queue_liveness uses undeclared {kind} members {extra}; see {_CARD}"
        assert not unused, f"queue_liveness: declared {kind} members never used: {unused}; see {_CARD}"


def test_the_recording_host_can_catch_an_undeclared_member():
    touched: set = set()
    rec = Recording(_Host(), touched)
    _ = rec.failed
    assert "failed" in touched and not touched <= DECLARED_HOST, "the recorder must notice an undeclared member"


def test_the_host_protocol_has_no_any():
    tree = ast.parse((_PKG / "liveness_host.py").read_text())
    names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    names += [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]
    assert "Any" not in names, "queue_liveness/liveness_host.py: no Any in the host contract"
    host_file = Path(inspect.getfile(queue_liveness.QueueLivenessHost)).resolve()
    assert host_file == _PKG / "liveness_host.py", "the host contract must live in queue_liveness/liveness_host.py"


# -- behaviour at the interface ---------------------------------------------------------------------

async def test_the_interface_end_to_end_on_a_recording_free_host():
    host = _Host()
    loop = asyncio.get_running_loop()
    slot = _Slot("s", SimpleNamespace(name="STAGED"), asyncio.Event(), completion_future=loop.create_future())
    host.queue.slots.append(slot)
    assert queue_liveness.client_gone(slot) is False
    queue_liveness.notify_client_gone(host, slot.disconnect_event)
    assert queue_liveness.client_gone(slot) is True and slot.is_evicted is False, "the event alone marks the client as gone"
    await asyncio.gather(*host.spawned)
    assert slot.is_evicted is True and host.queue.slots == [] and host.failed == ["s"]
    assert host.released == [("s", "disconnected")]
    assert await queue_liveness.evict_unstarted(host, slot) is False


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
    for src in ("import logging", "from . import liveness_host", "from .liveness_host import QueueLivenessHost",
                "from typing import Protocol"):
        assert not _project_imports(src), f"the import check flagged a legal import: {src!r}"


def test_no_module_file_imports_the_rest_of_the_project():
    for p in _py_files():
        assert _project_imports(p.read_text()) == [], f"queue_liveness/{p.name} imports the project; see {_CARD}"


# -- size and layout ---------------------------------------------------------------------------------

def test_the_package_is_exactly_the_stated_files():
    assert [p.name for p in _py_files()] == EXPECTED_FILES


def test_every_file_is_under_300_lines_and_ascii():
    for p in _py_files():
        assert len(p.read_text().splitlines()) < 300, f"queue_liveness/{p.name} is not under 300 lines"
    for p in _py_files() + [_PKG / "AGENTS.md", _PKG / "README.md"]:
        assert p.read_text().isascii(), f"queue_liveness/{p.name} must be ASCII"


# -- the card and the README -------------------------------------------------------------------------

def test_card_has_the_title_and_the_eight_sections_in_order():
    lines = _card().split("\n")
    assert lines[0] == "# Module: queue_liveness"
    assert [ln[3:] for ln in lines if ln.startswith("## ")] == CARD_HEADINGS, f"{_CARD}: sections out of order"


def test_card_public_interface_names_every_public_name():
    iface = _card().split("\n## Public Interface\n")[1].split("\n## Depends On\n")[0]
    for name in queue_liveness.__all__:
        assert f"`{name}" in iface, f"queue_liveness: Public Interface of {_CARD} does not list {name}"


def test_card_names_every_declared_host_member_under_depends_on():
    deps = _card().split("\n## Depends On\n")[1].split("\n## Invariants\n")[0]
    for name in sorted(DECLARED_HOST | DECLARED_RESIDENT | DECLARED_SLOT | DECLARED_QUEUE):
        assert name in deps, f"queue_liveness: Depends On of {_CARD} does not document member {name}"


def test_card_depends_on_nothing_inside_the_project_and_readme_exists():
    deps = _card().split("\n## Depends On\n")[1].split("\n## Invariants\n")[0]
    assert "Nothing inside TurboHaul" in deps
    assert (_PKG / "README.md").read_text().strip(), "queue_liveness: README.md is missing or empty"


def test_the_readme_names_every_source_file_and_the_inbox_condition():
    readme = (_PKG / "README.md").read_text()
    for name in EXPECTED_FILES:
        assert f"`{name}`" in readme, f"queue_liveness: README.md does not name {name}"
    assert "active_slot" in readme, "queue_liveness: README.md must state the mid-turn condition for the inbox removal"
