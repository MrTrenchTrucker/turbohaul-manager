"""Ownership of the engine load figure: the manager keeps a call site only.

The picker ``TurbohaulManager._pick_instance_for`` ranks the live engines of a model by readiness,
then by how much work each one has pending, then by how long it has been idle. How much work an
engine has pending is decided in the ``turbohaul.resident_identity`` module (see its card,
``AGENTS.md`` in that package), not in ``manager.py``. These tests read ``manager.py`` with ``ast``,
from the file of the ``turbohaul`` package that is imported, and assert three things, each in its own
test:

* the picker calls ``pending_load``;
* the picker itself reads none of the engine fields the load figure is made of (the finished slot,
  the riders in flight, the grace flag);
* ``manager.py`` imports ``pending_load`` from ``turbohaul.resident_identity``.

They check that the manager computes the figure only by calling the module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import turbohaul

MODULE = "turbohaul.resident_identity"
CARD = "turbohaul/resident_identity/AGENTS.md"
FUNCTION = "pending_load"
PICKER = "_pick_instance_for"
LOAD_FIELDS = {"active_slot", "inflight", "in_grace_loop"}
MANAGER_PATH = Path(turbohaul.__file__).resolve().parent / "manager.py"


def _where():
    return f"manager.py read from the imported package ({turbohaul.__file__})"


@pytest.fixture(scope="module")
def tree():
    return ast.parse(MANAGER_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def picker(tree):
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TurbohaulManager"), None)
    assert cls is not None, f"no class TurbohaulManager in {_where()}"
    found = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == PICKER]
    assert len(found) == 1, f"TurbohaulManager must define {PICKER} exactly once in {_where()}, found {len(found)}"
    return found[0]


def test_the_picker_calls_pending_load(picker):
    calls = []
    for node in ast.walk(picker):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == FUNCTION:
                calls.append(node.lineno)
            elif isinstance(func, ast.Attribute) and func.attr == FUNCTION:
                calls.append(node.lineno)
    assert calls, (
        f"{PICKER} never calls {FUNCTION}: the pending-work figure is not taken from {MODULE} "
        f"(card: {CARD}); {_where()}"
    )


def test_the_picker_reads_none_of_the_fields_the_load_figure_is_made_of(picker):
    found = []
    for node in ast.walk(picker):
        if isinstance(node, ast.Attribute) and node.attr in LOAD_FIELDS:
            found.append(f"line {node.lineno}: .{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in LOAD_FIELDS
        ):
            found.append(f"line {node.lineno}: {node.func.id}(..., {node.args[1].value!r})")
    assert not found, (
        f"{PICKER} still reads engine fields that belong to the load figure of {MODULE} "
        f"(card: {CARD}): {found}; {_where()}"
    )


def test_the_manager_imports_pending_load_from_the_module(tree):
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == MODULE and node.level == 0:
            imported.extend(alias.name for alias in node.names)
    assert FUNCTION in imported, (
        f"manager.py does not import {FUNCTION} from {MODULE} (card: {CARD}); "
        f"it imports from that module: {sorted(imported)}; {_where()}"
    )
