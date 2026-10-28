"""Resident identity: equivalence tests and an ownership test.

Part one pins what the manager does with the per-resident request identity and the handle lookups
(stash on admission, skipped keys, a swallowed failure, the status read, forgetting on eviction,
the serving-resident lookup, the spawn_seq lookup, the finished-request outcome line). Every test
drives the real manager through methods it already has. The expected values are literals.

Part two reads src/turbohaul/manager.py with ``ast`` and asserts the manager keeps call sites only:
the three methods are one call into turbohaul.resident_identity, and the dict is neither written
nor pruned from the manager. Those tests check that the logic lives in the module and the manager holds call sites only.
"""

from __future__ import annotations

import ast
from pathlib import Path

import _resident_identity_harness as H
import pytest
import pytest_asyncio

MODEL = H.MODEL
SECOND = H.SECOND
SINGLETON_KEY = "__phase0_singleton__"
MODULE = "turbohaul.resident_identity"
MANAGER_PATH = Path(__file__).resolve().parents[1] / "src" / "turbohaul" / "manager.py"

# ---- literals captured from the manager before the logic moved --------------------------
S1_STEPS = [{'duo': {'fastlane_rule': None,
          'ip': '10.0.0.5',
          'is_compression': None,
          'is_curator': None,
          'is_main': True,
          'is_sub_agent': None,
          'label': None,
          'model_tag': 'duo',
          'req_id': None,
          'resolved_class': 'main',
          'session_id': 's-a',
          'thread_id': 't-a'}},
 {'duo': {'fastlane_rule': None,
          'ip': '10.0.0.5',
          'is_compression': None,
          'is_curator': None,
          'is_main': True,
          'is_sub_agent': None,
          'label': None,
          'model_tag': 'duo',
          'req_id': None,
          'resolved_class': 'main',
          'session_id': 's-a',
          'thread_id': 't-a'},
  'duo#1': {'fastlane_rule': 'edge-gw',
            'ip': '10.0.0.6',
            'is_compression': None,
            'is_curator': None,
            'is_main': False,
            'is_sub_agent': True,
            'label': 'Gateway',
            'model_tag': 'duo',
            'req_id': 'req-b',
            'resolved_class': 'sub-agent',
            'session_id': 's-b',
            'thread_id': 't-b'}},
 {'duo': {'fastlane_rule': None,
          'ip': '10.0.0.5',
          'is_compression': None,
          'is_curator': None,
          'is_main': True,
          'is_sub_agent': None,
          'label': None,
          'model_tag': 'duo',
          'req_id': None,
          'resolved_class': 'main',
          'session_id': 's-a2',
          'thread_id': 't-a'},
  'duo#1': {'fastlane_rule': 'edge-gw',
            'ip': '10.0.0.6',
            'is_compression': None,
            'is_curator': None,
            'is_main': False,
            'is_sub_agent': True,
            'label': 'Gateway',
            'model_tag': 'duo',
            'req_id': 'req-b',
            'resolved_class': 'sub-agent',
            'session_id': 's-b',
            'thread_id': 't-b'}}]

S1_FRESH_OLD_DICT_AFTER = {'duo': {'fastlane_rule': None,
         'ip': '10.0.0.5',
         'is_compression': None,
         'is_curator': None,
         'is_main': True,
         'is_sub_agent': None,
         'label': None,
         'model_tag': 'duo',
         'req_id': None,
         'resolved_class': 'main',
         'session_id': 's-a',
         'thread_id': 't-a'}}

S2_SKIPPED = {'empty/blank': [None, {}],
 'empty/seeded': [None, {'duo': {'session_id': 's-keep'}}],
 'singleton/blank': [None, {}],
 'singleton/seeded': [None, {'duo': {'session_id': 's-keep'}}]}

S3_SWALLOWED = {'client_meta': {'a': {'model_tag': 'duo', 'session_id': 's-keep'}},
 'fastlane_label': {'a': {'model_tag': 'duo', 'session_id': 's-keep'}},
 'model_tag': {'a': {'model_tag': 'duo', 'session_id': 's-keep'}},
 'resident_key': {'a': {'model_tag': 'duo', 'session_id': 's-keep'}},
 'thread_id': {'a': {'model_tag': 'duo', 'session_id': 's-keep'}}}

S4_PIDS = {'duo': 41001, 'duo#1': 41002, 'lone': 41011, 'solo': 41010}

S4_BY_PID = {'duo': {'fastlane_rule': None,
         'ip': '10.0.0.5',
         'is_compression': None,
         'is_curator': None,
         'is_main': True,
         'is_sub_agent': None,
         'label': None,
         'model_tag': 'duo',
         'req_id': None,
         'resolved_class': 'main',
         'session_id': 's-a',
         'thread_id': 't-a'},
 'duo#1': {'fastlane_rule': 'edge-gw',
           'ip': '10.0.0.6',
           'is_compression': None,
           'is_curator': None,
           'is_main': False,
           'is_sub_agent': True,
           'label': None,
           'model_tag': 'duo',
           'req_id': 'req-b',
           'resolved_class': 'sub-agent',
           'session_id': 's-b',
           'thread_id': 't-b'},
 'lone': None,
 'solo': {'fastlane_rule': None,
          'ip': '10.0.0.9',
          'label': None,
          'model_tag': 'solo',
          'session_id': 's-solo'}}

S5_UNLOAD = {'duo': [[['duo', None]],
         {'duo#1': {'fastlane_rule': 'edge-gw',
                    'ip': '10.0.0.6',
                    'is_compression': None,
                    'is_curator': None,
                    'is_main': False,
                    'is_sub_agent': True,
                    'label': 'Gateway',
                    'model_tag': 'duo',
                    'req_id': 'req-b',
                    'resolved_class': 'sub-agent',
                    'session_id': 's-b',
                    'thread_id': 't-b'}}],
 'duo#1': [[['duo#1', None]],
           {'duo': {'fastlane_rule': None,
                    'ip': '10.0.0.5',
                    'is_compression': None,
                    'is_curator': None,
                    'is_main': True,
                    'is_sub_agent': None,
                    'label': None,
                    'model_tag': 'duo',
                    'req_id': None,
                    'resolved_class': 'main',
                    'session_id': 's-a',
                    'thread_id': 't-a'}}]}

S5_DRIVER = {'after_admissions': {'duo': {'fastlane_rule': None,
                              'ip': '10.0.0.5',
                              'is_compression': None,
                              'is_curator': None,
                              'is_main': True,
                              'is_sub_agent': None,
                              'label': None,
                              'model_tag': 'duo',
                              'req_id': None,
                              'resolved_class': 'main',
                              'session_id': 's-a',
                              'thread_id': 't-a'},
                      'duo#1': {'fastlane_rule': None,
                                'ip': '10.0.0.6',
                                'is_compression': None,
                                'is_curator': None,
                                'is_main': True,
                                'is_sub_agent': None,
                                'label': None,
                                'model_tag': 'duo',
                                'req_id': None,
                                'resolved_class': 'main',
                                'session_id': 's-b',
                                'thread_id': 't-b'}},
 'after_finally': {'duo': {'fastlane_rule': None,
                           'ip': '10.0.0.5',
                           'is_compression': None,
                           'is_curator': None,
                           'is_main': True,
                           'is_sub_agent': None,
                           'label': None,
                           'model_tag': 'duo',
                           'req_id': None,
                           'resolved_class': 'main',
                           'session_id': 's-a',
                           'thread_id': 't-a'}},
 'pops': [['duo#1', None]],
 'unload_calls_on_live_resident': 0}

S6_SERVING = {'dead': None,
 'none': None,
 'own_duo': 'duo',
 'own_duo_1': 'duo#1',
 'shared': 'other',
 'singleton': None,
 'stray': None}

S7_SPAWN_SEQ = {'dead': {'duo': 1, 'other': 3, 'zzz': 9},
 'none': {'duo': 1, 'other': 3, 'zzz': 9},
 'own_duo': {'duo': 1, 'other': 1, 'zzz': 1},
 'own_duo_1': {'duo': 2, 'other': 2, 'zzz': 2},
 'shared': {'duo': 3, 'other': 3, 'zzz': 3},
 'singleton': {'duo': 1, 'other': 3, 'zzz': 9},
 'stray': {'duo': 1, 'other': 3, 'zzz': 9}}

S7_MODEL_FALLBACK = {'duo': 1, 'other': 3, 'zzz': 9}

S8_OUTCOME = {'sibling': 'engine_reused_tokens=77777',
 'sibling_only_model_block': 'engine_reused_tokens=unavailable reason=no_matching_sample',
 'unknown_handle': 'engine_reused_tokens=11111'}
# ---- end of captured literals -------------------------------------------------------------


@pytest_asyncio.fixture
async def identity_world(tmp_path):
    world = H.IdentityWorld(tmp_path)
    try:
        yield world
    finally:
        await world.mgr.shutdown()


@pytest_asyncio.fixture
async def lookup_world(tmp_path):
    world = H.LookupWorld(tmp_path)
    try:
        yield world
    finally:
        await world.mgr.shutdown()


@pytest_asyncio.fixture
async def status_world(tmp_path):
    world = H.build_status_world(tmp_path)
    try:
        yield world
    finally:
        await world.mgr.shutdown()


# ---- S1: stash on admission ---------------------------------------------------------------

class TestStashUnderTheResidentsOwnKey:
    async def test_each_bind_writes_only_its_own_key(self, identity_world):
        steps, final = H.run_stash_sequence(identity_world)
        assert steps == S1_STEPS, f"dict after each bind: {steps!r}"
        assert final == S1_STEPS[-1], f"dict at the end: {final!r}"

    async def test_the_dict_is_read_from_the_manager_at_call_time(self, identity_world):
        """After one bind, a new dict replaces the old one. The later binds go to the new dict and the
        old one keeps only the first."""
        steps, old = H.run_stash_sequence(identity_world, fresh_dict=True)
        assert steps == S1_STEPS, f"writes landed elsewhere: {steps!r}"
        assert old == S1_FRESH_OLD_DICT_AFTER, f"the replaced dict was written to: {old!r}"


# ---- S2: keys that are never stored --------------------------------------------------------

@pytest.mark.parametrize("case", sorted(S2_SKIPPED))
async def test_empty_and_singleton_keys_are_not_stored(tmp_path, case):
    key = "" if case.startswith("empty") else SINGLETON_KEY
    preseed = {} if case.endswith("blank") else {"duo": {"session_id": "s-keep"}}
    mgr = H.new_manager(tmp_path)
    try:
        returned, after = H.run_skipped_key(mgr, key, preseed)
    finally:
        await mgr.shutdown()
    assert returned is None
    assert after == S2_SKIPPED[case][1], f"{case}: dict is now {after!r}"


# ---- S3: a failure while reading the slot is swallowed -------------------------------------

@pytest.mark.parametrize("case", H.SWALLOW_CASES)
async def test_a_failure_while_reading_is_swallowed_and_stores_nothing(tmp_path, case):
    mgr = H.new_manager(tmp_path)
    try:
        error, returned, after = H.run_swallow(mgr, case)
    finally:
        await mgr.shutdown()
    assert error is None, f"{case}: {error} escaped the stash"
    assert returned is None
    assert after == S3_SWALLOWED[case], f"{case}: dict is now {after!r}"


# ---- S4: the status read -------------------------------------------------------------------

@pytest.mark.parametrize("who", sorted(S4_PIDS))
async def test_status_reads_the_identity_under_the_residents_own_key(status_world, who):
    by_pid = H.status_request_identity_by_pid(status_world.mgr)
    pid = S4_PIDS[who]
    assert pid in by_pid, f"no status entry for {who}: {sorted(by_pid)}"
    assert by_pid[pid] == S4_BY_PID[who], f"{who}: {by_pid[pid]!r}"


# ---- S5: forgetting on eviction ------------------------------------------------------------

@pytest.mark.parametrize("evict", [MODEL, SECOND])
async def test_unload_forgets_only_the_evicted_residents_entry(identity_world, evict):
    pops, rest = H.run_unload(identity_world, evict)
    expected_pops, expected_rest = S5_UNLOAD[evict]
    assert pops == expected_pops, f"pop calls: {pops!r}"
    assert rest == expected_rest, f"entries left: {rest!r}"


async def test_driver_exit_forgets_only_its_own_entry(tmp_path):
    out = await H.run_driver_finally(tmp_path)
    assert out["after_admissions"] == S5_DRIVER["after_admissions"], out["after_admissions"]
    assert out["unload_calls_on_live_resident"] == 0, "the unload path ran, so its pop could be the one seen"
    assert out["pops"] == S5_DRIVER["pops"], f"pop calls: {out['pops']!r}"
    assert out["after_finally"] == S5_DRIVER["after_finally"], f"entries left: {out['after_finally']!r}"


# ---- S6 and S7: handle lookups --------------------------------------------------------------

HANDLES = sorted(S6_SERVING)
TAGS = ["duo", "zzz", "other"]


@pytest.mark.parametrize("handle_name", HANDLES)
async def test_serving_resident_for_handle_compared_by_identity(lookup_world, handle_name):
    world = lookup_world
    got = world.mgr._serving_resident_for_handle(world.handles[handle_name])
    expected_key = S6_SERVING[handle_name]
    expected = None if expected_key is None else world.residents[expected_key]
    assert got is expected, f"{handle_name}: got {world.key_of(got)!r}, expected {expected_key!r}"


@pytest.mark.parametrize("tag", TAGS)
@pytest.mark.parametrize("handle_name", HANDLES)
async def test_spawn_seq_for_handle(lookup_world, handle_name, tag):
    world = lookup_world
    got = world.mgr._spawn_seq_for_handle(world.handles[handle_name], tag)
    assert got == S7_SPAWN_SEQ[handle_name][tag], f"{handle_name}/{tag}: got {got!r}"
    if S6_SERVING[handle_name] is None:
        assert got == world.mgr._spawn_seq_for_model(tag), "no serving resident: must equal the model lookup"


@pytest.mark.parametrize("tag", TAGS)
async def test_model_lookup_values_are_the_captured_ones(lookup_world, tag):
    assert lookup_world.mgr._spawn_seq_for_model(tag) == S7_MODEL_FALLBACK[tag]


# ---- S8: which block the finished-request outcome line reads ---------------------------------

@pytest.mark.parametrize("case", sorted(S8_OUTCOME))
async def test_outcome_line_reads_the_serving_residents_block(lookup_world, caplog, case):
    got = H.run_outcome(lookup_world, case, caplog)
    assert got == S8_OUTCOME[case], f"{case}: {got!r}"


# ---- ownership: the manager keeps call sites only ---------------------------------------------

THIN_METHODS = ["_stash_resident_identity", "_serving_resident_for_handle", "_spawn_seq_for_handle"]
DICT_ATTR = "_last_request_identity_by_model"
FORBIDDEN_IN_THIN = (
    ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.Try, ast.Dict, ast.If, ast.IfExp, ast.Lambda,
)
DICT_WRITERS = {"pop", "popitem", "clear", "update", "setdefault"}


def _manager_tree():
    return ast.parse(MANAGER_PATH.read_text(encoding="utf-8"))


def _manager_class(tree):
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TurbohaulManager"), None)
    assert cls is not None, "manager.py has no class TurbohaulManager"
    return cls


def _method(name, tree=None):
    found = [n for n in _manager_class(tree or _manager_tree()).body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert len(found) == 1, f"TurbohaulManager must define {name} exactly once, found {len(found)}"
    return found[0]


def _module_imports(tree):
    """Local names bound by every ``from turbohaul.resident_identity import ...`` in manager.py."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == MODULE and node.level == 0:
            names.extend(a.asname or a.name for a in node.names)
    return names


def _is_dict_attr(node):
    return isinstance(node, ast.Attribute) and node.attr == DICT_ATTR


@pytest.mark.parametrize("name", THIN_METHODS)
def test_o1_the_method_is_one_call_into_the_module(name):
    method = _method(name)
    imported = set(_module_imports(_manager_tree()))
    body = list(method.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    assert len(body) == 1, (
        f"{name} must be a docstring plus ONE statement that calls into {MODULE}; "
        f"it has {len(body)} statements, so the logic is still in manager.py"
    )
    stmt = body[0]
    call = stmt.value if isinstance(stmt, (ast.Return, ast.Expr)) else None
    assert isinstance(call, ast.Call), (
        f"{name} must be one `return <call>` or one call statement into {MODULE}, found {type(stmt).__name__}"
    )
    callee = call.func
    assert isinstance(callee, ast.Name) and callee.id in imported, (
        f"{name} must call a name imported from {MODULE} (imported: {sorted(imported)}), "
        f"found {ast.unparse(callee)}"
    )
    bad = sorted({type(n).__name__ for n in ast.walk(method) if isinstance(n, FORBIDDEN_IN_THIN)})
    assert not bad, f"{name} still holds logic that belongs in {MODULE}: {bad}"


def test_o2_stash_no_longer_builds_the_identity_dict_itself():
    calls = [ast.unparse(n.func) for n in ast.walk(_method("_stash_resident_identity"))
             if isinstance(n, ast.Call) and "_identity_dict" in ast.unparse(n.func)]
    assert not calls, (
        f"_stash_resident_identity must not build the dict in manager.py; the module builds it "
        f"through the host. Found calls: {calls}"
    )


def test_o2_no_handle_identity_comparison_in_the_lookup_methods():
    found = []
    for name in THIN_METHODS:
        for node in ast.walk(_method(name)):
            if isinstance(node, ast.Compare) and any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
                operands = [node.left, *node.comparators]
                if any(isinstance(o, ast.Attribute) and o.attr == "handle" for o in operands):
                    found.append(f"{name}: {ast.unparse(node)}")
    assert not found, f"the handle match must live in {MODULE}, found in manager.py: {found}"


def test_o3_the_manager_never_writes_or_prunes_the_identity_dict():
    """Allowed in manager.py: reading it (``.get``), and creating it with a plain assignment of the whole
    dict inside ``TurbohaulManager.__init__``. Not allowed anywhere: subscript store or delete on it,
    pop/popitem/clear/update/setdefault on it, or assigning the attribute outside ``__init__``."""
    tree = _manager_tree()
    init = _method("__init__", tree)
    init_nodes = {id(n) for n in ast.walk(init)}
    found = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for t in targets:
            if isinstance(t, ast.Subscript) and _is_dict_attr(t.value):
                found.append(f"line {node.lineno}: subscript write/delete {ast.unparse(node)}")
            elif _is_dict_attr(t) and id(node) not in init_nodes:
                found.append(f"line {node.lineno}: assigns the attribute outside __init__")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in DICT_WRITERS and _is_dict_attr(node.func.value):
            found.append(f"line {node.lineno}: .{node.func.attr}(...) on the dict")
    assert not found, f"writes and prunes of {DICT_ATTR} belong in {MODULE}: {found}"


def test_o4_the_manager_imports_the_module_and_uses_every_name():
    tree = _manager_tree()
    imported = _module_imports(tree)
    assert imported, f"manager.py has no `from {MODULE} import ...`"
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    unused = [n for n in imported if n not in used]
    assert not unused, f"imported from {MODULE} but never used in manager.py: {unused}"
