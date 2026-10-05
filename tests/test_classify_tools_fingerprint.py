"""Frozen-prefix re-prefill guard: pure decision-table tier
for `_classify_tools_fingerprint` and its two helpers `_tools_fingerprint` /
`_tools_names`.

Two tiers, same discipline as every other guard in this suite
(a pure decision-table tier plus a wiring tier):
  * this file — the pure functions in isolation. Proves the table is right;
    proves nothing about whether the call site derives its inputs correctly.
  * the tools-fingerprint wiring test — drives the REAL `_probe_and_save_clean_kv`
    end to end; it checks both directions: a changed tool set must fire the
    guard and an unchanged one must not (this file's own imports make it collection-fail on any
    revert of manager.py, since the functions are brand new — vacuous by
    design, the wiring test carries the end-to-end weight).
"""
import pytest

from turbohaul.manager import (
    _classify_tools_fingerprint,
    _tools_fingerprint,
    _tools_names,
)


def _tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def test_tools_fingerprint_none_for_missing_or_empty():
    assert _tools_fingerprint(None) is None
    assert _tools_fingerprint([]) is None


def test_tools_fingerprint_stable_under_key_reordering():
    """The SAME tool, serialized with a different key order inside its schema,
    must hash identically -- sort_keys, not a positional/order-sensitive dump."""
    a = [{"type": "function", "function": {"name": "x", "parameters": {"a": 1, "b": 2}}}]
    b = [{"function": {"parameters": {"b": 2, "a": 1}, "name": "x"}, "type": "function"}]
    assert _tools_fingerprint(a) == _tools_fingerprint(b)


def test_tools_fingerprint_differs_on_real_content_change():
    a = [_tool("x")]
    b = [_tool("x"), _tool("y")]
    assert _tools_fingerprint(a) != _tools_fingerprint(b)


def test_tools_fingerprint_never_raises_on_unserializable():
    class Weird:
        def __repr__(self):
            return "weird!"
    assert _tools_fingerprint([{"type": "function", "function": {"name": "x"}, "junk": Weird()}])


def test_tools_names_openai_shape():
    assert _tools_names([_tool("alpha"), _tool("beta")]) == frozenset({"alpha", "beta"})


def test_tools_names_bare_shape():
    assert _tools_names([{"name": "gamma"}]) == frozenset({"gamma"})


def test_tools_names_missing_name_falls_back_to_unknown_marker():
    assert _tools_names([{"type": "function", "function": {}}]) == frozenset({"?"})
    assert _tools_names(["not-a-dict"]) == frozenset({"?"})


def test_classify_first_sight_never_changed():
    h = _tools_fingerprint([_tool("x")])
    changed, delta = _classify_tools_fingerprint(None, h, _tools_names([_tool("x")]))
    assert changed is False
    assert delta == {"prev_count": 0, "cur_count": 1, "entered": [], "left": []}


def test_classify_same_hash_never_changed():
    tools = [_tool("x"), _tool("y")]
    h = _tools_fingerprint(tools)
    names = _tools_names(tools)
    prev = {"hash": h, "names": names, "count": len(names)}
    changed, delta = _classify_tools_fingerprint(prev, h, names)
    assert changed is False
    assert delta["entered"] == [] and delta["left"] == []


def test_classify_added_tools_reports_entered():
    base = [_tool("a"), _tool("b")]
    grown = base + [_tool("c"), _tool("d")]
    prev = {"hash": _tools_fingerprint(base), "names": _tools_names(base), "count": 2}
    changed, delta = _classify_tools_fingerprint(
        prev, _tools_fingerprint(grown), _tools_names(grown))
    assert changed is True
    assert delta == {"prev_count": 2, "cur_count": 4, "entered": ["c", "d"], "left": []}


def test_classify_removed_tools_reports_left():
    grown = [_tool("a"), _tool("b"), _tool("c")]
    base = [_tool("a")]
    prev = {"hash": _tools_fingerprint(grown), "names": _tools_names(grown), "count": 3}
    changed, delta = _classify_tools_fingerprint(
        prev, _tools_fingerprint(base), _tools_names(base))
    assert changed is True
    assert delta == {"prev_count": 3, "cur_count": 1, "entered": [], "left": ["b", "c"]}


def test_classify_tools_to_no_tools_all_left_none_entered():
    tools = [_tool("a"), _tool("b")]
    prev = {"hash": _tools_fingerprint(tools), "names": _tools_names(tools), "count": 2}
    changed, delta = _classify_tools_fingerprint(prev, None, _tools_names(None))
    assert changed is True
    assert delta["entered"] == []
    assert sorted(delta["left"]) == ["a", "b"]


def test_classify_no_tools_to_tools_all_entered_none_left():
    prev = {"hash": None, "names": frozenset(), "count": 0}
    tools = [_tool("a")]
    changed, delta = _classify_tools_fingerprint(
        prev, _tools_fingerprint(tools), _tools_names(tools))
    assert changed is True
    assert delta["entered"] == ["a"]
    assert delta["left"] == []
