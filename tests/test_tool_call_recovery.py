"""spec tests for tool-call recovery post-processor.

Covers the 6 mandated vectors per RC body:
  1. OpenAI canonical text-JSON shape extracted
  2. Qwen <tool_call>...</tool_call> XML wrapper extracted
  3. Parallel calls (multiple JSONs in one response) — all picked up
  4. Idempotency: skip when tool_calls already populated
  5. Unknown name rejected (not in tools allowlist)
  6. Invalid JSON arguments rejected

Plus guard vectors:
  7. Pre-</think> JSON is NOT extracted (reasoning text guard)
  8. No tools advertised → no-op
  9. Nested-object args parsed correctly (brace-balancer)
"""
import json

import pytest

from turbohaul.api.tool_call_recovery import (
    _extract_known_names,
    _iter_candidates,
    maybe_recover_tool_calls,
)


def _tools(*names):
    return [
        {"type": "function", "function": {"name": n, "parameters": {}}}
        for n in names
    ]


def _result_with_content(content, tool_calls=None, finish_reason="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
    }


# ---------------------------------------------------------------------------
# 1. OpenAI canonical text-JSON shape
# ---------------------------------------------------------------------------

def test_canonical_text_json_extracted():
    result = _result_with_content(
        '<think>I should list the app dir.</think>\n\n'
        '{"name": "list_directory", "arguments": {"path": "/app"}}'
    )
    maybe_recover_tool_calls(result, _tools("list_directory"))
    msg = result["choices"][0]["message"]
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "list_directory"
    assert json.loads(tc["function"]["arguments"]) == {"path": "/app"}
    assert tc["id"].startswith("call_")
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    # JSON stripped from content; reasoning preserved.
    assert "list_directory" not in msg["content"]
    assert "<think>" in msg["content"]


# ---------------------------------------------------------------------------
# 2. Qwen <tool_call> XML wrapper
# ---------------------------------------------------------------------------

def test_qwen_xml_wrapper_extracted():
    result = _result_with_content(
        '<think>plan</think>\n'
        '<tool_call>{"name": "search_web", "arguments": {"q": "hello"}}</tool_call>'
    )
    maybe_recover_tool_calls(result, _tools("search_web"))
    msg = result["choices"][0]["message"]
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "search_web"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"q": "hello"}
    # Wrapper tags stripped entirely.
    assert "<tool_call>" not in msg["content"]
    assert "</tool_call>" not in msg["content"]
    assert result["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------------------
# 3. Parallel calls — all picked up (finditer, not single match)
# ---------------------------------------------------------------------------

def test_parallel_tool_calls_all_extracted():
    result = _result_with_content(
        '<think>two calls</think>\n'
        '{"name": "list_directory", "arguments": {"path": "/a"}}\n'
        '{"name": "list_directory", "arguments": {"path": "/b"}}'
    )
    maybe_recover_tool_calls(result, _tools("list_directory"))
    tcs = result["choices"][0]["message"]["tool_calls"]
    assert len(tcs) == 2
    paths = [json.loads(tc["function"]["arguments"])["path"] for tc in tcs]
    assert paths == ["/a", "/b"]
    # Synthetic IDs distinct.
    assert tcs[0]["id"] != tcs[1]["id"]


def test_parallel_mixed_xml_and_canonical():
    result = _result_with_content(
        '<think>plan</think>\n'
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n'
        '{"name": "b", "arguments": {"y": 2}}'
    )
    maybe_recover_tool_calls(result, _tools("a", "b"))
    tcs = result["choices"][0]["message"]["tool_calls"]
    names = [tc["function"]["name"] for tc in tcs]
    assert set(names) == {"a", "b"}


# ---------------------------------------------------------------------------
# 4. Idempotency — skip when tool_calls already populated
# ---------------------------------------------------------------------------

def test_idempotency_skip_when_populated():
    pre_existing = [{
        "id": "call_upstream",
        "type": "function",
        "function": {"name": "list_directory", "arguments": '{"path": "/x"}'},
    }]
    result = _result_with_content(
        # Even with text-JSON sitting in content, we must NOT add to tool_calls.
        '<think>x</think>\n{"name": "list_directory", "arguments": {"path": "/y"}}',
        tool_calls=pre_existing,
        finish_reason="tool_calls",
    )
    maybe_recover_tool_calls(result, _tools("list_directory"))
    msg = result["choices"][0]["message"]
    assert msg["tool_calls"] == pre_existing
    # Content untouched.
    assert '"path": "/y"' in msg["content"]
    assert result["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------------------
# 5. Unknown name rejected
# ---------------------------------------------------------------------------

def test_unknown_name_rejected():
    result = _result_with_content(
        '<think>x</think>\n{"name": "rm_rf", "arguments": {"path": "/"}}'
    )
    maybe_recover_tool_calls(result, _tools("list_directory"))
    msg = result["choices"][0]["message"]
    assert "tool_calls" not in msg
    # Content untouched.
    assert "rm_rf" in msg["content"]
    assert result["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# 6. Invalid JSON arguments rejected
# ---------------------------------------------------------------------------

def test_invalid_json_args_rejected():
    # Arguments object is malformed (trailing comma + missing close).
    result = _result_with_content(
        '<think>x</think>\n{"name": "list_directory", "arguments": {bogus}'
    )
    maybe_recover_tool_calls(result, _tools("list_directory"))
    msg = result["choices"][0]["message"]
    assert "tool_calls" not in msg


# ---------------------------------------------------------------------------
# 7. Pre-</think> guard
# ---------------------------------------------------------------------------

def test_pre_think_close_not_extracted():
    # The text-JSON appears INSIDE the <think> block — must be ignored.
    result = _result_with_content(
        '<think>I would call '
        '{"name": "list_directory", "arguments": {"path": "/app"}} '
        'if I needed it.</think>\n\nNo, I will just answer directly.'
    )
    maybe_recover_tool_calls(result, _tools("list_directory"))
    msg = result["choices"][0]["message"]
    assert "tool_calls" not in msg
    assert result["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# 8. No tools advertised → no-op
# ---------------------------------------------------------------------------

def test_no_tools_advertised_noop():
    result = _result_with_content(
        '<think>x</think>\n{"name": "list_directory", "arguments": {"path": "/a"}}'
    )
    maybe_recover_tool_calls(result, None)
    assert "tool_calls" not in result["choices"][0]["message"]
    maybe_recover_tool_calls(result, [])
    assert "tool_calls" not in result["choices"][0]["message"]


# ---------------------------------------------------------------------------
# 9. Nested-object arguments (brace-balancer correctness)
# ---------------------------------------------------------------------------

def test_nested_object_arguments_parsed():
    result = _result_with_content(
        '<think>x</think>\n'
        '{"name": "complex", "arguments": {"filter": {"k": "v", "n": {"a": 1}}}}'
    )
    maybe_recover_tool_calls(result, _tools("complex"))
    tc = result["choices"][0]["message"]["tool_calls"][0]
    args = json.loads(tc["function"]["arguments"])
    assert args == {"filter": {"k": "v", "n": {"a": 1}}}


# ---------------------------------------------------------------------------
# Helper sanity
# ---------------------------------------------------------------------------

def test_extract_known_names():
    assert _extract_known_names(_tools("a", "b")) == {"a", "b"}
    assert _extract_known_names(None) == set()
    assert _extract_known_names([]) == set()
    assert _extract_known_names([{"function": {"name": ""}}]) == set()
    assert _extract_known_names([{"type": "function"}]) == set()


def test_iter_candidates_empty_on_no_match():
    assert _iter_candidates("plain text, no tool calls here.") == []


# ---------------------------------------------------------------------------
# Edge: non-dict result / missing choices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, "string", 42, {}, {"choices": None}, {"choices": []}])
def test_malformed_result_noop(bad):
    # Just must not raise.
    maybe_recover_tool_calls(bad, _tools("x"))
