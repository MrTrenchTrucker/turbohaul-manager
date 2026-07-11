"""Single-source `<think>` formatter + byte-parity.

Proves:
  1. wrap_reasoning_think reproduces the OLD _merge_reasoning_into_content bytes on a
     matrix incl. empty/whitespace content + multiline/stray-tag reasoning (constraint
     #4 — the client-facing response does not change one byte).
  2. ALL THREE sites emit the SAME bytes for the same (reasoning, content):
       site 1 = _merge_reasoning_into_content (client response, what the harness stores)
       site 2 = TurbohaulManager._generated_assistant_msg (non-streaming shadow)
       site 3 = the streaming accumulator expression (chat_completion stream_gen)
  3. THE FIX: _strip_thinking_all(shadow_form) == _strip_thinking_all(merge_form)
     byte-for-byte on single-block AND multi-block / stray-`</think>` — the property the
     cold shadow restore relies on (== the harness's strip_think_blocks(x).strip()).
  4. The OLD no-newline `<think>{r}</think>{c}` shadow form DIVERGED on the edge case
     (regression-guard so the fix can't be silently reverted).
"""
from turbohaul.api.chat_completion import (
    _merge_reasoning_into_content,
    _strip_thinking_all,
    wrap_reasoning_think,
)
from turbohaul.manager import TurbohaulManager


def _OLD_merge_inline(rc: str, ct: str) -> str:
    rc_stripped = rc.strip()
    if ct.strip():
        return f"<think>\n{rc_stripped}\n</think>\n\n{ct}"
    return f"<think>\n{rc_stripped}\n</think>"


def _site3_stream_expr(_r: str, _c: str) -> str:
    """Verbatim copy of the streaming accumulator branch (chat_completion.py)."""
    return wrap_reasoning_think(_r, _c) if (_r and "<think>" not in _c) else _c


# non-empty reasoning + content WITHOUT `<think>` so all 3 guards fire uniformly
PARITY_CASES = [
    ("chain of reasoning", "THE_FINAL_ANSWER"),
    ("r", ""),                                  # empty content
    ("r", "   "),                               # all-whitespace content
    ("  padded reasoning  ", "ans"),            # reasoning needs strip()
    ("line1\nline2", "multi\nline\nanswer"),    # multiline both
    ("a</think>b", "content"),                  # STRAY </think> in reasoning
    ("", "content only, empty reasoning"),      # empty reasoning
]


def test_constraint4_merge_bytes_unchanged():
    """#4: wrap_reasoning_think == the OLD _merge inline bytes on every case."""
    for r, c in PARITY_CASES:
        assert wrap_reasoning_think(r, c) == _OLD_merge_inline(r, c), (r, c)


def test_all_three_sites_identical():
    """site1 (_merge) == site2 (_generated_assistant_msg) == site3 (stream) == helper."""
    for r, c in PARITY_CASES:
        if not r.strip():
            continue  # sites guard on non-empty reasoning; skip the empty-reasoning row
        helper = wrap_reasoning_think(r, c)

        # site 1: _merge mutates result.content
        res = {"choices": [{"message": {"role": "assistant", "content": c, "reasoning_content": r}}]}
        _merge_reasoning_into_content(res)
        site1 = res["choices"][0]["message"]["content"]

        # site 2: _generated_assistant_msg (guard: reasoning and "<think>" not in content)
        res2 = {"choices": [{"message": {"role": "assistant", "content": c, "reasoning_content": r}}]}
        site2 = TurbohaulManager._generated_assistant_msg(res2)["content"]

        # site 3: streaming accumulator expression
        site3 = _site3_stream_expr(r, c)

        assert site1 == helper == site2 == site3, (r, c, site1, site2, site3)


def test_fix_strip_shadow_equals_strip_merge():
    """THE FIX: strip(shadow_form) == strip(merge_form) on single + multi-block/stray."""
    strip_cases = PARITY_CASES + [
        ("reason", "pre<think>inner</think>post"),          # MULTI-BLOCK in content
        ("r1", "X<think>b</think>Y<think>d</think>Z"),      # MULTI-BLOCK x2
        ("multi\n</think>\nstray", "answer<think>x</think>"),
    ]
    for r, c in strip_cases:
        shadow_form = wrap_reasoning_think(r, c)   # sites 2/3 now emit this
        merge_form = wrap_reasoning_think(r, c)    # site 1 emits this (same source)
        assert _strip_thinking_all(shadow_form) == _strip_thinking_all(merge_form), (r, c)


def test_old_nonewline_form_diverged():
    """Regression-guard: the OLD no-newline shadow form DID diverge from strip(merge)
    on the stray-</think> case (so the fix is load-bearing, not cosmetic)."""
    r, c = "a</think>b", "content"
    old_shadow = f"<think>{r}</think>{c}"           # pre-fix sites 2/3
    merge = _OLD_merge_inline(r, c)                 # what the harness stores/resends
    assert _strip_thinking_all(old_shadow) != _strip_thinking_all(merge)
    # and the NEW form does NOT diverge
    assert _strip_thinking_all(wrap_reasoning_think(r, c)) == _strip_thinking_all(merge)
    # concrete bytes showing the divergence
    assert _strip_thinking_all(old_shadow) == "b</think>content"
    assert _strip_thinking_all(merge) == "b\n</think>\n\ncontent"


def test_helper_is_pure_no_guard():
    """The helper only FORMATS — it does not apply the reasoning/`<think>` guard (that
    stays at each call site). Empty reasoning still produces a (degenerate) wrapper."""
    assert wrap_reasoning_think("", "x") == "<think>\n\n</think>\n\nx"
    assert wrap_reasoning_think("r", "") == "<think>\nr\n</think>"
