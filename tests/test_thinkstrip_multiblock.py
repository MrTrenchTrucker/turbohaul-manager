"""thinkstrip-multiblock — unit tests for `_strip_thinking_all`.

`_strip_thinking_all` is the remove-ALL think-strip used by the shadow-save
(`manager._shadow_reprefill_and_save`) + byte-match probe
(`manager._record_shadow_bytematch_probe`). It must mirror the agent harness's
resend (`agent_runtime_helpers.strip_think_blocks(x).strip()`) so the shadow's
predicted assistant turn byte-matches what the harness resends next turn.

These tests pin:
  * NO regression vs the old rsplit-last `_strip_thinking_wrapper` on the canonical
    single-block reasoning pattern (`<think>...</think>ANSWER`);
  * the multi-block / pre-`<think>` fix (the previously observed live MISMATCH cases);
  * byte-parity with a LOCAL reference of the harness's `<think>` pass (kept in this
    file so the test needs no external harness dep);
  * None-safety + edge cases (empty-after-strip, unclosed, nested-ish, case).

Run standalone:  PYTHONPATH=<worktree>/src python3 tests/test_thinkstrip_multiblock.py
Run via pytest:  PYTHONPATH=<worktree>/src pytest tests/test_thinkstrip_multiblock.py
"""
import re

from turbohaul.api.chat_completion import (
    _strip_thinking_all,
    _strip_thinking_wrapper,
)


# Local reference of the harness's FIRST pass + its callers' trailing `.strip()`
# (agent_runtime_helpers.strip_think_blocks -> `re.sub(r'<think>.*?</think>', ...,
#  flags=re.DOTALL | re.IGNORECASE)` then caller `.strip()`). Kept here so parity is
# asserted without importing the (external, do-not-edit) agent harness.
def _harness_think_pass(content: str) -> str:
    return re.sub(
        r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def test_single_block_no_regression_vs_old_wrapper():
    """Canonical reasoning pattern: new remove-all == old rsplit-last, byte-identical."""
    for s in (
        "<think>reasoning here</think>ANSWER_TEXT",
        "<think>x</think>final",
        "<think>chain of reasoning</think>THE_FINAL_ANSWER",
        "<think>r</think>A",
    ):
        assert _strip_thinking_all(s) == _strip_thinking_wrapper(s) == s.split("</think>")[-1]


def test_multiblock_keeps_inter_block_text():
    """`<think>a</think>X<think>b</think>Y` -> `XY` (new), NOT just `Y` (old)."""
    s = "<think>a</think>X<think>b</think>Y"
    assert _strip_thinking_all(s) == "XY"
    # old rsplit-last dropped the inter-block `X` — this is the regression being fixed
    assert _strip_thinking_wrapper(s) == "Y"


def test_prethink_prose_preserved():
    """`intro<think>t</think>ans` -> `introans` (remove-all), NOT `ans` (rsplit-last)."""
    s = "intro<think>t</think>ans"
    assert _strip_thinking_all(s) == "introans"
    assert _strip_thinking_wrapper(s) == "ans"


def test_no_think_unchanged():
    """No think block -> content passes through (surrounding ws stripped)."""
    assert _strip_thinking_all("plain content") == "plain content"
    # harness always `.strip()`s; shadow callers never reach this branch (they pre-guard
    # on `</think>`), so the trailing/leading strip here is harmless + harness-faithful.
    assert _strip_thinking_all("  padded answer  ") == "padded answer"


def test_empty_after_strip():
    """Only a think block -> empty (drives the shadow `skipped_empty` guard)."""
    assert _strip_thinking_all("<think>only reasoning</think>") == ""
    assert _strip_thinking_all("<think>a</think><think>b</think>") == ""


def test_none_and_empty_safe():
    """None-safe (mirrors `_strip_thinking_wrapper`): non-str passes through; ''-> ''."""
    assert _strip_thinking_all(None) is None
    assert _strip_thinking_all(123) == 123
    assert _strip_thinking_all("") == ""


def test_case_insensitive_matches_harness():
    """Mixed-case think tags are removed (harness uses re.IGNORECASE)."""
    s = "<think>a</think>X<THINK>b</THINK>Y"
    assert _strip_thinking_all(s) == "XY"


def test_dotall_multiline_reasoning():
    """`.` spans newlines (re.DOTALL) so multi-line reasoning is fully removed."""
    s = "<think>line1\nline2\nline3</think>ANSWER"
    assert _strip_thinking_all(s) == "ANSWER"


def test_nongreedy_pairs_each_open_with_own_close():
    """Non-greedy `.*?` must NOT span from the FIRST open to the LAST close and eat
    the middle visible text. `<think>a</think>MID<think>b</think>END` -> `MIDEND`."""
    s = "<think>a</think>MID<think>b</think>END"
    assert _strip_thinking_all(s) == "MIDEND"


def test_unclosed_open_tag_documented_behavior():
    """EDGE (documented): an UNCLOSED `<think>` (no `</think>`) is NOT removed by the
    remove-all pass (the harness handles it in a SEPARATE unterminated-tag pass we
    intentionally do not port — the shadow callers pre-guard on a literal `</think>`,
    so an unclosed-only turn never reaches this fn). Left content passes through
    stripped; the leading `pre ` keeps its inner space, only the ends are stripped."""
    s = "pre <think>dangling reasoning with no close"
    assert _strip_thinking_all(s) == "pre <think>dangling reasoning with no close"


def test_nested_like_tags_documented_behavior():
    """EDGE (documented): non-greedy matches the FIRST `</think>`, so a pseudo-nested
    `<think>outer<think>inner</think>tail` closes at the first `</think>`, leaving
    `tail`. True XML nesting isn't a real model emission; documented for clarity."""
    s = "<think>outer<think>inner</think>tail"
    assert _strip_thinking_all(s) == "tail"


def test_byte_parity_with_harness_reference():
    """The load-bearing property: `_strip_thinking_all` == the harness's `<think>`
    pass + `.strip()`, byte-for-byte, across the representative corpus."""
    corpus = [
        "<think>r</think>ANSWER",
        "<think>a</think>X<think>b</think>Y",
        "intro<think>t</think>ans",
        "<think>only</think>",
        "plain content",
        "<think>a</think>X<THINK>b</THINK>Y",
        "<think>multi\nline</think>done",
        "pre<think>x</think>mid<think>y</think>post",
    ]
    for s in corpus:
        assert _strip_thinking_all(s) == _harness_think_pass(s), s


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


if __name__ == "__main__":
    passed, failed = [], []
    for t in _TESTS:
        try:
            t()
            passed.append(t.__name__)
            print("  ok  " + t.__name__)
        except AssertionError as e:  # noqa: PERF203
            failed.append((t.__name__, repr(e)))
            print(" FAIL " + t.__name__ + "  " + repr(e))
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
