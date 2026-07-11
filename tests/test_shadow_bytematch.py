"""Standalone unit test for the shadow byte-match self-check.

Run: PYTHONPATH=<worktree>/src python3 test_shadow_bytematch.py
Uses TurbohaulManager.__new__ to bypass the heavy __init__; the two helpers only
touch self._shadow_bytematch_{probe,counts} + the static hash/strip helpers.
"""
import sys
from types import SimpleNamespace

from turbohaul.manager import TurbohaulManager, _SHADOW_BYTEMATCH_CAP


def fresh_mgr():
    m = TurbohaulManager.__new__(TurbohaulManager)
    m._shadow_bytematch_probe = {}
    m._shadow_bytematch_counts = {}
    return m


def slot(thread_id="T1", streamed=None):
    return SimpleNamespace(thread_id=thread_id, streamed_assistant_text=streamed)


def result_msg(content=None, tool_calls=None, reasoning=None):
    msg = {}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg}]}


PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ok  " if cond else " FAIL ") + name)


# ---------------------------------------------------------------------------
# 1. RECORD non-streaming -> stash think-free hash
# ---------------------------------------------------------------------------
m = fresh_mgr()
th = TurbohaulManager._thread_hash("T1")
m._record_shadow_bytematch_probe(slot("T1"), result_msg("<think>reasoning here</think>ANSWER_TEXT"))
stash = m._shadow_bytematch_probe.get(th)
check("record non-streaming stashes entry", stash is not None)
check(
    "record non-streaming hashes THINK-FREE content",
    stash and stash["assistant_hash"] == TurbohaulManager._turn_hash("assistant", "ANSWER_TEXT"),
)
check("record non-streaming sample is think-free", stash and stash["sample"] == "ANSWER_TEXT")

# ---------------------------------------------------------------------------
# 2. COMPARE match (harness resends think-stripped == manager's strip)
# ---------------------------------------------------------------------------
ctx = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ANSWER_TEXT"},
       {"role": "user", "content": "next question"}]
m._compare_shadow_bytematch_probe("T1", ctx, None)
check("compare MATCH increments match", m._shadow_bytematch_counts.get("match") == 1)
check("compare MATCH pops the probe (consumed)", th not in m._shadow_bytematch_probe)

# ---------------------------------------------------------------------------
# 3. COMPARE mismatch (harness resend differs -> byte-delta detected)
# ---------------------------------------------------------------------------
m = fresh_mgr()
m._record_shadow_bytematch_probe(slot("T1"), result_msg("<think>r</think>ANSWER_TEXT"))
ctx_bad = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ANSWER_DIFFERENT"}]
m._compare_shadow_bytematch_probe("T1", ctx_bad, None)
check("compare MISMATCH increments mismatch", m._shadow_bytematch_counts.get("mismatch") == 1)
check("compare MISMATCH pops the probe", TurbohaulManager._thread_hash("T1") not in m._shadow_bytematch_probe)

# ---------------------------------------------------------------------------
# 4. skip: tool-call turn (non-streaming _generated_assistant_msg -> None)
# ---------------------------------------------------------------------------
m = fresh_mgr()
m._record_shadow_bytematch_probe(slot("T2"), result_msg(content=None, tool_calls=[{"id": "c1"}]))
check("skip tool-call -> skipped_toolcall", m._shadow_bytematch_counts.get("skipped_toolcall") == 1)
check("skip tool-call writes NO stash", len(m._shadow_bytematch_probe) == 0)

# ---------------------------------------------------------------------------
# 5. skip: content has no </think>
# ---------------------------------------------------------------------------
m = fresh_mgr()
m._record_shadow_bytematch_probe(slot("T3"), result_msg("plain answer, no think block"))
check("skip no-think -> skipped_no_think", m._shadow_bytematch_counts.get("skipped_no_think") == 1)
check("skip no-think writes NO stash", len(m._shadow_bytematch_probe) == 0)

# ---------------------------------------------------------------------------
# 6. skip: think-free strip empty (only a think block, nothing after)
# ---------------------------------------------------------------------------
m = fresh_mgr()
m._record_shadow_bytematch_probe(slot("T4"), result_msg("<think>only reasoning</think>"))
check("skip empty-after-strip -> skipped_empty", m._shadow_bytematch_counts.get("skipped_empty") == 1)
check("skip empty writes NO stash", len(m._shadow_bytematch_probe) == 0)

# ---------------------------------------------------------------------------
# 7. RECORD streaming path (result=None -> uses slot.streamed_assistant_text)
# ---------------------------------------------------------------------------
m = fresh_mgr()
m._record_shadow_bytematch_probe(slot("T5", streamed="<think>r</think>STREAMED_ANSWER"), None)
th5 = TurbohaulManager._thread_hash("T5")
check(
    "record STREAMING hashes think-free streamed text",
    m._shadow_bytematch_probe.get(th5, {}).get("assistant_hash")
    == TurbohaulManager._turn_hash("assistant", "STREAMED_ANSWER"),
)
# streaming with no stashed text -> skipped_empty (no false stash)
m._record_shadow_bytematch_probe(slot("T6", streamed=None), None)
check("record STREAMING empty -> skipped_empty", m._shadow_bytematch_counts.get("skipped_empty") == 1)

# ---------------------------------------------------------------------------
# 8. reasoning_content merged path (content w/o inline <think>) still records
# ---------------------------------------------------------------------------
m = fresh_mgr()
m._record_shadow_bytematch_probe(slot("T7"), result_msg(content="FINAL", reasoning="deep thoughts"))
th7 = TurbohaulManager._thread_hash("T7")
# _generated_assistant_msg wraps -> "<think>deep thoughts</think>FINAL"; strip -> "FINAL"
check(
    "record reasoning_content path hashes think-free 'FINAL'",
    m._shadow_bytematch_probe.get(th7, {}).get("assistant_hash")
    == TurbohaulManager._turn_hash("assistant", "FINAL"),
)

# ---------------------------------------------------------------------------
# 9. memory bound: never exceeds cap
# ---------------------------------------------------------------------------
m = fresh_mgr()
for i in range(_SHADOW_BYTEMATCH_CAP + 40):
    m._record_shadow_bytematch_probe(slot(f"thread-{i}"), result_msg(f"<think>r</think>ans{i}"))
check(f"stash bounded at cap ({_SHADOW_BYTEMATCH_CAP})", len(m._shadow_bytematch_probe) == _SHADOW_BYTEMATCH_CAP)

# ---------------------------------------------------------------------------
# 10. compare uses LAST assistant across multi-turn history
# ---------------------------------------------------------------------------
m = fresh_mgr()
m._record_shadow_bytematch_probe(slot("T8"), result_msg("<think>r</think>LATEST"))
multi = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "OLD_ANSWER"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "LATEST"},   # assistant-N == what we stashed
    {"role": "user", "content": "q3"},
]
m._compare_shadow_bytematch_probe("T8", multi, None)
check("compare picks LAST assistant (multi-turn) -> match", m._shadow_bytematch_counts.get("match") == 1)

# ---------------------------------------------------------------------------
# 11. compare no-op when no stash / no messages (dormant, no crash, no count)
# ---------------------------------------------------------------------------
m = fresh_mgr()
m._compare_shadow_bytematch_probe("nope", [{"role": "assistant", "content": "x"}], None)
check("compare with no stash is a no-op", m._shadow_bytematch_counts == {})
m._record_shadow_bytematch_probe(slot("T9"), result_msg("<think>r</think>A"))
m._compare_shadow_bytematch_probe("T9", None, None)  # no messages -> stash preserved
check("compare with no messages preserves stash", TurbohaulManager._thread_hash("T9") in m._shadow_bytematch_probe)

# ---------------------------------------------------------------------------
# 12. best-effort: never raises even on garbage input
# ---------------------------------------------------------------------------
m = fresh_mgr()
try:
    m._record_shadow_bytematch_probe(slot("T10"), "not-a-dict")
    m._record_shadow_bytematch_probe(object(), None)          # slot w/o attrs
    m._compare_shadow_bytematch_probe("T10", "garbage", 12345)
    check("best-effort swallows garbage (no raise)", True)
except Exception as e:  # pragma: no cover
    check(f"best-effort swallows garbage (no raise) [raised {e!r}]", False)

print()
print(f"PASSED {len(PASS)} / {len(PASS) + len(FAIL)}")
if FAIL:
    print("FAILURES:", FAIL)
    sys.exit(1)
print("ALL GREEN")
