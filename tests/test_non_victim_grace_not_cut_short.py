"""The NON-VICTIM grace defect, Loop A.

⛔ SCOPE, DELIBERATELY NARROW, because this file must NOT
duplicate the behavioural test:
`test_grace_fastlane_breakout.py::TestServeOnResidentGraceFastlaneBreakout::
test_non_victim_grace_runs_to_its_full_deadline_under_a_higher_ranked_claim`
already pins the BEHAVIOUR -- a non-victim's grace runs its full deadline and
writes no `grace_fastlane_breakout`. Nothing here re-asserts that.

THIS FILE PINS THE TWO THINGS THAT TEST CANNOT SEE:

1. THE RETENTION AND THE DIVERGENCE (structural). The fix does not add a
   guard -- it REMOVES a call site and RETAINS the code, uncalled, with a
   dead-code marker in its docstring. That is an ABSENCE, and an absence
   is exactly what rots silently: a future "cleanup" deleting the retained
   method, or a "tidy the two loops together" edit re-adding the Loop A call,
   would both be invisible to any behavioural test that is already green.
   Loop B keeps its copy ON PURPOSE, so the two loops differ -- pinned
   here so the divergence cannot be quietly "fixed".

2. THE CLAIM SURVIVES THE WAIT. The grace contract's "But" clause says the non-victim
   becomes evictable once its grace lapses in full. That is only worth
   anything if the higher-ranked claim is STILL THERE when the window ends.

★ WHY AST AND NOT `inspect.getsource(...)` SUBSTRINGS: the fix adds comment
prose that discusses `grace_fastlane_breakout` by name, and `getsource`
returns comments. A substring guard would read that prose and pass (or fail)
for reasons having nothing to do with the code. `ast` drops comments
entirely, so these assertions see CALLS and STRING LITERALS only.

NOT VERIFIED HERE, said plainly: end-to-end admission of the claimant through
the dispatcher/GPU-reservation machinery. That is
the integration-level acceptance test's job;
this file works at the `_serve_on_resident` seam and does not reach it.
"""

import ast
import asyncio
import pathlib
import time

import pytest

import turbohaul.manager as mgr_mod
from turbohaul.manager import Resident, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState

from test_grace_fastlane_breakout import (  # proven fixture builders, reused
    _boot_runtime,
    _make_fake_handle,
    _match,
    _TWO_RULES,
)

_EVENT = "grace_fastlane_breakout"
_RETAINED = "_legacy_fastlane_grace_breakout"


def _module_ast() -> ast.Module:
    src = pathlib.Path(mgr_mod.__file__).read_text()
    return ast.parse(src)


def _method(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in manager.py")


def _string_constants(node) -> list:
    return [
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def _attribute_calls(node) -> list:
    """Names of every `self.<name>(...)` call inside `node`."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            out.append(n.func.attr)
    return out


class TestTheRetentionAndTheDivergenceAreStructurallyPinned:
    def test_loop_A_no_longer_references_the_breakout_event_in_code(self):
        """Loop A (`_serve_on_resident`) must not emit or test for the
        breakout any more. AST-only, so the explanatory comment that names
        the event cannot satisfy or break this."""
        loop_a = _method(_module_ast(), "_serve_on_resident")
        assert _EVENT not in _string_constants(loop_a), (
            f"{_EVENT!r} still appears as a string literal inside "
            "_serve_on_resident -- the Loop A call site is back"
        )

    def test_the_retained_method_exists_and_has_ZERO_call_sites(self):
        """MARK AND RETAIN, not delete -- and retained means UNCALLED. Both
        halves fail loudly: deleting it, or re-attaching a caller."""
        tree = _module_ast()
        retained = _method(tree, _RETAINED)  # raises if deleted
        callers = [
            f.name
            for f in ast.walk(tree)
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
            and f is not retained
            and _RETAINED in _attribute_calls(f)
        ]
        assert callers == [], (
            f"{_RETAINED} is RETAINED DEAD CODE and must have no callers; "
            f"found {callers!r}. A caller here can only ever fire for a "
            "NON-victim, which is the non-victim grace defect returning."
        )

    def test_the_retained_method_carries_all_five_marker_parts(self):
        """The dead-code convention is only useful if the marker
        survives. Part 5 is the one that needed care -- nothing replaced
        this, and a blank there reads as an unfinished marker."""
        doc = ast.get_docstring(_method(_module_ast(), _RETAINED)) or ""
        for part in (
            "UNCALLED as of",
            "RETAINED, NOT DELETED",
            "WHY unreachable:",
            "NOT MAINTAINED",
            "WHAT REPLACED IT:",
        ):
            assert part in doc, f"5-part dead-code marker is missing {part!r}"
        assert "NOTHING" in doc, (
            "Part 5 must say explicitly that NOTHING replaced this -- a "
            "future reader needs 'replaced by nothing', not a blank"
        )

    def test_loop_B_deliberately_KEEPS_its_copy(self):
        """The two grace loops diverge ON PURPOSE (Loop B is the deprecated
        cap<=1 path, out of scope for this fix). If someone
        'tidies' them together this goes red -- which is the point."""
        loop_b = _method(_module_ast(), "_process_slot")
        assert _EVENT in _string_constants(loop_b), (
            "Loop B's breakout has been removed too. This fix scoped Loop A "
            "ONLY; Loop B keeps its copy on purpose and that copy "
            "must stay until that behaviour is deliberately changed."
        )


@pytest.mark.asyncio
class TestTheClaimSurvivesTheFullGraceWindow:
    async def test_the_higher_ranked_claim_is_still_queued_after_the_lapse(
        self, tmp_path
    ):
        """The grace contract's "But" clause is only worth anything if the claim is still
        waiting when the window ends. Making a claimant wait is correct;
        LOSING it while it waits would be a worse defect than the one this
        fix addresses, and no assertion in the behavioural test would notice."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            fastlane_enabled=True, fastlane_rules=_TWO_RULES,
        )

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )

        claim = Slot.new("m1", prompt="hi", thread_id="t2")
        claim.fastlane = _match(rule_index=0, rank=1)  # HIGH
        await mgr.queue.enqueue(claim)

        handle = _make_fake_handle("m1", 59500)
        r = Resident(
            model_tag="m1",
            handle=handle,
            port=handle.port,
            grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
        )
        anchor = Slot.new("m1", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED
        anchor.fastlane = _match(rule_index=1, rank=1)  # LOW -> non-victim

        started = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(r, anchor, handle),
            timeout=grace_seconds + 2.0,
        )
        elapsed = time.monotonic() - started

        # The window really did run out -- otherwise "survives the window"
        # would be true vacuously, having never been tested by one.
        assert elapsed >= grace_seconds - 1.0, (
            f"grace did not actually run its window ({elapsed:.2f}s) -- this "
            "test proves nothing unless the claim WAITED"
        )
        assert anchor.state is SlotState.POPPED, (
            f"anchor should have left grace at the natural lapse, got "
            f"{anchor.state!r}"
        )
        still_waiting = await mgr.queue.has_strictly_higher_priority_waiting(anchor)
        assert still_waiting is True, (
            "the higher-ranked claim vanished from the queue while it waited "
            "out the non-victim's grace -- the 'But' clause has nothing "
            "left to admit"
        )

        await mgr.shutdown()
