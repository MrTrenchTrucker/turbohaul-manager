"""The IDLE conjunct of the designation-restoration rule, plus a call-site
memoization guard.

WHAT ALREADY EXISTS AND IS DELIBERATELY NOT DUPLICATED HERE
-----------------------------------------------------------
The victim-enters-no-grace test already pins the GRACE half
of the restoration rule and the in-predicate half of the caching hazard:

  * `test_a_spared_resident_gets_a_full_fresh_window` -- the two-turn
    transition, asserting a FULL fresh grace window and not a remainder of
    one. That IS the restoration, behaviourally, for the grace conjunct.
  * `test_the_designation_predicates_store_no_state` -- an AST guard that
    fails on any attribute write INSIDE the two predicate bodies.

This file adds only what those two cannot reach. It does not restate them.

WHAT THIS FILE ADDS
-------------------
H1 (behavioural). The designation-restoration rule says the
de-designated client's "grace timer AND its idle-unload countdown are
restored". Only the grace conjunct was pinned. The idle conjunct's existing
test, the idle-unload-timer discriminator test, has a designated arm
and a NEVER-designated control arm -- neither performs the TRANSITION, and
its control resident was never a victim, so nothing was ever cached True for
a stale cache to hold. A stale-True cache sails straight through that file.

H2 (structural). The predicate-body guard inspects exactly two function bodies, so
a cache at a CALL SITE is textually outside it and structurally invisible to
it. `_drive_resident` calls `_is_designated_unload_target_locked` TWICE under a
single `_registry_lock` hold (in manager.py) -- two calls, same
lock, same resident, microseconds apart. That is the shape a speed-minded
refactorer memoizes, and it is the one place nothing watched.

⚠ COVERAGE STATED RATHER THAN IMPLIED (spelled out for H2).
Three memoization shapes, and which arm kills which:

  * a manager-level dict spanning call sites  -> H1 (behaviour) AND H2b
  * an `lru_cache` / caching decorator        -> H1 (behaviour) AND H2c
  * a hoisted local inside `_drive_resident`  -> H2a ONLY

  H1 CANNOT catch the hoisted local, and this is named rather than left for a
  reader to discover: that cache is created and discarded inside a single
  teardown and never spans the de-designation this file drives. H2a is the
  only arm that sees it. Conversely H2 alone would not prove the window is
  really restored -- it only proves nobody memoized. Both are needed.

`_serve_on_resident`'s own `skip_grace = self._is_designated_unload_target_locked(r)`
is NOT a violation and H2a is scoped so as not to flag it: that is a single
decision point consuming a value once, which is re-derivation, not caching.

Harness is imported from the idle-unload-timer test module rather than
re-declared, so the idle-branch classification this file asserts on is the
SAME instrument that the idle-unload-timer test's own two arms already use.
"""
import ast
import inspect
import pathlib
import textwrap

import pytest

from turbohaul.fastlane import FastLaneMatch
from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.manager import ResidentState, TurbohaulManager
import turbohaul.manager as manager_mod
from turbohaul.slot import Slot

from test_idle_unload_timer import (
    _boot_runtime,
    _drive_one_turn_and_classify,
    _mocks,
    _resident,
)

_PREDICATES = ("_is_designated_unload_target_locked", "_is_worst_ranked_loaded_locked")

_TWO_RULES = [
    FastLaneRule(address="9.9.9.9", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1)),
]


def _designation_calls(tree):
    """Every direct call to the designation predicate inside `tree`."""
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_is_designated_unload_target_locked"
    ]


def _predicate_calls(node):
    """Both designation predicates, in an arbitrary expression node."""
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in _PREDICATES
    ]


def _assign_targets(node):
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        return [node.target]
    return []


def _manager_source_tree():
    return ast.parse(pathlib.Path(manager_mod.__file__).read_text())


async def _one_victim_one_survivor(tmp_path, *, grace_seconds=5):
    """A worst-ranked resident made the designated victim by a live outranking
    claim -- the same construction the idle-unload-timer test's own discriminator uses, not a
    shortcut invented here. Returns the pieces the two H1 arms share."""
    boot, runtime = _boot_runtime(
        tmp_path, grace_seconds=grace_seconds,
        idle_hot_load_seconds=120, fastlane_rules=_TWO_RULES,
    )
    mgr = TurbohaulManager(boot, runtime, **_mocks())

    victim = _resident(
        mgr, "victim-model", 59961,
        rank_client_meta={"ip": "1.2.3.4"}, grace_seconds=grace_seconds,
    )
    victim.state = ResidentState.ACTIVE
    victim.last_active_monotonic = 1.0

    survivor = _resident(
        mgr, "survivor-model", 59962,
        rank_client_meta={"ip": "1.2.3.4"}, grace_seconds=grace_seconds,
    )
    survivor.state = ResidentState.ACTIVE
    survivor.last_active_monotonic = 1000.0

    claimant = Slot.new("claimant-model", prompt="hi", thread_id="claim")
    claimant.fastlane = FastLaneMatch(
        rule_index=0, raw_address="9.9.9.9", label="",
        effective_tag="main", rank=1,
    )
    mgr._register_fastlane_claim_locked(claimant, "make_room_starved_count_cap")
    return mgr, victim, claimant


@pytest.mark.asyncio
class TestIdleWindowIsRestoredWhenDesignationLifts:
    """H1 -- the idle-unload conjunct of the restoration rule, driven as a TRANSITION."""

    async def test_a_de_designated_resident_gets_its_idle_window_back(self, tmp_path):
        """★ THE CLAIM. A resident IS the designated victim (its idle window is
        withheld -- it would be torn down the instant its turn ends), then the
        claim is satisfied elsewhere mid-turn, so by the time its own turn
        completes it is NOT the victim. Per the restoration rule, its idle-unload
        countdown is restored, so it parks IDLE_EVICTABLE on its ordinary 120s window
        instead of being evicted.

        The release is hooked to the `grace_designated_victim_skip` audit
        event rather than a sleep, which makes it DETERMINISTIC and puts it
        exactly where the restoration rule puts it: after the designation check that decides
        this turn's grace, and before the idle branch that
        decides its window. That span is the whole point --
        a cache populated at the first call and read at the second reports a
        victim that no longer exists, and this resident dies for a claim that
        was already served.

        Observes the WINDOW, never the predicate: restoration is only an
        absence if you look at the flag, and a positive timeable fact if you
        look at what the client gets.
        """
        grace_seconds = 5
        mgr, victim, claimant = await _one_victim_one_survivor(
            tmp_path, grace_seconds=grace_seconds)

        # PRECONDITION -- holds in BOTH arms and on BOTH sides of the fix: the
        # fixture really does put the resident into the withheld state. Without
        # this the arm below could pass by never designating anything at all.
        assert mgr._is_designated_unload_target_locked(victim) is True

        released = []
        orig_audit = mgr._audit_async

        async def _audit_and_serve_the_claim_elsewhere(slot, event_type):
            await orig_audit(slot, event_type)
            if event_type == "grace_designated_victim_skip" and not released:
                released.append(True)
                # The restoration story: the claim is satisfied by another resident's
                # lapse, so this one stops being the victim mid-turn. The
                # registry lock is NOT held at this point (the grace-deciding `async
                # with` block has closed), so taking it here is safe.
                async with mgr._registry_lock:
                    mgr._release_fastlane_claim_locked(claimant, "served")

        mgr._audit_async = _audit_and_serve_the_claim_elsewhere

        try:
            outcome = await _drive_one_turn_and_classify(
                mgr, victim, "victim-model", timeout_s=grace_seconds + 3.0,
            )
            assert released, (
                "NON-VACUITY: the de-designation seam never fired, so this arm "
                "never exercised the transition it claims to test. The resident "
                "was not skipped as a designated victim at all."
            )
            assert outcome == "idle_evictable", (
                f"restoration rule: a client that is NO LONGER the designated victim gets "
                f"its idle-unload countdown back and parks on its ordinary 120s "
                f"window. outcome={outcome!r}. 'evicted' here means the idle "
                f"branch acted on a designation that had already lifted -- a "
                f"carried-forward verdict, which the no-carry-forward rule forbids."
            )
        finally:
            mgr._audit_async = orig_audit
            mgr._residents.pop("survivor-model", None)
            await mgr.shutdown()

    async def test_CONTROL_a_still_designated_resident_is_still_evicted(self, tmp_path):
        """★ CONTROL / DISCRIMINATOR -- passes on both sides, and it is what
        makes the arm above mean something. Same fixture, same drive, one
        difference: the claim is NEVER released, so the resident is still the
        designated victim when its turn ends and is evicted immediately.

        Without this, `idle_evictable` above could simply be what this fixture
        always produces. With it, the two outcomes differ on exactly the one
        variable under test.
        """
        grace_seconds = 5
        mgr, victim, _claimant = await _one_victim_one_survivor(
            tmp_path, grace_seconds=grace_seconds)
        assert mgr._is_designated_unload_target_locked(victim) is True

        try:
            outcome = await _drive_one_turn_and_classify(
                mgr, victim, "victim-model", timeout_s=grace_seconds + 3.0,
            )
            assert outcome == "evicted", (
                f"a STILL-designated victim keeps losing its idle window -- if "
                f"this parks IDLE_EVICTABLE then the fixture is not producing "
                f"designation at all and the sibling arm proves nothing. "
                f"outcome={outcome!r}"
            )
        finally:
            mgr._residents.pop("survivor-model", None)
            await mgr.shutdown()


class TestNoCallSiteMemoizesTheDesignationVerdict:
    """The call-site half. The predicate-body guard defends the inside of the
    two predicates; these three defend everywhere the verdict is CONSUMED.
    Green before and after this change: they defend an absence."""

    def test_H2a_each_idle_teardown_decision_point_re_derives(self):
        """Shape 1 of 3 -- THE HOISTED LOCAL, and the only shape H1 cannot see.

        `_drive_resident` asks the designation question at two separate
        decision points under one lock hold: whether to skip the idle window
        and whether to pre-drain the victim's inbox. Hoisting
        that into `is_unload_target = ...` and reusing it is the cheapest possible
        memoization and reads as a pure speed win in review.

        Fails if either the second call disappears or the verdict is bound to a
        name inside this function. Scoped to `_drive_resident` on purpose:
        `_serve_on_resident`'s `skip_grace = ...` binds the verdict too, but
        consumes it at ONE decision point, which is re-derivation per call, not
        a cache -- flagging it would be a false positive.
        """
        tree = ast.parse(textwrap.dedent(
            inspect.getsource(TurbohaulManager._drive_resident)))
        calls = _designation_calls(tree)
        assert len(calls) >= 2, (
            f"_drive_resident must ASK the designation question freshly at each "
            f"decision point, not once; found {len(calls)} direct call(s). The "
            f"no-carry-forward rule: 'no client name is carried forward from an earlier decision.' "
            f"If a decision point was deliberately removed, this guard is the "
            f"place to record that."
        )
        bound = []
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            if not targets or node.value is None:
                continue
            if _designation_calls(ast.Module(body=[ast.Expr(node.value)],
                                             type_ignores=[])):
                bound.extend(ast.unparse(t) for t in targets)
        assert not bound, (
            f"_drive_resident binds the designation verdict to {bound} and can "
            f"then reuse it at a later decision point -- that is a hoisted-local "
            f"cache. Each decision point must call the predicate itself."
        )

    def test_H2b_no_call_site_stores_the_verdict_in_attribute_or_subscript(self):
        """Shape 2 of 3 -- THE DICT ON THE MANAGER.

        `self._designation_cache[key] = self._is_designated_unload_target_locked(r)`
        survives the predicate-body guard completely: that guard walks the two
        predicate BODIES, and this assignment is outside both. Module-wide:
        the verdict may be consumed, never stored anywhere that outlives the
        call -- an attribute or a subscript.

        ⚠ TAINT-AWARE ON PURPOSE. The obvious version of this check looks for a
        predicate call in the assignment's VALUE, and a two-statement launder
        walks straight past it:

            v = self._is_designated_unload_target_locked(r)   # value is a call
            self._cache[id(r)] = v                     # value is only a Name

        Both statements are individually innocent. So this tracks, per
        function, the local names bound to a verdict and flags a store of
        EITHER a direct call or one of those names. Verified to flag both
        shapes and to leave `_serve_on_resident`'s legitimate
        `skip_grace = ...` alone -- that binds a verdict to a local and
        consumes it once, which is re-derivation, not storage.
        """
        stored = []
        tree = _manager_source_tree()
        functions = [n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for fn in functions:
            tainted = set()
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and node.value is not None \
                        and _predicate_calls(node.value):
                    tainted.update(t.id for t in node.targets
                                   if isinstance(t, ast.Name))
            for node in ast.walk(fn):
                targets = _assign_targets(node)
                value = getattr(node, "value", None)
                if not targets or value is None:
                    continue
                direct = bool(_predicate_calls(value))
                laundered = any(isinstance(x, ast.Name) and x.id in tainted
                                for x in ast.walk(value))
                if not (direct or laundered):
                    continue
                for t in targets:
                    if isinstance(t, (ast.Attribute, ast.Subscript)):
                        stored.append(
                            f"{fn.name}: {ast.unparse(t)} (line {t.lineno})")
        assert not stored, (
            f"the designation verdict is written to storage that outlives the "
            f"call: {stored}. Designation is DERIVED, never remembered (the no-carry-forward "
            f"rule), and the restoration rule works only while that holds -- a "
            f"stored verdict keeps costing a client its windows after the "
            f"designation that justified them has lifted."
        )

    def test_H2c_the_predicates_carry_no_caching_decorator_and_are_not_rebound(self):
        """Shape 3 of 3 -- THE `lru_cache`.

        Also invisible to the predicate-body guard, and for a subtler reason than the
        dict: a decorator is not an assignment at all, so a guard looking for
        attribute writes inside the body cannot see one no matter how it walks.
        An `@lru_cache` keyed on (self, resident) freezes the first answer for
        the life of the process.
        """
        tree = _manager_source_tree()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name in _PREDICATES:
                decorators = [ast.unparse(d) for d in node.decorator_list]
                assert not decorators, (
                    f"{node.name} carries decorator(s) {decorators}. The "
                    f"designation predicates must stay undecorated: a caching "
                    f"decorator freezes the verdict for the process lifetime "
                    f"and breaks designation restoration by construction."
                )
        rebound = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                name = t.attr if isinstance(t, ast.Attribute) else (
                    t.id if isinstance(t, ast.Name) else None)
                if name in _PREDICATES:
                    rebound.append(f"{ast.unparse(t)} (line {t.lineno})")
        assert not rebound, (
            f"a designation predicate is rebound at {rebound} -- wrapping it "
            f"(`f = lru_cache()(f)`) is the same freeze as decorating it."
        )

    def test_the_guard_this_file_relies_on_still_exists_and_is_narrower(self):
        """NON-VACUITY for the whole H2 class, and the reason it exists.

        Everything above is justified by the predicate-body guard being narrower than
        its name suggests. If that guard were ever widened to the call sites,
        these three would become duplicates and should be retired -- so this
        asserts the DIVISION OF LABOUR rather than assuming it, and fails
        loudly if the other file changes scope.

        ⚠ A SUBSTRING CHECK WOULD BE VACUOUS HERE. A check of the form
        `assert "test_the_designation_predicates_store_no_state" in src` -- a
        substring test on source text, which cannot tell a live guard from its
        own epitaph. Measured, not argued: deleting the guard outright and
        leaving `# test_the_designation_predicates_store_no_state was removed
        here` dropped the other file from 8 tests to 7 and a substring assertion
        would not notice. This check searches for a `FunctionDef` of that exact name.

        ⛔ WHAT THIS STILL DOES NOT CLOSE -- enumerated and MEASURED, so the
        assertion stops implying more than it checks. A name search proves a
        definition EXISTS; it proves nothing about whether it still runs or
        still asserts anything:
          · (4) body emptied to `pass`            -- name survives, NOT caught
          · (5) `@pytest.mark.skip` applied       -- name survives, NOT caught
          · (6) assertions weakened to `assert True` -- name survives, NOT caught
          · (7) def moved inside another function so pytest never collects it
                -- `ast.walk` still finds it, NOT caught
        What it DOES close: (1) the file deleted (via `exists()`), (2) the guard
        deleted with its name surviving in a comment, (3) a true rename. A
        SUPERSTRING rename now fails too, deliberately: a rename is an
        intentional act, so re-pointing this assertion should be intentional.

        Cases (4)-(7) are NOT defects to fix here -- pinning that another file's test
        still has teeth is a different and much larger claim than pinning that
        it still exists, and it belongs to whoever owns that file. They are
        written down so nobody reads this assertion as coverage it does not
        provide, which is the exact failure this whole file exists to prevent.
        """
        other = pathlib.Path(__file__).with_name(
            "test_victim_enters_no_grace.py")
        assert other.exists(), (
            "the victim-enters-no-grace test file is gone -- the grace half "
            "of the restoration rule and the in-predicate caching guard went with it.")
        src = other.read_text()
        tree = ast.parse(src)
        # A DEFINITION, not a substring: `"name" in src` also passes when the
        # guard has been deleted and its name survives only in a comment.
        guards = [n.name for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "test_the_designation_predicates_store_no_state"]
        assert guards, (
            "The in-predicate guard is gone as a DEFINITION -- a "
            "comment mentioning its name is not the guard. This file only ever "
            "covered the call sites and does NOT cover predicate bodies, so "
            "losing that one leaves in-body stores unwatched entirely. If it "
            "was renamed, re-point this assertion deliberately.")
        scoped = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr in _PREDICATES
        ]
        assert scoped, (
            "The in-predicate guard no longer names the predicates directly; "
            "re-read both files before trusting either.")
