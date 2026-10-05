"""The designated victim ENTERS no grace, and restoration
is a negative requirement.

THE RULE UNDER TEST. The fast-lane rule for victims: the designated victim
-- the lowest-priority loaded client -- gets NO grace timer and NO idle-unload
countdown. Both are surrendered the instant it is designated, not when its turn
ends. No grace state is ever entered for the victim.
A victim that skipped only the WAIT would still transition into
SlotState.GRACE, start the resident's GraceTimer and emit `grace_enter`.
The FSM table needs a legal ACTIVE->POPPED transition for that,
and the accompanying fsm.py change provides it.

WHAT IS NOT TESTED HERE, DELIBERATELY. The idle-unload half of the victim rule already
ships correctly at `_drive_resident`'s immediate-evict branch and is untouched.
The in-loop `grace_fastlane_breakout` check is unchanged. Loop B (`_process_slot`)
is out of scope here and is not exercised.

HARNESS. Direct `_serve_on_resident` calls against two loaded residents, the
same shape the grace-wiring test and
`test_grace_fastlane_breakout.py` already use -- it exercises exactly the branch
this change touched without the dispatcher/GPU-reservation machinery.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import time
from unittest.mock import MagicMock

import pytest

import turbohaul.manager as manager_mod
from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle

_GRACE_S = 2

# Two rules so a claimant can STRICTLY outrank a resident: 9.9.9.9 takes index
# 0 (the claimant), 1.2.3.4 index 1 (both residents). The victim rule's precondition is
# a live claim that strictly outranks -- with one rule nothing could.
_RULES = [
    FastLaneRule(address="9.9.9.9", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1)),
]


def _boot_runtime(tmp_path, *, grace_seconds=_GRACE_S):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False, grace_seconds=grace_seconds,
                          max_grace_extensions=50, idle_hot_load_seconds=0),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=_RULES),
    )
    return boot, runtime


def _handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 77_777
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _audit_events(boot, slot_id):
    conn = open_state_db(boot.storage.state_db_path)
    cur = conn.execute(
        "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
        (slot_id,),
    )
    events = [r["event_type"] for r in cur.fetchall()]
    conn.close()
    return events


def _resident(mgr, model_tag, port, *, last_active):
    h = _handle(model_tag, port)
    r = Resident(
        model_tag=model_tag, resident_key=model_tag, handle=h, port=port,
        grace=GraceTimer(grace_seconds=_GRACE_S, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=0),
        rank_client_meta={"ip": "1.2.3.4"},
        last_active_monotonic=last_active,
    )
    r.state = ResidentState.ACTIVE
    mgr._residents[model_tag] = r
    return r, h


def _mgr(boot, runtime):
    async def fake_complete(slot, handle):
        return {"ok": True}

    return TurbohaulManager(
        boot, runtime,
        spawn_fn=lambda *a, **k: None,
        health_fn=lambda *a, **k: True,
        sigterm_fn=lambda *a, **k: (True, "clean"),
        vram_fn=lambda **k: (True, 100),
        complete_fn=fake_complete,
    )


def _claimant():
    c = Slot.new("claimant-model", prompt="hi", thread_id="claim")
    c.fastlane = FastLaneMatch(
        rule_index=0, raw_address="9.9.9.9", label="",
        effective_tag="main", rank=1,
    )
    return c


async def _settle_bg(mgr, timeout_s=5.0):
    """Wait for the manager's fire-and-forget background tasks to finish."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = [t for t in list(mgr._bg_tasks) if not t.done()]
        if not pending:
            return
        await asyncio.wait(pending, timeout=0.25)
    raise AssertionError("background tasks did not settle")


def _spy_transitions(monkeypatch):
    """Record (slot_id, from_state, to_state) for EVERY FSM transition.

    This observes the state machine itself, not an audit row that describes it
    -- the design assertion is about the state being entered, and a test that
    reads only the audit event would pass against code that entered GRACE
    silently.
    """
    seen: list[tuple[str, SlotState, SlotState]] = []
    real = manager_mod.transition

    def spy(slot, to_state):
        seen.append((slot.slot_id, slot.state, to_state))
        return real(slot, to_state)

    monkeypatch.setattr(manager_mod, "transition", spy)
    return seen


async def _two_residents_one_claim(tmp_path, monkeypatch):
    """victim (least recently active, so the worst) + survivor, plus a live
    strictly-outranking claim so the victim rule actually engages."""
    boot, runtime = _boot_runtime(tmp_path)
    mgr = _mgr(boot, runtime)
    victim, v_handle = _resident(mgr, "victim-model", 59901, last_active=1.0)
    _resident(mgr, "survivor-model", 59902, last_active=1000.0)
    claimant = _claimant()
    mgr._register_fastlane_claim_locked(claimant, "make_room_starved_count_cap")
    # Registering a claim spawns a BACKGROUND audit write (_emit_fastlane_claim_
    # event -> asyncio.to_thread -> sqlite). Let it land before the turn starts:
    # without this, that thread and the turn's own `active` audit write race for
    # the same database and one of them loses with "database is locked". That is
    # a fixture artifact, not the behaviour under test, and it makes whichever
    # test happens to lose the race fail for the wrong reason.
    await _settle_bg(mgr)
    assert mgr._is_designated_unload_target_locked(victim) is True, (
        "fixture precondition: the resident under test must actually be the "
        "designated victim, or every assertion below is vacuous"
    )
    return boot, mgr, victim, v_handle, claimant


@pytest.mark.asyncio
class TestVictimEntersNoGrace:
    """The fix. Every test here is RED without the fix."""

    async def test_victim_emits_no_grace_enter_event(self, tmp_path, monkeypatch):
        """On the audit surface: no grace state is ever
        entered for the victim, so no `grace_enter` marker exists for it."""
        boot, mgr, victim, v_handle, _ = await _two_residents_one_claim(
            tmp_path, monkeypatch)
        anchor = Slot.new("victim-model", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED

        await asyncio.wait_for(
            mgr._serve_on_resident(victim, anchor, v_handle),
            timeout=_GRACE_S - 0.5,
        )
        events = _audit_events(boot, anchor.slot_id)
        assert "grace_enter" not in events, (
            "Victim rule: the designated victim gets NO grace timer -- it must not "
            f"emit grace_enter at all, events={events!r}"
        )
        assert "grace_designated_victim_skip" in events, (
            f"the victim's own marker must still fire, events={events!r}"
        )
        await mgr.shutdown()

    async def test_victim_slot_never_holds_the_grace_state(self, tmp_path, monkeypatch):
        """★ The literal assertion, observed on the STATE MACHINE
        rather than on an audit row that merely describes it."""
        seen = _spy_transitions(monkeypatch)
        _boot, mgr, victim, v_handle, _ = await _two_residents_one_claim(
            tmp_path, monkeypatch)
        anchor = Slot.new("victim-model", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED

        await asyncio.wait_for(
            mgr._serve_on_resident(victim, anchor, v_handle),
            timeout=_GRACE_S - 0.5,
        )
        mine = [(f, t) for sid, f, t in seen if sid == anchor.slot_id]
        assert not any(t is SlotState.GRACE for _f, t in mine), (
            "No grace state may ever be entered for the victim: "
            f"the victim's transitions were {mine!r}"
        )
        await mgr.shutdown()

    async def test_victim_goes_from_active_straight_to_popped(self, tmp_path, monkeypatch):
        """The consequence that made the fsm.py edge a precondition: with no
        GRACE waypoint the victim leaves ACTIVE for POPPED directly. Before
        the fsm.py change this transition raised InvalidTransition."""
        seen = _spy_transitions(monkeypatch)
        _boot, mgr, victim, v_handle, _ = await _two_residents_one_claim(
            tmp_path, monkeypatch)
        anchor = Slot.new("victim-model", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED

        await asyncio.wait_for(
            mgr._serve_on_resident(victim, anchor, v_handle),
            timeout=_GRACE_S - 0.5,
        )
        mine = [(f, t) for sid, f, t in seen if sid == anchor.slot_id]
        assert (SlotState.ACTIVE, SlotState.POPPED) in mine, (
            f"expected a direct ACTIVE->POPPED hop, got {mine!r}"
        )
        assert anchor.state is SlotState.POPPED
        await mgr.shutdown()

    async def test_victim_grace_timer_is_never_started(self, tmp_path, monkeypatch):
        """The victim rule's other half of the same sentence: NO grace TIMER, not just
        no audit event. `GraceTimer.start()` must never run for the victim, so
        the resident's timer stays unarmed."""
        _boot, mgr, victim, v_handle, _ = await _two_residents_one_claim(
            tmp_path, monkeypatch)
        anchor = Slot.new("victim-model", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED

        await asyncio.wait_for(
            mgr._serve_on_resident(victim, anchor, v_handle),
            timeout=_GRACE_S - 0.5,
        )
        assert victim.grace.thread_id is None, (
            "GraceTimer.start() stamps thread_id; a victim whose timer was "
            f"armed shows {victim.grace.thread_id!r}"
        )
        assert victim.grace.expired() is True
        assert anchor.grace_started_at == 0.0, (
            "grace_started_at must not be stamped for a slot that never "
            "entered grace"
        )
        await mgr.shutdown()


@pytest.mark.asyncio
class TestNonVictimIsUntouched:
    """★ CONTROLS -- must pass BOTH before and after the fix, and they share
    no branch with the class above: these drive the `else` arm. A change that
    broke the ordinary grace path would be invisible to the victim tests."""

    async def test_non_victim_still_enters_grace_and_waits(self, tmp_path, monkeypatch):
        seen = _spy_transitions(monkeypatch)
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mgr(boot, runtime)
        # A SOLO resident with no claim registered: ordinary-grace rule, grace is
        # unconditional in this state.
        r, h = _resident(mgr, "solo-model", 59901, last_active=1.0)
        anchor = Slot.new("solo-model", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED
        assert mgr._is_designated_unload_target_locked(r) is False

        started = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(r, anchor, h), timeout=_GRACE_S + 3,
        )
        elapsed = time.monotonic() - started

        events = _audit_events(boot, anchor.slot_id)
        assert "grace_enter" in events, (
            f"Ordinary-grace rule: every non-victim keeps its grace window, events={events!r}"
        )
        assert "grace_designated_victim_skip" not in events
        mine = [(f, t) for sid, f, t in seen if sid == anchor.slot_id]
        assert (SlotState.ACTIVE, SlotState.GRACE) in mine, (
            f"the ordinary path must still transit GRACE, got {mine!r}"
        )
        assert elapsed >= _GRACE_S - 0.5, (
            f"a non-victim must hold its FULL window; held {elapsed:.2f}s of "
            f"{_GRACE_S}s"
        )
        await mgr.shutdown()

    async def test_non_victim_grace_timer_is_started(self, tmp_path, monkeypatch):
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mgr(boot, runtime)
        r, h = _resident(mgr, "solo-model", 59901, last_active=1.0)
        anchor = Slot.new("solo-model", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED

        await asyncio.wait_for(
            mgr._serve_on_resident(r, anchor, h), timeout=_GRACE_S + 3,
        )
        assert r.grace.thread_id == "t1", (
            "the ordinary path must still arm the resident's GraceTimer"
        )
        assert anchor.grace_started_at > 0.0
        await mgr.shutdown()


@pytest.mark.asyncio
class TestRestorationIsANegativeRequirement:
    """Spared-victim rule: a spared victim returns to normal operation; its
    grace timer and idle-unload countdown are restored at that moment.

    DESIGN DECISION: a FULL fresh window, never a
    remainder. In the spared-victim scenario the victim is spared while still MID-TURN,
    so its grace clock never started and there is no remainder to compute
    against; and a partial window would be exactly the continuing cost the rule
    forbids: a designation that no longer holds must not go on
    costing the client the windows ordinary grace grants every loaded client.

    Restoration therefore needs NO code and NO state -- designation is already
    stateless and re-derived on every call, so once the claim is served the
    predicate simply answers False. These tests PIN that absence so a future
    change that adds a victim flag fails here instead of silently breaking restoration.
    """

    async def test_a_spared_resident_gets_a_full_fresh_window(self, tmp_path, monkeypatch):
        """Both halves in one run, so this is a real discriminator and not only
        a regression pin: turn 1 under designation takes NO grace, and turn 2
        after the claim is served takes a FULL window -- not a remainder of
        one, and not none at all."""
        boot, mgr, victim, v_handle, claimant = await _two_residents_one_claim(
            tmp_path, monkeypatch)

        # Turn 1 -- designated. No grace at all.
        a1 = Slot.new("victim-model", prompt="hi", thread_id="t1")
        a1.state = SlotState.STAGED
        t0 = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(victim, a1, v_handle),
            timeout=_GRACE_S - 0.5,
        )
        turn1 = time.monotonic() - t0
        assert "grace_enter" not in _audit_events(boot, a1.slot_id)
        assert turn1 < _GRACE_S - 0.5

        # The queued request has been served -- designation lifts. Nothing is
        # "restored"; the predicate simply stops answering True.
        mgr._release_fastlane_claim_locked(claimant, "served")
        await _settle_bg(mgr)
        assert mgr._is_designated_unload_target_locked(victim) is False

        # Turn 2 -- spared. Ordinary grace again, and the WHOLE window.
        victim.state = ResidentState.ACTIVE
        a2 = Slot.new("victim-model", prompt="hi", thread_id="t2")
        a2.state = SlotState.STAGED
        t0 = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(victim, a2, v_handle), timeout=_GRACE_S + 3,
        )
        turn2 = time.monotonic() - t0

        assert "grace_enter" in _audit_events(boot, a2.slot_id), (
            "a spared client returns to ordinary grace -- it gets its grace window back"
        )
        assert turn2 >= _GRACE_S - 0.5, (
            f"restoration is a FULL fresh window, not a remainder: turn 2 held "
            f"{turn2:.2f}s of a {_GRACE_S}s window. A remainder rule would "
            f"have produced a short window here."
        )
        assert victim.grace.thread_id == "t2"
        await mgr.shutdown()


class TestDesignationStaysDerivedNeverStored:
    """Sync sibling of the class above -- deliberately NOT asyncio-marked, so
    it stays a pure source-level guard with no runtime of its own."""

    def test_the_designation_predicates_store_no_state(self):
        """⛔ THE PIN FOR THIS RULE. Restoration works only while designation
        stays derived, never remembered. This fails if someone adds a "was
        designated" flag, a paused-clock remainder, or any other carried-forward
        state inside the predicates -- which is what the no-carried-forward rule forbids (no client
        name is carried forward from an earlier decision) and what would break
        the spared-victim restoration by construction.

        A pure guard: green before AND after the fix. It defends the absence.

        WHAT THIS TEST CLAIMS. One sentence, and it is deliberately
        wider than the list of shapes below: *no state of any kind is written
        inside these two bodies.* Its complement lives in
        the designation-restoration-not-cached test, which claims
        *the verdict is not stored at any call site.* "Is the caching hazard
        closed?" is answerable ONLY against those two sentences -- never
        against a shape list, because a shape list can let a guard
        imply coverage it does not have.

        WHY IT LOOKS LIKE THIS. A check of `isinstance(t,
        ast.Attribute)` on an assignment's target is not enough: Python's target grammar
        is RECURSIVE -- Tuple/List/Starred are INTERIOR nodes -- so it would be
        type-checking the ROOT of a tree and never descending, and a bare
        `self._a, self._b = 1, 2` would walk straight through it. On the
        real `_is_designated_unload_target_locked`: such a check misses most store shapes.
        The approach is not another type in the isinstance tuple (an unbounded
        list); it is flattening to the LEAVES (one bounded rule).

        NOT COVERED, STATED SO IT IS NOT IMPLIED. Three shapes are open and
        deliberately not defended, because catching them needs unbounded
        heuristics and a guard that cries wolf gets reasoned past:
          * a bare mutating method call -- `self._c.update({...})`,
            `.append(...)`. Needs an allowlist of mutating method names or type
            inference. Not attempted.
          * rule 4 below is NAME-based, so aliasing evades it:
            `f = setattr; f(self, '_x', v)`.
          * `exec("self._x = True")` defeats any AST guard whatsoever.

        WHAT ONE FAILING RUN REPORTS. Everything it found: both
        predicates and both rule categories, in a single assertion. Problems
        are accumulated across the loop and the categories are named in the
        message rather than split across separate `assert` statements, so no
        problem can be hidden behind an earlier one. This is deliberate and
        load-bearing -- asserting inside the predicate loop would
        report only the first predicate, and using
        one assert per rule would report only the first rule. A guard that
        reports a strict subset of what it found implies a coverage it does
        not have, which is the defect this whole file exists to prevent. Do
        not split these assertions.

        DELIBERATE EXCLUSION. A bare annotation (`self._x: bool`
        with no value) is NOT flagged. It binds nothing at runtime and stores
        no state, so flagging it would be a false positive -- and a false positive
        is what trains a reader to override a guard, after which a real hit
        gets the same treatment. Do not add it back.

        INCLUDED BY REASONING, NOT BY OBVIOUS STORAGE. `del
        self._c[k]` is flagged. A delete carries no verdict forward, so under a
        narrow reading ("stores a verdict") it would not belong. The claim is
        not that -- it is that the predicate is a PURE DERIVATION, which is why
        this guard covers any write and not only writes of the verdict. A
        `del` mutates a container that outlives the call, so it breaks purity
        by the same argument that admits `self._x = 1`. Secondary and weaker:
        you do not delete from a cache that does not exist. That was a deliberate decision;
        a future reader may reasonably disagree, and should be able to see the
        reasoning rather than re-derive it.
        """

        def _leaves(target):
            """Flatten the recursive target grammar to the nodes that can store.

            Tuple/List/Starred are interior nodes, not store types. Nesting is
            unbounded: `self._a, (self._b, self._c[0]) = ...` is three stores.
            """
            if isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    yield from _leaves(element)
            elif isinstance(target, ast.Starred):
                yield from _leaves(target.value)
            else:
                yield target

        # rule 4: a CLOSED set of four names, not "any call". Anything broader
        # false-positives on ordinary reads in these bodies.
        rebinding_builtins = {
            "setattr", "delattr", "__setattr__", "__delattr__",
        }

        # Accumulated ACROSS both predicates and asserted once,
        # below, rather than per-iteration. An assert inside this loop means a
        # failure in the first predicate leaves the second one never examined --
        # a guard that can only report one of its two subjects per run.
        writes = []
        rebinding_calls = []

        for fn in (TurbohaulManager._is_designated_unload_target_locked,
                   TurbohaulManager._is_worst_ranked_loaded_locked):
            tree = ast.parse(inspect.getsource(fn).lstrip())

            # a Name leaf only persists when it has been declared to reach an
            # enclosing scope; otherwise it is a local and dies with the call.
            escaping_names = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Global, ast.Nonlocal)):
                    escaping_names.update(node.names)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    called = (
                        func.id if isinstance(func, ast.Name)
                        else func.attr if isinstance(func, ast.Attribute)
                        else None
                    )
                    if called in rebinding_builtins:
                        rebinding_calls.append(
                            f"{fn.__name__}: {ast.unparse(node)} "
                            f"(line {node.lineno})"
                        )
                    continue

                # rule 1: every binding-target field the grammar defines. This
                # list is derived from ast's own `_fields`, not from recall.
                targets = []
                if isinstance(node, (ast.Assign, ast.Delete)):
                    targets = node.targets
                elif isinstance(node, ast.AugAssign):
                    targets = [node.target]
                elif isinstance(node, ast.AnnAssign):
                    # a bare annotation stores nothing -- see the docstring's
                    # DELIBERATE EXCLUSION before adding it back.
                    targets = [node.target] if node.value is not None else []
                elif isinstance(node, (ast.For, ast.AsyncFor,
                                       ast.comprehension, ast.NamedExpr)):
                    targets = [node.target]
                elif isinstance(node, ast.withitem):
                    targets = (
                        [node.optional_vars]
                        if node.optional_vars is not None else []
                    )

                for target in targets:
                    # rules 2 and 3: flatten, then classify the leaves. Any
                    # attribute or subscript write -- self.*, r.*, self._c[k]
                    # -- is persistence beyond the call and must not appear.
                    for leaf in _leaves(target):
                        if isinstance(leaf, (ast.Attribute, ast.Subscript)):
                            writes.append(
                                f"{fn.__name__}: {ast.unparse(leaf)} "
                                f"(line {leaf.lineno})"
                            )
                        elif (isinstance(leaf, ast.Name)
                                and leaf.id in escaping_names):
                            writes.append(
                                f"{fn.__name__}: global {leaf.id} "
                                f"(line {leaf.lineno})"
                            )

        # ONE assertion over BOTH rule categories, not one assert
        # each. Two asserts would mean a body carrying a target write and a
        # `setattr` reports only the write -- the same report-completeness
        # defect as asserting inside the predicate loop, one level over.
        # Naming the categories separately in the TEXT is what makes a failure
        # say which rule fired; that never required separate statements.
        problems = []
        if writes:
            problems.append(f"state writes: {writes}")
        if rebinding_calls:
            problems.append(
                f"attribute-rebinding builtins: {rebinding_calls}"
            )

        assert not problems, (
            "the designation predicates must stay pure and read-only -- "
            "designation is re-derived, never stored. One run reports every "
            "category, across both predicates, tagged with the function each "
            "entry came from:\n"
            + "\n".join(problems)
            + "\nA setattr IS a state write: the categories are named "
            "separately so the failure says which rule fired, not because "
            "they are different claims. See the no-carried-forward rule before "
            "changing this."
        )
