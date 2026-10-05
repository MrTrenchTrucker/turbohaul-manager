"""THE SIMULTANEOUS-VICTIM MEASUREMENT (clause B of the simultaneous-victim rule).

Measurement only. **No production file is changed by this file.** It builds no
selector, touches no designation logic and drives no eviction.

===========================================================================
WHAT IS BEING MEASURED, AND WHY IT IS NOT A MATTER OF OPINION
===========================================================================
The requirement:

    Only the designated victim is torn down, immediately. If 2 or more lower-ranked clients need to be
     evicted to make space for a larger model, that is the exception, because there are
     MULTIPLE DESIGNATED VICTIMS AT THAT POINT IN TIME.

And the governing design rule, the simultaneous-victim rule of the Fast Lane design:

    "Every victim is subject to the designated-victim rule INDIVIDUALLY: each is torn down at its own turn
     boundary, WITH NO GRACE TIMER."

The rule therefore has two separable clauses, and analysis showed that only one was broken:

  * CLAUSE A (the COUNT) — "two or more victims … until the claim can be satisfied" —
    ALREADY WORKS, iteratively: `_route_or_reserve` evicts one victim per admission pass
    and is re-entered, VRAM credit accumulating across passes.
  * CLAUSE B (the TREATMENT) — every victim designated, none of them holding a grace
    timer — IS THE DEFECT. `_is_worst_ranked_loaded_locked` takes a single `max()` and
    returns `victim is r`, so AT MOST ONE resident is ever the designated victim.

That conclusion was reached by READING. **This file turns it into a measurement.**

===========================================================================
⭐ WHY THIS FIXTURE LOOKS DELIBERATELY UNLIKE ITS SIBLING
===========================================================================
The sequential two-victim grace test measures the same rule and PASSES. It passes
because it parks victim B (`state = IDLE_EVICTABLE`) **before** driving victim A, and
`_is_worst_ranked_loaded_locked` filters to `{ACTIVE, GRACE}` — so with B gone from the
loaded pool, A is trivially "worst among one". That file is honest about it and says so in
its own words ("trivially True (worst-among-one)"), and its sequential
ordering is a faithful model of how eviction actually reaches residents.

But a sequential fixture can only ever observe TWO CONSECUTIVE SINGLE-VICTIM DESIGNATIONS.
It cannot observe the thing the requirement is about: **multiple designated victims
AT THAT POINT IN TIME.**

⛔ SO THE ONE RULE OF THIS FILE: **both victims stay loaded, simultaneously, for the whole
measurement.** Nobody is parked. Nothing is evicted. If a future edit makes this file park a
resident before probing, it has silently become its sibling and measures nothing new.

===========================================================================
THE SCENARIO
===========================================================================
    claimant  10.0.0.1  rule_index 0   strictly outranks both, layer-split, needs BOTH
    victim A  10.0.0.2  rule_index 1   BETTER-ranked  -> would be taken SECOND ("victim 2")
    victim B  10.0.0.3  rule_index 2   WORSE-ranked   -> would be taken FIRST  ("victim 1")

Both residents are `ResidentState.ACTIVE` at the instant every probe below is taken.

VERIFIED against the current code, and it is why ACTIVE is the only state that will do:
`ResidentState.GRACE` has **ZERO assignments anywhere in manager.py** — grep returns
nothing. Resident-level grace is a documented-legal state that nothing ever enters. The real
grace mechanism is per-SLOT (`SlotState.GRACE` + `slot.grace_started_at`) plus the
resident's own `GraceTimer` (`r.grace`). So the designation filter reduces to
`state is ACTIVE` by construction, and "victim 2 holds a live grace window" has to be
observed on the SLOT and the TIMER, never on the resident's state.

===========================================================================
WHAT THIS FILE DELIBERATELY DOES NOT DO
===========================================================================
* It does NOT touch, read for a conclusion, or infer from `grace_fastlane_breakout`
  (in either grace loop, including the deprecated one). That mechanism is tracked
  separately. This file therefore **asserts the narrow claim and OBSERVES the wide one**:
  whether victim 2 is DESIGNATED while its window is live is true-or-false independently of
  how long that window lasts. `_measure_both_victims_loaded` records the window's remaining
  time and the audit trail so the outcome is reported rather than assumed — see
  `test_victim_two_holds_a_live_grace_window_while_victim_one_is_designated`.
* It does NOT assert anything about the cap<=1 path (`_process_slot`), which is
  deprecated. cap>=2 only.
* It does NOT edit its sibling file. Not one character.

HARNESS REUSE, said plainly: the boot/runtime/manifest/handle/mocks scaffolding below is
MODELLED ON the sequential two-victim grace test's, deliberately, so the two files
are comparable. It is REUSED, not invented. What is new is the predicate recorder, the
simultaneous-probe helper, and the structural need-for-two control.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest
import yaml

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
from turbohaul.fastlane import match_fastlane
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle

# List index IS the priority. Claimant (index 0) strictly outranks both victims.
# Victim A (index 1) is BETTER-ranked than victim B (index 2), so B is "victim 1"
# (taken first, per the lowest-priority-first rule) and A is "victim 2".
_RULES = [
    FastLaneRule(address="10.0.0.1", tag_ranks=FastLaneTagRanks(main=1)),  # claimant
    FastLaneRule(address="10.0.0.2", tag_ranks=FastLaneTagRanks(main=1)),  # victim A (better)
    FastLaneRule(address="10.0.0.3", tag_ranks=FastLaneTagRanks(main=1)),  # victim B (worse)
]

_GRACE_SECONDS = 5

# The two ways `_serve_on_resident` can announce that it has DECIDED this
# anchor's grace outcome: it entered a window, or it was skipped for a designated
# victim. Both are announced through `_audit_async`, in that one function, and exactly
# one of them happens per serve.
#
# ⛔ WHY WAITING ON `SlotState.GRACE` ALONE IS WRONG. If victim 2 were not designated,
# `_serve_on_resident` would take the else-branch and the anchor would really
# enter GRACE — so a wait on that state would be reachable. Victim 2 IS
# designated, so the skip branch runs, and the anchor goes ACTIVE -> POPPED without ever
# being in GRACE. This file's own docstring describes exactly that, as
# what SIMULATING that designation produces. That designation is production behaviour, and a wait on
# GRACE alone would be a wait for something that cannot happen.
_GRACE_DECISION_EVENTS = ("grace_enter", "grace_designated_victim_skip")

# The budget for that wait. KEPT AT 3.0s ON PURPOSE, not tuned and not raised:
# the decision is reached within milliseconds, so this leaves ample headroom. It is worth
# saying why it was left alone — a wait whose condition cannot occur would make
# EVERY budget equivalent, because the loop would always fall
# through to its `task.done()` exit instead. Since the condition is reachable, this
# bound guards something real, and 3.0s is a generous bound on it.
_GRACE_DECISION_BUDGET_S = 3.0


def _boot_runtime(tmp_path):
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
            default_port_base=59800,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False,
            # LOAD-BEARING: cap<=1 routes to _process_slot, which has no designation
            # call anywhere in its body and is DEPRECATED. This rule needs
            # cap>=2. Do not "simplify" this to the QueueConfig default of 1.
            max_parallel_sidecars=2,
            grace_seconds=_GRACE_SECONDS,
            max_grace_extensions=50,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=_RULES),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag, *, expected_vram_mib=0, split_mode="none", main_gpu=0):
    (boot.storage.manifests_path / f"{model_tag}.yaml").write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": expected_vram_mib * 1024 * 1024,
        "context_size": 2048,
        "expected_vram_bytes": expected_vram_mib * 1024 * 1024,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


def _make_fake_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 88_000 + port
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _mocks():
    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return dict(
        spawn_fn=lambda *a, **k: None,
        health_fn=fake_health,
        sigterm_fn=fake_sigterm,
        vram_fn=fake_vram,
        complete_fn=fake_complete,
    )


def _audit_events(boot, slot_id):
    conn = open_state_db(boot.storage.state_db_path)
    cur = conn.execute(
        "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
        (slot_id,),
    )
    events = [r["event_type"] for r in cur.fetchall()]
    conn.close()
    return events


def _record_designation_answers(monkeypatch, mgr):
    """Wrap the REAL `_is_designated_unload_target_locked` and record every
    ``(model_tag, answer)`` it produces — production's own calls included.

    ⭐ WHY A RECORDER AND NOT ONLY DIRECT PROBES. A direct probe tells you what the
    predicate answers when THIS FILE asks it. It does not tell you what production
    asked or what production got. `_serve_on_resident` consults the predicate itself
    (its `skip_grace` decision), and the answer IT receives is the one that decides
    whether victim 2 enters a grace window at all. Recording production's own call is
    the difference between measuring the mechanism and measuring a bystander — the
    known lesson: spy on the mechanism's own entry point, not just the outcome.

    Wraps the CLASS attribute, so it captures calls made through `self` from inside
    manager code, not only calls this file makes on the instance.
    """
    log: list[tuple[str, bool]] = []
    real = type(mgr)._is_designated_unload_target_locked

    def recording(self, r):
        answer = real(self, r)
        log.append((r.model_tag, bool(answer)))
        return answer

    monkeypatch.setattr(type(mgr), "_is_designated_unload_target_locked", recording)
    return log


def _record_grace_decision(monkeypatch, mgr, anchor):
    """Record, IN PROCESS, the moment production decides this anchor's
    grace outcome — either branch — and return the list it appends to.

    ⛔ WHY THIS DOES NOT POLL THE AUDIT TABLE. `_audit_events` calls `open_state_db`,
    whose `PRAGMA journal_mode=WAL` in `src/turbohaul/state.py` is where this
    file's intermittent `OperationalError: database is locked` comes from (nearly all
    of its failures across repeated clean runs; tracked separately). Polling that in a 10ms loop would open
    the state database ~300 times per wait and multiply this file's exposure to the
    exact defect it is trying to keep out. **This spy opens no database and does no
    I/O.**

    ⭐ RECORDED BEFORE DELEGATING TO THE REAL METHOD, deliberately. The decision has
    already been made by the time the announcement is attempted, and the announcement
    is the part that can fail on the lock. Recording after would make this wait depend
    on the success of the very write that flakes — the wait would time out because the
    database was busy, which is precisely the misattribution this change exists to remove.

    Level-triggered by construction: a list, not an `asyncio.Event`, so a decision made
    before the first poll is still visible to it and there is no notify-before-wait race.

    Wraps the CLASS attribute, matching `_record_designation_answers` above, so it
    captures the call `_serve_on_resident` makes through `self`.
    """
    seen: list[str] = []
    real = type(mgr)._audit_async

    async def recording(self, slot, event_type):
        if slot is anchor and event_type in _GRACE_DECISION_EVENTS:
            seen.append(event_type)
        await real(self, slot, event_type)

    monkeypatch.setattr(type(mgr), "_audit_async", recording)
    return seen


async def _await_grace_decision(anchor, task, decisions):
    """Wait until production has decided this anchor's grace outcome, or the
    serving task has finished — and **FAIL, NAMING THE TIMEOUT, IF NEITHER HAPPENS.**

    ⛔ THE DEFECT THIS AVOIDS. A wait that polls to a 3.0s
    deadline and, on expiry, simply falls through lets everything after it run against
    a state the fixture never reached. Two failure modes come out of that, and the
    one that bites is NOT the loud one:

      * TODAY — a **FALSE PASS**. Forcing both deadlines to 0.0 so
        the wait always expires leaves this file **fully passing**. Designation is registry
        state that is already true before the serving task runs, so the simultaneous-victim assertion is
        satisfied whether or not the precondition ever held. **A test that passes when
        its precondition timed out is not measuring anything.**
      * LATENT — a **FALSE ACCUSATION**. The moment designation becomes dependent on the
        serving task's progress, the same fall-through reaches
        ``assert a_designated is True`` and prints "CLAUSE B OF THE SIMULTANEOUS-VICTIM RULE IS NOT IMPLEMENTED": an
        accusation about production code, for a timed-out precondition.

    Both are the same bug — continuing after a precondition times out — and the fix for
    both is to refuse to continue. Returns the elapsed seconds so a caller may report it;
    raises through `pytest.fail` on expiry so nothing downstream runs.
    """
    started = time.monotonic()
    deadline = started + _GRACE_DECISION_BUDGET_S
    timed_out = True
    while time.monotonic() < deadline:
        if decisions:
            timed_out = False
            break
        if task.done():
            timed_out = False
            break
        await asyncio.sleep(0.01)

    elapsed = time.monotonic() - started
    if timed_out:
        exc = None
        if task.done() and not task.cancelled():
            exc = task.exception()
        pytest.fail(
            "PRECONDITION TIMED OUT — THIS IS NOT A VERDICT ON THE SIMULTANEOUS-VICTIM RULE.\n"
            f"  waited .................. {_GRACE_DECISION_BUDGET_S:.1f}s "
            f"(elapsed {elapsed:.2f}s)\n"
            f"  waiting for ............. production to decide this anchor's grace "
            f"outcome, i.e. an `_audit_async` of one of {_GRACE_DECISION_EVENTS!r}\n"
            f"  decisions observed ...... {decisions!r}  (empty = never reached)\n"
            f"  anchor state ............ {anchor.state}\n"
            f"  anchor grace_started_at . {anchor.grace_started_at}\n"
            f"  serving task done ....... {task.done()}\n"
            f"  serving task exception .. {exc!r}\n"
            "  NOTHING BELOW THIS LINE RAN. Do not read this as 'clause B of the simultaneous-victim rule is not "
            "implemented' — that assertion was never reached and this says nothing "
            "about it.\n"
            "  ⚠ MOST LIKELY CAUSE: `src/turbohaul/state.py` — "
            "`PRAGMA journal_mode=WAL` raising `OperationalError: database is locked`, "
            "which kills the serving task. Across repeated clean runs of this file that "
            "accounted for every observed failure and this deadline accounted for none. "
            "Tracked separately.\n"
            "  ◊ AND IT IS NOT `busy_timeout`, so it is not re-derived needlessly: "
            "the retry-wait IS armed at that statement. `sqlite3.connect` sets it from "
            "its own `timeout` argument, which defaults to 5.0s, six lines before "
            "that statement — verified by reading `PRAGMA busy_timeout` on a fresh connection "
            "before any PRAGMA runs, where it already reads 5000. The explicit "
            "`PRAGMA busy_timeout=5000` is redundant, not late.\n"
            "  This is not this test, and it is not the simultaneous-victim rule."
        )
    return elapsed


async def _build_two_loaded_victims(tmp_path, monkeypatch, *, claimant_split_mode="layer"):
    """The scenario, with BOTH victims loaded and ACTIVE. Nobody is parked.

    Returns (boot, mgr, victim_a, victim_b, handle_a, log). Victim B is the
    worse-ranked one ("victim 1"); victim A is the better-ranked one ("victim 2").

    ``claimant_split_mode`` selects WHICH HALF of the contract the scenario is in,
    and it is the only difference between them — same residents, same ranks, same
    live claim. ``"layer"`` (the default, used by the two-victim tests) is the rule's own
    premise: a claimant that cannot co-reside,
    so the victim SET is taken. ``"none"`` is a claim that needs exactly ONE victim,
    where the single-victim branch must decide and a non-worst resident must still
    answer False. Parameterising the CLAIMANT rather than writing a second fixture
    is deliberate: it makes the two halves differ in exactly one input, so nothing
    else can be what explains a different answer.
    """
    boot, runtime = _boot_runtime(tmp_path)
    # "A model that does not fit on a single card" — the rule's own premise. A layer-split
    # claimant is the shape that cannot co-reside; see the need-for-two control for
    # the measured reason both victims are required.
    _seed_manifest(boot, "claimant", expected_vram_mib=15000,
                   split_mode=claimant_split_mode)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    log = _record_designation_answers(monkeypatch, mgr)

    # A REAL, live Fast Lane claim through the real registration function.
    claimant_slot = Slot.new("claimant", prompt="hi", thread_id="t-claimant",
                             client_meta={"ip": "10.0.0.1"})
    claimant_slot.fastlane = match_fastlane(
        mgr._fastlane_table(), "10.0.0.1", {"ip": "10.0.0.1"})
    assert claimant_slot.fastlane is not None, (
        "test setup bug: the claimant must genuinely match a Fast Lane rule, or the "
        "whole designated-victim-rule precondition is absent and every designation answer below is "
        "trivially False for the wrong reason"
    )
    async with mgr._registry_lock:
        mgr._register_fastlane_claim_locked(claimant_slot, "claim_setup")
    assert ("10.0.0.1", "claimant") in mgr._fastlane_claims, (
        "test setup bug: the claim did not register"
    )

    now = time.monotonic()
    handle_a = _make_fake_handle("modelA", 59800)
    handle_b = _make_fake_handle("modelB", 59801)
    victim_a = Resident(
        model_tag="modelA", resident_key="modelA", state=ResidentState.ACTIVE,
        main_gpu=0, split_mode="none", reserved_need_mib=10000,
        last_active_monotonic=now, rank_client_meta={"ip": "10.0.0.2"},
        handle=handle_a,
        grace=GraceTimer(grace_seconds=_GRACE_SECONDS, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=0),
    )
    victim_b = Resident(
        model_tag="modelB", resident_key="modelB", state=ResidentState.ACTIVE,
        main_gpu=1, split_mode="none", reserved_need_mib=10000,
        last_active_monotonic=now, rank_client_meta={"ip": "10.0.0.3"},
        handle=handle_b,
        grace=GraceTimer(grace_seconds=_GRACE_SECONDS, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=0),
    )
    # Insertion order deliberately OPPOSITE the expected pick order, so insertion
    # order cannot be what decides the answer.
    mgr._residents["modelA"] = victim_a
    mgr._residents["modelB"] = victim_b

    assert victim_a.state is ResidentState.ACTIVE
    assert victim_b.state is ResidentState.ACTIVE
    return boot, mgr, victim_a, victim_b, handle_a, log


@pytest.mark.asyncio
class TestTheInstrumentsCanReportBadNews:
    """Controls. Every zero and every False below is worthless without these."""

    async def test_the_recorder_actually_records_so_an_empty_log_means_something(
        self, tmp_path, monkeypatch
    ):
        """INSTRUMENT CONTROL. If the recorder is unwired, every assertion in this
        file about "what production answered" reads an empty list and passes for
        nothing. Prove it captures a real call before trusting a single entry."""
        boot, mgr, victim_a, victim_b, _handle_a, log = await _build_two_loaded_victims(
            tmp_path, monkeypatch)
        try:
            assert log == [], "nothing should have been recorded before the first probe"
            async with mgr._registry_lock:
                mgr._is_designated_unload_target_locked(victim_b)
            assert log, (
                "THE RECORDER IS UNWIRED. A call was made through the real predicate "
                "and nothing was captured, so every 'production answered X' assertion "
                "in this file is vacuous."
            )
            assert log[-1][0] == "modelB"
        finally:
            await mgr.shutdown()

    async def test_the_grace_decision_recorder_records_and_filters_by_anchor(
        self, tmp_path, monkeypatch
    ):
        """INSTRUMENT CONTROL for the grace-decision wait, and the sibling of the recorder
        control above.

        The two measurement tests now wait on `_record_grace_decision`. If that recorder
        were unwired the wait would fall back to its `task.done()` exit, everything would
        still pass, and the file would have quietly returned to waiting on nothing — the
        exact defect this wait was added to remove. **A recorder nobody proved can
        record is indistinguishable from one that cannot.**

        Two properties, and both are load-bearing:
          * it RECORDS — a real decision made by production is captured;
          * it FILTERS — a decision belonging to a DIFFERENT slot is not.
        Without the second, a recorder that captured every slot's events would satisfy
        the first while making the wait fire on somebody else's decision.

        ⛔ WHY THIS DRIVES NO SERVE. Proving these properties by running a third
        `_serve_on_resident` raises the intermittent-failure rate measurably, because
        a serve issues several audit writes and
        every one is another chance at the state-database lock in `state.py`. Adding
        exposure to the defect that this file already suffers from intermittently, in
        a change meant to stop it, is a bad trade for a control.
        So this exercises the recorder DIRECTLY, exactly as its sibling above exercises
        the predicate directly — and the complementary claim, that *production's own
        serve* announces through this seam, is asserted where a serve already happens,
        in `test_the_scenario_is_well_formed_with_both_victims_loaded_at_once`. Together
        they cover the seam at no extra cost.
        """
        boot, mgr, victim_a, victim_b, _handle_a, _log = await _build_two_loaded_victims(
            tmp_path, monkeypatch)
        anchor_a = Slot.new("modelA", prompt="hi", thread_id="t-a",
                            client_meta={"ip": "10.0.0.2"})
        bystander = Slot.new("modelB", prompt="hi", thread_id="t-bystander",
                             client_meta={"ip": "10.0.0.3"})
        try:
            # ⛔ THE REAL AUDIT WRITE IS STUBBED OUT FIRST, AND THAT IS THE POINT.
            # `_audit_async` opens the state database (see `state.py`). With
            # these three announcements writing for real, this file is
            # more failure-prone.
            # Everything this control asserts is a property of the RECORDER's own logic
            # — what it captures and what it refuses — and none of it needs a row on
            # disk. The complementary claim, that the recorder wraps the REAL
            # `_audit_async` and that production announces through it, is asserted
            # against the real thing in the scenario-control test, where a serve is
            # already happening and costs nothing extra.
            async def _no_write(self, slot, event_type):
                return None

            monkeypatch.setattr(type(mgr), "_audit_async", _no_write)

            # Armed in this order on purpose: the second wraps the first, so ONE event
            # stream feeds BOTH recorders and the include/exclude comparison is made
            # over identical input rather than over two different runs.
            mine = _record_grace_decision(monkeypatch, mgr, anchor_a)
            theirs = _record_grace_decision(monkeypatch, mgr, bystander)
            assert mine == [] and theirs == [], "nothing should be recorded yet"

            # RECORDS: a decision announced for this anchor is captured.
            await mgr._audit_async(anchor_a, "grace_designated_victim_skip")
            assert mine == ["grace_designated_victim_skip"], (
                "THE GRACE-DECISION RECORDER IS UNWIRED. A decision was announced "
                f"through the real `_audit_async` and {mine!r} was captured — so the "
                "wait in both measurement tests falls through to its `task.done()` "
                "exit and is once again waiting on nothing."
            )
            assert theirs == [], (
                "THE RECORDER DOES NOT FILTER BY ANCHOR. A slot that was never "
                f"announced about captured {theirs!r}, so the wait could be satisfied "
                "by a decision belonging to some other slot."
            )

            # FILTERS BOTH WAYS: the bystander's own decision goes to its recorder and
            # not to the anchor's. A recorder that captured everything would pass the
            # two assertions above and fail here.
            await mgr._audit_async(bystander, "grace_enter")
            assert theirs == ["grace_enter"], f"bystander recorder missed its own: {theirs!r}"
            assert mine == ["grace_designated_victim_skip"], (
                f"the anchor's recorder captured another slot's decision: {mine!r}"
            )

            # IGNORES NON-DECISIONS: the wait must not be satisfied by any other audit
            # event this anchor emits — `active` is emitted first, on every serve.
            await mgr._audit_async(anchor_a, "active")
            assert mine == ["grace_designated_victim_skip"], (
                "THE EVENT FILTER IS OPEN. A non-decision event was recorded as one, so "
                f"the wait would be satisfied before the decision is made: {mine!r}"
            )
        finally:
            await mgr.shutdown()

    async def test_the_claim_genuinely_needs_both_victims(self, tmp_path, monkeypatch):
        """ANTI-VACUITY FOR THE WHOLE FILE, and it is the load-bearing control.

        If this claim could be satisfied by evicting ONE resident, then victim 2 NOT
        being designated would be CORRECT behaviour and this file would be measuring
        nothing at all. The rule only speaks about claims that need two or more.

        ⭐ HOW IT IS PROVEN, AND WHY NOT BY VRAM ARITHMETIC. The obvious proof —
        compute `need_mib`, compare against what each eviction would release — would
        measure a mechanism this branch never consults. `_vram_admits_locked`
        (in manager.py) refuses ANY `split_mode != 'none'` claimant
        unconditionally while ANY sibling exists, and never reads `need` on that
        branch. So for a layer-split claimant the real stopping condition is "until
        ZERO siblings remain", and a need-arithmetic proof would be a plausible number
        describing the wrong mechanism (a known lesson). Verified
        here by reading the function directly, and measured below
        by calling the REAL function.

        Nobody is parked and nothing is evicted: the sibling count is varied by asking
        the real gate the same question against different registry contents.
        """
        boot, mgr, victim_a, victim_b, _handle_a, log = await _build_two_loaded_victims(
            tmp_path, monkeypatch)
        try:
            need, parallel, main_gpu, split_mode, _sleep, _auto = (
                mgr._read_model_footprint("claimant"))
            assert split_mode == "layer", (
                "test setup bug: the claimant must be the does-not-fit-on-one-card "
                f"shape the simultaneous-victim rule is about; got split_mode={split_mode!r}"
            )

            async with mgr._registry_lock:
                with_both = mgr._vram_admits_locked(need, parallel, main_gpu, split_mode)
                # One sibling: still refused. Evicting only victim 1 does NOT satisfy it.
                del mgr._residents["modelB"]
                with_one = mgr._vram_admits_locked(need, parallel, main_gpu, split_mode)
                # Zero siblings: admitted. So BOTH had to go.
                del mgr._residents["modelA"]
                with_none = mgr._vram_admits_locked(need, parallel, main_gpu, split_mode)
                mgr._residents["modelA"] = victim_a
                mgr._residents["modelB"] = victim_b

            assert with_both is False, (
                "premise broken: the claim is admissible with BOTH victims still "
                "loaded, so nothing needs evicting and this file measures nothing."
            )
            assert with_one is False, (
                "PREMISE BROKEN — THIS IS NOT A TWO-VICTIM SCENARIO. The claim became "
                "admissible after removing only victim 1, so victim 2 is not required "
                "and its not being designated is CORRECT, not a defect. Everything "
                "else in this file would be measuring the wrong thing."
            )
            assert with_none is True, (
                "premise broken in the other direction: the claim is refused even with "
                "zero siblings loaded, so the refusal is not about co-residence at all "
                "and 'both victims are needed' has not been shown."
            )
        finally:
            await mgr.shutdown()


@pytest.mark.asyncio
class TestClauseBBothVictimsLoadedSimultaneously:
    """THE MEASUREMENT. Both victims loaded at the same instant, under one live claim."""

    async def test_victim_one_is_designated_while_both_are_loaded(
        self, tmp_path, monkeypatch
    ):
        """ANTI-VACUITY PARTNER for the measurement below.

        A predicate that answered False for EVERYTHING — a dead claim, an unmatched
        rule, a broken registry — would make "victim 2 answers False" true for a
        reason that has nothing to do with the rule. Victim 1 answering True in the SAME
        snapshot, with both residents loaded, is what rules that out."""
        boot, mgr, victim_a, victim_b, _handle_a, log = await _build_two_loaded_victims(
            tmp_path, monkeypatch)
        try:
            async with mgr._registry_lock:
                b_designated = mgr._is_designated_unload_target_locked(victim_b)
                a_designated = mgr._is_designated_unload_target_locked(victim_a)
            assert b_designated is True, (
                "victim 1 (the WORSE-ranked resident, modelB) is not designated even "
                "though a strictly-outranking live claim exists and both residents are "
                "loaded. The designated-victim rule's precondition or the rank ordering is not what this "
                f"fixture believes. answers so far: {log!r}"
            )
            # Recorded, NOT asserted as correct — this is the defect under measurement
            # and the next test is where it is judged against the requirement.
            assert a_designated in (True, False)
        finally:
            await mgr.shutdown()

    async def test_the_scenario_is_well_formed_with_both_victims_loaded_at_once(
        self, tmp_path, monkeypatch
    ):
        """THE SCENARIO CONTROL. Everything asserted here stays TRUE AFTER THE FIX.

        Its job is to make the strict-xfail below MEANINGFUL: that test's failure is
        only evidence of a defect if the scenario genuinely put two victims in the
        loaded pool at once, under a live outranking claim, and genuinely asked the
        predicate about victim 2. This test proves all of that and nothing more.

        ⛔ WHY IT DELIBERATELY DOES NOT ASSERT `victim 2 answered False`, OR THAT IT
        ENTERED A GRACE WINDOW. Both are true today and both are properties OF THE
        DEFECT — with the fix simulated, victim 2 is seen becoming designated and
        emitting `grace_designated_victim_skip` instead of entering grace at all. An
        assertion on either would pin today's defect as expected behaviour and go red
        when the fix lands, which is a known failure mode: a test that asserts a
        defect has to be re-pointed by the very change that fixes it. Exactly ONE test
        in this file is allowed to flip when the fix lands, and it is the strict xfail
        below. Those two observations are RECORDED and PRINTED here, never asserted.

        ⚠ ASSERTED vs OBSERVED. The mechanism that could end a grace window early is
        tracked separately and is not read, inferred from, or asserted about here. The
        window's remaining time and the audit trail are reported so the outcome is
        visible rather than assumed.
        """
        boot, mgr, victim_a, victim_b, handle_a, log = await _build_two_loaded_victims(
            tmp_path, monkeypatch)
        anchor_a = Slot.new("modelA", prompt="hi", thread_id="t-a",
                            client_meta={"ip": "10.0.0.2"})
        anchor_a.state = SlotState.STAGED
        # Arm the recorder BEFORE the serving task starts, or a decision
        # reached during task startup would be missed. It is level-triggered, so arming
        # it early is always safe and arming it late is not.
        grace_decision = _record_grace_decision(monkeypatch, mgr, anchor_a)
        task = asyncio.create_task(mgr._serve_on_resident(victim_a, anchor_a, handle_a))
        try:
            # Wait for the grace DECISION — either branch — and fail naming
            # the timeout if it never comes. This replaces a poll for
            # `SlotState.GRACE`, a state the designated-victim skip means this anchor
            # no longer reaches, whose expiry fell through into the assertions below.
            await _await_grace_decision(anchor_a, task, grace_decision)

            async with mgr._registry_lock:
                a_designated = mgr._is_designated_unload_target_locked(victim_a)
                b_designated = mgr._is_designated_unload_target_locked(victim_b)

            # === ASSERTED: all of this stays true after the fix ===
            # Both still loaded at the instant of the probe. This is the whole point.
            assert victim_a.state is ResidentState.ACTIVE, (
                "victim 2 left the loaded pool before it was probed."
            )
            assert victim_b.state is ResidentState.ACTIVE, (
                "victim 1 left the loaded pool, so this fixture has silently become "
                "its sequential sibling and measures two consecutive single-victim "
                "designations instead of one simultaneous pair."
            )
            assert b_designated is True, (
                "victim 1 is not designated at the probe instant, so the designated-victim rule never "
                f"engaged and this scenario proves nothing. answers: {log!r}"
            )
            # PRODUCTION'S OWN CALL, not just this file's probes: `_serve_on_resident`
            # consulted the predicate about victim 2 and acted on whatever it returned.
            # Asserted on the QUESTION, never on the ANSWER — the question is asked
            # both before and after the fix; only the answer changes.
            assert any(tag == "modelA" for tag, _ in log), (
                "production never asked the predicate about victim 2, so anything this "
                "file concludes about victim 2 is this file talking to itself rather "
                f"than a measurement of the mechanism. answers: {log!r}"
            )
            # The OTHER half of the seam both waits depend on — that
            # production's own serve announces its grace decision through
            # `_audit_async`. Costs nothing: this test already ran a serve.
            # ⚠ GUARDED ON A CLEAN COMPLETION on purpose. A serve killed by the
            # `state.py:100` lock legitimately never reaches its decision, and firing
            # here on that would rebuild the misattribution this change removes.
            # ⛔ Asserts THAT a decision was announced, never WHICH — the branch is
            # allowed to change; the seam is not.
            if task.done() and not task.cancelled() and task.exception() is None:
                assert grace_decision, (
                    "a serve completed cleanly WITHOUT announcing a grace decision "
                    "through `_audit_async`. The seam that `_await_grace_decision` "
                    "waits on has moved, so both waits in this file are now falling "
                    "through to their `task.done()` exit and waiting on nothing. "
                    f"expected one of {_GRACE_DECISION_EVENTS!r}"
                )

            # === OBSERVED ONLY — recorded and printed, never asserted. Both of these
            # are properties of the DEFECT and both change when the fix lands. ===
            print(
                f"\nOBSERVED (not asserted):"
                f"\n  victim2 designated .......... {a_designated}"
                f"\n  victim2 slot state .......... {anchor_a.state}"
                f"\n  victim2 grace remaining_s ... {victim_a.grace.remaining_s():.2f}"
                f" of {_GRACE_SECONDS}"
                f"\n  victim2 audit trail ......... {_audit_events(boot, anchor_a.slot_id)!r}"
                f"\n  designation answers ......... {log!r}"
                # OBSERVED, never asserted — proof the in-process recorder
                # this file now waits on is LIVE. Asserting which branch fired would pin
                # today's post-fix behaviour and go red when it legitimately
                # changes, which is the disease the docstring above forbids.
                f"\n  grace decision observed ..... {grace_decision!r}"
            )
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await mgr.shutdown()

    async def test_victim_two_must_also_be_designated_under_the_simultaneous_victim_rule(
        self, tmp_path, monkeypatch
    ):
        """THE REQUIREMENT, ASSERTED — and since the designation fix it PASSES.

        Identical scenario to the test above. The only difference is what is asserted:
        there, what the code does today; here, what the requirement says it must do.
        The gap between the two WAS the defect, and it was measured rather than argued.

        DESIGN NOTE. While a defect of this kind exists, its test is carried as
        ``@pytest.mark.xfail(strict=True)`` — a measured defect rather than an
        accepted behaviour. ``strict=True`` is the point: once the designation fix
        lands, this XPASSES, the suite goes RED, and whoever lands it must come
        here and delete the marker, as the LAST step of
        the fix and only once the fix is green — the marker is never removed to make
        a build pass. The assertion below is the requirement itself, unchanged;
        this test carries no marker because the fix is in place.

        It is now a plain regression guard for the simultaneous-victim rule's clause B. If a future change
        makes ``_is_designated_unload_target_locked`` answer for only one resident again, this
        goes red on the requirement itself, with the same message it carried as a defect.
        """
        boot, mgr, victim_a, victim_b, handle_a, log = await _build_two_loaded_victims(
            tmp_path, monkeypatch)
        anchor_a = Slot.new("modelA", prompt="hi", thread_id="t-a",
                            client_meta={"ip": "10.0.0.2"})
        anchor_a.state = SlotState.STAGED
        # Arm the recorder BEFORE the serving task starts, or a decision
        # reached during task startup would be missed. It is level-triggered, so arming
        # it early is always safe and arming it late is not.
        grace_decision = _record_grace_decision(monkeypatch, mgr, anchor_a)
        task = asyncio.create_task(mgr._serve_on_resident(victim_a, anchor_a, handle_a))
        try:
            # Wait for the grace DECISION — either branch — and fail naming
            # the timeout if it never comes. This replaces a poll for
            # `SlotState.GRACE`, a state the designated-victim skip means this anchor
            # no longer reaches, whose expiry fell through into the assertions below.
            await _await_grace_decision(anchor_a, task, grace_decision)

            async with mgr._registry_lock:
                a_designated = mgr._is_designated_unload_target_locked(victim_a)
                b_designated = mgr._is_designated_unload_target_locked(victim_b)

            assert b_designated is True, (
                "victim 1 must be designated — if it is not, this scenario never "
                f"engaged the designated-victim rule at all. answers: {log!r}"
            )
            assert a_designated is True, (
                "CLAUSE B OF THE SIMULTANEOUS-VICTIM RULE IS NOT IMPLEMENTED. Victim 2 answered False to "
                "_is_designated_unload_target_locked while victim 1 answered True and BOTH "
                "were loaded at that instant — so this claim, which needs both, has "
                "exactly ONE designated victim. Per the requirement there must "
                "be 'multiple designated victims at that point in time', and per the simultaneous-victim "
                "rule every victim is subject to the designated-victim rule individually with no grace timer. "
                f"Victim 2 held a live grace window with "
                f"{victim_a.grace.remaining_s():.2f}s remaining. "
                f"designation answers: {log!r}"
            )
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await mgr.shutdown()


@pytest.mark.asyncio
class TestASingleVictimClaimStillTakesExactlyOneVictim:
    """THE OTHER HALF OF THE CONTRACT — the rule must NOT engage for a claim needing one.

    THIS GAP IS REAL: nothing else pinned it. Everything above
    measures the TWO-victim case, where the set branch is correct and ``any(...)``
    is the right answer. Nothing pinned the single-victim case, so a
    set-widening change — the guard in ``manager.py`` replaced with ``if False:``, forcing
    the SET branch for EVERY claim — would go unnoticed by the two-victim tests: they pass while every
    loaded resident answered True to any claim at all.

    That is exactly what the design constraint forbids: a change that can select a
    resident the single max() would never have selected is a violation, not
    a bonus.

    ⚠ AND IT IS A DIFFERENT QUANTITY FROM THE ONE THE POOL-WIDENING GUARD COVERS. Widening the
    POOL — which residents are candidates at all — is caught by the pre-existing
    ``test_idle_evictable_is_never_the_victim``. Set-widening leaves the pool alone and
    widens WHO INSIDE IT IS DESIGNATED. The fix's "the pool cannot widen,
    structurally" argument is sound for the pool and says nothing about the count —
    and the count is the quantity clause B actually governs. Same words, different
    quantity, and the shared-helper shape does not protect this half.

    SCOPE, STATED. This probes the predicate directly under ``_registry_lock``
    rather than through ``_serve_on_resident``. Production's own call is covered by
    the two-victim measurement above; the single-victim contract is a property of
    the predicate itself.

    ⚠ THE VERDICT IS ASSERTED AFTER TEARDOWN, AND THAT IS LOAD-BEARING. Asserting
    inside the ``try`` is wrong, for this reason:
    the target assertion fires correctly, then ``mgr.shutdown()`` in the ``finally``
    can raise ``sqlite3.OperationalError: database is locked`` from
    ``audit_db_session`` -> ``open_state_db`` (``state.py``; tracked separately).
    ◊ NOTE ON THE MECHANISM: the lock error is NOT a late-armed busy_timeout.
    The retry-wait
    is already armed at that statement — ``sqlite3.connect`` sets it from its own
    ``timeout`` argument, default 5.0s, six lines earlier, and a fresh connection
    reads ``PRAGMA busy_timeout`` = 5000 before any PRAGMA runs. The explicit
    ``PRAGMA busy_timeout=5000`` is redundant, not late. The point below
    is unaffected by which mechanism applies. The
    AssertionError is still in the report, but the TAIL of the traceback is the
    sqlite error, and on a PASSING run that teardown would redden this guard for a
    reason having nothing to do with designation. Capturing the answers, tearing
    down, and asserting afterwards means a cleanup failure can never replace or
    outrank the verdict.

    It does NOT swallow that error, and deliberately so: this file's other tests
    carry the same exposure, and hardening the observer would hide a real
    production fault whose repair belongs in ``state.py``. Residual, honestly
    named: if ``shutdown()`` raises, this guard goes red without reaching its
    assertions — false-red only, never false-green.
    """

    async def test_a_claim_needing_one_victim_designates_the_worst_and_only_the_worst(
        self, tmp_path, monkeypatch
    ):
        """Claim needs ONE. Two residents loaded. Worst True, non-worst False.

        Identical to the two-victim scenario in every input but one: the claimant's
        manifest says ``split_mode: none``, so it CAN co-reside and the rule's premise is
        absent. The victim set must therefore be exactly the single worst resident,
        byte-identically to the pre-fix behaviour.
        """
        boot, mgr, victim_a, victim_b, _handle_a, log = await _build_two_loaded_victims(
            tmp_path, monkeypatch, claimant_split_mode="none")
        try:
            async with mgr._registry_lock:
                governing = mgr._governing_claim_locked()
                claimant_is_split = (
                    None if governing is None
                    else mgr._claimant_unloads_every_sibling_locked(governing[1])
                )
                b_designated = mgr._is_designated_unload_target_locked(victim_b)
                a_designated = mgr._is_designated_unload_target_locked(victim_a)
        finally:
            await mgr.shutdown()

        # Answers captured above, teardown done. Everything below is a verdict on
        # values already taken, so no cleanup failure can replace it. Preconditions
        # first, in order, so a setup fault can never be read as the target firing.
        assert governing is not None, (
            "test setup bug: no live governing claim, so every designation answer "
            "below would be False for a reason with nothing to do with the simultaneous-victim rule"
        )
        assert claimant_is_split is False, (
            "test setup bug: the claimant reads as layer-split, so this is the "
            "TWO-victim scenario over again and the single-victim half is still "
            "unguarded — the exact gap this test exists to close"
        )
        assert b_designated is True, (
            "the WORST-ranked resident (modelB) is not designated, so this scenario "
            "never engaged the designated-victim rule at all and the False asserted below would prove "
            f"nothing. answers: {log!r}"
        )
        assert a_designated is False, (
            "DESIGN CONSTRAINT VIOLATED — THE VICTIM SET WIDENED PAST THE CLAIM. This "
            "claimant is not layer-split, so _claimant_unloads_every_sibling_locked "
            "is False and the single-victim branch must decide — yet modelA, which "
            "is NOT the worst-ranked resident, was ALSO designated. That is a "
            "resident the pre-fix max() would never have selected. The simultaneous-victim rule "
            "permits taking more than one member of the pool only for a claimant "
            "that cannot co-reside; it never permits taking a second victim from a "
            f"claim that needs one. answers: {log!r}"
        )
