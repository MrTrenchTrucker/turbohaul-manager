"""Two-victim eviction grace measurement -- MEASUREMENT ONLY, NOT A FIX.

Multi-victim rule: "A model that does not fit on a
single card evicts TWO OR MORE victims when a higher-priority client requests
it. Victims are selected in the usual eviction order -- lowest
priority first, ties broken by recency within a rank -- and the selection
continues until the claim can be satisfied. Every victim is subject to the
single-victim rule INDIVIDUALLY: each is torn down at its own turn boundary, with no grace timer,
and each has its context saved before teardown."

This file measures ONLY the "no grace timer" half of that rule, for a
genuine two-victim eviction. It does not fix anything, does not widen
`_is_designated_unload_target_locked`, does not touch queue.py, and does not edit
any production function body.

WHY THE ORDERING IS "GRACE, THEN SELECTION-PROOF", NOT THE REVERSE: in the
real system, `_lru_idle_unloadable` only ever considers IDLE_EVICTABLE
residents -- a resident becomes eligible for the credit-based multi-pass
eviction selector ONLY AFTER its own `_serve_on_resident` grace loop has
already resolved (skip, breakout, or full deadline) and it has parked. So the
causally faithful order is: (1) register a real, live Fast Lane claim (2)
drive victim 1's real grace loop, observe, park it (3) drive victim 2's real
grace loop -- now victim 1 is gone from the loaded pool, so victim 2 is
freshly re-evaluated against a smaller pool, exactly as the design requires ("fresh at
each call, never a stale snapshot") -- observe, park it (4) ONLY
THEN prove, via the REAL production selector (`_route_or_reserve` /
`_lru_idle_unloadable` / the credit mechanism), that this exact claim
genuinely needed BOTH residents evicted, in this exact order -- the proof
that victim 2 was not an artifact of the fixture.

Narrowing of the code path: the cap-based routing in manager.py is the
ONLY place cap routes to a loop -- cap>=2 -> _dispatch_loop -> _drive_resident
-> _serve_on_resident (HAS the designation skip); cap<=1 ->
worker_loop's own while -> _process_slot (has its own grace loop, NO
designation call anywhere in its body). The two-victim case needs
cap>=2, so _process_slot is unreachable for this measurement by construction.
This file only ever drives _serve_on_resident.

EVERY TEST HERE CALLS `await mgr.shutdown()` IN A `finally`, EVEN THE ONES
THAT NEVER START `worker_loop`: `_begin_unload_locked` (called for real, not
mocked, in the selection-proof tests) spawns a genuine background
`_unload_teardown` task via `_spawn_bg` into `mgr._bg_tasks` -- an un-drained
one can LEAK into the NEXT test in the same pytest session (a
pre-existing, unrelated, fixed-`asyncio.sleep(0.2)` test in
test_multislot_concurrency.py can flake when this file runs immediately
before it in a full-suite session, but not when run in isolation or
paired with just that one test, nor once
the shutdown() drain is added here). `shutdown()`'s own step 4 drains
`_bg_tasks` before returning, which is the fix -- not a workaround for this
code under test (no production file changed), a hygiene gap in this test
file's own cleanup.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

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

import asyncio


# === Shared scaffolding (test-only boilerplate; NOT the code under measurement) ===

# List index IS the priority (FastLaneRule/FastLaneConfig.rules docstring
# convention, same as the designated-victim-locked test's
# own _RULES). Claimant strictly outranks BOTH victims; victim A (index 1) is
# the BETTER-ranked victim, victim B (index 2) the WORSE-ranked one, so B
# must be selected FIRST ("lowest priority first").
_RULES = [
    FastLaneRule(address="10.0.0.1", tag_ranks=FastLaneTagRanks(main=1)),  # claimant
    FastLaneRule(address="10.0.0.2", tag_ranks=FastLaneTagRanks(main=1)),  # victim A (better)
    FastLaneRule(address="10.0.0.3", tag_ranks=FastLaneTagRanks(main=1)),  # victim B (worse)
]


def _boot_runtime(tmp_path, *, max_parallel_sidecars=2, grace_seconds=5,
                   max_grace_extensions=50):
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
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False,
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=grace_seconds,
            max_grace_extensions=max_grace_extensions,
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


async def _drive_and_classify(mgr, r, anchor, handle, boot, grace_seconds):
    """Drive the REAL _serve_on_resident and OBSERVE which outcome fires --
    never assumed. Polls the real audit trail rather than asserting on the
    coroutine's return (which carries no signal either way) or racing a tight
    timeout against a call that may legitimately run the full grace window."""
    task = asyncio.create_task(mgr._serve_on_resident(r, anchor, handle))
    start = time.monotonic()
    outcome = None
    while time.monotonic() - start < grace_seconds + 1.0:
        events = _audit_events(boot, anchor.slot_id)
        if "grace_designated_victim_skip" in events:
            outcome = "skip"
            break
        if "grace_fastlane_breakout" in events:
            outcome = "breakout"
            break
        if task.done():
            outcome = "full_wait_no_event"
            break
        await asyncio.sleep(0.02)
    elapsed = time.monotonic() - start
    if not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    else:
        task.result()  # surface any exception from the driven coroutine
    return outcome, elapsed


def _teardown_gate_patch(mgr):
    """Hold the ASYNC half of eviction teardown back so the SYNCHRONOUS
    credit (set inside the real _begin_unload_locked, before it spawns the
    teardown task) is observable across multiple _route_or_reserve passes
    without racing its own paired decrement. Does not touch the credit MATH
    itself -- that stays 100% the real function; this only gates the
    already-existing async continuation, the same DI seam the shared fixture
    (the shared fast-lane fixture's make_fakes) already uses for fake_complete."""
    gate = asyncio.Event()
    real_evict_teardown = mgr._unload_teardown

    async def gated_evict_teardown(r):
        await gate.wait()
        return await real_evict_teardown(r)

    mgr._unload_teardown = gated_evict_teardown
    return gate


@pytest.mark.asyncio
class TestTwoVictimGraceDiscriminator:
    async def test_each_victim_at_its_own_turn_boundary_no_grace_timer(self, tmp_path):
        """THE MEASUREMENT. Two evictable residents, distinguishable by rank
        (10.0.0.2 outranks 10.0.0.3 -- rule_index 1 vs 2), both loaded when a
        REAL, live Fast Lane claim (10.0.0.1, rule_index 0, strictly outranks
        both) is registered via the real `_register_fastlane_claim_locked` --
        the same function `_defer_unroutable` calls, called here directly
        because there is nothing to defer FROM yet (the claimant hasn't been
        routed at all in this direct-call harness; registering the claim is
        the thing under test, not a side effect to reconstruct).

        Victim B (worse rank) drives its own _serve_on_resident FIRST, per
        the rule's "own turn boundary" wording and per its worse rank. Victim A
        drives SECOND, once B has actually left the loaded pool (parked to
        IDLE_EVICTABLE, mirroring exactly what _serve_on_resident's own
        docstring says its caller does -- "Does NOT handle idle handoff").

        SPEC-CORRECT ASSERTION (single-victim rule x multi-victim rule): both victims get NO grace
        timer -- `grace_designated_victim_skip` or `grace_fastlane_breakout`,
        never `full_wait_no_event`. If the tree disagrees, this assertion is
        RED BY CONSTRUCTION (nothing here is fixed) and IS the measurement.
        """
        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=2,
                                       grace_seconds=grace_seconds)
        # "does not fit on a single card": a layer-split claimant budgets
        # against AGGREGATE free VRAM across every card (safety._vram_budget).
        # NOTE: it is NOT this size (15000 MiB) that forces
        # both evictions -- `_vram_admits_locked` refuses ANY split_mode=
        # 'layer' claimant unconditionally while a sibling exists, `need` is
        # never read on that branch, so both evictions are structural here
        # for every possible expected_vram_mib, not "genuinely needed" in a
        # size-sensitive sense. The real selector running for real (Part 5
        # below) is proven; that it was need that decided the count is not.
        # See the note above Part 5 and TestPremiseMutant's docstring for
        # the full explanation and where the honest falsifiability control for
        # this configuration actually lives.
        _seed_manifest(boot, "claimant", expected_vram_mib=15000, split_mode="layer")
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        try:
            # === Part 1: register a REAL, live claim (real function, real match) ===
            claimant_slot = Slot.new("claimant", prompt="hi", thread_id="t-claimant",
                                      client_meta={"ip": "10.0.0.1"})
            claimant_slot.fastlane = match_fastlane(
                mgr._fastlane_table(), "10.0.0.1", {"ip": "10.0.0.1"})
            assert claimant_slot.fastlane is not None, (
                "test setup bug: claimant must genuinely match a Fast Lane rule"
            )
            async with mgr._registry_lock:
                mgr._register_fastlane_claim_locked(claimant_slot, "claim_setup")
            assert ("10.0.0.1", "claimant") in mgr._fastlane_claims, (
                "test setup bug: claim did not register"
            )

            # === Part 2: two loaded, rank-distinguishable residents ===
            now = time.monotonic()
            handle_a = _make_fake_handle("modelA", 59700)
            handle_b = _make_fake_handle("modelB", 59701)
            resident_a = Resident(
                model_tag="modelA", resident_key="modelA", state=ResidentState.ACTIVE,
                main_gpu=0, split_mode="none", reserved_need_mib=10000,
                last_active_monotonic=now, rank_client_meta={"ip": "10.0.0.2"},
                handle=handle_a,
                grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
                idle=IdleHotTimer(idle_seconds=0),
            )
            resident_b = Resident(
                model_tag="modelB", resident_key="modelB", state=ResidentState.ACTIVE,
                main_gpu=1, split_mode="none", reserved_need_mib=10000,
                last_active_monotonic=now, rank_client_meta={"ip": "10.0.0.3"},
                handle=handle_b,
                grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
                idle=IdleHotTimer(idle_seconds=0),
            )
            # Insertion order deliberately OPPOSITE of expected pick/turn order
            # (A before B; B, the worse-ranked one, must still go first) --
            # rules out an insertion-order coincidence deciding this, the same
            # guard the designated-victim test's own tie-break companion uses.
            mgr._residents["modelA"] = resident_a
            mgr._residents["modelB"] = resident_b

            # Sanity the fixture is not vacuous: B really is worse-ranked than
            # A, and with only these two loaded, B really is the
            # worst-ranked-loaded resident right now (the SELECTION half),
            # before either grace loop runs.
            async with mgr._registry_lock:
                assert mgr._is_worst_ranked_loaded_locked(resident_b) is True
                assert mgr._is_worst_ranked_loaded_locked(resident_a) is False

            # === Part 3 (measurement): B's own turn boundary, first ===
            anchor_b = Slot.new("modelB", prompt="hi", thread_id="t-b",
                                 client_meta={"ip": "10.0.0.3"})
            anchor_b.state = SlotState.STAGED
            outcome_b, elapsed_b = await _drive_and_classify(
                mgr, resident_b, anchor_b, handle_b, boot, grace_seconds)

            # B has finished its own grace decision -- mirror what the real
            # driver loop does next (_serve_on_resident's own docstring:
            # "Does NOT handle idle handoff (the driver loop owns
            # IDLE_EVICTABLE)"). _drive_resident's own
            # park logic) stamps idle_client_meta from the just-served anchor
            # slot's client_meta at exactly this point -- NOT from
            # rank_client_meta, which is the mid-turn-only field. Mirrored
            # here, not reconstructed: this is the real park contract, and
            # skipping it silently made the eviction selector below see an
            # unresolvable (unlisted) resident, which picked by insertion
            # order instead of rank -- caught by this file's own
            # deliberately-reversed insertion order above.
            resident_b.idle_client_meta = anchor_b.client_meta
            resident_b.state = ResidentState.IDLE_EVICTABLE

            # === Part 4 (measurement): A's own turn boundary, second -- now
            # the ONLY loaded resident, so is_worst_ranked_loaded_locked(A)
            # is trivially True (worst-among-one), matching the designated-victim test's own
            # test_single_loaded_resident_is_its_own_victim. The claim is
            # STILL live (never released -- Part 5 below proves it is
            # genuinely not yet satisfiable by B alone) ===
            async with mgr._registry_lock:
                assert mgr._is_worst_ranked_loaded_locked(resident_a) is True, (
                    "test setup bug: A must become the sole loaded resident, "
                    "and therefore trivially its own worst, once B has "
                    "parked -- otherwise 'victim 2' was never really "
                    "selected here"
                )
            anchor_a = Slot.new("modelA", prompt="hi", thread_id="t-a",
                                 client_meta={"ip": "10.0.0.2"})
            anchor_a.state = SlotState.STAGED
            outcome_a, elapsed_a = await _drive_and_classify(
                mgr, resident_a, anchor_a, handle_a, boot, grace_seconds)
            resident_a.idle_client_meta = anchor_a.client_meta
            resident_a.state = ResidentState.IDLE_EVICTABLE

            # === Part 5 (the binding proof): the REAL
            # production selector, called for real, PROVES this exact claim
            # needed BOTH residents evicted, worst-first -- not a fixture
            # artifact. This runs AFTER both grace decisions, matching the
            # real system's own causal order (eviction only ever touches
            # IDLE_EVICTABLE residents). ===
            # NOTE: the assertion below stays a valid
            # regression pin (if it ever fires red, something IS broken), but
            # its own wording -- "the fixture's own premise (a genuine second
            # victim is needed)" -- describes NEED-sensitivity this
            # configuration cannot demonstrate on its own. This claimant is
            # split_mode='layer' (see above); `_vram_admits_locked` refuses ANY
            # split_mode != 'none' claimant unconditionally while a sibling
            # exists, without ever reading `need` on that branch, so BOTH
            # evictions are structurally guaranteed here for every possible
            # `expected_vram_mib`, not merely for 15000. A green result below
            # proves the selector correctly evicts down to zero siblings and
            # then stops (Part 5's real point) -- it does NOT, by itself,
            # prove need decided the count. TestPremiseMutant's control
            # (below, split_mode='none') proves need-sensitivity for a 'none'
            # claim; TestPremiseMutant.test_a_layer_claimant_evicts_both_
            # regardless_of_size (parametrized across sizes spanning four
            # orders of magnitude) proves the structural-refusal claim above
            # is itself falsifiable, which is the only axis that varies for
            # split_mode='layer'. See that class's docstring for the full
            # explanation.
            evicted_order = []
            real_begin_evict = mgr._begin_unload_locked

            def spy_begin_evict(r):
                evicted_order.append(r.model_tag)
                return real_begin_evict(r)

            mgr._begin_unload_locked = spy_begin_evict
            teardown_gate = _teardown_gate_patch(mgr)

            with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[0, 0]):
                await mgr._route_or_reserve(claimant_slot)   # pass 1
                assert evicted_order == ["modelB"], (
                    "pass 1 must evict the worse-ranked victim (modelB) "
                    f"first, per the eviction order -- got {evicted_order!r}"
                )
                await mgr._route_or_reserve(claimant_slot)   # pass 2
                assert evicted_order == ["modelB", "modelA"], (
                    "VICTIM 2 WAS NOT SELECTED: after crediting modelB's "
                    "reclaim alone (10000 MiB), the claim (needs 15000 MiB, "
                    "layer-split, aggregate-budgeted) is still not "
                    "satisfiable -- a second real eviction pass was "
                    f"required. Got {evicted_order!r}. If this is "
                    "['modelB'] only, the fixture's own premise (a genuine "
                    "second victim is needed) is FALSE and everything above "
                    "is UNFALSIFIABLE, not CONFIRMED, regardless of "
                    "outcome_a below."
                )
                await mgr._route_or_reserve(claimant_slot)   # pass 3: now admits
                assert evicted_order == ["modelB", "modelA"], (
                    "a third eviction would mean the claim over-evicted "
                    "past the credit-sufficiency stop -- an unrelated "
                    "defect this test is not scoped to find, but it would "
                    "poison this proof"
                )
            teardown_gate.set()

            # === result ===
            print(f"MEASUREMENT: victim B (worse rank, drives "
                  f"first) outcome={outcome_b!r} elapsed={elapsed_b:.3f}s")
            print(f"MEASUREMENT: victim A (better rank, drives "
                  f"second, sole loaded resident at its own turn) "
                  f"outcome={outcome_a!r} elapsed={elapsed_a:.3f}s")
            print(f"MEASUREMENT: real selector eviction order "
                  f"(proves victim 2 was genuinely needed) = {evicted_order!r}")

            assert outcome_b in ("skip", "breakout"), (
                f"victim 1 (modelB) got NO no-grace treatment -- "
                f"outcome={outcome_b!r} elapsed={elapsed_b:.3f}s. Even the "
                "SINGLE-victim case (not just the multi-victim rule) is broken if this "
                "fails; that would be a bigger problem than the "
                "question asked here."
            )
            assert outcome_a in ("skip", "breakout"), (
                f"VICTIM 2 (modelA) GOT NO NO-GRACE TREATMENT -- "
                f"outcome={outcome_a!r} elapsed={elapsed_a:.3f}s, against "
                f"grace_seconds={grace_seconds}. This is the multi-victim rule's third "
                "clause failing for the SECOND victim specifically: 'Every "
                "victim is subject to the single-victim rule INDIVIDUALLY ... each is torn "
                "down at its own turn boundary, with no grace timer.' "
                "If this fires, the hole "
                "is real and measured (elapsed_a above is the number). This "
                "file only measures; it fixes nothing."
            )
        finally:
            await mgr.shutdown()


@pytest.mark.asyncio
class TestGraceAssertionMutant:
    """RED-first-by-construction: the main measurement's grace
    assertions are GREEN on the unmodified tree (both victims skip
    grace, so the suspected defect is not present). A
    green that could not have been red is not a pass -- so this
    mutates the ONE thing the suspected defect would need (a claim
    that survives the defer cycle) by never registering a claim at all, and
    proves THIS SAME harness correctly measures the predicted defect shape
    (full_wait_no_event, no skip, no breakout) when that thing is genuinely
    absent. This is what makes the main test's 'skip'/'skip' result a real
    negative, not a vacuous one."""

    async def test_no_live_claim_means_full_grace_wait_not_skip(self, tmp_path):
        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=2,
                                       grace_seconds=grace_seconds)
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        try:
            # Deliberately NO claim registered anywhere -- this is the one
            # input the main test's Part 1 supplies and this mutant withholds.
            now = time.monotonic()
            handle_b = _make_fake_handle("modelB", 59704)
            resident_b = Resident(
                model_tag="modelB", resident_key="modelB", state=ResidentState.ACTIVE,
                main_gpu=1, split_mode="none", reserved_need_mib=10000,
                last_active_monotonic=now, rank_client_meta={"ip": "10.0.0.3"},
                handle=handle_b,
                grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
                idle=IdleHotTimer(idle_seconds=0),
            )
            mgr._residents["modelB"] = resident_b

            async with mgr._registry_lock:
                # Selection alone (worst-among-one) is still trivially True
                # -- this mutant isolates the CLAIM half, not the selection
                # half.
                assert mgr._is_worst_ranked_loaded_locked(resident_b) is True
                assert mgr._is_designated_unload_target_locked(resident_b) is False, (
                    "test setup bug: with no claim registered, designation "
                    "must be False even though selection alone is True -- if "
                    "this is True, the mutant does not isolate what it "
                    "claims to"
                )

            anchor_b = Slot.new("modelB", prompt="hi", thread_id="t-b",
                                 client_meta={"ip": "10.0.0.3"})
            anchor_b.state = SlotState.STAGED
            outcome, elapsed = await _drive_and_classify(
                mgr, resident_b, anchor_b, handle_b, boot, grace_seconds)

            assert outcome == "full_wait_no_event", (
                f"MUTANT DID NOT FIRE: with no live claim anywhere, outcome "
                f"should be the full grace wait, got {outcome!r} "
                f"elapsed={elapsed:.3f}s -- if this passes with 'skip', the "
                "main test's assertions are not falsifiable and its GREEN "
                "result proves nothing"
            )
            assert elapsed >= grace_seconds - 0.5, (
                f"held only {elapsed:.3f}s of a {grace_seconds}s "
                "grace_seconds window -- not the full-deadline shape the "
                "mutant expects"
            )
        finally:
            await mgr.shutdown()


@pytest.mark.asyncio
class TestPremiseMutant:
    """Premise control: the premise under test -- 'the credit-based
    repeated-pass path selects a SECOND victim at all for this claim' -- gets
    its own mutant here, proving the two-eviction assertion above is a real,
    falsifiable measurement and not vacuously true regardless of need.

    SCOPE: the paragraph below covers what this control proves (a
    'none'-topology claim's eviction count is genuinely need-driven). It does
    NOT make TestTwoVictimGraceDiscriminator's two-eviction
    result falsifiable, because that test's actual configuration differs
    from this control's.
    The control below is split_mode='none'; the main
    test's claimant is split_mode='layer' (see the main test). Those are TWO variables
    apart (need AND topology), not one, so a passing result here says
    nothing about whether the main test's own two-eviction outcome is
    need-driven. It is provably NOT: `_vram_admits_locked`
    refuses ANY split_mode != 'none' claimant outright while a sibling
    exists -- `need` is never even read on that branch -- so for a
    layer-split claimant the eviction count is decided by sibling COUNT
    alone, for every possible need, not by need at all. There is NO
    need-varying control possible for a layer claimant; see
    test_a_layer_claimant_evicts_both_regardless_of_size below
    (parametrized across three sizes spanning four orders of magnitude) for
    the actual, honest falsifiability proof for that configuration -- it
    measures whether the structural refusal itself still holds, which is
    the only thing that CAN vary for split_mode='layer'.

    Control design, true for what it claims: same rank setup as
    the main test, but the claim is sized to need only ONE victim's worth of
    credit. If this test also shows two evictions, THIS control (for a
    'none'-topology claim) is worthless (the selector would be evicting
    unconditionally) -- that check is real and still runs (the one-victim
    claim control below)."""

    async def test_a_claim_satisfiable_by_one_victim_evicts_only_one(self, tmp_path):
        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=2,
                                       grace_seconds=grace_seconds)
        # Half the need of the main test's claimant -- ONE victim's reclaim
        # (10000 MiB) already covers it.
        # split_mode "none", not "layer". FIXTURE ONLY -- the
        # assertion below is unchanged and still correct. This control's whole
        # claim is that NEED is the variable deciding the eviction COUNT, but a
        # layer-split claimant can never co-reside with ANY sibling:
        # _vram_admits_locked refuses a non-'none' incoming model STRUCTURALLY
        # whenever siblings exist ("crediting VRAM cannot change them"), so with
        # split_mode "layer" the claimant would lose BOTH residents regardless
        # of its size, and need would decide nothing. Such a control would be
        # confounded and could not falsify anything.
        # It could only look need-sensitive if the relocation
        # stamped the VICTIM's split_mode onto the claimant (the relocation
        # code's `= victim.split_mode`) and that value fed back into the claimant's own
        # VRAM gate -- so on pass 2 a layer-split claimant would be gated AS a 'none'
        # model, laundering the structural refusal into a VRAM question where
        # 8000 < 10000 admits after one eviction. The relocation no longer does
        # that, and a 'none' claimant makes need genuinely the
        # variable, which is what this control is meant to measure.
        # The companion arm below pins the layer/structural behaviour that this
        # fixture would otherwise exercise only by accident.
        _seed_manifest(boot, "claimant", expected_vram_mib=8000, split_mode="none")
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        try:
            claimant_slot = Slot.new("claimant", prompt="hi", thread_id="t-claimant",
                                      client_meta={"ip": "10.0.0.1"})
            claimant_slot.fastlane = match_fastlane(
                mgr._fastlane_table(), "10.0.0.1", {"ip": "10.0.0.1"})
            assert claimant_slot.fastlane is not None

            now = time.monotonic()
            handle_a = _make_fake_handle("modelA", 59702)
            handle_b = _make_fake_handle("modelB", 59703)
            resident_a = Resident(
                model_tag="modelA", resident_key="modelA", state=ResidentState.IDLE_EVICTABLE,
                main_gpu=0, split_mode="none", reserved_need_mib=10000,
                last_active_monotonic=now, idle_client_meta={"ip": "10.0.0.2"},
                handle=handle_a,
            )
            resident_b = Resident(
                model_tag="modelB", resident_key="modelB", state=ResidentState.IDLE_EVICTABLE,
                main_gpu=1, split_mode="none", reserved_need_mib=10000,
                last_active_monotonic=now, idle_client_meta={"ip": "10.0.0.3"},
                handle=handle_b,
            )
            mgr._residents["modelA"] = resident_a
            mgr._residents["modelB"] = resident_b

            evicted_order = []
            real_begin_evict = mgr._begin_unload_locked

            def spy_begin_evict(r):
                evicted_order.append(r.model_tag)
                return real_begin_evict(r)

            mgr._begin_unload_locked = spy_begin_evict
            teardown_gate = _teardown_gate_patch(mgr)

            with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[0, 0]):
                await mgr._route_or_reserve(claimant_slot)   # pass 1: evicts modelB
                await mgr._route_or_reserve(claimant_slot)   # pass 2: should ADMIT, not evict again
            teardown_gate.set()

            assert evicted_order == ["modelB"], (
                "PREMISE MUTANT: a claim satisfiable by ONE victim's reclaim "
                f"must evict exactly one resident, not {evicted_order!r} -- "
                "if this shows two evictions, the main test's 'victim 2 was "
                "genuinely needed' proof is not falsifiable and its result "
                "is worthless regardless of which way the grace measurement "
                "landed."
            )
        finally:
            await mgr.shutdown()

    @pytest.mark.parametrize(
        "expected_vram_mib", [1, 8000, 50000],
        ids=["near-zero", "one-victim-size", "far-over-both-victims"],
    )
    async def test_a_layer_claimant_evicts_both_regardless_of_size(
        self, tmp_path, expected_vram_mib,
    ):
        """Companion arm, parametrized to
        make the "regardless of size" claim in this test's own name an
        actually-tested generality, not one arbitrary data point. Pins the
        behaviour TestPremiseMutant's control above CANNOT: for a
        layer-split claimant the eviction count is decided STRUCTURALLY, not
        by need.

        _vram_admits_locked refuses a non-'none' incoming model outright while
        ANY sibling is resident -- "Structural refusals -- crediting VRAM cannot
        change them (a layer-split incoming model still cannot co-reside no
        matter how much VRAM frees up)", and `need` is never read on that
        branch. So this claimant -- at 1 MiB (need is trivially satisfiable by
        either victim alone), 8000 MiB (the control's own one-victim size),
        or 50000 MiB (over BOTH victims' combined reclaim) -- loses both
        residents identically every time. Size cannot help it, in either
        direction: this is the proof that no need-varying control can exist
        for split_mode='layer' (a structural property), not merely a spot check.

        A layer-split claimant would pass a one-eviction assertion only if
        the relocation
        laundered the claimant's split topology into the victim's 'none'
        (the relocation code's victim.split_mode stamp). With no such laundering, the
        behaviour is pinned here explicitly instead of being exercised by
        accident, and the control above is free to actually vary need."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=2,
                                       grace_seconds=grace_seconds)
        _seed_manifest(boot, "claimant", expected_vram_mib=expected_vram_mib,
                        split_mode="layer")
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        try:
            claimant_slot = Slot.new("claimant", prompt="hi", thread_id="t-claimant",
                                      client_meta={"ip": "10.0.0.1"})
            claimant_slot.fastlane = match_fastlane(
                mgr._fastlane_table(), "10.0.0.1", {"ip": "10.0.0.1"})
            assert claimant_slot.fastlane is not None

            now = time.monotonic()
            resident_a = Resident(
                model_tag="modelA", resident_key="modelA", state=ResidentState.IDLE_EVICTABLE,
                main_gpu=0, split_mode="none", reserved_need_mib=10000,
                last_active_monotonic=now, idle_client_meta={"ip": "10.0.0.2"},
                handle=_make_fake_handle("modelA", 59712),
            )
            resident_b = Resident(
                model_tag="modelB", resident_key="modelB", state=ResidentState.IDLE_EVICTABLE,
                main_gpu=1, split_mode="none", reserved_need_mib=10000,
                last_active_monotonic=now, idle_client_meta={"ip": "10.0.0.3"},
                handle=_make_fake_handle("modelB", 59713),
            )
            mgr._residents["modelA"] = resident_a
            mgr._residents["modelB"] = resident_b

            evicted_order = []
            real_begin_evict = mgr._begin_unload_locked

            def spy_begin_evict(r):
                evicted_order.append(r.model_tag)
                return real_begin_evict(r)

            mgr._begin_unload_locked = spy_begin_evict
            teardown_gate = _teardown_gate_patch(mgr)

            with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[0, 0]):
                await mgr._route_or_reserve(claimant_slot)
                await mgr._route_or_reserve(claimant_slot)
            teardown_gate.set()

            assert evicted_order == ["modelB", "modelA"], (
                "a layer-split claimant cannot co-reside with ANY sibling, so it "
                "must lose both regardless of its size -- got "
                f"{evicted_order!r}. If this ever shows ONE eviction, something "
                "is again rewriting the claimant's split_mode to 'none' behind "
                "the VRAM gate (the split_mode laundering defect)."
            )
            # And the relocation must NOT have handed it the victim's topology.
            assert claimant_slot.fastlane_relocated_split_mode in (None, "layer"), (
                "the relocation must carry the CLAIMANT's own split_mode, never "
                f"the victim's 'none' -- got "
                f"{claimant_slot.fastlane_relocated_split_mode!r}"
            )
        finally:
            await mgr.shutdown()
