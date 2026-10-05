"""The `candidate_key is None` VETO in the
widening of the placement path, and the multi-victim characterization, as ONE
change. RED-FIRST -- both the defect and its consequence were traced in the
code, NOT run; this file is what settles it.

Defect: at the widening's outrank check,
    if candidate_key is not None and (self.queue._fastlane_strictly_higher(...)):
`_resident_priority_key` returns `None` for an unlisted-or-unresolvable
resident. But the Fast Lane design states unlisted clients
rank BELOW ALL registered ones -- a worked example:
two unregistered clients loaded, any registered claimant (even
rank 9) is queued at capacity, and the UNREGISTERED clients are the
eviction pool. `_resident_unload_priority_key` already encodes exactly
this (unlisted sorts FIRST, tier 0, cheapest to evict) -- but the widening
reads `_resident_priority_key` instead, and treats its `None` as DECLINE,
so it refuses to evict the very resident that is, by the design's own
rule, the correct and cheapest victim -- and falls through to the on-card
path, evicting a LISTED on-card resident instead
with no outranking test at all.

NOTE: the claimant-side `claimant_key is not None`
guard is correct and unchanged by this fix -- a claimant
with no live claim must decline, full stop. The defect is that the SAME
`is None` check was reused for the CANDIDATE side, where `None` means the
opposite thing (lowest priority, not "no claim").

Fix (one line): `candidate_key is None` JOINS the outrank
condition (an unlisted candidate is trivially outranked by any live listed
claimant) instead of vetoing it.

Four proof arms, end-to-end through the REAL `_route_or_reserve` wiring
(same style as the existing by-construction template):
  - RED (this file's own name for it: test_RED_...): the exact fixture the
    scenario requires -- one unlisted idle-evictable resident OFF-card, all
    on-card residents LISTED and ranked ABOVE the claimant, one strong live
    Fast Lane claim placed explicitly (non-auto_place). Pre-fix: the
    off-card unlisted resident is NOT evicted; a listed on-card resident is
    evicted instead. The RED asserts on this BEHAVIOUR (who got evicted),
    never on a missing symbol.
  - GREEN: same fixture, same assertion, passing after the fix -- the
    off-card unlisted resident is evicted, the listed on-card resident
    survives.
  - THIRD ARM (auto_place=True, the placement function's own `if` branch, a
    sibling of the `else` arm the one-line edit lives entirely inside):
    proven BY CONSTRUCTION, same as that template -- the `_auto_place` branch
    is a different `if`-arm at a shallower indent than the edit, so it is
    structurally unreachable from the changed line. Confirmed by reading
    the diff context, not merely asserted.
  - NEGATIVE CONTROL: a non-outranking claimant (rank below the on-card
    listed residents) evicts nobody off-card -- the widening's outrank
    gate itself is untouched, only the None-handling within it changed.

CHARACTERIZATION of the multi-victim path, NOT a fix. Recorded
as observed behaviour, conformant-or-defect either way. By
design, the ordinary on-card CAPACITY path
(`if victim is None: victim = self._lru_idle_unloadable(
shielded_tags, main_gpu, split_mode)`) is deliberately NOT gated on
outranking -- gating it would let every resident outrank a low-priority
claimant forever on a full box, colliding with the design's one-turn floor.
The Fast Lane rules (preemption) and capacity (make-room) are different paths by
design; this section characterizes capacity's existing behaviour without
touching it.
  (a) does a claimant needing a second card's worth of room take a SECOND
      victim at all;
  (b) is the second victim selected worst-ranked-first;
  (c) is the second victim ever one the claimant does NOT outrank.
"""
import asyncio
import ipaddress
from unittest.mock import AsyncMock

import pytest

from turbohaul.config import (
    BootConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.fastlane import CompiledRule, FastLaneMatch
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.slot import Slot

NEED = 100  # arbitrary MiB unit for the fake VRAM ledger below


def _match(rule_index=0, rank=1):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag="main", rank=rank,
    )


def _rule(index, raw_address, tag_ranks=None):
    addr = ipaddress.ip_address(raw_address)
    return CompiledRule(
        index=index,
        raw_address=raw_address,
        address=addr,
        container_name=None,
        match_addresses=frozenset({addr}),
        label=f"rule{index}",
        tag_ranks=tag_ranks or {},
    )


# rule 0 = HIGH priority (the outranking claimant); rule 5 = MID (the
# on-card listed residents, ranked ABOVE the claimant per the scenario's
# fixture shape -- i.e. rule_index 5 is a LOWER rank number than the
# claimant would need to strictly outrank them, so they must NOT be the
# widening's chosen victim). tag_ranks pins the resolved rank to 1 for an
# unclassified {"ip": ...} meta, same convention as the existing template.
TABLE = [
    _rule(0, "10.0.0.1", tag_ranks={"unclassified": 1}),  # the claimant -- HIGHEST
    _rule(5, "10.0.0.5", tag_ranks={"unclassified": 1}),  # on-card listed residents
]


class _FakeVram:
    """Per-card free-MiB ledger. Not the real safety.py math -- this file is
    about the widening/veto decision, not VRAM sizing arithmetic."""

    def __init__(self, free):
        self.free = dict(free)

    def admits(self, need, parallel, main_gpu, split_mode, **_kw):
        return self.free.get(main_gpu, 0) >= need

    def free_card(self, card):
        self.free[card] = 10 ** 9


@pytest.fixture
def mgr(tmp_path):
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
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False, max_parallel_sidecars=8),
        pull=PullConfig(),
    )
    m = TurbohaulManager(boot, runtime)
    m._fastlane_table = lambda: TABLE
    return m


def _idle_resident(tag, gpu, holder_ip, last_active=1.0):
    return Resident(
        model_tag=tag, resident_key=tag,
        state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=last_active,
        main_gpu=gpu, split_mode="none",
        idle_client_meta={"ip": holder_ip},
    )


def _unlisted_idle_resident(tag, gpu, last_active=1.0):
    """No idle_client_meta resolvable against TABLE -- unlisted/unresolvable,
    `_resident_priority_key` returns None for this one."""
    return Resident(
        model_tag=tag, resident_key=tag,
        state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=last_active,
        main_gpu=gpu, split_mode="none",
        idle_client_meta={"ip": "10.0.0.99"},  # matches no rule in TABLE
    )


def _pinned_claimant(tag, rule_index=None):
    s = Slot.new(tag)
    if rule_index is not None:
        s.fastlane = _match(rule_index=rule_index)
    s.completion_future = asyncio.get_event_loop().create_future()
    return s


@pytest.mark.asyncio
class TestUnlistedCandidateVeto:
    async def test_RED_unlisted_offcard_veto_evicts_listed_oncard_instead(self, mgr):
        """The scenario's own fixture, watched: card 0 holds the claimant and a
        LISTED on-card resident (rule_index=5, ranked below the claimant's
        rule_index=0 -- i.e. the claimant DOES outrank it too, so this is
        the worst possible outcome: the widening had a cheaper, correct,
        off-card victim available and picked the wrong one anyway). Card 1
        holds ONE unlisted idle-evictable resident. Card 0 has nothing free.
        By design, the unlisted off-card resident is the
        eviction pool and must go first.

        PRE-FIX (expected RED): `candidate_key is not None` vetoes the
        off-card unlisted candidate -- the widening declines, falls through
        to the on-card-only lookup, and evicts the LISTED
        on-card resident instead. Assertion is on WHO got evicted, a real
        behavioural failure, not a missing symbol.

        POST-FIX (expected GREEN): the off-card unlisted resident is
        evicted; the on-card listed resident survives.
        """
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, False, False)
        mgr._residents["oncard-listed"] = _idle_resident(
            "oncard-listed", gpu=0, holder_ip="10.0.0.5")
        mgr._residents["offcard-unlisted"] = _unlisted_idle_resident(
            "offcard-unlisted", gpu=1)
        evicted = []
        mgr._begin_unload_locked = lambda r: (
            evicted.append(r.model_tag), vram.free_card(r.main_gpu))
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=0)  # strictly outranks rule_index=5

        await mgr._route_or_reserve(claimant)

        assert evicted == ["offcard-unlisted"], (
            f"expected the cheaper unlisted off-card resident evicted (the "
            f"design's own eviction pool), got {evicted!r} -- if this "
            f"fired on 'oncard-listed' instead, that is the traced defect: "
            f"the None-veto declined the correct candidate and fell through "
            f"to the on-card path, sacrificing a listed resident it should "
            f"have spared"
        )
        assert "oncard-listed" in mgr._residents, (
            "the on-card listed resident must survive -- it was never the "
            "correct victim"
        )

    async def test_NEGATIVE_CONTROL_non_outranking_claimant_declines_a_LISTED_offcard_candidate(self, mgr):
        """The genuine negative control for the `or` join: when the global
        widened candidate is LISTED (not unlisted), the strict-outrank test
        must still gate it exactly as before -- the fix only changes the
        UNLISTED (None) case unconditionally, it must not have accidentally
        turned the whole check into a no-op. Here the off-card candidate is
        LISTED at the SAME rank as the claimant (a tie, not a strict
        outrank) -- the widening must decline it and fall through to the
        on-card-only lookup, evicting the on-card resident instead.

        (An off-card UNLISTED candidate is deliberately NOT used here: per
        the design, an unlisted candidate is outranked by ANY live
        listed claimant unconditionally -- there is no claimant rank for
        which declining an unlisted candidate would be correct, so that
        shape cannot serve as this control.)

        NOTE: any test of
        the widening must construct a case where the global `min` and the
        on-card `min` are DIFFERENT OBJECTS -- if they tie on the whole
        eviction key `(tier, rule_index, last_active_monotonic)`, `min`
        returns the SAME object for both lookups and the widening/fall-
        through paths are indistinguishable no matter what is asserted.
        `offcard-listed-tie` is therefore given a STRICTLY lower
        `last_active_monotonic` (0.5) than `oncard-listed` (1.0) -- both
        tie on tier and rule_index, but `offcard-listed-tie` is strictly
        the more-LRU one, so the global `min` picks IT while the on-card
        `min` (scoped to card 0 only) still picks `oncard-listed` -- the
        two candidates the gate must be able to distinguish are no longer
        the same object. Verified with a mutant:
        `if True or (...)` (erasing the gate) turns this GREEN test RED,
        proving the control is now sensitive to the exact code it guards.
        A tie (both at 1.0) would let `if True or (...)` stay
        undetected -- global-min and on-card-min would be the same object in
        both worlds, so the assertion would hold either way; that is the
        vacuous-evidence defect a mutant exposes."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, False, False)
        mgr._residents["oncard-listed"] = _idle_resident(
            "oncard-listed", gpu=0, holder_ip="10.0.0.5", last_active=1.0)
        mgr._residents["offcard-listed-tie"] = _idle_resident(
            "offcard-listed-tie", gpu=1, holder_ip="10.0.0.5", last_active=0.5,
        )  # rule_index=5 ties the claimant's rank; strictly MORE-LRU than
           # oncard-listed so the two `min`s pick different objects (see the note above)
        evicted = []
        mgr._begin_unload_locked = lambda r: (
            evicted.append(r.model_tag), vram.free_card(r.main_gpu))
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=5)  # ties offcard-listed-tie -- no strict outrank

        await mgr._route_or_reserve(claimant)

        assert evicted == ["oncard-listed"], (
            f"a non-strictly-outranking claimant against a LISTED off-card "
            f"candidate must fall through to the pre-existing on-card-only "
            f"lookup, evicting the on-card resident -- got {evicted!r}; if "
            f"'offcard-listed-tie' fired instead, the `or` join erased the "
            f"strict-outrank gate for the listed case, not just fixed the "
            f"unlisted one"
        )
        assert "offcard-listed-tie" in mgr._residents, (
            "the off-card LISTED tied resident must survive -- the "
            "claimant does not strictly outrank it"
        )

    async def test_THIRD_ARM_auto_place_path_structurally_unreachable(self, mgr):
        """auto_place=True takes the `if _auto_place:` branch,
        a SIBLING of the `else:` arm the one-line edit
        lives entirely inside (confirmed by reading the diff context, not
        merely asserted here). Proven by construction, same style as
        the existing template: an UNLISTED claimant (no live Fast Lane claim at all,
        which would decline the new non-auto_place branch outright) still
        evicts the global-LRU victim unconditionally under auto_place,
        exactly as before this fix -- because the changed line is never
        even reached when `_auto_place` is True."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, True, False)  # auto_place=True
        mgr._residents["offcard-unlisted"] = _unlisted_idle_resident(
            "offcard-unlisted", gpu=1)
        evicted = []
        mgr._begin_unload_locked = lambda r: evicted.append(r.model_tag)
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=None)  # no live claim at all

        await mgr._route_or_reserve(claimant)

        assert evicted == ["offcard-unlisted"], (
            "auto_place=True must evict the global-LRU victim unconditionally "
            "-- unaffected by this fix, whose changed line sits in a "
            "structurally different if-arm"
        )


@pytest.mark.asyncio
class TestMultiVictimCharacterization:
    """CHARACTERIZATION ONLY -- records what the on-card capacity path
    (the `if victim is None:` fallback) DOES today. No assertion
    here should be read as endorsing or changing that behaviour; by
    design this path is deliberately NOT gated on outranking."""

    async def test_capacity_path_evicts_the_global_widened_offcard_candidate_not_a_second_victim(self, mgr):
        """(a) With TWO off-card unlisted idle residents and nothing on-card,
        a single claimant's single miss takes exactly ONE victim per
        `_route_or_reserve` call -- the widening's own single `_lru_idle_unloadable`
        call, not a second independent selection. There is no code path in
        this function that loops to a SECOND victim within one call; a
        second victim, if it ever happens, is a SEPARATE `_route_or_reserve`
        invocation (a second miss / a second retry), not a second search
        inside this one. Recorded as observed, not changed."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, False, False)
        mgr._residents["offcard-a"] = _unlisted_idle_resident("offcard-a", gpu=1, last_active=1.0)
        mgr._residents["offcard-b"] = _unlisted_idle_resident("offcard-b", gpu=1, last_active=2.0)
        evicted = []
        mgr._begin_unload_locked = lambda r: (
            evicted.append(r.model_tag), vram.free_card(r.main_gpu))
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=0)

        await mgr._route_or_reserve(claimant)

        assert len(evicted) == 1, (
            f"characterization: one _route_or_reserve call selects exactly "
            f"ONE victim via the widening's single _lru_idle_unloadable call, "
            f"got {evicted!r} -- confirms (a): the 'second victim' question "
            f"is about a SECOND miss/retry, not a loop within this call"
        )
        assert evicted == ["offcard-a"], (
            "(b) worst-ranked/oldest-idle-first: among two equally-unlisted "
            "(tier 0) off-card residents, the OLDER last_active_monotonic "
            "(offcard-a, 1.0 vs 2.0) is selected first -- LRU breaks the "
            "tie within tier 0, matching _resident_unload_priority_key's "
            "own documented contract (tier first, LRU only within-tier)"
        )

    async def test_capacity_path_second_call_can_select_a_non_outranked_oncard_victim(self, mgr):
        """(c) On a SEPARATE _route_or_reserve call (standing in for a
        second retry/miss after the first victim's card still doesn't fit),
        with NOTHING off-card left and only an on-card LISTED resident that
        OUTRANKS the claimant available, the capacity path
        still evicts it -- no outranking test gates this path, confirming
        (c): the priority gate is genuinely absent from ordinary capacity
        eviction, by design (the one-turn floor), not a defect
        introduced or touched by this change."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, False, False)
        # on-card resident outranks the claimant (rule_index=0 beats rule_index=5)
        mgr._residents["oncard-stronger"] = _idle_resident(
            "oncard-stronger", gpu=0, holder_ip="10.0.0.1")
        evicted = []
        mgr._begin_unload_locked = lambda r: (
            evicted.append(r.model_tag), vram.free_card(r.main_gpu))
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=5)  # LOWER priority than oncard-stronger

        await mgr._route_or_reserve(claimant)

        assert evicted == ["oncard-stronger"], (
            f"(c) characterization: the on-card capacity fallback evicted "
            f"{evicted!r} -- a resident the claimant does NOT outrank -- "
            f"with no outranking test at this path. This is CONFORMANT by "
            f"design (capacity != preemption, one-turn floor), "
            f"not something this change alters"
        )
