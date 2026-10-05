"""Fast Lane cross-card relocation: cross-card victim selection + claimant relocation,
landed as ONE slice because the two changes only make sense together:
widening the victim search without relocating the claimant would evict a
resident for nothing.

Defect: at the VRAM over-commit make-room site (manager.py's
_route_or_reserve, the `if not projected_fits:` branch), a non-auto_place
(manifest-pinned) claimant's victim search was scoped to its OWN card only
(`main_gpu=main_gpu` passed into `_lru_idle_unloadable`) -- so a Fast Lane
claim that strictly outranks a resident on a DIFFERENT card could never
select it, even though the claim has every right to. Widening the scope
alone (without also relocating the claimant onto the freed card) would be
WORSE than doing nothing: evicting an
off-card resident frees no VRAM on the card the claimant still targets, so
the claimant still fails to admit -- a resident died for nothing.

Fix (call-site only, `_lru_idle_unloadable`'s body untouched): when a live
Fast Lane claim strictly outranks the globally worst-ranked idle-evictable
resident, that resident is selected regardless of card (mirrors the
existing `_auto_place` scope at the same site) AND the claimant is
relocated onto the freed card via `Slot.fastlane_relocated_main_gpu` /
`_split_mode`, consumed by `_route_or_reserve`'s own placement-resolution
step and threaded into `_reserve_and_start_locked` through its
PRE-EXISTING `main_gpu_override` / `split_mode_override` params (the explicit
placement-override support) -- not a new mechanism, just a new caller of it. A claimant with
no live claim, or one that does not strictly outrank the global candidate,
falls through to the exact pre-existing on-card-only lookup -- ordinary
manifest-pinned (non-Fast-Lane) traffic never gains cross-card eviction
authority it never had (only a Fast Lane claim overrides the
manifest's card selection).

Three proof arms, all end-to-end through the REAL
_route_or_reserve wiring:
  - RED (this file's own name for it: test_RED_...): a PINNED claimant that
    strictly outranks an off-card resident evicts it and is, on the next
    routing pass, ADMITTED onto the freed card.
  - NEGATIVE CONTROL (two arms): a claimant that does NOT strictly outrank
    the global candidate, and a claimant with no live claim at all, both
    evict NOBODY.
  - THIRD ARM: the auto_place=True path is untouched -- same call, same
    unconditional global scope, no outrank gate, exactly as before this fix.
Companion regression: when the widened search's own candidate is already
on the claimant's card, no relocation override is set (byte-identical
outcome to the on-card-only lookup) -- proves the wiring is additive, not
a behavior change for the already-covered case.
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


# rule 0 = HIGH priority ("the outranking claimant" / "the strong resident");
# rule 9 = LOW priority ("the outranked off-card victim"). tag_ranks pins the
# resolved rank to 1 for an unclassified {"ip": ...} meta, same convention
# the shield carve-out test uses.
TABLE = [
    _rule(0, "10.0.0.1", tag_ranks={"unclassified": 1}),
    _rule(9, "10.0.0.9", tag_ranks={"unclassified": 1}),
]


class _FakeVram:
    """Per-card free-MiB ledger. Deliberately not the real safety.py math --
    this file is about the call-site widening/relocation decision, not the
    VRAM sizing arithmetic (unchanged, out of scope, covered elsewhere)."""

    def __init__(self, free):
        self.free = dict(free)

    def admits(self, need, parallel, main_gpu, split_mode, **_kw):
        return self.free.get(main_gpu, 0) >= need

    def free_card(self, card):
        self.free[card] = 10 ** 9  # "fully freed" for this test's purposes


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


def _pinned_claimant(tag, rule_index=None):
    s = Slot.new(tag)
    if rule_index is not None:
        s.fastlane = _match(rule_index=rule_index)
    s.completion_future = asyncio.get_event_loop().create_future()
    return s


@pytest.mark.asyncio
class TestCrossCardRelocation:
    async def test_RED_outranking_claimant_evicts_offcard_resident_and_is_admitted(self, mgr):
        """The core scenario, watched: a pinned (auto_place=False)
        claimant on card 0 strictly outranks an idle resident on card 1;
        card 0 has nothing evictable at all. Pre-fix this starves forever
        (the search never left card 0). Post-fix: tick 1 evicts the
        off-card resident and relocates the claim; tick 2 (the next routing
        pass, standing in for the real evict-pending retry) is ADMITTED
        onto the freed card via main_gpu_override."""
        vram = _FakeVram({0: 0, 1: 0})  # nothing free anywhere yet
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, False, False)
        mgr._residents["vic-1"] = _idle_resident("vic-1", gpu=1, holder_ip="10.0.0.9")
        evicted = []
        mgr._begin_unload_locked = lambda r: (evicted.append(r.model_tag), vram.free_card(r.main_gpu))
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=0)  # strictly outranks rule_index=9

        # Tick 1: nothing on-card 0, off-card vic-1 outranked -> evicted + relocated.
        await mgr._route_or_reserve(claimant)
        assert evicted == ["vic-1"], f"expected the off-card victim evicted, got {evicted}"
        assert claimant.fastlane_relocated_main_gpu == 1, (
            "the claim must be relocated onto the card the eviction freed"
        )
        assert claimant.fastlane_relocated_split_mode == "none"
        mgr._reserve_and_start_locked.assert_not_called(), (
            "tick 1 must not reserve in-band (evict-then-QUEUE)"
        )

        # Tick 2 (stands for the real evict-pending retry): the freed card
        # now fits, and the override must be threaded into the reserve call.
        await mgr._route_or_reserve(claimant)
        mgr._reserve_and_start_locked.assert_called_once_with(
            claimant, main_gpu_override=1, split_mode_override="none",
        )

    async def test_NEGATIVE_CONTROL_non_outranking_claimant_evicts_nobody(self, mgr):
        """A live claim that does NOT strictly outrank the global candidate
        (here: a tie/worse against the strongest possible resident) must
        fall through to the pre-existing on-card-only lookup -- and since
        nothing is evictable on-card, NOBODY is evicted. Without this arm
        the fix would be indistinguishable from deleting the card
        constraint outright."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, False, False)
        mgr._residents["vic-1"] = _idle_resident("vic-1", gpu=1, holder_ip="10.0.0.1")  # rule_index=0, HIGH
        evicted = []
        mgr._begin_unload_locked = lambda r: evicted.append(r.model_tag)
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=9)  # LOW -- does not outrank rule_index=0

        await mgr._route_or_reserve(claimant)

        assert evicted == [], f"a non-outranking claimant must evict nobody, got {evicted}"
        assert "vic-1" in mgr._residents
        assert claimant.fastlane_relocated_main_gpu is None
        mgr._reserve_and_start_locked.assert_not_called()

    async def test_NEGATIVE_CONTROL_no_live_claim_evicts_nobody(self, mgr):
        """Ordinary manifest-pinned traffic with NO Fast Lane claim at all
        must never gain cross-card eviction authority -- the scope override goes
        to a Fast Lane claim specifically, not to every miss."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, False, False)
        mgr._residents["vic-1"] = _idle_resident("vic-1", gpu=1, holder_ip="10.0.0.9")  # LOW, would be outranked by SOME claim
        evicted = []
        mgr._begin_unload_locked = lambda r: evicted.append(r.model_tag)
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=None)  # unlisted -- no .fastlane

        await mgr._route_or_reserve(claimant)

        assert evicted == [], f"an unlisted (non-Fast-Lane) claimant must evict nobody off-card, got {evicted}"
        assert "vic-1" in mgr._residents
        mgr._reserve_and_start_locked.assert_not_called()

    async def test_THIRD_ARM_auto_place_path_unchanged(self, mgr):
        """auto_place=True must remain byte-identical: unconditional global
        scope, no outrank gate, no relocation bookkeeping -- exactly the
        pre-existing behavior at this call site. An UNLISTED claimant (no
        live Fast Lane claim, which would block the new non-auto_place
        branch entirely) still evicts globally here, proving the auto_place
        arm of the `if` is untouched by the new gating logic added to the
        `else` arm."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, True, False)  # auto_place=True
        mgr._residents["vic-1"] = _idle_resident("vic-1", gpu=1, holder_ip="10.0.0.9")
        evicted = []
        mgr._begin_unload_locked = lambda r: evicted.append(r.model_tag)
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=None)  # unlisted

        await mgr._route_or_reserve(claimant)

        assert evicted == ["vic-1"], (
            "auto_place=True must evict the global-LRU victim regardless of "
            "any live claim or outrank comparison, unchanged by this fix"
        )
        assert claimant.fastlane_relocated_main_gpu is None, (
            "auto_place never sets a relocation override -- it was already "
            "global before this fix and needs none"
        )

    async def test_no_override_set_when_widened_victim_is_already_on_card(self, mgr):
        """Companion regression: when the globally-worst idle resident
        happens to already be on the claimant's own card, the widened
        search picks the SAME resident an on-card-only lookup would have --
        and must set no relocation override, proving the wiring is additive
        (only fires when it changes the outcome) rather than a behavior
        change for the already-covered on-card case."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, False, False)
        mgr._residents["vic-0"] = _idle_resident("vic-0", gpu=0, holder_ip="10.0.0.9")  # on-card, LOW
        evicted = []
        mgr._begin_unload_locked = lambda r: evicted.append(r.model_tag)
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=0)  # strictly outranks

        await mgr._route_or_reserve(claimant)

        assert evicted == ["vic-0"]
        assert claimant.fastlane_relocated_main_gpu is None, (
            "victim was already on the claimant's own card -- no relocation needed"
        )

    # --- the auto_place arm gets the same outrank gate -----------------------
    #
    # The TRAP: an UNLISTED claimant here (no live
    # Fast Lane claim at all) is ordinary non-Fast-Lane capacity traffic and
    # must keep evicting globally, unconditionally -- test_THIRD_ARM_auto_
    # place_path_unchanged above pins that and must keep passing UNCHANGED.
    # What the gap covers is specifically a LISTED, ranked claimant that does
    # NOT outrank the global-worst candidate: its eviction scope must narrow
    # to its own card, exactly as the pinned (else) arm already does.

    async def test_RED_auto_place_listed_non_outranking_claimant_narrows_to_own_card(self, mgr):
        """auto_place gap -- RED on the UNFIXED code: a LISTED auto_place claimant
        that does NOT outrank the global-worst idle resident (here:
        rule_index 9 against a rule-0 resident) must not evict that
        off-card resident -- it has no rank-based claim on it. Unfixed
        code: the auto_place arm ignores priority entirely and evicts the
        global-LRU victim regardless (vic-1 on card 1). Fixed code: the
        claim is live but not strictly higher, so the eviction search narrows to the
        claimant's own card (main_gpu=0); nothing there is evictable, so
        nobody is evicted and the slot is deferred (evict-then-QUEUE: no
        in-band reserve, no relocation bookkeeping)."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, True, False)  # auto_place=True
        mgr._residents["vic-1"] = _idle_resident("vic-1", gpu=1, holder_ip="10.0.0.1")  # rule 0 = HIGH
        evicted = []
        mgr._begin_unload_locked = lambda r: evicted.append(r.model_tag)
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=9)  # listed, LOW -- does not outrank rule 0

        await mgr._route_or_reserve(claimant)

        assert evicted == [], (
            f"an auto_place claimant that does not outrank the global candidate "
            f"must not evict the off-card resident it has no rank-based claim on, got {evicted}"
        )
        assert "vic-1" in mgr._residents
        assert claimant.fastlane_relocated_main_gpu is None
        mgr._reserve_and_start_locked.assert_not_called()

    async def test_auto_place_listed_outranking_claimant_still_evicts_global(self, mgr):
        """auto_place companion guard -- passes BEFORE and AFTER the fix: a LISTED
        auto_place claimant that STRICTLY OUTRANKS the global-worst idle
        resident keeps its cross-card eviction authority (it is granted
        to any claimant that outranks; the fix must not over-narrow this
        arm into on-card-only scope). auto_place still sets no relocation
        override: the re-routing auto-placer re-picks the card on the
        evict-pending retry -- the documented base-code behavior at this
        call site, unchanged by the fix."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, True, False)  # auto_place=True
        mgr._residents["vic-1"] = _idle_resident("vic-1", gpu=1, holder_ip="10.0.0.9")  # rule 9 = LOW
        evicted = []
        mgr._begin_unload_locked = lambda r: evicted.append(r.model_tag)
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=0)  # listed, HIGH -- strictly outranks rule 9

        await mgr._route_or_reserve(claimant)

        assert evicted == ["vic-1"], (
            f"a listed auto_place claimant that outranks the global candidate "
            f"keeps cross-card eviction authority, got {evicted}"
        )
        assert claimant.fastlane_relocated_main_gpu is None
    # --- two DISCRIMINATOR tests tightening the auto_place arm -------------
    # Discriminator 1: the first auto_place RED test left the claimant's own card
    # EMPTY, so "narrowed to on-card, found nothing" and "never looked
    # on-card at all, just deferred" were observationally identical. This
    # one puts an evictable resident ON the claimant's own card while the
    # global candidate sits OFF-card (victim ordering is priority-aware,
    # not LRU -- the highest rule_index / lowest priority goes first and
    # the idle timestamp never crosses tiers) -- the only GREEN outcome
    # is evicting the ON-CARD resident.
    # Discriminator 2: the unlisted-candidate clause
    # (candidate_key is None -> trivially outranked by any listed claim)
    # was unexercised -- every candidate in this file is listed.

    async def test_on_card_resident_evicted_when_claim_does_not_outrank(self, mgr):
        """Discriminator 1 -- RED on the UNPATCHED manager.py, GREEN on the fix:
        a LISTED auto_place claimant (rule_index 9, LOW) whose rank is EQUAL
        to the global-worst idle resident's -- equality is NOT strictly
        higher -- while its OWN card (main_gpu=0) hosts an evictable
        resident. The OFF-CARD rule-9 resident is the global candidate:
        victim ordering is priority-aware, not LRU -- the highest
        rule_index / lowest priority sorts first among listed and the idle
        timestamp only breaks ties within a tier -- so the unpatched
        unconditional-global arm evicts it OFF-CARD. The fix narrows the
        non-outranking scope to the claimant's own card and must evict the
        ON-CARD rule-0 resident instead; the off-card equal-rank resident
        SURVIVES -- an equal rank confers no rank-based claim."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, True, False)  # auto_place=True, main_gpu=0
        # ON-CARD (card 0) evictable resident, rule 0 = HIGH priority:
        # protected within the listed tier, but the ONLY on-card candidate,
        # so the narrowed scope must still take it.
        mgr._residents["vic-0"] = _idle_resident("vic-0", gpu=0, holder_ip="10.0.0.1")  # rule 0 = HIGH
        # OFF-CARD (card 1) rule-9 resident = global candidate: highest
        # rule_index (lowest priority) sorts first among listed, ahead of
        # the on-card rule-0 resident regardless of idle timestamp.
        mgr._residents["vic-1"] = _idle_resident("vic-1", gpu=1, holder_ip="10.0.0.9")  # rule 9 = LOW
        evicted = []
        mgr._begin_unload_locked = lambda r: evicted.append(r.model_tag)
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=9)  # listed, equal rank -- NOT strictly higher

        await mgr._route_or_reserve(claimant)

        assert evicted == ["vic-0"], (
            f"non-outranking auto_place claim: the scope must narrow to the "
            f"claimant's own card and evict the on-card resident, got {evicted}"
        )
        assert "vic-1" in mgr._residents, (
            "the off-card equal-rank resident the claimant does not strictly "
            "outrank must survive"
        )
        assert claimant.fastlane_relocated_main_gpu is None
        mgr._reserve_and_start_locked.assert_not_called()

    async def test_unlisted_offcard_candidate_outranked_by_any_listed_claim(self, mgr):
        """Discriminator 2 -- pins the unlisted-candidate clause: when
        the global candidate is UNLISTED (holder not in the Fast Lane rule
        table), its candidate_key is None, which the gate treats as
        'trivially outranked by any live listed claimant' -- it JOINS the
        outrank condition instead of vetoing it. EXPECTED TO PASS ON BOTH
        the unpatched and the patched tree (its job is to pin the clause,
        not to be RED): a listed rule-0 auto_place claimant evicts the
        unlisted off-card resident globally; auto_place still sets no
        relocation override."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, True, False)  # auto_place=True, main_gpu=0
        # UNLISTED off-card resident: 10.0.0.77 matches no rule in the table
        mgr._residents["vic-1"] = _idle_resident("vic-1", gpu=1, holder_ip="10.0.0.77")
        evicted = []
        mgr._begin_unload_locked = lambda r: evicted.append(r.model_tag)
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=0)  # listed, HIGH

        await mgr._route_or_reserve(claimant)

        assert evicted == ["vic-1"], (
            f"an unlisted global candidate is trivially outranked by any "
            f"listed claim (candidate_key None joins, never "
            f"vetoes), got {evicted}"
        )
        assert claimant.fastlane_relocated_main_gpu is None
