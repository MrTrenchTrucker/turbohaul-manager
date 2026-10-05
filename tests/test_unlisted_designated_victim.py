"""Designated-victim path, unlisted residents: the
UNLISTED-CLIENT VETO in the designated-victim path.

The earlier cross-card widening change closed ONE of the two
half-global halves. This file closes the other: the designated-victim
predicate still vetoes unlisted residents, so the machine-wide
victim set remains HALF-GLOBAL -- exactly the state the design
describes:

  "Per the design's worked example -- unregistered residents ARE
   the eviction pool -- the designated-victim predicate also needs it.
   The victim set is HALF-GLOBAL until that lands."

The design's worked example: two
unregistered clients loaded, any registered claimant (even a low rank)
queued at capacity -- "the UNREGISTERED clients are the eviction pool".
The design rule: "unregistered clients rank below all registered ones".

Two linked exclusion sites in manager.py (as read in the source):
  (1) the SELECTOR, ``_is_worst_ranked_loaded_locked``:
      ``base = self._resident_priority_key(cand, table); if base is None:
      continue`` -- an unresolvable resident never enters the candidate
      pool, so the machine's actual worst resident (an unlisted one) can
      never be selected; the selector names the worst LISTED resident
      instead.
  (2) the PREDICATE veto, ``_is_designated_unload_target_locked``:
      ``victim_key = self._resident_priority_key(r, table); if
      victim_key is None: return False`` -- even a candidate that reached
      the comparison is declared "never the victim" when unresolvable.
      Its own docstring cites the design rule ("unregistered clients rank
      below all registered ones") as the authority for EXCLUDING them --
      the opposite reading of the same sentence: ranking BELOW all
      registered ones means an unregistered resident is the WORST, not
      outside the pool. The design's worked example (above) is the
      controlling text, and ``_resident_unload_priority_key`` already
      encodes the correct reading (unlisted sorts FIRST, tier 0, the
      cheapest to evict) -- the claim-designation path alone reads the
      sentence backwards.

Consequence (shape mirrored from the widening fix's on-wire behaviour):
while a listed claimant strictly outranks every listed resident, the
designated-victim mechanism (grace removal at the turn boundary) can never
name the machine's actual worst resident. The claimant's preemption is
half-implemented: the cross-card widening reaches the unlisted
resident; the grace-skip / requeue-to-tail path (this predicate) does not.
The victim eventually leaves only via ordinary capacity make-room, after
grace has lapsed -- the extra turn the one-turn fairness floor and the
worked example do not require.

The fix:
  (1) SELECTOR: tier the candidate key so unresolvable residents sort as
      WORST than any listed resident -- key ``(tier, rule_index,
      -last_active_monotonic)`` with tier 0 = listed (byte-identical
      relative ordering among listed: ``(-last_active)`` term untouched,
      rank still excluded, as before), tier 1 = unlisted
      (rule_index slot zeroed; recency tie-break among unlisted in the
      same direction the eviction key uses: least-recently-active first).
      ``max`` then names the machine's true worst, listed or not.
  (2) PREDICATE: the ``victim_key is None`` branch becomes an AFFIRMATIVE
      -- an unlisted resident that the (now tier-aware) selector named as
      the machine's worst is the designated victim, because a live listed
      claim always strictly outranks it (the claim gate at the top of the
      method already guarantees ``claim_key is not None``; the grace-window exclusion's
      "no claim -> no victim" is preserved above it, untouched).

Note: the widening's own claimant-side ``claimant_key is not
None`` guard (in the widening fix) remains correct and is
untouched here -- a claimant with no resolvable identity has no ranking
standing, and that question is about the CLAIMANT, not the resident.

Proof arms (mirroring the earlier widening test this file follows):
  - RED (test names say so): the traced defect -- an unlisted
    ACTIVE resident on a machine that also holds a listed resident, with a
    live listed claim that strictly outranks the listed one. Pre-fix the
    selector and the predicate both answer as if the unlisted resident
    were not there. Assertions are on WHO is named (behaviour), never on
    a missing symbol.
  - GREEN: the same arms pass post-fix -- the unlisted resident IS the
    machine's worst AND IS the designated victim.
  - NEGATIVE CONTROLS (pass on both sides -- they are what the fix must
    NOT touch): no live claim -> no designated victim (the grace-window exclusion: grace
    unconditional "registered or not"); equal-ranked claim -> no bump
    (the queue's own strict comparator, equality is not strictly higher);
    listed-vs-listed ordering unchanged (rule 5 victim, rule 1 survivor,
    with or without an unlisted resident absent); the display field
    (``_likely_unload_target_model_tag``) tracks the selector's answer (the display-follows-decision rule: the
    display reflects the decision, not a second opinion).
"""
import asyncio
import ipaddress
import time
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

NEED = 100


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


# rule 0 = HIGHEST priority (the claimant); rule 5 = the listed resident;
# tag_ranks pins the resolved rank for an unclassified {"ip": ...} meta,
# same convention as the other fast-lane tests.
TABLE = [
    _rule(0, "10.0.0.1", tag_ranks={"unclassified": 1}),  # the claimant
    _rule(5, "10.0.0.5", tag_ranks={"unclassified": 1}),  # the listed resident
]


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


def _active_resident(tag, holder_ip, last_active=1.0):
    """ACTIVE resident (the designated-victim population is {ACTIVE, GRACE}).
    ``rank_client_meta`` is authoritative for an ACTIVE resident (the
    idle-meta fallback only applies to IDLE_EVICTABLE)."""
    return Resident(
        model_tag=tag, resident_key=tag,
        state=ResidentState.ACTIVE, last_active_monotonic=last_active,
        main_gpu=0, split_mode="none",
        rank_client_meta={"ip": holder_ip},
    )


def _unlisted_active_resident(tag, last_active=1.0):
    """ACTIVE resident whose identity matches no rule in TABLE --
    unlisted/unresolvable, ``_resident_priority_key`` returns None."""
    return Resident(
        model_tag=tag, resident_key=tag,
        state=ResidentState.ACTIVE, last_active_monotonic=last_active,
        main_gpu=0, split_mode="none",
        rank_client_meta={"ip": "10.0.0.99"},  # matches no rule in TABLE
    )


def _live_claim(mgr, rule_index=0):
    """Register a LIVE listed claim directly in the registry (the same
    shape ``_register_fastlane_claim_locked`` writes in manager.py):
    open completion future, no disconnect, TTL far out ->
    ``_claim_is_live`` returns None (live)."""
    slot = Slot.new("claimant")
    slot.fastlane = _match(rule_index=rule_index)
    slot.completion_future = asyncio.get_running_loop().create_future()
    now = time.monotonic()
    mgr._fastlane_claims["claimant"] = {
        "slot": slot,
        "reason": "test",
        "registered_at_monotonic": now,
        "registered_at_iso": "test",
        "ttl_deadline_monotonic": now + 3600,
    }
    return slot


@pytest.mark.asyncio
class TestUnlistedDesignatedVictim:
    async def test_RED_selector_names_the_unlisted_resident_as_machine_worst(self, mgr):
        """The traced selector defect. Machine: one listed ACTIVE
        resident (rule_index 5) + one unlisted ACTIVE resident. The
        design (the worked example) puts the unlisted one
        BELOW all registered -- it is the machine's worst.

        PRE-FIX (expected RED): the selector's ``base is None: continue``
        drops the unlisted resident from the candidate pool and names the
        listed one -- the machine's actual worst is invisible to
        designation. POST-FIX (expected GREEN): the unlisted resident is
        named the worst.
        """
        a = _active_resident("a-listed", "10.0.0.5", last_active=2.0)
        b = _unlisted_active_resident("b-unlisted", last_active=1.0)
        mgr._residents["a-listed"] = a
        mgr._residents["b-unlisted"] = b
        _live_claim(mgr, rule_index=0)

        assert mgr._is_worst_ranked_loaded_locked(b) is True, (
            "the unlisted resident is the machine's worst by the "
            "design's own worked example (unregistered "
            "clients rank below all registered ones); the selector "
            "excluded it from the candidate pool entirely"
        )
        assert mgr._is_worst_ranked_loaded_locked(a) is False, (
            "the listed resident is NOT the machine's worst while an "
            "unlisted resident is loaded -- naming it is the traced "
            "defect (it selects the worst LISTED, which is not the "
            "machine-wide worst)"
        )

    async def test_RED_predicate_designates_the_unlisted_machine_worst(self, mgr):
        """The traced predicate defect -- the arm for the predicate veto.
        Same machine as the selector arm, plus the trigger: a
        live listed claim (rule_index 0) strictly outranking the listed
        resident. The unlisted resident is the machine's worst AND
        outranked -> it is the designated victim (its grace must be
        removable at the turn boundary, exactly as the architecture's
        eviction pool requires).

        PRE-FIX (expected RED): ``_is_worst_ranked_loaded_locked`` names
        the listed resident, so the predicate answers False for the
        unlisted one; even if it reached the comparison, the
        ``victim_key is None: return False`` veto would decline
        it. POST-FIX (expected GREEN): True for the unlisted resident,
        False for the listed one.
        """
        a = _active_resident("a-listed", "10.0.0.5", last_active=2.0)
        b = _unlisted_active_resident("b-unlisted", last_active=1.0)
        mgr._residents["a-listed"] = a
        mgr._residents["b-unlisted"] = b
        _live_claim(mgr, rule_index=0)

        assert mgr._is_designated_unload_target_locked(b) is True, (
            "the unlisted machine-worst must be designable as the victim "
            "-- the design's worked example makes unregistered residents "
            "the eviction pool, and the live listed claim strictly "
            "outranks it. Pre-fix, the predicate's None-veto (and the "
            "selector's exclusion) make it undesignable: the victim set stays "
            "HALF-GLOBAL."
        )
        assert mgr._is_designated_unload_target_locked(a) is False

    async def test_RED_display_field_tracks_the_machine_worst(self, mgr):
        """``_likely_unload_target_model_tag`` is the dashboard's "why did my
        model get unloaded?" answer (per the design) and must
        track the selector (its docstring: it reads the SELECTOR, freshly,
        per the display-follows-decision rule -- the display reflects the decision, not a second
        opinion). Pre-fix it displays the listed resident as the likely
        victim while the machine's true worst is the unlisted one; post-
        fix it displays the unlisted one.
        """
        a = _active_resident("a-listed", "10.0.0.5", last_active=2.0)
        b = _unlisted_active_resident("b-unlisted", last_active=1.0)
        mgr._residents["a-listed"] = a
        mgr._residents["b-unlisted"] = b
        _live_claim(mgr, rule_index=0)

        assert mgr._likely_unload_target_model_tag() == "b-unlisted", (
            "the display field must show the machine's actual worst "
            "resident (the unlisted one) -- the decision must "
            "reflect reality, not a stale or filtered snapshot"
        )

    async def test_RED_unlisted_tie_breaks_by_recency(self, mgr):
        """Among UNLISTED residents (no listed one loaded), the least
        recently active is the machine's worst -- the same tie direction
        the eviction key uses (``last_active`` ascending within a tier).
        Two unlisted ACTIVE residents: the older one (last_active 1.0)
        must be the worst, the fresher one (9.0) must not be.

        PRE-FIX (expected RED, both False); POST-FIX (expected GREEN).
        """
        old = _unlisted_active_resident("b-old", last_active=1.0)
        new = _unlisted_active_resident("b-new", last_active=9.0)
        mgr._residents["b-old"] = old
        mgr._residents["b-new"] = new
        _live_claim(mgr, rule_index=0)

        assert mgr._is_worst_ranked_loaded_locked(old) is True
        assert mgr._is_worst_ranked_loaded_locked(new) is False
        assert mgr._is_designated_unload_target_locked(old) is True
        assert mgr._is_designated_unload_target_locked(new) is False

    # ------------------------------------------------------------------
    # NEGATIVE CONTROLS -- pass on BOTH sides of the fix. These are the
    # behaviours the fix must NOT touch; a fix that breaks any of them has
    # widened itself past its scope.
    # ------------------------------------------------------------------

    async def test_CONTROL_no_claim_no_designated_victim(self, mgr):
        """Grace is unconditional in this state: every loaded client
        gets it, registered or not. No live claim -> no split -> no
        designated victim. Holds for the unlisted resident too -- the fix
        must not create victims in a no-claim world. (Passes pre-fix AND
        must pass post-fix: the claim gate sits above the new branch.)"""
        b = _unlisted_active_resident("b-unlisted", last_active=1.0)
        a = _active_resident("a-listed", "10.0.0.5", last_active=2.0)
        mgr._residents["b-unlisted"] = b
        mgr._residents["a-listed"] = a
        assert mgr._fastlane_claims == {}

        assert mgr._is_designated_unload_target_locked(b) is False
        assert mgr._is_designated_unload_target_locked(a) is False

    async def test_CONTROL_equal_ranked_claim_does_not_bump(self, mgr):
        """The strict comparator: an EQUAL-ranked claimant does not bump
        an equal-ranked resident (``_fastlane_strictly_higher`` treats
        equality as NOT strictly higher). A rule-5 claim against a sole
        rule-5 listed resident -> no designated victim. Must hold
        pre-fix AND post-fix."""
        a = _active_resident("a-listed", "10.0.0.5", last_active=2.0)
        mgr._residents["a-listed"] = a
        _live_claim(mgr, rule_index=5)  # EQUAL to the resident

        assert mgr._is_designated_unload_target_locked(a) is False

    async def test_CONTROL_listed_ordering_unchanged(self, mgr):
        """The listed-vs-listed ordering is byte-for-byte the pre-fix
        behaviour (the rank-exclusion and recency tie-break are
        untouched): with rule-1 and rule-5 listed residents and a rule-0
        claim, the rule-5 resident is the designated victim and the
        rule-1 one survives. Must hold pre-fix AND post-fix -- the tier
        key must not reorder listed residents among themselves."""
        low = _active_resident("a-rule5", "10.0.0.5", last_active=2.0)   # rule 5
        high = _active_resident("c-rule1", "10.0.0.1", last_active=9.0)  # rule 0's address (listed, above the claimant); see the NOTE below
        # NOTE: TABLE has rules 0 and 5 only. The "higher-priority listed
        # resident" in this two-rule table is rule 0's address -- which is
        # the claimant's own address. To get a SECOND listed resident at a
        # DIFFERENT rule_index we need a third rule; the two-arm ordering
        # is therefore asserted with a rule-5 resident + a rule-0-listed
        # resident (rule 0 is listed in TABLE independently of being the
        # claimant's address -- a resident may match the same rule as the
        # claimant).
        high2 = Resident(
            model_tag="c-rule0", resident_key="c-rule0",
            state=ResidentState.ACTIVE, last_active_monotonic=9.0,
            main_gpu=0, split_mode="none",
            rank_client_meta={"ip": "10.0.0.1"},  # rule 0 -- listed, ABOVE the claimant
        )
        del high
        mgr._residents["a-rule5"] = low
        mgr._residents["c-rule0"] = high2
        _live_claim(mgr, rule_index=0)

        # rule 5 is the worst LISTED resident; the rule-0-listed resident
        # is NOT even outranked by the claim (equality is not strict).
        assert mgr._is_worst_ranked_loaded_locked(low) is True
        assert mgr._is_designated_unload_target_locked(low) is True
        assert mgr._is_designated_unload_target_locked(high2) is False
