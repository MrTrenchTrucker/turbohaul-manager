"""Tag rank (main=1, curator=3, ...) decides which loaded resident of one client is
evicted first, and whether a waiting claim outranks a resident of that client.

Expected direction asserted throughout: for ONE matched client (one rule), the
better-ranked tag is kept and the worse-ranked tag is evicted first. Each probe
OBSERVES which resident is named (never just that a rank value was computed).
Every group has a control with a different rule_index to prove the instrument
can see an ordering difference.

Within one client, the loaded-resident ordering, the claim-versus-resident check
and the idle-eviction key all agree with the pure queue comparator: the worse
tag rank is unloaded first, and a better-ranked claim outranks a worse-ranked
resident of the same client. Rank is never compared across clients.

Free VRAM: nothing here reads free VRAM (the checks are selection and comparison
over resident and claim keys). Both the `turbohaul.safety` and the
`turbohaul.manager` binding of the reader are pinned by an autouse fixture that
raises if it is ever called.
"""
import asyncio
import ipaddress
import time

import pytest

from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.fastlane import CompiledRule, FastLaneMatch
from turbohaul.manager import (
    Resident, ResidentState, TurbohaulManager, _unload_priority_key_for_meta,
)
from turbohaul.slot import Slot


# The tests in this file never read free VRAM. The manager's binding of the reader is
# pinned anyway, for both the manager and the safety module, so a future change that
# reaches it fails loudly here instead of reading the live GPU.
@pytest.fixture(autouse=True)
def _pin_free_vram_autouse(monkeypatch):
    import turbohaul.manager as manager_module
    import turbohaul.safety as safety_module

    def _must_not_read(*args, **kwargs):
        raise AssertionError("a test of this file read free VRAM")

    monkeypatch.setattr(manager_module, "_read_free_vram_all_mib", _must_not_read)
    monkeypatch.setattr(safety_module, "_read_free_vram_all_mib", _must_not_read)


MAIN_RANK = 1
CURATOR_RANK = 3
IP_RULE0 = "192.0.2.10"
IP_RULE2 = "192.0.2.30"
IP_UNLISTED = "198.51.100.99"
RANKS = {"main": MAIN_RANK, "curator": CURATOR_RANK, "unclassified": 5}

MAIN_META = {"is_main": True}
CURATOR_META = {"is_curator": True}


def _rule(index, raw_address):
    addr = ipaddress.ip_address(raw_address)
    return CompiledRule(
        index=index, raw_address=raw_address, address=addr, container_name=None,
        match_addresses=frozenset({addr}), label=f"rule{index}", tag_ranks=dict(RANKS),
    )


TABLE = [_rule(0, IP_RULE0), _rule(2, IP_RULE2)]


@pytest.fixture
def mgr(tmp_path):
    root = tmp_path / "state"
    root.mkdir()
    for d in ("blobs", "manifests", "import-staging"):
        (root / d).mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs", manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging",
            state_db_path=root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server", default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False, max_parallel_sidecars=8), pull=PullConfig(),
    )
    m = TurbohaulManager(boot, runtime)
    m._fastlane_table = lambda: TABLE
    return m


def _resident(tag, ip, tag_meta, last_active):
    """A loaded (ACTIVE) resident whose current-turn identity is ip + class labels."""
    return Resident(
        model_tag=tag, resident_key=tag, state=ResidentState.ACTIVE,
        last_active_monotonic=last_active, main_gpu=0, split_mode="none",
        rank_client_meta={"ip": ip, **tag_meta},
    )


def _load(mgr, *residents):
    for r in residents:
        mgr._residents[r.model_tag] = r


def _same_rule_pair(mgr, *, curator_last_active, main_last_active):
    """X = main (rank 1), Y = curator (rank 3), both matched to rule 0."""
    x = _resident("res-main", IP_RULE0, MAIN_META, main_last_active)
    y = _resident("res-curator", IP_RULE0, CURATOR_META, curator_last_active)
    _load(mgr, x, y)
    # Anti-vacuity: both identities resolve, to the SAME rule with DIFFERENT ranks.
    assert mgr._resident_priority_key(x, TABLE) == (0, MAIN_RANK)
    assert mgr._resident_priority_key(y, TABLE) == (0, CURATOR_RANK)
    return x, y


def _order(mgr):
    return [r.model_tag for r in mgr._ranked_loaded_candidates_locked()]


def _live_claim(mgr, rule_index, rank):
    """Register a live claim directly (open future, TTL far out)."""
    slot = Slot.new("claimant")
    slot.fastlane = FastLaneMatch(
        rule_index=rule_index, raw_address="192.0.2.10", label="", effective_tag="main", rank=rank,
    )
    slot.completion_future = asyncio.get_running_loop().create_future()
    now = time.monotonic()
    mgr._fastlane_claims["claimant"] = {
        "slot": slot, "reason": "test", "registered_at_monotonic": now,
        "registered_at_iso": "test", "ttl_deadline_monotonic": now + 3600,
    }
    return slot


def _unload_key(ip, tag_meta, last_active):
    return _unload_priority_key_for_meta({"ip": ip, **tag_meta}, TABLE, last_active)


@pytest.mark.asyncio
class TestVictimOrderWithinOneRule:
    async def test_worse_ranked_resident_listed_first_when_it_is_also_older(self, mgr):
        """The victim order is [curator, main]: the worse-ranked resident is listed first, and here it is also the older one."""
        _same_rule_pair(mgr, curator_last_active=1.0, main_last_active=2.0)
        assert _order(mgr) == ["res-curator", "res-main"]

    async def test_worse_ranked_resident_listed_first_when_it_is_more_recent(self, mgr):
        """The victim order is [curator, main] even though the curator ran more recently: within one client the worse-ranked resident is unloaded before the better-ranked one, regardless of recency."""
        _same_rule_pair(mgr, curator_last_active=2.0, main_last_active=1.0)
        order = _order(mgr)
        assert order == ["res-curator", "res-main"], (
            f"victim order {order}: the rank-3 (curator) resident was not named first "
            "while it was the more recently active one -- recency, not rank, decided"
        )

    async def test_victim_order_does_not_change_when_recency_is_flipped(self, mgr):
        """The victim order is the same under both recency assignments, because within one client rank alone decides between residents of different rank."""
        _same_rule_pair(mgr, curator_last_active=1.0, main_last_active=2.0)
        older_curator = _order(mgr)
        mgr._residents.clear()
        _same_rule_pair(mgr, curator_last_active=2.0, main_last_active=1.0)
        newer_curator = _order(mgr)
        assert older_curator == newer_curator, (
            f"flipping only last_active changed the order: {older_curator} -> {newer_curator}; "
            "recency alone decides between same-rule residents of different rank"
        )

    async def test_control_rule_index_flips_the_victim_with_recency_held_fixed(self, mgr):
        """Control: with identical recency, moving the same tag to rule 2 makes it the victim."""
        a = _resident("res-a", IP_RULE0, MAIN_META, 1.0)
        b = _resident("res-b", IP_RULE2, MAIN_META, 1.0)
        _load(mgr, a, b)
        assert _order(mgr) == ["res-b", "res-a"]
        mgr._residents.clear()
        a = _resident("res-a", IP_RULE2, MAIN_META, 1.0)
        b = _resident("res-b", IP_RULE0, MAIN_META, 1.0)
        _load(mgr, a, b)
        assert _order(mgr) == ["res-a", "res-b"]

    async def test_control_recency_alone_flips_the_victim_for_equal_rank_same_rule(self, mgr):
        """Control: the instrument sees recency -- the older of two equal (rule, rank) residents goes first."""
        _load(mgr, _resident("res-a", IP_RULE0, MAIN_META, 1.0),
              _resident("res-b", IP_RULE0, MAIN_META, 2.0))
        assert _order(mgr) == ["res-a", "res-b"]
        mgr._residents.clear()
        _load(mgr, _resident("res-a", IP_RULE0, MAIN_META, 2.0),
              _resident("res-b", IP_RULE0, MAIN_META, 1.0))
        assert _order(mgr) == ["res-b", "res-a"]


@pytest.mark.asyncio
class TestClaimVersusResident:
    async def test_better_ranked_claim_of_same_rule_outranks_the_worse_ranked_resident(self, mgr):
        """A rank-1 claim of rule 0 outranks the rank-3 resident of rule 0, so that resident is the designated victim: within one client a better-ranked claim outranks a worse-ranked resident."""
        x, y = _same_rule_pair(mgr, curator_last_active=1.0, main_last_active=2.0)
        _live_claim(mgr, rule_index=0, rank=MAIN_RANK)
        assert _order(mgr)[0] == "res-curator"  # selector is not the confound
        assert mgr._is_designated_unload_target_locked(y) is True, (
            "a rank-1 claim of rule 0 did not outrank the rank-3 resident of rule 0 "
            "(the predicate zeroes rank on both sides and sees an equal key)"
        )

    async def test_equal_rank_claim_of_same_rule_does_not_outrank_equal_rank_resident(self, mgr):
        """Control: an equal (rule, rank) claim does not bump the resident."""
        x = _resident("res-main", IP_RULE0, MAIN_META, 1.0)
        _load(mgr, x)
        _live_claim(mgr, rule_index=0, rank=MAIN_RANK)
        assert mgr._is_designated_unload_target_locked(x) is False

    async def test_control_claim_of_better_rule_outranks_resident(self, mgr):
        """Control: a rule-0 claim outranks a rule-2 resident, so the instrument sees rule_index."""
        r = _resident("res-rule2", IP_RULE2, MAIN_META, 1.0)
        _load(mgr, r)
        _live_claim(mgr, rule_index=0, rank=MAIN_RANK)
        assert mgr._is_designated_unload_target_locked(r) is True

    async def test_control_claim_of_worse_rule_does_not_outrank_resident(self, mgr):
        """Control: a rule-2 claim does not outrank a rule-0 resident."""
        r = _resident("res-rule0", IP_RULE0, CURATOR_META, 1.0)
        _load(mgr, r)
        _live_claim(mgr, rule_index=2, rank=MAIN_RANK)
        assert mgr._is_designated_unload_target_locked(r) is False

    async def test_manager_predicate_agrees_with_the_pure_comparator_on_the_same_pair(self, mgr):
        """The manager's claim-versus-resident predicate gives the same answer as the pure comparator queue._fastlane_strictly_higher for (0,1) vs (0,3): the better-ranked claim is strictly higher."""
        x, y = _same_rule_pair(mgr, curator_last_active=1.0, main_last_active=2.0)
        _live_claim(mgr, rule_index=0, rank=MAIN_RANK)
        pure = mgr.queue._fastlane_strictly_higher((0, MAIN_RANK), (0, CURATOR_RANK))
        manager = mgr._is_designated_unload_target_locked(y)
        assert pure is True  # the pure comparator reads rank
        assert manager == pure, (
            f"pure comparator says {pure} for (0,1) vs (0,3); the manager's "
            f"claim-vs-resident predicate says {manager} for the same pair"
        )


class TestPureComparator:
    def test_comparator_reads_rank_within_one_rule(self, mgr):
        """(0,1) is strictly higher than (0,3) and not the reverse."""
        higher = mgr.queue._fastlane_strictly_higher
        assert higher((0, 1), (0, 3)) is True
        assert higher((0, 3), (0, 1)) is False

    def test_control_comparator_reads_rule_index_and_treats_equal_as_not_higher(self, mgr):
        """Control: (0,3) beats (2,1) on rule_index; equal keys are not strictly higher."""
        higher = mgr.queue._fastlane_strictly_higher
        assert higher((0, 3), (2, 1)) is True
        assert higher((2, 1), (0, 3)) is False
        assert higher((0, 1), (0, 1)) is False


class TestIdleEvictionKey:
    def test_worse_ranked_idle_resident_sorts_before_better_ranked_at_equal_recency(self):
        """The curator (rank 3) key sorts below the main (rank 1) key at equal recency: within one client the worse-ranked resident is unloaded first."""
        main_key = _unload_key(IP_RULE0, MAIN_META, 5.0)
        curator_key = _unload_key(IP_RULE0, CURATOR_META, 5.0)
        assert curator_key < main_key, (
            f"eviction keys main={main_key} curator={curator_key}: rank is not part of the key"
        )

    def test_worse_ranked_idle_resident_sorts_first_even_when_it_is_more_recent(self):
        """The more recently active curator (rank 3) still sorts before an older main (rank 1): within one client rank outweighs recency."""
        main_key = _unload_key(IP_RULE0, MAIN_META, 1.0)
        curator_key = _unload_key(IP_RULE0, CURATOR_META, 2.0)
        assert curator_key < main_key, (
            f"eviction keys main={main_key} curator={curator_key}: recency decided, not rank"
        )

    def test_control_higher_rule_index_sorts_first_regardless_of_recency(self):
        """Control: a rule-2 resident sorts before a rule-0 resident whichever is more recent."""
        assert _unload_key(IP_RULE2, MAIN_META, 9.0) < _unload_key(IP_RULE0, MAIN_META, 1.0)
        assert _unload_key(IP_RULE2, MAIN_META, 1.0) < _unload_key(IP_RULE0, MAIN_META, 9.0)


@pytest.mark.asyncio
class TestRuleIndexAndUnlistedControls:
    async def test_higher_rule_index_resident_is_victim_regardless_of_recency(self, mgr):
        """Control: the rule-2 resident is listed first whether it is older or more recent."""
        for far_old, far_new in ((1.0, 2.0), (2.0, 1.0)):
            mgr._residents.clear()
            _load(mgr, _resident("res-rule0", IP_RULE0, MAIN_META, far_old),
                  _resident("res-rule2", IP_RULE2, MAIN_META, far_new))
            assert _order(mgr)[0] == "res-rule2"

    async def test_unlisted_resident_is_victim_before_any_listed_resident(self, mgr):
        """Control: an unlisted resident is listed first even against a rule-2 resident that is older."""
        _load(mgr, _resident("res-rule2", IP_RULE2, MAIN_META, 1.0),
              _resident("res-unlisted", IP_UNLISTED, MAIN_META, 9.0))
        assert _order(mgr) == ["res-unlisted", "res-rule2"]


# (resident client rule, resident tag meta, claim rule, claim rank, claim designates the resident?)
CROSS_CLIENT_CASES = [
    # claim from the WORSE client (rule 2) against a rule-0 resident: never, whatever the tags
    ("worse_client_better_tag_vs_best_tag", IP_RULE0, MAIN_META, 2, MAIN_RANK, False),
    ("worse_client_better_tag_vs_worst_tag", IP_RULE0, CURATOR_META, 2, MAIN_RANK, False),
    ("worse_client_worse_tag_vs_best_tag", IP_RULE0, MAIN_META, 2, CURATOR_RANK, False),
    # claim from the BETTER client (rule 0) against a rule-2 resident: always, whatever the tags
    ("better_client_worse_tag_vs_best_tag", IP_RULE2, MAIN_META, 0, CURATOR_RANK, True),
    ("better_client_worse_tag_vs_worst_tag", IP_RULE2, CURATOR_META, 0, CURATOR_RANK, True),
    ("better_client_better_tag_vs_worst_tag", IP_RULE2, CURATOR_META, 0, MAIN_RANK, True),
    # the same client: the tag decides
    ("same_client_better_tag", IP_RULE0, CURATOR_META, 0, MAIN_RANK, True),
    ("same_client_worse_tag", IP_RULE0, MAIN_META, 0, CURATOR_RANK, False),
    ("same_client_equal_tag", IP_RULE0, CURATOR_META, 0, CURATOR_RANK, False),
]


@pytest.mark.asyncio
class TestRankIsNeverComparedAcrossClients:
    @pytest.mark.parametrize(
        "case_id, res_ip, res_tag, claim_rule, claim_rank, designates",
        CROSS_CLIENT_CASES, ids=[c[0] for c in CROSS_CLIENT_CASES])
    async def test_claim_designates_a_resident_by_client_first_then_tag(
            self, mgr, case_id, res_ip, res_tag, claim_rule, claim_rank, designates):
        """Designation of a resident by a live claim: client order (rule_index) decides, and the
        tag rank only decides between a claim and a resident of the SAME client. The claim of
        a worse client never designates a better client's resident, whatever the tags; the
        claim of a better client always does."""
        r = _resident("res", res_ip, res_tag, 1.0)
        _load(mgr, r)
        _live_claim(mgr, rule_index=claim_rule, rank=claim_rank)
        assert mgr._is_designated_unload_target_locked(r) is designates, (
            f"{case_id}: resident key {mgr._resident_priority_key(r, TABLE)}, claim key "
            f"({claim_rule}, {claim_rank}): expected designated={designates}")


def _idle(tag, ip, tag_meta, last_active):
    """An idle, evictable resident whose last served turn carries ip + class labels."""
    return Resident(
        model_tag=tag, resident_key=tag, state=ResidentState.IDLE_EVICTABLE,
        last_active_monotonic=last_active, main_gpu=0, split_mode="none",
        idle_client_meta={"ip": ip, **tag_meta},
    )


def _claimant(rule_index, rank):
    slot = Slot.new("claimant")
    slot.fastlane = FastLaneMatch(
        rule_index=rule_index, raw_address="192.0.2.10", label="", effective_tag="main", rank=rank,
    )
    return slot


@pytest.mark.asyncio
class TestOutrankGate:
    """`_fastlane_outrank_unload_target_locked`: the gate of the VRAM make-room arms returns the
    globally worst idle resident only when the claimant strictly outranks it."""

    @pytest.mark.parametrize(
        "case_id, res_ip, res_tag, claim_rule, claim_rank, returns_resident",
        CROSS_CLIENT_CASES, ids=[c[0] for c in CROSS_CLIENT_CASES])
    async def test_gate_names_the_idle_resident_by_client_first_then_tag(
            self, mgr, case_id, res_ip, res_tag, claim_rule, claim_rank, returns_resident):
        _load(mgr, _idle("res", res_ip, res_tag, 1.0))
        got = mgr._fastlane_outrank_unload_target_locked(_claimant(claim_rule, claim_rank), "none")
        assert (got is not None) is returns_resident, (
            f"{case_id}: gate returned {getattr(got, 'model_tag', None)!r}, "
            f"expected a resident: {returns_resident}")

    async def test_gate_names_the_worse_tag_of_the_claimants_own_client_not_the_equal_one(self, mgr):
        """Two idle residents of one client: a main (rank 1) and a curator (rank 3). A main
        claimant of that client outranks only the curator, so the gate names the curator
        resident, whichever of the two ran more recently."""
        for main_last, cur_last in ((1.0, 2.0), (2.0, 1.0)):
            mgr._residents.clear()
            _load(mgr, _idle("res-main", IP_RULE0, MAIN_META, main_last),
                  _idle("res-curator", IP_RULE0, CURATOR_META, cur_last))
            got = mgr._fastlane_outrank_unload_target_locked(_claimant(0, MAIN_RANK), "none")
            assert got is not None and got.model_tag == "res-curator", (
                f"gate named {getattr(got, 'model_tag', None)!r} (main ran last: {main_last > cur_last})")

    async def test_gate_declines_when_the_only_idle_resident_has_an_equal_key(self, mgr):
        """Control: an equal (rule, rank) claimant does not outrank, so the gate names nothing."""
        _load(mgr, _idle("res-main", IP_RULE0, MAIN_META, 1.0))
        assert mgr._fastlane_outrank_unload_target_locked(_claimant(0, MAIN_RANK), "none") is None


class TestOneIdentityForRank:
    """The resident's rank is read from the identity of its LATEST turn, by one reader, for
    the designation key, the outrank gate and the idle-eviction key alike."""

    def _keys(self, mgr, r):
        return (mgr._resident_priority_key(r, TABLE),
                mgr._resident_unload_priority_key(r, TABLE))

    def test_a_stamped_latest_turn_wins_over_the_idle_stash_in_both_keys(self, mgr):
        """The latest turn is main (rank 1), the idle stash still says rank 5: both keys
        carry rank 1."""
        r = Resident(
            model_tag="res", resident_key="res", state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=1.0, rank_client_meta={"ip": IP_RULE0, **MAIN_META},
            idle_client_meta={"ip": IP_RULE0},
        )
        designation, eviction = self._keys(mgr, r)
        assert designation == (0, MAIN_RANK)
        assert eviction[:3] == (1, 0, -MAIN_RANK), f"eviction key {eviction} reads another turn"

    def test_a_stamped_worse_latest_turn_wins_over_a_better_idle_stash_in_both_keys(self, mgr):
        r = Resident(
            model_tag="res", resident_key="res", state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=1.0, rank_client_meta={"ip": IP_RULE0},
            idle_client_meta={"ip": IP_RULE0, **MAIN_META},
        )
        designation, eviction = self._keys(mgr, r)
        assert designation == (0, 5)
        assert eviction[:3] == (1, 0, -5), f"eviction key {eviction} reads another turn"

    def test_control_an_idle_resident_with_nothing_stamped_uses_its_idle_stash(self, mgr):
        r = Resident(
            model_tag="res", resident_key="res", state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=1.0, idle_client_meta={"ip": IP_RULE0, **CURATOR_META},
        )
        designation, eviction = self._keys(mgr, r)
        assert designation == (0, CURATOR_RANK)
        assert eviction[:3] == (1, 0, -CURATOR_RANK)

    def test_control_a_running_resident_with_nothing_stamped_is_unresolvable_in_both(self, mgr):
        r = Resident(
            model_tag="res", resident_key="res", state=ResidentState.ACTIVE,
            last_active_monotonic=1.0, idle_client_meta={"ip": IP_RULE0, **CURATOR_META},
        )
        designation, eviction = self._keys(mgr, r)
        assert designation is None
        assert eviction[:3] == (0, 0, 0), "an unresolvable resident must sort as unlisted"

    def test_control_an_unlisted_resident_stays_unlisted_in_both(self, mgr):
        r = Resident(
            model_tag="res", resident_key="res", state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=1.0, rank_client_meta={"ip": IP_UNLISTED, **MAIN_META},
            idle_client_meta={"ip": IP_RULE0, **MAIN_META},
        )
        designation, eviction = self._keys(mgr, r)
        assert designation is None
        assert eviction[:3] == (0, 0, 0)
