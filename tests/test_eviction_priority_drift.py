"""The delegate extraction (_resident_unload_priority_key ->
_unload_priority_key_for_meta, the one deliberate edit to an existing body) plus the
new _resident_priority_key (claim designation) must never disagree about which
resident is worst -- this is the drift test that guards that extraction.

Two properties, both required:
  1. REGRESSION: the delegate's output equals the oracle below, for every synthetic
     input: the inline logic from before the extraction (reconstructed below), with
     the rank slot the unload key carries so that, inside one client, a worse tag
     rank sorts ahead of a better one.
  2. DRIFT: _resident_priority_key (designation) and _resident_unload_priority_key
     (eviction) resolve "listed or not" and "how listed" identically, because both
     route through the same _resolve_fastlane_match -- this test proves the
     property, not just the implementation choice.

Watched RED on two independent mutants, then GREEN on the shipped code.
"""
import ipaddress

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
from turbohaul.fastlane import CompiledRule
from turbohaul.manager import (
    Resident,
    ResidentState,
    TurbohaulManager,
    _unload_priority_key_for_meta,
    _resolve_fastlane_match,
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


TABLE = [
    _rule(0, "10.0.0.1"),   # highest priority
    _rule(1, "10.0.0.2"),
    _rule(2, "10.0.0.3"),   # lowest priority of the three listed rules
]


def _old_inline_key(meta, table, last_active):
    """Reconstruction of _resident_unload_priority_key's body as it was
    BEFORE the delegate extraction (manager.py), carrying the rank slot the key
    has since gained (see the comment below) -- the regression oracle.
    Deliberately duplicated, not imported, so a change to the real
    implementation can't silently drag this oracle along with it."""
    from turbohaul.fastlane import match_fastlane
    match = match_fastlane(table, (meta or {}).get("ip"), meta) if meta else None
    # The key now has a rank slot between rule_index and recency: within one
    # rule_index the worse (larger) rank sorts first under min(). An unlisted
    # resident keeps the zeroed slots.
    if match is None:
        return (0, 0, 0, last_active)
    return (1, -match.rule_index, -match.rank, last_active)


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
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    m = TurbohaulManager(boot, runtime)
    # the unlisted-worst agreement arm exercises the REAL selector, which resolves
    # its table via self._fastlane_table() -- pin it to this module's TABLE.
    m._fastlane_table = lambda: TABLE
    return m


SYNTHETIC_RESIDENTS = [
    # (label, idle_client_meta, last_active_monotonic)
    ("listed_rule0", {"ip": "10.0.0.1"}, 100.0),
    ("listed_rule2_older", {"ip": "10.0.0.3"}, 10.0),
    ("listed_rule2_newer", {"ip": "10.0.0.3"}, 90.0),
    ("unlisted_ip", {"ip": "10.0.0.99"}, 50.0),
    ("unresolvable_none_meta", None, 5.0),
    ("unresolvable_no_ip_key", {}, 5.0),
]


class TestByteIdenticalDelegate:
    """Property 1: the delegate must never diverge from the oracle (the
    pre-extraction inline logic plus the rank slot), for every synthetic
    input above."""

    @pytest.mark.parametrize("label,meta,last_active", SYNTHETIC_RESIDENTS)
    def test_delegate_matches_old_inline_logic(self, mgr, label, meta, last_active):
        r = Resident(
            model_tag=label, state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=last_active, idle_client_meta=meta,
        )
        got = mgr._resident_unload_priority_key(r, TABLE)
        want = _old_inline_key(meta, TABLE, last_active)
        assert got == want, f"{label}: delegate {got} != pre-extraction oracle {want}"

    @pytest.mark.parametrize("label,meta,last_active", SYNTHETIC_RESIDENTS)
    def test_module_level_fn_matches_old_inline_logic(self, label, meta, last_active):
        got = _unload_priority_key_for_meta(meta, TABLE, last_active)
        want = _old_inline_key(meta, TABLE, last_active)
        assert got == want, f"{label}: {got} != {want}"


class TestDesignationCannotDriftFromEviction:
    """Property 2: _resident_priority_key (designation) and
    _resident_unload_priority_key (eviction) must agree about listed-ness
    and relative rank for every synthetic input."""

    @pytest.mark.parametrize("label,meta,last_active", SYNTHETIC_RESIDENTS)
    def test_none_agreement(self, mgr, label, meta, last_active):
        r = Resident(
            model_tag=label, state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=last_active, idle_client_meta=meta,
        )
        designation = mgr._resident_priority_key(r, TABLE)
        eviction = mgr._resident_unload_priority_key(r, TABLE)
        assert (designation is None) == (eviction[0] == 0), (
            f"{label}: designation is-None={designation is None} but "
            f"eviction tier={eviction[0]} disagree about listed-ness"
        )

    @pytest.mark.parametrize("label,meta,last_active", SYNTHETIC_RESIDENTS)
    def test_rule_index_agreement_when_listed(self, mgr, label, meta, last_active):
        r = Resident(
            model_tag=label, state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=last_active, idle_client_meta=meta,
        )
        designation = mgr._resident_priority_key(r, TABLE)
        eviction = mgr._resident_unload_priority_key(r, TABLE)
        if designation is None:
            # Unlisted/unresolvable: no rule_index exists to agree on --
            # listed-ness agreement is covered by test_none_agreement (the
            # frozen _resident_priority_key None contract, still intact after
            # the selector's tier change: that change lives in the SELECTOR, which
            # consumes this function's None as the worst tier), and the
            # tier-selectable + first-eviction-pick invariant is covered by
            # TestDesignationEvictionAgreementUnlisted in this file.
            pytest.skip("unlisted/unresolvable -- no rule_index to compare")
        assert designation[0] == -eviction[1], (
            f"{label}: designation rule_index={designation[0]} != "
            f"-eviction[1]={-eviction[1]}"
        )

    def test_worst_rule_index_is_the_min_eviction_pick(self, mgr):
        """End-to-end: among a pool of listed residents, the one designation
        calls worst (highest rule_index) is the SAME one min() over the real
        eviction key would pick -- the property the whole drift test exists
        to protect, exercised the way _lru_idle_unloadable actually uses it."""
        residents = [
            Resident(model_tag="best", state=ResidentState.IDLE_EVICTABLE,
                     last_active_monotonic=1.0, idle_client_meta={"ip": "10.0.0.1"}),
            Resident(model_tag="mid", state=ResidentState.IDLE_EVICTABLE,
                     last_active_monotonic=1.0, idle_client_meta={"ip": "10.0.0.2"}),
            Resident(model_tag="worst", state=ResidentState.IDLE_EVICTABLE,
                     last_active_monotonic=1.0, idle_client_meta={"ip": "10.0.0.3"}),
        ]
        by_designation_worst = max(
            residents, key=lambda r: mgr._resident_priority_key(r, TABLE)[0]
        )
        by_eviction_min = min(
            residents, key=lambda r: mgr._resident_unload_priority_key(r, TABLE)
        )
        assert by_designation_worst.model_tag == "worst"
        assert by_eviction_min.model_tag == "worst"
        assert by_designation_worst.model_tag == by_eviction_min.model_tag


# ---------------------------------------------------------------------------
# Unlisted machine-worst -- the binding-condition arm. The existing drift
# test pins EVICTION against _old_inline_key, an independent reconstruction --
# it does NOT compare DESIGNATION against EVICTION, so the selector's tier
# change passes it BY CONSTRUCTION and proves nothing. What is required is
# a proof that designation and eviction name the SAME resident when the
# machine-worst is UNLISTED, and that the agreement arm parametrizes over
# the SYNTHETIC_RESIDENTS unlisted cases.
#
# The class above does exactly that for listed pools only. This one
# parametrizes over the three unlisted/unresolvable cases of
# SYNTHETIC_RESIDENTS -- each as a pool's UNIQUE worst -- plus an
# all-unlisted pool and a listed-only control (the control proves the arm
# is not trivially unlisted-only).
# ---------------------------------------------------------------------------

AGREEMENT_POOLS = [
    # (pool label, [(label, meta, last_active), ...]); every pool has a
    # unique worst, so "name the same resident" is unambiguous.
    ("unlisted_ip_worst", [
        ("listed_rule0", {"ip": "10.0.0.1"}, 100.0),
        ("listed_rule2_older", {"ip": "10.0.0.3"}, 10.0),
        ("unlisted_ip", {"ip": "10.0.0.99"}, 50.0),
    ]),
    ("none_meta_worst", [
        ("listed_rule0", {"ip": "10.0.0.1"}, 100.0),
        ("listed_rule2_newer", {"ip": "10.0.0.3"}, 90.0),
        ("unresolvable_none_meta", None, 5.0),
    ]),
    ("no_ip_key_worst", [
        ("listed_rule0", {"ip": "10.0.0.1"}, 100.0),
        ("listed_rule2_older", {"ip": "10.0.0.3"}, 10.0),
        ("unresolvable_no_ip_key", {}, 5.0),
    ]),
    ("all_unlisted_older_worst", [
        ("unlisted_ip", {"ip": "10.0.0.99"}, 50.0),
        ("unresolvable_none_meta", None, 5.0),
        ("unresolvable_no_ip_key", {}, 50.0),
    ]),
    ("listed_worst_control", [
        ("listed_rule0", {"ip": "10.0.0.1"}, 100.0),
        ("listed_rule2_older", {"ip": "10.0.0.3"}, 10.0),
        ("listed_rule2_newer", {"ip": "10.0.0.3"}, 90.0),
    ]),
]


class TestDesignationEvictionAgreementUnlisted:
    """Binding condition: when the machine's
    worst is UNLISTED, DESIGNATION (the real
    ``_is_worst_ranked_loaded_locked`` over ACTIVE residents -- the
    selector the designated-victim predicate is built on) and EVICTION
    (min over the real ``_resident_unload_priority_key``, the delegate
    ``_lru_idle_unloadable`` uses) must name the SAME resident.

    Each side uses its own real production path and its own real
    population (ACTIVE mirrors for designation, IDLE_EVICTABLE mirrors for
    eviction -- the two states the two mechanisms actually filter on), so
    this is the agreement property exercised end-to-end, not a
    key-shape comparison that the old oracle already covers.
    """

    @pytest.mark.parametrize("pool_label, cases", AGREEMENT_POOLS)
    def test_designation_and_eviction_name_the_same_resident(self, mgr, pool_label, cases):
        # EVICTION side: the real delegate key (idle_client_meta is the
        # eviction path's meta source).
        eviction_key = {
            label: mgr._resident_unload_priority_key(
                Resident(model_tag=label, state=ResidentState.IDLE_EVICTABLE,
                         last_active_monotonic=la, idle_client_meta=meta),
                TABLE,
            )
            for label, meta, la in cases
        }
        by_eviction = min(eviction_key, key=eviction_key.get)

        # DESIGNATION side: the REAL selector over ACTIVE mirrors
        # (rank_client_meta is the ACTIVE path's meta source).
        for label, meta, la in cases:
            mgr._residents[label] = Resident(
                model_tag=label, state=ResidentState.ACTIVE,
                last_active_monotonic=la, rank_client_meta=meta,
            )
        designated = [
            r.model_tag for r in mgr._model_residents()
            if mgr._is_worst_ranked_loaded_locked(r)
        ]

        assert designated == [by_eviction], (
            f"{pool_label}: designation named {designated!r} but eviction's "
            f"min() is {by_eviction!r} -- when the machine's worst is "
            f"unlisted, designation and eviction MUST name the same "
            f"resident (binding condition; this is "
            f"the one way the selector's tier change could silently "
            f"diverge)"
        )



class TestTieBreakIsLeastRecentlyActive:
    """Independent re-derivation: on a real same-rule_index
    tie, min() picks the SMALLEST last_active_monotonic -- the LEAST-recently-
    active resident is evicted, not the most-recently-active."""

    def test_least_recently_active_loses_on_tie(self, mgr):
        older = Resident(
            model_tag="older", state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=10.0, idle_client_meta={"ip": "10.0.0.3"},
        )
        newer = Resident(
            model_tag="newer", state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=90.0, idle_client_meta={"ip": "10.0.0.3"},
        )
        victim = min(
            [older, newer], key=lambda r: mgr._resident_unload_priority_key(r, TABLE)
        )
        assert victim.model_tag == "older", (
            "tie-break must evict the LEAST-recently-active resident "
            "(smallest last_active_monotonic), not the most-recently-active"
        )


class TestResolveFastlaneMatchSharedByBoth:
    """_resolve_fastlane_match is the single resolution point -- assert both
    consumers actually call it (not a re-duplicated inline copy) by checking
    they agree on a case a duplicated-and-drifted copy would get wrong: an
    IP that matches rule 2 but with a tag_ranks entry affecting only `rank`,
    never `rule_index` -- rule_index must be identical either way."""

    def test_rank_only_difference_does_not_affect_rule_index_agreement(self, mgr):
        ranked_table = [
            _rule(0, "10.0.0.1"),
            _rule(1, "10.0.0.2", tag_ranks={"main": 1}),
        ]
        r = Resident(
            model_tag="ranked", state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=1.0,
            idle_client_meta={"ip": "10.0.0.2", "role": "user-message"},
        )
        designation = mgr._resident_priority_key(r, ranked_table)
        eviction = mgr._resident_unload_priority_key(r, ranked_table)
        assert designation is not None
        assert designation[0] == 1
        # The key is (tier, -rule_index, -rank, last_active): rule_index 1 gives -1; the
        # role "user-message" resolves to the unclassified tag, which this rule does not
        # list, so rank 6 gives -6; last_active is 1.0. The rank slot is separate
        # from the rule_index slot, which still agrees with designation.
        assert eviction == (1, -1, -6, 1.0)


def test_resolve_fastlane_match_is_the_shared_primitive():
    """Direct check on the extracted helper itself: same (meta, table) in,
    same FastLaneMatch (or None) out, for both intended call shapes."""
    meta = {"ip": "10.0.0.1"}
    match = _resolve_fastlane_match(meta, TABLE)
    assert match is not None
    assert match.rule_index == 0
    assert _resolve_fastlane_match(None, TABLE) is None
    assert _resolve_fastlane_match({}, TABLE) is None
    assert _resolve_fastlane_match({"ip": "10.0.0.99"}, TABLE) is None


class TestMidTurnIdentity:
    """Mid-turn identity: a mid-turn resident's identity
    for claim-driven designation must come from rank_client_meta (stamped at
    reserve + every ACTIVE transition), never from idle_client_meta -- that
    field reflects whoever was served BEFORE the resident last parked, which
    is a WRONG identity (not merely a missing one) for a resident now mid-
    turn for a DIFFERENT client. Violates the reflect-reality rule ("the decision must reflect
    reality, not a stale snapshot") otherwise.

    T-a: a listed resident on its FIRST turn ever (idle_client_meta still
    None -- it has never parked) must still resolve correctly once
    rank_client_meta is stamped at reserve.
    T-b: a resident whose CURRENT turn's client differs from the client its
    idle_client_meta remembers from a PREVIOUS park must resolve to the
    CURRENT client, not the stale one.
    """

    def test_ta_first_turn_ever_resolves_via_rank_client_meta(self, mgr):
        r = Resident(
            model_tag="fresh", state=ResidentState.ACTIVE,
            last_active_monotonic=1.0,
            idle_client_meta=None,  # never parked -- this is its first turn
            rank_client_meta={"ip": "10.0.0.1"},  # stamped at reserve
        )
        designation = mgr._resident_priority_key(r, TABLE)
        assert designation is not None, (
            "a first-turn resident with rank_client_meta stamped must NOT be "
            "unresolvable -- idle_client_meta being None is expected on a "
            "first turn, it must not be the only thing consulted"
        )
        assert designation == (0, 6)  # rule 0, rank UNRANKED (TABLE sets no tag_ranks)

    def test_tb_current_turn_client_wins_over_stale_idle_client_meta(self, mgr):
        r = Resident(
            model_tag="reused", state=ResidentState.ACTIVE,
            last_active_monotonic=1.0,
            # STALE: whoever this resident served on a PREVIOUS turn, before
            # it last parked -- rule 2 (worst of the three listed rules).
            idle_client_meta={"ip": "10.0.0.3"},
            # CURRENT: a DIFFERENT client is being served THIS turn, after
            # the resident was reclaimed from idle -- rule 0 (best).
            rank_client_meta={"ip": "10.0.0.1"},
        )
        designation = mgr._resident_priority_key(r, TABLE)
        assert designation == (0, 6), (  # rule 0, rank UNRANKED (TABLE sets no tag_ranks)
            f"got {designation}, expected rule 0 (the CURRENT client, from "
            "rank_client_meta) -- resolving to rule 2 (idle_client_meta, the "
            "PREVIOUS client) would be a WRONG answer, not a missing one, and "
            "violates the reflect-reality rule's \"reflect reality, not a stale snapshot\""
        )

    def test_parked_resident_still_uses_idle_client_meta(self, mgr):
        """The fallback exists FOR the parked case -- must not regress it.
        A parked resident has no current turn, so idle_client_meta (whoever
        it just finished serving) is the only meaningful identity."""
        r = Resident(
            model_tag="parked", state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=1.0,
            idle_client_meta={"ip": "10.0.0.3"},
            rank_client_meta=None,  # cleared/never restamped since parking
        )
        designation = mgr._resident_priority_key(r, TABLE)
        assert designation == (2, 6)  # rule 2, rank UNRANKED (TABLE sets no tag_ranks)

    def test_neither_present_is_unresolvable_not_a_guess(self, mgr):
        r = Resident(
            model_tag="blind", state=ResidentState.ACTIVE,
            last_active_monotonic=1.0, idle_client_meta=None, rank_client_meta=None,
        )
        assert mgr._resident_priority_key(r, TABLE) is None
