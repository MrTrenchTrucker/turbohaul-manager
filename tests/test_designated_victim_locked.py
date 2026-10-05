"""Selection predicate: `_is_worst_ranked_loaded_locked` -- the
SELECTION half of the designated-victim check.

RE-POINTED at the selector. Every assertion in this file was written against
`_is_designated_unload_target_locked` when that one function did BOTH jobs: it selected the
worst-ranked loaded resident AND was consulted as "is this the designated victim".
It never asked WHO was claiming, so with a single loaded resident it was
unconditionally True -- so it could evict `rule_index` 0 for a
`rule_index` 3 claimant (a priority inversion), and strip grace with nothing claiming at
all (grace stripped for no claimant).

The predicate is now split, so there is still exactly one place that decides each
question:
    `_is_worst_ranked_loaded_locked(r)`  -- SELECTION only (this file)
    `_is_designated_unload_target_locked(r)`    -- selection AND the unload rule's precondition,
                                            a LIVE claim that STRICTLY outranks r
                                            (the designated-victim claim test)

Every test below was ALWAYS testing selection -- worst-rank, tie-break, the loaded
pool, unresolvable exclusion, freshness -- so each is re-pointed at the selector
under its real name and its assertions are UNCHANGED, character for character. That
is deliberate: had they been left calling the gated predicate with no claim
registered, every `is True` would have gone red and, far worse, every `is False`
would have kept passing VACUOUSLY -- true because no claim exists rather than
because of rank. A suite full of greens that could not have been red is a worse
outcome than a suite that fails honestly.
One test, test_designation_agrees_with_eviction_when_same_rule_different_rank,
asserts that tag rank is part of the resident ordering inside one client:
its assertions state that, on purpose.
The original header follows.

"""

"""The designated-victim check, the manager-side half of
the shared queue/manager seam. The name is the contract: the queue side probes for it by
this exact name; a rename silently protects everyone and the queue-side arm passes for the
wrong reason.

Semantics: loaded = ResidentState in {ACTIVE, GRACE} (GRACE has 0 assignments today,
see TestSelectionLogic.test_grace_state_is_in_the_loaded_pool's note); worst =
max(pool, key=(rule_index, rank, -last_active_monotonic)) -- rule_index first, then,
inside one rule_index, the worse (numerically larger) tag rank, then recency.
An unresolvable resident ranks below every listed one and sorts worst. TAG RANK IS
COMPARED ONLY BETWEEN RESIDENTS OF THE SAME CLIENT (same rule_index), the way
Fast Lane compares clients: it decides which of one client's residents gives way
first, it never lifts one client above another, and equal ranks are a tie that
falls through to recency. This predicate and _unload_priority_key_for_meta (whose key
is (tier, -rule_index, -rank, last_active)) therefore order residents the same way,
so two residents matched to the SAME rule but classified to different tags (hence
different rank) can never make designation and eviction disagree. See
TestNeverDriftsFromEvictionAcrossRank for the pinning test. Ties between equal
(rule_index, rank) break by recency, same direction _unload_priority_key_for_meta
already breaks its own ties (the Fast Lane principle: the victim sort
already implements lowest-priority-first, recency breaking ties only
within a rank) -- applied inside this predicate's own local selection, NOT by
widening _resident_priority_key's frozen 2-tuple return shape (that shape stays
(rule_index, rank); fastlane_claims_snapshot still shows rank).

Three test styles, deliberately:
  - TestSelectionLogic: synthetic direct-construction, fast and deterministic,
    exercises the predicate's own selection/tie-break/exclusion logic in isolation
    (same style the eviction-priority drift test uses for its own synthetic residents).
  - TestNeverDriftsFromEvictionAcrossRank: the rank pinning test -- same rule_index,
    different rank via real tag classification (is_curator/is_main), proving
    designation and eviction never disagree even when rank varies.
  - TestAgainstDrivenResidents: a REAL driven-resident test using the shared
    Fast Lane fixture module -- drives ACTUAL mid-turn residents through
    submit_and_wait, so the predicate is proven against the real rank_client_meta stamping path,
    not just synthetic Resident(...) construction.

Verified to fail on a mutant of this new code, then to pass.
"""
import asyncio

import pytest

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
from turbohaul.manager import Resident, ResidentState, TurbohaulManager

# List index IS the priority (FastLaneConfig.rules docstring) -- 10.0.0.1 is rule_index
# 0 (highest), 10.0.0.3 is rule_index 2 (lowest of the three). main=1 for 10.0.0.2/
# 10.0.0.3: irrelevant to those tests, rule_index (list position) is what's exercised
# there. 10.0.0.1 ALSO carries curator=5, used only by the same-rule_index,
# different-rank pinning test below -- every other test on 10.0.0.1 sends a plain {"ip": ...} meta
# with no classification labels, which resolves to "unclassified" (unranked), so this
# addition changes nothing for them.
_RULES = [
    # main=1, curator=5: lets one client (10.0.0.1) resolve to TWO different
    # ranks depending on the REQUEST's own classification labels -- needed to
    # pin the same-rule_index, different-rank case.
    FastLaneRule(address="10.0.0.1", tag_ranks=FastLaneTagRanks(main=1, curator=5)),
    FastLaneRule(address="10.0.0.2", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address="10.0.0.3", tag_ranks=FastLaneTagRanks(main=1)),
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
            default_port_base=59600,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False), pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=_RULES),
    )
    return TurbohaulManager(boot, runtime)


def _put(mgr, label, *, state, meta, last_active):
    r = Resident(
        model_tag=label, resident_key=label, state=state,
        last_active_monotonic=last_active, rank_client_meta=meta,
    )
    mgr._residents[label] = r
    return r


class TestSelectionLogic:
    """Synthetic, direct-construction: the predicate's own selection logic,
    independent of the async driving machinery."""

    def test_worse_rank_is_the_victim(self, mgr):
        good = _put(mgr, "good", state=ResidentState.ACTIVE,
                    meta={"ip": "10.0.0.1"}, last_active=100.0)
        bad = _put(mgr, "bad", state=ResidentState.ACTIVE,
                   meta={"ip": "10.0.0.3"}, last_active=100.0)
        assert mgr._is_worst_ranked_loaded_locked(bad) is True
        assert mgr._is_worst_ranked_loaded_locked(good) is False

    def test_grace_state_is_in_the_loaded_pool(self, mgr):
        """A resident already past its own turn (GRACE) is still 'loaded' --
        it can itself be the worst-ranked resident in the pool."""
        active_good = _put(mgr, "active_good", state=ResidentState.ACTIVE,
                           meta={"ip": "10.0.0.1"}, last_active=100.0)
        grace_bad = _put(mgr, "grace_bad", state=ResidentState.GRACE,
                         meta={"ip": "10.0.0.3"}, last_active=100.0)
        assert mgr._is_worst_ranked_loaded_locked(grace_bad) is True
        assert mgr._is_worst_ranked_loaded_locked(active_good) is False

    def test_idle_evictable_is_never_the_victim(self, mgr):
        """IDLE_EVICTABLE is a different pool entirely (_lru_idle_unloadable's
        own concern) -- even the worst-ranked resident is excluded if idle."""
        idle_bad = _put(mgr, "idle_bad", state=ResidentState.IDLE_EVICTABLE,
                        meta={"ip": "10.0.0.3"}, last_active=1.0)
        active_good = _put(mgr, "active_good", state=ResidentState.ACTIVE,
                           meta={"ip": "10.0.0.1"}, last_active=100.0)
        assert mgr._is_worst_ranked_loaded_locked(idle_bad) is False
        # the only LOADED resident left is active_good -- it becomes its own
        # pool's worst by default, even though it is the higher-ranked client.
        assert mgr._is_worst_ranked_loaded_locked(active_good) is True

    def test_unresolvable_is_the_machine_worst(self, mgr):
        """Previously ``test_unresolvable_is_never_the_victim`` -- it pinned
        the EARLIER reading (unregistered clients "rank below everyone",
        read as "excluded from the comparison"). The design's own
        worked example (unregistered clients ARE the eviction
        pool) overrides it:
        "rank below all registered ones" places an unresolvable resident at
        the BOTTOM of the pool -- it is the machine's WORST, not outside
        the pool. Now: while an unresolvable resident is loaded, it is
        the machine's worst and the listed resident is not (the listed
        resident's old "trivially its own worst" status is exactly the
        backwards reading this test now rejects)."""
        unresolvable = _put(mgr, "unresolvable", state=ResidentState.ACTIVE,
                            meta={"ip": "10.9.9.9"}, last_active=1.0)
        listed = _put(mgr, "listed", state=ResidentState.ACTIVE,
                      meta={"ip": "10.0.0.1"}, last_active=100.0)
        assert mgr._is_worst_ranked_loaded_locked(unresolvable) is True
        assert mgr._is_worst_ranked_loaded_locked(listed) is False

    def test_no_loaded_residents_means_no_victim(self, mgr):
        only = _put(mgr, "only", state=ResidentState.IDLE_EVICTABLE,
                    meta={"ip": "10.0.0.3"}, last_active=1.0)
        assert mgr._is_worst_ranked_loaded_locked(only) is False

    def test_single_loaded_resident_is_its_own_victim(self, mgr):
        """Worst-among-one is trivially itself -- no other loaded client to
        compare against, matching the pairwise-at-each-transition framing."""
        solo = _put(mgr, "solo", state=ResidentState.ACTIVE,
                    meta={"ip": "10.0.0.1"}, last_active=100.0)
        assert mgr._is_worst_ranked_loaded_locked(solo) is True

    def test_same_rank_tie_breaks_to_least_recently_active(self, mgr):
        """Two residents matched by the SAME rule (same rank) -- the tie must
        break toward the LEAST recently active, same direction eviction's own
        tie-break already uses."""
        older = _put(mgr, "older", state=ResidentState.ACTIVE,
                     meta={"ip": "10.0.0.3"}, last_active=10.0)
        newer = _put(mgr, "newer", state=ResidentState.ACTIVE,
                     meta={"ip": "10.0.0.3"}, last_active=90.0)
        assert mgr._is_worst_ranked_loaded_locked(older) is True
        assert mgr._is_worst_ranked_loaded_locked(newer) is False

    def test_same_rank_tie_break_is_not_an_insertion_order_coincidence(self, mgr):
        """Companion to the test above, with insertion order REVERSED (newer
        put first). max() keeps the first-seen element on an exact tie, so an
        implementation that silently drops the recency term entirely would still
        pass the test above by insertion-order accident (older happens to be
        inserted first there) -- this arm proves the tie-break is doing real
        work, not riding insertion order."""
        newer = _put(mgr, "newer", state=ResidentState.ACTIVE,
                     meta={"ip": "10.0.0.3"}, last_active=90.0)
        older = _put(mgr, "older", state=ResidentState.ACTIVE,
                     meta={"ip": "10.0.0.3"}, last_active=10.0)
        assert mgr._is_worst_ranked_loaded_locked(older) is True
        assert mgr._is_worst_ranked_loaded_locked(newer) is False

    def test_fresh_at_each_call_not_a_stored_name(self, mgr):
        """Freshness: the decision is re-evaluated from live state every call -- no
        cached victim identity. Change the pool between two calls and the
        answer changes with it."""
        bad = _put(mgr, "bad", state=ResidentState.ACTIVE,
                   meta={"ip": "10.0.0.3"}, last_active=100.0)
        assert mgr._is_worst_ranked_loaded_locked(bad) is True
        # bad's turn completes and it leaves the loaded pool
        bad.state = ResidentState.IDLE_EVICTABLE
        assert mgr._is_worst_ranked_loaded_locked(bad) is False


class TestNeverDriftsFromEvictionAcrossRank:
    """Tag semantics: inside one client, tag rank ranks RESIDENTS as well as requests,
    the way Fast Lane ranks clients against each other; it is compared only between
    residents of the same rule_index and never lifts one client above another.
    `_unload_priority_key_for_meta` reads rank -- its tuple is
    (tier, -rule_index, -rank, last_active). If designation's own selection key
    left rank out, or placed it differently, two residents matched to the SAME rule
    but classified to DIFFERENT tags (hence different rank) could make designation
    and eviction name DIFFERENT residents as "worst" -- exactly the drift the
    shared `_resolve_fastlane_match` resolver exists to prevent, one level up: a
    shared RESOLVER does not make two ORDERINGS agree if one of them lacks a term
    the other has.
    """

    def test_designation_agrees_with_eviction_when_same_rule_different_rank(self, mgr):
        """Same client (10.0.0.1, rule_index=0 either way), two DIFFERENT requests
        classified to different tags -- curator (rank 5, worse) vs main (rank 1,
        better). The worse-ranked one is also the MORE recently active one, so a
        key that let recency dominate rank would pick the WRONG resident relative
        to eviction, which reads rank ahead of recency."""
        worse_rank_but_more_recent = _put(
            mgr, "worse_rank_but_more_recent", state=ResidentState.ACTIVE,
            meta={"ip": "10.0.0.1", "is_curator": True}, last_active=90.0,
        )
        better_rank_but_less_recent = _put(
            mgr, "better_rank_but_less_recent", state=ResidentState.ACTIVE,
            meta={"ip": "10.0.0.1", "is_main": True}, last_active=10.0,
        )
        # sanity: really the same rule, really different rank -- otherwise this
        # test would pass for the wrong reason (no divergence to detect at all).
        table = mgr._fastlane_table()
        worse_key = mgr._resident_priority_key(worse_rank_but_more_recent, table)
        better_key = mgr._resident_priority_key(better_rank_but_less_recent, table)
        assert worse_key[0] == better_key[0], "test setup bug: rule_index must match"
        assert worse_key[1] != better_key[1], "test setup bug: rank must differ"

        # eviction reads rank after rule_index -- for a same-rule_index pair the WORSE
        # rank wins min(), even when it is the more recently active one. Feed both
        # residents' idle_client_meta so the comparison isolates the FORMULA
        # (not meta-field resolution, already correct per the rank_client_meta stamping) --
        # same technique the eviction-priority drift test uses to compare the key functions.
        worse_rank_but_more_recent.idle_client_meta = worse_rank_but_more_recent.rank_client_meta
        better_rank_but_less_recent.idle_client_meta = better_rank_but_less_recent.rank_client_meta
        by_eviction_min = min(
            [worse_rank_but_more_recent, better_rank_but_less_recent],
            key=lambda r: mgr._resident_unload_priority_key(r, table),
        )
        assert by_eviction_min.model_tag == "worse_rank_but_more_recent", (
            "eviction must pick the worse-ranked resident on a same-rule_index "
            "pair, even though it is the more recently active one -- if this "
            "fails, the test's own understanding of eviction's key is wrong, "
            "not the predicate"
        )

        # designation must name the SAME resident -- within one rule_index the worse
        # rank is the victim ahead of recency, exactly as in eviction.
        assert mgr._is_worst_ranked_loaded_locked(worse_rank_but_more_recent) is True, (
            "designation picked the wrong resident -- rank was not part of the "
            "sort key ahead of recency, disagreeing with eviction"
        )
        assert mgr._is_worst_ranked_loaded_locked(better_rank_but_less_recent) is False


class TestAgainstDrivenResidents:
    """A REAL driven mid-turn resident, through the actual reservation path
    (the rank_client_meta stamping), not synthetic Resident(...) construction.

    Requires the shared Fast Lane fixture module in tests/
    (see module docstring) -- skipped if it is not present, so this
    file still collects cleanly on a tree where the shared fixture hasn't
    landed yet.
    """

    @pytest.mark.asyncio
    async def test_designated_victim_among_real_driven_residents(self, tmp_path):
        fx = pytest.importorskip(
            "_fastlane_fixture",
            reason="shared fast-lane fixture not present at tests/_fastlane_fixture.py",
        )
        boot, runtime = fx.boot_ranked_runtime(
            tmp_path,
            rules=fx.ranked_rules(("10.0.0.1", 1), ("10.0.0.3", 1)),
            max_parallel_sidecars=2,
        )
        fx.seed_manifest(boot, "good-model", main_gpu=0)
        fx.seed_manifest(boot, "bad-model", main_gpu=1)
        good_gate = asyncio.Event()
        bad_gate = asyncio.Event()
        fake_spawn, fake_health, fake_sigterm, fake_vram, fake_complete = fx.make_fakes({
            "good-model": good_gate, "bad-model": bad_gate,
        })
        m = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        with fx.high_vram():
            m._worker_task = asyncio.create_task(m.worker_loop())
            good_task = await fx.drive_to_active(
                m, "good-model", thread_id="t-good", client_meta={"ip": "10.0.0.1"},
            )
            bad_task = await fx.drive_to_active(
                m, "bad-model", thread_id="t-bad", client_meta={"ip": "10.0.0.3"},
            )
            good_key = fx.assert_resolves(m, "good-model")
            bad_key = fx.assert_resolves(m, "bad-model")
            assert bad_key[0] > good_key[0], "bad-model must resolve to the lower-priority rule"

            good_r = fx.resident_for(m, "good-model")
            bad_r = fx.resident_for(m, "bad-model")
            assert m._is_worst_ranked_loaded_locked(bad_r) is True
            assert m._is_worst_ranked_loaded_locked(good_r) is False

            good_gate.set()
            bad_gate.set()
            await good_task
            await bad_task
            await m.shutdown()
