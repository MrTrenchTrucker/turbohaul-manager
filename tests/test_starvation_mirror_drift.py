"""The drift-detection test `_starvation_reason`'s own
docstring claims exists -- "if this mirror and the real filter ever
disagree, that test fails immediately rather than this line silently
mis-explaining a miss" -- does not exist anywhere in the suite (a search of
the test tree, including indirect-symbol search and call-site counts, finds
no such test). This
file is that test.

⛔ CONSTRAINT: the comparison CALLS `_lru_idle_unloadable` itself --
no reconstruction, no hand-rolled `min`, no re-derived key.
The eviction-priority drift test's own end-to-end arm
(`test_worst_rule_index_is_the_min_eviction_pick`) hand-rolls `min()` over
`_resident_unload_priority_key` despite its docstring claiming it is
"exercised the way `_lru_idle_unloadable` actually uses it" -- copying that
shape here would compare the mirror against a SECOND MIRROR. This file
does not repeat it.

Two properties, both required:
  1. PER-CLAUSE (parametrized, one resident per case, every clause the
     mirror tracks plus the eligible baseline): the real filter and the
     mirror must agree on whether THIS ONE resident is excluded, and if so
     by which clause.
  2. END-TO-END (drift, minimum two residents, an excluded one declared
     BEFORE the survivor -- not optional; a single-resident fixture cannot
     distinguish "the excluded resident" from "the only resident", which is
     exactly how an earlier card_split test missed a real
     misalignment defect): the real filter's ACTUAL pick must be the same
     resident the mirror's per_resident attribution marks "eligible", not
     merely agree on isolated per-resident totals.

Mutation check: this test goes RED on a mutant of `_lru_idle_unloadable`'s OWN candidate
filter (not the mirror -- mutating the mirror would only prove the mirror
can be broken; mutating the real filter proves this test can see a genuine
drift between the two, which is the property the docstring claims and
nothing currently backs).

NOT VERIFIED: clause COMBINATIONS. Every case here trips exactly one
clause (or none). A resident excluded by two or more clauses simultaneously
is exercised by the existing `per_resident` tests (which build exactly
that shape for the totals) but not cross-checked against
`_lru_idle_unloadable`'s actual verdict here -- `_lru_idle_unloadable` only
ever returns "excluded" as a boolean, so a combination cannot expose
anything a single clause doesn't already, but it is a real, known gap
in this file's coverage rather than an implicit claim of exhaustiveness.

`_lru_idle_unloadable` and `_starvation_reason` themselves are not modified
by this file -- the mutation check above is a one-off experiment, not part
of the suite.
"""
import time

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import (
    Resident,
    ResidentState,
    TurbohaulManager,
    _fastlane_census_key,
    _STALENESS_GRANT_THRESHOLD_S,
)
from turbohaul.slot import Slot

SOLO_TAG = "solo"


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
    return TurbohaulManager(boot, runtime)


def _make(**kwargs):
    kwargs.setdefault("state", ResidentState.IDLE_EVICTABLE)
    kwargs.setdefault("last_active_monotonic", 1.0)
    return Resident(model_tag=SOLO_TAG, resident_key=SOLO_TAG, **kwargs)


def _r_state_not_idle(mgr):
    return _make(state=ResidentState.ACTIVE)


def _r_active_slot(mgr):
    return _make(active_slot=Slot.new(SOLO_TAG))


def _r_inflight(mgr):
    return _make(inflight=[Slot.new(SOLO_TAG)])


def _r_staleness_protected(mgr):
    # The staleness clause is gated on the Fast Lane feature flag, so this clause
    # only exists while the feature is ON -- turning it on HERE keeps this case
    # pointed at the clause it is named for, and deliberately leaves the other
    # six cases (which have nothing to do with Fast Lane) on the shared
    # feature-off fixture, unchanged. No rules are needed: the gate reads
    # `enabled`, not the rule table.
    mgr.runtime.fastlane = FastLaneConfig(enabled=True)
    # Same setup the turn-boundary handoff test already uses -- a hand-built
    # grant would test a reconstruction of the threshold, not the real one.
    ip = "203.0.113.5"
    mgr._fastlane_staleness_grants[_fastlane_census_key(ip)] = (
        time.monotonic() - (_STALENESS_GRANT_THRESHOLD_S + 1.0)
    )
    return _make(idle_client_meta={"ip": ip})


def _r_card_split(mgr):
    return _make(main_gpu=1, split_mode="none")


def _r_eligible(mgr):
    return _make()


# (label, resident_factory, main_gpu, split_mode, expected)
# Note: there is no listed_waiter row and no listed_waiter_tags column:
# that clause does not exist, and the filter is keyword-only and takes
# no such parameter at all.
# Every row below is a clause the filter has.
PER_CLAUSE_CASES = [
    ("state_not_idle", _r_state_not_idle, None, None, "state_not_idle"),
    ("active_slot", _r_active_slot, None, None, "active_slot"),
    ("inflight", _r_inflight, None, None, "inflight"),
    ("staleness_protected", _r_staleness_protected, None, None, "staleness_protected"),
    ("card_split", _r_card_split, 0, "none", "card_split"),
    ("eligible", _r_eligible, None, None, "eligible"),
]


class TestMirrorAgreesWithRealFilterPerClause:
    """Property 1: for every clause the mirror claims to track, one resident
    tripping EXACTLY that clause must be excluded by BOTH the real filter
    (called directly) and the mirror -- and the eligible baseline must be
    admitted by both."""

    @pytest.mark.parametrize(
        "label,make_resident,main_gpu,split_mode,expected",
        PER_CLAUSE_CASES,
    )
    def test_real_filter_and_mirror_agree(
        self, mgr, label, make_resident, main_gpu, split_mode, expected,
    ):
        r = make_resident(mgr)
        mgr._residents[SOLO_TAG] = r

        victim = mgr._lru_idle_unloadable(main_gpu=main_gpu, split_mode=split_mode)
        result = mgr._starvation_reason(main_gpu=main_gpu, split_mode=split_mode)

        if expected == "eligible":
            assert victim is not None and victim.model_tag == SOLO_TAG, label
            assert result["eligible"] == 1, label
        else:
            assert victim is None, label
            assert result["eligible"] == 0, label
        assert result["per_resident"] == f"{SOLO_TAG}:{expected}", (label, result["per_resident"])


class TestMirrorAgreesWithRealFilterEndToEnd:
    """Property 2 (drift, end-to-end): with >=2 residents -- one excluded,
    declared FIRST, one eligible, declared SECOND -- the mirror's
    per-resident attribution must agree with which one the real filter
    ACTUALLY picks, not just with isolated single-resident totals. Same
    shape as the alignment arm in the card_split test: a
    single-resident fixture cannot distinguish "the excluded resident" from
    "the only resident"."""

    def test_real_pick_matches_the_mirrors_eligible_entry(self, mgr):
        excluded = Resident(
            model_tag="busy", resident_key="busy",
            state=ResidentState.ACTIVE, last_active_monotonic=1.0,
        )
        survivor = Resident(
            model_tag="free", resident_key="free",
            state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=2.0,
        )
        mgr._residents["busy"] = excluded
        mgr._residents["free"] = survivor

        victim = mgr._lru_idle_unloadable()
        result = mgr._starvation_reason()
        by_tag = dict(entry.split(":", 1) for entry in result["per_resident"].split(","))

        assert victim is not None and victim.model_tag == "free"
        assert by_tag["free"] == "eligible"
        assert by_tag["busy"] == "state_not_idle"
        assert result["eligible"] == 1
