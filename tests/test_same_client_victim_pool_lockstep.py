"""The designated victim (first entry of the loaded-resident pool) and the
resident the idle-eviction key would unload must be the SAME resident, for
any mix of clients (rule_index), tag ranks and recencies.

Both sides are the REAL functions: designation is
``TurbohaulManager._ranked_loaded_candidates_locked()`` over ACTIVE residents,
eviction is ``TurbohaulManager._resident_unload_priority_key`` over the same
residents mirrored as IDLE_EVICTABLE, picked with ``min()`` exactly as
``_lru_idle_unloadable`` picks.

Rule under test: client order (rule_index) decides first; within ONE client
the worse (numerically larger) tag rank is the victim; rank is never compared
across different rule_index values; unlisted residents are the victims before
any listed one.

Every assertion names WHICH resident was picked.

Free VRAM: pure selection over resident keys; no path here reads free VRAM. Both
bindings of `_read_free_vram_all_mib` are pinned by an autouse fixture that raises
if it is ever called.
"""
import ipaddress

import pytest

from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.fastlane import CompiledRule
from turbohaul.manager import Resident, ResidentState, TurbohaulManager


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


IP0 = "192.0.2.10"      # client 0 (highest priority)
IP1 = "192.0.2.20"      # client 1
IP2 = "192.0.2.30"      # client 2 (lowest priority of the three)
IP_UNLISTED = "198.51.100.99"
RANKS = {"main": 1, "curator": 3, "unclassified": 5}

MAIN = {"is_main": True}            # rank 1
CURATOR = {"is_curator": True}      # rank 3
UNCLASSIFIED = {}                   # rank 5


def _rule(index, raw_address):
    addr = ipaddress.ip_address(raw_address)
    return CompiledRule(
        index=index, raw_address=raw_address, address=addr, container_name=None,
        match_addresses=frozenset({addr}), label=f"client{index}",
        tag_ranks=dict(RANKS),
    )


TABLE = [_rule(0, IP0), _rule(1, IP1), _rule(2, IP2)]


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


def _meta(ip, tag_meta):
    return None if ip is None else {"ip": ip, **tag_meta}


def _both_sides(mgr, cases):
    """cases: (label, ip-or-None, tag_meta, last_active). Returns
    (designation_order, eviction_order, designated, evicted) as labels."""
    for label, ip, tag_meta, last_active in cases:
        mgr._residents[label] = Resident(
            model_tag=label, resident_key=label, state=ResidentState.ACTIVE,
            last_active_monotonic=last_active, main_gpu=0, split_mode="none",
            rank_client_meta=_meta(ip, tag_meta),
        )
    designation_order = [r.model_tag for r in mgr._ranked_loaded_candidates_locked()]
    keys = {
        label: mgr._resident_unload_priority_key(
            Resident(model_tag=label, state=ResidentState.IDLE_EVICTABLE,
                     last_active_monotonic=last_active,
                     idle_client_meta=_meta(ip, tag_meta)),
            TABLE,
        )
        for label, ip, tag_meta, last_active in cases
    }
    eviction_order = sorted(keys, key=keys.get)
    return designation_order, eviction_order, designation_order[0], min(keys, key=keys.get)


# (id, residents, expected victim label)
CASES = [
    ("same_client_curator_newer",
     [("main", IP1, MAIN, 1.0), ("curator", IP1, CURATOR, 2.0)], "curator"),
    ("same_client_curator_older",
     [("main", IP1, MAIN, 2.0), ("curator", IP1, CURATOR, 1.0)], "curator"),
    ("same_client_three_ranks_worst_is_newest",
     [("main", IP1, MAIN, 1.0), ("curator", IP1, CURATOR, 2.0),
      ("uncl", IP1, UNCLASSIFIED, 3.0)], "uncl"),
    ("same_client_three_ranks_worst_is_oldest",
     [("main", IP1, MAIN, 3.0), ("curator", IP1, CURATOR, 2.0),
      ("uncl", IP1, UNCLASSIFIED, 1.0)], "uncl"),
    ("two_clients_ranks_crossed_c1_rank5_c2_rank1",
     [("c1_uncl", IP1, UNCLASSIFIED, 1.0), ("c2_main", IP2, MAIN, 2.0)], "c2_main"),
    ("two_clients_ranks_crossed_recency_flipped",
     [("c1_uncl", IP1, UNCLASSIFIED, 2.0), ("c2_main", IP2, MAIN, 1.0)], "c2_main"),
    ("two_clients_two_residents_each",
     [("c1_main", IP1, MAIN, 4.0), ("c1_curator", IP1, CURATOR, 3.0),
      ("c2_main", IP2, MAIN, 2.0), ("c2_curator", IP2, CURATOR, 1.0)], "c2_curator"),
    ("unlisted_beats_worst_ranked_listed",
     [("c2_uncl", IP2, UNCLASSIFIED, 1.0), ("stranger", IP_UNLISTED, MAIN, 9.0)], "stranger"),
    ("unlisted_mixed_with_same_client_ranks",
     [("c1_main", IP1, MAIN, 5.0), ("c1_curator", IP1, CURATOR, 6.0),
      ("stranger", IP_UNLISTED, MAIN, 7.0), ("nometa", None, {}, 8.0)], "stranger"),
    ("two_unlisted_older_goes_first",
     [("stranger_new", IP_UNLISTED, MAIN, 9.0), ("stranger_old", IP_UNLISTED, MAIN, 2.0),
      ("c0_uncl", IP0, UNCLASSIFIED, 1.0)], "stranger_old"),
]


@pytest.mark.parametrize("case_id, cases, expected", CASES, ids=[c[0] for c in CASES])
def test_designated_victim_is_the_resident_eviction_would_unload(mgr, case_id, cases, expected):
    designation_order, eviction_order, designated, evicted = _both_sides(mgr, cases)
    assert designated == expected, (
        f"{case_id}: designation named {designated!r}, expected {expected!r}; "
        f"pool order {designation_order}"
    )
    assert evicted == expected, (
        f"{case_id}: eviction min() named {evicted!r}, expected {expected!r}; "
        f"eviction order {eviction_order}"
    )
    assert designated == evicted, (
        f"{case_id}: designation named {designated!r} but eviction min() is {evicted!r}"
    )
    # Not only the first entry: the whole order is the same.
    assert designation_order == eviction_order, (
        f"{case_id}: pool order {designation_order} != eviction order {eviction_order}"
    )


class TestCrossClientControl:
    """Rank is never compared across rule_index: a client-2 resident with the
    BEST rank never outranks a client-1 resident."""

    @pytest.mark.parametrize("c1_last, c2_last", [(1.0, 2.0), (2.0, 1.0)])
    def test_best_ranked_client2_resident_is_still_victim_over_worst_ranked_client1(
        self, mgr, c1_last, c2_last,
    ):
        cases = [("c1_rank5", IP1, UNCLASSIFIED, c1_last), ("c2_rank1", IP2, MAIN, c2_last)]
        order, eviction_order, designated, evicted = _both_sides(mgr, cases)
        assert designated == "c2_rank1", (
            f"client-2 resident (rank 1) must be named before client-1 resident "
            f"(rank 5); pool order {order}: a rank was compared across rule_index"
        )
        assert evicted == "c2_rank1", f"eviction named {evicted!r}; order {eviction_order}"
        assert order == ["c2_rank1", "c1_rank5"], f"pool order {order}"

    def test_client1_resident_is_never_first_while_a_client2_resident_is_loaded(self, mgr):
        cases = [
            ("c1_curator", IP1, CURATOR, 1.0), ("c1_uncl", IP1, UNCLASSIFIED, 2.0),
            ("c2_main", IP2, MAIN, 3.0),
        ]
        order, _, designated, _ = _both_sides(mgr, cases)
        assert designated == "c2_main", f"pool order {order}"
        assert order[0] == "c2_main" and set(order[1:]) == {"c1_curator", "c1_uncl"}, (
            f"pool order {order}: client-2 resident must come before every client-1 one"
        )

    def test_instrument_sees_rank_within_one_client(self, mgr):
        """Negative control: swapping WHICH resident carries the worse tag, recency
        and client held fixed, flips the victim. A pool that ignored rank would name
        the same resident both times."""
        first = _both_sides(mgr, [("a", IP1, MAIN, 1.0), ("b", IP1, CURATOR, 2.0)])
        mgr._residents.clear()
        second = _both_sides(mgr, [("a", IP1, CURATOR, 1.0), ("b", IP1, MAIN, 2.0)])
        assert (first[2], second[2]) == ("b", "a"), (
            f"victim should follow the worse tag: got {first[2]!r} then {second[2]!r}"
        )
        assert first[2] == first[3] and second[2] == second[3], (
            f"designation/eviction disagree: {first[2:]} / {second[2:]}"
        )
