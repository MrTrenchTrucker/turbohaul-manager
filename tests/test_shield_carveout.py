"""THE LISTED-WAITER SHIELD DEFEATS THE
DESIGNATED-VICTIM MECHANISM.

Measured: a listed client's own queued work (``listed_waiter_model_tags``)
shields its resident from ``_lru_idle_unloadable`` UNCONDITIONALLY -- no rank
term anywhere. Since the designated-victim mechanism only ACCELERATES a
victim to IDLE_EVICTABLE for the EXISTING evictor to take, a victim shielded
by its own backlog is never actually evicted -- inverted from the feature's
purpose (the busier the low-priority client, the MORE permanent its
protection).

Design rule: the fix narrows the shield for a
tag iff
  (b) a resident holding that tag is IDLE_EVICTABLE right now, AND
  (c) the routed claimant's own Fast Lane priority STRICTLY outranks the
      BEST-RANKED QUEUED WAITER actually protecting that tag.
Every other listed tag keeps its shield in full.

Design note: (c) must not compare the claimant against
the resident's own stale ``idle_client_meta``
holder -- the WRONG party, since a resident's model_tag can be held by more
than one listed client over its lifetime, and the shield exists to protect
the tag's current QUEUED WAITER, not whoever last parked it. So it
compares against ``queue.listed_waiter_priority_keys()`` instead --
``_shield_carveout_tags`` does not read ``mgr._residents`` or any
``idle_client_meta`` at all; it is a pure function of
(claimant, listed_waiter_tags, listed_waiter_priority_keys). Condition (b)
is therefore NOT checked inside this function -- it has no residents to
check -- and is proven only at the end-to-end level (Group 2 below), via
``_lru_idle_unloadable``'s own pre-existing IDLE_EVICTABLE candidate filter,
unchanged.

STRICT MEMBERSHIP, never ``.get()``-then-compare:
a tag ABSENT from
``listed_waiter_priority_keys`` keeps its shield -- Group 1's
``test_tag_absent_from_priority_keys_keeps_shield`` exercises this
membership branch directly, distinct from the tie/worse-priority branches
(which are membership HITS that simply don't compare favorably).

No existing body is edited: ``_lru_idle_unloadable``, ``listed_waiter_model_tags``,
``_starvation_reason``, and ``_is_designated_unload_target_locked`` are all
unchanged by this fix. The fix lives entirely in the new
call-site-local helper, ``_shield_carveout_tags``, wired into
``_route_or_reserve``'s two real make-room call sites only.
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


def _match(rule_index=0, rank=1, tag="main"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag=tag, rank=rank,
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


# rule_index 0 = highest priority, rule_index 5 = low priority. tag_ranks
# pins the resolved rank to 1 for an unclassified client_meta (the shape
# {"ip": ...} resolves to, via resolve_rank/_class_from_label) -- otherwise
# an unranked resident resolves to fastlane.UNRANKED (6), which would make
# _match(rule_index=N, rank=1) NOT a genuine tie against it.
TABLE = [
    _rule(0, "10.0.0.1", tag_ranks={"unclassified": 1}),  # HIGH priority -- "the outranking claimant"
    _rule(5, "10.0.0.9", tag_ranks={"unclassified": 1}),  # LOW priority -- "the shielded resident"
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
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    m = TurbohaulManager(boot, runtime)
    m._fastlane_table = lambda: TABLE  # bypass the real resolver -- synthetic table, same style as the other fastlane tests
    return m


def _low_prio_idle_resident(tag="m1", resident_key=None, last_active=1.0, holder_ip="10.0.0.9"):
    """A parked resident. ``holder_ip`` (idle_client_meta) is IRRELEVANT to
    the carve-out decision -- kept as a field only so Group 2's
    end-to-end tests can prove the logic does not read it (see
    test_POSITIVE_shielded_resident_outranked_is_evicted, where this holder is
    HIGH-priority but the tag is still correctly carved because the QUEUED
    WAITER, not this holder, is what the comparison uses)."""
    return Resident(
        model_tag=tag, resident_key=resident_key or tag,
        state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=last_active,
        idle_client_meta={"ip": holder_ip},
    )


def _listed_waiter_slot(tag, rule_index, rank=1):
    s = Slot.new(tag)
    s.fastlane = _match(rule_index=rule_index, rank=rank)
    return s


# ---------------------------------------------------------------------------
# Group 1: _shield_carveout_tags in isolation -- (c), STRICT MEMBERSHIP, and
# NARROWNESS. (b) is not checked here at all -- see the module docstring.
# ---------------------------------------------------------------------------

class TestShieldCarveoutUnit:
    def test_empty_tags_passthrough(self, mgr):
        slot = Slot.new("m2")
        slot.fastlane = _match(rule_index=0)
        assert mgr._shield_carveout_tags(slot, set(), {}) == set()

    def test_unlisted_claimant_never_carves_anything(self, mgr):
        """claimant_key is None for an unlisted slot -- the raw shield set is
        returned UNCHANGED, even with a priority_keys entry that would
        otherwise carve it."""
        slot = Slot.new("m2")  # no .fastlane
        carved = mgr._shield_carveout_tags(slot, {"m1"}, {"m1": (9, 1)})
        assert carved == {"m1"}

    def test_tag_absent_from_priority_keys_keeps_shield(self, mgr):
        """Absent-tag arm: a shielded tag ABSENT from
        listed_waiter_priority_keys keeps its shield -- STRICT MEMBERSHIP
        (`if tag not in ...: continue`), not a `.get()`-then-None-check. The
        claimant here would strictly outrank essentially anything (rule_index
        0), so the ONLY thing keeping "m1" shielded is the membership branch
        itself, not a weak priority comparison."""
        slot = Slot.new("m2")
        slot.fastlane = _match(rule_index=0)
        carved = mgr._shield_carveout_tags(slot, {"m1"}, {})
        assert carved == {"m1"}, "absent tag must keep its shield via membership, not .get()"

    def test_condition_c_tie_priority_keeps_shield(self, mgr):
        """Claimant has the SAME priority as the best queued waiter -- not
        STRICTLY higher (an exact tie is deliberately not 'strictly higher',
        _fastlane_strictly_higher's own documented contract) -- tag stays."""
        slot = Slot.new("m2")
        slot.fastlane = _match(rule_index=5)  # same rule_index as the waiter
        carved = mgr._shield_carveout_tags(slot, {"m1"}, {"m1": (5, 1)})
        assert carved == {"m1"}

    def test_condition_c_worse_priority_keeps_shield(self, mgr):
        slot = Slot.new("m2")
        slot.fastlane = _match(rule_index=9)  # worse than the waiter's rule_index=5
        carved = mgr._shield_carveout_tags(slot, {"m1"}, {"m1": (5, 1)})
        assert carved == {"m1"}

    def test_condition_c_strictly_outranking_drops_the_tag(self, mgr):
        slot = Slot.new("m2")
        slot.fastlane = _match(rule_index=0)  # strictly outranks rule_index=5
        carved = mgr._shield_carveout_tags(slot, {"m1"}, {"m1": (5, 1)})
        assert carved == set(), "strictly-outranking claimant must lift the shield"

    def test_narrowness_outranked_tied_and_absent_each_resolve_independently(self, mgr):
        """Narrowness: the carve-out is NARROW, and the THREE ways
        a tag can stay shielded (tie, worse, absent) are all exercised
        together against the ONE way it can drop (strictly outranked)."""
        slot = Slot.new("m2")
        slot.fastlane = _match(rule_index=0)
        carved = mgr._shield_carveout_tags(
            slot,
            {"outranked", "tied", "absent_from_keys"},
            {"outranked": (5, 1), "tied": (0, 1)},  # "absent_from_keys" has no entry
        )
        assert carved == {"tied", "absent_from_keys"}, (
            "only the strictly-outranked tag drops; the tie and the absent tag both keep it"
        )


# ---------------------------------------------------------------------------
# Group 2: end-to-end through the REAL _route_or_reserve wiring.
# Both arms mandatory: a carve-out proven only on its positive arm
# is indistinguishable from deleting the shield outright.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRouteOrReserveEndToEnd:
    async def test_POSITIVE_shielded_resident_outranked_is_evicted(self, mgr):
        """A listed resident WITH its own queued work, facing a
        strictly-higher-ranked live claim, is EVICTED -- and the resident's
        own (irrelevant, HIGH-priority) idle_client_meta holder proves the
        comparison is against the QUEUED WAITER, not that stale
        field."""
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)
        mgr._vram_admits_locked = lambda *a, **k: True
        mgr._residents["m1"] = _low_prio_idle_resident("m1", holder_ip="10.0.0.1")  # HIGH-rank stale holder
        mgr.queue._staging.append(_listed_waiter_slot("m1", rule_index=5))  # m1's own queued listed work

        incoming = Slot.new("m2")
        incoming.fastlane = _match(rule_index=0)  # strictly outranks the WAITER's rule_index=5
        incoming.completion_future = asyncio.get_event_loop().create_future()
        await mgr._route_or_reserve(incoming)

        assert "m1" not in mgr._residents, (
            "a shielded resident facing a strictly-outranking live claim must be evicted, "
            "even though its own idle_client_meta holder (rule_index=0) would wrongly "
            "block eviction under a holder-based comparison"
        )
        mgr._reserve_and_start_locked.assert_called_once()

    async def test_NEGATIVE_CONTROL_condition_b_active_resident_never_evicted(self, mgr):
        """Condition (b) end-to-end: _shield_carveout_tags does not check
        resident state at all (it is a pure function over tag sets), so (b)
        is proven ONLY here -- an ACTIVE resident holding a heavily-carved
        tag must still never be evicted, because _lru_idle_unloadable's own
        pre-existing IDLE_EVICTABLE candidate filter excludes it regardless
        of what the shield-carveout tag set says."""
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)
        mgr._vram_admits_locked = lambda *a, **k: True
        mgr._residents["m1"] = Resident(
            model_tag="m1", resident_key="m1", state=ResidentState.ACTIVE,
            last_active_monotonic=1.0, idle_client_meta={"ip": "10.0.0.9"},
        )
        mgr.queue._staging.append(_listed_waiter_slot("m1", rule_index=5))

        incoming = Slot.new("m2")
        incoming.fastlane = _match(rule_index=0)  # vastly outranks the waiter -- tag WILL be carved
        incoming.completion_future = asyncio.get_event_loop().create_future()
        await mgr._route_or_reserve(incoming)

        assert "m1" in mgr._residents, "(b) IDLE_EVICTABLE is required -- ACTIVE must never be evicted"
        assert mgr._residents["m1"].state is ResidentState.ACTIVE
        mgr._reserve_and_start_locked.assert_not_called()


# Note: Group 3 (the three call-site
# drift-pins) and the two NEGATIVE_CONTROL wiring tests are not part of this
# file: they would pin the carve-out's wiring into _route_or_reserve, which
# is no longer wired in. The 7 TestShieldCarveoutUnit tests
# pin the retained (marked-dead) _shield_carveout_tags
# function itself, kept as marked-dead code.
