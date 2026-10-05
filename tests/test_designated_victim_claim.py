"""Claim-gated designation: `_is_designated_unload_target_locked` never asked WHO was claiming.

The shipped predicate took ONLY the resident and answered "are you the worst-ranked
loaded resident?". It never asked whether anyone was claiming, or whether that
claimant outranked the answer. With a single loaded resident `max()` returns that
resident, so the predicate was unconditionally True -- and a designated victim gets
NO grace and is torn down at its turn boundary.

Symptom: a resident at `rule_index` 0 (e.g. a chat platform, the HIGHEST-priority client
in the rule table) had its grace stripped and was evicted for a `rule_index` 3 claimant.

Two rules are violated, and they fail in opposite directions:
  The ranking rule -- "the lowest-priority client survives only one way" -- was applied to evict the
        HIGHEST-priority client for a lower-priority claimant.
  The grace rule -- "grace is unconditional in this state: every loaded client gets it, registered
        or not" -- every sole resident was a designated victim even with NOTHING
        claiming, so it never got the grace that rule guarantees.

The fix restores the designated-victim rule's own opening precondition, which the implementation dropped:
"A HIGHER-PRIORITY REQUEST IS WAITING IN THE QUEUE and the server is at capacity. From
this moment on, the loaded clients are split into two classes." No live claim => no
split => no designated victim.

Arms 1, 3, 4, 5 and 7 are watched RED against the shipped predicate before the
fix (arm 1 and arm 3 reproduce the defect described above); arms 2 and 6 must stay GREEN
across the change, so the fix is proven not to have simply disabled the mechanism.
"""
import ast
import asyncio
import inspect
import itertools

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
from turbohaul.fastlane import FastLaneMatch
from turbohaul.slot import Slot, SlotState
from turbohaul.state import state_db_session

# List index IS the priority: 10.0.0.1 -> rule_index 0 (highest) ... 10.0.0.5 -> 4.
_RULES = [
    FastLaneRule(address="10.0.0.1", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address="10.0.0.2", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address="10.0.0.3", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address="10.0.0.4", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address="10.0.0.5", tag_ranks=FastLaneTagRanks(main=1)),
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
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False), pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=_RULES),
    )
    m = TurbohaulManager(boot, runtime)
    # Same pre-warm as the sibling claims test: registration fires background
    # audit writes, and a brand-new sqlite file racing its first WAL pragma
    # produces a transient "database is locked" unrelated to anything here.
    with state_db_session(m.boot.storage.state_db_path):
        pass
    return m


_seq = itertools.count()


def _resident(mgr, label, *, rule_ip, last_active=100.0, state=ResidentState.ACTIVE):
    r = Resident(
        model_tag=label, resident_key=label, state=state,
        last_active_monotonic=last_active, rank_client_meta={"ip": rule_ip},
    )
    mgr._residents[label] = r
    return r


def _claim(mgr, *, rule_index, rank=1, model_tag="claimant", **slot_kw):
    """Register a REAL claim through the production path, not a hand-built dict.

    `_register_fastlane_claim_locked` is the only "registered" transition in the
    manager; going through it means this fixture cannot drift from the claim shape
    `_claim_is_live` and the governing-key helper actually read.
    """
    slot = Slot(
        slot_id=f"slot-{next(_seq)}",
        model_tag=model_tag,
        state=SlotState.RECEIVED,
        client_meta={"ip": f"10.0.0.{rule_index + 1}"},
        fastlane=FastLaneMatch(
            rule_index=rule_index, raw_address=f"10.0.0.{rule_index + 1}",
            label="", effective_tag="main", rank=rank,
        ),
        **slot_kw,
    )
    mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
    return slot


async def _drain_bg(mgr):
    """Await the background tasks `_spawn_bg` queued during registration.

    `_register_fastlane_claim_locked` fires the audit write and the bus publish
    off-lock, so these arms must run under a real event loop --
    a sync test raises `RuntimeError: no running event loop` INSIDE the fixture
    and never reaches its own assertion, which is a red that tested nothing.
    """
    tasks = [t for t in list(mgr._bg_tasks) if not t.done()]
    if tasks:
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
class TestDesignationRequiresAnOutrankingClaim:
    """The designated-victim rule's precondition: designation exists only while a higher-priority
    request is waiting. "Higher-priority" is the full (rule_index, rank) key:
    rule_index decides first, so the tag rank is only ever compared between a
    claim and a resident of the same client, and it never lifts one client
    above another."""

    async def test_lower_priority_claimant_does_not_designate_the_highest_client(self, mgr):
        """ARM 1 -- THE REPORTED DEFECT, RED against the shipped predicate.

        One resident, `rule_index` 0 (the highest-priority client on the machine).
        The claimant is `rule_index` 3, which it OUTRANKS. The ranking rule forbids evicting it
        for that claimant; the shipped predicate returned True because it never
        compared them at all.
        """
        owui = _resident(mgr, "owui", rule_ip="10.0.0.1")     # rule_index 0
        _claim(mgr, rule_index=3)                              # strictly WORSE
        await _drain_bg(mgr)
        assert mgr._is_designated_unload_target_locked(owui) is False

    async def test_higher_priority_claimant_still_designates(self, mgr):
        """ARM 2 -- MUST STAY GREEN. The mechanism still works when it should."""
        edge = _resident(mgr, "edge", rule_ip="10.0.0.5")  # rule_index 4
        _claim(mgr, rule_index=0)                              # strictly better
        await _drain_bg(mgr)
        assert mgr._is_designated_unload_target_locked(edge) is True

    async def test_no_claim_at_all_means_no_designated_victim(self, mgr):
        """ARM 3 -- THE GRACE-RULE HALF, RED against the shipped predicate.

        With nothing claiming there is no victim/survivor split, so the grace rule governs: "grace is
        unconditional in this state: every loaded client gets it, registered or
        not." The shipped predicate designated every sole resident regardless,
        which stripped the grace that rule guarantees.
        """
        sole = _resident(mgr, "sole", rule_ip="10.0.0.5")
        assert mgr._fastlane_claims == {}
        assert mgr._is_designated_unload_target_locked(sole) is False

    async def test_equal_priority_claimant_does_not_designate(self, mgr):
        """ARM 4 -- RED. An equal (rule_index, rank) key is a real tie and is deliberately NOT
        strictly higher (queue._fastlane_strictly_higher's own contract): an
        equal-ranked claimant must not bump an equal-ranked resident."""
        peer = _resident(mgr, "peer", rule_ip="10.0.0.3")     # rule_index 2
        # The resident carries no tag labels, so it resolves to the rule's unranked
        # rank 6 (the rules here list only main=1); the claim is given that same rank
        # so the two keys are identical, and a tag rank difference is not what is tested.
        _claim(mgr, rule_index=2, rank=6)                     # identical
        await _drain_bg(mgr)
        assert mgr._is_designated_unload_target_locked(peer) is False

    async def test_a_dead_claim_does_not_govern(self, mgr):
        """ARM 5 -- RED. A disconnected claimant would outrank the resident, but a
        dead claim must never drive an eviction (the rule that "the
        decision must reflect reality, not a stale snapshot")."""
        edge = _resident(mgr, "edge", rule_ip="10.0.0.5")  # rule_index 4
        ev = asyncio.Event()
        ev.set()
        slot = _claim(mgr, rule_index=0, disconnect_event=ev)  # would outrank
        await _drain_bg(mgr)
        assert mgr._claim_is_live(mgr._fastlane_claims[mgr._fastlane_claim_key(slot)]) is not None
        assert mgr._is_designated_unload_target_locked(edge) is False

    async def test_selection_still_binds_a_non_worst_resident_is_never_the_victim(self, mgr):
        """ARM 6 -- MUST STAY GREEN. The claim gate is ADDED to selection, it does
        not replace it: an outranking claim does not make every resident a victim."""
        good = _resident(mgr, "good", rule_ip="10.0.0.2")      # rule_index 1
        worst = _resident(mgr, "worst", rule_ip="10.0.0.5")    # rule_index 4
        _claim(mgr, rule_index=0)
        await _drain_bg(mgr)
        assert mgr._is_designated_unload_target_locked(worst) is True
        assert mgr._is_designated_unload_target_locked(good) is False


class TestTheCallSitesUseTheGatedPredicate:
    """The guard belongs in ONE place. These arms stop a later author from
    re-introducing the defect by pointing a behavioural call site at the raw
    selector, which would look harmless and would restore the exact bug."""

    def test_behavioural_call_sites_call_the_gated_predicate(self):
        """ARM 7 -- RED before the fix (the gated name and the raw selector are
        the same function today, so the two cannot yet be told apart).

        `_serve_on_resident` strips a resident's grace, and `_drive_resident`
        sends a victim's backlog to the TAIL instead of the head. Both are designated-victim
        consequences and both must ask the CLAIM-GATED question.

        DO NOT DELETE THIS AS REDUNDANT WITH THE DISPLAY-PATH ARM BELOW. Its
        SECOND assertion -- that neither site calls `_is_worst_ranked_loaded_
        locked` -- is the ONLY guard against the defect this slice fixes, and
        it is the guard that fails BY WORKING: pointing a behavioural site at
        the raw selector reintroduces the ranking/grace-rule inversion silently and greenly,
        where the display-path mistake fails loudly. Proven falsifiable by two
        separate mutants, because pytest stops at the first failing assert and
        a single mutant leaves the second assertion unexercised.
        """
        import turbohaul.manager as m
        tree = ast.parse(inspect.getsource(m))
        by_name = {
            n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "_is_worst_ranked_loaded_locked" in by_name, (
            "the raw selector must exist as its own named function -- if it does "
            "not, selection and designation are still the same function and the "
            "claim precondition cannot be enforced in one place"
        )
        for fn in ("_serve_on_resident", "_drive_resident"):
            called = {
                node.func.attr for node in ast.walk(by_name[fn])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            assert "_is_designated_unload_target_locked" in called, (
                f"{fn} must consult the claim-gated predicate"
            )
            assert "_is_worst_ranked_loaded_locked" not in called, (
                f"{fn} calls the RAW selector, which skips the designated-victim rule's precondition "
                f"-- this is exactly the defect that evicted rule_index 0 for a "
                f"rule_index 3 claimant"
            )

    def test_the_gated_predicate_keeps_its_contract_name(self):
        """The queue side probes for `_is_designated_unload_target_locked` by name. The
        split must not rename the contract."""
        assert hasattr(TurbohaulManager, "_is_designated_unload_target_locked")

    def test_the_new_helpers_introduce_no_await_under_the_registry_lock(self):
        """Both helpers run under `_registry_lock`; an await inside either would
        add a suspension point to a critical section that has none."""
        import turbohaul.manager as m
        tree = ast.parse(inspect.getsource(m))
        for name in ("_governing_claim_priority_key_locked", "_is_worst_ranked_loaded_locked"):
            fn = next(
                (n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name),
                None,
            )
            assert fn is not None, f"{name} missing"
            assert not isinstance(fn, ast.AsyncFunctionDef), f"{name} must be sync"
            assert not [x for x in ast.walk(fn) if isinstance(x, ast.Await)], (
                f"{name} contains an await"
            )

    def test_the_display_path_reads_the_selector_not_the_gated_predicate(self):
        """ARM 8 -- the drift guard pointing the OTHER way.

        `_likely_unload_target_model_tag` is DISPLAY-ONLY (by design:
        "the eviction decision is always the fresh under-lock evaluation, never
        the displayed name"). It must read the raw selector: the designated-victim rule's
        precondition governs grace and teardown, and a display causes neither,
        so gating it would blank the dashboard for no behavioural gain and
        remove the answer to "why did my model get unloaded?".

        SCOPE -- THIS ARM ASSERTS THE DISPLAY DIRECTION ONLY. It inspects
        `_likely_unload_target_model_tag` and nothing else, so a mutant that re-points
        a BEHAVIOURAL site at the selector does NOT make this test fail
        (measured: it goes 1 failed / 9 passed, and the one failure is ARM 7).

        THE OPPOSITE DIRECTION -- a behavioural site reading the raw selector,
        which restores the defect that evicted rule_index 0 for a rule_index 3
        claimant -- IS ASSERTED IN `test_behavioural_call_sites_call_the_gated_
        predicate` (ARM 7), by its second assertion. THAT IS THE ONLY GUARD
        AGAINST THE DEFECT THIS SLICE FIXES; it is not redundant with this one
        and must not be deleted as such.

        This scope note exists because the first version of this docstring
        claimed BOTH directions were asserted here. That is the same failure
        `_starvation_reason`'s docstring made -- a safety claim whose guard
        lives in a different place -- which is the whole reason the mirror
        drift test was required. A maintainer who believed the original wording
        could have deleted ARM 7 as duplicated coverage.
        """
        import turbohaul.manager as m
        tree = ast.parse(inspect.getsource(m))
        by_name = {
            n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        called = {
            node.func.attr for node in ast.walk(by_name["_likely_unload_target_model_tag"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_is_worst_ranked_loaded_locked" in called, (
            "the display must read the selector; gating it on a live claim blanks "
            "the dashboard whenever nothing happens to be claiming"
        )
        assert "_is_designated_unload_target_locked" not in called, (
            "the display is gated on the designated-victim rule's precondition, which governs grace "
            "and teardown -- neither of which a display causes"
        )
