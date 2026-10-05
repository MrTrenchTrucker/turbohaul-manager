"""Tests for TurbohaulManager (mocked subprocess/GPU; foundations only - worker_loop covered separately)."""
import ast
import asyncio
import inspect
import logging
import textwrap
import time
from pathlib import Path

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
from turbohaul.fastlane import FastLaneMatch, lint_rules
from turbohaul.manager import (
    TurbohaulManager,
    Resident,
    _SINGLETON_RESIDENT_KEY,
)
from turbohaul.queue import GraceTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.state import open_state_db


@pytest.fixture
def boot_and_runtime(tmp_path):
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
            llama_server_binary=tmp_path / "fake_llama_server",  # nonexistent but unused in tests
            default_port_base=59500,  # nothing on this range
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return boot, runtime


def _boot_and_runtime_with_fastlane_rule(tmp_path):
    """Same shape as boot_and_runtime, plus ONE Fast Lane rule (10.0.0.1) so
    a resident with a matching rank_client_meta genuinely resolves via
    _resident_priority_key -- needed by tests that must distinguish "the
    victim predicate is absent" from "the victim predicate is present but
    nothing resolves", which look identical against a bare, rule-less
    manager (see TestGraceActiveExclusions.test_victim_predicate_absent_protects_everyone).
    """
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
        queue=QueueConfig(safety_enabled=False), pull=PullConfig(),
        fastlane=FastLaneConfig(
            enabled=True,
            rules=[FastLaneRule(address="10.0.0.1", tag_ranks=FastLaneTagRanks(main=1))],
        ),
    )
    return boot, runtime


class TestConstructor:
    def test_init_wires_subsystems(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        assert mgr.queue is not None
        assert mgr.grace.grace_seconds == runtime.queue.grace_seconds
        assert mgr.idle.idle_seconds == runtime.queue.idle_hot_load_seconds
        assert mgr._active_slot is None
        assert mgr._active_handle is None

    def test_cold_start_idle_client_meta_initialized(self, boot_and_runtime):
        """Cold-start safety: _idle_client_meta must be initialized to None.

        Caught in early smoke test: first request after fresh manager startup would
        hit AttributeError when reading _idle_client_meta at line 3643, causing the
        request to be dropped. Root cause: __init__ read and wrote the attribute via
        _set_idle_holder() but never initialized it.

        Fix: Initialize to None alongside other _idle_* fields (lines 903-917).
        """
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Critical: attribute exists and is None (cold-start state, no idle holder yet)
        assert hasattr(mgr, "_idle_client_meta"), "Missing initialization in __init__"
        assert mgr._idle_client_meta is None, "Should be None on cold start"


class TestFastLaneCensusNormalization:
    """Address normalization: census/badge/age-out/lint must not compare RAW
    addresses while the matcher normalizes -- else false assigned:false,
    duplicate census rows, premature age-out for IPv4-mapped peers. The
    census dict KEY itself is normalized (not just the comparisons),
    row["address"] stays raw for display and
    row["normalized_address"] carries the normalized form."""

    def _mgr_with_rule(self, boot_and_runtime, raw_rule_address):
        boot, runtime = boot_and_runtime
        runtime = RuntimeConfig(
            queue=runtime.queue, pull=runtime.pull,
            fastlane=FastLaneConfig(
                enabled=True,
                rules=[FastLaneRule(
                    address=raw_rule_address,
                    tag_ranks=FastLaneTagRanks(main=1),
                )],
            ),
        )
        return TurbohaulManager(boot, runtime)

    def test_rule_addresses_helper_returns_normalized_form_not_the_raw_one(
        self, boot_and_runtime
    ):
        """_fastlane_rule_addresses() is the single source of truth for the
        addresses a rule claims. It must return the NORMALIZED CompiledRule
        field: comparing a raw display form against the normalized census
        keys is exactly the assigned:false bug that normalization prevents, and this expression
        would otherwise be copy-pasted at three separate census call sites.
        """
        RAW = "::ffff:192.0.2.3"
        NORMALIZED = "192.0.2.3"
        # Green control: if these two ever stop differing, the assertions
        # below are vacuous and say nothing about normalization at all.
        assert RAW != NORMALIZED

        mgr = self._mgr_with_rule(boot_and_runtime, RAW)
        addrs = mgr._fastlane_rule_addresses()

        assert addrs == {NORMALIZED}
        assert RAW not in addrs

    def test_rule_addresses_helper_skips_a_rule_that_has_no_address(
        self, boot_and_runtime
    ):
        """A container-name rule carries address=None. str(None) would
        put the literal "None" into this set, and every census call site
        would then treat a phantom address as rule-claimed -- age-out would
        protect a row that does not exist and the census would report
        assigned:true against nonsense. This guards that case
        for any rule that carries no address.
        """
        import dataclasses
        from turbohaul.fastlane import CompiledRule, normalize_address

        addr = normalize_address("192.0.2.5")
        real = CompiledRule(
            index=0,
            raw_address="192.0.2.5",
            address=addr,
            container_name=None,
            match_addresses=frozenset({addr}),
            label="Gateway",
            tag_ranks={},
        )
        # the shape a container_name rule compiles to: no address at all.
        # Its identity lives in container_name/match_addresses instead.
        nameless = dataclasses.replace(
            real, index=1, raw_address="", address=None,
            container_name="advisor", match_addresses=frozenset(),
            label="Advisor",
        )

        mgr = TurbohaulManager(*boot_and_runtime)
        mgr._fastlane_table = lambda: [real, nameless]  # type: ignore[method-assign]

        addrs = mgr._fastlane_rule_addresses()

        assert "None" not in addrs
        # green control: the addressed rule must still be present, or this
        # test would also pass on a helper that returned nothing at all.
        assert addrs == {"192.0.2.5"}

    def test_inbox_waiting_count_sees_requests_queue_py_cannot(self, boot_and_runtime):
        """A request handed straight to a live resident's inbox never
        calls queue.enqueue(), so it never enters _staging/_accept_buf. Hence
        staging_queue_depth would otherwise read 0 with two ACTIVE residents.
        """
        from turbohaul.manager import ResidentState

        class _Inbox:
            def __init__(self, n): self._n = n
            def qsize(self): return self._n

        class _R:
            def __init__(self, n, state=ResidentState.ACTIVE):
                self.inbox = _Inbox(n)
                self.state = state

        mgr = TurbohaulManager(*boot_and_runtime)
        mgr._residents = {"a": _R(2), "b": _R(3)}
        assert mgr._inbox_waiting_count() == 5

    def test_inbox_waiting_count_excludes_dead_residents(self, boot_and_runtime):
        """A DEAD resident's inbox is drained and RE-ENQUEUED into the queue,
        where _staging counts it. Counting it here too would double-count the
        same request in the number an operator reads to judge backlog.
        """
        from turbohaul.manager import ResidentState

        class _Inbox:
            def __init__(self, n): self._n = n
            def qsize(self): return self._n

        class _R:
            def __init__(self, n, state):
                self.inbox = _Inbox(n)
                self.state = state

        mgr = TurbohaulManager(*boot_and_runtime)
        mgr._residents = {
            "live": _R(4, ResidentState.ACTIVE),
            "dead": _R(99, ResidentState.DEAD),
        }
        # 4, not 103 -- and the control is that the live one still counts, so
        # this cannot pass by simply returning 0 for everything.
        assert mgr._inbox_waiting_count() == 4

    def test_inbox_waiting_count_zero_control_and_missing_inbox(self, boot_and_runtime):
        """Zero-control: without it, an always-on metric would pass the arms
        above for the wrong reason. Also: a resident mid-teardown may carry no
        inbox at all, and /status is the endpoint an operator reaches for when
        something is ALREADY wrong -- it must not raise.
        """
        class _NoInbox:
            state = None

        mgr = TurbohaulManager(*boot_and_runtime)
        mgr._residents = {}
        assert mgr._inbox_waiting_count() == 0
        mgr._residents = {"gone": _NoInbox()}
        assert mgr._inbox_waiting_count() == 0

    def test_status_snapshot_PUBLISHES_the_inbox_count_it_computed(
        self, boot_and_runtime
    ):
        """The count must actually TRAVEL to its production call site. An arm
        that only asserted the
        key exists and reads 0 with no residents -- which a wiring that always
        published 0, or that computed the count and discarded it, would also
        satisfy. This pins the join: what _inbox_waiting_count() returns is
        what /status reports.
        """
        mgr = TurbohaulManager(*boot_and_runtime)
        mgr._inbox_waiting_count = lambda: 6  # type: ignore[method-assign]

        q = mgr.status_snapshot()["queue"]

        assert q["queue_depth_total"] == 6, q
        # control: the pre-existing field must NOT absorb it -- that is the
        # whole reason this is an additional field (telemetry.py persists
        # staging_queue_depth verbatim into a durable log).
        assert q["staging_queue_depth"] == 0, q

    def test_status_snapshot_publishes_the_waiting_total(self, boot_and_runtime):
        """The wiring itself: status_snapshot builds its queue dict from
        explicitly named keys, so passing inbox_waiting through is not enough
        -- the field has to be published too. This arm pins that.
        """
        mgr = TurbohaulManager(*boot_and_runtime)
        q = mgr.status_snapshot()["queue"]
        assert "queue_depth_total" in q
        assert q["queue_depth_total"] == 0
        # the pre-existing fields must be untouched -- this is an ADDITIONAL
        # field, because telemetry.py persists staging_queue_depth verbatim.
        assert q["staging_queue_depth"] == 0

    def test_status_snapshot_fastlane_claims_snapshot_is_a_live_passthrough_not_a_stub(
        self, boot_and_runtime
    ):
        """Un-stubbed: calls the
        real fastlane_claims_snapshot(). Same wiring
        discipline as test_status_snapshot_PUBLISHES_the_inbox_count_it_
        computed above: a monkeypatched SENTINEL proves this actually
        TRAVELS through a real method call. A genuinely-empty live registry
        and a hardcoded [] are indistinguishable on a fresh manager --
        exactly the "cached snapshot with a friendly name" class of defect
        that recurs -- so an empty-registry check ALONE
        (see the control just below) would still pass against the un-fixed
        hardcoded stub for the wrong reason."""
        mgr = TurbohaulManager(*boot_and_runtime)
        mgr.fastlane_claims_snapshot = lambda: [{"sentinel": "claim-sentinel"}]  # type: ignore[method-assign]

        q = mgr.status_snapshot()["queue"]

        assert q["fastlane_claims_snapshot"] == [{"sentinel": "claim-sentinel"}], q
        # control: staging/acceptance are untouched (sentinel is not enqueued),
        # but queue_depth_total now INCLUDES claims_waiting:
        # the sentinel mock returns 1 claim, so total = 0+0+0+1 = 1.
        assert q["queue_depth_total"] == 1  # claims_waiting is part of the total
        assert q["staging_queue_depth"] == 0
        assert "staging_queue_max" in q

    def test_status_snapshot_fastlane_claims_snapshot_empty_registry_reads_empty_list(
        self, boot_and_runtime
    ):
        """Control: no claims registered -> a real, present, empty list --
        not an absent key. This alone is NOT proof of live wiring (see the
        sentinel test above for that); it only confirms the shape is still
        honest on a genuinely idle manager."""
        mgr = TurbohaulManager(*boot_and_runtime)
        q = mgr.status_snapshot()["queue"]
        assert "fastlane_claims_snapshot" in q, q
        assert q["fastlane_claims_snapshot"] == [], q

    def test_rule_addresses_includes_a_name_rules_RESOLVED_addresses(
        self, boot_and_runtime
    ):
        """_fastlane_rule_addresses() is what the census,
        age-out and /status paths use to answer "is this address claimed by a
        rule". It must not read only `.address`: every container_name rule would then
        contribute NOTHING -- a name-matched client's traffic would show as
        UNASSIGNED in the census and its rows would age out as if no rule
        referenced them. The whole point of name-based rules is that a client is now
        identified by name, so the name's resolved addresses must count.
        """
        import dataclasses
        from turbohaul.fastlane import CompiledRule, normalize_address

        a1 = normalize_address("192.0.2.5")
        a2 = normalize_address("2001:db8::5")
        addr_rule = CompiledRule(
            index=0, raw_address="203.0.113.9",
            address=normalize_address("203.0.113.9"), container_name=None,
            match_addresses=frozenset({normalize_address("203.0.113.9")}),
            label="off-network", tag_ranks={},
        )
        name_rule = dataclasses.replace(
            addr_rule, index=1, raw_address="", address=None,
            container_name="gateway-svc", match_addresses=frozenset({a1, a2}),
            label="Gateway",
        )

        mgr = TurbohaulManager(*boot_and_runtime)
        mgr._fastlane_table = lambda: [addr_rule, name_rule]  # type: ignore[method-assign]

        got = mgr._fastlane_rule_addresses()

        # BOTH families the name resolves to must be claimed...
        assert str(a1) in got, got
        assert str(a2) in got, got
        # ...and the address rule still is, which is the control: a helper
        # that returned only match_addresses would break the old path.
        assert "203.0.113.9" in got
        assert "None" not in got

    # --- The two lines that ARE the integration ---------------------------
    #
    # Each of these must turn a test red when gutted. fastlane_rule_names() is the
    # SOLE path by which a configured container name reaches the resolver;
    # the generation cache key is the ONLY reason a rolled docker lease
    # recompiles the rule table. Deleting either must make a test fail.

    def test_fastlane_rule_names_reports_configured_container_names(
        self, boot_and_runtime
    ):
        from turbohaul.config import FastLaneConfig, FastLaneRule, RuntimeConfig

        boot, runtime = boot_and_runtime
        runtime = RuntimeConfig(
            queue=runtime.queue, pull=runtime.pull,
            fastlane=FastLaneConfig(enabled=True, rules=[
                FastLaneRule(container_name="gateway-svc", label="Gateway"),
                FastLaneRule(address="203.0.113.9", label="off-network"),
                FastLaneRule(container_name="web-ui", label="Web UI"),
            ]),
        )
        mgr = TurbohaulManager(boot, runtime)

        names = mgr.fastlane_rule_names()

        # Mutant: `return set()` here permanently unresolves every name rule.
        assert names == {"gateway-svc", "web-ui"}, names
        # control: an ADDRESS rule contributes no name -- without this, a
        # function returning every string it could find would also pass.
        assert "203.0.113.9" not in names

    def test_fastlane_table_recompiles_when_the_resolver_generation_moves(
        self, boot_and_runtime
    ):
        """A container's addresses change when docker hands it a new
        lease, WITHOUT the config object changing. Keying the table cache on
        id(cfg) alone serves a stale table until the next PUT -- the exact
        drift name-based rules exist to remove. The generation is what makes a rolled
        lease recompile.
        """
        from turbohaul.config import FastLaneConfig, FastLaneRule, RuntimeConfig
        from turbohaul.fastlane import normalize_address

        boot, runtime = boot_and_runtime
        runtime = RuntimeConfig(
            queue=runtime.queue, pull=runtime.pull,
            fastlane=FastLaneConfig(enabled=True, rules=[
                FastLaneRule(container_name="gateway-svc", label="Gateway"),
            ]),
        )
        mgr = TurbohaulManager(boot, runtime)

        class _Resolver:
            def __init__(self):
                self.generation = 1
                self._addr = normalize_address("192.0.2.5")

            def addresses_for(self, name):
                return frozenset({self._addr})

            def roll(self, new_addr):
                # what docker actually does: same container, new lease
                self._addr = normalize_address(new_addr)
                self.generation += 1

        res = _Resolver()
        mgr._fastlane_resolver = res

        first = mgr._fastlane_table()
        assert normalize_address("192.0.2.5") in first[0].match_addresses

        res.roll("192.0.2.99")
        second = mgr._fastlane_table()

        # the CONFIG OBJECT NEVER CHANGED -- only the lease did.
        assert normalize_address("192.0.2.99") in second[0].match_addresses, (
            "the table did not recompile after the container's address rolled"
        )
        assert normalize_address("192.0.2.5") not in second[0].match_addresses

    def test_fastlane_table_does_NOT_recompile_on_a_quiet_tick(
        self, boot_and_runtime
    ):
        """Control for the arm above: generation must move ONLY when a
        resolved set actually changes. Without this, a cache key that
        recompiled on every call would pass that test for the wrong reason --
        and would throw away the cache this method exists to provide.
        """
        from turbohaul.config import FastLaneConfig, FastLaneRule, RuntimeConfig

        boot, runtime = boot_and_runtime
        runtime = RuntimeConfig(
            queue=runtime.queue, pull=runtime.pull,
            fastlane=FastLaneConfig(enabled=True, rules=[
                FastLaneRule(container_name="gateway-svc", label="Gateway"),
            ]),
        )
        mgr = TurbohaulManager(boot, runtime)

        class _Stable:
            generation = 7
            def addresses_for(self, name):
                return frozenset()

        mgr._fastlane_resolver = _Stable()
        a = mgr._fastlane_table()
        b = mgr._fastlane_table()
        assert a is b, "an unchanged generation must serve the CACHED table object"

    def test_census_key_collapses_duplicate_raw_forms(self, boot_and_runtime):
        # Two raw forms of the SAME host must collapse into ONE census
        # entry, not two -- this is the duplicate-row bug that normalization prevents
        # (the 256-cap becomes distinct-host, not distinct-raw-string).
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._fastlane_census_observe(
            client_meta={"ip": "192.0.2.10"}, thread_id="t1", model_tag="m"
        )
        mgr._fastlane_census_observe(
            client_meta={"ip": "::ffff:192.0.2.10"}, thread_id="t2", model_tag="m"
        )
        snapshot = mgr.fastlane_census_snapshot()
        assert len(snapshot["rows"]) == 1
        assert snapshot["rows"][0]["request_count"] == 2

    def test_assigned_true_for_rule_matching_different_raw_form(self, boot_and_runtime):
        # The headline bug: rule stored as "192.0.2.10", peer OBSERVED
        # as "::ffff:192.0.2.10" (the normal Starlette form for an IPv4
        # client on an IPv6 listener) -- same host, textually different,
        # matched and served as priority, yet census must not lie about it.
        mgr = self._mgr_with_rule(boot_and_runtime, "192.0.2.10")
        mgr._fastlane_census_observe(
            client_meta={"ip": "::ffff:192.0.2.10"}, thread_id="t", model_tag="m"
        )
        snapshot = mgr.fastlane_census_snapshot()
        assert len(snapshot["rows"]) == 1
        assert snapshot["rows"][0]["assigned"] is True

    def test_assigned_false_for_genuinely_unruled_address(self, boot_and_runtime):
        # Control: an address with NO matching rule at all must still show
        # assigned:false -- proves the fix isn't a blanket "always true".
        mgr = self._mgr_with_rule(boot_and_runtime, "192.0.2.10")
        mgr._fastlane_census_observe(
            client_meta={"ip": "198.51.100.5"}, thread_id="t", model_tag="m"
        )
        snapshot = mgr.fastlane_census_snapshot()
        assert snapshot["rows"][0]["assigned"] is False

    def test_row_address_stays_raw_and_pinned_to_first_seen(self, boot_and_runtime):
        # Display field must stay RAW (operator sees what actually hit the
        # server) and must NOT flicker to a later request's different raw
        # form of the same host -- would read as a new bug otherwise.
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._fastlane_census_observe(
            client_meta={"ip": "192.0.2.10"}, thread_id="t1", model_tag="m"
        )
        mgr._fastlane_census_observe(
            client_meta={"ip": "::ffff:192.0.2.10"}, thread_id="t2", model_tag="m"
        )
        row = mgr.fastlane_census_snapshot()["rows"][0]
        assert row["address"] == "192.0.2.10"  # first-seen raw form, pinned
        assert row["normalized_address"] == "192.0.2.10"

    def test_status_badge_unassigned_count_correct_for_mapped_peer(self, boot_and_runtime):
        mgr = self._mgr_with_rule(boot_and_runtime, "192.0.2.10")
        mgr._fastlane_census_observe(
            client_meta={"ip": "::ffff:192.0.2.10"}, thread_id="t", model_tag="m"
        )
        badge = mgr._fastlane_status_badge()
        assert badge["census_count"] == 1
        assert badge["unassigned_count"] == 0

    def test_status_badge_unassigned_count_control(self, boot_and_runtime):
        # Control: no rule at all -- the one observed address IS unassigned.
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._fastlane_census_observe(
            client_meta={"ip": "192.0.2.10"}, thread_id="t", model_tag="m"
        )
        badge = mgr._fastlane_status_badge()
        assert badge["unassigned_count"] == 1

    def test_no_premature_age_out_for_rule_referenced_mapped_peer(self, boot_and_runtime):
        # Age-out must recognize the observed row as rule-referenced even
        # though the raw forms differ -- a raw comparison would never see
        # it as "referenced" and would age it out regardless of the
        # operator explicitly caring about it (it's IN a rule).
        mgr = self._mgr_with_rule(boot_and_runtime, "192.0.2.10")
        mgr._fastlane_census_observe(
            client_meta={"ip": "::ffff:192.0.2.10"}, thread_id="t", model_tag="m"
        )
        far_future = time.time() + 10 * 365 * 24 * 3600  # far past any TTL
        mgr._fastlane_census_age_out_locked(far_future)
        assert len(mgr._fastlane_census) == 1  # kept: rule-referenced

    def test_age_out_control_unreferenced_address_still_ages_out(self, boot_and_runtime):
        # Control: an unreferenced address (no rule at all) still ages out
        # normally -- proves the fix isn't "nothing ever ages out".
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._fastlane_census_observe(
            client_meta={"ip": "192.0.2.10"}, thread_id="t", model_tag="m"
        )
        far_future = time.time() + 10 * 365 * 24 * 3600
        mgr._fastlane_census_age_out_locked(far_future)
        assert len(mgr._fastlane_census) == 0

    def test_lint_census_membership_normalized(self, boot_and_runtime):
        # lint_rules' census-membership check must normalize the
        # RULE's own address before checking membership -- the census dict
        # is ALWAYS keyed by normalized address (by manager.py's normalization), so
        # this specifically exercises whether lint_rules normalizes its
        # side too. The rule is stored in a NON-canonical raw form
        # ("::ffff:..."); if lint_rules compared the raw form directly, it
        # would never match the canonical census key regardless of manager
        # normalization -- this is the discriminating case
        # tests that use a canonical rule address
        # would miss (raw == normalized for "192.0.2.10" already).
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._fastlane_census_observe(
            client_meta={"ip": "192.0.2.10"}, thread_id="t", model_tag="m"
        )
        rule = FastLaneRule(address="::ffff:192.0.2.10", tag_ranks=FastLaneTagRanks(main=1))
        warnings = lint_rules([rule], census=mgr._fastlane_census)
        assert not any("never been observed" in w for w in warnings)

    def test_lint_census_membership_still_warns_for_truly_unobserved(self, boot_and_runtime):
        # Control: a rule address that was genuinely never observed (in
        # ANY form) must still warn -- proves the fix doesn't silence the
        # check entirely.
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        rule = FastLaneRule(address="192.0.2.10", tag_ranks=FastLaneTagRanks(main=1))
        warnings = lint_rules([rule], census=mgr._fastlane_census)
        assert any("never been observed" in w for w in warnings)

    def test_boot_reconcile_census_rebuild_collapses_duplicate_raw_forms(
        self, boot_and_runtime
    ):
        # boot_reconcile's census rebuild (reads state.sqlite's slots table)
        # is a SEPARATE code path from live _fastlane_census_observe --
        # normalizing one does not structurally guarantee the other is
        # normalized too. A mutation isolated to JUST this loop
        # (reverting only its norm_ip = _fastlane_census_key(ip) line, with
        # every other normalization site left correct) must turn a test red:
        # otherwise the whole suite would be blind to a
        # regression here. This test drives the REAL mgr.boot_reconcile(),
        # not a hand-rebuilt census, so it exercises the exact SQL-read +
        # rebuild loop the fix touches.
        from turbohaul.state import upsert_slot

        mgr = self._mgr_with_rule(boot_and_runtime, "192.0.2.10")
        conn = open_state_db(mgr.boot.storage.state_db_path)
        upsert_slot(conn, {
            "slot_id": "s1", "model_tag": "m", "state": "COLD",
            "thread_id": "t1", "client_meta": {"ip": "192.0.2.10"},
        })
        upsert_slot(conn, {
            "slot_id": "s2", "model_tag": "m", "state": "COLD",
            "thread_id": "t2", "client_meta": {"ip": "::ffff:192.0.2.10"},
        })
        conn.close()

        mgr.boot_reconcile()

        snapshot = mgr.fastlane_census_snapshot()
        assert len(snapshot["rows"]) == 1  # collapsed, not two duplicate rows
        row = snapshot["rows"][0]
        assert row["request_count"] == 2  # both slots counted into the one row
        assert row["assigned"] is True  # matches the rule despite raw-form mismatch


class TestLikelyVictimModelTag:
    """_likely_unload_target_model_tag() -- the manager-level
    resolver for queue.waiting[]'s likely_victim field."""

    def test_none_when_no_residents(self, boot_and_runtime):
        mgr = TurbohaulManager(*boot_and_runtime)
        assert mgr._likely_unload_target_model_tag() is None

    def test_resolves_the_only_resolvable_resident_as_its_own_trivial_victim(
        self, tmp_path
    ):
        """Same fixture shape TestGraceActiveExclusions.
        test_victim_predicate_absent_protects_everyone uses: a real Fast
        Lane rule + a matching rank_client_meta, so this resident genuinely
        resolves via _resident_priority_key rather than being an
        unresolvable bare Resident that would never enter the comparison
        either way."""
        boot, runtime = _boot_and_runtime_with_fastlane_rule(tmp_path)
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["r1"] = Resident(
            model_tag="model-a", resident_key="model-a",
            rank_client_meta={"ip": "10.0.0.1"},
        )
        assert mgr._likely_unload_target_model_tag() == "model-a"

    def test_unresolvable_sole_active_resident_is_displayed(self, boot_and_runtime):
        """An unresolvable ACTIVE resident IS the likely victim. The selector
        must not exclude unresolvable residents: if it did,
        the display would return None even though the
        eviction mechanism's own key ranks that resident FIRST to be evicted
        (tier 0 in ``_unload_priority_key_for_meta``) -- a display that
        contradicted the mechanism it describes (the display must
        reflect reality). By design (the
        eviction pool includes unregistered clients -- they
        are ranked first), an unresolvable ACTIVE resident IS the
        machine's worst, and the display must show it as the likely victim.
        (The IDLE_EVICTABLE population is still out of this field's scope --
        see test_none_when_no_residents' population filter via
        ``_is_worst_ranked_loaded_locked``'s own state filter.)"""
        from turbohaul.manager import ResidentState
        mgr = TurbohaulManager(*boot_and_runtime)
        mgr._residents["r1"] = Resident(model_tag="model-a",
                                        state=ResidentState.ACTIVE)
        assert mgr._likely_unload_target_model_tag() == "model-a"

    def test_never_reimplements_the_predicate_it_delegates_to(self, boot_and_runtime):
        """Non-vacuity guard: prove this calls _is_worst_ranked_loaded_locked
        rather than an inline copy of its selection logic, by monkeypatching
        the predicate to a trivial stub and observing the result change.

        Note: the one predicate is split
        into selection (_is_worst_ranked_loaded_locked) and
        designation (_is_designated_unload_target_locked, which adds the live-
        claim precondition). This field is DISPLAY-ONLY, so it reads the
        selector deliberately -- see the delegate's
        own docstring. The property under test is delegation, not
        reimplementation."""
        mgr = TurbohaulManager(*boot_and_runtime)
        mgr._residents["r1"] = Resident(model_tag="model-a")
        mgr._residents["r2"] = Resident(model_tag="model-b")
        mgr._is_worst_ranked_loaded_locked = lambda r: r.model_tag == "model-b"
        assert mgr._likely_unload_target_model_tag() == "model-b"


class TestQueueWaitingSurface:
    """/status's queue.waiting[] assembly --
    the manager-layer enrichment (likely_victim + LIVE label resolution)
    queue.py's queue_snapshot() itself cannot perform (no resident registry
    access, no _live_fastlane_label). See queue.py's TestQueueSnapshot for
    the row-shape/redaction tests; these are the manager-side wiring."""

    def test_waiting_key_present_and_empty_on_an_idle_manager(self, boot_and_runtime):
        mgr = TurbohaulManager(*boot_and_runtime)
        q = mgr.status_snapshot()["queue"]
        assert q["waiting"] == []

    async def test_a_staged_slot_appears_on_queue_waiting(self, boot_and_runtime):
        mgr = TurbohaulManager(*boot_and_runtime)
        s = Slot.new("m")
        await mgr.queue.enqueue(s)
        q = mgr.status_snapshot()["queue"]
        assert len(q["waiting"]) == 1
        assert q["waiting"][0]["slot_id"] == s.slot_id
        assert q["waiting"][0]["likely_victim"] is None  # no residents at all

    async def test_likely_victim_is_attached_to_every_row(self, tmp_path):
        boot, runtime = _boot_and_runtime_with_fastlane_rule(tmp_path)
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["r1"] = Resident(
            model_tag="model-a", resident_key="model-a",
            rank_client_meta={"ip": "10.0.0.1"},
        )
        await mgr.queue.enqueue(Slot.new("other-model"))
        await mgr.queue.enqueue(Slot.new("other-model-2"))
        q = mgr.status_snapshot()["queue"]
        assert len(q["waiting"]) == 2
        assert all(row["likely_victim"] == "model-a" for row in q["waiting"])

    async def test_likely_victim_is_recomputed_fresh_not_cached(self, tmp_path):
        """Design constraint: the field is "recomputed read-only under lock,
        NEVER stored". Two consecutive status_snapshot() calls, with the
        resident registry changed in between, must each reflect the state
        AT THAT CALL -- a cached value would show the first call's answer
        forever."""
        boot, runtime = _boot_and_runtime_with_fastlane_rule(tmp_path)
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["r1"] = Resident(
            model_tag="model-a", resident_key="model-a",
            rank_client_meta={"ip": "10.0.0.1"},
        )
        await mgr.queue.enqueue(Slot.new("other-model"))
        first = mgr.status_snapshot()["queue"]["waiting"][0]["likely_victim"]
        assert first == "model-a"

        del mgr._residents["r1"]

        second = mgr.status_snapshot()["queue"]["waiting"][0]["likely_victim"]
        assert second is None

    async def test_fastlane_label_is_resolved_live_not_the_frozen_matchs_stale_one(
        self, tmp_path
    ):
        """The redaction rule: "label = _live_fastlane_label
        (raw_address)" -- explicitly NOT the frozen FastLaneMatch's own
        .label, which was only ever true at admission time. Mutating the
        live rule's label AFTER the slot was admitted must be reflected on
        the next status_snapshot() read."""
        boot, runtime = _boot_and_runtime_with_fastlane_rule(tmp_path)
        mgr = TurbohaulManager(boot, runtime)
        s = Slot.new("m")
        s.fastlane = FastLaneMatch(
            rule_index=0, raw_address="10.0.0.1", label="stale-at-admission",
            effective_tag="main", rank=1,
        )
        await mgr.queue.enqueue(s)

        runtime.fastlane.rules[0].label = "live-after-admission"

        q = mgr.status_snapshot()["queue"]
        assert q["waiting"][0]["fastlane"]["label"] == "live-after-admission"

    async def test_fastlane_none_row_untouched_by_label_enrichment(self, boot_and_runtime):
        """Control: a row with no fastlane match must not crash or acquire
        a fastlane dict from the enrichment step."""
        mgr = TurbohaulManager(*boot_and_runtime)
        await mgr.queue.enqueue(Slot.new("m"))
        q = mgr.status_snapshot()["queue"]
        assert q["waiting"][0]["fastlane"] is None


class TestResidentsSnapshotInboxDepth:
    """Per-resident inbox_depth on /status's
    residents[] rows."""

    def test_inbox_depth_reflects_qsize(self, boot_and_runtime):
        mgr = TurbohaulManager(*boot_and_runtime)
        r = Resident(model_tag="model-a", inbox=asyncio.Queue())
        r.inbox.put_nowait(object())
        r.inbox.put_nowait(object())
        mgr._residents["model-a"] = r
        rows = mgr._residents_snapshot()
        assert len(rows) == 1
        assert rows[0]["inbox_depth"] == 2

    def test_inbox_depth_zero_when_no_inbox(self, boot_and_runtime):
        """Control: a resident with inbox=None (never assigned one) must
        read 0, not raise, matching the same getattr-style degrade the
        rest of this file already uses for an absent inbox."""
        mgr = TurbohaulManager(*boot_and_runtime)
        r = Resident(model_tag="model-a", inbox=None)
        mgr._residents["model-a"] = r
        rows = mgr._residents_snapshot()
        assert rows[0]["inbox_depth"] == 0


class TestQueueSurfaceStillAwaitFree:
    """Design constraint: ``status_snapshot`` and
    ``_likely_unload_target_model_tag`` must have zero suspension points, always.

    AST-based, not substring-based -- modelled on
    ``TestPriorityAdmitFromInboxStillAwaitFree``
    (in the turn-boundary handoff test module), which exists for
    exactly the same reason: a naive text search for "await" false-positives
    on prose in a docstring (this file's own ``_likely_unload_target_model_tag``
    docstring contains the literal word). Walking the parsed AST for real
    ``ast.Await``/``ast.AsyncFor``/``ast.AsyncWith`` nodes checks the actual
    code, not the comments about it.

    Why this specific pair, why now: both are called lock-free from
    ``/status`` (an async route handler on the event loop). Being sync with
    zero suspension points is what currently makes that lock-free read
    ATOMIC -- no other coroutine can run mid-read, so no torn view is
    possible. That invariant was previously true by construction only, with
    nothing in the suite asserting it; a single `await` added to either
    function in the future would silently reintroduce a torn-view race.
    See also ``TestQueueSnapshotStillAwaitFree`` in ``test_queue.py`` for
    the third function this same invariant covers (``queue_snapshot``,
    module boundary -- no resident-registry access from queue.py).
    """

    @staticmethod
    def _assert_no_suspension_points(fn, name):
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        offenders = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith))
        ]
        assert offenders == [], (
            f"{name} must have zero suspension points -- found {len(offenders)}"
        )

    def test_status_snapshot_has_zero_suspension_points(self):
        """Mutant this kills: turning status_snapshot into `async def` and
        adding a real `await asyncio.sleep(0)` anywhere in its body --
        `offenders` would become non-empty and the assertion would name
        status_snapshot."""
        self._assert_no_suspension_points(
            TurbohaulManager.status_snapshot, "status_snapshot",
        )

    def test_likely_victim_model_tag_has_zero_suspension_points(self):
        """Mutant this kills: turning _likely_unload_target_model_tag into
        `async def` and adding a real `await asyncio.sleep(0)` anywhere in
        its body -- `offenders` would become non-empty and the assertion
        would name _likely_unload_target_model_tag."""
        self._assert_no_suspension_points(
            TurbohaulManager._likely_unload_target_model_tag, "_likely_unload_target_model_tag",
        )


class TestFastlanePopKwargLoadedPriorityKeys:
    """Swap-budget exemption: TurbohaulManager._fastlane_pop_kwarg() resolves
    each live resident's fastlane match FRESH on every call (mirrors
    _resident_unload_priority_key's resolve-on-read style, never a cached copy)
    and threads the result into FastLanePopPolicy.loaded_priority_keys -- the set
    a cold-jump candidate's own key is compared against to decide the swap-budget
    exemption."""

    def _mgr_with_rules(self, boot_and_runtime, rules):
        boot, runtime = boot_and_runtime
        runtime = RuntimeConfig(
            queue=runtime.queue, pull=runtime.pull,
            fastlane=FastLaneConfig(enabled=True, rules=rules),
        )
        return TurbohaulManager(boot, runtime)

    # ========================================================================
    # FIXTURE SHAPE -- SELF-CONTAINED, NO ASSERTION INVOLVED.
    # Note: this block is separable from the producer fix in manager.py.
    # Reverting it does not revert that fix; it only re-reds these three tests.
    #
    # WHAT: three fixtures below use `rank_client_meta=` rather than
    # `idle_client_meta=`. One keyword each. Every assertion, every rule table,
    # every expected value is unaffected by that choice.
    #
    # WHY: a fixture must not build an ACTIVE resident (Resident.state defaults to
    # ResidentState.ACTIVE) carrying ONLY idle_client_meta. THAT STATE CANNOT
    # OCCUR IN PRODUCTION:
    #   * rank_client_meta is stamped at reserve and at every
    #     one of the four `r.state = ResidentState.ACTIVE` assignments
    #     -- all UNGATED -- and is never cleared
    #     anywhere in src/. So an ACTIVE resident ALWAYS carries it.
    #   * idle_client_meta is written on a model-keyed resident only at park
    #     and only when `parallel == 1`. (The Phase-0 singleton is also
    #     written, but _model_residents() excludes the singleton.)
    #   * manager.py says so directly: "reserve is this resident's
    #     FIRST turn ... idle_client_meta stays None until then."
    # The combination is exactly what _resident_priority_key's docstring calls
    # "a WRONG identity (not merely a missing one) for a resident now mid-turn
    # for a different client", and a mid-turn identity test
    # (in the eviction-priority drift test module) pinning the
    # OPPOSITE rule over the same field pair.
    #
    # The INTENT of all three tests -- a matched resident contributes its key,
    # an unmatched one contributes nothing, two residents aggregate by rule
    # index -- is what each test pins. Only the state they build
    # to express it has to be reachable.
    #
    # The third fixture (..._whose_meta_matches_no_rule) is NOT cosmetic: left
    # on idle_client_meta it still PASSES after the producer fix, but only
    # because the resolver never consults the field at all, so the rule table
    # is never exercised. A rule that points at the fixture's own IP makes a
    # live detector fire, so it is the right probe:
    #     unfixed producer + such a rule -> 1 failed  (detector alive)
    #     fixed producer + such a rule -> 1 passed  (DETECTOR DEAD)
    # That is a setup-only neutralisation with zero assertion deltas, so it
    # is repaired here
    # rather than left silently green.
    # ========================================================================

    def test_loaded_priority_keys_includes_a_matched_registered_resident(
        self, boot_and_runtime
    ):
        mgr = self._mgr_with_rules(boot_and_runtime, [
            FastLaneRule(address="192.0.2.5", tag_ranks=FastLaneTagRanks(main=1)),
        ])
        mgr._residents["r1"] = Resident(
            model_tag="loaded-model",
            # Was idle_client_meta. An ACTIVE resident's identity is
            # rank_client_meta; see the block comment above.
            rank_client_meta={"ip": "192.0.2.5", "is_main": True},
        )

        policy = mgr._fastlane_pop_kwarg()

        assert policy.loaded_priority_keys == ((0, 1),)

    def test_loaded_priority_keys_excludes_a_resident_with_no_client_meta(
        self, boot_and_runtime
    ):
        # A resident nobody has yet routed a fastlane-matched request to (cold-start
        # idle_client_meta still None) must contribute NOTHING -- not a phantom
        # (0, 0)-shaped sentinel that could make an ordinary candidate look exempt.
        mgr = self._mgr_with_rules(boot_and_runtime, [
            FastLaneRule(address="192.0.2.5", tag_ranks=FastLaneTagRanks(main=1)),
        ])
        mgr._residents["r1"] = Resident(model_tag="loaded-model", idle_client_meta=None)

        policy = mgr._fastlane_pop_kwarg()

        assert policy.loaded_priority_keys == ()

    def test_loaded_priority_keys_excludes_a_resident_whose_meta_matches_no_rule(
        self, boot_and_runtime
    ):
        mgr = self._mgr_with_rules(boot_and_runtime, [
            FastLaneRule(address="192.0.2.5", tag_ranks=FastLaneTagRanks(main=1)),
        ])
        mgr._residents["r1"] = Resident(
            model_tag="loaded-model",
            # Was idle_client_meta. On idle_client_meta this test
            # stays green after the producer fix WITHOUT EVER CONSULTING THE
            # RULE TABLE -- a dead detector. See the block comment above.
            rank_client_meta={"ip": "198.51.100.9", "is_main": True},  # not the rule's address
        )

        policy = mgr._fastlane_pop_kwarg()

        assert policy.loaded_priority_keys == ()

    def test_loaded_priority_keys_aggregates_multiple_loaded_residents_by_rule_index(
        self, boot_and_runtime
    ):
        # Two distinct residents, two distinct rules -- both keys must surface, and
        # each must carry ITS OWN rule's index (not e.g. both collapsing to whichever
        # resident _model_residents() happens to yield last).
        mgr = self._mgr_with_rules(boot_and_runtime, [
            FastLaneRule(address="192.0.2.5", tag_ranks=FastLaneTagRanks(main=1)),
            FastLaneRule(address="192.0.2.9", tag_ranks=FastLaneTagRanks(main=2)),
        ])
        mgr._residents["r1"] = Resident(
            model_tag="model-a",
            # Was idle_client_meta; see the block comment above.
            rank_client_meta={"ip": "192.0.2.5", "is_main": True},
        )
        mgr._residents["r2"] = Resident(
            model_tag="model-b",
            # Was idle_client_meta; see the block comment above.
            rank_client_meta={"ip": "192.0.2.9", "is_main": True},
        )

        policy = mgr._fastlane_pop_kwarg()

        assert set(policy.loaded_priority_keys) == {(0, 1), (1, 2)}

    def test_fastlane_pop_kwarg_is_none_when_feature_disabled(self, boot_and_runtime):
        # Control: with the feature off, queue.py's fastlane branch must see None
        # outright, not a policy carrying a vacuous loaded_priority_keys=() that
        # could be misread as "on, but nothing loaded".
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["r1"] = Resident(
            model_tag="loaded-model",
            idle_client_meta={"ip": "192.0.2.5", "is_main": True},
        )

        assert mgr._fastlane_pop_kwarg() is None


class TestFastlaneStatusBadgeSwapsByReason:
    """swaps_by_reason must surface on /status beside the already-shipped
    refusals_by_reason -- same accessor pair, same badge, same shape."""

    def test_swaps_by_reason_reflects_the_queues_own_counters(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr.queue._fastlane_swap_counts["exempt"] = 2
        mgr.queue._fastlane_swap_counts["warm"] = 5

        badge = mgr._fastlane_status_badge()

        assert badge["swaps_by_reason"] == {"exempt": 2, "warm": 5}
        # green control: refusals_by_reason must still be present alongside it,
        # or this test would also pass on a badge that dropped the older key.
        assert "refusals_by_reason" in badge

    def test_swaps_by_reason_empty_when_no_swaps_have_happened(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)

        badge = mgr._fastlane_status_badge()

        assert badge["swaps_by_reason"] == {}


class TestDispatchRoomHint:
    """`_dispatch_room_hint` is the un-locked, resolve-on-read
    twin of `_route_or_reserve`'s own `at_count_cap` fact
    (`len(self._model_residents()) >= cap`) -- direct unit coverage of the
    boundary, since the end-to-end regression proof
    (the turn-boundary handoff module's
    end-to-end real-park wake test)
    is expensive to run for every off-by-one case."""

    def test_true_when_live_residents_below_cap(self, boot_and_runtime):
        from turbohaul.manager import Resident
        boot, runtime = boot_and_runtime
        runtime.queue.max_parallel_sidecars = 3
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["r1"] = Resident(model_tag="model-a")
        assert mgr._dispatch_room_hint() is True  # 1 live < cap 3

    def test_false_exactly_at_cap_not_only_over_it(self, boot_and_runtime):
        """Boundary case: AT the cap (not merely over it) must already read
        as no-room -- `>=`, not `>`, mirrors _route_or_reserve's own
        at_count_cap comparison exactly."""
        from turbohaul.manager import Resident
        boot, runtime = boot_and_runtime
        runtime.queue.max_parallel_sidecars = 2
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["r1"] = Resident(model_tag="model-a")
        mgr._residents["r2"] = Resident(model_tag="model-b")
        assert mgr._dispatch_room_hint() is False  # 2 live == cap 2

    def test_true_when_no_residents_at_all(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        runtime.queue.max_parallel_sidecars = 1
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._dispatch_room_hint() is True


class TestAllocPort:
    """P1a: resident-registry-aware port allocation (max=1 identical to base)."""

    def test_returns_base_when_no_port_held(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        base = boot.runtime.default_port_base
        # Phase-0 singleton resident exists but its .port is the placeholder
        # (None) -> the window is empty of holds -> base is returned verbatim,
        # identical to the deployed hard-coded default_port_base.
        assert mgr._residents[_SINGLETON_RESIDENT_KEY].port is None
        assert mgr._alloc_port() == base

    def test_skips_a_held_port(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        base = boot.runtime.default_port_base
        # A second (Phase-1-style) resident holding the base port forces the
        # allocator to skip it and return the next free port in the window.
        mgr._residents["held-base"] = Resident(model_tag="held", port=base)
        assert mgr._alloc_port() == base + 1

    def test_skips_lower_held_returns_lowest_free(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        base = boot.runtime.default_port_base
        # Hold base+1 but leave base free -> lowest free is still base.
        mgr._residents["held-base1"] = Resident(model_tag="held", port=base + 1)
        assert mgr._alloc_port() == base
        # Now also hold base -> lowest free climbs to base+2.
        mgr._residents["held-base"] = Resident(model_tag="held2", port=base)
        assert mgr._alloc_port() == base + 2


class TestActiveSpawnSeq:
    """P1a: per-resident spawn_seq mirrors the global at max=1 (identical)."""

    def test_active_spawn_seq_mirrors_global_after_bump(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Fresh manager: both the global and the active resident start at 0.
        assert mgr._spawn_seq == 0
        assert mgr._active_spawn_seq() == 0
        # _bump_spawn_seq advances the global AND mirrors to the active resident
        # in lock-step, so the live-monitor generation_id input is unchanged.
        mgr._bump_spawn_seq()
        assert mgr._spawn_seq == 1
        assert mgr._residents[_SINGLETON_RESIDENT_KEY].spawn_seq == 1
        assert mgr._active_spawn_seq() == 1
        mgr._bump_spawn_seq()
        assert mgr._active_spawn_seq() == mgr._spawn_seq == 2


class TestBootReconcile:
    def test_boot_reconcile_returns_summary(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Boot reconcile uses /proc + nvidia-smi which are present on Linux server
        result = mgr.boot_reconcile()
        assert "orphans_reaped" in result
        assert "foreign_gpu_apps" in result
        assert "slots_reconciled_to_cold" in result

    def test_boot_reconcile_marks_stale_slots_cold(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Pre-populate state.sqlite with a fake-active slot whose pid is dead
        from turbohaul.state import upsert_slot
        conn = open_state_db(boot.storage.state_db_path)
        upsert_slot(
            conn,
            {
                "slot_id": "stale-1",
                "model_tag": "m",
                "state": "ACTIVE",
                "pid": 999_999_999,  # never alive
            },
        )
        conn.close()

        result = mgr.boot_reconcile()
        assert result["slots_reconciled_to_cold"] >= 1

        # Verify slot now COLD
        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute("SELECT state, end_reason FROM slots WHERE slot_id='stale-1'")
        row = cur.fetchone()
        assert row["state"] == "COLD"
        assert "boot-reconcile" in row["end_reason"]
        conn.close()

    def test_boot_reconcile_with_injected_pid_check(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Inject custom pid-alive function that says nothing is alive
        result = mgr.boot_reconcile(pid_is_alive_fn=lambda pid: False)
        assert "slots_reconciled_to_cold" in result


class TestVerifyBinary:
    def test_empty_sha_skips(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # boot.runtime.llama_server_binary_sha256 = "" by default → skip-OK
        assert mgr.verify_binary() is True

    def test_wrong_sha_fails(self, tmp_path):
        bin_path = tmp_path / "fake"
        bin_path.write_bytes(b"x" * 100)
        boot = BootConfig(
            server=ServerConfig(),
            storage=StorageConfig(
                blob_store_path=tmp_path / "b",
                manifests_path=tmp_path / "m",
                import_allowed_root=tmp_path / "i",
                state_db_path=tmp_path / "s.sqlite",
            ),
            runtime=RuntimePathsConfig(
                llama_server_binary=bin_path,
                llama_server_binary_sha256="deadbeef" * 8,  # wrong
                default_port_base=59500,
            ),
            ui=UIConfig(static_path=tmp_path / "ui"),
        )
        runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
        mgr = TurbohaulManager(boot, runtime)
        assert mgr.verify_binary() is False


@pytest.mark.asyncio
class TestSubmit:
    async def test_submit_returns_slot_with_auto_thread_id(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        slot = await mgr.submit(model_tag="qwen", prompt="hello world")
        assert slot.slot_id.startswith("slot-")
        assert slot.thread_id.startswith("auto-")
        assert slot.model_tag == "qwen"
        # Audit-logged
        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute("SELECT state FROM slots WHERE slot_id=?", (slot.slot_id,))
        row = cur.fetchone()
        assert row is not None
        assert row["state"] in ("STAGED", "ACCEPT_BUFFER")
        conn.close()

    async def test_submit_preserves_explicit_thread_id(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        slot = await mgr.submit(model_tag="qwen", prompt="hi", thread_id="custom-thread")
        assert slot.thread_id == "custom-thread"

    async def test_submit_with_grace_match_enqueues_head(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Prime: fill queue
        s_first = await mgr.submit(model_tag="qwen", prompt="prior", thread_id="thr-x")
        s_other = await mgr.submit(model_tag="qwen", prompt="other")
        # Manually start grace timer to simulate active slot popped + in grace
        mgr.grace.start("thr-x", "qwen")
        # Now follow-up should land at head
        s_followup = await mgr.submit(model_tag="qwen", prompt="next-turn", thread_id="thr-x")
        # Pop from queue → should be follow-up first (head)
        popped = await mgr.queue.pop_next()
        assert popped.slot_id == s_followup.slot_id

    async def test_submit_audit_logs(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        slot = await mgr.submit(model_tag="m", prompt="hi")
        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute(
            "SELECT event_type FROM audit_events WHERE slot_id=?", (slot.slot_id,)
        )
        events = [row["event_type"] for row in cur.fetchall()]
        assert "submit" in events
        conn.close()


class TestStatusSnapshot:
    def test_empty_status(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        snap = mgr.status_snapshot()
        assert snap["queue"]["acceptance_buffer_depth"] == 0
        assert snap["queue"]["staging_queue_depth"] == 0
        assert snap["active"] is None
        assert snap["grace"] is None
        assert snap["idle_hot"] is None
        assert snap["parallel_slots"]["used"] == 0
        assert snap["parallel_slots"]["max"] == runtime.queue.max_parallel_sidecars

    def test_grace_state_reflected(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr.grace.start("thr-abc12345", "qwen3.6-35b-moe")
        snap = mgr.status_snapshot()
        assert snap["grace"] is not None
        assert snap["grace"]["model_tag"] == "qwen3.6-35b-moe"
        # Redaction: only first 8 chars exposed
        assert snap["grace"]["thread_id_prefix"] == "thr-abc1"

    def test_idle_state_reflected(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr.idle.start("qwen-coder")
        snap = mgr.status_snapshot()
        assert snap["idle_hot"] is not None
        assert snap["idle_hot"]["model_tag"] == "qwen-coder"


@pytest.mark.asyncio
class TestShutdown:
    async def test_shutdown_closes_queue(self, boot_and_runtime):
        from turbohaul.queue import QueueClosed

        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        await mgr.submit(model_tag="m", prompt="hi")
        await mgr.shutdown()
        # After shutdown queue should refuse new submits
        with pytest.raises(QueueClosed):
            await mgr.submit(model_tag="m", prompt="another")

    async def test_shutdown_cancels_worker_task(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Start worker loop as a task
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.sleep(0.05)
        await mgr.shutdown()
        # worker task should be done
        assert mgr._worker_task.done() or mgr._worker_task.cancelled()


@pytest.mark.asyncio
class TestWorkerLoopSkeleton:
    async def test_worker_loop_consumes_queue(self, boot_and_runtime):
        """The loop dispatches a staged slot without relying on wall-clock load time."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        submitted = await mgr.submit(model_tag="m", prompt="hi")
        consumed = asyncio.Event()

        async def fake_process_slot(slot):
            assert slot.slot_id == submitted.slot_id
            consumed.set()

        # This is a worker-loop unit test, not a real llama-server integration
        # test. Replacing the downstream sidecar work makes the queue-dispatch
        # contract deterministic instead of assuming it finishes within 0.2s.
        mgr._process_slot = fake_process_slot
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.wait_for(consumed.wait(), timeout=1.0)
        await mgr.shutdown()
        assert consumed.is_set()


@pytest.mark.asyncio
class TestDispatchLoopWake:
    """The cap>=2 dispatcher (_dispatch_loop): wakes on the queue's
    on_enqueue hook instead of polling blind, with a bounded fallback so a
    wake that never fires still makes progress at the old poll cadence --
    never a hang. Both arms tested directly against _dispatch_loop,
    bypassing worker_loop's cap-based branch (out of scope here) and
    _route_or_reserve (real routing/GPU placement is not what this test
    exercises).

    Consuming the wake could race a resident's
    GRACE-loop pop_matched_thread poll for a same-thread follow-up, and
    could steal it into a cold re-route instead of warm ACTIVE_MATCH reuse
    (observed as a flake on
    tests/test_multislot_concurrency.py::TestKeepAliveActiveMatchGuard with
    the wake wired without the exclusion). The grace_active exclusion (passed to
    pop_next from this same loop) makes a same-thread follow-up inside an
    active, non-victim grace window invisible to this pop regardless of
    timing, closing that race structurally rather than by luck -- see
    manager.py's _dispatch_loop comment at the wait site."""

    async def test_wakes_promptly_on_enqueue_not_via_the_poll_bound(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        runtime.queue.max_parallel_sidecars = 2
        mgr = TurbohaulManager(boot, runtime)
        routed = asyncio.Event()

        async def fake_route_or_reserve(slot):
            routed.set()

        mgr._route_or_reserve = fake_route_or_reserve
        task = asyncio.create_task(mgr._dispatch_loop())
        await asyncio.sleep(0)  # let the loop start and block on the empty-queue wait

        start = time.monotonic()
        await mgr.submit(model_tag="m", prompt="hi")
        await asyncio.wait_for(routed.wait(), timeout=1.0)
        elapsed = time.monotonic() - start

        mgr._stop_event.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert elapsed < 0.02, (
            f"took {elapsed * 1000:.1f}ms to notice the enqueue -- should wake "
            "near-instantly via the Event, not wait out the poll bound"
        )

    async def test_a_wake_that_never_fires_still_makes_progress_via_the_bounded_fallback(
        self, boot_and_runtime,
    ):
        """Simulate the hook not firing (the dispatch loop is left waiting on
        an Event nothing sets, e.g. a coverage gap in on_enqueue wiring) and
        prove the bounded wait_for still makes progress at roughly the old
        poll cadence, rather than hanging. A revert to an unconditional
        `await self._dispatch_wake.wait()` here WOULD hang until this test's
        own outer timeout kills it -- that is the watched RED this test is
        built to catch."""
        boot, runtime = boot_and_runtime
        runtime.queue.max_parallel_sidecars = 2
        mgr = TurbohaulManager(boot, runtime)
        # Disconnect the wake the loop is actually waiting on from the one
        # the queue's on_enqueue hook sets -- exactly what "the hook does not
        # fire" looks like from _dispatch_loop's point of view.
        mgr._dispatch_wake = asyncio.Event()
        routed = asyncio.Event()

        async def fake_route_or_reserve(slot):
            routed.set()

        mgr._route_or_reserve = fake_route_or_reserve
        task = asyncio.create_task(mgr._dispatch_loop())
        await asyncio.sleep(0)

        start = time.monotonic()
        await mgr.submit(model_tag="m", prompt="hi")
        await asyncio.wait_for(routed.wait(), timeout=1.0)
        elapsed = time.monotonic() - start

        mgr._stop_event.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert elapsed >= 0.04, (
            f"took only {elapsed * 1000:.1f}ms despite a disconnected wake -- "
            "the fallback bound should have been the one making progress here"
        )
        assert elapsed < 0.5, (
            f"took {elapsed * 1000:.1f}ms -- the bounded fallback should cap "
            "this near _DISPATCH_DEFER_BACKOFF_S, not multiply it"
        )

    async def test_wake_event_is_cleared_after_being_consumed(self, boot_and_runtime):
        """A missed .clear() after consuming the wake would leave the Event
        permanently set, so every SUBSEQUENT empty-queue wait would return
        immediately (already set) instead of honoring the bounded fallback --
        a busy-loop the timeout exists specifically to prevent. Neither of
        this class's other two tests observes a SECOND empty-queue cycle, so
        neither would catch a dropped .clear()."""
        boot, runtime = boot_and_runtime
        runtime.queue.max_parallel_sidecars = 2
        mgr = TurbohaulManager(boot, runtime)
        routed = asyncio.Event()

        async def fake_route_or_reserve(slot):
            routed.set()

        mgr._route_or_reserve = fake_route_or_reserve
        task = asyncio.create_task(mgr._dispatch_loop())
        await asyncio.sleep(0)

        await mgr.submit(model_tag="m", prompt="hi")
        await asyncio.wait_for(routed.wait(), timeout=1.0)
        # Give the loop one more full iteration to reach the empty-queue
        # branch and, in fixed code, clear the wake it just consumed.
        await asyncio.sleep(0.01)

        mgr._stop_event.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not mgr._dispatch_wake.is_set(), (
            "wake Event left SET after being consumed -- every subsequent "
            "empty-queue wait would return immediately instead of honoring "
            "the bounded fallback, a busy-loop the timeout exists to prevent"
        )


class TestGraceActiveExclusions:
    """_grace_active_exclusions() resolves the set
    of (thread_id, model_tag) pairs pop_next must protect, fresh from live
    residents on every call, under _registry_lock -- and its victim carve-out
    seam is inert (protects everyone) until _is_designated_unload_target_locked
    (the companion predicate that identifies a designated victim)
    exists on the class."""

    async def test_no_residents_in_grace_returns_empty_set(self, boot_and_runtime):
        mgr = TurbohaulManager(*boot_and_runtime)

        result = await mgr._grace_active_exclusions()

        assert result == frozenset()

    async def test_resident_in_grace_contributes_its_pair(self, boot_and_runtime):
        mgr = TurbohaulManager(*boot_and_runtime)
        g = GraceTimer(grace_seconds=30.0)
        g.start("thread-a", "model-a")
        mgr._residents["r1"] = Resident(model_tag="model-a", grace=g)

        result = await mgr._grace_active_exclusions()

        assert result == frozenset({("thread-a", "model-a")})

    async def test_expired_grace_does_not_contribute(self, boot_and_runtime):
        mgr = TurbohaulManager(*boot_and_runtime)
        g = GraceTimer(grace_seconds=0.0)  # expires the instant it starts
        g.start("thread-a", "model-a")
        assert g.expired()  # green control: this fixture actually IS expired
        mgr._residents["r1"] = Resident(model_tag="model-a", grace=g)

        result = await mgr._grace_active_exclusions()

        assert result == frozenset()

    async def test_resident_with_no_grace_object_is_skipped(self, boot_and_runtime):
        mgr = TurbohaulManager(*boot_and_runtime)
        mgr._residents["r1"] = Resident(model_tag="model-a", grace=None)

        result = await mgr._grace_active_exclusions()

        assert result == frozenset()

    async def test_victim_predicate_absent_protects_everyone(self, tmp_path):
        """_is_designated_unload_target_locked is a permanent
        class method, so "absent" cannot occur by simple omission --
        a `not hasattr(...)` check cannot be used. Absence is
        simulated instead by
        shadowing the class method with an explicit instance-level None:
        `delattr` on an instance attribute that was never set to begin with
        is a no-op against a real class method, so it cannot represent
        "absent" any more than the bare-omission case could -- the
        shared fixture's own victim_predicate_absent() uses the same shadowing
        for the identical reason. None is what "absent" means to this
        consumer's own getattr(self, "_is_designated_unload_target_locked", None)
        probe: an instance attribute shadows the class method in normal
        attribute lookup, so getattr sees None either way.

        NON-VACUITY: a plain fixture (a bare
        boot_and_runtime manager, no Fast Lane config, r1 with no
        rank_client_meta) is unresolvable to _resident_priority_key
        regardless of whether the predicate runs at all -- it never becomes
        a victim CANDIDATE either way, so the shadow line would be
        inert (removing it would leave the test passing). This fixture instead
        configures a real Fast Lane rule and gives r1 a matching
        rank_client_meta, so it resolves and -- being the pool's only
        member -- IS its own trivial worst (same reasoning
        the designated-victim test module's
        test_unresolvable_is_never_the_victim documents for that case). A
        REAL, unshadowed predicate would exclude it; only the shadow proves
        "absent" still protects it.
        """
        boot, runtime = _boot_and_runtime_with_fastlane_rule(tmp_path)
        mgr = TurbohaulManager(boot, runtime)
        assert hasattr(mgr, "_is_designated_unload_target_locked"), (
            "expected a permanent class method on the manager -- "
            "if this ever starts failing, the predicate moved/was renamed and "
            "this simulated-absence fixture is stale, not correct"
        )
        mgr._is_designated_unload_target_locked = None
        g = GraceTimer(grace_seconds=30.0)
        g.start("thread-a", "model-a")
        mgr._residents["r1"] = Resident(
            # state defaults to ResidentState.ACTIVE -- left implicit rather
            # than importing ResidentState into this test for one default value.
            model_tag="model-a", resident_key="model-a",
            rank_client_meta={"ip": "10.0.0.1"}, grace=g,
        )

        result = await mgr._grace_active_exclusions()

        assert result == frozenset({("thread-a", "model-a")})

    async def test_victim_predicate_present_excludes_only_the_victim(self, boot_and_runtime):
        mgr = TurbohaulManager(*boot_and_runtime)
        victim_grace = GraceTimer(grace_seconds=30.0)
        victim_grace.start("victim-thread", "model-a")
        other_grace = GraceTimer(grace_seconds=30.0)
        other_grace.start("other-thread", "model-b")
        victim_r = Resident(model_tag="model-a", grace=victim_grace)
        other_r = Resident(model_tag="model-b", grace=other_grace)
        mgr._residents["victim"] = victim_r
        mgr._residents["other"] = other_r
        # Simulates the companion predicate landing -- named seam, not invented here.
        mgr._is_designated_unload_target_locked = lambda r: r is victim_r

        result = await mgr._grace_active_exclusions()

        assert result == frozenset({("other-thread", "model-b")})

    async def test_victim_predicate_raising_degrades_to_not_protected_and_logs_loudly(
        self, boot_and_runtime, caplog
    ):
        mgr = TurbohaulManager(*boot_and_runtime)
        g = GraceTimer(grace_seconds=30.0)
        g.start("thread-a", "model-a")
        mgr._residents["r1"] = Resident(model_tag="model-a", grace=g)

        def boom(_r):
            raise RuntimeError("simulated companion-predicate failure")

        mgr._is_designated_unload_target_locked = boom

        with caplog.at_level(logging.ERROR):
            result = await mgr._grace_active_exclusions()

        assert result == frozenset(), (
            "must degrade toward NOT protecting this one grace window, never "
            "crash the dispatcher's pop_next call"
        )
        failed = [r for r in caplog.records if "GRACE_VICTIM_CHECK_FAILED" in r.message]
        assert len(failed) == 1, "loud failure needs a greppable marker in the logs"
        assert failed[0].exc_info is not None, (
            "a loud failure needs exc_info, not a swallowed traceback"
        )

    async def test_registry_lock_is_held_while_the_victim_predicate_runs(self, boot_and_runtime):
        mgr = TurbohaulManager(*boot_and_runtime)
        g = GraceTimer(grace_seconds=30.0)
        g.start("thread-a", "model-a")
        mgr._residents["r1"] = Resident(model_tag="model-a", grace=g)
        observed = {}

        def check_lock(_r):
            observed["locked"] = mgr._registry_lock.locked()
            return False

        mgr._is_designated_unload_target_locked = check_lock

        await mgr._grace_active_exclusions()

        assert observed.get("locked") is True, (
            "the victim check must run under _registry_lock"
        )


class TestGraceActiveWiredAtTheRightCallSites:
    """_dispatch_loop and BOTH same-turn rider fan-out
    sites (the ones a probe proved CAN steal a grace-held pair) must pass
    grace_active; worker_loop's cap<=1 pop_next call must NOT -- there is no
    per-resident grace-loop task at cap<=1 for it to ever race, so wiring it
    there would be dead code, not a fix. Structural (AST-based), not a
    string grep, so reformatting the call doesn't create a false failure --
    but a dropped or wrongly-added kwarg at any one of the four IS caught.
    """

    def test_pop_next_call_sites_have_exactly_the_right_grace_active_wiring(self):
        import ast

        import turbohaul.manager as manager_module

        src = open(manager_module.__file__).read()
        tree = ast.parse(src)

        # Map: enclosing function name -> list of bool (has grace_active kwarg),
        # one entry per self.queue.pop_next(...) call found textually inside it.
        by_function: dict[str, list[bool]] = {}

        class _FuncScoper(ast.NodeVisitor):
            def visit_AsyncFunctionDef(self, node):
                self._visit_fn(node)

            def visit_FunctionDef(self, node):
                self._visit_fn(node)

            def _visit_fn(self, node):
                calls = []
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "pop_next"
                        and isinstance(sub.func.value, ast.Attribute)
                        and sub.func.value.attr == "queue"
                    ):
                        calls.append(
                            any(kw.arg == "grace_active" for kw in sub.keywords)
                        )
                if calls:
                    by_function.setdefault(node.name, []).extend(calls)
                self.generic_visit(node)

        _FuncScoper().visit(tree)

        expected_wired = {"_dispatch_loop", "_fan_out_and_drain", "_admit_nonstreaming_riders"}
        expected_unwired = {"worker_loop"}

        for fn in expected_wired:
            assert fn in by_function, f"expected a self.queue.pop_next(...) call inside {fn}"
            assert all(by_function[fn]), (
                f"{fn}'s pop_next call(s) must pass grace_active -- a probe "
                "proved this call shape can steal a grace-held pair"
            )
        for fn in expected_unwired:
            assert fn in by_function, f"expected a self.queue.pop_next(...) call inside {fn}"
            assert not any(by_function[fn]), (
                f"{fn}'s pop_next call is the cap<=1 site -- there is no "
                "per-resident grace-loop task at cap<=1, so wiring grace_active "
                "here would be dead code"
            )


import pytest
from turbohaul.manager import TurbohaulManager
from turbohaul.config import BootConfig, ServerConfig, StorageConfig, RuntimePathsConfig, UIConfig, RuntimeConfig, QueueConfig, PullConfig


@pytest.mark.asyncio
class TestShutdownFailsPending:
    """Shutdown must fail pending completion_futures."""

    async def test_shutdown_fails_staged_slot_completion_futures(self, tmp_path):
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
                default_port_base=59700,
            ),
            ui=UIConfig(static_path=tmp_path / "ui_dist"),
        )
        runtime = RuntimeConfig(
            queue=QueueConfig(safety_enabled=False, grace_seconds=0, idle_hot_load_seconds=0),
            pull=PullConfig(),
        )
        mgr = TurbohaulManager(boot, runtime)
        # Submit a slot via submit_and_wait WITHOUT starting worker_loop.
        # The slot sits in staging with an unresolved completion_future.
        # Wrap in a task so we can await shutdown concurrently.
        caller_task = asyncio.create_task(
            mgr.submit_and_wait("anymodel", "hi")
        )
        # Give the submit_and_wait task time to register the future
        await asyncio.sleep(0.05)
        assert not caller_task.done(), (
            "caller_task should be blocked on completion_future before shutdown"
        )
        # Shutdown should fail the pending future, not let the caller hang.
        await mgr.shutdown()
        # Caller now completes (with exception)
        with pytest.raises((asyncio.CancelledError, RuntimeError)):
            await asyncio.wait_for(caller_task, timeout=2.0)


class TestIdleWindowSeconds:
    """Per-resident idle window: _idle_window_seconds helper."""

    def test_none_keep_alive_uses_default(self, boot_and_runtime):
        """When keep_alive is None, the default (per-model or global) is used."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # With keep_alive=None, default=300 -> returns 300
        assert mgr._idle_window_seconds(None, 300) == 300

    def test_negative_keep_alive_pins(self, boot_and_runtime):
        """keep_alive < 0 -> pin-warm (KEEP_ALIVE_MAX_S)."""
        from turbohaul.config import KEEP_ALIVE_MAX_S
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._idle_window_seconds(-1, 120) == KEEP_ALIVE_MAX_S

    def test_positive_keep_alive_caps_at_max(self, boot_and_runtime):
        """keep_alive > KEEP_ALIVE_MAX_S is capped."""
        from turbohaul.config import KEEP_ALIVE_MAX_S
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._idle_window_seconds(999_999, 120) == KEEP_ALIVE_MAX_S

    def test_zero_keep_alive_disables_idle(self, boot_and_runtime):
        """keep_alive=0 -> idle_window=0 (unload immediately)."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._idle_window_seconds(0, 300) == 0


class TestPerModelIdleTimeout:
    """Per-model sleep_idle_seconds from manifest.

    A 35B model was evicting after 120s because the global
    idle_hot_load_seconds was used for everything. Each model's manifest
    sleep_idle_seconds should govern its own idle timeout.
    """

    def test_manifest_shorter_than_global_evicts_at_manifest(self, boot_and_runtime):
        """Model with sleep_idle_seconds=60 evicts at 60s, not the global 120s."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # Simulate a resident with a short per-model timeout
        r = Resident(model_tag="short-idle", sleep_idle_seconds=60)
        # The driver resolves: 60 > 0 -> per_model_idle = 60
        per_model_idle = (
            mgr.runtime.queue.idle_hot_load_seconds
            if r.sleep_idle_seconds == 0
            else r.sleep_idle_seconds
        )
        idle_window = mgr._idle_window_seconds(None, per_model_idle)
        assert idle_window == 60, "Should use manifest 60s, not global default"

    def test_manifest_longer_than_global_uses_manifest(self, boot_and_runtime):
        """Model with sleep_idle_seconds=600 stays warm 600s, not global 120s."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        r = Resident(model_tag="long-idle", sleep_idle_seconds=600)
        per_model_idle = (
            mgr.runtime.queue.idle_hot_load_seconds
            if r.sleep_idle_seconds == 0
            else r.sleep_idle_seconds
        )
        idle_window = mgr._idle_window_seconds(None, per_model_idle)
        assert idle_window == 600, "Should use manifest 600s, not global 120s"

    def test_manifest_pin_negative_ones_stays_warm(self, boot_and_runtime):
        """Model with sleep_idle_seconds=-1 is pinned (KEEP_ALIVE_MAX_S)."""
        from turbohaul.config import KEEP_ALIVE_MAX_S
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        r = Resident(model_tag="pinned", sleep_idle_seconds=-1)
        # -1 resolves to KEEP_ALIVE_MAX_S
        per_model_idle = (
            KEEP_ALIVE_MAX_S
            if r.sleep_idle_seconds == -1
            else (
                mgr.runtime.queue.idle_hot_load_seconds
                if r.sleep_idle_seconds == 0
                else r.sleep_idle_seconds
            )
        )
        idle_window = mgr._idle_window_seconds(None, per_model_idle)
        assert idle_window == KEEP_ALIVE_MAX_S, "Pinned model should stay warm"

    def test_manifest_zero_falls_back_to_global(self, boot_and_runtime):
        """Model with sleep_idle_seconds=0 (unset) uses global default."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        r = Resident(model_tag="default-idle", sleep_idle_seconds=0)
        # 0 means "use global default"
        per_model_idle = mgr.runtime.queue.idle_hot_load_seconds
        idle_window = mgr._idle_window_seconds(None, per_model_idle)
        assert idle_window == mgr.runtime.queue.idle_hot_load_seconds

    def test_request_keep_alive_overrides_manifest(self, boot_and_runtime):
        """A request's keep_alive_s overrides the manifest default.

        If the manifest says 60s but the client sends keep_alive=300,
        the 300s wins for that request cycle.
        """
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        r = Resident(model_tag="short-idle", sleep_idle_seconds=60)
        per_model_idle = r.sleep_idle_seconds  # 60s from manifest
        # Client sends keep_alive=300 -> should use 300, not manifest 60
        idle_window = mgr._idle_window_seconds(300, per_model_idle)
        assert idle_window == 300, "Request keep_alive should override manifest"

    def test_stale_keep_alive_no_leak_after_manifest_reset(self, boot_and_runtime):
        """B-after-A: B uses the manifest default, not A's stale keep_alive.

        Request A sends keep_alive=300, request B sends no keep_alive.
        B should use the manifest's sleep_idle_seconds, not A's 300.
        """
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        r = Resident(model_tag="short-idle", sleep_idle_seconds=60)
        per_model_idle = r.sleep_idle_seconds  # 60s from manifest
        # Request A: keep_alive=300 -> idle_window = 300
        idle_a = mgr._idle_window_seconds(300, per_model_idle)
        assert idle_a == 300
        # Request B: no keep_alive (None) -> idle_window = manifest 60, NOT 300
        idle_b = mgr._idle_window_seconds(None, per_model_idle)
        assert idle_b == 60, "B should use manifest 60s, not A's stale 300s"
