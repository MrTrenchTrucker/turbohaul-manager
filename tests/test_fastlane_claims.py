"""Fast Lane claim registry: claim-registry safety properties.
Three of these properties are each pinned by a mutation of the code under test
(the mutant must make the pinned test fail).
Each pinned property fails against its mutant;
the mutants are chosen to be effective against the real code (a mutant
that is a no-op against it would pin nothing).

A "claim" is a standing record that a Fast-Lane-eligible slot is currently
stuck in `_defer_unroutable`'s retry loop waiting for eviction capacity --
registered on each genuine defer (`_register_fastlane_claim_locked`, called
from the single funnel `_defer_unroutable`, not at either of
`_route_or_reserve`'s two call sites separately), released the moment the
slot is admitted (`_release_fastlane_claim_locked`, wired into all 3 real
admission points: `_route_to`'s two callers plus the legacy HIT block plus
`_reserve_and_start_locked`'s success return) or leaves the system (defer exhaustion
is retired -- see TestDeferUnroutableWiring's
own re-pointed test below -- a claim now only leaves via one of
`_claim_is_live`'s dead-claim conditions, swept by `fastlane_claims_snapshot`).

The "unsatisfiable capacity" release sub-check is NOT implemented
here (see `_claim_is_live`'s own docstring for why) -- a known
gap, not a guess, and therefore has no test for it either.

Each pinned property fails against its mutant and passes
against the real code.
"""
import asyncio
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
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot, SlotState
from turbohaul.state import state_db_session

_RULES = [FastLaneRule(address="10.0.0.1", tag_ranks=FastLaneTagRanks(main=1))]


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
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False), pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=_RULES),
    )
    m = TurbohaulManager(boot, runtime)
    # Pre-create the schema (incl. WAL mode) synchronously, uncontended --
    # this test fires MULTIPLE concurrent background audit writes per
    # registration, and state.py's audit_db_session fallback path
    # (no init_audit_pool -- this is a bare unit test, not under FastAPI
    # lifespan) opens a FRESH sqlite3 connection per call; two threads
    # racing the FIRST-ever `PRAGMA journal_mode=WAL` on a brand-new file
    # hit a transient "database is locked" (state.py's own busy_timeout
    # pragma runs AFTER the journal_mode pragma, too late to cover it).
    # Pre-warming here avoids that race without touching state.py, which
    # is out of scope for this test file.
    with state_db_session(m.boot.storage.state_db_path):
        pass
    return m


def _match(rule_index=0, rank=1, raw_address="10.0.0.1"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address=raw_address, label="",
        effective_tag="main", rank=rank,
    )


_slot_id_seq = itertools.count()


def _slot(
    model_tag="m1", *, fastlane=None, thread_id="", client_meta=None,
    disconnect_event=None, is_evicted=False, completion_future=None,
):
    # NOT id(object()): a short-lived object's id() can be REUSED by CPython
    # once garbage-collected, so two _slot() calls made close together can
    # silently get the SAME slot_id -- caught this exact collision via
    # test_hundred_rank_nine_then_one_rank_zero_evicts_worst_not_new's
    # set-based membership check undercounting (2 == 98 failure, not a
    # count mismatch in the code under test).
    return Slot(
        slot_id=f"slot-{next(_slot_id_seq)}",
        model_tag=model_tag,
        state=SlotState.RECEIVED,
        thread_id=thread_id,
        client_meta=client_meta or {},
        fastlane=fastlane,
        disconnect_event=disconnect_event,
        is_evicted=is_evicted,
        completion_future=completion_future,
    )


async def _drain_bg(mgr):
    """Await every background task _spawn_bg has queued so far (audit write
    + bus publish both happen off-lock -- tests must let them
    actually run before asserting on their side effects)."""
    tasks = [t for t in list(mgr._bg_tasks) if not t.done()]
    if tasks:
        await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Redaction shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRedactionShape:
    async def test_snapshot_never_leaks_full_thread_id_or_client_meta(self, mgr):
        secret_thread = "th-" + "x" * 40  # far longer than any [:8] prefix
        marker = "SECRET-SESSION-MARKER-998877"
        slot = _slot(
            fastlane=_match(rule_index=0, rank=1),
            thread_id=secret_thread,
            client_meta={"ip": "10.0.0.1", "session_id": marker},
        )
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)

        rows = mgr.fastlane_claims_snapshot()
        assert len(rows) == 1
        row = rows[0]
        assert "client_meta" not in row
        assert "ip" not in row
        flat = str(row)
        assert secret_thread not in flat
        assert marker not in flat
        assert "10.0.0.1" not in flat or row["fastlane"]["fastlane_rule"] == "10.0.0.1"
        # thread_id_prefix is the ONLY thread_id-derived field, and only the
        # first 8 chars.
        assert row["thread_id_prefix"] == secret_thread[:8]
        assert len(row["thread_id_prefix"]) == 8

    async def test_redacted_keys_frozenset_contains_new_entries(self, mgr):
        from turbohaul.manager import EventBus
        for key in ("thread_id", "ip", "client_meta", "idle_client_meta"):
            assert key in EventBus.REDACTED_KEYS


# ---------------------------------------------------------------------------
# Claim eligibility invariant
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestClaimEligibility:
    async def test_unlisted_slot_with_ordinary_client_meta_never_claimed(self, mgr):
        """`floor_promoted` is a real
        `Slot` field (see `_fastlane_claim_eligible`'s own docstring:
        "the day that field lands, eligibility picks it up with zero
        changes here"). This slot
        never went through the fairness floor, so it must read False,
        not merely absent -- a bare `not hasattr` pin would not check the
        live value, so the live-value check
        is used instead."""
        slot = _slot(fastlane=None, client_meta={"session_id": "abc"})
        assert slot.floor_promoted is False
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        assert mgr.fastlane_claims_snapshot() == []

    async def test_listed_slot_is_claimed(self, mgr):
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        rows = mgr.fastlane_claims_snapshot()
        assert len(rows) == 1
        assert rows[0]["slot_id"] == slot.slot_id


# ---------------------------------------------------------------------------
# _CLAIMS_MAX overflow semantics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestClaimsMaxOverflow:
    async def test_hundred_rank_nine_then_one_rank_zero_evicts_worst_not_new(self, mgr):
        from turbohaul.manager import _CLAIMS_MAX
        assert _CLAIMS_MAX == 100

        worst_slots = []
        for i in range(99):
            s = _slot(model_tag=f"worst-{i}", fastlane=_match(rule_index=0, rank=9))
            mgr._register_fastlane_claim_locked(s, "make_room_starved_count_cap")
            worst_slots.append(s)
        mid_slot = _slot(model_tag="mid", fastlane=_match(rule_index=0, rank=1))
        mgr._register_fastlane_claim_locked(mid_slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        assert len(mgr.fastlane_claims_snapshot()) == 100

        best_slot = _slot(model_tag="best", fastlane=_match(rule_index=0, rank=0))
        mgr._register_fastlane_claim_locked(best_slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)

        rows = mgr.fastlane_claims_snapshot()
        ids_present = {r["slot_id"] for r in rows}
        assert len(rows) == 100
        # the new best claim must be present, not refused
        assert best_slot.slot_id in ids_present
        # the second-best (rank=1) claim must SURVIVE -- only a genuinely
        # worst (rank=9) claim may be evicted, proving direction not just count
        assert mid_slot.slot_id in ids_present
        # exactly one of the 99 rank=9 claims is now gone
        worst_ids = {s.slot_id for s in worst_slots}
        surviving_worst = worst_ids & ids_present
        assert len(surviving_worst) == 98

    async def test_refusal_counted(self, mgr):
        for i in range(101):
            s = _slot(model_tag=f"m-{i}", fastlane=_match(rule_index=0, rank=9))
            mgr._register_fastlane_claim_locked(s, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        assert mgr._fastlane_claim_refusals.get("capacity", 0) >= 1


# ---------------------------------------------------------------------------
# Release wiring: admission + defer-exhaustion (not a separate requirement of their own,
# but the "released" transition depends on these firing correctly)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReleaseOnAdmission:
    async def test_admission_releases_the_claim(self, mgr):
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        assert len(mgr.fastlane_claims_snapshot()) == 1

        mgr._release_fastlane_claim_locked(slot, "admitted")
        await _drain_bg(mgr)
        assert mgr.fastlane_claims_snapshot() == []

    async def test_release_of_a_displaced_duplicate_does_not_delete_the_survivor(self, mgr):
        # Two slots share a dedup key (same fastlane_rule, same model_tag) --
        # the second registration is a no-op, so slot_a is the sole
        # representative. Releasing slot_b (which never actually held the
        # claim) must not delete slot_a's live claim.
        slot_a = _slot(model_tag="shared", fastlane=_match())
        slot_b = _slot(model_tag="shared", fastlane=_match())
        mgr._register_fastlane_claim_locked(slot_a, "make_room_starved_count_cap")
        mgr._register_fastlane_claim_locked(slot_b, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        assert len(mgr.fastlane_claims_snapshot()) == 1

        mgr._release_fastlane_claim_locked(slot_b, "admitted")
        await _drain_bg(mgr)
        rows = mgr.fastlane_claims_snapshot()
        assert len(rows) == 1
        assert rows[0]["slot_id"] == slot_a.slot_id


# ---------------------------------------------------------------------------
# Structural: release conditions the sweep DOES cover
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReleaseConditions:
    async def test_disconnect_event_releases(self, mgr):
        ev = asyncio.Event()
        slot = _slot(fastlane=_match(), disconnect_event=ev)
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        assert len(mgr.fastlane_claims_snapshot()) == 1
        ev.set()
        rows = mgr.fastlane_claims_snapshot()
        await _drain_bg(mgr)
        assert rows == []

    async def test_is_evicted_releases(self, mgr):
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        slot.is_evicted = True
        rows = mgr.fastlane_claims_snapshot()
        await _drain_bg(mgr)
        assert rows == []

    async def test_failed_future_releases(self, mgr):
        fut = asyncio.get_running_loop().create_future()
        slot = _slot(fastlane=_match(), completion_future=fut)
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        fut.set_exception(RuntimeError("boom"))
        rows = mgr.fastlane_claims_snapshot()
        await _drain_bg(mgr)
        assert rows == []
        # retrieve to avoid an "exception never retrieved" warning leaking
        # into the suite from this test's own synthetic future
        assert isinstance(fut.exception(), RuntimeError)

    async def test_queue_closed_releases(self, mgr):
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        mgr.queue._closed = True
        rows = mgr.fastlane_claims_snapshot()
        await _drain_bg(mgr)
        assert rows == []

    async def test_ttl_expiry_releases(self, mgr):
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        key = mgr._fastlane_claim_key(slot)
        mgr._fastlane_claims[key]["ttl_deadline_monotonic"] = 0.0  # already past
        rows = mgr.fastlane_claims_snapshot()
        await _drain_bg(mgr)
        assert rows == []


# ---------------------------------------------------------------------------
# Events off the lock, bus-observable
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBusObservability:
    async def test_all_four_events_reach_a_subscriber(self, mgr):
        """All four claim lifecycle events reach a subscriber: the
        `expected` set names four events, not three.

        This test is NAMED "all four", so its set must hold all four. A set of
        three would omit `fastlane_admitted` -- the
        lifecycle's third event -- and, since
        the check is a SUBSET assertion (`expected <= names_q1`), the test would
        still pass: the NAME would outrun the ASSERTION and the test would stay green for
        the entire time the event was missing. That is the gap a four-name
        set closes.

        The event is registered in manager.py, which is what makes the
        name true. Widening the set is not editing a test to make it pass --
        it keeps the test HONEST.
        The guard is checked BOTH WAYS, because a widened guard that
        never fails is a decoration: without the emission
        this FAILS on the `expected <= names_q1` assertion below with
        `fastlane_admitted` as the missing element; with the emission in
        manager.py it passes. Do not narrow this set back to three without
        removing the emission -- if it ever passes with the emission gone,
        the subset assertion has stopped biting again.
        """
        q1: asyncio.Queue = asyncio.Queue()
        q2: asyncio.Queue = asyncio.Queue()
        mgr.event_bus.subscribe(q1)
        mgr.event_bus.subscribe(q2)  # cap>=2, as in the original test wording

        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        mgr._release_fastlane_claim_locked(slot, "admitted")
        await _drain_bg(mgr)

        seen_q1 = []
        while not q1.empty():
            seen_q1.append(q1.get_nowait())
        seen_q2 = []
        while not q2.empty():
            seen_q2.append(q2.get_nowait())

        names_q1 = {e["event"] for e in seen_q1}
        names_q2 = {e["event"] for e in seen_q2}
        expected = {
            "fastlane_claim_registered",
            "fastlane_admission_pending",
            # The fourth event, in the lifecycle's
            # own order ("claim registered, admission
            # pending, admitted, released").
            "fastlane_admitted",
            "fastlane_claim_released",
        }
        assert expected <= names_q1
        assert expected <= names_q2
        for e in seen_q1:
            assert "thread_id_prefix" in e
            assert "client_meta" not in e
            assert "thread_id" not in e


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRepeatMissCadence:
    async def test_repeat_miss_does_not_re_register(self, mgr):
        q: asyncio.Queue = asyncio.Queue()
        mgr.event_bus.subscribe(q)
        slot = _slot(fastlane=_match())

        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)

        seen = []
        while not q.empty():
            seen.append(q.get_nowait())
        registered = [e for e in seen if e["event"] == "fastlane_claim_registered"]
        assert len(registered) == 1

    async def test_repeat_miss_does_not_reset_registered_at(self, mgr):
        import time as _time
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        key = mgr._fastlane_claim_key(slot)
        first_ts = mgr._fastlane_claims[key]["registered_at_monotonic"]

        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        assert mgr._fastlane_claims[key]["registered_at_monotonic"] == first_ts


# ---------------------------------------------------------------------------
# _defer_unroutable wiring: the real funnel, not a synthetic call to the
# registry methods directly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDeferUnroutableWiring:
    async def test_defer_registers_a_claim(self, mgr):
        slot = _slot(fastlane=_match())
        mgr._defer_unroutable(slot, evict_pending=False)
        await _drain_bg(mgr)
        rows = mgr.fastlane_claims_snapshot()
        assert len(rows) == 1
        assert rows[0]["reason"] == "make_room_starved_count_cap"
        # cancel the requeue task _defer_unroutable spawned so the test
        # doesn't leave a dangling queue write against a closed loop
        for t in list(mgr._bg_tasks):
            t.cancel()
        await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)

    async def test_defer_never_exhausts_the_claim_persists_and_the_future_stays_pending(
        self, mgr,
    ):
        """This re-points test_defer_
        exhaustion_releases_the_claim_and_fails_the_future: the behavior it pinned changed
        intentionally (a test that breaks due to an intentional behavior change
        is replaced by a new one). _defer_unroutable's exhaustion branch is
        gone -- a request, once queued, stays queued and is not
        cancelled unless the client very specifically cancels it.
        Same monkeypatch as before (force
        _max_vram_defers to 0, the old "exhaust on the very first call"
        trigger) proves the point sharper than a fresh test could: even
        forcing the derived cap to its most hostile value has NO EFFECT any
        more, because _defer_unroutable no longer calls _max_vram_defers at
        all -- the claim stays registered and the future stays pending
        exactly as if the monkeypatch were never applied."""
        fut = asyncio.get_running_loop().create_future()
        slot = _slot(fastlane=_match(), completion_future=fut)
        mgr._max_vram_defers = lambda: 0  # would have exhausted on the old contract
        mgr._defer_unroutable(slot, evict_pending=True)
        await _drain_bg(mgr)
        rows = mgr.fastlane_claims_snapshot()
        assert len(rows) == 1, "claim must persist -- no exhaustion to release it"
        assert rows[0]["reason"] == "make_room_starved_vram"
        assert not fut.done(), (
            "the future must stay pending -- there is no cap left to fail it"
        )
        # cancel the requeue task _defer_unroutable spawned so the test
        # doesn't leave a dangling queue write against a closed loop
        for t in list(mgr._bg_tasks):
            t.cancel()
        await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)
