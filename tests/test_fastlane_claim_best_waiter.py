"""One Fast Lane claim per (client, model), always keyed by the BEST request of that
(client, model) that still waits.

Rule under test: a strictly better later request upgrades the claim in place (slot and
key); when the request the claim points at starts its turn, disconnects or fails, the
claim moves to the next-best waiter of that (client, model) with that waiter's key, and is
released only when no waiter is left. The claim never points at a request that is gone and
never carries a key that nobody waiting has. The number of claims for N requests of one
(client, model) stays 1, so the claim-table cap sees the same counts as before.

The claim registry is driven through its real entry points (registration, release,
liveness check, parking in an inbox) with real manager state and real slot objects.

Free VRAM: nothing here reads free VRAM. Both the `turbohaul.safety` and the
`turbohaul.manager` binding of `_read_free_vram_all_mib` are pinned by an autouse
fixture that raises if it is ever called.
"""
import asyncio
import copy
import ipaddress
import time

import pytest

from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.fastlane import CompiledRule, FastLaneMatch
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.slot import Slot, SlotState


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


IP_1 = "192.0.2.10"
IP_2 = "198.51.100.20"
MODEL = "model-x"
KEY = (IP_1, MODEL)
MAIN, CURATOR, UNCLASSIFIED = 1, 3, 5   # tag ranks of the test rules


def _rule(index, raw_address):
    addr = ipaddress.ip_address(raw_address)
    return CompiledRule(
        index=index, raw_address=raw_address, address=addr, container_name=None,
        match_addresses=frozenset({addr}), label=f"client{index}",
        tag_ranks={"main": MAIN, "curator": CURATOR, "unclassified": UNCLASSIFIED},
    )


TABLE = [_rule(0, IP_1), _rule(1, IP_2)]


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


_n = [0]


def _slot(label, rank, *, rule_index=0, ip=IP_1, model=MODEL):
    """A queued request of client `rule_index` with a tag of rank `rank`."""
    _n[0] += 1
    return Slot(
        slot_id=f"{label}-{_n[0]}", model_tag=model, state=SlotState.RECEIVED,
        client_meta={"ip": ip}, created_at=float(_n[0]),
        disconnect_event=asyncio.Event(),
        fastlane=FastLaneMatch(
            rule_index=rule_index, raw_address=ip, label="", effective_tag="main", rank=rank,
        ),
    )


def _register(mgr, slot, reason="staging_arrival"):
    mgr._register_fastlane_claim_locked(slot, reason)


def _claim(mgr, key=KEY):
    assert key in mgr._fastlane_claims, f"there is no claim for {key}: {list(mgr._fastlane_claims)}"
    return mgr._fastlane_claims[key]


async def _settle(mgr):
    """Let the background audit/publish tasks registration queued finish."""
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
class TestClaimIsKeyedByTheBestWaiter:
    async def test_a_strictly_better_later_request_upgrades_the_claim_in_place(self, mgr):
        worse, better = _slot("worse", UNCLASSIFIED), _slot("better", MAIN)
        _register(mgr, worse)
        _register(mgr, better)
        assert _claim(mgr)["slot"] is better, "the claim still points at the worse request"
        assert mgr._governing_claim_priority_key_locked() == (0, MAIN)
        assert len(mgr._fastlane_claims) == 1
        await _settle(mgr)

    async def test_control_a_worse_later_request_leaves_the_better_claim_alone(self, mgr):
        better, worse = _slot("better", MAIN), _slot("worse", UNCLASSIFIED)
        _register(mgr, better)
        _register(mgr, worse)
        assert _claim(mgr)["slot"] is better
        assert mgr._governing_claim_priority_key_locked() == (0, MAIN)
        assert len(mgr._fastlane_claims) == 1
        await _settle(mgr)

    async def test_control_equal_tags_keep_the_first_request_as_the_claim(self, mgr):
        first, second = _slot("first", CURATOR), _slot("second", CURATOR)
        _register(mgr, first)
        _register(mgr, second)
        assert _claim(mgr)["slot"] is first, "an equal key must not take the claim over"
        assert mgr._governing_claim_priority_key_locked() == (0, CURATOR)
        assert len(mgr._fastlane_claims) == 1
        await _settle(mgr)

    async def test_the_claim_count_stays_one_for_many_requests_of_one_client_and_model(self, mgr):
        for i, rank in enumerate((UNCLASSIFIED, CURATOR, MAIN, CURATOR, UNCLASSIFIED, MAIN)):
            _register(mgr, _slot(f"r{i}", rank))
            assert len(mgr._fastlane_claims) == 1, f"after {i + 1} requests"
        assert mgr._governing_claim_priority_key_locked() == (0, MAIN)
        await _settle(mgr)

    async def test_control_other_models_and_other_clients_keep_their_own_claims(self, mgr):
        _register(mgr, _slot("a", MAIN))
        _register(mgr, _slot("b", CURATOR, model="model-y"))
        _register(mgr, _slot("c", CURATOR, rule_index=1, ip=IP_2))
        assert len(mgr._fastlane_claims) == 3
        await _settle(mgr)


@pytest.mark.asyncio
class TestClaimMovesToTheNextBestWaiter:
    async def test_when_the_better_request_disconnects_the_claim_falls_back_to_the_worse_one(
            self, mgr):
        worse, better = _slot("worse", UNCLASSIFIED), _slot("better", MAIN)
        _register(mgr, worse)
        _register(mgr, better)
        assert _claim(mgr)["slot"] is better, "setup: the better request must hold the claim first"
        better.disconnect_event.set()
        assert mgr._governing_claim_priority_key_locked() == (0, UNCLASSIFIED), (
            "the claim still governs with a key nobody waiting has")
        assert _claim(mgr)["slot"] is worse, "the claim still points at the request that is gone"
        assert len(mgr._fastlane_claims) == 1
        await _settle(mgr)

    async def test_when_the_better_request_fails_the_claim_falls_back_to_the_worse_one(self, mgr):
        worse, better = _slot("worse", CURATOR), _slot("better", MAIN)
        better.completion_future = asyncio.get_running_loop().create_future()
        _register(mgr, worse)
        _register(mgr, better)
        assert _claim(mgr)["slot"] is better, "setup: the better request must hold the claim first"
        better.completion_future.set_exception(RuntimeError("engine failed"))
        assert mgr._governing_claim_priority_key_locked() == (0, CURATOR)
        assert _claim(mgr)["slot"] is worse
        await _settle(mgr)

    async def test_when_the_better_request_starts_its_turn_the_worse_one_still_holds_a_claim(
            self, mgr):
        worse, better = _slot("worse", UNCLASSIFIED), _slot("better", MAIN)
        _register(mgr, worse)
        _register(mgr, better)
        assert _claim(mgr)["slot"] is better, "setup: the better request must hold the claim first"
        mgr._release_fastlane_claim_locked(better, "admitted")
        assert KEY in mgr._fastlane_claims, "the claim was deleted while a request still waits"
        assert _claim(mgr)["slot"] is worse
        assert mgr._governing_claim_priority_key_locked() == (0, UNCLASSIFIED)
        mgr._release_fastlane_claim_locked(worse, "admitted")
        assert mgr._fastlane_claims == {}, "the claim outlived its last waiter"
        assert mgr._governing_claim_priority_key_locked() is None
        await _settle(mgr)

    async def test_control_equal_tags_the_next_one_takes_over_when_the_first_starts(self, mgr):
        first, second = _slot("first", CURATOR), _slot("second", CURATOR)
        _register(mgr, first)
        _register(mgr, second)
        mgr._release_fastlane_claim_locked(first, "admitted")
        assert _claim(mgr)["slot"] is second
        await _settle(mgr)

    async def test_the_claim_moves_to_the_best_of_several_waiters(self, mgr):
        a, b, c = _slot("a", MAIN), _slot("b", UNCLASSIFIED), _slot("c", CURATOR)
        _register(mgr, b)
        _register(mgr, c)
        _register(mgr, a)
        assert _claim(mgr)["slot"] is a
        mgr._release_fastlane_claim_locked(a, "admitted")
        assert _claim(mgr)["slot"] is c, "not the best of the waiters that are left"
        c.disconnect_event.set()
        assert mgr._governing_claim_priority_key_locked() == (0, UNCLASSIFIED)
        assert _claim(mgr)["slot"] is b
        await _settle(mgr)

    async def test_a_waiter_that_is_gone_is_never_chosen(self, mgr):
        a, b = _slot("a", MAIN), _slot("b", CURATOR)
        _register(mgr, b)
        _register(mgr, a)
        b.disconnect_event.set()
        mgr._release_fastlane_claim_locked(a, "admitted")
        assert mgr._fastlane_claims == {}, "the claim points at a request that is gone"
        await _settle(mgr)

    async def test_when_no_waiter_is_left_the_claim_reads_dead_not_live(self, mgr):
        only = _slot("only", MAIN)
        _register(mgr, only)
        only.disconnect_event.set()
        assert mgr._claim_is_live(_claim(mgr)) == "disconnected"
        assert mgr._governing_claim_priority_key_locked() is None
        await _settle(mgr)


@pytest.mark.asyncio
class TestClaimsParkedInAnInbox:
    async def test_a_request_parked_in_an_inbox_keeps_its_own_claim_when_a_better_one_waits(
            self, mgr):
        parked, better = _slot("parked", CURATOR), _slot("better", MAIN)
        resident = Resident(model_tag="m-res", resident_key="res-1", state=ResidentState.ACTIVE)
        _register(mgr, parked)
        mgr._park_fastlane_claim_locked(parked, resident)
        assert _claim(mgr)["slot"] is parked and _claim(mgr).get("parked_on") == "res-1"
        _register(mgr, better)
        assert _claim(mgr)["slot"] is better and not _claim(mgr).get("parked_on"), (
            "the better waiting request did not take the shared claim")
        own = (f"parked:{parked.slot_id}", MODEL)
        assert mgr._fastlane_claims[own]["slot"] is parked
        assert mgr._fastlane_claims[own].get("parked_on") == "res-1"
        assert mgr._governing_claim_priority_key_locked() == (0, MAIN), (
            "a parked claim must not govern; the waiting better request must")
        await _settle(mgr)

    async def test_control_a_request_that_parks_leaves_the_waiting_claim_alone(self, mgr):
        waiting, other = _slot("waiting", MAIN), _slot("other", CURATOR)
        resident = Resident(model_tag="m-res", resident_key="res-1", state=ResidentState.ACTIVE)
        _register(mgr, waiting)
        _register(mgr, other)
        mgr._park_fastlane_claim_locked(other, resident)
        assert _claim(mgr)["slot"] is waiting
        mgr._release_fastlane_claim_locked(waiting, "admitted")
        assert KEY not in mgr._fastlane_claims, (
            "the parked request was left under the shared claim as a waiter")
        await _settle(mgr)


@pytest.mark.asyncio
class TestAHolderThatIsGoneButNotYetRepointed:
    """The request the claim points at disconnects and nobody reads the claim before the next
    request of the same (client, model) registers. The claim must still end up on the BEST
    request that waits, not on the newcomer."""

    async def test_a_worse_newcomer_does_not_take_the_claim_from_a_better_waiter(self, mgr):
        holder, better, newcomer = _slot("holder", MAIN), _slot("better", CURATOR), _slot("new", UNCLASSIFIED)
        _register(mgr, holder)
        _register(mgr, better)
        assert _claim(mgr)["slot"] is holder, "setup: the best request must hold the claim first"
        holder.disconnect_event.set()          # nobody reads the claim yet
        _register(mgr, newcomer, "make_room_starved_vram")
        assert _claim(mgr)["slot"] is better, "the claim sits on the newcomer while a better request waits"
        assert mgr._governing_claim_priority_key_locked() == (0, CURATOR)
        assert len(mgr._fastlane_claims) == 1
        await _settle(mgr)

    async def test_control_a_newcomer_that_is_the_best_takes_the_claim(self, mgr):
        holder, other, newcomer = _slot("holder", CURATOR), _slot("other", UNCLASSIFIED), _slot("new", MAIN)
        _register(mgr, holder)
        _register(mgr, other)
        assert _claim(mgr)["slot"] is holder, "setup: the claim starts on the first request"
        holder.disconnect_event.set()
        _register(mgr, newcomer, "make_room_starved_vram")
        assert _claim(mgr)["slot"] is newcomer
        assert mgr._governing_claim_priority_key_locked() == (0, MAIN)
        await _settle(mgr)

    async def test_control_a_worse_newcomer_and_no_other_waiter_holds_the_claim(self, mgr):
        holder, newcomer = _slot("holder", MAIN), _slot("new", UNCLASSIFIED)
        _register(mgr, holder)
        holder.disconnect_event.set()
        _register(mgr, newcomer, "make_room_starved_vram")
        assert _claim(mgr)["slot"] is newcomer, "the only request left must hold the claim"
        assert mgr._governing_claim_priority_key_locked() == (0, UNCLASSIFIED)
        await _settle(mgr)


def _resident_with_inbox(mgr, anchor_meta):
    """A busy resident (model m-res) whose latest turn is ranked from `anchor_meta`."""
    r = Resident(
        model_tag="m-res", resident_key="res-1", state=ResidentState.ACTIVE,
        rank_client_meta=anchor_meta,
    )
    r.inbox = asyncio.Queue()
    mgr._residents["res-1"] = r
    return r


@pytest.mark.asyncio
class TestAParkedClaimIsCheckedAgainAfterItsLivenessCheck:
    """Checking a claim for liveness can move it to another waiter. A claim counts as parked
    in an inbox only if the request it points at AFTER that check is in the inbox."""

    def _parked_shared_claim(self, mgr):
        """X (curator) holds the shared claim and sits in the inbox of a resident that serves
        an unclassified turn; Y (rank 4: worse than X, better than the resident) is a waiter
        under the same shared claim and is NOT in the inbox."""
        r = _resident_with_inbox(mgr, {"ip": IP_1})
        x, y = _slot("x", CURATOR), _slot("y", 4)
        _register(mgr, x)
        _register(mgr, y)
        r.inbox.put_nowait(x)
        mgr._park_fastlane_claim_locked(x, r)
        claim = _claim(mgr)
        assert claim["slot"] is x and claim.get("parked_on") == "res-1", "setup: X must hold the parked claim"
        assert y.slot_id in claim.get("waiters", {}), "setup: Y must wait under the shared claim"
        assert not mgr._inbox_holds_slot(r, y)
        return r, x, y

    async def test_a_parked_claim_that_is_dead_does_not_outrank_through_a_waiter_outside_the_inbox(
            self, mgr):
        r, x, y = self._parked_shared_claim(mgr)
        assert mgr._parked_claim_outranks_holder_locked(r) is True, "setup: a live parked claim outranks"
        x.disconnect_event.set()               # X leaves while it sits in the inbox
        assert mgr._parked_claim_outranks_holder_locked(r) is False, (
            "the check used the key of a waiter that is not parked in this inbox")
        await _settle(mgr)

    async def test_control_a_live_parked_claim_still_outranks_the_holder(self, mgr):
        r, x, y = self._parked_shared_claim(mgr)
        assert mgr._parked_claim_outranks_holder_locked(r) is True
        assert mgr._parked_claim_outranks_holder_locked(r, only_slot=x) is True
        await _settle(mgr)

    async def test_the_parked_count_does_not_count_a_claim_that_moved_out_of_the_inbox(self, mgr):
        r, x, y = self._parked_shared_claim(mgr)
        assert mgr._inbox_waiting_count() == 0, "setup: X's live parked claim is counted with the claims"
        x.disconnect_event.set()
        assert mgr._inbox_waiting_count() == 1, (
            "X is still physically in the inbox and its claim is no longer parked there")
        await _settle(mgr)

    async def test_control_a_live_parked_claim_is_not_counted_twice(self, mgr):
        r, x, y = self._parked_shared_claim(mgr)
        assert mgr._inbox_waiting_count() == 0
        r.inbox.put_nowait(_slot("plain", UNCLASSIFIED))
        assert mgr._inbox_waiting_count() == 1
        await _settle(mgr)


@pytest.mark.asyncio
class TestAnEqualRequestKeepsTheClaimWhereItIs:
    """The claim moves, or splits, only for a STRICTLY better request. A request with the same
    (rule_index, rank) key as the one the claim points at, or a worse one, is only recorded as a
    waiter: it does not take the claim, does not split it and does not change its priority.
    (The case where the claim's request is waiting for room is
    TestClaimIsKeyedByTheBestWaiter::test_control_equal_tags_keep_the_first_request_as_the_claim,
    which also asserts the claim count and the governing key.)"""

    def _parked_claim(self, mgr):
        r = _resident_with_inbox(mgr, {"ip": IP_1})
        holder = _slot("holder", CURATOR)
        _register(mgr, holder)
        r.inbox.put_nowait(holder)
        mgr._park_fastlane_claim_locked(holder, r)
        claim = _claim(mgr)
        assert claim["slot"] is holder and claim.get("parked_on") == "res-1", "setup: the claim is parked"
        return r, holder

    @pytest.mark.parametrize("rank", [CURATOR, UNCLASSIFIED], ids=["equal", "worse"])
    async def test_a_request_that_is_not_better_leaves_a_parked_claim_alone(self, mgr, rank):
        r, holder = self._parked_claim(mgr)
        keys_before = list(mgr._fastlane_claims)
        governing_before = mgr._governing_claim_priority_key_locked()
        newcomer = _slot("new", rank)
        _register(mgr, newcomer)
        assert list(mgr._fastlane_claims) == keys_before, "the parked claim was split"
        assert _claim(mgr)["slot"] is holder, "the claim moved to the newcomer"
        assert _claim(mgr).get("parked_on") == "res-1"
        assert mgr._governing_claim_priority_key_locked() == governing_before
        assert newcomer.slot_id in _claim(mgr).get("waiters", {}), "the newcomer is not recorded as a waiter"
        assert len(mgr._fastlane_claims) == 1
        await _settle(mgr)

    async def test_the_claim_passes_to_the_equal_request_when_the_parked_one_starts(self, mgr):
        r, holder = self._parked_claim(mgr)
        newcomer = _slot("new", CURATOR)
        _register(mgr, newcomer)
        mgr._release_fastlane_claim_locked(holder, "admitted")
        assert _claim(mgr)["slot"] is newcomer
        assert not _claim(mgr).get("parked_on"), "a waiting request holds a parked mark"
        assert mgr._governing_claim_priority_key_locked() == (0, CURATOR)
        await _settle(mgr)


@pytest.mark.asyncio
class TestTheClaimThatMovesToAWaiterStopsBeingParked:
    async def test_when_the_parked_holder_starts_the_claim_passes_to_a_waiter_as_a_waiting_claim(
            self, mgr):
        r = _resident_with_inbox(mgr, {"ip": IP_1})
        x, y = _slot("x", MAIN), _slot("y", CURATOR)
        _register(mgr, x)
        _register(mgr, y)
        r.inbox.put_nowait(x)
        mgr._park_fastlane_claim_locked(x, r)
        assert _claim(mgr)["slot"] is x and _claim(mgr).get("parked_on") == "res-1", "setup: X's claim is parked"
        assert y.slot_id in _claim(mgr).get("waiters", {}), "setup: Y waits under the shared claim"
        mgr._release_fastlane_claim_locked(x, "admitted")
        claim = _claim(mgr)
        assert claim["slot"] is y
        assert "parked_on" not in claim, "the claim of a waiting request still says it is parked"
        assert claim["ttl_deadline_monotonic"] > time.monotonic() + 1.0, "the time limit was not renewed"
        assert mgr._governing_claim_priority_key_locked() == (0, CURATOR), (
            "a waiting request's claim must govern; a claim marked as parked does not")
        await _settle(mgr)


@pytest.mark.asyncio
class TestReleasingANonHolderIsSilent:
    def _record_events(self, mgr):
        calls = []
        real = mgr._emit_fastlane_claim_event

        def spy(slot, event, **kw):
            calls.append((slot.slot_id, event))
            return real(slot, event, **kw)

        mgr._emit_fastlane_claim_event = spy
        return calls

    async def test_a_waiter_that_is_released_emits_no_claim_event(self, mgr):
        holder, waiter = _slot("holder", MAIN), _slot("waiter", CURATOR)
        _register(mgr, holder)
        _register(mgr, waiter)
        await _settle(mgr)
        calls = self._record_events(mgr)
        mgr._release_fastlane_claim_locked(waiter, "admitted")
        await _settle(mgr)
        assert calls == [], f"a request that never held the claim emitted {calls}"
        assert _claim(mgr)["slot"] is holder
        assert waiter.slot_id not in _claim(mgr).get("waiters", {"missing": None}), "the waiter is still recorded"

    async def test_control_the_holder_that_is_released_emits_admitted_and_released(self, mgr):
        holder, waiter = _slot("holder", MAIN), _slot("waiter", CURATOR)
        _register(mgr, holder)
        _register(mgr, waiter)
        await _settle(mgr)
        calls = self._record_events(mgr)
        mgr._release_fastlane_claim_locked(holder, "admitted")
        await _settle(mgr)
        assert calls == [(holder.slot_id, "fastlane_admitted"), (holder.slot_id, "fastlane_claim_released")]
        assert _claim(mgr)["slot"] is waiter


@pytest.mark.asyncio
class TestTheClaimsRequestIsMatchedByIdentity:
    async def test_a_waiter_that_merely_equals_the_claims_request_does_not_count_as_it(self, mgr):
        holder, other = _slot("holder", CURATOR), _slot("other", UNCLASSIFIED)
        _register(mgr, holder)
        _register(mgr, other)
        claim = _claim(mgr)
        assert "waiters" in claim, "setup: the claim records its waiters"
        twin = copy.copy(holder)               # equal field by field, a different object
        assert twin == holder and twin is not holder, "setup: a twin that is equal but not identical"
        claim["waiters"][holder.slot_id] = twin
        assert mgr._repoint_claim_to_best_waiter(claim) is True
        assert claim["slot"] is twin, "the claim's request was matched by equality, not identity"
        await _settle(mgr)


def _at(slot, created_at):
    """The same request with an explicit arrival time (the order ties are broken by)."""
    slot.created_at = created_at
    return slot


@pytest.mark.asyncio
class TestTiesAndTheInheritedClaim:
    async def test_an_equal_request_that_arrived_earlier_does_not_take_the_claim(self, mgr):
        holder = _at(_slot("holder", CURATOR), 10.0)
        earlier = _at(_slot("earlier", CURATOR), 5.0)
        _register(mgr, holder)
        _register(mgr, earlier)
        assert _claim(mgr)["slot"] is holder, "an equal key moved the claim to the earlier-created request"
        assert mgr._governing_claim_priority_key_locked() == (0, CURATOR)
        assert len(mgr._fastlane_claims) == 1
        await _settle(mgr)

    async def test_after_a_split_the_parked_request_is_not_a_waiter_of_the_shared_claim(self, mgr):
        r = _resident_with_inbox(mgr, {"ip": IP_1})
        parked, better = _slot("parked", 4), _slot("better", CURATOR)
        _register(mgr, parked)
        r.inbox.put_nowait(parked)
        mgr._park_fastlane_claim_locked(parked, r)
        _register(mgr, better)
        assert _claim(mgr)["slot"] is better, "setup: the better request must take the shared claim"
        mgr._release_fastlane_claim_locked(better, "admitted")
        assert list(mgr._fastlane_claims) == [(f"parked:{parked.slot_id}", MODEL)], (
            f"the parked request holds a second claim through the shared one: {list(mgr._fastlane_claims)}")
        assert len(mgr._fastlane_claims) == 1
        await _settle(mgr)

    async def test_ties_go_to_the_earliest_request_among_three(self, mgr):
        a = _at(_slot("a", CURATOR), 1.0)
        b = _at(_slot("b", CURATOR), 2.0)
        c = _at(_slot("c", CURATOR), 3.0)
        for slot in (a, b, c):
            _register(mgr, slot)
        assert _claim(mgr)["slot"] is a, "setup: the first of three equal requests holds the claim"
        mgr._release_fastlane_claim_locked(a, "admitted")
        assert _claim(mgr)["slot"] is b, "the claim skipped the earlier of the two equal waiters"
        await _settle(mgr)

    async def test_a_waiter_that_inherits_the_claim_then_parks_holds_one_parked_claim(self, mgr):
        r = _resident_with_inbox(mgr, {"ip": IP_1})
        first, inheritor = _slot("first", CURATOR), _slot("inheritor", CURATOR)
        _register(mgr, first)
        _register(mgr, inheritor)
        mgr._release_fastlane_claim_locked(first, "admitted")
        assert _claim(mgr)["slot"] is inheritor, "setup: the waiting request must inherit the claim"
        r.inbox.put_nowait(inheritor)
        mgr._park_fastlane_claim_locked(inheritor, r)
        assert len(mgr._fastlane_claims) == 1, f"claims: {list(mgr._fastlane_claims)}"
        assert _claim(mgr)["slot"] is inheritor
        assert _claim(mgr).get("parked_on") == "res-1", "the inherited claim is not marked as parked"
        mgr._release_fastlane_claim_locked(inheritor, "admitted")
        assert mgr._fastlane_claims == {}, "a claim is left after its request started its own turn"
        await _settle(mgr)
