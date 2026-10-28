"""Unit tests for the queue_liveness module, driven by small fakes (no manager, no queue).

The fake host has exactly the members the host contract declares. It keeps real FIFO buffers for
the queue and a real ``asyncio.Queue`` per resident inbox, and records every call the module makes,
so a test can see what the module asked of it and in which order.
"""
import ast
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

import turbohaul.queue_liveness as ql


def _state(name):
    return SimpleNamespace(name=name)


@dataclass(eq=False)
class FakeSlot:
    slot_id: str = "s1"
    state: object = field(default_factory=lambda: _state("STAGED"))
    disconnect_event: asyncio.Event | None = None
    is_evicted: bool = False
    completion_future: object | None = None


@dataclass(eq=False)
class FakeResident:
    state: object = field(default_factory=lambda: _state("IDLE_EVICTABLE"))
    active_slot: object | None = None
    inbox: asyncio.Queue | None = field(default_factory=asyncio.Queue)


class FakeQueue:
    def __init__(self, host):
        self.host = host
        self.staging: list = []
        self.accept: list = []

    def waiting_snapshot(self):
        return list(self.staging) + list(self.accept)

    async def remove(self, slot_id):
        self.host.calls.append(("queue.remove", slot_id, self.host.lock_held))
        await asyncio.sleep(0)
        for buf in (self.staging, self.accept):
            for i, s in enumerate(buf):
                if s.slot_id == slot_id:
                    del buf[i]
                    return s
        return None


class _TrackedLock:
    """An async lock that tells the host whether it is held."""

    def __init__(self, host):
        self._lock = asyncio.Lock()
        self._host = host

    async def __aenter__(self):
        await self._lock.acquire()
        self._host.lock_held = True

    async def __aexit__(self, *exc):
        self._host.lock_held = False
        self._lock.release()


class FakeHost:
    def __init__(self, residents=()):
        self.calls: list = []
        self.lock_held = False
        self.residents = list(residents)
        self._registry_lock = _TrackedLock(self)
        self.queue = FakeQueue(self)
        self.spawned: list = []

    def _model_residents(self):
        return tuple(self.residents)

    def _release_fastlane_claim_locked(self, slot, reason):
        self.calls.append(("release_claim", slot.slot_id, reason, self.lock_held))

    def _handle_unloaded_slot(self, slot):
        self.calls.append(("handle_unloaded", slot.slot_id))
        if slot.completion_future is not None and not slot.completion_future.done():
            slot.completion_future.set_exception(RuntimeError("evicted"))
            slot.completion_future.exception()  # mark retrieved

    def _spawn_bg(self, coro):
        task = asyncio.get_running_loop().create_task(coro)
        self.spawned.append(task)
        return task

    def n(self, name):
        return [c for c in self.calls if c[0] == name]


def _slot(slot_id="s1", *, event=True, state="STAGED", future=True):
    loop = asyncio.get_running_loop()
    return FakeSlot(
        slot_id=slot_id,
        state=_state(state),
        disconnect_event=asyncio.Event() if event else None,
        completion_future=loop.create_future() if future else None,
    )


def _busy():
    return FakeResident(state=_state("ACTIVE"), active_slot=FakeSlot(slot_id="running"))


def _parked():
    return FakeResident(state=_state("IDLE_EVICTABLE"), active_slot=None)


def _loading():
    return FakeResident(state=_state("RESERVED_LOADING"), active_slot=None)


def _inbox_ids(r):
    items = []
    while not r.inbox.empty():
        items.append(r.inbox.get_nowait())
    for s in items:
        r.inbox.put_nowait(s)
    return [s.slot_id for s in items]


# -- client_gone: truth table ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "event_state,evicted,expected",
    [
        ("none", False, False),
        ("clear", False, False),
        ("set", False, True),
        ("none", True, True),
        ("clear", True, True),
        ("set", True, True),
    ],
)
def test_client_gone_truth_table(event_state, evicted, expected):
    event = None if event_state == "none" else asyncio.Event()
    if event_state == "set":
        event.set()
    slot = FakeSlot(disconnect_event=event, is_evicted=evicted)
    assert ql.client_gone(slot) is expected
    assert slot.is_evicted is evicted, "client_gone is a pure read"


# -- stamp_if_disconnected -------------------------------------------------------------------------

def test_stamp_sets_the_flag_only_when_the_event_is_set():
    event = asyncio.Event()
    slot = FakeSlot(disconnect_event=event)
    assert ql.stamp_if_disconnected(slot) is False
    assert slot.is_evicted is False, "an unset event must not flag the slot"
    event.set()
    assert ql.stamp_if_disconnected(slot) is True
    assert slot.is_evicted is True


def test_stamp_never_clears_the_flag_and_tolerates_no_event():
    slot = FakeSlot(disconnect_event=None, is_evicted=True)
    assert ql.stamp_if_disconnected(slot) is True
    assert slot.is_evicted is True
    assert ql.stamp_if_disconnected(FakeSlot(disconnect_event=None)) is False


# -- drop_if_gone ----------------------------------------------------------------------------------

async def test_drop_if_gone_leaves_a_live_slot_alone():
    host, slot = FakeHost(), _slot()
    assert ql.drop_if_gone(host, slot) is False
    assert host.calls == [] and slot.is_evicted is False and not slot.completion_future.done()


async def test_drop_if_gone_fails_a_gone_slot_exactly_once():
    host, slot = FakeHost(), _slot()
    slot.disconnect_event.set()
    assert ql.drop_if_gone(host, slot) is True
    assert slot.is_evicted is True and slot.completion_future.done()
    assert ql.drop_if_gone(host, slot) is True, "still reports the client as gone"
    assert host.n("handle_unloaded") == [("handle_unloaded", "s1")], "the second call must not fail it again"


async def test_drop_if_gone_does_not_fail_a_slot_whose_future_is_already_done():
    host, slot = FakeHost(), _slot()
    slot.completion_future.set_result("served")
    slot.is_evicted = True
    assert ql.drop_if_gone(host, slot) is True
    assert host.calls == []


async def test_drop_if_gone_touches_no_queue_inbox_or_claim():
    resident = _busy()
    host = FakeHost([resident])
    slot = _slot()
    slot.disconnect_event.set()
    host.queue.staging.append(slot)
    resident.inbox.put_nowait(slot)
    ql.drop_if_gone(host, slot)
    assert host.queue.staging == [slot] and _inbox_ids(resident) == ["s1"]
    assert host.n("release_claim") == [] and host.n("queue.remove") == []


# -- evict_unstarted -------------------------------------------------------------------------------

async def test_evict_a_staged_slot_removes_releases_fails_and_flags():
    host, slot = FakeHost(), _slot()
    host.queue.staging.append(slot)
    assert await ql.evict_unstarted(host, slot) is True
    assert host.queue.staging == []
    assert slot.is_evicted is True and slot.disconnect_event.is_set()
    assert slot.completion_future.done()
    assert host.n("release_claim") == [("release_claim", "s1", "disconnected", True)], "released under the lock"
    assert host.n("queue.remove") == [("queue.remove", "s1", False)], "removed outside the lock"
    assert host.n("handle_unloaded") == [("handle_unloaded", "s1")]


async def test_evict_a_slot_in_the_acceptance_buffer():
    host, slot = FakeHost(), _slot()
    host.queue.accept.append(slot)
    assert await ql.evict_unstarted(host, slot) is True
    assert host.queue.accept == []


async def test_evict_is_idempotent_the_second_call_returns_false_and_does_nothing():
    host, slot = FakeHost(), _slot()
    host.queue.staging.append(slot)
    assert await ql.evict_unstarted(host, slot) is True
    before = list(host.calls)
    assert await ql.evict_unstarted(host, slot) is False
    assert host.calls == before


async def test_two_racing_evictions_fail_the_slot_once():
    host, slot = FakeHost(), _slot()
    host.queue.staging.append(slot)
    results = await asyncio.gather(ql.evict_unstarted(host, slot), ql.evict_unstarted(host, slot))
    assert sorted(results) == [False, True]
    assert len(host.n("handle_unloaded")) == 1


@pytest.mark.parametrize("state", ["LOADING", "ACTIVE", "GRACE", "ACTIVE_MATCH", "POPPED", "COLD"])
async def test_a_started_slot_is_untouched(state):
    host, slot = FakeHost(), _slot(state=state)
    assert await ql.evict_unstarted(host, slot) is False
    assert host.calls == []
    assert slot.is_evicted is False and not slot.disconnect_event.is_set() and not slot.completion_future.done()


@pytest.mark.parametrize("state", ["RECEIVED", "ACCEPT_BUFFER", "STAGED"])
async def test_every_unstarted_state_is_evictable(state):
    host, slot = FakeHost(), _slot(state=state)
    assert await ql.evict_unstarted(host, slot) is True


async def test_a_slot_whose_future_is_done_is_not_evicted():
    host, slot = FakeHost(), _slot()
    slot.completion_future.set_result("served")
    assert await ql.evict_unstarted(host, slot) is False
    assert host.calls == [] and slot.is_evicted is False


async def test_a_slot_without_an_event_is_still_evicted_and_flagged():
    host, slot = FakeHost(), _slot(event=False)
    host.queue.staging.append(slot)
    assert await ql.evict_unstarted(host, slot) is True
    assert slot.is_evicted is True and slot.completion_future.done()


async def test_the_flags_are_set_before_the_first_await():
    host, slot = FakeHost(), _slot()
    host.queue.staging.append(slot)
    task = asyncio.ensure_future(ql.evict_unstarted(host, slot))
    await asyncio.sleep(0)  # the task runs to its first suspension point
    assert slot.is_evicted is True and slot.disconnect_event.is_set()
    await task


async def test_a_mid_turn_resident_loses_the_slot_from_its_inbox_and_keeps_the_others_in_order():
    resident = _busy()
    host = FakeHost([resident])
    a, b, c = _slot("a"), _slot("b"), _slot("c")
    for s in (a, b, c):
        resident.inbox.put_nowait(s)
    assert await ql.evict_unstarted(host, b) is True
    assert _inbox_ids(resident) == ["a", "c"]
    assert b.completion_future.done() and not a.completion_future.done()
    assert host.n("release_claim") == [("release_claim", "b", "disconnected", True)]


async def test_a_parked_resident_keeps_the_slot_in_its_inbox_flagged_released_and_failed():
    resident = _parked()
    host = FakeHost([resident])
    slot = _slot()
    resident.inbox.put_nowait(slot)
    assert await ql.evict_unstarted(host, slot) is True
    assert _inbox_ids(resident) == ["s1"], "taking it would wedge a parked driver"
    assert slot.is_evicted and slot.disconnect_event.is_set() and slot.completion_future.done()
    assert host.n("release_claim") == [("release_claim", "s1", "disconnected", True)]


async def test_a_parked_resident_that_a_producer_flipped_to_active_is_not_mid_turn():
    """ACTIVE with no active request: the producer flipped it before putting. Not busy."""
    resident = FakeResident(state=_state("ACTIVE"), active_slot=None)
    host = FakeHost([resident])
    slot = _slot()
    resident.inbox.put_nowait(slot)
    assert await ql.evict_unstarted(host, slot) is True
    assert _inbox_ids(resident) == ["s1"]


@pytest.mark.parametrize("state", ["IDLE_EVICTABLE", "RESERVED_LOADING", "GRACE"])
async def test_only_an_active_resident_with_a_running_request_counts_as_mid_turn(state):
    """A set ``active_slot`` alone is not enough: the state must be ACTIVE too."""
    resident = FakeResident(state=_state(state), active_slot=FakeSlot(slot_id="running"))
    host = FakeHost([resident])
    slot = _slot()
    resident.inbox.put_nowait(slot)
    assert await ql.evict_unstarted(host, slot) is True
    assert _inbox_ids(resident) == ["s1"]


async def test_a_resident_still_loading_keeps_its_first_slot_in_the_inbox():
    resident = _loading()
    host = FakeHost([resident])
    slot = _slot()
    resident.inbox.put_nowait(slot)
    assert await ql.evict_unstarted(host, slot) is True
    assert _inbox_ids(resident) == ["s1"]
    assert slot.completion_future.done() and host.n("release_claim")


async def test_mid_turn_resident_without_the_slot_is_left_alone():
    resident = _busy()
    host = FakeHost([resident])
    other = _slot("other")
    resident.inbox.put_nowait(other)
    slot = _slot()
    host.queue.staging.append(slot)
    assert await ql.evict_unstarted(host, slot) is True
    assert _inbox_ids(resident) == ["other"]


async def test_a_resident_without_an_inbox_is_skipped():
    host = FakeHost([FakeResident(state=_state("ACTIVE"), active_slot=FakeSlot(), inbox=None)])
    slot = _slot()
    host.queue.staging.append(slot)
    assert await ql.evict_unstarted(host, slot) is True


async def test_a_slot_that_began_during_the_removal_is_not_failed():
    host, slot = FakeHost(), _slot()
    host.queue.staging.append(slot)
    real_remove = host.queue.remove

    async def remove_then_start(slot_id):
        out = await real_remove(slot_id)
        slot.state = _state("LOADING")
        return out

    host.queue.remove = remove_then_start
    assert await ql.evict_unstarted(host, slot) is False
    assert not slot.completion_future.done()


# -- notify_client_gone ------------------------------------------------------------------------------

def test_notify_client_gone_is_a_plain_function_not_a_coroutine_function():
    assert not asyncio.iscoroutinefunction(ql.notify_client_gone)


async def test_notify_sets_the_event_synchronously_and_schedules_through_the_host():
    host, slot = FakeHost(), _slot()
    host.queue.staging.append(slot)
    ret = ql.notify_client_gone(host, slot.disconnect_event)
    assert ret is None
    assert slot.disconnect_event.is_set(), "set before the call returns"
    assert len(host.spawned) == 1, "scheduled on the host, not awaited"
    assert host.calls == [], "nothing has run yet: the caller did not await"
    await asyncio.gather(*host.spawned)
    assert slot.is_evicted and slot.completion_future.done() and host.queue.staging == []


async def test_notify_evicts_every_waiting_slot_that_carries_the_event_and_only_those():
    resident = _busy()
    host = FakeHost([resident])
    event = asyncio.Event()
    staged, accepted, parked, other = _slot("a"), _slot("b"), _slot("c"), _slot("d")
    for s in (staged, accepted, parked):
        s.disconnect_event = event
    host.queue.staging.extend([staged, other])
    host.queue.accept.append(accepted)
    resident.inbox.put_nowait(parked)
    ql.notify_client_gone(host, event)
    await asyncio.gather(*host.spawned)
    assert all(s.is_evicted and s.completion_future.done() for s in (staged, accepted, parked))
    assert not other.is_evicted and not other.completion_future.done()
    assert host.queue.staging == [other] and host.queue.accept == []
    assert _inbox_ids(resident) == []


async def test_notify_is_idempotent():
    host, slot = FakeHost(), _slot()
    host.queue.staging.append(slot)
    ql.notify_client_gone(host, slot.disconnect_event)
    ql.notify_client_gone(host, slot.disconnect_event)
    await asyncio.gather(*host.spawned)
    assert len(host.n("handle_unloaded")) == 1


async def test_notify_with_no_matching_slot_only_sets_the_event():
    host = FakeHost()
    event = asyncio.Event()
    ql.notify_client_gone(host, event)
    await asyncio.gather(*host.spawned)
    assert event.is_set() and host.calls == []


async def test_one_failing_eviction_does_not_stop_the_others():
    host = FakeHost()
    event = asyncio.Event()
    first, second = _slot("a"), _slot("b")
    first.disconnect_event = second.disconnect_event = event
    host.queue.staging.extend([first, second])
    real = host._handle_unloaded_slot

    def flaky(slot):
        if slot is first:
            raise RuntimeError("boom")
        real(slot)

    host._handle_unloaded_slot = flaky
    ql.notify_client_gone(host, event)
    await asyncio.gather(*host.spawned)
    assert second.completion_future.done()


def test_notify_without_a_running_loop_sets_the_event_and_does_not_raise():
    class NoLoopHost(FakeHost):
        def _spawn_bg(self, coro):
            asyncio.get_running_loop()  # raises RuntimeError: no running loop
            return None

    event = asyncio.Event()
    ql.notify_client_gone(NoLoopHost(), event)
    assert event.is_set()


# -- purity ------------------------------------------------------------------------------------------

def test_the_module_imports_nothing_from_the_project():
    pkg = Path(ql.__file__).resolve().parent
    for p in sorted(pkg.glob("*.py")):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "turbohaul" for a in node.names), p.name
            elif isinstance(node, ast.ImportFrom):
                assert node.level < 2, f"{p.name} climbs above its package"
                assert not (node.level == 0 and (node.module or "").split(".")[0] == "turbohaul"), p.name
