"""Priority must govern the slot handed to a
PARKED driver, not only first admission.

`_drive_resident` has four admission moments. Two are already rank-aware (both
call `_priority_admit_from_inbox`). The third, the blocking
`await asyncio.wait_for(r.inbox.get(), timeout=idle_window)`, was not: a parked
`asyncio.Queue` getter is handed whatever was put FIRST and the driver served it
with no rank check at all -- exactly the failure to avoid:
handing the next slot to whichever client is nearest that
release rather than on rank.

These tests drive the REAL `_drive_resident` (only the sidecar spawn and the
per-turn serve are faked, so the admission path under test is the shipped one)
rather than calling the new helper directly. That is deliberate: a correct fix
wired into a site that never executes stays green. The structural
test at the bottom pins the wiring itself.

MUST FAIL ON UNMODIFIED CODE:
  - rank inversion: the parked getter takes the first-PUT (lower-ranked) slot.
  - the wedge: an admission change that returns to the INNER loop leaves a
    producer-flipped ACTIVE resident stuck there forever (see the regression
    test at the bottom of the first class).
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap

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
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.slot import Slot, SlotEvictedError, SlotState

def _boot_runtime(tmp_path):
    storage_root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (storage_root / sub).mkdir(parents=True)
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
        queue=QueueConfig(max_parallel_sidecars=2),
        pull=PullConfig(),
    )
    return boot, runtime


def _mk(tmp_path) -> TurbohaulManager:
    boot, runtime = _boot_runtime(tmp_path)
    return TurbohaulManager(boot, runtime)


def _match(rule_index: int, rank: int) -> FastLaneMatch:
    return FastLaneMatch(
        rule_index=rule_index, raw_address="10.0.0.1", label="test",
        effective_tag="main", rank=rank,
    )


def _slot(slot_id: str, created_at: float, *, fastlane=None, is_evicted=False) -> Slot:
    return Slot(
        slot_id=slot_id,
        model_tag="m1",
        state=SlotState.ACTIVE,
        created_at=created_at,
        fastlane=fastlane,
        is_evicted=is_evicted,
        client_meta={},
        completion_future=asyncio.Future(),
    )


async def _wait_until(pred, timeout=5.0, msg="condition never became true"):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if pred():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(msg)


async def _start_driver(mgr, r, served, monkeypatch, keep_alive_s=-1):
    """Run the REAL _drive_resident with only the spawn and the per-turn serve
    faked out. Everything on the admission path -- _maybe_defer_admission, the
    two _priority_admit_from_inbox grabs, and the blocking get under test --
    is the shipped code."""
    async def fake_spawn(r_, slot):
        return object()

    async def fake_serve(r_, slot, handle):
        served.append(slot)

    monkeypatch.setattr(mgr, "_spawn_for_resident", fake_spawn)
    monkeypatch.setattr(mgr, "_serve_on_resident", fake_serve)
    # keep the idle window effectively unbounded so the park under test is the
    # blocking get, never an idle-unload timeout racing the assertion
    r.latest_keep_alive_s = keep_alive_s
    r.sleep_idle_seconds = keep_alive_s
    task = asyncio.create_task(mgr._drive_resident(r))
    return task


async def _park(mgr, r, served, monkeypatch, keep_alive_s=-1):
    """Bring the driver up, let it finish one anchor turn, and leave it PARKED
    in the blocking `r.inbox.get()` with an EMPTY inbox -- the exact state in
    which the site under test is reached."""
    anchor = _slot("anchor", 0.0)
    r.inbox.put_nowait(anchor)
    task = await _start_driver(mgr, r, served, monkeypatch, keep_alive_s=keep_alive_s)
    await _wait_until(lambda: len(served) >= 1, msg="anchor turn never ran")
    await _wait_until(
        lambda: r.state is ResidentState.IDLE_EVICTABLE and r.inbox.empty(),
        msg="driver never parked in IDLE_EVICTABLE with an empty inbox",
    )
    return task


@pytest.mark.asyncio
class TestPriorityGovernsTheHandoverToAParkedDriver:
    async def test_higher_ranked_slot_wins_even_though_it_arrived_second(
        self, tmp_path, monkeypatch
    ):
        """THE rank-inversion defect. Driver is parked; a LOW-priority slot is put first
        and a HIGH-priority slot second. asyncio hands the parked getter the
        first-put item, so unmodified code serves LOW -- a lower-ranked client
        admitted ahead of a higher-ranked one that was already waiting.

        Both puts happen before any await, so they land while the getter is
        genuinely parked (this is the real dispatcher's shape: `_route_or_reserve`
        has a single caller and its HIT path has no suspension point, so it can
        issue two puts inside one event-loop callback)."""
        mgr = _mk(tmp_path)
        r = Resident(model_tag="m1", inbox=asyncio.Queue())
        served: list[Slot] = []
        task = await _park(mgr, r, served, monkeypatch)
        try:
            low = _slot("low", 1.0, fastlane=_match(rule_index=5, rank=5))
            high = _slot("high", 2.0, fastlane=_match(rule_index=0, rank=0))
            r.inbox.put_nowait(low)   # arrives FIRST
            r.inbox.put_nowait(high)  # arrives SECOND, outranks it

            await _wait_until(lambda: len(served) >= 2, msg="second turn never ran")
            assert served[1].slot_id == "high", (
                "rank inversion: the HIGHEST-ranked waiting slot must be admitted, not the "
                f"one that happened to arrive first -- served {served[1].slot_id!r}"
            )
            await _wait_until(lambda: len(served) >= 3, msg="third turn never ran")
            assert served[2].slot_id == "low", "the loser must still be served next"
        finally:
            task.cancel()

    async def test_equal_ranked_peer_does_not_displace_the_woken_slot(
        self, tmp_path, monkeypatch
    ):
        """⭐ NEGATIVE CONTROL, and the load-bearing half of this fix: making
        admission rank-aware must NOT start bumping a client for a peer of the
        SAME rank. `_fastlane_strictly_higher` treats an equal (rule_index,
        rank) as a real tie precisely so an agent's own follow-up keeps its warm
        slot instead of two equal agents ping-ponging context reloads.

        Fails if the fix ever selects on arrival time or on >= instead of >:
        both slots are rule0/rank0, so the FIRST-arrived must still win."""
        mgr = _mk(tmp_path)
        r = Resident(model_tag="m1", inbox=asyncio.Queue())
        served: list[Slot] = []
        task = await _park(mgr, r, served, monkeypatch)
        try:
            first = _slot("first", 1.0, fastlane=_match(rule_index=0, rank=0))
            peer = _slot("peer", 2.0, fastlane=_match(rule_index=0, rank=0))
            r.inbox.put_nowait(first)
            r.inbox.put_nowait(peer)

            await _wait_until(lambda: len(served) >= 2, msg="second turn never ran")
            assert served[1].slot_id == "first", (
                "an EQUAL-ranked peer must not displace the slot already handed "
                f"over -- served {served[1].slot_id!r}"
            )
        finally:
            task.cancel()

    async def test_evicted_woken_rider_is_still_served_this_path_does_not_reap(
        self, tmp_path, monkeypatch
    ):
        """⛔ PINS A DELIBERATE NON-CHANGE. Do not "fix" this.

        Reaping an evicted rider handed to a parked driver was considered and rejected,
        although `_priority_admit_from_inbox` does it for
        riders it drains itself, because
        skipping the serve means the driver never re-enters the
        OUTER loop, and `r.state = ResidentState.IDLE_EVICTABLE` is assigned in
        exactly ONE place in manager.py -- inside that outer loop. The resident
        would stay ACTIVE forever: never idle-unloading, never an LRU make-room
        victim, holding its VRAM slot.

        So this path deliberately keeps its pre-existing behaviour: the woken slot
        is SERVED, evicted or not. Reaping it is not a local improvement -- it
        requires a control-flow change to the four-part park cluster, which is a
        separate design decision.
        """
        mgr = _mk(tmp_path)
        r = Resident(model_tag="m1", inbox=asyncio.Queue())
        served: list[Slot] = []
        task = await _park(mgr, r, served, monkeypatch)
        try:
            ghost = _slot("ghost", 1.0, is_evicted=True)
            r.inbox.put_nowait(ghost)
            await _wait_until(lambda: len(served) >= 2, msg="second turn never ran")
            assert served[1].slot_id == "ghost", (
                "this path serves the woken slot as it always has -- reaping it "
                "here wedges the resident (see this test's docstring)"
            )
            # ⭐ The load-bearing half: "served" alone does not distinguish
            # "left alone" from "reaped AND then served anyway". A mutant that
            # removed the is_evicted exemption would still pass the assertion above,
            # because the fallback still hands the slot back -- while its
            # future would already be failed. A slot both failed and served is
            # strictly worse than either. Pin the future, not just the serve.
            assert not ghost.completion_future.done(), (
                "this path must not fail-complete the woken slot -- it is not "
                "the reaper, and a slot that is failed AND served is worse than "
                "either outcome alone"
            )
        finally:
            task.cancel()

    async def test_resident_still_idle_unloads_after_a_producer_flipped_it_active(
        self, tmp_path, monkeypatch
    ):
        """⭐⭐ THE REGRESSION TEST. Keep it whatever else changes.

        This is the shape the other fixtures did NOT have, and that gap is how a
        reaping variant would pass every other test while wedging a
        resident permanently. The real producer (`_route_to`) does not merely put
        into `r.inbox`: it takes `_registry_lock`, flips IDLE_EVICTABLE -> ACTIVE,
        and THEN puts. A fixture that puts directly leaves the resident
        IDLE_EVICTABLE and cannot observe the wedge at all.

        So: reproduce the producer faithfully, then assert the resident still
        completes its idle lifecycle. Any admission change that returns to the
        inner loop without paying off that ACTIVE flip leaves the resident stuck
        here and this test fails.
        """
        mgr = _mk(tmp_path)
        r = Resident(model_tag="m1", inbox=asyncio.Queue())
        served: list[Slot] = []
        # short idle window so the idle-unload actually fires inside the test
        task = await _park(mgr, r, served, monkeypatch, keep_alive_s=1)
        try:
            ghost = _slot("ghost", 1.0, is_evicted=True)
            async with mgr._registry_lock:      # exactly what _route_to does
                r.state = ResidentState.ACTIVE
                r.inbox.put_nowait(ghost)

            await _wait_until(
                lambda: r.state is not ResidentState.ACTIVE,
                timeout=8.0,
                msg=(
                    "resident WEDGED in ACTIVE with an empty inbox: it can never "
                    "idle-unload and can never be an LRU make-room victim"
                ),
            )
        finally:
            task.cancel()


class TestTheFixIsActuallyWiredIn:
    """A correct fix in a site that never runs is GREEN. These pin the wiring
    itself, independently of the behavioural tests above."""

    def test_drive_resident_calls_the_named_helper(self):
        src = textwrap.dedent(inspect.getsource(TurbohaulManager._drive_resident))
        calls = [
            n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_rank_admit_woken"
        ]
        assert len(calls) == 1, (
            "_drive_resident must contain exactly ONE call to _rank_admit_woken "
            f"-- found {len(calls)}"
        )

    def test_the_helper_is_await_free(self):
        """The selection must stay atomic in the single-threaded loop: an await
        between 'which is best' and 'admit it' reopens the double-admit window
        `_priority_admit_from_inbox`'s own await-free contract exists to close.
        The awaited `wait_for` stays OUTSIDE, at the call site, before this
        helper is entered."""
        src = textwrap.dedent(inspect.getsource(TurbohaulManager._rank_admit_woken))
        awaits = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Await)]
        assert awaits == [], (
            f"_rank_admit_woken must have zero suspension points -- found {len(awaits)}"
        )
