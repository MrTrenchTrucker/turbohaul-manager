"""Priority-select at the three inbox admit-grab
sites in manager.py (fan-out top-up, idle-wake grab, idle-timeout
regrab), via one shared await-free helper: TurbohaulManager._priority_admit_
from_inbox, following the multi-model priority design.

These are unit-level tests against the helper directly -- they construct a
Resident with a real asyncio.Queue() inbox and call the helper, without
spinning up a full dispatcher/driver loop (matching the design's own fixture
description).

Must-fail controls (admit order, keep-alive shift and the is_evicted design
decision) are proven separately, by mutation, NOT as standing
tests here -- a test that is SUPPOSED to fail would break CI. The
mutated-helper runs turn them RED, as expected.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this file> -v
"""
from __future__ import annotations

import asyncio
import inspect

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
from turbohaul.manager import Resident, TurbohaulManager
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
            default_port_base=59500,
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


def _slot(slot_id: str, created_at: float, *, fastlane=None, is_evicted=False,
          keep_alive_s=None, with_future=False) -> Slot:
    return Slot(
        slot_id=slot_id,
        model_tag="m1",
        state=SlotState.ACTIVE,
        created_at=created_at,
        fastlane=fastlane,
        is_evicted=is_evicted,
        client_meta={"keep_alive_s": keep_alive_s} if keep_alive_s is not None else {},
        completion_future=asyncio.Future() if with_future else None,
    )


def _resident() -> Resident:
    return Resident(model_tag="m1", inbox=asyncio.Queue())


async def _drain_ids(inbox: "asyncio.Queue[Slot]") -> list[str]:
    out = []
    while not inbox.empty():
        out.append((await inbox.get()).slot_id)
    return out


class TestAdmitOrder:
    """Priority beats FIFO: the exact given fixture."""

    async def test_admit_order_priority_over_fifo(self, tmp_path):
        mgr = _mk(tmp_path)
        r = _resident()
        s0 = _slot("s0", 0.0)                          # unlisted, arrives 1st
        s1 = _slot("s1", 1.0, fastlane=_match(1, 5))    # listed, rule1/rank5, 2nd
        s2 = _slot("s2", 2.0, fastlane=_match(0, 1))    # listed, rule0/rank1, 3rd
        s3 = _slot("s3", 3.0)                           # unlisted, arrives 4th (last)
        for s in (s0, s1, s2, s3):
            r.inbox.put_nowait(s)

        order = []
        while True:
            winner = mgr._priority_admit_from_inbox(r)
            if winner is None:
                break
            order.append(winner.slot_id)

        assert order == ["s2", "s1", "s0", "s3"], order
        assert r.inbox.empty()

    async def test_rule_index_is_primary_over_rank(self, tmp_path):
        """Every other fixture in this file has one candidate dominate in
        BOTH fastlane fields at once, which can't catch a swapped-field
        regression in the sort key (e.g. (rank, rule_index) instead of
        (rule_index, rank)). This one pits the two fields against each
        other directly: s_a has a worse rule_index but a better rank;
        s_b has a better rule_index but a worse rank. Per queue.py's
        _fastlane_priority_key contract (rule_index = address-list
        position, primary; rank = within-address sub-rank, secondary),
        rule_index must decide it -- s_b must win."""
        mgr = _mk(tmp_path)
        r = _resident()
        s_a = _slot("s_a", 0.0, fastlane=_match(1, 1))
        s_b = _slot("s_b", 1.0, fastlane=_match(0, 5))
        r.inbox.put_nowait(s_a)
        r.inbox.put_nowait(s_b)

        winner = mgr._priority_admit_from_inbox(r)

        assert winner.slot_id == "s_b", (
            "rule_index must be primary over rank in the select key"
        )


class TestRestoreOrder:
    """Partial drain must not invert the rest."""

    async def test_restore_order_no_inversion_on_partial_drain(self, tmp_path):
        mgr = _mk(tmp_path)
        r = _resident()
        s0 = _slot("s0", 0.0)
        s1 = _slot("s1", 1.0, fastlane=_match(1, 5))
        s2 = _slot("s2", 2.0, fastlane=_match(0, 1))
        s3 = _slot("s3", 3.0)
        for s in (s0, s1, s2, s3):
            r.inbox.put_nowait(s)

        winner = mgr._priority_admit_from_inbox(r)
        assert winner.slot_id == "s2"

        remaining = await _drain_ids(r.inbox)
        assert remaining == ["s0", "s1", "s3"], remaining


class TestKeepAliveShift:
    """The next idle window follows the LAST-SERVED slot,
    not the last-ARRIVED one. Fixture is deliberately built so the two differ:
    s0 and s2 share the same (rule_index, rank) -- s0 wins the tie (earlier
    created_at) and is served FIRST; s2 (listed, arrives LAST) is served
    SECOND; s1 (unlisted) is served LAST despite arriving in the middle."""

    async def test_keep_alive_follows_last_served_not_last_arrived(self, tmp_path):
        mgr = _mk(tmp_path)
        r = _resident()
        s0 = _slot("s0", 0.0, fastlane=_match(0, 0), keep_alive_s=100)  # served 1st
        s1 = _slot("s1", 1.0, keep_alive_s=200)                        # served LAST
        s2 = _slot("s2", 2.0, fastlane=_match(0, 0), keep_alive_s=300)  # arrives LAST, served 2nd
        for s in (s0, s1, s2):
            r.inbox.put_nowait(s)

        order = []
        while True:
            winner = mgr._priority_admit_from_inbox(r)
            if winner is None:
                break
            order.append(winner)
        assert [s.slot_id for s in order] == ["s0", "s2", "s1"]

        # The mechanism this composes with is manager.py's _serve_on_resident,
        # which sets r.latest_keep_alive_s from whatever slot it is actually
        # handed -- provenance-agnostic, doesn't care how that slot was
        # selected. This does NOT execute _serve_on_resident or prove the
        # composition end-to-end (that would need a full driver-loop
        # integration test); it's a drift tripwire only -- confirms the
        # exact statement this test's manual mirror below depends on still
        # exists verbatim in the real source, so a refactor that moves or
        # changes it fails loudly here instead of leaving this test
        # silently checking a stale assumption.
        src = inspect.getsource(TurbohaulManager._serve_on_resident)
        assert 'r.latest_keep_alive_s = (slot.client_meta or {}).get("keep_alive_s")' in src, (
            "the provenance-agnostic keep_alive assignment this test's mirror "
            "depends on has moved or changed -- update the mirror below"
        )
        for s in order:
            r.latest_keep_alive_s = (s.client_meta or {}).get("keep_alive_s")
        assert r.latest_keep_alive_s == 200, (
            "idle window should follow the LAST-SERVED slot (s1, keep_alive=200), "
            "not the last-ARRIVED slot (s2, keep_alive=300)"
        )


class TestEvictedRiders:
    """Design decision: evicted riders are fail-completed
    during the drain and never restored -- under priority-select a
    low-priority evicted rider would otherwise be drained-not-selected-
    restored every pass and never collected under sustained higher-priority
    traffic. It has its own test."""

    async def test_evicted_rider_fail_completed_not_restored_even_if_highest_priority(
        self, tmp_path
    ):
        mgr = _mk(tmp_path)
        r = _resident()
        # Evicted rider has the best REALISTICALLY ATTAINABLE priority key
        # (rank 1 is the best rank per config.py's documented 1-5/None=
        # unranked domain -- rank 0 is never produced by the real matching
        # pipeline) -- proves eviction disqualifies it regardless of rank,
        # not just "it happened to lose" against a token low-priority rival.
        s_evicted = _slot(
            "s_evicted", 0.0, fastlane=_match(0, 1), is_evicted=True, with_future=True
        )
        s_live = _slot("s_live", 1.0, with_future=True)  # unlisted, worst priority
        r.inbox.put_nowait(s_evicted)
        r.inbox.put_nowait(s_live)

        winner = mgr._priority_admit_from_inbox(r)

        assert winner is s_live, "the live rider must be admitted, not the evicted one"
        assert not winner.completion_future.done(), (
            "the admitted winner's own completion_future must be left untouched "
            "here -- it resolves later via _serve_on_resident, not inside the "
            "drain step. A mutation that fail-completes every drained slot "
            "unconditionally (not just evicted ones) would otherwise ship silently."
        )
        assert s_evicted.completion_future.done()
        exc = s_evicted.completion_future.exception()
        assert isinstance(exc, SlotEvictedError), exc
        assert r.inbox.empty(), "the evicted rider must be discarded, not restored"

    async def test_all_evicted_returns_none_and_drains_inbox(self, tmp_path):
        mgr = _mk(tmp_path)
        r = _resident()
        s1 = _slot("s1", 0.0, is_evicted=True, with_future=True)
        s2 = _slot("s2", 1.0, is_evicted=True, with_future=True)
        r.inbox.put_nowait(s1)
        r.inbox.put_nowait(s2)

        winner = mgr._priority_admit_from_inbox(r)

        assert winner is None
        assert r.inbox.empty()
        for s in (s1, s2):
            assert s.completion_future.done()
            assert isinstance(s.completion_future.exception(), SlotEvictedError)


class TestEmptyInbox:
    async def test_empty_inbox_returns_none(self, tmp_path):
        mgr = _mk(tmp_path)
        r = _resident()
        assert mgr._priority_admit_from_inbox(r) is None
