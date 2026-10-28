"""per-resident edge-case tests: three behaviours that no other test pinned.

GAP-1 (live_monitor.resident_gen_key fallback): a resident whose resident_key is empty, or that carries
no resident_key attribute at all, is keyed by its model_tag, so its block and poller live under "m".

GAP-2 (manager driver finally): when the driver of "m#1" exits through its own finally-deregister, it
pops ONLY "m#1"'s identity entry. The pop is proven to run from that finally: the driver task ends, the
registry entry is gone, and _begin_unload_locked (the other pop site) is never called.

GAP-3 (api/live_stream resident_key precedence): when a request sends BOTH resident_key and model_tag,
resident_key wins and streams that resident's block; the legacy model_tag alone still follows instance 0.

Helpers come from the two sibling per-resident test files and from test_multislot_concurrency (tests/ is on
pytest's pythonpath).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from _fastlane_fixture import wait_until
from _multiinstance_support import _manifest, _new_manager, _probe
from test_multislot_concurrency import _fake_handle
from test_resident_keying_identity import (
    ENGINE_A_IP,
    ENGINE_B_IP,
    MODEL,
    SECOND,
    SETUP_S,
    _admit,
    _finish,
    _Gated,
    _two_engines,
    _wait_card,
)
from test_resident_keying import (
    _Client,
    _first_sse_event,
    _gen_id,
    _manager,
    _register,
    _resident,
    _seed_two_blocks,
    _slots,
    _supervisor_env,
    _teardown,
    _two_residents,
)

from turbohaul.live_monitor import LiveResidentsSupervisor
from turbohaul.manager import ResidentState
from turbohaul.slot import Slot, SlotState


class TestResidentKeyFallback:
    async def test_generation_block_falls_back_to_model_tag_when_resident_key_is_empty(self, tmp_path):
        """per-resident GAP-1a: a resident with resident_key "" (instance 0 of a double) is keyed by its
        model_tag. The block, the poller and the snapshot read all sit under "m".
        If resident_gen_key returned r.resident_key with no fallback, the key would be
        "", the registry lookup would miss, and live_generations would stay empty, so this test would fail."""
        mgr = _manager(tmp_path)
        r = _resident("", 0, pid=41001, port=59500, spawn_seq=1)
        mgr._residents["m"] = r  # the registry key the supervisor looks up for this resident
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        try:
            with _supervisor_env(_Client(_slots(1000, 10))):
                assert [x is r for x in mgr._model_residents()] == [True], "per-resident setup: resident is not live"
                await sup._tick()
            assert set(mgr.live_generations) == {"m"}, (
                f"the empty-key resident is stored under {sorted(mgr.live_generations)}, expected ['m']")
            assert set(sup._pollers) == {"m"}, f"the empty-key poller is under {sorted(sup._pollers)}"
            snap = mgr._residents_snapshot()
            assert len(snap) == 1 and snap[0]["generation"] is not None, (
                f"the empty-key resident has no snapshot generation: {snap!r}")
            assert snap[0]["generation"]["generation_id"] == _gen_id(41001, 1, r), (
                "the snapshot did not read the block that the 'm' fallback key holds")
        finally:
            await _teardown(mgr, sup)

    async def test_generation_block_falls_back_when_resident_has_no_resident_key_attribute(self, tmp_path):
        """per-resident GAP-1b: a resident double with NO resident_key attribute (the old test doubles) is keyed
        by its model_tag. The snapshot is not read here: it reads r.resident_key directly.
        Without the fallback, resident_gen_key raises AttributeError inside _tick and this test fails
        with that error."""
        mgr = _manager(tmp_path)
        slot = Slot.new("m", thread_id="thread-bare")
        slot.state = SlotState.ACTIVE
        bare = SimpleNamespace(
            model_tag="m",
            state=ResidentState.ACTIVE,
            handle=_fake_handle("m", 59500, 41001),
            port=59500,
            spawn_seq=1,
            active_slot=slot,
            driver_task=None,
            last_active_monotonic=1.0,
        )
        assert not hasattr(bare, "resident_key"), "per-resident setup: the double must carry no resident_key"
        mgr._residents["m"] = bare
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        try:
            with _supervisor_env(_Client(_slots(1000, 10))):
                await sup._tick()
            assert set(mgr.live_generations) == {"m"}, (
                f"the attribute-less resident is stored under {sorted(mgr.live_generations)}, expected ['m']")
            assert set(sup._pollers) == {"m"}, f"its poller is under {sorted(sup._pollers)}, expected ['m']"
            assert mgr.live_generations["m"]["generation_id"] == _gen_id(41001, 1, bare), (
                "the block under 'm' is not the attribute-less resident's own generation")
        finally:
            await _teardown(mgr, sup)


class TestDriverFinallyIdentityPop:
    async def test_driver_finally_deregister_pops_only_that_residents_identity(self, tmp_path):
        """per-resident GAP-2: the driver of "duo#1" is cancelled while parked in its idle wait. Its finally
        (manager.py, the exactly-once claim block that deregisters the resident) deregisters "duo#1" and
        pops ONLY "duo#1"'s identity. "duo"'s identity stays.
        Reached-the-line evidence, asserted first: the driver task is done and "duo#1" left the registry.
        The other identity pop site (_begin_unload_locked) is only reached past its
        DEAD guard, so every call to it in this window must be on an
        already-DEAD resident (the finally marked it DEAD first). The pops are recorded with a dict
        subclass that calls the real pop.
        If the finally popped the model_tag, pops would be ['duo']; if it dropped the pop, pops would be [];
        either fails this test."""
        mgr, boot, _ = _new_manager(tmp_path, budget=2)
        _manifest(boot, MODEL, auto_place=True, llama_server_flags={"split_mode": "none"})
        mgr.runtime.queue.grace_seconds = 2
        mgr.runtime.queue.idle_hot_load_seconds = 600
        gated = _Gated(mgr, block={"s-a1", "s-b1"})
        p_free, p_probe = _probe([20000, 20000])
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        with p_free, p_probe:
            try:
                await _admit(mgr, gated, "s-a1", "t-a", ENGINE_A_IP)
                await _admit(mgr, gated, "s-b1", "t-b", ENGINE_B_IP)
                _two_engines(mgr)
                await _wait_card(mgr, MODEL, "s-a1", "setup: after both admissions")
                await _wait_card(mgr, SECOND, "s-b1", "setup: after both admissions")
                gated.release("s-a1")
                gated.release("s-b1")
                await wait_until(lambda: {"s-a1", "s-b1"} <= set(gated.done), timeout=SETUP_S)
                resident = mgr._residents[SECOND]
                await wait_until(lambda: resident.state is ResidentState.IDLE_EVICTABLE, timeout=SETUP_S)
                driver = resident.driver_task
                assert driver is not None and not driver.done(), "per-resident setup: 'duo#1' has no live driver"
                by_model = mgr._last_request_identity_by_model
                assert by_model.get(MODEL, {}).get("session_id") == "s-a1", (
                    f"per-resident setup: 'duo' must hold its own identity, got {by_model.get(MODEL)!r}")
                assert by_model.get(SECOND, {}).get("session_id") == "s-b1", (
                    f"per-resident setup: 'duo#1' must hold its own identity, got {by_model.get(SECOND)!r}")

                pops = []

                class _RecordingDict(dict):
                    def pop(self, key, *default):
                        pops.append(key)
                        return super().pop(key, *default)

                mgr._last_request_identity_by_model = _RecordingDict(by_model)
                real_begin = mgr._begin_unload_locked
                begin_calls = []  # (resident_key, state at call time) for each _begin_unload_locked call

                def _spy_begin(rr):
                    begin_calls.append((rr.resident_key, rr.state))
                    return real_begin(rr)

                mgr._begin_unload_locked = _spy_begin
                driver.cancel()
                await asyncio.wait({driver}, timeout=SETUP_S)
                await wait_until(lambda: SECOND not in mgr._residents, timeout=SETUP_S)

                assert driver.done(), "the cancelled driver of 'duo#1' did not finish"
                assert SECOND not in mgr._residents, "the driver exit did not deregister 'duo#1'"
                assert all(st is ResidentState.DEAD for _key, st in begin_calls), (
                    f"_begin_unload_locked ran on a live resident, so its pop could be the one that ran: "
                    f"{begin_calls!r}")
                assert pops == [SECOND], (
                    f"identity pops during the driver finally were {pops!r}, expected only ['duo#1']")
                by_model = mgr._last_request_identity_by_model
                assert SECOND not in by_model, f"'duo#1' still has its identity entry: {by_model!r}"
                assert by_model.get(MODEL, {}).get("session_id") == "s-a1", (
                    f"the living sibling 'duo' lost its identity, now {by_model.get(MODEL)!r}")
            finally:
                await _finish(mgr, gated)


class TestResidentKeyBeatsModelTag:
    async def test_resident_key_wins_over_model_tag_when_both_are_sent(self, tmp_path):
        """per-resident GAP-3: with BOTH ?resident_key=m#1 and ?model_tag=m, the stream follows the "m#1" block
        (resident_key wins). The reverse control ?model_tag=m alone still follows the "m" block.
        If the condition were `if resident_key and not model_tag:`, model_tag would win and the first
        frame would carry the "m" block's generation_id, so this test would fail."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        block_a, block_b = _seed_two_blocks(mgr)
        mgr.live_generation = None
        try:
            both = await _first_sse_event(mgr, {"resident_key": "m#1", "model_tag": "m"})
            assert both.get("reset") is True, f"the first frame is not the reset frame: {both!r}"
            assert both["generation_id"] == block_b["generation_id"], (
                f"resident_key=m#1 plus model_tag=m streamed {both.get('generation_id')!r}, "
                "expected the m#1 block")
            legacy = await _first_sse_event(mgr, {"model_tag": "m"})
            assert legacy["generation_id"] == block_a["generation_id"], (
                f"legacy ?model_tag=m streamed {legacy.get('generation_id')!r}, expected the m block")
        finally:
            await mgr.shutdown()


class TestNoneHandleResolvesNothing:
    async def test_none_handle_resolves_no_resident_even_when_a_resident_has_no_handle(self, tmp_path):
        """per-resident T8: a None handle must resolve NO resident, even when a model resident in the
        registry carries handle None (a RESERVED_LOADING sibling with no engine yet). Without the guard
        ``if handle is None: return None``, the match ``r.handle is handle`` finds that sibling, so the
        accessor and the spawn_seq fallback would read the wrong resident.
        Asserted first: the handle-less sibling is in _model_residents(), and its spawn_seq differs from the
        legacy _spawn_seq_for_model("m") value (instance 0's).
        If the None-handle guard were removed, the accessor would return the handle-less sibling and this test would fail."""
        from turbohaul.manager import Resident

        mgr = _manager(tmp_path)
        a = _resident("m", 0, pid=41001, port=59500, spawn_seq=1)
        booting = Resident(
            model_tag="m", handle=None, port=59502, spawn_seq=7,
            state=ResidentState.RESERVED_LOADING, resident_key="m#1", instance_idx=1)
        _register(mgr, a, booting)
        try:
            assert any(r is booting for r in mgr._model_residents()), (
                "per-resident setup: the handle-less RESERVED_LOADING sibling is not in _model_residents()")
            legacy = mgr._spawn_seq_for_model("m")
            assert legacy == 1, f"per-resident setup: the legacy fallback should be instance 0's spawn_seq 1, got {legacy}"
            assert legacy != booting.spawn_seq, (
                f"per-resident setup: the handle-less sibling must carry a different spawn_seq than {legacy}")
            assert mgr._serving_resident_for_handle(None) is None, (
                "a None handle resolved a resident, expected none")
            got = mgr._spawn_seq_for_handle(None, "m")
            assert got == legacy, (
                f"a None handle gave spawn_seq {got}, expected the legacy fallback {legacy}")
        finally:
            await mgr.shutdown()
