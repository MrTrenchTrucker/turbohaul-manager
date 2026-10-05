"""A barriered engine is never handed a new request by the multi-engine route.

While an engine is held by the spawn-time reclaim barrier, a same-model request that
arrives must not be put in its inbox: it would lose disconnect handling for the whole
hold. The single-engine path has its own guard. This file covers the OTHER one, inside
the closure the multi-engine branch of ``_route_or_reserve`` uses to hand a request to
the engine it picked.

The setup reaches that closure by intent, not by accident. The manifest asks for one
card per engine (``auto_place`` with ``split_mode: none``), so the branch is entered
because of the manifest; the tag's single live engine is keyed the way production keys
a first engine (its resident key is the model tag); and the box budget is spent
(``max_parallel_sidecars`` equals the live engine count), so the budget function answers
before any card probe and nothing can be spawned. The branch then skips spawning and
picks the existing engine.

The tests do not depend on the host: every card probe, the spawn step and the engine
reservation are replaced by spies that record a call and fail it, and each test asserts
the record is empty. They pass with or without a GPU tool on the path.

Everything else is real code: the real routing method, the real deferral funnel and the
real claim registry and release helper.
"""

import asyncio
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

_TREE = Path(__file__).resolve().parents[1]
for _p in (str(_TREE / "src"), str(_TREE / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from turbohaul.config import (  # noqa: E402
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
)
from turbohaul.fastlane import FastLaneMatch  # noqa: E402
from turbohaul.manager import (  # noqa: E402
    ResidentState,
    read_manifest_cached,
    _wants_one_card_per_engine,
)
from _multiinstance_support import (  # noqa: E402
    _manifest,
    _new_manager,
    _live_engine,
    _request,
)

TAG = "duo"
CLIENT = "203.0.113.7"

# Every seam that reads a card or places an engine. None may run on a spent budget.
_PROBE_SEAMS = (
    "turbohaul.manager._read_free_vram_all_mib",
    "turbohaul.manager._read_total_vram_all_mib",
    "turbohaul.safety._read_free_vram_all_mib",
    "turbohaul.safety._read_total_vram_all_mib",
)
_MANAGER_SEAMS = (
    "_cards_that_fit", "_card_avail_mib", "_auto_pick_gpu", "_resolve_placement_locked",
    "_pick_card_for_new_instance", "_vram_admits_locked", "_reserve_and_start_locked",
)


class _Scene:
    """One manager, one engine, one listed rider, and the spies around them."""

    def __init__(self, tmp_path, *, barriered):
        self.manager, self.boot, self.spawn_calls = _new_manager(tmp_path, budget=1)
        self.manager.runtime.fastlane = FastLaneConfig(
            enabled=True,
            rules=[FastLaneRule(address=CLIENT, tag_ranks=FastLaneTagRanks(main=1))],
        )
        _manifest(self.boot, TAG, auto_place=True,
                  llama_server_flags={"split_mode": "none"})
        self.engine = _live_engine(self.manager, TAG, gpu=0)
        self.engine.spawn_barrier_active = barriered
        self.rider = _request(TAG)
        self.rider.fastlane = FastLaneMatch(
            rule_index=0, raw_address=CLIENT, label="", effective_tag="main", rank=1,
        )
        self.forbidden_calls = []
        self.picked = []
        self.released = []

    def spies(self):
        """Patch the probe and spawn seams to record-and-fail, and spy on the picker and
        the claim release while delegating to the real methods."""
        mgr, stack = self.manager, ExitStack()

        def forbid(name):
            def _fail(*a, **k):
                self.forbidden_calls.append(name)
                raise AssertionError(f"{name} must not run on a spent box budget")
            return _fail

        for target in _PROBE_SEAMS:
            stack.enter_context(patch(target, side_effect=forbid(target)))
        for name in _MANAGER_SEAMS:
            stack.enter_context(patch.object(mgr, name, side_effect=forbid(name)))
        real_pick, real_release = mgr._pick_instance_for, mgr._release_fastlane_claim_locked

        def pick(slot, tag, insts):
            chosen = real_pick(slot, tag, insts)
            self.picked.append(chosen)
            return chosen

        def release(slot, reason):
            if slot is self.rider:
                self.released.append(reason)
            return real_release(slot, reason)

        stack.enter_context(patch.object(mgr, "_pick_instance_for", side_effect=pick))
        stack.enter_context(patch.object(mgr, "_release_fastlane_claim_locked",
                                         side_effect=release))
        return stack

    def check_the_intended_path(self):
        """Fail loudly if the setup no longer reaches the multi-engine route by intent."""
        mgr = self.manager
        assert _wants_one_card_per_engine(
            read_manifest_cached(self.boot.storage.manifests_path, TAG)
        ), (
            "the manifest no longer asks for one card per engine, so the multi-engine "
            "branch is not entered by intent"
        )
        insts = mgr._instances_for(TAG)
        assert insts == [self.engine] and self.engine.resident_key == TAG, (
            "the tag's only engine must be keyed by the model tag, as production keys a "
            "first engine; an off-key engine would enter the branch by accident"
        )
        assert len(mgr._model_residents()) == mgr.runtime.queue.max_parallel_sidecars, (
            "the box budget must be exactly spent, or a spawn becomes possible and the "
            "route no longer has to reuse the existing engine"
        )
        assert mgr._effective_cap(TAG) == 1, (
            "with the budget spent the cap is the live engine count; a larger value "
            "means the budget function started probing cards first"
        )
        assert mgr._fastlane_claim_key(self.rider) == (CLIENT, TAG), (
            "the rider must be listed so a deferral registers a claim under its key"
        )

    def assert_nothing_probed_or_spawned(self):
        assert self.forbidden_calls == [], (
            f"a card probe, placement or spawn step ran on a spent budget: {self.forbidden_calls}"
        )
        assert self.spawn_calls == [], "no engine may be spawned while the budget is spent"

    async def close(self):
        for t in list(self.manager._bg_tasks):
            t.cancel()
        await asyncio.gather(*list(self.manager._bg_tasks), return_exceptions=True)
        await self.manager.shutdown()


async def test_a_barriered_engine_defers_the_rider_and_keeps_its_claim(tmp_path):
    """The multi-engine route picks the only engine, finds it barriered and defers.

    The rider goes through the real deferral funnel: a claim is registered under the
    rider's key and survives an unchanged registration time across a second barriered
    pop, the claim release is never called for the rider, and the engine's inbox stays
    empty throughout.
    """
    s = _Scene(tmp_path, barriered=True)
    try:
        with s.spies():
            s.check_the_intended_path()
            await s.manager._route_or_reserve(s.rider)
            key = s.manager._fastlane_claim_key(s.rider)
            assert s.picked == [s.engine], (
                "the route must reach the pick-the-engine step of the multi-engine "
                "branch, not the single-engine path or a spawn"
            )
            assert key in s.manager._fastlane_claims, (
                "deferring a listed rider must register a real claim, so the wait is "
                "visible and the rider keeps its place"
            )
            claim = s.manager._fastlane_claims[key]
            assert claim["slot"] is s.rider and claim["reason"] == "make_room_starved_vram", (
                "the claim must be the rider's own, registered by the barrier's "
                "bounded-wait deferral regime"
            )
            first_ts = claim["registered_at_monotonic"]
            assert s.engine.inbox.empty(), (
                "a barriered engine must never receive a request: it would lose "
                "disconnect handling for the whole hold"
            )

            for t in list(s.manager._bg_tasks):
                t.cancel()
            await s.manager._route_or_reserve(s.rider)
            assert s.picked == [s.engine, s.engine], "the second pop must take the same route"
            assert getattr(s.rider, "_vram_defer_count", 0) == 2, (
                "both pops must have gone through the deferral funnel, once each"
            )
            assert s.manager._fastlane_claims[key]["registered_at_monotonic"] == first_ts, (
                "a repeat deferral while barriered must not reset the claim's age, or "
                "the rider would never age toward its turn"
            )
            assert s.engine.inbox.empty(), "the engine's inbox must stay empty on a repeat pop"
            assert s.released == [], (
                f"the claim must never be released for a deferred rider, got {s.released}"
            )
            assert s.engine.state is ResidentState.IDLE_EVICTABLE, (
                "a deferred request must not wake the engine"
            )
            s.assert_nothing_probed_or_spawned()
    finally:
        await s.close()


async def test_control_an_unbarriered_engine_takes_the_rider_and_keeps_the_claim_parked_on_the_engine(tmp_path):
    """Same setup with the barrier off: the rider is inboxed and its claim is parked.

    This shows the guard above is conditional: the route reaches the same step and
    hands the request over. The claim stays registered, marked as parked on the
    engine, until the rider's own turn starts.
    """
    s = _Scene(tmp_path, barriered=False)
    try:
        with s.spies():
            s.check_the_intended_path()
            await s.manager._route_or_reserve(s.rider)
            assert s.picked == [s.engine], (
                "the route must reach the pick-the-engine step of the multi-engine branch"
            )
            assert s.engine.inbox.qsize() == 1 and s.engine.inbox.get_nowait() is s.rider, (
                "with no barrier the rider must be handed to the engine's inbox"
            )
            key = s.manager._fastlane_claim_key(s.rider)
            # a parked request keeps its claim until its own turn starts
            assert key in s.manager._fastlane_claims, (
                "a request parked in the engine's inbox must keep its claim registered"
            )
            claim = s.manager._fastlane_claims[key]
            assert claim["slot"] is s.rider and claim.get("parked_on") == s.engine.resident_key, (
                "the kept claim must be the rider's own, marked as parked on the engine "
                "whose inbox holds it"
            )
            assert s.released == [], (
                f"the hand-off must not release the rider's claim as 'admitted', got {s.released}"
            )
            assert s.engine.state is ResidentState.ACTIVE, (
                "handing a request to an idle engine must mark it active"
            )
            assert getattr(s.rider, "_vram_defer_count", 0) == 0, (
                "an admitted request must not pass through the deferral funnel"
            )
            s.assert_nothing_probed_or_spawned()
    finally:
        await s.close()
