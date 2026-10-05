"""Fast Lane priority-inversion bug -- claim must be
visible at staging arrival, not only at failed-route time.

The bug: ``_register_fastlane_claim_locked`` has exactly ONE production
caller -- ``_defer_unroutable`` -- which fires only after ``pop_next`` has
picked a slot and ``_route_or_reserve`` determined it cannot be routed.
If a genuinely higher-priority request sits in the staging queue behind a
resident that is busy looping ``pop_matched_thread`` (same-thread grace
follow-ups), that request never gets popped, never reaches
``_defer_unroutable``, never gets a claim, and ``_governing_claim_locked``
returns None -- so ``_is_designated_unload_target_locked`` returns False
forever regardless of wait time (an unbounded stall).

The fix: register a fastlane claim the moment a
fastlane-eligible request LANDS in staging (the ``enqueue`` path in
``submit``), not only later at a failed route. This makes the
governing-claim mechanism cover the true waiting population without
touching the grace loop (no risk of reintroducing the earlier
over-firing on non-victims).

RED at unfixed code: ``submit()`` staggers a fastlane-eligible request
into staging but never calls ``_defer_unroutable`` (no routing attempt,
no failed route) -- so ``_governing_claim_locked`` returns None.
GREEN with the fix: ``submit()`` calls ``_register_staged_claim``
after staging-arrival enqueue, and the claim is visible immediately.
"""
import asyncio

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
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import SlotState
from turbohaul.state import state_db_session

_RULES = [
    FastLaneRule(address="10.0.0.5", tag_ranks=FastLaneTagRanks(main=1)),
]


@pytest.fixture
def mgr(tmp_path):
    """A real TurbohaulManager with fastlane enabled, cap=1.

    cap=1 is used because the staging-arrival claim registration lives in
    ``submit()`` itself -- before any dispatch loop runs -- so we need
    only the ``submit`` code path, not the full _route_or_reserve
    machinery. The dispatch loop is never started (stop_event is set);
    we are testing the CLAIM-REGISTRATION gate, not the routing engine.
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
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(max_parallel_sidecars=1, safety_enabled=False),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=_RULES),
    )
    m = TurbohaulManager(boot, runtime)
    # Pre-create the schema so state_db_session doesn't race on first WAL.
    with state_db_session(m.boot.storage.state_db_path):
        pass
    # _stop_event is left UNSET so submit() does not raise QueueClosed.
    # The dispatch loop (worker_loop) is not started by __init__ -- it is
    # spawned externally -- so submit() -> enqueue lands the slot in
    # staging without any routing attempt.
    return m


async def _drain_bg(mgr):
    """Await background tasks spawned by claim registration / release."""
    tasks = [t for t in list(mgr._bg_tasks) if not t.done()]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# RED: unfixed code -- a staged fastlane request is invisible
# ---------------------------------------------------------------------------

class TestStagedClaimVisibility:
    """The core regression test for staged-claim visibility.

    A fastlane-eligible request that has landed in staging (via submit ->
    enqueue) but has NOT been popped by pop_next (and therefore has NOT
    reached _defer_unroutable) must STILL produce a governing claim.

    Before the fix: _governing_claim_locked returns None (no claim).
    After the fix: _governing_claim_locked returns the staged slot.
    """

    @pytest.mark.asyncio
    async def test_staged_fastlane_request_is_visible_to_governing_claim(self, mgr):
        """RED at unfixed code: submit() staggers the request but never
        registers a claim (no deferred/miss path reached). GREEN with fix
        (staging_arrival claim registered immediately after enqueue)."""
        slot = await mgr.submit(
            model_tag="m_high",
            prompt="hi",
            thread_id="high-priority-thread",
            client_meta={"ip": "10.0.0.5"},
        )
        await _drain_bg(mgr)

        # The slot is now in staging (not yet popped by any dispatch loop).
        assert slot.state is SlotState.STAGED, (
            "slot should be staged, not yet routed (dispatch loop is stopped)"
        )

        # WITHOUT the fix: _governing_claim_locked returns None here,
        # because the claim is only registered at _defer_unroutable
        # (a failed route), and no route has been attempted yet.
        # WITH the fix: the staging_arrival claim makes it visible.
        governing = mgr._governing_claim_locked()
        assert governing is not None, (
            "FASTLANE INVISION: a fastlane-eligible request "
            "staged in the queue must be visible to _governing_claim_locked "
            "immediately -- otherwise _is_designated_unload_target_locked "
            "returns False forever while the request waits behind a "
            "same-thread grace loop, causing long priority inversions"
        )
        _, claim_slot = governing
        assert claim_slot is slot, (
            "governing claim must point at the staged fastlane request's slot"
        )

    @pytest.mark.asyncio
    async def test_non_fastlane_request_stays_invisible_at_staging(self, mgr):
        """A slot that does NOT match a fastlane rule must NOT register a
        claim at staging arrival -- the duty is scoped to the high-priority
        waiting population, not every routed request. This is the negative
        control that proves the staging_arrival gate is fastlane-scoped."""
        slot = await mgr.submit(
            model_tag="m_ordinary",
            prompt="hi",
            thread_id="ordinary-thread",
            client_meta={"ip": "10.0.0.9"},  # not in _RULES
        )
        await _drain_bg(mgr)

        assert slot.state is SlotState.STAGED
        assert mgr._fastlane_claims == {}, (
            "non-fastlane-eligible slot must not register a claim"
        )
        assert mgr._governing_claim_locked() is None, (
            "non-fastlane request must be invisible to the governing claim"
        )

    @pytest.mark.asyncio
    async def test_staged_claim_released_on_admission(self, mgr):
        """When the staged slot is admitted by a HIT route, the
        staging-arrival claim must be released -- otherwise we trade an
        invisible claimant for a permanent ghost. The existing 'admitted'
        release path handles this; staging-arrival must not create a
        second un-released claim."""
        slot = await mgr.submit(
            model_tag="m_high",
            prompt="hi",
            thread_id="high-priority-thread-2",
            client_meta={"ip": "10.0.0.5"},
        )
        await _drain_bg(mgr)

        assert mgr._governing_claim_locked() is not None

        # Simulate admission: _release_fastlane_claim_locked(slot, "admitted")
        # is what _route_to / _route_or_reserve's HIT paths already call.
        async with mgr._registry_lock:
            mgr._release_fastlane_claim_locked(slot, "admitted")
        await _drain_bg(mgr)

        assert mgr._governing_claim_locked() is None, (
            "staging-arrival claim must be released when the slot is "
            "admitted -- no ghost entry left behind"
        )

    @pytest.mark.asyncio
    async def test_repeat_staging_arrival_noop_refresh(self, mgr):
        """Dedup: re-registering the same key (e.g. the slot is
        re-enqueued at the staging head during a grace requeue) must be a
        no-op refresh -- preserving the original registered_at timestamp so
        waited_s keeps growing rather than resetting."""
        slot = await mgr.submit(
            model_tag="m_high",
            prompt="hi",
            thread_id="high-priority-thread-3",
            client_meta={"ip": "10.0.0.5"},
        )
        await _drain_bg(mgr)

        key = mgr._fastlane_claim_key(slot)
        assert key is not None
        first_ts = mgr._fastlane_claims[key]["registered_at_monotonic"]

        # Simulate the slot being re-enqueued (grace requeue at head) --
        # _register_fastlane_claim_locked's dedup makes this a no-op.
        async with mgr._registry_lock:
            mgr._register_fastlane_claim_locked(slot, "staging_arrival")
        await _drain_bg(mgr)

        assert mgr._fastlane_claims[key]["registered_at_monotonic"] == first_ts, (
            "repeat staging-arrival registration must not reset registered_at "
            "(dedup)"
        )
