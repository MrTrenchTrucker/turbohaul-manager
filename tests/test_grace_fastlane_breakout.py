"""Tests for the Fast Lane grace rule: Fast Lane, when enabled,
must outrank the grace period. A same-thread continuation must NOT be
granted out of GRACE when a STRICTLY higher-priority Fast Lane request is
waiting in staging -- the ACTIVE turn already in flight is never
interrupted (this code only runs after it has already finished).

Modeled directly on the grace-starvation breakout tests (the sibling
fix for the analogous "grace loop ignores a real signal" defect).
Same three-group shape, same manager/queue construction helpers,
same discriminator-test discipline (must fail on unmodified code).

Covers all three call sites of the shared comparator:
  - TurbohaulQueue._fastlane_priority_key / _fastlane_strictly_higher /
    has_strictly_higher_priority_waiting (queue.py)
  - TurbohaulManager._process_slot's grace loop ("Loop B")
  - TurbohaulManager._serve_on_resident's grace loop ("Loop A")
"""
import asyncio
import time
from unittest.mock import MagicMock

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
from turbohaul.manager import Resident, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer, TurbohaulQueue
from turbohaul.slot import Slot, SlotState
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle


def _match(rule_index=0, rank=1, tag="main"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag=tag, rank=rank,
    )


def _boot_runtime(tmp_path, *, grace_seconds=30, fastlane_enabled=False,
                   fastlane_rules=None, max_grace_extensions=50):
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
        queue=QueueConfig(safety_enabled=False,
            grace_seconds=grace_seconds,
            max_grace_extensions=max_grace_extensions,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(
            enabled=fastlane_enabled,
            rules=fastlane_rules or [],
            max_normal_wait_s=3600.0,
        ),
    )
    return boot, runtime


def _make_fake_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 88_888
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _audit_events(boot, slot_id):
    conn = open_state_db(boot.storage.state_db_path)
    cur = conn.execute(
        "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
        (slot_id,),
    )
    events = [r["event_type"] for r in cur.fetchall()]
    conn.close()
    return events


# Two rules: "high" (rule_index=0) always outranks "low" (rule_index=1).
_HIGH_ADDR = "1.1.1.1"
_LOW_ADDR = "2.2.2.2"
_TWO_RULES = [
    FastLaneRule(address=_HIGH_ADDR, tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address=_LOW_ADDR, tag_ranks=FastLaneTagRanks(main=1)),
]


# ---------------------------------------------------------------------------
# Group 1: queue.py — the shared comparator itself
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSharedPriorityComparator:
    async def test_locked_and_wrapper_agree(self):
        """has_strictly_higher_priority_waiting() (lock-acquiring, used by
        the grace loops) and _has_strictly_higher_priority_waiting_locked()
        (usable under an already-held lock) must return the identical
        answer -- ONE source of truth, not two copies that can drift."""
        q = TurbohaulQueue(staging_max=100)
        holder = Slot.new("m1")
        holder.fastlane = _match(rule_index=1, rank=1)
        higher = Slot.new("m1")
        higher.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(higher)

        via_wrapper = await q.has_strictly_higher_priority_waiting(holder)
        async with q._lock:
            via_locked = q._has_strictly_higher_priority_waiting_locked(holder)

        assert via_wrapper is True
        assert via_locked is True

    async def test_listed_beats_unlisted(self):
        q = TurbohaulQueue(staging_max=100)
        holder = Slot.new("m1")  # holder.fastlane is None -- unlisted
        higher = Slot.new("m1")
        higher.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(higher)
        assert await q.has_strictly_higher_priority_waiting(holder) is True

    async def test_unlisted_never_strictly_higher(self):
        q = TurbohaulQueue(staging_max=100)
        holder = Slot.new("m1")
        holder.fastlane = _match(rule_index=0, rank=1)
        unlisted = Slot.new("m1")  # no rule match
        await q.enqueue(unlisted)
        assert await q.has_strictly_higher_priority_waiting(holder) is False

    async def test_lower_rule_index_wins(self):
        q = TurbohaulQueue(staging_max=100)
        holder = Slot.new("m1")
        holder.fastlane = _match(rule_index=1, rank=1)
        candidate = Slot.new("m1")
        candidate.fastlane = _match(rule_index=0, rank=1)  # earlier in the list
        await q.enqueue(candidate)
        assert await q.has_strictly_higher_priority_waiting(holder) is True

    async def test_equal_priority_does_not_bump(self):
        """The design rule: an EQUAL (rule_index, rank) is
        a real tie and must NOT count as strictly higher -- an agent's own
        follow-up turns keep the warm slot, and two equal-ranked agents
        don't ping-pong reloading context every turn."""
        q = TurbohaulQueue(staging_max=100)
        holder = Slot.new("m1")
        holder.fastlane = _match(rule_index=0, rank=2)
        equal = Slot.new("m1")
        equal.fastlane = _match(rule_index=0, rank=2)  # identical tuple
        await q.enqueue(equal)
        assert await q.has_strictly_higher_priority_waiting(holder) is False

    async def test_created_at_never_manufactures_a_difference(self):
        """created_at is deliberately excluded from the comparison -- an
        equal-priority candidate that arrived EARLIER must still not count
        as strictly higher (unlike _pick_fastlane_locked's own sort, where
        created_at breaks ties between candidates already being compared
        for the same pick -- a different concern)."""
        q = TurbohaulQueue(staging_max=100)
        holder = Slot.new("m1")
        holder.fastlane = _match(rule_index=0, rank=2)
        earlier = Slot.new("m1")
        earlier.fastlane = _match(rule_index=0, rank=2)
        earlier.created_at = time.monotonic() - 1000.0
        await q.enqueue(earlier)
        assert await q.has_strictly_higher_priority_waiting(holder) is False

    async def test_agrees_with_pick_fastlane_locked_ordering(self):
        """Refactor-equivalence proof: the comparator's notion of 'higher'
        must agree with what _pick_fastlane_locked's own candidate sort
        would actually pick first, for the same staging contents."""
        from turbohaul.queue import FastLanePopPolicy
        q = TurbohaulQueue(staging_max=100)
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)
        low = Slot.new("m1", prompt="low")
        low.fastlane = _match(rule_index=1, rank=1)
        high = Slot.new("m1", prompt="high")
        high.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(low)
        await q.enqueue(high)

        assert await q.has_strictly_higher_priority_waiting(low) is True
        assert await q.has_strictly_higher_priority_waiting(high) is False

        picked = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert picked.slot_id == high.slot_id, (
            "the comparator said `high` outranks `low`, so pop_next's own "
            "picker must agree and serve `high` first"
        )

    async def test_sees_strictly_higher_claimant_in_accept_buffer(self):
        """A strictly-higher LISTED claimant that
        overflowed into the acceptance buffer is just as real a waiter as one in
        staging, and the comparator must see it.

        The population and the scan must cover the same buffers, exactly as for the
        fairness floor: `enqueue` (in queue.py) parks a
        fresh slot in `_accept_buf` whenever staging is already full, so a claimant
        can be waiting somewhere this predicate never looked. Scanning `_staging`
        only returns False -- and False is the PERMISSIVE direction, i.e. "nobody
        outranks the holder, go ahead and admit the lower-ranked client", which is
        an inversion rather than a safe default.

        Built through the REAL producer (`enqueue` under a full staging buffer)
        rather than by appending to `_accept_buf` directly, so the test pins the
        path a request actually takes to get there.
        """
        q = TurbohaulQueue(staging_max=1)  # forces the next enqueue to overflow
        holder = Slot.new("m1")
        holder.fastlane = _match(rule_index=1, rank=1)

        filler = Slot.new("m1")  # unlisted; takes the single staging seat
        await q.enqueue(filler)
        higher = Slot.new("m1")
        higher.fastlane = _match(rule_index=0, rank=1)  # earlier rule = higher
        await q.enqueue(higher)
        assert higher in q._accept_buf, (
            "test setup assumption: `higher` must have overflowed into the "
            "acceptance buffer, otherwise this test is not exercising the gap"
        )
        assert higher not in q._staging

        assert await q.has_strictly_higher_priority_waiting(holder) is True
        async with q._lock:
            assert q._has_strictly_higher_priority_waiting_locked(holder) is True

    async def test_accept_buffer_scan_has_its_own_examination_budget(self):
        """The `examined` bound must be PER-BUFFER, not
        shared across the two buffers' concatenation -- the same per-buffer rule used for the fairness floor,
        which applies here for the same reason it applied there. A shared counter
        is spent by staging before the acceptance buffer is examined even once,
        so the claimant below goes unseen and the predicate answers False, in the
        PERMISSIVE direction.

        ⚠ THE CAPS MUST DIFFER, AND acceptance_max MUST BE THE SMALLER ONE, or
        this test does not discriminate at all. Recorded because the obvious
        setup silently does not: with staging_max=3 and the default
        acceptance_max=10000, a shared counter arrives at the acceptance buffer
        holding 3, compares it against a cap of 10000, and scans anyway -- so the
        claimant is still found and the test passes even with a shared counter, so it
        could not catch the bug it exists to catch. That version is a trap: a
        hoisted counter would pass it. Hence staging_max=5 / acceptance_max=1: a shared
        counter reaches the acceptance buffer with examined=5 against cap=1 and
        breaks immediately, while a per-buffer counter resets to 0 and finds the
        claimant.
        """
        q = TurbohaulQueue(staging_max=5, acceptance_max=1)
        holder = Slot.new("m1")
        holder.fastlane = _match(rule_index=1, rank=1)

        for _ in range(5):  # fill staging to its own bound with unlisted slots
            await q.enqueue(Slot.new("m1"))
        higher = Slot.new("m1")
        higher.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(higher)
        assert len(q._staging) == 5 and higher in q._accept_buf, (
            "test setup assumption: staging filled to its bound and the listed "
            "claimant overflowed into the acceptance buffer"
        )

        assert await q.has_strictly_higher_priority_waiting(holder) is True


# ---------------------------------------------------------------------------
# Group 2: manager.py Loop B (_process_slot, single-sidecar path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestProcessSlotGraceFastlaneBreakout:
    async def test_discriminator_grace_breaks_on_higher_priority_not_full_deadline(
        self, tmp_path
    ):
        """THE defect this fix addresses. Worker sits in GRACE for the LOW-priority
        address with no follow-up; a HIGHER-priority Fast Lane request from a
        different address is staged. The grace loop must break out well
        before grace_seconds, not hold to the full deadline.

        MUST FAIL ON UNMODIFIED CODE: before the fix, this loop only polls
        pop_matched_thread and never consults Fast Lane priority, so it holds
        to the full grace_seconds deadline every time."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            fastlane_enabled=True, fastlane_rules=_TWO_RULES,
        )

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            return True, 100

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        slot_a = await mgr.submit(
            model_tag="m1", prompt="hi", thread_id="t1",
            client_meta={"ip": _LOW_ADDR},
        )

        mgr._worker_task = asyncio.create_task(mgr.worker_loop())

        # Wait for slot_a to actually reach GRACE before the higher-priority
        # request arrives -- enqueuing it up front would let the NORMAL
        # pop_next ladder (unrelated to this fix) admit it ahead of slot_a in
        # the first place, which proves nothing about the grace loop itself.
        # This mirrors the real "trickling arrival during an existing grace
        # window" shape seen in practice, and the same ordering the
        # negative-control test below uses for its follow-up submission.
        for _ in range(40):
            if "grace_enter" in _audit_events(boot, slot_a.slot_id):
                break
            await asyncio.sleep(0.05)
        assert "grace_enter" in _audit_events(boot, slot_a.slot_id)

        slot_b = Slot.new("m1", prompt="hi", thread_id="t2")
        slot_b.fastlane = _match(rule_index=0, rank=1)  # _HIGH_ADDR's rule_index
        await mgr.queue.enqueue(slot_b)

        started = time.monotonic()
        deadline = started + grace_seconds - 0.5
        breakout_seen = False
        while time.monotonic() < deadline:
            if "grace_fastlane_breakout" in _audit_events(boot, slot_a.slot_id):
                breakout_seen = True
                break
            await asyncio.sleep(0.05)
        elapsed = time.monotonic() - started

        await mgr.shutdown()

        assert breakout_seen, (
            "grace loop never broke out on the higher-priority Fast Lane "
            "request -- held toward the full grace_seconds deadline instead"
        )
        assert elapsed < grace_seconds - 1.0, (
            f"grace exited after {elapsed:.2f}s, not meaningfully sooner than "
            f"grace_seconds={grace_seconds}s -- priority was not consulted"
        )

    async def test_invariant_no_higher_priority_grace_continuation_still_granted(
        self, tmp_path
    ):
        """★ NEGATIVE CONTROL -- must PASS both before and after this change.
        With NO higher-priority request waiting, a same-thread follow-up
        must still be granted out of GRACE via ACTIVE_MATCH exactly as
        today. This is the test that proves the change did not simply break
        the warm-slot optimization, which is the main risk of this change.
        """
        grace_seconds = 5
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            fastlane_enabled=True, fastlane_rules=_TWO_RULES,
        )

        complete_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            return True, 100

        async def fake_complete(slot, handle):
            complete_calls.append(slot.slot_id)
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        # holder is the HIGH-priority address -- nothing waiting can ever
        # outrank it, so grace must behave exactly as before this change.
        slot_a = await mgr.submit(
            model_tag="m1", prompt="hi", thread_id="t1",
            client_meta={"ip": _HIGH_ADDR},
        )

        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        for _ in range(40):
            if "grace_enter" in _audit_events(boot, slot_a.slot_id):
                break
            await asyncio.sleep(0.05)
        assert "grace_enter" in _audit_events(boot, slot_a.slot_id)

        followup = await mgr.submit(
            model_tag="m1", prompt="again", thread_id="t1",
            client_meta={"ip": _HIGH_ADDR},
        )
        for _ in range(60):
            if followup.slot_id in complete_calls:
                break
            await asyncio.sleep(0.05)

        await mgr.shutdown()

        assert followup.slot_id in complete_calls, (
            "same-thread follow-up was not served via ACTIVE_MATCH with "
            "nothing higher-priority waiting -- the warm-slot optimization "
            "regressed"
        )
        assert "grace_fastlane_breakout" not in _audit_events(boot, slot_a.slot_id)

    async def test_fastlane_disabled_control_unaffected(self, tmp_path):
        """Fast Lane OFF: behavior must be byte-for-byte what it is today,
        even with a slot present that WOULD be higher-priority if the
        feature were on."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            fastlane_enabled=False, fastlane_rules=_TWO_RULES,
        )

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            return True, 100

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        slot_a = await mgr.submit(
            model_tag="m1", prompt="hi", thread_id="t1",
            client_meta={"ip": _LOW_ADDR},
        )
        slot_b = Slot.new("m1", prompt="hi", thread_id="t2")
        slot_b.fastlane = _match(rule_index=0, rank=1)
        await mgr.queue.enqueue(slot_b)

        mgr._worker_task = asyncio.create_task(mgr.worker_loop())

        started = time.monotonic()
        # Poll for slot_a reaching POPPED (natural grace expiry) -- must NOT
        # break out early, since Fast Lane is off.
        while time.monotonic() < started + grace_seconds + 2.0:
            if "grace_starvation_breakout" in _audit_events(boot, slot_a.slot_id):
                break  # unrelated mechanism, not what we're testing
            if "grace_fastlane_breakout" in _audit_events(boot, slot_a.slot_id):
                break
            await asyncio.sleep(0.1)
        elapsed = time.monotonic() - started

        await mgr.shutdown()

        assert "grace_fastlane_breakout" not in _audit_events(boot, slot_a.slot_id), (
            "grace_fastlane_breakout fired with Fast Lane OFF -- zero "
            "behavior change contract violated"
        )
        assert elapsed >= grace_seconds - 1.0, (
            f"grace exited early ({elapsed:.2f}s) with Fast Lane OFF -- "
            "should have held to the full deadline exactly as before this change"
        )


# ---------------------------------------------------------------------------
# Group 3: manager.py Loop A (_serve_on_resident, dispatcher path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestServeOnResidentGraceFastlaneBreakout:
    async def test_non_victim_grace_runs_to_its_full_deadline_under_a_higher_ranked_claim(
        self, tmp_path
    ):
        """Loop A: a NON-VICTIM's grace is NOT cut short by a
        strictly higher-ranked Fast Lane claim -- it counts down in full.

        NOTE, and it is not optional: this test asserts that a non-victim's grace is
        NOT cut short. It replaces (rather than deletes) an earlier test that
        asserted the opposite, because the behaviour changed intentionally; a
        flipped assertion otherwise reads like a red test turned green. The
        earlier version asserted that grace DID break early and that a
        `grace_fastlane_breakout` event WAS written, which was correct for the
        earlier rule. The rule is now SCOPED to the designated victim only,
        and this loop is entered only in the
        `else` of `skip_grace`, so every client it could fire for is by
        definition NOT the designated victim. The grace rule: such a client
        is *not* evicted for the waiting request and its grace counts down
        fully. The old test was inverted, not deleted, because a test that
        breaks due to an intentional behavior change is replaced by a new one
        rather than silently dropped.

        THE FIXTURE BELOW IS REUSED, DELIBERATELY: it is *proven* to
        construct the non-victim condition, because it is the fixture that
        drove the earlier breakout. A fresh one would only claim to.

        The Fast Lane grace rule is SCOPED, not deleted: it still governs the designated
        victim through `skip_grace`, and Loop B (`_process_slot`) keeps its
        own copy of the breakout entirely.
        """
        grace_seconds = 5
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            fastlane_enabled=True, fastlane_rules=_TWO_RULES,
        )

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )

        slot_other = Slot.new("m1", prompt="hi", thread_id="t2")
        slot_other.fastlane = _match(rule_index=0, rank=1)
        await mgr.queue.enqueue(slot_other)

        handle = _make_fake_handle("m1", 59500)
        r = Resident(
            model_tag="m1",
            handle=handle,
            port=handle.port,
            grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
        )
        anchor = Slot.new("m1", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED
        anchor.fastlane = _match(rule_index=1, rank=1)  # LOW priority

        started = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(r, anchor, handle),
            timeout=grace_seconds + 2.0,
        )
        elapsed = time.monotonic() - started
        events = _audit_events(boot, anchor.slot_id)

        # NON-VACUITY, load-bearing: without these two this test would ALSO
        # pass if the anchor had been the DESIGNATED VICTIM and skipped grace
        # altogether -- green for the wrong reason, on the one path this
        # clause does not govern.
        assert "grace_enter" in events, (
            f"anchor never entered a REAL grace window -- events={events!r}"
        )
        assert "grace_designated_victim_skip" not in events, (
            "anchor must be a NON-victim here -- that is the whole clause -- "
            f"events={events!r}"
        )

        assert "grace_fastlane_breakout" not in events, (
            "a higher-ranked Fast Lane claim ended a NON-VICTIM's grace: "
            f"non-victim grace violation -- events={events!r}"
        )
        assert elapsed >= grace_seconds - 1.0, (
            f"non-victim's grace exited early ({elapsed:.2f}s of "
            f"{grace_seconds}s) -- the grace rule requires it to count down fully"
        )

        await mgr.shutdown()

    async def test_invariant_no_higher_priority_still_grants_match(self, tmp_path):
        """★ NEGATIVE CONTROL for Loop A -- must PASS both before and after.
        A same-thread follow-up is already staged and NOTHING outranks the
        anchor; ACTIVE_MATCH must still fire exactly as today."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            fastlane_enabled=True, fastlane_rules=_TWO_RULES,
        )

        complete_calls = []

        async def fake_complete(slot, handle):
            complete_calls.append(slot.slot_id)
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )

        followup = Slot.new("m1", prompt="again", thread_id="t1")
        followup.fastlane = _match(rule_index=0, rank=1)
        await mgr.queue.enqueue(followup)

        handle = _make_fake_handle("m1", 59500)
        r = Resident(
            model_tag="m1",
            handle=handle,
            port=handle.port,
            grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
        )
        anchor = Slot.new("m1", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED
        anchor.fastlane = _match(rule_index=0, rank=1)  # SAME priority as followup

        task = asyncio.create_task(mgr._serve_on_resident(r, anchor, handle))
        for _ in range(60):
            if followup.slot_id in complete_calls:
                break
            await asyncio.sleep(0.05)
        assert followup.slot_id in complete_calls, (
            "matched follow-up was not served -- the priority break-out "
            "must never preempt an available match when nothing outranks "
            "the anchor (equal priority does not bump)"
        )
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_fastlane_disabled_control_unaffected(self, tmp_path):
        """Fast Lane OFF control for Loop A -- byte-identical to today even
        with a would-be-higher-priority slot present."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            fastlane_enabled=False, fastlane_rules=_TWO_RULES,
        )

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )

        slot_other = Slot.new("m1", prompt="hi", thread_id="t2")
        slot_other.fastlane = _match(rule_index=0, rank=1)
        await mgr.queue.enqueue(slot_other)

        handle = _make_fake_handle("m1", 59500)
        r = Resident(
            model_tag="m1",
            handle=handle,
            port=handle.port,
            grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
        )
        anchor = Slot.new("m1", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED
        anchor.fastlane = _match(rule_index=1, rank=1)

        started = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(r, anchor, handle),
            timeout=grace_seconds + 2.0,
        )
        elapsed = time.monotonic() - started

        assert "grace_fastlane_breakout" not in _audit_events(boot, anchor.slot_id)
        assert elapsed >= grace_seconds - 1.0, (
            f"grace exited early ({elapsed:.2f}s) with Fast Lane OFF"
        )

        await mgr.shutdown()
