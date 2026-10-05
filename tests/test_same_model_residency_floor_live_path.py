"""Port the same-model residency floor onto the LIVE path.

The floor already exists at `LiveSlotsPoller`-era `_process_slot` (manager.py, the
`_SAME_MODEL_QUEUED_HOLD_S` block) and is DEAD: `worker_loop` returns
`_dispatch_loop()` unconditionally (`config.py` pins `max_parallel_sidecars` to
`ge=1`), so `_process_slot` is unreachable at every legal cap. Its live twin is
`_drive_resident`'s post-turn idle decision:

    idle_window = self._idle_window_seconds(r.latest_keep_alive_s, per_model_idle)
    async with self._registry_lock:
        ...
        if idle_window <= 0 or self._is_designated_unload_target_locked(r):
            <immediate evict>
        <park IDLE_EVICTABLE for idle_window>

When `idle_window <= 0` (keep_alive=0 / zero per-model idle timeout) the resident is
torn down the instant its turn ends, so a request for the SAME model that is ALREADY
QUEUED pays a full teardown + respawn. That is the gap this change closes.

The rank-priority rule GOVERNS: the hold applies
ONLY IF no strictly-higher-ranked client is already waiting -- otherwise holding the
resident warm would occupy capacity a higher-ranked waiter needs, which is exactly
what "a lower-ranked client is not admitted ahead of a higher-ranked client that is
already waiting" forbids.

⚠ WHY THE OBVIOUS PORT IS INERT, and why these tests are shaped the way they are:
`_active_handle` is None forever on the live path (its only real writers are inside
the dead `_process_slot`), so a port that gates on it compiles, reviews clean, and
never fires. A test that merely asserts "the hold code exists" would pass against
that inert port. Every discriminator below therefore asserts an OBSERVABLE FSM
OUTCOME that is impossible without the new branch actually executing: at
`idle_window == 0` nothing else in the manager can produce IDLE_EVICTABLE.

Harness shape is lifted from the idle-unload-timer test -- drives
`_drive_resident` DIRECTLY, no dispatcher/worker_loop machinery, `safety_enabled`
False and no manifest on disk so no process is ever launched. That file already owns
this exact branch (an earlier change widened it), so this is the established harness for it,
not one invented here.
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
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle

GRACE_S = 1


def _boot_runtime(tmp_path, *, idle_hot_load_seconds, fastlane_rules=None):
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
            default_port_base=59960,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, max_parallel_sidecars=2,
            grace_seconds=GRACE_S, max_grace_extensions=50,
            idle_hot_load_seconds=idle_hot_load_seconds,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=fastlane_rules or []),
    )
    return boot, runtime


def _fake_handle(model_tag, port, pid):
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _mocks():
    pid = [91_000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **kw):
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True}

    return dict(
        spawn_fn=fake_spawn, health_fn=fake_health, sigterm_fn=fake_sigterm,
        vram_fn=fake_vram, complete_fn=fake_complete,
    )


def _listed(rule_index, rank, address="9.9.9.9"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address=address, label="",
        effective_tag="main", rank=rank,
    )


def _resident(mgr, model_tag, port):
    r = Resident(
        model_tag=model_tag, resident_key=model_tag, port=port,
        grace=GraceTimer(grace_seconds=GRACE_S, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=0),
        rank_client_meta=None,
        last_active_monotonic=time.monotonic(),
    )
    r.inbox = asyncio.Queue()
    mgr._residents[model_tag] = r
    return r


async def _drive_one_turn_and_classify(mgr, r, model_tag, *, timeout_s):
    """Put one anchor slot in r.inbox, run _drive_resident directly, and report
    the FIRST terminal outcome: 'evicted' (deregistered) or 'idle_evictable'
    (parked warm). Cancels the driver either way -- one turn's outcome is all
    this file needs, never the long-lived loop.

    The anchor is deliberately UNLISTED (no FastLaneMatch), so it is the
    `holder` against which `has_strictly_higher_priority_waiting` is compared:
    per queue.py's comparator an unlisted holder is outranked by ANY listed
    waiter, and ties/unlisted candidates are NOT strictly higher.
    """
    anchor = Slot.new(model_tag, prompt="hi", thread_id="t1")
    anchor.state = SlotState.STAGED
    await r.inbox.put(anchor)

    drive_task = asyncio.create_task(mgr._drive_resident(r))
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            cur = mgr._residents.get(model_tag)
            if cur is None:
                return "evicted"
            if cur.state is ResidentState.IDLE_EVICTABLE:
                return "idle_evictable"
            await asyncio.sleep(0.05)
        return "timeout"
    finally:
        drive_task.cancel()
        try:
            await drive_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
class TestSameModelResidencyFloorOnLivePath:
    """`idle_hot_load_seconds=0` forces `idle_window == 0` for every resident
    here, so the UNMODIFIED code takes the immediate-evict arm every time. Any
    'idle_evictable' outcome below is therefore attributable to the new floor
    and to nothing else -- that is what makes these discriminators, not
    assertions.
    """

    async def test_discriminator_same_model_queued_holds_resident_warm(self, tmp_path):
        """★ MUST FAIL on unmodified code (RED on the right reason).

        A same-model request is ALREADY QUEUED and no higher-ranked client is
        waiting. The resident must be held warm (parked IDLE_EVICTABLE for the
        bounded `_SAME_MODEL_QUEUED_HOLD_S` window) instead of torn down, so the
        queued request warm-inherits rather than paying teardown + respawn.

        Pre-fix this returns 'evicted' -- the floor is dead in `_process_slot`
        and nothing consults it here.
        """
        boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=0)
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        r = _resident(mgr, "held-model", 59961)
        r.state = ResidentState.ACTIVE

        # The NEXT queued request is the SAME model, and it is unlisted -- so it
        # is not strictly higher than the (also unlisted) holder. The rank-priority rule permits
        # the hold.
        queued = Slot.new("held-model", prompt="next", thread_id="t2")
        await mgr.queue.enqueue(queued)
        assert await mgr.queue.head_model_tag() == "held-model"

        try:
            outcome = await _drive_one_turn_and_classify(
                mgr, r, "held-model", timeout_s=GRACE_S + 4.0,
            )
            assert outcome == "idle_evictable", (
                "a resident whose NEXT queued request is the SAME model must be "
                "held warm for the bounded same-model window instead of being "
                "torn down at keep_alive=0 -- otherwise the queued request pays "
                f"a full teardown + respawn. outcome={outcome!r}"
            )
            assert r.idle_expires_at is not None, (
                "the held resident must carry a real bounded idle deadline, not "
                "an open-ended pin"
            )
        finally:
            await mgr.shutdown()

    async def test_discriminator_call_site_actually_executes(self, tmp_path):
        """★ MUST FAIL on unmodified code. Wiring proof: the
        test must prove THIS call site executes -- there are ZERO live
        callers of the rank predicate at this base, so nothing existing can
        vouch for the wiring.

        Wraps the named helper and asserts `_drive_resident` actually invoked it
        on the live path. Fails with a clear message (not an AttributeError
        cascade) if the helper was never added or never called.
        """
        boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=0)
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        r = _resident(mgr, "wired-model", 59962)
        r.state = ResidentState.ACTIVE

        queued = Slot.new("wired-model", prompt="next", thread_id="t2")
        await mgr.queue.enqueue(queued)

        real = getattr(mgr, "_same_model_residency_floor", None)
        assert real is not None, (
            "_same_model_residency_floor is not wired onto the manager -- the "
            "residency floor has not been ported to the live path at all"
        )

        calls = []

        async def spy(resident, window):
            calls.append((resident.model_tag, window))
            return await real(resident, window)

        mgr._same_model_residency_floor = spy

        try:
            await _drive_one_turn_and_classify(
                mgr, r, "wired-model", timeout_s=GRACE_S + 4.0,
            )
            assert calls, (
                "_drive_resident never called _same_model_residency_floor -- the "
                "helper exists but its call site is inert, which is precisely the "
                "failure mode this test exists to avoid"
            )
            assert calls[0][0] == "wired-model"
            assert calls[0][1] <= 0, (
                "the floor must only be consulted on the keep_alive=0 / "
                f"immediate-teardown edge, got window={calls[0][1]!r}"
            )
        finally:
            await mgr.shutdown()

    async def test_GREEN_CONTROL_higher_ranked_waiter_suppresses_the_hold(self, tmp_path):
        """★ RANK-PRIORITY GUARD -- must PASS before AND after the fix.

        The same model is queued at the HEAD, but a strictly-higher-ranked
        client is also waiting. Holding the resident warm would occupy capacity
        that higher-ranked waiter needs, so the floor must NOT fire and the
        resident must still be torn down.

        This is the arm that a naive port passes by accident and a rank-priority-violating
        port fails: it is the difference between "same model queued" and "same
        model queued AND nobody outranks it".
        """
        rules = [FastLaneRule(address="9.9.9.9", tag_ranks=FastLaneTagRanks(main=1))]
        boot, runtime = _boot_runtime(
            tmp_path, idle_hot_load_seconds=0, fastlane_rules=rules,
        )
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        r = _resident(mgr, "outranked-model", 59963)
        r.state = ResidentState.ACTIVE

        # HEAD is the same model (so the same-model condition is satisfied)...
        same_model = Slot.new("outranked-model", prompt="next", thread_id="t2")
        await mgr.queue.enqueue(same_model)
        # ...but a LISTED, strictly-higher-ranked client for a DIFFERENT model is
        # already waiting behind it. An unlisted holder is outranked by any
        # listed waiter (queue.py's comparator).
        higher = Slot.new("other-model", prompt="urgent", thread_id="t3")
        higher.fastlane = _listed(rule_index=0, rank=1)
        await mgr.queue.enqueue(higher)

        assert await mgr.queue.head_model_tag() == "outranked-model"

        try:
            outcome = await _drive_one_turn_and_classify(
                mgr, r, "outranked-model", timeout_s=GRACE_S + 4.0,
            )
            assert outcome == "evicted", (
                "The rank-priority rule governs: the same-model hold must NOT fire while a "
                "strictly-higher-ranked client is already waiting, otherwise the "
                "resident keeps capacity the higher-ranked client needs. "
                f"outcome={outcome!r}"
            )
        finally:
            await mgr.shutdown()

    async def test_higher_ranked_waiter_in_accept_buf_is_seen(self, tmp_path):
        """★ REAL CONTROL.

        A strictly-higher-ranked client waiting in the ACCEPTANCE BUFFER **is
        seen** by the rank predicate, so the same-model residency floor does NOT
        hold the resident and the higher-ranked waiter is not made to wait behind
        a lower-ranked same-model request.

        ⚠ BACKGROUND -- the assertion body below asserts the CORRECT behaviour
        (`evicted`) and is a live control, not an expected-failure marker.
        What it guards:
          * The predicate `queue._has_strictly_higher_priority_waiting_locked`
            scans BOTH `_staging` and `_accept_buf` with a PER-BUFFER budget.
          * A predicate that scanned `_staging` ONLY would miss a waiter in
            `_accept_buf` and answer False -- the PERMISSIVE direction -- so
            the floor would hold when the rank-visibility rule says it must
            not.
          * The floor CONSUMES that predicate and does not redefine it;
            widening the scan is a separate concern.
          * An expected-failure marker would be the wrong tool here: a strict
            xfail turns an unexpected pass into a run failure, so it cannot
            guard behaviour that is meant to stay correct over time, and it
            would fail the run as soon as the scan was widened.
          * It covers the case a narrow-scan predicate gets wrong: a
            strictly-higher-ranked waiter that sits in `_accept_buf` rather
            than `_staging`.
          * The control is asserted directly (not behind an xfail marker), so
            a regression in the scan shows up as an ordinary test failure.

        ⚠ SCOPE, retained deliberately: this is the OVERFLOW regime (rank-visibility rule), NOT
        ordinary load and NOT a rank-priority re-admission defect. There is exactly ONE
        producer of `_accept_buf` entries -- `queue.enqueue` -- reached only on
        the `len(self._staging) >= self.staging_max` path, i.e. once staging is
        FULL (default cap 100). This control covers the population that
        widening exists to serve.

        ◊ RESIDUE, DOCUMENTED AND DELIBERATELY NOT ASSERTED (by design):
        the widened scan is still BOUNDED per buffer, so a
        strictly-higher claimant sitting BEYOND either cap is still unseen and
        that residue still fails permissive -- see the predicate's own docstring.
        No assertion is written on it here, on purpose. A strict xfail is
        SELF-LIQUIDATING: it destroys itself when the defect closes, which is
        what suits it to a defect that is meant to close. An assertion that a limitation
        PERSISTS has no discharge condition -- it would calcify the bound and go
        red if anyone ever improved it, and a guard that cannot express intended
        divergence trains people to override it. If the residue is ever to be
        governed, that needs a decision first and then its own deliverable.
        """
        rules = [FastLaneRule(address="9.9.9.9", tag_ranks=FastLaneTagRanks(main=1))]
        boot, runtime = _boot_runtime(
            tmp_path, idle_hot_load_seconds=0, fastlane_rules=rules,
        )
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        r = _resident(mgr, "acceptbuf-model", 59966)
        r.state = ResidentState.ACTIVE

        same_model = Slot.new("acceptbuf-model", prompt="next", thread_id="t2")
        await mgr.queue.enqueue(same_model)

        # The higher-ranked waiter is in the ACCEPTANCE BUFFER -- the same
        # placement queue.enqueue itself produces once staging is full, and the
        # buffer the rank predicate did not scan before the widening.
        higher = Slot.new("other-model", prompt="urgent", thread_id="t3")
        higher.fastlane = _listed(rule_index=0, rank=1)
        async with mgr.queue._lock:
            mgr.queue._accept_buf.append(higher)

        assert await mgr.queue.head_model_tag() == "acceptbuf-model"

        # SUBJECT, asserted directly: the rank predicate must SEE the waiter
        # sitting in _accept_buf. `holder` is unlisted, mirroring what the floor
        # actually passes (r.grace_tip, which carries no FastLaneMatch in this
        # fixture) -- and per queue.py's comparator any LISTED waiter strictly
        # outranks an unlisted holder. Before the widening this answered False.
        holder = Slot.new("acceptbuf-model", prompt="holder", thread_id="t1")
        assert await mgr.queue.has_strictly_higher_priority_waiting(holder) is True, (
            "the rank predicate must see a strictly-higher-ranked LISTED waiter "
            "in _accept_buf; answering False is the PERMISSIVE direction and is "
            "the rank-visibility inversion the widening closed"
        )

        try:
            outcome = await _drive_one_turn_and_classify(
                mgr, r, "acceptbuf-model", timeout_s=GRACE_S + 4.0,
            )
            assert outcome == "evicted", (
                "The rank-priority rule governs regardless of WHICH queue buffer the "
                "higher-ranked waiter is sitting in -- the same-model hold must "
                f"not fire while it waits. outcome={outcome!r}"
            )
        finally:
            await mgr.shutdown()

    async def test_GREEN_CONTROL_no_same_model_queued_still_evicts(self, tmp_path):
        """★ CONTROL -- must PASS before AND after the fix. Nothing queued for
        this model, so the floor has no reason to fire and the keep_alive=0
        teardown must be completely unchanged. Proves the new branch does not
        over-fire for every idle resident.
        """
        boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=0)
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        r = _resident(mgr, "lonely-model", 59964)
        r.state = ResidentState.ACTIVE

        other = Slot.new("different-model", prompt="next", thread_id="t2")
        await mgr.queue.enqueue(other)
        assert await mgr.queue.head_model_tag() == "different-model"

        try:
            outcome = await _drive_one_turn_and_classify(
                mgr, r, "lonely-model", timeout_s=GRACE_S + 4.0,
            )
            assert outcome == "evicted", (
                "with no same-model request queued the keep_alive=0 teardown "
                f"must be unchanged -- outcome={outcome!r}"
            )
        finally:
            await mgr.shutdown()

    async def test_GREEN_CONTROL_ordinary_idle_window_unaffected(self, tmp_path):
        """★ CONTROL -- must PASS before AND after the fix. A resident with a
        normal (non-zero) idle window parks IDLE_EVICTABLE exactly as before;
        the floor is scoped to the `idle_window <= 0` edge and must not touch
        the ordinary path.
        """
        boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=120)
        mgr = TurbohaulManager(boot, runtime, **_mocks())
        r = _resident(mgr, "normal-model", 59965)
        r.state = ResidentState.ACTIVE

        try:
            outcome = await _drive_one_turn_and_classify(
                mgr, r, "normal-model", timeout_s=GRACE_S + 4.0,
            )
            assert outcome == "idle_evictable", (
                f"an ordinary idle window must be untouched -- outcome={outcome!r}"
            )
        finally:
            await mgr.shutdown()
