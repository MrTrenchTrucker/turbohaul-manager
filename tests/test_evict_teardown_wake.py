"""Eviction-teardown wake test (scoped to the
VRAM-over-commit and driver-death-reap paths): the post-teardown wake for a
VRAM-credited eviction lives in `_unload_teardown`'s VRAM-CONFIRMED block
and `_late_vram_reconcile`'s slow-path twin, NOT at
`_begin_unload_locked`'s call sites -- `credit_pending=True` is used only for
the evict-MORE decision, never the actual admission check (admission guard,
manager.py: "we must not spawn onto VRAM a teardown has not yet
physically released"), so a notify fired any earlier would wake a waiter into
a re-check that is structurally guaranteed to still fail.

★ Key point: GREEN must prove ADMISSION, not merely a wake. A
notify that fires while the immediate re-check still fails is a no-op that
looks like a fix and would pass a wake-counting test. So the killing
assertion here is real end-to-end admission (`submit_and_wait` resolving with
a genuine completion), driven through the actual dispatcher, with the
VRAM-over-commit backoff forced to an absurd 3600s -- admission that fast can
only be a real wake, never a coincidental timeout (six orders of magnitude,
same discriminator technique as the sibling eviction-wake test's own
Test A/E).

`_vram_verify` is exactly the injectable `vram_fn` constructor parameter
(manager.py: `self._vram_verify = vram_fn or verify_vram_cleared`) --
the fixture's own fake_vram IS the teardown-confirm probe, not a separate
mock. `_vram_admits_locked` (the admission decision) is stubbed to key off
the SAME `teardown_confirmed` event fake_vram sets right before reporting
cleared -- modelling the real causal chain (confirm-clear -> admission
becomes possible) instead of faking the two independently.
"""
import asyncio
import time
from unittest.mock import MagicMock

import pytest

from turbohaul.manager import ResidentState, TurbohaulManager
import turbohaul.manager as manager_mod
from turbohaul.subprocess_mgr import SidecarHandle

from test_eviction_gate_wake import _boot_runtime, _resident
from _fastlane_fixture import seed_manifest

_HUGE_BACKOFF_S = 3600.0


def _mocks_confirm_gated(teardown_confirmed):
    """Real spawn/health/sigterm fakes (byte-identical shape to the sibling
    test's own `_mocks()`), but `vram_fn` -- the actual `_vram_verify` the
    fixed code calls to decide `cleared_ok` -- sets `teardown_confirmed`
    right before reporting cleared, so admission and teardown-confirm share
    one real event instead of two independently-timed fakes."""
    pid = [92_000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **kw):
        pid[0] += 1
        proc = MagicMock()
        proc.pid = pid[0]
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        # Note: a real _vram_verify does real I/O (nvidia-smi polling)
        # -- this small, deliberate delay models that, so the claimant's own
        # background _requeue_after_backoff task has a real chance to reach
        # _wait_for_park_or_timeout's .wait() and register as a waiter
        # BEFORE this fires the notify. Without it, asyncio's own task
        # scheduling can let _unload_teardown (spawned first, minimal work
        # before this call) race ahead and complete -- including the notify
        # -- before the claimant's retry task has even started running,
        # which is the classic notify-before-wait CV race
        # `_wait_for_park_or_timeout`'s own docstring already documents and
        # tolerates (a lost wake degrades to the timeout) --
        # a race that reproduces when this delay is omitted (the
        # lost wake shows up as a timeout). This delay is test
        # construction matching production's real I/O timing, not a
        # workaround for a defect in the fix.
        await asyncio.sleep(0.05)
        teardown_confirmed.set()
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return dict(
        spawn_fn=fake_spawn, health_fn=fake_health, sigterm_fn=fake_sigterm,
        vram_fn=fake_vram, complete_fn=fake_complete,
    )


@pytest.mark.asyncio
class TestVramOvercommitAdmissionWake:
    async def test_A_SUBJECT_claimant_admitted_when_evict_teardown_confirms(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, fastlane_rules=[])
        seed_manifest(boot, "claimant-model", main_gpu=0)
        teardown_confirmed = asyncio.Event()
        mgr = TurbohaulManager(boot, runtime, **_mocks_confirm_gated(teardown_confirmed))
        original_backoff = manager_mod._VRAM_DEFER_BACKOFF_S
        manager_mod._VRAM_DEFER_BACKOFF_S = _HUGE_BACKOFF_S

        victim = _resident(
            mgr, "victim-model", 59980, rank_client_meta=None, grace_seconds=1,
        )
        victim.state = ResidentState.IDLE_EVICTABLE
        victim.reserved_need_mib = 4096

        # Admission is gated on the SAME event fake_vram sets -- before
        # teardown confirms, nothing fits (forces the VRAM-over-commit
        # branch, not count-cap); after, it genuinely does.
        mgr._vram_admits_locked = lambda *a, **k: teardown_confirmed.is_set()

        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        claim_task = asyncio.create_task(
            mgr.submit_and_wait(
                "claimant-model", "hi", thread_id="c1",
                client_meta={"ip": "203.0.113.9"},
            )
        )
        try:
            t0 = time.monotonic()
            slot, result = await asyncio.wait_for(claim_task, timeout=3.0)
            elapsed = time.monotonic() - t0
            assert result.get("model") == "claimant-model", (
                f"expected a real completion for the claimant, got {result!r} "
                f"-- this must be a genuine ADMISSION, not just a wake"
            )
            assert elapsed < 2.0, (
                f"admission took {elapsed:.3f}s with backoff_s={_HUGE_BACKOFF_S} -- "
                f"anything this fast can only be a real wake, never the timeout "
                f"(six orders of magnitude apart)"
            )
            assert victim.state is ResidentState.DEAD, (
                "victim was never evicted -- fixture problem, not this test's subject"
            )
        finally:
            manager_mod._VRAM_DEFER_BACKOFF_S = original_backoff
            if not claim_task.done():
                claim_task.cancel()
                try:
                    await claim_task
                except BaseException:
                    pass
            mgr._worker_task.cancel()
            try:
                await mgr._worker_task
            except BaseException:
                pass
            await mgr.shutdown()

    async def test_B_lost_wake_still_degrades_to_periodic_backoff_never_deadlocks(
        self, tmp_path,
    ):
        """A lost wake degrades to the periodic backoff and never hangs.
        The original control for this case (a lost wake must not hang the
        claimant) is restated here for the current contract, because the
        old exhaustion-based form no longer applies. Control: if
        _vram_verify never reports cleared (a permanently-stuck card), the
        claimant must not hang FOREVER waiting on a notify that never
        comes -- the OLD backstop was "fail via the max-defers exhaustion
        path." That backstop no longer exists -- once a request
        is queued it stays queued, and the server does not cancel it
        unless the client very specifically
        cancels it -- so staying queued forever
        with no notify is now the CORRECT outcome, not the bug this
        control used to guard against.

        What the control must still prove, restated for the new contract:
        a lost wake does not DEADLOCK the retry loop. _vram_defer_count
        must keep climbing across multiple real _VRAM_DEFER_BACKOFF_S-
        spaced cycles (proving the backstop backoff timer is genuinely
        alive, not silently stuck waiting on a notify that never arrives),
        and the completion future must still be unresolved at the end of
        that window -- there is no exhaustion path left to resolve it.
        """
        boot, runtime = _boot_runtime(tmp_path, fastlane_rules=[])
        seed_manifest(boot, "claimant-model", main_gpu=0)
        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not spawn")),
            health_fn=None, sigterm_fn=None,
            vram_fn=lambda **k: (False, 999999),  # never confirms cleared
            complete_fn=None,
        )
        # No victim registered at all -- nothing is evictable, so this
        # exercises the plain starvation path (no eviction, no teardown, no
        # notify anywhere) -- proves the backoff loop still ticks on its
        # own even with zero make-room activity and no wake ever firing.
        mgr._vram_admits_locked = lambda *a, **k: False
        # Forced small on purpose: this is also the discriminator against
        # the OLD contract -- the real derived _max_vram_defers() floor-
        # clamps to 60 (_VRAM_DEFER_MIN_DEFERS), which an 8-cycle sample
        # window would never reach either way, making the control vacuous
        # (pre-fix code would pass it by never reaching exhaustion, not
        # because the fix works). Forcing it to 3 means pre-fix code
        # exhausts and fails the future WELL inside this test's own
        # sampling window, so this exact test fails against the
        # pre-fix manager.py.
        mgr._max_vram_defers = lambda: 3
        original_backoff = manager_mod._VRAM_DEFER_BACKOFF_S
        manager_mod._VRAM_DEFER_BACKOFF_S = 0.05  # small -- several real cycles in a fast test

        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            slot = await mgr.submit(
                model_tag="claimant-model", prompt="hi", thread_id="c2",
                client_meta={"ip": "203.0.113.10"}, wait_for_completion=True,
            )
            counts = []
            for _ in range(8):
                await asyncio.sleep(manager_mod._VRAM_DEFER_BACKOFF_S * 1.5)
                counts.append(getattr(slot, "_vram_defer_count", 0))
            assert counts == sorted(counts) and counts[-1] > counts[0], (
                f"the backoff loop must keep making forward progress with "
                f"no wake ever firing -- defer count samples were {counts}, "
                f"expected a monotonic climb"
            )
            assert not slot.completion_future.done(), (
                "a lost wake must degrade to periodic backoff "
                "forever now, never to the old exhaustion failure -- "
                f"future resolved after defer count reached {counts[-1]}"
            )
        finally:
            manager_mod._VRAM_DEFER_BACKOFF_S = original_backoff
            mgr._worker_task.cancel()
            try:
                await mgr._worker_task
            except BaseException:
                pass
            await mgr.shutdown()
