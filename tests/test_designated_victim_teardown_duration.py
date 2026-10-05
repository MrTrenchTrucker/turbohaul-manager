"""The DURATION half of the designated-victim rule's "immediately" claim.

The designated-victim rule, quoted: "The instant its
turn's last byte is streamed ... its context is saved and the slot is torn down
IMMEDIATELY, and the waiting higher-priority client is loaded." "Immediately" is
a temporal claim. Every existing test that touches this behavior -- this repo's
own acceptance tests
and the idle-unload-timer discriminator test
-- assert only the eventual OUTCOME (did eviction happen before some generous
timeout / which branch did the code take), never the DURATION between turn-
complete and teardown. A scenario with an actively-polling contender (the one in the
acceptance tests, driven via submit_and_wait) makes eviction fast via a SEPARATE,
already-correct mechanism (_lru_idle_unloadable, the admission-time reclaim
picker invoked whenever room is needed -- in manager.py) regardless of how
quickly or slowly the designated-victim branch itself fires. That is why the
existing tests' contended arms pass unmodified: the contender's live presence is constitutive of
what they prove, so they cannot see this gap no matter how they're tuned. This
test removes that confound -- a claim is registered but nothing actively drives
a fresh admission attempt -- isolating _drive_resident's OWN post-turn decision,
the exact branch the immediate-teardown fix
touches (in manager.py, before the fix).

Direct-drives _drive_resident, same minimal harness shape as the
idle-unload-timer discriminator test -- the victim/survivor/claimant
construction and _register_fastlane_claim_locked usage mirror that file
closely, extended here to measure timing rather than classify an eventual
branch.

=== ASSERTION SHAPE: ORDER-RELATIVE, not a margin ===
Principle: a DURATION assertion is the one assertion class that can be BOTH flaky
AND vacuous in the same line, so prefer ORDER-RELATIVE checks where possible (eviction event
precedes window-expiry event) rather than a wall-clock margin: a contract
between two clocks, not a guess at a margin.

The primary assertion compares the REAL eviction timestamp against
`turn_complete_at + idle_window`, where BOTH inputs are captured by spying on
the exact production calls (_serve_on_resident's return, _idle_window_seconds's
call+return) -- the same two clocks manager.py itself would combine into
`r.idle_expires_at`, had this branch taken the path that sets it (it does not,
on the fix -- the immediate-evict branch never touches that field, which is
why this test computes the equivalent value from the code's own inputs rather
than reading it off the Resident object).

WHY THIS IS STRUCTURALLY ROBUST, NOT A CLOSE RACE: `asyncio.wait_for(fut,
timeout=idle_window)` cannot resolve -- by completion or by timeout -- before
`idle_window` real seconds have elapsed from ITS OWN start, which is always
AT OR AFTER `idle_window` was computed (a few synchronous statements, and one
bounded await, sit between them -- in manager.py). So on unpatched
code, eviction is GUARANTEED to land at or after `idle_window_computed_at +
idle_window` -- never before -- by asyncio's own timeout contract, not by
probability. This is what makes the comparison immune to CI/machine-speed
jitter regardless of how small idle_window is: the ordering is enforced by
control flow, not by out-racing a clock. On the fix, eviction happens within
the same locked step idle_window is computed in (ms-scale), landing far
before that boundary. idle_hot_load_seconds is kept at a modest, single-digit
value (this suite family's own convention for grace_seconds) purely so the
test runs fast -- the order-relative property does not depend on the window's
absolute size, only on wait_for's timeout contract, which holds at any scale.

A secondary, belt-and-braces bound is kept (if it costs
nothing, place it in the MIDDLE of the two regimes, never at either edge): half the
configured idle window, itself read from the same spy, not a separate
hand-picked constant.

Built for a single run: ONE file, no edits between runs -- pure
observation and comparison, no branching on which regime is expected. RED on
the unfixed code, GREEN on the fixed code.
"""
import asyncio
import time
from unittest.mock import MagicMock, patch

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

IDLE_WINDOW_S = 4.0


def _boot_runtime(tmp_path, *, grace_seconds=5, idle_hot_load_seconds=IDLE_WINDOW_S, fastlane_rules=None):
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
            grace_seconds=grace_seconds, max_grace_extensions=50,
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
    pid = [89_000]

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


_RULES = [FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1))]


def _resident(mgr, model_tag, port, *, rank_client_meta, grace_seconds):
    r = Resident(
        model_tag=model_tag, resident_key=model_tag, port=port,
        grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=0),
        rank_client_meta=rank_client_meta,
        last_active_monotonic=time.monotonic(),
    )
    r.inbox = asyncio.Queue()
    mgr._residents[model_tag] = r
    return r


async def _wait_for(cond, deadline, fail_msg):
    while True:
        if cond():
            return
        if time.monotonic() > deadline:
            pytest.fail(fail_msg)
        await asyncio.sleep(0.01)


def _make_serve_spy(orig_method, holder):
    """Wraps TurbohaulManager._serve_on_resident to timestamp the EXACT
    instant it returns -- 'turn complete' by the same definition production
    code uses (_drive_resident's own next line, in manager.py, computes
    idle_window immediately after this call returns). A poll-based
    alternative (watching r.active_slot flip busy->free) was tried first and
    is WRONG here: with instant fake mocks, an entire turn -- start, serve,
    complete -- can run inside a single poll gap, so the 'busy' state is
    never observed and the poll times out claiming the turn 'never started'
    even though it ran and finished. Wrapping the call itself has no such
    race: the timestamp is taken from inside the exact coroutine boundary,
    not inferred from state polled at arbitrary points in time.

    A genuine `async def` (not a callable class instance) is required here:
    set as a class attribute, a function is a descriptor and auto-binds
    `mgr_self` on `mgr._serve_on_resident(...)` the normal way; a callable
    object is not a descriptor and would NOT receive `mgr` as its first
    positional argument at all -- caught empirically on the first attempt
    (a callable-class version silently mis-bound arguments).
    """
    async def _spy(mgr_self, r, slot, handle):
        result = await orig_method(mgr_self, r, slot, handle)
        holder["turn_complete_at"] = time.monotonic()
        return result
    return _spy


def _make_idle_window_spy(orig_method, holder):
    """Wraps TurbohaulManager._idle_window_seconds to capture the exact
    moment it's called (in manager.py, unconditionally, immediately after
    _serve_on_resident returns -- fires on BOTH regimes, unlike the
    r.idle_expires_at field assignment a few lines later which only happens
    on the non-immediate branch) and its returned value. Together these are
    the same two inputs production code itself would combine into
    r.idle_expires_at (in manager.py) -- capturing them here gives the
    order-relative assertion its comparison basis from the code's own
    computation, without depending on which branch actually sets the field.
    """
    def _spy(mgr_self, keep_alive_s, default):
        called_at = time.monotonic()
        result = orig_method(mgr_self, keep_alive_s, default)
        holder["idle_window_computed_at"] = called_at
        holder["idle_window_s"] = result
        return result
    return _spy


async def _drive_one_turn_and_measure(mgr, r, model_tag, *, ceiling_s):
    """Puts one anchor slot in r.inbox, runs _drive_resident directly with
    _serve_on_resident and _idle_window_seconds spied (see the two spy
    factories above). Returns a dict with:
      outcome: 'evicted' or 'idle_evictable' (never reached teardown in
        ceiling_s -- itself the informative signal on unpatched code, if
        idle_window is set larger than this ceiling anticipates).
      eviction_at: real timestamp the resident was deregistered, or None.
      expected_idle_expires_at: idle_window_computed_at + idle_window_s --
        the order-relative comparison basis, computed from the code's own
        observed inputs.
      turn_complete_at, idle_window_s: raw components, for diagnostics.
    """
    anchor = Slot.new(model_tag, prompt="hi", thread_id="t1")
    anchor.state = SlotState.STAGED
    await r.inbox.put(anchor)

    holder = {"turn_complete_at": None, "idle_window_computed_at": None, "idle_window_s": None}
    serve_spy = _make_serve_spy(type(mgr)._serve_on_resident, holder)
    idle_spy = _make_idle_window_spy(type(mgr)._idle_window_seconds, holder)
    with patch.object(type(mgr), "_serve_on_resident", serve_spy), \
         patch.object(type(mgr), "_idle_window_seconds", idle_spy):
        drive_task = asyncio.create_task(mgr._drive_resident(r))
        try:
            deadline = time.monotonic() + ceiling_s
            await _wait_for(
                lambda: holder["idle_window_computed_at"] is not None, deadline,
                "turn never completed within the ceiling -- harness bug, not the assertion under test",
            )
            expected_idle_expires_at = holder["idle_window_computed_at"] + holder["idle_window_s"]

            deadline = holder["turn_complete_at"] + ceiling_s
            while time.monotonic() < deadline:
                cur = mgr._residents.get(model_tag)
                if cur is None:
                    return {
                        "outcome": "evicted",
                        "eviction_at": time.monotonic(),
                        "expected_idle_expires_at": expected_idle_expires_at,
                        "turn_complete_at": holder["turn_complete_at"],
                        "idle_window_s": holder["idle_window_s"],
                    }
                await asyncio.sleep(0.01)
            return {
                "outcome": "idle_evictable",
                "eviction_at": None,
                "expected_idle_expires_at": expected_idle_expires_at,
                "turn_complete_at": holder["turn_complete_at"],
                "idle_window_s": holder["idle_window_s"],
            }
        finally:
            drive_task.cancel()
            try:
                await drive_task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
class TestDesignatedVictimTeardownDuration:
    async def test_DISCRIMINATOR_victim_evicted_before_its_own_idle_expiry(self, tmp_path):
        """MUST FAIL on the unfixed code: a designated victim's eviction
        lands AT OR AFTER its own ordinary idle_expires_at -- it waits out
        `asyncio.wait_for(r.inbox.get(), timeout=idle_window)` inside
        _drive_resident before an idle-timeout eviction fires, structurally
        unable to land any earlier (wait_for cannot resolve before its own
        timeout). After the fix: eviction happens in the SAME
        locked step idle_window is computed in, ms-scale, landing far before
        that boundary -- via _begin_unload_locked instead."""
        grace_seconds = 5
        _rules = [
            FastLaneRule(address="9.9.9.9", tag_ranks=FastLaneTagRanks(main=1)),
            FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1)),
        ]
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds, fastlane_rules=_rules)
        mgr = TurbohaulManager(boot, runtime, **_mocks())

        victim = _resident(
            mgr, "victim-model", 59961,
            rank_client_meta={"ip": "1.2.3.4"}, grace_seconds=grace_seconds,
        )
        victim.state = ResidentState.ACTIVE
        victim.last_active_monotonic = 1.0
        survivor = _resident(
            mgr, "survivor-model", 59962,
            rank_client_meta={"ip": "1.2.3.4"}, grace_seconds=grace_seconds,
        )
        survivor.state = ResidentState.ACTIVE
        survivor.last_active_monotonic = 1000.0

        claimant = Slot.new("claimant-model", prompt="hi", thread_id="claim")
        claimant.fastlane = FastLaneMatch(
            rule_index=0, raw_address="9.9.9.9", label="",
            effective_tag="main", rank=1,
        )
        mgr._register_fastlane_claim_locked(claimant, "make_room_starved_count_cap")

        # NON-VACUITY: the claimant really is a live, observable waiting
        # claim (not just a side-effect of the locked call), and it really
        # does make `victim` -- specifically -- the designated victim, not
        # both/neither loaded residents.
        claims = mgr.fastlane_claims_snapshot()
        assert any(c.get("model_tag") == "claimant-model" for c in claims), (
            f"claimant must appear as a live waiting claim -- got {claims!r}"
        )
        assert mgr._is_designated_unload_target_locked(victim) is True
        assert mgr._is_designated_unload_target_locked(survivor) is False

        try:
            result = await _drive_one_turn_and_measure(
                mgr, victim, "victim-model", ceiling_s=IDLE_WINDOW_S + 1.0,
            )
            assert result["outcome"] == "evicted", (
                f"designated victim never reached teardown within the "
                f"ceiling -- outcome={result['outcome']!r}"
            )

            # PRIMARY, order-relative: the eviction
            # event must precede the window-expiry event -- a contract
            # between two real, spied timestamps, not a wall-clock margin.
            eviction_at = result["eviction_at"]
            expected_idle_expires_at = result["expected_idle_expires_at"]
            assert eviction_at < expected_idle_expires_at, (
                f"designated victim was evicted {eviction_at - expected_idle_expires_at:.3f}s "
                f"AFTER its own ordinary idle_expires_at, not before it -- the rule requires "
                f"eviction to PRECEDE the ordinary idle-expiry moment. "
                f"turn_complete_at+0={result['turn_complete_at']:.3f}, "
                f"idle_expires_at=+{expected_idle_expires_at - result['turn_complete_at']:.3f}s, "
                f"actual eviction=+{eviction_at - result['turn_complete_at']:.3f}s "
                f"(configured idle window: {result['idle_window_s']:.1f}s)"
            )

            # SECONDARY, belt-and-braces (if it costs nothing, place it in
            # the MIDDLE of the two regimes): half the OBSERVED idle
            # window, not a separate hand-picked constant. Redundant with
            # the primary assertion on the buggy path (both fail together);
            # exists to also catch a hypothetical "technically-ordered-
            # correct but still uncomfortably slow" regression the primary
            # assertion alone would miss.
            secondary_bound_s = result["idle_window_s"] / 2
            actual_duration_s = eviction_at - result["turn_complete_at"]
            assert actual_duration_s < secondary_bound_s, (
                f"designated victim took {actual_duration_s:.3f}s to tear down -- "
                f"still ordered before idle_expires_at, but not comfortably inside "
                f"the {secondary_bound_s:.2f}s belt-and-braces bound (half the "
                f"{result['idle_window_s']:.1f}s configured window)"
            )
        finally:
            mgr._residents.pop("survivor-model", None)
            await mgr.shutdown()

    async def test_GREEN_CONTROL_non_victim_still_resident_past_the_midpoint(self, tmp_path):
        """CONTROL -- must PASS both before and after the fix. Proves the
        primary assertion isn't trivially satisfiable by everything tearing
        down fast: a non-victim, checked at half its own OBSERVED idle
        window (the same reference point the discriminator's secondary
        bound uses), must still be resident and IDLE_EVICTABLE -- genuinely
        still waiting out its own ordinary idle window, not fast-tracked by
        the widened branch. Behavior here is identical on both regimes (a
        non-victim never takes the designated-victim branch either way), so
        this control's own pass/fail does not depend on which source it
        runs against."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds, fastlane_rules=_RULES)
        mgr = TurbohaulManager(boot, runtime, **_mocks())

        solo = _resident(
            mgr, "solo-model", 59963,
            rank_client_meta=None, grace_seconds=grace_seconds,
        )
        solo.state = ResidentState.ACTIVE
        assert mgr._is_designated_unload_target_locked(solo) is False  # non-vacuity

        holder = {"turn_complete_at": None, "idle_window_computed_at": None, "idle_window_s": None}
        serve_spy = _make_serve_spy(type(mgr)._serve_on_resident, holder)
        idle_spy = _make_idle_window_spy(type(mgr)._idle_window_seconds, holder)
        serve_patcher = patch.object(type(mgr), "_serve_on_resident", serve_spy)
        idle_patcher = patch.object(type(mgr), "_idle_window_seconds", idle_spy)
        serve_patcher.start()
        idle_patcher.start()
        drive_task = asyncio.create_task(mgr._drive_resident(solo))
        try:
            anchor = Slot.new("solo-model", prompt="hi", thread_id="t1")
            anchor.state = SlotState.STAGED
            await solo.inbox.put(anchor)

            # A non-victim holds its FULL grace_seconds wait before
            # _serve_on_resident returns (it is not the designated victim,
            # so grace is never skipped) -- the ceiling must cover that.
            turn_wait_deadline = time.monotonic() + grace_seconds + 2.0
            await _wait_for(
                lambda: holder["idle_window_computed_at"] is not None, turn_wait_deadline,
                "control's turn never completed -- harness bug",
            )
            midpoint_s = holder["idle_window_s"] / 2
            await asyncio.sleep(midpoint_s)
            elapsed = time.monotonic() - holder["turn_complete_at"]
            cur = mgr._residents.get("solo-model")
            assert cur is not None, (
                f"non-victim was already evicted at {elapsed:.2f}s (midpoint of its "
                f"own {holder['idle_window_s']:.1f}s window) -- the discriminator's "
                f"secondary bound would be vacuously satisfied by anything tearing "
                f"down this fast, victim or not"
            )
            assert cur.state is ResidentState.IDLE_EVICTABLE, (
                f"non-victim should still be warm-idle-holding at {elapsed:.2f}s, "
                f"got state={cur.state!r}"
            )
        finally:
            drive_task.cancel()
            try:
                await drive_task
            except asyncio.CancelledError:
                pass
            idle_patcher.stop()
            serve_patcher.stop()
            await mgr.shutdown()
