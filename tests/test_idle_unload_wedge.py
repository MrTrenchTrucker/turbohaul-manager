"""The idle-unload/eviction-eligibility WEDGE.

Trigger: a resident parked in _drive_resident's
inner loop, blocked in `await asyncio.wait_for(r.inbox.get(),
timeout=idle_window)`. A producer (the _route_or_reserve HIT path,
in manager.py) takes _registry_lock, flips IDLE_EVICTABLE -> ACTIVE,
and puts a slot. If that slot is EVICTED (the producer's own admission can
yield an evicted rider), the driver's TimeoutError handler
(in manager.py) drains it as a no-admit (_priority_admit_from_inbox
fail-completes evicted riders and returns None), finds r.state STILL ACTIVE
(the producer's flip), and hits the ELSE branch -- which, UNFIXED, re-parked
and `continue`d: timeout, drain nothing, ELSE, continue -- FOREVER. The
resident never re-reached the ONE place that flips it to IDLE_EVICTABLE
(manager.py, outer loop only). Invisible to
_lru_idle_unloadable (IDLE_EVICTABLE-only predicate) and therefore to
make-room -- a competing request starves behind it (MAKE_ROOM_STARVED
events keep firing).

THE FIX (applied to the real
tree): in that ELSE branch ONLY -- still under the SAME _registry_lock --
(1) _transition_resident_state(r.state, IDLE_EVICTABLE) (the guard,
never raises), (2) r.state = IDLE_EVICTABLE (the mirror), (3)
`continue` -- NOT `break`. The `continue` re-enters the inner loop
(no _maybe_defer_admission re-run; that sits before it), re-parks in get();
on the next idle-window timeout the IDLE_EVICTABLE branch runs
_begin_unload_locked -> DEAD -> the outer loop's own DEAD check
exits the driver before it can ever re-serve the stale slot. A `break`
would NOT be safe: the real _serve_on_resident ends with
transition(slot, POPPED), so the outer top's re-serve of the stale slot
would raise InvalidTransition (fsm.py) and kill the driver task (the
existing comment in manager.py documents the
same hazard, which is why the branch stays `continue`-shaped).

DETERMINISTIC HARNESS (why this does not race):
  Note: a put landing while the getter is PARKED in get() is
  consumed BY THE GETTER (the wake path -- _rank_admit_woken serves even
  evicted slots) -- the NON-wedge case. So the ghost put must land AFTER
  the getter's deadline, when no live getter exists. The driver's
  TimeoutError handler takes `async with self._registry_lock` --
  the ONLY suspension point between the get() timeout and the branch-1
  drain. So:
    T+0.00  holder task acquires _registry_lock (anchor turn done, driver
            parked in get(); a parked getter does not take the lock).
    T+0.10  (lock held) flip ACTIVE -- the producer shape; the
            holder IS the stand-in for the producer's lock.
    T~T_d   the driver's get() deadline fires (T_d in [T_settle+1.8,
            T_settle+2.1] -- see _park). wait_for CANCELS its getter and
            raises TimeoutError; the driver QUEUES on the held lock. From
            this moment NO GETTER EXISTS.
    L=T_d+  the GHOST (an EVICTED rider) is put. It cannot be consumed: no
            getter exists; the only pre-release drainer is the handler's
            branch-1 drain, and the holder owns the lock.
    L+1.0   holder releases; the handler runs: branch-1 drain
            FAIL-COMPLETES the ghost, returns None; state still ACTIVE;
            not DEAD; no stop.
                UNFIXED: ELSE -> re-park -> timeout -> ELSE -> ... FOREVER.
                FIXED:   ELSE -> guard + IDLE_EVICTABLE + continue ->
                        re-park -> next timeout -> IDLE_EVICTABLE branch
                        -> _begin_unload_locked -> DEAD -> driver exits
                        at the outer DEAD check. Never a stale re-serve.
  Every ordering the code could take is pinned by the held lock + the
  single-threaded loop: the ghost cannot be consumed before the handler
  (no getter exists in the window), and the handler cannot run before the
  put (the holder owns the lock). The outcome is a function of the code
  under test alone.

MUST FAIL WITHOUT THE FIX:
  - test_idle_unload_wedge_resident_becomes_evictable: stays ACTIVE for
    >= 2 idle windows with an EMPTY inbox and a fail-completed ghost, and
    is NOT a _lru_idle_unloadable candidate -- the wedge described in
    the module docstring.
  - test_competing_request_stars_then_is_served: make-room victim
    selection (_lru_idle_unloadable -- the predicate whose None return
    fires MAKE_ROOM_STARVED) returns None while the wedge holds.
The four-branch proofs (the OTHER four branches of the same handler are
byte-identical and behave as before) live in TestTheOtherFourBranchesAreUnchanged.
"""
from __future__ import annotations

import asyncio
import inspect
import textwrap
import time

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
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.slot import Slot, SlotState

# --- fixture: copied in shape from the rank-admit-woken test
# (the established way to drive the REAL _drive_resident hermetically: real
# admission path, faked spawn/serve). Kept local so this file stands alone in
# the repo (no shared fixture module needed).

W = 2  # idle_window seconds: keep_alive_s=2 -> _idle_window_seconds -> min(2, KEEP_ALIVE_MAX_S) = 2


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


def _slot(slot_id: str, created_at: float, *, is_evicted=False) -> Slot:
    return Slot(
        slot_id=slot_id,
        model_tag="m1",
        state=SlotState.ACTIVE,
        created_at=created_at,
        fastlane=None,
        is_evicted=is_evicted,
        client_meta={},
        completion_future=asyncio.Future(),
    )


async def _wait_until(pred, timeout=5.0, msg="condition never became true"):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.005)
    raise AssertionError(msg)


async def _park(mgr, r, served, monkeypatch, keep_alive_s=W):
    """Bring the driver up, let it finish one anchor turn, and leave it PARKED
    in the blocking `r.inbox.get()` with an EMPTY inbox. With keep_alive_s=2 the
    park window is idle_window=2s, so the TimeoutError branch under test is
    reached once per 2 seconds.

    SETTLE argument (why T_d is bracketed): the IDLE_EVICTABLE observation can
    precede the driver's get() entry by one event-loop turn (the outer top
    flips the state BEFORE the inner loop starts; the get() is entered within
    one tick after). The 0.2s settle sleeps past that tick, so by settle-end
    the driver is inside get() and its deadline T_d satisfies T_settle + 1.8
    <= T_d <= T_settle + 2.1. The harness schedules everything off T_settle
    with margins inside that bracket."""
    async def fake_spawn(r_, slot):
        return object()

    async def fake_serve(r_, slot, handle):
        served.append(slot)

    monkeypatch.setattr(mgr, "_spawn_for_resident", fake_spawn)
    monkeypatch.setattr(mgr, "_serve_on_resident", fake_serve)
    anchor = _slot("anchor", 0.0)
    r.inbox.put_nowait(anchor)
    r.latest_keep_alive_s = keep_alive_s
    r.sleep_idle_seconds = keep_alive_s
    task = asyncio.create_task(mgr._drive_resident(r))
    await _wait_until(lambda: len(served) >= 1, timeout=5.0, msg="anchor turn never ran")
    await _wait_until(
        lambda: r.state is ResidentState.IDLE_EVICTABLE and r.inbox.empty(),
        timeout=5.0,
        msg="driver never parked in IDLE_EVICTABLE with an empty inbox",
    )
    await asyncio.sleep(0.2)
    assert r.state is ResidentState.IDLE_EVICTABLE and r.inbox.empty(), (
        f"settle: the resident was not parked in IDLE_EVICTABLE (state {r.state}) -- precondition broken"
    )
    return task


class _WedgeHarness:
    """Drives the resident into the producer-race wedge window and holds it.
    START AFTER THE PARK (never before: the anchor turn needs _registry_lock
    freely, and the holder must not contend with it). See the module docstring
    for the full pinning argument."""

    def __init__(self, mgr, r, ghost):
        self.mgr = mgr
        self.r = r
        self.ghost = ghost
        self.release = asyncio.Event()
        self.flip_done = asyncio.Event()
        self.flip_wall = 0.0  # wall-clock of the producer flip (for the guard spy)
        self.holder = None
        self.sequencer = None

    async def __aenter__(self):
        mgr, r = self.mgr, self.r

        async def hold_lock():
            async with mgr._registry_lock:
                await self.release.wait()

        self.holder = asyncio.create_task(hold_lock())
        await asyncio.sleep(0.1)  # holder owns the lock; the driver is parked in get()

        async def sequence():
            # The driver's get() deadline T_d is in [T_settle+1.8, T_settle+2.1].
            # Wait past the UPPER edge so the flip+ghost land strictly after the
            # deadline has fired in every case: no getter exists, nothing can
            # consume the put, and the driver is queued on the handler's
            # _registry_lock -- the single suspension point between the
            # timeout and the branch-1 drain.
            await asyncio.sleep(W + 0.5)
            # (1) producer-flip, as in the real producer (lock, flip ACTIVE). The
            # holder IS the stand-in for that lock (it owns it), so this write
            # leaves exactly the state the real producer leaves: ACTIVE, no
            # put yet.
            r.state = ResidentState.ACTIVE
            self.flip_wall = time.monotonic()
            self.flip_done.set()
            # (2) the producer's put: an EVICTED rider, L+0.4 -- still no getter
            # (the getter was cancelled at T_d < L), so the ghost sits in the
            # inbox until the handler's branch-1 drain runs it.
            await asyncio.sleep(0.4)
            r.inbox.put_nowait(self.ghost)
            # (3) hold the lock a little longer so the handler -- queued on it
            # since T_d -- only runs its branch-1 drain AFTER the ghost is in
            # place (the holder owns the lock until the release below).
            await asyncio.sleep(0.6)
            self.release.set()

        self.sequencer = asyncio.create_task(sequence())
        # __aenter__ returns only AFTER the flip is applied: the test body's
        # flip-assertion must never race the sequencer (an early return here is
        # exactly what would make the harness misfire).
        await self.flip_done.wait()
        return self

    async def __aexit__(self, *exc):
        self.release.set()
        for t in (self.holder, self.sequencer):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


class TestTheOtherFourBranchesAreUnchanged:
    """Acceptance criterion: evidence that the OTHER FOUR branches of the
    same TimeoutError handler (admitted-from-inbox, IDLE_EVICTABLE->
    begin_unload, DEAD, stop_event) behave exactly as before. Two
    independent proofs: (a) a STRUCTURAL byte-identical proof over the source
    of the whole handler (the UNFIXED text is PINNED below, copied from the
    tree (manager.py), dedented exactly as
    inspect.getsource(textwrap.dedent) sees it -- not regenerated, so this
    cannot silently re-baseline if the tree moves; compared line-normalized
    so a cosmetic trailing-whitespace touch cannot mask a real change), and
    (b) behavioral tests for each branch, each of which fails if that
    branch's control flow changed (e.g. its `break` became a `continue`)."""

    # manager.py handler text, verbatim (dedented exactly as
    # inspect.getsource(textwrap.dedent) sees it).
    UNFIXED_HANDLER = """\
except TimeoutError:
                    async with self._registry_lock:
                        admitted = self._priority_admit_from_inbox(r)
                        if admitted is not None:
                            slot = admitted
                            break
                        if r.state is ResidentState.IDLE_EVICTABLE:
                            self._begin_unload_locked(r)
                            # Self-idle-unload
                            # eviction also releases capacity with no
                            # immediate claimant -- same third moment,
                            # same notify, same lock already held here.
                            # Wake the dispatch loop alongside the make-room signal
                            self._dispatch_wake.set()
                            self._make_room_signal.notify_all()
                            break
                        if r.state is ResidentState.DEAD:
                            # falls through safely: the outer loop's own
                            # DEAD check (before slot is ever touched)
                            # handles it -- nothing to do here.
                            break
                        if self._stop_event.is_set():
                            # let the outer while's own condition catch
                            # the stop on the next iteration.
                            break
                        # state changed to something else under us (see
                        # the paragraph above) -- retry, do not fall
                        # through with a stale slot.
                        continue
    """

    @staticmethod
    def _norm(text: str) -> list:
        """Line-aware normalization: rstrip each line (cosmetic trailing
        whitespace never meant anything), drop blank lines -- a real text or
        indentation change still surfaces."""
        return [ln.rstrip() for ln in text.split("\n") if ln.rstrip() != ""]

    def _handler_source(self) -> str:
        src = textwrap.dedent(inspect.getsource(TurbohaulManager._drive_resident))
        start = src.index("except TimeoutError:")
        end = src.index("except asyncio.CancelledError:")
        return src[start:end]

    def test_handler_first_four_branches_byte_identical_except_else(self):
        """The four non-ELSE branches must be line-identical to the unfixed
        code; the fix may appear ONLY in the ELSE position (the last
        statement group of the handler)."""
        cur = self._handler_source()
        else_marker = "# state changed to something else under us (see"
        assert self.UNFIXED_HANDLER.count(else_marker) == 1, (
            "test setup: the ELSE marker must be unique in the pinned handler"
        )
        base_unfixed = self.UNFIXED_HANDLER[:self.UNFIXED_HANDLER.index(else_marker)]
        idx = cur.find(else_marker)
        assert idx != -1, (
            "the pinned ELSE marker is gone -- the fix may rewrite "
            "the ELSE comment, but it must not delete it. Current handler (re-"
            f"pin required):\n{cur}"
        )
        base_cur = cur[:idx]
        assert self._norm(base_cur) == self._norm(base_unfixed), (
            "the TimeoutError handler's FIRST FOUR BRANCHES are not "
            "identical to the unfixed code.\n--- unfixed (pinned) ---\n"
            f"{base_unfixed}--- current ---\n{base_cur}"
        )
        # And the change must be confined to the ELSE: everything after the
        # marker in the current source is the ELSE (comment + guard + state
        # write + control statement). The final non-comment code line must be
        # `continue` -- the required shape. A `break` there is a
        # defect (stale-slot re-serve -> InvalidTransition ->
        # dead driver), so the pin must REQUIRE continue, not accept either.
        cur_else = cur[idx:]
        code_lines = [
            ln for ln in cur_else.split("\n")
            if ln.strip() and not ln.strip().startswith("#")
        ]
        assert code_lines, "the ELSE branch body is empty after the marker"
        last_stmt = code_lines[-1].strip()
        assert last_stmt == "continue", (
            "the ELSE branch must end the inner-loop cycle with "
            "`continue` (the required shape). A `break` re-serves the stale slot "
            "and kills the driver (a defect). Found instead: "
            f"{last_stmt!r}"
        )


class TestIdleUnloadWedge:
    async def test_idle_unload_wedge_resident_becomes_evictable(self, tmp_path, monkeypatch):
        """THE wedge. Without the fix: the resident is flipped ACTIVE by
        the producer-race window, STAYS ACTIVE for >= 2 idle windows with an
        EMPTY inbox and a fail-completed ghost (the exact drain shape), and is
        NOT a _lru_idle_unloadable candidate. With the fix:
        the ELSE branch flips it to IDLE_EVICTABLE (guard call witnessed) and
        re-parks; it becomes a make-room candidate; then its NEXT idle-window
        timeout takes the IDLE_EVICTABLE branch -> _begin_unload_locked ->
        DEAD -> the driver exits at the outer DEAD check -- a clean
        self-unload with NO stale re-serve (served stays 1)."""
        import turbohaul.manager as _tmgr

        # Guard-call witness: a dropped
        # _transition_resident_state call in the fix is behaviorally invisible
        # -- this spy makes it observable. Records (wall, from, to) for every
        # guard call and calls the original.
        guard_calls = []
        _orig_guard = _tmgr._transition_resident_state

        def _spy(state, new_state):
            guard_calls.append((time.monotonic(), state, new_state))
            return _orig_guard(state, new_state)

        monkeypatch.setattr(_tmgr, "_transition_resident_state", _spy)

        mgr = _mk(tmp_path)
        r = Resident(model_tag="m1", inbox=asyncio.Queue())
        mgr._residents["m1"] = r  # so _model_residents/_lru_idle_unloadable see it
        served: list[Slot] = []
        task = await _park(mgr, r, served, monkeypatch)
        try:
            ghost = _slot("ghost", 1.0, is_evicted=True)
            async with _WedgeHarness(mgr, r, ghost) as harness:
                # The flip must have landed before the put (harness invariant):
                # if it did not, the window was never entered -- a harness
                # misfire, not the wedge.
                assert r.state is ResidentState.ACTIVE, (
                    "harness misfire: the producer-flip did not take "
                    f"hold -- state is {r.state.value}, not ACTIVE."
                )
                # UNFIXED: stays ACTIVE for >= 2 full idle windows (the spin).
                # FIXED: leaves ACTIVE promptly (the ELSE flips on the first
                # handler run after the ghost drain).
                resolved = True
                try:
                    await _wait_until(
                        lambda: r.state is not ResidentState.ACTIVE,
                        timeout=2 * W + 1.0,
                        msg="harness misfire: resident never left ACTIVE?",
                    )
                except AssertionError:
                    resolved = False

                if not resolved:
                    # WEDGE HOLDS (expected on UNFIXED code). Prove the FULL
                    # wedge -- not just "still ACTIVE" -- then fail loudly.
                    assert r.inbox.empty(), (
                        "wedge shape: the inbox must be EMPTY once the "
                        f"ghost is drained -- a non-empty inbox is a different failure: "
                        f"{[s.slot_id for s in r.inbox._queue]}"
                    )
                    assert ghost.completion_future.done(), (
                        "wedge shape: the evicted ghost must have been "
                        "FAIL-COMPLETED by the handler's branch-1 drain -- its "
                        "future is not done, so this is not the wedge this test targets"
                    )
                    cand = mgr._lru_idle_unloadable()
                    assert cand is None, (
                        "wedge shape: an ACTIVE wedged resident must NOT "
                        f"be an LRU make-room candidate -- got {cand!r}"
                    )
                    pytest.fail(
                        "WEDGE REPRODUCED (expected without the fix): "
                        f"resident stuck in {r.state.value} with an empty inbox and a "
                        f"fail-completed ghost for >= {2 * W}s (>= 2 idle windows) -- "
                        "it can never idle-unload (the flip at manager.py is "
                        "only reachable via the outer loop, which this spin never "
                        "breaks to) and can never be an LRU make-room victim "
                        "(cand=None): the MAKE_ROOM_STARVED shape."
                    )

                # GREEN: the fix fired. It flips to IDLE_EVICTABLE and
                # re-parks; the resident is a make-room candidate from here.
                await _wait_until(
                    lambda: r.state is ResidentState.IDLE_EVICTABLE,
                    timeout=3 * W,
                    msg="the fix fired (left ACTIVE) but the resident "
                        "did not settle in IDLE_EVICTABLE",
                )
                assert r.inbox.empty(), "inbox must be empty once the wedge is resolved"
                cand = mgr._lru_idle_unloadable()
                assert cand is r, (
                    "the formerly-wedged resident must now be a valid "
                    f"LRU make-room candidate -- _lru_idle_unloadable() returned "
                    f"{cand!r}, expected {r!r}"
                )
                # Guard-call witness: the fix's guard call is
                # ACTIVE -> IDLE_EVICTABLE and must have happened AFTER the
                # producer flip (the anchor turn's own flip happened before
                # it, so "any" is not specific enough).
                post_flip = [
                    c for c in guard_calls
                    if c[1] is ResidentState.ACTIVE
                    and c[2] is ResidentState.IDLE_EVICTABLE
                    and c[0] >= harness.flip_wall
                ]
                assert post_flip, (
                    "guard-witness: no ACTIVE->IDLE_EVICTABLE "
                    "_transition_resident_state call after the producer flip -- "
                    "the fix's guard call did not run (a dropped guard is "
                    "behaviorally invisible; this spy exists to catch it)"
                )
                # SAFE SELF-UNLOAD (the heart of the fix): the resident's
                # NEXT idle-window timeout takes the IDLE_EVICTABLE branch
                # -> _begin_unload_locked -> DEAD, and the driver
                # exits at the outer loop's own DEAD check --
                # BEFORE it can re-serve the stale slot. A `break`-shaped
                # fix would instead re-serve the stale POPPED slot at the
                # outer top and raise InvalidTransition in the REAL
                # _serve_on_resident (fsm.py); here the fake serve would not
                # raise, but the stale re-serve itself would append to
                # `served` -- so served must stay exactly 1 (the anchor).
                await _wait_until(
                    lambda: r.state is ResidentState.DEAD,
                    timeout=2 * W + 4.0,
                    msg="the fix flipped IDLE_EVICTABLE but the "
                        "resident never self-unloaded to DEAD within the "
                        "next idle window + margin",
                )
                await _wait_until(
                    lambda: task.done(), timeout=3.0,
                    msg="driver task did not exit after DEAD",
                )
                assert not task.cancelled(), (
                    "the driver task ended by CANCELLATION, not a "
                    "clean exit at the outer DEAD check"
                )
                assert len(served) == 1, (
                    "the stale anchor must NOT be "
                    f"re-served at the outer top -- served {[s.slot_id for s in served]} "
                    f"(len {len(served)}), expected exactly the anchor"
                )
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_competing_request_stars_then_is_served(self, tmp_path, monkeypatch):
        """THE competing-request arm (the MAKE_ROOM_STARVED shape). A second resident
        (different model, ACTIVE with a live anchor -- NEVER evictable) means
        a third request's make-room has exactly ONE possible victim: the
        formerly-idle one. While the wedge holds, _lru_idle_unloadable() --
        the predicate whose None return is what fires MAKE_ROOM_STARVED --
        returns None: the competing request starves. After the fix it returns
        the formerly-wedged resident: the competing request's capacity frees."""
        mgr = _mk(tmp_path)
        r_wedge = Resident(model_tag="m1", inbox=asyncio.Queue())
        r_competing = Resident(model_tag="m2", inbox=asyncio.Queue())
        r_competing.active_slot = _slot("competing-anchor", 0.5)  # busy: never evictable
        r_competing.state = ResidentState.ACTIVE
        mgr._residents["m1"] = r_wedge
        mgr._residents["m2"] = r_competing
        served: list[Slot] = []
        task = await _park(mgr, r_wedge, served, monkeypatch)
        try:
            ghost = _slot("ghost", 1.0, is_evicted=True)
            async with _WedgeHarness(mgr, r_wedge, ghost):
                assert r_wedge.state is ResidentState.ACTIVE, (
                    "harness misfire: the producer-flip did not take "
                    f"hold -- state is {r_wedge.state.value}, not ACTIVE."
                )
                resolved = True
                try:
                    await _wait_until(
                        lambda: r_wedge.state is not ResidentState.ACTIVE,
                        timeout=2 * W + 1.0,
                        msg="harness misfire: resident never left ACTIVE?",
                    )
                except AssertionError:
                    resolved = False

                cand = mgr._lru_idle_unloadable()
                if not resolved:
                    assert cand is None, (
                        "with the wedge holding, make-room must find "
                        f"NO victim (the starvation) -- got {cand!r}"
                    )
                    pytest.fail(
                        "STARVATION REPRODUCED (expected without "
                        "the fix): the competing request's make-room found no evictable "
                        "resident (cand=None) for >= 2 idle windows while the wedged "
                        "resident held its slot -- the MAKE_ROOM_STARVED shape "
                        "of the wedge."
                    )
                # GREEN: the competing request's victim selection now finds the
                # formerly-wedged resident (well inside idle_window).
                await _wait_until(
                    lambda: mgr._lru_idle_unloadable() is r_wedge,
                    timeout=3 * W,
                    msg="after the fix, make-room never selected the "
                        "formerly-wedged resident",
                )
                # Extra gate: candidate selection alone
                # would also pass a malformed fix that makes the resident
                # evictable BUT still re-serves the stale anchor at the outer
                # top (the shape of that defect). This driver must have
                # served EXACTLY the anchor.
                assert len(served) == 1, (
                    f"the wedge driver served {len(served)} slots -- "
                    f"expected exactly the anchor. A stale re-serve means the "
                    f"ELSE branch does not end the inner cycle with `continue` "
                    f"(the defect's shape: the outer top re-serves the "
                    f"completed slot)."
                )
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


class TestBranchesAreUnchanged:
    async def test_branch1_admitted_from_inbox_still_breaks_with_slot(self, tmp_path, monkeypatch):
        """Branch 1: a LIVE (non-evicted) slot in the inbox at timeout time is
        admitted and the driver breaks to the outer loop and serves it. This
        is the normal idle-wake path through the TIMEOUT handler (the put
        lands after the getter's deadline, so the parked getter cannot take
        it) -- it must still work after the fix."""
        mgr = _mk(tmp_path)
        r = Resident(model_tag="m1", inbox=asyncio.Queue())
        mgr._residents["m1"] = r
        served: list[Slot] = []
        task = await _park(mgr, r, served, monkeypatch)
        try:
            live = _slot("live-rider", 1.0, is_evicted=False)
            async with mgr._registry_lock:
                r.inbox.put_nowait(live)
                # release the lock; the driver's NEXT timeout takes branch 1
            await _wait_until(
                lambda: len(served) >= 2,
                timeout=3 * W,
                msg="branch 1: a live rider in the inbox at timeout "
                    "was not admitted and served",
            )
            assert served[-1] is live, (
                f"branch 1: wrong slot served -- {served[-1].slot_id}"
            )
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_branch2_idle_evictable_still_begins_unload(self, tmp_path, monkeypatch):
        """Branch 2: a resident that is IDLE_EVICTABLE when its timeout fires
        begins unload (_begin_unload_locked -> DEAD) -- the ordinary idle
        path, unchanged. (With the fix, the wedged resident reaches THIS
        branch on its next window -- which is exactly the safe path.)"""
        mgr = _mk(tmp_path)
        r = Resident(model_tag="m1", inbox=asyncio.Queue())
        mgr._residents["m1"] = r
        served: list[Slot] = []
        task = await _park(mgr, r, served, monkeypatch)
        # _park already leaves r IDLE_EVICTABLE with an empty inbox; the next
        # idle-window timeout must take branch 2 and unload it.
        await _wait_until(
            lambda: r.state is ResidentState.DEAD,
            timeout=3 * W,
            msg="branch 2: an IDLE_EVICTABLE resident with an empty "
                "inbox did not begin unload on the next idle-window timeout",
        )
        await _wait_until(
            lambda: task.done(), timeout=3.0,
            msg="branch 2: driver task did not exit after DEAD",
        )
        task.cancel()

    async def test_branch3_dead_still_breaks_cleanly(self, tmp_path, monkeypatch):
        """Branch 3: if the resident is DEAD when the timeout handler runs,
        the handler breaks cleanly (the outer loop's own DEAD check finishes
        the job) -- unchanged."""
        mgr = _mk(tmp_path)
        r = Resident(model_tag="m1", inbox=asyncio.Queue())
        mgr._residents["m1"] = r
        served: list[Slot] = []
        task = await _park(mgr, r, served, monkeypatch)
        async with mgr._registry_lock:
            r.state = ResidentState.DEAD
        # DEAD: the next timeout hits branch 3 -> break; the outer loop's
        # DEAD check exits the driver without serving anything new.
        await _wait_until(
            lambda: task.done(), timeout=3 * W,
            msg="branch 3: a DEAD resident's driver did not break "
                "out cleanly on the next timeout",
        )
        assert len(served) == 1, (
            f"branch 3: the driver served {len(served)} slots; a DEAD "
            "resident must not start new serves"
        )
        task.cancel()

    async def test_branch4_stop_event_still_breaks(self, tmp_path, monkeypatch):
        """Branch 4: with _stop_event set, the handler breaks so the outer
        while's own condition ends the driver -- unchanged."""
        mgr = _mk(tmp_path)
        r = Resident(model_tag="m1", inbox=asyncio.Queue())
        mgr._residents["m1"] = r
        served: list[Slot] = []
        task = await _park(mgr, r, served, monkeypatch)
        mgr._stop_event.set()
        await _wait_until(
            lambda: task.done(), timeout=3 * W,
            msg="branch 4: a set _stop_event did not end the driver "
                "within the next idle window + margin",
        )
        assert not task.cancelled(), "branch 4: driver ended by cancellation"
        task.cancel()
