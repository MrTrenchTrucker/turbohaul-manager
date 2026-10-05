"""Deterministic repro skeleton for the evict-then-probe race.

Code under test: the src/turbohaul/... tree. Run this file against
that tree's checkout; each test pins either the fixed behavior of the
evict-then-probe race path or a guard around it that must stay green:

    pytest <this test file> -v

WHAT IS REAL (bound methods of the actual TurbohaulManager, via a __new__ harness —
no full __init__, no GPU, no network):
  _route_or_reserve, _vram_admits_locked, _begin_unload_locked, _unload_teardown,
  _defer_unroutable, _max_vram_defers, _lru_idle_unloadable,
  _reserve_and_start_locked, _resolve_placement_locked, _read_model_footprint,
  _auto_pick_gpu, _fail_completion_future, _spawn_bg, the Resident dataclass,
  the Slot dataclass, the real defer-budget math.

WHAT IS FAKE (and why):
  * nvidia-smi probe — ScriptedProbe (no GPU in CI). It models the residual
    CUDA-context window: the card keeps reporting the evicted victim's occupancy
    from process exit (t_exit) until the test calls set_card() — i.e. the test
    scripts t_freed explicitly (t_residual = t_freed - t_exit).
  * manifest reader — the manager-namespace `read_manifest` is patched to a
    fixed fake manifest (footprint values drive `need`).
  * `_requeue_after_backoff` — shadowed to an immediate, recorded re-enqueue.
    The 1.0s backoff sleep is NOT under test; each scripted
    `_route_or_reserve` re-entry stands for one backoff period (the
    interleaving is scripted explicitly). The budget counter + 503 failure live in the
    REAL `_defer_unroutable`, which is untouched.
  * `_drive_resident` — stubbed (spawn/health/safety-gates are out of scope for
    this repro; the race is entirely in the route/evict/credit/probe logic).
  * `_effective_cap` — shadowed to 1 (single-instance tags; the multi-instance
    branch of _route_or_reserve is out of scope).
  * victims have handle=None, booting_pid=None, idle_thread_id=None:
    `_unload_teardown` then takes its no-reap-branch path (both reap branches are
    gated on those) and drops the pending-reclaim credit in its `finally` the moment the
    task is pumped — i.e. the credit dies at "process reap" with ZERO drain
    delay: the widest residual window (t_exit as early as possible, t_residual
    fully scripted in the probe). Worst case for the over-eviction scenario.

WHAT EACH TEST PINS:
  * the residual-window over-eviction test: T_RESIDUAL_TICKS EXTRA victims
    would be evicted inside the window (one per re-route tick) if the pending-reclaim credit
    died at process reap, before the probe reflected the freed VRAM.
  * the credit-guard test: passes on any code —
    pins the intended guard (credit present -> projected fits -> no second
    eviction), so a change must not regress it.
  * test_credit_accounting_lifecycle: credit add -> drop
    on reap, clamped at 0, exactly-once.
  * the auto-place aggregate-credit test: the
    aggregate credit says "fits" (16000 >= 12000) while per-card reality can
    never fit (8000 < 12000); the auto-placer's layer fallback is
    refuse-blind against the co-residents (early return), no victims remain,
    and the slot stays queued with 16000 MiB genuinely freed
    (aggregate-versus-per-card mismatch).
  * test_bounded_budget_absorbs_plain_window: with the
    credit held alive, a plain residual window costs only
    extra defers: no over-eviction, no 503 (the budget absorbs the window).

The expected behavior in the evict-then-probe race is that the residual-window
over-eviction test sees exactly 1 eviction, an admit at the
first tick after t_freed, and a defer count == T_RESIDUAL_TICKS + 1.

Determinism: no real sleeps in the tested path (the requeue sleep is shadowed),
no GPU, no network, no /proc reads on the path under test. All interleavings are
explicit test steps + event-loop pumps.
"""
from __future__ import annotations

import asyncio
import contextlib
import types

import pytest

from turbohaul import manager as M
from turbohaul.manager import (
    Resident,
    ResidentState,
    Slot,
    TurbohaulManager,
)
from turbohaul.slot import SlotState


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class ScriptedProbe:
    """nvidia-smi `memory.free` fake: per-card free MiB, mutated by the test.

    Models the residual CUDA window: between the victim's process exit and
    driver-level VRAM release, nvidia-smi still reports the card occupied —
    the test keeps the card 'occupied' (free=0) until it calls set_card(),
    i.e. it scripts t_freed explicitly (t_residual = t_freed - t_exit).
    """

    def __init__(self, cards: dict[int, int]) -> None:
        self._free = dict(cards)  # card -> free MiB
        self.probe_calls: list[list[int]] = []

    def free_all(self) -> list[int]:
        vals = [self._free[g] for g in sorted(self._free)]
        self.probe_calls.append(list(vals))
        return vals

    def budget(self, split_mode: str = "layer", main_gpu: int = 0):
        # Mirrors safety._vram_budget semantics.
        vals = self.free_all()
        if (split_mode or "layer").lower() == "none":
            idx = main_gpu if 0 <= main_gpu < len(vals) else 0
            return vals[idx], vals[idx], 1
        return sum(vals), min(vals), len(vals)

    def set_card(self, g: int, free_mib: int) -> None:
        self._free[g] = free_mib


class FakeManifest:
    def __init__(self, expected_vram_mib: int, auto_place: bool = False) -> None:
        self.llama_server_flags = {
            "ctx_size": 0,          # keeps estimate_kv_cache_mib at 0
            "parallel": 1,
            "main_gpu": 0,
            "split_mode": "none",
            "sleep_idle_seconds": 0,
        }
        self.gguf_size_bytes = 0
        self.context_size = 0
        self.expected_vram_bytes = expected_vram_mib * 1024 * 1024
        self.hybrid_kv_ratio = 1.0
        self.auto_place = auto_place
        self.gguf_blob_sha256 = ""  # keeps _attn_kv_dims_for on its early None path


class FakeQueue:
    """TurbohaulQueue surface the race path touches (config + requeue sink)."""

    def __init__(self) -> None:
        # grace_seconds is kept small (real seconds).
        # The _late_vram_reconcile task sleeps `grace_seconds` before its one-shot
        # delayed re-verify -- at 30s, that background task would outlive every
        # test in this file (none call mgr.shutdown()), and keep running
        # uncancelled into whatever runs next, adding real wall-clock
        # to a full-suite run. A small real
        # value keeps the delayed re-verify's timing MODEL intact (something
        # genuinely happens after a real, bounded delay) while letting every
        # test that triggers a teardown finish in well under a second.
        self.grace_seconds = 0.05
        self.max_grace_extensions = 5
        self.safety_min_free_vram_mib = 512
        self.max_parallel_sidecars = 1
        self.requeued: list = []
        # Harness requirement: _reserve_and_start_locked
        # (unrelated to this repro's subject) needs this to construct
        # a Resident's IdleHotTimer. 0 keeps idle-window math inert
        # (matches this repro's own scope: it is entirely about the
        # evict/credit/probe race, not idle-timeout behavior).
        # It plays no part in the race itself.
        self.idle_hot_load_seconds = 0

    async def listed_waiter_model_tags(self):
        return set()

    async def listed_waiter_priority_keys(self):
        # Mirrors listed_waiter_model_tags() above --
        # this repro is entirely about the evict/credit/probe race, not Fast
        # Lane priority, so an empty map is the correct "no listed waiters"
        # answer for both, same as the real queue would give an idle staging
        # buffer.
        return {}

    async def enqueue_head(self, slot: Slot) -> None:
        self.requeued.append(slot)

    def _fastlane_priority_key(self, slot: Slot):
        # The real TurbohaulQueue's own contract
        # (queue.py) -- (rule_index, rank) for a listed slot, None for one
        # with no .fastlane. _route_or_reserve's VRAM-over-commit branch
        # calls this unconditionally (not gated behind a non-empty listed-
        # waiter set, unlike _shield_carveout_tags's own early return), so
        # this fixture needs the method. Every
        # Slot this file constructs (_miss()) leaves .fastlane unset, so
        # this always returns None here, same as production would for the
        # same input; this repro remains entirely about the evict/credit/
        # probe race, not Fast Lane priority (see the sibling methods'
        # comments above).
        return None


# ---------------------------------------------------------------------------
# Harness: real TurbohaulManager, __new__-constructed, race-path attributes only
# ---------------------------------------------------------------------------

def _make_manager(monkeypatch, probe: ScriptedProbe,
                  manifests: dict[str, FakeManifest]
                  ) -> tuple[TurbohaulManager, FakeQueue]:
    mgr = TurbohaulManager.__new__(TurbohaulManager)
    mgr._residents = {}
    mgr._registry_lock = asyncio.Lock()
    mgr._session_affinity = {}
    mgr._last_request_identity_by_model = {}
    mgr._bg_tasks = set()
    mgr._completion_cache = {}
    mgr._completion_inflight = {}
    mgr._hybrid_kv_announced = set()
    mgr._attn_dims_cache = {}
    # Harness requirement: __new__ bypasses
    # __init__ entirely, so NONE of __init__'s attributes exist here unless
    # this harness sets them. _pending_reclaim_mib is one of them, and
    # every test here references it directly, so the harness
    # must set it.
    mgr._pending_reclaim_mib = {}
    # Same __new__-harness requirement as _pending_reclaim_mib
    # above -- __new__ bypasses
    # __init__ entirely, so the harness must set it explicitly.
    mgr._card_release_tasks = {}
    # Same __new__-harness requirement again -- a sibling
    # dict to _pending_reclaim_mib, written by _begin_unload_locked whenever
    # a credit is raised (unconditionally, not just when the reconciler
    # path is under test), so every test that reaches that line needs it
    # present.
    mgr._pending_reclaim_raised_at = {}
    # __new__ bypasses __init__ so _idle_handle does not
    # exist unless the harness sets it. The teardown path in
    # _route_or_reserve reads self._idle_handle on the cross-model VRAM-miss
    # branch, so every test that reaches admission must have this present.
    mgr._idle_handle = None
    # Deliberately NOT setting _vram_total_mib here --
    # its absence on a __new__-built manager IS the scenario
    # the capacity-invariant-survives-a-new-harness-manager test and the
    # missing-capacity-keeps-credit test (both below) exist to prove is
    # handled safely.
    # Tests that need a real capacity set mgr._vram_total_mib explicitly.

    # _vram_verify + _gpu_used_mib are wired to
    # the SAME ScriptedProbe the file already scripts, so "confirmed cleared"
    # only becomes true once the probe's own t_freed is set (an
    # unconditional-cleared stub would hide the window).
    #
    # _gpu_used_mib: ScriptedProbe tracks FREE MiB; the real get_gpu_memory_
    # used_mib returns USED MiB (the opposite direction -- used falls as more
    # frees). Rather than invent a fictional per-card TOTAL capacity (the
    # probe has none), used is modeled as -free: a pre/post delta of
    # (-pre_free) - (-post_free) == post_free - pre_free, which is exactly
    # "how much MORE is free now than before" -- the correct sign and
    # magnitude for _unload_teardown's measured_delta math, with no invented
    # constant.
    mgr._gpu_used_mib = lambda *, device_index=0: -probe.free_all()[device_index]

    # _vram_verify: the real verify_vram_cleared polls for up to 30 REAL
    # seconds inside ONE call -- in this scripted-tick harness there is no
    # real time to poll across, so this polls the SAME probe on a bounded
    # asyncio.sleep(0) cadence instead. Critically, _unload_teardown's bg task
    # must NOT be pumped to full completion (via _pump's gather()) before the
    # test has had a chance to advance the probe on later ticks -- see
    # _drive_shallow below, used instead of _drive for every tick except the
    # last in the two window-shaped tests. Bounded at a generous multiple of
    # the scripted window so a genuinely-never-clears case still terminates.
    _VERIFY_MAX_POLLS = 64

    async def _stub_vram_verify(*, expected_drop_mib, timeout_s, device_index=0, **_ignored):
        # **_ignored absorbs the settle_floor_s kwarg --
        # _late_vram_reconcile's call passes settle_floor_s=0.0
        # explicitly, which this scripted-tick harness
        # (real time is not available to poll across) does not model.
        current = None
        for _ in range(_VERIFY_MAX_POLLS):
            vals = probe.free_all()
            current = vals[device_index] if device_index < len(vals) else None
            if current is not None and current >= expected_drop_mib * 0.9:
                return True, -current  # used-MiB sign convention, see above
            await asyncio.sleep(0)
        return False, (-current if current is not None else None)

    mgr._vram_verify = _stub_vram_verify

    fq = FakeQueue()
    mgr.queue = fq
    mgr.runtime = types.SimpleNamespace(queue=fq)
    mgr.boot = types.SimpleNamespace(
        storage=types.SimpleNamespace(manifests_path="/nonexistent"),
        runtime=types.SimpleNamespace(default_port_base=11500))

    # Probe + manifest seams (module-namespace names the real methods call).
    monkeypatch.setattr(M, "_vram_budget", probe.budget)
    monkeypatch.setattr(M, "_read_free_vram_all_mib", probe.free_all)
    monkeypatch.setattr(M, "read_manifest",
                        lambda path, tag: manifests[tag])

    # Documented shadows (see module docstring).
    mgr._effective_cap = lambda tag: 1

    async def _immediate_requeue(slot: Slot, *, backoff_s: float = 0.05) -> None:
        fq.requeued.append(slot)  # one scripted backoff period per re-entry

    mgr._requeue_after_backoff = _immediate_requeue

    async def _stub_driver(r: Resident) -> None:
        # Spawn/health/safety-gates out of scope for this repro.
        slot = await r.inbox.get()
        r.active_slot = slot
        r.state = ResidentState.ACTIVE
        r.routed_slot = slot  # test hook

    mgr._drive_resident = _stub_driver
    return mgr, fq


def _make_resident(model_tag: str, *, need: int, state: ResidentState,
                   gpu: int = 0, last_active: float = 0.0) -> Resident:
    r = Resident(
        model_tag=model_tag,
        resident_key=model_tag,
        port=10000 + len(model_tag),
        state=state,
        reserved_need_mib=need,
        main_gpu=gpu,
        split_mode="none",
        inflight=[],
        inbox=asyncio.Queue(),
        last_active_monotonic=last_active,
        idle_thread_id=None,  # skips the pre-SIGTERM KV-save block
    )
    r.torn_down = False
    return r


def _miss(tag: str) -> Slot:
    s = Slot(
        slot_id=f"miss-{tag}",
        model_tag=tag,
        state=SlotState.RECEIVED,
        prompt="p",
    )
    s.completion_future = None
    return s


async def _pump(mgr: TurbohaulManager, rounds: int = 8) -> None:
    """Let the detached bg tasks (teardown, immediate requeue) run to quiescence."""
    for _ in range(rounds):
        await asyncio.sleep(0)
    for _ in range(rounds):
        if mgr._bg_tasks:
            await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)
            break
        await asyncio.sleep(0)


def _routed(mgr: TurbohaulManager, tag: str) -> bool:
    return any(
        r.model_tag == tag and r.state in (
            ResidentState.RESERVED_LOADING, ResidentState.ACTIVE)
        for r in mgr._residents.values())


def _count_evictions(mgr: TurbohaulManager) -> list[str]:
    real_begin = mgr._begin_unload_locked
    seen: list[str] = []

    def counting(r):
        seen.append(r.model_tag)
        real_begin(r)

    mgr._begin_unload_locked = counting
    mgr._evictions = seen
    return seen


async def _drive(mgr: TurbohaulManager, miss: Slot) -> None:
    """One scripted re-route tick (stands for one backoff period later)."""
    await mgr._route_or_reserve(miss)
    await _pump(mgr)


async def _pump_shallow(mgr: TurbohaulManager, rounds: int = 8) -> None:
    """Like _pump, but never gather()s a bg
    task to completion. _pump's own gather() call blocks the CALLING
    coroutine (this test's own tick loop) until the bg task fully finishes --
    which would make a real, internally-polling _vram_verify (see
    _stub_vram_verify above) run to exhaustion on the very first tick, before
    the test has had any chance to call probe.set_card() on a later tick. A
    shallow pump instead just cooperatively yields: the bg task's own
    internal asyncio.sleep(0) polls and this test's tick loop's yields
    interleave naturally on the same event loop, each getting scheduled in
    turn, so the bg task can observe a probe state the test sets BETWEEN
    _drive_shallow calls. The bg task is only actually finished off by a
    real _pump (via _drive, not _drive_shallow) on the tick that closes the
    scripted window -- see its use in the two window-shaped tests below."""
    for _ in range(rounds):
        await asyncio.sleep(0)


async def _drive_shallow(mgr: TurbohaulManager, miss: Slot) -> None:
    """Like _drive, but pumps shallowly -- see _pump_shallow."""
    await mgr._route_or_reserve(miss)
    await _pump_shallow(mgr)


# ---------------------------------------------------------------------------
# Scenario A: single card, residual window, per-tick over-eviction
# ---------------------------------------------------------------------------

NEED = 8000            # incoming miss, MiB
VICTIM_NEED = 8000     # reserved_need_mib of each idle victim
T_RESIDUAL_TICKS = 5   # re-route ticks the card stays occupied after process exit
                       # (t_residual = 5 * 1.0s backoff in the scripted model)
CAPACITY = 10000       # test card's physical MiB capacity


@pytest.mark.asyncio
async def test_overeviction_in_residual_window(monkeypatch):
    """Fixed behavior of the residual window with a late clear.

    Design intent: model a late clear so the test reaches the target
    state stated below: exactly 1 eviction total, admit right after
    t_freed. That is the intended outcome; it is not (and cannot be) a
    series of per-tick "evicts victim-N every tick" assertions, which
    would describe a window in which the credit had been dropped at
    process reap. With the credit held alive, tick 1 alone already
    shows ZERO extra evictions -- the held-alive credit makes
    `projected_fits=True` from the very first residual tick onward,
    for ANY tick, regardless of when the probe eventually frees.
    There is no timing of "late" that reproduces multiple evictions
    once the credit correctly survives the window; asserting otherwise
    would assert the dropped-credit behavior instead. So this test
    checks the target state directly rather than forcing per-tick
    eviction assertions that the fixed behavior
    can never satisfy.

    How this differs from the never-clears/exactly-one-reverify
    test (which proves the SAME held-credit
    mechanism via a more controlled harness): THIS test models the probe
    freeing strictly BETWEEN the first verify's timeout and the delayed
    re-verify's own check -- i.e. genuinely "late" but still within the
    grace window -- so it is the delayed re-verify itself, not the credit-
    blind direct admit path, that must catch the clearance and release the
    credit. That is the credit-hold code path exercised end-to-end
    through the real _route_or_reserve/_defer_unroutable
    machinery.
    """
    probe = ScriptedProbe({0: 0})  # card 0: 0 MiB free (blocker 8000 + victim 8000)
    manifests = {
        "blocker": FakeManifest(NEED),
        "vic-0": FakeManifest(VICTIM_NEED),
        "miss-a": FakeManifest(NEED),
    }
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    evictions = _count_evictions(mgr)
    # No count cap in play: the final admit must go through the plain
    # vram_fits -> reserve path, not the count-cap
    # branch, whose in-band evict-then-reserve has different
    # victim semantics.
    mgr.queue.max_parallel_sidecars = 8
    # This test's own 5-tick loop + setup takes longer in real wall-clock
    # time than FakeQueue's global grace_seconds default (0.05s, sized for
    # the OTHER, simpler repro tests) -- override to a value comfortably
    # longer than that, so the "late clear" set below genuinely lands BEFORE
    # the delayed re-verify's own check, not after it has already given up.
    mgr.queue.grace_seconds = 0.5

    # Card 0 = 16000 MiB total: ACTIVE blocker holds 8000, idle victim 8000.
    mgr._residents["blocker"] = _make_resident(
        "blocker", need=NEED, state=ResidentState.ACTIVE)
    mgr._residents["vic-0"] = _make_resident(
        "vic-0", need=VICTIM_NEED, state=ResidentState.IDLE_EVICTABLE,
        last_active=0.0)

    miss = _miss("miss-a")

    # --- t=0: first miss. Probe occupied -> evict vic-0, defer. The first
    # verify (30s budget, this test's probe unchanged so far) times out ->
    # dec=0, credit stays FULLY alive (not dropped -- a
    # "credit == 0 after one pump" assertion would pin
    # the dropped-credit behavior instead).
    await _drive(mgr, miss)
    assert not _routed(mgr, "miss-a"), "miss must not route while over-committed"
    assert evictions == ["vic-0"], f"expected first evict of vic-0, got {evictions}"
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED, (
        "the first verify times out on the still-occupied probe "
        "-> the credit must stay fully alive, not drop to 0"
    )
    assert miss._vram_defer_count == 1
    assert probe.free_all() == [0], "residual window: probe still occupied"

    # --- t=1..5: scripted re-route ticks INSIDE the residual window. The
    # credit is held alive (above) -> projected_fits=True on every tick ->
    # NO extra eviction (without the credit hold this loop would over-evict a
    # new victim every tick -- see this test's docstring for why
    # that shape is not reachable here).
    #
    # Uses _drive_shallow, NOT _drive: _drive's full _pump() gather()s every
    # pending bg task to completion, including the delayed re-verify t=0
    # already scheduled -- gathering it here would BLOCK this loop for the
    # full grace_seconds (real time) before the "late clear" below is ever
    # set, defeating the "late" part of the model entirely (a plain
    # _drive call for tick 1 takes about one grace_seconds of wall-clock
    # time, for exactly this
    # reason). A shallow pump lets the reconcile task's own internal
    # asyncio.sleep(grace_seconds) interleave with -- not block -- these
    # ticks, so it is still pending when we get to setting the probe below.
    for tick in range(1, T_RESIDUAL_TICKS + 1):
        await _drive_shallow(mgr, miss)
        assert not _routed(mgr, "miss-a"), f"tick {tick}: probe occupied — defer"
        assert evictions == ["vic-0"], (
            f"tick {tick}: the held credit must suppress any further "
            f"eviction; got {evictions}"
        )

    # --- LATE CLEAR: the card actually freed sometime during the window
    # (modeled here, between the ticks above and the delayed re-verify's own
    # check below) -- but the credit doesn't know that yet; only the
    # delayed re-verify (scheduled at t=0 for grace_seconds later) will
    # observe it.
    probe.set_card(0, VICTIM_NEED)
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED, (
        "the credit must still read as held immediately after the probe "
        "changes -- nothing re-checks it synchronously"
    )

    # Wait for the delayed re-verify (grace_seconds, real time -- see
    # FakeQueue's own grace_seconds override, which keeps bg tasks
    # short-lived) to fire and observe the NOW-freed card.
    for _ in range(300):
        await asyncio.sleep(0.01)
        if mgr._pending_reclaim_mib.get(0, 0) == 0:
            break
    assert mgr._pending_reclaim_mib.get(0, 0) == 0, (
        "the delayed re-verify must catch the late clear and release the "
        "credit -- the actual mechanism, not the credit-blind "
        "direct admit path"
    )

    # A fresh re-route tick now admits via the plain (credit-blind) real
    # check, since the probe itself shows the card free. Spy on
    # _reserve_and_start_locked directly, and check evictions/reserved
    # IMMEDIATELY after -- NOT via _drive (route_or_reserve + a full pump)
    # and NOT via post-hoc residency state (_routed()): this file's
    # _stub_driver consumes exactly one inbox item then returns (unlike the
    # real _drive_resident, which loops forever), so once its driver_task
    # completes, _on_driver_done's callback treats that as an unexpected
    # exit and evicts/deregisters the JUST-reserved resident again
    # (evictions ends up
    # ["vic-0", "miss-a"], the resident vanishes from _residents). This is
    # the SAME harness gap bounded_budget's test is xfail'd for
    # (a known limitation) -- unrelated to what THIS test actually proves
    # (the late-clear reconcile). Reading `evictions`/`reserved` BEFORE any
    # pump gives the stub driver no chance to run and pollute either.
    reserved = []
    real_reserve = mgr._reserve_and_start_locked

    async def spy_reserve(slot, **kw):
        r = await real_reserve(slot, **kw)
        reserved.append(r)
        return r

    mgr._reserve_and_start_locked = spy_reserve
    await mgr._route_or_reserve(miss)
    assert reserved and reserved[-1] is not None, (
        "once the probe reflects the freed VRAM, _reserve_and_start_locked "
        "must succeed (admit) -- got no successful reserve call"
    )
    # Target state: exactly
    # ONE eviction total for the whole scenario -- it would be 1+T_RESIDUAL_TICKS
    # if the credit were dropped (the over-eviction failure). Read BEFORE any pump -- see
    # comment above.
    assert len(evictions) == 1, (
        f"exactly one eviction expected (the fix); got {evictions}"
    )


@pytest.mark.asyncio
async def test_credit_guard_holds_while_credit_alive(monkeypatch):
    """Pins the INTENDED guard (must stay green): while the
    pending-reclaim credit is present, the projected check must fit and suppress a second
    eviction. Synchronous — no teardown
    task is pumped, so the credit cannot have dropped."""
    probe = ScriptedProbe({0: 0})
    manifests = {"vic-0": FakeManifest(VICTIM_NEED),
                 "miss-a": FakeManifest(NEED)}
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    v0 = _make_resident("vic-0", need=VICTIM_NEED,
                        state=ResidentState.IDLE_EVICTABLE)
    mgr._residents["vic-0"] = v0

    # Evict v0 WITHOUT pumping (credit alive, probe still occupied).
    mgr._begin_unload_locked(v0)
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED
    # Projected WITH credit: fits -> no second eviction would be triggered.
    assert mgr._vram_admits_locked(NEED, 1, 0, "none",
                                   credit_pending=True) is True
    # Projected WITHOUT credit (the reserve gate's view): still refused.
    assert mgr._vram_admits_locked(NEED, 1, 0, "none",
                                   credit_pending=False) is False
    # The reserve gate must NEVER consume the credit: even with the credit at
    # 100x the need, credit_pending=False refuses while the probe says occupied.
    mgr._pending_reclaim_mib[0] = VICTIM_NEED * 100
    assert mgr._vram_admits_locked(NEED, 1, 0, "none",
                                   credit_pending=False) is False


@pytest.mark.asyncio
async def test_credit_accounting_lifecycle(monkeypatch):
    """Mechanism assertion -- NEVER-CLEARS
    shape: credit add -> the first verify times out on this
    test's unchanging probe (dec=0, the credit stays fully alive, NOT
    dropped) -> one delayed re-verify is scheduled -> it ALSO times
    out on the same unchanging probe -> the credit remains, exactly once,
    honest. The credit is never dropped unconditionally on reap. This is
    the same underlying
    mechanism as the timeout-never-clears-exactly-one-reverify test
    in test_multislot_concurrency.py, exercised here via this file's own
    __new__ harness + real _route/_unload machinery instead of the full
    worker_loop, which is why it is its own test
    (different harness, same mechanism,
    complementary coverage, not redundant)."""
    probe = ScriptedProbe({0: 0})
    manifests = {"vic-0": FakeManifest(VICTIM_NEED), "miss-a": FakeManifest(NEED)}
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    v0 = _make_resident("vic-0", need=VICTIM_NEED,
                        state=ResidentState.IDLE_EVICTABLE)
    mgr._residents["vic-0"] = v0

    # Spy on _vram_verify's call count: this scenario's probe never changes,
    # so `_pending_reclaim_mib` reads the SAME (still-credited) value whether
    # the delayed re-verify actually fires or not -- the dict-value
    # assertions alone do not discriminate the delayed re-verify (only
    # the base dec=0 mechanism). Counting calls proves the
    # delayed re-verify genuinely happened, not just that nothing changed.
    verify_calls = []
    real_verify = mgr._vram_verify

    async def counting_verify(**kw):
        verify_calls.append(kw)
        return await real_verify(**kw)

    mgr._vram_verify = counting_verify

    mgr._begin_unload_locked(v0)
    assert mgr._pending_reclaim_mib[0] == VICTIM_NEED
    assert v0.reclaim_credited_mib == VICTIM_NEED
    assert v0.state is ResidentState.DEAD
    assert v0.torn_down is False  # teardown task not pumped yet

    await _pump(mgr)  # _unload_teardown's finally runs; first verify times out
    assert v0.torn_down is True
    # dec=0 on the (unchanging) probe -> the credit
    # stays FULLY alive -- dropping it unconditionally on reap
    # would be the stuck-credit failure shape, not the
    # intended behavior.
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED
    assert v0.reclaim_credited_mib == 0  # per-resident marker still clears each run

    assert len(verify_calls) == 1, "the first verify must have fired synchronously"

    # Wait past the scripted grace_seconds for the ONE delayed re-verify to
    # actually fire (proving the delayed re-verify exists, not just that nothing
    # changed) and also observe the still-unchanging probe.
    for _ in range(300):
        await asyncio.sleep(0.01)
        if len(verify_calls) >= 2:
            break
    assert len(verify_calls) == 2, (
        f"expected exactly one delayed re-verify (2 total calls); got "
        f"{len(verify_calls)} -- the delayed re-verify must actually fire, not "
        f"just leave the credit unchanged by coincidence"
    )
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED, (
        "never-clears: the credit must remain after the delayed re-verify "
        "also fails to observe a freed card"
    )

    # Underflow safety: re-running the teardown on the same resident must not
    # touch the credit again (torn_down exactly-once guard, unrelated to the delayed re-verify).
    await mgr._unload_teardown(v0)
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED


# ---------------------------------------------------------------------------
# Scenario B: auto_place aggregate credit vs per-card landing -> stays queued
# ---------------------------------------------------------------------------

NEED_B = 12000   # miss needs 12000 on ONE card
VICTIM_B = 8000  # each victim frees 8000, on a DIFFERENT card


@pytest.mark.asyncio
async def test_auto_place_aggregate_credit_stays_queued_forever(monkeypatch):
    """A queued auto_place miss whose aggregate credit cannot land on any
    single card stays QUEUED; it is never refused with a 503,
    even though the aggregate credit says the miss would fit.

    The scenario: two 8000 MiB victims on cards 0/1 are evicted for
    an auto_place miss that needs 12000 on ONE card; the per-card
    reality can never fit (8000 < 12000); the auto-placer's
    aggregate layer fallback is refuse-blind against the ACTIVE
    co-residents (early return, which also bypasses the
    pending-reclaim credit); once both victims are gone there is
    nothing left to evict — 16000 MiB was genuinely freed but
    no single card can host 12000 of it.

    Terminal behavior: there is no defer-budget exhaustion path.
    A request that is queued stays queued until the client
    cancels it, so this slot stays QUEUED forever,
    re-deferring at its own 1s backoff with no ceiling,
    same as every other unroutable-for-now slot
    (it never ends in a retryable 503, VramOverCommitError).

    The adjacent question of what happens to a model that can
    never fit is not answered at the queue layer: the answer lives in
    manifest offload configuration (n_gpu_layers / cpu_moe /
    n_cpu_moe) or a genuine OOM, never a queue-layer refusal.
    A model that is too big either runs out of memory or is
    handled by the per-manifest configuration, where models can
    offload partially or in whole (a large MoE model can be
    offloaded even though it does not fit in VRAM, with the
    active experts still resident in VRAM).
    The auto-placer credit mismatch documented here is
    real and still open -- a future fix for it should
    expect THIS test to flip to "the miss routes," not to a
    503 assertion, because no
    503 path exists.

    Direct unit assertion of the credit mismatch below:
    with both victims' credits pending, the relocatable (aggregate)
    projection says FITS (16000 >= 12000) while the per-card reserve truth
    is NEVER fit — the credit scope (aggregate) cannot be satisfied by the
    landing scope (per-card).

    Driven well PAST the derived defer budget (floor of 60
    defers, at shrunk grace) to prove there is no ceiling left to hit, not
    just a slower one."""
    probe = ScriptedProbe({0: 0, 1: 0})  # both cards: 0 free (blockers+vitims)
    manifests = {
        "blk-a": FakeManifest(VICTIM_B),
        "blk-b": FakeManifest(VICTIM_B),
        "vic-a": FakeManifest(VICTIM_B),
        "vic-b": FakeManifest(VICTIM_B),
        "miss-b": FakeManifest(NEED_B, auto_place=True),
    }
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    evictions = _count_evictions(mgr)
    mgr.queue.max_parallel_sidecars = 8  # no count-cap; VRAM-only over-commit

    mgr._residents["blk-a"] = _make_resident(
        "blk-a", need=VICTIM_B, state=ResidentState.ACTIVE, gpu=0, last_active=9.0)
    mgr._residents["vic-a"] = _make_resident(
        "vic-a", need=VICTIM_B, state=ResidentState.IDLE_EVICTABLE,
        gpu=0, last_active=1.0)
    mgr._residents["blk-b"] = _make_resident(
        "blk-b", need=VICTIM_B, state=ResidentState.ACTIVE, gpu=1, last_active=9.0)
    mgr._residents["vic-b"] = _make_resident(
        "vic-b", need=VICTIM_B, state=ResidentState.IDLE_EVICTABLE,
        gpu=1, last_active=2.0)

    # --- Unit assertion of the credit mismatch (synchronous, no tasks).
    # The aggregate (relocatable) projection alone would wrongly CLAIM a
    # fit (16000 >= 12000) while the per-card reserve truth could never
    # fit. The check works on each card individually now: neither
    # card alone (0 free + 8000 own
    # credit = 8000) reaches the 12000 need, so this correctly returns
    # False.
    mgr._pending_reclaim_mib[0] = VICTIM_B
    mgr._pending_reclaim_mib[1] = VICTIM_B
    assert mgr._vram_admits_locked(NEED_B, 1, 0, "none", credit_pending=True,
                                   relocatable=True) is False, (
        "per-card check: neither card's own (free + its own credit) reaches "
        "12000 individually -- summing the two cards' credit together must "
        "not manufacture a fit that doesn't physically exist")
    probe.set_card(0, VICTIM_B)
    probe.set_card(1, VICTIM_B)
    assert mgr._vram_admits_locked(NEED_B, 1, 0, "none",
                                   credit_pending=False) is False, (
        "per-card truth: 8000 < 12000 on card 0 — the reserve can NEVER fit; "
        "the aggregate credit is unsatisfiable")
    mgr._pending_reclaim_mib.clear()
    probe.set_card(0, 0)
    probe.set_card(1, 0)

    # Shrink the defer budget via the REAL sizing math: grace 2s, 1 extension
    # -> window = 2*2 + 60 = 64s -> ceil(64/1.0) = 64 -> clamped to [60,1800]
    # -> 60 (the floor). Fast + deterministic.
    mgr.queue.grace_seconds = 2
    mgr.queue.max_grace_extensions = 1
    max_defers = mgr._max_vram_defers()
    assert max_defers >= 60

    miss = _miss("miss-b")
    fut = asyncio.get_running_loop().create_future()
    miss.completion_future = fut
    real_fail = mgr._fail_completion_future
    def fail_and_record(slot, exc):
        if slot is miss:
            real_fail(slot, exc)
            if not fut.done():
                fut.set_exception(exc)
        else:
            real_fail(slot, exc)
    mgr._fail_completion_future = fail_and_record

    # --- Phase 1 (residual window): both victims are evicted, one per tick.
    await _drive(mgr, miss)          # t=0: evict vic-a (global-LRU, oldest)
    assert evictions[-1] == "vic-a", f"t=0 should evict vic-a, got {evictions}"
    await _drive(mgr, miss)          # t=1: probe occupied, credit 0 -> evict vic-b
    assert evictions[-1] == "vic-b", f"t=1 should evict vic-b, got {evictions}"
    assert len(evictions) == 2
    assert miss._vram_defer_count == 2

    # --- Phase 2 (t_freed): the probe frees both cards — 8000 each.
    probe.set_card(0, VICTIM_B)
    probe.set_card(1, VICTIM_B)
    # No idle victims remain; the auto-placer now offers the aggregate
    # (0, 'layer') fallback, which the co-residence gate refuses blind;
    # nothing left to evict, so the slot just keeps re-deferring.
    # No exhaustion path -- drive well PAST the derived
    # cap (2x + margin) to prove there is no ceiling left to hit, not just
    # a slower one.
    ticks = 0
    old_cap_margin = (max_defers * 2) + 4
    while not fut.done() and ticks <= old_cap_margin:
        await _drive(mgr, miss)
        ticks += 1

    assert not fut.done(), (
        f"the slot must stay QUEUED forever, never resolve "
        f"via a defer-exhaustion 503 -- resolved after {ticks} ticks (derived "
        f"cap was {max_defers}), evictions={evictions}"
    )
    # The auto-placer credit mismatch is real and still open here (out of
    # scope -- see this test's own docstring for the answer to the adjacent
    # "unservable model" question): the
    # miss never routes either. It does not 503 -- it re-defers
    # forever instead, exactly like every other unroutable-for-now slot.
    assert miss._vram_defer_count > max_defers, (
        "defer count should have climbed well past the derived cap with no "
        "exhaustion firing"
    )
    assert len(evictions) == 2, "exactly the two victims were evicted -- no over-eviction"


# ---------------------------------------------------------------------------
# Scenario C: a PLAIN residual window (credit held alive) is absorbed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Provenance: this file's _stub_driver "
        "consumes exactly one inbox item then returns, unlike the real "
        "_drive_resident (which loops forever) -- once its driver_task "
        "completes, _on_driver_done's callback treats that as an "
        "unexpected exit and reaps/deregisters the just-reserved resident "
        "before this test's final _routed() check can observe it. "
        "Pre-existing harness gap in this file, unrelated to the fix "
        "under test (confirmed independently by test_overeviction_in_residual_"
        "window hitting the identical failure at its own final step, "
        "worked around there via a _reserve_and_start_locked spy instead "
        "of _routed() -- not applied here to keep this test's diff minimal "
        "and the xfail self-documenting). strict=False: a future harness "
        "fix (a looping stub driver, or the same spy-based check) should "
        "flip this back to a real pass, not silently stay xfail forever."
    ),
)
async def test_bounded_budget_absorbs_plain_window(monkeypatch):
    """When the pending-reclaim credit survives the reap (drop-on-probe-confirmed — the shape
    a credit hold takes; the idle-holder path's verify_vram_cleared semantics),
    a plain residual window costs only extra
    defers: no over-eviction, no 503 (the budget absorbs the window).

    The test holds the credit alive across the pump (simulating the
    credit-hold behavior) and drives the REAL route/defer code through the window. It passes
    because the test itself holds the credit; its job is to pin the
    CONSEQUENCE side (defer-only cost) so the target state is
    unambiguous."""
    probe = ScriptedProbe({0: 0})
    manifests = {"blocker": FakeManifest(NEED),
                 "vic-0": FakeManifest(VICTIM_NEED),
                 "miss-a": FakeManifest(NEED)}
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    evictions = _count_evictions(mgr)
    mgr.queue.max_parallel_sidecars = 8  # see Scenario A note

    mgr._residents["blocker"] = _make_resident(
        "blocker", need=NEED, state=ResidentState.ACTIVE, last_active=9.0)
    mgr._residents["vic-0"] = _make_resident(
        "vic-0", need=VICTIM_NEED, state=ResidentState.IDLE_EVICTABLE,
        last_active=1.0)

    miss = _miss("miss-a")
    await _drive(mgr, miss)
    assert evictions == ["vic-0"]

    # The real teardown dropped the credit during the pump; re-hold it to
    # simulate drop-on-probe-confirmed (the target behavior).
    mgr._pending_reclaim_mib[0] = VICTIM_NEED

    for tick in range(1, T_RESIDUAL_TICKS + 1):
        await _drive(mgr, miss)
        assert evictions == ["vic-0"], (
            f"tick {tick}: credit alive -> NO further eviction may happen "
            f"(another eviction would mean the credit hold is missing)")
        assert not _routed(mgr, "miss-a"), f"tick {tick}: probe occupied — defer"

    probe.set_card(0, VICTIM_NEED)  # t_freed
    await _drive(mgr, miss)
    assert _routed(mgr, "miss-a"), "miss must route once the probe frees"
    assert evictions == ["vic-0"]
    assert miss._vram_defer_count == T_RESIDUAL_TICKS + 1, (
        "extra defers = throughput cost only (budget absorbs)")


# ---------------------------------------------------------------------------
# Orphaned VRAM credit -- ROUTE 2 (teardown task dies
# before its finally's decrement), the still-held skip, and the
# non-empty-task-set skip. Route 1 (verify timeout, credit correctly stays
# alive) is already covered above by test_credit_accounting_lifecycle -- a
# test that only drove route 1 would pass against a per-card-
# counter design too, which is exactly the trap to avoid.
# ---------------------------------------------------------------------------


async def _begin_evict_and_get_teardown_task(mgr, r):
    """Drives _begin_unload_locked and returns the SINGLE teardown task it
    registered into _card_release_tasks -- exercising the
    registration itself, not a second, parallel bookkeeping path."""
    mgr._begin_unload_locked(r)
    tasks = mgr._card_release_tasks.get(r.main_gpu, set())
    assert len(tasks) == 1, (
        f"expected exactly 1 registered teardown task for card "
        f"{r.main_gpu}; got {len(tasks)}"
    )
    return next(iter(tasks))


@pytest.mark.asyncio
async def test_route2_orphaned_credit_is_reconciled_once_genuinely_free(
    monkeypatch,
):
    """ROUTE 2, the one that matters: the teardown task is cancelled WHILE
    suspended inside its own finally's _vram_verify await (the exact
    mechanism of the stuck credit -- except Exception at that await does
    not catch CancelledError, so everything after it, including the
    registry-lock decrement, never runs). One
    _reconcile_orphaned_vram_credit tick, with the card genuinely freed,
    releases it.

    The reconciler compares absolute free VRAM against the credit (see
    manager.py's _reconcile_orphaned_vram_credit) rather than a
    delta against a pre-teardown baseline -- this single-credit,
    cleanly-freed scenario passes under BOTH predicates
    (both-arms-green; the differentiating scenario, where a clobbered
    baseline makes a delta check fail where the absolute check
    succeeds, is the multi-teardown-credit reconciler test
    below). Needs _vram_total_mib populated
    and a real (non-sign-flipped) used-MiB reading for the absolute predicate to
    evaluate at all; without them this test would prove nothing
    about the reconciler."""
    probe = ScriptedProbe({0: 0})
    manifests = {"vic-0": FakeManifest(VICTIM_NEED), "miss-a": FakeManifest(NEED)}
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    mgr._vram_total_mib = [CAPACITY]
    mgr._gpu_used_mib = lambda *, device_index=0: CAPACITY - probe.free_all()[device_index]
    v0 = _make_resident("vic-0", need=VICTIM_NEED,
                        state=ResidentState.IDLE_EVICTABLE)
    mgr._residents["vic-0"] = v0

    teardown_task = await _begin_evict_and_get_teardown_task(mgr, v0)
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED

    # Let the task run far enough to enter _unload_teardown's finally and
    # suspend inside _stub_vram_verify's own polling loop (a real `await`,
    # not yet resolved) -- NOT far enough to finish (_pump_shallow, not
    # _pump/gather).
    await _pump_shallow(mgr, rounds=4)
    assert not teardown_task.done(), (
        "setup problem: the task finished before it could be killed "
        "mid-finally -- this test would prove nothing about route 2"
    )

    # ROUTE 2: kill it before its finally's decrement runs.
    teardown_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await teardown_task
    assert teardown_task.done() and teardown_task.cancelled()

    # The credit is stuck -- the stuck-credit failure, reproduced directly.
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED, (
        "credit must be stuck after its owning task died mid-finally"
    )
    # The task registry must have already self-discarded the dead task (its
    # done-callback fires on cancellation exactly like normal completion).
    assert mgr._card_release_tasks.get(0, set()) == set(), (
        "a cancelled task must still be discarded from the release-task "
        "registry -- otherwise the reconciler would see a permanently "
        "'live' task and never fire either"
    )
    # The actual observable damage: the make-room decision wrongly believes
    # this card fits, forever, absent a reconciler -- _vram_admits_locked itself
    # is not at fault; this proves the STUCK CREDIT feeds it a
    # false input, not that the gate's own logic is wrong.
    assert mgr._vram_admits_locked(
        NEED, 1, 0, "none", credit_pending=True,
    ) is True, "projected_fits wrongly True while the card is still occupied"

    # The card has now genuinely, fully freed.
    probe.set_card(0, VICTIM_NEED)

    await mgr._reconcile_orphaned_vram_credit()

    assert mgr._pending_reclaim_mib.get(0, 0) == 0, (
        "one reconciler tick must release a genuinely orphaned, genuinely "
        "free credit"
    )


@pytest.mark.asyncio
async def test_still_held_credit_is_kept_not_released(monkeypatch):
    """The keep arm, looked at first: an orphaned
    credit whose VRAM is STILL genuinely held must be KEPT, not released --
    a reconciler that just zeros every stuck credit passes every
    credit-cleared test and fails this one.

    Both-arms-green under the absolute predicate too
    (still-held VRAM means free_now stays well below 0.9*credit either way)
    -- so only a reconciler that releases unconditionally fails it."""
    probe = ScriptedProbe({0: 0})
    manifests = {"vic-0": FakeManifest(VICTIM_NEED), "miss-a": FakeManifest(NEED)}
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    mgr._vram_total_mib = [CAPACITY]
    mgr._gpu_used_mib = lambda *, device_index=0: CAPACITY - probe.free_all()[device_index]
    v0 = _make_resident("vic-0", need=VICTIM_NEED,
                        state=ResidentState.IDLE_EVICTABLE)
    mgr._residents["vic-0"] = v0

    teardown_task = await _begin_evict_and_get_teardown_task(mgr, v0)
    await _pump_shallow(mgr, rounds=4)
    assert not teardown_task.done()
    teardown_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await teardown_task
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED

    # The card is NOT actually free -- probe left at its original occupied
    # value (VRAM still genuinely held).

    await mgr._reconcile_orphaned_vram_credit()

    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED, (
        "the credit must be KEPT when VRAM does not appear free -- "
        "releasing it here would spawn into insufficient VRAM (the "
        "mirror-image failure mode)"
    )


@pytest.mark.asyncio
async def test_reconciler_skips_a_card_with_a_live_release_task(
    monkeypatch,
):
    """The double-payout guard's own directly-observable arm: a card whose
    credit is present but whose teardown task is STILL RUNNING (not done)
    must never be touched by the reconciler, even if the probe already
    shows the card fully free -- a live releaser might be about to pay out
    the SAME credit itself; the reconciler acting too would be the double-
    decrement hazard made real.

    This guard (the ``candidates`` computation's own
    ``not self._card_release_tasks.get(card)`` filter) is independent of the
    predicate used -- it excludes a live-task card from the sweep
    entirely, before either a delta or an absolute predicate
    ever runs. The predicate never touches this code, so this test
    checks only the filter itself."""
    probe = ScriptedProbe({0: 0})
    manifests = {"vic-0": FakeManifest(VICTIM_NEED), "miss-a": FakeManifest(NEED)}
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    mgr._vram_total_mib = [CAPACITY]
    v0 = _make_resident("vic-0", need=VICTIM_NEED,
                        state=ResidentState.IDLE_EVICTABLE)
    mgr._residents["vic-0"] = v0

    teardown_task = await _begin_evict_and_get_teardown_task(mgr, v0)
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED
    assert mgr._card_release_tasks.get(0, set()) == {teardown_task}

    # Advance the task far enough that it would be past any input-missing
    # early-continue if it were ever (wrongly) reached -- but NOT far enough
    # to finish. Capacity is explicitly populated (above) so a
    # missing-capacity keep -- not the live-task skip this test targets --
    # can never be the thing coincidentally producing the right answer here.
    await _pump_shallow(mgr, rounds=4)
    assert not teardown_task.done()
    assert mgr._vram_total_mib is not None, (
        "setup problem: capacity unavailable -- the missing-capacity guard, "
        "not the live-task guard this test targets, would be the thing "
        "skipping it, proving nothing about THIS guard"
    )

    # Card looks fully free to an outside observer -- if the reconciler
    # only checked VRAM state, it would release here. It must not, because
    # the teardown task itself is still live and owns this decrement.
    probe.set_card(0, VICTIM_NEED)
    assert not teardown_task.done()

    await mgr._reconcile_orphaned_vram_credit()

    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED, (
        "a card with a live release task must be skipped entirely, "
        "regardless of VRAM state -- this is what makes double-payout "
        "structurally impossible, not just unlikely"
    )

    # Let the real teardown finish normally afterward (own cleanup so this
    # test doesn't leak a pending task into the next one).
    await _pump(mgr)


@pytest.mark.asyncio
async def test_missing_capacity_keeps_credit_never_releases_on_absence(
    monkeypatch,
):
    """Missing capacity keeps the credit: the reconciler compares
    absolute free VRAM against the credit, which has no
    baseline to be missing -- but it has an input that can be
    missing instead: _vram_total_mib (capacity), which is None whenever
    neither live-monitor poller has populated it yet, or __init__'s
    boot-time probe found nvidia-smi unavailable. The missing input
    must be handled the way a missing baseline would be: a None
    capacity reading must read as "cannot verify", NOT as "no drop observed
    yet, keep waiting" and NEVER as grounds to release. This is the
    cheapest way to accidentally build a release-everything path --
    this pins that it
    does not exist."""
    probe = ScriptedProbe({0: 0})
    manifests = {"vic-0": FakeManifest(VICTIM_NEED), "miss-a": FakeManifest(NEED)}
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    # Deliberately NOT set -- simulating "neither poller has run yet, or
    # __init__'s boot-time probe found nvidia-smi unavailable".
    assert not hasattr(mgr, "_vram_total_mib") or mgr._vram_total_mib is None
    v0 = _make_resident("vic-0", need=VICTIM_NEED,
                        state=ResidentState.IDLE_EVICTABLE)
    mgr._residents["vic-0"] = v0

    teardown_task = await _begin_evict_and_get_teardown_task(mgr, v0)
    await _pump_shallow(mgr, rounds=4)
    assert not teardown_task.done()
    teardown_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await teardown_task
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED

    # The card looks fully free -- if a missing capacity reading were EVER
    # read as "assume cleared" or "trust the kill" (the dev-tolerance
    # shortcut _vram_verify itself uses elsewhere), this would wrongly
    # release.
    probe.set_card(0, VICTIM_NEED)

    await mgr._reconcile_orphaned_vram_credit()

    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED, (
        "a missing capacity reading must KEEP the credit -- absence of "
        "evidence must never become evidence for release"
    )


# ---------------------------------------------------------------------------
# Unit-mismatch regression (absolute predicate instead of a delta
# comparison) and the capacity invariant at the increment site.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconciler_releases_a_multi_teardown_credit_the_old_delta_check_could_not(
    monkeypatch,
):
    """The unit-mismatch scenario, reproduced directly: TWO
    teardowns accumulate credit on the SAME card before either drains (both
    die mid-finally, both orphaned -- ROUTE 2, twice, stacked). A
    delta check would compare CURRENT vram against whichever teardown's
    pre-snapshot was captured LAST -- unconditionally overwritten by the
    second teardown -- so once the SECOND teardown's (smaller-scope) snapshot
    clobbers the first's, the ACCUMULATED 2-credit total can never show
    enough of a "drop" against that undersized baseline, even once the card
    is ENTIRELY free.

    A delta check would wrongly keep a credit for a card
    that has completely cleared (needs 0.9*8000=7200, the clobbered
    baseline only shows a 5000 MiB delta). The absolute predicate
    has no baseline to clobber -- it reads the card's CURRENT state
    directly and releases correctly. Anchoring the baseline with
    setdefault/pop would also address the identical unit
    mismatch, but by a different
    mechanism.

    Asymmetric credits (2000 / 6000, not equal halves) are deliberate: each
    teardown's OWN internal _vram_verify only suspends (rather than
    resolving immediately) while the probe reads BELOW that teardown's own
    90% threshold -- v0 must stay cancellable before any probe change, and
    the probe value used to seed v1's clobbered snapshot must stay below
    v1's own threshold too, or v1 resolves on its own before it can be
    killed mid-finally, proving nothing about route 2."""
    probe = ScriptedProbe({0: 100})  # far below EITHER victim's own threshold
    manifests = {"vic-0": FakeManifest(2000), "vic-1": FakeManifest(6000),
                 "miss-a": FakeManifest(NEED)}
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    mgr._vram_total_mib = [CAPACITY]
    mgr._gpu_used_mib = lambda *, device_index=0: CAPACITY - probe.free_all()[device_index]

    v0 = _make_resident("vic-0", need=2000, state=ResidentState.IDLE_EVICTABLE)
    v1 = _make_resident("vic-1", need=6000, state=ResidentState.IDLE_EVICTABLE)
    mgr._residents["vic-0"] = v0
    mgr._residents["vic-1"] = v1

    # v0's teardown starts: pre-snapshot captured at used=9900 (free=100).
    # v0's OWN verify threshold is 0.9*2000=1800; 100 < 1800, so it suspends.
    mgr._begin_unload_locked(v0)
    t0 = next(iter(mgr._card_release_tasks[0]))
    await _pump_shallow(mgr, rounds=4)
    assert not t0.done(), "setup problem: t0 finished before it could be killed mid-finally"
    # Kill v0 NOW, before the probe moves -- otherwise v0's own suspended
    # poll would observe the later probe change and resolve on its own the
    # next time it is pumped, and this test would prove nothing about
    # ROUTE 2 for v0's share of the credit.
    t0.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await t0
    assert t0.done() and t0.cancelled()
    assert mgr._pending_reclaim_mib.get(0, 0) == 2000

    # Some of the card genuinely frees (simulating v0's SIGTERM having
    # landed) -- but only PARTIALLY, and deliberately still below v1's OWN
    # 0.9*6000=5400 threshold, so v1's own verify will ALSO suspend rather
    # than resolve immediately once it starts.
    probe.set_card(0, 5000)  # used=5000, understating the true 9900 start

    # v1's teardown starts SECOND, its OWN pre-snapshot captured NOW, AFTER
    # v0 already partially freed. A delta check that unconditionally overwrites
    # _card_pre_teardown_used_mib[0] with v1's smaller-scope 5000 reading
    # would clobber v0's earlier (truer) 9900.
    mgr._begin_unload_locked(v1)
    t1 = next(iter(mgr._card_release_tasks[0]))
    await _pump_shallow(mgr, rounds=4)
    assert not t1.done(), "setup problem: t1 finished before it could be killed mid-finally"
    t1.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await t1
    assert t1.done() and t1.cancelled()

    assert mgr._pending_reclaim_mib.get(0, 0) == 8000, (
        "both credits must be stuck and ACCUMULATED (2000 + 6000)"
    )

    # The card now genuinely, fully clears -- everything freed.
    probe.set_card(0, CAPACITY)

    await mgr._reconcile_orphaned_vram_credit()

    assert mgr._pending_reclaim_mib.get(0, 0) == 0, (
        f"a fully-freed card must release its ENTIRE accumulated credit -- "
        f"got {mgr._pending_reclaim_mib.get(0, 0)} MiB still stuck"
    )


@pytest.mark.asyncio
async def test_capacity_invariant_fires_when_credit_exceeds_card_capacity(
    monkeypatch,
):
    """PROVE the capacity invariant CAN FIRE, not merely that the
    code path is reached. Two victims evicted back-to-back on the SAME card
    accumulate PAST the card's own physical capacity (8000+8000=16000 > a
    10000 MiB test capacity) -- a card can never legitimately owe more VRAM
    than it holds.

    Without the invariant nothing stops the accumulation: credit sits at
    16000 forever (arithmetically unsatisfiable -- 0.9*16000=14400 > 10000
    capacity). With it, the invariant catches the SECOND
    increment crossing capacity and resets that card's credit to 0
    immediately, loud (log.error), not waiting on a reconcile tick.

    NOTE: the absolute predicate ALONE does NOT rescue this
    case -- free_now can never exceed capacity, so 0.9*16000 > capacity
    stays unsatisfiable under EITHER predicate. This invariant is what
    actually stops the runaway; the two mechanisms are complementary, not
    redundant."""
    probe = ScriptedProbe({0: 0})
    manifests = {"vic-0": FakeManifest(VICTIM_NEED), "vic-1": FakeManifest(VICTIM_NEED),
                 "miss-a": FakeManifest(NEED)}
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    mgr._vram_total_mib = [CAPACITY]

    v0 = _make_resident("vic-0", need=VICTIM_NEED, state=ResidentState.IDLE_EVICTABLE)
    v1 = _make_resident("vic-1", need=VICTIM_NEED, state=ResidentState.IDLE_EVICTABLE)
    mgr._residents["vic-0"] = v0
    mgr._residents["vic-1"] = v1

    mgr._begin_unload_locked(v0)
    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED  # 8000, <= capacity (10000)

    mgr._begin_unload_locked(v1)
    # 8000 + 8000 = 16000 > CAPACITY (10000) -- physically impossible; the
    # invariant must catch THIS increment and reset, not let it stand.
    assert mgr._pending_reclaim_mib.get(0, 0) == 0, (
        f"capacity invariant did not fire: credit sits at "
        f"{mgr._pending_reclaim_mib.get(0, 0)} MiB, exceeding the card's "
        f"own {CAPACITY} MiB physical capacity -- a card can never "
        f"legitimately owe more VRAM than it holds"
    )

    await _pump(mgr)  # let the two spawned teardowns run to quiescence, no leak


@pytest.mark.asyncio
async def test_capacity_invariant_survives_a_new_harness_manager_without_vram_total_mib(
    monkeypatch,
):
    """The __new__-harness landmine, proven not just
    avoided. This file and tests/test_shadow_bytematch.py build
    TurbohaulManager via __new__, bypassing __init__ entirely -- such a
    manager NEVER sets _vram_total_mib at all (not even to None; the
    attribute is simply absent). A capacity-invariant check written as
    self._vram_total_mib (no default) would raise AttributeError the
    instant _begin_unload_locked runs on a manager built this way --
    taking out every reclaim test in this file that builds its
    manager this way. This test builds exactly that manager and proves
    _begin_unload_locked does not explode.

    The guard is the getattr default: a source that replaced
    getattr(self, "_vram_total_mib", None)
    with a bare self._vram_total_mib (no default) would make this exact
    test fail with AttributeError, so the getattr default is
    load-bearing, not decorative; the test
    pins exactly that."""
    probe = ScriptedProbe({0: 0})
    manifests = {"vic-0": FakeManifest(VICTIM_NEED), "miss-a": FakeManifest(NEED)}
    mgr, fq = _make_manager(monkeypatch, probe, manifests)
    assert not hasattr(mgr, "_vram_total_mib"), (
        "setup problem: the __new__ harness must NOT set _vram_total_mib -- "
        "that absence is exactly the landmine this test proves is defused"
    )
    v0 = _make_resident("vic-0", need=VICTIM_NEED, state=ResidentState.IDLE_EVICTABLE)
    mgr._residents["vic-0"] = v0

    mgr._begin_unload_locked(v0)  # must not raise AttributeError

    assert mgr._pending_reclaim_mib.get(0, 0) == VICTIM_NEED
    await _pump(mgr)
