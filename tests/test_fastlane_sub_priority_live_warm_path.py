"""Does a better-ranked request that needs a model swap run before a worse-ranked
request of the SAME client whose model is already loaded (the warm path)?

Everything runs through the real manager and its real dispatcher loop: only the
sidecar process, health probe, teardown and completion call are faked, and the
GPU free-memory probe is a model of ONE card whose free memory shrinks by the
size of every loaded model. Two models are sized so that only one fits on the
card; each test asserts that, and records that two models were never loaded at
the same time.

Free VRAM: every test pins BOTH `turbohaul.safety._read_free_vram_all_mib` and
`turbohaul.manager._read_free_vram_all_mib` to the one-card model below, because the
capacity decision reads the manager's own binding.

Expected direction asserted throughout: for one client (one rule, main=1,
curator=3) the main request (rank 1) runs before the curator request (rank 3),
even when the curator request is the one that would be served warm.

Code paths that decide the outcome (src/turbohaul):
  _pick_fastlane_locked (queue) picks among waiting listed slots by
                  (rule_index, rank, arrival); this is the pop_next pick.
  _is_designated_unload_target_locked (manager) decides whether a resident may
                  be unloaded for a waiting claim; with the full (rule_index,
                  rank) key a better-ranked claim of the resident's own client
                  designates it, which ends its grace window.
  the grace loop (manager) checks that designation on every pass, before it
                  looks for a same-thread follow-up.
Rule: within ONE client the better (lower) tag rank is served first, including
ahead of a same-thread follow-up of a worse rank whose resident is in its grace
window (the claim ends the window); the worse-ranked resident of a client is
unloaded before its better-ranked one; a running turn is never cut; across
clients rank is never compared.
"""
import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
import yaml

from turbohaul.config import (
    BootConfig, FastLaneConfig, FastLaneRule, FastLaneTagRanks, PullConfig,
    QueueConfig, RuntimeConfig, RuntimePathsConfig, ServerConfig,
    StorageConfig, UIConfig,
)
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle

TAG_MAIN = "model-main"
TAG_CURATOR = "model-curator"
CARD_MIB = 100_000
NEED_MIB = 60_000  # two of these never fit on one card
IP_A = "192.0.2.10"
IP_B = "192.0.2.20"
MAIN_META = {"ip": IP_A, "is_main": True}
CURATOR_META = {"ip": IP_A, "is_curator": True}
LOADED = (ResidentState.ACTIVE, ResidentState.GRACE, ResidentState.IDLE_EVICTABLE)


class World:
    """A manager on a one-card box plus a record of what ran and what was loaded."""

    def __init__(self, tmp_path, rules, grace_seconds):
        self.started = []  # (label, model_tag) in the order completions began
        self.max_loaded = 0
        self.loaded_seen = set()
        self.hold = {}  # label -> asyncio.Event a completion waits on
        root = tmp_path / "state"
        root.mkdir()
        for d in ("blobs", "manifests", "import-staging"):
            (root / d).mkdir()
        for tag in (TAG_MAIN, TAG_CURATOR):
            (root / "manifests" / f"{tag}.yaml").write_text(yaml.safe_dump({
                "model_tag": tag, "gguf_blob_sha256": "a" * 64,
                "gguf_size_bytes": 0, "context_size": 2048,
                "expected_vram_bytes": NEED_MIB * 1024 * 1024,
                "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
            }))
        self.boot = BootConfig(
            server=ServerConfig(),
            storage=StorageConfig(
                blob_store_path=root / "blobs", manifests_path=root / "manifests",
                import_allowed_root=root / "import-staging",
                state_db_path=root / "state.sqlite",
            ),
            runtime=RuntimePathsConfig(
                llama_server_binary=tmp_path / "fake_llama_server",
                default_port_base=59900,
            ),
            ui=UIConfig(static_path=tmp_path / "ui_dist"),
        )
        self.runtime = RuntimeConfig(
            queue=QueueConfig(
                safety_enabled=False, max_parallel_sidecars=2,
                grace_seconds=grace_seconds, max_grace_extensions=50,
                idle_hot_load_seconds=120, drained_sigterm_window_active_s=1,
                drained_sigterm_window_cold_s=1, loading_health_timeout_s=10,
            ),
            pull=PullConfig(),
            fastlane=FastLaneConfig(enabled=True, rules=rules, max_normal_wait_s=3600.0),
        )
        pid = [70000]

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
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
            return True, 100

        async def fake_complete(slot, handle):
            self.started.append((slot.prompt, handle.model_tag))
            hold = self.hold.get(slot.prompt)
            if hold is not None:
                await hold.wait()
            return {"ok": True}

        self.mgr = TurbohaulManager(
            self.boot, self.runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        self.task = None
        self.sampler = None

    def free_mib(self):
        """One card: capacity minus every model that is already loaded."""
        used = sum(NEED_MIB for r in self.mgr._model_residents() if r.state in LOADED)
        return [CARD_MIB - used]

    def loaded_tags(self):
        return {r.model_tag for r in self.mgr._model_residents() if r.state in LOADED}

    async def _sample(self):
        while True:
            tags = self.loaded_tags()
            self.loaded_seen |= tags
            self.max_loaded = max(self.max_loaded, len(tags))
            await asyncio.sleep(0.002)

    def start(self):
        self.task = asyncio.create_task(self.mgr.worker_loop())
        self.mgr._worker_task = self.task
        if self.sampler is None:
            self.sampler = asyncio.create_task(self._sample())

    async def pause_dispatcher(self, attempts=20):
        """Stop the dispatcher loop and prove it stopped.

        A single cancel() can be lost: when the loop's bounded wait on its wake
        event completes in the same event-loop turn as the cancel, the cancel is
        absorbed and the loop keeps running (and keeps popping the queue, so
        the "paused" staging would not be staged at all). The cancel is therefore
        repeated until the task is observed done, and a dispatcher that still runs
        after `attempts` tries fails by assertion instead of waiting forever."""
        for _ in range(attempts):
            if self.task.done():
                break
            self.task.cancel()
            await asyncio.wait({self.task}, timeout=0.5)
        assert self.task.done(), (
            f"the dispatcher loop was still running after {attempts} cancels; "
            f"started={self.started}")
        try:
            await self.task
        except asyncio.CancelledError:
            pass

    async def close(self):
        """Tear down without being able to hang: release every held completion,
        stop the sampler, give shutdown a bounded time and cancel it (without
        awaiting the cancel) if it overruns."""
        for ev in self.hold.values():
            ev.set()
        if self.sampler is not None:
            self.sampler.cancel()
        stopper = asyncio.ensure_future(self.mgr.shutdown())
        try:
            done, _ = await asyncio.wait({stopper}, timeout=10)
        except BaseException:
            stopper.cancel()
            raise
        if not done:
            stopper.cancel()
        else:
            try:
                stopper.result()
            except Exception:
                pass

    async def send(self, label, tag, meta, thread):
        return await self.mgr.submit(
            model_tag=tag, prompt=label, thread_id=thread, client_meta=dict(meta))

    async def until(self, predicate, what, timeout=30.0):
        t0 = time.monotonic()
        while not predicate():
            if time.monotonic() - t0 > timeout:
                raise AssertionError(
                    f"timed out waiting for {what}; started={self.started} "
                    f"loaded={sorted(self.loaded_tags())}")
            await asyncio.sleep(0.005)

    async def warm(self, label, tag, meta, thread, state):
        """Run one request to completion and wait until its model is parked in `state`
        (the grace window is an ACTIVE resident whose grace loop is running)."""
        await self.send(label, tag, meta, thread)
        await self.until(lambda: any(s[0] == label for s in self.started), f"{label} to run")

        def parked(r):
            if state == "GRACE":
                return r.state is ResidentState.ACTIVE and r.in_grace_loop
            return r.state is state
        await self.until(
            lambda: any(r.model_tag == tag and parked(r) for r in self.mgr._model_residents()),
            f"{tag} to reach {state}")

    async def second_model_admitted(self):
        """The real capacity check's answer for one more model while exactly one is loaded."""
        loaded = self.loaded_tags()
        assert len(loaded) == 1, f"expected exactly one loaded model, got {sorted(loaded)}"
        async with self.mgr._registry_lock:
            return self.mgr._vram_admits_locked(NEED_MIB, 1, 0, "none")


def _world(tmp_path, *, rules=None, grace_seconds=0):
    rules = rules or [FastLaneRule(address=IP_A, tag_ranks=FastLaneTagRanks(
        main=1, curator=3, unclassified=5))]
    return World(tmp_path, rules, grace_seconds)


def _first(w, labels):
    """Of the labels, the one whose completion started first (None if none did)."""
    for label, _tag in w.started:
        if label in labels:
            return label
    return None


async def _race(w, order, *, warm_tag, warm_label, warm_meta, warm_thread, state,
                cur_thread="cur-thread", cur_meta=CURATOR_META, pause=True,
                hold_until_both_waiting=False):
    """Warm one model, submit both waiters in `order` and report who ran first.
    With `pause` the dispatcher is stopped while they are submitted, so both are
    staged together before it pops; without it the dispatcher runs throughout.
    With `hold_until_both_waiting` the dispatcher and the grace loop keep running,
    but the queue's two removal methods (pop_matched_thread, pop_next) answer
    "nothing" until both requests are observed waiting in the queue, so neither
    request can be served before the other has even arrived; the setup is
    asserted, then the real methods are restored and decide who runs first."""
    with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=w.free_mib), \
         patch("turbohaul.manager._read_free_vram_all_mib", side_effect=w.free_mib):
        w.start()
        await w.warm(warm_label, warm_tag, warm_meta, warm_thread, state)
        admits = await w.second_model_admitted()
        assert admits is False, "the real capacity check admitted a second model: the box holds both"
        w.hold["cur"] = asyncio.Event()
        w.hold["main"] = asyncio.Event()
        if pause:
            await w.pause_dispatcher()
        sends = {
            "cur": lambda: w.send("cur", TAG_CURATOR, cur_meta, cur_thread),
            "main": lambda: w.send("main", TAG_MAIN, MAIN_META, "main-thread"),
        }
        q = w.mgr.queue
        if hold_until_both_waiting:
            real_matched, real_next = q.pop_matched_thread, q.pop_next
            released = asyncio.Event()

            async def held_matched(*a, **k):
                if not released.is_set():
                    return None
                return await real_matched(*a, **k)

            async def held_next(*a, **k):
                if not released.is_set():
                    return None
                return await real_next(*a, **k)

            q.pop_matched_thread, q.pop_next = held_matched, held_next
        try:
            for name in order:
                await sends[name]()
            if hold_until_both_waiting:
                waiting = {s.prompt for s in list(q._staging) + list(q._accept_buf)}
                assert {"cur", "main"} <= waiting, (
                    f"setup: both requests must be waiting before either can be "
                    f"served; waiting={sorted(waiting)} started={w.started}")
                assert _first(w, {"cur", "main"}) is None, (
                    f"setup: a request was served before both were waiting; "
                    f"started={w.started}")
        finally:
            if hold_until_both_waiting:
                q.pop_matched_thread, q.pop_next = real_matched, real_next
                released.set()
        if pause:
            w.start()
        await w.until(lambda: _first(w, {"cur", "main"}) is not None, "either waiter to run",
                      timeout=60.0)
        first = _first(w, {"cur", "main"})
        w.hold["cur"].set()
        w.hold["main"].set()
        await w.until(lambda: {"cur", "main"} <= {s[0] for s in w.started},
                      "both waiters to run", timeout=60.0)
        print(f"order={order} started={w.started} max_loaded={w.max_loaded}")
        assert w.max_loaded <= 1, f"two models were loaded at once (max_loaded={w.max_loaded})"
        return first


@pytest.mark.asyncio
class TestWarmPathOneClient:
    @pytest.mark.parametrize("order", [("cur", "main"), ("main", "cur")])
    async def test_warm_curator_does_not_run_before_better_ranked_main(self, tmp_path, order):
        """The loaded, idle curator model (rank 3) is expected to yield to the main
        request (rank 1) that needs the swap; the dispatcher's pick orders by
        (rule_index, rank, arrival), so the main request is expected first and the
        card swaps to its model."""
        w = _world(tmp_path)
        try:
            first = await _race(
                w, order, warm_tag=TAG_CURATOR, warm_label="warmup",
                warm_meta=CURATOR_META, warm_thread="cur-thread",
                state=ResidentState.IDLE_EVICTABLE)
            assert first == "main", (
                f"arrival order {order}: the {first!r} request ran first; started={w.started}")
        finally:
            await w.close()

    @pytest.mark.parametrize("order", [("cur", "main"), ("main", "cur")])
    async def test_curator_follow_up_in_grace_does_not_run_before_better_ranked_main(
            self, tmp_path, order):
        """The loaded curator model sits in its grace window after a turn of the same
        thread the curator's follow-up uses. Within one client the better rank goes first:
        the main request (rank 1) designates the curator resident, which ends its grace
        window, so the main request is expected to run before the worse-ranked (rank 3)
        same-thread follow-up; the follow-up is served after the main request."""
        w = _world(tmp_path, grace_seconds=10)
        try:
            first = await _race(
                w, order, warm_tag=TAG_CURATOR, warm_label="warmup",
                warm_meta=CURATOR_META, warm_thread="cur-thread", state="GRACE",
                pause=False, hold_until_both_waiting=True)
            assert first == "main", (
                f"arrival order {order}: the {first!r} request ran first; started={w.started}")
        finally:
            await w.close()


@pytest.mark.asyncio
class TestWarmPathControls:
    async def test_control_warm_main_runs_before_cold_curator(self, tmp_path):
        """Control: the loaded model is the main's (rank 1) and the cold waiter is the
        curator's (rank 3), so the warm main request is expected first (the pick orders by rank);
        proves the instrument reports a served-first request."""
        w = _world(tmp_path)
        try:
            with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=w.free_mib), \
                 patch("turbohaul.manager._read_free_vram_all_mib", side_effect=w.free_mib):
                w.start()
                await w.warm("warmup", TAG_MAIN, MAIN_META, "main-thread",
                             ResidentState.IDLE_EVICTABLE)
                assert await w.second_model_admitted() is False, 'both models fit'
                await w.pause_dispatcher()
                await w.send("cur", TAG_CURATOR, CURATOR_META, "cur-thread")
                await w.send("main", TAG_MAIN, MAIN_META, "main-thread")
                w.start()
                await w.until(lambda: _first(w, {"cur", "main"}) is not None, "a waiter to run")
                first = _first(w, {"cur", "main"})
                await w.until(lambda: {"cur", "main"} <= {s[0] for s in w.started},
                              "both waiters to run", timeout=60.0)
            print(f"started={w.started} max_loaded={w.max_loaded}")
            assert first == "main", f"the {first!r} request ran first; started={w.started}"
            assert w.max_loaded <= 1, f"two models loaded at once (max_loaded={w.max_loaded})"
        finally:
            await w.close()

    async def test_control_better_rule_wins_the_swap(self, tmp_path):
        """Control: the cold waiter is a better RULE (rule_index 0) than the warm
        resident's client (rule_index 1) with equal ranks, so it is expected to win the
        swap (the pick compares rule_index first); proves the instrument sees
        rule_index."""
        rules = [
            FastLaneRule(address=IP_A, tag_ranks=FastLaneTagRanks(main=1, curator=3)),
            FastLaneRule(address=IP_B, tag_ranks=FastLaneTagRanks(main=1, curator=3)),
        ]
        w = _world(tmp_path, rules=rules)
        try:
            with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=w.free_mib), \
                 patch("turbohaul.manager._read_free_vram_all_mib", side_effect=w.free_mib):
                w.start()
                warm_meta = {"ip": IP_B, "is_curator": True}
                await w.warm("warmup", TAG_CURATOR, warm_meta, "b-thread",
                             ResidentState.IDLE_EVICTABLE)
                assert await w.second_model_admitted() is False, 'both models fit'
                await w.pause_dispatcher()
                await w.send("cur", TAG_CURATOR, warm_meta, "b-thread")
                await w.send("main", TAG_MAIN, {"ip": IP_A, "is_main": True}, "a-thread")
                w.start()
                await w.until(lambda: _first(w, {"cur", "main"}) is not None, "a waiter to run")
                first = _first(w, {"cur", "main"})
                await w.until(lambda: {"cur", "main"} <= {s[0] for s in w.started},
                              "both waiters to run", timeout=60.0)
            print(f"started={w.started} max_loaded={w.max_loaded}")
            assert first == "main", f"the {first!r} request ran first; started={w.started}"
            assert w.max_loaded <= 1, f"two models loaded at once (max_loaded={w.max_loaded})"
        finally:
            await w.close()
