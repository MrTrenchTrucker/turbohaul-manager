"""The routing decision never reads a manifest on the event loop.

``_route_or_reserve`` decides, under the registry lock, how a request is routed. Its
own rule for that block is that anything that touches the disk (a manifest read, a card
probe) must run in a worker thread: the cached manifest accessor stats the file on every
call, hit or miss, and a slow disk would stall every request and every other task on the
loop. The budget function is already called through ``asyncio.to_thread``. The placement
answer (does this manifest ask for one card per engine) needs the same manifest, so it
has to be computed in that same worker thread, not in a second read on the loop.

How the tests watch it: the cached manifest accessor is wrapped by a recorder that notes
the thread each call runs on and then calls the real function. A real
``await manager._route_or_reserve(slot)`` is driven for two kinds of tag, each with one
live engine, and the tests assert two things:

  1. no recorded call ran on the loop's own thread (the failure message says why);
  2. the accessor WAS called at least once (non-vacuity: assertion 1 alone would pass if
     the routing decision stopped reading manifests at all).

The tests do not depend on the host. The box budget is exactly spent (the sidecar limit
equals the live engine count), so the budget function answers from the engine count
before any card probe; every card probe is replaced by a spy that records and fails the
call, and each test asserts the record is empty. They pass with or without a GPU tool on
the path. Everything else is real code.
"""

import asyncio
import sys
import threading
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

_TREE = Path(__file__).resolve().parents[1]
for _p in (str(_TREE / "src"), str(_TREE / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import turbohaul.manager as manager_module  # noqa: E402
from _multiinstance_support import (  # noqa: E402
    _manifest,
    _new_manager,
    _live_engine,
    _request,
)

_PROBE_SEAMS = (
    "turbohaul.manager._read_free_vram_all_mib",
    "turbohaul.manager._read_total_vram_all_mib",
    "turbohaul.safety._read_free_vram_all_mib",
    "turbohaul.safety._read_total_vram_all_mib",
)

_WHY = (
    "the routing decision read a manifest on the event loop thread: the cached accessor "
    "stats the file on every call, so that read must run in the same worker thread as the "
    "budget function (asyncio.to_thread), never directly on the loop"
)


class _Scene:
    """One manager with one live engine of a tag, a spent budget and the recorders."""

    def __init__(self, tmp_path, tag, **manifest_fields):
        self.tag = tag
        self.manager, self.boot, self.spawn_calls = _new_manager(tmp_path, budget=1)
        _manifest(self.boot, tag, **manifest_fields)
        self.engine = _live_engine(self.manager, tag, gpu=0)
        self.loop_thread = threading.get_ident()
        self.reads = []          # (thread ident, model tag) of every accessor call
        self.probe_calls = []    # card probes that ran (there must be none)

    def check_the_setup(self):
        """Fail loudly if the setup stops being a spent budget with one live engine."""
        mgr = self.manager
        assert mgr._instances_for(self.tag) == [self.engine], "exactly one live engine"
        assert len(mgr._model_residents()) == mgr.runtime.queue.max_parallel_sidecars, (
            "the box budget must be exactly spent, or the budget function may probe cards"
        )

    async def route_one_request(self):
        """Drive the real routing decision for one request, recording every manifest read."""
        real_read = manager_module.read_manifest_cached

        def recording_read(*args, **kwargs):
            self.reads.append((threading.get_ident(), args[1] if len(args) > 1 else None))
            return real_read(*args, **kwargs)

        def probe_spy(name):
            def _fail(*a, **k):
                self.probe_calls.append(name)
                raise AssertionError(f"{name} must not run on a spent box budget")
            return _fail

        slot = _request(self.tag)
        with ExitStack() as stack:
            stack.enter_context(patch.object(manager_module, "read_manifest_cached",
                                             recording_read))
            for target in _PROBE_SEAMS:
                stack.enter_context(patch(target, side_effect=probe_spy(target)))
            try:
                await self.manager._route_or_reserve(slot)
            finally:
                for t in list(self.manager._bg_tasks):
                    t.cancel()
                await asyncio.gather(*list(self.manager._bg_tasks), return_exceptions=True)
                await self.manager.shutdown()
        return slot

    def assert_no_read_on_the_loop(self):
        assert self.probe_calls == [], f"a card probe ran on a spent budget: {self.probe_calls}"
        assert self.spawn_calls == [], "no engine may be spawned while the budget is spent"
        # Non-vacuity: a test that never reads would pass the loop-thread check below.
        assert self.reads, (
            "the routing decision never called the manifest accessor, so the check below "
            "proves nothing"
        )
        on_loop = [tag for ident, tag in self.reads if ident == self.loop_thread]
        assert on_loop == [], f"{_WHY}; tags read on the loop: {on_loop}"


async def test_a_one_card_per_engine_tag_is_routed_without_a_manifest_read_on_the_loop(
        tmp_path):
    """A tag that asks for one card per engine (the placement question is answered yes)."""
    s = _Scene(tmp_path, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    s.check_the_setup()
    slot = await s.route_one_request()
    assert s.engine.inbox.get_nowait() is slot, "the request must reach the live engine"
    s.assert_no_read_on_the_loop()


async def test_a_placement_free_tag_is_routed_without_a_manifest_read_on_the_loop(tmp_path):
    """A tag with no placement intent (the placement question is answered no)."""
    s = _Scene(tmp_path, "solo", auto_place=False)
    s.check_the_setup()
    slot = await s.route_one_request()
    assert s.engine.inbox.get_nowait() is slot, "the request must reach the live engine"
    s.assert_no_read_on_the_loop()
