"""The dashboard panel VANISHES for the whole duration of a sidecar spawn.

WHY THIS TEST EXISTS, AND WHAT BREAKS IN THE PRODUCT IF IT GOES RED
-------------------------------------------------------------------
Symptom: while watching the live frontend, the panel on the dashboard
disappears for a moment during model swaps or residency swaps, with no
visible reason. An earlier fix for a related symptom does not cover it,
because that fix did not touch the code path involved here; this is a
separate defect.

It is not the earlier grace/idle_hot defect. For as long as a sidecar spawn
lasts (tens of seconds for a large model) there is a blind window in which
`residents[]` is populated while `active`, `loading`, `grace` and `idle_hot` are
ALL None. During that window the resident state is RESERVED_LOADING and the slot
state is None on every frame. That is not a flicker; it is a blank dashboard
for the whole spawn.

THE MECHANISM (manager.py `status_snapshot`):
    slot = self._resolve_top_level_active_slot()
    if slot is not None:                      # <-- the ENTIRE active/loading block
        ...                                   #     lives inside this guard
        elif state_v in {"STAGED", "PRE_LOADING", "LOADING", "READY"}:
            loading_info = {...}

While a sidecar is spawning the RESIDENT is `RESERVED_LOADING` and its
`active_slot` is still None, so the guard is False and the whole block is SKIPPED.
`loading_info` never populates. The loading branch is gated on the **SLOT** state
enum, and `RESERVED_LOADING` is a **RESIDENT** state with no SLOT counterpart --
a vocabulary mismatch, not a stranded scalar.

THE PRODUCT CONSEQUENCE, traced to the consumer rather than assumed:
the frontend's resident-synthesis module builds its panel as
    data.active ?? data.loading ?? (grace ? {...} : null) ?? (idle_hot ? {...} : null)
All four None => it yields null => the panel renders NOTHING, while the resident
list beneath it is populated. The two halves of the dashboard disagree and the
panel disappears until the spawn completes.

WHY THE EARLIER GRACE/IDLE_HOT FIX DID NOT AND COULD NOT FIX THIS: it repointed `grace` and `idle_hot`
away from permanently-unwritten manager scalars (`self.grace`, `self._idle_handle`).
Both of those fixes are real -- the grace countdown steps 59 -> 0 with
model_tag and extension_count populated.
This is a THIRD, independent hole in the same function, and no amount of fixing
grace could have closed it.

IF THIS TEST GOES RED, THE DASHBOARD GOES BLANK DURING EVERY MODEL SWAP. Anyone
proposing to break it must first explain what the FE should draw while a resident
is spawning -- "nothing" is the answer that produced the original bug report.

How the test reaches the state:
  * The state is DRIVEN, never hand-set: a real `submit()` spawns a real resident
    and the health check is held open, which is precisely what a slow real spawn
    does. The key question ("can the state this fixture CONSTRUCTS actually ARISE in
    a real configuration?") is answered by driving the real code path:
    the resident stays RESERVED_LOADING for as long as the spawn takes.
  * Ground-truth / non-vacuity control: the resident's OWN state and the absence
    of a slot are asserted from the registry BEFORE status_snapshot is consulted.
    If that control fails, the fixture never reached the state under test and the
    failure is a fixture problem, not a product defect.
  * The cap is set EXPLICITLY. The env lies (TURBOHAUL_MAX_PARALLEL=2 in a typical deployment).
"""

import asyncio

import pytest

from turbohaul.manager import ResidentState, TurbohaulManager

from _fastlane_fixture import (
    boot_ranked_runtime,
    high_vram,
    make_fakes,
    seed_manifest,
    wait_until,
)


@pytest.mark.asyncio
class TestTheDashboardDoesNotGoBlankWhileASidecarSpawns:

    @pytest.mark.parametrize("cap", [1, 2])
    async def test_loading_surfaces_while_a_resident_is_reserved_loading(self, tmp_path, cap):
        """A resident in RESERVED_LOADING must put SOMETHING on the loading surface.

        MUST FAIL ON THE UNMODIFIED TREE: with a resident genuinely spawning and
        no slot yet, `status_snapshot()["loading"]` is None because the whole
        active/loading block sits behind `if slot is not None:`.

        Parametrised over BOTH caps deliberately: cap=1 is the shipping default
        (config.py `Field(default=1, ge=1, le=32)`) and cap=2 is a common parallel setting.
        The defect is cap-general -- the guard has nothing to do with the cap.
        """
        boot, runtime = boot_ranked_runtime(
            tmp_path, max_parallel_sidecars=cap, grace_seconds=10,
        )
        assert runtime.queue.max_parallel_sidecars == cap, (
            "explicit cap -- the env lies (TURBOHAUL_MAX_PARALLEL=2 in a "
            "typical deployment); a test that inherits a default risks measuring the "
            "wrong dispatch path"
        )
        seed_manifest(boot, "m1", main_gpu=0)
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({})

        # Hold the health check open. This is EXACTLY what a slow real spawn does:
        # the resident is registered RESERVED_LOADING and stays there until the
        # sidecar answers. A real spawn of a large model can take tens of seconds.
        release_spawn = asyncio.Event()

        async def held_health(*a, **k):
            await release_spawn.wait()
            return True

        with high_vram():
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=held_health,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            submit_task = asyncio.create_task(
                mgr.submit(model_tag="m1", prompt="hi", thread_id="t1", client_meta={})
            )
            try:
                def _spawning():
                    return any(
                        r.state is ResidentState.RESERVED_LOADING
                        for r in mgr._residents.values()
                    )

                await wait_until(_spawning, timeout=5.0)

                # ---- GROUND TRUTH / NON-VACUITY CONTROL, from the registry ----
                spawning = [
                    r for r in mgr._residents.values()
                    if r.state is ResidentState.RESERVED_LOADING
                ]
                assert spawning, (
                    "precondition failed: no resident is RESERVED_LOADING -- "
                    "fixture/timing problem, not a status_snapshot defect"
                )
                assert all(r.active_slot is None for r in spawning), (
                    "precondition failed: a RESERVED_LOADING resident already has "
                    "an active_slot, so the slot-derived path would legitimately "
                    "populate the surface and this test would be vacuous"
                )

                snap = mgr.status_snapshot()

                assert snap["residents"], (
                    "precondition failed: residents[] is empty, so there is no "
                    "disagreement for the FE to render"
                )

                # ---- THE ASSERTION THE ORIGINAL BUG REPORT IS ABOUT ----
                surfaces = {k: snap.get(k) for k in ("active", "loading", "grace", "idle_hot")}
                assert not all(v is None for v in surfaces.values()), (
                    "ALL FOUR of active/loading/grace/idle_hot are None while "
                    f"residents[] has {len(snap['residents'])} entry(ies) in state "
                    "RESERVED_LOADING. The frontend's resident-synthesis code builds its panel as "
                    "active ?? loading ?? grace ?? idle_hot, so it will render "
                    "NOTHING and the dashboard panel DISAPPEARS for the whole "
                    "spawn. A real spawn can take tens of seconds; this "
                    "is the dashboard blip."
                )
                assert snap["loading"] is not None, (
                    "status_snapshot()['loading'] is None while a resident is "
                    "genuinely RESERVED_LOADING with no slot. loading is the "
                    "field that should describe a spawning resident."
                )
                assert snap["loading"].get("model_tag") == "m1", (
                    f"loading surface names {snap['loading'].get('model_tag')!r}, "
                    "expected the spawning model 'm1' -- the FE prints this"
                )
                assert isinstance(snap["loading"].get("elapsed_s"), (int, float)), (
                    "loading.elapsed_s must be a number -- the FE renders it as "
                    "the spawn progress readout"
                )
            finally:
                release_spawn.set()
                submit_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(submit_task), timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
                await mgr.shutdown()
