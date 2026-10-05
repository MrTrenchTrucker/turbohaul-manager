"""The dashboard must not go blank while a card still holds a model.

WHY THIS TEST EXISTS, AND WHAT BREAKS IF IT GOES RED
-----------------------------------------------------
The symptom: the dashboard shows nearly full VRAM on one GPU and no resident card:

    RESIDENTS                                   slots 0/2
      VRAM
      GPU 0    22,000 / 24,000 MiB used     <- nearly full
      GPU 1       500 / 24,000 MiB used
    LIVE INFERENCE  -- no active generation --

A card holding a large model, and NO resident card at all. Expected behaviour: when a card is
loaded the dashboard must keep a resident box visible, showing that the new model is
loading rather than disappearing entirely. If any card is loaded with a model, the
panel must show it.

THE WINDOW, as seen in the manager log during a swap (illustrative times):
    t+0s  MAKE_ROOM_EVICTION model-a-35b   -> del self._residents[...]
    t+2s  KV_SAVE_DECLINE ... reap_resident_handle
    t+4s  spawning llama-server <model>...    -> next reservation, SECONDS LATER
Between the del (in manager.py) and the next reservation the registry is
EMPTY while the dying process still holds the card. `residents[]` is `[]`, every top-level
surface is None, and `synthesizeResident.ts` -- which is
`active ?? loading ?? grace ?? idle_hot` -- yields null and renders nothing.

WHY THE RESIDENT-BASED LOADING SURFACE DOES NOT COVER IT: it resolves a RESIDENT OBJECT in
RESERVED_LOADING. Here there is no resident object at all, so it has nothing to describe.

★ AND WHY A NON-EMPTY-REGISTRY INVARIANT MISSES IT: "residents NON-EMPTY implies
not all surfaces None" says NOTHING about residents being EMPTY -- which is precisely
this case. The hole is in how such an invariant is framed, not only in the code. This test is
the empty-registry half of that invariant.

Test class: behaviour change (regression test).
  * The question "can the state this fixture CONSTRUCTS actually ARISE in the deployed
    configuration?" -- yes, it can: the symptom above is that state, and a
    periodic sampler sees zero-resident windows of seconds
    during swaps. This is not a hypothetical state.
  * HONEST LIMIT, stated rather than hidden: the VRAM *reading* is injected. A real
    dying-process window cannot be held open in-process. What is NOT injected is the thing
    under test -- the registry is genuinely empty because nothing was ever registered, and
    `_residents_snapshot()` is asserted empty as a ground-truth control before
    `status_snapshot` is consulted.
  * The cap is set EXPLICITLY. The env can mislead (TURBOHAUL_MAX_PARALLEL=2 in some deployments,
    the config default is 1).
"""

import asyncio

import pytest

from turbohaul.manager import TurbohaulManager

from _fastlane_fixture import (
    boot_ranked_runtime,
    drive_to_active,
    high_vram,
    make_fakes,
    resident_for,
    seed_manifest,
)


@pytest.mark.asyncio
class TestTheDashboardDoesNotGoBlankWhileACardHoldsAModel:

    @pytest.mark.parametrize("cap", [1, 2])
    async def test_transition_surfaces_when_registry_is_empty_but_vram_is_held(self, tmp_path, cap):
        """Registry empty + card still loaded  =>  the panel must show SOMETHING.

        MUST FAIL WITHOUT THE FIX: with no residents, every surface is None and the
        FE renders nothing, which is exactly the symptom.

        Parametrised over both caps: cap=1 is the shipping default (see config.py) and cap=2
        is a common deployed setting. The teardown/reservation gap is not cap-specific.
        """
        boot, runtime = boot_ranked_runtime(tmp_path, max_parallel_sidecars=cap, grace_seconds=10)
        assert runtime.queue.max_parallel_sidecars == cap, (
            "explicit cap -- the env can mislead; a test that inherits a default risks measuring "
            "the wrong dispatch path"
        )
        seed_manifest(boot, "m1", main_gpu=0)
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({})
        with high_vram():
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            try:
                # ---- GROUND TRUTH: the registry really is empty. Not hand-emptied. ----
                assert not mgr._residents_snapshot(), (
                    "precondition failed: residents is not empty, so this is not the window "
                    "under test -- fixture problem, not a result"
                )

                # The ONE injected value: the VRAM *reading* a real prober would return for
                # a card still holding a large model. Mirrors the symptom
                # (GPU0 nearly full; GPU1 nearly free).
                mgr._vram_total_mib = [24467, 24467]
                mgr._vram_free_mib = [24467 - 22078, 24467 - 497]
                # What a REAL teardown records immediately before del self._residents[...]
                # destroys the entry. Not a display value -- the manager genuinely captures
                # this in manager.py, because after the del the outgoing
                # tag is unrecoverable.
                mgr._last_unloaded = {"model_tag": "outgoing-model", "main_gpu": 0}

                snap = mgr.status_snapshot()

                assert not snap["residents"], (
                    "precondition failed: residents[] is non-empty in the snapshot"
                )

                surfaces = {k: snap.get(k) for k in ("active", "loading", "grace", "idle_hot")}
                assert not all(v is None for v in surfaces.values()), (
                    "ALL FOUR of active/loading/grace/idle_hot are None while a GPU still "
                    "holds 22,078 MiB. synthesizeResident.ts is "
                    "`active ?? loading ?? grace ?? idle_hot`, so it yields null and the "
                    "dashboard panel DISAPPEARS while the VRAM bar still shows the card "
                    "nearly full. This is the symptom."
                )
                assert snap["loading"] is not None, (
                    "the loading surface is the one the FE draws during a load; it must "
                    "describe the transition when the registry is empty but a card is held"
                )
                assert snap["loading"].get("transition") is True, (
                    "the entry must mark itself a TRANSITION so a reader cannot mistake it "
                    "for a normal resident load"
                )
                # ---- THE LOGIC GATE ----
                # The panel keeps showing the old model name until the VRAM drops, even
                #  once it knows a new model is incoming.
                #  (The old name is the one on the card until then.)
                # The card is still holding the OUTGOING model, so that is the name the
                # panel must show. Naming the INCOMING model here would name a model that
                # is not on the card yet -- the opposite of the intended behaviour.
                assert snap["loading"].get("model_tag") == "outgoing-model", (
                    f"loading.model_tag={snap['loading'].get('model_tag')!r} -- during the "
                    "gap the card still holds the OUTGOING model, and that is the name the "
                    "panel must show until VRAM drops."
                )
                assert snap["loading"].get("state") == "UNLOADING", (
                    "state must read UNLOADING when the outgoing model is known, so a reader "
                    "cannot mistake it for a fresh load"
                )
                held = snap["loading"].get("vram_held_mib") or []
                assert any(h > 1024 for h in held), (
                    f"vram_held_mib={held!r} does not show a loaded card; the FE needs the "
                    "held figure to say WHICH card is still occupied"
                )
            finally:
                await mgr.shutdown()

    async def test_does_not_fire_while_a_resident_actually_exists(self, tmp_path):
        """NEGATIVE CONTROL #2 — pins the registry guard.

        THE GUARD: the transition entry ONLY fills a gap --
        if anything above already claimed active/loading, it is
        a no-op. The active/loading half IS pinned by the tests above. THE
        REGISTRY-IS-EMPTY HALF is pinned HERE: deleting
        `and not self._residents_snapshot()` from the guard would otherwise leave the whole suite GREEN.

        WHAT THAT WOULD COST IN THE PRODUCT: with the registry guard gone, the transition
        entry can populate while residents genuinely EXIST -- any case where a resident is
        present but active/loading failed to resolve, which is EXACTLY the loading-window case if
        its resolver ever misses. The dashboard would then say "swap in progress" over a
        live, serving model. That is a worse lie than the blank panel this whole fix exists
        to stop: blank is obviously wrong, whereas a confident wrong caption is not.

        A PROMISE IN A COMMENT WITH NO INSTRUMENT BEHIND IT IS NOT A GUARANTEE. This is the
        instrument.
        """
        boot, runtime = boot_ranked_runtime(tmp_path, max_parallel_sidecars=2, grace_seconds=10)
        seed_manifest(boot, "m1", main_gpu=0)
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({})
        with high_vram():
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            try:
                # A card genuinely loaded, AND a genuinely present resident. Both true at
                # once is the state the registry guard exists to exclude.
                mgr._vram_total_mib = [24467, 24467]
                mgr._vram_free_mib = [24467 - 22078, 24467 - 497]
                mgr._last_unloaded = {"model_tag": "outgoing-model", "main_gpu": 0}

                # The dispatcher must actually be RUNNING for drive_to_active to reach
                # ACTIVE. Without it the precondition control fires: the fixture
                # never gets to the state under test, and the control correctly says so
                # instead of letting a fixture failure read as a result.
                mgr._worker_task = asyncio.create_task(mgr.worker_loop())

                await drive_to_active(
                    mgr, "m1", thread_id="t-resident", client_meta={}, timeout=5.0,
                )
                # GROUND TRUTH: the resident is really there. If this fails the fixture never
                # reached the state under test and the result below means nothing.
                assert mgr._residents_snapshot(), (
                    "precondition failed: no resident present, so this test cannot "
                    "distinguish the guarded case -- fixture problem, not a result"
                )

                snap = mgr.status_snapshot()
                li = snap.get("loading")
                assert not (isinstance(li, dict) and li.get("transition") is True), (
                    f"loading={li!r} -- the TRANSITION entry fired while a resident is "
                    "actually present. The dashboard would caption a live, serving model as "
                    "'swap in progress'. The `not self._residents_snapshot()` guard is "
                    "load-bearing and this test pins it."
                )
            finally:
                await mgr.shutdown()

    async def test_a_real_teardown_actually_captures_the_outgoing_tag(self, tmp_path):
        """Pins the CAPTURE half, not just the rendering half.

        THE GAP: neutering BOTH _last_unloaded capture sites (assigned None at each)
        would not fail the other tests. Those
        tests INJECT `mgr._last_unloaded` and then assert on it, so they pin only the
        RENDERING half: GIVEN the value is populated, the transition entry reports it. Nothing
        proved the teardown actually WRITES it.

        AND THE CAPTURE IS THE LOAD-BEARING HALF AS ITS OWN DESIGN NOTE SAYS: "after
        that del the tag is UNRECOVERABLE, so it is captured there or not at all." Move or
        drop either capture in a future teardown refactor and the suite stays green while the
        design rule -- the panel shows the old model name until the VRAM drops -- silently
        stops working. That is a promise in a comment with no instrument behind it,
        one level down from the registry guard.

        THIS TEST EXERCISES THE PATH INSTEAD OF SUPPLYING ITS OUTPUT. It drives a real
        resident to ACTIVE and then puts it through the real unload, and asserts the tag was
        captured on the way past. Nothing about _last_unloaded is injected.
        """
        boot, runtime = boot_ranked_runtime(tmp_path, max_parallel_sidecars=2, grace_seconds=10)
        seed_manifest(boot, "m1", main_gpu=0)
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({})
        with high_vram():
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await drive_to_active(
                    mgr, "m1", thread_id="t-teardown", client_meta={}, timeout=5.0,
                )
                r = resident_for(mgr, "m1")
                # GROUND TRUTH before the act: the resident is really registered, and nothing
                # has written _last_unloaded yet. If either fails, the fixture never reached
                # the state under test and the result below means nothing.
                assert r is not None and mgr._residents_snapshot(), (
                    "precondition failed: no live resident to tear down"
                )
                assert getattr(mgr, "_last_unloaded", None) is None, (
                    "precondition failed: _last_unloaded is already populated BEFORE the "
                    "teardown, so a pass below would prove nothing about the capture"
                )

                # ---- THE REAL UNLOAD PATH. Not injected. ----
                async with mgr._registry_lock:
                    mgr._begin_unload_locked(r)

                captured = getattr(mgr, "_last_unloaded", None)
                assert captured is not None, (
                    "the teardown did NOT capture the outgoing tag. After `del "
                    "self._residents[...]` that tag is UNRECOVERABLE, so the dashboard can "
                    "never name the model still sitting on the card, and the design rule "
                    "-- show the old model name until VRAM drops -- silently stops working."
                )
                assert captured.get("model_tag") == "m1", (
                    f"captured {captured!r} -- the outgoing model tag must be the one that was "
                    "actually resident"
                )
                assert not mgr._residents_snapshot(), (
                    "the resident should be gone from the registry after the unload; if it is "
                    "still there this test is not exercising the window it claims to"
                )
            finally:
                await mgr.shutdown()

    async def test_does_not_fire_on_a_genuinely_idle_box(self, tmp_path):
        """NEGATIVE CONTROL -- the green must be reachable and the fix must not over-fire.

        An idle host with EMPTY cards must NOT get a phantom transition entry. Without this,
        the fix could satisfy the test above by always populating `loading`, which would put
        a permanent fake 'swapping' panel on an idle dashboard.
        """
        boot, runtime = boot_ranked_runtime(tmp_path, max_parallel_sidecars=2, grace_seconds=10)
        seed_manifest(boot, "m1", main_gpu=0)
        spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({})
        with high_vram():
            mgr = TurbohaulManager(
                boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
                sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
            )
            try:
                # idle-GPU readings: only a few MiB used
                mgr._vram_total_mib = [24467, 24467]
                mgr._vram_free_mib = [24467 - 2, 24467 - 12]
                snap = mgr.status_snapshot()
                assert snap["loading"] is None, (
                    f"loading={snap['loading']!r} on an IDLE host with empty cards -- the "
                    "transition entry is over-firing and would paint a permanent phantom "
                    "'swapping' panel on an idle dashboard"
                )
            finally:
                await mgr.shutdown()
