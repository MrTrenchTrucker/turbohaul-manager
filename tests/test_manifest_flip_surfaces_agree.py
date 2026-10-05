"""The removal reason for the "serving beats grace" guard, MEASURED
ACROSS A MANIFEST FLIP (a reason must survive a flag flip).

WHY THIS FILE EXISTS. `_resident_phase` deliberately carries NO
`r.state is ACTIVE and r.inflight` precedence guard. Adding one would leave
every other test green, so this file pins the reason. The recorded
removal reason is SURFACES-AGREE: the top-level `status_snapshot()` already reports
grace in this window, so a per-resident precedence would make the two surfaces
DISAGREE -- which is the defect the current design avoids.

⛔ NO SUCH GUARD EXISTS, WHICH IS THE REASON THIS FILE IS A FILE. The guard is absent
from the code; the only
line pairing `active` and `inflight` is PROSE IN A DOCSTRING.
No code path implements it. So the reason cannot be checked against
the tree by reading it; it can only be MEASURED on the artifact. That is what this does.

⛔ AND THE WEAKER REASON IS THE ONE THAT FAILS. "It never fires" is contingent:
`r.inflight`'s writer chain is LIVE, gated on a manifest flag, so "never" is a claim
about a DEPLOYMENT, not about the code. A reason must survive a flag flip. These
tests flip it and re-measure.

THE CHAIN THAT RESURRECTS `r.inflight` (each leg verified, not assumed):
    manifest `llama_server_flags.parallel: N`
      -> `manifest.flags_to_argv` emits `--parallel N`          (manifest.py)
      -> `spawn_sidecar` scans that argv and PINS `handle.parallel` (subprocess_mgr.py)
      -> `n_parallel = max(1, handle.parallel)`                  (manager.py)
      -> `if n_parallel > 1: _fan_out_on_resident(...)`          (manager.py)
      -> `r.inflight.append(s)` in `_launch`                     (manager.py)

⚠ TWO CAVEATS, both measured here:
  1. `manifest_parallel` (in manager.py) is NOT the reader that resurrects
     the guard -- it feeds the VRAM SAFETY GATE (`all_safety_gates(parallel=...)`).
     `handle.parallel` is pinned from the spawn ARGV and, per SidecarHandle's own
     docstring, "never a later manifest read".
  2. `parallel: 2` ALONE IS NOT A VALID MANIFEST. The validator rejects it:
     "parallel=2 requires kv_unified: true". The real-world flip is a PAIR of flags.
     A test that flipped only `parallel` would never boot -- and a fixture that never
     boots reports no inflight for a reason that has nothing to do with the guard.
"""
import asyncio
import time
from unittest.mock import MagicMock

import pytest
import yaml

from tests._fastlane_fixture import (
    boot_ranked_runtime,
    high_vram,
    make_fakes,
    resident_for,
    wait_until,
)
from turbohaul.manager import ResidentState, TurbohaulManager, _resident_phase
from turbohaul.subprocess_mgr import SidecarHandle

GRACE_SECONDS = 30.0   # long enough that the window cannot close mid-sample


def _seed_manifest_with_parallel(boot, model_tag, parallel, *, main_gpu=0):
    """The shared fixture's `seed_manifest` shape plus the flag under test.

    Written HERE, not by extending the shared fastlane fixture module: that fixture
    is shared with other tests and is not changed for this measurement.
    """
    flags = {"split_mode": "none", "main_gpu": main_gpu, "parallel": parallel}
    if parallel > 1:
        flags["kv_unified"] = True      # MEASURED requirement -- see module docstring
    (boot.storage.manifests_path / f"{model_tag}.yaml").write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 0,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "llama_server_flags": flags,
    }))


def _parallel_aware_fakes(gates, captured_argv):
    """`make_fakes()`, but the handle's `parallel` is pinned FROM THE ARGV.

    The shared fixture's `fake_spawn` builds `SidecarHandle(...)` on the default
    `parallel=1` and drops the argv, which would make the flip UNOBSERVABLE -- the
    measurement would return 0 for a fixture reason and read as a result about the
    guard. This reproduces `spawn_sidecar`'s `--parallel` scan (subprocess_mgr.py)
    and nothing else. The manifest->argv leg is NOT faked: the real argv the manager
    built is captured and asserted on.
    """
    _spawn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes(gates)
    pid = [90500]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        captured_argv.append(list(argv))
        parallel = 1
        for i, tok in enumerate(argv):
            if tok == "--parallel" and i + 1 < len(argv):
                try:
                    parallel = max(1, int(argv[i + 1]))
                except (TypeError, ValueError):
                    parallel = 1
        pid[0] += 1
        proc = MagicMock()
        proc.pid = pid[0]
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag,
                             parallel=parallel)

    return fake_spawn, health_fn, sigterm_fn, vram_fn, complete_fn


async def _poll_max_inflight(r, ticks=60, interval=0.005):
    """`r.inflight` is TRANSIENT -- it is cleared in `_fan_out_on_resident`'s
    `finally`. A single sample that reads 0 cannot distinguish "never populated"
    from "sampled between append and clear", so take the MAX over a window."""
    mx = 0
    for _ in range(ticks):
        mx = max(mx, len(r.inflight))
        await asyncio.sleep(interval)
    return mx


async def _measure(tmp_path, parallel):
    """Drive turn 1 (held, then completed -> grace opens) and turn 2 (a follow-up
    HELD inside that open grace window), sampling both surfaces throughout."""
    boot, runtime = boot_ranked_runtime(
        tmp_path,
        max_parallel_sidecars=2,        # EXPLICIT: the bug's precondition is cap>=2
        grace_seconds=GRACE_SECONDS,
        idle_hot_load_seconds=600,
    )
    _seed_manifest_with_parallel(boot, "m1", parallel)
    g1, g2 = asyncio.Event(), asyncio.Event()
    captured_argv: list[list[str]] = []
    gates = {"m1": g1}
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = _parallel_aware_fakes(
        gates, captured_argv)

    out: dict = {}
    with high_vram():
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
            sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        serve_calls: list = []
        orig_serve = mgr._serve_on_resident

        async def counting_serve(rr, slot, handle):
            serve_calls.append(getattr(handle, "parallel", None))
            return await orig_serve(rr, slot, handle)

        mgr._serve_on_resident = counting_serve
        try:
            # ---- turn 1: HELD mid-flight, then released so a REAL GraceTimer starts
            t1 = asyncio.create_task(
                mgr.submit_and_wait("m1", "p", thread_id="t1", client_meta={}))
            await wait_until(
                lambda: any(r.model_tag == "m1" and r.state is ResidentState.ACTIVE
                            for r in mgr._model_residents()), timeout=8.0)
            out["max_inflight_anchor_turn"] = await _poll_max_inflight(
                resident_for(mgr, "m1"))
            g1.set()
            await asyncio.wait_for(t1, timeout=8.0)

            r = resident_for(mgr, "m1")
            out["argv"] = captured_argv[0] if captured_argv else []
            out["handle_parallel"] = getattr(getattr(r, "handle", None),
                                             "parallel", None)
            out["grace_started"] = r.grace is not None and not r.grace.expired()

            # ---- turn 2: a FOLLOW-UP inside the still-open grace window, HELD
            gates["m1"] = g2
            t2 = asyncio.create_task(
                mgr.submit_and_wait("m1", "p2", thread_id="t2", client_meta={}))
            await wait_until(
                lambda: any(r2.model_tag == "m1" and r2.state is ResidentState.ACTIVE
                            for r2 in mgr._model_residents()), timeout=8.0)
            r = resident_for(mgr, "m1")
            out["max_inflight_followup_in_grace"] = await _poll_max_inflight(r)
            out["serve_calls"] = list(serve_calls)

            snap = mgr.status_snapshot()
            out["grace_open"] = r.grace is not None and not r.grace.expired()
            out["r_state"] = r.state.value
            out["snap_active_is_none"] = snap.get("active") is None
            out["snap_grace_is_set"] = snap.get("grace") is not None
            out["phase"] = _resident_phase(r, time.monotonic()).phase

            # ---- COUNTERFACTUAL: build the exact state the REMOVED guard was
            # written for (ACTIVE + non-empty inflight + an OPEN grace window).
            # r.inflight is genuinely empty here, so this is an INJECTION and is
            # labelled one: it is not a claim that the state arises on its own.
            injected = object()
            r.inflight.append(injected)
            try:
                snap2 = mgr.status_snapshot()
                top = ("GRACE" if snap2.get("grace") is not None
                       else "ACTIVE" if snap2.get("active") is not None
                       else "NEITHER")
                no_guard = _resident_phase(r, time.monotonic()).phase
                with_guard = ("ACTIVE"
                              if (r.state is ResidentState.ACTIVE and r.inflight)
                              else no_guard)
                out["cf_top_level"] = top
                out["cf_phase_no_guard"] = no_guard
                out["cf_phase_with_guard"] = with_guard
            finally:
                r.inflight.remove(injected)
            out["cf_inflight_restored"] = len(r.inflight)

            g2.set()
            try:
                await asyncio.wait_for(t2, timeout=8.0)
            except Exception:
                pass
        finally:
            mgr._worker_task.cancel()
            try:
                await mgr._worker_task
            except BaseException:       # CancelledError is BaseException on 3.11
                pass
    return out


@pytest.mark.parametrize("parallel", [1, 2])
def test_the_two_surfaces_agree_at_both_manifest_flag_values(parallel):
    """The removal reason must survive the flag flip. It does, at both values.

    In the window under test -- a resident serving a FOLLOW-UP inside an OPEN grace
    window -- the top-level surface reports GRACE (`active` is None, `grace` is set)
    and `_resident_phase` reports GRACE. They AGREE, at parallel=1 and parallel=2
    alike. The recorded reason is therefore not contingent on the deployed flag.
    """
    out = asyncio.run(_measure_tmp(parallel))

    # --- NON-VACUITY: the flip must actually have landed, or this proves nothing.
    assert out["handle_parallel"] == parallel, (
        f"manifest flip did not reach the handle: handle.parallel="
        f"{out['handle_parallel']!r}, wanted {parallel}")
    if parallel > 1:
        assert "--parallel" in out["argv"] and str(parallel) in out["argv"], (
            f"production argv never carried the flag: {out['argv']}")
    # --- NON-VACUITY: the window under test must genuinely be open.
    assert out["grace_started"], "turn 1 never opened a grace window"
    assert out["grace_open"], "the grace window closed before the follow-up sample"
    assert out["r_state"] == "ACTIVE", (
        f"the follow-up is not being served: r.state={out['r_state']!r}")

    # --- THE MEASUREMENT: both surfaces say GRACE, so they agree.
    assert out["snap_active_is_none"] is True
    assert out["snap_grace_is_set"] is True
    assert out["phase"] == ResidentState.GRACE.value


@pytest.mark.parametrize("parallel", [1, 2])
def test_the_removed_guard_would_split_the_surfaces_at_both_flag_values(parallel):
    """The counterfactual that makes SURFACES-AGREE the load-bearing reason.

    Injecting the guard's own precondition (a non-empty `r.inflight` while ACTIVE
    and in grace) and evaluating the removed clause by hand: WITHOUT it both
    surfaces say GRACE; WITH it the per-resident surface says ACTIVE while the top
    level still says GRACE. The guard MANUFACTURES the surface disagreement the
    current design avoids -- identically at parallel=1 and parallel=2.
    """
    out = asyncio.run(_measure_tmp(parallel))

    assert out["handle_parallel"] == parallel          # non-vacuity, as above
    assert out["cf_top_level"] == "GRACE"
    # WITHOUT the guard: agree.
    assert out["cf_phase_no_guard"] == "GRACE"
    # WITH the guard: disagree. This is the whole reason it is not in the tree.
    assert out["cf_phase_with_guard"] == "ACTIVE"
    assert out["cf_phase_with_guard"] != out["cf_top_level"]
    # the injection must not leak into any later assertion
    assert out["cf_inflight_restored"] == 0


def test_the_flag_IS_what_moves_inflight_and_the_grace_window_never_sees_it():
    """FIRING CONTROL + the result, in one differential.

    A measurement that reports "inflight is empty" is worthless unless the
    instrument can be shown to see a NON-empty one. It can: on a HELD ANCHOR turn,
    `r.inflight` reaches 1 at parallel=2 and stays 0 at parallel=1. So the writer
    chain is genuinely live and the flag is genuinely what moves it.

    And yet in the FOLLOW-UP-INSIDE-GRACE window it is 0 at BOTH values -- because
    that window is served by the ACTIVE_MATCH warm-reuse path, which never calls
    `_serve_on_resident` at all (exactly ONE call is recorded, turn 1's). So the
    removed guard could not do its intended job even with the flag flipped: it is
    inoperative in the one window it was written to discriminate.
    """
    p1 = asyncio.run(_measure_tmp(1))
    p2 = asyncio.run(_measure_tmp(2))

    # THE CONTROL FIRES: the instrument can see a non-empty r.inflight.
    assert p2["max_inflight_anchor_turn"] >= 1, (
        "FIRING CONTROL FAILED: r.inflight never went non-empty even at "
        "parallel=2 on a held anchor turn -- every zero below is uninterpretable")
    # ...and it is the FLAG that moves it, not the fixture.
    assert p1["max_inflight_anchor_turn"] == 0

    # THE RESULT: the grace-follow-up window never populates inflight, either way.
    assert p1["max_inflight_followup_in_grace"] == 0
    assert p2["max_inflight_followup_in_grace"] == 0
    # ...and the mechanism for that: the follow-up never re-enters _serve_on_resident.
    assert len(p1["serve_calls"]) == 1, p1["serve_calls"]
    assert len(p2["serve_calls"]) == 1, p2["serve_calls"]


async def _measure_tmp(parallel):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        return await _measure(Path(d), parallel)
