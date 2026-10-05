"""Listed-waiter eviction shield removal.

Design rule:

    "A client with queued work that has not started is not protected by that
    queue; it is protected by its rank, its grace window, its cooldown --
    and by nothing else."

The ``listed_waiter_tags`` shield in ``_lru_idle_unloadable`` (manager.py,
before its removal) excluded a resident from the eviction candidate set
BECAUSE a listed queued request existed for its model_tag -- exactly the
named, forbidden case. The shield was therefore REMOVED, and the
mirror clause was removed with it; the frozen ``why`` values are
unchanged and the unreachable probe is marked as dead code
(it can no longer be reached).

This file covers THAT change -- one command, both arms:

  PRE-fix:  the picker excludes the resident on the shield clause. The
            make-room MISS attribution (``_starvation_reason``, the
            per-resident mirror) names the resident under the
            ``listed_waiter`` clause -- so the victim assertion below fails
            and its failure message IS the right reason, not just a colour.
  POST-fix: the same resident is the victim; the mirror no longer carries
            the ``listed_waiter`` clause (drift invariant: the clause exists
            if and only if the filter parameter exists).

No spawn, no worker_loop, no wall clock: the unit under change is the
candidate filter and its mirror, driven through the REAL manager, the REAL
submit() listing path (``slot.fastlane = match_fastlane(...)``,
manager.py), and the REAL queue snapshot producer
(``queue.listed_waiter_model_tags``). The resident is seeded IDLE_EVICTABLE
directly into ``mgr._residents`` -- the same registry the filter reads via
``_model_residents``. Harness convention (spawn/health/sigterm/vram hooks)
per tests/test_multislot_concurrency.py, reused through the acceptance fixture's
``_boot_runtime`` / ``_seed_manifest`` / ``_gated_mocks``.
"""
from __future__ import annotations

import inspect
import time

import pytest

import test_ranked_eviction_acceptance as ranked_acceptance
from turbohaul.manager import Resident, ResidentState, TurbohaulManager

_TAG = "shield-model"
_WAITER_IP = ranked_acceptance._P1_ADDR  # rule index 0 -- the highest-ranked listed waiter


def _has_param(fn, name: str) -> bool:
    return name in inspect.signature(fn).parameters


@pytest.mark.asyncio
async def test_listed_waiter_is_no_eviction_shield(tmp_path):
    spawn_calls: list = []
    sigterm_calls: list = []
    boot, runtime = ranked_acceptance._boot_runtime(tmp_path, grace_seconds=1)
    ranked_acceptance._seed_manifest(boot, _TAG, main_gpu=0)
    mgr = TurbohaulManager(
        boot, runtime, **ranked_acceptance._gated_mocks(spawn_calls, sigterm_calls, {})
    )

    # One parked, idle-evictable, UNLISTED resident for model _TAG.
    r = Resident(
        model_tag=_TAG,
        state=ResidentState.IDLE_EVICTABLE,
        last_active_monotonic=time.monotonic() - 3600.0,
        idle_client_meta=None,
    )
    mgr._residents[_TAG] = r

    # A LISTED queued waiter for the SAME model, through the real submit path
    # (admission-time match sets slot.fastlane; the slot lands in staging).
    await mgr.submit(
        _TAG, "hi", thread_id="t-waiter", client_meta={"ip": _WAITER_IP}
    )

    tags = await mgr.queue.listed_waiter_model_tags()
    assert tags == {_TAG}, (
        f"precondition failed: the listed-waiter snapshot must be live for "
        f"{_TAG!r}, got {tags!r} -- the shield's input does not exist in "
        "this configuration and the test proves nothing"
    )

    # Attribution FIRST, so the victim assertion's failure message carries the
    # right reason (the per-resident mirror), not just a colour.
    attr = (
        mgr._starvation_reason(tags)
        if _has_param(mgr._starvation_reason, "listed_waiter_tags")
        else mgr._starvation_reason()
    )
    per_resident = attr["per_resident"]

    filter_shields = _has_param(mgr._lru_idle_unloadable, "listed_waiter_tags")
    victim = (
        mgr._lru_idle_unloadable(tags) if filter_shields
        else mgr._lru_idle_unloadable()
    )

    # The change under test: a queued listed request must NOT shield the
    # resident (design rule, "and by nothing else").
    assert victim is not None, (
        "the listed-waiter shield is still in _lru_idle_unloadable: the "
        f"IDLE_EVICTABLE resident for {_TAG!r} was excluded although its only "
        "blocker is a queued LISTED waiter (pre-fix arm -- design rule, "
        f"'and by nothing else'). Attributing mirror: why={attr['why']!r} "
        f"per_resident={per_resident!r}"
    )
    assert victim.model_tag == _TAG

    # Drift invariant (the mirror must never disagree
    # with the filter it mirrors): the mirror's listed_waiter clause exists
    # IFF the filter parameter exists. Pre-fix: both present. Post-fix: both
    # gone. A mismatch is the instrumented invariant failing.
    mirror_shields = "listed_waiter" in str(attr)
    assert filter_shields == mirror_shields, (
        f"mirror drift: filter takes listed_waiter_tags={filter_shields} but "
        f"the mirror reports listed_waiter={mirror_shields}; "
        f"per_resident={per_resident!r}"
    )


@pytest.mark.asyncio
async def test_unlisted_waiter_shielded_input_is_live_and_complete(tmp_path):
    """Sanity half of the pair: the producer this change preserves
    (queue.listed_waiter_model_tags) sees BOTH listed waiters and stays
    empty for unlisted traffic -- the data plane survives the shield."""
    spawn_calls: list = []
    sigterm_calls: list = []
    boot, runtime = ranked_acceptance._boot_runtime(tmp_path, grace_seconds=1)
    ranked_acceptance._seed_manifest(boot, _TAG, main_gpu=0)
    ranked_acceptance._seed_manifest(boot, "plain-model", main_gpu=0)
    mgr = TurbohaulManager(
        boot, runtime, **ranked_acceptance._gated_mocks(spawn_calls, sigterm_calls, {})
    )

    assert await mgr.queue.listed_waiter_model_tags() == set()

    await mgr.submit(
        _TAG, "hi", thread_id="t-w", client_meta={"ip": ranked_acceptance._P3_ADDR}
    )
    await mgr.submit(
        "plain-model", "hi", thread_id="t-u", client_meta={"ip": "10.9.9.9"}
    )

    tags = await mgr.queue.listed_waiter_model_tags()
    assert tags == {_TAG}, (
        f"producer regression: exactly the listed model_tag must appear, "
        f"got {tags!r}"
    )
