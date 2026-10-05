"""The hot path of ``_effective_cap``: no card probe while the box budget is spent.

``_effective_cap`` runs on every route, including a request for a tag that already has a
live engine. Measuring the cards means an ``nvidia-smi`` subprocess and an uncached
footprint read, so it must only happen when a further engine is actually within reach.

When the box-wide ``queue.max_parallel_sidecars`` budget is already spent, no further
engine can be admitted whatever the cards say, so the answer is the number of engines the
tag already runs (0 for a cold tag) and it must come back BEFORE any card probe:
``_cards_that_fit``, ``_card_avail_mib``, ``_read_model_footprint`` and the
``nvidia-smi`` reader under them. These tests spy on all of them (wrapping the real
functions) and assert none is reached, and a control shows the spies do fire when a box
slot is free.

Everything runs against a real manager and real manifest files on disk.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from _multiinstance_support import _live_engine, _manifest, _new_manager, _request
from test_placement_multiinstance import _add, _mk, _res, _seed
from turbohaul.manager import ResidentState

PROBES = ("_cards_that_fit", "_card_avail_mib", "_read_model_footprint")


@contextmanager
def _spied(mgr, free_vram=(20000, 20000)):
    """Wrap each card-probe method of ``mgr`` (and the nvidia-smi reader under them)
    with a call recorder that delegates to the real one. Yields {name: [calls]}."""
    calls = {name: [] for name in (*PROBES, "nvidia_smi_reader")}
    for name in PROBES:
        real = getattr(mgr, name)

        def make(name=name, real=real):
            def spy(*a, **k):
                calls[name].append((a, k))
                return real(*a, **k)
            return spy

        setattr(mgr, name, make())

    def reader():
        calls["nvidia_smi_reader"].append(())
        return list(free_vram)

    with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=reader), \
         patch("turbohaul.manager._read_free_vram_all_mib", side_effect=reader):
        yield calls


def _nothing_probed(calls):
    return not any(calls.values())


MANIFESTS = {
    "one-card-per-engine": dict(auto_place=True, split_mode="none"),
    "placement-free": dict(auto_place=False, split_mode="layer"),
}


@pytest.mark.parametrize("kind", list(MANIFESTS))
def test_a_live_tag_holding_the_only_slot_is_answered_without_a_probe(tmp_path, kind):
    # Budget 1, this tag's own engine holds it: nothing more can be admitted, so the
    # answer is the one engine the tag runs. This is the shape of every request for a
    # live tag on a box at its sidecar limit.
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=1)
    _seed(boot, "borage", expected_vram_mib=3000, **MANIFESTS[kind])
    _add(mgr, _res("borage", "borage", gpu=0, state=ResidentState.ACTIVE))
    with _spied(mgr) as calls:
        assert mgr._effective_cap("borage") == 1
    assert _nothing_probed(calls), f"a probe ran on the spent-budget path: {calls}"


@pytest.mark.parametrize("kind", list(MANIFESTS))
def test_a_live_tag_holding_every_slot_is_answered_without_a_probe(tmp_path, kind):
    # Budget 2, two engines of this tag: it keeps both, no probe.
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000, **MANIFESTS[kind])
    _add(mgr, _res("borage#0", "borage", gpu=0), _res("borage#1", "borage", gpu=1))
    with _spied(mgr) as calls:
        assert mgr._effective_cap("borage") == 2
    assert _nothing_probed(calls), f"a probe ran on the spent-budget path: {calls}"


@pytest.mark.parametrize("kind", list(MANIFESTS))
def test_a_cold_tag_while_other_tags_fill_the_budget_is_answered_zero_without_a_probe(
        tmp_path, kind):
    # Budget 2, two OTHER tags hold both slots, this tag has nothing live: 0 (an engine
    # here would breach the box-wide budget) and no probe.
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000, **MANIFESTS[kind])
    _add(mgr, _res("alpha", "alpha", gpu=0), _res("beta", "beta", gpu=1))
    with _spied(mgr) as calls:
        assert mgr._effective_cap("borage") == 0
    assert _nothing_probed(calls), f"a probe ran on the spent-budget path: {calls}"


def test_an_unreadable_manifest_is_answered_without_a_probe(tmp_path):
    # The fail-safe: a manifest that cannot be read returns 1 and never reaches the
    # cards, even with a free box slot.
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    (boot.storage.manifests_path / "borage.yaml").write_text("{ not: valid: yaml")
    _add(mgr, _res("borage", "borage", gpu=0))
    with _spied(mgr) as calls:
        assert mgr._effective_cap("borage") == 1
    assert _nothing_probed(calls), f"a probe ran for an unreadable manifest: {calls}"


# ---- control: the spies do fire when a further engine is within reach ----------

def test_control_the_probe_is_reached_when_a_box_slot_is_free(tmp_path):
    # Budget 2, one engine live: one slot is free, so the cards are measured. If the
    # spies were not wired to the functions _effective_cap really calls, the tests
    # above would pass for the wrong reason; this one cannot pass unless they fire.
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000, **MANIFESTS["one-card-per-engine"])
    _add(mgr, _res("borage", "borage", gpu=0, state=ResidentState.ACTIVE))
    with _spied(mgr) as calls:
        assert mgr._effective_cap("borage") == 2
    assert len(calls["_read_model_footprint"]) == 1
    assert len(calls["_cards_that_fit"]) >= 1
    assert len(calls["_card_avail_mib"]) >= 1
    assert len(calls["nvidia_smi_reader"]) >= 1


# ---- a tag that does not ask for one card per engine skips the probe ----------------
#
# A manifest can only be placed as more than one engine when it lets the manager pick the
# card and keeps each engine on one card. For any other manifest the cards cannot change
# the answer, so the probe is skipped even while a box slot is free. The value is the
# engines the tag already runs, and at least 1: a cold tag may start its one engine, but
# never a second.

# Manifest overrides for ``_manifest`` (``auto_place``, ``llama_server_flags``): every
# shape here is one that ``_wants_one_card_per_engine`` rejects.
NOT_ONE_CARD_PER_ENGINE = {
    "placement-free": dict(auto_place=False, llama_server_flags={"split_mode": "layer"}),
    "auto-place-no-split-mode": dict(auto_place=True, llama_server_flags={}),
    "auto-place-layer-split": dict(auto_place=True,
                                   llama_server_flags={"split_mode": "layer"}),
    "pinned-card": dict(auto_place=False, llama_server_flags={"split_mode": "none"}),
}

# (box budget, live engines as (registry key, card), expected cap). In every state the
# box has a free slot (live engines < budget), so the spent-budget path is not what
# answers: only the early return keeps the probe from running.
LIVE_STATES = {
    "cold": (2, [], 1),
    "one-live-engine": (2, [("borage", 0)], 1),
    "two-live-engines": (3, [("borage#0", 0), ("borage#1", 1)], 2),
    "off-key-survivor": (2, [("borage#1", 1)], 1),
}


@pytest.mark.parametrize("state", list(LIVE_STATES))
@pytest.mark.parametrize("shape", list(NOT_ONE_CARD_PER_ENGINE))
def test_a_tag_not_asking_for_one_card_per_engine_is_answered_without_a_probe(
        tmp_path, shape, state):
    budget, live, expected = LIVE_STATES[state]
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=budget)
    _manifest(boot, "borage", **NOT_ONE_CARD_PER_ENGINE[shape])
    _add(mgr, *(_res(key, "borage", gpu=gpu) for key, gpu in live))
    with _spied(mgr) as calls:
        cap = mgr._effective_cap("borage")
    assert len(mgr._model_residents()) < budget, (
        "test setup: a box slot is free, so the spent-budget path cannot be what answers"
    )
    assert cap == expected, f"{shape} / {state}: expected {expected}, got {cap}"
    assert _nothing_probed(calls), (
        f"{shape} / {state}: a probe ran for a tag that cannot be placed as a second "
        f"engine: {calls}"
    )


# ---- control: a one-card-per-engine tag still measures the cards --------------------

@pytest.mark.parametrize("live_keys", [[], ["borage"]], ids=["cold", "one-live-engine"])
def test_control_a_one_card_per_engine_tag_reads_the_footprint_once_and_probes_the_cards(
        tmp_path, live_keys):
    # Same free box slot as the tests above, but a manifest that asks for one card per
    # engine: the footprint is read exactly once and the cards are measured. This is the
    # arm that shows the early return is keyed on the manifest and not on the state.
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000, **MANIFESTS["one-card-per-engine"])
    _add(mgr, *(_res(key, "borage", gpu=0, state=ResidentState.ACTIVE) for key in live_keys))
    assert len(mgr._model_residents()) < 2, "test setup: a box slot is free"
    with _spied(mgr) as calls:
        mgr._effective_cap("borage")
    assert len(calls["_read_model_footprint"]) == 1
    assert len(calls["_cards_that_fit"]) >= 1
    assert len(calls["_card_avail_mib"]) >= 1
    assert len(calls["nvidia_smi_reader"]) >= 1


# ---- the same through the real routing path ----------------------------------------

async def test_routing_a_placement_free_tag_with_two_live_engines_spawns_nothing(tmp_path):
    # Budget 3, two engines of a placement-free tag: a box slot is free, yet the tag
    # cannot be placed as a third engine. Nothing is spawned and the request is handed
    # to one of the live engines.
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=3)
    _manifest(boot, "borage", **NOT_ONE_CARD_PER_ENGINE["placement-free"])
    engines = [_live_engine(manager, "borage", gpu=0, key="borage#0"),
               _live_engine(manager, "borage", gpu=1, key="borage#1")]
    slot = _request("borage")
    with _spied(manager) as calls:
        try:
            await manager._route_or_reserve(slot)
            await asyncio.sleep(0.1)
            assert len(manager._model_residents()) < 3, "test setup: a box slot is free"
            assert spawn_calls == [], f"an engine was spawned: {spawn_calls}"
            assert len(manager._instances_for("borage")) == 2
            handed = [e for e in engines if not e.inbox.empty()]
            assert len(handed) == 1 and handed[0].inbox.get_nowait() is slot, (
                "the request must go to one of the live engines"
            )
        finally:
            await manager.shutdown()
    assert _nothing_probed(calls), f"a probe ran on the routing path: {calls}"
