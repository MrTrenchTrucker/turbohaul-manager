"""Engine count and card placement for one model tag: _effective_cap /
_cards_that_fit / _auto_pick_gpu_excluding / _pick_card_for_new_instance.

The number of engines one model may run is bounded by two things only: the box-wide
``queue.max_parallel_sidecars`` budget and the cards the model fits on. The manifest
has no per-model engine limit. The context windows each engine serves
(``llama_server_flags.parallel``) never change the engine count.

Builds a real TurbohaulManager and drives the real _auto_pick_gpu against a patched
free-VRAM probe (nvidia-smi absent in CI - patch BOTH turbohaul.safety and
turbohaul.manager bindings, same as test_auto_placer.py). _instances_for is stubbed
here (same tag filter over the live residents as the real method) so the unit runs
independently of the registry. Mock Residents live in mgr._residents so the real
_model_residents()/_auto_pick_gpu see them.

Every expectation below is derived from the rule, not from running the code:
engines = min(cards that can host one copy, own engines + free box slots), where a
free box slot is one the OTHER tags are not holding, and a spent box budget keeps
exactly the engines already live (0 for a cold tag).
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.engine_budget import effective_engine_cap
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime(tmp_path, *, max_parallel_sidecars=2, safety_min_free_vram_mib=512):
    root = tmp_path / "state"
    root.mkdir()
    for sub in ("blobs", "manifests", "import-staging"):
        (root / sub).mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs", manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging", state_db_path=root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server", default_port_base=59500),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(max_parallel_sidecars=max_parallel_sidecars, grace_seconds=0,
                          idle_hot_load_seconds=0,
                          safety_min_free_vram_mib=safety_min_free_vram_mib),
        pull=PullConfig(),
    )
    return boot, runtime


def _mocks():
    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        proc = MagicMock(); proc.pid = 1; proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)
    async def fake_health(*a, **k): return True
    async def fake_sigterm(h, **k): return True, "clean"
    async def fake_vram(**k): return True, 100
    async def fake_complete(s, h): return {"ok": True}
    return dict(spawn_fn=fake_spawn, health_fn=fake_health, sigterm_fn=fake_sigterm,
                vram_fn=fake_vram, complete_fn=fake_complete)


def _mk(tmp_path, **kw):
    boot, runtime = _boot_runtime(tmp_path, **kw)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    mgr.runtime.queue.safety_enabled = False
    # Same tag filter as the real _instances_for, stubbed so this unit runs without
    # the registry.
    mgr._instances_for = lambda tag: [
        r for r in mgr._model_residents() if r.model_tag == tag]
    return mgr, boot


def _seed(boot, tag, *, expected_vram_mib=0, auto_place=True, split_mode="none",
          extra_flags=None):
    """Write a manifest. The defaults describe a one-card-per-engine tag: the manager
    picks the card (auto_place) and each engine stays on one card (split_mode none)."""
    p = boot.storage.manifests_path / f"{tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": expected_vram_mib * 1024 * 1024,
        "context_size": 2048,
        "expected_vram_bytes": expected_vram_mib * 1024 * 1024,
        "auto_place": auto_place,
        "llama_server_flags": {"split_mode": split_mode, **(extra_flags or {})},
    }))


def _res(rk, tag, *, gpu=0, state=ResidentState.ACTIVE, need=0):
    return SimpleNamespace(resident_key=rk, model_tag=tag, main_gpu=gpu, state=state,
                           reserved_need_mib=need)


@contextmanager
def _vram(vals):
    with patch("turbohaul.safety._read_free_vram_all_mib", return_value=vals), \
         patch("turbohaul.manager._read_free_vram_all_mib", return_value=vals):
        yield


def _add(mgr, *residents):
    for r in residents:
        mgr._residents[r.resident_key] = r


# ---- _effective_cap ------------------------------------------------------------
def test_effective_cap_missing_manifest(tmp_path):
    mgr, boot = _mk(tmp_path)
    assert mgr._effective_cap("nope") == 1


def test_effective_cap_bug_scenario_second_card_spawns(tmp_path):
    # max_parallel=2, a one-card-per-engine tag, 2 cards, 1 engine already up on
    # card0. Counting the tag's own engine against the box budget (max_parallel -
    # residents = 2-1 = 1) would cap at 1 and strand card1. Adding the tag's own
    # engines back (current + free_global = 1+1 = 2) lets the 2nd engine spawn: the
    # tag may hold every box slot that no other tag holds, and card1 fits a copy.
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000)
    _add(mgr, _res("borage#0", "borage", gpu=0, state=ResidentState.ACTIVE))
    with _vram([4000, 20000]):                          # card1 free, card0 partly used
        assert mgr._effective_cap("borage") == 2


def test_effective_cap_cold_start_two_cards(tmp_path):
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000)
    with _vram([20000, 20000]):
        assert mgr._effective_cap("borage") == 2


def test_effective_cap_degrade_to_one_card(tmp_path):
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=3)
    _seed(boot, "borage", expected_vram_mib=3000)
    with _vram([20000, 500]):                           # only card0 fits
        assert mgr._effective_cap("borage") == 1


def test_effective_cap_probe_down_cold_is_one(tmp_path):
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000)
    with _vram(None):
        assert mgr._effective_cap("borage") == 1        # safe degrade


def test_effective_cap_probe_down_keeps_existing(tmp_path):
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000)
    _add(mgr, _res("borage#0", "borage", gpu=0), _res("borage#1", "borage", gpu=1))
    with _vram(None):
        assert mgr._effective_cap("borage") == 2        # budget spent: keep both, no false shrink


def test_effective_cap_global_starved_by_other_model(tmp_path):
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000)
    _add(mgr, _res("other#0", "other", gpu=0), _res("borage#0", "borage", gpu=1))
    with _vram([20000, 20000]):
        assert mgr._effective_cap("borage") == 1        # 2nd global slot held by 'other'


def test_effective_cap_budget_full_cold_is_zero(tmp_path):
    # 0 instances of the tag, TWO other-tag engines already fill the box budget
    # (max_parallel=2), both cards VRAM-free. cap MUST be 0: the routing code spawns while
    # `len(instances) < cap`, so a floor of 1 (0 < 1) would admit a 3rd engine past
    # the box-wide limit.
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000)
    _add(mgr, _res("alpha#0", "alpha", gpu=0), _res("beta#0", "beta", gpu=1))
    with _vram([20000, 20000]):                          # both cards free for the tag
        assert mgr._effective_cap("borage") == 0


@pytest.mark.parametrize("probe", [None, [100, 100]], ids=["probe-down", "no-card-fits"])
def test_effective_cap_never_shrinks_below_the_live_engines(tmp_path, probe):
    # Budget 3, two engines of the tag live on cards 0 and 1, one box slot free. The
    # card probe says nothing useful (down, or no card has room for another copy), so
    # no ADDITIONAL engine is admitted, but the two running ones sit on two distinct
    # cards that already host the model: cap = 2 live cards + 0 new cards = 2. It must
    # never drop below the engines already running (that would make the routing code treat a
    # live engine as surplus).
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=3)
    _seed(boot, "borage", expected_vram_mib=3000)
    _add(mgr, _res("borage#0", "borage", gpu=0), _res("borage#1", "borage", gpu=1))
    with _vram(probe):
        assert mgr._effective_cap("borage") == 2


def _spy_budget(monkeypatch):
    """Record every EngineBudgetInputs _effective_cap hands to the budget function,
    delegating to the real one."""
    seen = []
    real = effective_engine_cap

    def spy(inp):
        seen.append(inp)
        return real(inp)

    monkeypatch.setattr("turbohaul.manager.effective_engine_cap", spy)
    return seen


@pytest.mark.parametrize("parallel", [1, 2, 4])
def test_effective_cap_engine_count_ignores_windows_per_engine(tmp_path, parallel):
    # budget 2, cold tag, both cards fit a copy -> 2 engines. The windows each
    # engine serves (llama_server_flags.parallel) are a different setting and must
    # not scale the engine count: 2 engines whatever --parallel is, never 2*parallel.
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000,
          extra_flags={"parallel": parallel, "kv_unified": True})
    with _vram([20000, 20000]):
        assert mgr._effective_cap("borage") == 2


def test_effective_cap_passes_the_manifests_windows_per_engine_to_the_budget(
        tmp_path, monkeypatch):
    # The windows number the budget function reports a context total from is the
    # manifest's llama_server_flags.parallel (default 1 when the flag is absent).
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    seen = _spy_budget(monkeypatch)
    _seed(boot, "wide", expected_vram_mib=3000,
          extra_flags={"parallel": 4, "kv_unified": True})
    _seed(boot, "plain", expected_vram_mib=3000)
    with _vram([20000, 20000]):
        mgr._effective_cap("wide")
        mgr._effective_cap("plain")
    assert [i.parallel_width for i in seen] == [4, 1]


@pytest.mark.parametrize("unusable", [0, -3, "2", True, 2.5, None])
def test_effective_cap_counts_an_unusable_parallel_as_one_window(
        tmp_path, monkeypatch, unusable):
    # Anything that is not an int >= 1 counts as one window and never raises, so a
    # bad value cannot break routing. The manifest read is replaced (the real
    # validator refuses these values) with an object carrying the odd flag.
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "odd", expected_vram_mib=3000)
    seen = _spy_budget(monkeypatch)
    monkeypatch.setattr(
        "turbohaul.manager.read_manifest_cached",
        lambda root, tag: SimpleNamespace(
            auto_place=True,
            llama_server_flags={"split_mode": "none", "parallel": unusable}),
    )
    with _vram([20000, 20000]):
        assert mgr._effective_cap("odd") == 2
    assert [i.parallel_width for i in seen] == [1]
    assert [type(i.parallel_width) for i in seen] == [int]


def test_effective_cap_ignores_a_retired_key_left_in_the_manifest_file(tmp_path):
    # An old manifest still carrying max_instances: 1 loads (the key is dropped) and
    # must not hold the tag to one engine: budget 2, one engine up on card0, card1
    # fits a copy -> 2.
    mgr, boot = _mk(tmp_path, max_parallel_sidecars=2)
    _seed(boot, "borage", expected_vram_mib=3000)
    path = boot.storage.manifests_path / "borage.yaml"
    body = yaml.safe_load(path.read_text())
    body["max_instances"] = 1
    path.write_text(yaml.safe_dump(body))
    _add(mgr, _res("borage#0", "borage", gpu=0, state=ResidentState.ACTIVE))
    with _vram([4000, 20000]):
        assert mgr._effective_cap("borage") == 2


# ---- _cards_that_fit / _auto_pick_gpu_excluding --------------------------------
def test_cards_that_fit(tmp_path):
    mgr, _ = _mk(tmp_path)
    with _vram([20000, 1000, 20000]):                   # floor 512
        assert mgr._cards_that_fit(3000) == 2
        assert mgr._cards_that_fit(3000, exclude={0}) == 1
        assert mgr._cards_that_fit(3000, exclude={0, 2}) == 0


def test_card_avail_subtracts_reserved_loading_sibling(tmp_path):
    # A RESERVED_LOADING sibling's reserved_need_mib is subtracted from
    # ITS card's available VRAM (mirrors _auto_pick_gpu's per-card accounting) so
    # _cards_that_fit / placement see the reduced headroom of a booting neighbour.
    mgr, boot = _mk(tmp_path)
    _add(mgr, _res("x#0", "x", gpu=0, state=ResidentState.RESERVED_LOADING, need=18000))
    with _vram([20000, 20000]):
        # card0 avail = 20000 - 18000 - 512 = 1488 < 3000 -> excluded; card1 fits
        assert mgr._cards_that_fit(3000) == 1
        assert mgr._auto_pick_gpu_excluding(3000, set())[0] == 1


def test_auto_pick_gpu_excluding(tmp_path):
    mgr, _ = _mk(tmp_path)
    with _vram([20000, 5000, 30000]):
        g, sp = mgr._auto_pick_gpu_excluding(3000, {2})  # exclude most-free card2
        assert g == 0 and sp == "none"                    # most-free of the rest
        assert mgr._auto_pick_gpu_excluding(3000, {0, 1, 2})[0] is None
    with _vram(None):
        assert mgr._auto_pick_gpu_excluding(3000, set())[0] is None


# ---- _pick_card_for_new_instance (drives the REAL _auto_pick_gpu) --------------
def test_pick_card_even_split_nudges_to_distinct(tmp_path):
    mgr, boot = _mk(tmp_path)
    _seed(boot, "borage", expected_vram_mib=3000)
    # instance #1 ACTIVE on card0; card0 kept most-free so _auto_pick_gpu returns 0
    # (a used card) -> _pick_card_for_new_instance must nudge to card1.
    _add(mgr, _res("borage#0", "borage", gpu=0, state=ResidentState.ACTIVE))
    with _vram([20000, 15000]):
        g, sp = mgr._pick_card_for_new_instance("borage", 3000)
        assert g == 1 and sp == "none"                    # even split


def test_pick_card_first_instance_uses_auto_pick(tmp_path):
    mgr, boot = _mk(tmp_path)
    _seed(boot, "borage", expected_vram_mib=3000)
    with _vram([15000, 20000]):                          # no instances yet
        g, sp = mgr._pick_card_for_new_instance("borage", 3000)
        assert g == 1                                      # most-free, nothing excluded


def test_pick_card_pile_on_when_no_distinct_fits(tmp_path):
    mgr, boot = _mk(tmp_path)
    _seed(boot, "borage", expected_vram_mib=3000)
    _add(mgr, _res("borage#0", "borage", gpu=0, state=ResidentState.ACTIVE))
    with _vram([20000, 200]):                            # only card0 fits, card0 used
        g, sp = mgr._pick_card_for_new_instance("borage", 3000)
        assert g == 0                                      # keep _auto_pick_gpu pile-on


def test_pick_card_none_when_nothing_fits(tmp_path):
    mgr, boot = _mk(tmp_path)
    _seed(boot, "borage", expected_vram_mib=3000)
    with _vram([100, 100]):                              # nothing fits anywhere
        g, sp = mgr._pick_card_for_new_instance("borage", 3000)
        assert g is None                                   # caller must NOT spawn


def test_pick_card_never_emits_layer_split(tmp_path):
    # When only an aggregate layer-split would fit (no single card), _auto_pick_gpu
    # returns (0,'layer'). A one-card-per-engine tag is split:none, so a card-spanning
    # engine is NOT a valid even-split target -> return (None,'none') (route to an
    # existing engine), never spawn a layer-split engine.
    mgr, boot = _mk(tmp_path)
    _seed(boot, "borage", expected_vram_mib=3000)
    mgr._auto_pick_gpu = lambda need: (0, "layer")       # force the aggregate fallback
    g, sp = mgr._pick_card_for_new_instance("borage", 3000)
    assert g is None and sp == "none"
