"""KV-cache hard RAM cap: the save-time gate and the GC's SOFT/HARD protection split.

The RAM-tier KV ceiling (``ram_cache_max_bytes``, default 20 GiB) must be a real limit:

  * The GC reclaims idle bins. Idle-held and resident-idle KV bins are SOFT: they are
    kept apart from the protected set of live threads, so ``_gc_kv_cache``'s age prune
    and its ceiling prune can reclaim them instead of giving up at the cap.
  * The per-turn save funnel ``_save_slot_kv_inner`` checks the cap before it saves.

Save gate: after the existing refusals and before the engine save POST, if the dir is
already at or over ``ram_cache_max_bytes`` the existing GC (ceiling only) runs to make
room; if a HARD bin still holds the floor, the snapshot is declined (live KV stays
in-engine).

GC: protection is HARD (active and in-flight bins, never evicted) or SOFT (idle-held,
resident-idle and recency clean-pins, reclaimable). The age prune unlinks SOFT bins past
``age_cutoff``; the ceiling prune falls back to SOFT bins, oldest first, before it gives
up.

These are behaviour tests of the HARD/SOFT split, the SOFT age-prune and ceiling
fallback, and the save-time gate.
"""
import os
import time

import pytest

import turbohaul.manager as manager_mod
import turbohaul.subprocess_mgr as subprocess_mgr
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
from turbohaul.slot import Slot
from turbohaul.manager import TurbohaulManager

_QWEN = "example-27b"


# ---------------------------------------------------------------------------
# fixtures (mirror tests/test_kvcache_ram_ceiling_config.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def mgr(tmp_path):
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
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


# ---------------------------------------------------------------------------
# bin writers
# ---------------------------------------------------------------------------

def _total_bin_bytes(kv_dir) -> int:
    """Sum of .bin (+ .bin.ckpt) sizes on disk, computed independently of the code
    under test (it does not use _kvcache_dir_accounting)."""
    total = 0
    for fn in os.listdir(str(kv_dir)):
        if not fn.endswith(".bin"):
            continue
        p = os.path.join(str(kv_dir), fn)
        try:
            total += os.stat(p).st_size
            total += os.stat(p + ".ckpt").st_size
        except OSError:
            pass
    return total


def _write_bin(kv_dir, name: str, size: int, mtime: float) -> str:
    """Plain, un-pinned .bin, NO sidecar — never pin-elected, never
    thread-protected. A pure candidate for any prune axis."""
    p = kv_dir / name
    p.write_bytes(b"\x00" * size)
    os.utime(p, (mtime, mtime))
    return name


def _write_kv_bin(kv_dir, name: str, size: int, mtime: float,
                  thread_hash: str | None = None, clean_prefix: bool = False,
                  model_tag: str = _QWEN, port: int = 59500) -> str:
    """A .bin + .json sidecar carrying ``thread_hash`` (so the GC's
    ``hash_by_path`` sees it and the protection tiers can classify it). No
    ``bin_bytes``/``prompt_tokens`` (keeps the save T-GUARD estimate off) and,
    unless asked, no ``clean_prefix`` (so it is protected ONLY by its thread
    tier, not by the clean-anchor pin)."""
    import json as _json
    p = kv_dir / name
    p.write_bytes(b"\x00" * size)
    os.utime(p, (mtime, mtime))
    meta = {"model_tag": model_tag, "port": port, "slot_id": 0}
    if thread_hash is not None:
        meta["thread_hash"] = thread_hash
    if clean_prefix:
        meta["clean_prefix"] = True
    (kv_dir / (name[:-4] + ".json")).write_text(_json.dumps(meta))
    return name


def _set_hard_active(mgr, thread_id: str) -> None:
    """Make ``thread_id`` a live HARD thread (active slot)."""
    mgr._active_slot = Slot.new(
        _QWEN, thread_id=thread_id, admission_ctx_len=100,
        client_meta={"session_id": f"sess-{thread_id}"},
    )


def _set_soft_idle(mgr, thread_id: str) -> None:
    """Make ``thread_id`` the idle-held SOFT thread."""
    mgr._idle_thread_id = thread_id
    mgr._idle_client_meta = {"session_id": f"sess-{thread_id}"}


# ===========================================================================
# GC SOFT/HARD hierarchy
# ===========================================================================

@pytest.mark.asyncio
async def test_gc_age_prunes_soft_idle_bin_but_never_hard_active(mgr, kv_dir):
    """Age prune. A SOFT idle-held bin PAST ``age_cutoff`` is reclaimed; a
    HARD active-turn bin (even older) is NEVER age-evicted.

    Idle-held bins must be SOFT, not part of the live-thread protected set: the
    idle bin is age-pruned and the active bin (HARD) is kept. If the idle bin sat
    in the protected set, both bins would survive (the `soft gone` assertion fails).
    """
    active_tid = "active-tid"
    idle_tid = "idle-tid"
    _set_hard_active(mgr, active_tid)
    _set_soft_idle(mgr, idle_tid)

    t0 = 10_000.0
    now = t0 + 7 * 3600  # 7h old -> past the default 6h age limit
    # HARD active bin (older on disk, but NEVER age-evicted)
    hard = _write_kv_bin(kv_dir, "m.hard.active.bin", 1024, mtime=t0 - 3600,
                         thread_hash=mgr._thread_hash(active_tid))
    # SOFT idle bin, past the age cutoff
    soft = _write_kv_bin(kv_dir, "m.soft.idle.bin", 1024, mtime=t0,
                         thread_hash=mgr._thread_hash(idle_tid))

    # cap huge + max_files huge -> isolate the AGE axis (no ceiling/count)
    mgr.runtime.kv.ram_cache_max_bytes = 10 ** 18

    real_time = time.time
    try:
        time.time = lambda: now
        deleted = await mgr._gc_kv_cache()  # no args -> config default age (6h)
    finally:
        time.time = real_time

    assert deleted >= 1
    assert not (kv_dir / soft).exists(), (
        "SOFT idle-held bin past age_cutoff was NOT reclaimed — the RAM-cap age "
        "prune is missing (it stayed pinned behind the live-thread set)."
    )
    assert (kv_dir / hard).exists(), (
        "HARD active-turn bin was age-evicted — HARD must NEVER be evicted."
    )


@pytest.mark.asyncio
async def test_gc_ceiling_falls_back_to_soft_before_over_cap_giveup(mgr, kv_dir):
    """Ceiling prune. Under an over-cap pass, after evicting every
    unprotected bin it falls back to SOFT bins oldest-first BEFORE giving up;
    the HARD active bin is never taken.

    The oldest SOFT bin is reclaimed, bringing the total under cap; HARD + the
    newer SOFT survive. If SOFT idle bins were protected like live threads, the
    ceiling prune would give up at the soft cap and all three would survive
    (the `exactly one SOFT gone` assertion fails).
    """
    active_tid = "active-tid"
    idle_tid = "idle-tid"
    _set_hard_active(mgr, active_tid)
    _set_soft_idle(mgr, idle_tid)

    ten_mib = 10 * 1024 * 1024
    # HARD active bin — 10 MiB, and the OLDEST file on disk (must still survive)
    hard = _write_kv_bin(kv_dir, "m.hard.active.bin", ten_mib, mtime=1000.0,
                         thread_hash=mgr._thread_hash(active_tid))
    soft_old = _write_kv_bin(kv_dir, "m.soft.old.bin", ten_mib, mtime=2000.0,
                             thread_hash=mgr._thread_hash(idle_tid))
    soft_new = _write_kv_bin(kv_dir, "m.soft.new.bin", ten_mib, mtime=3000.0,
                             thread_hash=mgr._thread_hash(idle_tid))

    # 30 MiB total, cap 25 MiB -> exactly ONE 10 MiB bin must go. Age disabled
    # so ONLY the ceiling prune acts; the victim must be the OLDEST SOFT bin.
    mgr.runtime.kv.ram_cache_max_bytes = int(25 * 1024 * 1024)

    deleted = await mgr._gc_kv_cache(max_age_hours=1e9, max_files=1_000_000_000)

    assert deleted == 1, (
        f"expected exactly one eviction to get 30 MiB under a 25 MiB cap, got "
        f"{deleted} — if the SOFT idle bins are protected, the ceiling "
        f"prune gives up over cap."
    )
    assert not (kv_dir / soft_old).exists(), "oldest SOFT bin should be reclaimed"
    assert (kv_dir / soft_new).exists(), "newer SOFT bin should survive (only 1 needed)"
    assert (kv_dir / hard).exists(), "HARD active bin must NEVER be evicted"
    assert _total_bin_bytes(kv_dir) <= int(25 * 1024 * 1024)


@pytest.mark.asyncio
async def test_gc_never_evicts_hard_active_bin_even_single_bin_over_cap(mgr, kv_dir):
    """Guard: a lone HARD active bin that ALONE exceeds the
    cap is never evicted; the ceiling prune gives up and leaves it. Ensures the
    SOFT/HARD split kept HARD fully exempt."""
    _set_hard_active(mgr, "active-tid")
    big = _write_kv_bin(kv_dir, "m.big.active.bin", 1024, mtime=1000.0,
                        thread_hash=mgr._thread_hash("active-tid"))
    mgr.runtime.kv.ram_cache_max_bytes = 10  # absurdly small
    deleted = await mgr._gc_kv_cache(max_age_hours=1e9, max_files=1_000_000_000)
    assert deleted == 0
    assert (kv_dir / big).exists()


# ===========================================================================
# Save-time hard-cap gate
# ===========================================================================

class _SaveHarness:
    """httpx fake: GET /slots returns N populated slots; POST writes the bin.
    Records every POST so a test can tell 'the save was attempted' from 'the
    gate declined it before any engine call'."""

    def __init__(self, kv_dir, slots_payload, bin_bytes: int = 50):
        self.kv_dir = kv_dir
        self.slots_payload = slots_payload
        self.bin_bytes = bin_bytes
        self.posts: list = []
        self.gets: list = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
        def raise_for_status(self):
            return None
        def json(self):
            return self._payload

    def _client(self):
        outer = self

        class _Client:
            async def __aenter__(self_):
                return self_
            async def __aexit__(self_, *a):
                return False
            async def get(self_, url):
                outer.gets.append(url)
                if "/slots" in url and "action=save" not in url:
                    return _SaveHarness._Resp(outer.slots_payload)
                return _SaveHarness._Resp({})
            async def post(self_, url, json=None, **kw):
                outer.posts.append((url, json))
                if "action=save" in url and json and "filename" in json:
                    target = os.path.join(str(outer.kv_dir), json["filename"])
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with open(target, "wb") as f:
                        f.write(b"\x00" * outer.bin_bytes)
                return _SaveHarness._Resp({"n_saved": 1, "n_written": outer.bin_bytes})

        return _Client()


class _FakeHttpx:
    @staticmethod
    def AsyncClient(*a, **k):
        # captured per-call via the active harness on the class
        return _FakeHttpx._active._client()
    Timeout = staticmethod(lambda *a, **k: None)


def _install_httpx(monkeypatch, harness):
    _FakeHttpx._active = harness
    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)


def _slots_payload(thread_id: str, n_prompt_tokens: int = 100):
    return [{"id": 0, "n_prompt_tokens": n_prompt_tokens, "thread_id": thread_id,
             "n_context": 4096, "n_remaining": 2048}]


@pytest.mark.asyncio
async def test_save_gate_skips_snapshot_when_hard_floor_holds(mgr, kv_dir, monkeypatch):
    """Save gate skip. The dir is at/over cap and a HARD active bin holds the floor, so
    the GC cannot make room: the save-time gate DECLINES the on-disk snapshot
    (the live KV stays in-engine) — no engine POST, no new bin, returns False.

    Without the gate the engine save POST would fire and a new bin would be
    written (the cap would only be checked at the GC): the result would be
    truthy and a POST recorded.
    """
    active_tid = "active-tid"
    _set_hard_active(mgr, active_tid)
    mgr.runtime.kv.ram_cache_max_bytes = 100  # tiny
    # 200-byte HARD bin -> dir already 200 >= cap 100, and it cannot be evicted
    _write_kv_bin(kv_dir, "m.hard.active.bin", 200, mtime=1000.0,
                  thread_hash=mgr._thread_hash(active_tid))

    harness = _SaveHarness(kv_dir, _slots_payload(active_tid), bin_bytes=50)
    _install_httpx(monkeypatch, harness)

    slot = Slot.new(_QWEN, thread_id=active_tid, admission_ctx_len=100,
                    client_meta={"session_id": f"sess-{active_tid}"})
    result = await mgr._save_slot_kv(59500, _QWEN, slot)

    assert result is False, (
        "save-time gate did NOT decline the over-cap snapshot — if the save "
        "funnel never checks ram_cache_max_bytes, the over-cap snapshot is written."
    )
    assert harness.posts == [], "gate must decline BEFORE any engine save POST"
    bins = sorted(f for f in os.listdir(str(kv_dir)) if f.endswith(".bin"))
    assert bins == ["m.hard.active.bin"], (
        f"no new snapshot bin should be written when the save is declined; "
        f"found {bins}"
    )


@pytest.mark.asyncio
async def test_save_gate_makes_room_by_evicting_unprotected_then_saves(mgr, kv_dir, monkeypatch):
    """Save gate make-room. The dir is over cap with only UNPROTECTED bins: the gate
    drives the GC to evict them, then the save proceeds — total ends <= cap.

    Without the gate the save would add its bin without reclaiming, so the dir
    would end over cap (existing 200 + new 50 = 250 > 150) and the existing bin
    would not be reclaimed.
    """
    mgr.runtime.kv.ram_cache_max_bytes = int(150)
    old = _write_bin(kv_dir, "m.old.unpinned.bin", 200, mtime=1000.0)  # unprotected, over cap

    harness = _SaveHarness(kv_dir, _slots_payload("some-active-tid"), bin_bytes=50)
    _install_httpx(monkeypatch, harness)

    slot = Slot.new(_QWEN, thread_id="some-active-tid", admission_ctx_len=100,
                    client_meta={"session_id": "sess-some"})
    result = await mgr._save_slot_kv(59500, _QWEN, slot)

    assert result is True, "save should proceed once room is made"
    assert not (kv_dir / old).exists(), (
        "gate should have evicted the over-cap unprotected bin to make room"
    )
    assert _total_bin_bytes(kv_dir) <= int(150), (
        f"after make-room the dir must be <= cap; got "
        f"{_total_bin_bytes(kv_dir)} bytes (the gate did not make room)."
    )
    assert harness.posts, "save must have reached the engine after making room"
