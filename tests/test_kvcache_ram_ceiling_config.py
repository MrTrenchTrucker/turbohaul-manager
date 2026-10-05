"""RAM KV-cache byte ceiling as a real config field.

Before this fix, `_gc_kv_cache`'s total-bytes ceiling was read directly from
`os.environ.get("TURBOHAUL_KVCACHE_MAX_BYTES")` inside a bespoke free function
(`_kvcache_ceiling_bytes()`, in manager.py) -- invisible to `/api/config`, the
schema endpoint, and the Settings UI, unlike its disk-tier sibling
`persist.max_bytes` (config.py, already a real field). This change promotes it to
`KVConfig.ram_cache_max_bytes`, mirroring `persist.max_bytes` exactly (same
`_ENV_MAP` mechanism, same env var name, same default).

Covers:
  1. The ceiling is readable/settable through the config surface
     -- proven at the model level (KVConfig field exists with the
     right default) AND at the env-override level (the SAME env var
     TURBOHAUL_KVCACHE_MAX_BYTES still works, now via apply_env_overrides).
  2. The GC still enforces the ceiling, oldest-first, reading the value LIVE
     from `self.runtime.kv.ram_cache_max_bytes` -- proven by writing bins of
     known size/age, setting a small ceiling via runtime.kv directly (the same
     mutation PUT /api/config performs when it replaces mgr.runtime wholesale),
     and confirming oldest-first eviction with NO restart of the manager
     (the live-reload behaviour).

Non-vacuity: test_ceiling_is_a_real_config_field FAILS on unmodified
code (KVConfig has no such field -> AttributeError / extra="forbid"
ValidationError), PASSES after the fix.
"""
import os

import pytest

from turbohaul.config import (
    BootConfig,
    KVConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
    apply_env_overrides,
    TurbohaulConfig,
)
from turbohaul.manager import TurbohaulManager


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
    import turbohaul.subprocess_mgr as subprocess_mgr
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


def _write_bin(kv_dir, name: str, size: int, mtime: float) -> str:
    """A plain, un-pinned .bin with no .json sidecar -- no clean_prefix meta,
    no thread_hash, so it is never pin-elected and never live-protected
    (`_live_protected_thread_hashes()` is empty on a fresh manager with no
    active state) -- a pure, simple candidate for the bytes-ceiling scan."""
    p = kv_dir / name
    p.write_bytes(b"\x00" * size)
    os.utime(p, (mtime, mtime))
    return name


# === Readable/settable through the config surface ===

def test_ceiling_is_a_real_config_field_mirroring_persist_max_bytes():
    """FAILS on unmodified code (no such field on KVConfig -- either
    AttributeError reading it, or a ValidationError from extra='forbid' when
    constructing with it). PASSES after the fix: same default (20 GiB) as the
    old bespoke code fallback -- this promotion does not change the value."""
    kv = KVConfig()
    assert kv.ram_cache_max_bytes == 20 * 1024 ** 3
    # settable, mirroring persist.max_bytes's own shape exactly
    kv2 = KVConfig(ram_cache_max_bytes=5 * 1024 ** 3)
    assert kv2.ram_cache_max_bytes == 5 * 1024 ** 3


def test_env_var_still_works_same_name_via_standard_override_path(monkeypatch):
    """The SAME env var (TURBOHAUL_KVCACHE_MAX_BYTES, unrenamed) still sets the
    ceiling -- now via the standard apply_env_overrides mechanism every other
    config knob uses (mirrors TURBOHAUL_KVCACHE_PERSIST_MAX_BYTES for
    persist.max_bytes), not a bespoke os.environ.get() inside manager.py."""
    monkeypatch.setenv("TURBOHAUL_KVCACHE_MAX_BYTES", "25769803776")  # 24 GiB, an example value
    cfg = TurbohaulConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path="/tmp/x/blobs", manifests_path="/tmp/x/manifests",
            import_allowed_root="/tmp/x/import", state_db_path="/tmp/x/state.sqlite",
        ),
        runtime=RuntimePathsConfig(llama_server_binary="/tmp/x/fake"),
        ui=UIConfig(static_path="/tmp/x/ui_dist"),
        queue=QueueConfig(),
        pull=PullConfig(),
    )
    overridden = apply_env_overrides(cfg)
    assert overridden.kv.ram_cache_max_bytes == 25769803776


def test_config_surface_exposes_kv_section_including_new_field():
    """/api/config's live view is mgr.runtime.model_dump() -- confirm the new
    field round-trips through that exact call (what get_config() in
    api/main.py actually does), and appears in KVConfig's own schema dump
    (what config_schema.py's get_config_schema() iterates)."""
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    dumped = runtime.model_dump(mode="json")
    assert "ram_cache_max_bytes" in dumped["kv"]
    assert dumped["kv"]["ram_cache_max_bytes"] == 20 * 1024 ** 3
    schema_defaults = KVConfig().model_dump(mode="json")
    assert "ram_cache_max_bytes" in schema_defaults


# === GC still enforces it, live, no restart ===

@pytest.mark.asyncio
async def test_gc_ceiling_still_enforced_oldest_first_via_runtime_kv(mgr, kv_dir):
    """Three bins, ages oldest->newest, each 10 MiB. Ceiling set small enough
    that exactly the two oldest must go. Reads self.runtime.kv.ram_cache_max_bytes
    live -- proves the config plumbing actually drives the GC, not just that
    the field exists in isolation.

    SCOPE -- WHICH AXIS THIS PINS, AND WHICH IT DOES NOT. The
    production sort key is ``(".shadow." in fn, mtime)`` -- TWO axes. Every
    fixture below is non-shadow, so all three land in one tier and this test
    sees the MTIME axis only. Measured, not assumed: it FAILS if the order is
    reversed to newest-first, and it passes unchanged whether the tier
    component is present or absent. That is not a defect in this test --
    oldest-first is exactly what its name claims and exactly what it proves.
    The TIER axis is covered separately by
    ``test_shadow_tier_is_evicted_last_change_detector``; do not read this
    test's green as saying anything about it."""
    ten_mib = 10 * 1024 * 1024
    _write_bin(kv_dir, "m.a.slot0.bin", ten_mib, mtime=1000.0)   # oldest
    _write_bin(kv_dir, "m.b.slot0.bin", ten_mib, mtime=2000.0)   # middle
    _write_bin(kv_dir, "m.c.slot0.bin", ten_mib, mtime=3000.0)   # newest

    # Ceiling = 15 MiB: 30 MiB total -> must evict oldest-first until <= 15 MiB
    # -> exactly the two oldest bins go, the newest survives.
    mgr.runtime.kv.ram_cache_max_bytes = int(15 * 1024 * 1024)

    deleted = await mgr._gc_kv_cache(max_age_hours=1e9, max_files=1_000_000)

    assert deleted == 2
    assert not (kv_dir / "m.a.slot0.bin").exists()
    assert not (kv_dir / "m.b.slot0.bin").exists()
    assert (kv_dir / "m.c.slot0.bin").exists()


@pytest.mark.asyncio
async def test_ceiling_change_is_picked_up_live_no_restart(mgr, kv_dir):
    """Live reload: mutate self.runtime.kv.ram_cache_max_bytes the same way PUT
    /api/config does (config_put.py's `mgr.runtime = new_runtime` -- replacing
    the whole runtime object) BETWEEN two GC calls on the SAME manager
    instance, with no re-construction of TurbohaulManager -- proves the next
    GC pass honors the new ceiling immediately."""
    ten_mib = 10 * 1024 * 1024
    _write_bin(kv_dir, "m.a.slot0.bin", ten_mib, mtime=1000.0)
    _write_bin(kv_dir, "m.b.slot0.bin", ten_mib, mtime=2000.0)

    # Ceiling generous (30 MiB) -> first pass evicts nothing.
    mgr.runtime.kv.ram_cache_max_bytes = int(30 * 1024 * 1024)
    deleted_first = await mgr._gc_kv_cache(max_age_hours=1e9, max_files=1_000_000)
    assert deleted_first == 0
    assert (kv_dir / "m.a.slot0.bin").exists()
    assert (kv_dir / "m.b.slot0.bin").exists()

    # Simulate a live PUT /api/config: REPLACE mgr.runtime wholesale (exactly
    # what config_put.py's `mgr.runtime = new_runtime` does), tightening the
    # ceiling to 15 MiB -- 20 MiB total (2x10 MiB) must drop to <= 15 MiB, so
    # exactly the oldest one bin goes. No TurbohaulManager re-construction, no
    # restart.
    new_runtime = mgr.runtime.model_copy(deep=True)
    new_runtime.kv.ram_cache_max_bytes = int(15 * 1024 * 1024)
    mgr.runtime = new_runtime

    deleted_second = await mgr._gc_kv_cache(max_age_hours=1e9, max_files=1_000_000)
    assert deleted_second == 1
    assert not (kv_dir / "m.a.slot0.bin").exists()  # oldest evicted
    assert (kv_dir / "m.b.slot0.bin").exists()       # newest survives


# === The TIER axis of the eviction sort — CHANGE DETECTOR ===

@pytest.mark.asyncio
async def test_shadow_tier_is_evicted_last_change_detector(mgr, kv_dir):
    """⚠ THIS TEST ASSERTS CURRENCY, NOT CORRECTNESS. Read this before you fix it.

    WHAT IT OBSERVES. The ceiling GC's sort key is
    ``evictable.sort(key=lambda x: (".shadow." in x[0], x[2]))`` -- TIERED: every
    non-shadow bin is evicted before any ``.shadow.`` bin, and mtime orders only
    WITHIN a tier. So the OLDEST file on disk survives an over-cap pass if it is
    a shadow and a younger non-shadow is available to take instead. That is what
    the fixture below builds and what the assertions record.

    WHY IT IS DELIBERATE. The ceiling GC is tiered so that the think-free
    ``.shadow`` KV bin is not over-cap-evicted before reuse while many pinned
    DEAD-session clean anchors hold a large protected floor; otherwise a
    returning client pays a full re-prefill instead of the fast restore the
    lifecycle promises. Reordering the ceiling eviction to reap dead
    anchors before shadows is what prevents that. **If this key is ever "simplified"
    back to plain mtime, that production failure returns and nothing
    else in this suite notices.** That is the whole reason this test exists.

    ⛔ WHY IT DOES NOT ASSERT THAT THIS IS CORRECT. The documentation does not state
    this ordering. It says only that the oldest entries are pushed out
    first -- which, read literally, describes the STRICT
    sort, i.e. the one production does NOT do. This is a documentation gap:
    the tiered ordering is not yet settled as the contract, and
    **it has not been decided**
    whether it is the intended behaviour. Asserting the tiered order as
    REQUIRED would therefore pin behaviour that contradicts the documented
    literal text -- and pinning today's behaviour as intended is a known
    failure mode of change-detector tests.

    ⇒ SO IF THIS TEST GOES RED, THAT IS NOT AUTOMATICALLY A BUG. It means
    somebody changed the tier component of the sort. The correct response is to
    settle the intended ordering contract FIRST and then decide, not to edit this assertion
    and not to "fix" the code to match it.

    ★ EXIT CONDITION -- this paragraph is what stops it being a change detector
    forever. **When the tiered ordering is documented as the contract, this test is upgraded
    to assert the tiered order as REQUIRED, its name loses the
    ``_change_detector`` suffix, it cites the documented contract instead of this
    paragraph, and THIS PARAGRAPH IS DELETED.** If the ordering is REJECTED
    instead, this test is deleted TOGETHER WITH the behaviour it records -- it
    must never outlive the code it describes. Same discipline as an
    ``xfail(strict=True)`` marker: the graduation is written down, so the state
    cannot become permanent by inattention.

    ◊ WHAT THIS TEST DOES **NOT** SEE, and where the rest is covered. It forces
    exactly ONE eviction, so it observes only the MINIMUM of the sort: it learns
    THAT the shadow was demoted and never HOW FAR. A merely *bounded* demotion
    (an mtime bonus rather than a tier) leaves it green, and no other test
    in the suite would catch that either. The universal form
    ("EVERY non-shadow before ANY shadow") is pinned separately by
    ``test_shadow_tier_is_total_not_bounded_change_detector`` below; neither
    test asserts that the ordering is CORRECT.

    (The sort key demotes every non-shadow bin, and the comment above the key
    describes exactly that, not only dead clean_prefix anchors. The key and
    the comment therefore agree, so there is no divergence
    to hunt for here. Prose only;
    no assertion here
    depends on this note.)
    """
    ten_mib = 10 * 1024 * 1024
    # The shadow is the OLDEST file on disk. That is what makes the two orderings
    # disagree: strict oldest-first takes it, the tiered key spares it.
    _write_bin(kv_dir, "m.old.shadow.slot0.bin", ten_mib, mtime=1000.0)
    _write_bin(kv_dir, "m.mid.slot0.bin", ten_mib, mtime=2000.0)
    _write_bin(kv_dir, "m.new.slot0.bin", ten_mib, mtime=3000.0)

    # NON-VACUITY PRECONDITIONS, asserted before the target: if any of these
    # stops holding, the two orderings agree again and the assertions below go
    # green without discriminating anything.
    assert (kv_dir / "m.old.shadow.slot0.bin").exists()
    assert (kv_dir / "m.mid.slot0.bin").exists()
    assert (kv_dir / "m.new.slot0.bin").exists()
    ages = {p.name: p.stat().st_mtime for p in kv_dir.glob("*.bin")}
    assert min(ages, key=ages.get) == "m.old.shadow.slot0.bin", (
        "fixture no longer discriminates: the shadow bin must be the OLDEST on "
        "disk, or strict-oldest-first and the tiered key would pick the same "
        f"victim and this test would prove nothing. ages: {ages!r}"
    )
    assert sum(".shadow." in n for n in ages) == 1, (
        f"fixture must contain exactly one .shadow. bin. names: {sorted(ages)!r}"
    )

    # 30 MiB total, ceiling 25 MiB -> exactly ONE bin must go.
    mgr.runtime.kv.ram_cache_max_bytes = int(25 * 1024 * 1024)
    deleted = await mgr._gc_kv_cache(max_age_hours=1e9, max_files=1_000_000)

    assert deleted == 1, (
        f"expected exactly one eviction to get 30 MiB under a 25 MiB ceiling, "
        f"got {deleted}"
    )
    assert not (kv_dir / "m.mid.slot0.bin").exists(), (
        "TIER COMPONENT OF THE EVICTION SORT HAS CHANGED. The oldest NON-SHADOW "
        "bin (m.mid) should have been taken, leaving the older .shadow. bin "
        "alone. This is the behaviour the tiered eviction exists to stop "
        "shadow bins being reaped ahead of dead anchors. See this test's "
        "docstring before changing anything: a red here means the intended ordering "
        "needs a decision, NOT that this assertion needs editing."
    )
    assert (kv_dir / "m.old.shadow.slot0.bin").exists(), (
        "the .shadow. bin was evicted even though it is protected by TIER, not "
        "by age -- it is the OLDEST file here, so a strict oldest-first sort "
        "would take it and the tiered sort must not. This is the exact "
        "regression the tiered eviction addresses; see the docstring."
    )
    assert (kv_dir / "m.new.slot0.bin").exists()


# === The STRENGTH of that tier — total, not bounded. CHANGE DETECTOR ===

@pytest.mark.asyncio
async def test_shadow_tier_is_total_not_bounded_change_detector(mgr, kv_dir):
    """⚠ THIS TEST ASSERTS CURRENCY, NOT CORRECTNESS. Read this before you fix it.

    WHAT IT OBSERVES, AND WHY IT IS NOT A DUPLICATE OF THE TEST ABOVE. The sort
    key ``(".shadow." in x[0], x[2])`` makes the tier DOMINANT: every non-shadow
    goes before ANY shadow, no matter how much older the shadow is. Its sibling
    ``test_shadow_tier_is_evicted_last_change_detector`` forces exactly ONE
    eviction, so it can only see the MINIMUM of the sort -- that the shadow was
    not taken first. It cannot see the shadow's rank against the REST.

    ⇒ THE HOLE THAT LEAVES, MEASURED NOT ARGUED. Replace the tier
    with a merely BOUNDED demotion -- ``key=lambda x: x[2] + (1500.0 if
    ".shadow." in x[0] else 0.0)``, which lifts this fixture's shadow to an
    effective 2500, i.e. BETWEEN m.mid (2000) and m.new (3000) -- and the shadow
    stops being last while still not being first. In that case the sibling
    above stays GREEN, and so does every other test in the suite: a shadow
    could be reclaimed ahead of a newer ordinary bin with nothing red anywhere.
    This test is the instrument for that half.

    HOW IT DISCRIMINATES. Three 10 MiB bins under a 15 MiB ceiling force exactly
    TWO evictions, so the sort is observed two deep instead of one:
      * TIERED (production): both non-shadows go, and the OLDEST FILE ON DISK --
        the shadow -- is the SOLE SURVIVOR.
      * BOUNDED demotion: m.mid then the SHADOW go, and m.new survives.
    Note ``deleted == 2`` holds either way; it is a fixture check, not the
    discriminator. The discriminator is WHICH bin is left.

    WHY THE BEHAVIOUR IS DELIBERATE. Same rationale as the sibling: the
    ceiling GC keeps the
    think-free ``.shadow`` bin from being over-cap-evicted before reuse while many pinned
    DEAD-session clean anchors hold a large protected floor, which would cost a returning
    client the fast restore the lifecycle promises.

    ⛔ WHY IT DOES NOT ASSERT THAT THIS IS CORRECT. The documentation does not state
    the ordering: it says only that the oldest entries are pushed out
    first, which read literally describes the STRICT sort production does NOT
    do. The tiered ordering has NOT been settled as the contract. The
    universal form is also what the comment above the sort describes --
    and it is an open question
    whether the boundary should be narrowed to the dead-anchor case. Asserting
    it as REQUIRED would pre-empt that decision.

    ⇒ SO IF THIS TEST GOES RED, THAT IS NOT AUTOMATICALLY A BUG. It means the
    tier stopped being total -- someone weakened it to a bounded demotion, or
    removed it. Get that open question settled FIRST. Do not edit this
    assertion and do not "fix" the code to match it.

    ★ EXIT CONDITION. **If the tiered order is ADOPTED as the contract, this test is upgraded to
    assert it as REQUIRED, loses its ``_change_detector`` suffix, cites the
    documented contract, and THIS PARAGRAPH IS DELETED. If that question is resolved by
    NARROWING the boundary, this test is deleted together with the behaviour it
    records** -- it must never outlive the code it describes, and on that path
    the comment above the sort and any documentation of the
    two-tier eviction come down with it.
    """
    ten_mib = 10 * 1024 * 1024
    # Same shape as the sibling: the shadow is the OLDEST file on disk. Here it
    # must survive TWO evictions, not one -- that is the whole difference.
    _write_bin(kv_dir, "m.old.shadow.slot0.bin", ten_mib, mtime=1000.0)
    _write_bin(kv_dir, "m.mid.slot0.bin", ten_mib, mtime=2000.0)
    _write_bin(kv_dir, "m.new.slot0.bin", ten_mib, mtime=3000.0)

    # NON-VACUITY PRECONDITIONS. If any stops holding, a bounded demotion and a
    # total tier pick the same survivor and this test discriminates nothing.
    ages = {p.name: p.stat().st_mtime for p in kv_dir.glob("*.bin")}
    assert len(ages) == 3, f"fixture must be exactly three bins. names: {sorted(ages)!r}"
    assert sum(".shadow." in n for n in ages) == 1, (
        f"fixture must contain exactly one .shadow. bin. names: {sorted(ages)!r}"
    )
    assert min(ages, key=ages.get) == "m.old.shadow.slot0.bin", (
        "fixture no longer discriminates: the shadow must be the OLDEST bin, or "
        "surviving is explained by age and not by the tier. "
        f"ages: {ages!r}"
    )
    assert ages["m.old.shadow.slot0.bin"] < ages["m.mid.slot0.bin"] < ages["m.new.slot0.bin"], (
        "the two non-shadow bins must BOTH be younger than the shadow and "
        "distinct from each other, or 'the shadow outranks every non-shadow' is "
        f"not observable two evictions deep. ages: {ages!r}"
    )

    # 30 MiB total, ceiling 15 MiB -> exactly TWO bins must go (one is not
    # enough: 20 MiB is still over). Two is what makes the sort observable
    # beyond its minimum.
    mgr.runtime.kv.ram_cache_max_bytes = int(15 * 1024 * 1024)
    deleted = await mgr._gc_kv_cache(max_age_hours=1e9, max_files=1_000_000)

    assert deleted == 2, (
        f"fixture check, not the discriminator: 30 MiB under a 15 MiB ceiling "
        f"needs exactly two 10 MiB evictions, got {deleted}"
    )
    assert (kv_dir / "m.old.shadow.slot0.bin").exists(), (
        "THE TIER IS NO LONGER TOTAL. The .shadow. bin was reclaimed while a "
        "non-shadow bin survived, so `.shadow.` is now at best a BOUNDED "
        "demotion (an age bonus) rather than a dominant sort component. The "
        "sibling one-eviction test cannot catch this and neither can the rest "
        "of the suite. See this test's docstring: a red here means the open "
        "ordering question needs a decision, NOT that this assertion needs editing."
    )
    assert not (kv_dir / "m.mid.slot0.bin").exists(), (
        "the oldest NON-shadow bin should have been the first victim"
    )
    assert not (kv_dir / "m.new.slot0.bin").exists(), (
        "m.new is the YOUNGEST file on disk and still a non-shadow, so under a "
        "dominant tier it must be evicted ahead of the much older shadow. It "
        "survived, which means age outranked the tier for this pair."
    )
    assert [p.name for p in kv_dir.glob("*.bin")] == ["m.old.shadow.slot0.bin"], (
        "the shadow must be the SOLE survivor of the two-eviction pass"
    )
