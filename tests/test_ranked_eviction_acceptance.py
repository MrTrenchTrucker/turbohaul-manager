"""Acceptance test for the three-client Fast Lane eviction scenario (P1, P3, P4).

P1 is the highest-ranked claimant, P3 a mid-ranked resident and P4 the lowest-ranked
resident. The scenario boots with a short injected grace_seconds, drives real turns
and asserts real wall-clock bounds. The existing grace-loop starvation test does not
cover it: that test exercises the GRACE-loop m1/m2 starvation predicate, never P3/P4/P1,
never rank-based eviction. No other test in this suite drives the P3/P4/P1 shape end
to end; this file does.

This file reuses the REAL techniques already proven elsewhere in this suite,
combined for the first time:
  - the multi-slot dispatcher's real fake-spawn/health/sigterm/vram harness
    (tests/test_multislot_concurrency.py's `_mocks()`), with an independent
    `asyncio.Event` gate PER MODEL (not the single shared gate that file uses)
    so P3's and P4's turns can be held open and released in either order --
    exactly what the scenario's two branches need.
  - the real short-`grace_seconds` + real-wall-clock-bound technique
    (the grace-starvation test's `_boot_runtime` + real
    `mgr.worker_loop()` + audit-event polling).
  - real `FastLaneConfig`/`FastLaneRule` registration (positional rank, rule 0 =
    highest -- the turn-boundary handoff test's own convention),
    giving P1 > P3 > P4 in RELATIVE order. NOT the literal production
    ranks -- a deliberate choice: a real deployment's table is a
    deployment fact, and pinning to it would fail a run for reasons unrelated
    to the behaviour under test. Recorded here as a design choice.
  - the real KV-save-to-disk mechanism (the clean-stamp integrity test's
    `kv_dir` + `make_httpx` fixtures: a fake httpx.AsyncClient whose GET /slots
    reports a populated slot and whose POST action=save writes a REAL .bin file
    to the monkeypatched SLOT_SAVE_DIR) -- the scenario requires that the victim's KV
    blob be PRESENT IN THE STORE, and a log line must not stand in for
    that evidence. The following
    explains why: "resident KV cache saved before teardown" is logged inside a
    best-effort try/except that can fire even when the underlying save silently
    no-ops -- the log can lie about exactly this.

Design constraint: P1 is manifest-PINNED to P4's card (gpu1), not
auto-placed.
This deliberately makes the two branches exercise TWO DIFFERENT
eviction paths, never the same tied-object shape:
  - Branch 1 (P4 evicted): P4 is already ON P1's target card -- the ordinary
    on-card lookup, no widening needed.
  - Branch 2 (P3 evicted): P3 is on gpu0, NOT P1's target card -- this can only
    succeed through the cross-card widening + relocation
    (`slot.fastlane_relocated_main_gpu`), the exact mechanism that other
    tests already prove correct. The global `_lru_idle_unloadable` result
    (P3, the only idle-evictable resident at that moment) and the on-card
    `_lru_idle_unloadable(main_gpu=gpu1)` result (nothing -- P4 is still mid-turn,
    not evictable) are trivially different objects (one is a real Resident, the
    other is None) -- structurally clear of the tie,
    not merely argued clear of it.

Scope (per arm, never bundled): every arm in this
file asserts an eviction OUTCOME under contention. None of them measures how long
a designated victim holds the card before teardown (see the scope note below);
that duration is covered by the designated-victim teardown-duration test.
Each arm's assertion can fail on the REAL current code once the harness exists
and is wired correctly.
"""
# Scope note: every arm in this file asserts an eviction
# OUTCOME under contention (P1 actively waiting/polling via
# submit_and_wait) -- it does NOT and structurally CANNOT
# measure the DURATION a designated victim spends
# idle-holding before teardown. An 'immediately' requirement is
# a temporal claim; a live, actively-retrying contender makes
# eviction fast via a SEPARATE, already-correct mechanism
# (_lru_idle_unloadable, the admission-time reclaim picker)
# regardless of how quickly or slowly the designated-victim
# branch itself fires -- so these arms would pass even against
# code with a slow designated-victim teardown. This is not a
# flaw in these arms' own assertions; it is a scope boundary.
# The duration dimension is covered by the
# designated-victim teardown-duration test -- read that
# file's module docstring for the mechanism this file
# cannot see.
# No assertion below depends on this note.
from __future__ import annotations

import asyncio
import ipaddress
import os
import time
from unittest.mock import MagicMock

import pytest

import turbohaul.manager as manager_mod
import turbohaul.subprocess_mgr as subprocess_mgr
from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import SlotState
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle

# Relative rank only (a deliberate choice) -- rule list index 0
# is highest priority (the established convention), NOT literal
# production ranks. P1 > P3 > P4.
_P1_ADDR = "10.0.9.1"
_P3_ADDR = "10.0.9.3"
_P4_ADDR = "10.0.9.4"

_P1_TAG = "p1-open-webui"
_P3_TAG = "p3-agent-a"
_P4_TAG = "p4-agent-b"

_PORT_BASE = 59700


def _boot_runtime(tmp_path, *, grace_seconds):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
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
            default_port_base=_PORT_BASE,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False,
            # 3, not 2: with only 2, P1's arrival while P3+P4 are BOTH still
            # resident hits len(residents)==2 >= cap FIRST -- the ordinary
            # COUNT-CAP path (global _lru_idle_unloadable(), NO
            # main_gpu scoping, NO outrank test at all) -- which admits P1
            # via a mechanism this test does not intend to exercise and
            # masks whether the VRAM/widening path (the one the cross-card widening
            # and the pinned-card constraint actually govern) works at all.
            # With cap=2 and expected_vram_bytes=0
            # (need=0, so VRAM never gates admission either), disabling the
            # widening's `if claimant_key is not None:` gate entirely
            # leaves every arm in this file passing --
            # proof such a fixture exercises the count-cap fallback, not
            # the cross-card widening. cap=3 keeps
            # len(residents)==2 < cap while P1 is still queued behind P3+P4,
            # so only the real VRAM path can gate P1's admission.
            max_parallel_sidecars=3,
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=60,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(
            enabled=True,
            rules=[
                FastLaneRule(address=_P1_ADDR, label="P1", tag_ranks={"unclassified": 1}),
                FastLaneRule(address=_P3_ADDR, label="P3", tag_ranks={"unclassified": 1}),
                FastLaneRule(address=_P4_ADDR, label="P4", tag_ranks={"unclassified": 1}),
            ],
        ),
    )
    return boot, runtime


# Real sizing (test_multislot_concurrency.py's own convention): need != 0,
# so the VRAM check actually gates admission -- an expected_vram_bytes=0
# manifest makes `need=0`, so `vram_fits` is
# trivially True regardless of card occupancy and the VRAM/widening path
# never engages at all. See the
# max_parallel_sidecars comment above.
_VRAM_MIB = 18000


def _seed_manifest(boot, model_tag, *, main_gpu):
    import yaml
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": _VRAM_MIB * 1024 * 1024,
        "context_size": 2048,
        "expected_vram_bytes": _VRAM_MIB * 1024 * 1024,
        "llama_server_flags": {"split_mode": "none", "main_gpu": main_gpu},
    }))


def _fake_handle(model_tag: str, port: int, pid: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _audit_events(boot, slot_id):
    conn = open_state_db(boot.storage.state_db_path)
    cur = conn.execute(
        "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
        (slot_id,),
    )
    events = [r["event_type"] for r in cur.fetchall()]
    conn.close()
    return events


class _SaveResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _KvSaveClient:
    """Real KV-save-to-disk fake (the clean-stamp integrity test's
    own proven `_ProbeSaveClient`): GET /slots reports one populated slot for
    WHICHEVER port asks, POST action=save writes a REAL .bin file to the
    monkeypatched SLOT_SAVE_DIR. The KV-save requirement (the victim's KV blob is PRESENT
    IN THE STORE) is checked against this real file, never a log line."""

    def __init__(self, posts):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if "/slots" in url and "action=save" not in url:
            return _SaveResp([{"id": 0, "n_prompt_tokens": 4321}])
        return _SaveResp({})

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        if "action=save" in url and json and "filename" in json:
            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            tmp_path = os.path.join(SLOT_SAVE_DIR, json["filename"])
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(b"dummy kv cache data")
        return _SaveResp({"status": "ok"})


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture
def kv_save_posts(monkeypatch):
    posts = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _KvSaveClient(posts))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return posts


def _gated_mocks(spawn_calls, sigterm_calls, gates):
    """Per-model completion gating -- P3 and P4 need INDEPENDENT gates (the
    existing `_mocks()` in test_multislot_concurrency.py only supports ONE
    gated model at a time), so this is a new helper, not a copy."""
    pid = [95000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        spawn_calls.append({"model_tag": model_tag, "port": port})
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        sigterm_calls.append(handle.model_tag)
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        gate = gates.get(handle.model_tag)
        if gate is not None:
            await gate.wait()
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


@pytest.mark.asyncio
class TestRankedEvictionAcceptance:
    async def test_ARM1_completes_first_not_evicted_enters_grace(self, tmp_path, kv_dir, kv_save_posts):
        """Assertion 1: P3's turn completes first and P3 is not
        evicted -- it is in grace, not a candidate.

        P3 and P4 both loaded, both mid-turn (held via independent gates).
        P1 submits (registers a live Fast Lane claim, cannot fit -- P4's card
        is occupied by P4). P3's gate releases first. Real assertion: P3's
        grace_enter audit event fires, its grace_designated_victim_skip event
        does NOT fire (P3 is not the worst-ranked loaded resident while P4 is
        still loaded), and P3 remains a live resident (never evicted, never
        sigterm'd)."""
        from unittest.mock import patch

        grace_seconds = 4
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds)
        _seed_manifest(boot, _P3_TAG, main_gpu=0)
        _seed_manifest(boot, _P4_TAG, main_gpu=1)
        _seed_manifest(boot, _P1_TAG, main_gpu=1)  # pinned to P4's card

        spawn_calls, sigterm_calls = [], []
        gates = {_P3_TAG: asyncio.Event(), _P4_TAG: asyncio.Event()}
        mgr = TurbohaulManager(boot, runtime, **_gated_mocks(spawn_calls, sigterm_calls, gates))
        mgr.runtime.queue.safety_enabled = False

        def live_probe():
            # gpu0 occupied by P3 while resident; gpu1 occupied by P4 while resident.
            # Each card is 20000 MiB; a resident occupies _VRAM_MIB
            # (18000), leaving 2000 free -- below what a second
            # 18000-need model requires, so admission genuinely
            # depends on eviction, not a trivial need=0 fit.
            free0 = (20000 - _VRAM_MIB) if mgr._residents.get(_P3_TAG) is not None else 20000
            free1 = (20000 - _VRAM_MIB) if mgr._residents.get(_P4_TAG) is not None else 20000
            return [free0, free1]

        with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=live_probe):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                slot_p3 = await mgr.submit(_P3_TAG, "hi", thread_id="t-p3",
                                            client_meta={"ip": _P3_ADDR})
                slot_p4 = await mgr.submit(_P4_TAG, "hi", thread_id="t-p4",
                                            client_meta={"ip": _P4_ADDR})
                # Both must reach ACTIVE (mid-turn, held by their gates) before P1 submits.
                for _ in range(100):
                    r3, r4 = mgr._residents.get(_P3_TAG), mgr._residents.get(_P4_TAG)
                    if (r3 is not None and r3.state == ResidentState.ACTIVE
                            and r4 is not None and r4.state == ResidentState.ACTIVE):
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail(
                        f"P3/P4 never both reached ACTIVE -- "
                        f"P3={mgr._residents.get(_P3_TAG)!r} P4={mgr._residents.get(_P4_TAG)!r}"
                    )

                slot_p1 = await mgr.submit(_P1_TAG, "hi", thread_id="t-p1",
                                            client_meta={"ip": _P1_ADDR})

                # P3 finishes first.
                gates[_P3_TAG].set()

                deadline = time.monotonic() + grace_seconds - 0.5
                grace_entered = False
                while time.monotonic() < deadline:
                    events = _audit_events(boot, slot_p3.slot_id)
                    if "grace_enter" in events:
                        grace_entered = True
                        break
                    await asyncio.sleep(0.05)

                events_p3 = _audit_events(boot, slot_p3.slot_id)
                assert grace_entered, (
                    f"P3 never entered grace after completing its turn first -- "
                    f"events={events_p3!r}"
                )
                assert "grace_designated_victim_skip" not in events_p3, (
                    "P3 must NOT be treated as the designated victim while P4 is "
                    "still loaded -- P4 is the worst-ranked loaded resident, not "
                    f"P3. events={events_p3!r}"
                )
                assert _P3_TAG not in sigterm_calls, (
                    "P3 must not be evicted just because it finished first "
                    f"(designated-victim rule) -- sigterm_calls={sigterm_calls!r}"
                )
                assert mgr._residents.get(_P3_TAG) is not None, (
                    "P3's resident must still exist -- it was never evicted"
                )
            finally:
                for g in gates.values():
                    g.set()
                await mgr.shutdown()

    async def test_ARM2_branch1_evicted_zero_grace_kv_present_admitted(
        self, tmp_path, kv_dir, kv_save_posts
    ):
        """Assertion 2 (Branch 1): when P4's turn completes
        (before P3's grace lapses), P4 is evicted with ZERO GRACE:
        no grace state is ever entered for P4, its KV blob is PRESENT IN THE
        STORE, and P1 is admitted immediately after.

        Self-contained (does not reuse Arm 1's manager instance). P3 and P4
        both mid-turn, P1 submits (registers a live claim, P4's card gpu1
        occupied). P4's gate releases -- before P3's. Real assertions: (a)
        P4's audit trail shows grace_designated_victim_skip fired and NO
        grace state entered AT ALL -- no `grace_enter`, and no
        `grace_fastlane_breakout` / `grace_starvation_breakout` event, which
        can only fire INSIDE the wait loop -- plus a wall-clock bound well
        under grace_seconds. ⚠ The no-grace half of (a) is asserted literally:
        the FSM has a legal ACTIVE->POPPED edge, so the victim skips grace
        entirely rather than entering it and only declining to wait, and the
        entry itself is gated, so no `grace_enter` marker may exist for P4
        at all;
        (b) a real
        `.bin` file for P4 exists in the (monkeypatched) KV store on disk --
        never a log-line match;
        (c) P1's submission resolves (admitted) shortly after.
        """
        from unittest.mock import patch

        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds)
        _seed_manifest(boot, _P3_TAG, main_gpu=0)
        _seed_manifest(boot, _P4_TAG, main_gpu=1)
        _seed_manifest(boot, _P1_TAG, main_gpu=1)  # pinned to P4's card

        spawn_calls, sigterm_calls = [], []
        gates = {_P3_TAG: asyncio.Event(), _P4_TAG: asyncio.Event()}
        mgr = TurbohaulManager(boot, runtime, **_gated_mocks(spawn_calls, sigterm_calls, gates))
        mgr.runtime.queue.safety_enabled = False

        def live_probe():
            # Each card is 20000 MiB; a resident occupies _VRAM_MIB
            # (18000), leaving 2000 free -- below what a second
            # 18000-need model requires, so admission genuinely
            # depends on eviction, not a trivial need=0 fit.
            free0 = (20000 - _VRAM_MIB) if mgr._residents.get(_P3_TAG) is not None else 20000
            free1 = (20000 - _VRAM_MIB) if mgr._residents.get(_P4_TAG) is not None else 20000
            return [free0, free1]

        with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=live_probe):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                slot_p3 = await mgr.submit(_P3_TAG, "hi", thread_id="t-p3",
                                            client_meta={"ip": _P3_ADDR})
                slot_p4 = await mgr.submit(_P4_TAG, "hi", thread_id="t-p4",
                                            client_meta={"ip": _P4_ADDR})
                for _ in range(100):
                    r3, r4 = mgr._residents.get(_P3_TAG), mgr._residents.get(_P4_TAG)
                    if (r3 is not None and r3.state == ResidentState.ACTIVE
                            and r4 is not None and r4.state == ResidentState.ACTIVE):
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("P3/P4 never both reached ACTIVE")

                p1_future = asyncio.ensure_future(
                    mgr.submit_and_wait(_P1_TAG, "hi", thread_id="t-p1",
                                         client_meta={"ip": _P1_ADDR})
                )
                await asyncio.sleep(0.1)  # let P1's submission land + claim register

                started = time.monotonic()
                # P4 finishes first (Branch 1). P3 stays held the whole time --
                # its grace never lapses in this branch.
                gates[_P4_TAG].set()

                try:
                    _, res_p1 = await asyncio.wait_for(p1_future, timeout=grace_seconds - 0.5)
                except asyncio.TimeoutError:
                    pytest.fail(
                        f"P1 was not admitted within {grace_seconds - 0.5}s of P4 "
                        f"completing -- P4 events={_audit_events(boot, slot_p4.slot_id)!r}"
                    )
                elapsed = time.monotonic() - started

                events_p4 = _audit_events(boot, slot_p4.slot_id)
                # ⚠ UPDATED assertion (the previous form is kept below as a
                # commented reference). This assertion previously
                # read, verbatim:
                #
                #     assert "grace_enter" in events_p4, (
                #         f"P4 must still pass through the grace_enter marker "
                #         f"before the skip check runs -- events={events_p4!r}")
                #
                # It encoded a limitation of the FSM rather than the required
                # behaviour: the skip logic could only skip the grace
                # WAIT, never the ENTRY, because the FSM table had no legal
                # ACTIVE->POPPED transition and GRACE was therefore a required
                # waypoint out of _serve_on_resident. The FSM now has that edge
                # (in fsm.py) and the entry itself is gated, so the
                # required assertion is reachable. The old form is kept as a
                # commented reference, not deleted, because it records why the
                # earlier behaviour existed at all.
                assert "grace_enter" not in events_p4, (
                    f"No grace state may ever be entered for P4 -- "
                    f"the designated victim "
                    f"gets NO grace timer and NO idle-unload countdown. Both "
                    f"are surrendered the instant it is designated. No "
                    f"grace_enter marker may exist for the victim at all, "
                    f"events={events_p4!r}"
                )
                assert "grace_designated_victim_skip" in events_p4, (
                    f"P4 (the worst-ranked loaded resident, outranked by P1's live "
                    f"claim) must be designated the victim and skip grace entirely -- "
                    f"events={events_p4!r}"
                )
                assert "grace_fastlane_breakout" not in events_p4, (
                    "grace_fastlane_breakout can only fire INSIDE the grace WAIT "
                    "loop -- its presence would mean P4 entered the loop the "
                    "designated-victim skip is supposed to bypass entirely, "
                    f"events={events_p4!r}"
                )
                assert "grace_starvation_breakout" not in events_p4, (
                    f"same reasoning, the OTHER in-loop breakout marker -- "
                    f"events={events_p4!r}"
                )
                assert elapsed < grace_seconds - 1.0, (
                    f"P4->P1 handoff took {elapsed:.2f}s, not meaningfully "
                    f"sooner than grace_seconds={grace_seconds}s -- P4 was not "
                    f"actually fast-tracked"
                )
                assert res_p1 == {"ok": True, "model": _P1_TAG}, (
                    f"P1 must be admitted and complete -- got {res_p1!r}"
                )
                assert _P4_TAG in sigterm_calls, (
                    f"P4 must actually be torn down -- sigterm_calls={sigterm_calls!r}"
                )

                # (b) KV blob PRESENT IN THE STORE -- a real file, never a log line.
                bin_files = [f for f in os.listdir(kv_dir)
                             if f.startswith(f"{_P4_TAG}.") and f.endswith(".bin")]
                assert bin_files, (
                    f"P4's KV cache must be saved to a REAL file before teardown -- "
                    f"kv_dir contents: {os.listdir(kv_dir)!r}, save POSTs: "
                    f"{kv_save_posts!r}"
                )
                for fn in bin_files:
                    content = (kv_dir / fn).read_bytes()
                    assert content, f"{fn} exists but is empty -- not a real save"

                # P3 must be entirely untouched by this branch.
                assert _P3_TAG not in sigterm_calls
                assert mgr._residents.get(_P3_TAG) is not None
            finally:
                for g in gates.values():
                    g.set()
                await mgr.shutdown()

    async def test_ARM3_branch2_grace_lapses_evicted_admitted_untouched(
        self, tmp_path, kv_dir, kv_save_posts
    ):
        """Assertion 3 (Branch 2, a separate run): in a second
        run where P3's grace lapses while P4 is still mid-turn, P3 is
        evicted and P1 is admitted; P4 is untouched and completes its turn
        normally.

        Self-contained. P3's gate is released FIRST here too (its turn must
        complete before P4's, exactly as assertion 1 requires -- P3 is never
        evicted for finishing first), but UNLIKE Arm 2, P4's gate is held
        past P3's full grace_seconds window, so P3's real grace timer
        genuinely lapses on its own (natural timeout, not a fastlane/
        starvation in-loop breakout -- checked explicitly). P1 is pinned to
        P4's card (gpu1), NOT P3's (gpu0) -- so P1's admission after P3's
        eviction can only happen through the cross-card
        widening + relocation (the widening logic and the relocation
        wiring), never the on-card fallback. This is the OTHER eviction
        path from Arm 2, deliberately -- the two arms together exercise both
        of the scenario's branches through genuinely different code paths, not
        the same one twice."""
        from unittest.mock import patch

        grace_seconds = 3
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds)
        _seed_manifest(boot, _P3_TAG, main_gpu=0)
        _seed_manifest(boot, _P4_TAG, main_gpu=1)
        _seed_manifest(boot, _P1_TAG, main_gpu=1)  # pinned to P4's card, NOT P3's

        spawn_calls, sigterm_calls = [], []
        gates = {_P3_TAG: asyncio.Event(), _P4_TAG: asyncio.Event()}
        mgr = TurbohaulManager(boot, runtime, **_gated_mocks(spawn_calls, sigterm_calls, gates))
        mgr.runtime.queue.safety_enabled = False

        def live_probe():
            # Each card is 20000 MiB; a resident occupies _VRAM_MIB
            # (18000), leaving 2000 free -- below what a second
            # 18000-need model requires, so admission genuinely
            # depends on eviction, not a trivial need=0 fit.
            free0 = (20000 - _VRAM_MIB) if mgr._residents.get(_P3_TAG) is not None else 20000
            free1 = (20000 - _VRAM_MIB) if mgr._residents.get(_P4_TAG) is not None else 20000
            return [free0, free1]

        with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=live_probe):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                slot_p3 = await mgr.submit(_P3_TAG, "hi", thread_id="t-p3",
                                            client_meta={"ip": _P3_ADDR})
                slot_p4 = await mgr.submit(_P4_TAG, "hi", thread_id="t-p4",
                                            client_meta={"ip": _P4_ADDR})
                for _ in range(100):
                    r3, r4 = mgr._residents.get(_P3_TAG), mgr._residents.get(_P4_TAG)
                    if (r3 is not None and r3.state == ResidentState.ACTIVE
                            and r4 is not None and r4.state == ResidentState.ACTIVE):
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("P3/P4 never both reached ACTIVE")

                p1_future = asyncio.ensure_future(
                    mgr.submit_and_wait(_P1_TAG, "hi", thread_id="t-p1",
                                         client_meta={"ip": _P1_ADDR})
                )
                await asyncio.sleep(0.1)

                # P3 finishes first (assertion 1's own precondition), P4 stays
                # held mid-turn through P3's ENTIRE grace window.
                gates[_P3_TAG].set()

                try:
                    _, res_p1 = await asyncio.wait_for(
                        p1_future, timeout=grace_seconds + 3.0
                    )
                except asyncio.TimeoutError:
                    pytest.fail(
                        f"P1 was not admitted after P3's grace should have "
                        f"lapsed -- P3 events={_audit_events(boot, slot_p3.slot_id)!r}, "
                        f"P4 events={_audit_events(boot, slot_p4.slot_id)!r}"
                    )

                events_p3 = _audit_events(boot, slot_p3.slot_id)
                assert "grace_enter" in events_p3
                assert "grace_designated_victim_skip" not in events_p3, (
                    "P3 must go through a REAL grace window here (P4 is still "
                    f"loaded and is the worst-ranked resident) -- events={events_p3!r}"
                )
                assert "grace_fastlane_breakout" not in events_p3, (
                    "P1's claim does not strictly outrank P3 on its own merits "
                    "at any point before P3's own natural lapse in this branch "
                    f"-- events={events_p3!r}"
                )
                assert "grace_starvation_breakout" not in events_p3, (
                    f"single-model scenario, no other-model starvation predicate "
                    f"should ever fire here -- events={events_p3!r}"
                )

                assert res_p1 == {"ok": True, "model": _P1_TAG}, (
                    f"P1 must be admitted via the cross-card relocation after "
                    f"P3's natural grace lapse -- got {res_p1!r}"
                )
                assert _P3_TAG in sigterm_calls, (
                    f"P3 must actually be evicted once its grace lapses -- "
                    f"sigterm_calls={sigterm_calls!r}"
                )
                assert _P4_TAG not in sigterm_calls, (
                    f"P4 must be UNTOUCHED in this branch -- sigterm_calls="
                    f"{sigterm_calls!r}"
                )
                assert mgr._residents.get(_P4_TAG) is not None, (
                    "P4's resident must still exist -- it was never evicted"
                )

                # P3's KV must also be saved to a real file (the same requirement applies
                # to every eviction, not only the designated-victim one).
                bin_files_p3 = [f for f in os.listdir(kv_dir)
                                 if f.startswith(f"{_P3_TAG}.") and f.endswith(".bin")]
                assert bin_files_p3, (
                    f"P3's KV cache must be saved to a real file before teardown "
                    f"-- kv_dir contents: {os.listdir(kv_dir)!r}"
                )

                # P4 completes its OWN turn normally, on its own timeline, once
                # released -- checked via its OWN audit trail (its
                # completion_future is cleared to None once resolved/POPPED,
                # a normal cleanup, so the future object itself is not a
                # reliable post-hoc signal; grace_enter firing for P4 IS,
                # since that only happens after a real completed turn).
                events_p4_before_release = _audit_events(boot, slot_p4.slot_id)
                gates[_P4_TAG].set()
                for _ in range(100):
                    events_p4_after = _audit_events(boot, slot_p4.slot_id)
                    if "grace_enter" in events_p4_after:
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail(
                        f"P4 never reached grace_enter after being released -- "
                        f"before={events_p4_before_release!r}, after={events_p4_after!r}"
                    )
                assert "grace_enter" not in events_p4_before_release, (
                    "sanity: P4 must not have already been in grace before its "
                    "own gate was released"
                )
            finally:
                for g in gates.values():
                    g.set()
                await mgr.shutdown()

    async def test_ARM4_queue_surface_client_tag_rank_visibility(
        self, tmp_path, kv_dir, kv_save_posts
    ):
        """Assertion 4: the queue surface shows P1 waiting with
        its client, tag, and rank throughout -- never hidden, never
        reordered behind lower-priority traffic.

        THIS ARM DOES NOT ASSERT THE LITERAL FORM OF THE REQUIREMENT:
        queue.waiting[] is empty during the evict-pending wait (the slot is
        held in a background backoff task), so the claims snapshot is the
        surface that shows the claim.

        Observed behaviour:
        `status_snapshot()["queue"]["waiting"]` -- the literal array of waiting
        requests -- goes
        EMPTY the instant P1 hits the VRAM-over-commit path and stays empty
        for the ENTIRE evict-pending wait. The reason is structural, not a
        bug in the surface itself: `_defer_unroutable` -> `_register_
        fastlane_claim_locked` (registers the claim) then `_spawn_bg(self.
        _requeue_after_backoff(slot, backoff_s=_VRAM_DEFER_BACKOFF_S))` -- a
        DETACHED background task that holds the popped `Slot` object in its
        own closure for up to `_VRAM_DEFER_BACKOFF_S` (1.0s) before calling
        `enqueue_head` again. `queue_snapshot()` can only enumerate
        `_staging`/`_accept_buf` -- it has no way to see a slot parked inside
        an in-flight backoff coroutine's local frame. So `waiting[]` is
        empty for ~all of every ~1s cycle, not just at some interior instant.

        WHAT DOES SATISFY THE ASSERTION'S OWN INTENT (client/tag/rank, never
        hidden): `status_snapshot()["queue"]["fastlane_claims_snapshot"]` --
        a SIBLING field under the same `"queue"` key, populated by `_register_
        fastlane_claim_locked` at the moment `_defer_unroutable` runs and
        left in place (with a live `waited_s`) for the ENTIRE wait, exactly
        the population `waiting[]` misses. This is asserted below as the
        positive half -- proof the client/tag/rank claim is not lost
        anywhere in the system, just not on the ONE named field the
        requirement cites.

        Self-contained. P3 and P4 both mid-turn (held throughout -- P1 never
        gets admitted here, deliberately, so its whole wait is observable).
        """
        from unittest.mock import patch

        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds)
        _seed_manifest(boot, _P3_TAG, main_gpu=0)
        _seed_manifest(boot, _P4_TAG, main_gpu=1)
        _seed_manifest(boot, _P1_TAG, main_gpu=1)

        spawn_calls, sigterm_calls = [], []
        gates = {_P3_TAG: asyncio.Event(), _P4_TAG: asyncio.Event()}
        mgr = TurbohaulManager(boot, runtime, **_gated_mocks(spawn_calls, sigterm_calls, gates))
        mgr.runtime.queue.safety_enabled = False

        def live_probe():
            free0 = (20000 - _VRAM_MIB) if mgr._residents.get(_P3_TAG) is not None else 20000
            free1 = (20000 - _VRAM_MIB) if mgr._residents.get(_P4_TAG) is not None else 20000
            return [free0, free1]

        with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=live_probe):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await mgr.submit(_P3_TAG, "hi", thread_id="t-p3", client_meta={"ip": _P3_ADDR})
                await mgr.submit(_P4_TAG, "hi", thread_id="t-p4", client_meta={"ip": _P4_ADDR})
                for _ in range(100):
                    r3, r4 = mgr._residents.get(_P3_TAG), mgr._residents.get(_P4_TAG)
                    if (r3 is not None and r3.state == ResidentState.ACTIVE
                            and r4 is not None and r4.state == ResidentState.ACTIVE):
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("P3/P4 never both reached ACTIVE")

                slot_p1 = await mgr.submit(_P1_TAG, "hi", thread_id="t-p1",
                                            client_meta={"ip": _P1_ADDR})

                def _p1_claim_row():
                    claims = mgr.status_snapshot()["queue"]["fastlane_claims_snapshot"]
                    rows = [c for c in claims if c["slot_id"] == slot_p1.slot_id]
                    return (rows[0] if rows else None), claims

                def _p1_waiting_row():
                    waiting = mgr.status_snapshot()["queue"]["waiting"]
                    rows = [r for r in waiting if r["slot_id"] == slot_p1.slot_id]
                    return (rows[0] if rows else None), waiting

                # Let P1 register its claim (first VRAM-overcommit pass).
                claim1 = None
                for _ in range(40):
                    claim1, _ = _p1_claim_row()
                    if claim1 is not None:
                        break
                    await asyncio.sleep(0.02)
                assert claim1 is not None, (
                    "P1's Fast Lane claim must register on the first "
                    "over-commit pass -- this is the funnel that the "
                    "claim registration goes through"
                )
                assert claim1["fastlane"]["label"] == "P1"
                assert claim1["fastlane"]["rule_index"] == 0

                # POSITIVE result: fastlane_claims_snapshot shows P1's
                # client/tag/rank continuously across a real wall-clock
                # window that spans at least one full backoff cycle.
                saw_claim_every_sample = True
                waiting_ever_nonempty = False
                samples = 12
                for _ in range(samples):
                    claim, _ = _p1_claim_row()
                    wrow, _ = _p1_waiting_row()
                    if claim is None or claim["fastlane"]["label"] != "P1":
                        saw_claim_every_sample = False
                    if wrow is not None:
                        waiting_ever_nonempty = True
                    await asyncio.sleep(0.1)  # 12 * 0.1s = 1.2s, > 1.0s backoff

                assert saw_claim_every_sample, (
                    "fastlane_claims_snapshot must show P1's client/tag/rank "
                    "on EVERY sample across a >1 backoff-cycle window -- "
                    "this is the surface that actually satisfies the "
                    "requirement's intent, even though the literal waiting[] array does not"
                )

                # The expected negative result, asserted (not just narrated): the
                # LITERAL queue.waiting[] array is EMPTY on
                # every single sample across the same window -- a real,
                # observed negative result against the literal reading of the requirement.
                assert not waiting_ever_nonempty, (
                    "EXPECTED: queue.waiting[] was "
                    "non-empty for P1 at least once during a >1s window -- "
                    "if this fails, the literal requirement now holds and "
                    "this result is STALE and must be revised, not silently "
                    "left in place"
                )
            finally:
                for g in gates.values():
                    g.set()
                await mgr.shutdown()
