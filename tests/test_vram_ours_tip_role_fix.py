"""The vram-anchor thread_hash collision fix.

The problem: a curator review client (thread=bg-review,
persist_disabled, reusing main's session_id) computed the SAME thread_hash
as main. Root cause: _bin_identity(thread_id, client_meta,
chain) falls back to the RAW thread_id whenever session_id/role-label/chain
aren't ALL simultaneously present (manager.py, _bin_identity) -- and the
"curator reuse-main" design deliberately sends the curator's thread_id =
main's thread_id, with an empty admission_hash_chain at the point
_maybe_force_clean_restore runs. _vram_ours (inside _maybe_force_clean_restore)
compared model_tag+thread_hash+pid only -- all three trivially match for a
same-session curator riding main's warm slot by design, so the classifier
incorrectly trusted foreign VRAM content as main's own "native reuse".

An AND-with-_kv_dirty_tail direction was tried and found insufficient
(by analysis, then re-confirmed by
direct code trace): _kv_dirty_tail is only SET by THIS SAME turn's own
_probe_and_save_clean_kv call, which in the grace-loop matched path always
runs AFTER _maybe_force_clean_restore's read of it (streaming and
non-streaming alike). On the curator's FIRST touch the flag is still
whatever the PREVIOUS (main) occupant left it (cleared) -- the collision
sails through unblocked.

Fix: a "tip_role" field, stamped on the VRAM anchor
itself at the same synchronous instant thread_hash/chain already are (three
write sites in _probe_and_save_clean_kv), compared against the CURRENT
call's own ZERO-LAG role (admission_role, stamped at submit() time, else
_bin_role(client_meta) -- no dependency on any write that hasn't happened
yet). Closes the FIRST-touch case because the anchor's last-written
tip_role reflects main's own completed prior write, not curator's current
one.

Also folds in a wording fix: the disposable-role gate's "clean-prefix save
GATED... outgoing main parked" log line hardcoded "main" regardless of what
was ACTUALLY being displaced (e.g. a curator displacing another curator) --
now names the real outgoing role via the same tip_role field.

Covers:
  - TestVramOursTipRoleFix: the discriminator (must fail on unmodified base)
    + legitimate-reuse preservation (main-only and same-role continuations)
  - TestAnchorWriteCompleteness: the anchor-write completeness question --
    is the anchor written on EVERY main turn, or can tip_role go
    stale-by-absence on some path? (confirmed: yes, gated by the same
    40k-char min-ctx bar as everything else -- a PRE-EXISTING characteristic
    of the whole anchor/dirty-tail mechanism, not introduced by this fix.
    UPDATE: a later check found that an intervening SHORT
    foreign-role turn is NOT side-effect-free once the same-model gate
    exists -- see test_intervening_short_curator_turn_now_correctly_marks_
    anchor_foreign, which supersedes this class's original "no new
    regression" premise with the corrected, verified-safe behavior)
  - TestGatedWordingFix: the log-line correction

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this file> -v
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

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
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot
from turbohaul.subprocess_mgr import SidecarHandle

_QWEN = "qwen3.6-27b"
_PORT = 59500


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
            default_port_base=_PORT,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


def _fake_handle(model_tag=_QWEN, port=_PORT, pid=77_777):
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _msgs(k, marker="content"):
    filler = (marker + "-") * 100
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-{filler}"} for i in range(k)]


async def _stamp_anchor_directly(mgr, handle, slot):
    """Drive the real _probe_and_save_clean_kv per-turn-off park path, which
    is what actually stamps _kv_vram_anchor[port] (including tip_role) --
    mirrors what a real turn does, no test-only shortcuts."""
    await mgr._probe_and_save_clean_kv(handle, slot)


# ================================================================================
# 1. The discriminator + legitimate-reuse preservation
# ================================================================================
@pytest.mark.asyncio
class TestVramOursTipRoleFix:
    async def test_discriminator_curator_sharing_main_thread_id_first_touch_denied(
        self, mgr,
    ):
        """MUST FAIL on unmodified base: a curator sharing main's raw
        thread_id, with NO session_id of its own on THIS request (the
        precondition that makes _bin_identity fall back to the raw
        thread_id via its FIRST check, `if not session_id: return
        thread_id` -- independent of chain content, confirmed empirically:
        an empty admission_hash_chain instead trips
        _maybe_force_clean_restore's OWN earlier `if not inc_chain: return`
        guard before _vram_ours is ever evaluated, so that construction
        cannot reproduce this bug at all -- caught while writing this test,
        not assumed) must NOT be trusted as "native reuse" of main's own
        VRAM state on its first touch."""
        handle = _fake_handle()
        shared_thread_id = "bg-review-shares-main-tid"

        main_slot = Slot.new(
            _QWEN, thread_id=shared_thread_id, admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="MAINCONTENT")},
        )
        await _stamp_anchor_directly(mgr, handle, main_slot)
        anchor = mgr._kv_vram_anchor.get(_PORT)
        assert anchor is not None and anchor.get("tip_role") in (None, "main"), (
            "sanity: main's own stamp did not land as expected"
        )

        curator_slot = Slot.new(
            _QWEN, thread_id=shared_thread_id,  # SAME raw thread_id -- the collision shape
            client_meta={"is_curator": True, "persist_disabled": True},  # NO session_id
            admission_hash_chain=["curator-own-turn-hash-1", "curator-own-turn-hash-2"],
        )
        curator_slot.pid = handle.pid  # curator rides main's own live engine, by design

        decision = await mgr._maybe_force_clean_restore(
            handle.port, _QWEN, curator_slot, warm_chain=[],
        )

        assert decision["resolved_from"] != "warm-vram-anchor-native-reuse", (
            "the classifier trusted the anchor's chain as the curator's own "
            "native reuse -- the thread_hash collision was not closed"
        )

    async def test_legitimate_main_only_continuation_still_trusted(self, mgr):
        """No regression: main's own grace follow-up must still get native
        VRAM-anchor reuse when nothing foreign touched the port."""
        handle = _fake_handle()

        main_slot = Slot.new(
            _QWEN, thread_id="main-solo-1", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="SOLOCONTENT")},
        )
        await _stamp_anchor_directly(mgr, handle, main_slot)

        followup_chain = mgr._kv_vram_anchor[_PORT]["chain"]
        followup = Slot.new(
            _QWEN, thread_id="main-solo-1", admission_ctx_len=45100,
            client_meta={"messages": _msgs(31, marker="SOLOCONTENT")},
            admission_hash_chain=followup_chain + ["extra-turn-hash"],
        )
        followup.pid = handle.pid

        decision = await mgr._maybe_force_clean_restore(
            handle.port, _QWEN, followup, warm_chain=[],
        )

        assert decision["resolved_from"] == "warm-vram-anchor-native-reuse", (
            "a legitimate main-only continuation was denied native VRAM "
            "reuse -- this is exactly the regression the fix must not cause"
        )

    async def test_main_to_main_inconsistent_is_main_labeling_does_not_regress(
        self, mgr,
    ):
        """LOAD-BEARING (the "real regression" signature: a
        main turn declining an anchor that legitimately WAS its own).
        _bin_role treats an EXPLICIT is_main=True label and a fully
        UNLABELED request as the SAME main-class bucket everywhere else in
        this file (`role not in (None, "main")`).
        If the harness is inconsistent turn-to-turn about sending that
        label, a naive strict-equality tip_role comparison would see
        "main" != None and wrongly deny a genuinely-main-to-main
        continuation. Tests BOTH directions explicitly."""
        handle = _fake_handle()

        # Direction 1: explicit is_main=True, then a follow-up with NO label at all.
        explicit_main = Slot.new(
            _QWEN, thread_id="main-labeling-1", admission_ctx_len=45000,
            client_meta={"is_main": True, "messages": _msgs(30, marker="EXPLICITMAINCONTENT")},
        )
        await _stamp_anchor_directly(mgr, handle, explicit_main)
        assert mgr._kv_vram_anchor[_PORT]["tip_role"] == "main"

        unlabeled_followup = Slot.new(
            _QWEN, thread_id="main-labeling-1", admission_ctx_len=45100,
            client_meta={"messages": _msgs(31, marker="EXPLICITMAINCONTENT")},  # no is_main
            admission_hash_chain=mgr._kv_vram_anchor[_PORT]["chain"] + ["extra-turn-hash"],
        )
        unlabeled_followup.pid = handle.pid

        decision_1 = await mgr._maybe_force_clean_restore(
            handle.port, _QWEN, unlabeled_followup, warm_chain=[],
        )
        assert decision_1["resolved_from"] == "warm-vram-anchor-native-reuse", (
            "explicit-is_main -> unlabeled-follow-up (same main identity) "
            "was wrongly denied -- the None/'main' split was not normalized"
        )

        # Direction 2: unlabeled, then a follow-up with an explicit is_main=True.
        handle2 = _fake_handle(port=59501, pid=88_888)
        unlabeled_main = Slot.new(
            _QWEN, thread_id="main-labeling-2", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="UNLABELEDMAINCONTENT")},
        )
        await _stamp_anchor_directly(mgr, handle2, unlabeled_main)
        assert mgr._kv_vram_anchor[59501]["tip_role"] == "main"

        explicit_followup = Slot.new(
            _QWEN, thread_id="main-labeling-2", admission_ctx_len=45100,
            client_meta={"is_main": True, "messages": _msgs(31, marker="UNLABELEDMAINCONTENT")},
            admission_hash_chain=mgr._kv_vram_anchor[59501]["chain"] + ["extra-turn-hash"],
        )
        explicit_followup.pid = handle2.pid

        decision_2 = await mgr._maybe_force_clean_restore(
            handle2.port, _QWEN, explicit_followup, warm_chain=[],
        )
        assert decision_2["resolved_from"] == "warm-vram-anchor-native-reuse", (
            "unlabeled -> explicit-is_main follow-up (same main identity) "
            "was wrongly denied -- the None/'main' split was not normalized"
        )

    async def test_legitimate_same_role_continuation_still_trusted(self, mgr):
        """No regression: a curator's own second turn (same identity, same
        role) must still be trusted -- role==role even though the identity
        is disposable."""
        handle = _fake_handle()

        curator_1 = Slot.new(
            _QWEN, thread_id="curator-solo-1", admission_ctx_len=42000,
            client_meta={"is_curator": True, "session_id": "s-cur-solo",
                         "messages": _msgs(30, marker="CURSOLOCONTENT")},
        )
        await _stamp_anchor_directly(mgr, handle, curator_1)
        anchor = mgr._kv_vram_anchor.get(_PORT)
        assert anchor.get("tip_role") == "curator"

        followup_chain = anchor["chain"]
        curator_2 = Slot.new(
            _QWEN, thread_id="curator-solo-1", admission_ctx_len=42100,
            client_meta={"is_curator": True, "session_id": "s-cur-solo"},
            admission_hash_chain=followup_chain + ["extra-turn-hash"],
        )
        curator_2.pid = handle.pid

        decision = await mgr._maybe_force_clean_restore(
            handle.port, _QWEN, curator_2, warm_chain=[],
        )

        assert decision["resolved_from"] == "warm-vram-anchor-native-reuse", (
            "a legitimate same-role (curator-after-curator) continuation "
            "was denied native VRAM reuse"
        )

    async def test_curator_after_main_correctly_denied_even_with_matching_hash(
        self, mgr,
    ):
        """Symmetric direction check: main resuming after a curator (or vice
        versa) with a genuinely foreign anchor must stay denied -- confirms
        the fix denies in BOTH directions of a role mismatch, not just the
        one direction the discriminator above exercises."""
        handle = _fake_handle()
        shared_thread_id = "bg-review-shares-main-tid-2"

        curator_slot = Slot.new(
            _QWEN, thread_id=shared_thread_id, admission_ctx_len=42000,
            client_meta={"is_curator": True, "session_id": "s-main-2",
                         "messages": _msgs(30, marker="CURATORFIRSTCONTENT")},
        )
        await _stamp_anchor_directly(mgr, handle, curator_slot)
        anchor = mgr._kv_vram_anchor.get(_PORT)
        assert anchor.get("tip_role") == "curator"

        main_slot = Slot.new(
            _QWEN, thread_id=shared_thread_id, admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="MAINAFTERCURATOR")},
            # non-empty -- an empty admission_hash_chain trips
            # _maybe_force_clean_restore's OWN earlier `if not inc_chain:
            # return` guard, which would make this test pass vacuously
            # without ever reaching _vram_ours at all (caught while
            # writing these tests, not assumed -- see the discriminator
            # test's docstring for the same trap).
            admission_hash_chain=["main-after-curator-turn-hash"],
        )
        main_slot.pid = handle.pid

        decision = await mgr._maybe_force_clean_restore(
            handle.port, _QWEN, main_slot, warm_chain=[],
        )

        assert decision["resolved_from"] != "warm-vram-anchor-native-reuse", (
            "main incorrectly trusted a curator's anchor as its own native "
            "reuse -- the fix must deny in this direction too"
        )


# ================================================================================
# 2. Anchor-write completeness (can tip_role go stale by absence)
# ================================================================================
@pytest.mark.asyncio
class TestAnchorWriteCompleteness:
    async def test_short_main_turn_does_not_update_anchor(self, mgr):
        """CONFIRMS the gap in question: a short-context main
        turn (< the 40000-char min-ctx bar) does NOT reach the anchor-stamp
        code at all -- _kv_vram_anchor[port] is left completely untouched,
        so tip_role (like every other anchor field) is stale-by-absence on
        this path. This is a PRE-EXISTING characteristic of the whole
        _probe_and_save_clean_kv chokepoint (the min-ctx bar gates ALL
        anchor/dirty-tail writes uniformly), not introduced by this fix --
        documented here, not silently fixed (out of this item's scope)."""
        handle = _fake_handle()
        short_main = Slot.new(
            _QWEN, thread_id="short-main-1", admission_ctx_len=500,
            client_meta={"messages": [{"role": "user", "content": "hi"}]},
        )
        await mgr._probe_and_save_clean_kv(handle, short_main)
        assert mgr._kv_vram_anchor.get(_PORT) is None, (
            "a short main turn unexpectedly stamped the anchor -- if this "
            "assertion ever fails, the min-ctx-bar gating changed and the "
            "staleness analysis above needs re-deriving"
        )

    async def test_intervening_short_curator_turn_now_correctly_marks_anchor_foreign(
        self, mgr,
    ):
        """Short intervening curator turn: an executable repro against THIS
        exact scenario shows that asserting the anchor stays unchanged would
        encode the bug this test guards against.
        The min-ctx-bar gate alone does NOT
        make a short intervening curator turn side-effect-free once
        the same-model gate exists (a no-unload displacement):
        That gate is hoisted ABOVE the min-ctx-bar specifically so a short
        visitor can still trigger a save of main's OWN outgoing KV -- and
        that nested save stamps this port's anchor with tip_role="main",
        correct at that instant. An assertion that the anchor was
        COMPLETELY UNCHANGED after the short
        curator's own turn would hold only without that gate, and hides
        the real danger: the anchor is left claiming tip_role="main" even
        though the curator, not main, is the one actually about to occupy
        the engine's VRAM. The fix corrects tip_role to the
        SHORT VISITOR's own role right at the point its own turn would
        otherwise return before ever reaching its own tip_role stamp --
        this test asserts THAT correction lands, and that main's own
        later continuation correctly stops trusting the anchor as native
        VRAM reuse once it no longer matches."""
        handle = _fake_handle()

        main_1 = Slot.new(
            _QWEN, thread_id="stale-check-1", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="STALECHECKCONTENT")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_1)
        anchor_after_main = dict(mgr._kv_vram_anchor[_PORT])
        assert anchor_after_main["tip_role"] == "main"

        short_curator = Slot.new(
            _QWEN, thread_id="stale-check-1", admission_ctx_len=200,
            client_meta={"is_curator": True, "session_id": "s-stale",
                         "messages": [{"role": "user", "content": "hi"}]},
        )
        await mgr._probe_and_save_clean_kv(handle, short_curator)
        anchor_after_curator = mgr._kv_vram_anchor[_PORT]
        assert anchor_after_curator["tip_role"] == "curator", (
            "MUST FAIL before the fix: the anchor must now "
            "reflect the short curator's own occupancy, not a stale "
            "'main' claim left behind by the gate's nested save of main's "
            "own outgoing KV"
        )
        # Every OTHER field this fix does not touch stays exactly as
        # main's own stamp left it (this harness has no real SLOT_SAVE_DIR
        # wired, so the gate's own nested save itself does not persist here —
        # only the tip_role correction does).
        assert {k: v for k, v in anchor_after_curator.items() if k != "tip_role"} == {
            k: v for k, v in anchor_after_main.items() if k != "tip_role"
        }

        main_2 = Slot.new(
            _QWEN, thread_id="stale-check-1", admission_ctx_len=45100,
            client_meta={"messages": _msgs(31, marker="STALECHECKCONTENT")},
            admission_hash_chain=anchor_after_main["chain"] + ["extra-turn-hash"],
        )
        main_2.pid = handle.pid

        decision = await mgr._maybe_force_clean_restore(
            handle.port, _QWEN, main_2, warm_chain=[],
        )

        assert decision["resolved_from"] != "warm-vram-anchor-native-reuse", (
            "main's own continuation must NOT trust native VRAM reuse here "
            "-- the engine's real VRAM tip, after the short curator's own "
            "turn ran, actually holds the curator's tokens, not main's"
        )
        assert decision["action"] == "fresh"


# ================================================================================
# 3. The GATED log-wording fix
# ================================================================================
@pytest.mark.asyncio
class TestGatedWordingFix:
    async def test_gated_log_names_actual_outgoing_role_not_hardcoded_main(
        self, mgr, caplog,
    ):
        """The disposable-role gate's own log line must name the ACTUAL
        outgoing role, not unconditionally say "main" -- e.g. curator #2
        displacing curator #1 must say "outgoing curator parked", not the
        factually-wrong "outgoing main parked"."""
        import logging
        handle = _fake_handle()

        curator_1 = Slot.new(
            _QWEN, thread_id="wording-cur-1", admission_ctx_len=42000,
            client_meta={"is_curator": True, "session_id": "s-wording-1",
                         "messages": _msgs(30, marker="WORDINGCONTENT1")},
        )
        await mgr._probe_and_save_clean_kv(handle, curator_1)

        curator_2 = Slot.new(
            _QWEN, thread_id="wording-cur-2", admission_ctx_len=42000,
            client_meta={"is_sub_agent": True, "session_id": "s-wording-2",
                         "messages": _msgs(30, marker="WORDINGCONTENT2")},
        )
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, curator_2)

        gated_lines = [r.message for r in caplog.records if "clean-prefix save GATED" in r.message]
        assert gated_lines, "sanity: the GATED log line never fired"
        assert "outgoing main parked" not in gated_lines[-1], (
            "the log line still hardcodes 'main' even though a curator, "
            "not main, was actually displaced"
        )
        assert "outgoing curator parked" in gated_lines[-1]

    async def test_gated_log_still_says_main_when_main_actually_displaced(
        self, mgr, caplog,
    ):
        """Precision guard: when main genuinely was the outgoing identity,
        the wording fix must not have flipped this to something else."""
        import logging
        handle = _fake_handle()

        main_slot = Slot.new(
            _QWEN, thread_id="wording-main-1", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="WORDINGMAINCONTENT")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_slot)

        curator = Slot.new(
            _QWEN, thread_id="wording-cur-3", admission_ctx_len=42000,
            client_meta={"is_curator": True, "session_id": "s-wording-3",
                         "messages": _msgs(30, marker="WORDINGCONTENT3")},
        )
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, curator)

        gated_lines = [r.message for r in caplog.records if "clean-prefix save GATED" in r.message]
        assert gated_lines
        assert "outgoing main parked" in gated_lines[-1]
