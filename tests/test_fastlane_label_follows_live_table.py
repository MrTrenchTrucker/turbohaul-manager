"""Regression test: a resident card's Fast Lane label must follow the LIVE rule table.

The dashboard shows each resident's last-served identity. The `label` on that
record is stamped at admission from whichever rule matched, and nothing marks it
stale when the operator changes the rule table underneath it. Two resident cards can then
show "Gateway" after the table has already been
corrected -- the stamps predate the correction by a few seconds, so the label was
true when it was written and wrong when it is read.

That is a silent-drift defect, one layer up from the
matcher: the operator cannot trust what that screen says about who is being served.

⛔ WHY ARM 1 IS NOT DECORATION. Arms 2 and 3 both assert that a label is ABSENT or
CHANGED. A probe that reached the wrong dict, or a fixture where fastlane never
matched at all, would satisfy both of them for free while measuring nothing. Arm 1
requires the label to be PRESENT and correct on an unchanged table, so the probe
has to be able to see a label before its absence is allowed to mean anything.

ARMS
  1. rule present, table untouched -> card shows that rule's label   [GREEN CONTROL, passes today]
  2. the rule is RELABELLED        -> card must show the NEW label    [RED today: shows the old one]
  3. the rule is REMOVED entirely  -> card must show None, not a name [RED today: still shows it]

Run:
    cd <repo> && python3 -m pytest tests/test_fastlane_label_follows_live_table.py -v
(pyproject sets pythonpath=["src"], which is ROOTDIR-relative and beats PYTHONPATH --
so this must be run from the repo root, not with PYTHONPATH pointed at some other tree.)
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

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
from turbohaul.manager import TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle

CLIENT_IP = "192.0.2.55"      # TEST-NET-1 (RFC 5737), documented for tests
ORIGINAL_LABEL = "Gateway"     # a label that sticks after the table changes
NEW_LABEL = "Advisor"


# --- fixtures: local copies, per this suite's established per-file convention ---

def _boot_runtime(tmp_path, *, rules):
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
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            max_parallel_sidecars=2,  # cap<=1 uses the singleton path where _residents_snapshot() is [] by design (covered by the cap-one tests). Typical setups run max 2.
            grace_seconds=0,
            idle_hot_load_seconds=120,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=rules),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag):
    import yaml
    (boot.storage.manifests_path / f"{model_tag}.yaml").write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 0,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
    }))


def _fake_handle(model_tag, port, pid):
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _mocks():
    """Constructor-injected fakes. The manager takes these as kwargs
    (spawn_fn/health_fn/...), NOT as attributes assigned after construction --
    assigning attributes leaves the real spawn path live, which fails on
    /var/lib/turbohaul with a PermissionError long before any assertion runs.
    A fixture that crashes has tested nothing; this comment is here so the
    next person does not rediscover it."""
    pid = [91000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


def _mk(boot, runtime, **mocks):
    mgr = TurbohaulManager(boot, runtime, **mocks)
    mgr.runtime.queue.safety_enabled = False
    return mgr


def _label_for(mgr, model_tag):
    """The label the DASHBOARD would render on that resident's card."""
    for entry in mgr._residents_snapshot():
        if entry["model_tag"] == model_tag:
            ident = entry["request_identity"]
            return None if ident is None else ident.get("label")
    raise AssertionError(f"resident {model_tag!r} not in the snapshot at all")


def _rule(label):
    return FastLaneRule(address=CLIENT_IP, label=label)


def _retable(mgr, rules):
    """Change the rule table exactly the way PUT /api/config does it:
    replace runtime wholesale, then bust the compiled-table cache
    (api/config_put.py)."""
    mgr.runtime = mgr.runtime.model_copy(
        update={"fastlane": FastLaneConfig(enabled=True, rules=rules)}
    )
    mgr.invalidate_fastlane()


async def _serve_one(mgr, model_tag="m1"):
    await asyncio.wait_for(
        mgr.submit_and_wait(
            model_tag, "hello", thread_id="t1",
            client_meta={"session_id": "sess-1", "ip": CLIENT_IP, "is_main": True},
        ),
        timeout=5,
    )


@pytest.mark.usefixtures("tmp_path")
class TestLabelFollowsLiveTable:

    async def _stamped(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, rules=[_rule(ORIGINAL_LABEL)])
        _seed_manifest(boot, "m1")
        mgr = _mk(boot, runtime, **_mocks())
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000]):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            await _serve_one(mgr)
        return mgr

    async def test_arm1_green_control_unchanged_table_shows_the_rules_label(self, tmp_path):
        """GREEN CONTROL. If this fails, the fixture never produced a Fast Lane
        match and arms 2 and 3 prove nothing -- they would pass on an empty dict."""
        mgr = await self._stamped(tmp_path)
        try:
            assert _label_for(mgr, "m1") == ORIGINAL_LABEL, (
                "the probe cannot see a label even on an UNCHANGED table -- arms 2 "
                "and 3 are unfalsifiable until this passes"
            )
        finally:
            await mgr.shutdown()

    async def test_arm2_relabelling_the_rule_updates_the_card(self, tmp_path):
        """RED on today's code: the card keeps rendering the label that was
        stamped at admission, so the operator reads a name that is no longer
        what that rule says."""
        mgr = await self._stamped(tmp_path)
        try:
            assert _label_for(mgr, "m1") == ORIGINAL_LABEL  # precondition
            _retable(mgr, [_rule(NEW_LABEL)])
            assert _label_for(mgr, "m1") == NEW_LABEL, (
                f"stale label: the card still shows {_label_for(mgr, 'm1')!r} after the "
                f"rule was relabelled to {NEW_LABEL!r}"
            )
        finally:
            await mgr.shutdown()

    async def test_arm3_removing_the_rule_blanks_the_card(self, tmp_path):
        """RED on today's code: the rule is gone, so the manager has no basis
        for claiming that client's identity -- but the card still names it.
        This is the shape an operator would see."""
        mgr = await self._stamped(tmp_path)
        try:
            assert _label_for(mgr, "m1") == ORIGINAL_LABEL  # precondition
            _retable(mgr, [])
            assert _label_for(mgr, "m1") is None, (
                f"stale label: the rule was removed but the card still shows "
                f"{_label_for(mgr, 'm1')!r}"
            )
        finally:
            await mgr.shutdown()

    async def test_arm4_removing_a_rule_ABOVE_the_match_does_not_shift_the_label(
        self, tmp_path
    ):
        """THE ARM THAT DISCRIMINATES THIS DESIGN FROM THE ALTERNATIVE DESIGN.

        Rules are matched in list order and `rule_index` is POSITIONAL, so an
        implementation that stored the index would, after rule[0] is deleted,
        resolve index 1 to what USED to be rule[2] and render a neighbouring
        client's label with total confidence. Keying on the rule's own
        identifying string is immune: the matched rule is still in the table,
        just at a different position.

        MEASURED, not asserted: _live_fastlane_label was mutated into the
        rule_index implementation and the suite re-run. Arms 1, 2, 3 and 5
        PASSED; this arm FAILED with

            assert 'NEIGHBOUR BELOW' != 'NEIGHBOUR BELOW'

        -- the neighbour that shifted into the matched rule's old index
        rendering on that client's card. (The mutant parsed; an unparseable
        one kills everything and fails open.)

        ⇒ THIS ARM EXISTS TO KILL THE POSITIONAL VARIANT. Without it both
        designs look equally green and the decision between them is untested.
        Do not delete it as redundant with arms 2 and 3: it is the only test
        in this suite that distinguishes the design that shipped from the
        one that was rejected.
        """
        boot, runtime = _boot_runtime(tmp_path, rules=[
            FastLaneRule(address="192.0.2.10", label="NEIGHBOUR ABOVE"),
            _rule(ORIGINAL_LABEL),                       # <- index 1, the one that matches
            FastLaneRule(address="192.0.2.30", label="NEIGHBOUR BELOW"),
        ])
        _seed_manifest(boot, "m1")
        mgr = _mk(boot, runtime, **_mocks())
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000]):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            await _serve_one(mgr)
            try:
                assert _label_for(mgr, "m1") == ORIGINAL_LABEL  # precondition
                # delete the rule ABOVE the match: everything below shifts up one
                _retable(mgr, [_rule(ORIGINAL_LABEL),
                               FastLaneRule(address="192.0.2.30", label="NEIGHBOUR BELOW")])
                got = _label_for(mgr, "m1")
                assert got != "NEIGHBOUR BELOW", (
                    "POSITIONAL RESOLUTION: the label shifted to the neighbour that "
                    "moved into the matched rule's old index -- confidently wrong, "
                    "which is worse than the stale value it replaced"
                )
                assert got == ORIGINAL_LABEL, (
                    f"the matched rule is still in the table and still carries "
                    f"{ORIGINAL_LABEL!r}; the card shows {got!r}"
                )
            finally:
                await mgr.shutdown()

    async def test_arm5_retabling_disturbs_the_label_and_nothing_else(self, tmp_path):
        """Additive only. Re-resolving the label must not
        touch ip / session_id / req_id / model_tag / resolved_class, which are
        not config-derived and were true when stamped.

        The `!=` on `label` is this arm's own control: without it the whole
        assertion would hold for a no-op that changed nothing at all."""
        mgr = await self._stamped(tmp_path)
        try:
            before = next(e for e in mgr._residents_snapshot()
                          if e["model_tag"] == "m1")["request_identity"]
            _retable(mgr, [_rule(NEW_LABEL)])
            after = next(e for e in mgr._residents_snapshot()
                         if e["model_tag"] == "m1")["request_identity"]
            assert after["label"] != before["label"], (
                "control: the label must actually have moved, or this arm would "
                "pass against a change that did nothing"
            )
            for field in ("ip", "session_id", "req_id", "model_tag",
                          "resolved_class", "thread_id", "is_main"):
                assert after[field] == before[field], (
                    f"{field} changed across a rule-table edit: "
                    f"{before[field]!r} -> {after[field]!r}"
                )
            assert set(after) == set(before), "no key added or dropped at the read surface"
        finally:
            await mgr.shutdown()
