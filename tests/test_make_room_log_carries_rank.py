"""The two make-room log lines carry the tag rank of the client they name.

``MAKE_ROOM_EVICTION`` (a victim was chosen) and ``MAKE_ROOM_STARVED`` (no
victim could be chosen) both print ``rule_index=``. Within one client the tag
rank decides which model goes first, so a line that names the client but not
the rank cannot say why one tag's model was unloaded or kept. Both lines now
end with ``rank=<n>``. On the starved line it is the incoming request's rank,
from the same Fast Lane match that yields ``rule_index=``. On the eviction line
it is the victim's rank as the unload decision read it: the identity of the
resident's latest turn, through ``_resident_rank_meta`` (the one reader the
designation, the outrank gate, the victim pool and the eviction key use), not
the idle stash that ``fastlane_client=`` and ``rule_index=`` still read.
``rank=unresolvable`` means the resident has no identity the lookup can
resolve; in a state where only one of the two identities resolves, ``rank=``
and ``rule_index=`` therefore disagree about it. The field is APPENDED after
every existing field: the existing fields keep their text and their order.

The live cases drive the real routing path (``submit_and_wait`` through the
dispatcher into ``_route_or_reserve``) with a fake engine, and read the line
back from the log. Waits are polls on a predicate with a timeout; no sleep
decides a result. Controls: every tag of one client prints its own rank (a
constant would pass a single case), a tag with no configured rank prints the
unranked value, and a client that matches no rule prints ``unresolvable`` for
both fields. The identity cases call the real eviction emitter on a manager
with a stand-in resident whose stamped identity (latest turn) and idle stash
are set apart.

Free VRAM: the live cases pin both `turbohaul.safety._read_free_vram_all_mib`
and `turbohaul.manager._read_free_vram_all_mib` (the manager imports its own
copy of the reader, so pinning only the first leaves the decision reading the
real GPUs). The identity cases never reach a path that reads free VRAM.

The ranked client sits at 192.0.2.10 (RFC 5737 documentation range).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle


# The identity tests in this file never read free VRAM. The reader is pinned anyway,
# in both the manager and the safety module, so a future change that reaches it fails
# loudly here instead of reading the live GPU. The live tests patch both bindings
# themselves inside the test, which replaces this pin for their duration.
@pytest.fixture(autouse=True)
def _pin_free_vram_autouse(monkeypatch):
    import turbohaul.manager as manager_module
    import turbohaul.safety as safety_module

    def _must_not_read(*args, **kwargs):
        raise AssertionError("a test of this file read free VRAM")

    monkeypatch.setattr(manager_module, "_read_free_vram_all_mib", _must_not_read)
    monkeypatch.setattr(safety_module, "_read_free_vram_all_mib", _must_not_read)


RANKED_IP = "192.0.2.10"
UNLISTED_IP = "198.51.100.77"

# The rule ranks four tags and leaves "unclassified" unset, so a request with
# no class label resolves to the unranked value (6).
RANK_OF = {
    "main": 1,
    "curator": 2,
    "compression": 3,
    "sub_agent": 4,
    "unclassified": 6,
}
META_OF = {
    "main": {"ip": RANKED_IP, "is_main": True},
    "curator": {"ip": RANKED_IP, "is_curator": True},
    "compression": {"ip": RANKED_IP, "is_compression": True},
    "sub_agent": {"ip": RANKED_IP, "is_sub_agent": True},
    "unclassified": {"ip": RANKED_IP},
}
UNLISTED = {"ip": UNLISTED_IP}
TAGS = list(RANK_OF)


def _boot_runtime(tmp_path):
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
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            max_parallel_sidecars=2,
            grace_seconds=0,
            idle_hot_load_seconds=120,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(
            enabled=True,
            rules=[
                FastLaneRule(
                    address=RANKED_IP,
                    label="ranked-client",
                    tag_ranks=FastLaneTagRanks(
                        main=1, curator=2, compression=3, sub_agent=4,
                    ),
                ),
            ],
        ),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag, main_gpu):
    import yaml
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 0,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "llama_server_flags": {"split_mode": "none", "main_gpu": main_gpu},
    }))


def _mocks(gate, gated_models, started):
    pid = [90000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        pid[0] += 1
        proc = MagicMock()
        proc.pid = pid[0]
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        if handle.model_tag in gated_models:
            started[handle.model_tag].set()
            await gate.wait()
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


def _pin_free_vram(stack):
    """Both bindings of the free-VRAM reader, as the module docstring says."""
    stack.enter_context(
        patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000]))
    stack.enter_context(
        patch("turbohaul.manager._read_free_vram_all_mib", return_value=[80000, 80000]))


async def _wait_until(predicate, what, *, timeout=10.0, interval=0.01):
    t0 = time.monotonic()
    while not predicate():
        if time.monotonic() - t0 > timeout:
            raise AssertionError(f"never became true within {timeout}s: {what}")
        await asyncio.sleep(interval)


def _lines(caplog, name):
    return [r.getMessage() for r in caplog.records if name in r.getMessage()]


async def _eviction_scenario(tmp_path, caplog, victim_meta):
    """m1 busy, m2 (served for ``victim_meta``) idle, box at its model cap; a
    request for m3 makes room by unloading idle m2. Returns the
    MAKE_ROOM_EVICTION lines."""
    boot, runtime = _boot_runtime(tmp_path)
    for tag, gpu in (("m1", 0), ("m2", 1), ("m3", 1)):
        _seed_manifest(boot, tag, gpu)
    gate = asyncio.Event()
    started = {"m1": asyncio.Event()}
    mgr = TurbohaulManager(boot, runtime, **_mocks(gate, {"m1"}, started))
    mgr.runtime.queue.safety_enabled = False
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    with ExitStack() as stack:
        _pin_free_vram(stack)
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        f1 = f3 = None
        try:
            f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
            await asyncio.wait_for(started["m1"].wait(), timeout=10)  # m1 is mid-turn
            await asyncio.wait_for(
                mgr.submit_and_wait("m2", "b", thread_id="t2", client_meta=victim_meta),
                timeout=10,
            )
            await _wait_until(
                lambda: (
                    mgr._residents.get("m2") is not None
                    and mgr._residents["m2"].state is ResidentState.IDLE_EVICTABLE
                ),
                "m2 is idle-evictable (the scenario's start state)",
            )
            f3 = asyncio.create_task(mgr.submit_and_wait("m3", "c", thread_id="t3"))
            await _wait_until(
                lambda: bool(_lines(caplog, "MAKE_ROOM_EVICTION")),
                "a MAKE_ROOM_EVICTION line was logged",
            )
            lines = _lines(caplog, "MAKE_ROOM_EVICTION")
        finally:
            gate.set()
            # Teardown only: what m1 and m3 do after the gate opens is not under test.
            pending = [t for t in (f1, f3) if t is not None]
            if pending:
                await asyncio.wait(pending, timeout=5)
            for t in pending:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await mgr.shutdown()
    return lines


async def _starved_scenario(tmp_path, caplog, claimant_meta):
    """m1 and m2 both busy, box at its model cap; a request for m3 finds no
    victim. Returns the MAKE_ROOM_STARVED lines logged by then."""
    boot, runtime = _boot_runtime(tmp_path)
    for tag, gpu in (("m1", 0), ("m2", 1), ("m3", 1)):
        _seed_manifest(boot, tag, gpu)
    gate = asyncio.Event()
    started = {"m1": asyncio.Event(), "m2": asyncio.Event()}
    mgr = TurbohaulManager(boot, runtime, **_mocks(gate, {"m1", "m2"}, started))
    mgr.runtime.queue.safety_enabled = False
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    with ExitStack() as stack:
        _pin_free_vram(stack)
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        f1 = f2 = f3 = None
        try:
            f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
            f2 = asyncio.create_task(mgr.submit_and_wait("m2", "b", thread_id="t2"))
            await asyncio.wait_for(started["m1"].wait(), timeout=10)
            await asyncio.wait_for(started["m2"].wait(), timeout=10)  # both mid-turn
            f3 = asyncio.create_task(
                mgr.submit_and_wait("m3", "c", thread_id="t3", client_meta=claimant_meta)
            )
            await _wait_until(
                lambda: bool(_lines(caplog, "MAKE_ROOM_STARVED")),
                "a MAKE_ROOM_STARVED line was logged",
            )
            assert mgr._residents.get("m3") is None, (
                "scenario: m3 was admitted, so nothing starved"
            )
            lines = _lines(caplog, "MAKE_ROOM_STARVED")
        finally:
            gate.set()
            # Teardown only: what m3 does after the gate opens is not under test.
            pending = [t for t in (f1, f2, f3) if t is not None]
            if pending:
                await asyncio.wait(pending, timeout=5)
            for t in pending:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await mgr.shutdown()
    return lines


# The existing fields, in their existing order, exactly as they were before
# rank= was added (the line minus its trailing " rank=<n>" must still match).
EVICTION_EXISTING = re.compile(
    r"^MAKE_ROOM_EVICTION reason=(?P<reason>\S+) model_tag=(?P<tag>\S+) "
    r"fastlane_client=(?P<client>\S+) rule_index=(?P<rule>\S+)$"
)
STARVED_EXISTING = re.compile(
    r"^MAKE_ROOM_STARVED reason=(?P<reason>\S+) why=(?P<why>\S+) "
    r"model_tag=(?P<tag>\S+) fastlane_client=(?P<client>\S+) "
    r"rule_index=(?P<rule>\S+) residents=\d+ .* "
    r"traffic_class=(?P<traffic>registered|unregistered)$"
)


def _split_rank(line):
    """(line without its trailing rank field, rank value); AssertionError when
    the line does not END with exactly one ``rank=<value>``."""
    assert line.count("rank=") == 1, f"expected exactly one rank= field in: {line!r}"
    head, sep, value = line.rpartition(" rank=")
    assert sep and " " not in value and value, (
        f"rank= is not the last field of: {line!r}"
    )
    return head, value


class TestEvictionLineCarriesRank:
    @pytest.mark.parametrize("tag", TAGS)
    async def test_victim_rank_follows_its_tag_and_existing_fields_survive(
        self, tmp_path, caplog, tag,
    ):
        lines = await _eviction_scenario(tmp_path, caplog, META_OF[tag])
        assert len(lines) == 1, lines
        head, value = _split_rank(lines[0])
        m = EVICTION_EXISTING.match(head)
        assert m is not None, f"existing fields changed or reordered: {head!r}"
        assert m["tag"] == "m2"
        assert m["client"] == RANKED_IP
        assert m["rule"] == "0"
        assert value == str(RANK_OF[tag]), lines[0]

    async def test_unresolvable_client_prints_unresolvable_rank(
        self, tmp_path, caplog,
    ):
        lines = await _eviction_scenario(tmp_path, caplog, UNLISTED)
        assert len(lines) == 1, lines
        head, value = _split_rank(lines[0])
        m = EVICTION_EXISTING.match(head)
        assert m is not None, f"existing fields changed or reordered: {head!r}"
        assert m["client"] == "unresolvable"
        assert m["rule"] == "unresolvable"
        assert value == "unresolvable", lines[0]

    async def test_rank_comes_after_every_existing_field(self, tmp_path, caplog):
        lines = await _eviction_scenario(tmp_path, caplog, META_OF["curator"])
        assert len(lines) == 1, lines
        names = [tok.partition("=")[0] for tok in lines[0].split()[1:]]
        assert names == [
            "reason", "model_tag", "fastlane_client", "rule_index", "rank",
        ], lines[0]


class TestStarvedLineCarriesRank:
    @pytest.mark.parametrize("tag", TAGS)
    async def test_claimant_rank_follows_its_tag_and_existing_fields_survive(
        self, tmp_path, caplog, tag,
    ):
        lines = await _starved_scenario(tmp_path, caplog, META_OF[tag])
        assert lines, "no MAKE_ROOM_STARVED line was logged"
        for line in lines:
            head, value = _split_rank(line)
            m = STARVED_EXISTING.match(head)
            assert m is not None, f"existing fields changed or reordered: {head!r}"
            assert m["tag"] == "m3"
            assert m["client"] == RANKED_IP
            assert m["rule"] == "0"
            assert m["traffic"] == "registered"
            assert value == str(RANK_OF[tag]), line

    async def test_unresolvable_client_prints_unresolvable_rank(
        self, tmp_path, caplog,
    ):
        lines = await _starved_scenario(tmp_path, caplog, UNLISTED)
        assert lines, "no MAKE_ROOM_STARVED line was logged"
        for line in lines:
            head, value = _split_rank(line)
            m = STARVED_EXISTING.match(head)
            assert m is not None, f"existing fields changed or reordered: {head!r}"
            assert m["client"] == "unresolvable"
            assert m["rule"] == "unresolvable"
            assert m["traffic"] == "unregistered"
            assert value == "unresolvable", line

    async def test_rank_comes_after_every_existing_field(self, tmp_path, caplog):
        lines = await _starved_scenario(tmp_path, caplog, META_OF["curator"])
        assert lines, "no MAKE_ROOM_STARVED line was logged"
        for line in lines:
            names = [tok.partition("=")[0] for tok in line.split()[1:]]
            # The line's last field is rank, traffic_class (the previous last
            # field) is directly before it, and the contract fields lead.
            assert names[-1] == "rank", line
            assert names[-2] == "traffic_class", line
            assert names[:5] == [
                "reason", "why", "model_tag", "fastlane_client", "rule_index",
            ], line


def _emit_eviction_line(tmp_path, caplog, *, state, stamp, stash):
    """Call the real eviction emitter for a stand-in victim: ``stamp`` is its
    ``rank_client_meta`` (latest turn), ``stash`` its ``idle_client_meta``."""
    boot, runtime = _boot_runtime(tmp_path)
    mgr = TurbohaulManager(boot, runtime)
    victim = SimpleNamespace(
        model_tag="m2", state=state, rank_client_meta=stamp, idle_client_meta=stash,
    )
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr._log_make_room_unload(victim, "make_room_count_cap")
    lines = _lines(caplog, "MAKE_ROOM_EVICTION")
    assert len(lines) == 1, lines
    return lines[0]


# (state, stamp tag or None/"unlisted", stash tag or None/"unlisted") ->
# (fastlane_client=, rule_index=, rank=)
UNRES = "unresolvable"
IDENTITY_CASES = {
    # the stamp is the later turn and wins in every state
    "idle_stamp_wins_over_the_stash": (
        ResidentState.IDLE_EVICTABLE, "curator", "sub_agent", (RANKED_IP, "0", "2")),
    # an idle resident with nothing stamped reads the stash
    "idle_nothing_stamped_reads_the_stash": (
        ResidentState.IDLE_EVICTABLE, None, "sub_agent", (RANKED_IP, "0", "4")),
    # a non-idle resident with nothing stamped has no identity for the decision
    "active_nothing_stamped_has_no_rank_identity": (
        ResidentState.ACTIVE, None, "sub_agent", (RANKED_IP, "0", UNRES)),
    # stamp resolves, stash missing: rank= resolves where rule_index= cannot
    "grace_stamp_resolves_stash_missing": (
        ResidentState.GRACE, "curator", None, (UNRES, UNRES, "2")),
    # stamp resolves, stash is an unlisted client: same disagreement
    "grace_stamp_resolves_stash_unlisted": (
        ResidentState.GRACE, "main", "unlisted", (UNRES, UNRES, "1")),
    # stamp is an unlisted client, stash resolves: the other disagreement
    "grace_stamp_unlisted_stash_resolves": (
        ResidentState.GRACE, "unlisted", "curator", (RANKED_IP, "0", UNRES)),
    "idle_nothing_anywhere": (
        ResidentState.IDLE_EVICTABLE, None, None, (UNRES, UNRES, UNRES)),
}


def _meta(tag):
    if tag is None:
        return None
    return UNLISTED if tag == "unlisted" else META_OF[tag]


class TestEvictionRankFollowsTheDecisionsIdentity:
    def test_a_grace_victim_prints_the_stamps_rank_not_the_idle_stashs(
        self, tmp_path, caplog,
    ):
        # The stamp (latest turn) is a curator, rank 2; the stash holds the
        # earlier sub_agent turn, rank 4. Both are the same client, so the
        # existing fields read the same on either identity; only rank= tells
        # which identity was read.
        line = _emit_eviction_line(
            tmp_path, caplog, state=ResidentState.GRACE,
            stamp=META_OF["curator"], stash=META_OF["sub_agent"],
        )
        head, value = _split_rank(line)
        m = EVICTION_EXISTING.match(head)
        assert m is not None, f"existing fields changed or reordered: {head!r}"
        assert (m["client"], m["rule"]) == (RANKED_IP, "0"), line
        assert value == "2", f"rank= is not the stamped turn's rank: {line!r}"

    @pytest.mark.parametrize("case", list(IDENTITY_CASES))
    def test_rank_is_read_through_the_decisions_identity_reader(
        self, tmp_path, caplog, case,
    ):
        state, stamp, stash, (client, rule, rank) = IDENTITY_CASES[case]
        line = _emit_eviction_line(
            tmp_path, caplog, state=state, stamp=_meta(stamp), stash=_meta(stash),
        )
        head, value = _split_rank(line)
        m = EVICTION_EXISTING.match(head)
        assert m is not None, f"existing fields changed or reordered: {head!r}"
        # Both values are stated: fastlane_client= and rule_index= come from
        # the stash, rank= from the stamp-first reader.
        assert (m["client"], m["rule"], value) == (client, rule, rank), line
