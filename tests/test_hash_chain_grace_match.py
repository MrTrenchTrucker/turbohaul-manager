"""Grace-window match on the content-derived identity (admission_hash_chain)
instead of the client ADDRESS, keyed on
kv_policy._is_prefix_match (not re-implemented -- second consumer of the same
chokepoint the KV-restore gate already uses).

GROUP A drives the real TurbohaulQueue + real Slot + the REAL
kv_policy._prefix_hash_chain (never a hand-typed fake hash) against
drain_inbox_and_staging_match_hash_chain directly -- the same level of rigor
the drain-examined-invariant test uses for the IP sibling this
method replaces.

⭐ Core requirement: a test that only asserts the chain is populated goes GREEN
while every request still cold-loads, so assert that the returning turn's chain
PREFIX-MATCHES the resident's saved chain and that the slot is consequently
matched. TestPrefixMatchIsTheDiscriminator is that assertion: two candidates
both carry a non-empty admission_hash_chain; only the one whose chain is a
genuine extension of the anchor's is served.

GROUP B drives the real manager end-to-end (real TurbohaulQueue,
TurbohaulManager, FastAPI app, real grace loop in _serve_on_resident) with a
GROWING multi-turn message history -- the shape a real client actually sends,
and the shape that makes admission_hash_chain a genuine rolling prefix chain
rather than a series of unrelated single-message hashes. Reuses the fake
sidecar / spawn harness from test_interturn_race_gate.py as-is.
"""
from __future__ import annotations

import asyncio
import ipaddress
import time
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

import turbohaul.telemetry as telemetry_module
from turbohaul.api.main import create_app
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
from turbohaul.kv_policy import _prefix_hash_chain
from turbohaul.queue import TurbohaulQueue
from turbohaul.slot import Slot
from turbohaul.subprocess_mgr import SidecarHandle

_MODEL = "test-model"


def _slot_with_chain(messages, *, model_tag: str = _MODEL, created_at: float) -> Slot:
    """Real Slot, real chain -- computed by the actual production function,
    never a hand-typed placeholder string."""
    s = Slot.new(model_tag, context=list(messages),
                 admission_hash_chain=_prefix_hash_chain(messages))
    s.created_at = created_at
    return s


async def _drain_chain(q: TurbohaulQueue, anchor_chain, inbox, grace_started_at: float):
    # No decline_sink: the new drain does not take
    # one -- production never read the IP sibling's sink back (AST-verified,
    # write-only), so this method does not inherit that dead plumbing. The
    # return value is the oracle everywhere below.
    return await q.drain_inbox_and_staging_match_hash_chain(
        anchor_chain, _MODEL, grace_started_at, inbox=inbox,
    )


# ===========================================================================
# GROUP A -- direct, real-queue-level evidence.
# ===========================================================================
@pytest.mark.asyncio
class TestDrainMatchHashChainDirect:
    async def test_empty_anchor_chain_never_matches_fail_safe(self):
        """kv_policy._is_prefix_match([], x) is False by construction -- an
        admission site that never threaded a chain (there is exactly one:
        api/embeddings.py) must never claim a grace window, same fail-safe
        posture the KV restore gate already holds to."""
        q = TurbohaulQueue(staging_max=10)
        t0 = time.monotonic()
        candidate = _slot_with_chain(
            [{"role": "user", "content": "hello"}], created_at=t0 + 1,
        )
        inbox: "asyncio.Queue" = asyncio.Queue()
        inbox.put_nowait(candidate)
        matched = await _drain_chain(q, [], inbox, t0)
        assert matched is None
        assert inbox.qsize() == 1, "an empty anchor chain must not even peek/consume the inbox"

    async def test_genuine_extension_matches_via_staging(self):
        """The core positive case: a candidate whose chain is the anchor's
        chain PLUS one more real turn -- exactly what a growing conversation
        produces -- must match."""
        q = TurbohaulQueue(staging_max=10)
        t0 = time.monotonic()
        history = [
            {"role": "user", "content": "what is the capital of france"},
            {"role": "assistant", "content": "paris"},
        ]
        anchor_chain = _prefix_hash_chain(history)
        follow_up_history = history + [{"role": "user", "content": "and of germany"}]
        candidate = _slot_with_chain(follow_up_history, created_at=t0 + 1)
        await q.enqueue(candidate)
        matched = await _drain_chain(q, anchor_chain, None, t0)
        assert matched is candidate

    async def test_genuine_extension_matches_via_inbox(self):
        """Same positive case, but the candidate sits in the resident's
        inbox (the HIT route) rather than _staging -- the branch
        drain_inbox_and_staging_match_ip exists for."""
        q = TurbohaulQueue(staging_max=10)
        t0 = time.monotonic()
        history = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}]
        anchor_chain = _prefix_hash_chain(history)
        candidate = _slot_with_chain(history + [{"role": "user", "content": "three"}], created_at=t0 + 1)
        inbox: "asyncio.Queue" = asyncio.Queue()
        inbox.put_nowait(candidate)
        # Prove the fixture put the slot in r.inbox and NOT in
        # _staging, rather than assuming it -- a staging-only fixture cannot
        # discriminate the real drain from a chain predicate wrongly bolted
        # onto the staging-only pop_matched_thread chokepoint (both would see
        # it there). This is the SUBJECT population; the staging
        # test above is the CONTROL.
        assert candidate not in q.staging_snapshot(), "fixture bug: candidate leaked into _staging"
        assert inbox.qsize() == 1, "fixture bug: candidate never reached the inbox"
        matched = await _drain_chain(q, anchor_chain, inbox, t0)
        assert matched is candidate
        assert inbox.qsize() == 0

    async def test_PREFIX_MATCH_IS_THE_DISCRIMINATOR_not_mere_presence(self):
        """⭐ The core bar. Two candidates, BOTH carry a non-empty
        admission_hash_chain (both would pass a "chain is populated" test).
        Only ONE is a genuine extension of the anchor's own chain; the other
        is a real, unrelated conversation that merely happens to also have
        turns. A presence-only assertion cannot tell these apart -- prefix
        comparison must."""
        q = TurbohaulQueue(staging_max=10)
        t0 = time.monotonic()
        anchor_history = [
            {"role": "user", "content": "what is the capital of france"},
            {"role": "assistant", "content": "paris"},
        ]
        anchor_chain = _prefix_hash_chain(anchor_history)

        genuine_followup = _slot_with_chain(
            anchor_history + [{"role": "user", "content": "and of germany"}],
            created_at=t0 + 1,
        )
        unrelated_conversation = _slot_with_chain(
            [{"role": "user", "content": "totally different topic, different first turn"},
             {"role": "assistant", "content": "sure"}],
            created_at=t0 + 1,
        )
        assert unrelated_conversation.admission_hash_chain, (
            "test invalid if the negative candidate's chain is empty -- it must be "
            "PRESENT but non-extending, or this proves nothing about prefix-vs-presence"
        )

        await q.enqueue(unrelated_conversation)
        await q.enqueue(genuine_followup)
        matched = await _drain_chain(q, anchor_chain, None, t0)
        assert matched is genuine_followup, (
            f"matched {matched!r} -- a presence-only check would have accepted "
            f"either candidate; only the genuine extension may win"
        )
        # The unrelated one is still sitting in staging, untouched, FIFO-preserved --
        # not consumed, not lost, just correctly not this grace window's match.
        remaining = q.staging_snapshot()
        assert unrelated_conversation in remaining

    async def test_diverged_history_does_not_match(self):
        """A candidate that shares the anchor's FIRST turn but diverges at
        the second (a real compression/rewrite/branch, not a clean
        extension) must NOT match -- _is_prefix_match compares element-wise,
        not just chain[0]."""
        q = TurbohaulQueue(staging_max=10)
        t0 = time.monotonic()
        anchor_history = [
            {"role": "user", "content": "shared first turn"},
            {"role": "assistant", "content": "shared reply"},
        ]
        anchor_chain = _prefix_hash_chain(anchor_history)
        diverged_history = [
            {"role": "user", "content": "shared first turn"},
            {"role": "assistant", "content": "a DIFFERENT reply -- the chain diverges here"},
        ]
        candidate = _slot_with_chain(diverged_history, created_at=t0 + 1)
        await q.enqueue(candidate)
        matched = await _drain_chain(q, anchor_chain, None, t0)
        assert matched is None

    async def test_wrong_model_tag_never_matches(self):
        q = TurbohaulQueue(staging_max=10)
        t0 = time.monotonic()
        history = [{"role": "user", "content": "hi"}]
        anchor_chain = _prefix_hash_chain(history)
        candidate = _slot_with_chain(
            history + [{"role": "user", "content": "again"}],
            model_tag="other-model", created_at=t0 + 1,
        )
        await q.enqueue(candidate)
        matched = await _drain_chain(q, anchor_chain, None, t0)
        assert matched is None

    async def test_temporal_guard_rejects_pre_existing_slot(self):
        """Same guard_anchor discipline as pop_matched_ip: a slot already
        staged BEFORE this grace window opened is not "my own next turn"."""
        q = TurbohaulQueue(staging_max=10)
        t0 = time.monotonic()
        history = [{"role": "user", "content": "hi"}]
        anchor_chain = _prefix_hash_chain(history)
        candidate = _slot_with_chain(
            history + [{"role": "user", "content": "again"}],
            created_at=t0 - 5,  # arrived BEFORE grace_started_at
        )
        await q.enqueue(candidate)
        matched = await _drain_chain(q, anchor_chain, None, t0)
        assert matched is None


@pytest.mark.asyncio
class TestBoundedDrainInvariantDifferentOracle:
    """The new drain does NOT take a decline_sink
    (production never reads the IP sibling's back, AST-verified write-only),
    so it cannot inherit that method's "every slot examined"
    invariant test AS WRITTEN -- that test's oracle IS the sink. The
    invariant itself is still enforced (same bounded peek-drain-restore
    loop, same inline `raise AssertionError` on a head-only-drain), so it
    still needs a test -- just pinned by a DIFFERENT oracle: the RETURN
    VALUE (which candidate matched) plus the INBOX's own resulting contents
    and FIFO ORDER, observed directly, never via a sink dict.

    A head-only-drain bug (examines only the first inbox item, silently
    stops) would surface here as either the WRONG slot matching (a real
    match sitting behind a non-matching head) or the non-matching slots
    failing to reappear in the inbox afterward -- both directly observable
    without any sink.
    """

    async def test_match_behind_non_matching_head_is_found_and_examined(self):
        """Three candidates in the inbox: two do NOT extend the anchor's
        chain, one (in the MIDDLE, not the head) does. A head-only-drain
        bug would stop at the first non-match and never reach it."""
        q = TurbohaulQueue(staging_max=10)
        t0 = time.monotonic()
        anchor_history = [{"role": "user", "content": "root turn"}]
        anchor_chain = _prefix_hash_chain(anchor_history)

        not_a_match_1 = _slot_with_chain(
            [{"role": "user", "content": "unrelated conversation one"}], created_at=t0 + 1,
        )
        the_real_match = _slot_with_chain(
            anchor_history + [{"role": "user", "content": "genuine follow-up"}], created_at=t0 + 1,
        )
        not_a_match_2 = _slot_with_chain(
            [{"role": "user", "content": "unrelated conversation two"}], created_at=t0 + 1,
        )
        inbox: "asyncio.Queue" = asyncio.Queue()
        inbox.put_nowait(not_a_match_1)  # head -- must NOT stop the scan
        inbox.put_nowait(the_real_match)  # middle -- the one a head-only bug misses
        inbox.put_nowait(not_a_match_2)  # tail

        matched = await _drain_chain(q, anchor_chain, inbox, t0)

        assert matched is the_real_match, (
            f"matched {matched!r} instead of the middle candidate -- a head-only-drain "
            f"bug stops examining after the first non-match"
        )
        # Both non-matches must have been examined AND restored -- not dropped,
        # not left behind, FIFO order preserved (the IP sibling's own restore contract).
        assert inbox.qsize() == 2, (
            f"expected both non-matching candidates restored to the inbox, got "
            f"qsize={inbox.qsize()} -- a slot was silently dropped"
        )
        remaining = []
        while not inbox.empty():
            remaining.append(inbox.get_nowait())
        assert remaining == [not_a_match_1, not_a_match_2], (
            f"restored order {remaining!r} does not preserve original FIFO order"
        )

    async def test_all_candidates_examined_when_none_match(self):
        """No sink to read staging_depth/inbox_depth from -- the oracle here
        is that every one of five non-matching candidates reappears in the
        inbox, none dropped, none duplicated."""
        q = TurbohaulQueue(staging_max=10)
        t0 = time.monotonic()
        anchor_chain = _prefix_hash_chain([{"role": "user", "content": "root"}])
        candidates = [
            _slot_with_chain([{"role": "user", "content": f"unrelated {i}"}], created_at=t0 + 1)
            for i in range(5)
        ]
        inbox: "asyncio.Queue" = asyncio.Queue()
        for c in candidates:
            inbox.put_nowait(c)

        matched = await _drain_chain(q, anchor_chain, inbox, t0)

        assert matched is None
        assert inbox.qsize() == 5, f"expected all 5 candidates restored, got {inbox.qsize()}"
        remaining = []
        while not inbox.empty():
            remaining.append(inbox.get_nowait())
        assert remaining == candidates, "restore must preserve exact identity and FIFO order"


# ===========================================================================
# GROUP B -- the real manager, real FastAPI app, real grace loop
# (_serve_on_resident), real fake-sidecar harness (same pattern, reused
# as-is from test_interturn_race_gate.py). GROWING multi-turn history, so
# admission_hash_chain is a genuine rolling prefix chain -- the shape a real
# client sends -- not a series of unrelated single-message hashes.
# ===========================================================================
class _FakeStreamResponse:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self):
        return b""


class _FakeStreamCM:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return _FakeStreamResponse(self._chunks)

    async def __aexit__(self, *exc):
        return False


class _FakeAsyncClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, *a, **k):
        return _FakeStreamCM(_SSE_CHUNKS)


_SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"hi"},"index":0}]}\n\n',
    b"data: [DONE]\n\n",
]


def _make_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 12345
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _write_manifest_yaml(manifests_root, tag: str):
    (manifests_root / f"{tag}.yaml").write_text(
        f"""model_tag: {tag}
gguf_blob_sha256: "{'a' * 64}"
gguf_size_bytes: 1000
"""
    )


def _boot_and_runtime(tmp_path, *, grace_seconds=10, shared_addresses=""):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    _write_manifest_yaml(storage_root / "manifests", _MODEL)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake",
            default_port_base=59800,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            max_parallel_sidecars=2,  # the only live site, per manager.py's own comment
            safety_enabled=False,
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=300,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            grace_ip_match_shared_addresses=shared_addresses,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _boot_app(tmp_path, *, grace_seconds, shared_addresses=""):
    telemetry_module._telemetry = None
    boot, runtime = _boot_and_runtime(tmp_path, grace_seconds=grace_seconds, shared_addresses=shared_addresses)
    app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
    mgr = app.state.manager

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        return _make_handle(model_tag, port)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def fake_sigterm(handle, **kwargs):
        return True, "sigterm-clean"

    async def fake_vram(**kwargs):
        return True, 100

    async def fake_complete(slot, handle):
        return {
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1700000000,
            "model": slot.model_tag,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    mgr._spawn = fake_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = fake_sigterm
    mgr._vram_verify = fake_vram
    mgr._complete_fn = fake_complete
    return app, mgr


def _stream_history(client, messages):
    body = b""
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": _MODEL, "messages": messages, "stream": True},
    ) as r:
        status = r.status_code
        for chunk in r.iter_bytes():
            body += chunk
    return status, body


class TestDrivenGraceMatchViaRealManager:
    """⭐ The load-bearing arms: prove the
    admissions that used to rely on the client-ADDRESS fallback are STILL
    served -- by the content-derived path, through the real grace loop, with
    state that ARISES from driven HTTP traffic (never a hand-set `matched`
    or a hand-placed slot in the "convenient" container).

    Both tests below use a GROWING history (each turn = prior history + one
    new short user turn) so the whole-prompt thread_id churns EVERY turn
    (a correction to slot.py's own `else normalized`
    branch: true only <=256 words) while admission_hash_chain accumulates a
    genuine rolling prefix -- the real shape of client traffic.
    """

    def test_RED_GREEN_churn_plus_ip_denylisted_still_served_by_chain(self, tmp_path, monkeypatch):
        """THE CORE FIX. The client's own address is DENYLISTED
        (grace_ip_match_shared_addresses), so grace_ip_match_eligible refuses
        it -- on the base tree this leaves NO fallback at all once
        pop_matched_thread misses, and the follow-up strands for the full
        grace window (~10s, RED against the <7s bar). On the fixed tree the
        content-derived match does not consult IP eligibility at all and
        still serves it fast (GREEN) -- proving the new path is not merely
        piggy-backing on IP still being reachable.
        """
        app, mgr = _boot_app(tmp_path, grace_seconds=10, shared_addresses="testclient,172.16.0.1")
        with TestClient(app) as client:
            monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

            history = [{"role": "user", "content": "capital of france"}]
            status, body = _stream_history(client, history)
            assert status == 200 and b"hi" in body
            history.append({"role": "assistant", "content": "hi"})

            max_inter_turn = 0.0
            for i in range(1, 6):
                history.append({"role": "user", "content": f"and turn {i}"})
                t0 = time.monotonic()
                status, body = _stream_history(client, history)
                t1 = time.monotonic()
                max_inter_turn = max(max_inter_turn, t1 - t0)
                assert status == 200, f"turn {i}: {status}"
                assert b"hi" in body, f"turn {i} produced no token"
                history.append({"role": "assistant", "content": "hi"})

            assert max_inter_turn < 7.0, (
                f"inter-turn time {max_inter_turn:.2f}s -- IP-denylisted + churning thread_id "
                f"stranded the follow-up for the grace window instead of matching by "
                f"admission_hash_chain. On the base tree (no chain fallback), this assertion "
                f"is RED at ~grace_seconds=10s; the fix makes it GREEN."
            )
        telemetry_module._telemetry = None

    def test_STILL_SERVED_inbox_population_not_lost(self, tmp_path, monkeypatch, caplog):
        """⭐ THE INBOX POPULATION: the dominant real
        case sits in r.inbox (the HIT route), not _staging, and the IP is
        NOT denylisted here -- this is exactly the traffic that worked via
        `via=pop_matched_ip_inbox` before. Proves two things at once: (1)
        still served fast (not lost), and (2) served by the NEW mechanism,
        not coincidentally still by IP -- the old via= label cannot appear
        post-fix (its call site no longer exists), and caplog confirms
        which label actually fired rather than inferring it from latency
        alone.
        """
        import logging
        app, mgr = _boot_app(tmp_path, grace_seconds=10, shared_addresses="")
        caplog.set_level(logging.INFO, logger="turbohaul.queue")
        # TestClient's host is the literal string "testclient", which
        # ipaddress.ip_address() rejects -- on the BASELINE tree that breaks
        # drain_inbox_and_staging_match_ip's own parse (`except ValueError:
        # return None`) before the IP key is ever compared, which would make
        # this test "pass" on old code for the WRONG reason (identity refused,
        # not identity matched) and understate what is actually being
        # retired. Patched exactly as test_interturn_race_gate.py
        # does, so old code gets a fair, working IP path to be compared against.
        _real_ip_address = ipaddress.ip_address

        def _patched_ip(ip):
            if ip == "testclient":
                return _real_ip_address("172.16.0.1")
            return _real_ip_address(ip)

        monkeypatch.setattr(ipaddress, "ip_address", _patched_ip)
        with TestClient(app) as client:
            monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

            history = [{"role": "user", "content": "capital of germany"}]
            status, body = _stream_history(client, history)
            assert status == 200 and b"hi" in body
            history.append({"role": "assistant", "content": "hi"})

            for i in range(1, 4):
                history.append({"role": "user", "content": f"followup {i}"})
                status, body = _stream_history(client, history)
                assert status == 200
                assert b"hi" in body
                history.append({"role": "assistant", "content": "hi"})

        telemetry_module._telemetry = None
        assert "via=pop_matched_hash_chain_inbox" in caplog.text, (
            "the dominant real population (requests sitting in the resident inbox) "
            "must be served via the new content-derived path -- caplog shows what actually "
            "fired, not what the latency alone would suggest.\n--- captured log ---\n"
            + caplog.text
        )
        assert "via=pop_matched_ip" not in caplog.text, (
            "the retired via= labels must never fire again -- their call site is gone"
        )

    def test_GREEN_CONTROL_stable_thread_unaffected(self, tmp_path, monkeypatch):
        """A conversation whose first message alone exceeds the 256-word
        prefix cutoff keeps a STABLE derived thread_id
        (derive_thread_id_prefix_hash's `if len(words) > n` branch) --
        pop_matched_thread hits every time and this change is never
        consulted. Must stay fast on both trees; if a machine-wide slowdown
        were the real cause, this would fail too."""
        app, mgr = _boot_app(tmp_path, grace_seconds=10)
        long_prefix = " ".join(f"sysword{i}" for i in range(300))
        with TestClient(app) as client:
            monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
            history = [{"role": "user", "content": f"{long_prefix} turn0"}]
            status, body = _stream_history(client, history)
            assert status == 200 and b"hi" in body
            history.append({"role": "assistant", "content": "hi"})
            for i in range(1, 4):
                history.append({"role": "user", "content": f"{long_prefix} turn{i}"})
                t0 = time.monotonic()
                status, body = _stream_history(client, history)
                inter_turn = time.monotonic() - t0
                assert status == 200
                assert inter_turn < 6.0, f"same-thread turn {i} took {inter_turn:.2f}s"
                history.append({"role": "assistant", "content": "hi"})
        telemetry_module._telemetry = None


# ===========================================================================
# GROUP C -- embeddings parity: embeddings stays
# excluded from BOTH the retired path and the new one -- byte-identical.
# ===========================================================================
class TestEmbeddingsParityUnaffected:
    def test_embeddings_admission_carries_no_hash_chain(self, tmp_path, monkeypatch):
        """api/embeddings.py is the one submit* call site that is known
        to never thread admission_hash_chain.
        Pinned here at the Slot the manager actually admits -- if a future
        change threads a chain in (something this change must not
        introduce), this goes RED."""
        app, mgr = _boot_app(tmp_path, grace_seconds=10)
        # The shared manifest fixture doesn't declare embeddings support;
        # this route's own capability gate refuses before admission
        # otherwise. Re-declare it for this one test only.
        (mgr.boot.storage.manifests_path / f"{_MODEL}.yaml").write_text(
            f"""model_tag: {_MODEL}
gguf_blob_sha256: "{'a' * 64}"
gguf_size_bytes: 1000
llama_server_flags:
  embeddings: true
"""
        )
        captured = {}
        real_submit = mgr.submit_for_streaming

        async def spy_submit_for_streaming(*a, **kw):
            captured["admission_hash_chain"] = kw.get("admission_hash_chain")
            captured["called"] = True
            return await real_submit(*a, **kw)

        mgr.submit_for_streaming = spy_submit_for_streaming
        with TestClient(app) as client:
            monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
            try:
                # Only the ADMISSION kwargs matter here (captured by the spy
                # above, before any sidecar I/O runs) -- the fake httpx client
                # models the chat-completion SSE shape, not embeddings', so a
                # downstream error past admission is expected and irrelevant.
                client.post("/v1/embeddings", json={"model": _MODEL, "input": "embed this text"})
            except Exception:
                pass
        telemetry_module._telemetry = None
        assert captured.get("called"), "embeddings route did not reach submit_for_streaming -- test proves nothing"
        assert not captured.get("admission_hash_chain"), (
            f"embeddings threaded a non-empty admission_hash_chain "
            f"({captured.get('admission_hash_chain')!r}) -- this is a behaviour change "
            f"FORBIDDEN by this change (embeddings must stay excluded from the identity "
            f"mechanism by construction, same as today)"
        )
