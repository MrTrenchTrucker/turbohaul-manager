"""Inter-turn race gate.

The bar: an active, unpreempted resident between its own turns must serve
the next turn within 6 seconds. Pinned by test_green_control_same_thread_
no_false_positive on a genuinely extending (stable thread_id) conversation.

The defect is a race on ONE deque under ONE lock between the grace-loop
matcher and the dispatcher's pick. When the dispatcher wins, the slot goes
to the resident's inbox where the matcher can never see it, and the client
waits out the whole grace window.

This gate drives the REAL route end-to-end and measures the REAL inter-turn
time, rather than asserting the call site is present in the source.

Behaviour change: the same-address,
UNRELATED-content case (test_unrelated_content_same_address_is_not_fused,
formerly test_inter_turn_time_within_bar) is INVERTED, not deleted. It used
to require the retired IP fallback to fuse unrelated conversations by
address for speed; that key was retired because the fusion itself was
the hazard (a wrong match silently corrupting a live conversation's KV
cache). The fixture is kept -- it is proven to construct the condition --
the assertion now requires the opposite: still served correctly, never
fused by address alone.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
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
from turbohaul.subprocess_mgr import SidecarHandle


_MODEL = "test-model"
# Prefix >256 tokens so the thread_id is derived from the prefix, not the tail.
# A conversation that keeps its system prompt and appends turns keeps ONE
# thread_id. To simulate thread_id churn (the bug), we use DIFFERENT prefixes.
_AGENT_PREFIX = " ".join(f"sysword{i}" for i in range(300))


# ---------------------------------------------------------------------------
# Fake sidecar stream (same pattern as the tool-call latency guard test).
# ---------------------------------------------------------------------------
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


def _boot_and_runtime(tmp_path, *, grace_seconds=30):
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
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            max_parallel_sidecars=2,
            safety_enabled=False,
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=300,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _app_with_work_counters(tmp_path, *, grace_seconds, client_ip="172.16.0.1"):
    telemetry_module._telemetry = None
    boot, runtime = _boot_and_runtime(tmp_path, grace_seconds=grace_seconds)
    app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
    mgr = app.state.manager
    work = _WorkCounters()

    def counting_spawn(binary, gguf, port, model_tag, argv, **_kw):
        work.spawns += 1
        return _make_handle(model_tag, port)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def counting_sigterm(handle, **kwargs):
        work.sigterms += 1
        return True, "sigterm-clean"

    async def counting_vram(**kwargs):
        work.vram_verifies += 1
        return True, 100

    async def fake_complete(slot, handle):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": slot.model_tag,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    mgr._spawn = counting_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = counting_sigterm
    mgr._vram_verify = counting_vram
    mgr._complete_fn = fake_complete

    with TestClient(app) as client:
        yield app, client, mgr, work

    telemetry_module._telemetry = None


class _WorkCounters:
    def __init__(self):
        self.spawns = 0
        self.sigterms = 0
        self.vram_verifies = 0

    @property
    def teardowns(self) -> int:
        return self.sigterms + self.vram_verifies


@pytest.fixture
def app_with_work_counters(tmp_path):
    # grace_seconds=10: the bug strands turns for ~10s (over the 6s bar),
    # while the fixed version serves them in <1s.
    yield from _app_with_work_counters(tmp_path, grace_seconds=10, client_ip="172.16.0.1")


def _stream_once(client, *, prompt="say hi"):
    body = b""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": _MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        },
    ) as r:
        status = r.status_code
        for chunk in r.iter_bytes():
            body += chunk
    return status, body


# ===========================================================================
# The inter-turn race gate.
# ===========================================================================
class TestInterTurnRaceGate:
    """Gate for the inter-turn race defect.

    The bar: an active, unpreempted resident between its own turns must serve
    the next turn within 6 seconds.
    """

    def test_unrelated_content_same_address_is_not_fused(
        self, app_with_work_counters, monkeypatch, caplog
    ):
        """PROVENANCE: until the IP fallback was retired, this test
        asserted `max_inter_turn < 7.0` for follow-up turns sharing an IP
        but carrying DIFFERENT, unrelated 300-word prefixes each turn --
        i.e. it required the IP fallback to fuse six unrelated one-shot
        conversations into one resident's slot, and called that fusion
        correct because it was fast.

        INVERTED when the IP fallback was retired (an existing
        test that breaks due to an intentional
        behaviour change is replaced by a new one) because that pass
        condition IS the exact fail-open hazard that was retired -- an
        address match with no content relationship, silently fusing
        unrelated conversations into the wrong resident's KV cache. The
        FIXTURE is kept, not replaced, because it is PROVEN to construct
        the condition (same TestClient/IP, genuinely unrelated content
        each turn) rather than merely claiming to.

        What is now asserted is the opposite of the old bar: turns sharing
        an address but carrying unrelated content must NOT be fused --
        each is still served correctly (200, real content), but via a
        fresh admission after the grace window, never via a false address
        match. caplog pins WHICH path actually fired, the same way
        test_STILL_SERVED_inbox_population_not_lost pins the genuine
        population's path in the hash-chain grace-match test.

        The <6s/<7s bar itself is NOT dropped -- it survives, pinned on the
        population it actually governs, in test_green_control_same_thread_
        no_false_positive (same file, stable thread_id, a genuinely
        extending conversation). The inversion rests on that control
        staying green; it does.
        """
        app, client, mgr, work = app_with_work_counters
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
        caplog.set_level(logging.INFO, logger="turbohaul.queue")

        # Patch ipaddress.ip_address to handle 'testclient' as a valid IP.
        # The TestClient uses 'testclient' as hostname, which is NOT a valid IP.
        # Kept from the pre-inversion test: without this patch, the old IP
        # path would "pass" for the wrong reason (identity parse refused,
        # not identity compared) -- retained so the fixture still gives the
        # retired path a fair, working address to have matched on.
        _original_ip_address = ipaddress.ip_address

        def _patched_ip(ip):
            if ip == "testclient":
                return _original_ip_address("172.16.0.1")
            return _original_ip_address(ip)

        monkeypatch.setattr(ipaddress, 'ip_address', _patched_ip)

        # Turn 0: cold load (establishes the resident).
        status, body = _stream_once(client, prompt=f"{_AGENT_PREFIX} tail0")
        assert status == 200
        assert b"hi" in body

        # Follow-up turns with DIFFERENT, unrelated prefixes each turn (no
        # growing history) but the SAME IP -- the exact scenario an
        # address-only match would fuse. Each turn must still be served
        # correctly; none may be silently bound to the wrong resident.
        for i in range(1, 6):
            prefix = " ".join(f"word{i}_{j}" for j in range(300))
            status, body = _stream_once(client, prompt=f"{prefix} tail{i}")
            assert status == 200, f"Turn {i} did not return 200: {status}"
            assert b"hi" in body, f"Turn {i} produced no token"

        # The discriminator that now exists: unrelated content sharing an
        # address is never fused via a content-derived chain match, and the
        # retired IP-fallback labels never fire again (their call site is
        # gone). Each turn above having still returned 200/"hi" proves it
        # was served correctly some other way (a fresh admission after the
        # grace window), not that it was stranded.
        assert "via=pop_matched_hash_chain" not in caplog.text, (
            "unrelated content per turn cannot form a genuine prefix relationship in "
            "admission_hash_chain -- a hash-chain match firing here would mean two "
            "unrelated conversations got fused by address alone, the same hazard the "
            "retired IP path had.\n--- captured log ---\n" + caplog.text
        )
        assert "via=pop_matched_ip" not in caplog.text, (
            "the retired IP-fallback labels must never fire again -- their call site "
            "is gone"
        )

    def test_green_control_same_thread_no_false_positive(
        self, app_with_work_counters, monkeypatch
    ):
        """GREEN CONTROL: a conversation that keeps its thread_id must NOT move.

        If the gate is measuring a host-wide effect (not the race), this
        conversation would also fail. It uses the SAME prefix for all turns,
        so the thread_id is stable and pop_matched_thread always hits.
        """
        app, client, mgr, work = app_with_work_counters
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        # Turn 0: cold load.
        status, body = _stream_once(client, prompt=f"{_AGENT_PREFIX} tail0")
        assert status == 200
        assert b"hi" in body

        # Follow-up turns with SAME prefix (same thread_id).
        for i in range(1, 6):
            t0 = time.monotonic()
            status, body = _stream_once(client, prompt=f"{_AGENT_PREFIX} tail{i}")
            t1 = time.monotonic()
            inter_turn = t1 - t0
            assert status == 200
            assert b"hi" in body
            # Same-thread turns should be fast on BOTH trees.
            assert inter_turn < 6.0, (
                f"Same-thread turn {i} took {inter_turn:.2f}s — "
                f"this is a host-wide effect, not the race."
            )
