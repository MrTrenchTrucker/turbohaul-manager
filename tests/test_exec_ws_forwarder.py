"""Unit tests for the WS exec forwarder (hop 1) — tests/test_exec_ws_forwarder.py.

Covers the WS exec contract surface without a live container: the resolution
chokepoint (upgrade refused pre-accept on every read_manifest/resolve failure
mode, target never reached), byte-identical frame pass-through, run_id
extraction, the dedicated target-lost error frame (protocol code 101),
client-disconnect cleanup, and the IPv6 URL-bracket helper.

The websockets client is replaced with a scripted fake; the ASGI app is real
(FastAPI + TestClient, starlette's websocket test session).
"""
import asyncio
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from types import SimpleNamespace

from turbohaul.api import exec_ws
from turbohaul.config import PluginEndpoint
from turbohaul.manifest import _safe_manifest_path
from starlette.websockets import WebSocketDisconnect


def _app(manifests_root, registry):
    app = FastAPI()
    mgr = SimpleNamespace(
        boot=SimpleNamespace(
            storage=SimpleNamespace(manifests_path=manifests_root),
            plugins=SimpleNamespace(registry=registry),
        ),
        runtime=SimpleNamespace(
            plugin_runtime=SimpleNamespace(no_progress_timeout_s=None)
        ),
    )
    app.state.manager = mgr
    app.include_router(exec_ws.router)
    return app


def _write_manifest(root, tag, resource_key="whisperx"):
    path = _safe_manifest_path(root, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"kind: plugin\nmodel_tag: {tag}\nlane: cpu\n"
        f"resource_key: {resource_key}\ncapabilities: [video-decode]\n"
    )
    return path


class _FakeClosed(Exception):
    """Stands in for websockets ConnectionClosed in the fake target."""


class FakeTarget:
    def __init__(self, frames, fail_after=None, hang=False):
        self.frames = list(frames)
        self.fail_after = fail_after  # raise _FakeClosed after N frames yielded
        self.hang = hang  # stay open (live idle target) until close() is called
        self.yielded = 0
        self.received = []
        self.closed = False
        self._closed = asyncio.Event()
        # Protocol faithfulness: the real endpoint produces NOTHING until the
        # client has finished its input phase (stdin_eof) — it only then
        # probes, spawns, and starts streaming. Gating the scripted frames on
        # stdin_eof (not just on the first client frame) keeps two hazards
        # out of the tests: (a) the target cannot finish its script before
        # the client's upload frames have been forwarded (the forwarder would
        # tear the session down mid-upload), and (b) the run_id extraction
        # race is closed as a consequence.
        self._eof_seen = asyncio.Event()

    async def send(self, data):
        self.received.append(data)
        if isinstance(data, str) and '"stdin_eof"' in data:
            self._eof_seen.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.fail_after is not None and self.yielded >= self.fail_after:
            raise _FakeClosed("simulated target death")
        if self.frames:
            await self._eof_seen.wait()  # speak only after the client's input
            self.yielded += 1
            return self.frames.pop(0)
        if self.hang:
            await self._closed.wait()
        raise StopAsyncIteration  # clean close (terminal frame done / after close)

    async def close(self):
        self.closed = True
        self._closed.set()


@pytest.fixture
def fake_connect(monkeypatch):
    """Replaces websockets.asyncio.client.connect with a scripted fake."""
    import websockets.asyncio.client as _mod

    state = {"target": None, "url": None, "kwargs": None}

    async def _fake_connect(url, **kwargs):
        state["url"] = url
        state["kwargs"] = kwargs
        return state["target"] or FakeTarget([])

    monkeypatch.setattr(_mod, "connect", _fake_connect)
    return state


def _wait_for(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# _host_for_url (pure)
# ---------------------------------------------------------------------------

def test_host_for_url_brackets_ipv6_and_leaves_others_alone():
    assert exec_ws._host_for_url("172.16.0.8") == "172.16.0.8"
    assert exec_ws._host_for_url("whisperx-diarize") == "whisperx-diarize"
    assert exec_ws._host_for_url("fe80::1") == "[fe80::1]"
    # an already-bracketed host is never double-wrapped (invoke_plugin precedent)
    assert exec_ws._host_for_url("[fe80::1]") == "[fe80::1]"


# ---------------------------------------------------------------------------
# resolution chokepoint
# ---------------------------------------------------------------------------

def test_upgrade_refused_unknown_tag_target_never_reached(tmp_path, fake_connect):
    root = tmp_path / "manifests"
    app = _app(root, {"whisperx": PluginEndpoint(host="whisperx-diarize", port=8080)})
    client = TestClient(app)
    # a refused upgrade (pre-accept close) surfaces as WebSocketDisconnect at
    # session ENTER — and the target endpoint must NEVER be contacted.
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/api/plugins/no-such-tag/exec"):
            pass
    assert exc_info.value.code == 1008
    assert fake_connect["url"] is None  # the load-bearing assertion


def test_upgrade_refused_when_registry_entry_missing(tmp_path, fake_connect):
    root = tmp_path / "manifests"
    _write_manifest(root, "ffmpeg-video", resource_key="no-such-resource")
    # registry without that key -> resolve_endpoint raises unknown_resource
    app = _app(root, {})
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/api/plugins/ffmpeg-video/exec"):
            pass
    assert exc_info.value.code == 1008
    assert fake_connect["url"] is None


def test_upgrade_refused_when_tag_traverses(tmp_path, fake_connect):
    root = tmp_path / "manifests"
    app = _app(root, {"whisperx": PluginEndpoint(host="h", port=1)})
    client = TestClient(app)
    # validate_tag (inside read_manifest) must reject traversal BEFORE any
    # filesystem access; the upgrade is refused and no target is reached.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/plugins/..%2Fetc/exec"):
            pass
    # whichever layer refused (route 404 or validate_tag), the target is
    # never reached
    assert fake_connect["url"] is None


# ---------------------------------------------------------------------------
# happy path: byte-identical pass-through
# ---------------------------------------------------------------------------

def test_happy_path_frames_pass_through_byte_identical(tmp_path, fake_connect):
    root = tmp_path / "manifests"
    _write_manifest(root, "ffmpeg-video")
    start = json.dumps({"type": "start", "binary": "ffmpeg",
                        "argv": ["-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-i", "pipe:0", "-f", "rawvideo", "pipe:1"],
                        "probe_first": True, "run_id": "run-abc"})
    stdin_eof = json.dumps({"type": "stdin_eof"})
    meta = json.dumps({"type": "meta", "width": 320, "height": 240,
                       "frame_bytes": 230400, "nb_frames": 10,
                       "duration": 4.2, "run_id": "run-abc"})
    ready = json.dumps({"type": "ready", "run_id": "run-abc"})
    chunk1, chunk2 = b"\x00" * 1000 + b"AB", b"XY" * 500
    exitf = json.dumps({"type": "exit", "code": 0, "signal": None,
                        "stderr": "", "run_id": "run-abc"})

    fake_connect["target"] = FakeTarget([meta, ready, chunk1, chunk2, exitf])
    app = _app(root, {"whisperx": PluginEndpoint(host="172.16.0.8", port=8080)})
    client = TestClient(app)
    with client.websocket_connect("/api/plugins/ffmpeg-video/exec") as session:
        session.send_text(start)
        session.send_bytes(b"INPUT-A")
        session.send_bytes(b"INPUT-B")
        session.send_text(stdin_eof)
        assert json.loads(session.receive_text()) == json.loads(meta)
        assert json.loads(session.receive_text()) == json.loads(ready)
        assert session.receive_bytes() == chunk1
        assert session.receive_bytes() == chunk2
        assert json.loads(session.receive_text()) == json.loads(exitf)

    target: FakeTarget = fake_connect["target"]
    # client -> target: byte-identical, in order (text stays text, bytes bytes)
    assert target.received[0] == start
    assert target.received[1] == b"INPUT-A"
    assert target.received[2] == b"INPUT-B"
    assert target.received[3] == stdin_eof
    assert len(target.received) == 4
    # target URL: resolved endpoint + the contract path (generic /ws/exec)
    assert fake_connect["url"] == "ws://172.16.0.8:8080/ws/exec"


# ---------------------------------------------------------------------------
# failure paths
# ---------------------------------------------------------------------------

def test_target_lost_mid_stream_sends_dedicated_error_frame(tmp_path, fake_connect):
    root = tmp_path / "manifests"
    _write_manifest(root, "ffmpeg-video")
    start = json.dumps({"type": "start", "binary": "ffmpeg", "argv": ["-i", "pipe:0"],
                        "probe_first": True, "run_id": "run-777"})
    meta = json.dumps({"type": "meta", "width": 2, "height": 2,
                       "frame_bytes": 12, "nb_frames": 1, "duration": 0.1,
                       "run_id": "run-777"})
    # two good frames (meta, OUT), then the target connection dies
    fake_connect["target"] = FakeTarget([meta, b"OUT"], fail_after=2)
    app = _app(root, {"whisperx": PluginEndpoint(host="172.16.0.8", port=8080)})
    client = TestClient(app)
    with client.websocket_connect("/api/plugins/ffmpeg-video/exec") as session:
        session.send_text(start)
        session.send_text(json.dumps({"type": "stdin_eof"}))
        assert json.loads(session.receive_text()) == json.loads(meta)
        assert session.receive_bytes() == b"OUT"
        err = json.loads(session.receive_text())
        assert err["type"] == "error"
        assert err["code"] == exec_ws._TARGET_LOST_CODE == 101
        assert err["run_id"] == "run-777"
        assert "target" in err["stderr"].lower()
        with pytest.raises(Exception):  # 1011 close follows the error frame
            session.receive_text()


def test_client_disconnect_closes_target(tmp_path, fake_connect):
    root = tmp_path / "manifests"
    _write_manifest(root, "ffmpeg-video")
    start = json.dumps({"type": "start", "binary": "ffprobe",
                        "argv": ["-i", "pipe:0"], "probe_first": True,
                        "run_id": "run-dc"})
    # target script stays open (idle) — the client goes away first; the
    # forwarder must close the target in response
    fake_connect["target"] = FakeTarget([], hang=True)
    app = _app(root, {"whisperx": PluginEndpoint(host="172.16.0.8", port=8080)})
    client = TestClient(app)
    with client.websocket_connect("/api/plugins/ffmpeg-video/exec") as session:
        session.send_text(start)
        session.close()
    target: FakeTarget = fake_connect["target"]
    assert _wait_for(lambda: target.closed), "forwarder must close the target " \
        "when the client disconnects (so the endpoint reaps its child)"


def test_run_id_not_in_start_frame_stays_opaque(tmp_path, fake_connect):
    """A malformed start frame (no run_id) must not break pass-through: the
    forwarder forwards it verbatim and logs run_id '?'; the endpoint is the
    one that validates the protocol."""
    root = tmp_path / "manifests"
    _write_manifest(root, "ffmpeg-video")
    bad = json.dumps({"type": "start", "binary": "ffmpeg",
                      "argv": ["-i", "pipe:0"], "probe_first": True})
    fake_connect["target"] = FakeTarget(["garbage-but-forwarded"], fail_after=1)
    app = _app(root, {"whisperx": PluginEndpoint(host="172.16.0.8", port=8080)})
    client = TestClient(app)
    with client.websocket_connect("/api/plugins/ffmpeg-video/exec") as session:
        session.send_text(bad)
        # the (fake) target holds its frames until stdin_eof (protocol
        # faithfulness) — complete the input phase, then receive.
        session.send_text(json.dumps({"type": "stdin_eof"}))
        # the (fake) target's frame comes back through untouched
        assert session.receive_text() == "garbage-but-forwarded"
    assert fake_connect["target"].received[0] == bad


# ---------------------------------------------------------------------------
# provides_executables consumption (manifest schema field;
# the field is declared with the manifest schema — this file only CONSUMES it)
# ---------------------------------------------------------------------------

def test_manifest_without_field_imposes_no_hop1_restriction(tmp_path, fake_connect):
    """Back-compat (today's manifests have no provides_executables): the
    hop-1 declaration check must be inert — the target's allow-list is the
    only gate. A start frame is forwarded untouched."""
    root = tmp_path / "manifests"
    _write_manifest(root, "ffmpeg-video")
    fake_connect["target"] = FakeTarget([
        json.dumps({"type": "meta", "width": 0, "height": 0, "frame_bytes": 0,
                    "nb_frames": 0, "duration": 0.0, "run_id": "run-hc1"}),
        json.dumps({"type": "ready", "run_id": "run-hc1"}),
        json.dumps({"type": "exit", "code": 0, "signal": None, "stderr": "",
                    "run_id": "run-hc1"}),
    ])
    app = _app(root, {"whisperx": PluginEndpoint(host="172.16.0.8", port=8080)})
    client = TestClient(app)
    start = json.dumps({"type": "start", "binary": "ffmpeg",
                        "argv": ["-i", "pipe:0"], "probe_first": False,
                        "run_id": "run-hc1", "input_index": 1})
    with client.websocket_connect("/api/plugins/ffmpeg-video/exec") as session:
        session.send_text(start)
        session.send_text(json.dumps({"type": "stdin_eof"}))
        assert json.loads(session.receive_text())["type"] == "meta"  # forwarded
    # the binary WAS forwarded to the target despite no declaration field
    assert fake_connect["target"].received[0] == start


def test_manifest_declares_executables_violation_refused_at_hop1(tmp_path, fake_connect, monkeypatch):
    """The manifest DECLARES provides_executables: a start frame naming a
    binary OUTSIDE the declaration is refused at hop 1 — error frame 127 to
    the client, NOTHING forwarded to the target (the target's allow-list
    remains the security property; this check only fails earlier)."""
    from turbohaul.api import exec_ws as ew
    root = tmp_path / "manifests"
    _write_manifest(root, "ffmpeg-video")
    # today's PluginManifest may not have such a field yet; it can be declared separately.
    # Simulate its shape by attaching the attribute to the real
    # manifest object read_manifest returns (consumption uses getattr, so
    # this is exactly what a declared field looks like at runtime).
    real_read = ew.read_manifest

    def _read_with_field(root_p, tag):
        m = real_read(root_p, tag)
        object.__setattr__(m, "provides_executables", ["ffmpeg"])
        return m

    monkeypatch.setattr(ew, "read_manifest", _read_with_field)
    fake_connect["target"] = FakeTarget([], hang=True)
    app = _app(root, {"whisperx": PluginEndpoint(host="172.16.0.8", port=8080)})
    client = TestClient(app)
    start = json.dumps({"type": "start", "binary": "ffprobe",  # NOT declared
                        "argv": ["-i", "pipe:0"], "probe_first": False,
                        "run_id": "run-hc2", "input_index": 1})
    with client.websocket_connect("/api/plugins/ffmpeg-video/exec") as session:
        session.send_text(start)
        err = json.loads(session.receive_text())
        assert err["type"] == "error" and err["code"] == 127
        assert "not declared" in err["stderr"]
        assert err["run_id"] == "run-hc2"
        # the server closes right after the refusal frame:
        with pytest.raises(WebSocketDisconnect):
            session.receive_text()
    # NOTHING was forwarded to the target
    assert fake_connect["target"].received == []
    # control: the DECLARED binary passes the hop-1 check (same manifest)
    fake_connect["target"] = FakeTarget([
        json.dumps({"type": "meta", "width": 0, "height": 0, "frame_bytes": 0,
                    "nb_frames": 0, "duration": 0.0, "run_id": "run-hc3"}),
        json.dumps({"type": "ready", "run_id": "run-hc3"}),
        json.dumps({"type": "exit", "code": 0, "signal": None, "stderr": "",
                    "run_id": "run-hc3"}),
    ])
    with client.websocket_connect("/api/plugins/ffmpeg-video/exec") as session:
        session.send_text(start.replace('"ffprobe"', '"ffmpeg"')
                          .replace("run-hc2", "run-hc3"))
        session.send_text(json.dumps({"type": "stdin_eof"}))
        assert json.loads(session.receive_text())["type"] == "meta"
    assert fake_connect["target"].received[0] is not None


def test_declared_violation_pure_function():
    """The check itself: field absent / empty / non-list = inert; declared
    binary passes; undeclared binary is flagged; non-str binary is inert
    (garbage-in is the TARGET's validation job, not hop 1's)."""
    from turbohaul.api import exec_ws as ew

    class M:  # manifest stand-in (the check only uses getattr)
        pass
    m = M()
    assert ew._declared_violation(m, "ffmpeg") is None  # no field at all
    m2 = M(); m2.provides_executables = []
    assert ew._declared_violation(m2, "ffmpeg") is None  # empty = no restriction
    m3 = M(); m3.provides_executables = "not-a-list"
    assert ew._declared_violation(m3, "ffmpeg") is None  # malformed = inert
    m4 = M(); m4.provides_executables = ["ffmpeg", "ffprobe"]
    assert ew._declared_violation(m4, "ffmpeg") is None  # declared
    assert ew._declared_violation(m4, "rm") == "rm"  # undeclared -> flagged
    assert ew._declared_violation(m4, None) is None  # non-str: not hop 1's job
