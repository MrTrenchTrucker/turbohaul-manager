"""Tests for the engine's generic PATH shim (engine/ffshim/ffshim.py).

One shim implementation installed under any executable name (by design:
the binary is a PARAMETER; ffmpeg/ffprobe are two installs of this one file).

Unit layer (no I/O): the TailDrop clean-multiple invariant (the
honest-failure bar), the input-operand locator (flag-parameterised), and the
install-config loader.

Integration layer (subprocess + a real local websockets server): the shim's
full protocol against a scripted fake manager+endpoint — byte-exact stdout,
the tail-drop on a deliberate mid-stream death, the pre-network path-input
fail-fast, and the remote error-frame exit code.

LOCAL-FIRST layer: a real binary of the same name on PATH (not the shim
itself) is execv'd directly — zero hops, zero network (proven against a dead
port).

The shim is executed as a COPY named after the binary it serves (its
self-identification is by basename(argv[0])); sys.executable supplies the
interpreter so the test needs no specific venv. Integration tests run with a
HERMETIC PATH (/nonexistent) so the local-first scan is deterministic: no
real binary is ever found in these tests.
"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SHIM = Path(__file__).resolve().parent.parent / "engine" / "ffshim" / "ffshim.py"

pytest.importorskip("websockets")  # the shim's only dependency, and ours here

sys.path.insert(0, str(SHIM.parent))
import ffshim  # noqa: E402


# ---------------------------------------------------------------------------
# TailDrop — the honest-failure invariant
# ---------------------------------------------------------------------------

def _run_taildrop(writes, frame_bytes):
    """Drive a TailDrop over an os.pipe and return (bytes_read, td, dropped)."""
    r, w = os.pipe()
    td = ffshim.TailDrop(w, frame_bytes)
    for chunk in writes:
        td.write(chunk)
    dropped = td.finish()
    os.close(w)
    out = b""
    while True:
        b = os.read(r, 65536)
        if not b:
            break
        out += b
    os.close(r)
    return out, td, dropped


def test_taildrop_passes_whole_frames_only():
    fb = 100
    data = os.urandom(10 * fb + 37)  # 10 whole frames + 37-byte partial
    out, td, dropped = _run_taildrop([data], fb)
    assert len(out) == 10 * fb
    assert td.frames == 10
    assert dropped == 37
    assert out == data[: 10 * fb]  # byte-exact, in order


def test_taildrop_handles_frames_split_across_write_boundaries():
    fb = 1000
    data = os.urandom(3 * fb + 7)
    # write in awkward 137-byte slices — frames cross many write boundaries
    slices = [data[i:i + 137] for i in range(0, len(data), 137)]
    out, td, dropped = _run_taildrop(slices, fb)
    assert len(out) == 3 * fb
    assert td.frames == 3
    assert dropped == 7
    assert out == data[: 3 * fb]


def test_taildrop_empty_stream_emits_zero_bytes():
    out, td, dropped = _run_taildrop([], 100)
    assert out == b""
    assert td.frames == 0
    assert dropped == 0


def test_taildrop_single_partial_frame_emits_zero_bytes():
    """The pre-first-frame failure case: < frame_bytes total => ZERO bytes out."""
    out, td, dropped = _run_taildrop([b"xyz"], 100)
    assert out == b""
    assert dropped == 3


# ---------------------------------------------------------------------------
# operand locator
# ---------------------------------------------------------------------------

def test_find_input_operand_last_i_wins():
    argv = ["-f", "rawvideo", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-i", "cache:pipe:0", "pipe:1"]
    assert ffshim._find_input_operand(argv) == 9


def test_find_input_operand_absent():
    assert ffshim._find_input_operand(["-f", "rawvideo"]) is None
    assert ffshim._find_input_operand(["-i"]) is None  # no operand after -i


def test_find_input_operand_flag_parameterised():
    # The input flag is per-install (a custom tool's convention), not fixed.
    #
    # NOTE: the locator is NO LONGER FLAG-ONLY. When the flag is
    # absent it falls back to a trailing SERVABLE_INPUT_OPERANDS token, because
    # one install of this file serves binaries with DIFFERENT argv conventions
    # and ffprobe's input is positional. A flag-only description of this
    # locator would therefore be stale: the tests below pin both the flagged
    # and the bare-positional shapes.
    #
    # So "the flag is honoured" is now proved with an operand the fallback
    # CANNOT claim, rather than merely by the flag's absence.
    assert ffshim._find_input_operand(["--in", "pipe:0", "--out", "pipe:1"],
                                      "--in") == 1
    # Flag absent AND nothing servable -> still None. This REPLACES the old
    # ["--in", "pipe:0"] case (which asserted None) and is strictly stronger:
    # it proves the flag is not being ignored AND that the fallback does not
    # grab an arbitrary trailing token.
    assert ffshim._find_input_operand(["--in", "somefile.mp4"], "-i") is None
    # WIDENED. Kept visible here on purpose instead of being hidden by a
    # deletion -- flag absent but a servable operand present now RESOLVES.
    # This is exactly ffprobe's real command-line shape, and asserting the old
    # `is None` here is what made the ffprobe install permanently inert.
    assert ffshim._find_input_operand(["--in", "pipe:0"], "-i") == 1
    assert ffshim._find_input_operand(["-i"], "-i") is None  # no operand after


def test_install_config_loader(tmp_path, monkeypatch):
    cfgdir = tmp_path / "cfg"
    cfgdir.mkdir()
    (cfgdir / "mytool.env").write_text(
        "# comment line\n"
        "PROBE=0\n"
        "INPUT_FLAG=--in\n"
        "FRAME_ALIGNED=0\n"
        "\n"
        "GARBAGE-LINE-NO-EQUALS\n")
    monkeypatch.setenv("FFSHIM_CONFIG_DIR", str(cfgdir))
    cfg = ffshim._load_install_config("mytool")
    assert cfg == {"PROBE": "0", "INPUT_FLAG": "--in", "FRAME_ALIGNED": "0"}
    # no config file for this name: defaults apply (empty cfg)
    assert ffshim._load_install_config("nosuchtool") == {}


# ---------------------------------------------------------------------------
# integration: shim (subprocess) vs a scripted fake endpoint
# ---------------------------------------------------------------------------

class FakeEndpoint:
    """A websockets server implementing the contract with scripted behavior.
    mode:
      echo   : meta, ready, echo the staged input back in 70KB S2C chunks,
               exit 0
      die    : meta, ready, echo echo_bytes, then close 1011 early (target
               death: no exit frame)
      refuse : send an error frame (code 2) before ready, close 1011
    """

    def __init__(self, mode, echo_bytes=None, meta=None):
        self.mode = mode
        self.echo_bytes = echo_bytes or 0
        self.meta = meta or {"type": "meta", "width": 10, "height": 10,
                             "frame_bytes": 300, "nb_frames": 99,
                             "duration": 0.5, "run_id": "?"}
        self.saw = {"start": None, "input": b"", "stdin_eof": False, "argv": None}
        self.port = None
        self._server = None
        self._loop = None
        self._thread = None

    async def _handler(self, ws):
        start = json.loads(await ws.recv())
        self.saw["start"] = start
        self.saw["argv"] = start.get("argv")
        if self.mode == "refuse":
            await ws.send(json.dumps({"type": "error", "code": 2,
                                      "stderr": "fake refusal",
                                      "run_id": start.get("run_id")}))
            await ws.close(code=1011)
            return
        while True:
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                self.saw["input"] += bytes(msg)
            else:
                if json.loads(msg).get("type") == "stdin_eof":
                    self.saw["stdin_eof"] = True
                    break
        meta = dict(self.meta)
        meta["run_id"] = start.get("run_id")
        await ws.send(json.dumps(meta))
        await ws.send(json.dumps({"type": "ready", "run_id": start.get("run_id")}))
        if self.mode == "die":
            await ws.send(self.saw["input"][: self.echo_bytes])
            await asyncio.sleep(0.1)
            await ws.close(code=1011)  # early close: no exit frame = target death
            return
        staged = self.saw["input"]
        step = 70000  # force multiple S2C frames
        for i in range(0, len(staged), step):
            await ws.send(staged[i:i + step])  # websockets asyncio API: send()
        await ws.send(json.dumps({"type": "exit", "code": 0, "signal": None,
                                  "stderr": "", "run_id": start.get("run_id")}))
        await ws.close(code=1000)

    def start(self):
        self._loop = asyncio.new_event_loop()
        self._error = None
        from websockets.asyncio.server import serve

        def _run():
            try:
                asyncio.set_event_loop(self._loop)

                async def _start():
                    # serve() must be AWAITED (create_task on a not-yet-running
                    # loop raises "no running event loop" on 3.10+).
                    return await serve(self._handler, "127.0.0.1", 0)

                server = self._loop.run_until_complete(_start())
                self._server = server
                self.port = server.sockets[0].getsockname()[1]
                self._loop.run_forever()
            except BaseException as e:  # noqa: BLE001 — surface it in the test thread
                self._error = e

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        for _ in range(100):
            if self.port:
                return self
            time.sleep(0.05)
        raise RuntimeError(
            f"fake endpoint did not start: {self._error!r}")

    def stop(self):
        if self._loop and self._thread:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            try:
                self._loop.close()
            except Exception:
                pass


def _install_shim_as_ffmpeg(tmp_path: Path) -> Path:
    """The shim self-identifies by basename(argv[0]) — copy it to <tmp>/ffmpeg
    so the subprocess presents the name the engine's PATH resolution would."""
    dst = tmp_path / "ffmpeg"
    dst.write_bytes(SHIM.read_bytes())
    dst.chmod(0o755)
    return dst


def _run_shim(shim_path: Path, env_extra, stdin_bytes, argv, timeout=30):
    env = dict(os.environ)
    # HERMETIC PATH unless the test provides one: the local-first scan must
    # find nothing in the network-path tests (deterministic, no real binary
    # interference from whatever image the suite happens to run in).
    env["PATH"] = "/nonexistent-hermetic"
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(shim_path), *argv],
        input=stdin_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=timeout,
    )


FFMPEG_ARGV = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-i", "pipe:0",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]


def test_raw_script_invocation_refused(tmp_path):
    """Running `python ffshim.py`
    has NO binary identity — the shim must refuse locally and loudly (exit
    127, zero stdout) rather than sending its own script name to the
    network, where the endpoint would 127 it as 'unmappable binary
    ffshim.py'. The installed-name contract (the file is named after the
    binary it serves) is how the name travels."""
    shim_src = Path(__file__).parent.parent / "engine" / "ffshim" / "ffshim.py"
    assert shim_src.exists()
    p = _run_shim(shim_src,
                  {"FFSHIM_TMPDIR": str(tmp_path / "staged")},
                  b"", FFMPEG_ARGV)
    assert p.returncode == 127, p.stderr.decode()[-2000:]
    assert p.stdout == b""  # zero stdout, always
    err = p.stderr.decode()
    assert "ffshim" in err and "install" in err.lower()


def test_integration_echo_byte_exact(tmp_path):
    """Full chain: engine-stdin -> shim -> fake endpoint -> back. The fake
    echoes the staged input; the shim must emit EXACTLY the complete-frame
    prefix of it (frame_bytes=300: 100_003 bytes -> 333 frames = 99_900 bytes,
    103-byte tail dropped)."""
    fake = FakeEndpoint("echo").start()
    shim = _install_shim_as_ffmpeg(tmp_path)
    payload = os.urandom(100_003)
    try:
        p = _run_shim(shim,
                      {"FFSHIM_PORT": str(fake.port),
                       "FFSHIM_TMPDIR": str(tmp_path / "staged"),
                       "FFSHIM_DEBUG": "1"},
                      payload, FFMPEG_ARGV)
    finally:
        fake.stop()
    assert p.returncode == 0, p.stderr.decode()[-2000:]
    assert p.stdout == payload[: 333 * 300]  # byte-exact clean-multiple prefix
    assert len(p.stdout) % 300 == 0
    # the fake saw the protocol: start (argv verbatim), full input, stdin_eof
    assert fake.saw["argv"] == FFMPEG_ARGV
    assert fake.saw["input"] == payload
    assert fake.saw["stdin_eof"] is True
    assert b"tail-drop: discarded 103 partial-trailing bytes" in p.stderr


def test_integration_mid_stream_death_clean_eof(tmp_path):
    """The death arm: endpoint closes early after 350 bytes of a 300-byte-frame
    stream -> the shim must emit exactly 1 frame (300 bytes), exit 1, and log
    the connection loss. ZERO partial bytes on stdout — the invariant."""
    fake = FakeEndpoint("die", echo_bytes=350).start()
    shim = _install_shim_as_ffmpeg(tmp_path)
    try:
        p = _run_shim(shim,
                      {"FFSHIM_PORT": str(fake.port),
                       "FFSHIM_TMPDIR": str(tmp_path / "staged")},
                      b"V" * 5000, FFMPEG_ARGV)
    finally:
        fake.stop()
    assert p.returncode == 1, p.stderr.decode()[-2000:]
    assert p.stdout == b"V" * 300  # exactly one whole frame, then clean EOF
    assert b"connection lost mid-stream" in p.stderr


def test_integration_remote_error_code_propagates(tmp_path):
    fake = FakeEndpoint("refuse").start()
    shim = _install_shim_as_ffmpeg(tmp_path)
    try:
        p = _run_shim(shim,
                      {"FFSHIM_PORT": str(fake.port),
                       "FFSHIM_TMPDIR": str(tmp_path / "staged")},
                      b"V" * 1000, FFMPEG_ARGV)
    finally:
        fake.stop()
    assert p.returncode == 2  # the endpoint's error code, propagated
    assert p.stdout == b""  # pre-first-frame failure: ZERO bytes
    assert b"fake refusal" in p.stderr


def test_path_input_fails_before_any_network(tmp_path):
    """Path input must die BEFORE the network (no stdin is ever fed for path
    input; the file cannot exist remotely). A closed port proves the
    fail-fast: no connect attempt, immediate exit 1, loud stderr, zero stdout."""
    shim = _install_shim_as_ffmpeg(tmp_path)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()
    argv = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-i", "/nonexistent/in.mp4",
            "-f", "rawvideo", "pipe:1"]
    p = _run_shim(shim,
                  {"FFSHIM_PORT": str(dead_port),
                   "FFSHIM_TMPDIR": str(tmp_path / "staged"),
                   "FFSHIM_CONNECT_TIMEOUT_S": "2"},
                  b"", argv, timeout=10)
    assert p.returncode == 1
    assert p.stdout == b""
    assert b"cannot serve filesystem input" in p.stderr
    assert b"/nonexistent/in.mp4" in p.stderr


def test_local_first_execs_real_binary_no_network(tmp_path):
    """If the REAL binary is available locally, the shim
    execv's it — zero hops, zero protocol, zero dependency on plugin config.
    Proven against a DEAD port: if the shim had routed, it would have failed
    with 'cannot reach'; instead the real binary's marker lands on stdout."""
    realbin = tmp_path / "realbin"
    shimbin = tmp_path / "shimbin"
    realbin.mkdir()
    shimbin.mkdir()
    real = realbin / "ffmpeg"
    real.write_text(f"#!{sys.executable}\n"
                    "import sys\n"
                    "sys.stdout.buffer.write(b'LOCAL-FFMPEG-MARKER')\n")
    real.chmod(0o755)
    shim = shimbin / "ffmpeg"  # the shim, installed FIRST in PATH
    shim.write_bytes(SHIM.read_bytes())
    shim.chmod(0o755)

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()

    p = _run_shim(shim,
                  {"PATH": f"{shimbin}{os.pathsep}{realbin}",
                   "FFSHIM_PORT": str(dead_port),
                   "FFSHIM_CONNECT_TIMEOUT_S": "2"},
                  b"", ["-version"], timeout=10)
    assert p.returncode == 0, p.stderr.decode()[-2000:]
    assert p.stdout == b"LOCAL-FFMPEG-MARKER"
    assert b"local-first: real 'ffmpeg' found" in p.stderr
    assert b"cannot reach turbohaul manager" not in p.stderr  # no routing at all


def test_generic_name_and_input_flag_from_install_config(tmp_path):
    """The binary is a PARAMETER: the same shim file, installed as 'mytool'
    with its own install config (INPUT_FLAG=--in), accepts a non-ffmpeg name
    and a non--i input convention, and proceeds all the way to the network
    step — failing ONLY there (dead port), which is exactly where it must
    fail when no manager is up."""
    shim = tmp_path / "mytool"
    shim.write_bytes(SHIM.read_bytes())
    shim.chmod(0o755)
    cfgdir = tmp_path / "cfg"
    cfgdir.mkdir()
    (cfgdir / "mytool.env").write_text("PROBE=0\nINPUT_FLAG=--in\n"
                                       "FRAME_ALIGNED=0\n")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()

    p = _run_shim(shim,
                  {"FFSHIM_PORT": str(dead_port),
                   "FFSHIM_TMPDIR": str(tmp_path / "staged"),
                   "FFSHIM_CONFIG_DIR": str(cfgdir),
                   "FFSHIM_CONNECT_TIMEOUT_S": "2"},
                  b"junk-data",
                  ["--in", "pipe:0", "--out", "pipe:1"], timeout=10)
    assert p.returncode == 1, p.stderr.decode()[-2000:]
    assert p.stdout == b""
    # reached the connect step: name accepted, --in flag honored (else it
    # would have died earlier with 'no input operand')
    assert b"cannot reach turbohaul manager" in p.stderr
    assert b"no input operand" not in p.stderr


# ---------------------------------------------------------------------------
# ffprobe's REAL argv shape
#
# One install of this file serves ffmpeg AND ffprobe (Dockerfile.cuda-multi
# :137-140), but the two binaries do not share a command line:
#     ffmpeg  : ... -i cache:pipe:0 ...              FLAGGED
#     ffprobe : ... -of default=noprint_wrappers=1 pipe:0    BARE POSITIONAL
# `ffprobe [options] INPUT` is ffprobe's real CLI; `-i` is ffmpeg's convention.
# The flag-only locator therefore made the ffprobe install permanently
# unservable -- exit 1, zero bytes, on every call -- while the ffmpeg install
# worked and hid it. Probe runs FIRST, so the working leg was never reached and
# any test aimed at ffmpeg passed forever while video stayed broken.
#
# Behaviour before the fix: exit=1, stdout 0
# bytes, stderr "no input operand after '-i' ...". A control run: the same binary
# with `-i pipe:0` produced a DIFFERENT error, proving the flagged leg was alive
# and the fault was specifically the ARGV SHAPE.
#
# Tests aimed only at the flagged (ffmpeg) leg cannot catch this: the ffprobe
# shape ('show_entries', 'select_streams', 'noprint_wrappers') needs its own
# coverage, which the tests below provide.
# ---------------------------------------------------------------------------

# The engine's real ffprobe invocation shape: options, then a bare positional
# input, and no '-i' anywhere in it.
FFPROBE_ARGV = ["-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "default=noprint_wrappers=1", "pipe:0"]


def _install_shim_as(tmp_path: Path, name: str) -> Path:
    """The shim self-identifies by basename(argv[0]) -- install it under the
    name whose convention is under test (the binary is a PARAMETER)."""
    dst = tmp_path / name
    dst.write_bytes(SHIM.read_bytes())
    dst.chmod(0o755)
    return dst


def test_find_input_operand_bare_positional_ffprobe_shape():
    """ffprobe passes the input as a trailing POSITIONAL with no '-i'.

    Flag-only lookup returned None here, which main() turned into exit 1 with
    zero stdout -- the whole video feature, inert.
    """
    assert "-i" not in FFPROBE_ARGV, "fixture must have no flag at all"
    idx = ffshim._find_input_operand(FFPROBE_ARGV)
    assert idx is not None, (
        "ffprobe's real argv resolved to no input operand -- this is the argv-shape defect: the "
        f"ffprobe install exits 1 with zero bytes on every call: {FFPROBE_ARGV}")
    assert FFPROBE_ARGV[idx] == "pipe:0", FFPROBE_ARGV[idx]


def test_positional_fallback_never_takes_an_ffmpeg_output_operand():
    """THE REGRESSION GUARD for the fallback's one real hazard.

    An ffmpeg command line ends in its OUTPUT ('pipe:1'). A fallback that took
    "the trailing bare positional" would grab that output and try to serve it
    as input. The fallback is restricted to SERVABLE_INPUT_OPERANDS, so it
    cannot -- and the flagged leg wins here regardless, since ffmpeg passes -i.

    ⛔ This test CANNOT fail on the pre-fix code: with '-i' present the flag
    path returns first either way. It is a guard against a WRONG VERSION OF THE
    FIX, namely a fallback that takes any trailing token rather than only a
    servable input operand.

    """
    idx = ffshim._find_input_operand(FFMPEG_ARGV)
    assert FFMPEG_ARGV[idx] == "pipe:0", (
        "the input operand must be the -i operand, never the trailing output: "
        f"got {FFMPEG_ARGV[idx]!r} from {FFMPEG_ARGV}")
    assert FFMPEG_ARGV[-1] == "pipe:1", "fixture must end in an OUTPUT operand"
    assert idx != len(FFMPEG_ARGV) - 1, (
        "resolved to the LAST token -- that is ffmpeg's OUTPUT being served as "
        "input, which is exactly what restricting the fallback prevents")

    # And with the flag absent, the fallback must still refuse an output-only
    # command line rather than inventing an input from it.
    assert ffshim._find_input_operand(["-f", "rawvideo", "pipe:1"]) is None, (
        "a command line whose only operand is an OUTPUT must resolve to no "
        "input, not to the output")


def test_ffprobe_real_argv_is_accepted_not_rejected_on_shape(tmp_path):
    """END-TO-END at the same layer as the other fail-fast tests.

    Before the fix this died at the argv-shape gate: exit 1, zero stdout, stderr
    "no input operand after '-i'". After it, the shape is ACCEPTED and the run
    proceeds to the next real check -- here the empty-stdin one, because the
    test feeds no bytes. Asserting the shape gate is passed (rather than a full
    success) keeps this honest: no endpoint exists in this test.
    """
    shim = _install_shim_as(tmp_path, "ffprobe")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()
    p = _run_shim(shim,
                  {"FFSHIM_PORT": str(dead_port),
                   "FFSHIM_TMPDIR": str(tmp_path / "staged"),
                   "FFSHIM_CONNECT_TIMEOUT_S": "2"},
                  b"", FFPROBE_ARGV, timeout=10)

    assert b"no input operand" not in p.stderr, (
        "ffprobe's real argv was rejected on SHAPE -- the argv-shape defect: the "
        f"install is inert. stderr: {p.stderr[:400]!r}")
    # POSITIVE CONTROL: it must have gone PAST the shape gate and reached a
    # later, different check -- otherwise "no rejection message" could just
    # mean the process never got that far.
    assert b"stdin was empty" in p.stderr, (
        "expected the run to reach the empty-stdin check, proving the argv "
        f"shape was accepted and the run proceeded. stderr: {p.stderr[:400]!r}")
