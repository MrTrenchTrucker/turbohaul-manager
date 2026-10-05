#!/usr/bin/env python3
"""ffshim — the engine's generic PATH stand-in for a local binary.

ONE implementation, installed under whatever executable name it serves.
The turbohaul engine image ships WITHOUT ffmpeg; the engine's mtmd video path
forks `ffmpeg`/`ffprobe` from PATH (mtmd-helper.cpp:921-924). This file is
installed as /usr/local/bin/ffmpeg AND /usr/local/bin/ffprobe (self-identifies
by basename(argv[0])) — ffmpeg and ffprobe are two installs of this one
generic shim (design rule: the binary is a PARAMETER, the
mechanism serves ANY plugin, not just ffmpeg).

LOCAL-FIRST: if the REAL binary is available locally
(anywhere on PATH, not this shim), the shim execv's it and routes NOWHERE —
an operator who bakes ffmpeg into the image must not pay a network hop nor
depend on any plugin configuration. Only when no local binary exists does the
shim forward the invocation over two WebSocket hops:

  shim --WS hop1--> turbohaul manager  /api/plugins/$FFSHIM_PLUGIN_TAG/exec
       --WS hop2--> plugin container   /ws/exec  (runs an allow-listed binary)

Per-install semantics (how one implementation serves many binaries) live in a
config file: /etc/ffshim/<name>.env (FFSHIM_CONFIG_DIR override), KEY=VALUE:
  PROBE=1|0        probe_first default (1). Non-video installs: 0.
  INPUT_FLAG=-i    the argv flag whose following operand is the input
                   (default -i; the engine's convention).
  FRAME_ALIGNED=1|0  apply the tail-drop clean-multiple invariant when the
                   endpoint reports frame_bytes>0 (default 1; 0 = raw
                   write-through for non-frame streams).
Env overrides for testing / one-offs: FFSHIM_PROBE, FFSHIM_INPUT_FLAG,
FFSHIM_FRAME_ALIGNED, plus FFSHIM_PORT (11401), FFSHIM_PLUGIN_TAG
(ffmpeg-video), FFSHIM_TMPDIR (/tmp), FFSHIM_NO_PROGRESS_S (300),
FFSHIM_CONNECT_TIMEOUT_S (5), FFSHIM_AUTH_TOKEN (default empty; Bearer when
set — the loopback hop is unauthenticated at present; if the manager's auth
ever gates the route, the token comes from here, never hardcoded),
FFSHIM_DEBUG (1 = progress trace on stderr).

THE ENGINE'S CONTRACT (verified against mtmd-helper.cpp):
- the engine NEVER reads the child's exit code — stdout EOF is its only death
  signal; read_next_frame treats a partial frame byte-identical to clean EOF
  (LOG_DBG only). Therefore the shim's stdout is the product and:
  * when frame alignment applies (FRAME_ALIGNED + endpoint-reported
    frame_bytes>0) stdout is ALWAYS a clean multiple of frame_bytes — the
    TAIL-DROP invariant (the honest-failure bar): a pre-first-frame
    failure emits ZERO bytes; a mid-stream death emits N whole frames + clean
    EOF, never a partial frame;
  * every failure is loud on stderr (the engine's stderr = the turbohaul log);
  * the engine's blindness to mid-stream death is a documented, accepted risk
    (a separate nb_frames guard would be needed), not something this shim
    pretends to close.
- the probe step (mtmd-helper.cpp:679-757) runs ffprobe with input "pipe:0"
  (PLAIN pipe); the decode step (:759-800) uses "cache:pipe:0" (seekable,
  moov-at-end). Both arrive here as stdin + the same argv shape; the operand
  at the input position is what the remote side replaces with its staged file.
- PATH input ("-i /some/local/path") is UNSERVABLE REMOTELY: the engine feeds
  no stdin for path input and the file does not exist in the remote container.
  The shim fails IMMEDIATELY (before touching the network) with a loud stderr.
  (LOCAL-FIRST makes this a non-issue when the binary is actually local.)

Protocol: a frozen frame protocol. Identical frames on both hops; the
manager route is a pure forwarder. websockets (v17 in the engine venv) sync
client — the shim is a dedicated single-threaded child process.

Timing design (why there is no watchdog thread): the websockets SYNC client is
not thread-safe, so a separate thread calling ws.close() while the main thread
sits in recv() is a data race. Instead every recv carries its own
timeout=no_progress_s (default 300): a wedged forwarder makes the next recv raise
TimeoutError -> loud exit 1 -> the pipe to the engine breaks -> the engine's
fread unblocks. connect uses open_timeout (default 5s). The one unbounded call
is send() during the input upload — a blocked send means a dead network, and
it is then bounded by the kernel's TCP retransmit timeout, not by us; the
upload is a fast container-bridge transfer anyway (typically tens of MB/s).
"""
import json
import os
import secrets
import sys
import time

S2C_C = 256 * 1024  # C2S input chunk (input phase)


def _e(msg: str) -> None:
    """stderr is the ONLY log channel (the engine's stderr = the turbohaul log).
    Never, ever write to stdout — stdout is the byte product."""
    sys.stderr.write("ffshim: " + msg + "\n")
    sys.stderr.flush()


def _load_install_config(name: str) -> dict:
    """Per-install semantics from /etc/ffshim/<name>.env (KEY=VALUE lines).
    Missing file = defaults (the common case: the engine image, where the
    install is purely 'route to the plugin')."""
    cfg = {}
    d = os.environ.get("FFSHIM_CONFIG_DIR", "/etc/ffshim")
    p = os.path.join(d, name + ".env")
    if os.path.isfile(p):
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip()
        except OSError as e:
            _e(f"WARN: cannot read install config {p}: {e}")
    return cfg


def _find_local_binary(name: str):
    """LOCAL-FIRST: scan PATH for an executable with this
    name that is NOT this shim. Found -> the real binary exists locally ->
    execv it (zero hops, zero protocol). None -> route over the network.
    The realpath comparison excludes the shim itself (however it was
    invoked); a separately-installed copy at another path is indistinguishable
    from a real binary and is treated as one (documented edge)."""
    self_path = None
    try:
        self_path = os.path.realpath(sys.argv[0])
    except (OSError, TypeError):
        pass
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d or d.startswith("/proc/"):
            continue
        cand = os.path.join(d, name)
        try:
            if not (os.path.isfile(cand) and os.access(cand, os.X_OK)):
                continue
            rp = os.path.realpath(cand)
            if self_path is not None and rp == self_path:
                continue  # that's us (or a hardlink of us)
            return cand
        except OSError:
            continue
    return None


# The only operands this shim can serve: the engine's stdin pipe. Everything
# else (a filesystem path, a URL, or an OUTPUT pipe such as 'pipe:1') is
# unservable remotely and is refused in main(). Named here rather than inlined
# because _find_input_operand's positional fallback below must scan for exactly
# this set -- one definition, two users, no drift.
SERVABLE_INPUT_OPERANDS = ("pipe:0", "cache:pipe:0")


def _find_input_operand(argv, flag: str = "-i"):
    """Index of this invocation's input operand, or None.

    TWO SHAPES, because ONE install of this file serves binaries with DIFFERENT
    command-line conventions:
      1. FLAGGED  -- the operand after the LAST occurrence of `flag` (default
         '-i'; per-install override via INPUT_FLAG). ffmpeg's convention.
      2. POSITIONAL -- if the flag is absent entirely, the LAST argv token that
         is one of SERVABLE_INPUT_OPERANDS. ffprobe's real CLI is
         `ffprobe [options] INPUT` with no '-i' anywhere, so a flag-only lookup
         made the ffprobe install permanently unservable: it exited 1 with zero
         bytes on every call, while the ffmpeg install worked and hid it.

    WHY BACKWARDS, stated rather than inherited: for the flagged shape the LAST
    '-i' is the effective one (a later flag overrides an earlier), and for the
    positional shape the input is the trailing operand. Both conventions put
    the answer at the END, so a reverse scan is correct for both -- not merely
    carried over from the original flag-only implementation.

    WHY THE FALLBACK IS RESTRICTED TO SERVABLE_INPUT_OPERANDS rather than
    taking any trailing bare token: an ffmpeg invocation's trailing operand is
    its OUTPUT ('pipe:1', a path, a URL). Matching only the stdin-pipe operands
    makes "mistake an output for an input" impossible by construction instead
    of relying on main()'s later refusal to catch it. The fallback is also
    dormant for ffmpeg, which always passes '-i'.
    """
    for i in range(len(argv) - 1, -1, -1):
        if argv[i] == flag and i + 1 < len(argv):
            return i + 1
    for i in range(len(argv) - 1, -1, -1):
        if argv[i] in SERVABLE_INPUT_OPERANDS:
            return i
    return None


class TailDrop:
    """stdout wrapper enforcing the honest-failure invariant: everything
    written is a whole number of frames. Holds back a trailing < frame_bytes
    remainder; finish() DISCARDS it (the partial frame is dropped, counted,
    logged). Applied only when the install declares FRAME_ALIGNED and the
    endpoint actually reports frame_bytes>0 — alignment is a SEMANTICS, not a
    name: text outputs (ffprobe) and non-frame streams write through."""

    def __init__(self, fd: int, frame_bytes: int):
        self.fd = fd
        self.fb = frame_bytes
        self.tail = b""
        self.frames = 0
        self.bytes_out = 0

    def write(self, data: bytes) -> int:
        buf = self.tail + data
        if self.fb > 0:
            n = len(buf) // self.fb
            if n:
                head = buf[: n * self.fb]
                self._flush_all(head)
                self.frames += n
                self.bytes_out += len(head)
                buf = buf[n * self.fb:]
        self.tail = buf
        return len(data)

    def _flush_all(self, data: bytes) -> None:
        while data:
            w = os.write(self.fd, data)
            data = data[w:]

    def finish(self) -> int:
        """Discard the partial tail; returns the number of bytes dropped."""
        dropped = len(self.tail)
        self.tail = b""
        return dropped


def _write_all(fd: int, data: bytes) -> None:
    while data:
        w = os.write(fd, data)
        data = data[w:]


def main() -> int:
    t_start = time.monotonic()
    argv0 = os.path.basename(sys.argv[0]) if sys.argv else ""
    if argv0.endswith(".py"):
        argv0 = argv0[:-3]  # dev invocations like `python ffshim.py`
    if not argv0:
        _e("no argv[0]: cannot determine the executable name, refusing")
        return 127
    if argv0 == "ffshim":
        _e("refusing: invoked as the canonical script name 'ffshim' — a raw "
           "`python ffshim.py` has no binary identity (the name IS "
           "the binary parameter). Install the shim under the executable name "
           "it serves (e.g. /usr/local/bin/ffmpeg); the engine resolves it "
           "from PATH and the name carries the contract")
        return 127
    binary = argv0  # the binary is a PARAMETER
    argv = sys.argv[1:]

    # ---- LOCAL-FIRST: real binary available locally -> execv, no routing.
    # This runs BEFORE env parsing/staging/network: a locally-baked binary
    # serves the invocation natively (including path inputs), paying zero
    # protocol overhead.
    local = _find_local_binary(binary)
    if local:
        _e(f"local-first: real {binary!r} found at {local} — execv, no routing "
            "(operator-baked binary wins over the plugin route)")
        os.execvpe(local, [local] + argv, os.environ)  # never returns on success
        # execvpe returns only on failure:
        _e(f"local-first execv of {local} failed — continuing to network path")

    cfg = _load_install_config(binary)
    port = os.environ.get("FFSHIM_PORT", "11401")
    tag = os.environ.get("FFSHIM_PLUGIN_TAG", "ffmpeg-video")
    tmpdir = os.environ.get("FFSHIM_TMPDIR", "/tmp")
    probe_first = os.environ.get(
        "FFSHIM_PROBE", cfg.get("PROBE", "1")) != "0"
    input_flag = os.environ.get("FFSHIM_INPUT_FLAG", cfg.get("INPUT_FLAG", "-i"))
    frame_aligned = os.environ.get(
        "FFSHIM_FRAME_ALIGNED", cfg.get("FRAME_ALIGNED", "1")) != "0"
    no_progress_s = float(os.environ.get("FFSHIM_NO_PROGRESS_S", "300"))
    connect_timeout = float(os.environ.get("FFSHIM_CONNECT_TIMEOUT_S", "5"))
    auth_token = os.environ.get("FFSHIM_AUTH_TOKEN", "")
    debug = os.environ.get("FFSHIM_DEBUG", "") == "1"

    run_id = secrets.token_hex(8)

    # ---- input operand: the value the remote side will replace with its
    # staged file. Must be a stdin-pipe operand; a filesystem path is
    # unservable remotely (engine feeds no stdin for path inputs; the file
    # cannot exist in the remote container). Immediate fail-fast, zero bytes.
    input_idx = _find_input_operand(argv, input_flag)
    if input_idx is None:
        # This message must name what was ACTUALLY searched for. Naming only
        # the flag ("no input operand after '-i'") would present the flag as
        # the whole cause -- and once the positional fallback exists that is a
        # confidently wrong diagnosis pointing the reader at the flag when the
        # flag is no longer the only thing tried. Naming a narrow cause for a
        # wider search is a misleading diagnosis, so the message lists
        # everything that was tried.
        _e(f"no input operand in argv {argv!r}: found neither an operand after "
           f"{input_flag!r} nor a trailing "
           f"{'/'.join(SERVABLE_INPUT_OPERANDS)} operand; nothing to serve, exiting")
        return 1
    operand = argv[input_idx]
    if operand not in SERVABLE_INPUT_OPERANDS:
        _e(f"cannot serve filesystem input {operand!r}: remote exec only "
           f"serves stdin inputs (pipe:0 / cache:pipe:0) and the engine "
           f"feeds no stdin for path inputs. If this engine was given a file "
           f"path for input, that path is not reachable from the plugin "
           f"container (and no local {binary!r} was found to run it with).")
        return 1

    # ---- stage the engine's stdin to a temp file (the feeder does ONE fwrite
    # of the whole buffer then closes — mirror the engine's own behavior).
    # No websockets needed before this point: a missing dependency with nothing
    # to send is a staging error, not an import error.
    # makedirs: FFSHIM_TMPDIR may point at a path the operator pre-created or
    # not; creating it is harmless (it is our temp home) and beats a
    # FileNotFoundError on the very first byte.
    try:
        os.makedirs(tmpdir, exist_ok=True)
    except OSError as e:
        _e(f"cannot prepare FFSHIM_TMPDIR {tmpdir!r}: {e}")
        return 1
    tmp_path = os.path.join(tmpdir, f"ffshim-{run_id}-{os.getpid()}.tmp")
    staged = 0
    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = sys.stdin.buffer.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                staged += len(chunk)
        if debug:
            _e(f"staged {staged} bytes input -> {tmp_path}")
        if staged == 0:
            _e("stdin was empty (engine fed no input for a pipe:0 operand): "
               "nothing to decode, exiting")
            return 1
    except Exception as e:
        _e(f"staging stdin failed: {type(e).__name__}: {e}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return 1

    try:
        # NOTE: `import websockets` does NOT expose the sync subpackage as a
        # top-level attribute — websockets.sync.client must be imported
        # explicitly or connect() raises AttributeError at runtime.
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect as ws_sync_connect
    except ImportError as e:
        _e(f"ffshim dependency missing: {e} — the engine venv must carry the "
           f"'websockets' package (the build-time import check should have "
           f"caught this)")
        os.unlink(tmp_path)
        return 127

    try:
        # ---- connect (hop 1: the manager's own loopback port)
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
        ws_url = f"ws://127.0.0.1:{port}/api/plugins/{tag}/exec"
        try:
            ws = ws_sync_connect(
                ws_url, open_timeout=connect_timeout, additional_headers=headers)
        except Exception as e:
            _e(f"cannot reach turbohaul manager at {ws_url}: {type(e).__name__}: "
               f"{e} (check FFSHIM_PORT/FFSHIM_PLUGIN_TAG and that the manager "
               f"is up)")
            return 1

        def _send(data: bytes) -> None:
            ws.send(data)

        def _send_json(obj: dict) -> None:
            ws.send(json.dumps(obj))

        def _recv():
            # timeout=no_progress_s is the watchdog: a wedged forwarder raises
            # TimeoutError here -> loud exit -> pipe breaks -> engine unblocks.
            return ws.recv(timeout=no_progress_s)

        # ---- start (the binary is a parameter; input_index tells the endpoint
        # exactly which operand to replace with its staged file — the -i
        # heuristic is the shim's convention, the index is the contract)
        _send_json({"type": "start", "binary": binary, "argv": argv,
                    "probe_first": probe_first, "input_index": input_idx,
                    "run_id": run_id})

        # ---- input phase: stream the staged file C2S, then stdin_eof.
        # Target/forwarder death DURING the upload is the same terminal
        # failure as mid-output death: loud line, non-zero exit, zero stdout
        # (nothing was written yet), and no traceback (honest-failure
        # contract — stderr is a line, not a stack).
        t_up = time.monotonic()
        sent = 0
        try:
            with open(tmp_path, "rb") as f:
                while True:
                    chunk = f.read(S2C_C)
                    if not chunk:
                        break
                    _send(chunk)
                    sent += len(chunk)
            _send_json({"type": "stdin_eof"})
        except ConnectionClosed as e:
            _e("connection lost mid-stream (during input upload): "
               f"{type(e).__name__}: {e} — the target or the forwarder died "
               "before the run could start; zero bytes served (run "
               f"{run_id})")
            return 1
        if debug:
            _e(f"upload done: {sent} bytes in {time.monotonic() - t_up:.2f}s "
               f"(run {run_id})")

        # ---- meta (probe-first) then ready
        frame_bytes = 0
        nb_frames = 0
        while True:
            msg = _recv()
            if isinstance(msg, (bytes, bytearray)):
                continue  # no data frames before ready in this protocol
            m = json.loads(msg)
            mtype = m.get("type")
            if mtype == "meta":
                frame_bytes = int(m.get("frame_bytes") or 0)
                nb_frames = int(m.get("nb_frames") or 0)
                if debug:
                    _e(f"meta: {m} (run {run_id})")
            elif mtype == "ready":
                break
            elif mtype == "error":
                _e(f"remote error before ready: code={m.get('code')} "
                   f"stderr={str(m.get('stderr', ''))[:2000]}")
                return int(m.get("code") or 1)
            else:
                _e(f"unexpected frame before ready: {m!r}")
                return 1

        # Alignment is semantics, not a name: tail-drop ON when the install
        # declares frame alignment AND the endpoint actually reported a frame
        # size. frame_bytes==0 (probe disabled or non-video) = write-through,
        # no alignment claim.
        td = None
        if frame_aligned and frame_bytes > 0:
            td = TailDrop(1, frame_bytes)
        elif frame_bytes > 0:
            _e("meta reports frame alignment but this install is "
               "FRAME_ALIGNED=0 — raw write-through, no invariant")

        # ---- output phase: S2C BINARY -> stdout, until exit/error/close.
        exit_code = None
        err_code = 1
        t_first = None
        last = time.monotonic()
        try:
            while True:
                msg = _recv()
                now = time.monotonic()
                if debug and now - last > 1.0:
                    b = td.bytes_out if td is not None else 0
                    fr = td.frames if td is not None else 0
                    _e(f"progress: {b} bytes / {fr} frames so far (run {run_id})")
                    last = now
                if isinstance(msg, (bytes, bytearray)):
                    if t_first is None:
                        t_first = time.monotonic()
                        if debug:
                            _e(f"first output byte at {t_first - t_start:.2f}s "
                               f"after shim start (run {run_id})")
                    if td is not None:
                        td.write(bytes(msg))
                    else:
                        _write_all(1, bytes(msg))
                    continue
                m = json.loads(msg)
                mtype = m.get("type")
                if mtype == "exit":
                    exit_code = int(m.get("code") or 0)
                    stderr_out = str(m.get("stderr", ""))
                    if stderr_out:
                        _e(f"remote stderr: {stderr_out[:4000]}")
                    break
                if mtype == "error":
                    _e(f"remote error mid-stream: code={m.get('code')} "
                       f"stderr={str(m.get('stderr', ''))[:2000]}")
                    err_code = int(m.get("code") or 1)
                    break
                _e(f"unexpected frame in output phase: {m!r}")
                break
        except (TimeoutError, ConnectionClosed) as e:
            _e(f"connection lost mid-stream before exit frame: "
               f"{type(e).__name__} (target lost — engine sees clean EOF, "
               f"which it cannot distinguish from a short video: documented "
               f"blind spot)")
            err_code = 1

        # ---- finish: discard the partial tail (honest-failure invariant),
        # report the diagnostic, exit with the REMOTE's code on a clean exit
        # frame (the engine ignores it — logs and tests are the consumers).
        dropped = td.finish() if td is not None else 0
        frames_out = td.frames if td is not None else -1
        if dropped:
            _e(f"tail-drop: discarded {dropped} partial-trailing bytes "
               f"(clean-multiple invariant held; stdout = whole frames only)")
        code = exit_code if exit_code is not None else err_code
        _e(f"done: run {run_id} binary={binary} exit={code} "
           f"frames_emitted={frames_out} "
           f"frames_expected={nb_frames if nb_frames else '?'} "
           f"bytes_out={td.bytes_out if td is not None else '?'} "
           f"first_output={f'{t_first - t_start:.2f}s' if t_first else 'never'} "
           f"elapsed={time.monotonic() - t_start:.2f}s")
        if (nb_frames and frames_out >= 0 and frames_out < nb_frames
                and exit_code == 0):
            _e(f"NOTE: remote exited 0 but emitted {frames_out} of {nb_frames} "
               f"expected frames — the engine cannot distinguish this from a "
               f"genuinely shorter video (documented blind spot)")
        return code

    except Exception as e:  # noqa: BLE001 — loud, never silent
        import traceback
        traceback.print_exc(file=sys.stderr)
        _e(f"unexpected: {type(e).__name__}: {e}")
        return 1
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
