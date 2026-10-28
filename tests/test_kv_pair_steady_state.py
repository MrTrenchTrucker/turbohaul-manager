"""A live thread whose saved pair keeps growing must keep saving while the directory sits at the cap.

The save directory is a RAM disk with a byte cap (``kv.ram_cache_max_bytes``). Other threads keep
arriving and fill it up to that cap, and the one live thread saves a pair that grows a few percent
every turn. The only thing that may stop such a save is the cap gate at the top of the save path,
which first lets one cleanup pass free idle pairs of other threads and declines only when the live
thread's own pair holds the directory at or over the cap.

A gate that estimated the size of the incoming pair and compared it with the cap before the save
would decline this thread on its very first growing turn: the other threads' pairs fill the
directory, and the thread's own pair never gets to be bigger than it was. Nothing would ever free
room, so the thread would stay frozen. The first test below catches that freeze. The second test is
the control: a live thread alone in the directory, with its own pair at or over the cap, is
declined by design, so a decline from the cap itself is told apart from a freeze.

The fixtures are copied from the other save-path tests and nothing from the cap or pair modules is
imported, so the file depends only on the manager. One thread, no
processes.
"""
import json
import os
import time

import pytest

import turbohaul.manager as manager_mod
import turbohaul.subprocess_mgr as subprocess_mgr
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
from turbohaul.kv_policy import kv_meta_fn, kv_save_fn
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot

MODEL = "example-27b"
PORT = 59500
LIVE_THREAD = "sess-live-growing"
MSGS = [{"role": "user", "content": "hi"}]

# Sizes are in bytes and small on purpose. A pair is a bin plus a checkpoint sidecar of half its size.
BASE_TOKENS = 1100
BIN_PER_TOKEN = 4  # 1100 tokens -> a 4400 byte bin, a 2200 byte sidecar, a 6600 byte pair
GROWTH = 1.03
IDLE_BIN = 4000
IDLE_PAIR = IDLE_BIN + IDLE_BIN // 2
N_IDLE = 5
CAP = (N_IDLE + 1) * IDLE_PAIR  # 36000: five idle pairs and a live pair of about one idle pair's size fill it
TURNS = 8


def _tokens(turn):
    return round(BASE_TOKENS * GROWTH ** turn)


def _bin_bytes(tokens):
    return tokens * BIN_PER_TOKEN


def _pair_bytes(tokens):
    return _bin_bytes(tokens) + _bin_bytes(tokens) // 2


@pytest.fixture
def mgr(tmp_path):
    root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (root / sub).mkdir(parents=True)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs", manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging", state_db_path=root / "state.sqlite"),
        runtime=RuntimePathsConfig(llama_server_binary=tmp_path / "fake_llama_server", default_port_base=59500),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    return TurbohaulManager(boot, RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig()))


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    monkeypatch.setenv("TURBOHAUL_CKPT_SIDECAR", "1")
    return d


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


class _Engine:
    """Fake engine. GET /slots lists one populated slot at ``tokens``; POST action=save writes a bin
    sized for those tokens and its sidecar, and reports the bytes it wrote."""

    def __init__(self, kv_dir):
        self.kv_dir = kv_dir
        self.tokens = BASE_TOKENS
        self.save_posts = []

    def install(self, monkeypatch):
        outer = self

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                return _Resp([{"id": 0, "n_prompt_tokens": outer.tokens, "thread_id": LIVE_THREAD,
                               "n_context": 4096, "n_remaining": 2048}])

            async def post(self, url, json=None, **kw):
                if "action=save" not in url or not json or "filename" not in json:
                    return _Resp({})
                outer.save_posts.append(url)
                size = _bin_bytes(outer.tokens)
                tmp = os.path.join(str(outer.kv_dir), json["filename"])
                with open(tmp, "wb") as f:
                    f.write(b"k" * size)
                with open(tmp + ".ckpt", "wb") as f:
                    f.write(b"c" * (size // 2))
                return _Resp({"n_saved": outer.tokens, "n_written": size})

        class _FakeHttpx:
            AsyncClient = staticmethod(lambda *a, **k: _Client())
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)


def _write_pair(kv_dir, mgr, tid, bin_size, tokens, mtime=None):
    """Lay down a complete saved pair (bin, sidecar, metadata) the way a finished save leaves it."""
    th = mgr._thread_hash(tid)
    bin_fn = kv_save_fn(MODEL, 0, th, PORT)
    (kv_dir / bin_fn).write_bytes(b"k" * bin_size)
    (kv_dir / (bin_fn + ".ckpt")).write_bytes(b"c" * (bin_size // 2))
    (kv_dir / kv_meta_fn(MODEL, 0, th, PORT)).write_text(json.dumps({
        "thread_hash": th, "model_tag": MODEL, "port": PORT, "slot_id": 0,
        "prompt_tokens": tokens, "bin_bytes": bin_size, "clean_prefix": False}))
    if mtime is not None:
        for name in (bin_fn, bin_fn + ".ckpt"):
            os.utime(kv_dir / name, (mtime, mtime))
    return bin_fn


def _usage(kv_dir):
    """The bytes the cap counts: every bin plus its sidecar."""
    total = 0
    for n in os.listdir(str(kv_dir)):
        if n.endswith(".bin") or n.endswith(".bin.ckpt"):
            total += os.path.getsize(str(kv_dir / n))
    return total


def _live_slot(mgr):
    """The live turn: the active slot of the manager, which the cap treats as never evictable."""
    slot = Slot.new(MODEL, thread_id=LIVE_THREAD, admission_ctx_len=100, client_meta={"messages": MSGS})
    mgr._active_slot = slot
    return slot


async def _save(mgr, slot):
    return await mgr._save_slot_kv(
        PORT, MODEL, slot, thread_id_override=LIVE_THREAD, admission_ctx_len_override=100,
        client_meta_override={"messages": MSGS})


@pytest.mark.asyncio
async def test_growing_live_thread_keeps_saving_with_the_directory_at_the_cap(mgr, kv_dir, monkeypatch):
    mgr.runtime.kv.ram_cache_max_bytes = CAP
    old = time.time() - 10000.0
    arrivals = 0

    def _another_thread_arrives():
        nonlocal arrivals
        _write_pair(kv_dir, mgr, f"sess-idle-{arrivals}", IDLE_BIN, 900, mtime=old + arrivals)
        arrivals += 1

    for _ in range(N_IDLE):
        _another_thread_arrives()
    live_bin = _write_pair(kv_dir, mgr, LIVE_THREAD, _bin_bytes(BASE_TOKENS), BASE_TOKENS)
    assert CAP < _usage(kv_dir) < CAP + IDLE_PAIR, "fixture: the directory must start full, just over the cap"

    engine = _Engine(kv_dir)
    engine.install(monkeypatch)
    slot = _live_slot(mgr)
    previous_size = _bin_bytes(BASE_TOKENS)
    for turn in range(1, TURNS + 1):
        engine.tokens = _tokens(turn)
        before = len(engine.save_posts)
        await _save(mgr, slot)
        expected = _bin_bytes(engine.tokens)
        assert len(engine.save_posts) == before + 1, (
            f"turn {turn}: the growing live thread was FROZEN, its save never reached the engine "
            f"(its pair was {previous_size} bin bytes, would be {expected}; directory usage "
            f"{_usage(kv_dir)} against a cap of {CAP})")
        on_disk = os.path.getsize(str(kv_dir / live_bin))
        assert on_disk == expected, (
            f"turn {turn}: the bin on disk is {on_disk} bytes, expected the new {expected} bytes")
        assert on_disk > previous_size, f"turn {turn}: the bin did not grow ({previous_size} -> {on_disk})"
        assert os.path.getsize(str(kv_dir / (live_bin + ".ckpt"))) == expected // 2, (
            f"turn {turn}: the sidecar on disk is not the new one")
        assert not [n for n in os.listdir(str(kv_dir)) if ".tmp" in n], f"turn {turn}: a temporary file is left"
        previous_size = on_disk
        _another_thread_arrives()  # other threads keep arriving, the directory is full again


@pytest.mark.asyncio
async def test_a_hard_live_thread_alone_at_the_cap_is_still_declined(mgr, kv_dir, monkeypatch):
    # No other thread to free: the live thread's own pair is the whole directory and it is over the cap.
    own_pair = _pair_bytes(BASE_TOKENS)
    mgr.runtime.kv.ram_cache_max_bytes = own_pair - 600
    live_bin = _write_pair(kv_dir, mgr, LIVE_THREAD, _bin_bytes(BASE_TOKENS), BASE_TOKENS)
    assert _usage(kv_dir) == own_pair
    engine = _Engine(kv_dir)
    engine.install(monkeypatch)
    slot = _live_slot(mgr)
    engine.tokens = _tokens(1)
    before = {n: (kv_dir / n).read_bytes() for n in os.listdir(str(kv_dir))}
    await _save(mgr, slot)
    assert engine.save_posts == [], (
        "a live thread whose own pair holds the directory over the cap is declined by design: "
        "the engine save must not be sent")
    assert {n: (kv_dir / n).read_bytes() for n in os.listdir(str(kv_dir))} == before, (
        "a declined save must leave the directory byte-identical")
    assert os.path.getsize(str(kv_dir / live_bin)) == _bin_bytes(BASE_TOKENS)
