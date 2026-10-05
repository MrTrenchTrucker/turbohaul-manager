"""Observability hardening: the classifier's verdict is logged with
its own inputs (so "saved_chain is None" is visible as the CAUSE of a
sub-agent verdict, not inferred by a human cross-referencing two lines), and
the shape/role vocabulary collision is resolved via two separately-labelled
fields (shape_verdict, role) rather than a rename. classify_event /
EVENT_SUB_AGENT / EVENT_COMPRESSION and every other EVENT_* constant's VALUE
is untouched -- verified directly below, because that string is a live
lookup key (EVENT_TO_CLASS -> POLICIES -> save_ok), not a label.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this test file> -v
"""
from __future__ import annotations

import json
import logging

import pytest

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
from turbohaul.kv_policy import _prefix_hash_chain, kv_meta_fn
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot

_QWEN = "qwen3.6-27b"
_PORT = 60000


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


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture(autouse=True)
def _warm_force_on(monkeypatch):
    monkeypatch.setenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", "1")


_SYS = {"role": "system", "content": "system prompt long enough to matter"}
_U1 = {"role": "user", "content": "first user turn"}
_U2 = {"role": "user", "content": "second user turn"}


def _write_clean_bin(kv_dir, model_tag, port, thread_id, sid, chain, prompt_len=40000):
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": 12345,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": True,
    }))
    return bin_fn


def _warm_classify_lines(caplog):
    return [r.message for r in caplog.records if r.message.startswith("WARM_CLASSIFY")]


@pytest.mark.asyncio
class TestClassifyInputsVisible:
    async def test_discriminator_no_anchor_saved_chain_fp_none_visible_as_cause(
        self, mgr, kv_dir, caplog
    ):
        """No clean bin on disk, no VRAM anchor -> 'sub-agent' verdict. The
        classify line must show saved_chain_fp=None as the reason, not just
        the bare verdict string."""
        inc = _prefix_hash_chain([_SYS, _U1])
        slot = Slot.new(model_tag=_QWEN, thread_id="t-noanchor", admission_hash_chain=inc)
        with caplog.at_level(logging.INFO):
            d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, [])
        assert d["event_type"] == "sub-agent"
        assert d["resolved_from"] == "warm-no-clean-bin"

        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "saved_chain_fp=None" in line, line
        assert "shape_verdict=sub-agent" in line, line
        assert "resolved_from=warm-no-clean-bin" in line, line

    async def test_discriminator_classify_line_carries_all_three_chain_fingerprints(
        self, mgr, kv_dir, caplog
    ):
        """The 'normal' path (clean bin found + evaluated): all three inputs
        present and non-null, plus role as its own field."""
        clean_chain = _prefix_hash_chain([_SYS, _U1])
        _write_clean_bin(kv_dir, _QWEN, _PORT, "t-fullclassify", 0, clean_chain)
        warm = _prefix_hash_chain(
            [_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
        inc = _prefix_hash_chain(
            [_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
        slot = Slot.new(model_tag=_QWEN, thread_id="t-fullclassify",
                         admission_hash_chain=inc)
        slot.admission_role = "main"
        with caplog.at_level(logging.INFO):
            d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, warm)
        assert d["event_type"] == "user-message"

        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "saved_chain_fp=" in line and "saved_chain_fp=None" not in line, line
        assert "inc_chain_fp=" in line and "inc_chain_fp=None" not in line, line
        assert "warm_chain_fp=" in line and "warm_chain_fp=None" not in line, line
        assert "shape_verdict=user-message" in line, line
        assert "role=main" in line, line

    async def test_discriminator_shape_verdict_and_role_are_independent_fields(
        self, mgr, kv_dir, caplog
    ):
        """The collision this change exists to prevent: a 'sub-agent' SHAPE verdict
        (saved_chain is None) on a request whose actual ROLE is 'curator' --
        the two must appear as separately-labelled fields, never collapsed
        into one, so a reader cannot mistake the shape for the role."""
        inc = _prefix_hash_chain([_SYS, _U1])
        slot = Slot.new(
            model_tag=_QWEN, thread_id="t-collision", admission_hash_chain=inc,
            client_meta={"is_curator": True},
        )
        with caplog.at_level(logging.INFO):
            d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, [])
        assert d["event_type"] == "sub-agent"  # the SHAPE verdict

        lines = _warm_classify_lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "shape_verdict=sub-agent" in line, line
        assert "role=curator" in line, line
        assert "role=sub-agent" not in line, line

    async def test_good_shape_survives_guard_skip_emits_no_classify_line(
        self, mgr, kv_dir, caplog
    ):
        """Control: guard-skip (no incoming chain at all) never reaches a
        classify call -- proves the WARM_CLASSIFY line is conditioned on the
        classifier actually having run, not emitted unconditionally on
        every decision."""
        slot = Slot.new(model_tag=_QWEN, thread_id="", admission_hash_chain=[])
        with caplog.at_level(logging.INFO):
            d = await mgr._maybe_force_clean_restore(_PORT, _QWEN, slot, [])
        assert d["resolved_from"] == "warm-no-incoming-chain"
        assert _warm_classify_lines(caplog) == []


class TestNoRenameOfEventConstants:
    def test_discriminator_event_constant_values_byte_identical(self):
        """The exact check required -- these values are a LIVE
        LOOKUP KEY (EVENT_TO_CLASS -> POLICIES -> save_ok in kv_classify.py,
        plus two hardcoded-literal call sites in manager.py); renaming them
        would silently move a save decision. This change does not touch them."""
        from turbohaul.kv_classify import (
            EVENT_COMPRESSION,
            EVENT_CONTINUATION,
            EVENT_SUB_AGENT,
            EVENT_USER_MESSAGE,
        )
        assert EVENT_SUB_AGENT == "sub-agent"
        assert EVENT_COMPRESSION == "compression"
        assert EVENT_CONTINUATION == "continuation"
        assert EVENT_USER_MESSAGE == "user-message"

    def test_discriminator_hardcoded_literal_call_sites_still_present(self):
        """The two call sites that bypass the EVENT_* constants entirely
        (manager.py) -- named here so a future rename attempt trips this
        test instead of silently desyncing them from the classifier."""
        import inspect

        import turbohaul.manager as manager_mod
        src = inspect.getsource(manager_mod)
        assert '"event_type") == "compression"' in src
        assert '"continuation" if restore_ok else "guard-skip"' in src
