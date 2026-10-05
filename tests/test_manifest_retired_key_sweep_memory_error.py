"""A file that runs the parser out of memory must not stop the retired-key sweep.

`migrate_retired_keys` promises that a problem with one stored manifest never stops
the sweep of the others. A very large file, or one built to blow up when its
aliases are expanded, can make the YAML parser raise MemoryError. That file must
be left byte-for-byte as it is with one warning, and every other manifest must
still be swept.
"""
from __future__ import annotations

import logging
import os

import pytest
import yaml

import turbohaul.manifest as manifest_mod
from turbohaul.manifest import ModelManifest, migrate_retired_keys

MANIFEST_LOGGER = "turbohaul.manifest"
OLD_MTIME_NS = 1_500_000_000_000_000_000  # 2017; any rewrite would move it
SHA = "a" * 64


def _put_old_style(root, tag, max_instances=2):
    """Store a manifest that still carries the retired key; return its fields."""
    data = ModelManifest(
        model_tag=tag,
        display_name=f"Display {tag}",
        gguf_blob_sha256=SHA,
        gguf_size_bytes=1024,
        context_size=2048,
        llama_server_flags={"ctx_size": 2048, "n_gpu_layers": 7},
    ).model_dump(mode="json")
    data["max_instances"] = max_instances
    path = root / f"{tag}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    os.chmod(path, 0o644)
    os.utime(path, ns=(OLD_MTIME_NS, OLD_MTIME_NS))
    return data


def _warnings(caplog):
    return [
        r.getMessage() for r in caplog.records
        if r.name == MANIFEST_LOGGER and r.levelno >= logging.WARNING
    ]


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "manifests"
    d.mkdir()
    return d


def test_memory_error_on_one_file_leaves_it_alone_and_still_sweeps_the_others(
    root, caplog, monkeypatch
):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    # "aaa-huge" sorts first, so a sweep that stops at it never reaches "bbb-ok".
    huge = _put_old_style(root, "aaa-huge", 3)
    ok = _put_old_style(root, "bbb-ok", 2)
    huge_path = root / "aaa-huge.yaml"
    huge_bytes = huge_path.read_bytes()
    huge_mtime = huge_path.stat().st_mtime_ns

    real_safe_load = yaml.safe_load
    calls = []

    def safe_load_running_out_of_memory(text, *args, **kwargs):
        # The sweep hands the parser the file text; pick the file by its tag.
        calls.append("model_tag: aaa-huge" in text)
        if "model_tag: aaa-huge" in text:
            raise MemoryError()
        return real_safe_load(text, *args, **kwargs)

    monkeypatch.setattr(manifest_mod.yaml, "safe_load", safe_load_running_out_of_memory)

    try:
        result = migrate_retired_keys(root)
    except MemoryError:
        pytest.fail("MemoryError escaped from the sweep and stopped it")

    assert sorted(calls) == [False, True]          # the parser really saw both files
    assert result == ["bbb-ok"]
    # The file that ran out of memory: same bytes, same mtime, retired key still there.
    assert huge_path.read_bytes() == huge_bytes
    assert huge_path.stat().st_mtime_ns == huge_mtime
    assert real_safe_load(huge_bytes.decode())["max_instances"] == huge["max_instances"]
    # The other file: swept, every other field unchanged.
    stored = real_safe_load((root / "bbb-ok.yaml").read_text())
    assert "max_instances" not in stored
    assert stored == {k: v for k, v in ok.items() if k != "max_instances"}
    # One warning each, the first naming the error type and nothing else about the file.
    assert _warnings(caplog) == [
        "retired-key sweep: leaving 'aaa-huge' untouched, cannot read it (MemoryError)",
        "manifest 'bbb-ok': removed retired key(s) max_instances from the stored file",
    ]
    # No stray tempfile.
    assert sorted(p.name for p in root.iterdir()) == ["aaa-huge.yaml", "bbb-ok.yaml"]
