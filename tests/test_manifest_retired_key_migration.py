"""One-time sweep that removes the retired `max_instances` key from stored manifests.

`migrate_retired_keys(manifests_root)` rewrites only the files the store itself
loads as manifests, and only when they carry a retired key. The rewrite is a pure
key removal: every other field and the `revision` stay exactly as they were, so a
client holding the old ETag can still save. Anything else in the directory
(backups, hidden files, notes, symlinks, files that do not parse) is left
byte-for-byte alone.
"""
from __future__ import annotations

import logging
import os
import stat

import pytest
import yaml

import turbohaul.manifest as manifest_mod
from turbohaul.manifest import (
    ModelManifest,
    list_manifests,
    manifest_etag,
    migrate_retired_keys,
    read_manifest,
    write_manifest_atomic,
)

MANIFEST_LOGGER = "turbohaul.manifest"
OLD_MTIME_NS = 1_500_000_000_000_000_000  # 2017; any rewrite would move it
SHA = "a" * 64


def _body(tag, **over):
    """A full valid manifest body as the writer would store it."""
    m = ModelManifest(
        model_tag=tag,
        display_name=f"Display {tag}",
        gguf_blob_sha256=SHA,
        gguf_size_bytes=1024,
        context_size=2048,
        llama_server_flags={"ctx_size": 2048, "n_gpu_layers": 7},
        **over,
    )
    return m.model_dump(mode="json")


def _put_raw(root, name, text, mode=0o644):
    p = root / name
    if isinstance(text, bytes):
        p.write_bytes(text)
    else:
        p.write_text(text)
    os.chmod(p, mode)
    os.utime(p, ns=(OLD_MTIME_NS, OLD_MTIME_NS))
    return p


def _put_old_style(root, tag, max_instances=2, **over):
    """A stored manifest that still carries the retired key."""
    data = _body(tag, **over)
    data["max_instances"] = max_instances
    _put_raw(root, f"{tag}.yaml", yaml.safe_dump(data, sort_keys=False))
    return data


def _snapshot(root):
    """{name: (bytes, mtime_ns)} for every entry, symlinks not followed."""
    snap = {}
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if entry.is_symlink():
            snap[entry.name] = ("symlink", os.readlink(entry.path))
        elif entry.is_file(follow_symlinks=False):
            st = entry.stat(follow_symlinks=False)
            with open(entry.path, "rb") as f:
                snap[entry.name] = (f.read(), st.st_mtime_ns)
    return snap


def _sweep_records(caplog):
    return [
        r for r in caplog.records
        if r.name == MANIFEST_LOGGER and r.levelno >= logging.WARNING
    ]


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "manifests"
    d.mkdir()
    return d


def test_sweep_rewrites_only_files_with_the_key(root, caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    old = {
        "old-a": _put_old_style(root, "old-a", 2, revision=1),
        "old-b": _put_old_style(root, "old-b", 7, revision=5),
        "old-c": _put_old_style(root, "old-c", 1, revision=9, hidden=True),
    }
    # Two key-free manifests: one written by the real writer, one raw.
    write_manifest_atomic(root, ModelManifest(**_body("new-a")))
    clean_raw = _body("new-b")
    _put_raw(root, "new-b.yaml", yaml.safe_dump(clean_raw, sort_keys=False))
    os.utime(root / "new-a.yaml", ns=(OLD_MTIME_NS, OLD_MTIME_NS))
    _put_raw(root, "broken.yaml", "model_tag: [unclosed\n  - x: : :\n")
    _put_raw(root, "listroot.yaml", "- model_tag: x\n- max_instances: 2\n")
    before = _snapshot(root)

    result = migrate_retired_keys(root)

    assert result == ["old-a", "old-b", "old-c"]
    after = _snapshot(root)
    for tag, original in old.items():
        stored = yaml.safe_load((root / f"{tag}.yaml").read_text())
        expected = {k: v for k, v in original.items() if k != "max_instances"}
        assert "max_instances" not in stored
        assert stored == expected                      # every field, revision included
        assert stored["revision"] == original["revision"]
        assert after[f"{tag}.yaml"][0] != before[f"{tag}.yaml"][0]
    for name in ("new-a.yaml", "new-b.yaml", "broken.yaml", "listroot.yaml"):
        assert after[name] == before[name], name        # bytes AND mtime_ns
    # No stray tempfile and no new file of any kind.
    assert sorted(after) == sorted(before)

    records = _sweep_records(caplog)
    messages = sorted(r.getMessage() for r in records)
    assert messages == sorted([
        "manifest 'old-a': removed retired key(s) max_instances from the stored file",
        "manifest 'old-b': removed retired key(s) max_instances from the stored file",
        "manifest 'old-c': removed retired key(s) max_instances from the stored file",
        "retired-key sweep: leaving 'broken' untouched, cannot read it (ParserError)",
        "retired-key sweep: leaving 'listroot' untouched, its root is not a mapping",
    ])
    # Tag and key only: no value of the old files appears in any record.
    joined = " ".join(messages)
    assert "Display" not in joined
    assert "7" not in joined.replace("max_instances", "")


def test_rewritten_file_has_mode_0600_and_loads_with_no_warning(root, caplog):
    _put_old_style(root, "old-a", 3)
    assert stat.S_IMODE(os.stat(root / "old-a.yaml").st_mode) == 0o644
    migrate_retired_keys(root)
    assert stat.S_IMODE(os.stat(root / "old-a.yaml").st_mode) == 0o600
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    m = read_manifest(root, "old-a")
    assert m.model_tag == "old-a"
    assert _sweep_records(caplog) == []


def test_second_run_returns_empty_and_changes_nothing(root, caplog):
    _put_old_style(root, "old-a", 2)
    _put_old_style(root, "old-b", 3)
    _put_raw(root, "clean.yaml", yaml.safe_dump(_body("clean"), sort_keys=False))
    assert migrate_retired_keys(root) == ["old-a", "old-b"]
    snap = _snapshot(root)
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    assert migrate_retired_keys(root) == []
    assert _snapshot(root) == snap                      # bytes and mtime_ns
    assert _sweep_records(caplog) == []


def test_uses_tempfile_in_same_dir_and_os_replace(root, monkeypatch):
    _put_old_style(root, "old-a", 2)
    real_replace = os.replace
    calls = []

    def spy(src, dst, *a, **kw):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(manifest_mod.os, "replace", spy)
    assert migrate_retired_keys(root) == ["old-a"]
    assert len(calls) == 1
    src, dst = calls[0]
    assert dst == str((root / "old-a.yaml").resolve())
    assert os.path.dirname(src) == os.path.dirname(dst)
    assert os.path.basename(src).startswith(".tmp_")


def test_failure_between_tempfile_write_and_replace_is_atomic(root, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    _put_old_style(root, "aaa-fails", 2)
    old_b = _put_old_style(root, "bbb-ok", 3)
    failing_before = _snapshot(root)["aaa-fails.yaml"]
    real_replace = os.replace

    def flaky(src, dst, *a, **kw):
        if os.path.basename(str(dst)) == "aaa-fails.yaml":
            raise OSError("simulated failure at replace")
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(manifest_mod.os, "replace", flaky)
    result = migrate_retired_keys(root)             # must not raise

    assert result == ["bbb-ok"]                     # failing tag not reported
    assert _snapshot(root)["aaa-fails.yaml"] == failing_before   # bytes + mtime_ns
    assert sorted(os.listdir(root)) == ["aaa-fails.yaml", "bbb-ok.yaml"]  # no tempfile
    stored_b = yaml.safe_load((root / "bbb-ok.yaml").read_text())
    assert stored_b == {k: v for k, v in old_b.items() if k != "max_instances"}
    messages = [r.getMessage() for r in _sweep_records(caplog)]
    assert messages == [
        "retired-key sweep: could not rewrite 'aaa-fails', left as it was (OSError)",
        "manifest 'bbb-ok': removed retired key(s) max_instances from the stored file",
    ]
    # Once the fault is gone a later sweep finishes the job.
    monkeypatch.setattr(manifest_mod.os, "replace", real_replace)
    assert migrate_retired_keys(root) == ["aaa-fails"]


def test_missing_root_returns_empty_list(tmp_path):
    assert migrate_retired_keys(tmp_path / "does-not-exist") == []


def test_empty_root_returns_empty_list(root):
    assert migrate_retired_keys(root) == []


def test_symlinked_manifest_is_left_alone(root, tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    outside = tmp_path / "outside.txt"
    outside.write_text(yaml.safe_dump({**_body("link"), "max_instances": 2}, sort_keys=False))
    os.utime(outside, ns=(OLD_MTIME_NS, OLD_MTIME_NS))
    os.symlink(outside, root / "link.yaml")
    outside_before = (outside.read_bytes(), outside.stat().st_mtime_ns)
    assert migrate_retired_keys(root) == []
    assert (root / "link.yaml").is_symlink()
    assert os.readlink(root / "link.yaml") == str(outside)
    assert (outside.read_bytes(), outside.stat().st_mtime_ns) == outside_before
    assert _sweep_records(caplog) == []


def test_directory_named_like_a_manifest_is_skipped(root):
    (root / "folder.yaml").mkdir()
    assert migrate_retired_keys(root) == []
    assert (root / "folder.yaml").is_dir()


def test_non_utf8_file_is_left_untouched_and_does_not_stop_the_sweep(root, caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    _put_raw(root, "aaa-bytes.yaml", b"model_tag: \xff\xfe\xfa\nmax_instances: 2\n")
    _put_old_style(root, "bbb-ok", 2)
    before = _snapshot(root)["aaa-bytes.yaml"]
    assert migrate_retired_keys(root) == ["bbb-ok"]
    assert _snapshot(root)["aaa-bytes.yaml"] == before
    messages = [r.getMessage() for r in _sweep_records(caplog)]
    assert messages[0] == (
        "retired-key sweep: leaving 'aaa-bytes' untouched, cannot read it "
        "(UnicodeDecodeError)"
    )
    assert len(messages) == 2


def test_empty_file_is_left_untouched(root, caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    _put_raw(root, "empty.yaml", "")
    before = _snapshot(root)
    assert migrate_retired_keys(root) == []
    assert _snapshot(root) == before
    assert [r.getMessage() for r in _sweep_records(caplog)] == [
        "retired-key sweep: leaving 'empty' untouched, its root is not a mapping"
    ]


def test_file_failing_validation_for_another_reason_still_loses_only_the_key(root):
    # The sweep does not validate: an unrelated problem must neither block the
    # key removal nor be "fixed" on the way through.
    data = {"model_tag": "odd", "max_instances": 2, "context_size": -5,
            "mystery_field": {"x": 1}, "revision": 4}
    _put_raw(root, "odd.yaml", yaml.safe_dump(data, sort_keys=False))
    assert migrate_retired_keys(root) == ["odd"]
    stored = yaml.safe_load((root / "odd.yaml").read_text())
    assert stored == {"model_tag": "odd", "context_size": -5,
                      "mystery_field": {"x": 1}, "revision": 4}


def test_nested_max_instances_is_not_a_retired_key_and_is_kept(root):
    data = _body("nest")
    data["llama_server_flags"]["max_instances"] = 5      # nested: not top level
    _put_raw(root, "nest.yaml", yaml.safe_dump(data, sort_keys=False))
    before = _snapshot(root)
    assert migrate_retired_keys(root) == []
    assert _snapshot(root) == before


def test_key_order_of_the_other_fields_is_preserved(root):
    data = _put_old_style(root, "ordered", 2)
    migrate_retired_keys(root)
    stored = yaml.safe_load((root / "ordered.yaml").read_text())
    assert list(stored) == [k for k in data if k != "max_instances"]


def test_etag_is_unchanged_by_the_sweep_and_the_old_etag_can_still_save(root):
    _put_old_style(root, "etag-tag", 2, revision=3)
    etag_before = manifest_etag(root, "etag-tag")
    assert etag_before == '"3"'
    assert migrate_retired_keys(root) == ["etag-tag"]
    assert manifest_etag(root, "etag-tag") == etag_before
    written = write_manifest_atomic(
        root, read_manifest(root, "etag-tag"), if_match=etag_before
    )
    assert written.revision == 4
    stored = yaml.safe_load((root / "etag-tag.yaml").read_text())
    assert stored["revision"] == 4
    assert "max_instances" not in stored


# --- only the files the store loads as manifests ------------------------------


def test_backups_hidden_and_other_files_are_never_touched(root, caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    old = _put_old_style(root, "a", 2, revision=2)
    manifest_like = yaml.safe_dump({**_body("a"), "max_instances": 4}, sort_keys=False)
    others = [
        "a.yaml.bak-1",
        "a.yaml.bak",
        "a.pre-fix",
        ".hidden.yaml",
        "notes.txt",
        "Upper.yaml",       # stem is not a valid tag, so the store never loads it
        "a.yaml~",
        "a.yml",
    ]
    for name in others:
        _put_raw(root, name, manifest_like)
    before = _snapshot(root)

    assert migrate_retired_keys(root) == ["a"]

    after = _snapshot(root)
    stored = yaml.safe_load((root / "a.yaml").read_text())
    assert stored == {k: v for k, v in old.items() if k != "max_instances"}
    for name in others:
        assert after[name] == before[name], name         # bytes AND mtime_ns
    assert sorted(after) == sorted(before)
    assert [r.getMessage() for r in _sweep_records(caplog)] == [
        "manifest 'a': removed retired key(s) max_instances from the stored file"
    ]


def test_rewritten_text_is_block_style_in_the_writers_format(root, tmp_path):
    # The sweep must write what write_manifest_atomic would write for the same
    # content: block-style YAML, original key order, no flow-style braces.
    data = _put_old_style(root, "fmt-tag", 2)
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    write_manifest_atomic(
        reference_root,
        ModelManifest(**{k: v for k, v in data.items() if k != "max_instances"}),
    )
    migrate_retired_keys(root)
    text = (root / "fmt-tag.yaml").read_text()
    assert text == (reference_root / "fmt-tag.yaml").read_text()
    assert not text.startswith("{")
    assert "{" not in text
    top_level_keys = [
        line.split(":")[0] for line in text.splitlines() if line and not line[0].isspace()
    ]
    assert top_level_keys == [k for k in data if k != "max_instances"]
    assert "llama_server_flags:\n  ctx_size: 2048\n" in text


# --- per-file problems and the exact write sequence ---------------------------


def test_directory_named_like_a_manifest_is_skipped_without_a_warning(root, caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    (root / "adir.yaml").mkdir()
    _put_old_style(root, "good", 2)
    assert migrate_retired_keys(root) == ["good"]
    assert (root / "adir.yaml").is_dir()
    assert os.listdir(root / "adir.yaml") == []
    messages = [r.getMessage() for r in _sweep_records(caplog)]
    assert messages == [
        "manifest 'good': removed retired key(s) max_instances from the stored file"
    ]


def test_non_oserror_failure_while_writing_is_logged_by_type_and_the_sweep_goes_on(
    root, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    _put_old_style(root, "aaa-fails", 2)
    old_b = _put_old_style(root, "bbb-ok", 3)
    failing_before = _snapshot(root)["aaa-fails.yaml"]
    real_write = manifest_mod._write_text_atomic

    def flaky(target, text):
        if target.name == "aaa-fails.yaml":
            raise ValueError("secret detail that must not be logged")
        return real_write(target, text)

    monkeypatch.setattr(manifest_mod, "_write_text_atomic", flaky)
    assert migrate_retired_keys(root) == ["bbb-ok"]          # must not raise
    assert _snapshot(root)["aaa-fails.yaml"] == failing_before
    assert yaml.safe_load((root / "bbb-ok.yaml").read_text()) == {
        k: v for k, v in old_b.items() if k != "max_instances"
    }
    assert [r.getMessage() for r in _sweep_records(caplog)] == [
        "retired-key sweep: could not rewrite 'aaa-fails', left as it was (ValueError)",
        "manifest 'bbb-ok': removed retired key(s) max_instances from the stored file",
    ]


def test_list_manifests_takes_only_plain_yaml_files_with_a_valid_tag(root):
    for name in (
        "a.yaml", "b.yaml.bak", "c.yaml.orig", "d.yaml~", ".hidden.yaml",
        "Upper.yaml", "e.yml", "notes.txt",
    ):
        _put_raw(root, name, "model_tag: x\n")
    assert list_manifests(root) == ["a"]


def test_a_path_resolution_error_is_logged_by_type_and_the_sweep_goes_on(
    root, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    denied = _put_old_style(root, "aaa-denied", 2)
    _put_old_style(root, "bbb-ok", 3)
    denied_before = _snapshot(root)["aaa-denied.yaml"]
    real_safe = manifest_mod._safe_manifest_path

    def fake_safe(manifests_root, tag):
        if tag == "aaa-denied":
            raise PermissionError("denied detail that must not be logged")
        return real_safe(manifests_root, tag)

    monkeypatch.setattr(manifest_mod, "_safe_manifest_path", fake_safe)
    assert migrate_retired_keys(root) == ["bbb-ok"]          # must not raise
    assert _snapshot(root)["aaa-denied.yaml"] == denied_before
    assert denied["max_instances"] == 2
    assert [r.getMessage() for r in _sweep_records(caplog)] == [
        "retired-key sweep: cannot resolve 'aaa-denied' (PermissionError)",
        "manifest 'bbb-ok': removed retired key(s) max_instances from the stored file",
    ]


def test_deeply_nested_yaml_is_left_untouched_and_does_not_stop_the_sweep(root, caplog):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    _put_raw(root, "aaa-deep.yaml", "[" * 5000 + "]" * 5000 + "\n")
    _put_old_style(root, "bbb-ok", 2)
    deep_before = _snapshot(root)["aaa-deep.yaml"]
    assert migrate_retired_keys(root) == ["bbb-ok"]          # must not raise
    assert _snapshot(root)["aaa-deep.yaml"] == deep_before
    assert "max_instances" not in yaml.safe_load((root / "bbb-ok.yaml").read_text())
    assert [r.getMessage() for r in _sweep_records(caplog)] == [
        "retired-key sweep: leaving 'aaa-deep' untouched, cannot read it (RecursionError)",
        "manifest 'bbb-ok': removed retired key(s) max_instances from the stored file",
    ]


def test_rewrite_syncs_the_file_before_the_replace_and_the_directory_after(root, monkeypatch):
    _put_old_style(root, "old-a", 2)
    root_inode = os.stat(root).st_ino
    real_fsync = os.fsync
    real_replace = os.replace
    events = []

    def spy_fsync(fd):
        mode = os.fstat(fd).st_mode
        if stat.S_ISREG(mode):
            events.append("fsync-file")
        elif stat.S_ISDIR(mode):
            events.append("fsync-dir-is-root" if os.fstat(fd).st_ino == root_inode
                          else "fsync-other-dir")
        else:
            events.append("fsync-other")
        return real_fsync(fd)

    def spy_replace(src, dst, *a, **kw):
        events.append("replace")
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(manifest_mod.os, "fsync", spy_fsync)
    monkeypatch.setattr(manifest_mod.os, "replace", spy_replace)
    assert migrate_retired_keys(root) == ["old-a"]
    assert events == ["fsync-file", "replace", "fsync-dir-is-root"]
