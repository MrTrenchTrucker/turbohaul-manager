"""Whatever one stored manifest does to the parser, the retired-key sweep must carry on.

`migrate_retired_keys` promises that a problem with one stored file never stops
the sweep of the others. The per-file guard has to cover the whole class of
ordinary errors, not a list of known ones. Two cases are checked here:

* a real YAML text that the real parser rejects with a ValueError (an impossible
  calendar date), which is not a YAMLError;
* a parser that raises some other ordinary error for one file.

In both, the bad file is left byte-for-byte as it is with one warning naming the
file and the error class, and the good file next to it is still swept.
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
IMPOSSIBLE_DATE_LINE = "released: 2001-13-45\n"  # month 13 does not exist


def _put_old_style(root, tag, max_instances=2, extra_text=""):
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
    path.write_text(yaml.safe_dump(data, sort_keys=False) + extra_text)
    os.chmod(path, 0o644)
    os.utime(path, ns=(OLD_MTIME_NS, OLD_MTIME_NS))
    return data


def _warnings(caplog):
    return [
        r.getMessage() for r in caplog.records
        if r.name == MANIFEST_LOGGER and r.levelno >= logging.WARNING
    ]


def _check_outcome(root, caplog, result, bad_tag, good_tag, good, bad_bytes, bad_mtime, exc_name):
    """The shared assertions: bad file untouched and named once, good file swept."""
    assert result == [good_tag], (
        "only the good file may be reported as rewritten; the bad file was left alone"
    )
    bad_path = root / f"{bad_tag}.yaml"
    assert bad_path.read_bytes() == bad_bytes, "the bad file must stay byte-for-byte as it was"
    assert bad_path.stat().st_mtime_ns == bad_mtime, (
        "the bad file must not be rewritten, even with identical bytes (its mtime moved)"
    )
    stored = yaml.safe_load((root / f"{good_tag}.yaml").read_text())
    assert "max_instances" not in stored, "the good file must lose the retired key"
    assert stored == {k: v for k, v in good.items() if k != "max_instances"}, (
        "every other field of the good file must be unchanged"
    )
    bad_warning = f"retired-key sweep: leaving {bad_tag!r} untouched, cannot read it ({exc_name})"
    good_warning = f"manifest {good_tag!r}: removed retired key(s) max_instances from the stored file"
    expected = [bad_warning, good_warning] if bad_tag < good_tag else [good_warning, bad_warning]
    assert _warnings(caplog) == expected, (
        f"exactly one warning for the bad file, naming its tag and {exc_name}, plus the good file's notice"
    )
    assert sorted(p.name for p in root.iterdir()) == sorted(
        [f"{bad_tag}.yaml", f"{good_tag}.yaml"]
    ), "no stray temporary file may be left in the manifest directory"


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "manifests"
    d.mkdir()
    return d


def test_the_real_parser_rejects_the_impossible_date_with_a_value_error(root):
    """Precondition: the stored text below really makes the real parser raise ValueError."""
    _put_old_style(root, "probe", extra_text=IMPOSSIBLE_DATE_LINE)
    with pytest.raises(ValueError) as info:
        yaml.safe_load((root / "probe.yaml").read_text())
    assert not isinstance(info.value, yaml.YAMLError), (
        "the case only means something if the error is NOT a YAMLError"
    )


@pytest.mark.parametrize(
    "bad_tag, good_tag",
    [("aaa-bad-date", "mmm-ok"), ("zzz-bad-date", "mmm-ok")],
    ids=["bad-file-sorts-first", "bad-file-sorts-last"],
)
def test_impossible_date_in_one_file_leaves_it_alone_and_still_sweeps_the_other(
    root, caplog, bad_tag, good_tag
):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    _put_old_style(root, bad_tag, 3, extra_text=IMPOSSIBLE_DATE_LINE)
    good = _put_old_style(root, good_tag, 2)
    bad_path = root / f"{bad_tag}.yaml"
    bad_bytes = bad_path.read_bytes()
    bad_mtime = bad_path.stat().st_mtime_ns

    try:
        result = migrate_retired_keys(root)
    except ValueError:
        pytest.fail("ValueError from the real parser escaped the sweep and stopped it")

    _check_outcome(root, caplog, result, bad_tag, good_tag, good, bad_bytes, bad_mtime, "ValueError")


@pytest.mark.parametrize("exc_type", [TypeError, KeyError, ZeroDivisionError])
def test_other_ordinary_errors_from_the_parser_do_not_stop_the_sweep(
    root, caplog, monkeypatch, exc_type
):
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    # "aaa-odd" sorts first, so a sweep that stops at it never reaches "bbb-ok".
    _put_old_style(root, "aaa-odd", 3)
    good = _put_old_style(root, "bbb-ok", 2)
    bad_path = root / "aaa-odd.yaml"
    bad_bytes = bad_path.read_bytes()
    bad_mtime = bad_path.stat().st_mtime_ns

    real_safe_load = yaml.safe_load
    calls = []

    def safe_load_failing_on_one_file(text, *args, **kwargs):
        # The sweep hands the parser the file text; pick the file by its tag.
        calls.append("model_tag: aaa-odd" in text)
        if "model_tag: aaa-odd" in text:
            raise exc_type("simulated parser failure")
        return real_safe_load(text, *args, **kwargs)

    monkeypatch.setattr(manifest_mod.yaml, "safe_load", safe_load_failing_on_one_file)

    try:
        result = migrate_retired_keys(root)
    except exc_type:
        pytest.fail(f"{exc_type.__name__} escaped the sweep and stopped it")

    assert sorted(calls) == [False, True], "the parser must have seen both files"
    _check_outcome(
        root, caplog, result, "aaa-odd", "bbb-ok", good, bad_bytes, bad_mtime, exc_type.__name__
    )
