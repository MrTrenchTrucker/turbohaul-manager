"""Tests for manifest schema + atomic write + ETag/If-Match concurrency."""
import os
import time

import pytest
from pydantic import ValidationError as PydanticValidationError

from turbohaul.manifest import (
    ConcurrencyError,
    Manifest,
    ModelManifest,
    ManifestValidationError,
    cache_reuse_inert_on_mmproj,
    delete_manifest,
    flags_to_argv,
    list_manifests,
    manifest_etag,
    read_manifest,
    read_manifest_cached,
    validate_tag,
    write_manifest_atomic,
)

SAMPLE_TAG = "qwen3.6-35b-moe"
SAMPLE_SHA = "1a2b3c4d" + "0" * 56  # 64 hex chars


def make_manifest(**overrides) -> Manifest:
    base = dict(
        model_tag=SAMPLE_TAG,
        display_name="Qwen 3.6 35B-A3B MoE Q4",
        description="test",
        gguf_blob_sha256=SAMPLE_SHA,
        gguf_size_bytes=22_000_000_000,
        context_size=131072,
        expected_vram_bytes=22_500_000_000,
        revision=1,
        llama_server_flags={"ctx_size": 131072, "n_gpu_layers": 999, "mlock": True},
    )
    base.update(overrides)
    return ModelManifest(**base)


class TestTagValidation:
    def test_valid_tags(self):
        for t in ["qwen3.6-35b-moe", "abc", "a", "tag-name_v1.0", "model123"]:
            validate_tag(t)

    def test_invalid_tags(self):
        for bad in [
            "",
            "../etc/passwd",
            "tag/path",
            "tag\\back",
            "Tag-Upper",
            ".dotted",
            "a" * 65,
            "tag with space",
            "tag\x00null",
        ]:
            with pytest.raises(ManifestValidationError):
                validate_tag(bad)


class TestManifestSchema:
    def test_valid_manifest_parses(self):
        m = make_manifest()
        assert m.model_tag == SAMPLE_TAG
        assert m.llama_server_flags["ctx_size"] == 131072

    def test_reject_unknown_flag(self):
        with pytest.raises(PydanticValidationError, match="not in the closed allowlist"):
            make_manifest(llama_server_flags={"evil_unknown": "x"})

    def test_reject_path_bearing_flag_mmproj(self):
        with pytest.raises(PydanticValidationError, match="explicitly denied"):
            make_manifest(llama_server_flags={"mmproj": "/etc/passwd"})

    def test_reject_path_bearing_flag_lora(self):
        with pytest.raises(PydanticValidationError, match="explicitly denied"):
            make_manifest(llama_server_flags={"lora": "/root/private/secrets.env"})

    def test_reject_path_bearing_flag_log_file(self):
        with pytest.raises(PydanticValidationError, match="explicitly denied"):
            make_manifest(llama_server_flags={"log_file": "/etc/cron.d/pwn"})

    def test_reject_hf_token_override(self):
        with pytest.raises(PydanticValidationError, match="explicitly denied"):
            make_manifest(llama_server_flags={"hf_token": "attacker-token"})

    def test_reject_model_override(self):
        with pytest.raises(PydanticValidationError, match="explicitly denied"):
            make_manifest(llama_server_flags={"model": "/tmp/evil.gguf"})

    def test_log_verbosity_accepts_engine_debug_level(self):
        # engine LOG_LEVEL_DEBUG is 5 (common/log.h) — must be reachable
        # through a manifest, not capped one below it.
        m = make_manifest(llama_server_flags={"log_verbosity": 5})
        assert m.llama_server_flags["log_verbosity"] == 5

    def test_reject_log_verbosity_above_new_max(self):
        with pytest.raises(PydanticValidationError, match="above max 5"):
            make_manifest(llama_server_flags={"log_verbosity": 6})

    def test_reject_bad_sha256(self):
        with pytest.raises(PydanticValidationError, match="64 hex"):
            make_manifest(gguf_blob_sha256="notahash")

    def test_reject_short_sha256(self):
        with pytest.raises(PydanticValidationError):
            make_manifest(gguf_blob_sha256="abc123")

    def test_reject_tag_with_traversal(self):
        with pytest.raises(PydanticValidationError):
            make_manifest(model_tag="../../etc/passwd")

    def test_reject_top_level_unknown_field(self):
        with pytest.raises(PydanticValidationError):
            ModelManifest(
                model_tag=SAMPLE_TAG,
                gguf_blob_sha256=SAMPLE_SHA,
                evil_field=1,  # type: ignore[call-arg]
            )

    def test_bool_type_strict(self):
        # int 1 must NOT be accepted as bool for mlock
        with pytest.raises(PydanticValidationError, match="expects bool"):
            make_manifest(llama_server_flags={"mlock": 1})

    def test_kv_bytes_per_token_floor(self):
        # kv_bytes_per_token is the highest-precedence KV-fit
        # override, used VERBATIM as effective BYTES/token. Its only guard is a
        # 1 KiB/token floor (Field ge=1024.0) that rejects a KiB-vs-bytes typo
        # (a <1 KiB value would silently under-count → the gate would always pass).
        with pytest.raises(PydanticValidationError):
            make_manifest(kv_bytes_per_token=13.5)     # typed KiB, meant bytes → rejected
        with pytest.raises(PydanticValidationError):
            make_manifest(kv_bytes_per_token=1023.9)   # just below the floor → rejected
        assert make_manifest(kv_bytes_per_token=1024.0).kv_bytes_per_token == 1024.0  # floor accepts
        assert make_manifest(kv_bytes_per_token=13824.0).kv_bytes_per_token == 13824.0  # measured value
        assert make_manifest().kv_bytes_per_token is None  # unset default → byte-identical


SAMPLE_MMPROJ_SHA = "5a6b7c8d" + "0" * 56  # 64 hex chars


class TestCacheReuseMmprojVisibility:
    """cache_reuse is inert (not wrong) on a multimodal manifest --
    the engine unconditionally zeroes it for mtmd requests. Make that visible
    without a save-time reject (some real manifests set both)."""

    def test_negative_control_cache_reuse_no_mmproj_no_warning(self, caplog):
        # cache_reuse alone, text-only model: nothing is inert, no log line. A
        # signal that always fires is not a signal.
        with caplog.at_level("INFO", logger="turbohaul.manifest"):
            m = make_manifest(llama_server_flags={"cache_reuse": 256})
        assert m.mmproj_blob_sha256 == ""
        assert not any("cache_reuse" in r.message and "inert" in r.message for r in caplog.records)

    def test_positive_control_cache_reuse_and_mmproj_fires(self, caplog):
        # a realistic combination: cache_reuse=256 + an mmproj set.
        with caplog.at_level("INFO", logger="turbohaul.manifest"):
            make_manifest(
                llama_server_flags={"cache_reuse": 256},
                mmproj_blob_sha256=SAMPLE_MMPROJ_SHA,
            )
        matches = [r for r in caplog.records if "inert" in r.message and SAMPLE_TAG in r.message]
        assert len(matches) == 1
        assert matches[0].levelname == "INFO"  # not WARNING -- see manifest.py docstring

    def test_save_and_load_still_work(self):
        # Acceptance test: saving/loading a manifest with both set must not
        # raise -- this is explicitly NOT a save-time reject.
        m = make_manifest(
            llama_server_flags={"cache_reuse": 256},
            mmproj_blob_sha256=SAMPLE_MMPROJ_SHA,
        )
        assert m.llama_server_flags["cache_reuse"] == 256
        assert m.mmproj_blob_sha256 == SAMPLE_MMPROJ_SHA

    def test_no_mmproj_no_cache_reuse_key_at_all_not_inert(self):
        m = make_manifest(mmproj_blob_sha256=SAMPLE_MMPROJ_SHA)
        assert cache_reuse_inert_on_mmproj(m.llama_server_flags, m.mmproj_blob_sha256) is False

    def test_cache_reuse_zero_with_mmproj_not_inert(self):
        # cache_reuse: 0 means "off" already -- nothing for the engine to
        # disable, so this is not the inert case.
        m = make_manifest(
            llama_server_flags={"cache_reuse": 0}, mmproj_blob_sha256=SAMPLE_MMPROJ_SHA
        )
        assert cache_reuse_inert_on_mmproj(m.llama_server_flags, m.mmproj_blob_sha256) is False

    def test_predicate_matches_log_trigger(self):
        # Same predicate drives both the log line and the API marker alike
        # -- assert the pure function agrees with itself both ways.
        inert = make_manifest(
            llama_server_flags={"cache_reuse": 256}, mmproj_blob_sha256=SAMPLE_MMPROJ_SHA
        )
        not_inert = make_manifest(llama_server_flags={"cache_reuse": 256})
        assert cache_reuse_inert_on_mmproj(inert.llama_server_flags, inert.mmproj_blob_sha256) is True
        assert cache_reuse_inert_on_mmproj(not_inert.llama_server_flags, not_inert.mmproj_blob_sha256) is False


class TestAtomicWriteAndRead:
    def test_write_then_read(self, tmp_path):
        m = make_manifest()
        out = write_manifest_atomic(tmp_path, m)
        assert out.revision == 1
        read = read_manifest(tmp_path, SAMPLE_TAG)
        assert read.model_tag == SAMPLE_TAG
        assert read.gguf_blob_sha256 == SAMPLE_SHA
        assert read.revision == 1

    def test_etag_after_write(self, tmp_path):
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        etag = manifest_etag(tmp_path, SAMPLE_TAG)
        assert etag == '"1"'

    def test_second_write_correct_if_match_increments(self, tmp_path):
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        out2 = write_manifest_atomic(tmp_path, make_manifest(display_name="Updated"), if_match='"1"')
        assert out2.revision == 2
        # Verify on-disk
        read = read_manifest(tmp_path, SAMPLE_TAG)
        assert read.revision == 2
        assert read.display_name == "Updated"

    def test_second_write_wrong_if_match_raises(self, tmp_path):
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        with pytest.raises(ConcurrencyError, match="If-Match"):
            write_manifest_atomic(tmp_path, make_manifest(display_name="Stale"), if_match='"99"')

    def test_no_if_match_on_update_raises(self, tmp_path):
        """PUT without If-Match on existing manifest must raise."""
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        with pytest.raises(ConcurrencyError):
            write_manifest_atomic(tmp_path, make_manifest(display_name="No-ETag"))

    def test_file_mode_is_0o600(self, tmp_path):
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        path = tmp_path / f"{SAMPLE_TAG}.yaml"
        mode = path.stat().st_mode & 0o777
        assert mode & 0o600 == 0o600
        assert mode & 0o077 == 0  # no group/other access

    @pytest.mark.parametrize(
        "hostile_umask",
        [
            # 0o022: a common default umask, and the
            # single most useful value to test -- a plain open()/write_text()
            # write (0o666 base) under 0o022 produces exactly 0o644, which is
            # a common wrong mode. This is the
            # umask value that actually discriminates "explicit 0600" from
            # "umask-dependent default" for THIS bug shape.
            0o022,
            # 0o000: maximally permissive umask (clears nothing) -- a plain
            # open() write here would be 0o666, the worst case. If 0o022
            # alone somehow didn't catch a regression, this would.
            0o000,
            # 0o077 (another common umask value): included for
            # completeness, but NOTE -- verified empirically this does NOT
            # discriminate this specific mutant. 0o666 & ~0o077 == 0o600,
            # a coincidence of the arithmetic (0o077 clears exactly the
            # group+other bits that 0o600 also lacks), so a plain open()
            # write under this ONE umask value happens to land on 0o600 too.
            # Kept in the matrix (it's still a legitimate umask to prove
            # correctness under) but 0o022/0o000 above are what actually
            # catch the "never sets the mode" mutant -- a single-umask test
            # using only 0o077 would have passed against that regression.
            0o077,
        ],
    )
    def test_file_mode_is_0o600_under_hostile_umask(self, tmp_path, hostile_umask):
        """The test above (test_file_mode_is_0o600) runs under
        whatever umask the test process happens to have -- if that default
        is already restrictive enough not to leak anything, the test would
        pass whether or not the code actually forces the mode. That proves
        nothing about the claim, because the claim IS about umask
        independence, not about what one ambient umask happens to produce.

        Sets an explicit umask around the real write and confirms the
        result is 0o600 regardless -- restores the process umask in a
        finally so this test cannot leak state into any other test in the
        same process (umask is process-global, not per-thread).
        """
        old_umask = os.umask(hostile_umask)
        try:
            m = make_manifest()
            write_manifest_atomic(tmp_path, m)
        finally:
            os.umask(old_umask)
        path = tmp_path / f"{SAMPLE_TAG}.yaml"
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600, f"expected 0o600 under umask {oct(hostile_umask)}, got {oct(mode)}"


class TestReadManifestCached:
    """mtime-keyed cache wrapper for
    _effective_cap's hot-path manifest reads. read_manifest itself (and its
    ~24 other call sites, including the manifest API's own read-modify-write
    routes) is untouched -- only the NEW read_manifest_cached wrapper caches."""

    def setup_method(self):
        # Module-level cache -- clear between tests so no test can observe a
        # stale entry left by a prior one (paths are tmp_path-unique in
        # practice, but this removes any doubt).
        import turbohaul.manifest as _mod
        _mod._manifest_cache.clear()

    def test_cache_hit_same_mtime_does_not_reparse(self, tmp_path, monkeypatch):
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        calls = []
        real_read_manifest = read_manifest

        def spy_read_manifest(root, tag):
            calls.append(tag)
            return real_read_manifest(root, tag)

        monkeypatch.setattr("turbohaul.manifest.read_manifest", spy_read_manifest)

        first = read_manifest_cached(tmp_path, SAMPLE_TAG)
        assert calls == [SAMPLE_TAG], "first call must be a real (uncached) read"
        second = read_manifest_cached(tmp_path, SAMPLE_TAG)
        assert calls == [SAMPLE_TAG], (
            "second call at the SAME mtime must be served from cache, not re-parsed"
        )
        assert second.model_tag == first.model_tag == SAMPLE_TAG
        assert second.revision == first.revision == 1

    def test_mtime_change_triggers_reparse_with_new_content(self, tmp_path):
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        path = tmp_path / f"{SAMPLE_TAG}.yaml"
        os.utime(path, (1_000_000_000, 1_000_000_000))  # pin a known mtime

        calls = []
        real_read_manifest = read_manifest

        def spy_read_manifest(root, tag):
            calls.append(tag)
            return real_read_manifest(root, tag)

        # Spy is scoped tightly around each read_manifest_cached call only --
        # write_manifest_atomic ALSO calls the module-level read_manifest
        # internally (its own If-Match pre-read), which would otherwise
        # inflate this count and have nothing to do with the cache.
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("turbohaul.manifest.read_manifest", spy_read_manifest)
            first = read_manifest_cached(tmp_path, SAMPLE_TAG)
        assert first.display_name == "Qwen 3.6 35B-A3B MoE Q4"
        assert len(calls) == 1

        # Overwrite with different content AND a distinct (later) mtime.
        write_manifest_atomic(
            tmp_path, make_manifest(display_name="Updated"), if_match='"1"',
        )
        os.utime(path, (1_000_000_100, 1_000_000_100))  # advance mtime

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("turbohaul.manifest.read_manifest", spy_read_manifest)
            second = read_manifest_cached(tmp_path, SAMPLE_TAG)
        assert len(calls) == 2, "mtime change must trigger a real re-parse"
        assert second.display_name == "Updated", (
            "cache must not serve the stale pre-edit content after mtime changed"
        )

    def test_fresh_parse_error_still_raises_every_time(self, tmp_path):
        """A corrupt/invalid manifest must not be cached -- read_manifest_cached
        re-raises on every call, matching uncached read_manifest's behavior, so
        _effective_cap's existing degrade-to-1 except-clause is unaffected."""
        path = tmp_path / f"{SAMPLE_TAG}.yaml"
        path.write_text("not: [valid, manifest, shape}")  # malformed YAML
        with pytest.raises(Exception):
            read_manifest_cached(tmp_path, SAMPLE_TAG)
        # Second call must ALSO raise -- not silently return a cached None/
        # stale value from the failed first attempt.
        with pytest.raises(Exception):
            read_manifest_cached(tmp_path, SAMPLE_TAG)

    def test_effective_cap_uses_cache_not_uncached_read_manifest(self, tmp_path, monkeypatch):
        """Wiring check: _effective_cap must call read_manifest_cached, not
        the uncached read_manifest, directly (grep-level guard against a
        regression that reverts the wiring without touching the cache
        function itself)."""
        import inspect
        from turbohaul.manager import TurbohaulManager
        src = inspect.getsource(TurbohaulManager._effective_cap)
        assert "read_manifest_cached(" in src, (
            "_effective_cap must call read_manifest_cached -- the wiring "
            "moved or was reverted"
        )
        assert "read_manifest(" not in src.replace("read_manifest_cached(", ""), (
            "_effective_cap must not ALSO call the uncached read_manifest "
            "directly (would defeat the cache)"
        )

    def test_same_second_mtime_ns_still_triggers_reparse(self, tmp_path):
        """st_mtime (a float) cannot always represent
        the SAME two nanosecond-precision mtimes distinctly -- a Unix
        timestamp today has ~10 integer digits, and float64 has ~15-17
        significant digits total, leaving only microsecond-ish precision
        for the fractional part. A live manifest edit landing within
        that collision window would have been invisible to an st_mtime-keyed
        cache. This test PROVES the collision is real (not hypothetical)
        before proving the fix: two mtimes 1ns apart are shown to produce
        the IDENTICAL float st_mtime, then the cache (now keyed on the
        integer st_mtime_ns) is shown to still re-parse correctly across
        that exact pair."""
        m = make_manifest()
        write_manifest_atomic(tmp_path, m)
        path = tmp_path / f"{SAMPLE_TAG}.yaml"

        base_ns = time.time_ns()
        # Round down to a value whose float(seconds) representation is
        # dense enough that a +1ns neighbor collides -- empirically true
        # for any current real timestamp (see docstring); assert it rather
        # than assume it, so a future Python/OS float precision change
        # would fail this test loudly instead of silently testing nothing.
        os.utime(path, ns=(base_ns, base_ns))
        first_mtime = path.stat().st_mtime
        first_mtime_ns = path.stat().st_mtime_ns

        os.utime(path, ns=(base_ns + 1, base_ns + 1))
        second_mtime = path.stat().st_mtime
        second_mtime_ns = path.stat().st_mtime_ns

        assert first_mtime == second_mtime, (
            "precondition failed: expected the two 1ns-apart mtimes to "
            "collide under float st_mtime -- if this fails, float precision "
            "on this platform is finer than assumed and the "
            "motivating scenario doesn't reproduce here (not a cache bug)"
        )
        assert first_mtime_ns != second_mtime_ns, (
            "precondition failed: st_mtime_ns should still distinguish "
            "the two writes even though st_mtime collided"
        )

        calls = []
        real_read_manifest = read_manifest

        def spy_read_manifest(root, tag):
            calls.append(tag)
            return real_read_manifest(root, tag)

        os.utime(path, ns=(base_ns, base_ns))
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("turbohaul.manifest.read_manifest", spy_read_manifest)
            read_manifest_cached(tmp_path, SAMPLE_TAG)
        assert len(calls) == 1

        os.utime(path, ns=(base_ns + 1, base_ns + 1))
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("turbohaul.manifest.read_manifest", spy_read_manifest)
            read_manifest_cached(tmp_path, SAMPLE_TAG)
        assert len(calls) == 2, (
            "st_mtime_ns must still trigger a re-parse across a mtime pair "
            "that collides under float st_mtime -- got a cache hit instead"
        )


class TestListAndDelete:
    def test_list_manifests(self, tmp_path):
        write_manifest_atomic(tmp_path, make_manifest())
        write_manifest_atomic(tmp_path, make_manifest(model_tag="second"))
        tags = list_manifests(tmp_path)
        assert SAMPLE_TAG in tags
        assert "second" in tags
        assert len(tags) == 2

    def test_list_empty_dir(self, tmp_path):
        assert list_manifests(tmp_path) == []

    def test_list_ignores_hidden(self, tmp_path):
        write_manifest_atomic(tmp_path, make_manifest())
        # Write a hidden file (simulating partial-write leftover)
        (tmp_path / ".tmp_garbage.yaml").write_text("oops")
        tags = list_manifests(tmp_path)
        assert ".tmp_garbage" not in tags
        assert SAMPLE_TAG in tags

    def test_delete(self, tmp_path):
        write_manifest_atomic(tmp_path, make_manifest())
        assert delete_manifest(tmp_path, SAMPLE_TAG) is True
        assert delete_manifest(tmp_path, SAMPLE_TAG) is False  # idempotent

    def test_read_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path, "nope")


class TestFlagsToArgv:
    def test_basic_mapping(self):
        argv = flags_to_argv({"ctx_size": 4096, "n_gpu_layers": 100})
        assert "--ctx-size" in argv
        assert "4096" in argv
        assert "--n-gpu-layers" in argv
        assert "100" in argv

    def test_bool_true_no_value(self):
        argv = flags_to_argv({"mlock": True})
        assert argv == ["--mlock"]

    def test_bool_false_omitted(self):
        argv = flags_to_argv({"mlock": False})
        assert argv == []

    def test_denied_flag_rejected_at_argv(self):
        with pytest.raises(ManifestValidationError, match="blocked at argv-build"):
            flags_to_argv({"mmproj": "/etc/passwd"})

    def test_unknown_flag_rejected_at_argv(self):
        with pytest.raises(ManifestValidationError, match="blocked at argv-build"):
            flags_to_argv({"random_unknown_thing": "foo"})

    def test_snake_to_kebab(self):
        argv = flags_to_argv({"n_cpu_moe": True})
        assert "--n-cpu-moe" in argv
