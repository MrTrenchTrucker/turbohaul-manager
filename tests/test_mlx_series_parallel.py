"""Series-parallel on MLX: derive_mlx_parallel maps mlx_lm's concurrency flags
onto SidecarHandle.parallel, which is Turbohaul's fan-out cap.

Context: the manager computes
    n_parallel = max(1, getattr(handle, "parallel", 1))
and only enters _fan_out_and_drain when n_parallel > 1. mlx_spawn used to pin
parallel=1 ("mlx-lm is single-slot per process"), which confined MLX to the
single-series residency mode.

That comment is stale: mlx_lm.server is a ThreadingHTTPServer driving a real
BatchGenerator (completion_batch_size=--decode-concurrency,
prefill_batch_size=--prompt-concurrency). Measured on Qwen3.5-9B-MLX-8bit with
--decode-concurrency 4: four simultaneous requests finished in 2.0x a single
request's wall time (serial would be ~4x).

The one hard constraint is mlx_lm's own batchability rule in
ModelProvider._load: `is_batchable = draft_model is None and ...`. With
speculative decoding every request goes through _serve_single, so a width > 1
would have the manager dispatch riders the engine then serializes.
"""

import pytest

from turbohaul.mlx_spawn import derive_mlx_parallel


class TestDefaultsToSerial:
    def test_no_flags_is_one(self):
        """Byte-identical to the old hardcoded parallel=1 for existing manifests."""
        assert derive_mlx_parallel({}) == 1

    def test_unrelated_flags_are_one(self):
        assert derive_mlx_parallel({"max_tokens": 4096, "temp": 0.7}) == 1

    def test_explicit_one_is_one(self):
        assert derive_mlx_parallel({"decode_concurrency": 1}) == 1


class TestDerivesWidth:
    def test_decode_concurrency_sets_width(self):
        assert derive_mlx_parallel({"decode_concurrency": 4}) == 4

    def test_prompt_concurrency_sets_width(self):
        assert derive_mlx_parallel({"prompt_concurrency": 3}) == 3

    def test_takes_the_max_of_both(self):
        assert derive_mlx_parallel(
            {"decode_concurrency": 4, "prompt_concurrency": 2}
        ) == 4
        assert derive_mlx_parallel(
            {"decode_concurrency": 2, "prompt_concurrency": 8}
        ) == 8


class TestSpeculativeDecodingClamps:
    """mlx_lm sets is_batchable=False when a draft model is configured, so the
    engine serializes regardless of the concurrency flags."""

    def test_draft_model_clamps_to_one(self):
        assert derive_mlx_parallel(
            {"draft_model": "mlx-community/Qwen3-0.6B-4bit",
             "decode_concurrency": 8}
        ) == 1

    def test_empty_draft_model_does_not_clamp(self):
        assert derive_mlx_parallel(
            {"draft_model": "", "decode_concurrency": 4}
        ) == 4


class TestDegenerateValues:
    """Never return < 1 -- the manager would divide the fan-out cap by it."""

    @pytest.mark.parametrize("bad", [0, -1, -99])
    def test_non_positive_floors_to_one(self, bad):
        assert derive_mlx_parallel({"decode_concurrency": bad}) == 1

    @pytest.mark.parametrize("bad", ["4", None, "abc", 3.9])
    def test_non_int_never_raises(self, bad):
        """A manifest is user input; a bad value must not kill the spawn."""
        assert derive_mlx_parallel({"decode_concurrency": bad}) >= 1

    def test_string_digit_is_coerced(self):
        assert derive_mlx_parallel({"decode_concurrency": "4"}) == 4


class TestHandleWiring:
    """The value must actually reach SidecarHandle.parallel -- that field, not
    max_parallel_sidecars, is what gates _fan_out_and_drain."""

    def test_spawn_sets_handle_parallel(self, monkeypatch, tmp_path):
        import turbohaul.mlx_spawn as ms

        monkeypatch.setattr(ms, "_check_mlx_preconditions", lambda python: None)

        captured = {}

        class FakeProc:
            pid = 4242

            def poll(self):
                return None

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return FakeProc()

        handle = ms.mlx_spawn(
            port=11500,
            model_tag="tag",
            model_repo="",
            model_path=str(tmp_path),
            mlx_flags={"decode_concurrency": 4},
            popen_factory=fake_popen,
        )
        assert handle.parallel == 4
        # and the flag still reaches the child argv
        assert "--decode-concurrency" in captured["cmd"]

    def test_spawn_defaults_parallel_one(self, monkeypatch, tmp_path):
        import turbohaul.mlx_spawn as ms

        monkeypatch.setattr(ms, "_check_mlx_preconditions", lambda python: None)

        class FakeProc:
            pid = 4242

            def poll(self):
                return None

        handle = ms.mlx_spawn(
            port=11500,
            model_tag="tag",
            model_repo="",
            model_path=str(tmp_path),
            mlx_flags={},
            popen_factory=lambda cmd, **kw: FakeProc(),
        )
        assert handle.parallel == 1
