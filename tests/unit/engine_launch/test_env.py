"""Unit tests for turbohaul.engine_launch.env.

Hermetic: no nvidia-smi, no real engine, no live GPU state (project test conventions /
known gotchas) -- every test builds its own argv/base_env and asserts on launch_env's
or preset_device_mismatch's return value directly, never on live process state.
"""
from turbohaul.engine_launch import launch_env, preset_device_mismatch

TEXT_ONLY_ARGV = ["--port", "11500", "-m", "/x/model.gguf", "--ctx-size", "4096"]
VISION_NONE_MAIN1_ARGV = [
    "--port", "11500", "-m", "/x/model.gguf",
    "--split-mode", "none", "--main-gpu", "1",
    "--mmproj", "/x/proj.gguf",
]
VISION_ROW_MAIN1_ARGV = [
    "--port", "11500", "-m", "/x/model.gguf",
    "--split-mode", "row", "--main-gpu", "1",
    "--mmproj", "/x/proj.gguf",
]
VISION_LAYER_MAIN1_ARGV = [
    "--port", "11500", "-m", "/x/model.gguf",
    "--split-mode", "layer", "--main-gpu", "1",
    "--mmproj", "/x/proj.gguf",
]
VISION_NONE_NO_MAIN_GPU_ARGV = [
    "--port", "11500", "-m", "/x/model.gguf",
    "--split-mode", "none",
    "--mmproj", "/x/proj.gguf",
]
BASE_ENV = {"PATH": "/usr/bin", "HOME": "/root"}


class TestLaunchEnv:
    def test_vision_single_card_gives_exact_cuda1(self):
        """The primary proof: a main_gpu 1 vision argv gives exactly 'CUDA1' --
        the VALUE, not just presence (a bad name silently falls back to CUDA0)."""
        env = launch_env(VISION_NONE_MAIN1_ARGV, BASE_ENV)
        assert env["MTMD_BACKEND_DEVICE"] == "CUDA1"
        assert env == {**BASE_ENV, "MTMD_BACKEND_DEVICE": "CUDA1"}

    def test_vision_row_split_gives_exact_cuda1(self):
        """Row split: row -> MTMD_BACKEND_DEVICE=CUDA<main_gpu>, same as none."""
        env = launch_env(VISION_ROW_MAIN1_ARGV, BASE_ENV)
        assert env["MTMD_BACKEND_DEVICE"] == "CUDA1"

    def test_vision_layer_split_sets_nothing(self):
        """Layer split: layer -> today's behaviour, base_env unchanged."""
        env = launch_env(VISION_LAYER_MAIN1_ARGV, BASE_ENV)
        assert env == BASE_ENV

    def test_text_only_returns_base_env_unchanged(self):
        env = launch_env(TEXT_ONLY_ARGV, BASE_ENV)
        assert env == BASE_ENV
        assert env is BASE_ENV or env == BASE_ENV  # same keys/values, nothing added/filtered

    def test_vision_single_card_no_main_gpu_defaults_to_cuda0(self):
        """--main-gpu absent -> llama.cpp's own compiled default (0), not left unset."""
        env = launch_env(VISION_NONE_NO_MAIN_GPU_ARGV, BASE_ENV)
        assert env["MTMD_BACKEND_DEVICE"] == "CUDA0"

    def test_repeated_main_gpu_flag_last_one_wins(self):
        """llama.cpp itself is last-wins for a
        repeated flag (arg.cpp), not first-wins -- _flag_value must match.
        Unreachable today (TurboHaul emits each flag once), but must hold if that
        ever changes."""
        argv = [
            "--port", "11500", "-m", "/x/model.gguf",
            "--split-mode", "none",
            "--main-gpu", "0", "--main-gpu", "1",
            "--mmproj", "/x/proj.gguf",
        ]
        env = launch_env(argv, BASE_ENV)
        assert env["MTMD_BACKEND_DEVICE"] == "CUDA1"

    def test_no_op_paths_return_a_new_dict_not_the_same_object(self):
        """launch_env's no-op paths must not alias
        base_env -- a caller mutating the returned dict must never mutate the
        caller's own env dict too. Covers all three no-op return sites: text-only,
        layer-split, and preset-already-set."""
        env = launch_env(TEXT_ONLY_ARGV, BASE_ENV)
        assert env == BASE_ENV
        assert env is not BASE_ENV

        env2 = launch_env(VISION_LAYER_MAIN1_ARGV, BASE_ENV)
        assert env2 == BASE_ENV
        assert env2 is not BASE_ENV

        preset_env = {**BASE_ENV, "MTMD_BACKEND_DEVICE": "CUDA1"}
        env3 = launch_env(VISION_NONE_MAIN1_ARGV, preset_env)
        assert env3 == preset_env
        assert env3 is not preset_env

    def test_operator_preset_is_never_overridden(self):
        """An operator preset in base_env wins, launch_env never overrides it,
        even when it names a different card than --main-gpu."""
        preset_env = {**BASE_ENV, "MTMD_BACKEND_DEVICE": "CUDA5"}
        env = launch_env(VISION_NONE_MAIN1_ARGV, preset_env)
        assert env == preset_env
        assert env["MTMD_BACKEND_DEVICE"] == "CUDA5"

    def test_operator_preset_kept_verbatim_when_it_matches(self):
        preset_env = {**BASE_ENV, "MTMD_BACKEND_DEVICE": "CUDA1"}
        env = launch_env(VISION_NONE_MAIN1_ARGV, preset_env)
        assert env == preset_env

    def test_no_mmproj_ignores_split_mode_and_main_gpu(self):
        argv = ["--split-mode", "none", "--main-gpu", "1"]  # no --mmproj
        env = launch_env(argv, BASE_ENV)
        assert env == BASE_ENV


class TestPresetDeviceMismatch:
    def test_no_preset_returns_none(self):
        assert preset_device_mismatch(VISION_NONE_MAIN1_ARGV, BASE_ENV) is None

    def test_preset_equals_main_gpu_returns_none(self):
        """Preset equal -> no warning."""
        preset_env = {**BASE_ENV, "MTMD_BACKEND_DEVICE": "CUDA1"}
        assert preset_device_mismatch(VISION_NONE_MAIN1_ARGV, preset_env) is None

    def test_preset_differs_from_main_gpu_names_both(self):
        """Preset differing -> exactly one warning naming both."""
        preset_env = {**BASE_ENV, "MTMD_BACKEND_DEVICE": "CUDA5"}
        msg = preset_device_mismatch(VISION_NONE_MAIN1_ARGV, preset_env)
        assert msg is not None
        assert "CUDA5" in msg
        assert "CUDA1" in msg

    def test_preset_differs_on_row_split_also_warns(self):
        """The rule applies on row placement too, not only split-mode none."""
        preset_env = {**BASE_ENV, "MTMD_BACKEND_DEVICE": "CUDA5"}
        msg = preset_device_mismatch(VISION_ROW_MAIN1_ARGV, preset_env)
        assert msg is not None
        assert "CUDA5" in msg and "CUDA1" in msg

    def test_preset_differs_on_layer_split_returns_none(self):
        """layer has no home card to compare against -- never warns, even with a preset
        present, per the 'single-card or row placement' scope."""
        preset_env = {**BASE_ENV, "MTMD_BACKEND_DEVICE": "CUDA5"}
        assert preset_device_mismatch(VISION_LAYER_MAIN1_ARGV, preset_env) is None

    def test_text_only_returns_none_even_with_preset(self):
        preset_env = {**BASE_ENV, "MTMD_BACKEND_DEVICE": "CUDA5"}
        assert preset_device_mismatch(TEXT_ONLY_ARGV, preset_env) is None

    def test_pure_no_io_does_not_log(self, caplog):
        """Invariant: pure functions, no I/O (logging included)."""
        preset_env = {**BASE_ENV, "MTMD_BACKEND_DEVICE": "CUDA5"}
        caplog.clear()
        preset_device_mismatch(VISION_NONE_MAIN1_ARGV, preset_env)
        assert caplog.records == []
