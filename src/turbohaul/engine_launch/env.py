"""engine_launch.env -- the per-launch environment for one llama-server engine.

Both functions are pure: no I/O (logging included), no GPU access, no reading of live
state (a module invariant). `spawn_sidecar` is the caller that acts on their output --
it applies `launch_env`'s dict as `env=`, and logs `preset_device_mismatch`'s result at
WARNING when it is not None. This module never logs.

The CUDA<N> naming and the --main-gpu/CUDA-device-order equivalence this relies on are
verified at source (clip.cpp, ggml-backend-reg.cpp, ggml-cuda.cu,
llama.cpp's default device-selection path) -- and hold only under the exact
conditions (no --rpc, no --device).
"""

_MMPROJ_FLAG = "--mmproj"
_SPLIT_MODE_FLAG = "--split-mode"
_MAIN_GPU_FLAGS = ("--main-gpu", "-mg")
_PRESET_VAR = "MTMD_BACKEND_DEVICE"
_SINGLE_CARD_SPLIT_MODES = ("none", "row")


def _flag_value(argv: list[str], names: tuple[str, ...]) -> str | None:
    """Return the value following the LAST occurrence of any flag in `names`, or None --
    matching llama.cpp's own last-wins behaviour for a repeated flag (arg.cpp), not
    first-wins. Unreachable today (TurboHaul emits each flag once), kept correct for
    when it stops being unreachable."""
    value = None
    for i, tok in enumerate(argv):
        if tok in names and i + 1 < len(argv):
            value = argv[i + 1]
    return value


def _target_card(argv: list[str]) -> str:
    """The CUDA<N> name for this launch's --main-gpu (default 0, matching llama.cpp's own
    compiled default in llama-model.cpp when --main-gpu is absent from argv)."""
    main_gpu = _flag_value(argv, _MAIN_GPU_FLAGS)
    if main_gpu is None:
        main_gpu = "0"
    return f"CUDA{main_gpu}"


def launch_env(argv: list[str], base_env: dict[str, str]) -> dict[str, str]:
    """The full environment for one engine launch, built from its final argv and the
    manager's environment. `base_env` is never filtered: every key in it reaches the
    engine (existing behaviour); this only ever ADDS one key, and only when no
    value is already present for it.

    A vision engine (--mmproj present) placed so that one card is unambiguously its
    home (--split-mode none, or --split-mode row where --main-gpu still names the card
    holding intermediate results and the KV cache) gets MTMD_BACKEND_DEVICE=CUDA<N> for
    that card's index N -- UNLESS the operator already set MTMD_BACKEND_DEVICE in
    base_env, which always wins.

    A layer-split vision engine (--split-mode layer, or any other/absent split-mode) has
    no single home card (the model spans every card), so nothing is added: the engine
    keeps its own default (the first GPU).

    A text-only engine (no --mmproj) returns base_env completely unchanged.
    """
    if _MMPROJ_FLAG not in argv:
        return dict(base_env)
    if _flag_value(argv, (_SPLIT_MODE_FLAG,)) not in _SINGLE_CARD_SPLIT_MODES:
        return dict(base_env)
    if _PRESET_VAR in base_env:
        return dict(base_env)
    return {**base_env, _PRESET_VAR: _target_card(argv)}


def preset_device_mismatch(argv: list[str], base_env: dict[str, str]) -> str | None:
    """One-line warning text when an operator's preset MTMD_BACKEND_DEVICE names a
    different card than --main-gpu on a single-card or row placement with a projector;
    otherwise None. `launch_env` never overrides the preset -- this is the
    visibility half of that rule: without it, an operator override that
    silently reintroduces the crash this variable prevents would go unnoticed.
    """
    if _MMPROJ_FLAG not in argv:
        return None
    if _flag_value(argv, (_SPLIT_MODE_FLAG,)) not in _SINGLE_CARD_SPLIT_MODES:
        return None
    preset = base_env.get(_PRESET_VAR)
    if preset is None:
        return None
    expected = _target_card(argv)
    if preset == expected:
        return None
    return (
        f"engine_launch: operator preset {_PRESET_VAR}={preset!r} differs from "
        f"--main-gpu's card ({expected!r}); the preset wins and is not overridden"
    )
