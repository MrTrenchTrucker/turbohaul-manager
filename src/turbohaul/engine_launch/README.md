# engine_launch

When TurboHaul starts an engine (a llama-server process) for a model, this module works out the environment that
process starts with.

Most of the time that is simply the manager's own environment, passed through unchanged. The one addition today is for
vision models. A vision model loads an image projector next to the model. The engine puts that projector on the card
named by the environment variable `MTMD_BACKEND_DEVICE`, and if the variable is not set it uses the first GPU, even when
the model itself is on another card. So for a vision model placed on one card (or split by row, where `--main-gpu` is the home
card), this module sets `MTMD_BACKEND_DEVICE=CUDA<card>` to the same card as `--main-gpu`. The projector then sits on the
model's own card.
Two exceptions: a model split by layer across every card keeps the engine's default (the first GPU), and a value the operator
already set is never overridden.

Without it, a vision model placed on GPU 1 still puts its projector on GPU 0. That uses memory on a card the model was not
placed on, and when GPU 0 is full, the model can fail to load.

Public interface: two functions. `launch_env(argv, base_env)` returns the full environment for one launch.
`preset_device_mismatch(argv, base_env)` returns a one-line warning when an operator's preset projector card differs from
the model's card, and nothing otherwise; `spawn_sidecar` writes that warning to the log. The process itself is started by
`subprocess_mgr.spawn_sidecar`; this module never starts anything and never logs.
