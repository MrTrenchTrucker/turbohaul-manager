# ADR-001: Adopt the module system one fix at a time; first module engine_launch

**Status:** accepted
**Date:** 2026-09-25

## Context
TurboHaul has no module registry. manager.py is a very large file (over 24,000 lines) and ARCHITECTURE.md is a long
single document. A vision model could not load because the engine put its image projector on GPU 0 while the model was
placed on GPU 1, and the failed load was returned to the caller as an error instead of waiting in the queue. The fix was
built as part of TurboHaul's first module system.

## Decision
- Adopt a module registry: a repo-root `modules.toml`, one module per fix, and a module check (`tools/sop_check.py`) that
  looks at registered modules (plus legacy code importing past a module's entry point). Everything not in a registered
  module is legacy until a later change moves it in, on touch, never all at once.
- The first module is `engine_launch` (src/turbohaul/engine_launch): the environment each engine process starts with,
  including the projector's card.
- The requeue half of the same fix stays in manager.py for now. It lives in the admission and dispatch code, which is far
  too tangled to extract as part of a bug fix.
- `architecture_word_budget` (in `modules.toml`) is a no-growth ratchet: `tools/sop_check.py` fails when ARCHITECTURE.md
  has more words than the budget, and the budget may be lowered but not raised. Trimming the document to a much smaller
  size is its own later piece of work.

## Reasons
- The projector rule is a small, pure decision with a real interface (argv and environment in, environment out), and it
  is exactly where the bug was: the right shape for a first module.
- One fix at a time keeps a stable production system stable. The rest of the repo is converted only when work touches it.

## Consequences
- Any change to how an engine's environment is built now goes through engine_launch and its card.
- Legacy code must import engine_launch only through its package entry point; the check fails a deep import.
- Future fixes that touch other areas register their own modules the same way.
