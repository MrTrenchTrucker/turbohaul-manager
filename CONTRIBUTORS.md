# Contributors

Turbohaul-Manager is a one-operator project with an AI-augmented build pipeline.
This file records the humans and named AI agents who shaped it.

## Project Lead

**MrTrench** -- architect, operator.
- Founder + sole human contributor on v0.2.1.
- Develops on a Blackwell-class GPU workstation.
- Authored the design intent for the FIFO + grace + IDLE_HOT + model-swap state machine
  (the "trucking dispatch" mental model that gives Turbohaul-Manager its name).
- Directed every load-bearing decision: licensing (MIT), safety guardrails posture, 5-min IDLE_HOT
  hot-hold, immediate model-swap on different model_tag, no copyleft deps.
- Reviewed and approved every commit landing in v0.2.1 in real time.

## How to contribute

This project is in early days. If you want to contribute:

1. Open an issue first describing what you want to change and why.
2. Keep PRs scoped to one concern (fix, feature, or doc -- not all three).
3. Run `pytest` locally; CI gate is 348/348 green at v0.2.1 ship.
4. Follow the existing comment style: no comments on the WHAT (the code says that),
   only on the WHY (intent, invariant, or non-obvious constraint).
5. New runtime dependencies must be MIT-compatible. No copyleft (GPL/AGPL/LGPL).
   Add the new entry to `THIRD_PARTY_NOTICES.md` in the same PR.

## Recognition

Future contributors will be added below in the order their first commit lands on `main`.

- **lmist** -- favicon set + project logo PNG, frontend integration (PR #1), started 2026-05

<!-- Add new contributors above this line, format:
- **Handle** -- one-line description, started YYYY-MM
-->
