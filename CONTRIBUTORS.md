# Contributors

Turbohaul-Manager is an independently developed project with an AI-augmented build pipeline.
This file records the humans and named AI agents who shaped it.

## Project Lead

**MrTrench** -- lead architect, operator, maintainer.
- Founder and lead maintainer.
- Developed on a dual-GPU workstation (Ryzen 9 9900X, 128 GiB DDR5, 2x RTX PRO 4000 Blackwell SFF Edition).
- Authored the design intent for the FIFO + grace + IDLE_HOT + model-swap state machine
  (the "trucking dispatch" mental model that gives Turbohaul-Manager its name).
- Directed every load-bearing decision: licensing (MIT), safety guardrails posture, the IDLE_HOT
  hot-hold, immediate model-swap on different model_tag, no copyleft deps.
- Reviewed and approved every commit landing in v0.2.1 in real time.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

## Recognition

Future contributors will be added below in the order their first commit lands on `main`.

- **lmist** -- favicon set + project logo PNG, frontend integration (PR #1), started 2026-05
- **rahlquist** -- offline engine-build fix: disable llama.cpp prebuilt-UI download
  in the Dockerfiles (GitHub PR #8), started 2026-07

- **Sahil-SS9** -- started 2026-07 (GitHub PRs #9-#12)
  - Engine lifecycle hardening: every spawned engine's identity (pid, port,
    start time) is recorded, and cleanup only ever touches a process that
    matches a record, so an unrelated process on a managed port is reported
    rather than killed.
  - Test determinism and contract accuracy: replaced a timing-dependent
    worker-loop assertion with an event-backed one, and aligned the keep-alive
    parser tests with the documented capped pin-warm semantics.
  - Main-lane admission reservation: an interactive request is admitted ahead
    of queued auxiliary work once the active engine frees, as a queue-level
    reservation that never interrupts a running engine.
  - Reported, not authored (GitHub issue #21): `cache_reuse` is accepted by a
    manifest and then silently does nothing on multimodal models. The report
    was correct; the cause turned out to be a limitation in llama.cpp rather
    than the KV cache type.
  - Reported, not authored (GitHub issue #15): `GET /models` (without the
    `/v1` prefix) returned 404, so OpenAI-compatible clients mis-detected a
    model's context length and silently fell back to a wrong default. Our
    own investigation found that `/v1/models` also served only four fields
    and no context length, and the same defect on two further surfaces the
    report did not name -- `/api/tags` and `/api/show` -- so the reach was
    wider than filed.
  - Reported, not authored (GitHub issue #24): the OpenAI `reasoning_effort`
    request parameter was accepted and then silently discarded, reaching no
    engine setting at all, so a client asking for more or less reasoning got
    neither and was never told (the defect is addressed in this release).
  - Reported, not authored (GitHub issue #19): asked whether a manifest
    carrying an unknown top-level field could fail to load. The question was
    put conditionally rather than as a claim, and the field it named,
    `measured_vram`, proved never to have existed in this repository -- but
    the investigation it prompted found a real and separate defect, which is
    that a schema-invalid manifest crashed with an uncaught 500 on every
    inference route (the defect is addressed in this release).
  - Reported, not authored (GitHub issue #20): the frontend `ResidentModel`
    interface was missing `engine_op`, breaking the typecheck.

<!-- Add new contributors above this line, format:
- **Handle** -- one-line description, started YYYY-MM
-->
