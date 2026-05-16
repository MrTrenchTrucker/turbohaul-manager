# Turbohaul-Manager

Ollama-shape inference manager using TurboQuant llama.cpp.

Fleet-internal local inference one-stop-shop. FIFO queue, grace + idle hot-load, BYOM blob store, React+Vite FE.

**Architecture:** see [ARCHITECTURE.md](./ARCHITECTURE.md) (v0.2 lock).
**Phase tracker:** see [TODO.md](./TODO.md).

## Status

- Phase 0 (Forgejo + license audit): ✓ DONE 2026-05-16
- Phase 1 (Architecture + RBSRS critique): ✓ DONE 2026-05-16 (v0.2 commit 5a001eb)
- Phase 2 (Core queue + slot manager + supervision): IN PROGRESS

## License

TBD. Inference backend uses llama-server built from Tom's TurboQuant fork of llama.cpp (MIT). Ollama-compatible HTTP API surface (nominative use only).
