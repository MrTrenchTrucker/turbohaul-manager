# Role Tags & Session Identity — how Turbohaul Manager knows who's calling (and what it does when it doesn't)

Turbohaul Manager keeps **one precomputed KV-cache copy per (session, role)** and reuses it whenever physics allows (see [ARCHITECTURE.md §4](../ARCHITECTURE.md)). To do that, it wants to know *who* each request is. This guide covers the tagging contract for multi-agent setups, **and** the fallback ladder that makes Turbohaul work out of the box with clients that send no tags at all — a plain chat app, an OpenAI SDK script, or any Ollama-aware tool.

**TL;DR:** tags unlock the full role-aware KV contract (isolation + differentiated persistence + save/restore across model swaps). No tags still gets you warm-slot reuse and grace-window follow-ups out of the box — Turbohaul "guesses" a stable identity from what it can see. Add an explicit `thread_id` (one field) and untagged conversations also get KV saved at the unload seam and restored when they return.

---

## 1. The tagging contract (`client_meta`)

Identity fields ride the request payload — **top-level keys win, with a nested `client_meta` object as fallback** — on both `POST /v1/chat/completions` and `POST /api/chat`. None of these fields are ever forwarded to the inference engine.

| Field | Type | Meaning |
|---|---|---|
| `session_id` | string | The conversation/agent-session this request belongs to. Required for role-keyed KV. |
| `is_main` | bool | This is the primary agent's turn. Its KV is **always saved** and restored when it returns. |
| `is_sub_agent` | bool | A spawned worker. Its KV is isolated and **thrown away** after the job (unless `save_kv`). |
| `is_curator` | bool | A background reviewer. Gets its own isolated bin by default; its KV is never saved, and it can never overwrite main's saved copy. |
| `is_compression` | bool | A context-compression pass. Marks the session's saved main KV stale so the next main turn re-anchors at the new compressed baseline. |
| `save_kv` | bool | Per-role persistence override (see §3). |
| `role` | string | Back-compat literal (`"sub-agent"`, `"curator"`, …). Prefer the boolean flags. |
| `thread_id` | string | (top-level) Explicit conversation identity; skips the guess ladder entirely. |

Example (OpenAI surface):

```json
{
  "model": "example-27b-mtp",
  "messages": [...],
  "thread_id": "my-agent-main",
  "client_meta": {
    "session_id": "sess-2026-07-10-001",
    "is_main": true
  }
}
```

**Flag overlap is fine.** Flags are not required to be mutually exclusive; they resolve by a fixed priority: `is_curator` > `is_compression` > `is_sub_agent` > `is_main`. A double-labelled curator resolves to curator, never sub-agent.

### 1.1 Where exactly Turbohaul looks in your request

Turbohaul reads identity from **two places in the JSON body**, in this order:

1. **Top-level keys** on the request payload itself — checked **first**; a non-null top-level value always wins.
2. **The nested `client_meta` object** — the fallback when the top-level key is absent/null.

| You want to set… | Top-level path (wins) | Fallback path |
|---|---|---|
| session | `$.session_id` | `$.client_meta.session_id` |
| role flags | `$.is_main`, `$.is_sub_agent`, `$.is_curator`, `$.is_compression` | `$.client_meta.is_main`, … |
| persistence toggle | `$.save_kv` | `$.client_meta.save_kv` |
| conversation identity | `$.thread_id` | *(top-level only)* |

Nothing else is scanned — headers, query strings, and message content are **never** used for identity. The same two-location rule applies on both `POST /v1/chat/completions` and `POST /api/chat`.

**Copy-paste examples.** Raw HTTP (both locations shown — pick either):

```bash
curl http://localhost:11401/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "example-27b-mtp",
  "thread_id": "assistant-alice",
  "client_meta": { "session_id": "sess-001", "is_main": true },
  "messages": [{"role": "user", "content": "hello"}]
}'
```

OpenAI SDKs reject unknown top-level fields client-side, so pass them through `extra_body`:

```python
client.chat.completions.create(
    model="example-27b-mtp",
    messages=[...],
    extra_body={
        "thread_id": "assistant-alice",
        "client_meta": {"session_id": "sess-001", "is_main": True},
    },
)
```

Ollama-style (`/api/chat`) accepts the same fields at the same two locations alongside the usual `model`/`messages`/`keep_alive`.

**Verify it landed** (§5): grep the manager log for `R2B_REQ_IDENTITY` right after your test call — if `session_id` and `resolved_class` show what you sent, the placement is correct.

**Bin identity.** Tagged requests key their KV as `(session_id, role)` — one copy per pair. Sub-agents and curators additionally get a conversation fingerprint so concurrent same-role siblings in one session stay in distinct bins.

## 2. What each role gets

| Role | KV while running | KV after the job | Why |
|---|---|---|---|
| **main** | native VRAM reuse between tool calls | **saved** at the model-swap seam (system RAM), persisted to SSD after full unload, **restored on return** — only the new suffix is prefilled | the main agent's context is the product; it must never silently recompute |
| **sub-agent** | native VRAM reuse on its own isolated bin | **thrown away** (unless `save_kv: true`) | disposable by contract; can never muddy main, the curator, or sibling sub-agents |
| **curator** | its own isolated per-conversation bin (default); an opt-in route (`TURBOHAUL_CURATOR_REUSE_MAIN`, off by default) lets a labeled curator restore main's saved state read-only instead | **thrown away**; structurally prevented from overwriting main's saved copy either way | reviews must not corrupt the thing they review |
| **compression** | ordinary serve | not saved; marks main's saved bin stale | after compression the old prefix is wrong by design |

**Give every spawned sub-agent its own `session_id`.** The recommended pattern (used by the reference agent integration) is `{parent_session_id}-sub-<nonce>` per spawn — distinct sessions mean distinct bins, which is what makes context bleed between concurrent agents structurally impossible.

## 3. The `save_kv` override

`client_meta["save_kv"]` (bool) is the per-request control over disposable-role persistence — deliberately a request field, never an environment variable, so it is runtime-switchable per role from your client's settings. Absent → sub-agent/curator/compression KV is not saved (main always saves). Plenty of VRAM/RAM and long-lived sub-agents? Send `save_kv: true` on that role and its KV starts being kept — no restart, no rebuild. (One configuration-level exception: with the opt-in `TURBOHAUL_CURATOR_REUSE_MAIN` route enabled, curator saves are forced off regardless of the toggle.)

## 4. No tags? The guess ladder

Untagged clients work out of the box. Turbohaul derives a stable identity in this order:

1. **Explicit `thread_id`** in the payload — used as-is.
2. **IP + first-message fingerprint** — for tag-less clients at single-model residency: `agent-ip-<ip>-auto-<hash>` (the hash covers the model tag + the first message), degrading to bare `agent-ip-<ip>` when no usable first message exists. Distinguishes different personas behind one IP while keeping one conversation's follow-ups together.
3. **Prompt prefix-hash** (`auto-<hash>`) — a hash of the model tag + the first ~256 words of the prompt. Because a growing conversation keeps the same prefix, **follow-up turns map to the same identity** with zero client cooperation.

What each level gets:

| Client sends | Warm-slot reuse + grace follow-ups | KV saved at unload seam + restored after a swap | Role-differentiated handling |
|---|---|---|---|
| nothing (guess ladder, `agent-ip-*`/`auto-*`) | ✅ | ❌ — auto-derived identities are deliberately treated as walk-in/disposable at the persistence seam | ❌ |
| explicit `thread_id` | ✅ | ✅ per conversation | ❌ |
| `thread_id` + tags | ✅ | ✅ | ✅ full §2 contract |

So a completely naive client is fast between follow-ups on the live model; adding **one field** (`thread_id`) buys cross-swap KV persistence; adding tags buys the full multi-agent contract. By design, unlabeled traffic keys into a different bin namespace than tagged sessions, so well-formed untagged traffic cannot collide with, restore, or reset a tagged session's bins.

**How your IP is used.** Turbohaul records the source IP of every request — that is how it knows *who sent what* even when the request carries nothing else. For tag-less clients the IP anchors the derived identity (rung 2), so two different machines talking to the same server never blur into one conversation; and the IP is displayed for the operator (the identity strip on the Dashboard, the `R2B_REQ_IDENTITY` log, `/status`). It is **not** authentication — Turbohaul trusts its network perimeter (see `ARCHITECTURE.md` §8) — and it never appears on the redacted WebSocket event feed.

This is not theoretical: non-agent clients (plain headless CLI tools driving the OpenAI surface) have been validated against Turbohaul — they ride the identity ladder and reuse warm state across turns without any integration work.

## 5. Verifying your tags land

- **Log:** every admitted request emits one greppable line — `R2B_REQ_IDENTITY {"ip":…,"model_tag":…,"session_id":…,"is_main":…,"resolved_class":…,"thread_id":…}`. Grep for it while testing your integration.
- **API:** `GET /status` → `request_identity` shows the last request's resolved identity.
- **UI:** the Dashboard's residents box renders the identity strip (`ip · model · ROLE · session`) live per slot.

If `resolved_class` shows the role you intended and `session_id` is non-null, the full KV contract in §2 is in effect.

## 6. Tags × cache matching — two different jobs

Tags and **cache matching** are independent axes, and you need both for the fast path:

- **Tags decide *which* KV copy** a request may touch (one bin per session + role, owner-checked before any restore).
- **Matching decides *how much* of that copy is reusable**: the saved state must still be a valid prefix of what you're about to send — turn-level at the manager's gate, byte/token-level at the engine.

Wrong tags → you prefill fresh in your *own* bin (isolation working as intended). Right tags but mutated history (edited earlier turns, unstable system prompt, nondeterministic tool serialization) → same owner, **no reuse**. The full matching contract — what must stay byte-stable, what the manager defends automatically, and how to diagnose a miss — lives in [KV_CACHE_MATCHING.md](KV_CACHE_MATCHING.md).

## 7. Gotchas

- **`session_id` is required for role keying.** Flags without a session fall back to raw thread-identity behavior (still safe, just not role-differentiated).
- **Send the flags, not just `role` strings.** The boolean flags are the contract; the literal `role` field exists for back-compat.
- **One session ID per sub-agent.** Reusing the parent's `session_id` for a spawned sub-agent would make it a sibling of main in the same session — mint a distinct child session instead (§2).
- **Compression must be labeled.** An unlabeled compression pass looks like an ordinary turn; label it `is_compression` so the stale-marking contract fires and the next main turn re-anchors cleanly.
