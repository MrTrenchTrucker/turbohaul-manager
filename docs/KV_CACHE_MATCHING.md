# KV-Cache Matching — how reuse is decided, what breaks it, and how to see it

Turbohaul-Manager's speed story rests on one question asked at every restore: **does the saved KV state still match what the client is about to send?** This document explains exactly how that matching works at both levels (manager and engine), what client behavior preserves it, and how to diagnose the result. Companion docs: [ARCHITECTURE.md §4](../ARCHITECTURE.md) (the full KV lifecycle) and [TAGS_AND_IDENTITY.md](TAGS_AND_IDENTITY.md) (who owns which cache).

**TL;DR — does matching still matter?** Yes, completely — but only for **speed, never for safety**. A match means near-zero prefill (measured: 154,647 tokens reused with a 29-token prefill — 629× less prefill work). A mismatch means a re-prefill — and *nothing worse*: every mismatch path fails safe (skip, fresh prefill, or a bounded engine recovery). The era when a mismatch could poison saved state or crash the engine is closed by construction (canonical seam saves, the prefix-validity gate, checkpoint validation, 3-strikes quarantine). What remains entirely in the **client's** hands is whether you get the 629× path or pay the re-prefill.

---

## 1. Two match levels

| Level | Who | Granularity | Decides |
|---|---|---|---|
| **1 — Manager gate** | `kv_policy.resolve_kv` | **turn-level** (one hash per message) | *whether* a saved bin is offered to the engine at all |
| **2 — Engine reuse** | `llama-server` (`get_common_prefix`) | **token-level** (byte-exact) | *how much* of the restored state is actually reused |

They compose with identity (tags): **tags pick which bin; level 1 authorizes the restore; level 2 pays out the reuse.**

## 2. Level 1 — the manager's turn-level gate

**The prefix-hash chain.** At admission and at save time, the same function computes a rolling per-turn hash chain over the messages: `H_i = SHA256(H_{i-1} ⊕ role_i ⊕ content_i)` (`kv_policy.py` `_prefix_hash_chain`). Properties that matter to you:

- It hashes **role + content only**. Structured content (lists/dicts) is canonicalized with sorted-key JSON so multimodal blocks hash stably.
- It is **order-sensitive** — every turn's hash folds in all previous turns. Editing one character in an early message changes that turn's hash *and every hash after it*: the chain diverges at that index and the whole tail of the cache is unmatchable.
- Some turns are **hash-invisible**: an assistant turn whose payload is `tool_calls` (with null content) and the serialization details of `role: "tool"` results don't reach the hash. The manager knows this and treats tool-opaque spans with extra caution (dedicated guards exist), but the practical consequence is rule 6 below: serialize tool turns deterministically, because the chain can't catch drift there — only the engine can, at re-prefill cost.

**The restore decision.** Every candidate bin runs through `resolve_kv("restore", …)` in strict order: owner-identity match → saved tokens > 0 → incoming identity present → admission size recorded → admission chain present → **physics belt** (a bin *longer* than the incoming request can never restore — the engine would have to clear it) → **prefix validity**: the saved chain must be an element-wise prefix of the incoming chain (equal, or the incoming strictly extends it). Anything else → fresh prefill. Every skip rule exists because restoring on doubt used to cost more than recomputing; the worst case of every rule is **one fresh prefill**.

## 3. Level 2 — the engine's token-level reuse

The manager's restore `POST /slots/{id}?action=restore` loads the saved tokens back into the slot — **no matching happens at restore time**. On the *next decode*, the engine computes `n_past = get_common_prefix(restored_tokens, incoming_tokens)` — a pure token-by-token longest common prefix (one differing token ends it) — then applies a three-case policy on the stale tail (`stale = restored − matched`):

| Case | Engine behavior | Cost |
|---|---|---|
| `stale ≤ 0` — incoming covers the restored state | **STRICT EXTENSION** (the fast path; no state surgery) | near-zero: decode only the new tail (measured 521 ms / 29 tokens vs a 327 s full prefill) |
| `0 < stale ≤ n_rs_seq` — tiny stale tail | bounded rollback (`seq_rm`) | delta prefill |
| `stale > n_rs_seq` — real divergence | **ladder-rewind** to a validated `.ckpt` checkpoint at/before the divergence point, if available; else **CLEAR** + full re-prefill | delta from the checkpoint, or the full context (~330–420 s at 150–220k tokens — the cost this whole system exists to avoid) |

Why the bound exists: hybrid-recurrent models (MTP-class) keep state that *cannot be partially erased mid-sequence* beyond `n_rs_seq` per-token snapshots — whole-sequence erase is always legal, surgical mid-trims are not. The **checkpoint-ladder sidecars** (`<bin>.ckpt`, `TURBOHAUL_CKPT_SIDECAR`) exist precisely to give a cold-restored slot rewind points; each sidecar passes a validation gauntlet (magic, format, engine build id, model hash, per-entry size + content hashes) and *any* mismatch degrades gracefully to the slow path — never a crash.

The manager is deliberately honest about this division of labor: its restore log says *"engine determines actual n_past"*, and the engine's own log lines (`strict extension … FAST path` vs `large stale … CLEAR + reprefill`) are the ground truth for whether a match paid out.

## 4. How matching composes with tags

[Tags](TAGS_AND_IDENTITY.md) decide *which* bin a request may touch — one KV copy per `(session, role)`, owner-checked at rule 1 of the gate. Matching then decides *how much of that bin is reusable*. The two are independent failure axes: wrong/missing tags → the gate rejects at **owner match** (`restore-owner-mismatch` / `restore-no-anchor-for-identity`) and you prefill fresh in your own bin; right tags but mutated history → the gate rejects at **prefix validity** (`restore-diverged-fresh`) or the engine clears — same owner, no reuse. You need both right to ride the fast path.

## 5. What breaks matching — client rules

Each rule below traces to a shipped mechanism (not folklore). The manager defends against *systematic* mismatch sources automatically — think-scaffold divergence (save-side strip probe + preserved-reasoning-is-DATA rule), disposable-role tip pollution (canonical seam saves), compression (stale-marking) — but it cannot fix a client that renders unstable bytes.

1. **DO keep history append-only.** Never insert, delete, reorder, or edit earlier turns mid-session — the rolling chain breaks at the edit and everything after it is unmatchable.
2. **DON'T re-render dynamic content into old turns.** No timestamps, live counters, or rotating content re-rendered into already-sent messages. Earlier turns must be **byte-stable**, not just semantically stable.
3. **DO pick one think-handling convention and keep it.** Either always resend assistant turns think-stripped (the default the save probe is built for), or always preserve `reasoning_content` (the preserved-reasoning rule then keeps those blocks in the save). Never alternate.
4. **DO keep the system prompt / first message byte-stable for the session's life.** Turn 0 is doubly load-bearing: for tag-less clients its fingerprint *is* the identity (mutate it → new identity → every saved bin orphaned), and it is position 0 of the prefix (mutate it → zero reuse). No manager defense compensates.
5. **DO label compression passes** (`is_compression` + `session_id`). Labeled, the session's saved main state is stale-marked *at admission* so post-compression renders never fight pre-compression bins; unlabeled, the mismatch is only detected after the reuse is already lost.
6. **DO serialize tool-call turns and tool results deterministically** — stable key order, stable formatting, identical bytes on every resend. These spans are hash-invisible to the level-1 chain; nondeterminism there is caught only by the engine, as a re-prefill.
7. **DO send identity labels every request** (see [TAGS_AND_IDENTITY.md](TAGS_AND_IDENTITY.md)) — labels drive bin keying, the persistence gates, and the compression contract.
8. **DON'T expect the manager to fix an inconsistent client.** The guard rails safe-degrade to full reprocessing — they prevent wrong answers and crashes, not slow turns. Only byte-stable behavior gets the 629× path.

## 6. Diagnosing match results

Six surfaces, from decision intent down to ground truth:

**`resolved_from` provenance** — every restore decision logs one token. The important ones:

| Signal | Verdict |
|---|---|
| `restore-prefix-valid`, `wave-return-clean-restore`, `wave-return-shadow-restore`, `warm-force-clean-restore` | **match success** — a bin was restored |
| `warm-native-reuse-longer`, `warm-vram-fresher-skip`, `warm-anchor-natural-skip`, `warm-force-gated-native-reuse` | **intentional non-restore** — the live VRAM state already covers the request; not restoring IS the correct outcome |
| `restore-diverged-fresh` | **match failure** — an earlier turn changed (see rules 1/2/5) |
| `restore-physics-belt-saved-longer` | **match failure** — saved state overshoots the incoming request |
| `restore-owner-mismatch`, `restore-no-anchor-for-identity` | not a match failure — correct isolation (another identity's bin) |
| `restore-no-bins` | first contact for this identity — nothing to match yet |
| `restore-no-incoming-size/-chain`, `restore-no-identity` | wiring/guard skips — fail-safe, investigate the client integration |
| `restore-post-failed` | mechanical failure of the restore POST, not a matching miss |

**The `KV_RESTORE` diag line** (one per cold decision): `chosen=` (fresh / chose_clean / chose_shadow), `common_prefix_turns=` vs `incoming_turns=` (how deep the turn-level match went), `divergence_pos=`, plus KV-pressure fields. **`/status.kv_classifier`**: the running counters — `wave_return_restores` (cold successes) and `forced_clean_restores` (warm), plus the per-event-type tallies. **Engine log** (ground truth): `restored slot: strict extension (stale=…) FAST path` = the match paid out; `large stale=… > n_rs_seq…, CLEAR + reprefill` = it didn't; `ladder-rewind to checkpoint` = partial salvage. A small `prompt eval` count right after a restore is the definitive receipt. **LOAD_VERIFY** `kv_restore` records report the engine-actual `n_past` (V1 caveat: with no expected value threaded yet, `kv_restore_ok` only proves a numeric read — depth verification is a queued follow-up). **FE**: the blue `kv_restore` engine-op pill during the operation, and the prefill bar's *restored-from-KV* numerator showing recovered context live.

Quick recipe: send your follow-up, then check the engine log for `strict extension` and `/status.kv_classifier.wave_return_restores` for the bump. If you see `restore-diverged-fresh` instead, diff the exact bytes of your resent history against what you sent before — rules 1–6 above name every known cause.

## 7. Why this document exists

Matching failures were this project's hardest bug class: silent byte-drift between saved state and resent renders once caused full-context re-prefills on every turn, cross-conversation cache collisions, and (at worst) restore-linked engine aborts. That war produced the current design: **matching is now validated turn-level before any restore, verified token-level by the engine, guarded at every save seam, and every failure mode degrades to a fresh prefill.** The receipts live in the v0.6.0 [CHANGELOG](../CHANGELOG.md) entry.
