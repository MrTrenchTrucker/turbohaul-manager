# KV-Cache Matching — how reuse is decided, what breaks it, and how to see it

Turbohaul-Manager's speed story rests on one question asked at every restore: **does the saved KV state still match what the client is about to send?** This document explains exactly how that matching works at both levels (manager and engine), what client behavior preserves it, and how to diagnose the result. Companion docs: [ARCHITECTURE.md §4](../ARCHITECTURE.md) (the full KV lifecycle) and [TAGS_AND_IDENTITY.md](TAGS_AND_IDENTITY.md) (who owns which cache).

**TL;DR — does matching still matter?** Yes, completely — but only for **speed, never for safety**. A match means near-zero prefill (only the new tail is prefilled). A mismatch means a re-prefill — and is designed to cost *nothing worse*: the mismatch paths skip the restore, fall back to a fresh prefill, or take a bounded engine recovery. The protections against a mismatch poisoning saved state or crashing the engine are canonical seam saves, the prefix-validity gate, checkpoint validation and 3-strikes quarantine. What remains entirely in the **client's** hands is whether you get the fast path or pay the re-prefill.

---

## 1. Two match levels

| Level | Who | Granularity | Decides |
|---|---|---|---|
| **1 — Manager gate** | `kv_policy.resolve_kv` | **turn-level** (one hash per message) | *whether* a saved bin is offered to the engine at all |
| **2 — Engine reuse** | `llama-server` (`get_common_prefix`) | **token-level** (byte-exact) | *how much* of the restored state is actually reused |

They compose with identity (tags): **tags pick which bin; level 1 authorizes the restore; level 2 pays out the reuse.**

## 2. Level 1 — the manager's turn-level gate

**The prefix-hash chain.** At admission and at save time, the same function computes a rolling per-turn hash chain over the messages: `H_i = SHA256(H_{i-1} + role_i + content_i + tool-call fields)` (NUL-delimited concatenation) (`kv_policy.py` `_prefix_hash_chain`). Properties that matter to you:

- It hashes **role + content**, plus tool-call identity. An assistant turn's `tool_calls` and legacy `function_call` payloads — canonicalized as sorted-key JSON — and a `role: "tool"` result's `tool_call_id` are folded into that turn's hash. Two requests differing only in which tool was called, with what arguments, or which call a result answers therefore produce different chains and cannot mis-match. A turn carrying none of those three fields hashes exactly as it did before they were added, so an ordinary text-only conversation's chain is unaffected.
- Structured content (lists/dicts) is canonicalized with sorted-key JSON so multimodal blocks hash stably.
- It is **order-sensitive** — every turn's hash folds in all previous turns. Editing one character in an early message changes that turn's hash *and every hash after it*: the chain diverges at that index and the whole tail of the cache is unmatchable.
- What the chain still cannot see is how a chat template *renders* a tool span into tokens. The manager classifies a tool-call turn, a tool result (or legacy `function` turn), and an empty-content turn as **tool-opaque**. An optional guard (`TURBOHAUL_TOOLTAIL_RESTORE_SKIP`, off by default) skips a restore whose divergent tail contains one — the whole covered span when `TURBOHAUL_TOOLTAIL_SCAN_COVERED` is also on — rather than risk an undetected token drift; the practical consequence is rule 6 below.

**The restore decision.** Every candidate bin runs through `resolve_kv("restore", …)` in strict order: owner-identity match → saved tokens > 0 → incoming identity present → admission size recorded → admission chain present → **physics belt** (a bin *longer* than the incoming request can never restore — the engine would have to clear it) → **prefix validity**: the saved chain must be an element-wise prefix of the incoming chain (equal, or the incoming strictly extends it). Anything else → fresh prefill. Every skip rule exists because restoring on doubt used to cost more than recomputing; the worst case of every rule is **one fresh prefill**.

## 3. Level 2 — the engine's token-level reuse

The manager's restore `POST /slots/{id}?action=restore` loads the saved tokens back into the slot — **no matching happens at restore time**. On the *next decode*, the engine computes `n_past = get_common_prefix(restored_tokens, incoming_tokens)` — a pure token-by-token longest common prefix (one differing token ends it) — then applies a three-case policy on the stale tail (`stale = restored − matched`):

| Case | Engine behavior | Cost |
|---|---|---|
| `stale ≤ 0` — incoming covers the restored state | **STRICT EXTENSION** (the fast path; no state surgery) | near-zero: decode only the new tail |
| `0 < stale ≤ n_rs_seq` — tiny stale tail (any stale tail when the model has no recurrent bound, `n_rs_seq = 0`) | bounded rollback (`seq_rm`) | delta prefill |
| `stale > n_rs_seq` (only when `n_rs_seq > 0`) — real divergence | **CLEAR** + full re-prefill. The checkpoint ladder does *not* rescue this case: the same branch wipes the sequence **and discards the ladder**, because a ladder entry describes the pre-clear sequence and a later checkpoint search keys only on positions — a surviving entry could be matched against KV content that no longer corresponds to it | the full context (the cost this whole system exists to avoid) |

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
6. **DO serialize tool-call turns and tool results deterministically** — stable key order, stable formatting, identical bytes on every resend. The level-1 chain hashes `tool_calls`, `function_call` and `tool_call_id`, so a *changed* tool call or result is caught at the gate. What it cannot see is drift in how the chat template renders that span into tokens — that is caught only by the engine, as a re-prefill, which is why the optional tool-tail guard described in Level 1 exists.
7. **DO send identity labels every request** (see [TAGS_AND_IDENTITY.md](TAGS_AND_IDENTITY.md)) — labels drive bin keying, the persistence gates, and the compression contract.
8. **DON'T expect the manager to fix an inconsistent client.** The guard rails safe-degrade to full reprocessing — they prevent wrong answers and crashes, not slow turns. Only byte-stable behavior gets the fast path.

## 6. Diagnosing match results

After a follow-up turn, the question is: **did it reuse the saved prefix, and if not, why?** This section shows where to look; for a quick check, jump to the quick recipe at the end.

Seven surfaces, from decision intent down to ground truth:

| Surface | What it tells you |
|---|---|
| **`resolved_from` provenance** | the manager's decision for each restore, as one token |
| **The `KV_RESTORE` diag line** | one line per cold decision: which bin was chosen and how deep the turn-level match went |
| **`/status.kv_classifier`** | running counters of the restores |
| **`/status.shadow_diag`** | best-effort diagnostics that name *why* a cold restore went the way it did |
| **Engine log** | the ground truth: whether the match paid out |
| **LOAD_VERIFY `kv_restore` records** | the restored depth the manager expected against the depth the engine reports |
| **FE** | the `KV_RESTORE` pill and the prefill progress line on the resident card |

### `resolved_from` provenance

Every restore decision logs one token. The important ones:

| Outcome | Signal | What it means |
|---|---|---|
| **match success** | `restore-prefix-valid`, `wave-return-clean-restore`, `wave-return-shadow-restore`, `warm-force-clean-restore` | A bin was restored. |
| **intentional non-restore** | `warm-native-reuse-longer`, `warm-vram-fresher-skip`, `warm-anchor-natural-skip`, `warm-force-gated-native-reuse` | The live VRAM state already covers the request, so not restoring IS the correct outcome. |
| **match failure** | `restore-diverged-fresh` | An earlier turn changed (see rules 1/2/5 in section 5). |
| **match failure** | `restore-physics-belt-saved-longer` | The saved state overshoots the incoming request. |
| not a match failure — correct isolation | `restore-owner-mismatch`, `restore-no-anchor-for-identity` | The bin belongs to another identity, so it was correctly left alone. |
| first contact for this identity | `restore-no-bins` | Nothing is saved for this identity yet, so there is nothing to match. |
| wiring/guard skips | `restore-no-incoming-size/-chain`, `restore-no-identity` | A fail-safe skip. Investigate the client integration. |
| mechanical failure of the restore POST | `restore-post-failed` | The restore POST itself failed. It is not a matching miss. |

### The `KV_RESTORE` diag line

One line is logged per cold decision. Its `chosen=` field says what happened:

| `chosen=` | What it means |
|---|---|
| `chose_clean` | A think-free clean anchor was restored. |
| `chose_shadow` | A think-free shadow bin won. |
| `chose_clean_withthink` | A bin was restored but still carries generated `<think>` tokens. This is the value to look for when a restore reports success and the engine clears anyway. |
| `fresh` | No restore, or the restore POST failed. |

The other fields on the line:

- `resolved_from=` — the token from the table above.
- `clean_bin_present=`, `shadow_bin_present=` — whether a clean bin and a shadow bin were present.
- `common_prefix_turns=` vs `incoming_turns=` — how deep the turn-level match went.
- `divergence_pos=` — the divergence position.
- The KV-pressure fields.

### `/status.kv_classifier`

The running counters:

- `wave_return_restores` — cold successes.
- `forced_clean_restores` — warm restores.
- The per-event-type tallies.

### `/status.shadow_diag`

Best-effort diagnostics with the fields `saves`, `restores`, `byteparity`, `evictions` and `kvgc`. Three of them name *why* a cold restore went the way it did:

| Field | What it tells you |
|---|---|
| `restores` | Which bin each cold pass chose (`chose_clean` / `chose_shadow` / `chose_clean_withthink` / `fresh`). |
| `byteparity` | Compares the manager's reconstructed think-free render against the harness's resend of that same turn: `match` / `diverge` / `no_recon_src`. This separates *the right bin was chosen but its bytes diverge* from *the wrong bin was chosen*, a distinction no other surface can make. |
| `evictions` | How many shadow bins the byte-ceiling GC reaped: the bin was saved correctly and then deleted under memory pressure. |

These drive no decisions; every writer is a log/counter belt.

### Engine log (ground truth)

| Log line | What it means |
|---|---|
| `restored slot: strict extension` | The match paid out. |
| `restored slot: small stale=… <= n_rs_seq=…, seq_rm` | A bounded rollback absorbed the difference. |
| `restored slot: large stale=… > n_rs_seq=…, CLEAR + reprefill` | It didn't. |

The checkpoint ladder is a separate axis:

| Log line | What it means |
|---|---|
| `restored checkpoint ladder for …` | Confirms a `.ckpt` sidecar loaded at restore time. |
| `restored context checkpoint (pos_min = …, pos_max = …)` | An actual ladder rewind on the prompt-processing path. |
| `forcing full prompt re-processing due to lack of cache data` | That path's negative. |

A small `prompt eval` count right after a restore is the definitive receipt.

### LOAD_VERIFY

LOAD_VERIFY `kv_restore` records report the expected restored depth (`kv_expected_tokens`) and the depth the engine reports (`kv_actual_n_past`). `kv_restore_ok` is true when the actual depth is at least 98% of the expected one, and null when there was nothing to check against.

### FE

- The `KV_RESTORE` engine-op pill on the resident card, shown during the operation.
- The resident card's prefill progress line, whose token count includes the cache-reused tokens (`n_prompt_cache`) the engine reports.

### Quick recipe

1. Send your follow-up.
2. Check the engine log for `strict extension`.
3. Check `/status.kv_classifier.wave_return_restores` for the bump.
4. If you see `restore-diverged-fresh` instead, diff the exact bytes of your resent history against what you sent before — rules 1–6 above name every known cause.

## 7. Why this document exists

Matching failures are subtle: silent byte-drift between saved state and resent renders once caused full-context re-prefills on every turn, cross-conversation cache collisions, and (at worst) restore-linked engine aborts. That history produced the current design: **matching is now validated turn-level before any restore, verified token-level by the engine, guarded at every save seam, and failure modes are designed to degrade to a fresh prefill.** The v0.7.0 [CHANGELOG](../CHANGELOG.md) entry records the hardening work.
