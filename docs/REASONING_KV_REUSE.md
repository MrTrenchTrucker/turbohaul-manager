# Reasoning-Model (`<think>`) KV Reuse

**See also:** [KV_CACHE_MATCHING.md](./KV_CACHE_MATCHING.md) (how a match is decided and diagnosed) and [ARCHITECTURE.md](../ARCHITECTURE.md) (the full KV lifecycle). This page is the **reasoning-model angle** — why models that emit `<think>...</think>` are the hard case for cache reuse, and the benchmark that proves Turbohaul-Manager solves it. It complements those docs rather than repeating them.

---

## 1. Why reasoning models are the interesting case

Modern reasoning models don't answer directly — they first emit a hidden chain of thought wrapped in `<think>...</think>`, then emit the user-visible reply. Those `<think>` tokens can be enormous: thousands of tokens of scratch work per turn. They are also **disposable**: nearly every agent harness strips the `<think>` block out of the assistant message before it resends the conversation on the next turn, because the model is not supposed to condition on its own prior scratch work.

That strip is exactly what makes reasoning models a trap for KV-cache reuse. Turbohaul-Manager exists to make a long conversation's precomputed context nearly free to resume — across a model swap, and even after the model has been fully unloaded from the GPU. For an ordinary model that resend-what-you-sent-before, that resume is cheap. For a reasoning model, the state you saved and the bytes the harness sends next **don't match** — and a naive cache would silently fall back to recomputing everything. This page explains the mismatch, why the obvious fixes fail, the fix Turbohaul actually ships, and the measured payoff.

## 2. The problem — the saved state carries `<think>`, the next turn does not

Walk through one turn of a reasoning agent:

1. The client sends the conversation. The engine prefills it into the GPU KV cache and the model generates: first a long `<think>...</think>` block, then the visible answer. **Both are now in the live KV state** — the cache physically contains the reasoning tokens the model just produced.
2. Turbohaul saves that KV state at the model-swap seam (or persists it to SSD when the model is unloaded) so the conversation can resume later without re-reading it.
3. Next turn, the harness resends the conversation — but with the previous assistant turn **think-stripped**. The `<think>` block is gone.

Now the saved state and the incoming request diverge *inside the assistant turn*: the saved prefix contains reasoning tokens that the new render does not. Cache reuse is decided by a longest-common-prefix comparison (see [KV_CACHE_MATCHING.md](./KV_CACHE_MATCHING.md)); the common prefix ends at the first `<think>` token, and everything after it — potentially the entire rest of a 150,000-token conversation — is unmatchable. The saved bin is **no longer a clean prefix** of what the client will send.

The cost of that miss is not subtle. On a large reasoning conversation, falling off the fast path means the engine re-prefills the whole context from scratch: **a measured ~327 seconds** (roughly 330–420 s at 150k–220k tokens on the reference hardware) of pure prefill before the model can even begin the next turn. That is the entire cost this system exists to avoid — and a naive cache pays it on *every* turn of *every* reasoning conversation.

## 3. Why you can't just trim the `<think>` tokens out

The obvious fix is to trim the reasoning tail out of the saved state so the remainder matches. It doesn't work — and the reason is architectural, not a bug.

The reference deployment runs a reasoning model with two *independent* properties: **MTP** (multi-token prediction) — extra heads that speculatively draft several tokens per step — and a **hybrid-recurrent memory** that carries an SSM-style **recurrent state** alongside the attention KV. It's the recurrence, not MTP itself, that creates the trap. Recurrent state has a hard property — it **cannot be partially erased mid-sequence** beyond a small window (`n_rs_seq`, which is 2 in the reference config; that window is sized to the MTP draft depth, *not* to prompt divergence, which is exactly why it's so narrow). Whole-sequence erase is always legal; surgical mid-sequence trims are not.

So if you let the model generate its `<think>` tail into the live state and *then* try to drop that tail to recover a clean prefix, you hit the recurrent-state bound. The engine can't cleanly rewind past that window — a save of the partially-trimmed state persists an internally inconsistent bin that later **aborts the engine** when it tries to extend from it. You can't retroactively subtract the reasoning tokens.

Why the asymmetry with an ordinary KV cache? An attention KV cache is an explicit per-position ledger: each token gets its own `(K, V)` entry, written once and never mutated, so truncating to a prefix is just slicing an array. Recurrent state is the opposite — a single fixed-size buffer into which every token is mixed and then decayed by a non-invertible gate. There is no sub-range that corresponds to "the first N tokens," and the decay can't be run backwards, so the state can only be restored *whole* at the position it was captured — never sliced back to an earlier one.

The consequence is the design constraint that drives the whole solution: **the clean prefix must be captured *before* the model generates its `<think>` block, not trimmed out afterward.** Once reasoning tokens are in the recurrent state, they're baked in.

## 4. The fix — capture a clean prefix before `<think>` exists

Turbohaul's answer is a **prefill-only clean-prefix probe** plus per-conversation identity and an owner-gated restore that prefers the clean copy. Three pieces:

**1. Prefill-only clean-prefix capture (before any `<think>` is generated).** At the save seam, instead of dumping the live think-carrying state, the manager re-renders the historical transcript **think-stripped** and prefills only that (engine `/apply-template` → strip the assistant `<think>` scaffold → prefill with `n_predict=0`, i.e. no generation). Because generation never runs, no reasoning tokens and no recurrent-state contamination ever enter this snapshot. The bin it saves **byte-matches the exact think-stripped bytes the harness will resend next turn** — a clean prefix by construction, not by after-the-fact surgery. This behavior is on by default (`TURBOHAUL_COVERED_SCAFFOLD_STRIP`) and only kicks in above a save floor (~13k tokens), since tiny contexts are cheaper to re-prefill than to manage. The bin is stamped `clean_prefix: true` in its metadata.

**2. Per-conversation identity.** Each conversation (and each agent role within it) owns exactly one KV copy, keyed by `(session, role)`. This is what lets the manager find *your* clean bin later and never confuse it with another agent's — the identity contract is covered in depth in [KV_CACHE_MATCHING.md](./KV_CACHE_MATCHING.md).

**3. Owner-gated, single-best-bin restore preferring the clean bin.** On resume, the manager lists this identity's candidate bins, runs each through the restore gate (owner-identity match first — another agent's bins are never touched), and restores **only the single best bin**, sorting `clean_prefix` bins ahead of everything else and then by token count. The think-free clean prefix wins. Restoring one carefully-chosen bin — rather than several — is deliberate: it's what keeps a stale copy from ever clobbering the good one.

There is also a **warm** variant for the case where the model is still resident but the harness's next render is about to diverge from the engine's natural think-carrying state: the manager force-restores the think-free clean bin in place — but only when that clean chain is a valid prefix of the incoming request and the live state doesn't already cover it. If the live VRAM state is already longer and matches, that wins instead; the manager never restores over something better.

## 5. The proof — 154,647 tokens reused, 29-token prefill

The headline result is a **cold restore after a full model unload** — the hardest case, where the model was entirely evicted from the GPU and its state survived only on SSD:

| Metric | Value |
|---|---|
| Tokens reused from the restored clean bin | **154,647** |
| Tokens actually prefilled on resume | **29** |
| Prefill work vs. recomputing | **~629× less** |
| Wall-clock: full re-prefill (the avoided slow path) | **~327 s** |
| Wall-clock: clean restore (the fast path) | **~0.5 s** |

The conversation came back in about half a second instead of recomputing for over five minutes — after the model had been fully unloaded. The engine confirmed it took the fast path: after the restore, its own log reports a *strict extension* (only the 29 new tokens decoded) rather than a *clear + reprefill*. That engine-level line is the ground-truth receipt that the match paid out.

**It holds under real multi-agent load, not just a single-conversation microbenchmark.** In hardening runs, one session's main conversation was seam-saved at a model swap and wave-return-restored at **~96% common-prefix reuse** — while sub-agents interleaved on a *second* model correctly went fresh (their bins belong to a different identity and are never cross-restored). The exact heavy-context scenario that used to storm the engine with aborts now runs with **zero engine deaths and zero aborts**.

**Streaming works too.** The clean-prefix probe runs at the save seam, entirely independent of how the next turn is served — so a streaming harness gets the same instant resume. The restore happens once when the model comes back; from then on the turn streams normally, decoding only the new suffix.

**The owner gate really rejects.** If a different agent (a different `(session, role)`) asks for a bin that isn't theirs, the restore is refused at the owner-match step and that caller prefills fresh in its *own* bin — it never rides another conversation's clean prefix. Cross-agent isolation is a hard boundary, not a best-effort hint.

## 6. What this means for you

If you're deploying reasoning-model agents — anything that emits `<think>...</think>` — behind Turbohaul-Manager:

- **Your agents resume instantly across model swaps and idle unloads.** A long reasoning conversation that would cost ~327 s to recompute comes back in ~0.5 s, because the saved state is a clean, think-free prefix that matches exactly what your harness resends.
- **You don't have to do anything special about `<think>`.** The clean-prefix probe handles the strip-mismatch for you; you just send your normal think-stripped renders (the default convention). Keep your history append-only and byte-stable and you stay on the fast path — the client rules in [KV_CACHE_MATCHING.md](./KV_CACHE_MATCHING.md) apply here too.
- **It works with streaming harnesses.** The reference validation ran on the MIT-licensed [Hermes](https://github.com/nousresearch/hermes-agent) harness, which streams — the reasoning-model resume path is proven end-to-end there.
- **Multiple agents share one GPU without stepping on each other's context.** Each conversation and role owns its own clean bin; the owner gate keeps them isolated even when a sub-agent wave takes the GPU and the main agent returns afterward.

The net effect: a reasoning-model agent never has to re-read its own conversation. The scratch work it discards every turn no longer poisons the cache, and the expensive part — the precomputed context — is reused down to the last near-clean prefix.
