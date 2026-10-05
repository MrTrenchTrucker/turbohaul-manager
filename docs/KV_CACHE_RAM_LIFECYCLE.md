# Turbohaul-Manager — KV Cache RAM Lifecycle

**Audience:** Operators and developers who want to understand when and why a slot's pre-computed KV
cache is moved into system RAM, when it comes back, and why the checkpoint defaults are set the way
they are.

**Goal:** Name the main situations in which a slot's KV cache is displaced into the RAM tier and later
restored, explain the two-tier storage model behind it, and give an honest account of what is proven
and what is not — so you can predict latency and durability correctly instead of guessing from the
directory name.

---

## TL;DR

A slot's KV cache moves into RAM in three main situations. All three save into the
same RAM-backed tier, whose retention is capped at 20 GiB by default. None of them writes to disk
directly. A cache reaches the cold disk tier only as a later copy from the RAM tier, when a different
conversation takes over the same engine slot (ownership transfer). A copy when a model unloads (idle expiry, grace expiry, shutdown, a failure teardown, or an idle model released to make room for another) is
work in progress and currently not working.

| Situation | What triggers it | Is the model unloaded? | Where the cache ends up |
|---|---|---|---|
| **Curator displacement** | A curator pass (or a same-model sub-agent pass) displaces an ongoing conversation | No | RAM tier, under the main conversation's own key — a curator never overwrites it |
| **Sub-agent / compression wave** | A sub-agent or compression pass needs a different model | Yes — a full model swap | RAM tier, then restored when the main conversation returns |
| **Unload seam** | Any model swap-out or teardown | Depends on the trigger | RAM tier for anything eligible; a copy to the disk tier on unload (idle expiry, grace expiry, shutdown, a failure teardown, an idle model released to make room) is work in progress and currently not working |

If you remember one thing from this doc: **"saved" does not mean "written to disk."** The tier a cache
lands in changes both how fast it comes back and how long it survives.

---

## The two storage tiers

A displaced KV cache can land in one of two places, and they behave very differently.

| Tier | What it is | When it's used | Round trip |
|---|---|---|---|
| **RAM tier** | `/var/lib/turbohaul/kvcache`, expected to be a **tmpfs** mount — it lives in system RAM, not on disk. Retention is bounded by a byte ceiling (`kv.ram_cache_max_bytes`, default **20 GiB**) alongside the age and file-count prunes | Every displacement described in this doc | Fast: the whole point of this tier is a cheap restore |
| **Disk tier** | `/var/lib/turbohaul/kvcache_persist`, a plain directory expected to be on a non-volatile filesystem, used purely as **cold storage**. Bounded by its own ceiling (`persist.max_bytes`, default **40 GiB**, adjustable from the Settings UI) | Written as a copy from the RAM tier when a different conversation takes over the same engine slot (ownership transfer); a copy when a model unloads (idle expiry, grace expiry, shutdown, a failure teardown, or an idle model released to make room for another) is work in progress and currently not working | Read back into the RAM tier once, when the next matching request looks for it (at most once while the RAM tier stays without that cache) — only when the RAM tier doesn't already hold a copy |

That the RAM tier is a tmpfs is a property of how the directory is mounted, not of its path or its
name — don't infer durability from the directory name alone. At boot the manager checks the
directory's filesystem type (`kv.kv_save_expected_fstype`, default `tmpfs`) and logs an error, without
refusing to start, if it differs; the disk tier gets the inverse check (`kv.kv_persist_forbidden_fstypes`,
default `tmpfs` and `ramfs`). If the host loses power, anything sitting in the RAM
tier is gone; anything already flushed to the disk tier survives.

A disk-cooled cache is not restored directly into video memory. The manager reads it back into the
RAM tier first, when the next matching request looks for it — and only when the RAM tier doesn't
already hold a copy of it; an existing RAM copy always wins. That RAM-tier detour is attempted at
most once per period in which the RAM tier lacks the cache, at the moment it's needed
again — not on a schedule, and not ahead of a real request.

---

## Retention — what gets reclaimed, and in what order

The RAM tier is bounded by **three prunes**, applied in one pass, in this order:

| Prune | Bound | What it takes |
|---|---|---|
| **Age** | `kv.ram_cache_max_age_hours`, default **6 hours** | entries older than the limit |
| **File count** | `max_files`, default **100** | the oldest entries beyond that count |
| **Byte ceiling** | `kv.ram_cache_max_bytes`, default **20 GiB** | entries until the tier is back under the ceiling |

The age and file-count prunes run first; the ceiling prune acts on whatever survives them.

### Two classes of entry are exempt from the prunes

An entry is **protected** — excluded from the age, file-count and ceiling prunes alike — if either
of these is true.

**It is owned by a live thread:** the active slot, any in-flight slot, the idle-held conversation, or
a resident's idle conversation. This holds for as long as the thread lives — such an entry is not
age-released late, it is not age-released at all.

**It is the elected clean anchor for its conversation:** the single best entry per
model-and-conversation, kept as that conversation's cheap restore point. ⚠ **This election is gated,
and the gate is easy to miss.** A conversation is given an anchor only while it is live, *or* while
the candidate entry is newer than `queue.idle_hot_load_seconds`. Once a conversation is dead **and**
its newest clean entry has aged past that window, it has no elected anchor at all, and its entries
fall to the age prune, the file-count prune and the ceiling like any other.

Note that the two overlap: for a live conversation the anchor adds nothing, because the live-thread
rule already covers every one of its entries. **The anchor's only independent effect is a short grace
window for a conversation that has just gone quiet** — which is exactly when a returning client is
most likely to want it back.

**The live-thread exemption is deliberate, and honouring the age limit literally would be worse.**
Releasing a live conversation's cache on a timer would delete the saved artifact out from under a
session still using it, and dropping its anchor means the next turn re-prefills from scratch instead
of restoring. The cost is real and is stated here rather than left to be discovered: **a busy,
long-lived session can hold RAM-tier entries past every bound for as long as it stays busy.** That is
the intended trade, not a leak. A *quiet* session does not get that treatment — see the gate above.

### The ceiling prune takes entries in two tiers, not in age order

When the tier is over its byte ceiling, unprotected entries are reclaimed in this order:

1. **every non-`.shadow.` entry**, oldest first;
2. **then** `.shadow.` entries, oldest first.

**So the oldest entry on the RAM disk is not necessarily the first one taken.** Between two
*unprotected* entries, a three-hour-old shadow one outlives a five-second-old ordinary one, because
the tier boundary is applied before age is considered at all. If you are reasoning about
which entry is about to disappear, the tier matters more than the timestamp.

**Why it works this way.** Anchors belonging to long-dead conversations must not
stay pinned and hold a large protected floor while the shadow entries that make a
returning client cheap are reclaimed ahead of them: a returning client would then
pay a full re-prefill instead of the sub-second restore this tier exists to
provide.

**Two mechanisms work together.** A protected entry is removed from the candidate list
*before* the ceiling prune sorts anything, so no sort order can ever reclaim one.
What makes a dead anchor reclaimable is gating the anchor election on liveness
and recency, the rule described above, so a conversation that goes quiet stops
being pinned. The reorder is what then stops shadow entries being taken ahead of
ordinary ones.

**The limitation:** the tier boundary is broader than the dead-anchor case it targets.
The ceiling prune takes **every** non-shadow entry first — including a fresh one
written by an ordinary save, which is restore-useful. Under sustained ceiling
pressure a recent entry can therefore be reclaimed ahead of a much older shadow.

---

## Instance 1 — a curator pass displaces the main conversation

> **Note:** the main conversation and its curator must select the identical model manifest. Curator
> displacement may not work as described here in this release; read this section as the intended behaviour.

A **curator** request is a background reviewer pass over an ongoing conversation (the `is_curator`
role; see [TAGS_AND_IDENTITY.md](TAGS_AND_IDENTITY.md)). It carries the **same conversation identity as
the main conversation** and, in the case described here, runs on the **same model**, so it takes over
the engine slot the main conversation is holding without any model being unloaded.

When a curator request arrives while a main conversation holds the slot, the manager:

1. Saves the main conversation's slot KV cache into the RAM tier.
2. **Waits for that save to finish** before letting the curator prefill. This ordering is
   load-bearing — the save completing first is what makes the later restore faithful.
3. Runs the curator. The model is **not** unloaded.
4. Restores the main conversation's cache from RAM when its next turn needs it (the warm-path
   force-clean restore described in [ARCHITECTURE.md](../ARCHITECTURE.md), section 4.6; no restore is
   made when the engine's live state already covers the turn).

This displacement save is made for a curator or sub-agent request that lands on the same model as the
main conversation. A compression request does not trigger it, because a compression pass is about to
rewrite the main conversation's context. A displaced context shorter than about 40,000 characters
(roughly 13k tokens) is not saved this way, since it is cheaper to re-prefill. In the background, the
outgoing conversation's RAM copy is also copied to the disk tier when a different conversation takes
over the engine slot (ownership transfer).

**This behaves as a fork, not an override.** Saved caches are keyed by both conversation and role (for
requests that carry a session id and a role label), so a labelled curator has a key of its own,
distinct from the main conversation's, while the main conversation keeps its own, untouched. By
default a curator's own cache is not kept after the pass: a request opts a disposable role in to
saving with `save_kv: true` (see [TAGS_AND_IDENTITY.md](TAGS_AND_IDENTITY.md)). Either way a curator
cannot overwrite the main conversation's saved cache: by default the two never share a key, and on
the opt-in reuse-main route (`TURBOHAUL_CURATOR_REUSE_MAIN`, off by default) the curator's save is
skipped.

This separation covers the *saved* copies. It does not cover the live, resident sequence on the GPU
while both roles are active: a curator pass shares that resident sequence with the main conversation,
and that sharing can leave the curator's tokens on the live sequence, in which case the main
conversation's own context may have to be rebuilt (see the dirty-tip tracking in
[ARCHITECTURE.md](../ARCHITECTURE.md), section 4.7). The saved-bin guarantee above still holds — this
concerns the live sequence itself, not what gets saved.

---

## Instance 2 — a sub-agent wave displaces the main conversation

Sub-agent and context-compression work is often routed to a **different model**, a choice made by the
client or deployment rather than by the manager. When the concurrency limit
(`queue.max_parallel_sidecars`, default 1) leaves no room to keep both models resident, that forces a
genuine model swap, which evicts the main conversation's cache from video memory entirely — a much
harder displacement than instance 1, where the model never moves.

This has a useful property: after a real model
swap, the main conversation's next turn **cannot** be served by a cache that merely happened to still
be warm, because there is no warm cache left to happen upon. It can only succeed if the cache was
truly saved to RAM and truly restored. In that sense, a forced model swap is the system's own honesty
check on its save-and-restore path — there's no way to fake a fast return after one.

---

## Instance 3 — the unload seam

Whenever a model is swapped out or torn down, a save fires at this **unload seam** for any eligible
resident cache — "eligible" here means the same conversation-identity requirement covered in
[Eligibility](#eligibility--not-every-requests-cache-is-saved) below.

This is the mechanism that makes instance 2 possible: the save that happens at swap-out is what
instance 2's later restore reads back. It is also meant to feed the disk tier on a full
unload — once a model's grace timer and its idle timer have both run out with no follow-up request in
flight — but that disk copy is work in progress and currently not working, so after a full unload the
cache is in the RAM tier only.
The unload seam is the one save point shared by
"moving to a different model for a moment" and "shutting this model down for good," which is why it's
worth naming as its own instance rather than folding it into instance 2.

Instance 1 does **not** go through the unload seam — no model swap or teardown happens there, since
the model stays loaded throughout a curator pass. Instance 1's save is a save "in place," not a save
"on the way out."

Put end to end, the cold round trip looks like this, for a cache that reached the disk tier by
ownership transfer (the disk copy at unload is work in progress and currently not working): it sits in the disk tier — cold — until the next request that matches it
arrives. At that point, as part of that request's load sequence, the cache is read back into the RAM
tier and restored from there. That happens on demand — only while the RAM tier lacks the cache, and at
most once per such gap — not on any other schedule.

---

## What actually gets saved

A displacement doesn't just save the cache bytes. Up to three files travel together as one set:

| Component | What it is |
|---|---|
| **KV cache** (`.bin`) | The pre-computed attention state for the conversation so far |
| **Checkpoint ladder** (`.bin.ckpt`) | The engine-state sidecar: intermediate rungs that allow a *partial* restore when the incoming conversation has changed slightly, instead of forcing a full recompute. Written only when the engine's checkpoint-sidecar flag is on |
| **Metadata** (`.json`) | Owner identity, the turn-hash chain, the saved token count, and the `clean_prefix` / `stale` / `shadow` markers — this is what the restore gate reads to decide whether the bin may be offered at all |

The cache and its ladder are treated as one unit throughout: a save is rejected (and the previous pair
kept) if the ladder is under a tenth of its bin's size, or is missing where an earlier ladder already
existed, and the garbage collector removes the two together, with a separate sweep for any orphaned
ladder, so an eviction does not leak a multi-gigabyte orphan ladder. The ladder itself is gated on
the engine's checkpoint-sidecar flag; with that flag off a bin still saves and restores correctly,
but every near-miss costs a full recompute because there is no rung to fall back to.

---

## Checkpoints — why the default is 2

The checkpoint ladder is what makes a *partial* restore possible instead of an all-or-nothing one.
Each rung on the ladder is a point the engine can fall back to when the incoming conversation doesn't
match the saved one exactly — for example, because the last turn or two changed.

| Ladder size | What happens on a near-miss |
|---|---|
| **1** | There is effectively a single rung, and the engine's own fallback rung is discarded. A near-miss costs a **full recompute**. |
| **2** (the default) | The fallback rung survives. A near-miss can be served by a **cheap partial restore** instead. |

**The honest limitation:** a default of 2 is aimed at **warm** restore. **Cold restore at this setting
has not been characterised at scale.** This gap is stated here deliberately rather than left out: a
doc that oversells a default is worse than one that says where its evidence stops.

---

## Eligibility — not every request's cache is saved

A request that arrives without a stable conversation identifier is treated as **disposable walk-in
traffic**, and its cache is **not** saved.

This is deliberate, not an oversight: preserving a cache for every anonymous, one-off request would
evict caches that are actually going to be reused, for the sake of ones that almost certainly won't
be. It's also the most common reason an operator sees no save happen and assumes something is broken.
If you want a request's cache preserved across turns, send a stable conversation identifier with it —
without one, the manager has no way to know a later request is the same conversation, so it can't
safely keep the cache around for it.

---

## Exception — vision-capable models cannot be saved at all

Everything above assumes the model being displaced is text-only. A model served with a multimodal
projector (a manifest carrying `mmproj_blob_sha256`) is a different case entirely: **none of the three
save mechanisms in this doc apply to it.**

The engine itself refuses the save, restore, and erase actions outright whenever a projector is
loaded — this is a limitation in the engine, not a bug in the manager's save path. A vision-capable
client's precomputed context does not survive any of the three displacement instances above; its next
turn pays a full re-prefill regardless of which instance triggered the displacement. The
`--slot-save-path` flag that enables saving for text-only models is present on every manager-spawned
engine, vision or not — the projector is what makes the difference, not a missing flag.

This is not a hypothetical edge case: it affects every eviction of every vision-capable resident.
Text-only models are unaffected and behave exactly as described in the rest of this document.

The code-side fix for this (skip the doomed save attempt, log the reason plainly instead of a generic
failure, and — separately — actual engine-level support for saving a projector-loaded slot) is not
implemented as of this writing.

---

## What to expect, and how to tell healthy from unhealthy

Everything below follows directly from the mechanisms above — it's the same behavior, described from
where an operator is standing.

- **Around a curator pass:** expect (when the displacement conditions above hold) a save of the main
  conversation immediately before the curator runs, and a restore of that same cache when the main conversation's next turn arrives. The model
  should not be reloaded for this — a curator pass never unloads it. If the main conversation's
  context looks like it was rebuilt from scratch right after a curator turn, that is not the expected
  behavior described here.
- **Around a sub-agent or compression wave:** expect a full model swap, a save of the main
  conversation's cache before the swap, and a restore afterward in which the large majority of tokens
  come back from the saved cache rather than being recomputed — recomputing only a small remainder,
  not the whole context, is the expected shape of a healthy restore.
- **If a given client's traffic never shows a save at all:** the most likely explanation is that the
  requests aren't carrying a stable conversation identifier, which puts them in the disposable
  walk-in category described above. That is expected behavior for that kind of traffic, not a fault —
  the fix, if the cache should be preserved, is to start sending a stable identifier. **If the client
  is served by a vision-capable manifest, check that first** — a projector-loaded model never saves,
  regardless of conversation identity; see [Exception — vision-capable models](#exception--vision-capable-models-cannot-be-saved-at-all).
- **If a near-miss restore recomputes far more than a couple of thousand tokens' worth of change:**
  that is consistent with the checkpoint ladder's fallback rung not being available — check that the
  checkpoint count is at the default of 2, not 1.
- **Warm restores are the case this design targets.** The behavior described here for a model that's
  still resident in some form is the expected shape. Cold restores — where the model has been
  fully unloaded and reloaded — follow the same design, but their latency has not been characterised
  at the default checkpoint setting; treat it as plausible, not guaranteed.

---

## Summary

- A KV cache is displaced into RAM in three main situations: a **curator pass** (model stays
  loaded, the curator never overwrites the main conversation's entry), a **sub-agent or compression wave** (a real model
  swap, the main conversation's cache is fully evicted from VRAM and must be genuinely restored), and
  the **unload seam** (the shared save point behind both a swap and a full teardown).
- There are two storage tiers, and they are not interchangeable: a **RAM-backed tier** for fast round
  trips, capped at **20 GiB** by default, and a **disk tier** for cold storage, capped at **40 GiB**
  by default, which a cache reaches only as a later copy from the RAM tier (when another conversation
  takes over the engine slot; the copy on unload is work in progress and currently not working) — and which is read back into
  the RAM tier on demand, when the next matching request looks for it and the RAM tier lacks a copy.
- A save carries the cache **together with its checkpoint ladder and its metadata sidecar** — the
  ladder is written only when the engine's checkpoint-sidecar flag is on, and cache and ladder are
  saved, copied, and evicted as a pair.
- The checkpoint ladder defaults to **2 rungs** because that is what keeps a near-miss cheap instead
  of forcing a full recompute, aimed at warm restore; cold restore at that setting has not been
  characterised at scale.
- A cache is only ever saved for a request carrying a **stable conversation identifier** — anonymous,
  one-off traffic is treated as disposable by design, not by accident.
- **None of this applies to vision-capable models.** A manifest carrying a multimodal projector cannot
  be saved, restored, or erased at all — the engine refuses all three outright whenever a projector is
  loaded. Every eviction of a vision-capable resident costs a full re-prefill; this is an engine
  limitation, not a manager defect, and text-only models are unaffected.

Taken together, these are the answer to "what happens to a slot's pre-computed KV cache in the main
situations where it has to move" — not just the common case, but all three, and the one case where none
of them apply.
