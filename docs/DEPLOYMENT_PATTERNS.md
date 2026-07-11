# Deployment & Model Patterns — from one agent to a whole fleet

Turbohaul-Manager turns a single GPU into a persistent, cache-aware inference layer. This guide is about *how you deploy it*: powering one agent from start to finish through every model swap, scaling up to a fleet of agents, and — the decision most teams care about — **which model to run for which job**. For the mechanics underneath, see [ARCHITECTURE.md](../ARCHITECTURE.md) (how it works), [TAGS_AND_IDENTITY.md](TAGS_AND_IDENTITY.md) (how requests are identified), and [KV_CACHE_MATCHING.md](KV_CACHE_MATCHING.md) (how cache reuse is decided). For the step-by-step harness wiring, see [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md).

---

## 1. One agent, start to finish

A single long-running agent is the simplest deployment, and it shows off the whole point of Turbohaul: **the agent's context is computed once and reused for the life of the conversation, even across model swaps.**

Walk the lifecycle of one agent on one GPU:

1. **First turn.** The agent sends its opening request labeled as the *main* agent. Turbohaul loads the model and prefills the prompt — the one and only time the full context is read from scratch.
2. **Tool-call turns.** The agent calls tools and continues. Each follow-up reuses the warm cache already in GPU memory, so only the new tokens are processed — responses come back at full speed with no re-reading of the conversation.
3. **A sub-agent wave.** The agent spawns sub-agents (say, to research or draft in parallel). If they run on a different model, Turbohaul saves the main agent's precomputed cache to system RAM at the swap, serves the sub-agents, and then — when the main agent returns — restores that cache. The main agent picks up where it left off having re-read *almost nothing* (in validation, a 154,000-token context resumed with a 29-token prefill).
4. **The model unloads.** If the agent goes idle long enough for the model to unload entirely, its cache is written down to SSD first, so even a full unload doesn't cost a re-prefill on the next turn — the context is restored from disk.
5. **Repeat, indefinitely.** The agent can run for hours or days, swapping models and spawning sub-agents as needed, and its main context stays warm the whole time.

The result: a single agent runs on a single GPU as if the model never left, no matter how much model-swapping and sub-agent work happens in between.

## 2. A fleet of agents

The same machinery scales to *many* agents sharing one GPU (or a few). Every agent — and every sub-agent it spawns — gets its own isolated cache keyed to its identity, so **no agent can read, overwrite, or muddy another agent's context.** (The one place a cache is deliberately shared is within a single agent: an off-by-default option lets a curator restore its *own* agent's main context read-only, to review it — never another agent's, and never with the ability to overwrite it.) Turbohaul serializes the shared GPU across the fleet, in whichever mode the hardware supports:

- **Single series** — one model resident, one request at a time. The fleet's requests queue up and are served in turn, each riding its own warm cache. This is the proven single-GPU shape.
- **Series parallel** — one model resident, several context windows served *at the same time* on that one engine. Same-model requests from different agents run concurrently.
- **Double parallel** — several models resident at once (bounded by VRAM), each serving its own agents, and each able to run series-parallel itself.

You don't rewrite anything to move up the ladder — the mode is configuration, so the same deployment grows from one card time-slicing a whole fleet up to a multi-model box running several engines at once. (See [ARCHITECTURE.md §1.1](../ARCHITECTURE.md) for a diagram of each mode.)

## 3. Which model for which job

An agent isn't one model — it's a *main* reasoner plus a set of background helpers. Turbohaul lets you assign a different model to each role and handles each one's cache correctly:

| Role | What it does | Recommended model | Cache treatment |
|---|---|---|---|
| **Main agent** | The primary reasoning loop — plans, calls tools, drives the task | A **frontier model** (top-tier reasoning) — or a strong local model | Saved and restored; kept warm across swaps |
| **Sub-agents** | Spawned workers running subtasks in parallel | **AUX** (smaller, local) models | Isolated per sub-agent; thrown away when the job finishes |
| **Curator** | Background review of the main agent's work | An **AUX** model | Never saved over the main agent's cache |
| **Compression** | Summarizes/compacts the conversation when it grows | An **AUX** model | Marks the old cache stale so the compacted context re-anchors cleanly |

The insight: the *main* reasoning wants the best model you can afford, but the *high-volume, disposable* work — sub-agents, review, compaction — runs perfectly well on cheaper local models. Turbohaul's role tags and disposable-cache contract (see [TAGS_AND_IDENTITY.md](TAGS_AND_IDENTITY.md)) are what make this split safe: the aux work can never pollute the main agent's carefully-kept context.

## 4. Two supported deployment shapes

**Shape 1 — Turbohaul runs everything (fully local).** The main agent *and* all the aux roles run on Turbohaul-managed local models on your own hardware. Nothing leaves the machine. Choose this when you want everything local, offline, and private, and you have the VRAM to back the main model you want. Turbohaul swaps between the main and aux models on one GPU and keeps every role's cache straight.

**Shape 2 — Frontier main + Turbohaul aux (recommended).** The main agent calls a **frontier model** (a hosted frontier API) for its top-tier reasoning, while the sub-agents, curator, and compression passes all run on **Turbohaul-managed local models**. This is the best cost-for-quality balance for most teams: you pay frontier prices only for the main reasoning that actually needs them, and run the high-volume disposable work locally for free. Turbohaul becomes your **aux inference layer** — the local, cache-aware backend for everything except the main reasoning.

**Either way, the setup is the same on Turbohaul's side** — you point the roles you want it to serve at Turbohaul and label each request with its role (main / sub-agent / curator / compression). Whether the main agent's requests go to Turbohaul or to a frontier API is a choice you make in your harness, not a different Turbohaul configuration.

> **Recommendation:** unless you specifically need the main model local, run **Shape 2** — a frontier model for the main agent, Turbohaul for the aux roles. You get frontier-quality reasoning where it matters and free, private, cache-accelerated local inference for the rest.

## 5. Why it pays off — cost and speed

Splitting roles this way isn't just architectural tidiness — it's money and latency, at a scale that surprises most teams.

**Cost: the disposable work is where the spend explodes.** In an agentic system the main agent's own turns are a small fraction of total inference. The volume lives in the *disposable* roles — every sub-agent a wave spawns, every research or council persona, every compression pass, every background review. A single complex task can fan out to dozens of sub-agent calls, each carrying its own full context. If all of that goes to a frontier API, you're paying frontier per-token prices on the **highest-volume** part of the workload — the part whose output is often discarded once it's synthesized. Move that volume onto local aux models and its marginal token cost drops to effectively **zero**: the hardware is a fixed cost and the tokens are free. At production scale — thousands of tasks, each fanning out — this is the difference between a token bill in the **millions** and one that's a rounding error. You keep paying frontier prices only for the main agent's reasoning, which is exactly where that quality earns its cost.

**Speed: local can beat the round-trip, especially with MoE.** Sending disposable work to a hosted API means a network round-trip on every call, plus queueing behind everyone else's traffic. A local model has neither. And modern **Mixture-of-Experts (MoE) models** tilt this further toward local: an MoE model activates only a small fraction of its parameters per token, so it runs far faster than its total size suggests while still punching well above its weight on quality — ideal for high-throughput sub-agent and council work. Layer Turbohaul's warm-cache reuse on top (each aux call reuses its own precomputed context instead of re-prefilling), and the local aux layer is frequently *faster* end-to-end than farming the same work out to a frontier API — not just cheaper.

**The net:** run the main agent on the best model you can justify, and run everything else — the sub-agents, the council, compression, review — locally on aux models through Turbohaul. You get frontier-grade reasoning where it counts, and free, fast, private inference for the workload that would otherwise dominate your bill. A working example of this coordination pattern — one persistent orchestrator that holds the goal and context, dispatching disposable, scoped sub-agents and councils that report back and are discarded — is described in the agent-setup guide below.

## 6. Why it holds together

Two mechanisms make all of the above work, and both are worth understanding before you deploy:

- **KV-cache orchestration** ([ARCHITECTURE.md §4](../ARCHITECTURE.md)) keeps the main agent's context nearly free to resume — across tool calls, sub-agent waves, model swaps, and even full unloads.
- **Role identity** ([TAGS_AND_IDENTITY.md](TAGS_AND_IDENTITY.md)) keeps every agent's and every role's cache isolated, so concurrency and aux work never corrupt the main context. It also means Turbohaul works out of the box for non-agent clients that send no tags at all — they simply get warm reuse without the role-aware extras.

## 7. Setting it up

The harness-side wiring — declaring Turbohaul as an inference provider, pointing each role at the right model, and passing the role labels on each request — is covered step by step in [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md), with a working reference configuration you can copy. That guide also includes a complete orchestrator/sub-agent coordination template you can adapt: one persistent main agent holding the goal, dispatching disposable research, coding, and council sub-agents that run on aux models and report back.
