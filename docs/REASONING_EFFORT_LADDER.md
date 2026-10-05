# `reasoning_effort` — the five-rung ladder, resolved at spawn

**`reasoning_effort` is not an ordinary per-request parameter: it is read once, when a
model's engine starts.** Two things to know before you send it, in this order:

1. It takes effect only on the first request that starts a model's engine, and is then
   fixed until that engine goes away. This holds at every `max_parallel_sidecars` value,
   the default of 1 included.
2. It needs something to scale: the model's manifest must have a positive
   `reasoning_budget`. With none (absent, `0` or `-1`) the value is validated and then
   has no effect.

If you read nothing else on this page, read those two.

## What it does, and why the value locks in at spawn

`reasoning_effort` on a chat request to Turbohaul Manager selects one of five rungs —
`low`, `medium`, `high`, `xhigh`, `max` — each a fraction of the model's own configured
reasoning budget. Send it as a normal field in a chat completion request body — both the
OpenAI-compatible and Ollama-compatible routes accept it:

```json
{
  "model": "your-model",
  "messages": [ ... ],
  "reasoning_effort": "high"
}
```

```
low .25 · medium .50 · high .75 · xhigh 1.0 · max 1.25
applied at launch as: --reasoning-budget = ceil(fraction × the model's configured budget)
# locked at spawn -- a later request asking for a different value does not change it
```

**These five are the entire accepted set** (matched case-insensitively, ignoring surrounding
whitespace; JSON `null` is treated as no value). Any other value — including OpenAI's `minimal`,
which this API does not accept — is rejected with an HTTP 400.

(As an example, one tested model resolved these five rungs to 798 / 1,596 / 2,394 / 3,192 /
3,990 tokens. The fractions are fixed; the token counts are not — they depend on the budget
configured for that particular model.)

`xhigh` and `max` go beyond `high`. A client that only ever sends `low` through
`high` never has to ask for more than that — but not asking for `xhigh` or `max` does not
guarantee it never receives one: if some other client already caused the same model's engine
to spawn at a higher rung, this client's request can still be served at that higher rung
instead (see below).

**The value is read once, not per request — and this is the part to know before calling
this endpoint at all.** It is applied when the engine behind that model starts up, as a
launch setting, and stays locked for as long as that engine keeps running. Once the engine
is up, it is up: the manager cannot load a new value onto an engine that is already
running, and it will not tear down that engine just to change the thinking value. A request
asking for a different rung does **not** cause a reload — it reuses the engine that is
already there, still running at whatever value it was launched with.

**This means the rung has to be sent on the first request for that model** — the request
that actually causes the engine to start. That is the only request whose value takes effect,
until the engine goes away.

A later request asking for a different rung is not rejected: it succeeds — HTTP 200, not an
error — and the response comes back generated at the rung the engine is already locked to,
not the one that request asked for. For example: if some other client already caused a
model's engine to spawn at `max` — Open WebUI, or any other harness — a later request asking
for `low` against that same model still gets `max`, silently, with an ordinary successful
response. Nothing releases that lock except the engine itself going away: a model swap that
evicts it, or its own idle timeout. Once that happens, the next request to spawn the engine
becomes the new first request — a request that simply lands on a still-running engine never
gets to set anything.

## Where the value is applied: every box budget

Every engine start goes through the same resident dispatcher, whatever
`max_parallel_sidecars` is (1 to 32; 1 is the default). At a budget of 1 it starts one
engine at a time and queues the rest; the ladder described above is resolved and baked
into the launch arguments the same way at every budget. The single-engine path
(`_process_slot`) is retained in the source as dead code and is never run.

The `main_gpu`/`split_mode` auto-placement override is applied on the same path, so it
is not affected by the box budget either.

## Why the value can't be changed per request

The reasoning: `reasoning_effort` is applied as an engine launch flag, so it takes effect
when the engine starts up and is then locked in for as long as it keeps running — there is
no way to change it mid-conversation without tearing the engine down and rebuilding it.

In the maintainers' own testing, on two different models (called model A and model B here),
the value was sent both in the request body and at engine launch:

| where the value was sent | value | result |
|---|---|---|
| body `thinking_budget_tokens` | 1 / 2 / 4 | RUNAWAY to the output cap (model A) |
| body `thinking_budget_tokens` | 0 | WORSE than `1/2/4` — removes the cap entirely (18,757 reasoning chars) |
| body `thinking_budget_tokens` | 3 | **STOP** — unexplained (see below) |
| body `thinking_budget_tokens` | 8 / 32 | stops cleanly (both models) |
| LAUNCH `--reasoning-budget` | 0 | RUNAWAY, worse still (23,244 reasoning chars) |
| LAUNCH `--reasoning off` | — | `reasoning_budget=0` on both models, but model A ran to the output cap anyway; model B stopped cleanly (1,209 reasoning chars) |

**Note:** the "reasoning chars" figures above count characters of reasoning output produced,
not tokens — they are a different unit from the token-based ladder values described earlier
in this document and are not directly comparable to them.

⚠ **Those runaway figures were measured at budgets of 0 to 4 tokens — far below the lowest
rung** (`low` resolved to 798 tokens on the tested model), which is why deciding the value at
launch time removes that mechanism. **Separately, and not explained:** a failure to terminate
has also been observed at the top of the range, and with no `reasoning_effort` sent at all.
Its scope is not established. If a request of yours does
not terminate, report it rather than assuming the cause is your `reasoning_effort`.

The engine's own help text documents budget `0` as "immediate end." In testing it was
not — the engine's own log printed `"forcing immediately"` while the model reasoned on
regardless. **Do not trust that documented contract.**

## An unexplained result from the per-request measurements

`thinking_budget_tokens` is not part of this feature's supported interface — it is a
request-body field that `reasoning_effort` never writes. It is mentioned here only because
one measurement taken in the request-body tests does not fit the pattern the rest of the
testing suggests.

One measurement in the table above does not fit the pattern the others suggest (small
budgets running away): `thinking_budget_tokens=3` stopped cleanly. No theory for why is
offered here. It is recorded as measured, not explained.

## `minimal` is rejected with HTTP 400

`minimal` (the OpenAI-spec value below `low`) is not special-cased and not mapped to
anything — it is simply absent from the five-entry ladder, so it falls into the same
unrecognized-value 400 any other bad string gets:

```json
{
  "detail": {
    "error": "reasoning_effort_unsupported_value",
    "message": "reasoning_effort must be one of: low, medium, high, xhigh, max",
    "received": "minimal"
  }
}
```
 **This deviates from the OpenAI spec
deliberately**: `minimal` is a legal, documented OpenAI value, and a spec-compliant client
sending it gets a 400 instead of a resolution, not an oversight to be discovered later.

`low` is the floor of the ladder because the measurements above show suppressing a
model's thinking does not reliably make it answer faster — some models do not reliably
answer at all with less thinking room. **Which model tolerates reduced thinking is an
operator/end-user decision.** Turbohaul cannot know in advance which model a caller will
pick, and does not pretend to by offering a value it cannot honor uniformly.

## Checking whether a request landed on a locked value

The distinction between what a request asked for and what it actually got served matters:
a request that gets silently served at someone else's locked-in value still comes back with
a normal, plausible answer — just not the one it asked for — which is exactly what makes it
easy to miss.

The reliable way to check is the status listing, not the log:

- **`GET /status`** — each running engine's entry carries a `reasoning_budget_override`
  field: the exact token-count budget its launch value is locked to. **It is an integer, such
  as `3990` — or `null` if no override is in effect. It is not the rung name.** It sits beside
  `main_gpu` and `split_mode`, the same kind of field: decided once when the engine starts,
  and unchanged until it is torn down.

  ⚠ **If you sent a `reasoning_effort` on the request that started the engine and this reads
  `null`, that is not a mistake in your request** - it also reads `null` when the model's
  manifest has no positive `reasoning_budget`, because there is nothing to take a fraction
  of. Check that before assuming you sent the value wrongly.

  ⚠ **This is per-engine, not per-request.** The completion response carries no field
  identifying which rung served it, so there is nothing to look for there:

  ```
  GET /status
  "reasoning_budget_override": 3990   // or null -- an integer token count, not a rung name
  ```

- **The log is a weaker, secondary signal, and one an API caller cannot see at all.** A
  warning line (written by `_warn_if_reasoning_effort_mismatch`, at most once per
  model, or per parallel instance of it, and requested rung) is written only the first time a
  mismatched rung arrives for a given model — and that memory belongs to the manager
  process, not the engine. Restarting or reloading the engine does not bring the warning
  back: the same mismatch against a freshly respawned engine is silent. Only the manager
  process itself restarting, or that pairing aging out of the most recent 64 remembered,
  lets the line be written again. Either way, every one of those requests is still served
  the locked value, whether or not a line was written for it that time.

## What this does NOT do

- It does not write anything into a request payload. The `thinking_budget_tokens` field of
  that payload is never touched by `reasoning_effort`, for any of the five
  rungs.
- **You may send both `thinking_budget_tokens` and `reasoning_effort` on the same request.
  They do not interact**, and neither is rejected for disagreeing with the other: your explicit
  `thinking_budget_tokens` is forwarded unchanged, whatever `reasoning_effort` says.
- A separate per-request mechanism (`clamp_reasoning_budget_for_ceiling`) is
  active and independent of this one: it reduces a model's reasoning budget on an
  individual request when the configured budget would leave it no room to answer, regardless
  of what got baked into the launch flag.

## In short

First, check that the feature can apply: the model's manifest needs a positive
`reasoning_budget`. Send `reasoning_effort` on the first request you make for a model -
that is the only request whose value takes effect. Then read `reasoning_budget_override`
in `GET /status` to confirm what a running engine is actually locked to.
