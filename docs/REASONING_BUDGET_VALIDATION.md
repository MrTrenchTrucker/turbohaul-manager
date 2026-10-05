# Cross-field validation for `reasoning_budget`

## The defect

A per-model manifest can set a thinking budget larger than the output ceiling the caller actually receives.
The model then spends its whole allowance inside `<think>` and never emits a completion. Both values are
individually legal, nothing compares them, and nothing warns — so it presents as "the model is broken".

Example: a 35B-class MTP reasoning model with `reasoning_budget: 3192` called with a caller cap of
**2048** can stop at exactly 2048 tokens without ever leaving `<think>`; the ceiling is caller-supplied, not the engine's.

## Why it was invisible

| layer | what it checks | what it missed |
|---|---|---|
| manifest flag bounds | `reasoning_budget` within `-1 .. 1_000_000` | nothing about the ceiling |
| manifest flag bounds | `n_predict` valid independently | `-1` is legal and means "unlimited" |
| request path | caller's `max_tokens` can set a lower ceiling than `n_predict` | the effective ceiling is not knowable from the manifest |

Whether the defect bites is a property of the **caller**, not the model. The same budget is fine on one model
and fatal on another:

- A 27B-class model — budget 3192, `n_predict` 20480 ⇒ satisfiable.
- A 35B-class MTP model — budget 3192, caller cap 2048 ⇒ unsatisfiable.

**Any model whose manifest carries a locked budget (`reasoning_budget > 0`) is exposed, so this is a general shape,
not a one-model quirk.**

## Why -1 is not a safe value

`is_thinking_payload()` (`api/chat_completion.py`) detects thinking mode **solely** via
`reasoning_budget > 0`. Setting it to `-1` therefore switches off thinking-aware handling entirely, including
the thinking-mode validate-and-retry path for `json_schema` requests.

**`-1` is not a safe remediation.** A positive value below the ceiling is. The request-time guards below act
only on a positive budget, so `-1` (unlimited thinking) escapes them too. Clearing a budget by setting `-1`
silently drops that handling.

## What ships today

1. **Save-time check** — a manifest validator **rejects** a manifest whose `reasoning_budget >= n_predict`
   while `n_predict > 0`, naming both values and the fix. `n_predict <= 0` (unbounded) is exempt: there is
   no ceiling to violate.
2. **Request-time warning** — the budget is compared against the *effective* ceiling (the smaller of a
   bounded manifest `n_predict` and the caller's `max_tokens`) and logged once per
   `(model, caller)` (the caller is the thread id, else the client IP), behind a bounded latch of 64 entries so a
   caller-influenced key cannot grow without limit; once a key is evicted from the latch it can be logged again.
3. **Request-time clamp** — the effective budget is reduced so an answer
   floor survives. The floor is half the effective ceiling, bounded to `[32, 2048]` tokens; when even the
   minimum floor would consume the whole ceiling, thinking is switched off for that one request by
   returning `0` — never `-1`, for the reason in the trap above. When nothing is pathological the request
   payload is left completely untouched.
4. **FE hint** — the per-model editor mirrors the full flag set and shows an inline conflict hint when
   `reasoning_budget >= n_predict`, programmatically associated with both fields. It is deliberately
   non-blocking: the caller's `max_tokens` is invisible to the editor, so a manifest-only check can be
   wrong in either direction.

The warning and the clamp share one computation of the effective ceiling, so they can never disagree about
what counts as pathological.

The request-time guards (2 and 3) are the ones that catch this failure on a model with
`n_predict: -1`, which the save-time check exempts by design. The save-time check and the FE hint are
guard rails.

**Still open:** whether a locked budget should be expressed as a *fraction* of the effective ceiling rather
than an absolute token count, given the ceiling varies per caller.
