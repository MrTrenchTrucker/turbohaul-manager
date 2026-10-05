# The Tool-Call Latency Guard

**What this document is for.** On one deployment of this system, a change
collapsed the wait that agent turns spent on their first token after a tool
call. This document explains where that wait comes from and what now stops a
future change from quietly bringing it back. The measurements came from that
operator's persisted telemetry and cannot be reproduced from this repository
(see section 9). It is written
for someone who has never seen the background and has no context at all. If you
are here because a test called
`tests/test_tool_call_latency_guard.py` went red and you want to know
whether you may edit it, read **"May I change the test?"** at the bottom. The
short answer is almost certainly no.

---

## 1. What happened

An agent using this system runs in a loop: it calls a tool, reads the result,
calls another tool. Between those calls it waits for the model to produce its
first token. That wait is the thing users actually feel.

At that deploy the wait collapsed: the median time to first token fell from
tens of seconds to a few seconds, and the share of requests taking over a
minute fell from a clear majority to a small minority. It stayed there
afterwards, across restarts.

### It was real, and it was not a measurement artifact

Three things were checked before anyone believed it:

- **The config was byte-identical** either side of the deploy.
- **The request rate was identical** either side.
- **There was a control.** A model that had no reason to move sat at the same
  value on both sides and did not move. A box-wide effect — a quieter machine,
  a faster disk, a kinder GPU — would have moved the control too. It didn't.

---

## 2. Where the time actually lived

This is the part that decides everything about how the guard is built.

The engine was never the problem. Prefill compute (the engine thinking) was
flat across the step change; the time lived in queue and placement (the manager
deciding). The time was **manager-side admission overhead** — everything that
happens *before* the engine is asked to do anything. A second, independent
instrument agrees: manager-side TTFT minus the engine's own reported
`prompt eval time` was far larger than the engine's own work in the slow era.
Two instruments, one conclusion.

### The unit of the loss is one model load

Averages over stages can mislead. The decisive measurement is the one that
splits requests into **two populations**:

| population | what it means | time to first token |
|---|---|---|
| **COLD** | the request needed a fresh slot assign — a model had to be loaded | much longer |
| **WARM** | the request reused a live, already-loaded slot | much shorter |

**The gap is about one cold model load** (tens of seconds for a large GGUF).
The entire regression, in a single unit.

So the win was **not** achieved by making anything faster. It was
achieved by **stopping doing something**: residents being torn down and then
reloaded for the very next request. The mechanism indicator confirms it — the
`grace_designated_victim_skip` event (a loaded model denied its grace window
and torn down) became far rarer after the change. Fewer teardowns → fewer cold
loads → the wait collapses.

---

## 3. Why the guard counts work instead of timing it

This is the design decision most likely to be questioned later, so here is the
full argument.

A unit test cannot measure real time-to-first-token. There is no GPU, no model
and no real engine in the test environment, and a timing assertion that depends on hardware is a
flake generator. So the guard does not measure time **at all**. It counts
**units of work**.

**The manager cannot lose tens of seconds by being slightly slow.** The dispatcher's
wake timeout and busy-path retry backoff are `_DISPATCH_DEFER_BACKOFF_S = 0.05`
seconds; the longer waits on the admission path (for example
`_VRAM_DEFER_BACKOFF_S = 1.0` seconds while an eviction is pending) are tied to
evictions. To lose tens of seconds at 50 milliseconds a time, a request would
have to go round the loop about a thousand times. It loses that time by doing
**unnecessary work**, and that work has a known unit price: **one cold model
load**. That is exactly what the cold-vs-warm split above says.

So the guard asserts that a request which *could* be served warm *is* served
warm, and costs nothing. Two properties follow, and both matter:

1. **It cannot flake.** There is no clock in any assertion. A loaded CI box, a
   slow disk and a fast laptop all produce the same integers.

2. **It cannot be quietly weakened, because there is no constant in it to
   bump. You cannot loosen zero.**

That second point is the whole reason for this shape. A duration budget — "warm
admission must complete in under 200ms" — has a number in it. The first time it
flakes on a loaded CI box, someone raises the number to 500ms, truthfully
reports "the test still passes", and the protection is gone with nobody having
lied and nobody having noticed. A test asserting *zero cold loads* has no such
dial.

A wall-clock margin is easy to get wrong: a 3.0-second margin can **pass against
unfixed, buggy code**, because ordinary test-environment overhead pushes the
buggy path's real timing past the chosen margin, so the test goes green without
proving anything. A timing guard is the shape most likely to end up weakened or
vacuous — which makes it the wrong shape for the one test that must never be
weakened.

---

## 4. What the guard actually asserts

Four counts, all in `tests/test_tool_call_latency_guard.py`.

### The first-token emitter still fires

`telemetry.on_first_token(slot, ttft_ms)` is the emitter that produces the
first-token measurement. It writes the `first_token` event, and it is the only
instrument that can **date** a future regression — that is, tell us not just
that we got slower but *which deploy did it*.

> ⚠ **Without this guard, nothing in the test suite would enforce the emitter's
> existence.** Nothing would go red if `on_first_token` was deleted, renamed,
> made conditional, or moved out of the streaming loop.

The guard drives a real streaming request end to end and requires the real
`first_token` event to appear, carrying a usable `ttft_ms` and a `slot_id`.
Note that an emitter *existing* is not the same as an emitter *firing on the
path production runs*: an emitter wired only into a deprecated code path can
sit unused while a different path actually serves traffic.
So the guard drives the route rather than inspecting the source.

### A warm turn triggers zero cold loads

The cost of one cold model load. If this goes red, agents are paying a model
load per tool call again.

### A warm turn triggers zero teardowns

The mechanism indicator, caught one step earlier than the cold-load count.
Teardowns are what *cause* cold loads.

### The warm admission path executes zero unconditional sleeps

The "somebody quietly adds an `await`" shape. Deliberately narrow: see
section 6.

### The anti-vacuity rule

"No work happened" is also true when **nothing** happened — a refused request, a
fixture that never reached the code, a typo in a model tag. So every zero-count
assertion in that file is paired with a **positive assertion** that the work
which was supposed to happen did happen: the request returned 200 and a real
token reached the client, and (for the sleep count) the admission function genuinely
executed.

This is not decoration. A *continuing* conversation, for example, never runs the
admission function at all, so a "no sleep ran under admission" assertion driven
by one would pass on an empty scenario. The recorder would be correct; the
scenario empty; and from the outside a green from an empty scenario is
byte-identical to a green from a real one. The sleep-count test therefore
proves the mechanism it judges actually executed.

---

## 5. What this guard CANNOT protect

Being honest about the gap is part of the deliverable. This guard does **not**
catch:

1. **Anything engine-side.** Prefill regressions are invisible to it, by
   design. Prefill was flat across the step change and is not what this
   protects.

2. **Real wall-clock time to first token.** If the median drifts upward while the
   work counts stay the same, this guard stays green. **It protects the shape
   of the win, not the size of it.** → See section 7: that half has its own
   instrument.

3. **Work that gets slower without getting more numerous.** A VRAM settle
   floor going from 5s to 30s would not be caught either: it is a sleep, but it
   sits on the eviction path, which this guard deliberately does not cover (see
   section 6). Nor would an `nvidia-smi` probe that simply becomes slow.

   ⚠ **A concrete example of this gap.** A warm turn can reach the same live
   resident by *two* independent routes: the resident's own grace loop pops the
   same-thread follow-up, or `_route_or_reserve` finds the still-live resident.
   If a change broke only the first route, the follow-up would sit in the queue
   until the grace window lapsed and then be served by that same resident —
   **later, but with no extra spawn and no extra teardown.** The counts stay at
   zero and this guard stays green. That delay is real latency and it is
   section 7's job, not this file's. It is named here because it is a real
   case, not a guess: removing either one of the two routes on its own leaves
   the guard correctly green, because the property under test — a warm turn
   costs no cold load — is genuinely still true.

4. **Scenarios the fixture does not build.** It protects the shapes it drives —
   a continuing conversation, and a new conversation joining a resident model —
   not "the admission path" as an abstraction.

5. **Non-streaming requests, entirely.** The TTFT instrument only exists on the
   streaming path: `on_first_token` has exactly one call site, inside the
   streaming handler. A non-streaming request emits no `first_token` event at
   all, so the measured population is **streaming requests only**. This is
   fine for agent workloads, which stream — but the guard cannot protect a
   number that is never measured. *(A small, unmeasured slice of admission is
   also excluded from `ttft_ms` itself, because the clock starts just after
   submission rather than at arrival.)*

6. **Deployment.** It runs on code, not on the image actually serving traffic.
   Code can land without being deployed.

---

## 6. What the guard deliberately does NOT re-assert

The dispatcher's wake-on-enqueue behaviour — that a waiting request is woken by
an event rather than by a poll timeout — is a genuine part of this protection,
but it is **already covered elsewhere**, and this guard does not restate it.
That coverage lives in:

| file | class |
|---|---|
| `tests/test_manager.py` | `TestDispatchLoopWake` |
| `tests/test_evict_teardown_wake.py` | `TestVramOvercommitAdmissionWake` |
| `tests/test_eviction_gate_wake.py` | `TestThirdMomentWake` |
| `tests/test_turn_boundary_handoff.py` | `TestParkFiresNotifyAndDeferredRetryWakesEarly` |

> ⛔ **Those four tests are part of this protection.** Deleting one of them
> removes wake coverage that the latency guard was deliberately written *not*
> to duplicate. If you are pruning tests, they are not spare.

Duplicating them would have let the guard claim protection it did not actually
add — and a guard that claims more than it can prove is worse than a narrow one
that is honest.

**The sleep count is also scoped to the *warm* path on purpose.** The eviction path
legitimately sleeps: VRAM was observed not to begin dropping for more than two
seconds after a model unload, so a one-time settle floor before polling skips
samples that are guaranteed misses. Three of the four `verify_vram_cleared` call
sites in `manager.py` pass zero; the fourth keeps the default floor (5
seconds), where it gates a notification a waiting request depends on. Widening it to cover eviction would make it fail on correct,
deliberate code — and a guard that fails on correct work teaches people to
disable guards.

---

## 7. The other half: the smoke check

This guard protects the **shape** of the win in the test suite, deterministically,
on every run. It cannot see the **size** of the win, because the test environment
has no GPU.

The other half is a **smoke check** run outside the unit tests, against **live
telemetry** on a real deployment: it fails when the median time to first token,
computed from the `first_token` events, exceeds a threshold. That check is not
part of this unit-test suite. Together:

| | what it watches | where | when |
|---|---|---|---|
| **this guard** | the shape — no cold loads, no teardowns, no sleeps, yardstick alive | unit tests, no hardware | every test run |
| **smoke check** | the size — real median TTFT from live telemetry | a real deployment | at release |

Neither replaces the other. If you weaken this guard, the smoke check will
eventually notice — *after* the regression has shipped and users have felt it.
That is strictly worse than catching it in the test suite, which is the entire
point.

---

## 8. May I change the test?

> **What this test protects**
>
> This test may be changed **only to make the system faster**, and **only with
> proven results**. It may **not** be changed to make a build pass. It may
> **not** be changed because it is inconvenient.
>
> **If it goes red, the change is the suspect — not the test.**

Concretely:

**If the guard goes red on your branch,** the default explanation is that your
change made agents wait for a model load they used to avoid. Read the failure
message; it names the counts. Fix the change.

**You may edit this test if** you are making the system genuinely faster and you
have measured proof — a before and an after, on the same workload, with a control
that should not move and did not.

**You may not edit this test to:** make CI green, unblock a release, remove a
flake (it has no clock and cannot flake — if it "flakes", that is a real
intermittent regression and it is telling you something), or because the
fixture configuration looks arbitrary. It is not arbitrary: several fixture
values are load-bearing and each is commented in place explaining what breaks
if it changes.

**If you believe the test is genuinely wrong,** do not quietly re-point it.
Raise it with the maintainers and record the reasoning with the change, the
same way this document records its own.

---

## 9. Provenance

- The first-token measurement, the control, the cold/warm population split
  and the `grace_designated_victim_skip` rates were measured from persisted
  telemetry (the `flap_*.jsonl` files `telemetry.py` writes, by default under
  `/var/lib/turbohaul/telemetry`) on one deployment. They are cited here in
  general terms, not re-derived — this repository cannot read those files.
- The test shape (count work, never time it) was chosen deliberately, for the
  reasons in section 3.
