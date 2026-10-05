# Fast Lane — request priority

**Which Fast Lane doc do you want?** The architecture (how Fast Lane behaves, the specification) is [design/FAST_LANE_ARCHITECTURE.md](design/FAST_LANE_ARCHITECTURE.md), the architecture doc for Fast Lane. The dashboard page is [frontend/03-fast-lane.md](frontend/03-fast-lane.md). This page is setup and day-to-day use.

![Fast Lane - the rig running the priority lane past queued traffic](fastlane-banner.png)

Fast Lane lets you decide **which waiting request gets the GPU next**.

By default the manager serves requests in the order they arrive. That is fair, but it is not always what you
want: an interactive chat can end up queued behind a burst of background automation, and the person waiting is
the one who notices. Fast Lane lets you put named callers — and specific kinds of traffic from those callers —
ahead of the rest.

It is **off by default**. While it is off, Fast Lane has no effect on request ordering.

---

## What it does, and what it deliberately does not

**Fast Lane is non-preemptive priority scheduling: priority admission and turn-boundary eviction. It changes which
waiting request is served next and, when the server is full, which idle model is unloaded to make room for a
higher-priority request. A request's priority is looked up once, when the request is accepted, and Fast Lane never
interrupts a turn that is already running.**

| effect | what it means |
|---|---|
| ✅ Changes which *waiting* request is served next | Only requests that are already queued are reordered. A listed caller is served ahead of unlisted ones that were waiting before it, and two listed callers are ordered by their rule's position in the table first, then by tag rank — `(rule_index, rank)`, oldest first on a tie. Nothing already running is affected. The priority is looked up once, when the request is admitted; the order among the requests that are waiting is worked out again each time the next one is picked. |
| ✅ Ends a grace hold early for a model that has to give way, **at a turn boundary** | When a higher-priority request is waiting for room, the loaded model with the worst effective rank that the request strictly outranks (the *designated victim*) gives up its grace window as soon as its current turn finishes, instead of sitting out the rest of the window. When the incoming model spans every card (layer-split, see [Placement and co-residence](PLACEMENT_AND_CORESIDENCE.md)) every loaded model has to go, so several models can be designated victims at once, each one the request strictly outranks giving up its grace at its own turn boundary. A model with a strictly better request parked in its own queue also ends its grace then, and admits that request next. **A model that is not designated keeps its full grace window**, however long a higher-priority request waits. The **idle-unload countdown** needs no such step: once grace expires the resident sits in `IDLE_EVICTABLE` and can be unloaded on demand for the whole countdown, apart from the few protections listed under [What protects a resident from eviction](#what-protects-a-resident-from-eviction). See [Grace and idle-hot holds](#grace-and-idle-hot-holds). |
| ❌ Never interrupts a reply that is already being generated | Whatever is running finishes, always. You become next in line, not next token — a grace hold is only ended, and a model is only unloaded, *between* turns. |
| ❌ Never starves unlisted traffic **beyond the floor you set** | A wall-clock fairness floor promotes the longest-waiting *unlisted* request once it has waited too long. "Never starves" is only as strong as that number, and out of the box it is set to the highest value it accepts, an hour — so on a fresh install this row promises very little until you lower it. The [Configuration reference](#configuration-reference) gives the shipped value; `GET /api/config` gives what your own box is running. A listed address that is simply outranked by another listed address is not eligible for the floor at all — see [Set the fairness floor](#3-set-the-fairness-floor). |
| ⚠ May force a model swap, and that swap is not rationed | Serving a request for a model that is not currently loaded means unloading whichever model is loaded and loading that one instead — seconds of work, sometimes considerably more. The per-minute swap budget applies to *unlisted* traffic, not to Fast Lane traffic, so a priority request for an unloaded model can trigger a swap at any budget value, including `0`. See [Cross-model jumps](#cross-model-jumps). |

Because it only reorders waiting requests and chooses which idle model gives way, Fast Lane helps when requests
are genuinely **competing** for the GPU. If nothing else is waiting, there is nothing to reorder and you will see
no difference.

### What protects a resident from eviction

A loaded model counts as an idle candidate for being unloaded to make room only while all three of these are true:

- it is idle and evictable (its grace window is over), not still in grace or loading;
- it is not running a turn — a resident that is mid-turn, including mid-prefill, is never unloaded under
  that turn, and one that was chosen earlier but then accepted a follow-up is unloaded only at the turn
  boundary;
- it does not hold an unexpired staleness grant. While Fast Lane is on, each time a client's idle model is
  unloaded the manager records the eviction against that client's address. From 20 seconds after the latest
  eviction, and for the next 60 seconds, that client's idle model is protected from being unloaded again; the
  grant is spent when a turn for that client completes.

For an idle candidate, which is the list the placement below counts from, those three are the only
protections. A listed request that is merely *waiting* for a model does **not** protect that model's
resident. So if the only idle model on the box is the one a listed request is waiting
for, a different request that needs that memory can unload it, and the waiting request loads it again
afterwards.

### Unloading idle models to keep a model on one card

This section is about where a model that is set up for one card is placed when it is loaded cold.

**The order.** The manager first takes the automatic placer's own pick: the most-free single card that fits the
model on the live free-memory reading, which needs no unload. When the placer finds no such card (it would
spread the model across all the cards as a layer split, or nothing fits and the manifest's pin is kept), the
manager looks for one card that would hold the model once the idle models on it are unloaded. For each card it
counts as free memory the idle one-card models on that card (idle and past their grace window, not running a
turn, not under a staleness grant, so the ones that pass the protections above) and any unloads already in
flight on it. A card with neither is not considered. If such a card fits the model and its context, the model is
sent to that card and the idle models on it are unloaded to make room. Each one's cache is saved first when it
can be (the save needs a live engine and a recorded thread, and is skipped with no log line without them; a
refused or failed save is logged, and the unload goes ahead without it in every case), and a running turn is
never cut. Each retry of the request unloads one more idle model
while the card still does not fit. Only when no single card fits even so does the model run split across the
cards, as it otherwise would.

**Only the chosen card.** The unload lookup is limited to the chosen card, so a model on another card is left
alone. When no idle model is left on that card, the existing lookup can instead take a model that was
designated to give way, whose grace window has ended and which is not running a turn.

**When a split model is loaded.** This placement does not act while a model that is split across cards
(layer, row or tensor) is loaded: the usual fit check refuses every card for a one-card model then, so no
card is chosen, and the usual placement and make-room paths apply. Such a model is not counted toward any
card's free memory either, because its share on each card is not recorded.

**Who it applies to.** It applies to every request whose card the automatic placer decides: a model whose
manifest is set for one card and is auto-placed, which means the manifest sets `auto_place` or the lane is
on. That covers listed Fast Lane clients and unlisted clients while the lane is on, and requests for an
`auto_place` model while the lane is off. A model that is not auto-placed while the lane is off stays on its
pinned card. A model whose manifest is itself split across cards (layer, row or tensor) is never moved to one
card this way.

**What it counts.** Memory reserved by models that are still starting up on a card is netted off that card's free
figure. Unloads already in flight are counted once.

**The safety checks.** The chosen card must be admitted by the usual fit check with its idle models counted
as free, and it must still have the safety floor free after the model and its context are placed. A one-card
model also leaves roughly 1.3 GB of CUDA context on every card it is not placed on (see
[Placement and co-residence](PLACEMENT_AND_CORESIDENCE.md)). So on every other card, the free memory left,
net of models still starting up there and after that 1.3 GB allowance, must still clear the safety floor. A
card that fails either check is skipped.

**When several cards fit.** Rank never decides whether a card fits; it only orders cards that already fit. The
cards are first compared by how well protected the models are that each would have to unload. For each card, take
its idle one-card models in the unload order (models of unlisted clients first, then listed clients from the
lowest priority up, and inside one client the model with the worse class first; recency only orders models inside
one card) and keep taking them until their memory covers what still has to be freed on that card. The card is
ranked by the most protected model among those. A card that needs nothing unloaded, because it fits as it stands
or through unloads already in flight, unloads nobody and is best. The card whose most protected model is least
protected wins, so a lower-priority client's idle model can be unloaded to keep a higher-priority client's idle
model on another card, even if that costs more MiB. Only when two cards rank equal (same tier, same rule position
and same tag rank) do the tie-breaks apply: the card that still needs the fewest MiB unloaded, then the card with
the most headroom, then the lowest card number. With no Fast Lane rules every model counts as unlisted, so all
cards rank equal and only the tie-breaks apply. Once a card is chosen, the worst-ranked idle model on it goes
first, then the least recently used.

**In the log.** When no idle model can be unloaded and the request waits, the make-room starvation line carries
`card_scope=`: `all` for requests that look at every card (auto-placed ones and the ones this placement
covers), otherwise the request's own card.

**Retrying.** The decision is taken again every time the request is retried. When it no longer holds, for
example because the placer now finds a card that fits on its own, or no card fits any more, it is dropped and
the placement is exactly what it would have been without it.

---

## How priority is decided

Two levels.

### Level 1 — which client it is (the baseline)

An **ordered list** of rules. **Position in the list is the priority**: the first entry is served first, the
second next, and so on. A client matching no rule is normal priority and is served after every listed one (unless the fairness floor promotes it).

Each rule identifies its client by **exactly one** of two things, and which one you should use depends on
where the caller is:

| | use | why |
|---|---|---|
| **On the docker network** | `container_name` | Docker assigns container addresses dynamically unless you pin them, so a container's address can change when it restarts. A rule pinned to today's address silently starts prioritising whichever container inherits it next — the rule keeps working, on the wrong client. A name does not move. |
| **Off the docker network** - LAN, a VPN or any other overlay network, anything with a fixed address | `address` | There is no container name to resolve. The address is the only stable identifier, and off-network addresses are normally static anyway. |

**Names are resolved forward, in the background, and never on the request path.** A refresher periodically
looks up each configured name and caches the set of addresses it currently holds — every address, on every
shared network, in both IPv4 and IPv6 form. Matching a request is then a set-membership test against that
cache. **No name lookup ever happens while a request is being admitted**, so a slow or dead DNS resolver
cannot add latency to a request or block one. A lookup that fails clears that name's cached addresses, so its rule
matches nothing until the name resolves again (a warning is logged), and a name that resolves to nothing yields an
*empty* set.

An empty set matches nothing. That is deliberate and it is the safe direction: a rule whose container is
misspelled, stopped, or not yet started simply does not match, and the request falls through to the next rule
— it can never be promoted into a *different* client's lane. **The manager warns rather than failing
silently**: a name that resolves to no addresses produces "rule N names container 'X', which currently
resolves to NO addresses -- that rule matches nothing and the client has no Fast Lane priority. Check the container
name and whether it is running." in the `warnings` list of the `PUT /api/config` response and in the log.

Two other config errors are reported the same way, both naming the two rules involved:

- **two rules on the same address** — the second is unreachable;
- **two rules naming the same container** — likewise. The first rule already claims every address that name
  resolves to, so the ordered scan never reaches the second. It sits in the table looking configured and does
  nothing.

Container names are matched **exactly**, including case, because that is how they are handed to the resolver.
`Gateway-Svc` and `gateway-svc` are two different names here.

### Level 2 — the traffic class, within one address

The same caller often does several different jobs. A single process might send interactive turns, background
summarisation, and automated sub-tasks — all from one address. So each address entry carries a small table of
**ranks from 1 to 5** (1 is served first) for the classes the manager can distinguish:

| Class | Meaning |
|---|---|
| `main` | a primary interactive conversation |
| `curator` | context-curation work |
| `compression` | history-compaction cycles |
| `sub_agent` | delegated background tasks |
| `unclassified` | requests arriving with no class markings at all |

**See also:** [TAGS_AND_IDENTITY.md](TAGS_AND_IDENTITY.md) — how a caller labels its requests, so the manager can tell these classes apart.

Leave the whole table blank and everything from that address is treated equally — the address alone carries
the priority. Fill it in and the classes of that address are ranked against each other the way addresses are
ranked against each other: the address decides first, and the class rank is only ever compared between two
things that belong to the **same** address. It never lifts one address above another.

Ranks may repeat. Two classes sharing a rank simply means first-come-first-served between them.

**Where the class rank applies.** Inside one address, the class rank is used in four places, and in each of
them it works the way the address rank works between addresses:

- **Waiting requests.** A waiting request with a better class is served before a waiting request of the same
  address with a worse class (see [The effective order](#the-effective-order)).
- **Same-thread follow-ups.** A follow-up of the thread that holds a model's grace window normally goes
  straight back to the warm model. It no longer does when a request of the same address with a strictly better
  class is waiting, whether that request sits in the queue or is parked in the busy loaded model's own queue:
  the follow-up waits behind it. A follow-up of an equal class keeps the warm path, and a request that is
  already running is never affected.
- **Making room.** A waiting request with a better class can displace a loaded model of the same address that
  holds a worse class, under exactly the rules that apply between addresses. It happens only at that model's
  own turn boundary: a running turn (prefill included) always finishes first, then the model is unloaded, then
  the better request's model is loaded. Other loaded models keep their grace. The choice is made fresh at each
  decision point, the waiting request is woken rather than left to poll, the displaced model's context is saved
  before it is unloaded (when it can be), and the swap-budget exemption, the protections listed under
  [What protects a resident from eviction](#what-protects-a-resident-from-eviction) and a re-queued request
  keeping its place apply as they do for addresses. A better-class request parked in the holder's own queue
  ends the holder's grace at its turn boundary; the holder is not unloaded and admits that request next.
- **Which model is unloaded first.** Among the idle models, an unlisted client goes first, then the listed
  address with the lowest priority; inside one address the model that holds the worse class goes first.
  Least-recently-used breaks a tie only between equal classes.

A loaded model's class is the class of the latest request it served. A follow-up served inside the grace
window updates it, and that one value is used both for choosing what gives way and for the unload order.

**A worked example.** Address A is listed first, with `main` at 1 and `sub_agent` at 3. Address B is listed
second, with `main` at 1. A model Q is loaded and its latest request was an A `sub_agent` turn; a second model R
is waiting to be loaded for an A `main` request, and there is not room for both models.

- A's `main` request may displace Q once Q's turn is over, because it is the same address and 1 beats 3. Q's
  context is saved (when it can be), Q is unloaded and R is loaded.
- A B `main` request does not jump ahead of an A `sub_agent` request in the queue, although 1 beats 3: the
  address decides first, and B is listed after A. The class rank never lifts one address above another.
- If Q's latest request had been an A `main` request instead, the two classes are equal, so there is no
  displacement: a tie.

### The effective order

```
(position of the matching address in the list, then the class rank, then arrival time)
```

Anything matching no address is served after everything that does, unless the fairness floor promotes it.

The same pair, address position first and class rank second, also decides between a waiting request and a
loaded model of the same address (a better class may displace a worse one), and which of an address's loaded
models is unloaded first (the worse class goes first).

### Priority, as a decision table

The same rule, stated as comparisons rather than prose — useful if you're checking a specific ordering by hand
or parsing this doc programmatically:

| Comparing | Result |
|---|---|
| A listed address vs. an unlisted address | The listed one wins, unless the fairness floor has promoted the unlisted request. |
| Two different listed addresses | The one earlier in the `rules` list wins. Two *different* listed addresses can never tie — list position is unique. |
| Two requests from the **same** listed address, different tag rank | The lower rank number wins (`1` beats `2`). |
| Two requests from the same address, **equal** tag rank | A real tie — first-come-first-served between them. Equal rank does **not** grant priority. |
| Two unlisted addresses | Fast Lane does not reorder them; the queue's ordinary ordering applies (arrival order, plus the queue's own main-lane, compression and loaded-model preferences). |

`created_at` (arrival time) only ever breaks a tie between candidates already equal on address-position and
rank — it can never manufacture a priority difference between two otherwise-equal requests.

---

## Why the address is the baseline and the class only ranks within an address

The class markings are **supplied by the caller**. Anything able to reach the API can claim to be whatever it
likes. The source address is **observed by the manager** and cannot be set by the caller.

So trust flows in the sensible direction: the address decides *how much* you trust a caller, and the class only
sorts traffic *within* a caller you have already chosen to list. A class marking on its own never grants
priority — an unlisted address is normal priority no matter what it claims to be.

> Fast Lane is a **scheduling preference, not an access control**. It decides service order. It does not decide
> who may connect, and it must not be used as though it did.

### One class per request

Class markings are not mutually exclusive — a curation pass may legitimately also mark itself a sub-task. Each
request is therefore resolved to exactly **one** class first, using the same resolver the rest of the manager
uses, and only then given that row's rank. This guarantees the priority you see can never disagree with the
classification the rest of the system applies.

---

## Grace and idle-hot holds

Two separate mechanisms normally keep a model resident on the GPU after a reply finishes, so a same-thread
follow-up can reuse the warm process instead of paying reload cost: a short **grace window** right after a
reply, and a longer **idle-hot hold** once grace expires. Grace is ordinarily deaf to anything else — a thread
holding grace keeps holding it, whatever else is waiting.

**Fast Lane cuts a grace window short only for the model or models that have to give way, and only at the boundary
between turns.** If a higher-priority request is waiting for room and the loaded model with the worst effective rank is
one that request strictly outranks (the designated victim), that model gives up its grace window — at the turn
boundary, or the moment it is designated if that happens inside the window — and becomes the model that is
unloaded to make room. When the incoming model spans every card (layer-split), several loaded models can be
designated at once, and each one the request strictly outranks gives up its grace the same way. A model with a
strictly better request parked in its own queue also ends its grace at the turn boundary, but it is not unloaded:
it admits that request next. Every model that is not designated keeps its full grace window, even while a
higher-priority request waits. **A turn that is already generating is never touched** — the
override only changes what happens *next*, never what is happening *now*.

Inside one address the same comparison uses the class rank, and a better-class request parked in the holder's own
queue ends the holder's grace at the turn boundary; see
[Level 2](#level-2--the-traffic-class-within-one-address).

A thread that keeps sending follow-ups can extend its grace window, but only up to `queue.max_grace_extensions`
times (default `5`).

**Grace and idle-hot are configured independently of Fast Lane**, in the same runtime-mutable `queue` config
section:

| Setting | Default | Overridable via |
|---|---|---|
| `queue.grace_seconds` | `30` | `PUT /api/config` (runtime), or `TURBOHAUL_GRACE_S` env var (boot) |
| `queue.idle_hot_load_seconds` | `600` | `PUT /api/config` (runtime), or `TURBOHAUL_IDLE_HOT_S` env var (boot) |

Both can also be changed at runtime the same way Fast Lane's own settings are (see
[Setting it up without the UI](#4-setting-it-up-without-the-ui--api-and-config-file)) — they live under the
`queue` key rather than `fastlane`, so read/write that section instead. **If a deployment's actual grace or
idle-hot behavior doesn't match the numbers above, check for an env var override before assuming the default is
wrong** — a container's env is a common place for this to differ from a fresh install.

---

## Setting it up

### 1. Turn it on

**Settings → General → Fast Lane.** One switch. It takes effect on the next request; no restart.

The Fast Lane screen itself lives at **Queue → Fast Lane**. It stays visible when the feature is off, greyed
out, so you can always find it.

### 2. Add addresses from the Discovered list

**You never type an address.** Any address the manager has not seen before that makes a request appears in the
**Discovered** list — whatever markings it carried. Each entry shows:

- how many requests it has sent
- when it was first and last seen (the first-seen time is hidden on narrow screens)
- which classes it has actually used
- which models it has asked for
- the container name behind it, once the manager has confirmed it (a dash in the desktop table until then; no name line on the mobile card)

Click an entry to add it to the priority list, then set its ranks.

> **Why clicking and not typing.** The manager compares the address it actually observed. Depending on how a
> client reaches the service, that may be recorded in an IPv4-mapped IPv6 form rather than plain dotted-quad.
> A hand-typed address can therefore fail to match **silently, forever**. Clicking stores exactly what was
> observed. (Matching itself normalises both forms, so a rule added either way will match — but the Discovered
> list removes the chance of a typo that matches nothing.)

When the manager has confirmed a container name for an address, the **Name** column shows it, and **Add to Fast
Lane** on that row saves a rule by container name, so the rule keeps matching when the container restarts and gets
a new address. The lookup runs in the background and handles at most four addresses every five seconds, so a name can take
longer to appear when the Discovered list is long; an address that does not resolve is retried after 30 seconds, and
the wait doubles on each miss up to ten minutes. For a client with no name (not a container, or a
name that does not resolve from the manager's network), the column shows a dash and **Add to Fast Lane** saves a
rule by address; the manager may then warn that the address looks docker-assigned, and once the name is known you
can add that client by name in the **Add by container name** box below the **Rules (priority order)** heading,
above the list of rules. The name is only shown and used to build the rule; it is never used to match a request.

Some rows may be the manager itself or a local tool rather than a client, for example a loopback address. A name
comes from a reverse lookup on the host, so what it shows depends on your resolver. Do not add rows that are not
clients.

### 3. Set the fairness floor

One number, in seconds: **the longest any *unlisted* request may wait** while priority traffic goes first. When
a waiting unlisted request crosses it, it is served next regardless of priority. This is what stops a busy
period from starving background work, and it is measured on the clock rather than by counting how many requests
jumped.

> ⚠ Every default in this document is the value the software ships with, which is not necessarily
> the value your deployment is running. Someone may have tuned this box, or it may still be
> carrying settings saved by an earlier version. This is the single number deciding how long
> ordinary background work can be held behind priority traffic, so before relying on it, read what
> is actually set rather than what is written here:
> `curl -s http://<host>:11401/api/config | jq .fastlane`

This floor rescues **unlisted** requests only. A *listed* address that is simply outranked by a higher-priority
listed address is not eligible for it — it waits until the higher-priority address stops sending, exactly as
the priority table says it should. If a lower-ranked listed caller is waiting too long, change the order of the
`rules` list or its tag ranks; lowering this number does not help it.

### 4. Setting it up without the UI — API and config file

Everything above is doable over HTTP, which is how an agent or a provisioning script should do it. Settings
apply to the next request; no restart.

**Read the current state.** The Fast Lane block lives under the `fastlane` key:

```bash
curl -s http://<host>:11401/api/config | jq .fastlane
```

**Find the addresses — do not invent them.** The same rule as the UI applies, for the same reason: match on
what the server actually observed.

```bash
curl -s http://<host>:11401/api/fastlane/census | jq '.entries[] | {address, served_count, tag_classes_seen}'
```

Copy the `address` value verbatim out of that response into your rule.

**Write it back.** `PUT` the whole `fastlane` section. Position in `rules` is the priority — index 0 is served
first:

```bash
curl -s -X PUT http://<host>:11401/api/config \
  -H 'Content-Type: application/json' \
  -d '{
    "fastlane": {
      "enabled": true,
      "max_normal_wait_s": 3600.0,
      "cross_model_switches_per_min": 2,
      "census_ttl_hours": 168,
      "rules": [
        { "address": "<copied from the census>", "label": "primary agent",
          "tag_ranks": { "main": 1, "curator": 2, "compression": 3,
                         "sub_agent": 4, "unclassified": null } }
      ]
    }
  }'
```

> ⚠ **`GET` returns a field that `PUT` refuses.** The read response carries a read-only `config_error`
> diagnostic key. If you fetch the section, edit it, and send it straight back, the write is rejected with
> `Extra inputs are not permitted`. **Strip `config_error` before writing.** This bites anyone doing the
> natural read-modify-write loop.

> ⚠ **Use the key `fastlane` for both.** `GET` and `PUT` both use the top-level key `fastlane`. A `PUT` under any
> other top-level key that the API does not know is rejected with HTTP 400 (`unknown section(s)`), so a misnamed
> section fails loudly instead of being ignored.

**Confirm it took**, rather than assuming the `200` meant what you wanted:

```bash
curl -s http://<host>:11401/api/config | jq '.fastlane | {enabled, rules: [.rules[] | {address, tag_ranks}]}'
```

Check the rule *content*, not just the count — a rule list of the right length can still be the wrong rules.

---

## Addresses that cannot be told apart

Several distinct callers can arrive under a single address — for example when they reach the service through a
shared gateway or proxy rather than connecting directly. When the manager sees one address carrying two or more
different classes, the census response (`GET /api/fastlane/census`) marks that entry `ambiguous` and carries this
text in `ambiguity_reason`:

> *A rule on the address alone will apply to all of them - if these are different callers they cannot be told
> apart here; add a tag to narrow it.*

(A "tag" here is a class rank.) The Fast Lane screen does not read the `ambiguous` field: under a rule on an
address it shows a similar warning, worked out from the census entries that carry that address. The field itself
is only in the census response. It is derived from what has actually been observed, not from any assumption about your network. If you
need callers behind such an address to be prioritised separately, they must reach the manager in a way that
preserves their own address; otherwise use class ranks to distinguish their traffic instead.

**How much one address can carry.** A single bridge-gateway address can accumulate requests spanning every traffic
class and many different models. One address, many callers, no way to tell them apart from the address alone —
that is the whole limitation, visible as one row of the Discovered table.

⭐ **A container-name rule does not fix this, and it is worth being clear about why.** Naming the container
resolves that name to the addresses the container itself holds. It does not recover the identity of a caller
that reached the manager *through* a gateway, because that caller's own address never arrived. What the name
does fix is the opposite problem — a caller whose address is real but *moves*. Use `container_name` for drift;
use class ranks for callers you cannot tell apart. They solve different problems and neither substitutes for
the other.

**A related case a name rule DOES fix outright:** the same client reaching the manager over both IPv4 and
IPv6. With only one of the two addresses listed, that client is prioritised on one path and silently unlisted
on the other — the misses are easy to overlook precisely because most of its traffic *is* matching. A name rule
covers every address that container holds, in both families, so this class of miss disappears.

---

## Cross-model jumps

Serving a priority request for a model that is **not currently loaded** means unloading the current one and
loading another. That costs seconds, sometimes considerably more, and a rapid succession of such jumps will
thrash the GPU and slow everything down — including the request that jumped.

`cross_model_switches_per_min` rations those swaps, but **it does not apply to Fast Lane traffic**. A request that
matches a rule is never refused a swap on budget grounds, at any value of the setting, including `0`. The budget
gates *unlisted* requests instead: while Fast Lane is on, an unlisted request that needs a cross-model swap while
no room is free is refused once the budget for the minute is spent, and stays queued. Each refusal is counted and
shown (see [Seeing whether it is working](#seeing-whether-it-is-working)). The fairness floor never applies to a
listed request either way, since that floor only ever promotes a waiting *unlisted* request.

What you are trading, for a priority caller: a swap can leave the very request that triggered it **worse off**. A
priority request that jumps to an unloaded model waits for the current model to be unloaded and its own to be
loaded before it can start, and that wait can easily exceed the wait it skipped by going first. No setting avoids
that cost for a Fast Lane request. Whether the trade is worth it depends on how often your priority callers ask
for a model that is not already loaded — if they mostly share one model, it rarely comes into play.

---

## Seeing whether it is working

### The number that says how many requests are waiting

The dashboard's Queue card leads with the **waiting total**, and the Queue tab shows the same number as
**Requests waiting**. Read that one; it is the only number that counts everything actually waiting.

**Why there is more than one number, and why the older one reads low.** A request whose model already has a
live resident is handed straight to that resident's inbox and never enters the staging queue at all. The
staging figure — the familiar `N / 100` — therefore counts only requests in that one holding area, and on a
busy box it can sit at **0** while real work is queued behind busy residents. Requests typically spend only a short
time in staging before a free resident takes them, so on a busy box that number can read 0 and blip to 1 rather
than hold steady. It is not wrong about staging; it answers a narrower question than the one being asked of it.
Staging and the acceptance buffer are still shown underneath, because they are real and they are useful when you
want to know where in the pipeline something is sitting.

⚠ **What the inbox route does and does not keep.** A request that goes straight to a resident's inbox is still
ordered by Fast Lane priority — that ordering is not lost. What it does not get is the **fairness floor**,
which promotes a long-waiting *unlisted* request and is applied by the queue those requests bypassed. So on a
box serving mostly warm residents, the floor protects less traffic than the fairness section above might
suggest.

**If the number shows an em dash instead of a value**, the manager you are talking to does not report a
waiting total — it predates the field. That is shown as unknown rather than as `0`, deliberately: a fabricated
zero is indistinguishable from "nothing is waiting", which is wrong in the one direction that matters.

A scheduler whose effects are invisible is indistinguishable from a broken one, so the screens show what they can:

- **Queue tab → Waiting requests** — each request that has reached staging, with its client (the rule's label,
  or a thread id prefix when unlisted), its tag rank and its status; the status names the likely victim when a
  turn has to complete first. Beneath it, **Fast Lane claims** lists requests that are waiting for room and have
  not reached staging yet, with their model, client, rank, reason and how long they have waited.
- **Fast Lane screen → Counters** — refusals by reason, which are the swap-budget refusals described under
  [Cross-model jumps](#cross-model-jumps). The served/jumped-per-class and fairness-firing counters are shown as
  "not yet available".
- **Banners on the Fast Lane screen** — one when the feature is off, and one when a saved configuration was
  rejected at boot and Fast Lane started disabled, with the rejection text.

### From the logs — for scripts, agents, and anyone without the UI open

The manager logs a specific line for each outcome. Grepping for these is the ground truth:

| Log line | Fires when | Proves |
|---|---|---|
| `QUEUE_PRIORITY served_class=... reason=fastlane ...` | A listed rule actually won a pick. | Fast Lane changed which request was served next. |
| `QUEUE_PRIORITY served_class=... reason=fastlane-floor ...` | The wall-clock fairness floor rescued an unlisted request. | The floor is doing its job — this is expected behavior, not a bug. |
| `FASTLANE_DECLINED reason=staging_empty ...` (DEBUG-only) | The picker ran with nothing queued. | Silence elsewhere is explained — there was nothing to pick from. |
| `FASTLANE_DECLINED reason=no_candidate ...` (DEBUG-only) | The picker ran, but no *listed* request was in the scanned window. | Fast Lane looked and found nothing eligible — not that it failed to look. |
| `QUEUE_ADMITTED_NO_PICK via=pop_matched_thread ...` | A same-thread continuation was served without going through the priority picker. It yields to a waiting request of the same address with a strictly better class, and stays waiting in that case. | This request's ordering wasn't decided by the picker - expected for same-thread follow-ups, not a malfunction. |
| `MATCH_DECLINED_BETTER_RANK_WAITING via=... staging_depth=... accept_depth=...` (DEBUG-only) | A same-thread or same-address follow-up was skipped because a request of the same address with a strictly better class is waiting. | Explains why a follow-up did not take the warm path: the better request is served first. |
| `MAKE_ROOM_EVICTION reason=... model_tag=... fastlane_client=... rule_index=... rank=...` | A loaded model was unloaded to make room. `rank=` is the class rank, inside its address, of the model that was unloaded; it is `unresolvable` when the model has no identity that lookup can resolve, and `rank=` and `rule_index=` do not always resolve together. | Shows which class the unloaded model held, so a same-address unload can be read off the log. |
| `MAKE_ROOM_STARVED reason=... why=... model_tag=... fastlane_client=... rule_index=... ... rank=...` | No idle model could be unloaded and the request waits. `rank=` is the last field: the class rank of the incoming request inside its address, or `unresolvable` exactly where `rule_index=` is. | Shows the class of the request that is waiting for room. |
| `grace_designated_victim_skip slot=... model=... resident=...` | A model chosen to give way, or one with a strictly better request parked in its own queue, finished a turn and skipped its grace window. | The turn-boundary override described above fired. |
| `grace_designated_unload_target_break slot=... model=... resident=...` | A model was designated to give way while already inside its grace window, and the wait ended early. | The same override, fired mid-window. |

A single request's outcome is usually explained by one of these lines. If none of them appear across an interval
where you expected activity, that is itself informative — see the next section.

---

## Configuration reference

Settings live in the `fastlane` section of runtime configuration and are editable from the UI. Changes apply to
the next request; no restart.

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch. Off = Fast Lane has no effect on ordering. |
| `rules` | `[]` | Ordered list of address entries. **Position is the priority.** |
| `max_normal_wait_s` | `3600.0` | Fairness floor: longest an *unlisted* request may wait before it is served next regardless of priority. Accepts `1.0`–`3600.0`, and the shipped value is the top of that range — an hour — so the floor rescues very little until you lower it. Does not rescue a listed-but-lower-ranked address. |
| `cross_model_switches_per_min` | `2` | How many times a minute an *unregistered* (no-rule-matched) request may cause a cross-model swap — an unload, a load, and a re-prefill. Fast Lane traffic is exempt from this cap entirely, at any value including `0`. Accepts `0`–`3`. A number above `3` is rejected outright rather than reduced to fit, so a configuration carrying a larger value will fail to load. |
| `census_ttl_hours` | `168` | How long an unreferenced Discovered entry is retained. |

Each entry in `rules`:

| Field | Meaning |
|---|---|
| `address` | A single host address, stored exactly as observed. Ranges and prefixes are rejected. |
| `container_name` | A docker container name, resolved to its current addresses in the background (see "Level 1 — which client it is" above). Exactly one of `address`/`container_name` is required per rule — never both, never neither. |
| `label` | Your own note, e.g. `"workstation"`. Not used for matching. |
| `tag_ranks` | `main`, `curator`, `compression`, `sub_agent`, `unclassified` — each `1`–`5`, or unset. |

```yaml
fastlane:
  enabled: true
  max_normal_wait_s: 3600.0          # the shipped value, and the highest this may be set to
  cross_model_switches_per_min: 2    # 0–3; gates unregistered traffic only — Fast Lane is exempt at any value
  rules:
    - address: "192.0.2.10"          # added by clicking a Discovered entry
      label: "workstation"
      tag_ranks:
        compression: 1               # short, and finishing one unblocks a conversation
        main: 2
        curator: 3
        unclassified: 2              # unmarked traffic treated as interactive
        # sub_agent left unset — ranks after everything above
    - address: "192.0.2.20"
      label: "automation host"
      tag_ranks: {}                  # everything from here is equal
```

Addresses must be single hosts. Prefixes are rejected at validation: the list is meant to be populated by
clicking observed entries, and a prefix silently widens as an environment changes.

---

## Limits worth knowing

- **Embedding requests carry no source address** and are always normal priority. The Fast Lane screen states
  this rather than leaving it to be discovered.
- **`cross_model_switches_per_min` accepts at most three.** A configuration that sets a higher value is refused when
  it loads rather than quietly reduced to fit, so the service starts with Fast Lane disabled and shows you the
  rejection.
- **Duplicate address entries are a configuration error** and are reported rather than silently resolved.
- **A malformed saved configuration will not take the service down.** It is parsed in isolation. A rule whose
  address is not a single IP is read as a container name instead, and a rule with a blank identity is replaced by
  an inert placeholder; both keep their position, so the rules below them keep their rank. A list longer than the
  rule cap (10) keeps its highest-priority rules. A section that cannot be repaired is dropped: the manager logs the
  error, starts with Fast Lane disabled, and shows the rejection text on the Fast Lane screen.
- **The Discovered list is bounded.** If the cap is reached (256 addresses), new addresses are refused rather than
  quietly evicting existing ones, and the refusal is counted, reported in the `overflow` block of the census
  response and logged once at error level — a silently truncated list would be worse than a visibly full one.
- **When more than one model may be resident at a time**, each model's own resident inbox is still Fast-Lane-ordered, not first-in-first-out — the same priority pick used on the staging queue runs again whenever that resident wakes for its next request (see [What the inbox route does and does not keep](#seeing-whether-it-is-working)). It only serves in arrival order once nothing currently waiting for that model is a listed match.
- **A load that fails with an out-of-memory error does not cost a request its place or its Fast Lane standing.** It goes back into the queue at the same priority instead of erroring out, retried only when something real changes (see [MODEL_LOAD_TIMEOUT.md](MODEL_LOAD_TIMEOUT.md#an-out-of-memory-load-failure-does-not-end-the-request)). Re-admission re-evaluates placement fresh, the same as any other admission.

---

## Troubleshooting

### Why is nothing happening?

This is the question worth answering carefully, because **absence of activity is not, by itself, evidence of a
broken feature.** There are two ordinary, healthy reasons you will see no `grace_designated_*` or
`QUEUE_PRIORITY reason=fastlane` lines at all:

1. **Nothing is actually competing.** Fast Lane only reorders requests that are waiting at the same time. If your
   listed address's requests always arrive when the queue is otherwise empty, there is nothing to overtake and
   nothing to log. Check the **Waiting requests** table on the Queue tab: if it is consistently empty or
   single-item when your traffic lands, that is the explanation.
2. **There was room.** Fast Lane only ends a grace hold, or unloads an idle model, when a higher-priority request
   is waiting for room that is not free. If the priority request's model fits alongside what is already loaded,
   nothing has to give way, and no `grace_designated_victim_skip` or `grace_designated_unload_target_break` line
   is expected.

Silence at INFO cannot tell these apart from a broken lane. `FASTLANE_DECLINED` fires on every poll tick the
picker runs and finds nothing to do (`reason=staging_empty` / `reason=no_candidate`), which is why it logs at
**DEBUG, not INFO**: on every tick it would flood the log. Read its absence at the default level as "not logged at
this level," never as "the picker stopped running"; to check, turn DEBUG on temporarily (`--log-level` or
`TURBOHAUL_LOG_LEVEL`).

**I turned it on and nothing changed.**
Work through the two cases above first. If neither explains it, confirm that `enabled` is `true` in
`GET /api/config | jq .fastlane` and that your client's address is in the rules.

**My requests are not getting priority.**
Check the Discovered list for the address the manager actually sees for your client. If it does not appear
there, your requests are arriving under a different address than you expect — that is the first thing to
confirm, before adjusting ranks.

**A same-thread follow-up did not take the warm path.**
Check whether a request of the same address with a better class is waiting, in the queue or parked in the loaded
model's own queue. The follow-up yields to it, which is intended. Equal classes never cause this.

**Background work has stalled.**
Lower `max_normal_wait_s`: the fairness floor is what bounds how long *unlisted* traffic waits behind priority
traffic. Other queue settings, such as `queue.max_other_model_wait_s` (default `20.0`) and
`queue.max_consecutive_same_model` (default `3`), also limit how long a request waits behind work for another
model. If the stalled traffic is on a *listed* address ranked below another listed address, the floor does not
apply to it (see [Set the fairness floor](#3-set-the-fairness-floor)) — re-rank it instead.

**A priority request seems slower than before.**
This is not `cross_model_switches_per_min` — Fast Lane is exempt from that cap entirely, at any value
including `0` (see [Cross-model jumps](#cross-model-jumps)). A priority claimant that requires a cold jump pays
the swap's real cost (one model unloaded, another loaded, before the turn can begin) every time; nothing in this
setting reduces that. The refusals-by-reason counter counts *unregistered* traffic only, so it will never explain
a slow priority request.

**A model did not give up its grace window when I expected it to.**
Only a designated victim, or a model with a strictly better request parked in its own queue, loses its grace
window early; every other model keeps the full window. Check what your box is actually running with —
`GET /api/config | jq .queue` — before assuming the defaults above apply: a boot-time env var
(`TURBOHAUL_GRACE_S`, `TURBOHAUL_IDLE_HOT_S`) or an earlier runtime `PUT` can leave different numbers than a
fresh install.
