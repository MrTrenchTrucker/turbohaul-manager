# Media Hook

**Status:** work in progress — feedback and contributions welcome. The manifest variant, the boot registry, the invoke route, the WebSocket exec forwarder
and the Plugins tab are all in the codebase today. This page is the conceptual companion — why the hook exists
and what its two lanes are for. For the shipped wire contract see [PLUGIN_REFERENCE.md](PLUGIN_REFERENCE.md),
and for the operator walkthrough see [PLUGINS_SETUP.md](PLUGINS_SETUP.md).

---

## 1. What it is

An agent asks Turbohaul for a media capability — transcription, OCR, text-to-speech, image/video generation —
the same way it already asks for a model. Turbohaul resolves the plugin's `resource_key` to the operator-declared
container, forwards the request to it, and returns the result inline to the caller.

**What it is not:** Under this design, the plugin path ships **no media tooling** — no ffmpeg, no binaries, no
vendored source. The operator either bakes the tool (ffmpeg, WhisperX, whatever) into their own image, or
points Turbohaul at a container already reachable on the network.

The engine images no longer apt-install ffmpeg: `Dockerfile` and `Dockerfile.cuda-multi`
install a small generic PATH shim under the names `ffmpeg` and `ffprobe`, which runs a real local binary if
the operator baked one in and otherwise routes the invocation to a plugin container through the manager.
`Dockerfile.engine-src` still installs a distribution ffmpeg, because it is the from-source build variant.
See [VISION_MODELS.md](VISION_MODELS.md) for what that means for video input.

## 2. The two lanes

| | CPU lane (intended) | GPU lane (intended) |
|---|---|---|
| Members | ffmpeg, WhisperX, Marker (OCR), Kokoro (TTS) | a second engine, for image/video generation |
| VRAM arbitration | none — the main agent stays resident on the cards | the existing queue + Fast Lane, unchanged |
| Admission | bypasses admission | normal admission, ranked by Fast Lane |
| Rationale | running off the CPU is fine to leave unarbitrated | this requires a GPU slot, so it has to queue properly |

**Today `lane` is declared but not load-bearing.** It is surfaced on `GET /api/plugins` and selects no admission
path: every plugin call is a direct HTTP call that does not traverse the queue, the Fast Lane ranking or the VRAM
fit gate (see [ARCHITECTURE.md](../ARCHITECTURE.md) §9.2). The intended CPU-lane guard is a cheap capacity check — a
semaphore with a high cap, not a real queue; `plugin_runtime.max_concurrent` is the declared knob for it and is
not enforced yet ([PLUGIN_REFERENCE.md](PLUGIN_REFERENCE.md) §3).

## 3. How an agent addresses it

An agent addresses a plugin by its `model_tag`: `POST /api/plugins/{model_tag}/invoke` with a body of
`{"path": ..., "payload": ...}` (see [PLUGIN_REFERENCE.md](PLUGIN_REFERENCE.md) §4.2). The invoke route reads no
identity: it does not parse `client_meta` or the `role`, `session_id`, `is_main`, `is_sub_agent`, `is_curator` and
`is_compression` fields, so a plugin call carries no per-session identity today. Those fields are read on the chat
routes, where one shared parser (`_derive_client_meta_identity` in `api/chat_completion.py`) is used at all 3
`client_meta` build sites (OpenAI non-stream / OpenAI stream / `/api/chat`). There is no media-specific identity
field.

## 4. Result delivery

Turbohaul returns the result to whoever called it. No storage layer, no path handoff. Which context the
answer lands in is the **harness's** decision, not Turbohaul's, and Turbohaul does not implement or observe that
routing. The intended client-side handling is:

- vision / OCR / audio / ffmpeg results are serviced for the agent's **main** model.
- image / video / any *generation* result is routed by the harness to an **AUX model on its own separate
  context.**

Because the AUX model's context is separate, a large generation result need not enter the main model's context
— this removes work from the context budget rather than adding to it.

## 5. Plugin configuration

A resource plugin is one arm of a **discriminated union** on `kind: model | plugin`, resolved through the
single existing validation chokepoint. The two arms are **sibling classes, not a superset**: `PluginManifest`
declares none of the model-only fields, so the model-side flag protections (the closed allowlist, the
denied-flags list, the suffix forward-defense) are not inherited — they are unreachable here, because the
field they guard does not exist. What the plugin arm does inherit from the shared base is the closed schema
(`extra="forbid"`), the `model_tag` identity and its traversal guard, and the bookkeeping fields. Safety on
this path comes from that closed schema plus the `resource_key` indirection, which keeps every address in
operator-written boot config.

The design follows four rules:

1. **One schema, one discriminator.** Plugin and model manifests share one validation
   chokepoint and one discriminator; a second, parallel schema would re-scatter validation
   logic across two chokepoints and add a variant the wrong way.
2. **Never a path.** A plugin must never carry `endpoint_url`, `container_path`, or `binary_path`. The
   manifest schema is closed (extra fields forbidden), and the project has already answered "how do we point
   at an external artifact" twice — once for a projector blob, once for a speculative-decoding draft model —
   and both times the answer was a **content-addressed SHA resolved server-side**, never a path, URL, or
   repo, with the raw forms already denied outright and a suffix guard on path-shaped keys. A plugin follows
   the same precedent: it references its container by an **operator-declared key into a boot-config
   registry** — never a URL or path inside the manifest itself.
3. **Adding a plugin is one registry entry** (plus its manifest), not a new `if kind == …` branch. Open for extension, closed
   for modification — skip it and every new plugin type re-opens dispatch logic to edit, the same scatter
   the first rule above already rules out.
4. **Provenance is kept.** Every call resolves through the one chokepoint (`resolve_endpoint` in
   `plugin_invoke.py`), and failure log lines name the `resource_key` involved; a successful response body is the
   plugin's own JSON, returned verbatim, with no added provenance field.

This guarantee holds at the manifest/config surface. The invocation routes are a separate surface and do not
inherit it automatically; `plugin_invoke.py` carries its own egress guard (see [ARCHITECTURE.md](../ARCHITECTURE.md) §8 and §9.2).

**Hostname trust model (read before registering a name instead of an IP).** `resolve_endpoint` performs NO
IP validation on hostname endpoints — none at registration, none at request time: the SSRF check applies to
IP-literal hosts only, and httpx resolves the name at connect time, which is after that check and
independent of it. A registered name is therefore trusted at registration time and may resolve to any
address at invocation time — the "DNS rebinding" gap is stronger than it sounds: hostnames are not
"checked-then-re-resolved", they are *never* checked. That is deliberate under the operator-boot threat
model (only an operator editing the boot registry can introduce a name). Two compensating controls:
(1) loopback IP literals are hard-blocked in both families, so a mis-registered *literal* cannot reach the
manager's own API even though a hostname could (`127.0.0.0/8` and `::1` are in the blocked set in
`plugin_invoke.py`); (2) the invoke client does not follow redirects (the httpx default; `follow_redirects` is
never set), so a plugin's 302 cannot move the request to a new host. If trust ever extends beyond the operator's boot
registry, pin resolution (resolve once, connect to the pinned IP) before relying on this note.

## 6. The Plugins tab

The Plugins tab is in the web interface today, on the frontend and the backend. It is its own tab — not the blob tab (content-addressed model storage, files by hash; a plugin is not a blob) and not the config tab. Like the rest of the plugin feature it is **work in progress — feedback and contributions welcome**: the navigation item reads **Plugins (WIP)**, and a few parts of the page are not wired up yet (they are called out below).

**What it shows.** The tab is served at `/ui/plugins`. It lists every plugin manifest that is not marked `hidden`, one row each, with one shared block of runtime settings under the list. The list comes from `GET /api/plugins`, which never carries a host or a port, so the page never shows one.

| Column | Meaning |
|---|---|
| **Model tag** | How agents address the plugin |
| **Lane** | The lane the manifest declares, CPU or GPU. It is descriptive: the manager only reports it |
| **Capabilities** | The list the manifest declares (an em-dash when it is empty) |
| **Status** | **Configured**, or **Not configured yet** when the plugin's key has no usable registry entry |
| **Enabled** | A checkbox, on by default, saved as soon as you change it |

**Configured** does not mean reachable. It means the plugin's key resolves in the boot registry and the address is allowed; the page does not probe the container. When no plugin is configured, the page says so and explains how to add one.

**What an operator can change.** The tab writes only through `PUT /api/config`, and only to `plugin_runtime`. Every save sends the whole `plugin_runtime` object, including the full `enabled` map.

| Setting (a field of `plugin_runtime`) | Default | Takes effect |
|---|---|---|
| **Enabled** (`enabled`) | on | Stored and shown, but not enforced yet: a plugin switched off here can still be called by its tag. Deleting its manifest is what makes it uncallable |
| **Max concurrent** (`max_concurrent`) | 4 | Stored and shown, but not enforced yet |
| **No-progress timeout (s)** (`no_progress_timeout_s`) | 600 | Live on the next call, no restart. It is how long a call may go without a byte sent or received; `0` means the call times out immediately, not that the timeout is off |

**Max concurrent** and **No-progress timeout (s)** are shared by every plugin and save when you leave the field. The page reads the bounds from `GET /api/config/schema`, and a value outside them is refused with a message.

**What it cannot change.** The boot-only configuration sections — server, storage, runtime, ui, and **`plugins` itself** — are never touched by the Plugins write path, and the tab says so on the page. The write path reaches `plugin_runtime` only; a `PUT /api/config` carrying a `plugins` key is refused with HTTP 403 before any validation or merge, so the registry can never be edited over the API. To change the registry, edit the boot configuration and restart the manager. The tab does not invoke a plugin, and it does not edit manifests: those are written with `PUT /api/manifests/{tag}`, the same route model manifests use, and take effect without a restart.

**Where to read more.** [PLUGINS_SETUP.md](PLUGINS_SETUP.md) walks an operator through declaring a plugin. [PLUGIN_REFERENCE.md](PLUGIN_REFERENCE.md) has the details: the tab in section 8, the runtime settings in section 3 and what needs a restart in section 7. The tab's page-by-page guide, with its screenshot, is [frontend/06-plugins.md](frontend/06-plugins.md).

## 7. Errors

**No duration timeout.** A duration cap is the wrong axis — it kills legitimate long jobs. A crash or OOM in
the plugin returns a **loud, recoverable** error to the harness instead of the expected result. No harness
changes are required to consume it.

**A hang is neither a crash nor slowness.** A crashed container refuses the connection — catchable. A
working one produces output. A hung one is alive, connection open, producing nothing, forever — crash
detection cannot see it. The answer is not a duration timeout, it is a **no-progress check**: not "has it run an
hour" but "has it produced anything in N minutes." It is implemented as `plugin_runtime.no_progress_timeout_s`
(default 600 seconds), applied as the read and write inactivity timeouts of the plugin call, and reported as
`no_progress` / HTTP 504.

## 8. The caller-side ceiling

A calling harness usually has its own request or stream-read timeout, for example 30 minutes.

That is the **caller's** limit, not Turbohaul's (§7). A three-hour media job that Turbohaul would happily keep
waiting on still fails when the caller's timeout expires, and not because of anything in this design. The
feature is designed for jobs that finish inside the caller's timeout; the ceiling is recorded here, not solved
here. Raising a caller-side timeout is client configuration, outside the scope of this feature.

## 9. Out of scope

This feature does not cover workflow engines such as ComfyUI, leasing or other non-model tenants, 3D tools such as Blender or FreeCAD, or client-side harness work.

**See also:** [VISION_MODELS.md](VISION_MODELS.md) — serving a vision model with its multimodal projector. A vision model reads images as input; it is not a plugin.
