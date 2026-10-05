# Using the Turbohaul Manager web interface

The manager ships a web interface at `/ui` on the same port it serves the API from
(`http://<host>:11401/ui` by default). It is read-mostly: it shows you what the server is
doing right now, and it lets you change the handful of things that are safe to change while
the server is running.

- **See** what is loaded, what is running, how fast, and how much memory is left, and what is waiting and why.
- **Manage** model files and their manifests, and the Fast Lane rules that change the order requests are served in.
- **No login.** Anyone who can reach the port can use every control; see [Before you start](#before-you-start).

There are five tabs. Three of them have sub-pages.

| Tab | What it answers | Page |
|---|---|---|
| **Dashboard** | What is loaded, what is running, how fast, and how much memory is left | [Dashboard](01-dashboard.md) |
| **Queue** | What is waiting, and why this request rather than that one | [Queue](02-queue.md) · [Fast Lane](03-fast-lane.md) |
| **Blob** | What model files are on disk, and what manifests point at them | [Blob](04-blob.md) · [Models](05-models.md) |
| **Plugins** (work in progress) | Which external plugin containers are configured for agents to reach | [Plugins](06-plugins.md) |
| **Settings** | Version and general switches, configuration, a schema builder, and the audit log | [General and Config](07-settings-config.md) · [Schema](08-settings-schema.md) · [Logs](09-settings-logs.md) |

## All nine pages

- [Dashboard](01-dashboard.md) — four questions at a glance: is anything queued, what is loaded, how fast is it going, and how much GPU memory is left.
- [Queue](02-queue.md) — what is waiting, what is being served, and how long a finished model will be held before it is unloaded.
- [Fast Lane](03-fast-lane.md) — changes the order requests are served in.
- [Blob](04-blob.md) — every model file the server has on disk, addressed by content rather than by filename.
- [Models](05-models.md) — the model-first view, and the second sub-page of the Blob tab: one tile per model file, and the manifests that configure it one click in.
- [Plugins](06-plugins.md) (work in progress) — how an agent reaches tools the manager does not contain — transcription, OCR, text-to-speech, image generation.
- [General and Config](07-settings-config.md) — every setting the server is running with, split into what you can change now and what needs a restart.
- [Schema](08-settings-schema.md) — a builder for the `response_format` envelope, so you can constrain what a model returns.
- [Logs](09-settings-logs.md) — the audit trail: what the server did, in order, with filters.

## Backend docs these pages pair with

Each page above describes the screen. These are the main docs that cover the same subject from the server side.

| Page | Read alongside | What it covers |
|---|---|---|
| Dashboard | [ARCHITECTURE.md](../../ARCHITECTURE.md) §11 | The front-end: how it is built and served at `/ui` |
| | [ARCHITECTURE.md](../../ARCHITECTURE.md) §10 | Observability: the `/status` snapshot that feeds the dashboard |
| | [SAFETY_GATE_VRAM_MATH.md](../SAFETY_GATE_VRAM_MATH.md) | The arithmetic behind a model's VRAM estimate |
| Queue | [ARCHITECTURE.md](../../ARCHITECTURE.md) §3 | The request lifecycle: admit, queue, route, serve, grace, idle window, unload |
| | [FAST_LANE.md](../FAST_LANE.md) | Grace and idle-hot holds |
| | [MULTI_AGENT_SHARING.md](../MULTI_AGENT_SHARING.md) | Several agents sharing one GPU through the same queue |
| Fast Lane | [FAST_LANE.md](../FAST_LANE.md) | Request priority: which waiting request gets the GPU next |
| | [FAST_LANE_ARCHITECTURE.md](../design/FAST_LANE_ARCHITECTURE.md) | The specification of how Fast Lane behaves |
| Blob | [MODELS_AND_MANIFESTS.md](../MODELS_AND_MANIFESTS.md) | Getting a model file into the blob store, and the manifest that runs it |
| | [API_REFERENCE.md](../API_REFERENCE.md) | The routes for pulling, importing and deleting blobs |
| Models | [MODELS_AND_MANIFESTS.md](../MODELS_AND_MANIFESTS.md) | Adding and configuring a model, from the backend and from the dashboard |
| | [MODEL_CONFIG_REFERENCE.md](../MODEL_CONFIG_REFERENCE.md) | Per-model manifest fields and flags |
| Plugins | [MEDIA_HOOK.md](../MEDIA_HOOK.md) | Why the media hook exists and what its two lanes are for |
| | [PLUGINS_SETUP.md](../PLUGINS_SETUP.md) | The operator setup guide |
| | [PLUGIN_REFERENCE.md](../PLUGIN_REFERENCE.md) | The shipped wire contract |
| General and Config | [ARCHITECTURE.md](../../ARCHITECTURE.md) §9 | How the configuration is loaded and split into boot and runtime settings |
| | [API_REFERENCE.md](../API_REFERENCE.md) | The routes that read and update the configuration |
| | [MODEL_CONFIG_REFERENCE.md](../MODEL_CONFIG_REFERENCE.md) | Per-model manifest fields and flags |
| Schema | [API_REFERENCE.md](../API_REFERENCE.md) | The chat route that accepts `response_format` |
| | [ARCHITECTURE.md](../../ARCHITECTURE.md) §7 | How `response_format` is bounded |
| | [MODEL_CONFIG_REFERENCE.md](../MODEL_CONFIG_REFERENCE.md) | Per-model manifest fields and flags |
| Logs | [ARCHITECTURE.md](../../ARCHITECTURE.md) §10 | Observability |
| | [API_REFERENCE.md](../API_REFERENCE.md) | The logging route: paginated audit events with server-side redaction |
| | [MODEL_CONFIG_REFERENCE.md](../MODEL_CONFIG_REFERENCE.md) | Per-model manifest fields and flags |

## Before you start

**The identifiers in these screenshots are examples.** Model tags, agent names, session ids and
addresses are sample values; addresses such as `198.51.100.x` and `2001:db8::` come from the
ranges reserved for documentation. A screenshot shows the layout at the time it was taken, so
labels and small details can differ slightly from the current interface; where they do, the
text on each page describes the current interface.

**Nothing here is authenticated.** The interface has no login because the server has no
app-layer auth — the boundary is the address it binds to. Anyone who can reach the port can
use every control on every page. If that is not what you want, put a reverse proxy in front
of the whole thing rather than in front of one page.

**A screen showing a number is not the same as the server agreeing with it.** Several pages
say plainly what they cannot tell you. Those notes are worth reading; they are the difference
between debugging the system and debugging your assumption about it.
