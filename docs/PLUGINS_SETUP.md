# Turbohaul-Manager — Plugin Setup Guide (Operator)

**Audience:** the operator of a Turbohaul-Manager instance who wants a non-LLM tool — a transcriber, an OCR/document converter, anything HTTP-shaped — reachable through the manager's API.

**Goal:** one plugin answering `POST /api/plugins/<model_tag>/invoke` after two files and one restart, and enough detail to tell which of the two files is wrong when an error appears.

**Status:** work in progress — feedback and contributions welcome.

---

## TL;DR — two files, one restart

The plugin path ships **no** media tooling. A plugin is an HTTP service *you* run. Turbohaul holds two things about it, in two different places, with two different reload behaviours:

| # | File | What it holds | Reload |
|---|---|---|---|
| 1 | `turbohaul.yaml`, `plugins.registry.<resource_key>` | `host`, `port`, `health_path`, `auth_token_file` — **the only place an address may live** | **Boot-only. Restart the manager.** |
| 2 | `<manifests_path>/<model_tag>.yaml` | `kind: plugin`, `lane`, `resource_key`, `capabilities`, `provides_routes`, `provides_executables` | Read off disk on every request. **No restart.** |

That split is the sharp edge of this feature and the reason most first-time setups look half-broken:
**adding the registry entry needs a restart; adding the manifest does not.** Do the registry edit first, restart once, then iterate on the manifest freely.

Minimum viable pair. The first block is a **fragment to add to your existing `turbohaul.yaml`**, not a config file in its own right — `turbohaul.yaml` also requires `server`, `storage`, `runtime`, `ui`, `queue` and `pull`, none of which have a default. A file containing only the `plugins:` block below does not boot; it fails with six `Field required` errors.

```yaml
# ADD to turbohaul.yaml (default /etc/turbohaul/turbohaul.yaml) as a new top-level
# section, alongside the existing server:/storage:/runtime:/ui:/queue:/pull:
plugins:
  registry:
    whisperx:
      host: whisperx
      port: 8080
```

```yaml
# <manifests_path>/whisperx-fast.yaml
model_tag: whisperx-fast
kind: plugin
lane: cpu
resource_key: whisperx
```

Restart the manager, then:

```bash
curl -s -X POST http://<manager-host>:11401/api/plugins/whisperx-fast/invoke \
  -H 'Content-Type: application/json' \
  -d '{"path":"/transcribe","payload":{"audio_path":"/data/interview.wav"}}'
```

The rest of this doc is the two full worked examples and the confirmation steps. Field-by-field schemas, the full error table and the complete list of known limits live in [PLUGIN_REFERENCE.md](PLUGIN_REFERENCE.md); this page does not restate them.

---

## Getting your edits into `turbohaul.yaml`

Every step below says "edit `turbohaul.yaml`". On a containerized deployment you cannot do that in place, and nothing else in this guide works until you have solved it once:

- The images built from the Dockerfiles in this repository **bake the config into the image**: each base Dockerfile here does `COPY docker/turbohaul.default.yaml /etc/turbohaul/turbohaul.yaml` at build time, and any image you build `FROM` one of them inherits that copy. Edits to it inside a running container are lost on the next recreate.
- The `docker-compose.yml` in this repository carries an optional, commented-out line; uncomment it to mount your own copy over that path **read-only**: `./turbohaul.yaml:/etc/turbohaul/turbohaul.yaml:ro`. Edits attempted from inside the container are refused.

**No environment variable can set the plugin registry.** The `TURBOHAUL_*` override map covers a fixed list of scalar knobs under `server`, `queue`, `persist`, `kv` and `http`; nothing in it reaches `plugins` or `plugin_runtime`. `TURBOHAUL_CONFIG_PATH` selects *which file* is loaded, not what is in it. **A config file on disk that the manager can read is the only route to a registry entry.**

Do this once, before PATH A or PATH B:

```bash
# 1. Start from the config the image is really running, not from a doc snippet.
docker cp <manager-container>:/etc/turbohaul/turbohaul.yaml ./turbohaul.yaml

# 2. Edit ./turbohaul.yaml on the host — add the plugins: section (step A1 below).

# 3. Bind-mount it over the baked copy. The compose file in this repository has this line commented out; uncomment it:
#      - ./turbohaul.yaml:/etc/turbohaul/turbohaul.yaml:ro
#    Read-only is correct: the manager only ever reads this file.

# 4. Restart, so the boot config is re-read.
docker compose up -d          # or: docker restart <manager-container>
```

Then confirm the manager loaded the file you meant — it logs the path at startup:

```bash
docker logs <manager-container> 2>&1 | grep 'loading config'
# loading config: /etc/turbohaul/turbohaul.yaml
```

If that path does not exist the manager does **not** fall back to defaults: it logs `config not found: <path>` and exits with status 2. The usual cause is a bind-mount naming a host file that is not there — Docker creates a *directory* at that path instead, and the manager will not start.

The shipped `docker/turbohaul.default.yaml` carries **no** `plugins:` or `plugin_runtime:` block — both sections default to empty, so the file that ships is a working config without them. Step 1 therefore hands you the real running config to add to, not a template to uncomment: append the `plugins:` block from step A1 as a new top-level section alongside the `server:` / `storage:` / `queue:` sections already in the file.

---

## What the invoke route does and does not do

`POST /api/plugins/{model_tag}/invoke` with body `{"path": ..., "payload": ...}`:

1. Reads `<manifests_path>/<model_tag>.yaml` from disk (traversal-guarded before any filesystem access).
2. Rejects it with 400 if it is not a plugin manifest.
3. Looks the manifest's `resource_key` up in the **boot** registry and applies the address guard.
4. Builds `http://<host>:<port><path>` and issues **one** `POST` with `payload` as the JSON body.
5. Returns the plugin's parsed JSON body **inline**, verbatim, to the caller.

Two consequences decide whether a given container is usable at all, and you should check both before you write any config:

- **The call is always JSON-in, JSON-object-out.** `invoke_plugin` posts `json=payload` and the route is annotated `-> dict`. A container whose endpoint expects `multipart/form-data` is **not reachable** through this route, and a plugin that answers a top-level JSON *array* produces a bare 500. Both cases want a thin JSON shim in front of the tool — PATH B below is exactly that pattern.
- **`path` is a route, never a URL,** and the hop is plain `http://` with no TLS, and no auth unless you set `auth_token_file` on the registry entry. A caller never supplies a host, port, or scheme; that is the entire point of the registry indirection.

The exact request/response shapes, the timeout model, and the redirect behaviour are in [PLUGIN_REFERENCE.md §4](PLUGIN_REFERENCE.md#4--http-endpoints).

**There is a second calling convention, and it is not JSON.** `WS /api/plugins/{model_tag}/exec` forwards a WebSocket stream to the plugin container's `/ws/exec` so the plugin can run an **executable** on the manager's behalf. That is how the engine images that ship without local media binaries get video decode: they install a small shim on `PATH` under the binary's own name, and the shim connects to this route when no real binary is present locally. If you are running one of those images and want video input to work, the thing to configure is a plugin whose container serves `/ws/exec`, under the tag the shim is pointed at — see [PLUGIN_REFERENCE.md §4.3](PLUGIN_REFERENCE.md#43-ws-apipluginsmodel_tagexec). The registry half of the setup is identical to PATH A; the manifest gains a `provides_executables` list, which advertises the binary to agents and, when non-empty, limits the binary names the exec route will forward (it does not gate the JSON invoke route), and the container must serve `/ws/exec` rather than a JSON route.

---

## Getting the file to the plugin — you arrange this, Turbohaul does not

This is the single most common thing to get wrong, so it is stated once, here, in full.

**`invoke_plugin` makes exactly one call: `client.post(url, json=payload, headers=headers)`.** A JSON body and nothing else (`headers` holds only the optional bearer token from `auth_token_file`). There is no `multipart/form-data`, no file streaming, no upload endpoint, no chunking, no shared staging directory the manager creates, and no field anywhere in the manifest or the registry that names a file. Turbohaul never opens, reads, copies, or even sees your media file. It forwards a JSON object to an address and hands you back whatever JSON comes out. **Getting the bytes in front of the plugin container is entirely yours to arrange**, and there are exactly three ways to do it:

1. **Base64 the file into the payload** (`{"audio_b64": "<...>"}`). Needs no infrastructure at all, which is its only advantage. The encoded file is buffered whole in the manager's memory and again in the plugin's, base64 adds about a third to its size, and **there is no request-body size ceiling on this route** — the body-size middleware covers only the chat endpoints. Fine for a voice memo; a bad idea for a 40-minute recording.
2. **Put a URL in the payload that the plugin fetches itself** (`{"audio_url": "https://…"}`). The manager does **not** fetch it — it is an opaque string inside a payload it forwards verbatim. Whether that URL resolves, whether it needs credentials, and what SSRF exposure it creates are all questions about *your plugin container*, not about Turbohaul. Turbohaul's address guard protects the hop to the plugin; it has no visibility past it.
3. **Put a path in the payload that points at a volume you mounted into both containers** (`{"audio_path": "/data/interview.wav"}`). No copy, no size limit, no encoding overhead — and it only works if you mounted the same volume into the manager (or wherever the file is produced) *and* into the plugin, at a path both agree on. Turbohaul does not create that mount, verify it, or warn you when it is missing; a wrong path is a plugin-side error that comes back as `plugin_error` / 502.

The key name in every one of those examples — `audio_b64`, `audio_url`, `audio_path` — is **your plugin's, not Turbohaul's**. Nothing in this project defines, validates, or even reads the inside of `payload`.

**These docs use option 3 throughout**, with a volume mounted at `/data` in both containers, because it is the only one that stays workable at real media sizes. Every example you see below follows it. Pick whichever suits your deployment; just pick one and be consistent, because the three are not interchangeable at the plugin end.

**This is a known limitation, not an oversight to work around.** It is listed as such in [PLUGIN_REFERENCE.md §9](PLUGIN_REFERENCE.md#9--known-gaps-and-accepted-risks). One practical corollary: several widely-used speech-to-text images expose their transcription route as a `multipart/form-data` upload — those cannot be called through this route at all, whatever you put in `payload`. Front them with a shim.

---

## The two identifiers: `model_tag` vs `resource_key`

`model_tag` names the manifest (*what to call*); `resource_key` names the registry entry (*where it lives*). Mixing them up is the most common validation failure, because **they use different regexes and only `model_tag` allows dots** — exact patterns and the rejection message in [PLUGIN_REFERENCE.md §1.4](PLUGIN_REFERENCE.md#14-the-two-identifiers-use-different-regexes).

The one shape worth knowing while you set up: `resource_key` is deliberately **many-to-one**. A `whisperx-fast` and a `whisperx-accurate` manifest can both point at `resource_key: whisperx`, differing only in the `path`/`payload` your caller sends.

---

## PATH A — point at an external container already on your network

This is the normal case: the tool runs as its own container/host, the manager reaches it by service name over the container network.

**Example: WhisperX for transcription, on a service named `whisperx`, port 8080.**

### A1. Registry entry (boot config)

Edit `turbohaul.yaml` — the manager loads `/etc/turbohaul/turbohaul.yaml` by default, overridable with `--config` or `TURBOHAUL_CONFIG_PATH`. Add a top-level `plugins:` section (it sits alongside `server:`, `storage:`, `queue:` …):

```yaml
plugins:
  registry:
    whisperx:
      host: whisperx        # bare hostname or IP literal — NEVER a URL
      port: 8080
      health_path: /health  # optional, defaults to /health
```

Those four fields are the whole schema and the models are `extra="forbid"`, so a typo fails the boot loudly rather than at first invoke. `auth_token_file` is optional and holds a **path** to a file containing a bearer token, never the token itself — if you set it, the file must exist and be non-empty at boot or the manager will not start. Field types, bounds, and every boot-time validation message: [PLUGIN_REFERENCE.md §2](PLUGIN_REFERENCE.md#2--boot-registry-schema).

**One rule to apply while you are naming things: make sure the host's last dot-separated label is not all digits.** A label starting with a letter always satisfies it. `whisperx`, `marker`, `media-host-1` register; `svc.2` and `999` are refused, with a message about shorthand IP forms that will look baffling if you do not know the rule. Why that rule exists, and the full accept/refuse list: [PLUGIN_REFERENCE.md §6](PLUGIN_REFERENCE.md#6--host-value-rules).

### A2. Plugin manifest

Manifests live in `storage.manifests_path` (`/var/lib/turbohaul/manifests` in the shipped default), one file per tag, named `<model_tag>.yaml`.

Write it through the API — the same route model manifests use:

```bash
curl -s -X PUT http://<manager-host>:11401/api/manifests/whisperx-fast \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "plugin",
    "display_name": "WhisperX (fast)",
    "description": "CPU-lane speech-to-text",
    "lane": "cpu",
    "resource_key": "whisperx",
    "capabilities": ["transcribe", "align"]
  }'
```

Response: `{"status":"ok","model_tag":"whisperx-fast","revision":1,"restart_required":false}` plus an `ETag: "1"` header.

**Leave `hidden` off for now.** It defaults to `false`, which is what makes the confirmation step below able to show you anything. There is a real decision to make about it, and it comes *after* you have proof the plugin works — step A4.

The file that lands on disk is exactly this (round-tripped through the real write path — field order is the writer's, not yours):

```yaml
model_tag: whisperx-fast
display_name: WhisperX (fast)
description: CPU-lane speech-to-text
revision: 1
hidden: false
kind: plugin
lane: cpu
resource_key: whisperx
capabilities:
- transcribe
- align
provides_routes: []
provides_executables: []
```

Both `provides_*` keys are written even though you omitted them: they are declared fields with empty-list defaults, and the writer serialises the whole model. Fill them in to make the plugin self-describing to agents — see [PLUGIN_REFERENCE.md §1.1](PLUGIN_REFERENCE.md#11-field-table).

You may also just drop that file into `manifests_path` yourself; it is read off disk per request either way.

The schema is closed at eleven keys — four required, seven optional, anything else rejected. Field-by-field, with types, defaults and bounds: [PLUGIN_REFERENCE.md §1.1](PLUGIN_REFERENCE.md#11-field-table).

#### Sharp edge — `kind: plugin` must be written literally

There is no default. A file that omits `kind` is validated as a **model** manifest and comes back complaining about `gguf_blob_sha256`, with `lane` and `resource_key` reported as *"Extra inputs are not permitted"* — nothing in the error mentions plugins ([§1.2](PLUGIN_REFERENCE.md#12-kind-plugin-must-be-written-literally)).

Note **where** you see it. `PUT /api/manifests/<tag>` gives you that 400 with the field list. A file already sitting on disk without `kind:` does not: it fails to parse on read, and the invoke route answers a **500** (`manifest for '<tag>' is present but unreadable`) while the listing skips it, logging a warning. Those two symptoms together — *absent from the listing* **and** *500 on invoke* — mean an unparseable manifest, not a missing one; a missing file gives a clean 404. Write manifests through the API and you get the readable error instead.

### A3. Restart, then confirm

```bash
# registry changed -> restart required
docker restart <manager-container>       # or however you run it
```

Confirm from the API, in this order:

```bash
# 1. Did the registry entry load? (addresses are omitted by design — keys only)
curl -s http://<manager-host>:11401/api/config | python3 -c 'import json,sys; print(json.load(sys.stdin)["plugins"])'
# {'registry_keys': ['whisperx']}

# 2. Does the manifest see it?
curl -s http://<manager-host>:11401/api/plugins
# {"plugins":[{"model_tag":"whisperx-fast","lane":"cpu",
#              "capabilities":["transcribe","align"],"configured":true,
#              "provides_routes":[],"provides_executables":[],
#              "invoke":{"method":"POST",
#                        "url":"/api/plugins/whisperx-fast/invoke",
#                        "body":{"path":"<one of provides_routes>","payload":{}}}}],
#  "total":1}

# 3. Does it actually work?
curl -s -X POST http://<manager-host>:11401/api/plugins/whisperx-fast/invoke \
  -H 'Content-Type: application/json' \
  -d '{"path":"/transcribe","payload":{"audio_path":"/data/interview.wav"}}'
```

Step 2 lists the plugin because you left `hidden` at its default of `false`. `configured: true` there means **the `resource_key` is present in the registry and its address passed the guard** — it does **not** mean the container is up, because nothing probes it. Step 3 is the only real proof.

The listing is self-describing on purpose: an agent that can see the entry can also see what to call and how, without being told out of band. `provides_routes` and `provides_executables` are echoed from the manifest and are empty until you declare them. (B4's output has the same shape.)

`path` and `payload` are whatever *your* container's API expects. Turbohaul does not interpret either; it only checks that `path` starts with `/`, contains no `..`, and is printable. `/data/interview.wav` is a path on a volume mounted into both containers — the convention set out in [Getting the file to the plugin](#getting-the-file-to-the-plugin--you-arrange-this-turbohaul-does-not). Check your container's route actually takes JSON before you copy this; if it wants a `multipart/form-data` upload, front it with a shim (PATH B's `marker_shim.py` is exactly that pattern, and works just as well in a separate container).

### A4. Decide about `hidden`

There is no trade to make here. `hidden` affects exactly one surface: the agent-facing listing.

- `hidden: false` (the default) — the plugin appears in `GET /api/plugins` and in the Plugins tab, and is invokable by exact `model_tag`.
- `hidden: true` — the plugin is absent from `GET /api/plugins` and the Plugins tab, and is **still** fully invokable by exact `model_tag`. Your agents then have to be told their tags out of band.

Neither setting touches `GET /api/tags` or `GET /v1/models`: those are model listings and exclude every plugin manifest by `kind`, whatever `hidden` says. Leave it at `false` unless you specifically want the tag kept out of discovery.

```bash
# read the current revision, then flip it
curl -si http://<manager-host>:11401/api/manifests/whisperx-fast | grep -i '^etag:'
curl -s -X PUT http://<manager-host>:11401/api/manifests/whisperx-fast \
  -H 'Content-Type: application/json' -H 'If-Match: "1"' \
  -d '{"kind":"plugin","lane":"cpu","resource_key":"whisperx",
       "capabilities":["transcribe","align"],"hidden":true}'
```

The measured behaviour of both settings on all three surfaces is in [PLUGIN_REFERENCE.md §9](PLUGIN_REFERENCE.md#9--known-gaps-and-accepted-risks).

### A5. The Plugins tab

`http://<manager-host>:11401/ui/plugins` (nav item **Plugins (WIP)**), for non-hidden plugin manifests only. It is a read/write view over `GET /api/plugins` plus the config surface — it never invokes anything, and it never shows host or port. What each column and badge means: [PLUGIN_REFERENCE.md §8](PLUGIN_REFERENCE.md#8--the-plugins-tab-ui).

If the tab is empty and you expected a row, the usual answer is `hidden: true` — see A4.

---

## PATH B — bake the tool into your own image and point at it locally

Use this when you would rather ship one image than run one more container. The tool runs inside the manager's own container, listening on loopback, and the registry points at it there.

**Example: Marker for PDF/OCR → markdown, co-resident, listening on port 8001.**

### B1. Add the tool to the image

```dockerfile
FROM <your-turbohaul-manager-image>

USER root
RUN pip install --no-cache-dir marker-pdf uvicorn fastapi
COPY marker_shim.py /opt/plugins/marker_shim.py
COPY entrypoint-with-marker.sh /usr/local/bin/entrypoint-with-marker.sh
RUN chmod +x /usr/local/bin/entrypoint-with-marker.sh
ENTRYPOINT ["/usr/local/bin/entrypoint-with-marker.sh"]
```

`marker_shim.py` is a small JSON-in/JSON-out wrapper. The manager posts JSON and requires a JSON **object** back, so the shim's whole job is to bridge those two facts to whatever API the tool actually has. A complete skeleton — only the marked line is yours to write:

```python
# marker_shim.py — POST /convert {"file_path": "..."} -> {"markdown": "...", "pages": 12}
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ConvertRequest(BaseModel):
    file_path: str

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/convert")
def convert(req: ConvertRequest) -> dict:
    markdown, page_count = run_your_converter(req.file_path)   # <- your call into the tool
    return {"markdown": markdown, "pages": page_count}         # MUST be an object, never a list
```

Two rules this skeleton encodes, both enforced by the manager and neither negotiable: the response is a JSON **object** (a bare list 500s the invoke route), and any non-2xx you return comes back to the caller as `plugin_error` / 502 with your response body attached — so keep failure bodies short and free of anything you would not hand to the caller.

`entrypoint-with-marker.sh` starts the tool, then hands off to the manager's normal entrypoint:

```bash
#!/bin/sh
set -e
uvicorn marker_shim:app --host 127.0.0.1 --port 8001 --app-dir /opt/plugins &
exec turbohaul-manager "$@"
```

`turbohaul-manager` is the console-script entrypoint the Dockerfiles in this repository use (`ENTRYPOINT ["turbohaul-manager"]`). If you are extending an image you did not build, confirm it first rather than assuming:

```bash
docker inspect --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}' <your-turbohaul-manager-image>
```

Whatever that prints is what your last line must `exec`.

**Pick a port nothing else owns.** The manager itself is on `server.port` (default `11401`), and engine children are allocated from the 100-port window starting at `runtime.default_port_base` (default `11500`, so `11500-11599`). `8001` is clear of both.

### B2. Registry entry — use `localhost`, not `127.0.0.1`

```yaml
plugins:
  registry:
    marker:
      host: localhost     # NOT 127.0.0.1 — see below
      port: 8001
      health_path: /health
```

This is the one place PATH B differs materially from PATH A, and it will bite you if nobody says it out loud:

**`host: 127.0.0.1` registers cleanly and then fails every single invoke with `blocked_target` / HTTP 400**, because the address guard refuses loopback *IP literals*. **`host: localhost` works**, because that guard inspects IP literals only and deliberately never resolves a hostname. The full reasoning and the exact refused set: [PLUGIN_REFERENCE.md §6.3–§6.4](PLUGIN_REFERENCE.md#63-resolution-the-address-policy).

Be precise about the consequence while you are here: because `localhost` is never resolved or checked, a registry entry pointing at the wrong loopback port will happily reach **the manager's own API**. Double-check the port you write.

### B3. Plugin manifest

```bash
curl -s -X PUT http://<manager-host>:11401/api/manifests/marker-ocr \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "plugin",
    "display_name": "Marker (OCR)",
    "description": "GPU-lane PDF/OCR to markdown",
    "lane": "gpu",
    "resource_key": "marker",
    "capabilities": ["convert", "ocr"]
  }'
```

Same as PATH A: leave `hidden` off until you have confirmed the plugin works, then make the A4 decision — it applies identically here.

On disk:

```yaml
model_tag: marker-ocr
display_name: Marker (OCR)
description: GPU-lane PDF/OCR to markdown
revision: 1
hidden: false
kind: plugin
lane: gpu
resource_key: marker
capabilities:
- convert
- ocr
provides_routes: []
provides_executables: []
```

`lane: gpu` here is **descriptive only**. Nothing in the manager keys admission, queueing, or scheduling off `lane` today ([§9](PLUGIN_REFERENCE.md#9--known-gaps-and-accepted-risks)); why the lane concept exists at all is in [MEDIA_HOOK.md](MEDIA_HOOK.md). It does not reserve a GPU, and it does not stop your OCR container and a resident model from contending for the same card. Plan capacity yourself.

### B4. Rebuild, restart, confirm

```bash
docker build -t <your-image-with-marker> .       # build the B1 Dockerfile into a new image
# set `image:` in your compose file to that tag (the compose file in this repository also has a `build:` section; `docker compose up -d` without `--build` does not rebuild from it), then:
docker compose up -d                              # recreate the container on the new image

curl -s http://<manager-host>:11401/api/config | python3 -c 'import json,sys; print(json.load(sys.stdin)["plugins"])'
# {'registry_keys': ['marker', 'whisperx']}

curl -s http://<manager-host>:11401/api/plugins
# {"plugins":[{"model_tag":"marker-ocr","lane":"gpu",
#              "capabilities":["convert","ocr"],"configured":true, ...}, ...],"total":2}

curl -s -X POST http://<manager-host>:11401/api/plugins/marker-ocr/invoke \
  -H 'Content-Type: application/json' \
  -d '{"path":"/convert","payload":{"file_path":"/data/report.pdf"}}'
```

---

## Where each file lives

| File | Default location | Written by | Effective when |
|---|---|---|---|
| `turbohaul.yaml` (`plugins.registry`) | `/etc/turbohaul/turbohaul.yaml`, or `--config` / `TURBOHAUL_CONFIG_PATH` | you, by hand | **manager restart** |
| plugin manifest | `<storage.manifests_path>/<model_tag>.yaml`, default `/var/lib/turbohaul/manifests/` | `PUT /api/manifests/<tag>`, or by hand | **immediately** |
| `runtime_config.yaml` (`plugin_runtime`) | next to `storage.state_db_path` | `PUT /api/config` — do not hand-edit | **immediately**, and survives restart |

---

## What needs a restart, and what does not

Only one thing on this page needs a restart: **the `plugins.registry` entry**. Manifests are read from disk on every request, and `plugin_runtime` is swapped in place by `PUT /api/config`. The full matrix, with the reason for each row, is in [PLUGIN_REFERENCE.md §7](PLUGIN_REFERENCE.md#7--what-needs-a-restart).

Two consequences worth internalising while you are setting up:

1. **The two halves of one fix have different latencies.** A manifest naming a `resource_key` with no registry entry *saves successfully* — there is deliberately no write-time gate, so first-time setup is not a chicken-and-egg. The only symptom is `configured: false` and a 400 `unknown_resource` at invoke. Fixing it means editing boot config, which needs a restart, while the manifest half you were just iterating on did not.
2. **You cannot write the registry through the API.** `plugins` is a boot-only section, so `PUT /api/config` carrying a `plugins` key is refused with 403 before any validation or merge, and a hand-tampered `runtime_config.yaml` carrying a `plugins:` block is structurally ignored at boot. Edit `turbohaul.yaml` and restart.

---

## Runtime knobs

One knob on this path does anything today — `plugin_runtime.no_progress_timeout_s`, the per-operation inactivity timeout — and it is live, no restart:

```bash
curl -s -X PUT http://<manager-host>:11401/api/config \
  -H 'Content-Type: application/json' \
  -d '{"plugin_runtime":{"no_progress_timeout_s":900}}'
```

`enabled` and `max_concurrent` are stored, shown in the UI, and read by nothing. Defaults, bounds, the shallow-merge trap on the `enabled` map, and why `0.0` means *immediate* timeout rather than *disabled*: [PLUGIN_REFERENCE.md §3](PLUGIN_REFERENCE.md#3--runtime-knobs).

---

## Troubleshooting

Every failure reason this path can emit, the HTTP status it maps to, what the caller sees versus what is logged, and a symptom-keyed table are in [PLUGIN_REFERENCE.md §5](PLUGIN_REFERENCE.md#5--complete-failure-table) and [§10](PLUGIN_REFERENCE.md#10--troubleshooting). They are not repeated here — one copy, kept correct.

The one thing a setup guide has to add: several of those rows say *"see the server log"*, and they mean the manager's own stdout. For a container deployment:

```bash
docker logs --tail 200 <manager-container>
```

The redacted half of every address failure — the actual `host:port` — is only ever there, never in the HTTP response. If you are debugging a `configured: false` badge or a `blocked_target`, that log is the only surface that names the address.

---

## Known limits

This path ships with real, measured sharp edges: knobs that are stored and displayed but enforced nowhere,
an address check that covers IP literals and not hostnames, and one failure that forwards the plugin's own
body to the caller unredacted. They are enumerated once, with the measurement behind each, in
[PLUGIN_REFERENCE.md](PLUGIN_REFERENCE.md) §9 — read
that list before you put this path in front of anything that matters. The address ranges that are and are
not refused, and the host-name rules the validator enforces, are in the same document, §6.

---

## Where to go next

- `docs/AI_AGENT_SETUP.md` — connecting agents to the chat surfaces (note: "provider plugin" there means something unrelated to this document).
- `docs/PLUGIN_REFERENCE.md` — the field-by-field reference and **the single owner of every schema, the failure table, the host rules and the known-limits list**. Use this page to get one plugin working; use that one to look anything up.
- `docs/MEDIA_HOOK.md` — why the hook exists, how it sits against the rest of the manager, and what the two lanes are for. Conceptual; no config in it.
- `docs/MODEL_CONFIG_REFERENCE.md` — the *model* manifest schema, for contrast. Plugin and model manifests are sibling schemas: apart from the `kind` discriminator that tells them apart, they share only `model_tag`, `display_name`, `description`, `revision`, `hidden`.
- `ARCHITECTURE.md` — queue, slot lifecycle, and where the plugin path deliberately sits outside all of it.
- [frontend/06-plugins.md](frontend/06-plugins.md) — the Plugins tab in the web interface.
- [VISION_MODELS.md](VISION_MODELS.md) — serving a vision model with its multimodal projector. A vision model reads images as input; it is not a plugin.

---

*This doc describes the plugin path as it is built today. If you hit an error string or a config key that is not explained here, that is a doc bug — open an issue.*
