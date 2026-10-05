# Turbohaul-Manager — Resource Plugin Reference

**Audience:** Operators wiring an external media/tooling container (transcription, OCR/PDF conversion, TTS, image generation) into a Turbohaul-Manager instance, and agent developers calling one.

**Goal:** Every manifest field, every registry field, every runtime knob, all three HTTP endpoints, and every failure reason — with the exact type, bound, default, and status code the manager enforces today.

**Status:** work in progress — feedback and contributions welcome. It applies to plugins (resource plugins and the Media Hook).

Companion docs: [PLUGINS_SETUP.md](PLUGINS_SETUP.md) (the step-by-step operator walkthrough — start there if you have no plugin working yet), [MEDIA_HOOK.md](MEDIA_HOOK.md) (why the hook exists, the two lanes, and how it fits), [MODEL_CONFIG_REFERENCE.md](MODEL_CONFIG_REFERENCE.md) (the *model* manifest, a sibling schema — not a superset of this one), [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md) (wiring an agent to Turbohaul), [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## Preface — how to read this reference

**Turbohaul ships no media tooling.** A resource plugin is an *operator-declared external container* that Turbohaul forwards a JSON POST to. Turbohaul owns the addressing, the guard rails, and the error contract; the container owns the work.

**Ground truth is six files.** `src/turbohaul/manifest.py` (`PluginManifest`), `src/turbohaul/config.py` (`PluginEndpoint`, `PluginsConfig`, `PluginRuntimeConfig`), `src/turbohaul/plugin_invoke.py` (`resolve_endpoint`, `invoke_plugin`), the JSON route layer in `src/turbohaul/api/plugins.py`, the WebSocket exec forwarder in `src/turbohaul/api/exec_ws.py`, and the background health probe in `src/turbohaul/plugin_health.py` (`PluginHealthMonitor`). Where this doc and those files disagree, the files win and this doc is the bug.

**A plugin is two halves that live in two different places, with two different edit latencies.**

| Half | Where it lives | Who writes it | Takes effect |
|---|---|---|---|
| **Manifest** — identity, lane, capabilities, and the `resource_key` it points at | `<manifests_path>/<model_tag>.yaml` | `PUT /api/manifests/{tag}` or on disk | **Immediately** — `read_manifest` hits disk on every request |
| **Registry entry** — the `host`, `port`, `health_path` that `resource_key` resolves to | `plugins.registry.<resource_key>` in `turbohaul.yaml` | Boot config only | **After a manager restart** — `BootConfig` is frozen at boot |

**A caller never supplies a host, a port, or a URL.** An agent names a `model_tag` on the route and a `path` on the target container. Everything else is resolved server-side from the boot registry. Every connection Turbohaul makes to a plugin container goes to an address taken from that registry: the JSON invoke path (`plugin_invoke.py`), the exec forwarder (`api/exec_ws.py`, which resolves through `resolve_endpoint`), and the background health probe (`plugin_health.py`).

---

## TL;DR — the minimal working setup

**1. Boot config (`turbohaul.yaml`) — declare where the container is:**

```yaml
plugins:
  registry:
    whisperx:
      host: whisperx        # service name on your container network
      port: 8080
      health_path: /health  # optional, defaults to /health
```

**2. Restart the manager.** The registry is boot-only.

**3. Save a plugin manifest naming that key:**

```bash
TH=http://<turbohaul-host>:11401

curl -s -X PUT "$TH/api/manifests/whisperx-fast" \
  -H 'Content-Type: application/json' \
  -d '{
        "kind": "plugin",
        "display_name": "WhisperX (fast)",
        "description": "CPU-lane transcription",
        "lane": "cpu",
        "resource_key": "whisperx",
        "capabilities": ["transcribe", "align"]
      }'
# {"status":"ok","model_tag":"whisperx-fast","revision":1,"restart_required":false}
```

**4. Confirm it is configured, then call it:**

```bash
curl -s "$TH/api/plugins"
# {"plugins":[{"model_tag":"whisperx-fast","lane":"cpu","capabilities":["transcribe","align"],"configured":true,"provides_routes":[],"provides_executables":[],"invoke":{"method":"POST","url":"/api/plugins/whisperx-fast/invoke","body":{"path":"<one of provides_routes>","payload":{}}}}],"total":1}

curl -s -X POST "$TH/api/plugins/whisperx-fast/invoke" \
  -H 'Content-Type: application/json' \
  -d '{"path": "/transcribe", "payload": {"audio_path": "/data/interview.wav"}}'
# the plugin's own JSON body, returned inline and verbatim
```

`/data/interview.wav` is a path **you** made visible to the plugin container by mounting a volume into it. Turbohaul does not put the file there and does not move it — the invoke body is JSON and only JSON. Read [PLUGINS_SETUP.md — "Getting the file to the plugin"](PLUGINS_SETUP.md#getting-the-file-to-the-plugin--you-arrange-this-turbohaul-does-not) before you design a payload; it is the single authoritative treatment of that limitation, and every example in these docs follows the convention it sets.

**One thing to know before you finish:** `hidden` is a free choice on a plugin manifest. It affects only `GET /api/plugins` and the Plugins tab — `hidden: true` removes the plugin from the agent-facing listing while leaving it fully invokable by its exact `model_tag` (§1.1). It has no effect on `GET /api/tags` or `GET /v1/models`, which exclude every plugin manifest by `kind` regardless of this field ([§9 — Known gaps](#9--known-gaps-and-accepted-risks)). Leave it at `false` unless you specifically want the tag kept out of discovery.

---

## 1 — Plugin manifest schema

One YAML file per plugin at `<manifests_path>/<model_tag>.yaml`. Parsed by `manifest.parse_manifest()` into `PluginManifest`, which inherits `model_config = ConfigDict(extra="forbid")` from `HardenedManifestBase`. **There are exactly eleven legal keys — the five from `HardenedManifestBase` plus the six this variant adds. Any other key is a validation error, not an ignored one.**

### 1.1 Field table

| Field | Type | Required | Default | Meaning | Constraints |
|---|---|---|---|---|---|
| `kind` | `Literal["plugin"]` | **yes** | *none* | Discriminator. Selects `PluginManifest` over `ModelManifest`. | Must be the literal string `plugin`, written explicitly in the YAML. There is **no default** — see §1.2. |
| `model_tag` | `str` | **yes** | *none* | Primary key, filename stem, and the tag an agent names on the invoke route. | `TAG_RE` = `^[a-z0-9][a-z0-9._-]{0,63}$`. Lowercase ASCII, 1–64 chars, dots allowed but not as the first character, no `/`. Forced to match the URL segment by `PUT /api/manifests/{tag}`. Re-validated at every path resolution. |
| `lane` | `Literal["cpu","gpu"]` | **yes** | *none* | Declares which execution lane the plugin's work belongs to. | Exactly `cpu` or `gpu`. **Metadata only today** — see §9. |
| `resource_key` | `str` | **yes** | *none* | The key looked up in the boot registry to find the container's host and port. | `RESOURCE_KEY_RE` = `^[a-z0-9][a-z0-9_-]{0,63}$`. **Narrower than `TAG_RE` — no dots.** It is a registry lookup key, not a filename stem. |
| `capabilities` | `list[str]` | no | `[]` | Free-text list of operations the plugin offers, e.g. `["transcribe","align"]`. | No per-item validation. **Never compared against the `path` a caller invokes** — see §9. |
| `provides_routes` | `list[str]` | no | `[]` | Routes this plugin serves, e.g. `["/transcribe","/align"]`. Published by `GET /api/plugins` so an agent can discover what to call without being told out of band. | Each entry must start with `/`, must not start with `//` (protocol-relative resolves off-host), must not contain `..`, and must match `^/[A-Za-z0-9._~/-]{0,127}$` — no scheme (`:`), no percent-encoding, no userinfo (`@`), no query or fragment, no backslash, no whitespace. No duplicates. Max 64 entries. **Not an allowlist:** the invoke route does not check a caller's `path` against it, and must not be assumed to — see §9. |
| `provides_executables` | `list[str]` | no | `[]` | Bare binary names this plugin can run, e.g. `["ffmpeg","ffprobe"]`. Published by `GET /api/plugins`, and checked as a first-hop declaration on the exec route (§4.3). | Bare names only: no `/`, no `\`, no `..`, and must match `^[A-Za-z0-9_][A-Za-z0-9._-]{0,63}$` — it may not start with `-` (it would pose as an argv flag) or `.`. No duplicates. Max 64 entries. An empty list imposes no first-hop restriction; the target container's own allow-list remains the security property either way. |
| `display_name` | `str` | no | `""` | Human label for the UI. | Cosmetic. |
| `description` | `str` | no | `""` | Free-text notes. | Cosmetic. |
| `revision` | `int` | no | `1` | The ETag value for optimistic-concurrency writes. Server-incremented on every atomic update. | `ge=1`. You do not hand-manage it: `GET` returns `ETag: "<revision>"`, you echo it in `If-Match`, the server bumps it. |
| `hidden` | `bool` | no | `false` | Listings-only visibility. `true` omits the manifest from `GET /api/plugins`. | **Does not affect invocability** — a hidden plugin is still fully callable by its exact `model_tag`. Plugin manifests are absent from `GET /api/tags` and `GET /v1/models` at all times, filtered by `kind`, whatever this field says. |

### 1.2 `kind: plugin` must be written literally

`parse_manifest()` does `payload.setdefault("kind", "model")` before handing the dict to the discriminated-union adapter, so **every existing on-disk manifest with no `kind` key still loads as a model with zero migration** — and a plugin file that forgets `kind` is validated as a `ModelManifest`. The errors you get back then talk about the *model* schema, not about being a plugin:

```
4 validation errors for tagged-union[...,PluginManifest]
model.gguf_blob_sha256
  Field required
model.lane
  Extra inputs are not permitted
model.resource_key
  Extra inputs are not permitted
model.capabilities
  Extra inputs are not permitted
```

If you see `gguf_blob_sha256 Field required` on a file you believe is a plugin, you omitted `kind`.

### 1.3 `PluginManifest` is a sibling class, not a superset

`PluginManifest` declares **none** of `gguf_blob_sha256`, `llama_server_flags`, `context_size`, `expected_vram_bytes`, `gguf_size_bytes`, `arch`, `kv_bytes_per_token`. Those attributes do not exist on the object — `plugin_manifest.llama_server_flags` raises `AttributeError`. The two variants share only `HardenedManifestBase`: `model_tag`, `display_name`, `description`, `revision`, `hidden`, and the `extra="forbid"` closure.

The practical consequence: a plugin manifest **cannot carry a `llama-server` flag, a path, or a URL, because there is no field to put one in.** It also means the model-side protections (the closed flag allowlist, `DENIED_FLAGS`, the suffix forward-defense) are not inherited — they are unreachable, because the field they guard does not exist here. Safety on this path comes from a different mechanism: a closed eleven-key schema, plus the `resource_key` indirection that keeps every address in operator-written boot config.

### 1.4 The two identifiers use different regexes

| Value | Regex | Dots | Example legal | Example rejected |
|---|---|---|---|---|
| `model_tag` | `^[a-z0-9][a-z0-9._-]{0,63}$` | **allowed** | `whisper.x`, `whisperx-fast`, `marker-pdf` | `Whisperx`, `whisper/x`, `.hidden` |
| `resource_key` | `^[a-z0-9][a-z0-9_-]{0,63}$` | **rejected** | `whisperx`, `whisper_x`, `marker` | `whisper.x`, `WhisperX`, `whisper/x` |

```
resource_key 'whisper.x' fails regex ^[a-z0-9][a-z0-9_-]{0,63}$ - ASCII lowercase only,
no path separators, no traversal, max 64 chars, no dots (it is a registry lookup key,
not a filename)
```

### 1.5 Exact on-disk form

Round-tripped through `write_manifest_atomic` → `read_manifest`, `<manifests_path>/whisperx-fast.yaml` is exactly:

```yaml
model_tag: whisperx-fast
display_name: WhisperX (fast)
description: CPU-lane transcription
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

The two `provides_*` keys are written even when you omit them from the request body: they are declared fields with empty-list defaults, and the writer serialises the whole model.

### 1.6 Writing a manifest over HTTP

A plugin manifest goes through the **same** route as a model one: `PUT /api/manifests/{tag}`. The route forces `payload["model_tag"] = tag` (so a mismatched body field can never confuse the filename), calls `parse_manifest`, then `write_manifest_atomic`.

| Call | If-Match | Result |
|---|---|---|
| First write of a tag | omitted | `200 {"status":"ok","model_tag":...,"revision":1,"restart_required":false}`, header `ETag: "1"` |
| Update | omitted | `412` `ConcurrencyError` |
| Update | `If-Match: "1"` (current) | `200`, `revision` bumps to `2`, header `ETag: "2"` |
| Update | `If-Match: "1"` (stale) | `412` |

```bash
# read current revision
curl -si "$TH/api/manifests/whisperx-fast" | grep -i '^etag:'
# ETag: "1"

# update using it
curl -s -X PUT "$TH/api/manifests/whisperx-fast" \
  -H 'Content-Type: application/json' -H 'If-Match: "1"' \
  -d '{"kind":"plugin","lane":"cpu","resource_key":"whisperx","hidden":true,"capabilities":["transcribe"]}'
# {"status":"ok","model_tag":"whisperx-fast","revision":2,"restart_required":false}
```

`restart_required: false` is accurate **for the manifest half only**. If you also had to add a registry entry, that half needs a restart (§7).

### 1.7 Removing a plugin

`DELETE /api/manifests/{tag}` — the same route model manifests use. **No `If-Match` is required or consulted**, so a delete is not concurrency-checked the way a `PUT` is.

```bash
curl -s -X DELETE "$TH/api/manifests/whisperx-fast"
# {"status":"deleted","model_tag":"whisperx-fast"}

# already gone:
# 404 {"detail":"manifest not found: whisperx-fast"}
```

Deleting the manifest is the direct way to make a plugin uncallable — `plugin_runtime.enabled` does not do it (§9); removing its registry entry from boot config (and restarting) does too. The registry entry the manifest named is untouched and survives in boot config; an orphan registry entry is harmless, although the health probe (§2.2) keeps checking it until you remove it.

---

## 2 — Boot registry schema

`BootConfig.plugins` is a `PluginsConfig`, whose only field is `registry: dict[str, PluginEndpoint]`. **This is the only place in the system a plugin host or port may live.** Both models are `frozen=True, extra="forbid"`.

### 2.1 `PluginsConfig`

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `registry` | `dict[str, PluginEndpoint]` | no | `{}` | Every **key** must match `^[a-z0-9][a-z0-9_-]{0,63}$` — the same register as a manifest's `resource_key`. A key like `Whisper.X` fails at boot. |

### 2.2 `PluginEndpoint`

| Field | Type | Required | Default | Meaning | Constraints |
|---|---|---|---|---|---|
| `host` | `str` | **yes** | *none* | Where the container lives: a bare hostname or a bare IP literal. | Never a URL. Full accept/refuse rules in §6. |
| `port` | `int` | **yes** | *none* | TCP port on that host. | `ge=1, le=65535`. |
| `health_path` | `str` | no | `/health` | Route requested (`GET`) by the manager's background health probe. | Must start with `/`; must not contain `..`. Used only by that probe (`plugin_health.py`): one pass at boot, then every 60 s per entry (backing off to 600 s while an entry keeps failing), logging status changes to the server log. The result is kept in-process only: it is not served over HTTP and does not feed `configured` or the invoke route. See §9. |
| `auth_token_file` | `Path \| None` | no | `null` | Path to a file holding a bearer credential for this plugin — **never the credential itself**, because the config file is bind-mounted and copied around while a token is a secret. When set, the invoke path reads the file fresh on every call and sends `Authorization: Bearer <token>`. Absent means no `Authorization` header, byte-identical to a registry entry that predates the field. | Validated at boot: the file must exist, be readable, and be non-empty, or the manager fails to start with a message naming the **path** only, never the content. The value is never cached on the frozen endpoint object. A read failure at call time is `plugin_error` / 502 with the path in the server log only. |

### 2.3 Worked boot config

```yaml
# turbohaul.yaml — top-level key, sibling of server/storage/runtime/ui
plugins:
  registry:
    whisperx:
      host: whisperx
      port: 8080
      health_path: /health
    marker:
      host: marker
      port: 8081
      # health_path omitted -> "/health"
```

### 2.4 Boot-registry validation errors

| Mistake | Error at boot |
|---|---|
| `registry: {Whisper.X: {...}}` | `plugins.registry` — `plugins registry key 'Whisper.X' must match '^[a-z0-9][a-z0-9_-]{0,63}$' (same register as manifest.py's TAG_RE)` — **ignore that parenthetical**, it is a source-side misnomer: the pattern quoted in the message is the real one, and it is *narrower* than `TAG_RE` (§1.4) |
| `endpoint_url: http://...` inside an entry | `plugins.registry.<key>.endpoint_url` — `Extra inputs are not permitted` |
| `port` omitted | `plugins.registry.<key>.port` — `Field required` |
| `port: 0` | `plugins.registry.<key>.port` — `Input should be greater than or equal to 1` |
| `health_path: health` | `plugins.registry.<key>.health_path` — `health_path must start with '/': 'health'` |

### 2.5 The registry cannot be written through the API

`plugins` is a member of `api/config_put.py`'s `BOOT_SECTIONS = {"server","storage","runtime","ui","plugins"}`. A `PUT /api/config` whose body contains a `plugins` key is rejected **before** any validation or merge:

```bash
curl -s -X PUT "$TH/api/config" -H 'Content-Type: application/json' -d '{"plugins":{"registry":{}}}'
# HTTP 403
# {"detail":"sections ['plugins'] are BOOT-ONLY; restart manager to change
#   (...)"}
```

The 403 is not the only guard. The boot merge in `__main__.py` only merges a persisted `runtime_config.yaml` section when `section in RUNTIME_SECTIONS`, so a hand-edited override file carrying a `plugins:` block is **structurally ignored** at boot rather than merged. Consequently `GET /api/config`'s `_provenance` can never label a boot-section field `persisted`.

### 2.6 The registry is never disclosed over HTTP

`GET /api/config` serves the section as key names only:

```json
"plugins": { "registry_keys": ["marker", "whisperx"] }
```

Every endpoint field — `host`, `port`, `health_path`, `auth_token_file` — is withheld, on the same rationale as the storage/runtime path-basename redaction: the registry holds internal container topology. `GET /api/plugins` withholds it too (§4.1), and so does every error body (§5.3). `plugin_runtime` is the one plugin-related section served in full.

---

## 3 — Runtime knobs

`RuntimeConfig.plugin_runtime` is a `PluginRuntimeConfig`. It is runtime-mutable through `PUT /api/config` and **carries no host and no port.**

| Field | Type | Default | Valid range | Meaning | Live? |
|---|---|---|---|---|---|
| `enabled` | `dict[str, bool]` | `{}` | any keys | Intended per-plugin on/off switch. **Keyed by `model_tag`, not `resource_key`.** | **Stored, never read — inert.** See §9. |
| `max_concurrent` | `int` | `4` | `1`–`64` | Intended ceiling on simultaneous plugin invocations. | **Stored, never read — inert.** See §9. |
| `no_progress_timeout_s` | `float` | `600.0` | `0.0`–`86400.0` | Per-operation inactivity timeout on the plugin call: seconds since the last byte was sent or received. | **Yes** — read live off `mgr.runtime.plugin_runtime` on every invoke. No restart. |

`enabled` is keyed by `model_tag` deliberately: `resource_key` is many-to-one, so two manifests (say a fast profile and an accurate profile) may share one registry entry, and keying by `resource_key` would make disabling one silently disable the other.

### 3.1 Worked runtime config

```yaml
# turbohaul.yaml — top-level key, sibling of queue/pull/kv/http/fastlane
plugin_runtime:
  enabled:
    whisperx-fast: true
    marker-pdf: false
  max_concurrent: 4
  no_progress_timeout_s: 600.0
```

### 3.2 Changing it at runtime

```bash
curl -s -X PUT "$TH/api/config" -H 'Content-Type: application/json' \
  -d '{"plugin_runtime": {"no_progress_timeout_s": 1800}}'
```

**Merge semantics, precisely.** `put_config` does `merged[section] = {**current[section], **sec_payload}` — a **shallow, one-level merge against the live runtime**. So:

| You PUT | What happens |
|---|---|
| `{"plugin_runtime":{"max_concurrent":8}}` | `max_concurrent` changes; `enabled` and `no_progress_timeout_s` **keep their live values**. A partial PUT of top-level fields is safe. |
| `{"plugin_runtime":{"enabled":{"whisperx-fast":false}}}` | The **entire `enabled` map is replaced.** Every tag you omitted is dropped. Always send the full map. |
| `{"plugin_runtime":{"no_progress_timeout_s":100000}}` | `400` — `Input should be less than or equal to 86400`. |
| `{"plugins":{...}}` | `403` — boot-only (§2.5). |
| `{"nosuchsection":{...}}` | `400` — `unknown section(s)`. |

The one-level-deep replacement of `enabled` is the trap; the shallow merge of the three top-level fields is not.

`GET /api/config/schema` publishes the bounds, so a client can validate before it PUTs:

```json
"plugin_runtime": {
  "enabled":               {"type":"object",  "default":{},    "minimum":null, "maximum":null},
  "max_concurrent":        {"type":"integer", "default":4,     "minimum":1,    "maximum":64},
  "no_progress_timeout_s": {"type":"number",  "default":600.0, "minimum":0.0,  "maximum":86400.0}
}
```

### 3.3 `no_progress_timeout_s: 0.0` is not "disabled"

The bound is `ge=0.0`, so `0.0` is accepted. It is passed straight into `httpx.Timeout(read=0.0, write=0.0)`, which is an **immediate** timeout — every invoke then fails with `no_progress` / HTTP 504. "No timeout" is `None`, which the HTTP path can never send (the route always passes the configured float). Do not set this to zero.

---

## 4 — HTTP endpoints

Three routes, all under `/api/plugins`: a discovery listing, a JSON invoke, and a WebSocket exec forwarder. **None requires authentication** — see §9.

### 4.1 `GET /api/plugins`

Agent-facing discovery. Lists only manifests where `isinstance(m, PluginManifest)`; skips `hidden`; skips (with a server-side warning) any manifest that fails to load.

**Request**

```bash
curl -s "$TH/api/plugins"
```

**Response** — HTTP 200

```json
{
  "plugins": [
    {
      "model_tag": "whisperx-fast",
      "lane": "cpu",
      "capabilities": ["transcribe", "align"],
      "configured": true,
      "provides_routes": ["/transcribe", "/align"],
      "provides_executables": [],
      "invoke": {
        "method": "POST",
        "url": "/api/plugins/whisperx-fast/invoke",
        "body": {"path": "<one of provides_routes>", "payload": {}}
      }
    }
  ],
  "total": 1
}
```

| Field | Type | Meaning |
|---|---|---|
| `model_tag` | `str` | The tag to pass on the invoke route. |
| `lane` | `"cpu"\|"gpu"` | Echoed from the manifest. |
| `capabilities` | `list[str]` | Echoed from the manifest. |
| `configured` | `bool` | Computed at read time by actually calling `resolve_endpoint(m.resource_key, registry)` inside a `try/except PluginInvokeError` — the same guard `invoke` uses, not a hand-rolled `key in registry` check. **Registry state only; nothing is probed.** |
| `provides_routes` | `list[str]` | Echoed from the manifest — the routes an agent can pass as `path`. Empty when the manifest declares none. |
| `provides_executables` | `list[str]` | Echoed from the manifest — the binaries the exec route will accept (§4.3). Empty when the manifest declares none. |
| `invoke` | `object` | How to call the routes above: `method`, a **relative** `url`, and the body shape. Relative by construction — this endpoint states *what* to call, never *where*; the registry stays the only place a host or port lives. |
| `total` | `int` | Length of `plugins`. |

**`host` and `port` are never present.** There is no field for them, by design.

**`configured: false` collapses two different failures.** The `except` clause does not discriminate, so *"you never registered this `resource_key`"* (`unknown_resource`) and *"you registered it at a refused address"* (`blocked_target`) look identical here. The distinguishing detail is in the server log only. The UI renders `false` as amber **"Not configured yet"**, never as an error, because a manifest that exists before its registry entry is a legitimate expected state (§7).

**`configured: true` does not mean the plugin is up — and the name says so.** It means the `resource_key` is in the registry and its address is not in the refused set. Nothing is probed.

`resolve_endpoint` is a registry lookup plus an address-policy check that **never opens a socket**, so `configured` is honest about what it measures but says nothing about reachability: it can be `true` for every capability while every request to them fails. Reachability is established only by invoking (`unreachable` → 502), never by this listing — and deliberately so. Probing six endpoints on a polled, agent-facing listing would trade a silent wrong answer for a slow one, and caching the result would be the same claim with a timestamp on it.

### 4.2 `POST /api/plugins/{model_tag}/invoke`

The only way an agent reaches a plugin. Synchronous: the plugin's JSON body is returned **inline and verbatim**. There is no job store, no file handoff, no polling, and no envelope.

**Request body**

| Field | Type | Required | Default | Meaning | Constraints |
|---|---|---|---|---|---|
| `path` | `str` | **yes** | *none* | A route on the already-resolved container, e.g. `/transcribe`. **Not a URL.** | Must start with `/`; must not contain `..`; must be `str.isprintable()`. |
| `payload` | `dict` | no | `{}` | The JSON body forwarded to the plugin unchanged. | Must be a JSON object. No size ceiling — see §9. |

**Worked call**

```bash
curl -s -X POST "$TH/api/plugins/whisperx-fast/invoke" \
  -H 'Content-Type: application/json' \
  -d '{
        "path": "/transcribe",
        "payload": {"audio_path": "/data/interview.wav", "language": "en"}
      }'
```

**Success** — HTTP 200, the plugin's own body:

```json
{"text": "hello world", "segments": []}
```

**Getting the file itself to the plugin is not part of this contract.** `invoke_plugin` issues exactly one `client.post(url, json=payload, headers=headers)` — a JSON body and, at most, the optional bearer header described below, nothing else. There is no multipart, no file streaming, no upload route, and no staging directory the manager arranges. The single authoritative treatment of what that leaves you, and the convention every example in these docs follows, is [PLUGINS_SETUP.md — "Getting the file to the plugin"](PLUGINS_SETUP.md#getting-the-file-to-the-plugin--you-arrange-this-turbohaul-does-not). It is a known limitation, listed in §9.

**What the manager does, in order:**

1. `read_manifest(root, model_tag)` — which runs `validate_tag` before any filesystem access, so a traversal or malformed tag is rejected first.
2. `isinstance(m, PluginManifest)` — a model manifest is a 400.
3. `resolve_endpoint(m.resource_key, registry)` — registry lookup, then the IP-literal guard.
4. `_validate_path(body.path)`.
5. If the registry entry sets `auth_token_file`, read it fresh (§2.2). Bracket an IPv6 literal host, build `url = f"http://{host}:{port}{path}"` — **plain HTTP, not HTTPS**, by design for an internal operator-declared container.
6. One `await client.post(url, json=payload, headers=headers)` inside `httpx.AsyncClient(timeout=...)`.
7. Non-2xx → `plugin_error`. Non-JSON → `plugin_error`. Otherwise return `response.json()`.

Step 3 precedes step 4: **a bad `path` against an unknown `resource_key` reports `unknown_resource`, not `blocked_target`.**

**Timeouts.** `httpx.Timeout(connect=10.0, read=no_progress_timeout_s, write=no_progress_timeout_s, pool=10.0)`. httpx's read/write timeouts are per-operation *inactivity* timeouts (time since the last byte), which is genuinely a no-progress mechanism rather than a repurposed duration cap. Connect and pool sit on a fixed 10.0 s floor independent of the configured value, so a very small or absent no-progress value is never read as "wait forever to open the socket". **There is no total-duration timeout anywhere on this path** — see §9.

**Redirects are not followed.** `invoke_plugin` never passes `follow_redirects`, and the httpx version in use defaults it to `False`. Measured: a plugin answering `307 Location: http://<some-other-host>/…` causes exactly one outbound request (to the registered endpoint), the redirect target is never fetched, and the caller gets `plugin_error` / HTTP 502 with `returned HTTP 307`.

**What your plugin container actually receives.** Measured on the wire: a `POST` to `http://<registered-host>:<registered-port><path>`, body = the `payload` object serialised as-is (`{"a": 1}` in, `{"a": 1}` out — no envelope, no added keys), `Content-Type: application/json`, and nothing else but httpx's own defaults (`host`, `accept`, `accept-encoding`, `connection`, `user-agent`, `content-length`). **No header from the original caller is forwarded** — not `Authorization`, not a trace ID, not a cookie. The manager can nonetheless present a credential of its own: if the registry entry sets `auth_token_file` (§2.2), the invoke path reads that file fresh on every call and adds `Authorization: Bearer <token>` to the outbound request. That is the one header the manager originates, it comes from operator-written boot config rather than from the caller, and the file holds the secret while the config holds only its path.

**Hidden plugins are still invokable** by exact `model_tag`, matching the `/v1/models` and `/api/tags` contract for hidden models.

### 4.3 `WS /api/plugins/{model_tag}/exec`

A bidirectional WebSocket forwarder for running an **executable** inside a plugin container, rather than calling an HTTP route on it. It exists so an engine image can ship without local media binaries: the image installs a small generic shim on `PATH` under the executable's own name, and the shim connects here when no real binary is present locally.

**Resolution is the same chokepoint as invoke.** `read_manifest` → `isinstance(m, PluginManifest)` → `resource_key` → registry → `resolve_endpoint`, all of it **before** the WebSocket is accepted, so a bad tag, a non-plugin manifest, a missing, unparseable or schema-invalid manifest or a refused address closes the upgrade (the ASGI server answers HTTP 403) instead of accepting and then failing.

**The target path is fixed by the contract**, not supplied by the caller: the forwarder dials `ws://<registered-host>:<registered-port>/ws/exec`. There is no path injection on this route.

**The forwarder is near-opaque.** Frames pass through unmodified in both directions. It only: extracts `run_id` from the first text frame for its own log lines; applies the first-hop declaration check below; injects a dedicated target-lost error frame (`code: 101`) if the target connection dies mid-stream; and closes both ends when either side goes away.

**First-hop declaration check.** If the plugin manifest declares a non-empty `provides_executables`, a start frame naming a binary outside that list is refused at the forwarder with `{"type":"error","code":127,…}` and a `1008` close. A manifest with an empty list imposes no restriction here. This is defence in depth in front of the target container's own allow-list, which remains the security property.

**Optional outbound credential.** If `EXEC_PLUGIN_AUTH_TOKEN` is set in the manager's environment, the forwarder sends `Authorization: Bearer <token>` to the target. The route itself still authenticates no inbound caller.

**Failure shapes the client sees:** upgrade refused (missing, unparseable or schema-invalid manifest / not a plugin / unknown or blocked `resource_key`); `{"type":"error","code":1}` then a `1011` close when the target is unreachable — the registry address goes to the server log only, never to the client; `{"type":"error","code":101}` then `1011` when the target is lost mid-stream; a clean `1000` close after the target's own terminal frame.

---

## 5 — Complete failure table

### 5.1 Plugin-invocation failures (`PluginInvokeError`)

These all return the body `{"detail": {"reason": <reason>, "detail": <text>}}`. There are exactly five reasons across thirteen raise sites.

| Reason | HTTP | Raised when | Body the caller sees (`detail.detail`) | Logged server-side instead |
|---|---|---|---|---|
| `unknown_resource` | **400** | `registry.get(resource_key)` is `None` — the manifest names a key with no registry entry. | `no registry entry for resource_key 'marker'` | *(nothing extra — no address exists to withhold)* |
| `blocked_target` | **400** | The registry host is an IP literal inside the refused set (§6.3). | `resource_key 'x' is registered at an address that is never a valid plugin destination (multicast / reserved / link-local-IMDS / NAT64 / IPv4-mapped range; see server log for the address)` — **it does not say which range matched** | `log.error` — `plugin endpoint blocked at resolve: resource_key=%s endpoint=%s:%s` |
| `blocked_target` | **400** | `path` does not start with `/`. | `path 'transcribe' must start with '/'` | *(nothing — the path came from the caller, so echoing it leaks nothing)* |
| `blocked_target` | **400** | `path` contains `..`. | `path '/a/../b' must not contain '..'` | *(nothing)* |
| `blocked_target` | **400** | `path` contains a non-printable character (tab, newline, NUL). | `path '/a\tb' contains a non-printable character` | *(nothing)* |
| `blocked_target` | **400** | `httpx.InvalidURL` while building the request — the safety net for a host that passed registration but is not URL-safe. | `could not build a request URL for plugin 'whisperx' (see server log for details)` | `log.error` — `plugin invoke invalid URL: resource_key=%s endpoint=%s:%s error=%s` |
| `unreachable` | **502** | `httpx.ConnectError` / `ConnectTimeout` / `PoolTimeout` — including DNS failure for a hostname endpoint. | `could not reach plugin 'whisperx' (see server log for the address)` | `log.error` — `plugin invoke unreachable: resource_key=%s endpoint=%s:%s error=%s` |
| `unreachable` | **502** | Any other `httpx.TransportError` (read/write/close error) — the connection dropped mid-flight. | `transport error talking to plugin 'whisperx' (see server log)` | `log.error` — `plugin invoke transport error: resource_key=%s endpoint=%s:%s error=%s` |
| `no_progress` | **504** | `httpx.ReadTimeout` / `WriteTimeout` — no byte moved for `no_progress_timeout_s`. | `plugin 'whisperx' produced no activity within 600.0s` | `log.error` — `plugin invoke no progress: resource_key=%s endpoint=%s:%s error=%s` |
| `plugin_error` | **502** | The plugin answered outside `200 <= code < 300`. | `'whisperx' returned HTTP 500: 'Traceback: boom'` — **the plugin's own response body, first 500 chars, verbatim** | **nothing** — this raise site writes no server log |
| `plugin_error` | **502** | `response.json()` raised `json.JSONDecodeError`. | `'whisperx' returned invalid JSON: Expecting value: line 1 column 1 (char 0)` | **nothing** |
| `plugin_error` | **502** | The endpoint declares an `auth_token_file` and it could not be read (`OSError`) at call time. | `could not read the auth token file for plugin 'whisperx' (see server log for the path)` | `log.error` — `plugin invoke auth token file unreadable: resource_key=%s auth_token_file=%s error=%s` |
| `plugin_error` | **502** | The endpoint's `auth_token_file` exists but is empty (or whitespace only). | `the auth token file for plugin 'whisperx' is empty (see server log for the path)` | `log.error` — `plugin invoke auth token file empty: resource_key=%s auth_token_file=%s` |
| *(anything unrecognised)* | **502** | A future reason `api/plugins.py` has not been taught. `_REASON_TO_STATUS.get(reason, 502)` — a deliberate fallback, never a `KeyError`/500. | as raised | as raised |

The two credential-file rows are call-time failures only. A file that is missing or empty **at boot** fails `BootConfig` construction instead and the manager does not start (§2.2).

Real bodies, measured:

```json
400 {"detail":{"reason":"unknown_resource","detail":"no registry entry for resource_key 'marker'"}}
400 {"detail":{"reason":"blocked_target","detail":"path 'transcribe' must start with '/'"}}
502 {"detail":{"reason":"unreachable","detail":"could not reach plugin 'whisperx' (see server log for the address)"}}
502 {"detail":{"reason":"plugin_error","detail":"'whisperx' returned HTTP 500: 'Traceback: boom'"}}
504 {"detail":{"reason":"no_progress","detail":"plugin 'whisperx' produced no activity within 600.0s"}}
```

### 5.2 Pre-invocation failures — a different body shape

These happen before `invoke_plugin` is ever called and **do not carry a `reason` key**. A client that blindly reads `detail.reason` will get `undefined` here.

| Condition | HTTP | Body |
|---|---|---|
| No manifest with that tag | **404** | `{"detail":"plugin not found: nope"}` |
| Tag fails `TAG_RE` | **400** | `{"detail":"tag 'BAD..TAG' fails regex ^[a-z0-9][a-z0-9._-]{0,63}$ - ASCII lowercase only, no path separators, no traversal, max 64 chars"}` |
| Tag names a **model** manifest | **400** | `{"detail":"'real-model' is not a plugin manifest (kind='model')"}` |
| Manifest file's YAML root is not a mapping (a list, a scalar) | **400** | `{"detail":"manifest root must be mapping, got list"}` |
| Manifest on disk fails **schema** validation, its YAML will not parse, or the file cannot be read or decoded | **500** | `{"detail":"manifest for 'tag' is present but unreadable"}` — a fixed message with no `reason` key; the details go to the server log only — see §9 |
| Request body omits `path` | **422** | FastAPI's standard `{"detail":[{"type":"missing","loc":["body","path"],…}]}` |

The split between the two "manifest on disk" rows is not cosmetic. The invoke route handles `FileNotFoundError` (404) and `ManifestValidationError` (400) on their own. `ManifestValidationError` is what `validate_tag`, the symlink/path-escape guard, and the root-must-be-a-mapping check raise *outside* pydantic — those become clean 400s. But the moment pydantic itself is running, it **wraps** a field validator's `ManifestValidationError` into a `pydantic.ValidationError`, which is not a `ManifestValidationError`; neither is `yaml.YAMLError`. Those two, plus `UnicodeDecodeError` and `OSError`, share one handler that logs the full exception server-side and returns a **500** with the fixed message shown above — never the exception text, because pydantic echoes manifest field values and YAML errors quote file content. Checked: an unknown key, a dotted `resource_key`, a missing `lane`, and a file with broken YAML syntax each raise one of those exception types.

### 5.3 What is withheld, and what is not

The contract is that **a caller never learns the registry's internal `host:port`**, even when it is mis-registered — that address is exactly what a rebind-pivoting attacker wants. Seven raise sites log the full diagnostic and return a redacted string. Five of them withhold the registry's internal `host:port` — the `blocked_target`-at-resolve, `unreachable`, `no_progress`, transport-error, and `httpx.InvalidURL` paths. Two more withhold the `auth_token_file` **path** on the two credential-file failures (§5.1). The credential's **content** is stricter still: it is never written to a log line and never placed in a `PluginInvokeError.detail`, because both of those are forwarded to the caller or to the server log. Four of the five address-redacting sites are regression-pinned by tests asserting the exception detail contains neither the host nor the port while the captured log does; the `InvalidURL` safety net has a test for its `reason` but none for its redaction.

**The redaction covers the registry address only.** The `plugin_error` path forwards the plugin's own response body — the first 500 characters, verbatim — into the client's 502. If your plugin container puts a stack trace, an internal path, or its own upstream URL in a 500 body, the caller sees it. And because that raise site does not log, the operator sees nothing. Treat your plugin containers' error bodies as caller-visible.

---

## 6 — Host-value rules

Two independent layers apply to `plugins.registry.<key>.host`.

| Layer | When | What it checks | Failure |
|---|---|---|---|
| **Registration** — `PluginEndpoint._host_is_bare_hostname_or_ip` | At boot, when the config is parsed | *Spelling*: not URL-shaped, and either parseable by `ipaddress` or a hostname whose final label is not all-digits | Boot-time `ValueError`; manager does not start |
| **Resolution** — `plugin_invoke.resolve_endpoint` | On every `GET /api/plugins` and every invoke | *Address policy*: if the host is an IP **literal**, is it in the refused set? | `blocked_target` → HTTP 400 |

Neither layer checks that an accepted *hostname* is real or even host-shaped — see §6.4 and §6.5.

### 6.1 Registration: what is accepted

Rejected outright if the string contains any of `://`, `/`, `?`, `@`. Then one matching bracket pair is stripped and the remainder must be either (a) parseable by `ipaddress.ip_address`, or (b) a hostname whose final dot-separated label is **not** all digits.

| Host value | Registration | Why |
|---|---|---|
| `whisperx` | **accept** | hostname, final label alphabetic |
| `whisperx.internal` | **accept** | hostname |
| `plugin.example.com` | **accept** | hostname |
| `host-1` | **accept** | final label `host-1` is not all-digits |
| `localhost` / `LOCALHOST` | **accept** | hostname — but see §6.4 |
| `my_host` | **accept** | underscore is not checked |
| `example.com.` | **accept** | trailing dot ⇒ final label is `""`, which is not all-digits |
| `a b` | **accept** | embedded space is not checked |
| `""` (empty) | **accept** | emptiness is not checked |
| `127.0.0.1`, `::1`, `[::1]` | **accept** at registration | valid literals — **refused later at resolution** (§6.3) |
| `2606:4700::1` | **accept** | valid literal, not in the refused set |
| `http://whisperx` | **reject** | contains `://` |
| `whisperx/api` | **reject** | contains `/` |
| `user@whisperx` | **reject** | contains `@` |
| `whisperx?x=1` | **reject** | contains `?` |

### 6.2 Registration: the shorthand address spellings, and why they are refused

These **look like addresses but are not valid literals**. `ipaddress.ip_address()` refuses to parse them — so `resolve_endpoint`'s guard would never see an address at all and would wave them through as "just a hostname". The operating system's own `inet_aton` **does** expand them, to a real target. That gap is real: a registration of `127.1` resolves to `127.0.0.1`, so the loopback block would be a false guarantee for exactly those spellings.

The fix is at the **spelling** layer, at registration, with **no DNS resolution** (resolving here would change the trust model and add a time-of-check/time-of-use gap).

| Rejected host | What it looks like | What it would actually reach |
|---|---|---|
| `127.1` | short-form dotted quad | `127.0.0.1` (loopback — the manager's own API) |
| `2130706433` | 32-bit decimal | `127.0.0.1` |
| `0177.0.0.1` | leading-zero octal | `127.0.0.1` |
| `0x7f.0.0.1` | hex first octet | `127.0.0.1` |
| `[2130706433]` | bracketed decimal | nothing valid — brackets mark an IPv6 literal, so URL construction would fail; rejected on spelling regardless |
| `0` | bare zero | `0.0.0.0` (measured) |
| `1.2.3.4.5` | five octets | nothing — the resolver rejects it (measured `gaierror`); rejected on spelling regardless |

All produce the same boot-time error:

```
plugins endpoint host '127.1' is not a bare IP literal or hostname -- shorthand IP forms
(decimal/octal/hex/short, e.g. '127.1') are rejected so the SSRF guard cannot be
bypassed by spelling
```

**The cost of that rule: a legitimate hostname whose final label is all digits is also refused.** The test is purely `final label is not all-digits`, so `v1.2`, `svc.2`, `999`, and `my.host.8` are all rejected at boot with the "shorthand IP" message even though they are ordinary container names. **Name your plugin containers so the last dot-separated label starts with a letter.** `host-1` and `whisperx-2b` are fine; `host.1` is not.

### 6.3 Resolution: the address policy

`resolve_endpoint` checks **IP literals only**. The policy is a deliberate *subset* of the manager's general outbound SSRF denylist, because the two problems are inverted: the general guard exists to stop an arbitrary user-supplied URL from reaching internal infrastructure, whereas here the host **is** the operator's own boot config and the whole point of the registry is to name a container on the internal network.

**Refused — `blocked_target`:**

| Family | Ranges | Rationale |
|---|---|---|
| IPv4 | `0.0.0.0/8`, `127.0.0.0/8`, `169.254.0.0/16`, `224.0.0.0/4`, `240.0.0.0/4` | "this network", loopback, link-local/metadata, multicast, reserved |
| IPv6 | `::/96` (covers `::1`), `64:ff9b::/96`, `fe80::/10`, `ff00::/8`, `2001:db8::/32` | loopback/unspecified, NAT64 bypass class, link-local, multicast, documentation |

**Permitted — deliberately dropped from the general denylist:** the three RFC 1918 private-use IPv4 blocks, RFC 6598 CGNAT (`100.64.0.0/10`), RFC 4193 IPv6 unique-local (`fc00::/7`), and all global addresses. That is exactly the address space real plugin containers live in; blanket-applying the general denyset would refuse every plugin this feature exists for.

**Two link-local exceptions are kept on purpose.** `169.254.0.0/16` and its IPv6 analogue `fe80::/10` stay blocked in both families: no operator legitimately registers a plugin on the cloud metadata endpoint, and there is no legitimate-use cost.

**IPv4-mapped IPv6 is unwrapped, not blanket-blocked.** `::ffff:0:0/96` is deliberately absent from the refused set. `_is_never_valid_ip` extracts `ip.ipv4_mapped` and re-checks the embedded IPv4 against the IPv4 policy, so `::ffff:<private-address>` behaves identically to the bare form (permitted) while `::ffff:127.0.0.1` is refused.

**Brackets are normalised on both sides.** `_parse_ip_literal` strips one matching bracket pair before parsing, so `[::1]` is checked exactly like `::1` instead of falling through as an opaque hostname that skips the check. `invoke_plugin` then re-brackets an IPv6 literal for URL construction, never double-wrapping.

Measured behaviour:

| Host | Resolution |
|---|---|
| RFC 1918 / CGNAT / `fc00::1` / global addresses | **resolves** |
| `127.0.0.1`, `127.1.2.3`, `0.0.0.0` | `blocked_target` |
| `::1`, `[::1]`, `::ffff:127.0.0.1` | `blocked_target` |
| `169.254.169.254`, `fe80::1` | `blocked_target` |
| `224.0.0.1`, `240.0.0.1`, `ff02::1`, `64:ff9b::1`, `2001:db8::1` | `blocked_target` |
| `whisperx`, `localhost`, `""` | **resolves** — not an IP literal, so never checked |

### 6.4 Accepted limitation: a hostname is never address-checked

`resolve_endpoint` **deliberately performs no DNS resolution**. If `_parse_ip_literal` returns `None`, the endpoint is returned unchecked. The reason is contract honesty: this function is documented to raise only `unknown_resource` or `blocked_target`, and a DNS failure fits neither — hostname reachability, DNS failure included, is `invoke_plugin`'s `unreachable` concern. Resolving here would also introduce a time-of-check/time-of-use gap.

**The concrete consequence: `host: localhost` is accepted at registration and resolves cleanly, and it reaches loopback — including the manager's own API.** The IP-literal loopback block does not cover it. Read "loopback is blocked in both families" as *"loopback **spelled as an IP literal** is blocked in both families"*. This is a deliberate, accepted trade, not an oversight — but it is the most obvious spelling an operator would type, so it is worth stating plainly.

It cuts both ways. The hostname bypass is the only reason a plugin co-resident in the manager's own container is addressable at all — the deployment PLUGINS_SETUP.md calls PATH B, where `host: 127.0.0.1` registers cleanly and then fails every invoke with `blocked_target`/400 while `host: localhost` works. So: for a plugin in its **own** container, register the service name, never `localhost`; for a deliberately co-resident one, `localhost` is the supported spelling, and the port is the thing to double-check, because a mistyped loopback port reaches the manager's own API. **And if this registry ever stops being operator-only, pin resolution before relying on the loopback block** — resolve the name once and connect to the pinned IP, so the address that was checked is the address that is dialled. What makes the omission safe today is the operator-boot threat model, not a property of the code.

### 6.5 Accepted limitation: the host validator accepts some non-hosts

The registration layer never checks that an accepted hostname is *shaped* like a hostname. An empty string, a string with a space, an underscore, a trailing dot, and a `name:port` string all register cleanly — none is a config error at boot. They fail at first invoke instead, and **not all the same way**, which matters because the two outcomes point you at different things:

| Registered `host` | Registration | First invoke | Why |
|---|---|---|---|
| `"name:1234"` | accept | **`blocked_target` / 400** — *"could not build a request URL for plugin …"* | The embedded colon makes the port unparseable; `httpx.InvalidURL` hits the safety net. |
| `""` (empty) | accept | **`unreachable` / 502** | The URL becomes `http://:<port><path>`, which the client rejects as having no protocol — `httpx.UnsupportedProtocol`, a `TransportError`, so it lands on the transport branch, **not** the `InvalidURL` one. |
| `"a b"`, `"my_host"`, `"example.com."` | accept | **`unreachable` / 502** | A valid URL is built and the string is handed to DNS like any other hostname. A typo of this shape is a name that does not resolve. |
| host containing a control character | accept | **`blocked_target` / 400** | `httpx.InvalidURL` — *"Invalid non-printable ASCII character in URL"*. |

All measured on the pinned client. The consequence to internalise: **a 502 on this path does not prove the container is down** — it equally means you wrote a hostname that cannot resolve, or an empty one. The address is never in the response, so check `GET /api/config` → `registry_keys` and the server log before you go looking at the container.

---

## 7 — What needs a restart

| Change | Restart? | Why |
|---|---|---|
| Add / edit / delete a **plugin manifest** | **No** | `read_manifest` goes to disk on every request; `GET /api/plugins` and the invoke route both re-read. |
| Add / edit / remove a **registry entry** (`plugins.registry`) | **Yes** | `BootConfig` is frozen after `TurbohaulConfig.split()` at boot, and the routes read `mgr.boot.plugins.registry`. |
| `plugin_runtime.no_progress_timeout_s` | **No** | Read live off `mgr.runtime.plugin_runtime` on every invoke; `PUT /api/config` replaces `mgr.runtime` in place. |
| `plugin_runtime.enabled` / `max_concurrent` | **No** (but see §9 — nothing reads them) | Same live-runtime object. |

**The two halves of a first-time setup therefore have different latencies.** A manifest naming a `resource_key` with no registry entry **saves successfully** — there is deliberately no write-time gate, because one would make first-time setup a chicken-and-egg (you could not save the manifest before the registry entry existed, or vice versa). The only symptoms are `configured: false` in the listing and `unknown_resource` / HTTP 400 at invoke. Fixing it means editing boot config **and restarting**, while the manifest half you already saved was live instantly.

---

## 8 — The Plugins tab (UI)

The UI is a thin read/write view over `GET /api/plugins` plus the config surface (`GET /api/config`, `GET /api/config/schema`, `PUT /api/config`) — it never calls the invoke route. It shows, per plugin: `model_tag`, `lane` (uppercased), `capabilities` (joined; em-dash when empty), a Status badge, and an Enabled checkbox. Below the list sits one shared **Runtime settings** block with **Max concurrent** and **No-progress timeout (s)**.

- **It never shows host or port.** The client-side type has no such field, by design, and the backend redacts server-side.
- **Status** is `configured: true` → emerald **"Configured"**, `configured: false` → amber **"Not configured yet"**, never red. The data genuinely cannot distinguish a typo from a half-finished setup (§4.1), and the backend itself treats a manifest-before-registry-entry as a legitimate state, so the UI does not paint a fault the backend refuses to assert.
- **Bounds validation is schema-driven** — it reads `minimum`/`maximum` from `GET /api/config/schema` and simply skips a bound when the schema is unavailable. The server stays the bounds authority.
- **Every save PUTs the full `plugin_runtime` object**, including the complete `enabled` map. That is the correct behaviour for the nested map (§3.2).
- The tab states in-page that the boot registry is not editable there.

---

## 9 — Known gaps and accepted risks

Everything in this section is true of the shipped code today. None of it is theoretical.

**Plugin manifests are excluded from the model listings by `kind`, not by `hidden`.** `api/ollama.py`'s `get_tags` and `api/models.py`'s `list_models` both run `if not isinstance(m, ModelManifest): continue` *before* the `hidden` check and before any model-only field access, so a plugin manifest is never enumerated as a chat model and never reaches `gguf_size_bytes` / `gguf_blob_sha256` / `context_size`. A visible plugin manifest does **not** break `GET /api/tags`, and `GET /v1/models` does not list plugins. `hidden` therefore affects exactly one surface on this path: `GET /api/plugins` and the Plugins tab that reads it.

**`POST /api/manifests/{tag}/restore-defaults` on a plugin tag is an unhandled HTTP 500.** The route does `stored = dict(existing.llama_server_flags)` with no `isinstance` guard, and `AttributeError` is not in its `except` list. Do not call it on a plugin tag.

**A manifest that fails schema validation makes its own invoke an HTTP 500, while the listing quietly hides it.** The two routes handle the identical failure in different ways, and neither reports the cause to the caller. `GET /api/plugins` catches six exception types (`FileNotFoundError`, `ManifestValidationError`, `yaml.YAMLError`, `pydantic.ValidationError`, `OSError`, `UnicodeDecodeError` — every way `read_manifest` can fail), logs `skipping unreadable manifest`, and omits the entry — so a plugin with a typo, a permissions problem, or non-UTF-8 bytes simply *is not in the list*, indistinguishable from one that was never created. `POST /api/plugins/{model_tag}/invoke` answers the first two itself (404 and 400) and sends the other four to one handler that logs the full exception and returns `500` with the fixed body `{"detail":"manifest for '<tag>' is present but unreadable"}` — a plain-string `detail` with no `reason` key, outside the documented `{"reason", "detail"}` contract (§5.2). Applies to an unknown key, a dotted `resource_key`, a missing `lane`, and broken YAML syntax. **The server log is the only place either symptom is explained** — grep it for `skipping unreadable manifest` (listing) or `is present but unreadable` (invoke) before concluding a manifest was never saved.

**`plugin_runtime.enabled` and `max_concurrent` are inert.** The only consumer of `plugin_runtime` anywhere in the backend is `no_progress_timeout_s`, read by the invoke route. There is no semaphore or concurrency limiter on this path at all, and the invoke route never consults `enabled`. **Unchecking "Enabled" in the UI stores the intent and reports "Saved", but the plugin remains fully invokable.** Treat both as recorded intent, not enforcement.

**Turbohaul arranges no file transport of any kind.** The invoke body is JSON and only JSON — `client.post(url, json=payload, headers=headers)` (the only extra is the optional `Authorization` header), no multipart, no streaming, no upload endpoint, no shared staging directory. Getting a media file in front of the plugin is entirely the operator's problem, and the three ways to do it each cost something. Full treatment, with the convention these docs use: [PLUGINS_SETUP.md — "Getting the file to the plugin"](PLUGINS_SETUP.md#getting-the-file-to-the-plugin--you-arrange-this-turbohaul-does-not).

**`health_path` feeds a log-only background probe, nothing else.** The invoke path and the listing never use it. At boot, and then periodically (each registry entry every 60 s, backing off to at most 600 s while it keeps failing), the manager resolves the host name, opens a TCP connection, and issues `GET <health_path>` (with the entry's bearer credential, if one is configured). It logs a `WARNING` when an entry turns unhealthy and an `INFO` when it recovers, and restates a still-unhealthy entry about every 30 minutes; the address is never in those lines. The result is held in process and published nowhere: it does not feed `configured`, the Plugins tab, or any endpoint. A Status of "Configured" therefore still means "the `resource_key` resolves to a permitted address", never "the container is up".

**`lane`, `capabilities`, and `provides_routes` are echoed metadata only.** Nothing keys admission, queueing, or scheduling off `lane`, and neither `capabilities` nor `provides_routes` is ever compared against the `path` a caller invokes — a caller may POST any path shape that passes the three shape checks. `provides_routes` widens what an agent can *discover*, never what it can *reach*.

**The invoke route has no authentication and no request-body size ceiling.** The body-size middleware guards only `/v1/chat/completions` and `/api/chat`; the plugins router is registered with no auth dependency. Any caller that can reach the manager's port can POST an arbitrarily large `payload`, which is fully buffered and forwarded to the plugin container. Keep the manager's port on a trusted network.

**A plugin returning a non-object JSON body produces a raw HTTP 500.** `invoke_plugin` returns `response.json()` with no `isinstance` check and the route is annotated `-> dict`, which the web framework enforces as a response model. A plugin that legitimately answers a top-level JSON **array** yields `500 Internal Server Error` with no `reason` key — outside the documented error contract. Have your plugin wrap arrays in an object.

**Two of the five reasons blame the caller for an operator's mistake.** `unknown_resource` and `blocked_target`-at-resolve both map to HTTP **400**, but the agent supplies only `model_tag` and `path` and cannot fix either — both are boot-config errors. An agent retry policy that treats 4xx as "don't retry, my request was wrong" will silently mask a misconfigured registry.

**`unknown_resource` and `blocked_target` are indistinguishable by status code.** Both are 400. The `detail.reason` field is the only discriminator on the wire; the actual address is in the server log only.

**There is no total-duration timeout.** Only the per-operation no-progress read/write timeout plus the fixed 10 s connect floor. A plugin that keeps dribbling bytes can hold a request open indefinitely. The calling harness's own timeout is the real ceiling, and that is deliberately out of scope for this route.

**Redirect safety is implicit, not pinned.** The sibling pull path constructs its client with an explicit `follow_redirects=False`; `invoke_plugin` relies on the library default instead. It is correct on the version in use — measured: a 307 was never followed — but nothing in the module or its tests asserts it, and the project declares `httpx>=0.27.0`: a lower bound, not an upper pin, so a fresh dependency resolve is free to pick up a release with a different default. A dependency bump that flipped that default would turn every plugin into an open redirector past the address guard, surfacing only as a changed status code.

**The host validator is imprecise in both directions, and neither direction fails at boot.** It refuses legitimate container names whose final label is all digits (§6.2), and it accepts strings that are not hosts at all — empty, spaced, underscored, or carrying a control character (§6.5). The accepted ones fail on a caller's first invoke instead: `unreachable`/502 for most, `blocked_target`/400 only for a control character. **A mistyped registry host is a runtime 502, never a failed startup** — full measured breakdown in §6.5.

**The address policy is coupled to the general SSRF denylist by range *name*.** `plugin_invoke.py` filters the shared deny lists by string and asserts at import that every kept name still exists. Renaming a network in the shared module breaks the import loudly; **adding** a new deny range there silently does nothing here.

---

## 10 — Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| A model is missing from `GET /api/tags` | Its manifest is unreadable — corrupt YAML, a schema failure, or bad bytes — and was skipped | Grep the server log for `skipping unreadable manifest`; fix the file and re-`PUT` it (§9). Plugin manifests are excluded from this listing by `kind` and are never the cause |
| Validation errors about `gguf_blob_sha256 Field required` on a plugin file | `kind: plugin` omitted — it was parsed as a model | Add `kind: plugin` literally (§1.2) |
| `resource_key ... fails regex` | Dots in `resource_key` | Dots are legal in `model_tag`, never in `resource_key` (§1.4) |
| `Extra inputs are not permitted` on a manifest | An extra key — the schema is closed at eleven fields | Remove it; there is no `endpoint_url` field, by design (§1.1) |
| Manager will not boot: `host ... is not a bare IP literal or hostname` | The host's final dot-separated label is all digits, or is a shorthand IP spelling | Rename so the last label starts with a letter (§6.2) |
| `configured: false` in `GET /api/plugins` | Either no registry entry for that `resource_key`, or it is registered at a refused address | Check the server log — that is the only place the two are distinguished (§4.1) |
| Added the registry entry, still `configured: false` | The registry is boot-only | Restart the manager (§7) |
| `400 unknown_resource` at invoke, but the manifest saved fine | Manifest and registry are decoupled by design | Add the registry entry and restart (§7) |
| `400 blocked_target` on a working-looking address | Registered on loopback / link-local / multicast / reserved as an IP literal | Use the container's service name or a permitted address (§6.3) |
| `400 blocked_target` quoting the `path` you sent | `path` is not `/`-prefixed, contains `..`, or contains a control character | Fix the caller's `path` — this is the one `blocked_target` that is the caller's mistake, not the operator's (§5.1) |
| `502 unreachable` | DNS failure, connection refused, or a mid-flight drop | Server log has `endpoint=<host>:<port>` and the underlying exception |
| `504 no_progress` | The plugin sent no byte for `no_progress_timeout_s` | Raise `plugin_runtime.no_progress_timeout_s` (max 86400), or fix the plugin. Never set it to `0` (§3.3) |
| Every invoke 504s instantly | `no_progress_timeout_s` was set to `0.0` | `0.0` is an immediate timeout, not "disabled" (§3.3) |
| `502 plugin_error ... returned HTTP 307` | The plugin issued a redirect | Redirects are not followed, by design (§4.2) |
| Raw `500 Internal Server Error`, no `reason` in the body | The plugin returned a top-level JSON array | Wrap it in an object (§9) |
| `500` with `manifest for '<tag>' is present but unreadable` at invoke, and the tag is missing from `GET /api/plugins` | The manifest file on disk fails schema validation or will not parse as YAML | Server log has the full exception (invoke) and `skipping unreadable manifest '<tag>'` (listing); fix the file, then re-`PUT` it (§5.2, §9) |
| Tag missing from `GET /api/plugins`, but invoking it works fine | The manifest has `hidden: true` | Working as designed — `hidden` hides from listings only and never affects invocability (§1.1) |
| `412` on `PUT /api/manifests/{tag}` | Missing or stale `If-Match` on an update | `GET` the manifest, read its `ETag`, retry with it (§1.6) |
| Disabled a plugin in the UI, agents still call it | `enabled` is inert | Delete the manifest, or set `hidden: true` and stop publishing the tag (§9) |
| `403 sections ['plugins'] are BOOT-ONLY` | Tried to `PUT` the registry | Edit `turbohaul.yaml` and restart (§2.5) |
| A `plugin_runtime.enabled` entry vanished after a save | The nested `enabled` map is replaced wholesale | Always PUT the complete map (§3.2) |

---

## 11 — Where to go next

- [PLUGINS_SETUP.md](PLUGINS_SETUP.md) — the operator walkthrough: two files, one restart, and the error each mistake produces. This document is the lookup table; that one is the procedure.
- [MEDIA_HOOK.md](MEDIA_HOOK.md) — the conceptual companion: why a manifest can never carry an address, what the two lanes are for, and why a result comes back inline instead of as a job id. No schemas in it, by design.
- [MODEL_CONFIG_REFERENCE.md](MODEL_CONFIG_REFERENCE.md) — the model manifest: a sibling schema with its own required fields and its own flag allowlist.
- [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md) — wiring an agent to the chat surfaces. Note the naming collision: "provider plugin" there means a harness-side client shim, not a resource plugin.
- [ARCHITECTURE.md](../ARCHITECTURE.md) — the manager's queueing and slot lifecycle. Plugin invocation does not participate in either.
- [frontend/06-plugins.md](frontend/06-plugins.md) — the Plugins tab in the web interface.
- [VISION_MODELS.md](VISION_MODELS.md) — serving a vision model with its multimodal projector. A vision model reads images as input; it is not a plugin.

---

*This document is a contract. If you find a wire-shape detail — a field, a bound, a status code, a body key — that is not documented here, that is a doc bug.*
