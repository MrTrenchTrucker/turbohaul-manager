---
name: turbohaul-use
description: Use when operating a running TurboHaul Manager — changing settings, downloading a model into the blob store, writing per-model manifest config, or calling its HTTP API.
---

# Use TurboHaul Manager

You have a TurboHaul Manager running (if not, set it up first — see the `turbohaul-setup-and-launch`
skill). This skill is how you drive it: change settings, get a model in, configure that model, call it.

## The three objects — learn these first, they are the whole model

| Object | What it is | Where it lives |
|---|---|---|
| **A blob** | The model bytes. One file, named after the SHA256 of its own contents. No name, no tag, no settings. | `<blob_store_path>/sha256/<first-2-hex>/<full-64-hex>` |
| **A manifest** | The per-model configuration. One YAML file per model, named after the tag you choose. Names the blob, the context size, placement, and flags. | `<manifests_path>/<tag>.yaml` |
| **An endpoint** | What you call. The tag you chose in the manifest is the model name you pass to the chat routes. | `http://127.0.0.1:11401` |

Three consequences worth knowing before you touch anything:

- A blob can exist with **zero** manifests — bytes on disk that serve nothing. Getting a model in and
  making it callable are two separate steps.
- A manifest can name a blob that is not in the store, and the save still succeeds. The failure surfaces
  at load time, not save time.
- A blob's identity is its **hash**. A blob and a model have two different names and you must not mix
  them: the blob is the digest, the model is the manifest tag.

## There is no authentication

Default bind is loopback only, and the manager has no app-layer auth of its own. The bind address is the
entire security boundary. Keep it on loopback unless you have a network policy in front. Do not treat
"it answered without a token" as a misconfiguration.

## The workflow

**1. See what is already there before you add anything.**

```
curl -s http://127.0.0.1:11401/api/blobs    | jq '.total, .blobs[]'
curl -s http://127.0.0.1:11401/api/manifests | jq '.total, .hidden_count, .manifests[]'
curl -s http://127.0.0.1:11401/v1/models    | jq '.data[].id'
```

A blob with no matching manifest is not callable. A manifest whose blob is missing will fail at load.

**2. Get the model bytes into the blob store.** There are two flows. The staging flow is the normal one;
the direct pull is a convenience.

**2a. The staging flow — the designed path, and how the web UI's import works.**

A file must be somewhere the manager can read before it can be imported, and the designed place is the
staging directory (`/var/lib/turbohaul/import-staging` by default, and both launch paths mount your host
`./models` there). So the flow is two steps: the file lands in staging, then you import it into the blob.

```
# step 1 — put the file in staging (host side; the mount makes it appear inside the container)
curl -sL -o ./models/model.gguf https://example.invalid/model.gguf

# step 2 — import it into the blob store
curl -s -X POST http://127.0.0.1:11401/api/import \
  -H 'content-type: application/json' \
  -d '{"path": "/var/lib/turbohaul/import-staging/model.gguf"}'
```

The import path must be absolute, **the container's own view of the path**, under the import root, not
a symlink, and must resolve to an existing regular file whose first bytes are the model-file magic.
Anything else is a 400. The path you name is inside the container — if you already have the file on the
host, either put it in the mounted `./models` directory (it appears at
`/var/lib/turbohaul/import-staging`), or copy it in with
`docker cp ./model.gguf <container>:/var/lib/turbohaul/import-staging/model.gguf`.

**2b. The direct pull routes — the manager fetches the file for you.** These stream straight into the
blob store and do **not** use staging.

```
# from a URL
curl -s -X POST http://127.0.0.1:11401/api/pull-url \
  -H 'content-type: application/json' \
  -d '{"url": "https://example.invalid/model.gguf", "expected_sha256": "<optional>"}'

# from a model host — NOTE the field names
curl -s -X POST http://127.0.0.1:11401/api/pull-hf \
  -H 'content-type: application/json' \
  -d '{"repo_id": "owner/name", "filename": "model.gguf", "revision": "main"}'
```

The field names on the model-host route are `repo_id` and `filename`. Missing either one is a 400.
`expected_sha256` is optional on both; supply it when you have it. Whichever flow you use, the response
carries a `sha256` — that digest is the blob's identity, and it is what the manifest names.

**Do not call `POST /api/pull`.** It exists and answers **501 unconditionally** — it is a stub, and the
web UI's empty-state text points at it anyway. Use the routes above.

One caveat on the web UI: its **model-host form** sends `repo`/`file` while the backend requires
`repo_id`/`filename`, so that one form fails with a 400. Its **URL form** and its **import form** both
work. If you use the UI, use one of those two, or call the API directly.

**3. Write a manifest that names the blob.** The tag you choose here is the model name you will call.

```
curl -s -X PUT http://127.0.0.1:11401/api/manifests/my-model \
  -H 'content-type: application/json' \
  -d '{"model_tag": "my-model", "gguf_blob_sha256": "<digest from step 2>",
       "context_size": 8192, "gguf_size_bytes": 4294967296,
       "llama_server_flags": {"n_gpu_layers": "all"}}'
```

Only two fields are required: `model_tag` and `gguf_blob_sha256`. Everything else has a default — see
the field reference before you set anything you do not need to. `n_gpu_layers` is worth setting
deliberately: it is what asks the engine to offload layers to the GPU, and a manifest that omits it
does not request any offload. `"all"` means all layers; an integer is also accepted.

Re-sending the same `PUT` for an existing tag is refused with **412** until you send the entity tag the
manager returned. Read the manifest first, take its `ETag`, and pass it back:

```
ETAG=$(curl -sD - -o /dev/null http://127.0.0.1:11401/api/manifests/my-model | awk -F'"' '/[Ee][Tt]ag/{print $2}')
curl -s -X PUT http://127.0.0.1:11401/api/manifests/my-model \
  -H 'content-type: application/json' -H "If-Match: $ETAG" -d '{...}'
```

**4. Call it.** The manifest tag is the model name.

```
curl -s http://127.0.0.1:11401/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "my-model", "messages": [{"role": "user", "content": "hello"}]}'
```

**5. If it does not load, read the error before you change anything.** A model that does not fit is
refused at load, not at save, and a large cold load legitimately takes a couple of minutes — do not
restart during the first load. Failures arrive as HTTP 503 with a body of
`{"detail": "<reason>"}`; the reason names which limit was hit. Some 503 responses carry a
`Retry-After` header — back off for the number of seconds it states rather than guessing. If the
reason is memory, see placement and sizing below; if it is a missing blob, the digest in your manifest
does not match anything in the store.

## Changing settings

Settings come from four layers; highest wins:

    remembered runtime settings  >  environment variable  >  the config file  >  built-in defaults

Two things that surprise people:

- **An API change persists and keeps winning — including over an environment variable.** Settings
  written through the API are stored in a separate file beside the state database and applied LAST at
  every boot, after the environment variables. So an API edit outranks both the config file and the
  environment, even after a restart. If "my config says one thing and the manager does another", this
  is why, and `runtime_config.yaml` is where to look.
- **Some sections cannot be changed without a restart.** `server`, `storage`, `runtime`, `ui` and
  `plugins` are boot-only and are refused by the API. `queue`, `pull`, `persist`, `monitor`, `kv`,
  `http`, `fastlane` and `plugin_runtime` are runtime-mutable.

```
curl -s http://127.0.0.1:11401/api/config          | jq   # effective values
curl -s http://127.0.0.1:11401/api/config/schema   | jq   # types, defaults, bounds
curl -s -X PUT http://127.0.0.1:11401/api/config \
  -H 'content-type: application/json' -d '{"kv": {"ram_cache_max_bytes": 8589934592}}'
```

A `PUT` is refused with 403 for a boot-only section and 400 for an unknown key. After a write, read
`GET /api/config` back and confirm the value is in force — do not assume the write landed.

## Reference

The exhaustive detail lives in two reference files. Load the one you need:

- `references/settings-and-layout.md` — every settings key with type, default and description, the
  environment-variable list, and where everything lives on disk (including the paths that cannot be
  moved by config or environment at all).
- `references/models-manifests-and-api.md` — the full blob and manifest field references, the manifest
  lifecycle and its error cases, the validation rules that will refuse your save, placement and VRAM
  sizing, and the complete HTTP endpoint inventory with error and retry behaviour.

## Traps that will cost you time

- **A manifest save does not check the blob exists.** A typo in the digest saves fine and fails at load.
- **The web UI's model-host form does not work.** It sends `repo`/`file`; the backend needs `repo_id`/
  `filename`, so it 400s. The UI's URL form and import form both work, or call the API directly.
- **`POST /api/pull` is a 501 stub**, despite what the UI's empty state says.
- **The reasoning-effort parameter is inert more often than you would expect.** It does nothing when
  the model has no reasoning budget, a zero budget, or a negative one — the request is still accepted,
  the parameter is simply ignored. One of its accepted values, `minimal`, is deliberately rejected with
  a 400 rather than silently treated as "very low".
- **The request body ceiling is enforced by `Content-Length` only**, so a chunked body is not measured
  against it. Do not rely on it as a hard limit.
- **Errors are `{"detail": ...}`.** Some 503 and 504 responses carry a `Retry-After` header; its value
  is derived from the configured model-load timeout rather than a fixed constant, so honour the header
  rather than assuming a number.
- **Back up the whole state tree, not just the database file.** The database runs in a mode that writes
  recent changes to two extra files beside it.