# Part 2 — Getting a Model In, Configuring It, and Calling It

## The three objects

Three objects, and confusing them is the most common source of wasted time:

- **A blob** is the model bytes: one file on disk, named after the SHA256 of its own contents. It has no name, no tag, and no settings.
- **A manifest** is the per-model configuration: one YAML file per model, named after the tag you choose. It says which blob is the weights, how large they are, how much video memory to expect, and every tuning flag the engine is started with.
- **An endpoint** is what you call. The tag you pick in the manifest is the name you pass to the chat routes.

A blob can exist with **zero** manifests — bytes on disk that serve nothing. A manifest can name a blob that is not in the store, and the save still succeeds. A pull or an import gives you a blob and stops; **writing a manifest is the separate step that makes a model servable.** Do the two in order.

## Authentication: there is none

There is **no application-layer authentication on any endpoint** — no API key, no bearer token, no login. The only request-side middleware is a body-size limiter on the chat routes. Every credential you will find in this product is *outbound*: the model-hosting key used for gated downloads, the plugin invocation token.

The protection is the network perimeter. The default bind is `127.0.0.1:11401` (loopback only), and a public bind is refused from configuration unless it is explicitly enabled **and** the manager is also launched with the public-bind flag. So: on loopback, every caller with local access is fully privileged. Do not expose the port to a shared or untrusted network without putting your own access control in front of it — there is nothing in the product that will stop a remote caller from importing a file or rewriting a manifest.

Throughout this document, examples use `BASE=http://127.0.0.1:11401`.

## The blob store

### What it is, and how a model is identified

The blob store is **a directory of files on local disk**. Not a database, not an object store, not an index. It is content-addressed: one flat file per model, sharded two levels deep:

```
<blob_store_path>/sha256/<first-2-hex>/<full-64-hex-sha256>     final, mode 0400 (read-only)
<blob_store_path>/sha256/incoming/pull_<random>.tmp             transient, only during a transfer
```

Only a well-formed lowercase 64-hex digest can ever resolve to a path.

**A blob's identity is its hash.** There is no name, tag, or filename in the store. A blob and a *model* have two different names and you must not mix them: the *file* has only its 64-character hash; the *model* has the short tag you choose in its manifest. The tag is what you pass to chat endpoints.

The relationship to a manifest is one-way and loose. A manifest never names a path — it names a **hash** (the weights, and optionally a vision projector and a standalone draft model, each a 64-hex content address into the same store). The manager resolves hash to path itself when it builds the engine command line. Many manifests may point at the same blob; one blob may have no manifests at all and still be listed.

### Listing what is already there

```bash
curl -s $BASE/api/blobs
```

```json
{"blobs": [{"digest": "…64 hex…", "size_bytes": 1234567, "description": null}], "total": 1}
```

This is a pure read-only walk of the shard directories. It **deliberately does not join against manifests**, which is exactly why a blob nobody has written a manifest for is still visible here. Each size is stat'ed in its own error handler, so one file vanishing mid-listing cannot blank the whole response.

Do not confuse it with the other two listings:

| Call | Lists | Includes blobs with no manifest? | Includes hidden models? |
|---|---|---|---|
| `GET /api/blobs` | raw stored files | **yes** | n/a |
| `GET /v1/models` | models (manifests) | no | **no** |
| `GET /api/tags` | models (manifests), Ollama shape | no | **no** |

There is no command-line equivalent of `GET /api/blobs` — the manager program is a server launcher, not a tool with subcommands.

To attach a human label to a blob:

```bash
curl -s -X PUT $BASE/api/blobs/<digest>/description \
  -H 'content-type: application/json' \
  -d '{"description": "14B instruct, Q4_K_M"}'
```

An empty string or `null` clears it. Maximum 2000 characters. A malformed digest is 400; an absent blob is 404. The label map is a single JSON sidecar next to the manifests directory, keyed by digest — it is the only blob-level metadata that exists.

### Getting a model in — the staging flow and the two direct pulls

The designed flow is **two steps: a file lands in the staging directory, then you import it into the
blob store.** The staging directory is the configured import root (default
`/var/lib/turbohaul/import-staging`), and both launch paths mount your host `./models` at that path, so
a file you fetch or drop on the host appears inside the container ready to import.

```
# step 1 — the file lands in staging (host side; the mount exposes it to the manager)
curl -sL -o ./models/model.gguf https://example.invalid/model.gguf

# step 2 — import it into the blob store
curl -s -X POST $BASE/api/import -H 'content-type: application/json' \
  -d '{"path": "/var/lib/turbohaul/import-staging/model.gguf"}'
```

There are also two **direct pull** routes where the manager fetches the file itself; these stream
straight into the blob store and do **not** touch staging. All paths end the same way: you get a
`sha256`, and that digest is the blob's identity.

There is no file-upload endpoint (no multipart, no PUT-bytes), so you can never hand the manager raw
bytes over HTTP: either the manager fetches them, or you put the file where the manager can see it and
tell it the path.

**1. Download from an arbitrary HTTPS address** — the manager fetches the URL for you.

```bash
curl -s -X POST $BASE/api/pull-url -H 'content-type: application/json' \
  -d '{"url": "https://example.invalid/model.gguf", "expected_sha256": "<optional 64-hex>"}'
```

```json
{"pull_id": "pull-…", "status": "complete", "sha256": "…", "bytes_written": 4294967296, "host": "example.invalid"}
```

The URL must be `https://` and is checked against a double-resolve SSRF guard before anything is opened — 400 if it fails. `expected_sha256` is optional; supply it and the manager aborts and discards the partial file if the bytes do not match. Errors: 400 missing URL or unsafe URL or hash mismatch, 413 over the size ceiling, 502 upstream fetch failure, 500 other write failure.

**2. Download a file from a model host** — the manager builds the address itself; you never paste a URL.

```bash
curl -s -X POST $BASE/api/pull-hf -H 'content-type: application/json' \
  -d '{"repo_id": "owner/name", "filename": "model.gguf", "revision": "main"}'
```

Field names are `repo_id` and `filename` — **not** `repo` and `file`. `revision` defaults to `main`; `expected_sha256` is optional. Missing `repo_id` or `filename` is 400. A resolved host outside the configured allowlist is **403** (the shipped allowlist covers the main hosts and, by subdomain match, their CDN hosts). For a private or gated repository, set the API-key environment variable on the manager — its name defaults to `HF_API_KEY` and is configurable. Never put the key value in a manifest, a request body, or a log. The key is read only after the host allowlist check passes, and it is stripped from the request on any cross-host redirect hop.

**3. Copy a file already on the manager's filesystem.**

```bash
curl -s -X POST $BASE/api/import -H 'content-type: application/json' \
  -d '{"path": "/var/lib/turbohaul/import-staging/model.gguf"}'
```

```json
{"pull_id": "import-…", "status": "complete", "sha256": "…", "bytes_written": 4294967296, "source": "local-import"}
```

The path sandbox is strict and you should not expect anything outside it to work: the path must be **absolute**, must be under the configured import root, must not itself be a symbolic link, must not touch a denied system prefix, and must resolve to an existing regular file. The first four bytes must be the model-file magic. A rejection is 400, over the ceiling is 413, a hash mismatch is 400. To import a file, place it in the staging directory first (for example with a bind mount or `docker cp`).

**The advertised fourth route is a stub.** `POST /api/pull` — the Ollama-registry-shaped pull — exists and answers **501 not implemented**, unconditionally, with a message telling you to use one of the two working routes. It is the single most likely wrong turn for someone following older instructions, and the empty-state copy in the shipped web UI points at it. Do not call it.

### The web UI's download forms: which work, and the one that does not

The UI's Blob tab is wired to the real routes. Two of its three forms work and one does not:

- Its **URL form** (`/api/pull-url`) sends `url`, which the backend reads. **Works.**
- Its **import form** (`/api/import`) sends `path`, which the backend reads. **Works.**
- Its **model-host form** (`/api/pull-hf`) sends `repo` and `file`, while the backend requires `repo_id`
  and `filename`. That call is **rejected with 400** and cannot succeed as shipped. Use the URL form, the
  import form, or call `/api/pull-hf` directly with the right field names.

Separately, note that **none** of these forms creates a manifest, and a `tag` field some callers send is
not read by the backend on any route. That is not a fault in the UI: a blob and a servable model are two
different things by design. Getting bytes in is step one; writing the manifest that names them is step
two (see the manifest sections above). A blob with no manifest will not appear in any model listing —
that is expected, not an error.

### Limits, integrity, and housekeeping

- **Size ceiling.** One transfer is capped at 100 GiB by default, adjustable at runtime through the `pull` config section. Exceeding it is 413 and the partial file is discarded — you get a clear "too large" error rather than a full disk.
- **Integrity.** Every write streams to a temp file, hashes as it goes in a single pass, optionally compares against your expected hash, fsyncs, atomically renames into place, fsyncs the parent directory, then marks the file read-only. If you supply no expected hash, the computed hash becomes canonical — the store names the file, you never do.
- **No resume.** There is no range request and no partial-file reuse. A dropped transfer deletes its temp file and restarts from zero. On a flaky connection, fetch the file yourself with a resuming client and use `POST /api/import` instead.
- **Deduplication is free but there is no skip.** Two identical files hash the same, so the second write lands on the same path. But there is no "already present, skip the download" short-circuit: re-pulling a blob you already have **re-downloads the whole file** and then overwrites the same path.
- **Nothing is ever evicted.** No quota, no LRU, no size-based reaping. The only way a stored file is removed is an explicit delete.
- **Crashed transfers can leak.** A stale-tempfile cleanup function exists in the store but **has no runtime caller** — a transfer killed by a crash or a signal can leave a multi-gigabyte `.tmp` sitting in the `sha256/incoming/` staging directory indefinitely. If disk pressure appears after a failed pull, look there and delete the file by hand.

### Deleting a blob

```bash
curl -s -X DELETE $BASE/api/delete -H 'content-type: application/json' \
  -d '{"digest": "sha256:<64-hex>"}'
```

`{"sha256": "<64-hex>"}` is accepted in place of the prefixed form. Existence is checked first, so a missing blob is **404**. If any manifest still names the digest — in any of its three blob-hash fields — the delete is refused with **409**, and the detail names the offending tags and tells you to delete or re-point them first. The guard reads the raw stored YAML rather than the validated manifest, so even a corrupt manifest file blocks the delete. That is deliberate: a manifest is a text file an operator repairs in a minute, a blob is a multi-gigabyte download. On success the description sidecar entry is pruned.

Delete the manifests first, then the blob. The reverse order always 409s.

## The manifest

### Where it lives and what it is

One YAML file per model, at `<manifests_path>/<tag>.yaml`, where the tag is the model's name. Files on disk, not a database table — the directory is scanned for `*.yaml`, and the filename stem is the tag. State that must be queryable (audit, queue bookkeeping) lives in a separate SQLite file; the manifests themselves do not.

Two variants, discriminated by a `kind` field: `model`, the one you care about, and `plugin`, a resource-plugin manifest that is still a work in progress. A file with no `kind` key loads as `model`, so no migration is needed.

The schema is **closed**: any unknown top-level key is a hard 400, not an ignored extra. This is a deliberate safety property, and the error names the offending key.

The tag must match `^[a-z0-9][a-z0-9._-]{0,63}$` — lowercase ASCII only, must start with a letter or digit, no path separators, no traversal, at most 64 characters. It is re-validated on every read, write, and delete, not just on create.

### Field reference — `kind: model`

| Field | Type | Default | Description |
|---|---|---|---|
| `model_tag` | string | **REQUIRED** | The model's name. Lowercase ASCII, 1–64 chars, starts with a letter or digit. Becomes the filename and the name you pass to chat. The URL always wins over any value in the body. |
| `gguf_blob_sha256` | string | **REQUIRED** | Which blob is the weights: exactly 64 lowercase hex chars. A content address, never a path, repo, or URL. |
| `kind` | string | `"model"` | Discriminator. Omit it; a manifest with no `kind` is a model manifest. |
| `display_name` | string | `""` | Cosmetic label shown in listings. |
| `description` | string | `""` | Cosmetic notes. |
| `revision` | integer | `1` (min 1) | Optimistic-concurrency counter, and the ETag value. The product increments it on every successful write. |
| `hidden` | boolean | `false` | Listings-only visibility. Hides the model from `/v1/models` and `/api/tags` while leaving it **fully loadable and servable by its exact tag** — nothing in the serve path reads this. |
| `mmproj_blob_sha256` | string | `""` | Optional vision projector blob: empty, or 64 lowercase hex. Empty means text-only. |
| `spec_draft_gguf_blob_sha256` | string | `""` | Optional standalone draft model for speculative decoding: empty, or 64 lowercase hex. |
| `gguf_size_bytes` | integer | `0` (min 0) | Declared on-disk weights size. Feeds the key-cache fit estimate as the model-body term. **At 0 the detailed fit check is skipped entirely** — put the real size here. |
| `context_size` | integer | `2048` (min 1) | Declared model context length. Drives the fit estimate, so bumping it without headroom is refused at spawn. |
| `expected_vram_bytes` | integer | `0` (min 0) | Your honest answer to "how much video memory does this model need on one card", in bytes. See *Placement and VRAM* — 0 has real consequences. |
| `auto_place` | boolean | `false` | Top-level placement opt-in, **not** a flag inside `llama_server_flags`. `false` honours your pinned card verbatim. `true` only delegates card choice when `split_mode` is `none`. |
| `arch` | string | `""` | Architecture identifier. One designated hybrid architecture activates a different key-cache fit branch. |
| `hybrid_kv_ratio` | float | `1.0` (0.0–1.0) | Fraction of layers that grow a per-token key cache. `1.0` = pure attention. Only multiplies the legacy file-size estimate path. |
| `kv_bytes_per_token` | float or null | `null` (min 1024.0) | Operator-measured effective key-cache cost **in bytes per token** (13.5 KiB/token = `13824.0`). Used verbatim, bypassing the closed-form estimate. The 1024 floor exists to reject a KB-vs-bytes typo. |
| `llama_server_flags` | object | `{}` | The closed allowlist of engine flags. See below. |
| `prompt_template` | object | `{}` | `{system_default: string = "", stop_tokens: list[string] = []}`. Unknown keys inside it are rejected too. |

### Field reference — `kind: plugin`

Listed so you can recognise these rows in a listing; the variant is incomplete. It carries none of the model's fields — no flags, no weights hash, no context size.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | string | **REQUIRED** | Must be `"plugin"`. |
| `lane` | string | **REQUIRED** | `cpu` or `gpu`. |
| `resource_key` | string | **REQUIRED** | Registry lookup key: `^[a-z0-9][a-z0-9_-]{0,63}$`. Never a URL or a path. |
| `capabilities` | list[string] | `[]` | Declared capabilities. |
| `provides_routes` | list[string] | `[]` | Max 64 entries, each `^/[A-Za-z0-9._~/-]{0,127}$`. **This is not an allowlist**: adding a route here grants nothing and removing one revokes nothing. It widens what a client can *discover*, never what it can *reach*. |
| `provides_executables` | list[string] | `[]` | Each `^[A-Za-z0-9_][A-Za-z0-9._-]{0,63}$`. |

### One flag you must set to serve embeddings

If you want a model to answer `POST /v1/embeddings`, the manifest must say so:

```
"llama_server_flags": {"embeddings": true}
```

Without it the route refuses the request with **400** and a message naming
`manifest.llama_server_flags.embeddings`. This is the first check that runs, so a manifest that looks
otherwise correct will still fail every embeddings call. `embeddings` is on the allowlist, and no other
setting enables it.

### The flag allowlist

`llama_server_flags` is a **closed** map of 108 engine flags, each `snake_case` in the manifest and `--kebab-case` on the engine's command line. Anything not on the list is a 400 naming the offending key and the allowlist size. There is no flag that names a GPU, and no flag that can smuggle in a path, URL, repository, or credential: a family of flag *name patterns* (anything ending in `_file`, `_path`, `_dir`, `_url`, `_repo`, `_key`, `_model`, plus a set of prefixes such as `hf_`, `lora`, `control_vector`, `lookup_cache_`, `ssl_`, `api_key`, `slot_save_`, `webui_`, `docker_`) is permanently refused, alongside an explicit denylist that includes a tool-execution flag. That is why the projector and the draft model are configured by pasting a content hash and never a path.

Do not try to memorise or hand-transcribe the list — it is long and it changes. Get the authoritative list one of two ways: the structured editor in the web UI mirrors it, or submit an unknown flag and read the error. Values are type-checked (a boolean is never accepted for an integer flag and vice versa, though int promotes to float), string flags are checked against their enumerations, and numeric flags are range-checked. Special cases: the attention flag takes a boolean or a named mode; the GPU-layer count takes an integer or `all`/`auto` and rejects a boolean; the chat-template flag must be a built-in name or a short plain identifier and is rejected if it contains templating constructs; the multi-GPU split string must be a strict comma-separated list of 2–16 non-negative decimals.

Representative bounds, so you can catch an obviously wrong value before saving: context window `1..2000000`; GPU layers `-1..999` plus the two strings; max output tokens `-1..1000000`; thinking budget `-1..1000000`; concurrent slots `1..256`; main GPU index `0..16`; log verbosity `0..5`; speculative draft tokens `0..64`.

### Defaults you do not have to write down

Three flags carry an injected built-in default: the checkpoint count, and the type of the key and value halves of the cache (a purpose-built low-bit cache, not a general-purpose quantisation applied to cache data). Leave them out and the model follows the built-in default, and a later product update changes it for you automatically.

Injected defaults are **never written to the file**. An explicit value always wins, including a deliberate `0`. This means a manifest you never touched will start following a *new* default after an upgrade. Read the exact current values back from the restore-defaults response rather than assuming them; the value strings are internal implementation names and are deliberately not reproduced here.

## Manifest lifecycle

Every route is under the `/api/manifests` prefix.

| Operation | Method and path | Body | Notes |
|---|---|---|---|
| List | `GET /api/manifests` | — | Management listing: **every** manifest, hidden ones included, both kinds. `{manifests: [...], total, hidden_count}`. |
| Read one | `GET /api/manifests/{tag}` | — | Full manifest body plus an `ETag` response header holding the quoted revision, plus a derived-only `cache_reuse_inert_mmproj` marker on model manifests. |
| Create / full replace | `PUT /api/manifests/{tag}` | the whole manifest | Full replace, not a merge. `If-Match` optional on create, required on update. |
| Toggle visibility | `PATCH /api/manifests/{tag}` | `{"hidden": true}` | Accepts `hidden` and nothing else. |
| Reset defaults | `POST /api/manifests/{tag}/restore-defaults` | — | Drops this model's overrides of the three defaulted flags only. |
| Delete | `DELETE /api/manifests/{tag}` | — | Removes the file. 404 if absent. |

**Create a model.** This is the step that turns a blob into a servable, named model — the pull routes return a bare hash and stop.

```bash
curl -s -X PUT $BASE/api/manifests/my-model \
  -H 'content-type: application/json' \
  -d '{
        "model_tag": "my-model",
        "gguf_blob_sha256": "<64-hex from the pull response>",
        "gguf_size_bytes": 4294967296,
        "context_size": 8192,
        "expected_vram_bytes": 6442450944,
        "display_name": "My 14B instruct",
        "llama_server_flags": {"n_gpu_layers": -1, "ctx_size": 8192}
      }'
```

```json
{"status": "ok", "model_tag": "my-model", "revision": 1, "restart_required": false}
```

The `model_tag` in the body is overwritten from the URL, so a mismatch is impossible. Writes are atomic (temp file in the same directory, fsync, rename, fsync the parent) and each successful write bumps `revision` by one.

**Update an existing model** — read first, send the ETag back:

```bash
ETAG=$(curl -s $BASE/api/manifests/my-model | python3 -c 'import sys,json;print(json.load(sys.stdin)["revision"])')
curl -s -X PUT $BASE/api/manifests/my-model -H 'content-type: application/json' \
  -H "If-Match: \"$ETAG\"" -d @full-manifest.json
```

The header must carry the **quoted** form exactly as returned — `If-Match: "3"`. The listing gives you that quoted form directly as an `etag` field; use it verbatim. Sending the bare integer, a stale value, or nothing at all on a model that already exists is **412**. That refusal is the lost-update guard: without it, two callers could silently overwrite each other.

An `If-Match` sent on a **create** is ignored: the comparison only happens when the file already exists, so a create with a stale or wrong `If-Match` still succeeds. The rule is the simple one: on create, do not send it; on update, send it.

**Toggle visibility** — a narrow, race-free write:

```bash
curl -s -X PATCH $BASE/api/manifests/my-model -H 'content-type: application/json' -d '{"hidden": true}'
```

It exists so that flipping visibility never has to round-trip a whole manifest through `PUT`, which would be a read-modify-write race. Its three 400s: a key other than `hidden`, a body with no `hidden` key at all, or a `hidden` that is not a boolean. It does not require `If-Match`.

**Reset defaults:**

```bash
curl -s -X POST $BASE/api/manifests/my-model/restore-defaults
```

```json
{"cleared": ["ctx_checkpoints"], "now_defaults": {"…": "…"}, "restart_required": false, "takes_effect": "next model spawn"}
```

It is scoped deliberately: it removes your per-model *values* for those three flags so the model follows the global default again, and it does **not** write the default in. It does not touch context size, the split ratio, or anything else you tuned.

### Error cases you can actually trip

| Situation | Status | What it means |
|---|---|---|
| Bad tag, unknown top-level key, bad hash shape, a flag not on the allowlist, a cross-field conflict | **400** | Validation. The detail text names the offending field and value. |
| Corrupt YAML or non-UTF-8 bytes in the file | **400** | The detail carries the parser's line/column of *your* typo. |
| Manifest present but unreadable for another reason | **400** | A fixed message with no filesystem path; the real reason goes to the log. |
| Manifest does not exist (read, patch, delete) | **404** | `manifest not found: <tag>` |
| `If-Match` missing or not matching on an update | **412** | Re-read the model and retry with the fresh ETag. |
| A blob-hash that is well-formed but names nothing in the store | **not an error** | The save succeeds. The failure surfaces later, at spawn, as the engine being handed a path to a file that does not exist. Check the digest against `GET /api/blobs` yourself. |

One caution: values you paste are **echoed back in the error message** for a manifest you own, because that is what you need to fix it. Be aware when pasting an error verbatim somewhere shared.

**Unreadable files do not break the listing.** A manifest that fails to parse still appears in `GET /api/manifests`, as a row with its three blob-hash fields `null` and `error: "unreadable"`. One broken file never blanks the view, and `hidden_count` only counts rows whose `hidden` is exactly `true`, so unreadable rows do not inflate it.

**Change notifications carry identifiers only.** Every write publishes a `manifest_changed` event over the state WebSocket carrying the tag and, when known, the revision. Never the body, the flags, the display name, or the description.

**Most changes are not instant.** A model that is already loaded keeps running with the settings it started with. Manifests hot-reload on the next load; restore-defaults reports `takes_effect: "next model spawn"` for exactly this reason.

## Validation: the cross-field rules that will refuse your save

Two rules reject an otherwise well-formed manifest because two fields disagree.

**More than one concurrent slot requires the unified key-cache pool.** Setting concurrent slots above 1 without also setting the unified pool flag to true is a 400, with the fix in the message. The per-slot context divisibility check below it is currently unreachable while this requirement stands; it reactivates automatically if the requirement is ever relaxed, so do not rely on it being dead.

**A thinking budget must sit strictly below the output ceiling.** If you cap the thinking budget at a value greater than or equal to the maximum output length, the save is refused — the model would spend its whole allowance inside its thinking and return nothing. Fix it either way: lower the thinking budget below the output cap, **or** set the output length to its unlimited value (`-1`), which exempts you from the rule, since there is then no ceiling to violate. Both fields accept `-1..1000000`, where `-1` means unbounded for each in its own sense.

**A third combination is tolerated, not refused.** Asking to reuse the prefix cache on a model that has a vision projector is inert for that model, and the save is **allowed** with an informational log. It is not wrong, and refusing it would break a manifest that works today.

**A retired key is tolerated; every other unknown key is an error.** Exactly one legacy key — a former per-model instance-limit field — is silently dropped with a warning rather than failing the save, and a one-time migration rewrites stored files without changing the revision, so a client holding the old ETag can still save. Unknown keys that are not that one are untouched and still fail.

**Not reachable as a save-time error:** a blob that does not exist, and a mismatch between the declared GPU-layer count and the visible cards. The first is not checked at write time. The second is checked at **spawn** time, not save time, along with the requirement that a multi-GPU split ratio list have exactly as many entries as there are visible cards.

## Placement and VRAM

A manifest does not name a GPU. It names *how the weights are split* and *whether to delegate the card choice*.

| To do this | Set |
|---|---|
| Pin a model to one specific card | `llama_server_flags.split_mode: "none"` and `llama_server_flags.main_gpu: <index>` |
| Spread across cards yourself | `split_mode: "layer"` (or `"row"` / `"tensor"`) and, optionally, a `tensor_split` ratio list |
| Let the manager choose the card | top-level `auto_place: true` **and** `llama_server_flags.split_mode: "none"` |

**`auto_place` is only a delegation when `split_mode` is `none`.** With the default `auto_place: false`, the pinned card is honoured verbatim. With `auto_place: true` **and** `split_mode: "none"`, the manager at admit time picks the card with the most free video memory (most-free, not first-fit, which is what spreads load); if no single card fits but the aggregate does, it falls back to a layer split across all cards; and if nothing fits it **refuses** rather than blindly admitting a model that will crash. An absent `split_mode` is not the same as `"none"`, so a manifest with `auto_place: true` and no explicit split mode keeps the single-engine path.

**`expected_vram_bytes` is your declared footprint, and it is inert at 0.** It engages in two places:

1. **The free-memory floor.** The required free threshold is `max(box-wide minimum free MiB, expected_vram_bytes ÷ 1 MiB)`. Declared `0` contributes nothing, so only the box-wide floor guards you. That floor defaults to 512 MiB. The refusal message names both terms, so you can see which one bound.
2. **Expert-offload configurations — the one place your number is authoritative rather than advisory.** For a model configured with the expert-offload flag, your `expected_vram_bytes` is treated as the operator's *measured* footprint (the reduced on-GPU body plus its cache) and **replaces** the closed-form estimate outright. **Both conditions must hold:** the expert-offload flag must be set *and* `expected_vram_bytes > 0`. Leave it at 0 and the closed form is used, which over-counts for offload configurations and can refuse a model that would actually fit.

The same pattern appears in the admission path and in the drop-planning estimate, so a wrong number has effects beyond the initial spawn.

**`gguf_size_bytes: 0` disables the detailed fit check.** The key-cache fit gate derives its prediction from the context size, the declared weights size, and the quantisation — precisely so that raising `context_size` from 4096 to 65536 is refused if the resulting cache will not fit. With `gguf_size_bytes` at 0 there is nothing to compute from, and the gate passes itself as having insufficient input. So the two numbers that decide whether a model will start are `gguf_size_bytes` and `context_size`, and the three that decide *how* are `expected_vram_bytes`, the KV fields, and the split settings.

**KV estimate precedence**, in order: a measured `kv_bytes_per_token` override, then parsed dimensions, then a legacy file-size heuristic. A set `kv_bytes_per_token` is used verbatim — total bytes = `kv_bytes_per_token × context_size`, with no quantisation scaling and no hybrid multiplier, because those paths already reflect only the growing attention layers and re-applying `hybrid_kv_ratio` would double-discount. `hybrid_kv_ratio` multiplies only the legacy file-size path.

**A thinking budget is a launch setting, not a per-request one.** Set it in the manifest. A separate request-time clamp can *reduce* the effective budget so it fits the output ceiling for a given request; it never raises it. Do not try to drive thinking depth from a request field and expect the manifest to follow.

## The HTTP API

### Endpoint inventory

| Surface | Endpoints |
|---|---|
| Health and status | `GET /health` → `{"status":"ok","version":"…"}`; `GET /status` → queue, active, loading, grace, idle-hot, evictions, pending-reclaim counters; `GET /api/version` |
| Configuration | `GET /api/config` (effective settings, with per-field provenance); `GET /api/config/schema`; `PUT /api/config` |
| OpenAI-compatible | `POST /v1/chat/completions`; `GET /v1/models`; `GET /v1/models/{model}`; `POST /v1/embeddings`; `GET /v1/logging`; `GET /v1/telemetry/events`; `GET /v1/telemetry/status` |
| The same two model paths without the prefix | `GET /models`, `GET /models/{model}` — one handler, identical responses |
| Ollama-compatible | `POST /api/chat`; `GET /api/tags`; `GET /api/show?name=<tag>` |
| Fast Lane | `GET /api/fastlane/census` — the discovered-address census |
| Manifests | `GET /api/manifests`; `GET /api/manifests/{tag}`; `PUT`/`PATCH`/`DELETE /api/manifests/{tag}`; `POST /api/manifests/{tag}/restore-defaults` |
| Blobs | `POST /api/pull-hf`; `POST /api/pull-url`; `POST /api/import`; `GET /api/blobs`; `PUT /api/blobs/{digest}/description`; `DELETE /api/delete` |
| Plugins | `GET /api/plugins`; `POST /api/plugins/{model_tag}/invoke` |
| Streaming and events | `WS /ws/state`; `GET /ui/live/output/stream` (SSE); `WS /api/plugins/{model_tag}/exec` |

Use `GET /health` for a first-boot wait loop, not `/status` — it is the cheapest possible liveness answer.

`GET /api/config` deliberately redacts absolute storage paths to bare file names and never reveals where plugins run. That is intentional disclosure control, not a bug; do not file it as one. Boot-only sections are rejected with 403 and require a restart; unknown sections with 400.

`WS /ws/state` sends one `connected` event carrying a status snapshot on accept, then redacted state changes. It **never** broadcasts prompt text, response text, stderr lines, full thread ids, or addresses — so it is safe to consume for load/unload reactions without polling.

### The four calls an agent actually makes

```bash
# 1. is it up
curl -s $BASE/health

# 2. what can I call
curl -s $BASE/v1/models

# 3. talk to it
curl -s -X POST $BASE/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "my-model", "messages": [{"role":"user","content":"hi"}], "stream": false}'

# 4. what is configured
curl -s $BASE/api/config
```

`GET /v1/models` returns exactly `{"object": "list", "data": [...]}` — **there is no count field**, so if you need a number, count the array yourself. Each model object is not a raw manifest: it carries `id`, `object`, `created` (the file's mtime, best-effort, `0` on failure), `owned_by`, `context_length`, `context_length_source` (`"flag"` or `"manifest"`), and a `turbohaul` sub-object with `display_name` and a `multimodal` flag. **Treat `context_length` as a report of intent, not an enforced limit** — nothing at that layer rejects a longer request. Non-model (plugin) manifests are skipped from the listing rather than causing an error.

### Where OpenAI compatibility is partial

Five deliberate gaps, each of which will cost you a debugging cycle if you assume the spec:

1. **No `/v1/completions` and no `/api/generate`.** Use `/v1/chat/completions` or `/api/chat`.
2. **`reasoning_effort` does not accept `minimal`.** It returns 400. The accepted values are `low`, `medium`, `high`, `xhigh`, `max` — case-insensitive, surrounding whitespace ignored. This departure from the OpenAI spec is intentional and documented in the code, not an oversight.
3. **`reasoning_effort` is a launch setting, and it is silently inert twice over.** It is read **once**, when the engine is first spawned, and locked for that engine's life — the first caller wins, and a later request asking for a different effort on an already-warm engine gets the earlier value with **no error and nothing in the response to indicate it**. And it only does anything at all if that model's manifest sets a positive thinking budget; with none set, absent, `0`, or unbounded, the value is validated and then discarded, again silently. Send it on the request that actually starts the model, and treat it as a property of that load.
4. **Embeddings reject `encoding_format: "base64"`** (use `float`) and **any `dimensions` value** (the dimension is model-defined). Both are 400.
5. **`POST /api/pull` is a 501 stub**, as covered above.

Accepted on `/v1/chat/completions`: `model` and a non-empty `messages` list (both required, 400 otherwise), `stream`, `thread_id`, `session_id`, `role`, `response_format`, `max_tokens`, `keep_alive`, and — forwarded verbatim to the engine when present — `temperature`, `top_p`, `top_k`, `min_p`, `seed`, the presence/frequency/repeat penalties, `repeat_last_n`, `typical_p`, the mirostat family, `n_predict`, `thinking_budget_tokens`, `reasoning_budget`, `reasoning`, `tools`, `tool_choice`, `parallel_tool_calls`, `function_call`, `functions`. `stream_options` is streaming-only.

### Freeing memory: there is no unload endpoint, only `keep_alive`

There is **no route that stops a loaded model.** Nothing in the API unloads, evicts, or frees a slot on
demand. The only lever is the `keep_alive` field on a chat request, which sets how long the model stays
resident after that request. Its accepted values:

| `keep_alive` | Effect |
|---|---|
| omitted, or unparseable | Use the server default (`queue.idle_hot_load_seconds`, 600 in code) |
| `0` (also `"0"`, `0.0`, `false`) | **Unload immediately** after the request — this is how you free memory |
| `-1` | Pin the model — held for the maximum, not unloaded |
| a positive number of seconds | Stay resident that long, clamped to a ceiling of **1800 s** |
| `"30s"` / `"5m"` / `"2h"` | Same, in suffix form |
| `true` | Treated as "use the server default", the same as omitting it |

So to reclaim VRAM deterministically, send a small request with `"keep_alive": 0` and the model unloads
when it finishes. Note the ceiling: you cannot ask for longer than 1800 seconds, and `-1` means pinned
rather than "forever".

Two response-shape differences to know:

- `/v1/chat/completions` returns the upstream engine's body **verbatim** — no reshaping. `/api/chat` is internally forwarded as OpenAI and then re-shaped to the Ollama shape on return, with `finish_reason` mapped to `done_reason`.
- **Streaming keeps HTTP 200 even when it fails.** A mid-stream failure arrives as a final `data: {"error": {...}}` chunk followed by `data: [DONE]`. **A streaming client must inspect chunk content, not just the status code.** While a model is cold-loading you will also get `: keep-alive` comment lines — **skip any line starting with `:`**, and stop at `data: [DONE]`.

One route is rarely mentioned in the product's own prose but is registered: an Ollama-style single-model show, `GET /api/show?name=<tag>`. It is a real route, not a rumour. `GET /api/tags` returns `{"models": [...]}` with **no count field**; each entry carries `name`, `model`, `size`, `digest` (the blob hash with a `sha256:` prefix), `modified_at`, `revision`, and a `details` object. Note that `/api/tags` lists **manifests**, not raw blobs — it is not a substitute for `GET /api/blobs`.

### Errors and retries

Every error body is the same envelope: `{"detail": ...}`, where `detail` is either a plain string (most 400/404/500s) or a nested object (the typed error families on the chat routes). There is no application-level exception handler, so read `detail` rather than assuming a shape.

| Status | Meaning | Retry? |
|---|---|---|
| 400 | Missing/invalid `model` or `messages`, bad tag, bad `reasoning_effort`, bad `response_format`, validation failure | **No** — fix the request |
| 404 | `model not found: <name>` | No |
| 422 | `json_schema` validation failed (`{"error":"schema_validation_failed", …}`) | No |
| 413 | Request body over the ceiling, or a transfer over the size ceiling | No — shrink it |
| 499 | Client closed the request | No |
| 500 | `sidecar failed: …`, or `manifest for '<name>' is present but unreadable` | Depends — see below |
| 502 | Upstream engine answered with its own error | Usually **no** — read the echoed upstream status and body |
| 503 | Engine unavailable mid-response, capacity/VRAM over-commit, or no backend wired | **Yes**, honour `Retry-After` |
| 504 | Timeout | **Yes**, honour `Retry-After` |

Typed `detail` objects on the chat routes: `{"error": "client_closed_request", …}`; `{"error": {"type": "sidecar_unavailable", "cause", "message"}}`; `{"error": "sidecar_timeout", "message"}`; `{"error": "upstream_sidecar_error", "upstream_status", "upstream_body", "message"}` (the upstream body is capped at 500 characters); `{"error": {"type": "capacity_unavailable", "cause", "message", "retry_after"}}`; and nested `{"error": "<code>", "message", "received"}` for `reasoning_effort` and `response_format` rejections. A couple of 503s are flat strings instead.

**`Retry-After` appears on exactly three conditions**, and the value is computed, not fixed:

1. **503 engine unavailable** (crash or disconnect mid-response). The header is derived: `max(the exception's own default of 30s, the live model-load timeout setting)`. The value tracks a config change, and the manager deliberately never advertises a wait shorter than the one it would itself perform.
2. **504 timeout.** The exception's own value, default 60s.
3. **503 capacity / VRAM over-commit.** The exception's own value.

Two exceptions worth internalising: the 503 that reports *no completion backend wired* carries **no** `Retry-After` at all. And the embeddings route's residual-failure arm uses a **hardcoded** `Retry-After: 5` rather than the computed value — a known, recorded divergence in the product.

**Back off for the number in the header, not a number you hardcoded.** Its value is derived from live configuration — specifically the model-load timeout setting — so it changes when that setting does.

Retry rules, in order:

- **Always honour `Retry-After` on a 503 or 504.** These are genuinely retryable — the model was out of memory capacity, or the engine behind it crashed mid-response. Nothing is broken.
- **A 500 whose text starts `sidecar failed: safety gates refused spawn` is retryable.** The host was under memory or IO pressure; the text after the colon names the specific failed check.
- **A 500 reading `manifest for '<name>' is present but unreadable` is not retryable by waiting.** The configuration file on disk is corrupt. Fix the file.
- **A 502 is usually not a blind-retry case** — most often the prompt exceeded the context window. The upstream status and body are echoed in the response; shorten the input.
- **On a 5xx during streaming you already have HTTP 200.** Check the last chunk before `[DONE]`.
- Read the body text before retrying anything. These shapes differ enough that a blind retry loop will hammer a request that can never succeed.

### Request size

Chat requests are capped at **64 MiB** by default, adjustable through the `http.max_body_bytes` setting or the equivalent environment variable. Over the cap is a 413 with the same `{"detail": string}` envelope. **The cap is checked against the declared body length only** — a chunked request with no `Content-Length` falls through unenforced. Separately, embeddings accept at most 64 inputs per call, and that over-cap is *also* a 413 with a different message; read `detail` to tell them apart. An embeddings call can legitimately block a long time waiting for a slot.

## Verify it worked

Check each step; do not assume the previous one succeeded.

```bash
# 1. the bytes arrived, and it is the size you expected
curl -s $BASE/api/blobs | python3 -m json.tool

# 2. the manifest parses and the hash resolves to a real blob
curl -s $BASE/api/manifests/my-model | python3 -m json.tool
#    cross-check: gguf_blob_sha256 above must appear in the digest list from step 1

# 3. it is discoverable, and the model really answers
curl -s $BASE/v1/models | python3 -m json.tool
curl -s -X POST $BASE/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"reply with the single word: ok"}],"stream":false}'

# 4. a spawn refusal names the check that failed
#    read the detail text of any 500/503 before retrying
```

If step 3 returns 404, the manifest is not there under that exact tag. If it returns 500 `manifest … present but unreadable`, the file is corrupt — fix it, do not retry. If the model is missing from `/v1/models` but answers a chat request, check the `hidden` flag on `GET /api/manifests`; hidden models are omitted from the discovery listings by design while staying fully servable. And if a chat request returns 503 with a capacity message, your `expected_vram_bytes` and `gguf_size_bytes` are the two numbers to fix before you retry anything.
