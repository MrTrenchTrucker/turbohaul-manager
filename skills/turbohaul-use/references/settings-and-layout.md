# Part 1 — The settings surface and where everything lives

## The config file

One YAML file, `turbohaul.yaml`, read from `/etc/turbohaul/turbohaul.yaml`. A container image bakes
that file into place and points the config-path variable at the same location.

Point the manager at a different file two ways, either of which beats the baked-in default:

```
turbohaul-manager --config /path/to/turbohaul.yaml
TURBOHAUL_CONFIG_PATH=/path/to/turbohaul.yaml
```

The file must exist at startup. A missing file is fatal — the manager stops and says so rather than
starting on defaults.

**The file is strict.** The root must be a mapping, and every section and every key inside it must be
a known name. A misspelled section, a misspelled key, a misspelled top-level word — any of these is a
hard validation failure at startup, not a warning. If startup fails complaining about a setting you
have never heard of, look for a typo in a section or key name first.

Cross-field rules beyond the per-key ones:

- A wildcard listen address is rejected in the file unless the separate public-bind opt-in is also set
  — and that opt-in is a start-up switch, so the file alone can never ask for a public bind.
- Every plugin registry key must match `^[a-z0-9][a-z0-9_-]{0,63}$`.
- Each fast-lane priority rule must name its client by network address **or** by container name —
  exactly one, never both and never neither.
- A fast-lane rule list is capped at 10 entries.

One deliberate exception to strictness: a partly-bad or over-length `fastlane` section is coerced at
boot rather than refused. Everything else is refused.

### Start-up switches

These are command-line switches, not settings keys. Do not put them in a config file.

| Switch | Default | Purpose |
|---|---|---|
| `--config` | `/etc/turbohaul/turbohaul.yaml`, or `TURBOHAUL_CONFIG_PATH` | Which config file to load. |
| `--allow-public-bind` | set when `TURBOHAUL_ALLOW_PUBLIC_BIND=1` | The ONLY thing that widens the bind. The `server.allow_public_bind` file key is never read. |
| `--log-level` | `TURBOHAUL_LOG_LEVEL`, else `info` | One of `critical`, `error`, `warning`, `info`, `debug`, `trace`. |

Exposing the manager beyond the host needs `--allow-public-bind` on the command line, or
`TURBOHAUL_ALLOW_PUBLIC_BIND=1` in the environment — either one alone suffices. The
`server.allow_public_bind` key in the config file exists but is **never read at bind time**: setting it
does nothing, and it is not a second switch you must also throw. Do not rely on the file key to expose
or to protect the service.

## Precedence: four layers, highest wins

```
code default  <  your YAML file  <  environment variable  <  a change made through the settings API
```

The API layer is not in memory only. Every successful `PUT /api/config` is also written to
**`runtime_config.yaml`**, sitting next to the state database, and that file is merged **last** at
every boot. So an API change keeps winning:

- after a restart,
- after you edit the file you bind-mounted, and
- after you change the matching environment variable.

That is the usual explanation for "my config file says one thing and the manager does another".
Before debugging anything else, check whether `runtime_config.yaml` exists next to the state
database.

The persisted file records only the sections and fields a request actually touched, so a partial
edit never blanks its neighbours, and it is written temp-file-then-rename, so it is never half
written. Boot sections can never appear in it — a file carrying a boot section is ignored for that
section.

Ask the running manager who won, per setting:

```
GET /api/config
```

The reply carries `_provenance`, which names the layer that supplied each effective value
(`default` / `yaml` / `env` / `persisted`) and is recomputed on every call. It also carries
`_provenance_stamp`, the UTC time of the most recent API write that touched each field. The stamp is
forward-only: it can tell you when a field was last written, and nothing about older history.

## Boot-only versus runtime-mutable

Five of the thirteen sections are **boot-only**. They are read once, frozen, and changing one needs a
restart. Eight are **runtime-mutable** and can be changed while the manager keeps serving.

| Class | Sections |
|---|---|
| Boot-only (restart to change) | `server`, `storage`, `runtime`, `ui`, `plugins` |
| Runtime-mutable (changeable live) | `queue`, `pull`, `persist`, `monitor`, `kv`, `http`, `fastlane`, `plugin_runtime` |

What each governs:

- **Boot-only** — `server`: listen address and port, and permission to be reachable off-host.
  `storage`: the four on-disk locations. `runtime`: the inference-engine binary path, its checksum
  pin, and the port range handed to per-model workers. `ui`: whether the bundled web interface is
  served and where its files are. `plugins`: the registry of external-service endpoints — the only
  place in the whole system where a plugin's address may live, so that a model's own config never
  carries a URL.
- **Runtime-mutable** — `queue`: how much runs at once, fair ordering, warm-worker reuse, the
  resource gates checked before a launch, and the timers. `pull`: download endpoints and their
  safety limits. `persist`: the byte ceiling on the disk-tier saved-state cache. `monitor`: the
  live progress monitor's on/off switch and poll cadence. `kv`: saved-conversation cache behaviour
  plus the RAM and disk tier ceilings. `http`: the request-body size ceiling. `fastlane`: the
  two-level request-priority feature. `plugin_runtime`: per-plugin operational knobs.

Decide this **before** you change anything — it decides whether the edit takes effect now or after a
restart. A `PUT` that names a boot section is refused with HTTP 403 and the offending section names
in the message. Change those in the file and restart.

On a successful write the manager shallow-merges each named section, re-validates the whole runtime
object, swaps the live object atomically, refreshes the grace and idle timer objects, and invalidates
the fast-lane cache. The new values apply without a restart.

## Environment variables

The prefix is `TURBOHAUL_`, but there is **no mechanical mapping** from a variable name to a config
path. The mapping is a hand-maintained table of 15 entries, and the names do not mirror the file's
section.field structure. An environment variable that is not in that table is invisible to the
configuration — it is not a warning, it is simply ignored. Do not invent a name.

**The 15 variables that override a setting:**

| Environment variable | Overrides |
|---|---|
| `TURBOHAUL_HOST` | `server.host` |
| `TURBOHAUL_PORT` | `server.port` |
| `TURBOHAUL_MAX_PARALLEL` | `queue.max_parallel_sidecars` |
| `TURBOHAUL_STAGING_DEPTH` | `queue.staging_queue_depth` |
| `TURBOHAUL_ACCEPT_MAX` | `queue.acceptance_buffer_max` |
| `TURBOHAUL_GRACE_S` | `queue.grace_seconds` |
| `TURBOHAUL_IDLE_HOT_S` | `queue.idle_hot_load_seconds` |
| `TURBOHAUL_MAX_GRACE_EXT` | `queue.max_grace_extensions` |
| `TURBOHAUL_PREFIX_TOKEN_COUNT` | `queue.prefix_token_count` |
| `TURBOHAUL_GRACE_IP_SHARED_ADDRESSES` | `queue.grace_ip_match_shared_addresses` |
| `TURBOHAUL_MAX_CONSECUTIVE_SAME_MODEL` | `queue.max_consecutive_same_model` |
| `TURBOHAUL_MAX_OTHER_MODEL_WAIT_S` | `queue.max_other_model_wait_s` |
| `TURBOHAUL_KVCACHE_PERSIST_MAX_BYTES` | `persist.max_bytes` |
| `TURBOHAUL_KVCACHE_MAX_BYTES` | `kv.ram_cache_max_bytes` |
| `TURBOHAUL_MAX_BODY_BYTES` | `http.max_body_bytes` |

No other setting responds to an environment variable.

**Abbreviations that will bite you.** The names are prefix-plus-abbreviation, and several do not
follow from the key they set. `TURBOHAUL_MAX_PARALLEL` sets `max_parallel_sidecars`, not a key
called `max_parallel`. `TURBOHAUL_ACCEPT_MAX` sets `acceptance_buffer_max`.
`TURBOHAUL_KVCACHE_PERSIST_MAX_BYTES` sets `persist.max_bytes` — the section is `persist`, not
`kv` — while `TURBOHAUL_KVCACHE_MAX_BYTES` sets `kv.ram_cache_max_bytes`. `TURBOHAUL_HOST` sets
`server.host`, not a key called `bind`. Use the table; do not derive a name from the key.

Three more variables are read outside the table and are not settings keys: `TURBOHAUL_CONFIG_PATH`,
`TURBOHAUL_LOG_LEVEL`, `TURBOHAUL_ALLOW_PUBLIC_BIND`. A fourth, `TURBOHAUL_DIVERGENCE_DEBUG`,
enables the opt-in debug reports described under the disk layout below.

One setting stores the **name** of an environment variable rather than a value:
`pull.hf_api_key_env` names the variable that holds the download API token (default `HF_API_KEY`).
The token never goes in the config file.

Roughly two dozen other `TURBOHAUL_*` variables exist in the program as internal feature flags.
They have no config section, are not in the table, and never appear in `GET /api/config`. Do not
document them as settings.

## The settings API

Three endpoints on the same prefix.

### `GET /api/config` — the effective configuration

Read live, so it reflects API writes. Returns all thirteen sections plus `_provenance` and
`_provenance_stamp`.

It redacts on purpose, so do not expect a round trip:

- Paths come back as **file names only** — a basename, not a full path. Publishing absolute paths
  would hand a rebind-pivoting caller the exact write targets on disk.
- `plugins.registry` comes back as the **sorted list of registry key names**, with every endpoint
  field withheld. The registry holds addresses of internal containers, which is exactly the network
  topology an attacker wants.

To read a real path back, go to the status surface or the filesystem, not to this endpoint.

### `PUT /api/config` — write the mutable settings

Body shape: one top-level key per section, an object value, shallow field merge.

```
PUT /api/config
{"queue": {"grace_seconds": 60}, "monitor": {"poll_interval_s": 2.0}}
```

Error behaviour:

| Condition | Status | Body |
|---|---|---|
| A boot-only section is named | 403 | Names the refused sections and says a restart is required |
| An unknown section is named | 400 | Names the unknown sections |
| Any value fails validation | 400 | The validation error text |
| Missing file named by `auth_token_file` | — | Refused at boot, naming the path only |

The 403 on boot sections is deliberate: it is what stops an HTTP caller from repointing the
inference binary. Change those in the file and restart.

### `GET /api/config/schema` — types, defaults, bounds

Per field: `type`, `default`, `minimum`, `maximum`, for every runtime-mutable section. A client uses
it to render typed inputs and offer a reset-to-default. Defaults are read by instantiating the model
rather than from the JSON schema, so list-valued defaults come back correct instead of null.

**Prefer this over hard-coding defaults in your own tooling.** Ask the running manager.

### Verify a write landed

```
GET /api/config
```

Read back the field, then check its `_provenance` entry — it should say `persisted`. Then check the
file next to the state database:

```
cat /var/lib/turbohaul/runtime_config.yaml
```

Only the fields in your request should appear.

## Settings reference

62 top-level keys across 13 sections, plus 13 keys in nested per-entry objects.

**Reading the default column.** Most keys have the same default in the program and in the shipped
config file. Four do not, and for those both are shown: the value a fresh install actually gets is
marked **(shipped)**, and the program's own fallback is marked **(code)**. Quote the shipped number
for "what happens out of the box" and the code number for "what the fallback is".

Six keys have **no default at all** and must be present in every config file: the four storage
paths, the engine binary path, and the UI static path. A file missing any of them will not load.

### `server` — boot-only

| Key | Type | Default | What it does |
|---|---|---|---|
| `host` | str | `"127.0.0.1"` | Interface the manager listens on. `0.0.0.0` is rejected from the file. |
| `port` | int | `11401` (1–65535) | TCP port to listen on. |
| `allow_public_bind` | bool | `false` | **Never read at bind time.** The start switch is the `--allow-public-bind` flag or `TURBOHAUL_ALLOW_PUBLIC_BIND=1`; this file key does nothing. |

### `storage` — boot-only, all four mandatory

| Key | Type | Default | What it does |
|---|---|---|---|
| `blob_store_path` | Path | **required** | Directory holding downloaded model files. |
| `manifests_path` | Path | **required** | Directory holding per-model config files. |
| `import_allowed_root` | Path | **required** | The only directory a file may be imported from. |
| `state_db_path` | Path | **required** | SQLite file holding queue and slot state. |

Shipped: `/var/lib/turbohaul/blobs`, `/var/lib/turbohaul/manifests`,
`/var/lib/turbohaul/import-staging`, `/var/lib/turbohaul/state.sqlite`.

### `runtime` — boot-only

| Key | Type | Default | What it does |
|---|---|---|---|
| `llama_server_binary` | Path | **required** | Path to the inference engine executable launched per model. |
| `llama_server_binary_sha256` | str | `""` (empty) | Expected checksum. **Empty skips verification** — dev only; pin it in production. |
| `default_port_base` | int | `11500` (1024–65000) | First port of the range given to per-model workers. |

Shipped: `/opt/turbohaul/bin/llama-server`, empty checksum.

### `ui` — boot-only

| Key | Type | Default | What it does |
|---|---|---|---|
| `enabled` | bool | `true` | Whether the built-in web interface is served. |
| `static_path` | Path | **required** | Directory holding the web interface files. |

Shipped: `/opt/turbohaul/ui_dist`.

### `queue` — runtime-mutable, 26 keys

*How much runs at once*

| Key | Type | Default | What it does |
|---|---|---|---|
| `max_parallel_sidecars` | int | `1` (1–32) | Model worker processes allowed at once. |
| `staging_queue_depth` | int | `100` (1–10000) | Queue depth before new requests are refused. |
| `acceptance_buffer_max` | int | `10000` (≥1) | Total accepted-but-unfinished requests. |

*Fair ordering*

| Key | Type | Default | What it does |
|---|---|---|---|
| `max_consecutive_same_model` | int | `3` (1–1000) | Back-to-back requests for one model before switching. `1` disables batching. |
| `max_other_model_wait_s` | float | `20.0` (0.0–3600.0) | Starvation window before a waiting different-model request jumps the queue. `0.0` is strict FIFO. |
| `main_lane_reserved` | bool | `true` | Reserve priority for interactive work. |
| `main_lane_identity_keys` | list[str] | `["is_main"]` | Request labels that count as interactive. |

*Warm-worker reuse and stay-loaded time*

| Key | Type | Default | What it does |
|---|---|---|---|
| `grace_seconds` | int | `30` (0–3600) | How long a finished slot stays reusable by a follow-up. |
| `max_grace_extensions` | int | `5` (0–1000) | How many times that window may be extended. |
| `prefix_token_count` | int | `256` (1–8192) | Prompt tokens hashed to auto-identify a conversation. |
| `idle_hot_load_seconds` | int | **120 (shipped)** / 600 (code) (0–86400) | Slot stay-warm window while a model sits idle. |
| `grace_ip_match_shared_addresses` | str | `""` (comma-separated) | Addresses known to be shared by several clients, excluded from IP-based grace matching. |

The shared-address list is **fail-open by construction**: an address that is not listed is matched
silently. Empty means "nobody has listed one", never "there are none".

*Resource gates, checked before launching a new worker*

| Key | Type | Default | What it does |
|---|---|---|---|
| `safety_enabled` | bool | `true` | Master switch for the pre-launch checks. |
| `safety_min_free_ram_mib` | int | `1024` (≥0 MiB) | Refuse to launch below this much free system RAM. |
| `safety_min_free_vram_mib` | int | `512` (≥0 MiB) | Same, for graphics memory. |
| `safety_max_cpu_busy_percent` | float | `90.0` (0.0–100.0) | Refuse to launch above this percent CPU actually busy. **Percent of host CPU, not load per core** — do not reuse a per-core number here. |
| `safety_cpu_util_sample_window_s` | float | `0.4` (0.05–5.0) | Sampling window for the CPU reading. |
| `safety_max_iowait_percent` | float | `30.0` (0.0–100.0) | Refuse to launch above this percent disk wait. |
| `safety_iowait_sample_window_s` | float | `0.4` (0.05–5.0) | Sampling window for the disk-wait reading. |

**No `safety_*` key appears in the shipped file at all** — not this one, and not `safety_enabled`, `safety_min_free_ram_mib`, `safety_min_free_vram_mib`, `safety_cpu_util_sample_window_s`, `safety_max_iowait_percent` or `safety_iowait_sample_window_s`. The values below are the code defaults, which is what a fresh install actually runs. `queue.safety_max_cpu_busy_percent` specifically is absent, so a fresh install gets the
code value of 90.0.

*Timeouts*

| Key | Type | Default | What it does |
|---|---|---|---|
| `loading_health_timeout_s` | int | **120 (shipped)** / 60 (code) (10–7200) | How long a fresh worker has to report healthy before the load counts as failed. The shipped file deliberately raises this above the program default so a stall warning does not fire on every healthy large-model load. |
| `spawn_reclaim_wait_max_s` | float | `10.0` (0.0–60.0) | Wait for an in-flight memory reclaim before a final refusal. `0.0` means no waiting. |
| `sidecar_complete_timeout_s` | float | `3600.0` (1.0–86400.0) | Timeout for a whole non-streaming request. Matches the streaming timeout. |
| `drained_sigterm_window_active_s` | int | `15` (1–300) | Graceful-stop window for an active worker. |
| `drained_sigterm_window_cold_s` | int | `5` (1–300) | Graceful-stop window for an idle worker. |

*Background cleanup*

| Key | Type | Default | What it does |
|---|---|---|---|
| `background_sweep_interval_s` | int | `60` (1–86400) | How often a sweeper finalises leftover records. |
| `background_sweep_min_age_s` | int | `86400` (60–2592000, 24h) | Minimum record age before sweeping, so live work is never reaped. |

### `pull` — runtime-mutable

| Key | Type | Default | What it does |
|---|---|---|---|
| `hf_api_key_env` | str | `"HF_API_KEY"` | **Name** of the variable holding the download token. The token itself is never in this file. |
| `hf_host_allowlist` | list[str] | **2 hosts (shipped)** / 5 (code) | Only these hosts may be downloaded from. A security boundary. |
| `pull_url_https_only` | bool | `true` | Refuse plain-HTTP downloads. |
| `pull_concurrency` | int | `2` (1–16) | Parallel download streams. |
| `pull_chunk_size_mb` | int | `64` (1–1024) | Download chunk size in MiB. |
| `per_stream_max_bytes` | int | `107374182400` (100 GiB, ≥1) | Hard cap per download. |

The shipped allowlist is **narrower** than the program fallback, so out of the box a download from a
host the shipped list omits is refused.

### `persist` — runtime-mutable

| Key | Type | Default | What it does |
|---|---|---|---|
| `max_bytes` | int | `42949672960` (40 GiB, ≥0) | Byte ceiling for the on-disk saved-state cache. `0` removes the cap. |

### `monitor` — runtime-mutable

| Key | Type | Default | What it does |
|---|---|---|---|
| `enabled` | bool | `true` | Operator kill-switch for the live progress monitor. |
| `poll_interval_s` | float | `1.0` (>0.0–60.0) | Cadence for the slot-status feed. One poller regardless of client count. |

There are only two keys. The rest of the monitor's tuning — smoothing, stall thresholds, text-tail
size — is hardcoded in the program and is deliberately not configurable. Do not go looking for it.

### `kv` — runtime-mutable

| Key | Type | Default | What it does |
|---|---|---|---|
| `covered_scaffold_strip` | bool | `true` | Strip internal reasoning scaffolding from saved history. |
| `ram_cache_max_bytes` | int | `21474836480` (20 GiB, ≥0) | Byte ceiling for the fast tier. `<= 0` disables the ceiling. |
| `ram_cache_max_age_hours` | float | `6.0` (≥0) | Age ceiling for fast-tier entries, in hours. |
| `kv_save_expected_fstype` | str \| None | `"tmpfs"` | Assert the fast tier is on this filesystem type. `None` disables the check. |
| `kv_persist_forbidden_fstypes` | list[str] | `["tmpfs", "ramfs"]` | The inverse: cold storage must **not** be on a volatile type. |

The two filesystem keys are deliberate opposites — the fast tier must be RAM-backed, the durable tier
must not be. At boot the manager complains loudly if either assertion fails, and keeps serving. A
fast tier that is not actually in RAM will quietly wear the disk instead. Leave the checks on.

These bound the **size and age** of the cache, never its **location**.

### `http` — runtime-mutable

| Key | Type | Default | What it does |
|---|---|---|---|
| `max_body_bytes` | int | `67108864` (64 MiB, ≥1, **no upper bound**) | Largest accepted request body. Sized for generic multimodal input, not media ingestion. |

No ceiling is set on purpose, so an operator can raise it for very large multimodal input without the
field itself blocking them.

### `fastlane` — runtime-mutable, off by default

| Key | Type | Default | What it does |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch for two-level request priority. **Off out of the box.** |
| `rules` | list[rule] | `[]` | Per-client priority rules. **List order is the priority** — earlier wins. Hard cap **10**. |
| `max_normal_wait_s` | float | `3600.0` (1.0–3600.0) | How long an unmatched client waits before priority traffic may bump it. |
| `cross_model_switches_per_min` | int | `2` (0–3) | Per-minute model switches for **unregistered** clients. Priority-matched traffic is exempt from this budget entirely, at any value including `0`. |
| `census_ttl_hours` | int | `168` (1–8760, 7 days) | How long an observed address is remembered as a candidate client. |

`fastlane.rules[].*`:

| Sub-key | Type | Default | What it does |
|---|---|---|---|
| `address` | str \| None | `None` | A single client IP. Exactly one IP; CIDR ranges and hostnames are rejected. |
| `container_name` | str \| None | `None` | Identify the client by container name instead, resolved at runtime. |
| `label` | str | `""` | The operator's own note. |
| `tag_ranks` | object | all `None` | Optional per-workload-type priority ranks, 1 (served first) to 5, or unset. |

`address` and `container_name` are mutually exclusive and exactly one is required. **Use
`container_name` for anything on the container network**: an `address` rule silently drifts to the
wrong client on every container restart, because that network has no static address allocation.

`tag_ranks` has five sub-keys, one per workload type. Describe the types in plain words; the
on-disk spelling you must type is:

| On-disk key | Workload type |
|---|---|
| `main` | the primary interactive conversation |
| `curator` | background housekeeping and curation |
| `compression` | background summarisation and compaction |
| `sub_agent` | delegated helper agents |
| `unclassified` | anything not matched to one of the above |

Write the section as `fastlane`. An old top-level section spelled `fastline` is silently renamed when
the **file** is read, and discarded if both spellings are present — that tolerance is for old saved
files only. Over the API the section is `fastlane` and only `fastlane`.

### `plugins` — boot-only

| Key | Type | Default | What it does |
|---|---|---|---|
| `registry` | dict[name, endpoint] | `{}` | Named external-service endpoints. **The only place in the system a plugin address may live** — a model's own config carries a name from here, never a URL. |

`plugins.registry.<name>.*`:

| Sub-key | Type | Default | What it does |
|---|---|---|---|
| `host` | str | **required** | Bare hostname or IP. Must **not** be a URL, and shorthand IP spellings are rejected. |
| `port` | int | **required** (1–65535) | Endpoint port. |
| `health_path` | str | `"/health"` | Liveness path. Must start with `/` and contain no `..`. |
| `auth_token_file` | Path \| None | `None` | **Path to** a file holding a bearer credential — never the credential itself. Read fresh on every call. Absent means no auth header is sent. |

A missing, unreadable, or empty `auth_token_file` aborts startup and the message names the **path
only**, so a config mistake surfaces at boot instead of as a baffling 401 later. Treat that path as
a secret location.

### `plugin_runtime` — runtime-mutable

| Key | Type | Default | What it does |
|---|---|---|---|
| `enabled` | dict[str, bool] | `{}` | Per-model on/off switches. **Keyed by the model's own name, not by registry entry** — deliberately many-to-one, so disabling one model does not silently disable another sharing the same endpoint. |
| `max_concurrent` | int | `4` (1–64) | Plugin calls allowed at once. |
| `no_progress_timeout_s` | float | `600.0` (0.0–86400.0) | Seconds without progress before a plugin call is abandoned. |

This is the section the web interface's plugins tab edits.

### Which sections the shipped file actually contains

`server`, `storage`, `runtime`, `ui`, `queue`, `pull`, `persist`, `kv`. The other five —
`monitor`, `http`, `fastlane`, `plugins`, `plugin_runtime` — are absent from it and fall back to the
program defaults.

### Verify your edit was actually honoured

At every boot, the manager logs one line for **every** key whose fully-resolved value differs from
what a fresh install would get:

```
config diverges from shipped default: <section>.<key> effective=<value> shipped_default=<value>
```

That line is the fastest way to confirm an edit took, and to see a remembered API change overriding
your file. If a setting is not behaving as you expect and this line is silent, suspect the value was
never changed rather than never applied.

## Where everything lives

A running manager keeps its mutable state in exactly **two roots**.

**`/var/lib/turbohaul/` — the data volume.** Almost everything lives here. This is the one path to
back up, bind-mount, or persist as a container volume. Back it up, or you lose downloaded models,
per-model configs, queue state, and the saved fast-start cache.

| Path | What lives there | Moved by |
|---|---|---|
| `blobs/` | Model file store | `storage.blob_store_path` — config file only |
| `blobs/sha256/incoming/` | Downloads in progress | follows the blob root |
| `blobs/sha256/<2 chars>/<full fingerprint>` | A finished model file | follows the blob root |
| `manifests/<model-tag>.yaml` | One settings file per model | `storage.manifests_path` — config file only |
| `model_meta.json` | Display names and descriptions | follows the manifests path |
| `import-staging/` | Drop-box for local-file import | `storage.import_allowed_root` — config file only |
| `state.sqlite` (+ `-wal`, `-shm`) | Queue snapshot, slot history, audit events, live slot registry | `storage.state_db_path` — config file only |
| `runtime_config.yaml` | Settings remembered from API writes | follows the state database |
| `kvcache/` | Fast cache tier — per-turn saves | **not movable** |
| `kvcache_persist/` | Durable cache tier — unload-time archive | **not movable** |
| `kv_quarantine/` | Cache files moved aside after repeat failures | **not movable** |
| `engine_logs/` | One detailed log per model load | **not movable** |
| `telemetry/` | Degradation reports | **not movable** |
| `/etc/turbohaul/turbohaul.yaml` | The config file | `--config` or `TURBOHAUL_CONFIG_PATH` |

**`/var/log/turbohaul/` — a second, separate root**, used only by an off-by-default divergence-debug
facility. Different root, so it is **not** covered by the data volume and will not be backed up with
it.

### The two surprises

**1. The four storage paths have no environment variable.** Not one. There is no `TURBOHAUL_*`
variable that moves the blob store, the manifests directory, the import sandbox, or the state
database. They are boot-only *and* config-file-only. This is the single most common thing a user
tries and fails to do. Edit the file and restart.

**2. The two cache directories cannot be moved at all.** `kvcache/` and `kvcache_persist/` are
fixed strings in the program. No config key, no environment variable, no start-up switch. To put
them somewhere else you must mount over `/var/lib/turbohaul` as a whole. What you *can* tune is size
and age — `persist.max_bytes` (40 GiB, or `TURBOHAUL_KVCACHE_PERSIST_MAX_BYTES`),
`kv.ram_cache_max_bytes` (20 GiB, or `TURBOHAUL_KVCACHE_MAX_BYTES`), and
`kv.ram_cache_max_age_hours` (6 hours, config file only). Setting either byte cap to zero disables
that cap.

The same applies to `kv_quarantine/`, `engine_logs/`, and `telemetry/`: no config key and no
environment variable moves them. Only the whole-root mount does.

### How the two cache tiers work

Per turn, the cache is written to the fast tier only, so the disk takes no wear. When a model is
unloaded, the cache is copied to the durable tier. At startup the fast tier is refilled **from the
durable tier**. So only the durable tier survives a restart: give the fast tier real disk and expect
it to be rebuilt, not preserved.

### Files on disk are not named after models

Model files are content-addressed. The final file is named with the full fingerprint and filed under
a directory named by the first two characters of that same fingerprint. There is no model name in
the path. Downloads land in `blobs/sha256/incoming/` and are moved into place only after the
download verifies. The directory layout is created on demand.

Per-model settings are plain files, one per model tag, at `manifests/<model-tag>.yaml`. Nothing about
a model lives in a database. `model_meta.json`, a sibling of the manifests directory, holds the
human-readable names and descriptions.

### Back up the whole state database, not just the file

`state.sqlite` runs in write-ahead-log mode, so recent changes are still in `state.sqlite-wal` and
`state.sqlite-shm` beside it. Copying `state.sqlite` on its own can lose the most recent activity.
The database file is created readable only by its owner. It does **not** stop a second manager from
starting on the same data directory: a lock routine for that exists in the code, but it is not called
from the startup path, so two managers pointed at the same state directory will both start. Run one
manager per data directory by convention, not by protection.

### Importing a model you already have

Copy the file into `import-staging/` first. Nothing in the import path creates that directory for
you — make it yourself. The manager refuses any path outside it, including via a symlink escape.

### Logs

**The manager writes no log file of its own.** It logs to stdout and stderr only. Set verbosity with
`--log-level` or `TURBOHAUL_LOG_LEVEL`; `trace` maps to the internal debug level. To keep logs,
redirect the container's output through your normal logging driver.

Separately, each time a model is loaded, the engine underneath it is told to write its own detailed
file into `engine_logs/`, named after the model, its port, and the time
(`engine_<model>_p<port>_<timestamp>.log`). The manager's own load and verify records go to the
manager's stdout stream, not to a separate file. Engine stdout and stderr are deliberately discarded.

### Debug sentinels and the opt-in debug root

Setting `TURBOHAUL_DIVERGENCE_DEBUG=1` makes the manager write verbose difference reports into
`/var/log/turbohaul/`. Leave it off in production. One of those two files is size-capped and rolled
over to a single `.1` backup; the other can grow without limit.

Three sentinel files are read by the manager and created by nobody — put an empty one directly in
`/var/lib/turbohaul/` and the behaviour changes:

| Sentinel | Effect |
|---|---|
| `.ram_reanchor_off` | Disables the RAM re-anchor behaviour |
| `.warm_force_clean_off` | Disables the forced clean warm path |
| `.divergence_debug` | Enables the divergence capture path |

Treat these as a debugging escape hatch, not supported configuration.

### State that grows without bound

- **`kv_quarantine/`** — when the same saved cache misbehaves three times in a row, the manager moves
  its files out of both tiers into this directory rather than deleting them, so the next load starts
  fresh, and the evidence is kept for inspection. There is no automatic cleanup. If a user reports
  slow cold loads and disk usage creeping up, look here first.
- **`engine_logs/`** — one file per model load, and no rotation or cleanup was found. The file count
  grows with the number of loads. Prune it yourself.
- **`telemetry/`** — the degradation reports *do* have retention pruning of old files, so this one
  is bounded.

Verify before relying on the growth claims:

```
du -sh /var/lib/turbohaul/kv_quarantine /var/lib/turbohaul/engine_logs
ls -1 /var/lib/turbohaul/engine_logs | wc -l
```

A related check that is **not** a setting: the maximum honoured client keep-alive value is a program
constant, not a key. Neither is the monitor's smoothing, stall, and text-tail tuning. If a control
you want is not in the reference above, it is one of those — do not go looking for a key to set.
