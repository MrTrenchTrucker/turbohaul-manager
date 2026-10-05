---
name: turbohaul-setup-and-launch
description: Use when setting up, installing, launching, or verifying a first boot of TurboHaul Manager. Build the image, pass the config gates, start it, confirm /health.
---

# Set up and launch TurboHaul Manager

## What it is

TurboHaul Manager is a control plane for local inference: it stores models, starts one engine
process per loaded model, and serves an OpenAI-style API, an Ollama-shaped API, and a web UI —
all from the same process on the same port. It ships as a Python package inside a Docker image.

Once it is up and answering, the companion skill `turbohaul-use` covers operating it: changing
settings, getting a model into the blob store, writing the per-model manifest, and calling the API.

## Prerequisites

Required:

- A container runtime (Docker) able to build from source.
- **The NVIDIA container toolkit on the host.** Both launch paths below reserve a GPU — `--gpus all`
  on `docker run`, and a hard device reservation in the compose file. Without the toolkit the
  **container will not start at all**, so you never reach the GPU probe described below. Install the
  toolkit before your first run.
- GPU access declared at run time: `--gpus all` on `docker run`, or an equivalent device
  reservation in a compose file. Both images need an NVIDIA base image at build time.
- `curl` and `jq` on the machine you drive it from — every example in these skills uses them.

Not required at boot — these are advisory, logged, and ignored, so do not chase them:

- **A GPU or a driver.** The probe shells out to `nvidia-smi`; if it is missing you get
  `nvidia-smi unavailable; skipping GPU compute-apps scan` and the manager carries on. A GPU only
  becomes mandatory when a request needs a model loaded.
- **A working engine binary.** The shipped config leaves the expected checksum blank, which means
  "skip verification". A missing or wrong binary is therefore *not* reported at boot; it surfaces on
  the first model load. If you pin a checksum and it does not match, that is logged as an error and
  the manager still serves.
- **Any models.** Zero models is a valid boot. `/health` answers happily and every inference request
  fails until you add one.
- **Free disk space.** Nothing checks it. Only the state database's parent directory is created for
  you; the blob store, manifests, and telemetry directories appear on first use.

Documented hardware guidance is sizing advice, not a code-enforced minimum: an RTX-class card with
8 GB is the stated floor, 24 GB is recommended, and CPU-only operation is supported but recommended
only for embedding models.

Run exactly **one manager per data directory**. The code contains a lock intended to enforce that,
but at this release it is not wired into the startup path — two managers pointed at the same
`/var/lib/turbohaul` will both start.

## Build the image

There is nothing to install first: clone, build one image, run that image. That is the whole install.

```
git clone https://github.com/MrTrenchTrucker/turbohaul-manager.git
cd turbohaul-manager
docker build -f Dockerfile.engine-src -t turbohaul-manager:local .
```

- The first build compiles the inference engine from source inside the image and takes a while.
  It is fully offline: the engine is vendored, Python comes from vendored wheels, and the web bundle
  is a committed build artifact. No PyPI, no npm, no external clone.
- The build runs two guard scripts that fail it on a mismatched or mixed engine, so a passing build
  cannot produce a corrupt image.
- If your GPU is outside the architecture set the default build targets, build the wider one instead
  — same image, broader NVIDIA support (the product README describes it as covering Turing through
  Blackwell):
  ```
  docker build -f Dockerfile.cuda-multi -t turbohaul-manager:local .
  ```
- Do **not** build the bare `Dockerfile`. It is a slim management-plane image with no engine binary
  in it; its header points at a runtime mount of the engine binary that nothing in the shipped
  compose file actually provides. Booting it alone gives you an API with no engine behind it.
- **Build from source.** This repository contains no automation that builds or publishes a container
  image, so the image tag quoted in the quick-start examples is not produced by anything you can see
  here. If that image is not available to you, building from source as above is the supported path and
  is the one the rest of this skill assumes.

## The config file — the two hard boot gates

The config file must exist at `/etc/turbohaul/turbohaul.yaml`. A usable default is baked into the
image at build time, so the container works with no setup. Point somewhere else with the `--config`
flag or `TURBOHAUL_CONFIG_PATH`.

**Gate 1 — the file must exist.** It is never generated for you. If it is missing the manager logs
`config not found: <path>` and exits with status 2 without starting.

**Gate 2 — the file must contain no key the running version does not know.** The top-level settings
model and every section inside it reject unknown keys. A typo, a key from a newer release, or a
section placed at the wrong level fails validation and **the manager refuses to boot**; the error
names the offending section and key. Fix it or delete it. The one exception is the optional
request-priority section: if every complaint is inside that one section, the manager repairs or
disables it and keeps serving (a build without that salvage treats the section as fatal, and note
that turning the feature off does not help — validation still runs).

Also fatal, and easy to hit when hand-writing a file:

- **A root that is not a key/value mapping.** A top-level list or an empty file yields
  `config root must be mapping, got <type>`. Rewrite it as a mapping.
- **Missing required paths.** Six fields have no defaults: the four storage locations
  (`blob_store_path`, `manifests_path`, `import_allowed_root`, `state_db_path`), the engine binary
  path (`llama_server_binary`), and the UI bundle path (`ui.static_path`). Omit any one and boot
  fails.
- **A declared plugin credential file that is missing, unreadable, or empty.** This check is
  deliberately fatal. Put only the path in the config, never the credential.
- **`server.host: 0.0.0.0` in the file.** Rejected on purpose. The `allow_public_bind` key in the
  file looks like it should work and does not — only the `--allow-public-bind` flag or
  `TURBOHAUL_ALLOW_PUBLIC_BIND=1` widens the bind. The manager has no authentication of its own, so
  the bind address is the only boundary; keep it on loopback unless you have a network policy.

### Overriding the shipped config

Settings precedence, highest first: **remembered runtime settings** (`runtime_config.yaml`, written
by settings-UI/API edits) → environment variable → the config file → built-in defaults. The full
ladder is in the `turbohaul-use` skill's settings reference; it is stated once there so it cannot
drift. Note the order: a setting changed through the API or the Settings UI outranks an environment
variable, because the persisted layer is applied last at boot.

The compose file ships the override mount **commented out**. If you supply your own config with
compose, uncomment this line yourself or your file is ignored and the manager quietly runs on the
baked-in defaults:

```
# - ./turbohaul.yaml:/etc/turbohaul/turbohaul.yaml:ro
```

There is **no environment variable that moves the storage paths** — not the blob store, not the
manifests directory, not the import sandbox, not the state database. Those change only in the YAML
file. The environment variables that do exist include `TURBOHAUL_CONFIG_PATH`,
`TURBOHAUL_LOG_LEVEL`, `TURBOHAUL_ALLOW_PUBLIC_BIND`, `TURBOHAUL_HOST`, `TURBOHAUL_PORT`,
`TURBOHAUL_MAX_PARALLEL`, `TURBOHAUL_GRACE_S`, `TURBOHAUL_IDLE_HOT_S`,
`TURBOHAUL_KVCACHE_MAX_BYTES`, and `TURBOHAUL_KVCACHE_PERSIST_MAX_BYTES`. The seconds ones are
abbreviated — `_S`, not `_SECONDS`.

## Launch

Compose, which builds the same self-contained image and starts it:

```
docker compose up -d --build
docker compose logs -f
```

Plain Docker:

```
docker run --gpus all -p 11401:11401 \
    -v $(pwd)/state:/var/lib/turbohaul \
    -v $(pwd)/models:/var/lib/turbohaul/import-staging \
    --tmpfs /var/lib/turbohaul/kvcache:rw \
    turbohaul-manager:local
```

The `--tmpfs` line is not optional decoration: the fast KV tier must be its own in-memory mount or the
manager logs a warning on every boot (see the KV entry in Troubleshooting). The shipped compose file
does **not** set it, so if you use compose, add a `tmpfs:` entry for
`/var/lib/turbohaul/kvcache` under the service yourself.

The image's entrypoint is the `turbohaul-manager` console script, so there is no wrapper and no
default argument list. Append flags if you need them — `--config`, `--allow-public-bind`,
`--log-level` — but the image already sets the environment equivalents.

## Ports and volumes

- **One port: 11401.** The OpenAI-style and Ollama-shaped surfaces are both on it, served by the same
  process. There is no second port.
- **Leave a small range above 11401 open.** The manager hands each loaded model its own engine port,
  allocated upward from 11500. One model loaded at a time is the default.
- **The two shipped launch paths expose the port differently.** Compose publishes it on the local
  machine only (`127.0.0.1:11401:11401`); the manual `docker run` publishes it on all network
  interfaces. If a remote agent cannot connect, this is almost always why.
- **`/var/lib/turbohaul` is the one volume that matters.** It holds the state database, the model
  manifests, the model files, and the saved fast-start cache. Without it everything lives in the
  container's writable layer and dies with the container. Point it at a real directory; the shipped
  docs call this required for production even though nothing in the code enforces it.
- **`/var/lib/turbohaul/import-staging` is the model intake directory.** This is the designed place to
  put a model file you already have: you drop or fetch it here (the compose and `docker run` examples
  mount your host `./models` at this path), then import it with one API call. The manager refuses any
  import path outside this directory. Nothing creates it for you — make it yourself.
- Back up the whole `/var/lib/turbohaul` tree, not just `state.sqlite`. The database runs in a mode
  that writes recent changes to two extra files beside it, so copying the single file can lose the
  most recent activity.

## Confirm the boot

Boot runs, in order: read the config file → apply environment overrides → layer on the remembered
runtime settings → lint and repair the request-priority section (never fatal) → log every setting
that differs from a fresh install → choose the bind address → print the readiness line → open the
state and audit database → reconcile stale processes and slots → verify the engine checksum if one
is pinned → start the queue worker and background sweeps → run name resolution and plugin probes
(best-effort) → bind and serve.

Almost every step after the config load is wrapped so a failure is logged and boot continues. That
is why a healthy `/health` can sit next to a log full of errors.

Verify in this order:

1. **Look for the readiness line** in the log: it begins `ready:`, and names the address, the port,
   and whether the web UI is on. It means configuration was accepted and the server is about to
   listen.
2. **Read the `boot_reconcile:` line.** One summary of what startup found: stale processes reaped,
   slots reset to cold, ports still occupied, and whether the cache directories are on the expected
   kind of filesystem. Problems there mean startup found something to clean up.
3. **Call `GET /health` on port 11401.** Good: HTTP 200 with a short JSON body carrying a status of
   `ok` and a version. No login, no token. Bad: connection refused (nothing is listening — the
   process exited) or any non-200 (something in front is failing).
4. **Do not use `/healthz`.** It is filtered out of the access log as though it were a health
   endpoint, but no such route is registered — probing it returns 404. For the same reason, missing
   `/health` lines in the access log are normal and prove nothing.
5. **Open the web UI at `http://127.0.0.1:11401/ui`** if the readiness line said it is on. The console
   is served at the `/ui` path, not at the bare host root, and it shares the API's one port. Nothing
   else needs configuring to reach it.
6. **Read `GET /status`** for real state: queue depth, whether a model is active, still loading, in
   its hold window, or idle. A manager that has not served yet shows under `loading` — that is a
   normal cold load, not a hang. Then `GET /api/config` to confirm the values actually in force, then
   send one small real request and confirm it completes.
7. **Scan for `config diverges from shipped default:` lines.** Each names a setting whose effective
   value is not what a fresh install would use — this is how you confirm a saved or environment
   override is really in force.

Know the limit of step 3: `/health` does no work. No database query, no slot interaction, no GPU
query. It cannot tell you a model will load, that the engine binary works, that the GPU is usable,
or that the config sections you care about are valid. It proves the HTTP server is up and the app
was built. On a fresh install the engine binary's integrity is genuinely unchecked — pin the
expected checksum if that matters to you.

The images ship a container health check that curls `/health` on loopback every 15 seconds with a
3-second timeout and 5 retries, so an orchestrator will mark the container unhealthy if that
endpoint stops answering.

Logs go to stdout/stderr only — the manager writes no log file of its own. Set verbosity with
`--log-level` or `TURBOHAUL_LOG_LEVEL` (`critical`, `error`, `warning`, `info`, `debug`, `trace`;
default `info`) and capture the container's output with `docker logs` or your logging driver.

## Troubleshooting

| What you see | Meaning | Do this |
|---|---|---|
| `could not select device driver` / the container exits immediately | The NVIDIA container toolkit is not installed or not configured on the host | Install and configure it; both launch paths require a GPU reservation, so nothing starts without it |
| `config not found: <path>`, exit code 2 | No config file at that path | Provide the file, or point at the right path with `--config` / `TURBOHAUL_CONFIG_PATH` |
| Validation error naming a section and key (any section other than request-priority), process exits | Bad value, missing required field, or an **unknown key** | Read the named section and key; fix the value or delete the key |
| `config root must be mapping, got <type>` | Valid YAML that is not a key/value document | Rewrite the file as a mapping |
| `llama_server_binary sha256 mismatch` (logged, boot continues) | A pinned checksum does not match the file on disk | Correct the pinned value, or clear it to skip verification (development only) |
| `nvidia-smi unavailable; skipping GPU compute-apps scan` | No GPU tooling visible to the container | Expected without a GPU; fine at boot, matters at first model load |
| `fastlane config coerced at boot from ...` (logged, boot continues) | Only the request-priority section was bad and the manager repaired it | Nothing to do; check the section if the repair was not what you wanted |
| `fastlane config in ... is invalid and could not be salvaged; booting with fastlane DISABLED ...` (logged, boot continues) | The request-priority section was bad beyond repair | That feature is off; repair it in Settings or remove the section |
| `config diverges from shipped default: <section>.<key>` | Effective value differs from a fresh install | Informational — tells you where a persisted or environment override is in force |

Three more that look like failures but are not:

- **The first model load looks stuck.** The manager waits a couple of minutes for a model to report
  healthy before calling it failed, so a large cold load is not misreported as an error. Do not
  restart the container during a first load.
- **Errors in the boot log beside a healthy `/health`.** Startup reconciliation, the checksum check,
  the cache sweep, and the plugin probes are all non-fatal by design. Read the log; do not assume
  the process is broken.
- **KV cache tier complaints.** Startup asserts the fast tier is RAM-backed and the durable tier is
  not, and complains loudly but keeps serving if the check fails. A fast tier that is not actually in
  RAM silently wears the disk instead. The two tier directories are fixed in the program — no config
  key and no environment variable moves them. Mounting `/var/lib/turbohaul` as a whole is **not**
  enough: the fast tier must be its own in-memory mount *inside* that tree, or the check fires on
  every boot. Give the fast tier its own tmpfs, as the launch examples above do
  (`--tmpfs /var/lib/turbohaul/kvcache:rw`, or the matching `tmpfs:` entry in compose). What you can
  tune is size and age: the durable tier caps at 40 GB, the fast tier at 20 GB, and fast-tier entries
  are discarded after 6 hours by default.
