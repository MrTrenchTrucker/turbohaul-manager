# Turbohaul Persistence Checklist

**Scope:** the persistence and recovery hardening required for a production Turbohaul deployment — what must survive a container removal, a host reboot, and a disk loss.

---

## Checklist

What must be true before a Turbohaul deployment is safe to leave running unattended.

| Item | Requirement | Why |
|---|---|---|
| Restart policy | `--restart unless-stopped` | Survives host reboot and Docker daemon restart. |
| State volume | Bind-mount a host directory to `/var/lib/turbohaul` | Without it, `state.sqlite`, `manifests/*.yaml` and `blobs/` live in the container layer and die with `docker rm`. |
| Image provenance | Built from `Dockerfile.engine-src`, tagged, and reproducible from a commit | A container patched in place cannot be recreated; the next recovery starts from the base image and silently loses the patches. |
| Config | `turbohaul.yaml` mounted read-only from version control | The effective config must be recoverable, not just observable via `GET /api/config`. |
| Blob inventory | Keep the list of blob hashes, and where each file came from (Hugging Face repo and filename, or URL), off-host | GGUFs are content-addressed, so the hash list identifies every file and lets a re-pull be verified (`expected_sha256`). |
| Off-host copy | Mirror `state.sqlite` and the `manifests/` directory to a second machine | The state volume is the only thing here that cannot be rebuilt from the repository. |

## Image provenance

**Rebuild from `Dockerfile.engine-src` against the current tree.** That is the canonical production build: it compiles the vendored, TurboQuant-modified engine from source, so what ships is always what is actually in the repository.

Do **not** rebuild from a Dockerfile that bypasses the vendored engine: it would not compile the engine source in this repository, so a non-compiling engine could reach the mainline with no signal. The slim `Dockerfile` ships the manager only and does not include the engine binary.

Patching a running container in place, or committing it and re-saving a tarball, is not a deploy path: a container whose code was laid over the running filesystem cannot be reproduced from any commit, and the next recreation from its base image silently reverts every one of those patches — with the only symptom appearing later, as a request the recreated container rejects.

## Survival matrix

| Event | Survives? |
|---|---|
| Host reboot / Docker daemon restart | Yes — restart policy |
| `docker rm` of the container | Yes — state is on the host volume |
| Container-layer corruption | Yes — recreate from the tagged image |
| Loss of the host state volume | Partial — manifests and `state.sqlite` restore from backup; blobs must be re-pulled from their sources and verified by hash |
| Loss of the host and every off-host copy | Total — accepted as a full-incident scenario |

Blobs are the one deliberate gap. GGUFs are content-addressed by SHA256, so an off-host list of blob hashes together with each file's source (repo and filename, or URL) is a complete re-pull recipe: recovery costs wall-clock time (GGUF files are large) but loses no information.

## Restore

1. Recreate the container from the tagged image with the same state bind-mount and config mount.
2. If the state volume was lost: restore `state.sqlite` and the `manifests/` directory from backup, then re-pull each blob from its source via `POST /api/pull-hf` or `POST /api/pull-url`, passing `expected_sha256` so the download is verified against the recorded hash.
3. Confirm with `GET /health`, then `GET /api/tags` — every model manifest that is not marked hidden should list (`GET /api/manifests` lists all manifests, hidden ones included).

## See also

- [DEPLOYMENT_PATTERNS.md](./DEPLOYMENT_PATTERNS.md) — deployment topologies and model-selection guidance.
- [TURBOQUANT_FLAGS.md](./TURBOQUANT_FLAGS.md) — flag doctrine (the manifests this checklist persists).
- [../README.md](../README.md) — quickstart, including the required state bind-mount.
