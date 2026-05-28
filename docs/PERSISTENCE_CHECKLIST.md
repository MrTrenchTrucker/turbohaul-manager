# Turbohaul Persistence Checklist

A deployment persistence checklist for running Turbohaul-Manager durably in production.

**Audit date:** 2026-05-19.
**Migration executed:** 2026-05-19 — bind-mount migration to make state persistent.

**Headline:** All applicable checklist items PASS. Bind-mount migration complete. New image tag baked + tarballed. Auto-recovery armed + mirrored to multiple sites.

---

## Final state (post-migration)

| Checklist item | Status | Details |
|---|---|---|
| Restart Policy | PASS | `unless-stopped` |
| Script Sync | N/A | container deployment |
| Persistent Volumes | **PASS** | `/var/lib/turbohaul` bind-mounted to a host data path on durable storage |
| Auto-Recovery | **PASS** | recovery function defined + wired into the host recovery flow; `bash -n` PASS; mirrored to multiple sites |
| Off-host Backup | PASS | config + state mirrored to a separate host |
| Image Backup | **PASS** | 2 tarballs: pre-migration + post-migration (with patches baked) |

## Migration log

| Phase | Wall | Result |
|---|---|---|
| Pre-flight (inactive-agent check, disk space, container size 106 GB) | < 1 min | GREEN |
| `docker stop <container>` | 10.9 s | GREEN |
| `mkdir -p <host-data-path>` | < 1 s | GREEN |
| `docker cp <container>:/var/lib/turbohaul/.` → host (106 GB) | 11.6 min (152 MB/s) | GREEN — 7 blobs + 9 manifests + state.sqlite + WAL all intact |
| Cleanup duplicate import-staging in host path (21 GB; overlaid by inner ro mount) | < 1 s | GREEN — 21 GB freed |
| `docker rm <container>` | < 1 s | GREEN |
| `docker run` with new `-v <host-data-path>:/var/lib/turbohaul` | < 1 s | GREEN — container started, /status responsive at 2 s |
| **Issue:** `/api/manifests/{tag}` 500 — `llama_server_flags.reasoning` rejected by base-image validator | (caught) | **finding: image-vs-patches debt** (see below) |
| Surgical fix: `docker cp src/turbohaul/.` → container site-packages (25 .py files, md5-verified match dev tree) | < 1 s | GREEN |
| `docker restart <container>` | < 5 s | GREEN — boot_reconcile clean, periodic_terminal_park_sweep started |
| Verify `/api/tags` lists all manifests + `/api/manifests/qwen3.6-27b-dense` returns full TurboQuant flags | < 1 s | GREEN |
| `docker commit` (bake patches into new image layer) | 2 s | GREEN |
| `docker save` new tag → tarball | 2 min | GREEN — 2.59 GB tarball |
| Mirror tarball to off-host backup | < 30 s | GREEN |
| Update recovery function to use new tag + new tarball | < 1 s | GREEN |
| Mirror updated script to backup sites | < 5 s | GREEN |

**Total migration wall:** ~18 min including verification. Inference interruption: ~15 min (from `docker stop` until verified `/api/manifests` GREEN).

## Image-vs-patches debt finding (banked)

The pre-migration image `turbohaul-manager:v0.2-cuda` was baked back at Phase 6 (174a847, mid-May 2026). Subsequent code changes — /v1/logging, /v1/embeddings, response_format, Ollama-tools, net-host guard, DB pool, streaming test, IPv6 RCA, the sweeper — were applied to the **running container** via `docker cp` overlays, not baked into a new image.

The bind-mount migration recreated the container from the base image and lost those overlays. The `reasoning` flag in the manifest schema (added after the bake) was rejected by the base-image validator. Surgical re-apply via `docker cp src/turbohaul/.` restored runtime correctness. The new `docker commit` → `v0.2.3-cuda-bindmount` tag bakes those patches into a proper image layer so future recoveries do not require the re-apply step.

**Going forward:** ANY non-trivial production deploy should `docker commit` + re-save tarball + update auto-recovery references, OR rebuild from `Dockerfile.cuda` against current dev-tree HEAD. The "patch live container only" pattern is brittle and should not be repeated.

## Survival matrix (current)

| Event | Survival |
|---|---|
| Server reboot | ✓ (restart policy) |
| Docker daemon restart | ✓ (restart policy) |
| Kernel panic (any kind) | ✓ (state on the bind-mount) |
| `docker rm <container>` | ✓ (state on disk; auto-recovery recreates from tarball) |
| Container layer corruption | ✓ (state on disk; auto-recovery recreates) |
| Disk corruption on the host data path | ⚠ partial — off-host backup has state.sqlite snapshot (pre-migration) + manifests.tar.gz; blobs LOST (re-pull from HuggingFace) |
| All backup sites gone | total loss (acceptable — multi-site loss = full incident) |

The blob backup gap is acceptable: GGUFs are deterministic + addressable by SHA256, so re-pull from HuggingFace recovers them given enough wall time (hours per 17-21 GB file). The `blobs_inventory.txt` in the off-host backup preserves the SHA list so we know exactly what to re-pull.

## Recovery procedures

### Container missing
```bash
# Manual recovery (also automatable via your host's auto-recovery script)
docker load -i <backup-path>/turbohaul-manager_v0.2.3-cuda-bindmount.tar.gz
docker run -d --name turbohaul \
    --restart unless-stopped \
    --runtime nvidia --gpus all \
    -p 11401:11401 \
    -v <host-data-path>:/var/lib/turbohaul \
    -v <model-staging-path>:/var/lib/turbohaul/import-staging:ro \
    -e TURBOHAUL_CONFIG_PATH=/etc/turbohaul/turbohaul.yaml \
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    turbohaul-manager:v0.2.3-cuda-bindmount
```

### Host data path lost (disk corruption)
1. Restore `state.sqlite` + `manifests.tar.gz` from the off-host backup.
2. `tar xzf manifests.tar.gz` into the restore path.
3. Recreate container per "Container missing" above — it will start with state + manifests but missing blobs.
4. Use `blobs_inventory.txt` to re-pull each GGUF from HuggingFace via `POST /api/pull-hf` (deterministic by SHA — re-creates the blob store).

### Whole server gone
Restore from the off-host backup onto replacement hardware, then follow "Container missing" above.

## See also

- [MULTI_AGENT_SHARING.md](./MULTI_AGENT_SHARING.md) — multi-agent multiplexing context.
- [TURBOQUANT_FLAGS.md](./TURBOQUANT_FLAGS.md) — flag doctrine (the persisted manifests).
