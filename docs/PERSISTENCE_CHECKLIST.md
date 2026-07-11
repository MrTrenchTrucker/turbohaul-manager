# Turbohaul Persistence Checklist

The full hardening checklist for a production Turbohaul deployment. It covers
restart policy, persistent volumes, image tarball backups, config mirroring,
auto-recovery, and a survival matrix that maps each failure mode to whether
your state and manifests survive it.

Follow this after your first successful `docker run` to take a demo/dev
deployment to a production-grade, restart-and-crash-survivable one.

---

## Persistence audit items

Work down this list; every item should be **PASS** before you consider the
deployment durable.

| Item | Goal | How to satisfy |
|---|---|---|
| Restart Policy | Container comes back after reboot / daemon restart | Run with `--restart unless-stopped` |
| Persistent Volumes | Blobs, manifests, and DB survive container recreation | Bind-mount `/var/lib/turbohaul` to a directory on the host data volume |
| Auto-Recovery | A missing container is recreated automatically | A recovery hook that reloads the image tarball and re-runs the container (see below) |
| Config Backup | Container `docker run` config is reproducible | Mirror the run config + env to a backup location |
| Off-box Backup | State survives loss of the primary host | Copy `state.sqlite`, `manifests.tar.gz`, and the image tarball to a second machine |
| Image Backup | You can rebuild the exact container without a network pull | `docker save` the image to a tarball and keep it with the off-box backup |

### Recommended final state

- Restart policy: `unless-stopped`.
- Persistent volume: `/var/lib/turbohaul` bind-mounted to a directory on the
  host data volume (sized for your models — a handful of GGUFs is easily tens
  of GB).
- Auto-recovery: a recovery function that reloads the image tarball and re-runs
  the container, mirrored to at least one off-box location.
- Backups: `docker run` config + env mirrored; `state.sqlite`, a manifests
  archive, and two image tarballs (pre/post any config change) copied off-box.

## Migrating an existing demo container to a bind-mount

If you started with an anonymous/managed volume (e.g. from a quick demo run)
and want to move to a host bind-mount without losing your blob store, the
following sequence moves the data in place. Inference is interrupted only for
the duration (roughly 15-20 minutes for ~100 GB of blobs on a fast SSD).

1. **Pre-flight.** Confirm no active inference, check free disk space, and note
   the container size (`docker ps -s`).
2. `docker stop turbohaul`.
3. `mkdir -p /var/lib/turbohaul` on the host (your chosen bind-mount target).
4. Copy the container's data out to the host:
   `docker cp turbohaul:/var/lib/turbohaul/. /var/lib/turbohaul/`.
   This preserves the blobs, manifests, `state.sqlite`, and its WAL.
5. Remove any duplicate import-staging data from the host copy if you mount a
   separate read-only staging directory (it would otherwise be overlaid).
6. `docker rm turbohaul`.
7. Re-run the container with the new bind-mount (see "Container missing" below
   for the full `docker run`). Confirm `/status` responds.
8. Verify `/api/tags` lists all your manifests and that
   `/api/manifests/{tag}` returns the full flag set for a representative model.

### Bake your patches into the image

If you have been applying code changes to a **running** container (via
`docker cp` overlays) rather than rebuilding the image, recreating the
container from the base image will drop those overlays. Symptom: a manifest
that uses a newer schema field (for example a `reasoning` flag) is rejected by
the older base image's validator after recreation.

Fix and prevent it:

1. Re-apply your current source into the running container if needed:
   `docker cp src/turbohaul/. <container>:/opt/venv/lib/python3.11/site-packages/turbohaul/`,
   then `docker restart <container>`.
2. Bake the running state into a fresh image layer:
   `docker commit turbohaul turbohaul-manager:<new-tag>`.
3. `docker save` the new tag to a tarball and update your auto-recovery
   reference to it.

**Going forward:** any non-trivial production deploy should `docker commit` +
re-save the tarball + update the auto-recovery reference, OR rebuild from
`Dockerfile.cuda-multi` against your current source tree. Patching a live container
only is brittle and should not be your steady-state workflow.

## Survival matrix

With the recommended final state above, each failure mode maps to whether your
data survives:

| Event | Survival |
|---|---|
| Server reboot | Yes (restart policy) |
| Docker daemon restart | Yes (restart policy) |
| Kernel panic (any kind) | Yes (state on the host bind-mount) |
| `docker rm` of the container | Yes (state on host; auto-recovery recreates from the image tarball) |
| Container layer corruption | Yes (state on host; auto-recovery recreates) |
| Disk corruption on the host data volume | Partial — the off-box copy has a `state.sqlite` snapshot + manifests archive; blobs are lost and must be re-pulled |
| Off-box backup + host + volume all gone | Total loss (acceptable — losing every copy at once is a full disaster) |

The blob-backup gap is acceptable by design: GGUFs are deterministic and
addressable by SHA256, so they can be re-pulled and recover byte-for-byte given
enough wall time (hours per large file). Keep a `blobs_inventory.txt` (the list
of blob SHAs) with your off-box backup so you know exactly what to re-pull.

## Recovery procedures

### Container missing

```bash
# Manual recovery (also automatable via a recovery hook)
docker load -i /var/lib/turbohaul-backups/turbohaul-manager_<tag>.tar.gz
docker run -d --name turbohaul \
    --restart unless-stopped \
    --runtime nvidia --gpus all \
    -p 11401:11401 \
    -v /var/lib/turbohaul:/var/lib/turbohaul \
    -v /path/to/models/_smoke:/var/lib/turbohaul/import-staging:ro \
    -e TURBOHAUL_CONFIG_PATH=/etc/turbohaul/turbohaul.yaml \
    -e TURBOHAUL_ALLOW_PUBLIC_BIND=1 \
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    turbohaul-manager:<tag>
```

### Host data volume lost (disk corruption)

1. Restore `state.sqlite` and the manifests archive from your off-box backup.
2. `tar xzf manifests.tar.gz` into the restore path.
3. Recreate the container per "Container missing" above — it starts with state
   and manifests but missing blobs.
4. Use `blobs_inventory.txt` to re-pull each GGUF via `POST /api/pull-hf`
   (deterministic by SHA — this re-creates the blob store).

### Whole server gone

Rebuild the host, install Docker + the NVIDIA runtime, restore the off-box
backup, and follow "Host data volume lost" above.

## See also

- [MULTI_AGENT_SHARING.md](./MULTI_AGENT_SHARING.md) — multi-agent multiplexing context.
- [TURBOQUANT_FLAGS.md](./TURBOQUANT_FLAGS.md) — flag doctrine (the flags that persist in your manifests).
