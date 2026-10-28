# Module: kv_ram_cap

## Purpose
Keep the slot KV save directory under its RAM cap. Before a snapshot is written, decide whether the save may
proceed, after at most one GC pass. Decide which live bins are HARD (a live turn's, never evicted) and which are SOFT
(idle-held, reclaimable under pressure). Pick the bins the GC may take. The caller measures the directory, gathers the
live state, drives the GC and does all the logging.

## Owns
- The byte accounting of the save directory: each `.bin` plus its `.bin.ckpt` sidecar (`dir_accounting`).
- Splitting live thread identities into the HARD and SOFT protection sets (`split_protection`, `ProtectionSplit`).
- The save-time cap decision: whether a save proceeds, and whether a GC pass made room (`cap_exceeded`,
  `enforce_save_cap`, `SaveGate`).
- The two lists the GC walks: the age-prune targets and the SOFT ceiling order (`soft_age_targets`,
  `soft_ceiling_order`).

## Does Not Own
- Gathering the live state (the active slot, the in-flight teardowns, the resident idle threads): manager.py
  (`_live_kv_protection`), legacy.
- Running the GC (`_gc_kv_cache`), the unlink loop and `_unlink_pair`: manager.py, legacy.
- The snapshot write itself (`_save_slot_kv_inner`), and every log line, including the RAM-cap SAVE SKIP and SHADOW_EVICT
  lines: manager.py.
- Choosing the save directory and the cap value: the runtime config and subprocess_mgr.py. The caller passes both in.

## Public Interface
- `dir_accounting(save_dir, on_error=None) -> tuple[int, list]`: `(total_bytes, bins)` for the `.bin` files in one
  directory. A missing directory gives `(0, [])`.
- `ProtectionSplit`: the HARD and SOFT hash sets, and `protected`, their union as a new set.
- `split_protection(hard_threads, soft_threads, thread_hash, bin_identity, on_error=None) -> ProtectionSplit`.
- `cap_exceeded(total_bytes, cap_bytes) -> bool`: True when the total is at or over a positive cap.
- `SaveGate`: the outcome of the gate: `proceed`, `total_bytes`, `cap_bytes`, `made_room`.
- `enforce_save_cap(cap_bytes, measure, make_room) -> SaveGate`: async; the one-pass save gate.
- `soft_age_targets(candidates, entries, hash_by_path, soft_hashes) -> list`.
- `soft_ceiling_order(surviving, deleted_paths, hard_hashes, soft_hashes, pinned_paths, hash_by_path) -> list`.

## Depends On
- Nothing inside TurboHaul. The module uses the standard library only. Plain values and callables in, a decision or a
  list out.

## Invariants
- `cap_exceeded` uses `>=`: a directory exactly at its cap counts as full. A cap of zero or less means no cap and is
  never exceeded.
- `enforce_save_cap` with a cap of zero or less returns proceed without calling `measure` or `make_room`.
- `enforce_save_cap` measures first. Only if that is at or over the cap does it call `make_room`, at most once, and
  measure again. A save proceeds only when the second measurement is under the cap.
- `soft_ceiling_order` never returns a HARD bin, and never returns a bin that is already deleted. It returns oldest
  first, and it does not change its inputs.
- `soft_age_targets` adds only SOFT bins to the candidates it is given. It does not check the candidates itself, and
  it does not remove duplicates.
- Each thread is protected under both its raw thread-id hash and its identity hash, so a bin named either way stays
  protected.
- `dir_accounting` and `split_protection` never raise. A failed scan returns what it gathered so far, and calls
  `on_error` from inside the except block so the caller can attach the traceback.
- Pure apart from the directory scan in `dir_accounting`: no logging, no globals, no locks, no clock.

## Test Locations
- Unit: tests/unit/kv_ram_cap/
- Contract: tests/contract/test_kv_ram_cap_contract.py
- Wiring (drives the real manager through the save gate and the GC): tests/test_kv_hardcap_save_and_gc.py. Its
  docstring still says "manager.py-only"; that wording predates the module split. The logic's own tests are the two
  files above.

## Known Gotchas
- The GC (`_gc_kv_cache`), the unlink loop, `_unlink_pair`, the live-state gathering (`_live_kv_protection`) and all
  logging stay in manager.py (legacy) by design. Only the decisions moved.
- `soft_age_targets` and `soft_ceiling_order` need the GC's own state passed in: the candidates, the scanned entries,
  the hash of each path, the HARD and SOFT hash sets, the pinned and deleted paths. They look nothing up themselves.
- `dir_accounting` and `enforce_save_cap` take a path or callables, so the module imports nothing from turbohaul.
- The RAM-cap rules this module carries: SOFT idle bins are reclaimable under pressure; HARD active and in-flight bins are
  never evicted; a save is declined when the HARD floor holds the directory at or over the cap; in that case the live
  KV stays in the engine and only the on-disk copy is skipped.
- `split_protection` keeps the partial split when the scan fails, on purpose. A failed scan must not stop the cap from
  working, and the caller's `on_error` records it.
