# Module: kv_stack_wiring

## Purpose
Be the manager-facing side of the slot KV save directory's RAM cap and its one-pair-per-thread rule. The two
decision modules are pure: they log nothing and read no live state. This module does what they may not. It reads the
live manager, writes the log lines, awaits the GC the cap gate asks for, and hands the decision modules their inputs.
It holds the manager-facing side of the KV stack, as it behaved inside manager.py.

## Owns
- Gathering the live thread hashes for the GC: the HARD and SOFT tiers (`live_kv_protection`).
- Gathering the engine ports a live turn serves, for the stale-pair drop (`live_turn_groups`).
- The directory accounting with its DEBUG log hook (`dir_accounting_logged`).
- The save-time cap gate with its warning and its call to the GC (`save_declined_by_cap`).
- The stale-pair drop after a save, with its log lines and its swallow of any failure (`drop_stale_pairs_after_save`).
- The GC's SOFT-tier fallback loop (`evict_soft_floor`).

## Does Not Own
- The decisions themselves: the cap gate, the protection split, the SOFT lists (`kv_ram_cap`), and which pairs are
  stale and how they are removed (`kv_stale_pairs`).
- The GC itself, its unlink function and its other prune steps: manager.py, legacy.
- The snapshot write, the engine calls and the scan-cache implementation: manager.py, legacy.
- The save directory and the cap value: the runtime config and subprocess_mgr.py. The caller passes them in.

## Public Interface
- `dir_accounting_logged(save_dir) -> tuple[int, list]`: `(total_bytes, bins)` for one directory. A failed scan logs one
  DEBUG record and returns what it has.
- `live_turn_groups(host, bin_identity)`: `[(port, [hash_set, ...]), ...]` for the live turns, or None when the scan
  fails.
- `live_kv_protection(host, bin_identity) -> tuple[set, set]`: the `(hard, soft)` thread-hash sets.
- `save_declined_by_cap(host) -> bool` (async): True when the save must be declined; it has already written the warning.
- `drop_stale_pairs_after_save(host, drop_fn, save_dir, model_tag, thread_hash, port, final_path, n_written, bin_fn)`:
  returns None; runs the drop and its log lines.
- `evict_soft_floor(surviving, deleted_paths, hard_hashes, soft_hashes, pinned_paths, hash_by_path, total, max_bytes, unlink_pair) -> tuple[int, int]`:
  `(removed_count, total)`.

## Depends On
- `kv_ram_cap`, through its package entry point: `dir_accounting`, `split_protection`, `enforce_save_cap`,
  `soft_ceiling_order`.
- `kv_stale_pairs` is the module whose drop function the manager hands in as `drop_fn`. This module imports nothing
  from it; it is named here because the stale-pair drop is what this wiring serves.
- The standard library.

## Invariants
- Each function behaves as it did inside the manager, with the same predicates, the same order
  of reads and calls, the same log text, level and arguments, and the same logger name, `turbohaul.manager`.
- What was swallowed stays swallowed (`live_turn_groups`, `live_kv_protection`, `drop_stale_pairs_after_save`,
  `dir_accounting_logged`). What propagated still propagates (`save_declined_by_cap` lets a failure of the accounting
  or the GC through).
- `live_kv_protection` runs the split after the gather even when the gather failed part-way, so the protection gathered
  so far is kept.
- `save_declined_by_cap` writes its warning only when it returns True.
- `evict_soft_floor` stops as soon as `total` is at or under `max_bytes`, and takes a bin's size off `total` only when
  the unlink removed it.
- The manager's own functions come in as arguments (`bin_identity`, `drop_fn`, `unlink_pair`), because tests replace
  them in the manager's namespace at run time.
- No state, no tasks, no clock.

## Test Locations
- Unit: tests/unit/kv_stack_wiring/
- Contract: tests/contract/test_kv_stack_wiring_contract.py
- Through the real manager (legacy code): tests/test_kv_ram_cap_wiring.py,
  tests/test_kv_hardcap_save_and_gc.py, tests/test_kv_port_move_live_scan.py

## Known Gotchas
- `host` is the manager, read through a small protocol. A rename of a manager member this module reads breaks the
  call at run time, not at import; the wiring tests above drive the real manager for that reason.
- `drop_stale_pairs_after_save` evaluates its inputs in this order: the landed size, then the live-turn scan, then the
  drop. The live-turn scan runs even when the landed size could not be read.
- The log text keeps its original wording, because the text is behaviour.
- Logging is under the manager's logger name on purpose. Filters and tests that watch `turbohaul.manager` still see
  these lines.
