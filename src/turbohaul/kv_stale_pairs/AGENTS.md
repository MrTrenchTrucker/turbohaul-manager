# Module: kv_stale_pairs

## Purpose
Keep one saved KV pair per thread on the RAM-disk save directory. A thread that moves to a new engine port leaves its
old pairs behind on the ports it left. After the new pair has landed, this module finds those pairs and removes them,
sparing any port a live turn still serves. The caller supplies the proof that the new pair is real and the ports that are
live, and does all the logging.

## Owns
- Checking that the new pair's size is proven: the file size and the engine's byte count are the same positive int
  (`drop_stale_pairs`).
- Deciding which served ports are known, and which pairs of the thread are stale: exact model tag and thread hash,
  not on the new port, not on a served port (`drop_stale_pairs`, `StalePair`).
- Removing each stale pair: its bin, then its checkpoint sidecar, then its metadata file, and reporting what went
  (`DropResult`).

## Does Not Own
- Gathering the live state (which ports a live turn serves this thread on): manager.py (`_live_turn_groups`), legacy.
- The snapshot write itself and measuring the size of the file that landed: manager.py, legacy.
- Every log line, and clearing the scan cache after a drop: manager.py.
- The save-time size gate on the RAM disk: `kv_ram_cap`.
- The file names of the save directory: `turbohaul.kv_policy`. The module reads names of the form
  `<model_tag>.p<port>.<thread_hash>.slot<id>.bin` and does not build them.

## Public Interface
- `drop_stale_pairs(save_dir, model_tag, thread_hash, new_port, live_turns, landed_bytes, n_written, remove=os.remove) -> DropResult`:
  remove the thread's pairs on ports other than `new_port` and the served ports, and return what happened.
- `DropResult`: `dropped` (tuple of `StalePair`, sorted by name), `skipped` (None, or the reason nothing ran),
  `landed_bytes` and `n_written` (the arguments, repeated).
- `StalePair`: `name`, `port`, `slot_id`, `pair_bytes` (bin bytes plus checkpoint bytes).

## Depends On
- Nothing inside TurboHaul. The module uses the standard library only.

## Invariants
- `skipped` is None when the drop ran, otherwise exactly one of `landed_size_mismatch`, `served_unknown`,
  `dir_missing`, `scan_failed`. When it is not None, nothing was removed.
- The landed check comes first. `n_written` and `landed_bytes` must both be ints (a bool is not one), equal, and
  positive. Anything else keeps the old pair.
- `live_turns` is None, or a sequence of `(port, slots)` where `slots` holds one collection of thread hashes per live
  turn. An entry serves the thread when any collection holds its hash. `live_turns` None, or a serving entry whose port is
  None or not an int, gives `served_unknown`.
- A pair counts as stale only on an exact match of model tag and thread hash. Another thread, another model, or a tag
  that merely starts with this one never matches. A pair on `new_port` is never stale, whatever its slot.
- A stale pair is removed in name order, bin first, then `.ckpt`, then `.json`. A failure on one target is ignored for
  that target and the others are still tried. A pair counts as dropped only when the bin removal did not raise.
- The function never raises. An unexpected failure gives `scan_failed`.
- Standard library only: no logging, no globals, no locks, no clock, no import from turbohaul.

## Test Locations
- Unit: tests/unit/kv_stale_pairs/
- Contract: tests/contract/test_kv_stale_pairs_contract.py
- The manager's use of the module (legacy code): tests/test_kv_pair_steady_state.py

## Known Gotchas
- The module trusts the facts it is given. The landed size comes from the caller's `getsize` and the engine's
  `n_written`; the served ports come from the caller's scan of live turns. Wrong inputs give a wrong drop, which is why
  a missing or unequal size keeps the old pair.
- One pair per thread holds per (port, engine slot). A thread that changes engine slot on the SAME port keeps the older
  slot's pair, because pairs on the new port are never stale.
- A bin name embeds the model tag, which may itself contain dots. The match is on the full name, not on splitting at dots.
- The `.json` file name is the bin name without `.bin`; the checkpoint name is the bin name plus `.ckpt`.
- A bin already gone when `remove` is called is not counted as dropped by this call, because the removal raised.
