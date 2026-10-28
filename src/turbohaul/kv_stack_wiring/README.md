# kv_stack_wiring

The manager-facing side of the slot KV save directory's RAM cap and its one-pair-per-thread rule.

## What it does
TurboHaul saves a conversation's KV cache to a RAM disk. Two modules decide how that disk is kept in bounds:
`kv_ram_cap` decides whether a save may proceed and which saved bins the GC may take, and `kv_stale_pairs` decides which
old pairs a thread left behind on other engine ports. Both are pure on purpose: they log nothing and look at no live
state. This module is the glue between them and the manager. It reads the live manager, writes the log lines and awaits
the GC.

## How it works
1. **Measure.** `dir_accounting_logged` counts the bins in the save directory and logs a failed scan at DEBUG.
2. **Gate.** `save_declined_by_cap` asks `kv_ram_cap` whether the save may go ahead, driving the GC once to make room.
   When the answer is no, it writes the warning and returns True so the manager skips the snapshot.
3. **Protect.** `live_kv_protection` reads the manager's live slots and splits the threads into HARD (a live turn's) and
   SOFT (idle-held). `evict_soft_floor` is the GC's last step when a HARD floor still holds the directory over the cap:
   it takes SOFT bins, oldest first, until the total is back under the cap.
4. **Drop.** After a save lands, `drop_stale_pairs_after_save` calls the manager's stale-pair drop, writes its log
   lines, and never lets a failure reach the save. `live_turn_groups` supplies the ports a live turn still serves.

## What it does not do
- It does not decide anything the two decision modules decide.
- It does not run the GC, write snapshots or keep any state.

## Who calls it
manager.py, and only there, through short methods that keep their old names. The manager passes in the functions it
looks up in its own namespace at call time. The public names are listed in AGENTS.md.
