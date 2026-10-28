# kv_ram_cap

Keeps the slot KV save directory under its RAM cap.

## What it does
TurboHaul saves a conversation's KV cache to a RAM disk, so the conversation can be restored later without a full
re-prefill. Without a cap, repeated saves can fill that RAM disk well past its limit. This module holds the rules that
keep it under the limit: when a save may proceed, which saved bins are safe to delete, and in what order.

## How it works
1. **Measure.** `dir_accounting` counts the `.bin` files in the save directory. Each file is counted with its `.bin.ckpt`
   sidecar, because the sidecar is about as large as the bin.
2. **Gate.** Before a save, `enforce_save_cap` checks the total against the cap. If the total is at or over the cap, the
   caller's GC runs once to make room, and the directory is measured again. If it is still at or over the cap, the save
   is declined. The live KV stays in the engine; only the on-disk copy is skipped.
3. **Protect.** `split_protection` sorts the live threads into two groups. HARD bins belong to a live turn (the active
   slot and any in-flight teardown), and the GC never deletes them. SOFT bins belong to idle threads, and the GC may
   delete them under pressure.
4. **Choose.** `soft_age_targets` lists what the age prune may delete: the unprotected candidates, then the SOFT bins.
   `soft_ceiling_order` lists the SOFT bins the ceiling prune may delete once every unprotected bin is gone, oldest
   first.

## What it does not do
- It does not delete files. The GC's unlink loop does that, in manager.py.
- It does not look at live state. The caller gathers the threads and passes them in.
- It does not log. A caller that wants a log line passes an `on_error` hook.

## Who calls it
manager.py, and only there. It calls the module from the save path, from its directory accounting, from its live
protection scan and from the GC's selection. The public names are listed in AGENTS.md.
