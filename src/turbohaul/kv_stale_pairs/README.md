# kv_stale_pairs

Removes the old saved KV pairs of a thread that has moved to another engine port.

## What it does
TurboHaul saves a conversation's KV cache to a RAM disk, so the conversation can be restored later without a full
re-prefill. The save is named after the model, the engine port, the conversation (a thread hash) and the engine slot.
When a conversation moves to a different port, its old save stays behind and wastes RAM. This module deletes those
leftovers once the new save is safely on disk, so each conversation keeps one save per (port, engine slot).

## How it works
1. **Prove the new save.** The caller passes the size of the file that landed and the byte count the engine reported.
   They must be the same positive whole number. If not, nothing is deleted and the old save stays.
2. **Know the live ports.** The caller says which ports a live turn is still using for this conversation. If that is
   unknown, nothing is deleted. A port in use is never touched.
3. **Scan once.** The module reads the save directory once and sizes each bin together with its checkpoint sidecar.
4. **Pick.** A bin is stale when its name matches this model and this conversation exactly, and its port is neither the
   new port nor a port in use.
5. **Remove.** For each stale bin, in name order, it removes the bin, then the checkpoint, then the metadata file. A
   failure on one file does not stop the others. A bin counts as dropped when its removal went through.

The result says what was dropped, or why nothing ran (`landed_size_mismatch`, `served_unknown`, `dir_missing`,
`scan_failed`), and repeats the two sizes it was given so the caller can log them.

## What it does not do
- It does not gather the live state, write the save or measure the new file. The caller does.
- It does not log and does not raise. The caller logs from the result.
- It does not decide whether a save is allowed. The RAM cap gate in `kv_ram_cap` does that.
- One save per conversation holds per (port, engine slot). A conversation that changes engine slot on the same port
  keeps the older slot's save.

## Who calls it
manager.py, and only there, right after a slot save has landed. The public names are listed in AGENTS.md.
