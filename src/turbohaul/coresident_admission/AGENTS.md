# Module: coresident_admission

## Purpose
Decide whether one more tensor-isolated model (split_mode 'none') may load onto a card that already holds other such models.
The decision is one inequality: free VRAM minus the reserve for siblings still booting, plus the credit for unloads already in flight, must cover the need. The module also gives the bounded retry that the manager's admission gate uses when the free-VRAM probe flakes.

## Owns
- The inequality `free - reserve + credit >= need`, and the single place it is written (`fits`).
- The reserve: the sum of the reserved need of siblings that are still booting, on one card (`reserved_loading_mib`).
- The per-card decision from a free-VRAM reading, including refusing an empty reading (`admit`).
- The bounded retry of a flaky free-VRAM read, returning the first usable reading or an empty list (`retry_read_free_vram`).

## Does Not Own
- Reading nvidia-smi, and the primary free-VRAM read with its card fallback warning: safety.py (`_vram_budget`).
- Choosing the card or the split of a model: manager placement in manager.py.
- The credit for in-flight unloads: the manager computes the number and passes it in.
- Choosing which resident to evict.
- The structural refusals: a layer-split model never co-resides, and a sibling that is not split 'none' refuses. The manager keeps those checks ahead of this module.

## Public Interface
- `SiblingLoad`: one resident sibling as the manager sees it, with its card, its reserved need and whether it is still booting.
- `retry_read_free_vram(read, attempts=2) -> list[int]`: the first non-empty reading from at most `attempts` calls, or an empty list.
- `reserved_loading_mib(siblings, gpu) -> int`: the reserve for booting siblings on one card.
- `fits(free_mib, reserve_mib, credit_mib, need_mib) -> bool`: the one inequality.
- `admit(vals, main_gpu, siblings, credit, need) -> bool`: the per-card decision from a reading.

## Depends On
- Nothing inside TurboHaul. Plain numbers and a plain callable in, a decision out. The standard library only.

## Invariants
- Pure: no I/O, no logging, no locks, no globals, and no read of the GPU. The caller supplies the reading and the reader.
- `retry_read_free_vram` calls `read` at most `attempts` times and stops at the first non-empty reading, so a healthy first read costs exactly one call. `attempts` below 1 raises ValueError.
- An empty or missing reading never admits: `admit` returns False for it.
- Only booting siblings are reserved, and only those on the card asked about. A loaded sibling is inside the live reading and is never subtracted twice.
- `fits` is the only place the comparison is written, and it uses `>=`: exactly enough room admits.

## Test Locations
- Unit: tests/unit/coresident_admission/
- Contract: tests/contract/test_coresident_admission_contract.py
- The manager's call sites (legacy wiring test): tests/test_vram_coresident_probe_retry.py

## Known Gotchas
- The module cannot read the probe itself. The manager injects it. A test that patches the probe must patch the manager's name for the retry reads AND the safety module's name for the primary read. These are two namespaces.
- `admit` reserves on `main_gpu` but reads the card it fell back to. With `main_gpu` out of range it reads card 0 and still counts siblings whose `main_gpu` equals the out-of-range number. This matches the manager's earlier inline check.
- The normal path gets its scalar free figure from the primary read in safety._vram_budget. For a layer split that figure is the sum over the cards. That path calls `fits` directly rather than `admit`, and it makes no extra read.
- The three uses are one inequality: the probe-unavailable path (retry, then admit), the relocatable evict branch (retry, then reserve and fits per card) and the normal path (reserve and fits on the scalar free figure). The credit term differs per caller, because the manager builds it, and it is not this module's business. The manager's credit-only shortcut asks `fits` with zero free and zero reserve, so the comparison is still written once.
