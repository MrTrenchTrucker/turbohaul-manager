# ADR-004: One co-residence admission rule and one retry read for the free-VRAM probe (module coresident_admission)

**Status:** accepted
**Date:** 2026-10-22

## Context
When a model asks to load next to a model that is already resident, the manager reads the free VRAM of each card with
nvidia-smi. That read can flake and return nothing. The request was then refused outright on one failed read, even when
the card had room. The eviction path already read twice before it gave up, so the same flake was handled two ways. The
refuse path and the eviction path each carried their own copy of the two-read idiom and of the sum of still-booting
siblings on the card. The rule existed in several copies that could drift apart.

## Decision
- One pure module, `coresident_admission` (src/turbohaul/coresident_admission), holds the retry read and the admission
  arithmetic. The reserve is the sum of still-booting siblings on the target card. A request is admitted when
  free - reserve + credit >= need.
- The module does no I/O. The caller passes in its own free-VRAM reader, so the module does not own the nvidia-smi read.
- If the primary read is unreadable, the credit already pledged for the request is checked first. If that credit alone does
  not cover the need, a bounded retry of the same reader runs. If every read fails, the request is refused, never
  admitted blind.
- The credit-only shortcut (admit when the pledged credit alone covers the need) is the same inequality with zero free
  and zero reserve, so that comparison is written once.
- The evict path uses the same retry read, reserve and inequality.
- The existing primary read is unchanged and still runs first. The manager keeps the structural refusals, the credit
  computation and the call sites only.

## Reasons
- One rule in one place. A change to the admission arithmetic is then made once.
- The module can be tested without a manager or a GPU, because the reader is passed in.
- Passing in the reader keeps the existing test seams on the manager's reader working.
- Behaviour does not change: the admit or refuse result and the number of reads on each path stay as they were.

## Consequences
- A new module, registered in modules.toml with its own card. The manager keeps only the call sites.
- The probe-available path keeps its single reading and uses the module's reserve and inequality.
- A future change to the admission arithmetic or to the retry is made in the module only.
