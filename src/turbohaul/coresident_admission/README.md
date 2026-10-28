# coresident_admission

Whether one more model may share a graphics card with the models already on it.

## What it decides
Some models are pinned to a single card (split_mode 'none'). Several of them can sit on the same card, as long as the card has room. This module answers one question: is there room for this model on this card?

The answer comes from one rule:

    free - reserve + credit >= need

A worked example with numbers:

- The live reading says the card has 10000 MiB free.
- One sibling model on that card is still booting. It is reserved to take 2000 MiB, and its weights are not in the reading yet.
- An unload already in flight will free 1000 MiB. That is the credit.
- The new model needs 9000 MiB.

So 10000 - 2000 + 1000 = 9000, and 9000 >= 9000. The model fits. If it needed 9001 MiB it would not.

Two details keep the numbers honest:

- A sibling that is already loaded is inside the 10000 MiB reading, so it is not subtracted again.
- A booting sibling on a different card does not reduce this card's room.

## The three parts
- **The reading.** Free MiB per card, from the caller's probe. If the probe gives no usable answer, the retry reads at most two more times after the first read has failed. The first usable answer wins. If every try fails, there is no answer, and nothing is admitted.
- **The reserve.** The sum of the booting siblings that sit on the card being checked.
- **The comparison.** The one inequality above, written in one place.

## What it does not decide
- **Reading the GPU.** The safety module runs nvidia-smi. This module never does.
- **Placement.** Which card a model goes on, and how a model is split across cards, is the manager's placement code.
- **The credit.** The number of MiB that unloads in flight will free is worked out by the manager, and passed in.
- **Eviction.** Which resident to unload is the manager's choice.
- **Layer-split models.** A model split across several cards never shares a card. The manager refuses that before this module is asked.

## How the manager uses it
The manager calls this module in three places:

1. **The probe is unavailable.** It retries the read, then asks whether any card admits the model.
2. **The relocatable evict branch.** It retries the read, then checks each card with the reserve and the comparison.
3. **The normal path.** It checks the reserve and the comparison against the single free figure its own primary read produced. This path makes no extra read.

The module never runs nvidia-smi. The manager passes its own reader in, so the module stays pure. It is tested on its own, without a running manager.
