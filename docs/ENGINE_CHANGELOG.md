# Engine changelog

Turbohaul-Manager serves models through a vendored build of `llama.cpp`. This
file records the changes carried in that vendored engine on top of its upstream
base, so that engine behaviour visible through the manager can be traced to a
specific change and, where one exists, to its upstream discussion.

Entries are grouped by the area they affect. Each entry states what the problem
was, what the change does, and why it was adopted. Changes taken from upstream
link to the upstream pull request; changes developed here for this engine carry
no external reference.

This file describes the engine source. Which engine build a given deployment
runs is a separate question, answered by the engine pin recorded in
`engine.lock`. Note that the pin records the commit this source was taken from,
not a commit the shipped tree matches.

---

## Selected set

The changes below are carried in the vendored engine source, on top of its
base. A running engine contains them only if it was built from this source.
`engine.lock` records the base commit this source was taken from; because the
shipped tree carries further work, that pin does not by itself identify a
build. This file does not change the pin.

Every engine change we cherry-pick or make ourselves is recorded in this file when it is completed.

## Upstream commits (llama.cpp)
https://github.com/ggml-org/llama.cpp

#### MoE / stability

### Fix tensor-parallel and CPU-offloaded MoE expert crash on MoE models

Combining tensor parallelism with CPU-offloaded MoE experts aborted during warm-up on MoE models with a contiguity assertion failure in the backend buffer transfer path. The failing tensor is the MoE router output, which is replicated identically across every device and stored as a non-contiguous view; the assertion checked contiguity before checking whether the tensor was mirrored, even though the mirrored case was already handled just below it. The check order is corrected so a mirrored tensor reaches the existing handling instead of aborting, and tensor parallelism without expert offload is unaffected.

Upstream: https://github.com/ggml-org/llama.cpp/pull/25028

### Fix out-of-bounds read in the n-gram lookahead map on prompt shrink

The n-gram map's stale-entry invalidation used the previous generation's start offset instead of the new prompt's length, so when a slot was reused for a shorter prompt, keys extending past the new prompt length were kept and later read out of bounds during draft lookup. Invalidation now uses the new prompt length and drops any key whose span extends past it.

Upstream: https://github.com/ggml-org/llama.cpp/pull/23936

### Reject a decreasing sequence position within a batch

A batch could silently accept a token position lower than the sequence's current position, because the per-sequence current-position tracker was never populated, so nothing enforced monotonic positions. The tracker is now populated and checked, so a batch that would move a sequence's position backward is rejected instead of silently accepted.

Upstream: https://github.com/ggml-org/llama.cpp/pull/25449

### Fix crash when using draft-simple speculative decoding

Using the draft-simple speculative decoding strategy could abort at runtime. The draft context was created with a smaller maximum-output capacity than the verification batch the target model subsequently required, so an assertion on that capacity failed and terminated the process. The draft decode is now handed its own copy of the batch with the target's per-token logits selection cleared, and the engine reads a null logits selection as "output for the last token only", so the draft context derives its own output positions instead of inheriting the verification batch's, and never needs output slots for them. Draft-simple decoding completes without aborting, and nothing about the draft context's capacities is resized. The speculative-decoding tests can now name the strategy explicitly and set draft bounds directly, and assert that drafting actually engaged.

Upstream: https://github.com/ggml-org/llama.cpp/pull/25720

### Fix stale tensor-split parameters for draft models

Tensor-split mode combined with certain speculative decoding drafts could throw once the draft model was created, because the tensor-split parameters used to build it were not retained by the model and went out of scope while still required. The parameters are now copied into the model structure itself so they stay valid for its lifetime; the added storage is a small fixed-size array and its overhead is negligible.

Upstream: https://github.com/ggml-org/llama.cpp/pull/24814

### Remove Top-N-Sigma's redundant softmax and sort

The Top-N-Sigma sampler always performed a softmax and sort over the token distribution before returning, even when the next sampler in the chain does not need that ordering. When Top-N-Sigma is immediately followed by a sampler that already handles ordering itself, such as Dist, this work was pure overhead. The softmax and sort are now removed outright: the sampler masks candidates below its threshold and returns, leaving ordering and renormalisation to whatever sampler runs next.

Upstream: https://github.com/ggml-org/llama.cpp/pull/22645

#### CUDA correctness

### Address integer overflows in the CUDA binary-ops implementation

Large tensors processed by binary operations on the CUDA backend could overflow 32-bit signed integer index and size calculations. Index and size types were widened, multiplications that could overflow were given explicit safe casts, and launch-time assertions were added to catch any value still out of range before a kernel runs.

Upstream: https://github.com/ggml-org/llama.cpp/pull/24706

### Reduce temporary buffer memory usage in CUDA top-k and argsort

The CUDA implementations of top-k and argsort allocated their full working buffers up front, sized to the entire input, which could use a large amount of temporary memory for large tensors. Both operations now process the input in smaller chunks instead of sizing their working buffers to the whole input, reducing peak memory usage. (In the top-k fallback path the temporary destination buffer is allocated once before the chunk loop; argsort's chunks each use their own working buffers.)

Upstream: https://github.com/ggml-org/llama.cpp/commit/074944998d3f25e7001ede30d152b59dff741c8c (#24776)

### Fix external compilation of the q1_0 CUDA MMQ kernel

The CUDA MMQ (matrix multiplication quantized) kernel was missing an external template declaration for the q1_0 quantization type, causing that template specialization to be compiled twice: once in the shared compilation unit and once in its own per-type instance file. The missing declaration is added and the ordering of the per-type declarations is made consistent with the rest of the file.

Upstream: https://github.com/ggml-org/llama.cpp/pull/25778

### Fix incorrect matrix indexing in the CPU SIMD GEMM scalar tail-column path

The scalar tail-column path in the SIMD GEMM microkernel indexed the first matrix using an offset scheme left over from before the kernel was changed to receive already row-block-offset pointers for each block. Applying the old offset scheme against the new, already-offset pointers produced an incorrect index in that tail path, yielding wrong results for row counts that do not divide evenly into full blocks. The index is corrected to match the current offset pointers.

Upstream: https://github.com/ggml-org/llama.cpp/pull/25390

#### Server / deploy

### Enforce a strict prompt-cache RAM limit

Previously `--cache-ram` was a soft target: the cache always kept at least one entry even if it exceeded the configured RAM/token limits, and old entries were evicted only after the new one was saved, so total cache size could briefly exceed the limit even when every individual entry was within bounds on its own. Saving now fails outright if a new entry alone would exceed the limit, older entries are evicted first to make room, and token-limit cleanup may now evict the last remaining entry rather than always keeping one. `--cache-ram` is now a hard ceiling instead of a best-effort target.

Upstream: https://github.com/ggml-org/llama.cpp/pull/25070

### Fix a crash in the edit_file tool on end-of-file appends

The `edit_file` server tool crashed when a model appended content at the end of a file using `line_start: -1`: the value was normalized to one line past the buffer's length instead of to its length, so the insert wrote past the end of the vector. The fix restricts `-1` to append operations only, rejecting it for replace and delete, and inserts at the buffer's true end, including for empty files.

Upstream: https://github.com/ggml-org/llama.cpp/pull/24893

### Fix read_file line-number formatting breaking edit_file matches

`read_file` prefixes each line with `{n}→ ` in its output; the space after the arrow is only a separator, but a model reusing that text as `edit_file`'s target string couldn't tell the difference and kept the leading space, so the match failed every time, since fuzzy matching only trims trailing whitespace. That forced repeated failed retries and, eventually, a full file rewrite. Removing the separator space means the returned line now matches the file's real content exactly, so the failure can no longer happen.

Upstream: https://github.com/ggml-org/llama.cpp/pull/25705

### Add an id field to Responses API tool-call output

The OpenAI Responses API defines both an item-level `id` and a `call_id` on `function_call` output objects, but the server's `/v1/responses` endpoint only emitted `call_id`. Some clients require the item-level `id` field to deserialize a tool call and failed outright without it. The server now emits both fields, matching the documented Responses API shape.

Upstream: https://github.com/ggml-org/llama.cpp/pull/24882

### Add missing adaptive-sampling parameters to generation_settings

The `adaptive_target` and `adaptive_decay` task parameters were omitted when task parameters were serialized into a response's `generation_settings` block, so callers inspecting that field never saw them even though the parameters were active. Both fields are now included, with a regression test covering the omission.

Upstream: https://github.com/ggml-org/llama.cpp/pull/25830

### Disable proxy buffering on streaming endpoints

Streaming endpoints did not send the `X-Accel-Buffering: no` header, so an Nginx reverse proxy placed in front of the server would buffer responses by default and break streaming for clients expecting incremental output. The header is now set on streaming responses only; it has no effect when Nginx isn't involved, and non-streaming endpoints are unaffected.

Upstream: https://github.com/ggml-org/llama.cpp/pull/24774

#### Multimodal

### Fix silent prompt truncation on an embedded NUL byte

A NUL byte embedded in a chat message silently truncated everything after it before the prompt reached the model, with no error or log line. This is low severity on its own, but it can break reliability in agentic use, since the truncated tail may have carried instructions or context the model never actually saw. The prompt is now built without treating an embedded NUL as a string terminator, so the full message reaches the model as sent.

Upstream: https://github.com/ggml-org/llama.cpp/pull/25548

### Fix multimodal token-count miscalculation

The token-count logic for merged multimodal embeddings was convoluted enough to produce incorrect counts in some configurations. It has been rewritten around two explicit cases: with temporal merging disabled, the output token count is the product of the two spatial dimensions and the batch size; with it enabled, that count is divided by the merge factor and rounded up.

Upstream: https://github.com/ggml-org/llama.cpp/pull/24656

### Stop dropping image blocks in tool results during Anthropic-to-OpenAI conversion

When converting an Anthropic-format `tool_result` that contained image blocks into the server's internal OpenAI-style representation, the image content was silently discarded and only the text was kept, so a client that expected a tool result to include an image saw the model behave as though none had been provided. Tool results containing images are now converted into OpenAI multimodal content parts, text plus an image array, covering both base64-embedded and URL-referenced images; results with no images continue to pass through as plain text.

Upstream: https://github.com/ggml-org/llama.cpp/pull/22536

#### Model support

### Fix double-escaping in the LFM2 tool-call parser

The LFM2 tool-call parser re-escaped function-argument strings that were already escaped by the underlying JSON encoding, so a literal newline in an argument round-tripped as a doubled escape sequence in the final tool-call JSON instead of a single one. The parser no longer re-escapes already-escaped content, and test coverage was added for the case.

Upstream: https://github.com/ggml-org/llama.cpp/pull/24667

### Load GLM-DSA indexer tensors as optional

Loading a GLM-5.2 GGUF failed with a missing-tensor error because the loader created the model's five DSA lightning-indexer tensors as required on every layer, while GLM-5.2 only ships the indexer on a subset of layers. Since this architecture runs as plain MLA attention and the indexer runtime isn't implemented yet, those tensors are loaded but unused regardless of whether they're present; marking them optional lets layers without an indexer load with a null tensor instead of failing, and leaves DeepSeek-V3.2's uniform per-layer indexer unaffected.

Upstream: https://github.com/ggml-org/llama.cpp/pull/24770

### Fix speculative decoding for qwen3next

The qwen3next graph never populated its per-layer input-tensor slot, so speculative-decoding drafters that read target hidden states from that slot, including DFlash and EAGLE3, aborted at runtime on a null-tensor assertion. The graph now registers the layer input tensor as other architectures already do, so those drafters work against qwen3next as expected.

Upstream: https://github.com/ggml-org/llama.cpp/pull/25141

## Upstream commits (TurboQuant fork)
https://github.com/TheTom/llama-cpp-turboquant

#### MoE / stability

### Fix draft-model load abort when every device reports zero free memory

When a draft model loads after the primary model has already filled the available VRAM, every device can report zero bytes free. The default memory-proportional device split then divides by that zero sum, producing NaN split points that push a bounds check past the end of the device list and abort the draft-model load. The split now falls back to a uniform division across devices whenever the free-memory weights sum to zero, with an added bounds clamp on the resulting device index.

Fork: https://github.com/TheTom/llama-cpp-turboquant/commit/8e3c455b9cc00ee31aec69ff4a6933592a46a390

### Default the quantized KV cache's attention rotation off, with per-side overrides

The quantized KV cache can rotate keys and values as they are written and read. The cache now carries controls for that, and this entry records them because they were previously unlisted: rotation is off by default on both sides, and is enabled per side by `LLAMA_ATTN_ROT_K_OVERRIDE` and `LLAMA_ATTN_ROT_V_OVERRIDE`, while `LLAMA_ATTN_ROT_DISABLE` forces it off on both sides and blocks the per-side overrides, so a known-bad combination can be locked out without changing anything else. Rotation only applies where it can — a quantized cache type and a head dimension that is a multiple of 64 — and one architecture family still receives key rotation for its own indexer where its key head size matches its indexer head size, subject to the same lock-out. The default is off because reported testing across several model families on the quantized cache types found the best policy to be model- and quantization-specific: value rotation helps substantially on some combinations, hurts on others, fails outright on a few, and is within noise on the rest, with no single answer even within one architecture family at different sizes, so a per-architecture rule would silently regress untested variants. The key rotation tile size is selectable with `LLAMA_ATTN_ROT_K_NROT`: unset keeps this engine's default of 64, and zero selects the upstream rule instead, the largest power of two that divides the head dimension, which is not the same as the default except at a head dimension of 64. The value must be a power of two of at least 64 that the head dimension is a multiple of; any other value builds and then fails an assertion on the first decode. A cache constructed with a shared cache inherits both sides' settings from the cache it shares with, and does not read the environment itself; the resolved values are reported in the cache's construction log line at startup.

Fork: https://github.com/TheTom/llama-cpp-turboquant/commit/db3595a755a9260cb76dbb3db18a8ca8fc1a9662

#### CUDA correctness

### Support the TurboQuant WHT operation under tensor-split mode

Using tensor-split mode together with TurboQuant KV cache quantization aborted, because the backend's tensor-split dispatch had no case for the TurboQuant WHT operation. A case for this operation is added to the tensor-split path so multi-GPU TurboQuant KV works correctly with tensor splitting; it reuses the same generic handler as the other non-scalar, multi-source operations in that switch, so the operation splits like an elementwise op instead of needing an operation-specific rule, and the operation's kernels themselves are unchanged.

Fork: https://github.com/TheTom/llama-cpp-turboquant/commit/a1fafdc44b7a15a3adf9f8b867b1d1c5df9b2799

#### Server / deploy

### Route a request to the slot bound to its prompt-cache key

When a request carries a prompt-cache key, the slot it runs on is now chosen with regard for the slot that key was previously bound to, so a request can start on a cold slot while the warm one sits idle and the cached prefix the key identifies goes unused; this behaviour was previously unlisted. The server keeps a map from prompt-cache key to slot index and offers a keyed request to its bound slot first. The key comes from the request's `cache_key` field, falling back to `session_id`, so a client already sending `session_id` gets this routing whether or not it asked for it. The binding is honoured only when the slot is idle and its prompt still agrees with the incoming one: it must share at least `--slot-cache-key-min-prefix` tokens with it (default 32, which 0 disables) and, unless the ratio check is disabled, at least `--slot-cache-key-similarity` of it (default 0.5, which 0.0 disables). Because the default floor is 32 tokens, a short prompt never reuses its bound slot. A binding whose slot is gone, or whose prompt has been emptied, is dropped rather than followed, and binding a key to a slot clears that slot's previous keys, so a key never outlives the slot it named. The dispatch order verified here is explicit slot id, then the key-bound slot if it qualifies, then a free slot; a keyed request that cannot use its bound slot takes a free slot with prompt-similarity matching disabled, so it is not steered onto a merely similar slot. No claim is made about other request paths beyond that dispatch sequence.

Fork: https://github.com/TheTom/llama-cpp-turboquant/commit/8e12d35d0e4ef8141cba169165db903a02b77e83

## Our engine changes

#### CUDA correctness

### Make a failed CUDA buffer free non-fatal

The CUDA backend's buffer wrapper released its device memory through the checked wrapper whose error path ends in a process abort. That destructor does not only run at shutdown: it also runs on every ordinary buffer release while the process is still serving, so a failed free at any point in the process's life could take down a live server over a diagnostic that had already succeeded in every other respect. The destructor now calls the free directly and, when it fails, logs two warnings naming the error, the current device and the source location instead of aborting; the free is still attempted every time, only its fatality is gone, and the message says explicitly that it came from a destructor so a reader does not have to already know the symbol to judge the blast radius. Every other checked call in that file is unchanged and still aborts on failure.

#### Server / deploy

### Measure a draft model against its target when the draft borrows the target's embeddings

Device-memory measurement builds a throwaway context for the model being measured and reports the bytes it needs per device. Some draft models leave the token-embedding and output tensors out of their own checkpoint to save disk and borrow them from the target when their graph is built, so measuring one without its target present failed during graph construction and reported nothing; automatic device sizing could not price such a drafter, and the failure surfaced as an aborted measurement rather than a number. A measurement entry point was added that loads the parent model and context first, hands the parent to the child's context parameters, measures the child, and then releases the parent — on failure as well as on success. The parent is loaded without allocating buffers, mapped or locked memory, and its load is logged at the same reduced verbosity as the child's, because the parent is a full-size target model whose load would otherwise flood the output of every measurement. The server's pre-fit reservation for a draft model uses this entry point; a drafter that runs on the target model itself keeps the parentless call.

### Always print a backtrace when the engine aborts

A native backtrace was printed only on the abort path taken when no abort callback was installed. Any abort while a callback was registered printed a bare message and stopped with no stack behind it — and the serving path is precisely the case where a callback is registered, so the crashes that most needed a stack were the ones that produced none. Backtrace printing now runs on every abort: the callback's message, or the default message when there is no callback, is printed first, then the backtrace, then the process stops.

#### Model support

### Register per-layer input tensors on the qwen2 graph

Drafters that read the target's hidden states out of the per-layer input slot — the DFlash and EAGLE3 drafters both do, as the qwen3next entry above describes — hit a null-tensor check and aborted at runtime on any architecture whose graph left that slot unpopulated. The qwen2 graph was one such architecture, so speculative decoding against a qwen2 target aborted instead of drafting. The qwen2 graph now registers the layer input tensor in that slot, as the other architectures it can be targeted from already do.

#### Speculative decoding

### Turn a draft-model context failure into a structured downgrade instead of an abort

A draft model whose checkpoint omits a tensor its architecture needs could abort the process while its context was being created, and an assertion raised that way cannot be intercepted by the check that only handles a context coming back null; the old path logged a warning about a failed memory measurement and then walked into the abort, so one incompatible drafter took the server down. With automatic device fitting on — the default, and disabled with `--fit off` — a failure while measuring the draft or multi-token-prediction context before the target is loaded now marks speculation as downgraded, records which component failed and why, drops every model-based speculative strategy from the request, clears the draft model's path, and lets the target load and serve on its own. Once the target has loaded, at which point its architecture is known, a single structured log line records the downgrade: `[spec] SPEC_DOWNGRADED arch=<architecture> component=<draft|mtp> reason=draft_context_init_failed detail=<text>`. The reason is a fixed code for machine consumption; the component is `draft` or `mtp`; the detail is free text. A draft context that genuinely fails to construct is now reported as an error and the server declines to start, instead of continuing with a null draft context.

### Support the DFlash draft-model architecture end to end

There was no way to run a draft model of the DFlash family in this engine: the architecture had no id or name in the registry, no model class or graph, and no speculative strategy that could select it, so such a checkpoint could not be loaded, drafted against, or produced. DFlash is now a first-class draft architecture: registered under its own id and name, instantiated by the model loader, reported as an encoder-input architecture, and given a model class and graph that run in two modes — an embedding batch that projects the target's hidden states and injects them as keys and values into the draft context's cache, rotating them into the cache's rotated space on the way, and a token batch that denoises a masked block to produce draft tokens. Its encoder input width is derived from the target layers it consumes rather than read from the draft's own input width, and its rotary embedding type is the half-split NeoX variant, which is what its weights are laid out for. On the speculative side a `draft-dflash` strategy drives it: the target layers named by the draft checkpoint have their inputs extracted, the draft context is told to produce masked embeddings and to attend non-causally, and each sequence drafts one block per step. The block size the model was trained with must be present in the checkpoint as a positive integer and is now rejected when it is missing, unparsable or out of range, rather than defaulted; a requested draft size above what the trained block can yield is clamped with a warning. A checkpoint with no target-layer list is refused at load, and one whose list is present but empty fails an assertion at speculative init; the shipped converter writes the list only when it is non-empty and defaults the block size to 16 when the config omits it, so both of those are reached only from a hand-edited checkpoint. Converting a DFlash checkpoint requires `--target-model-dir`, pointing at the target model's directory: the draft checkpoint carries no vocabulary of its own, so the tokenizer is taken from the target and converted against the target's own vocab handler. Finally, a strategy that needs its own draft model may only be configured once at a time — draft-simple, EAGLE3, multi-token prediction, DFlash and DSpark all share a single draft context slot — so asking for more than one is refused by name. Note that the refusals here and in the block-size checks all surface the same way at the server: the speculative decoder is not initialised, one error naming the reason is logged at startup, and the server runs and serves normally without speculative decoding.

### Support the DSpark draft variant's Markov and confidence heads

DSpark is DFlash plus a semi-autoregressive Markov head — an embedding of the previous token and a projection that turns it into a bias — and a confidence head that scores each drafted position; it is selected with `draft-dspark`. Nothing existed for it: no tensor ids, names, registry entries or model storage slots, so a checkpoint carrying those tensors could not load; and its configuration is published in a flatter shape than DFlash's, and its checkpoints are written without the vocabulary tensors a plain drafter carries. The head's tensors are now registered as output tensors with their load and operation types, the loader recognises the head by the presence of the Markov weight and creates it with the confidence bias optional, and the graph builds the head and its confidence output alongside the DFlash path. Drafting then uses an anchor-first layout: instead of predicting from the last position of a masked block, it predicts from the first and takes a full block of draft tokens rather than one fewer, truncating the block at the first position whose confidence falls below the threshold. That truncation only applies when the speculative probability floor is set above zero — `--spec-draft-p-min`, which defaults to 0.0 and is therefore off, and which is shared with the other drafters, so raising it for draft-simple reasons also shortens a DSpark block. On the conversion side the variant is registered, its flatter configuration is normalised into the nested form DFlash expects, its head tensor names are mapped, and the vocabulary tensors are deliberately not written, since the draft borrows the target's and conversion requires `--target-model-dir` for the same reason as DFlash.

### Reserve the target's rollback sequences only for the drafters that need them

Models with recurrent state can only partially remove a bounded number of tokens from the end of a sequence, so a drafter that produces several tokens per step against that state requires the target to reserve rollback sequences in advance, which costs real device memory in the target context. The reservation is now requested when multi-token prediction or either of the two DFlash-family drafters is configured, at the size of the configured draft window. The EAGLE3-style drafter is deliberately left out: it is a separate small model that consumes hidden-state features from selected target layers and never reads or writes the target's recurrent state, so it needs no reservation.

#### Savestate / checkpoint ladder

### Persist the slot's checkpoint ladder alongside its saved state

Saving a slot wrote the sequence state to a file, but the ladder of intermediate recurrent-state snapshots the slot built up while processing its prompt existed only in memory, so a restored slot came back with an empty ladder and had to reprocess its whole prompt. A ladder restored from a file is also unsafe to trust blindly: a stale or mismatched snapshot can reach the load path and fail late, and with a drafter active each ladder entry's draft-state size depends on how many tokens have been processed, so comparing entries against a single live snapshot of the current draft context rejects every multi-entry drafter sidecar. A sidecar file named `<save-path>.ckpt` is now written next to the state file. It carries a format version, an engine build identifier, a hash of the target model, the context and sequence-count settings, whether a drafter is configured and its draft size, the ladder spacing settings, the byte count of the state file it accompanies, and per entry the token count and position range, both state sizes and a content hash for each. It is written to a temporary file and renamed into place, and a failure to write it never fails the save. On restore it is validated field by field — version, build identifier, model hash, every setting, the state file's size, and per entry the target-state size and content hash, with the draft-state size required to be non-empty when a drafter is configured and zero when it is not — and anything that does not match rejects the whole sidecar, leaving the slot with an empty ladder rather than a partially parsed one. The ladder is then used: a later request in that slot searches it and can restore one of its snapshots instead of starting over. The in-memory ladder is cleared before it is repopulated, so a previous tenant's entries cannot leak in, and a slot whose state has just been discarded and re-prefilled from scratch also drops its ladder, whose entries describe the state that was discarded. A save that writes zero bytes is now a rejected save rather than a silent success, reporting an invalid-request error and no token count; a mid-session state read that produces zero bytes is instead skipped rather than cached, so the next request re-prefills instead of restoring a truncated state. A fresh save supersedes any previous sidecar: if the ladder is empty or the sidecar write failed, a stale `.ckpt` left over from an earlier state file is removed. The sidecar is written only when `TURBOHAUL_CKPT_SIDECAR` is set to exactly the string `1`; other values, including `true`, `yes` and `0`, leave it off, with no log line either way. The format version was raised because the draft context's key rotation changed: draft state written by an earlier build occupies a different rotational space, and the build identifier cannot tell two builds apart when both were made from a source tree without version-control metadata, so the version check is the only thing standing between a stale draft sidecar and a mis-rotated restore.

### Add a capped, prefix-only sequence-save entry point to the C API

Saving a sequence's state wrote the whole sequence, so an embedder that wanted a bounded, self-consistent prefix had no way to ask for one. A new public entry point is now exported that writes only the first N tokens and only the state below a caller-supplied position cap, producing an internally consistent file: the token header and token array are bounded by the token count and the key-value payload is bounded by the cap, and passing an unbounded cap reproduces the previous full-sequence behaviour byte for byte. The cap is threaded through the whole state-write interface — the sequence save entry point, the context and memory interfaces, and every cache implementation in the tree — and the cap is applied by sequence position rather than by storage index, so a truncated save keeps the state at the first N positions. Because this changes the signature of a pure-virtual state-write method, out-of-tree memory implementations must add the parameter to keep compiling; and because the entry point is a new exported symbol, embedders that want the prefix form must relink. The server is the first caller: it uses the cap when a save request asks for one, and reports the capped token count back to the caller.

### Reuse a restored slot's state instead of re-prefilling it

A restored slot's key state is already populated, but the prompt-processing path treated it like any other slot, so this capability was previously unlisted and its behaviour is recorded here. The restored slot now computes its reusable prefix as the longest common prefix of the restored tokens and the incoming prompt, which is correct whether the prompt repeats what was saved, extends it, or diverges from it, and the checkpoint search and reset logic is skipped for restored slots so nothing discards the position bookkeeping the restore just established. An empty prompt on a restored slot releases the slot rather than falling into stale-state handling, further mid-prompt snapshots are not taken while the restored marker is set, and the marker is cleared once the slot has been handled, so the next request on that slot takes the ordinary path.

### Stop discounting the reasoning budget from restored-slot stale-tail accounting

The stale-tail computation for a restored slot worked out how much of the restored state fell beyond the reusable prefix, then subtracted the configured reasoning budget from that number as an allowance for reasoning tokens. The budget is an upper bound rather than an actual count, so when a resumed prompt did not carry the reasoning tokens that had been saved with it — a common case, since reasoning markup is often stripped before the prompt is re-sent — the subtraction drove the stale count negative, and a negative count was read as "the restored state is entirely reusable", so the tail trim was skipped entirely. The stale count is now simply the restored token count minus the length of the reusable prefix, so the no-removal fast path is taken only when there is genuinely nothing to remove. A restored slot is only treated this way when it actually has a restored token count: one restore path marks a slot as restored without recording how many tokens were restored, which previously made the stale count always negative there and skipped the tail trim entirely; that case is now handled by the ordinary trim path. Cold restores, which set both values together, are unaffected.
