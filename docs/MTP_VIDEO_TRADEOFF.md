# MTP vs. video length — the trade-off, and what actually decides it

**A video is billed to the engine by the frame, and every frame is billed at its own resolution.** A
full-resolution camera frame costs about nine times what a 640×360 frame does, and that is the single most
important thing on this page. Read the [frame math](#it-is-frames-not-duration-or-file-size) section before
anything else if you only have two minutes.

The short version: MTP (speculative decoding's bundled draft head) makes text generation faster, and it
does that by reserving part of the context window for its own draft KV cache. A video eats context by the
frame. The two compete for the same room. Turn MTP on and text gets faster; turn it on for a long video and
the video may not fit. This page has the measured numbers so you don't have to guess which way to set it
for your workload.

Everything below was measured on one 27B model (hybrid `qwen35` arch, bundled MTP head, 250,000 ctx,
turbo3 KV), one variable changed per comparison, same box, same prompt where applicable. These are one
setup's numbers; measure your own.

---

## The trade-off, plainly

| | speculation (MTP) ON | speculation (MTP) OFF |
|---|---|---|
| Text generation | **faster** | slower |
| Max video length that fits in context | **shorter** | longer |

MTP's draft context is sized off the target's context window and allocated at its own KV type (f16 by
default; `spec_draft_type_k` / `spec_draft_type_v`), regardless of what the target itself uses. That allocation is fixed overhead — it's there whether or not
a request has a video in it, and it's why long videos are the thing that runs out of room first.

## 1. MTP is a real speed win on text

Same long realistic prompt, same 4000-token budget, same box, one variable:

| arm | gen tok/s | acceptance |
|---|---|---|
| MTP on | 21.37 | 40.08% (4439 drafted / 1779 accepted) |
| MTP off (control) | 15.20 | `draft_n: None` — proves no drafter ran |

**1.41x faster with MTP on.** (21.37 / 15.20 = 1.406.)

## 2. MTP costs video length

Same clip (600 frames, 640px width), one variable:

| arm | result |
|---|---|
| MTP on | **dies** at 125,830 prompt tokens (94% through) — `decode: failed to find a memory slot for batch of size 220` |
| MTP off | **completes** at 134,080 prompt tokens, answers correctly |

The MTP draft context ate the room those last frames needed.

## It is frames, not duration or file size

Qwen3-VL's vision encoder is patch 16, spatial merge 2 — one token per 32×32 source pixels, per frame.
Frame *count* drives the token cost. The engine decodes video at 4 frames per second by default,
whatever the clip's own frame rate, so frames ≈ 4 × seconds of video (14 frames ≈ 3.5 s, 240 frames = 1 min,
600 frames = 2.5 min). File size does not matter directly — only how many frames, at what size, you end up
sending.

| clip | frames | resolution | est. vision tokens¹ | measured `prompt_tokens`² | result |
|---|---|---|---|---|---|
| full camera resolution | 210 | 1920×1080 | 415,800 | — *(no count recorded)* | **exceeds the 250,000-token context** (estimate) |
| 640px width | 14 | 640×360 | 3,080 | — *(succeeded; `prompt_tokens` not recorded)* | fine |
| 640px width | 240 | 640×360 | 52,800 | **53,806** | fine |
| 640px width | 600 | 640×360 | 132,000 | **134,080** (MTP off) | fine *only with MTP off* — see [§2](#2-mtp-costs-video-length) |

¹ *Est. vision tokens* = `(width // 32) × (height // 32) × frames` — the vision-encoder cost alone, with
no text and no chat template. This is a computed estimate, not a server measurement.

² *Measured `prompt_tokens`* is what the server actually reported for a completed request — vision tokens
**plus** the text question **plus** the chat-template wrapper. Where a row shows both columns, the gap is
that non-vision overhead: 53,806 − 52,800 ≈ 1,006 tokens of text/template for the 240-frame case. Where a
row has no measured value — the 1080p case has no recorded count; the 14-frame case
succeeded but `prompt_tokens` wasn't recorded — that's marked with a dash rather than a made-up number.
Don't read the *est. vision tokens* column as something the server confirmed; it's the formula, nothing
more.

**The comparison that matters:** 210 frames at full camera resolution come to an estimated 415,800 vision
tokens before a single word of the prompt — more than the whole 250,000-token context. 600 frames at
640×360 are a *measured* 134,080 total. Fewer frames, at native resolution, cost roughly three times the
tokens of more frames, downsampled. Frame count and per-frame resolution are what you're paying for, not
runtime.

## The practical recipe: pre-sample before sending

Your decoder's equivalent of this — the flags matter more than the specific tool:

```sh
ffmpeg -i input.mp4 -vf "scale=640:-2" -an output.mp4
```

- **keep the clip short** — the engine decodes at 4 frames per second by default, so every second of video
  is 4 frames of vision-model context; trim to the part that matters. A lower frame rate baked into the file
  does not reduce what the engine samples, because it asks the decoder for 4 fps whatever the clip contains.
- **cap the resolution** (`scale=640:-2` above) — `-2` keeps the aspect ratio and rounds to an even height,
  which most codecs need; the equivalent flag exists in essentially any decoder.
- **drop audio** (`-an` above) — the vision path doesn't use it, so there's no reason to pay for muxing it
  back out.

Do this **before** the clip ever reaches the engine, with whatever decoder you're using. The engine
resamples only the frame rate; it does not scale frames down before the projector's own per-image limit (4096
tokens per image by default for Qwen-VL projectors), so whatever resolution you send is what gets tokenized,
up to that limit.

## The two-manifest pattern

Ship both, pointed at the same model blob so it costs no extra disk:

```yaml
model_tag: my-model-27b
gguf_blob_sha256: <same 64-hex sha256 for both manifests>
llama_server_flags:
  spec_type: draft-mtp
  spec_draft_n_max: 3
```

```yaml
model_tag: my-model-27b-video
gguf_blob_sha256: <same 64-hex sha256 as above>
# no spec_type — MTP off, full context available for frames
```

`my-model-27b` is the default: fast text, short-video-or-none. `my-model-27b-video` is the same weights with
speculation off, for long clips. Route by request, not by re-downloading or re-quantizing anything.

## Known gap

A request whose total token count exceeds the context window is refused by the engine up front with an
exceeds-context-size error. The failing case is the one in between: with MTP on, a video that fits the
window by itself can still run out of room partway through the prompt (the `failed to find a memory slot`
failure in §2), because MTP's draft context takes part of the same budget. Staying under the limits above (or
using the `-video` manifest for long clips) is how to avoid it.

---

## See also

- [VISION_MODELS.md](./VISION_MODELS.md) — wiring a vision model and its projector through the manager.
- [SPECULATIVE_DECODING.md](./SPECULATIVE_DECODING.md) — MTP, D-Flash and D-Spark in general; when to use
  which, and why results vary by model, hardware and workload.
