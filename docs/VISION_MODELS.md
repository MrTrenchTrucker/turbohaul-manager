# Vision Models (multimodal projectors)

Turbohaul-Manager can serve multimodal (vision) models. A vision model needs two pieces: the language-model GGUF and a **multimodal projector** (the `mmproj` file) that turns image embeddings into tokens the model understands. This guide shows how to wire the projector through the manager.

For general manifest fields see [MODEL_CONFIG_REFERENCE.md](MODEL_CONFIG_REFERENCE.md).

---

## Content-addressed projector: `mmproj_blob_sha256`

The projector is referenced by its **content hash**, not a filesystem path. Store the `mmproj` GGUF in the content-addressed blob store (the same store the model weights live in) and reference it from the manifest:

```yaml
model_tag: my-vision-model
gguf_blob_sha256: <64-hex sha256 of the model GGUF>
mmproj_blob_sha256: <64-hex sha256 of the mmproj GGUF>
context_size: 8192
llama_server_flags:
  ctx_size: 8192
  split_mode: none
```

`mmproj_blob_sha256` must be empty or exactly 64 hex characters (it is validated at load). **Empty means the model is text-only.**

At spawn, the manager resolves the hash to its location in the blob store and appends `--mmproj <resolved-path>` to the engine command line for you.

## Why a hash instead of a path

The raw, path-bearing `mmproj` command-line flag is **not accepted** in a manifest — it stays on the denied-flags list. Using a content hash instead of an arbitrary filesystem path means:

- **Portability** — the manifest is not tied to any host's directory layout; the manager derives the path from the blob store.
- **Integrity** — the projector is identified by its exact content; there is no ambiguity about which file is loaded.
- **Safety** — manifests cannot point the engine at arbitrary files on the host.

## Adding a projector to the blob store

Put the `mmproj` GGUF into the content-addressed blob store so its hash resolves, then reference that hash from the manifest as above. Once the blob is present and the manifest carries `mmproj_blob_sha256`, the model serves image inputs; requests without images are handled exactly like a text model.

## A vision-capable resident cannot save its KV cache on eviction

This is an engine limitation, not a manifest setting: any model with `mmproj_blob_sha256` set is
refused outright by the engine on slot save, restore, and erase, whenever the projector is loaded.
Eviction still tears the slot down as normal, but the precomputed context does not survive it — the
client's next turn pays a full re-prefill. Text-only models are unaffected. See
[KV_CACHE_RAM_LIFECYCLE.md](KV_CACHE_RAM_LIFECYCLE.md#exception--vision-capable-models-cannot-be-saved-at-all)
for the full mechanism.

## Serving alongside other models

A vision model is a normal resident: it obeys the same placement and co-residency rules as any other model (see [MULTI_GPU_PLACEMENT.md](MULTI_GPU_PLACEMENT.md)), including `auto_place`. The projector is loaded with the model on its chosen card.

## Video input requires `ffmpeg` in the image

Video decode is a separate capability from image/vision decode and has its own runtime dependency: the engine shells out to `ffmpeg`/`ffprobe` at request time to decode a video into frames. If those binaries are not on `PATH` inside the container, a video-capable model still loads and still serves image requests normally, but a video request fails at decode time.

**The `/props` capability probe reports a build-time fact, not a runtime one.** The engine's video-support flag depends on whether video support was compiled into the engine binary (and on a vision projector being loaded); it does not check for the presence of `ffmpeg`/`ffprobe` on the host. That means `/props` can report `video: true` on an image where `ffmpeg` is missing, and the failure only surfaces when a real video is submitted, as a decode error rather than a clear "unsupported" response. Runtime images built from this repo's Dockerfiles handle that gap in two different ways, and it matters which one you are on. `Dockerfile.engine-src` installs a distribution `ffmpeg` in its runtime stage, so video decode works locally. `Dockerfile` and `Dockerfile.cuda-multi` deliberately do **not**: they install a small generic shim under the names `ffmpeg` and `ffprobe`, which execs a real binary if the operator baked one onto `PATH` and otherwise forwards the invocation to a plugin container through the manager (resource plugins are a work in progress). On those images, video decode works only if you either bake in a real decoder or configure a plugin to serve one. If you build or maintain a different image for this engine, make one of those two arrangements true, or expect a silent mismatch between the advertised and actual capability.

**See also:** media tools such as transcription, OCR and image generation, reached through the manager as plugins (a work in progress), are covered in [PLUGINS_SETUP.md](PLUGINS_SETUP.md) (the operator setup guide for a plugin), [PLUGIN_REFERENCE.md](PLUGIN_REFERENCE.md) (every manifest field, registry field, runtime knob, endpoint and failure reason), [MEDIA_HOOK.md](MEDIA_HOOK.md) (why the hook exists and what its two lanes are for) and [frontend/06-plugins.md](frontend/06-plugins.md) (the Plugins tab in the web interface).

### The decoder's license is not this project's license

The `ffmpeg`/`ffprobe` install above is a build-time dependency pulled from the Linux distribution's own
package repository, under that package's own license terms — not Turbohaul-Manager's. Building and running
an image locally from this repo's Dockerfiles doesn't change that; it's still the distro's package, on the
distro's terms.

**If you redistribute a built image** — publish it, hand it to someone else, host it for others to pull —
you take on whatever obligations that decoder's license carries for you, the same way you would for any
other distro package baked into an image you hand out. That's a decision for whoever does the
redistributing to make with their own license terms in front of them; this project doesn't make it for you
and doesn't assure you either way.

If you'd rather not rely on the distro's build at all, you don't have to: the engine runs the decoder
binaries by name from `PATH`, and looks for exactly two executables — `ffmpeg` and `ffprobe`. Put a build
from any source you choose on `PATH`; the engine doesn't care where it came from, only that those two
names resolve.

## Video and speculative decoding compete for context

If the model also serves video and you're considering turning on MTP (speculative decoding) for faster text, read [MTP_VIDEO_TRADEOFF.md](MTP_VIDEO_TRADEOFF.md) first — MTP's draft context and long video both draw from the same context window, and the measured numbers there are counter-intuitive (a short full-resolution clip can cost more tokens than a much longer downscaled one).
