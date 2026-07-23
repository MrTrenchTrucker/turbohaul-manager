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

## Serving alongside other models

A vision model is a normal resident: it obeys the same placement and co-residency rules as any other model (see [MULTI_GPU_PLACEMENT.md](MULTI_GPU_PLACEMENT.md)), including `auto_place`. The projector is loaded with the model on its chosen card.
