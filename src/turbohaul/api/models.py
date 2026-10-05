"""GET /v1/models + /v1/models/{model} — OpenAI-compatible discovery surface.

Clients polling /v1/models would get nothing without this module (the route
would not exist -- /api/tags is a separate route).
Discovery-only: model SWITCHING already works via /v1/chat/completions
(the chat route validates `model` via read_manifest). This does not widen that.

Same list_manifests/read_manifest source /api/tags uses (ollama.py), same
local FileNotFoundError/ManifestValidationError handling pattern — not
consolidated into a shared helper (only two call sites exist, so a
shared helper is not warranted). No GGUF model-card parse: the OpenAI model
object carries none of those fields, and that parse is expensive (offloaded
off the event loop elsewhere for real work) — irrelevant here.

Security note: /v1/models/{model} takes a
caller-controlled path segment, unlike /api/tags' disk-sourced tags.
read_manifest -> _safe_manifest_path -> validate_tag already rejects
traversal/invalid tags by raising ManifestValidationError
before any filesystem access, so no redundant validate_tag call is added
here. But that makes the `except ManifestValidationError` branch below
security-load-bearing: it is what turns a traversal attempt into a clean
400 instead of an unhandled 500. Keep it explicit and narrow — never widen
to a bare `except Exception`.

Deliberate asymmetry, not a bug: the LIST endpoint SKIPS a
ManifestValidationError (tags come off disk; one corrupt manifest must not
500 the whole listing, as ollama.py's tag listing does). The SINGLE-model
endpoint SURFACES the same exception as 400 (that tag came from the
caller). Same exception type, opposite handling, on purpose.
"""
import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from turbohaul.manifest import (
    ManifestValidationError,
    ModelManifest,
    list_manifests,
    read_manifest,
)


log = logging.getLogger(__name__)

router = APIRouter(tags=["openai-compat"])

# read_manifest can fail SIX ways, and the LIST path catches all of them:
#   - FileNotFoundError
#   - ManifestValidationError
#   - yaml.YAMLError (a .yaml with broken SYNTAX)
#   - pydantic ValidationError (a well-formed mapping that no longer
#     satisfies the Manifest schema, e.g. a manifest written by an older
#     turbohaul that predates a newly-required field; after any schema
#     change, every stale manifest raises it)
#   - OSError (an unreadable file)
#   - UnicodeDecodeError (a file that is not valid UTF-8; it is a
#     ValueError, so catching OSError alone does not cover it)
#
# A single bad file must not 500 the ENTIRE listing: that would defeat the
# whole point of skipping bad entries. The last two classes are easy to
# miss, so the tuple names all six explicitly. It is kept file-local
# rather than shared, like the same tuple in ollama.py.
#
# NOTE this is the LIST path only. The single-model path below still
# surfaces ManifestValidationError as a 400 on purpose -- see the module
# docstring. (See the read_manifest docstring for the full set of classes
# it raises.)
#
_UNREADABLE_MANIFEST = (
    FileNotFoundError,
    ManifestValidationError,
    yaml.YAMLError,
    ValidationError,
    OSError,
    UnicodeDecodeError,
)


def _manifest_created_at(manifests_root: Path, tag: str) -> int:
    """Best-effort int unix timestamp of a model's manifest .yaml mtime.

    Never raises — falls back to 0. Mirrors ollama.py's
    _manifest_modified_at_iso, but returns the int form OpenAI's schema
    expects instead of ISO-8601.
    """
    try:
        return int((Path(manifests_root) / f"{tag}.yaml").stat().st_mtime)
    except OSError:
        return 0


def _to_openai_model(m: ModelManifest, manifests_root: Path) -> dict:
    # `m` is a ModelManifest, not the wider Manifest union: `Manifest` covers
    # both model and plugin manifests, and only the model variant carries
    # context_size, llama_server_flags and mmproj_blob_sha256. Both callers
    # isinstance-check before calling here.
    #
    # context_length reports the context the model is CONFIGURED for:
    # llama_server_flags.ctx_size when that flag is set, otherwise the
    # manifest's context_size. The flag wins because context_size alone is
    # never passed to llama-server -- publishing it would report a number the
    # engine does not use, which would be misleading.
    # (The manager resolves the context the same way, flag first.)
    #
    # context_length_source says where that number came from, not whether
    # it's enforced -- nothing at this layer rejects a longer request, so
    # "enforced" would be the wrong word. "flag" means
    # llama_server_flags.ctx_size was set; "manifest" means it wasn't, and
    # llama.cpp is free to pick its own context at load time, so the number
    # is configured intent rather than a hard limit either way.
    #
    # context_length and context_length_source
    # both derive from ONE lookup (_ctx_flag below) so they cannot disagree.
    # Otherwise the value would fall through on a falsy ctx_size (`or`) while
    # the source keyed on bare presence -- a ctx_size:0 manifest would then
    # report source="flag" alongside a manifest-derived value. Currently
    # UNREACHABLE: manifest.py's SAFE_LLAMA_FLAG_BOUNDS["ctx_size"] = (1,
    # 2_000_000) rejects any manifest with ctx_size <= 0 at read_manifest,
    # before this function ever runs. If that bound ever moves, revisit what
    # ctx_size=0 actually means at the engine level.
    #
    # multimodal is derived from the vision-projector digest -- the same
    # condition the manager uses to decide whether to pass --mmproj. The
    # digest itself is never published.
    _ctx_flag = m.llama_server_flags.get("ctx_size")
    return {
        "id": m.model_tag,
        "object": "model",
        "created": _manifest_created_at(manifests_root, m.model_tag),
        "owned_by": "turbohaul",
        "context_length": int(_ctx_flag or m.context_size),
        "context_length_source": "flag" if _ctx_flag else "manifest",
        "turbohaul": {
            "display_name": m.display_name or m.model_tag,
            "multimodal": bool(m.mmproj_blob_sha256),
        },
    }


@router.get("/v1/models")
@router.get("/models")
async def list_models(request: Request) -> dict:
    """OpenAI-compat: list every model in the blob store.

    Served at both ``/v1/models`` and ``/models``; the two paths share this
    handler and return identical responses.
    """
    mgr = request.app.state.manager
    manifests_root = mgr.boot.storage.manifests_path
    data = []
    for tag in list_manifests(manifests_root):
        try:
            m = read_manifest(manifests_root, tag)
        except _UNREADABLE_MANIFEST as exc:
            # One unreadable manifest must never 500 the whole listing — this
            # is the discovery surface a client polls, so a single bad file on
            # disk cannot be allowed to hide every other model.
            # But a skip must never be silent either -- an
            # operator debugging "why did my model vanish" needs a trace.
            # Type name only, never str(exc) (same leak rule as the 500
            # branch below: yaml.YAMLError leaks parser position/content,
            # pydantic ValidationError echoes manifest field VALUES).
            log.warning("skipping unreadable manifest %r: %s", tag, type(exc).__name__)
            continue
        if not isinstance(m, ModelManifest):
            # list_manifests() returns every manifest tag regardless of
            # kind. This is a MODEL listing -- without this check a plugin
            # manifest would be advertised as a selectable chat model;
            # this mirrors the isinstance(m, PluginManifest) filter in
            # api/plugins.py, in the inverse direction.
            continue
        if m.hidden:
            # Listings-only: the model stays fully loadable by exact tag via
            # /v1/models/{model} and the completion routes, which read the
            # manifest directly rather than enumerating this list.
            continue
        data.append(_to_openai_model(m, manifests_root))
    return {"object": "list", "data": data}


@router.get("/v1/models/{model}")
@router.get("/models/{model}")
async def get_model(model: str, request: Request) -> dict:
    """OpenAI-compat: fetch one model by tag. 404 unknown, 400 invalid tag.

    Served at both ``/v1/models/{model}`` and ``/models/{model}`` --
    the two paths share this handler and return identical
    responses, same pattern as list_models above.
    """
    mgr = request.app.state.manager
    manifests_root = mgr.boot.storage.manifests_path
    try:
        m = read_manifest(manifests_root, model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"model not found: {model}") from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (yaml.YAMLError, ValidationError, UnicodeDecodeError, OSError) as e:
        # UnicodeDecodeError and OSError belong to the same case this arm
        # describes -- present but unreadable -- and without them an
        # undecodable manifest would produce an UNHANDLED 500: bare
        # "Internal Server Error", no log line, none of the redaction
        # below. Listing them here keeps this arm a complete list of the
        # present-but-unreadable cases. They are not wrapped at the producer:
        # they would land on the ManifestValidationError arm above, which
        # renders str(e) -- the one thing the redaction rule below forbids.
        # The tag is valid and the model isn't absent -- the file
        # is present but broken (corrupt YAML syntax, or a stale manifest
        # that predates a schema change). Not a 4xx (client did nothing
        # wrong) and not a 404 (conflating "you typo'd it" with "your
        # manifest is corrupt" destroys an operator's ability to debug a
        # vanished model). A handled 5xx: the server failed to fulfil a
        # valid request because of bad server-side data. Fixed message only
        # -- never str(e): yaml.YAMLError leaks parser position/content,
        # pydantic ValidationError echoes manifest field VALUES (verified
        # empirically). Full detail logged server-side instead.
        log.exception("manifest for %r is present but unreadable", model)
        raise HTTPException(
            status_code=500, detail=f"manifest for '{model}' is present but unreadable"
        ) from e
    if not isinstance(m, ModelManifest):
        # A plugin manifest is not a model: like list_models above (a
        # metadata lookup, not the hot inference path), this check keeps a
        # plugin tag from returning 200 as if it were a chat model. It gets
        # the same 404 an unknown tag gets.
        raise HTTPException(status_code=404, detail=f"model not found: {model}")
    return _to_openai_model(m, manifests_root)
