"""Manifests CRUD routes.

GET/PUT /api/manifests/{tag} with ETag/If-Match concurrency control + closed flag
allowlist enforcement.
"""
import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import ValidationError

from turbohaul.manifest import (
    ConcurrencyError,
    MANIFEST_FLAG_DEFAULTS,
    Manifest,
    ManifestValidationError,
    cache_reuse_inert_on_mmproj,
    delete_manifest,
    manifest_etag,
    parse_manifest,
    read_manifest,
    validate_tag,
    write_manifest_atomic,
)


log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/manifests", tags=["manifests"])

# Every way read_manifest can fail to hand back a Manifest -- see its
# docstring, which is the contract this enumerates. It must cover all six
# classes, including (yaml.YAMLError, UnicodeDecodeError), so an ordinary
# YAML typo or invalid UTF-8 never blanks the listing. A narrower tuple
# would let either of those classes 500 the whole listing.
#
# File-local ON PURPOSE, not imported and not exported, exactly as
# api/ollama.py and api/plugins.py record for their own copies: the
# no-shared-helper constraint stands, and the real consolidation candidate
# runs through chat_completion.py and is a separate refactor. Naming it here
# rather than leaving it inline is what lets the drift test hold all four
# copies to ONE coverage without coupling the four modules to one object --
# a shared ASSERTION instead of shared CODE.
_UNREADABLE_MANIFEST = (
    FileNotFoundError,
    ManifestValidationError,
    yaml.YAMLError,
    ValidationError,
    OSError,
    UnicodeDecodeError,
)


def _publish_manifest_changed(mgr, tag: str, revision: int | None = None) -> None:
    """Announce that a manifest changed. IDENTIFIER ONLY -- never a body.

    Every write route in this module publishes this event (pull.py, import_.py and
    live_monitor.py publish theirs the same way), so a manifest written here
    becomes visible to every other tab without a manual Refresh. Without it,
    the per-model config flow would only LOOK two-way, because the editor
    re-GETs and re-seeds itself after its own save.

    ⛔ ws_state.py's module docstring is a security boundary, not a style rule: the state
    channel "NEVER broadcasts: prompt text, response text, stderr lines, full
    thread_ids, IPs." A tag and a revision are identifiers -- the same class
    as import_started's pull_id/path_basename. The manifest body, its flags,
    its display_name and its description must never appear here. Consumers
    that need content re-GET it through the authenticated route.

    One helper rather than four copies of the same three lines, specifically
    so that boundary is stated and enforced in ONE place.
    """
    event: dict = {"event": "manifest_changed", "model_tag": tag}
    if revision is not None:
        event["revision"] = revision
    mgr.event_bus.publish_nowait(event)


@router.get("/{tag}")
async def get_manifest(tag: str, request: Request, response: Response) -> dict:
    """Read a manifest by tag. Returns ETag header for subsequent PUT."""
    try:
        validate_tag(tag)
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    mgr = request.app.state.manager
    try:
        m = read_manifest(mgr.boot.storage.manifests_path, tag)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"manifest not found: {tag}") from e
    except ValidationError as e:
        # Stored file parses as YAML but fails Manifest validation (e.g. the
        # reasoning_budget >= n_predict cross-field rule). A pydantic
        # ValidationError, NOT a ManifestValidationError -- catching only the
        # latter would let it 500.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        # The last two of read_manifest's six classes to reach this
        # route. Without them an ordinary YAML typo, or a manifest that is not
        # valid UTF-8, would produce an UNHANDLED 500 -- the very outcome
        # the arms above exist to prevent, arriving through two neighbouring
        # classes instead of a neighbouring file.
        #
        # str(e) here, matching the two arms above, is deliberate and is NOT
        # the leak that the discovery route in models.py forbids. The distinction is
        # ENTITLEMENT, and it is what makes these two routes answer
        # differently on purpose: THIS route hands the caller the entire
        # manifest body on success (model_dump below), so naming the line and
        # column of their own typo tells them nothing they could not already
        # read -- and it is exactly what an operator needs to fix it. The
        # OpenAI-compat discovery route does not return the body, so there the
        # same string would be a genuine disclosure.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except OSError as e:
        # Also unreadable, but an ENVIRONMENT fault rather than a file one --
        # and str(OSError) carries the absolute path, which no other arm on
        # this route emits. Fixed message out, full detail to the log, same
        # split models.py uses. (FileNotFoundError is an OSError and is
        # already taken by the 404 arm above, which is why that one must stay
        # first.)
        log.exception("manifest %r is present but unreadable", tag)
        raise HTTPException(
            status_code=400, detail=f"manifest {tag!r} is present but unreadable"
        ) from e
    response.headers["ETag"] = f'"{m.revision}"'
    out = m.model_dump(mode="json")
    # Derived-only marker, never a real field on the model (a
    # pydantic field would round-trip through model_dump() back into
    # parse_manifest() at restore-defaults time and hit extra="forbid").
    # This marker is a proxy for INERTNESS: it detects multimodal models via
    # mmproj, which covers the common case but is not the test llama.cpp
    # actually runs. llama.cpp disables cache_reuse for ANY context it cannot
    # shift -- the STEP35 architecture, or M-ROPE / I-M-ROPE position
    # encodings -- whatever the KV cache type is. See
    # cache_reuse_inert_on_mmproj() in manifest.py for the full explanation.
    # PluginManifest carries neither llama_server_flags nor mmproj_blob_sha256
    # -- scoped to kind == "model" so a plugin GET is untouched.
    if m.kind == "model":
        out["cache_reuse_inert_mmproj"] = cache_reuse_inert_on_mmproj(
            m.llama_server_flags, m.mmproj_blob_sha256
        )
    return out


@router.put("/{tag}")
async def put_manifest(
    tag: str,
    payload: dict,
    request: Request,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> dict:
    """Write a manifest. ETag/If-Match required for updates.

    First write (no existing manifest) succeeds without If-Match.
    Subsequent updates require If-Match: "<current-revision>"; mismatch → 412.
    """
    try:
        validate_tag(tag)
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Force model_tag in payload to match URL to prevent tag-mismatch confusion
    payload = dict(payload)  # copy
    payload["model_tag"] = tag

    try:
        manifest = parse_manifest(payload)
    except (ValidationError, ManifestValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    mgr = request.app.state.manager
    try:
        written = write_manifest_atomic(
            mgr.boot.storage.manifests_path, manifest, if_match=if_match
        )
    except ConcurrencyError as e:
        raise HTTPException(status_code=412, detail=str(e)) from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    response.headers["ETag"] = f'"{written.revision}"'
    _publish_manifest_changed(mgr, written.model_tag, written.revision)
    return {
        "status": "ok",
        "model_tag": written.model_tag,
        "revision": written.revision,
        "restart_required": False,  # per-model yaml hot-reloads on next stage
    }


@router.delete("/{tag}")
async def delete_manifest_route(tag: str, request: Request) -> dict:
    """Remove a manifest."""
    try:
        validate_tag(tag)
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    mgr = request.app.state.manager
    removed = delete_manifest(mgr.boot.storage.manifests_path, tag)
    if not removed:
        raise HTTPException(status_code=404, detail=f"manifest not found: {tag}")
    # No revision to report -- the manifest is gone. The tag alone is the
    # identifier, and "changed" covers deletion: consumers re-fetch and find
    # it absent rather than being told what it used to contain.
    _publish_manifest_changed(mgr, tag)
    return {"status": "deleted", "model_tag": tag}


@router.post("/{tag}/restore-defaults")
async def restore_manifest_defaults(
    tag: str,
    request: Request,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> dict:
    """Drop this model's per-model overrides for every defaulted flag.

    Scoped deliberately to the flags in MANIFEST_FLAG_DEFAULTS -- it does NOT
    touch ctx_size, tensor_split, or anything else an operator has tuned. It
    removes the per-model VALUES so the model follows the global default again;
    it does not write the default in. That distinction is the whole point: a
    written-in default is a frozen copy that stops tracking, which is exactly
    the state this endpoint exists to undo.

    Affects ONE model. Every other manifest is untouched.
    """
    try:
        validate_tag(tag)
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    mgr = request.app.state.manager
    root = mgr.boot.storage.manifests_path
    try:
        existing = read_manifest(root, tag)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"manifest {tag!r} not found") from e
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Compare against what the FILE stores, not the loaded object: the loaded
    # object always carries the defaults, so diffing it would report "nothing
    # to do" on every call.
    stored = dict(existing.llama_server_flags)
    injected = set(getattr(existing, "_defaulted_flags", set()) or set())
    # Report only the genuine per-model overrides -- an injected default was
    # never the operator's choice, so listing it as "cleared" would be a lie.
    cleared = {
        k: v for k, v in stored.items()
        if k in MANIFEST_FLAG_DEFAULTS and k not in injected
    }
    # ...but REMOVE every defaulted flag, injected ones included. They arrived
    # on the loaded object, so leaving them here would hand them to parse_manifest()
    # below as if the file had set them -- re-freezing the very defaults this
    # endpoint exists to unfreeze.
    for key in MANIFEST_FLAG_DEFAULTS:
        stored.pop(key, None)

    payload = existing.model_dump(mode="json")
    payload["llama_server_flags"] = stored
    try:
        manifest = parse_manifest(payload)
    except (ValidationError, ManifestValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        written = write_manifest_atomic(root, manifest, if_match=if_match)
    except ConcurrencyError as e:
        raise HTTPException(status_code=412, detail=str(e)) from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    response.headers["ETag"] = f'"{written.revision}"'
    _publish_manifest_changed(mgr, written.model_tag, written.revision)
    return {
        "status": "ok",
        "model_tag": written.model_tag,
        "revision": written.revision,
        # What was actually removed, and what the model now follows. An empty
        # "cleared" means it was already on the defaults -- not a failure.
        "cleared": cleared,
        "now_defaults": dict(MANIFEST_FLAG_DEFAULTS),
        # Spawn-time flags: a resident engine keeps its old argv until it
        # respawns, so the change is not visible on a loaded model until then.
        "restart_required": False,
        "takes_effect": "next model spawn",
    }


@router.get("")
async def list_manifests(request: Request) -> dict:
    """Management listing: EVERY manifest, including hidden ones.

    Deliberately separate from the discovery endpoints. /v1/models and
    /api/tags exist to tell a client what it can call, and they OMIT hidden
    models by design -- so they can never be used to manage visibility: a
    hidden model is absent from them, leaving no row to un-hide. Any UI that
    toggles visibility needs a list that shows what is hidden, which is this.

    Returns tag, hidden, revision and etag per model. Send the `etag` value
    verbatim as If-Match on a subsequent PATCH -- it is the quoted form the
    header requires; the bare `revision` integer is for display and will be
    rejected with a 412 if sent as a header.
    """
    mgr = request.app.state.manager
    root = mgr.boot.storage.manifests_path
    out = []
    for path in sorted(Path(root).glob("*.yaml")):
        tag = path.stem
        try:
            m = read_manifest(root, tag)
        except _UNREADABLE_MANIFEST:
            # One unreadable manifest must not blank the whole management view.
            # ValidationError is pydantic's class (e.g. a cross-field
            # rule); it is NOT a ManifestValidationError, and missing it here
            # would 500 the entire list and blank the Models page.
            # The enumeration must also cover two neighbouring classes:
            # an ordinary YAML typo (yaml.YAMLError) and a file that
            # is not valid UTF-8 (UnicodeDecodeError, which is a ValueError
            # and so slips past the OSError above), or the listing fails the same way
            # through those classes. The full enumeration
            # matches read_manifest's documented failure surface;
            # the drift test pins it there.
            # The blob fields must appear HERE too, as
            # None, or a consumer hits a missing key on exactly the row that is
            # already broken -- the same shape rule that governs the readable
            # rows.
            out.append({"model_tag": tag, "hidden": None, "revision": None,
                        "kind": None, "display_name": None,
                        "gguf_blob_sha256": None, "mmproj_blob_sha256": None,
                        "spec_draft_gguf_blob_sha256": None,
                        "error": "unreadable"})
            continue
        out.append({
            "model_tag": m.model_tag,
            # Deliberately NOT filtered by kind -- this IS the
            # general manifest listing, both model and plugin belong here.
            # Exposed so a consumer that mixes both kinds together can tell
            # them apart (m.kind is on HardenedManifestBase's discriminated
            # union, always present, safe to read generically).
            "kind": m.kind,
            "hidden": bool(m.hidden),
            "revision": m.revision,
            # The revision as an If-Match header value, ready to send. An ETag
            # is a QUOTED string per the HTTP spec, so If-Match: 3 is a
            # mismatch against ETag "3" and returns 412 -- a caller who reads
            # the bare integer above and sends it gets a confusing rejection on
            # a perfectly current revision. Handing back the exact string the
            # header wants removes the guess.
            "etag": f'"{m.revision}"',
            # The Models-page tile grid needs these for
            # every row without a per-manifest refetch. Off the SAME `m`
            # already loaded above -- zero extra I/O. display_name is on
            # HardenedManifestBase (every kind, default ""), safe direct read.
            # gguf_blob_sha256 is ModelManifest-ONLY -- PluginManifest
            # declares none of ModelManifest's fields (manifest.py's own
            # docstring: "no llama_server_flags, no gguf_blob_sha256, no
            # context_size"), and this listing is deliberately NOT
            # kind-filtered, so a direct m.gguf_blob_sha256 would
            # AttributeError on every plugin row. getattr(..., None) keeps
            # plugins in the list with the field simply absent-as-None,
            # matching the unreadable-row shape rather than reintroducing
            # a whole-listing failure.
            "display_name": m.display_name,
            "gguf_blob_sha256": getattr(m, "gguf_blob_sha256", None),
            # mmproj and spec_draft are the OTHER two ways a
            # manifest can name a blob -- a vision projector and a speculative
            # draft model. Their absence from this row would make a blob
            # referenced ONLY as one of those look unreferenced on the
            # Models page: it would render as an empty model and be offered for
            # deletion. Same getattr guard and the same reason as
            # gguf_blob_sha256 above: PluginManifest declares none of them, and
            # this listing is deliberately not kind-filtered.
            "mmproj_blob_sha256": getattr(m, "mmproj_blob_sha256", None),
            "spec_draft_gguf_blob_sha256": getattr(
                m, "spec_draft_gguf_blob_sha256", None
            ),
        })
    return {
        "manifests": out,
        "total": len(out),
        "hidden_count": sum(1 for x in out if x.get("hidden") is True),
    }


@router.patch("/{tag}")
async def patch_manifest(
    tag: str,
    payload: dict,
    request: Request,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> dict:
    """Narrow field update. Currently accepts `hidden` only.

    Exists so a visibility toggle does not have to round-trip a full manifest
    through PUT: a full-replace write is a read-modify-write race, and it makes
    every unrelated field in the payload something the caller can corrupt by
    accident. This touches one field and nothing else.
    """
    try:
        validate_tag(tag)
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    allowed = {"hidden"}
    unknown = set(payload) - allowed
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"PATCH accepts {sorted(allowed)} only; got {sorted(unknown)}",
        )
    if "hidden" not in payload:
        raise HTTPException(status_code=400, detail="payload must set 'hidden'")
    if not isinstance(payload["hidden"], bool):
        raise HTTPException(
            status_code=400,
            detail=f"hidden must be a bool, got {type(payload['hidden']).__name__}",
        )

    mgr = request.app.state.manager
    root = mgr.boot.storage.manifests_path
    try:
        existing = read_manifest(root, tag)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"manifest {tag!r} not found") from e
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    updated = existing.model_copy(update={"hidden": payload["hidden"]})
    # Carry the injected-default set across the copy, or the write path will
    # re-persist the defaults this branch just stopped persisting.
    object.__setattr__(
        updated, "_defaulted_flags",
        set(getattr(existing, "_defaulted_flags", set()) or set()),
    )
    try:
        written = write_manifest_atomic(root, updated, if_match=if_match)
    except ConcurrencyError as e:
        raise HTTPException(status_code=412, detail=str(e)) from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    response.headers["ETag"] = f'"{written.revision}"'
    _publish_manifest_changed(mgr, written.model_tag, written.revision)
    return {
        "status": "ok",
        "model_tag": written.model_tag,
        "revision": written.revision,
        "hidden": bool(written.hidden),
    }
