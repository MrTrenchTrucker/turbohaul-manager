"""PUT /api/config with boot-vs-runtime split per v0.2 §7.1.

security finding F7 must-fix: runtime-mutable fields PUT-able; boot fields → HTTP 403
(prevents the binary-swap attack: PUT /api/config with runtime.llama_server_binary
pointed at /tmp/evil.sh).
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from turbohaul.config import RuntimeConfig


router = APIRouter(prefix="/api", tags=["config"])


# Boot-only sections (per v0.2 §7 split)
BOOT_SECTIONS = {"server", "storage", "runtime", "ui"}


@router.put("/config")
async def put_config(payload: dict, request: Request) -> dict:
    """Apply runtime-mutable config updates.

    Accepts JSON like {"queue": {"grace_seconds": 60}, "pull": {...}}.
    Boot sections (server, storage, runtime, ui) → HTTP 403; restart required.
    Unknown sections → HTTP 400.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be JSON object")

    boot_attempts = set(payload.keys()) & BOOT_SECTIONS
    if boot_attempts:
        raise HTTPException(
            status_code=403,
            detail=(
                f"sections {sorted(boot_attempts)} are BOOT-ONLY; restart manager "
                "to change (v0.2 §7.1 - prevents binary-swap attack class)"
            ),
        )

    valid_sections = {"queue", "pull"}
    unknown = set(payload.keys()) - valid_sections
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"unknown section(s): {sorted(unknown)}"
        )

    mgr = request.app.state.manager
    current = mgr.runtime.model_dump(mode="json")

    # Merge: payload sections override; sub-fields merged (shallow)
    merged = dict(current)
    for section, sec_payload in payload.items():
        if not isinstance(sec_payload, dict):
            raise HTTPException(
                status_code=400, detail=f"section {section} must be object"
            )
        merged[section] = {**current[section], **sec_payload}

    try:
        new_runtime = RuntimeConfig(**merged)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Apply atomically: replace runtime + refresh derived timer config
    mgr.runtime = new_runtime
    if "queue" in payload:
        mgr.grace.grace_seconds = new_runtime.queue.grace_seconds
        mgr.grace.max_extensions = new_runtime.queue.max_grace_extensions
        mgr.idle.idle_seconds = new_runtime.queue.idle_hot_load_seconds

    return {
        "status": "ok",
        "restart_required": False,
        "applied_sections": sorted(payload.keys()),
        "current": new_runtime.model_dump(mode="json"),
    }
