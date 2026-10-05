"""GET /api/config/schema — additive, read-only field metadata for the FE.

Serves {type, default, minimum, maximum} per field for every runtime-mutable
section, so the FE can render typed inputs and a per-setting reset-to-default
button. Zero change to any existing endpoint.

Section list: _SECTIONS is derived FROM config_put.py's RUNTIME_SECTIONS
(imported, not copied) plus _ALL_SECTION_MODELS (the name->model-class
registry every section, boot or runtime, comes from). A separately
hand-typed list would drift from RUNTIME_SECTIONS (fastlane, persist,
monitor and http are all runtime sections beyond queue, pull and kv), and
any section missing from it would be PUT-able but have no schema:
writable, undiscoverable, no bounds, no reset-to-default. Deriving the list
means there is exactly one place a new RuntimeConfig section needs
registering (RuntimeConfig itself), and the assert below is a fail-fast
tripwire in case a hand-typed list is ever reintroduced here.


Defaults MUST come from instantiating the model (`Model().model_dump(...)`),
not from `model_json_schema()`'s 'default' key: pull.hf_host_allowlist uses
default_factory, which the JSON schema reports as null. Sourcing that null
would ship a reset button that wipes the HF host allowlist to empty.
"""
from fastapi import APIRouter

from turbohaul.api.config_put import RUNTIME_SECTIONS
from turbohaul.config import _ALL_SECTION_MODELS

router = APIRouter(prefix="/api", tags=["config"])

# Reachable BEFORE _SECTIONS is built, not after: _SECTIONS is constructed
# directly from RUNTIME_SECTIONS below, so a check comparing the two AFTER
# construction would be a tautology that can never fail (a bogus
# RUNTIME_SECTIONS entry makes construction raise a bare KeyError, so
# execution never reaches that
# after-the-fact assert). This one actually guards the drift: if
# RUNTIME_SECTIONS ever names a section _ALL_SECTION_MODELS doesn't know
# about, this fires with a message instead of leaving a bare KeyError to
# whoever's debugging a broken boot.
assert RUNTIME_SECTIONS <= set(_ALL_SECTION_MODELS), (
    "config_put.RUNTIME_SECTIONS names a section absent from "
    "_ALL_SECTION_MODELS -- config_schema.py cannot build a schema entry "
    "for it: " + str(RUNTIME_SECTIONS - set(_ALL_SECTION_MODELS))
)

_SECTIONS = tuple((name, _ALL_SECTION_MODELS[name]) for name in sorted(RUNTIME_SECTIONS))

_EXCLUSIVE = {"minimum": "exclusiveMinimum", "maximum": "exclusiveMaximum"}


def _bound(spec: dict, key: str) -> float | int | None:
    """Inclusive bound if present, else the exclusive one; None if unbounded."""
    value = spec.get(key)
    return spec.get(_EXCLUSIVE[key]) if value is None else value


@router.get("/config/schema")
async def get_config_schema() -> dict:
    """Per-field {type, default, minimum, maximum} for every runtime-mutable
    section (queue, pull, persist, monitor, kv, http, fastlane)."""
    schema = {}
    for name, model in _SECTIONS:
        properties = model.model_json_schema()["properties"]
        defaults = model().model_dump(mode="json")
        # Iterate the DUMP, not the schema properties: a field that is present in
        # the JSON schema but absent from the dump (e.g. Field(exclude=True)) would
        # KeyError here and 500 the whole endpoint. Driving off the dump means such
        # a field is simply omitted -- the FE degrades, it does not break.
        schema[name] = {
            field: {
                "type": properties.get(field, {}).get("type"),
                "default": default,
                # ge/le land as minimum/maximum; gt/lt land as exclusiveMinimum/
                # exclusiveMaximum. Fall back so an exclusive bound is never served
                # as "no bound" -- an absent bound would silently drop the FE's
                # client-side range check. monitor.poll_interval_s (gt=0.0, le=60.0)
                # is a real field (the monitor section is one of _SECTIONS) that exercises the exclusive-fallback path, and
                # the monkeypatch tests exercise the same path synthetically.
                #
                # The bound is surfaced on the inclusive key by one epsilon; the BE
                # remains the validation authority and still rejects the boundary
                # value.
                "minimum": _bound(properties.get(field, {}), "minimum"),
                "maximum": _bound(properties.get(field, {}), "maximum"),
            }
            for field, default in defaults.items()
        }
    return schema
