"""PUT /api/config with boot-vs-runtime config split.

Hardening rule: runtime-mutable fields PUT-able; boot fields -> HTTP 403
(prevents the binary-swap attack: PUT /api/config with runtime.llama_server_binary
pointed at /tmp/evil.sh).

Runtime config durability -- PUT mutations persist to a runtime
override file (state_db_path.parent / "runtime_config.yaml"). At boot,
__main__ loads this file on top of the YAML defaults so edits survive
restart/recreate and are honored long-term (not reset by boot-config re-merge).
"""
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from turbohaul.config import _ALL_SECTION_MODELS, _ENV_MAP, RuntimeConfig, migrate_legacy_fastlane_key
from turbohaul.fastlane import lint_rules

log = logging.getLogger(__name__)

# Top-level sibling key in the SAME runtime_config.yaml
# file, holding {section: {field: iso8601_utc}} for every (section, field) an
# actual PUT /api/config request has explicitly touched. NOT a RUNTIME_SECTIONS
# name, so it is structurally invisible to the boot merge in __main__.py
# (which only processes `section in RUNTIME_SECTIONS`) and to
# compute_config_provenance's per-section lookups -- neither can mistake it
# for a config value. load_runtime_override (every existing and future
# caller's contract) actively strips this key before returning; only
# load_runtime_provenance_stamp reads it, by design -- see that function's
# docstring for what this can and cannot tell you.
_PROVENANCE_STAMP_KEY = "_provenance_stamp"


router = APIRouter(prefix="/api", tags=["config"])


# Boot-only sections (part of the boot-vs-runtime config split)
BOOT_SECTIONS = {"server", "storage", "runtime", "ui", "plugins"}

# Runtime-mutable sections (part of the boot-vs-runtime config split). Derived from RuntimeConfig's own
# fields rather than hand-typed, so the set cannot drift: a hand-typed copy
# could fall out of step with config_schema.py's own hand-typed _SECTIONS
# (fewer sections advertised than are actually PUT-able), leaving fastlane/persist/
# monitor/http writable but absent from GET /api/config/schema. Adding
# a new RuntimeConfig section only requires touching RuntimeConfig
# itself (turbohaul/config.py) -- this set, and config_schema.py's
# _SECTIONS, both follow automatically.
RUNTIME_SECTIONS = set(RuntimeConfig.model_fields)


def _runtime_override_path(boot_storage_state_db_path: Path) -> Path:
    """Path to the runtime config override file (lives next to state_db)."""
    return boot_storage_state_db_path.parent / "runtime_config.yaml"


def _stamp_newer(a: str, b: str) -> bool:
    """True if timestamp `a` is strictly newer than `b`. Both are expected to
    be UTC ISO8601 strings from `datetime.now(timezone.utc).isoformat()`.

    Parsed as real datetimes, NOT string-compared: `isoformat()` drops the
    trailing `.ffffff` microseconds component entirely when it's exactly
    zero, so two timestamps at the same wall-clock second can have different
    string LENGTHS -- a lexicographic compare would then order them by which
    character comes first at the point they diverge (`+` before `.` in
    ASCII), which is wrong in a way that looks fine on almost every real
    input (microsecond==0 exactly at a stamped instant is rare, so this bug
    class survives ordinary testing).

    Malformed input (should not happen -- these are always internally generated
    timestamps, but tampered/hand-edited data is possible) is treated as
    "not newer", never raises.
    """
    try:
        return datetime.fromisoformat(a) > datetime.fromisoformat(b)
    except (TypeError, ValueError):
        return False


def _migrate_legacy_fastlane_stamp_section(stamp: dict) -> dict:
    """`_provenance_stamp` is itself keyed by section name, so
    the config-VALUE migration (which only rewrites the top-level
    "fastline"->"fastlane" key `existing` is stored under) leaves a stale
    `stamp["fastline"]` sibling sitting next to the correct
    `stamp["fastlane"]` entry forever -- same failure shape as the config-value case,
    one level deeper, inside the stamp's own content rather than in
    `existing`'s top-level keys.

    Unlike the config value (a single blob where the new key simply wins
    outright), the stamp's two entries are independently true per-field
    facts: the "fastline"-keyed fields really were last written at their
    recorded time, just under the old section name. Merging is therefore
    more honest than discarding one wholesale -- per field, the LATER
    timestamp wins (see `_stamp_newer` -- this is a real per-field
    comparison, not "fastlane always wins because that's the new name").
    Only "fastlane" survives in the returned dict; "fastline" is always
    removed when present, even where it "loses" every field, so the stamp
    never carries a section name that doesn't exist in RUNTIME_SECTIONS.

    Returns a NEW dict; never mutates the input. A stamp with no legacy
    "fastline" entry is returned unchanged (true no-op).

    Does NOT change what a stamp value MEANS or how it's read --
    `load_runtime_provenance_stamp` and `compute_config_provenance` read stamps
    exactly as before; this only normalizes what `save_runtime_config` writes.
    """
    if "fastline" not in stamp:
        return stamp
    stamp = dict(stamp)
    legacy = stamp.pop("fastline")
    current = dict(stamp.get("fastlane") or {})
    if isinstance(legacy, dict):
        for field, legacy_ts in legacy.items():
            current_ts = current.get(field)
            if current_ts is None or _stamp_newer(legacy_ts, current_ts):
                current[field] = legacy_ts
    stamp["fastlane"] = current
    return stamp


def _strip_null_container_names(payload: dict) -> dict:
    """Drop `container_name` from persisted rules when it carries no value.

    Rollback safety: a rolled-back older build rejects unknown keys
    under extra="forbid" and boots the feature disabled. A rule that actually
    NAMES a container keeps its key -- that deployment has opted into the new
    schema; only the null placeholder is removed. Non-destructive: returns a
    copy, never mutates the caller's payload.
    """
    fl = payload.get("fastlane")
    if not isinstance(fl, dict) or not isinstance(fl.get("rules"), list):
        return payload
    rules = []
    for r in fl["rules"]:
        if isinstance(r, dict) and "container_name" in r and not r["container_name"]:
            r = {k: v for k, v in r.items() if k != "container_name"}
        rules.append(r)
    return {**payload, "fastlane": {**fl, "rules": rules}}


def save_runtime_config(boot_storage_state_db_path: Path, payload: dict) -> None:
    """Persist ONLY the sections/fields present in `payload`
    -- merged onto whatever is already on disk, never a full
    snapshot of the in-memory RuntimeConfig.

    `payload` is the raw PUT request body (e.g. {"queue": {"grace_seconds":
    60}}), not the fully-merged RuntimeConfig -- a full-object dump would
    re-persist every field on every PUT, silently pinning many TURBOHAUL_*
    env-derived values
    after the first PUT on a deployment. The caller (put_config) has
    already validated payload's values as part of constructing the full
    merged RuntimeConfig before calling this -- no re-validation here.

    Untouched sections/fields are left exactly as they already are on disk
    (present from an earlier PUT, or simply absent, in which case that field
    keeps resolving from whichever of env/yaml/default it already did) --
    this is a read-merge-write, not an overwrite.

    Also stamps, into the SAME file's
    `_provenance_stamp` sibling key, the UTC timestamp of this request for
    every (section, field) `payload` explicitly names -- the exact set
    `save_runtime_config` is already being told is an explicit set, simply
    kept instead of discarded once merged into the flat section dict (the
    same "value already computed, merely unread" shape as n_restored and
    the content chain). One atomic write covers both the
    config delta and its stamp, so they can never drift apart -- the
    reason this lives in-band rather than a sibling file or state.sqlite
    (the short version is that an operator can
    hand-restore a backup of just
    this one file, and a stamp that doesn't travel with the restore is a
    stamp that silently lies).

    FORWARD-ONLY, deliberately: a field already
    on disk before stamping existed gets NO retroactive stamp -- there is
    nothing to infer it from, and guessing would be worse than the
    unlabeled status quo. This labels every field
    written from here on; it does not, and cannot, heal history. Precedence
    is not affected by this: persisted always wins, and
    compute_config_provenance's "persisted" label logic does not consult
    this function.
    """
    override_path = _runtime_override_path(boot_storage_state_db_path)
    # _load_runtime_override_raw is deliberately UNfiltered (kept
    # that way so _provenance_stamp survives this read-merge-write), so a
    # legacy "fastline" block from a pre-rename build would otherwise ride
    # along in `existing` untouched forever -- the merge loop below only ever
    # looks at keys `payload` names, and PUT only ever sends "fastlane". Same
    # shared translation load_config_yaml / load_runtime_override already use
    # -- applied here so read and write agree: the very next save,
    # regardless of which section it touches, re-serializes `existing` with
    # the legacy key gone and its value carried onto "fastlane" (a no-op when
    # no legacy key is present).
    existing = migrate_legacy_fastlane_key(_load_runtime_override_raw(boot_storage_state_db_path) or {})
    now = datetime.now(timezone.utc).isoformat()
    # The stamp's own CONTENT is section-keyed too -- a legacy
    # "fastline" entry inside it survives migrate_legacy_fastlane_key above
    # (that call only rewrites `existing`'s top-level keys, never recurses
    # into _provenance_stamp's nested per-field dict). Same opportunistic
    # "next save cleans it up regardless of which section it touches" seam
    # as the config-value migration.
    stamp = _migrate_legacy_fastlane_stamp_section(dict(existing.get(_PROVENANCE_STAMP_KEY) or {}))
    for section, sec_payload in payload.items():
        existing[section] = {**existing.get(section, {}), **sec_payload}
        sec_stamp = dict(stamp.get(section) or {})
        for field in sec_payload:
            sec_stamp[field] = now
        stamp[section] = sec_stamp
    existing[_PROVENANCE_STAMP_KEY] = stamp
    tmp = override_path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(existing, default_flow_style=False, sort_keys=True))
    tmp.replace(override_path)
    log.info(
        "runtime config persisted (sparse) to %s: sections=%s",
        override_path, sorted(payload.keys()),
    )


def _load_runtime_override_raw(boot_storage_state_db_path: Path) -> dict | None:
    """Internal: the file's FULL parsed content, including the
    `_provenance_stamp` bookkeeping key if present. Not for config
    consumption directly -- use `load_runtime_override` (config sections
    only, `_provenance_stamp` stripped) or `load_runtime_provenance_stamp`
    (the stamp only). Kept as one shared read so both views can never
    disagree about what's actually on disk."""
    override_path = _runtime_override_path(boot_storage_state_db_path)
    if not override_path.exists():
        return None
    try:
        data = yaml.safe_load(override_path.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        log.exception("failed to load runtime override %s", override_path)
    return None


def load_runtime_override(boot_storage_state_db_path: Path) -> dict | None:
    """Load runtime config override file if it exists -- CONFIG SECTIONS
    ONLY. Returns the parsed dict with the `_provenance_stamp` bookkeeping
    key stripped, so this function's contract is
    independent of that key, for every existing and future caller
    -- nobody reading config through this function
    is ever exposed to it. Does NOT filter anything else (a tampered boot
    section in the file, for instance, still passes through unfiltered --
    the __main__.py boot merge's own RUNTIME_SECTIONS check is what guards
    against that, by design, not this function).

    Returns None if no override file exists (first boot / no PUT mutations
    yet).

    Also runs migrate_legacy_fastlane_key on the way out, so a
    runtime_config.yaml written by a pre-rename build (its Fast Lane section
    still keyed "fastline") loads under "fastlane" with identical effective
    settings -- every caller of this function sees only the new key, same as
    load_config_yaml's own contract for the shipped/boot yaml. ONE shared
    translation for both disk-read sites, not two copies that could drift.
    """
    data = _load_runtime_override_raw(boot_storage_state_db_path)
    if data is None:
        return None
    data = {k: v for k, v in data.items() if k != _PROVENANCE_STAMP_KEY}
    return migrate_legacy_fastlane_key(data)


def load_runtime_provenance_stamp(boot_storage_state_db_path: Path) -> dict:
    """Load the PUT-time provenance stamp -- which
    (section, field) pairs an actual PUT /api/config request has explicitly
    set, and when (UTC ISO8601 of the most recent such PUT).

    FORWARD-ONLY -- READ THIS BEFORE USING THE RESULT. A field's absence
    here is NOT evidence that it was never explicitly set; it only means
    no PUT touched it SINCE stamping began. A field can be
    `compute_config_provenance`-labelled "persisted" (present as a key in
    the override file) while having NO entry here at all, if it was
    written before this stamp existed -- an existing deployment can have
    ALL of its runtime-mutable fields already "persisted" with
    zero way to tell which were deliberate. This function does not, and
    structurally cannot, answer that question for anything already on
    disk. Its only claim is about writes from here forward.

    Returns {} (never None) when the override file is absent or carries no
    stamp yet, so callers can iterate/lookup unconditionally without a
    None-check.
    """
    data = _load_runtime_override_raw(boot_storage_state_db_path)
    if data is None:
        return {}
    stamp = data.get(_PROVENANCE_STAMP_KEY)
    return stamp if isinstance(stamp, dict) else {}


# Reverse of config._ENV_MAP -- (section, field) ->
# the TURBOHAUL_* env var that can override it, for fields that have one.
_FIELD_TO_ENV: dict[tuple[str, str], str] = {
    (section, field): env_key for env_key, (section, field, _cast) in _ENV_MAP.items()
}


def compute_config_provenance(
    shipped_yaml_raw: dict, boot_storage_state_db_path: Path
) -> dict[str, dict[str, str]]:
    """Per field across all 11 GET /api/config sections, which precedence
    layer supplied the effective value: 'default' | 'yaml' | 'env' |
    'persisted' (highest wins, matching the real boot merge in __main__.py).

    Recomputed fresh on every call (one small file read + a handful of
    os.environ / dict lookups) so it can never go stale relative to a live
    PUT -- same "reads live state" contract GET /api/config already has for
    the values themselves.

    Boot sections (BOOT_SECTIONS) can NEVER report 'persisted': RUNTIME_SECTIONS
    structurally excludes them from the persisted-override merge in __main__.py
    (the boot-section guard -- a tampered runtime_config.yaml must not be able to
    re-open the binary-swap attack PUT already blocks with HTTP 403). A
    fabricated 'persisted' label on a boot key would be worse than no
    provenance at all, on an endpoint whose entire purpose is telling the
    truth about where a value came from.
    """
    runtime_override = load_runtime_override(boot_storage_state_db_path) or {}
    provenance: dict[str, dict[str, str]] = {}
    for section, model_cls in _ALL_SECTION_MODELS.items():
        provenance[section] = {}
        section_yaml = shipped_yaml_raw.get(section) or {}
        section_override = (
            runtime_override.get(section) if section in RUNTIME_SECTIONS else None
        )
        for field in model_cls.model_fields:
            if (
                section in RUNTIME_SECTIONS
                and isinstance(section_override, dict)
                and field in section_override
            ):
                provenance[section][field] = "persisted"
                continue
            env_key = _FIELD_TO_ENV.get((section, field))
            if env_key is not None and os.environ.get(env_key) is not None:
                provenance[section][field] = "env"
                continue
            if isinstance(section_yaml, dict) and field in section_yaml:
                provenance[section][field] = "yaml"
                continue
            provenance[section][field] = "default"
    return provenance


@router.put("/config")
async def put_config(payload: dict, request: Request) -> dict:
    """Apply runtime-mutable config updates.

    Accepts JSON like {"queue": {"grace_seconds": 60}, "monitor": {...}}.
    Boot sections (server, storage, runtime, ui) -> HTTP 403; restart required.
    Unknown sections -> HTTP 400.
    Mutations are persisted to runtime_config.yaml so they survive restart.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be JSON object")

    boot_attempts = set(payload.keys()) & BOOT_SECTIONS
    if boot_attempts:
        raise HTTPException(
            status_code=403,
            detail=(
                f"sections {sorted(boot_attempts)} are BOOT-ONLY; restart manager "
                "to change (prevents binary-swap attack class)"
            ),
        )

    unknown = set(payload.keys()) - RUNTIME_SECTIONS
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
    _invalidate_fastlane = getattr(mgr, "invalidate_fastlane", None)
    if _invalidate_fastlane is not None:
        _invalidate_fastlane()

    # Persist so edits survive restart --
    # sparse, only the sections/fields THIS request actually touched (the
    # raw payload, already validated above as part of building new_runtime,
    # not new_runtime itself, which is the full merged state).
    # ROLLBACK SAFETY: never persist `container_name: null`.
    # An older build's FastLaneRule uses extra="forbid", so an unknown key makes
    # it reject the whole section and boot Fast Lane DISABLED -- and deployments
    # can need a rollback step. A note would tell the operator to expect
    # breakage; omitting the key means there is none. An address-only rule is
    # written without any extra key.
    #
    # ⛔ THE PERSIST RUNS HERE, IMMEDIATELY AFTER THE APPLY, AND MUST STAY
    # ABOVE EVERY `await` IN THIS FUNCTION.
    # This handler is `async`, so any statement between `mgr.runtime = ...`
    # and this call sits in a window where a second concurrent PUT can run.
    # This function has a single suspension point, the fastlane resolve pass
    # below; with no await before it, apply-then-persist is atomic BY CONSTRUCTION --
    # a property that needs no upkeep as long as no await precedes the persist.
    # If the persist trailed the resolve pass's suspension point, two overlapping
    # PUTs could suspend there,
    # resume in either order, and leave the live runtime disagreeing with the
    # persisted file -- both requests returning 200, and the next restart
    # silently adopting the value the operator watched LOSE.
    # Only ONE of the two PUTs has to suspend for that to happen: the other
    # runs to completion inside the first one's suspension.
    # Writing before any await preserves that property. If you add an
    # await between the apply above and this line, you reintroduce the race.
    payload = _strip_null_container_names(payload)
    save_runtime_config(mgr.boot.storage.state_db_path, payload)

    # lint_rules (fastlane.py) reports duplicate address entries as a
    # configuration warning: they are reported rather than silently
    # resolved, never rejected. Wired here as a NON-FATAL channel --
    # a 400 would contradict that, and a scheduling-preference lint
    # must never block an otherwise-valid config write. (The same lint also
    # runs at boot, from the structural-only call in __main__.py; this is the
    # PUT-path call.)
    #
    # Gated on `"fastlane" in payload`, not run on every PUT: an ungated
    # lint would re-report the SAME already-existing duplicate on every
    # unrelated PUT forever, training operators to ignore the warning
    # channel this lint exists to create. Gated, it fires at the moment a
    # duplicate is introduced -- which is what "reported" means. The gap
    # this leaves (a duplicate baked into the shipped yaml, never PUT) is
    # covered separately by the boot-time lint in __main__.py.
    #
    # census= wired here, not at boot -- the
    # boot-time lint call (__main__.py, right after cfg.split()) runs
    # before a TurbohaulManager instance exists, so there is no live
    # census to pass there. `mgr._fastlane_census`
    # is keyed by NORMALIZED address (see manager.py), matching what
    # lint_rules' own census-membership check compares against.
    fastlane_warnings: list[str] = []
    if "fastlane" in payload:
        # A resolve pass must run BEFORE lint
        # here, or a brand-new rule saved in THIS SAME payload is judged
        # against a cache that has never heard of it and gets reported as
        # unresolvable on its first save. Reached through the manager's own
        # resolver (getattr, below), not a module import, so this file never
        # depends on fastlane_resolve at import time -- a module-level import
        # would make every route in this file depend on that module being
        # importable, not just this one gated branch.
        # NOTE: this pass does not touch mgr._fastlane_census.
        # FastLaneNameResolver writes only its own per-name address
        # cache, so a PUT-time resolve cannot stop a brand-new address
        # from being reported "never observed" -- the census is
        # populated by requests arriving at admission, and nothing
        # else populates it.
        #
        # WHAT THE PASS IS FOR: it REFRESHES THE ADDRESS CACHE
        # that lint reads through resolve_name= below. Without the
        # refresh, a container_name rule saved in THIS payload is
        # linted against a cache that has never heard of it and is
        # reported as unresolvable on its own first save. (The cache is
        # the resolver's own per-name address cache, as noted above;
        # it is not the census.)
        try:
            # Integration: use the manager's OWN resolver, do not construct one.
            # The resolver class is FastLaneNameResolver(names_fn);
            # it takes a callable, not a manager
            # (see fastlane_resolve.py).
            # Constructing one here
            # would start from an EMPTY cache and throw its work away --
            # the instance owns the generation counter the rule-table cache
            # keys on. If the resolver is absent (a manager built without the
            # lifespan, as in tests), skip rather than fail the write.
            resolver = getattr(mgr, "_fastlane_resolver", None)
            if resolver is not None:
                await resolver.resolve_once()
        except Exception:
            # Same non-fatal doctrine as the lint call itself: a scheduling
            # preference (and the resolve pass that keeps its address cache fresh)
            # must never block an otherwise-valid config write. Falls back
            # to linting against the last-known census -- identical to this
            # function's behavior without the resolve pass -- rather than 500ing the PUT.
            log.exception(
                "fastlane census resolve pass failed before lint; linting "
                "against last-known census instead of failing the write"
            )
        # resolve_name= must be passed here, or the collision check
        # (address vs resolved name), the shared-prefix check and the unresolved-name
        # warning would all be unreachable in the shipped product -- three guarded
        # blocks that only tests could enter. The resolver is already in hand
        # from the refresh above; this is the one line that makes them real.
        # getattr on the METHOD too, not just the resolver: a lint is a
        # diagnostic and must never take down the write it is diagnosing. A
        # resolver that is present but does not expose addresses_for (a
        # partially-constructed one, or a future shape change) would otherwise
        # raise HERE, outside the try/except above, and 500 an
        # otherwise-valid config write. Degrade to no-resolver instead.
        _resolver = getattr(mgr, "_fastlane_resolver", None)
        _resolve_name = getattr(_resolver, "addresses_for", None) if _resolver else None
        fastlane_warnings = lint_rules(
            new_runtime.fastlane.rules,
            census=mgr._fastlane_census,
            resolve_name=_resolve_name,
        )
        for warning in fastlane_warnings:
            log.warning("fastlane config lint: %s", warning)

    return {
        "status": "ok",
        "restart_required": False,
        "applied_sections": sorted(payload.keys()),
        "current": new_runtime.model_dump(mode="json"),
        "warnings": fastlane_warnings,
    }
