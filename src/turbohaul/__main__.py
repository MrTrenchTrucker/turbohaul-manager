"""Turbohaul-Manager CLI entry.

Loads /etc/turbohaul/turbohaul.yaml (overridable via --config / TURBOHAUL_CONFIG_PATH),
applies TURBOHAUL_* env overrides per config.apply_env_overrides, and starts uvicorn.

For container deployment where binding 0.0.0.0 is needed, pass --allow-public-bind
or TURBOHAUL_ALLOW_PUBLIC_BIND=1. The yaml ServerConfig still validates as 127.0.0.1
(the documented loopback default); the public-bind override only changes the uvicorn host argument,
not the loaded BootConfig.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
import yaml

from pydantic import ValidationError

from turbohaul.api.config_put import (
    RUNTIME_SECTIONS,
    _runtime_override_path,
    compute_config_provenance,
    load_runtime_override,
)
from turbohaul.api.main import create_app
from turbohaul.config import (
    MAX_FASTLANE_RULES,
    FastLaneConfig,
    TurbohaulConfig,
    apply_env_overrides,
    compute_shipped_baseline,
    load_config_yaml,
    migrate_legacy_fastlane_key,
)
from turbohaul.fastlane import lint_rules


log = logging.getLogger("turbohaul.main")


# uvicorn's access log floods docker-logs with ~1/sec health-poll
# `GET /status ... 200` lines (~22:1 vs the real decision logs), burying the
# per-turn decisions. Drop access records whose request path is EXACTLY a health
# endpoint. Safe-degrade: any unexpected record shape returns True (record kept),
# so this can never crash the log path or suppress a real request.
class _HealthPollAccessFilter(logging.Filter):
    _HEALTH_PATHS = frozenset({"/status", "/health", "/healthz"})

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn access record: args = (client_addr, method, full_path,
        # http_version, status_code) -> index 2 is the request target.
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True
        target = args[2]
        if not isinstance(target, str):
            return True
        path = target.split("?", 1)[0]  # drop ?query before exact-matching
        return path not in self._HEALTH_PATHS


# The manager's own ~1Hz LiveSlotsPoller/ResidentSlotsPoller
# GET /slots poll rides httpx, whose own request-complete log
# call (httpx/_client.py, `logger = logging.getLogger("httpx")`) fires an INFO
# line for EVERY completed transaction -- by far the largest share of the log in a busy
# container. Filter the POLL, not the LOGGER: `getLogger("httpx").setLevel(
# WARNING)` would "work" and would ALSO blind us to every other genuine httpx
# client error in the process (chat_completion.py's sidecar forwarder among
# them) -- the exact alerting-reports-success-while-failing trap that a
# blanket level change walks into. Mirrors _HealthPollAccessFilter's shape:
# exact-match on the identifying fields, safe-degrade (any unrecognized
# record shape or a non-200 outcome returns True = keep), so this can never
# crash the log path or suppress a real request OR a real poll failure.
class _SlotsPollHttpxFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # httpx's request-complete log call (httpx/_client.py, two call sites):
        #   logger.info('HTTP Request: %s %s "%s %d %s"',
        #                method, url, http_version, status_code, reason_phrase)
        # args = (method, url, http_version, status_code, reason_phrase);
        # url is an httpx.URL object (NOT a plain str) -- .path already strips
        # any query string, same normalization _HealthPollAccessFilter had to
        # do by hand via .split("?", 1)[0]. status_code is an int, not a str
        # -- a string comparison would silently never match and ship a filter
        # that drops nothing.
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True
        method, url, _http_version, status_code, _reason_phrase = args[:5]
        if method != "GET":
            return True
        if getattr(url, "path", None) != "/slots":
            return True
        if status_code != 200:
            return True
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="turbohaul-manager",
        description="Ollama-shape inference manager.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.environ.get("TURBOHAUL_CONFIG_PATH", "/etc/turbohaul/turbohaul.yaml")
        ),
        help="Path to turbohaul.yaml (default /etc/turbohaul/turbohaul.yaml).",
    )
    p.add_argument(
        "--allow-public-bind",
        action="store_true",
        default=os.environ.get("TURBOHAUL_ALLOW_PUBLIC_BIND") == "1",
        help=(
            "Override uvicorn host to 0.0.0.0 (container public bind). "
            "The default is 127.0.0.1; enable only inside an "
            "explicit network-policy boundary (e.g., a container with port mapping)."
        ),
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("TURBOHAUL_LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    return p


def _cap_fastlane_rules(candidate: dict) -> dict:
    """Truncate a persisted fastlane rule list to the documented rule cap
    BEFORE it ever reaches FastLaneConfig's own cap validator -- an
    over-length list where every individual rule is otherwise well-formed
    would still fail validation on count alone, and without this the
    per-rule salvage below can't help (it never shortens the list, so a
    too-long list stays too-long and the whole feature would be disabled
    over a length problem, not a content one). Keeps the FIRST
    MAX_FASTLANE_RULES entries -- rule_index is positional, so trimming the
    TAIL is the only truncation that cannot re-rank a kept rule, same
    precedent as the per-rule salvage's own index-preservation rule below.
    A scheduling preference must never take the whole manager down over
    this; log at WARNING (recoverable, feature stays on), naming both the
    original count and the cap, mirroring the coercion notes' own severity.
    """
    rules = candidate.get("rules")
    if not isinstance(rules, list) or len(rules) <= MAX_FASTLANE_RULES:
        return candidate
    log.warning(
        "persisted fastlane config had %d rules, exceeding the %d-rule cap; "
        "keeping the %d highest-priority and dropping the lowest-priority %d",
        len(rules), MAX_FASTLANE_RULES, MAX_FASTLANE_RULES,
        len(rules) - MAX_FASTLANE_RULES,
    )
    return {**candidate, "rules": rules[:MAX_FASTLANE_RULES]}


def _salvage_fastlane_rules(candidate: dict) -> "tuple[dict | None, list[str]]":
    """Make a persisted fastlane block loadable without dropping any rule.

    Returns (salvaged_candidate, notes) or (None, []) when the LIST ITSELF is
    malformed (not a list, or an entry that is not even a dict -- there is no
    rule-shaped thing to preserve a slot for). The rule LIST LENGTH AND ORDER
    ARE PRESERVED EXACTLY -- rule_index is positional, so removing an entry
    would re-rank every rule below it.

    A BLANK IDENTITY -- `address` and `container_name`
    both empty/whitespace/absent, whether that arrived directly (a bare
    `container_name: ""`) or was MANUFACTURED by the coercion arm below
    stripping a blank `address` down to `""` -- would bail this WHOLE
    function, and the callers' only recourse to an unsalvageable section is
    dropping it entirely. With the NEITHER arm blank-aware (config.py),
    an operator leaving a rule half-filled becomes exactly that:
    one blank identity taking every VALID sibling rule down with it, which is
    the precise failure "taking every VALID sibling rule with it and booting
    the feature disabled" this whole salvage mechanism exists to prevent --
    just for a different input shape than the one it guards.

    The approach is REPLACEMENT, not removal: a blank-identity rule is swapped IN
    PLACE for an inert placeholder -- same slot, so rule_index for every
    OTHER rule is untouched (the length/order guarantee above still holds
    exactly) -- carrying a synthetic `container_name` that cannot collide
    with a real docker container, so it resolves to nothing, fail-closed,
    the same shape the address-coercion arm below already uses for "we don't
    know what this was, so make it inert rather than dangerous." Its `label`
    says so too, which the Fast Lane rules table already renders per-rule --
    the operator sees which slot is broken without reading the boot log.

    Scope, deliberately narrow: ONLY a blank identity gets this treatment.
    Any OTHER unsalvageable shape (both fields set, an unexpected extra
    field, ...) still bails the whole list exactly as before -- each of
    those is its own failure class and widening past blank-identity is not
    warranted, as no other input shape is known to need it.

    Shared by BOTH callers -- the persisted runtime_config.yaml override
    (main(), machine-written by PUT) and the bind-mounted turbohaul.yaml
    (_load_boot_config_with_fastlane_salvage, hand-typed) -- deliberately.
    Both already share the coercion arm below; this one additional failure
    class is not being treated MORE gently for a human's file, because
    replacement here never discards information the section's OWNER
    supplied -- the sibling rules keep working, this one slot is reported
    (log + UI), and the alternative (fail loud on the hand-typed file
    specifically) reopens the exact crash-loop
    _load_boot_config_with_fastlane_salvage's own docstring documents: a
    scheduling preference permanently taking down a manager serving many
    models, with no way to fix it from inside a container whose config is
    mounted read-only.
    """
    from turbohaul.config import FastLaneRule

    rules = candidate.get("rules")
    if not isinstance(rules, list):
        return None, []

    out: list = []
    notes: list[str] = []
    for idx, raw in enumerate(rules):
        if not isinstance(raw, dict):
            return None, []
        try:
            FastLaneRule(**raw)
            out.append(raw)
            continue
        except Exception:
            pass
        # coerce a non-parsing address into container_name, same position
        coerced = dict(raw)
        bad_addr = coerced.pop("address", None)
        if bad_addr is not None and not coerced.get("container_name"):
            coerced["container_name"] = str(bad_addr).strip()
            try:
                FastLaneRule(**coerced)
            except Exception:
                pass
            else:
                notes.append(
                    f"rule index {idx} had an address that is not a single IP "
                    f"({bad_addr!r}); read as a container_name instead. It matches "
                    "nothing until it resolves -- fix it in Settings."
                )
                out.append(coerced)
                continue

        if (
            not (raw.get("address") or "").strip()
            and not (raw.get("container_name") or "").strip()
        ):
            notes.append(
                f"rule index {idx} sets neither address nor container_name to "
                "anything but blank/whitespace (even after attempting to coerce "
                f"a usable identity out of it: raw content {raw!r}); replaced "
                "with an inert placeholder so every OTHER rule still loads. It "
                "matches nothing until fixed -- fix it in Settings."
            )
            out.append({
                "container_name": f"__unsalvageable_fastlane_rule_{idx}__",
                "label": f"INVALID at boot (rule {idx}): blank identity -- see "
                         "server log, fix in Settings",
            })
            continue

        return None, []

    salvaged = {**candidate, "rules": out}
    try:
        FastLaneConfig(**salvaged)
    except Exception:
        return None, []
    return salvaged, notes


def _is_fastlane_only_error(exc: ValidationError) -> bool:
    """True only when EVERY complaint in `exc` is inside the fastlane section.

    Deliberately all-or-nothing. The salvage below exists to stop a scheduling
    preference killing the manager -- it must NOT become a general "boot
    anyway" path. A bad `storage` or `runtime` section is boot config: it has
    no salvage, no safe default, and failing loudly on it is correct.
    """
    errors = exc.errors()
    return bool(errors) and all(
        e.get("loc") and e["loc"][0] == "fastlane" for e in errors
    )


def _load_boot_config_with_fastlane_salvage(path):
    """Load turbohaul.yaml, giving its fastlane section the SAME cap-and-
    salvage treatment the persisted runtime_config.yaml already gets.

    Returns (config, fastlane_error_or_None).

    Why this exists: the two files travel different paths into the same
    validator, and only one of them is guarded. A persisted block containing
    a bare container name in the `address` field, or an eleventh rule, is
    capped, coerced, logged and served. The very same content in the operator's
    hand-authored turbohaul.yaml would reach `TurbohaulConfig(**data)` unguarded,
    raise out of `main()`, and exit the process -- and because a compose
    file typically restarts the container and mounts that file read-only, nothing inside
    the container could fix it. A scheduling preference would take down a manager
    serving many models, permanently, which is the exact outcome the persisted
    path's own comments say must never happen.

    Worse, `enabled: false` does not help: both validators run at model
    construction, so an operator who has TURNED FAST LANE OFF but left rules in
    the file still gets a dead manager.

    The bind-mounted file is the one a human types by hand, with no UI
    validating it first -- the PUT path rejects both bad shapes at the door --
    so it is if anything the more likely of the two to be wrong.
    """
    try:
        return load_config_yaml(path), None
    except ValidationError as exc:
        if not _is_fastlane_only_error(exc):
            raise
        # Bound to a name that OUTLIVES the except block: `exc` itself is
        # unbound on exit, and a bare `raise` below would have no active
        # exception to re-raise.
        original = exc

    data = migrate_legacy_fastlane_key(yaml.safe_load(path.read_text()) or {})
    section = data.get("fastlane")
    if not isinstance(section, dict):
        raise original

    salvaged, notes = _salvage_fastlane_rules(_cap_fastlane_rules(section))
    if salvaged is not None:
        for note in notes:
            log.warning("fastlane config coerced at boot from %s: %s", path, note)
        return TurbohaulConfig(**{**data, "fastlane": salvaged}), None

    # Unsalvageable: boot WITHOUT the fastlane section rather than not at all,
    # matching the persisted path's disable-and-log outcome. The operator loses
    # a scheduling preference and is told so twice -- once in the log, once in
    # the Fast Lane tab's banner, which renders this same string.
    error = str(_first_fastlane_error(section))
    log.error(
        "fastlane config in %s is invalid and could not be salvaged; booting "
        "with fastlane DISABLED so the manager still serves: %s",
        path, error,
    )
    return TurbohaulConfig(**{k: v for k, v in data.items() if k != "fastlane"}), error


def _first_fastlane_error(section: dict) -> str:
    """Re-validate the section alone so the operator-facing message names the
    fastlane problem, not whatever TurbohaulConfig says about the whole file."""
    try:
        FastLaneConfig(**section)
    except ValidationError as e:
        return str(e)
    return "unknown fastlane configuration error"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper() if args.log_level != "trace" else "DEBUG",
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if not args.config.exists():
        log.error("config not found: %s", args.config)
        return 2

    log.info("loading config: %s", args.config)
    # The raw shipped yaml, kept around (never mutated)
    # for the provenance endpoint and the divergence log below -- a second,
    # cheap parse of the same small file, not a second source of truth. The
    # boot-effective config below is still built the normal way, from
    # load_config_yaml/apply_env_overrides; this is read-only bookkeeping
    # alongside it, never a substitute input.
    # The same legacy-key migration load_config_yaml applies to the
    # authoritative parse below -- this is a second read of the SAME file, so
    # an old "fastline:" block must resolve the same way here too, or the
    # divergence log a few lines down would misreport a shipped-yaml value
    # under "fastlane" as "no shipped default" just because this bookkeeping
    # copy never learned the section moved.
    shipped_yaml_raw = migrate_legacy_fastlane_key(yaml.safe_load(args.config.read_text()) or {})
    _boot_cfg, _boot_fastlane_error = _load_boot_config_with_fastlane_salvage(args.config)
    cfg = apply_env_overrides(_boot_cfg)
    # Load runtime config override (persisted by PUT /api/config) on top
    # of YAML defaults so runtime edits survive restart/recreate.
    fastlane_config_error: str | None = _boot_fastlane_error
    runtime_override = load_runtime_override(cfg.storage.state_db_path)
    if runtime_override:
        log.info("loading runtime config override from %s",
                 _runtime_override_path(cfg.storage.state_db_path))
        cfg_data = cfg.model_dump()
        for section, values in runtime_override.items():
            # Boot-section guard: only merge runtime-mutable sections. A tampered
            # runtime_config.yaml carrying boot sections (e.g. storage or
            # runtime.llama_server_binary) must NOT override boot config at
            # boot -- that would re-open the binary-swap attack
            # that PUT /api/config blocks with HTTP 403.
            if section in RUNTIME_SECTIONS and section in cfg_data and isinstance(values, dict):
                if section == "fastlane":
                    # Isolated fail-loud parse: a scheduling preference must
                    # never take down a manager serving many models. A bad
                    # persisted fastlane block is rejected on its own, logged
                    # once, and the feature boots OFF (the code default) --
                    # it does not abort the whole runtime_config.yaml merge.
                    candidate = _cap_fastlane_rules({**cfg_data[section], **values})
                    try:
                        FastLaneConfig(**candidate)
                    except ValidationError as e:
                        # ONE bad rule must not take the feature
                        # down. Pydantic rejects the SECTION, not the rule, so
                        # a single typo'd address would otherwise flip
                        # enabled:true to the code default False and take every
                        # GOOD rule with it -- signalled by one log.error. That
                        # is disproportionate and it fails quiet, which is the
                        # failure class this coercion exists to prevent. The file's
                        # own migrate_legacy_fastlane_key sets the coerce
                        # precedent.
                        #
                        # ⛔ THE RULE IS NEVER REMOVED FROM THE LIST. rule_index
                        # is POSITIONAL (compile_fastlane enumerates), so
                        # dropping a bad rule at [0] would silently PROMOTE
                        # every rule below it -- a priority change caused by a
                        # validation error, which is exactly what this
                        # mitigation is meant to avoid.
                        #
                        # A bad `address` is coerced to `container_name`
                        # holding the same text: valid, position-preserving,
                        # and FAIL-CLOSED (it resolves to nothing unless it is
                        # genuinely a container, in which case the coercion is
                        # the correct reading of what the operator meant). It
                        # then announces itself through the fast-lane lint's
                        # "resolves to NO addresses" warning.
                        salvaged, notes = _salvage_fastlane_rules(candidate)
                        if salvaged is None:
                            fastlane_config_error = str(e)
                            log.error(
                                "persisted fastlane config rejected, booting with "
                                "fastlane disabled: %s", e,
                            )
                            continue
                        for note in notes:
                            log.warning("fastlane config coerced at boot: %s", note)
                        candidate = salvaged
                    cfg_data[section] = candidate
                else:
                    cfg_data[section] = {**cfg_data[section], **values}
        cfg = type(cfg)(**cfg_data)
    boot, runtime = cfg.split()

    # Lint the FINAL merged fastlane rule list --
    # not just the persisted-override branch above (which only fires when a
    # runtime_config.yaml override touches "fastlane"). A duplicate baked
    # into the shipped turbohaul.yaml, with no runtime override at all,
    # would never reach that branch; linting the post-split() `runtime`
    # covers shipped-yaml, env, and persisted-override sources uniformly,
    # exactly once. try/except-guarded per boot posture: a scheduling
    # preference must never take down a manager serving many models --
    # never fail boot on lint, only log.
    try:
        for _fastlane_warning in lint_rules(runtime.fastlane.rules):
            log.warning("fastlane config lint: %s", _fastlane_warning)
    except Exception:
        log.exception("fastlane config lint failed at boot (non-fatal, continuing)")

    # The one-turn fairness window's default equals its own upper
    # bound (~1 hour); an operator running on the code default rather than an
    # explicit choice should see that plainly at boot, not only by reading
    # GET /api/config's "_provenance" field after the fact.
    try:
        _fastlane_provenance = compute_config_provenance(
            shipped_yaml_raw, boot.storage.state_db_path
        )
        if _fastlane_provenance.get("fastlane", {}).get("max_normal_wait_s") == "default":
            log.info(
                "fastlane.max_normal_wait_s using the code default (%.0fs) -- "
                "no shipped yaml, env, or persisted override sets it",
                runtime.fastlane.max_normal_wait_s,
            )
    except Exception:
        log.exception("fastlane max_normal_wait_s provenance check failed at boot (non-fatal)")

    # Log every key whose fully-resolved effective
    # value (shipped yaml + env override + persisted runtime override, the
    # merge above) differs from what a FRESH deployment would get (shipped
    # yaml + code default only). This is the "silent config drift" case,
    # made visible at the one point where an operator is most
    # likely to be watching -- boot.
    _baseline = compute_shipped_baseline(shipped_yaml_raw)
    _effective = {**boot.model_dump(mode="json"), **runtime.model_dump(mode="json")}
    for _section, _fields in _effective.items():
        _base_fields = _baseline.get(_section, {})
        for _field, _value in _fields.items():
            _base_value = _base_fields.get(_field)
            if _value != _base_value:
                log.info(
                    "config diverges from shipped default: %s.%s effective=%r shipped_default=%r",
                    _section, _field, _value, _base_value,
                )

    bind_host = boot.server.host
    if args.allow_public_bind:
        bind_host = "::"  # noqa: S104 -- dual-stack container bind override (IPv6)
        log.warning(
            "--allow-public-bind in effect: uvicorn binding :: dual-stack "
            "(BootConfig.server.host=%s preserved)",
            boot.server.host,
        )

    log.info(
        "ready: %s:%d (ui.enabled=%s ui.static_path=%s)",
        bind_host,
        boot.server.port,
        boot.ui.enabled,
        boot.ui.static_path,
    )

    app = create_app(boot, runtime, fastlane_config_error=fastlane_config_error, shipped_yaml_raw=shipped_yaml_raw)
    # Silence /status /health /healthz poll spam on the access log.
    # Registered on the logger (not a handler); survives uvicorn's dictConfig
    # (disable_existing_loggers=False preserves logger-attached filters).
    logging.getLogger("uvicorn.access").addFilter(_HealthPollAccessFilter())
    # Silence the /slots poll's INFO flood on httpx's own
    # request logger, without touching its level (a real httpx error must
    # still log -- see _SlotsPollHttpxFilter's docstring).
    logging.getLogger("httpx").addFilter(_SlotsPollHttpxFilter())
    _lvl = args.log_level if args.log_level != "trace" else "debug"
    if args.allow_public_bind:
        # httpx resolves the container IPv6 (AAAA) first and, unlike curl, does
        # NOT fall back to IPv4 -> an IPv4-only bind gives ConnectError. A bare "::"
        # uvicorn host goes IPv6-ONLY under docker (breaks the IPv4 host-port forward),
        # so bind an explicit dual-stack socket (IPV6_V6ONLY=0) serving BOTH families.
        import socket as _socket

        _sock = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
        _sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        _sock.setsockopt(_socket.IPPROTO_IPV6, _socket.IPV6_V6ONLY, 0)
        _sock.bind(("::", boot.server.port))
        uvicorn.Server(
            uvicorn.Config(app, log_level=_lvl, access_log=True)
        ).run(sockets=[_sock])
    else:
        uvicorn.run(
            app,
            host=bind_host,
            port=boot.server.port,
            log_level=_lvl,
            access_log=True,
            # No --reload in production; restart the process to pick up changes.
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
