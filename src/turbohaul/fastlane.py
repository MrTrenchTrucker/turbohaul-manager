"""Fast Lane: two-level request-priority matcher.

Level 1 = source IP address (an ordered rule list; list index is priority).
Level 2 = tag, ranked 1-5 within one address. Effective sort key on the pop
path is (rule index, tag rank, arrival order) -- built here, applied by
queue.py's _pick_fastlane_locked.

This module is pure and side-effect-free: it never touches client_meta,
never talks to the queue, and never raises out of match_fastlane. Off by
default upstream (RuntimeConfig.fastlane may not exist yet, or
.enabled=False) -- callers degrade to "no match" via getattr, not by
importing anything conditional here.
"""
from __future__ import annotations

import dataclasses
import ipaddress
from typing import Optional

from .kv_classify import (
    CLASSES,
    CLASS_COMPRESSION,
    CLASS_CURATOR,
    CLASS_MAIN,
    CLASS_SUB_AGENT,
    CLASS_USER_MESSAGE,
    _class_from_label,
)

# One past the max configured rank (1-5); an unset/unmapped tag sorts after
# every ranked one, so an all-unset rank table makes every request from that
# address equal (first-come-first-served). Single owner: any other module
# needing this value should import it from here rather than redefining it,
# to avoid cross-module drift.
UNRANKED = 6

# "unclassified" is a Fast Lane tag, NOT a kv_classify class -- it has no
# corresponding CLASS_* constant. It is the fold target for any class that
# has no dedicated Fast Lane rank field (today: CLASS_USER_MESSAGE, and any
# future class kv_classify adds that this module hasn't been taught yet).
TAG_UNCLASSIFIED = "unclassified"

# Every kv_classify class maps to exactly one Fast Lane tag. CLASS_USER_MESSAGE
# folds to "unclassified" by design decision: the `role` field reaching
# _class_from_label is caller-controlled (any client can send
# {"role": "user-message"} or an arbitrary POLICIES-registered role), and the
# tag-rank config model only has five rank fields with a closed schema --
# adding a sixth for a class carrying no real traffic today is not worth the
# churn. The lookup below is total via .get(), so this map can never raise
# regardless of what class resolves.
_TAG_BY_CLASS = {
    CLASS_MAIN: "main",
    CLASS_CURATOR: "curator",
    CLASS_COMPRESSION: "compression",
    CLASS_SUB_AGENT: "sub_agent",
    CLASS_USER_MESSAGE: TAG_UNCLASSIFIED,
}
# Coverage assert, not equality: this fires only if kv_classify grows a class
# this module hasn't been taught to fold -- it can never fire spuriously
# because every current CLASSES member has an explicit entry above.
assert set(_TAG_BY_CLASS) == set(CLASSES), (
    f"_TAG_BY_CLASS {sorted(_TAG_BY_CLASS)} != CLASSES {sorted(CLASSES)} — every "
    "kv_classify class must map to exactly one Fast Lane tag."
)


@dataclasses.dataclass(frozen=True)
class CompiledRule:
    """One rule, ready to match, identified by exactly one of `address` or
    `container_name` (config.py's FastLaneRule enforces this at validation
    time). `raw_address` is the operator's original identifying string, kept
    verbatim (the UI shows what he clicked) -- the address he typed for an
    address rule, or the container_name he typed for a name rule.

    `address` is the parsed single-host object for an address rule, None for
    a name rule. `match_addresses` is what `match_fastlane` actually tests
    membership against: a one-element frozenset for an address rule, or the
    name resolver's current (possibly empty, possibly multi-address) set for
    a name rule."""

    index: int
    raw_address: str
    address: "ipaddress.IPv4Address | ipaddress.IPv6Address | None"
    container_name: "str | None"
    match_addresses: "frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address]"
    label: str
    tag_ranks: dict  # tag -> rank int (1-5), only set tags present


@dataclasses.dataclass(frozen=True)
class FastLaneMatch:
    """Result of matching one request against a compiled table.

    `effective_tag` is the class-derived tag used to resolve `rank` -- it is
    for display/logging ONLY and must never be written back into
    client_meta (writing it would move the KV bin identity, see the module
    docstring above). `matched_by` is "address" or "name" -- the
    dimension that decided this match, so logs/labels can say honestly which
    one fired rather than inferring it from raw_address's shape."""

    rule_index: int
    raw_address: str
    label: str
    effective_tag: str
    rank: int
    # defaults to "address", the original match type -- every
    # FastLaneMatch(...) construction that predates `container_name` rules
    # is implicitly an address match; only match_fastlane itself ever needs
    # to pass this explicitly.
    matched_by: str = "address"  # "address" | "name"


def normalize_address(raw: str) -> "ipaddress.IPv4Address | ipaddress.IPv6Address":
    """Parse an address string into a comparable ipaddress object.

    - Strips any `%zone` suffix (`fe80::1%eth0` -> `fe80::1`) BEFORE parsing,
      since ipaddress preserves scope_id in equality (`fe80::1%eth0` !=
      `fe80::1` even though they're the same host for matching purposes).
    - Unwraps IPv4-mapped IPv6 (`::ffff:192.0.2.10`) to its IPv4 form, so a
      rule stored as the plain dotted quad matches the mapped form observed
      on the published-port path, and vice versa.
    - Raises ValueError on anything that isn't a single exact host (CIDR,
      prefix, or unparseable) -- exact single hosts only.
    """
    zone_stripped = raw.split("%", 1)[0]
    addr = ipaddress.ip_address(zone_stripped)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def resolve_rank(client_meta: Optional[dict], tag_ranks: dict) -> tuple[str, int]:
    """Resolve (effective_tag, rank) for a request's client_meta against one
    rule's tag_ranks. Goes through kv_classify._class_from_label (never raw
    booleans) -- the is_* labels are not mutually exclusive (a curator
    carries both is_curator=True and is_sub_agent=True), and this is the
    same resolver the KV bin uses, so the rank an operator sees can never
    disagree with the class that actually governs the request's KV bin."""
    cls = _class_from_label(client_meta)
    tag = _TAG_BY_CLASS.get(cls, TAG_UNCLASSIFIED)
    rank = tag_ranks.get(tag, UNRANKED)
    return tag, rank


def _tag_ranks_of(rule) -> dict:
    tag_ranks_obj = getattr(rule, "tag_ranks", None)
    if tag_ranks_obj is None:
        return {}
    if isinstance(tag_ranks_obj, dict):
        return {k: v for k, v in tag_ranks_obj.items() if v is not None}
    # Pydantic model instance (FastLaneTagRanks) -- read its fields.
    return {
        k: v
        for k, v in getattr(tag_ranks_obj, "__dict__", {}).items()
        if v is not None
    }


def compile_fastlane(rules: list, resolve_name=None) -> list[CompiledRule]:
    """Compile a validated rule list (config-layer FastLaneRule-shaped
    models, or anything duck-typed the same way:
    .address/.container_name/.label/.tag_ranks) into CompiledRule objects.
    List index IS priority -- position is preserved. Invalid/unresolvable
    entries are skipped or degrade to an empty match set, never raised past
    this function -- validation-time rejection is lint_rules'/the config
    layer's job; this is the hot admission path and must degrade, not
    crash, on bad input that slipped through.

    `resolve_name`: an OPTIONAL zero-arg-per-call
    `name -> frozenset[address]` cache reader (fastlane_resolve.py's
    `FastLaneNameResolver.addresses_for`, or any callable shaped the same
    way) -- CACHE ONLY, never performs I/O itself, so calling it here never
    adds DNS to whatever calls compile_fastlane. Defaults to None so any
    EXISTING call site (e.g. manager.py's `compile_fastlane(cfg.rules)`)
    keeps compiling exactly as it did before name rules existed: a
    container_name rule compiled with no resolver degrades to an empty,
    fail-closed match set (same posture as a name genuinely unresolved)
    rather than erroring.
    """
    compiled = []
    for idx, rule in enumerate(rules):
        container_name = getattr(rule, "container_name", None)
        label = getattr(rule, "label", "") or ""
        tag_ranks = _tag_ranks_of(rule)
        if container_name:
            match_addresses = (
                frozenset(resolve_name(container_name))
                if resolve_name is not None
                else frozenset()
            )
            compiled.append(
                CompiledRule(
                    index=idx,
                    raw_address=container_name,
                    address=None,
                    container_name=container_name,
                    match_addresses=match_addresses,
                    label=label,
                    tag_ranks=tag_ranks,
                )
            )
            continue
        try:
            addr = normalize_address(rule.address)
        except (ValueError, AttributeError, TypeError):
            continue
        compiled.append(
            CompiledRule(
                index=idx,
                raw_address=rule.address,
                address=addr,
                container_name=None,
                match_addresses=frozenset({addr}),
                label=label,
                tag_ranks=tag_ranks,
            )
        )
    return compiled


def match_fastlane(
    compiled_rules: list[CompiledRule],
    ip: Optional[str],
    client_meta: Optional[dict],
) -> Optional[FastLaneMatch]:
    """Match one request's (ip, client_meta) against a compiled table.
    Returns None on no match, feature off (empty table), or any internal
    failure to normalize the request's own ip -- never raises. Callers wrap
    this in a best-effort try/except regardless (belt + suspenders), since
    this is on the hot admission path.
    """
    if not compiled_rules or not ip:
        return None
    try:
        req_addr = normalize_address(ip)
    except ValueError:
        return None
    for rule in compiled_rules:
        if req_addr in rule.match_addresses:
            tag, rank = resolve_rank(client_meta, rule.tag_ranks)
            return FastLaneMatch(
                rule_index=rule.index,
                raw_address=rule.raw_address,
                label=rule.label,
                effective_tag=tag,
                rank=rank,
                matched_by="name" if rule.container_name is not None else "address",
            )
    return None


def _shares_docker_lease_prefix(
    a: "ipaddress.IPv4Address | ipaddress.IPv6Address",
    b: "ipaddress.IPv4Address | ipaddress.IPv6Address",
) -> bool:
    """True if `a` and `b` share a /24 (IPv4) or /64 (IPv6) prefix -- the
    granularity docker's own bridge/overlay subnets are typically cut at.
    Used only by lint_rules' (d) check below. Cross-family (v4 vs v6)
    never matches -- an IPv4 and an IPv6 address share no meaningful prefix
    regardless of their numeric values."""
    if type(a) is not type(b):
        return False
    prefix_len = 24 if isinstance(a, ipaddress.IPv4Address) else 64
    return b in ipaddress.ip_network(f"{a}/{prefix_len}", strict=False)


def lint_rules(
    rules: list, census: Optional[dict] = None, resolve_name=None
) -> list[str]:
    """Report-only warnings for a rule list. Never auto-fixes, never
    reorders, never picks a winner -- the project's doctrine is to fail loud so a
    problem lands in the logs and gets diagnosed, not silently resolved.

    The duplicate check compares NORMALIZED
    addresses, not raw strings -- match_fastlane itself normalizes both
    sides (this module's normalize_address), so two textually different
    rules that resolve to the same host (`192.0.2.10` vs
    `::ffff:192.0.2.10`) are the exact duplicate this function exists to
    catch; a raw string compare would pass right over it and the later
    rule's tag_ranks would stay silently dead. An address that fails to
    normalize (unparseable) falls back to comparing its own raw form --
    same behavior as a raw compare for anything already-invalid, which
    is compile_fastlane's problem to skip, not lint's to crash on.

    `census`, when passed, is keyed by
    NORMALIZED address (manager.py's `_fastlane_census`) -- the
    membership check below compares the rule's own normalized form against
    it, not the raw `addr`, or a legitimately-observed IPv4-mapped peer
    would get a false "never observed" warning.

    The container_name checks below are all DELIBERATELY zero
    new I/O and no new dependency -- this function stays pure and sync,
    called from both a no-event-loop boot path and an async route, and a
    diagnostic that can itself hang or fail is worse than one with a
    stated blind spot (the same doctrine already applied to the
    config_put fallback):

    (b) A label-vs-container_name disagreement check is deliberately NOT
        performed here. `label` is documented in config.py as "operator's
        own note" and the schema never asserts any relationship between it
        and the container name, so any label can legitimately sit beside any
        container name and such a check would only raise false warnings.
        A check founded on
        a relationship nothing asserts cannot be tuned into usefulness, only
        made quieter -- and every false warning here trains an operator to
        ignore the channel the unresolved-name warning depends on. The slot
        is kept reserved so the letters of the other checks stay stable in the
        discussion below.

    (c) an address-only rule whose address appears in the RESOLVED set of
        a DIFFERENT rule's container_name -- a real collision, one client
        claimed twice, once by name and once by a raw address. This reuses
        the exact same `resolve_name` cache-reader shape compile_fastlane
        takes (e.g. fastlane_resolve.FastLaneNameResolver.addresses_for) --
        CACHE ONLY, no lookup is triggered here either. Only runs when a
        caller passes `resolve_name`; omitted (as at the boot-time
        structural-only site, which predates any resolver) it silently
        skips this check, matching compile_fastlane's own degrade pattern.

    (d) an address-only rule whose address shares a docker-lease-sized
        prefix (/24 for IPv4, /64 for IPv6 -- `ipaddress` is already
        imported, no new dependency) with any address the resolver
        ALREADY holds for a DIFFERENT rule's container_name: a softer
        signal than (c) -- not necessarily the SAME address, just the same
        docker subnet -- warning that this looks like a docker-assigned
        lease that can roll on restart, and that container_name is
        preferred. Skipped for any address (c) already flagged exactly, to
        avoid double-warning the same rule. An off-network address (LAN,
        VPN, any static IP -- the operator's own fallback)
        numerically shares no prefix with anything docker resolves, so it
        stays silent BY CONSTRUCTION, not by a special case -- this is a
        requirement this check must preserve, not just incidentally true.

    *** TWO DISCLOSED BLIND SPOTS *** -- read both before
    trusting a clean lint as full coverage:

    1. An address-only rule whose address belongs to a container NO OTHER
       RULE NAMES cannot be checked here. Catching that needs address ->
       name resolution, which this function deliberately does not use: the
       names the Discovered list now shows come from a reverse lookup that
       is display only, confirmed by a forward lookup (see
       fastlane_resolve.py), and they never reach this lint. Such a rule
       can drift silently forever and this function will stay silent
       about it -- the census-membership check above is the closest
       existing signal, and it can only ever say "unobserved", never
       "observed, but under someone else's name".

    2. (d) is INERT on a fully address-keyed table -- i.e. it cannot
       diagnose the exact drifted state container_name rules exist to fix
       (a table whose rules are all address-only has nothing resolved
       for (d) to compare against); it can only prevent REGRESSION
       back into that state once at least one rule has been migrated to
       container_name. This is a fail-QUIET blind spot -- there is no
       warning that signals the check is currently inert; it
       has to be understood from the code, which is why it is written
       down here explicitly."""
    warnings: list[str] = []
    seen: dict[str, tuple[int, str]] = {}  # normalized addr -> (first idx, first raw addr)
    # Duplicate container_name check. A host can be named two ways: by address
    # (the duplicate-address loop further down covers that) or by
    # container_name (covered here). Without this loop a name pasted twice
    # would be DEAD and silent: the first rule claims every
    # address that name resolves to, the ordered scan in match_fastlane never
    # reaches the second, and the operator sees it sitting in the table with
    # ranks configured. Same failure as a duplicate address -- a rule the
    # screen shows and the matcher ignores -- so it gets the same message.
    #
    # Compared case-sensitively and unnormalized: docker container names are
    # case-sensitive and are handed to getaddrinfo verbatim, so "Gateway-Svc"
    # and "gateway-svc" are genuinely different names here and folding them
    # together would invent a duplicate that does not exist.
    seen_names: dict[str, int] = {}
    for idx, rule in enumerate(rules):
        name = getattr(rule, "container_name", None)
        if name:
            if name in seen_names:
                warnings.append(
                    f"duplicate container_name: rule {seen_names[name] + 1} and "
                    f"rule {idx + 1} both name {name!r} -- the second can never "
                    "match, since the first already claims every address that "
                    "name resolves to. Config error, not auto-resolved."
                )
            else:
                seen_names[name] = idx
    for idx, rule in enumerate(rules):
        addr = getattr(rule, "address", None)
        if addr is None:
            continue
        try:
            norm_addr = str(normalize_address(addr))
        except ValueError:
            norm_addr = addr
        if norm_addr in seen:
            first_idx, first_addr = seen[norm_addr]
            warnings.append(
                f"duplicate address: rule {first_idx + 1} ({first_addr!r}) and "
                f"rule {idx + 1} ({addr!r}) both target the same host -- "
                "config error, not auto-resolved."
            )
        else:
            seen[norm_addr] = (idx, addr)
        if census is not None and norm_addr not in census:
            warnings.append(
                f"rule {idx + 1} (address {addr!r}) has never been observed "
                "in the discovered-address census."
            )

    name_rule_addresses: dict[int, tuple[str, "frozenset"]] = {}
    for idx, rule in enumerate(rules):
        container_name = getattr(rule, "container_name", None)
        if not container_name:
            continue
        if resolve_name is not None:
            resolved = frozenset(resolve_name(container_name))
            name_rule_addresses[idx] = (container_name, resolved)
            # A name that resolves to NOTHING must WARN, not go silent.
            # Fail-closed means such a rule matches nothing, so the client
            # silently loses its Fast Lane priority -- a typo'd or
            # not-yet-started container is indistinguishable from a
            # correctly-configured one that simply never gets served. That
            # silence is the exact drift class container_name rules exist to
            # prevent.
            #
            # Only when a resolver was SUPPLIED: with none we have no evidence
            # either way, and accusing every name rule on every boot would
            # train operators to ignore this channel.
            if not resolved:
                warnings.append(
                    f"rule {idx + 1} names container {container_name!r}, which "
                    "currently resolves to NO addresses -- that rule matches "
                    "nothing and the client has no Fast Lane priority. Check the "
                    "container name and whether it is running."
                )

    # address-only rule colliding with a DIFFERENT rule's
    # resolved container_name -- only when a resolver was supplied.
    if resolve_name is not None:
        addr_owner: dict = {}  # resolved address -> (name_rule_idx, container_name)
        for name_idx, (container_name, addrs) in name_rule_addresses.items():
            for resolved_addr in addrs:
                addr_owner.setdefault(resolved_addr, (name_idx, container_name))
        for idx, rule in enumerate(rules):
            addr = getattr(rule, "address", None)
            if addr is None or getattr(rule, "container_name", None):
                continue
            try:
                norm_addr_obj = normalize_address(addr)
            except ValueError:
                continue
            owner = addr_owner.get(norm_addr_obj)
            if owner is not None:
                name_idx, container_name = owner
                warnings.append(
                    f"rule {idx + 1} (address {addr!r}) collides with rule "
                    f"{name_idx + 1}'s container_name {container_name!r} -- "
                    "the same client may be listed twice, once by address "
                    "and once by name."
                )

    # address-only rule sharing a docker-lease-sized prefix
    # (/24 v4, /64 v6) with an already-resolved container_name address --
    # only when a resolver was supplied; skips anything (c) already
    # flagged exactly, to avoid double-warning the same rule.
    if resolve_name is not None:
        all_name_addrs: list[tuple] = [
            (a, name_idx, container_name)
            for name_idx, (container_name, addrs) in name_rule_addresses.items()
            for a in addrs
        ]
        for idx, rule in enumerate(rules):
            addr = getattr(rule, "address", None)
            if addr is None or getattr(rule, "container_name", None):
                continue
            try:
                norm_addr_obj = normalize_address(addr)
            except ValueError:
                continue
            if norm_addr_obj in addr_owner:
                continue  # (c) already gave the stronger exact-collision warning
            for name_addr, name_idx, container_name in all_name_addrs:
                if _shares_docker_lease_prefix(norm_addr_obj, name_addr):
                    warnings.append(
                        f"rule {idx + 1} (address {addr!r}) shares a "
                        "docker-lease-sized prefix with rule "
                        f"{name_idx + 1}'s resolved container_name "
                        f"{container_name!r} -- this looks like a "
                        "docker-assigned address that can roll on restart; "
                        "prefer container_name for this rule. If this is a "
                        "deliberate off-network address that happens to "
                        "share a prefix by coincidence, that intent is "
                        "unverifiable by this check."
                    )
                    break  # one warning per address-rule is enough
    return warnings
