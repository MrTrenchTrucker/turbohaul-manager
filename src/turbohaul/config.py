"""Top-level configuration: BootConfig (read-only at runtime) vs RuntimeConfig (PUT-mutable).

See ARCHITECTURE.md §9 for the boot-vs-runtime split rationale.

BootConfig fields require restart to change (server bind, storage paths, binary path).
RuntimeConfig fields are mutable via PUT /api/config (queue timings, pull params).
"""
import ipaddress
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


# Maximum honored client `keep_alive` value (cap).
# Module constant, not a QueueConfig field — operational policy, not per-deployment
# knob. Bump here if your hardware profile shifts.
KEEP_ALIVE_MAX_S = 1800


class ServerConfig(BaseModel):
    """Boot-only: server bind config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=11401, ge=1, le=65535)
    allow_public_bind: bool = False

    @field_validator("host")
    @classmethod
    def host_safe_default(cls, v: str) -> str:
        if v == "0.0.0.0":
            raise ValueError(
                "server.host cannot be 0.0.0.0 from yaml; set allow_public_bind: true "
                "AND pass --allow-public-bind CLI flag explicitly to bind public"
            )
        return v


class StorageConfig(BaseModel):
    """Boot-only: storage paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blob_store_path: Path
    manifests_path: Path
    import_allowed_root: Path
    state_db_path: Path


class RuntimePathsConfig(BaseModel):
    """Boot-only: binary path + sha256 pin + child port base."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    llama_server_binary: Path
    llama_server_binary_sha256: str = ""  # empty = skip verify (dev only)
    default_port_base: int = Field(default=11500, ge=1024, le=65000)


class UIConfig(BaseModel):
    """Boot-only: UI static path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    static_path: Path


class QueueConfig(BaseModel):
    """Runtime-mutable: queue + timing constants."""

    model_config = ConfigDict(extra="forbid")

    max_parallel_sidecars: int = Field(default=1, ge=1, le=32)
    staging_queue_depth: int = Field(default=100, ge=1, le=10000)
    acceptance_buffer_max: int = Field(default=10000, ge=1)
    # Model-affinity pop tuning (single-mutator-safe parallelism support).
    # Both default to behavior that is a strict-FIFO no-op unless the
    # worker_loop opts in by passing warm_model_tag to pop_next:
    #   - max_consecutive_same_model bounds the run-length of one model the
    #     affinity path will cluster before forcing the FIFO head (fairness).
    #     A value of 1 disables batching entirely (every pop honors FIFO head).
    #   - max_other_model_wait_s is the age (via Slot.created_at monotonic
    #     clock) past which a starved other-model head request forces a swap,
    #     overriding affinity. 0.0 means "starve immediately" => strict FIFO.
    max_consecutive_same_model: int = Field(default=3, ge=1, le=1000)
    max_other_model_wait_s: float = Field(default=20.0, ge=0.0, le=3600.0)
    # Queue-level reservation for interactive main traffic. This is admission
    # control, not a second sidecar: when main is queued, it is popped before
    # auxiliary work after the current request completes.
    main_lane_reserved: bool = True
    main_lane_identity_keys: list[str] = Field(default_factory=lambda: ["is_main"])
    grace_seconds: int = Field(default=30, ge=0, le=3600)
    # Default bumped 120 → 300 so multi-turn agents (OpenAI-SDK class clients)
    # with client-side tool-exec / reflection gaps in the 2-5min range keep their
    # slot warm without needing to send keep_alive. OpenAI-SDK clients can't send
    # keep_alive natively (see Ollama Issue #11458).
    # Later bumped 300 → 600 because a reasoning model with a large reasoning
    # budget on complex compare prompts can produce 5-7min client-side inter-turn
    # gaps. The 300s window was eaten by the client's reasoning chain on the FIRST
    # tool-result reflection, not by Turbohaul itself.
    idle_hot_load_seconds: int = Field(default=600, ge=0, le=86400)
    # Safety guardrails -- mirror Ollama's pre-spawn safety posture
    safety_enabled: bool = True
    safety_min_free_ram_mib: int = Field(default=1024, ge=0)
    safety_min_free_vram_mib: int = Field(default=512, ge=0)
    # Percent of host CPU actually busy (not tasks per core).
    # A load-average-per-core metric (tasks per core) is a different scale and
    # is not used by this gate.
    # 90.0 is a saturation ceiling and a policy pick, not a measured
    # optimum, and is revisable on saturated telemetry.
    #
    # This is a DIFFERENT SCALE from a load-average-per-core threshold (percent
    # of host CPU actually busy, not tasks per core) and the two do not
    # convert, so 90.0 makes no continuity claim on an old 0.9 threshold;
    # that value could not be reused as-is.
    #
    # The ceiling deliberately relaxes the gate, which is correct
    # here: a load-average metric can refuse spawns on a mostly idle
    # host, while a percent-busy ceiling only refuses when the CPU is
    # actually saturated.
    #
    safety_max_cpu_busy_percent: float = Field(default=90.0, ge=0.0, le=100.0)
    safety_cpu_util_sample_window_s: float = Field(default=0.4, ge=0.05, le=5.0)
    safety_max_iowait_percent: float = Field(default=30.0, ge=0.0, le=100.0)
    safety_iowait_sample_window_s: float = Field(default=0.4, ge=0.05, le=5.0)
    max_grace_extensions: int = Field(default=5, ge=0, le=1000)
    # prefix_token_count controls how many tokens of the prompt are
    # hashed for auto-derived thread_id. Hashing only the prefix (not the full
    # prompt) ensures that conversation extensions — same prefix, more tokens —
    # produce the SAME thread_id, allowing the grace window to match and the KV
    # cache restore to fire. Default 256 captures a typical system prompt + initial
    # context; adjust via TURBOHAUL_PREFIX_TOKEN_COUNT if needed.
    prefix_token_count: int = Field(default=256, ge=1, le=8192)
    # Addresses KNOWN to be shared by several distinct clients — a
    # container gateway, a NAT egress. Comma-separated; parsed by
    # slot.parse_shared_addresses. These are refused an IP-based GRACE match,
    # because that consumer fails PERMISSIVE (a fused address serves one client's
    # request on another's warm resident) where the fairness floor's identical
    # `ip:` key fails RESTRICTIVE and is therefore safe to fuse. Do not "harmonise"
    # the two — see slot.grace_ip_match_eligible for why they must differ.
    # ⛔ FAIL-OPEN BY CONSTRUCTION: a shared address that is not listed here is
    # matched silently. Empty means "nobody has listed one", never "there are none".
    grace_ip_match_shared_addresses: str = ""
    # How long to wait for a freshly-spawned engine to report healthy before the
    # load is treated as failed. 60s comfortably covers a normal model swap; a
    # stalled load surfaces in ~1 minute instead of blocking the caller for
    # much longer. Deployments that genuinely need more (very large models on
    # slow storage) can raise it — the upper bound stays generous on purpose so
    # that remains possible without a rebuild.
    loading_health_timeout_s: int = Field(default=60, ge=10, le=7200)
    # Spawn-time reclaim barrier. On a VRAM-only refusal with a
    # bounded reclaim ALREADY in flight on the gated card(s) (a live entry in
    # _card_release_tasks), wait up to this many seconds -- re-running the full
    # safety gate every _SPAWN_RECLAIM_REGATE_TICK_S -- before the terminal
    # refusal. 0.0 is the kill switch (disables the barrier entirely: zero
    # waits, terminal on the first gate run). The le=60.0 clamp deliberately
    # ties this to loading_health_timeout_s's own default (above) -- the
    # reservation-hold envelope this code path already takes AFTER a
    # successful spawn, so the barrier can't hold a reservation longer than
    # the spawn path itself already does post-spawn.
    spawn_reclaim_wait_max_s: float = Field(default=10.0, ge=0.0, le=60.0)
    # httpx timeout for the non-streaming sidecar completion request
    # (chat_completion.py's make_llama_server_complete_fn). A hardcoded default
    # argument would be the ONLY value that ever runs in production, since main.py's
    # no-argument call site never overrides it -- so no operator
    # could raise it without editing code. Default 3600.0 gives literal
    # parity with the streaming path's STREAM_TIMEOUT_S, defined in
    # chat_completion.py; NOT derived from it, because config.py has zero imports from
    # turbohaul.api today and chat_completion.py already imports FROM
    # config.py (KEEP_ALIVE_MAX_S) -- deriving here would invert that
    # dependency into a circular import. The two values CAN drift if
    # STREAM_TIMEOUT_S is raised later and this field is forgotten; that
    # drift is the same mismatch shape this field removes. The clean fix is moving
    # STREAM_TIMEOUT_S into config.py as a real field too, eliminating the
    # asymmetry structurally -- a larger change, not made here
    # (it touches the streaming path)
    # and noted here as a known limitation rather than a bare TODO.
    sidecar_complete_timeout_s: float = Field(default=3600.0, ge=1.0, le=86400.0)
    drained_sigterm_window_active_s: int = Field(default=15, ge=1, le=300)
    drained_sigterm_window_cold_s: int = Field(default=5, ge=1, le=300)
    # Background sweeper cadence — finalizes state-row for
    # evictions that landed audit-only via _audit_event_only_async pool path.
    # 60s aligns with the audit pool rhythm. Sweeper requires staleness ≥ 24h
    # (background_sweep_min_age_s) so in-flight slots are never reaped.
    background_sweep_interval_s: int = Field(default=60, ge=1, le=86400)
    background_sweep_min_age_s: int = Field(default=86400, ge=60, le=2592000)  # floor stays at 60s — actual SQL gate is `state=STAGED` (NOT grace-rematch states), so operator misconfig cannot reap in-flight grace-rematch slots; gate-filter is sufficient defense; 60s floor preserved for synthetic-age test boundary


class PullConfig(BaseModel):
    """Runtime-mutable: pull endpoints + safety constraints."""

    model_config = ConfigDict(extra="forbid")

    hf_api_key_env: str = "HF_API_KEY"
    hf_host_allowlist: list[str] = Field(default_factory=lambda: ["huggingface.co", "hf.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co"])
    pull_url_https_only: bool = True
    pull_concurrency: int = Field(default=2, ge=1, le=16)
    pull_chunk_size_mb: int = Field(default=64, ge=1, le=1024)
    per_stream_max_bytes: int = Field(default=107_374_182_400, ge=1)


class PersistConfig(BaseModel):
    """Runtime-mutable: SSD persist tier config."""

    model_config = ConfigDict(extra="forbid")

    max_bytes: int = Field(default=42949672960, ge=0)  # 40 GiB default


class MonitorConfig(BaseModel):
    """Runtime-mutable: live inference monitor (tok/s + progress + live output).

    enabled is an ops kill-switch; poll_interval_s is the single-poller /slots
    cadence (one reader regardless of FE client count). The remaining tuning
    (smoothing, stall thresholds, text-tail size) are module-level constants in
    live_monitor.py — not speculative config surface.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    poll_interval_s: float = Field(default=1.0, gt=0.0, le=60.0)


class HttpConfig(BaseModel):
    """Runtime-mutable: HTTP-layer request admission limits."""

    model_config = ConfigDict(extra="forbid")

    # Single Content-Length ceiling shared by
    # api/body_size_limit.py (_MAX_BODY_BYTES) and
    # api/embeddings.py (_MAX_REQUEST_BYTES), kept as one real config field in
    # a one-field section like persist.max_bytes and the
    # KVConfig.ram_cache_max_bytes byte ceiling (every existing ceiling in
    # this codebase lives in a runtime section — none of them
    # live in BootConfig). Default (64 MiB, generously sized: comfortably covers a
    # 1-minute 720p video clip base64-inline, ~half of a 5-minute clip) is a
    # generic-multimodal-request default, NOT a media-ingestion ceiling:
    # very long video needs a different
    # ingestion path rather than an ever-bigger value here. No upper bound
    # (`le=`) is set deliberately: an operator must be able to raise this
    # arbitrarily high (no second hardcoded cap anywhere else in
    # the stack silently overrides a raised value) without this field being
    # the thing standing in the way of their own choice.
    max_body_bytes: int = Field(default=64 * 1024 * 1024, ge=1)


class KVConfig(BaseModel):
    """Runtime-mutable: KV-cache behavior toggles."""

    model_config = ConfigDict(extra="forbid")

    covered_scaffold_strip: bool = True
    # The RAM-tier KV-cache save-dir byte ceiling
    # (mirrors PersistConfig.max_bytes, the disk-tier sibling).
    # Default 20 GiB.
    # <= 0 disables the ceiling (age/count knobs still
    # apply). Environment override:
    # TURBOHAUL_KVCACHE_MAX_BYTES.
    ram_cache_max_bytes: int = Field(default=20 * 1024 ** 3, ge=0)
    # The RAM-tier KV-cache AGE ceiling: entries in the RAM-disk tier
    # are released after this many hours.
    #
    # It is a config key, not an environment variable, exposed on the
    # backend and in the frontend Settings tab. Like its byte-ceiling
    # sibling above, it is a real config field.
    #
    # Default 6.0 hours, the same as the `max_age_hours` default of
    # manager.py's `_gc_kv_cache(...)`. The value is reachable
    # through the config API.
    ram_cache_max_age_hours: float = Field(default=6.0, ge=0)
    # Boot-time KV-dir mount/fstype assertion
    # (FAIL LOUD, KEEP SERVING — see manager.py's boot_reconcile /
    # _check_kv_dir_mounts). Config key with a default, never an env var
    # (project policy) — both config models use extra="forbid"
    # and a persisted runtime_config.yaml on the data volume outranks both the
    # shipped file AND the code default, so a REQUIRED field here would fail
    # validation against an already-persisted file on a live deployment.
    #
    # SLOT_SAVE_DIR (the RAM/tmpfs tier): expected to be its OWN mount with
    # this fstype — this is the directory where a misconfiguration is most
    # costly (a missing tmpfs mount would silently redirect every RAM-tier
    # write to the backing filesystem instead of erroring). None disables the
    # assertion entirely (own-mount-ness is not checked either, in that case).
    kv_save_expected_fstype: str | None = "tmpfs"
    # SLOT_PERSIST_DIR (the cold-storage/disk tier): the INVERSE assertion — a
    # FORBIDDEN-fstype list, not an expected one. By design, kvcache_persist is
    # a plain directory on the underlying filesystem, NOT
    # its own mount -- asserting own-mount there would fire LOUD on
    # a correctly configured production deployment on its very first boot. What IS
    # assertable: cold storage must never land on a volatile filesystem, or
    # it silently evaporates on restart (the same failure shape as the
    # RAM-tier problem above, on the opposite tier).
    kv_persist_forbidden_fstypes: list[str] = Field(
        default_factory=lambda: ["tmpfs", "ramfs"]
    )


class FastLaneTagRanks(BaseModel):
    """Rank 1-5 WITHIN one address slot; lower is served first. None = unranked."""

    model_config = ConfigDict(extra="forbid")

    main: int | None = Field(default=None, ge=1, le=5)
    curator: int | None = Field(default=None, ge=1, le=5)
    compression: int | None = Field(default=None, ge=1, le=5)
    sub_agent: int | None = Field(default=None, ge=1, le=5)
    unclassified: int | None = Field(default=None, ge=1, le=5)


class FastLaneRule(BaseModel):
    """A client is named by exactly one of `address` (off-network: LAN, VPN,
    any static IP) or `container_name` (on-network: the docker
    container name, resolved forward-only at runtime by fastlane_resolve.py
    -- see that module for why this is forward name->address resolution and
    never reverse DNS). The docker bridge network
    has no static IPAM, so an `address` rule silently drifts to the wrong
    client on every container restart; `container_name` is the fix for the
    on-network case. `address` remains required for anything NOT on the
    docker network, where there is no name to resolve."""

    model_config = ConfigDict(extra="forbid")

    address: str | None = None  # VERBATIM as observed; never retyped
    container_name: str | None = None  # docker container name; resolved, not stored as an IP
    label: str = ""  # operator's own note
    tag_ranks: FastLaneTagRanks = Field(default_factory=FastLaneTagRanks)

    @field_validator("address")
    @classmethod
    def _address_must_be_single_ip_or_none(cls, v: str | None) -> str | None:
        """Reject any non-single-host address, not just a CIDR/prefix.
        The name states the full check: the value must be one IP address
        or None, so a name that implied only a prefix check would
        mislead."""
        if v is None:
            return None
        if "/" in v:
            raise ValueError(
                f"fastlane rule address must be a single host, not a CIDR/prefix: {v!r}"
            )
        # A bare hostname such as "gateway-svc" must be rejected here, not only
        # a value containing "/": compile_fastlane's normalize_address try/except
        # would silently drop it, and lint_rules would then misreport it as
        # "never observed" -- a misleading diagnosis pointing at the wrong cause.
        # Same host-parsing
        # shape normalize_address (fastlane.py) accepts -- strip a %zone
        # suffix before parsing -- reimplemented locally rather than imported,
        # same precedent as this file's own _RESOURCE_KEY_RE inlining above
        # (config.py does not import runtime-hot-path modules, to keep the
        # boot-time validation layer independent of it).
        zone_stripped = v.split("%", 1)[0]
        try:
            ipaddress.ip_address(zone_stripped)
        except ValueError:
            raise ValueError(
                f"fastlane rule address {v!r} is not a valid IP address -- if you "
                "meant a docker container name, set container_name instead"
            ) from None
        return v

    @model_validator(mode="after")
    def _exactly_one_of_address_or_container_name(self) -> "FastLaneRule":
        identity = repr(self.label) if self.label else "(unlabeled)"
        if self.address is not None and self.container_name is not None:
            raise ValueError(
                f"fastlane rule {identity} sets BOTH address and container_name "
                "-- exactly one is required"
            )
        # A BLANK container_name is not an identity, and it has to be refused
        # HERE rather than anywhere downstream. `""` is `is not None`, so a
        # null test counts it as SET at this gate, while
        # _strip_null_container_names (api/config_put.py) tests truthiness and
        # counts the same value as UNSET on the way to disk -- so the rule
        # persists carrying NEITHER field. On the next boot FastLaneConfig
        # rejects the section, _salvage_fastlane_rules declines it (its
        # coercion arm needs a bad ADDRESS to work with, and there is none),
        # and the boot merge's `continue` drops the section whole -- taking
        # every VALID sibling rule with it and booting the feature disabled.
        #
        # Handled on this side rather than in the strip deliberately: the strip
        # exists for rollback safety, removing the `container_name: null`
        # placeholder so a rolled-back earlier release (extra="forbid") can still
        # load the section. Narrowing it to `is None` would let `""` reach disk
        # as a real key and defeat exactly that guarantee.
        #
        # Testing emptiness also makes this file agree with the field's two
        # other readers: compile_fastlane and lint_rules (fastlane.py) both
        # already treat a falsy container_name as absent. `is not None` here
        # would be the odd one out among three readers of the same field.
        #
        # Only the NEITHER arm is emptiness-aware. The BOTH arm above is left
        # on `is not None` on purpose -- making it blank-aware would start
        # ACCEPTING address-plus-blank, a loosening this case does not need.
        if self.address is None and not (self.container_name or "").strip():
            if self.container_name is not None:
                raise ValueError(
                    f"fastlane rule {identity} sets container_name to a blank "
                    "string, which is not an identity -- give it a real "
                    "container name, or set address instead"
                )
            raise ValueError(
                f"fastlane rule {identity} sets NEITHER address nor container_name "
                "-- exactly one is required"
            )
        return self


MAX_FASTLANE_RULES = 10


class FastLaneConfig(BaseModel):
    """Runtime-mutable: Fast Lane two-level request-priority feature. OFF by default."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    rules: list[FastLaneRule] = Field(default_factory=list)  # LIST INDEX IS THE PRIORITY
    # ~1 hour: the one-turn fairness guarantee's own window, so a client that
    # gets no follow-up inside it has already had its fair turn.
    max_normal_wait_s: float = Field(default=3600.0, ge=1.0, le=3600.0)
    # Ceiling set to the operationally
    # sane range for this feature. Fast Lane is exempt from this budget
    # entirely — a Fast-Lane-matched claimant is never refused on budget
    # grounds, at any value including 0 (by design). The cap
    # governs unregistered (no-rule-matched) cross-model swaps instead.
    cross_model_switches_per_min: int = Field(default=2, ge=0, le=3)
    census_ttl_hours: int = Field(default=168, ge=1, le=8760)

    @model_validator(mode="after")
    def _rules_within_cap(self) -> "FastLaneConfig":
        """Fast Lane designates AT MOST MAX_FASTLANE_RULES clients -- the
        list-index-is-priority design has no room for a client outside that
        range to be meaningfully ranked. Message names both the attempted
        count and the cap so a rejection (PUT) or a boot-time salvage log
        is self-explanatory without cross-referencing this file."""
        if len(self.rules) > MAX_FASTLANE_RULES:
            raise ValueError(
                f"fastlane.rules has {len(self.rules)} entries, exceeding the "
                f"{MAX_FASTLANE_RULES}-rule cap"
            )
        return self


class PluginEndpoint(BaseModel):
    """Boot-only: one operator-declared external-container endpoint. Never
    editable through the API -- see PluginsConfig. host/port/health_path
    describe WHERE to connect; the manifest side only ever names this entry
    by resource_key, never by URL or path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str
    port: int = Field(ge=1, le=65535)
    health_path: str = "/health"
    # OPTIONAL path to a file holding a bearer credential for this
    # plugin -- NEVER the credential itself. The config file this model loads
    # from is bind-mounted, copied around by operators, and the kind of thing
    # that ends up in a paste; a path is not a secret, a token is. Read by
    # invoke_plugin at CALL time (turbohaul.plugin_invoke), fresh every call --
    # not cached here, not read via an import of this module into that one
    # (config.py and plugin_invoke.py deliberately avoid a real runtime import
    # cycle today; see plugin_invoke.py's TYPE_CHECKING-only PluginEndpoint
    # reference). Absent (None) = no Authorization header is sent, the same as
    # a registry entry that declares no token file.
    auth_token_file: Path | None = None

    @model_validator(mode="after")
    def _auth_token_file_readable_at_boot(self) -> "PluginEndpoint":
        """Fail LOUDLY at boot (BootConfig construction) on a missing or
        unreadable secret file, rather than silently sending no Authorization
        header and turning a config mistake into a baffling 401 far from its
        cause. The read value is discarded immediately -- this
        proves the file CAN be read and is non-empty, it does not cache the
        token anywhere on this frozen, long-lived object. Error messages below
        name the PATH only, never the file's content.
        """
        if self.auth_token_file is None:
            return self
        try:
            content = self.auth_token_file.read_text()
        except OSError as e:
            raise ValueError(
                f"plugins endpoint auth_token_file {str(self.auth_token_file)!r} "
                f"could not be read at boot: {e}"
            ) from e
        if not content.strip():
            raise ValueError(
                f"plugins endpoint auth_token_file {str(self.auth_token_file)!r} "
                "is empty"
            )
        return self

    @field_validator("host")
    @classmethod
    def _host_is_bare_hostname_or_ip(cls, v: str) -> str:
        for bad in ("://", "/", "?", "@"):
            if bad in v:
                raise ValueError(
                    f"plugins endpoint host must be a bare hostname or IP, never a URL "
                    f"(found {bad!r} in {v!r})"
                )
        # Bare IP literal or hostname ONLY. A shorthand IP spelling (127.1,
        # 2130706433, 0177.0.0.1, 0x7f.0.0.1 -- bracketed or not) is NEITHER:
        # ipaddress rejects it, so resolve_endpoint's guard never sees it,
        # and the OS inet_aton would still expand it to its real target --
        # a registration of 127.1 would reach 127.0.0.1 and bypass
        # the guard's loopback coverage, which would then be a false
        # guarantee for those spellings. Contract, enforced here: parseable
        # by ipaddress, or a hostname whose final label is not all-numeric
        # (no real DNS name ends in one). No DNS resolution here -- that
        # would change the documented trust model and add TOCTOU.
        unwrapped = v[1:-1] if v.startswith("[") and v.endswith("]") else v
        try:
            ipaddress.ip_address(unwrapped)
            return v
        except ValueError:
            pass
        if unwrapped.rsplit(".", 1)[-1].isdigit():
            raise ValueError(
                f"plugins endpoint host {v!r} is not a bare IP literal or hostname -- "
                f"shorthand IP forms (decimal/octal/hex/short, e.g. '127.1') are "
                f"rejected so the SSRF guard cannot be bypassed by spelling"
            )
        return v

    @field_validator("health_path")
    @classmethod
    def _health_path_shape(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError(f"health_path must start with '/': {v!r}")
        if ".." in v:
            raise ValueError(f"health_path must not contain '..': {v!r}")
        return v


# Same key register as manifest.py's TAG_RE (resource_key is
# validated against this identical pattern) -- inlined rather than imported: config.py
# does not import manifest.py today and must not start (import-cycle risk).
_RESOURCE_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class PluginsConfig(BaseModel):
    """Boot-only: the plugin endpoint registry. THE ONLY PLACE in the system a
    host/port for an external plugin container may live -- it exists so a
    manifest never carries a URL. Must be listed in api/config_put.py's
    BOOT_SECTIONS so it can never be written through PUT /api/config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    registry: dict[str, PluginEndpoint] = Field(default_factory=dict)

    @field_validator("registry")
    @classmethod
    def _keys_are_resource_key_shaped(
        cls, v: "dict[str, PluginEndpoint]"
    ) -> "dict[str, PluginEndpoint]":
        for key in v:
            if not _RESOURCE_KEY_RE.match(key):
                raise ValueError(
                    f"plugins registry key {key!r} must match "
                    f"{_RESOURCE_KEY_RE.pattern!r} (same register as manifest.py's TAG_RE)"
                )
        return v


class PluginRuntimeConfig(BaseModel):
    """Runtime-mutable: per-plugin operational knobs (the FE Plugins tab
    edits this). Never carries a host or port -- see PluginsConfig."""

    model_config = ConfigDict(extra="forbid")

    # KEYED BY model_tag, NOT resource_key. `enabled` is a
    # POLICY question -- should this plugin run -- and policy belongs to manifest
    # identity. resource_key is a TRANSPORT question and is deliberately MANY-TO-ONE:
    # two plugin manifests may share one registry entry (e.g. a fast and an accurate
    # profile against the same container), so keying here by resource_key would make
    # disabling one silently disable the other.
    enabled: dict[str, bool] = Field(default_factory=dict)
    max_concurrent: int = Field(default=4, ge=1, le=64)
    no_progress_timeout_s: float = Field(default=600.0, ge=0.0, le=86400.0)


class BootConfig(BaseModel):
    """Top-level boot-only configuration (frozen after load)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    server: ServerConfig
    storage: StorageConfig
    runtime: RuntimePathsConfig
    ui: UIConfig
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)


class RuntimeConfig(BaseModel):
    """Top-level runtime-mutable configuration (PUT-able)."""

    model_config = ConfigDict(extra="forbid")

    queue: QueueConfig
    pull: PullConfig
    persist: PersistConfig = Field(default_factory=PersistConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    kv: KVConfig = Field(default_factory=KVConfig)
    fastlane: FastLaneConfig = Field(default_factory=FastLaneConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    plugin_runtime: PluginRuntimeConfig = Field(default_factory=PluginRuntimeConfig)


class TurbohaulConfig(BaseModel):
    """Full config = boot + runtime, used for yaml load/save."""

    model_config = ConfigDict(extra="forbid")

    server: ServerConfig
    storage: StorageConfig
    runtime: RuntimePathsConfig
    ui: UIConfig
    queue: QueueConfig
    pull: PullConfig
    persist: PersistConfig = Field(default_factory=PersistConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    kv: KVConfig = Field(default_factory=KVConfig)
    fastlane: FastLaneConfig = Field(default_factory=FastLaneConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    plugin_runtime: PluginRuntimeConfig = Field(default_factory=PluginRuntimeConfig)

    def split(self) -> tuple[BootConfig, RuntimeConfig]:
        boot = BootConfig(
            server=self.server,
            storage=self.storage,
            runtime=self.runtime,
            ui=self.ui,
            plugins=self.plugins,
        )
        runtime = RuntimeConfig(
            queue=self.queue, pull=self.pull, persist=self.persist,
            monitor=self.monitor, kv=self.kv, http=self.http, fastlane=self.fastlane,
            plugin_runtime=self.plugin_runtime,
        )
        return boot, runtime


def migrate_legacy_fastlane_key(data: dict) -> dict:
    """Back-compat: accept the legacy top-level key "fastline" as "fastlane".
    A persisted config -- the shipped/boot turbohaul.yaml OR a
    runtime_config.yaml override written by an earlier version -- may still carry its Fast Lane section under the old
    top-level key "fastline". Translate it to "fastlane" so old data keeps
    loading with IDENTICAL effective settings: the Pydantic model underneath
    (FastLaneConfig) is unchanged, only the section's key moved.

    A dict that already has "fastlane" wins outright -- "fastline" is never
    consulted for its VALUE in that case, so an explicit new-key value is
    never silently overridden by a stale old-key one. But "fastline" is
    ALWAYS popped out of the returned dict when present, win-or-not: both
    TurbohaulConfig and RuntimeConfig use ConfigDict(extra="forbid"), so
    leaving an unrecognized "fastline" key sitting alongside a correct
    "fastlane" one would still raise on construction -- pathological
    (both keys in the same file), but a migration that fixes the common case
    while introducing a new crash in the rare one is not a migration.
    Returns a NEW dict; never mutates the input in place, since both call
    sites (config.py's own load_config_yaml, api/config_put.py's
    load_runtime_override) hand this a dict they still use afterward.

    Server-side only, by design:
    there is no such tolerance anywhere on the live wire -- GET /api/config
    serves "fastlane" only, PUT /api/config accepts "fastlane" only. This
    function exists SOLELY for the two places a persisted config is read off
    disk, so already-saved rules keep loading no matter which
    key wrote them.
    """
    if "fastline" in data:
        data = dict(data)
        legacy = data.pop("fastline")
        if "fastlane" not in data:
            data["fastlane"] = legacy
    return data


def load_config_yaml(path: Path) -> TurbohaulConfig:
    """Load + validate turbohaul.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"config root must be mapping, got {type(data).__name__}")
    data = migrate_legacy_fastlane_key(data)
    return TurbohaulConfig(**data)


_ENV_MAP: dict[str, tuple[str, str, type]] = {
    "TURBOHAUL_HOST": ("server", "host", str),
    "TURBOHAUL_PORT": ("server", "port", int),
    "TURBOHAUL_MAX_PARALLEL": ("queue", "max_parallel_sidecars", int),
    "TURBOHAUL_STAGING_DEPTH": ("queue", "staging_queue_depth", int),
    "TURBOHAUL_ACCEPT_MAX": ("queue", "acceptance_buffer_max", int),
    "TURBOHAUL_GRACE_S": ("queue", "grace_seconds", int),
    "TURBOHAUL_IDLE_HOT_S": ("queue", "idle_hot_load_seconds", int),
    "TURBOHAUL_MAX_GRACE_EXT": ("queue", "max_grace_extensions", int),
    "TURBOHAUL_PREFIX_TOKEN_COUNT": ("queue", "prefix_token_count", int),
    "TURBOHAUL_GRACE_IP_SHARED_ADDRESSES": ("queue", "grace_ip_match_shared_addresses", str),
    "TURBOHAUL_MAX_CONSECUTIVE_SAME_MODEL": ("queue", "max_consecutive_same_model", int),
    "TURBOHAUL_MAX_OTHER_MODEL_WAIT_S": ("queue", "max_other_model_wait_s", float),
    "TURBOHAUL_KVCACHE_PERSIST_MAX_BYTES": ("persist", "max_bytes", int),
    # Environment override for kv.ram_cache_max_bytes, handled by the standard
    # config + env-override mechanism every other knob (including the
    # sibling persist.max_bytes above) uses. The field is visible
    # through /api/config,
    # the schema endpoint,
    # and PUT.
    "TURBOHAUL_KVCACHE_MAX_BYTES": ("kv", "ram_cache_max_bytes", int),
    # Environment override for http.max_body_bytes (same mechanism as
    # TURBOHAUL_KVCACHE_MAX_BYTES above); a request-body ceiling that is too
    # small blocks multimodal input.
    "TURBOHAUL_MAX_BODY_BYTES": ("http", "max_body_bytes", int),
}


def apply_env_overrides(cfg: TurbohaulConfig) -> TurbohaulConfig:
    """Apply TURBOHAUL_* env var overrides. Env beats yaml."""
    data: dict[str, Any] = cfg.model_dump()
    for env_key, (section, field, cast) in _ENV_MAP.items():
        v = os.environ.get(env_key)
        if v is not None:
            # Optional sections (e.g. persist) may be
            # absent from a user yaml; an env override must not KeyError.
            data.setdefault(section, {})[field] = cast(v)
    return TurbohaulConfig(**data)


# The sections GET /api/config reports, keyed the
# same way. Shared by the provenance resolver (api/config_put.py, which needs
# RUNTIME_SECTIONS + load_runtime_override and so imports from here rather
# than the reverse) and compute_shipped_baseline below.
_ALL_SECTION_MODELS: dict[str, type[BaseModel]] = {
    "server": ServerConfig,
    "storage": StorageConfig,
    "runtime": RuntimePathsConfig,
    "ui": UIConfig,
    "queue": QueueConfig,
    "pull": PullConfig,
    "persist": PersistConfig,
    "monitor": MonitorConfig,
    "kv": KVConfig,
    "http": HttpConfig,
    "fastlane": FastLaneConfig,
    "plugins": PluginsConfig,
    "plugin_runtime": PluginRuntimeConfig,
}


def compute_shipped_baseline(shipped_yaml_raw: dict) -> dict[str, dict[str, Any]]:
    """Per-field value a FRESH deployment would get: code default, overridden
    by the shipped yaml if present -- no env var, no persisted runtime
    override. This is the baseline the startup divergence log
    compares the real effective config against ("effective value differs
    from the shipped default").
    """
    baseline: dict[str, dict[str, Any]] = {}
    for section, model_cls in _ALL_SECTION_MODELS.items():
        section_yaml = shipped_yaml_raw.get(section) or {}
        try:
            baseline[section] = model_cls(**section_yaml).model_dump(mode="json")
        except ValidationError:
            # This function is READ-ONLY BOOKKEEPING for the provenance
            # endpoint and the startup divergence log -- it is the second
            # place the shipped yaml gets validated, and it must never be the
            # reason a manager fails to start. The boot-effective config is
            # built elsewhere, from load_config_yaml, which has its own
            # handling for a salvageable section; a raise here would defeat
            # that entirely and abort the boot anyway, from bookkeeping.
            # Falling back to the pure code defaults keeps the comparison
            # meaningful: the divergence log then reports the section as
            # differing from the shipped default, which is exactly what an
            # unloadable shipped section means.
            baseline[section] = model_cls().model_dump(mode="json")
    return baseline
