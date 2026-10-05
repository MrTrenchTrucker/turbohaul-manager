"""The staleness protection must be INERT while the
Fast Lane feature is switched off.

Fast Lane ships OFF (`config.py` `FastLaneConfig.enabled: bool = False`) with an
empty rule table. A feature that ships disabled must not change behaviour on a
host that never enabled it. Before this gate, it did: `_record_staleness_unload`
was called unconditionally from `_begin_unload_locked`, and
`_fastlane_staleness_protects` was an unconditional clause of
`_lru_idle_unloadable`'s candidate filter. Neither read `cfg.enabled` or the rule
table, so a grant armed by an ordinary LRU eviction excluded a SURVIVING SIBLING
resident from the eviction candidate set on a deployment with Fast Lane off.

⚠ WORDING THAT MATTERS, because the obvious phrasing is false: it is NOT "the
evicted resident is protected." That resident is deleted from `self._residents`
one line after the grant is armed, and `_model_residents()` filters DEAD. Grants
are keyed on the CLIENT IP (`_fastlane_census_key`), not on the resident -- so the
resident that gets excluded is a DIFFERENT, surviving one that last parked
holding the same client identity. Every fixture below is built that way: the
grant is armed for an ip, and the resident it protects is a separate object.

⛔ CONSTRAINT, same one the starvation-mirror drift test
imposes: these tests CALL the real `_lru_idle_unloadable`, `_starvation_reason`,
`_fastlane_staleness_protects` and `_record_staleness_unload`. No
reconstruction of the filter, no hand-rolled `min`, no re-derived threshold --
`_STALENESS_GRANT_THRESHOLD_S` and `_STALENESS_GRANT_WINDOW_S` are imported from
the module under test so a retune cannot silently orphan these assertions.

⛔ These tests are written entirely in terms of PRE-EXISTING symbols
(`_record_staleness_unload`, `_fastlane_staleness_protects`,
`_lru_idle_unloadable`, `_starvation_reason`, `_fastlane_staleness_grants`).
Nothing here calls the new `_fastlane_enabled` helper. That is deliberate: it
makes the pre-fix failure a genuine WRONG-VALUE failure on the unpatched tree, not an
AttributeError on a symbol that does not exist yet.

Behaviour changed: these tests fail on the unpatched code (wrong-value
failures) and pass with the gate.

TWO COVERAGE LIMITS, disclosed rather than implied away:
  1. These are pure-unit tests against the manager's own methods. They do not
     spawn a sidecar and do not exercise the cap>=2 dispatcher end to end, so
     they do not independently re-prove that `_begin_unload_locked` is the sole
     eviction entry point -- they take that from the call-graph reading of
     the source.
  2. `TestStalenessProtectionStillFiresWhenFastLaneEnabled` uses an ENABLED-but-EMPTY rule table
     throughout. That is not an accident of fixture convenience; it is the
     design decision made here, pinned as a test -- see
     `test_staleness_protection_still_fires_when_enabled_with_an_empty_rule_table`.
"""
import time

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import (
    Resident,
    ResidentState,
    TurbohaulManager,
    _fastlane_census_key,
    _STALENESS_GRANT_THRESHOLD_S,
    _STALENESS_GRANT_WINDOW_S,
)

GRANTED_IP = "203.0.113.7"
GRANTED_TAG = "granted-model"
OTHER_TAG = "other-model"


def _boot(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    return BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )


def _manager(tmp_path, fastlane):
    """`fastlane=None` omits the section entirely -- a host that never touched the
    setting. Otherwise a FastLaneConfig is passed explicitly."""
    kwargs = dict(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    if fastlane is not None:
        kwargs["fastlane"] = fastlane
    return TurbohaulManager(_boot(tmp_path), RuntimeConfig(**kwargs))


# The two ways a real deployment is "Fast Lane off": never configured at all, and
# configured explicitly off. Both must behave identically.
OFF_CASES = [
    ("never_configured", None),
    ("explicitly_disabled", FastLaneConfig(enabled=False)),
]


def _resident(tag, *, ip=None, last_active=1.0):
    return Resident(
        model_tag=tag,
        resident_key=tag,
        state=ResidentState.IDLE_EVICTABLE,
        last_active_monotonic=last_active,
        idle_client_meta={"ip": ip} if ip is not None else None,
    )


def _arm_grant(mgr, ip, *, age):
    """Seed a grant of a given age directly, the idiom the shipped staleness
    tests already use -- no clock faking exists in this suite."""
    mgr._fastlane_staleness_grants[_fastlane_census_key(ip)] = time.monotonic() - age


LIVE_AGE = _STALENESS_GRANT_THRESHOLD_S + 1.0


class TestStalenessProtectionIsInertWhenFastLaneDisabled:
    """The bar: an operator who never enabled Fast Lane gets pre-staleness-protection eviction
    behaviour, byte for byte."""

    @pytest.mark.parametrize("label,fastlane", OFF_CASES)
    def test_an_eviction_arms_no_grant_when_disabled(self, tmp_path, label, fastlane):
        mgr = _manager(tmp_path, fastlane)
        evicted = _resident("evicted-model", ip=GRANTED_IP)

        # PRECONDITION, so a green cannot be vacuous: the table starts empty and
        # the identity this call would record IS resolvable. If the resident had
        # no resolvable ip, `_record_staleness_unload` would return early for a
        # reason that has nothing to do with the feature gate.
        assert mgr._fastlane_staleness_grants == {}, label
        assert (evicted.idle_client_meta or {}).get("ip") == GRANTED_IP, label

        mgr._record_staleness_unload(evicted)

        assert mgr._fastlane_staleness_grants == {}, (
            f"[{label}] the protection armed a staleness grant on a host with Fast Lane off -- "
            "a feature that ships disabled must record nothing"
        )

    @pytest.mark.parametrize("label,fastlane", OFF_CASES)
    def test_a_live_grant_does_not_protect_when_disabled(self, tmp_path, label, fastlane):
        mgr = _manager(tmp_path, fastlane)
        _arm_grant(mgr, GRANTED_IP, age=LIVE_AGE)
        sibling = _resident(GRANTED_TAG, ip=GRANTED_IP)

        # PRECONDITION: the grant really is seeded and really is inside the
        # window by age, so a False below is the GATE talking and not an
        # expired/absent grant.
        assert _fastlane_census_key(GRANTED_IP) in mgr._fastlane_staleness_grants, label
        assert LIVE_AGE < _STALENESS_GRANT_THRESHOLD_S + _STALENESS_GRANT_WINDOW_S, label

        assert mgr._fastlane_staleness_protects(sibling) is False, (
            f"[{label}] a staleness grant protected a resident with Fast Lane off"
        )

    @pytest.mark.parametrize("label,fastlane", OFF_CASES)
    def test_granted_resident_is_still_an_eviction_candidate_when_disabled(
        self, tmp_path, label, fastlane
    ):
        """The behavioural claim, through the real selector.

        Two residents. The one holding the granted identity is the
        least-recently-active, so plain global LRU picks IT. With Fast Lane off
        that must still be the pick -- the protection must not have moved the victim.
        """
        mgr = _manager(tmp_path, fastlane)
        _arm_grant(mgr, GRANTED_IP, age=LIVE_AGE)
        mgr._residents[GRANTED_TAG] = _resident(GRANTED_TAG, ip=GRANTED_IP, last_active=1.0)
        mgr._residents[OTHER_TAG] = _resident(OTHER_TAG, ip="203.0.113.9", last_active=2.0)

        victim = mgr._lru_idle_unloadable()

        assert victim is not None, f"[{label}] the whole pool became unevictable"
        assert victim.model_tag == GRANTED_TAG, (
            f"[{label}] the protection moved the eviction victim on a Fast-Lane-off host: "
            f"expected the LRU resident {GRANTED_TAG!r}, got {victim.model_tag!r}"
        )

    @pytest.mark.parametrize("label,fastlane", OFF_CASES)
    def test_starvation_mirror_does_not_attribute_staleness_protected_when_disabled(
        self, tmp_path, label, fastlane
    ):
        """`_fastlane_staleness_protects` has a SECOND consumer: the
        `_starvation_reason` mirror. The gate lives inside the predicate rather
        than at the filter's call site precisely so both go inert together --
        `_starvation_reason`'s own docstring makes mirror/filter agreement an
        instrumented invariant, and a call-site gate would have broken it."""
        mgr = _manager(tmp_path, fastlane)
        _arm_grant(mgr, GRANTED_IP, age=LIVE_AGE)
        mgr._residents[GRANTED_TAG] = _resident(GRANTED_TAG, ip=GRANTED_IP)

        victim = mgr._lru_idle_unloadable()
        result = mgr._starvation_reason()

        assert victim is not None and victim.model_tag == GRANTED_TAG, label
        assert result["eligible"] == 1, (label, result)
        assert result["per_resident"] == f"{GRANTED_TAG}:eligible", (
            f"[{label}] the mirror still blames staleness_protected with the "
            f"feature off: {result['per_resident']!r}"
        )


class TestStalenessProtectionStillFiresWhenFastLaneEnabled:
    """The positive arm. A gate that kills the protection outright is not the fix -- every
    assertion here must survive the change."""

    @pytest.fixture
    def mgr_on(self, tmp_path):
        return _manager(tmp_path, FastLaneConfig(enabled=True))

    def test_an_eviction_arms_a_grant_when_enabled(self, mgr_on):
        evicted = _resident("evicted-model", ip=GRANTED_IP)
        assert mgr_on._fastlane_staleness_grants == {}

        mgr_on._record_staleness_unload(evicted)

        assert _fastlane_census_key(GRANTED_IP) in mgr_on._fastlane_staleness_grants, (
            "the protection recorded nothing with Fast Lane ENABLED -- the gate killed the feature"
        )

    def test_a_live_grant_protects_when_enabled(self, mgr_on):
        _arm_grant(mgr_on, GRANTED_IP, age=LIVE_AGE)
        sibling = _resident(GRANTED_TAG, ip=GRANTED_IP)

        assert mgr_on._fastlane_staleness_protects(sibling) is True

    def test_granted_resident_is_excluded_from_candidates_when_enabled(self, mgr_on):
        """The exact mirror image of the disabled case above: same fixture, same
        LRU order, opposite verdict. This pair is what makes both arms
        discriminating rather than merely green."""
        _arm_grant(mgr_on, GRANTED_IP, age=LIVE_AGE)
        mgr_on._residents[GRANTED_TAG] = _resident(GRANTED_TAG, ip=GRANTED_IP, last_active=1.0)
        mgr_on._residents[OTHER_TAG] = _resident(OTHER_TAG, ip="203.0.113.9", last_active=2.0)

        victim = mgr_on._lru_idle_unloadable()

        assert victim is not None, "the protection wedged the whole pool"
        assert victim.model_tag == OTHER_TAG, (
            "with Fast Lane ENABLED the protected resident must be skipped and the "
            f"next candidate taken, got {victim.model_tag!r}"
        )

    def test_mirror_attributes_staleness_protected_when_enabled(self, mgr_on):
        _arm_grant(mgr_on, GRANTED_IP, age=LIVE_AGE)
        mgr_on._residents[GRANTED_TAG] = _resident(GRANTED_TAG, ip=GRANTED_IP)

        victim = mgr_on._lru_idle_unloadable()
        result = mgr_on._starvation_reason()

        assert victim is None
        assert result["eligible"] == 0, result
        assert result["per_resident"] == f"{GRANTED_TAG}:staleness_protected", result

    @pytest.mark.parametrize(
        "label,age,expected",
        [
            ("below_threshold", _STALENESS_GRANT_THRESHOLD_S - 1.0, False),
            ("inside_window", _STALENESS_GRANT_THRESHOLD_S + 1.0, True),
            (
                "past_window",
                _STALENESS_GRANT_THRESHOLD_S + _STALENESS_GRANT_WINDOW_S + 1.0,
                False,
            ),
        ],
    )
    def test_window_boundaries_are_unchanged_when_enabled(self, mgr_on, label, age, expected):
        """The gate must not have retuned the protection's own [threshold, threshold+window)
        semantics -- only decided whether it runs at all."""
        _arm_grant(mgr_on, GRANTED_IP, age=age)
        sibling = _resident(GRANTED_TAG, ip=GRANTED_IP)

        assert mgr_on._fastlane_staleness_protects(sibling) is expected, label

    def test_staleness_protection_still_fires_when_enabled_with_an_empty_rule_table(self, mgr_on):
        """THE DESIGN DECISION OF THIS CHANGE, pinned so it cannot regress silently.

        The gate is keyed on `enabled` ALONE, not on `_fastlane_table()` being
        non-empty. Those are not the same predicate: the table is ALSO empty for
        enabled-with-no-rules. That configuration is live and behaviour-bearing
        here, not degenerate -- with an empty table every staged slot is
        "normal", which is exactly the population `queue._pick_fastlane_locked`'s
        wall-clock fairness floor governs. This protection is the resident-side member of that
        same anti-starvation family: it keys on client identity and never calls
        `match_fastlane`. Gating it on the rule table would switch it off in a
        configuration where it still has work to do.

        If someone later "simplifies" the gate to `not self._fastlane_table()`,
        this test is the one that fails.
        """
        assert mgr_on._fastlane_table() == [], (
            "fixture precondition: this manager is ENABLED with NO rules, so the "
            "compiled table must be empty -- that is the whole point of the case"
        )
        _arm_grant(mgr_on, GRANTED_IP, age=LIVE_AGE)
        sibling = _resident(GRANTED_TAG, ip=GRANTED_IP)

        assert mgr_on._fastlane_staleness_protects(sibling) is True, (
            "the protection went inert with Fast Lane ENABLED but no rules configured -- the "
            "gate is reading the rule table instead of the enabled flag"
        )


class TestUnresolvableIdentityIsUnaffectedByTheGate:
    """Regression guard on the pre-existing fail-open contract: a resident with
    no resolvable identity was never protected and never recorded, with Fast Lane
    on OR off. The gate must not have changed that in either direction."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_no_ip_is_never_protected_and_never_recorded(self, tmp_path, enabled):
        mgr = _manager(tmp_path, FastLaneConfig(enabled=enabled))
        anonymous = _resident("anon-model", ip=None)

        assert mgr._fastlane_staleness_protects(anonymous) is False
        mgr._record_staleness_unload(anonymous)
        assert mgr._fastlane_staleness_grants == {}
