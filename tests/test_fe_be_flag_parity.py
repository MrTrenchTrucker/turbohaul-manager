"""FE/BE flag-parity guard.

The FE mirrors the BE allowlist by hand: ``flagsSchema.ts`` carries a
``FLAGS_SCHEMA`` entry per flag in ``manifest.SAFE_LLAMA_FLAGS``. Nothing
enforced that, and it drifted -- the BE reached 108 flags while the FE mirror
held 106, leaving ``tensor_split`` and ``no_kv_offload`` allowlisted and
validated on the backend with no control in the per-model editor at all.

``test_kv_cache_type_parity.py`` already guards the KV *cache-type value list*.
This guards the *flag-name set*, which is the surface that actually drifted:
that test would have stayed green through the entire regression.

Deliberately implemented in Python against the checked-in TypeScript rather
than as a front-end test. The frontend has no test runner at all, so a JS test
could not run in CI as it stands; parsing the schema keeps the guard in the
suite that already executes.
"""
import re
from pathlib import Path

from turbohaul.manifest import SAFE_LLAMA_FLAGS

FRONTEND_ROOT = Path(__file__).resolve().parent.parent / "src" / "frontend"
FLAGS_SCHEMA_PATH = FRONTEND_ROOT / "src" / "flagsSchema.ts"

# Matches the `name: 'flag_name'` key of each FLAGS_SCHEMA entry.
_FE_NAME_RE = re.compile(r"name:\s*'([a-z0-9_]+)'")


def _fe_flag_names() -> set:
    assert FLAGS_SCHEMA_PATH.is_file(), f"expected FE schema at {FLAGS_SCHEMA_PATH}"
    names = set(_FE_NAME_RE.findall(FLAGS_SCHEMA_PATH.read_text()))
    # Guard the guard: if the regex stops matching the file's shape this test
    # would pass vacuously by comparing two empty-ish sets.
    assert len(names) > 50, (
        f"parsed only {len(names)} flag names from {FLAGS_SCHEMA_PATH.name}; "
        "the schema format probably changed and this test is no longer reading it"
    )
    return names


class TestFeBeFlagParity:
    def test_flag_name_sets_are_identical(self):
        be = set(SAFE_LLAMA_FLAGS)
        fe = _fe_flag_names()

        missing_from_fe = sorted(be - fe)
        absent_from_be = sorted(fe - be)

        assert not missing_from_fe, (
            "allowlisted on the BE but with no FE control: "
            f"{missing_from_fe}. Add a FLAGS_SCHEMA entry in "
            "src/frontend/src/flagsSchema.ts, or remove the flag from "
            "SAFE_LLAMA_FLAGS if it should not be settable."
        )
        assert not absent_from_be, (
            "present in the FE schema but NOT allowlisted on the BE: "
            f"{absent_from_be}. The editor would offer a flag the manager "
            "rejects. Remove it from flagsSchema.ts or allowlist it."
        )
        assert be == fe

    def test_both_sides_are_non_trivial(self):
        """A parity assertion between two tiny sets proves nothing."""
        assert len(SAFE_LLAMA_FLAGS) > 100
        assert len(_fe_flag_names()) > 100
