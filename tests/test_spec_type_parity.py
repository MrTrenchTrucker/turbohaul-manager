"""FE/BE parity guard for the spec_type enum value list.

The backend and frontend spec_type lists do not stay
in sync automatically through flagsSchema.ts. The flag-name parity test
guards the flag-NAME set, the KV cache type parity test
guards enum VALUES but only for 4 hardcoded fields (KV_TYPED_FE_FIELDS), and
spec_type is not among them. Nothing else would catch manifest.py's SAFE_LLAMA_FLAG_STRING_ENUMS
and flagsSchema.ts's enumValues disagreeing on which spec_type strings are
accepted. This closes that specific gap the same way the KV cache type parity test
closes it for KV cache types: read the FE source directly, assert exact value-set
equality against the BE source of truth.
"""
import re
from pathlib import Path

from turbohaul.manifest import SAFE_LLAMA_FLAG_STRING_ENUMS

FRONTEND_ROOT = Path(__file__).resolve().parent.parent / "src" / "frontend"
FLAGS_SCHEMA_PATH = FRONTEND_ROOT / "src" / "flagsSchema.ts"
DIST_ASSETS_DIR = FRONTEND_ROOT / "dist" / "assets"


def _extract_enum_values(source: str, field_name: str) -> set[str]:
    match = re.search(
        r"name:\s*'" + re.escape(field_name) + r"'.*?enumValues:\s*\[([^\]]*)\]",
        source,
        re.DOTALL,
    )
    assert match is not None, f"could not find enumValues for '{field_name}' in flagsSchema.ts"
    values_blob = match.group(1)
    return set(re.findall(r"'([^']*)'", values_blob))


class TestSpecTypeFeParity:
    def test_flags_schema_exists(self):
        assert FLAGS_SCHEMA_PATH.is_file(), f"expected FE schema at {FLAGS_SCHEMA_PATH}"

    def test_fe_spec_type_matches_be_source_of_truth(self):
        be_values = SAFE_LLAMA_FLAG_STRING_ENUMS["spec_type"]
        source = FLAGS_SCHEMA_PATH.read_text()
        fe_values = _extract_enum_values(source, "spec_type")

        missing_from_fe = be_values - fe_values
        extra_in_fe = fe_values - be_values
        assert fe_values == set(be_values), (
            "'spec_type' enumValues diverged from "
            "turbohaul.manifest.SAFE_LLAMA_FLAG_STRING_ENUMS['spec_type']\n"
            f"  missing from FE: {sorted(missing_from_fe)}\n"
            f"  extra in FE:     {sorted(extra_in_fe)}"
        )

    def test_both_sides_are_non_trivial(self):
        """A parity assertion between two tiny/empty sets proves nothing."""
        assert len(SAFE_LLAMA_FLAG_STRING_ENUMS["spec_type"]) >= 3
        source = FLAGS_SCHEMA_PATH.read_text()
        assert len(_extract_enum_values(source, "spec_type")) >= 3

    def test_built_asset_ships_every_spec_type(self):
        """Guards the SHIPPED bundle, not just source -- catches a forgotten rebuild.

        Same rationale as the equivalent KV cache type parity test: this repo
        commits src/frontend/dist despite it being gitignored, so source parity
        alone doesn't guarantee the shipped bundle agrees.
        """
        built_assets = sorted(DIST_ASSETS_DIR.glob("*.js"))
        assert built_assets, f"no built JS assets found under {DIST_ASSETS_DIR} -- run npm run build"
        bundle_text = "\n".join(p.read_text() for p in built_assets)

        missing_from_bundle = sorted(
            name for name in SAFE_LLAMA_FLAG_STRING_ENUMS["spec_type"]
            if bundle_text.count(name) == 0
        )
        assert not missing_from_bundle, (
            "built asset is missing spec_type value(s) present in "
            f"SAFE_LLAMA_FLAG_STRING_ENUMS['spec_type']: {missing_from_bundle} "
            f"(checked {[p.name for p in built_assets]}) -- rebuild the frontend "
            "(cd src/frontend && npm run build) and stage dist/"
        )
