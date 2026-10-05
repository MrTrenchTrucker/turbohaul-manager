"""FE/BE parity guard for KV cache quant type enums.

The type list lives once, in turbohaul.safety.KV_CACHE_TYPES. flagsSchema.ts
is a static, hand-maintained frontend file with no BE endpoint behind it, so
this test is the anti-drift mechanism: it reads the FE source directly and
asserts every KV-typed field's enumValues matches the BE constant exactly.

This repo commits its build output (src/frontend/dist is tracked despite
being gitignored), so source parity alone isn't enough -- the SHIPPED bundle
can silently disagree with source+BE if a rebuild is forgotten. The second
check below guards the built asset directly.
"""
import re
from pathlib import Path

from turbohaul.safety import KV_CACHE_TYPES

FRONTEND_ROOT = Path(__file__).resolve().parent.parent / "src" / "frontend"
FLAGS_SCHEMA_PATH = FRONTEND_ROOT / "src" / "flagsSchema.ts"
DIST_ASSETS_DIR = FRONTEND_ROOT / "dist" / "assets"

# Every FE field whose enumValues must equal KV_CACHE_TYPES. Add a name here
# when a new KV-typed flag is introduced -- no other change needed.
KV_TYPED_FE_FIELDS = [
    "cache_type_k",
    "cache_type_v",
    "spec_draft_type_k",
    "spec_draft_type_v",
]


def _extract_enum_values(source: str, field_name: str) -> set[str]:
    match = re.search(
        r"name:\s*'" + re.escape(field_name) + r"'.*?enumValues:\s*\[([^\]]*)\]",
        source,
        re.DOTALL,
    )
    assert match is not None, f"could not find enumValues for '{field_name}' in flagsSchema.ts"
    values_blob = match.group(1)
    return set(re.findall(r"'([^']*)'", values_blob))


class TestKvCacheTypeFeParity:
    def test_flags_schema_exists(self):
        assert FLAGS_SCHEMA_PATH.is_file(), f"expected FE schema at {FLAGS_SCHEMA_PATH}"

    def test_fe_kv_typed_fields_match_be_source_of_truth(self):
        source = FLAGS_SCHEMA_PATH.read_text()
        for field_name in KV_TYPED_FE_FIELDS:
            fe_values = _extract_enum_values(source, field_name)
            missing_from_fe = KV_CACHE_TYPES - fe_values
            extra_in_fe = fe_values - KV_CACHE_TYPES
            assert fe_values == set(KV_CACHE_TYPES), (
                f"'{field_name}' enumValues diverged from turbohaul.safety.KV_CACHE_TYPES\n"
                f"  missing from FE: {sorted(missing_from_fe)}\n"
                f"  extra in FE:     {sorted(extra_in_fe)}"
            )

    def test_built_asset_ships_every_kv_cache_type(self):
        """Guards the SHIPPED bundle, not just source -- catches a forgotten rebuild.

        Literal-string presence, not line matching: minified bundles have no
        stable line structure, but the type names are runtime enum values
        (dropdown options / API payload strings) so they survive minification
        as intact string literals.
        """
        built_assets = sorted(DIST_ASSETS_DIR.glob("*.js"))
        assert built_assets, f"no built JS assets found under {DIST_ASSETS_DIR} -- run npm run build"
        bundle_text = "\n".join(p.read_text() for p in built_assets)

        missing_from_bundle = sorted(
            name for name in KV_CACHE_TYPES if bundle_text.count(name) == 0
        )
        assert not missing_from_bundle, (
            "built asset is missing KV cache type string(s) present in "
            f"turbohaul.safety.KV_CACHE_TYPES: {missing_from_bundle} "
            f"(checked {[p.name for p in built_assets]}) -- rebuild the frontend "
            "(cd src/frontend && npm run build) and stage dist/"
        )
