// Drift test: it must FAIL if the set
// of sections the FE can render ever diverges from the set the backend
// advertises.
//
// Deliberately NOT written as "assert deriveRenderableSections(realKeys) ==
// [a literal 7-name array]": that array would itself be a THIRD hand-typed
// section list (config_put.py's RUNTIME_SECTIONS and config_schema.py's
// _SECTIONS were the first two), subject to exactly the
// drift this test exists to prevent. Instead this tests the STRUCTURAL PROPERTY:
// deriveRenderableSections gates inclusion on nothing but "not a boot
// section, not a `_`-prefixed bookkeeping key" -- never on an allowlist of
// specific known names. A synthetic section name invented for this test
// file, never seen in any hardcoded union anywhere in this codebase, proves
// that: if it renders, no allowlist is gating things; if a real future
// backend section name were somehow rejected, this test would catch that a
// name-based allowlist crept back in.
//
// Run against a Config.tsx that hardcodes SectionKey to 'queue' | 'pull'
// (deriveRenderableSections missing), this file fails to even import;
// against the current Config.tsx it passes.
import test from 'node:test';
import assert from 'node:assert/strict';
import { deriveRenderableSections, renderableFields } from './Config';

// --- deriveRenderableSections: structural property, not a name allowlist ---

test('deriveRenderableSections: an unknown/synthetic section name renders — nothing gates on a name allowlist', () => {
  const result = deriveRenderableSections(['zzz_test_section_never_hardcoded_anywhere', 'queue']);
  assert.ok(result.includes('zzz_test_section_never_hardcoded_anywhere'));
  assert.ok(result.includes('queue'));
});

test('deriveRenderableSections: BOOT_SECTIONS never render, even mixed with real and synthetic sections', () => {
  const result = deriveRenderableSections([
    'server', 'storage', 'runtime', 'ui',
    'queue', 'zzz_test_section_never_hardcoded_anywhere',
  ]);
  assert.deepEqual(result, ['queue', 'zzz_test_section_never_hardcoded_anywhere']);
});

test('deriveRenderableSections: `_`-prefixed bookkeeping keys are excluded, not treated as sections', () => {
  const result = deriveRenderableSections(['queue', '_provenance', '_provenance_stamp']);
  assert.deepEqual(result, ['queue']);
});

test('deriveRenderableSections: output is sorted (stable render order)', () => {
  const result = deriveRenderableSections(['pull', 'fastlane', 'queue']);
  assert.deepEqual(result, ['fastlane', 'pull', 'queue']);
});

test('deriveRenderableSections: empty input -> empty output, never throws', () => {
  assert.deepEqual(deriveRenderableSections([]), []);
});

// --- Sanity check against today's REAL shape (illustrative, not the enforcement mechanism) ---

test('deriveRenderableSections: today\'s real GET /api/config key set renders exactly the 7 RuntimeConfig sections', () => {
  const realConfigKeys = [
    'server', 'storage', 'runtime', 'ui',
    'queue', 'pull', 'persist', 'monitor', 'kv', 'http', 'fastlane',
    '_provenance', '_provenance_stamp',
  ];
  assert.deepEqual(
    deriveRenderableSections(realConfigKeys),
    ['fastlane', 'http', 'kv', 'monitor', 'persist', 'pull', 'queue'],
  );
});

// --- renderableFields: the two data-loss shapes SKIP_FIELDS exists to prevent ---

test('renderableFields: persist.max_bytes is skipped — owned by Settings.tsx', () => {
  const result = renderableFields('persist', { max_bytes: 42949672960 });
  assert.ok(!result.includes('max_bytes'));
  assert.deepEqual(result, []);
});

test('renderableFields: fastlane skips enabled/rules/max_normal_wait_s/cross_model_switches_per_min, keeps census_ttl_hours', () => {
  const result = renderableFields('fastlane', {
    enabled: false,
    rules: [],
    max_normal_wait_s: 45.0,
    cross_model_switches_per_min: 0,
    census_ttl_hours: 168,
  });
  assert.deepEqual(result, ['census_ttl_hours']);
});

test('renderableFields: a section with no skip entry (e.g. monitor) returns every field unfiltered', () => {
  const result = renderableFields('monitor', { enabled: true, poll_interval_s: 1.0 });
  assert.deepEqual(result.sort(), ['enabled', 'poll_interval_s']);
});

test('renderableFields: kv (no skip entry) is fully generic — a section with many fields', () => {
  const result = renderableFields('kv', {
    covered_scaffold_strip: true,
    ram_cache_max_bytes: 21474836480,
    kv_save_expected_fstype: 'tmpfs',
    kv_persist_forbidden_fstypes: ['tmpfs', 'ramfs'],
  });
  assert.deepEqual(
    result.sort(),
    ['covered_scaffold_strip', 'kv_persist_forbidden_fstypes', 'kv_save_expected_fstype', 'ram_cache_max_bytes'],
  );
});
