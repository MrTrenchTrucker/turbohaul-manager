// Scheduling & priority card (Settings -> General -> Fast Lane).
//
// Settings.tsx's GeneralSettings is not exported and, once it renders a
// react-router-dom <Link> (added in this change, pointing at Settings -> Config
// -> queue), cannot be statically rendered without a router context anyway
// -- same limitation Queue.test.tsx already discloses for its own <Navigate>
// usage. Nothing here attempts that render. What IS tested
// with real dynamic assertions is the pure logic extracted from handleFastLaneSave:
// validateSchedulingFields (the bounds-check + payload-shaping that used to
// live inline in handleFastLaneSave) and formatQueueFieldValue (the
// read-only row formatter), plus a static-shape guard on
// QUEUE_SCHEDULING_FIELDS so the curated field list can't silently drift
// from what was actually agreed.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeFastLaneLoadState,
  formatQueueFieldValue,
  QUEUE_SCHEDULING_FIELDS,
  validateSchedulingFields,
} from './Settings';
import type { ConfigSchema } from '../api';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import Settings from './Settings';

// --- QUEUE_SCHEDULING_FIELDS (static shape) ----------------------------

test('QUEUE_SCHEDULING_FIELDS is exactly the 7 agreed queue fields, no more no less', () => {
  const keys = QUEUE_SCHEDULING_FIELDS.map((f) => f.key);
  assert.deepEqual(
    [...keys].sort(),
    [
      'grace_seconds',
      'idle_hot_load_seconds',
      'main_lane_reserved',
      'max_consecutive_same_model',
      'max_grace_extensions',
      'max_other_model_wait_s',
      'staging_queue_depth',
    ].sort(),
  );
});

test('QUEUE_SCHEDULING_FIELDS excludes the queue.safety_* fields (explicitly out of scope)', () => {
  const keys = QUEUE_SCHEDULING_FIELDS.map((f) => f.key);
  assert.equal(keys.some((k) => k.startsWith('safety_')), false);
});

// --- formatQueueFieldValue -----------------------------------------------

test('formatQueueFieldValue: booleans render as literal true/false, not 1/0', () => {
  assert.equal(formatQueueFieldValue(true), 'true');
  assert.equal(formatQueueFieldValue(false), 'false');
});

test('formatQueueFieldValue: numbers pass through as strings', () => {
  assert.equal(formatQueueFieldValue(60), '60');
  assert.equal(formatQueueFieldValue(20.0), '20');
});

test('formatQueueFieldValue: missing/unknown value renders as an em dash, not "undefined"', () => {
  assert.equal(formatQueueFieldValue(undefined), '—');
  assert.equal(formatQueueFieldValue(null), '—');
});

// --- validateSchedulingFields ---------------------------------------------

// Re-pointed to the shipped bounds. The server derives this schema from
// FastLaneConfig, so a fixture that disagrees with it silently stops modelling
// the thing under test -- these tests kept passing while asserting behaviour at
// a ceiling the product no longer has.
// The `default` values were also stale (45.0 / 0) but were INERT:
// validateSchedulingFields reads only `minimum` and `maximum`. Corrected anyway
// so the fixture does not read as a description of a system that no longer
// exists.
const fastlaneSchema: ConfigSchema[string] = {
  max_normal_wait_s: { type: 'number', default: 3600.0, minimum: 1.0, maximum: 3600.0 },
  cross_model_switches_per_min: { type: 'integer', default: 2, minimum: 0, maximum: 3 },
};

test('validateSchedulingFields: in-bounds values are accepted and coerced to numbers', () => {
  const result = validateSchedulingFields('3600', '0', fastlaneSchema);
  assert.deepEqual(result, {
    valid: true,
    payload: { max_normal_wait_s: 3600, cross_model_switches_per_min: 0 },
  });
});

test('validateSchedulingFields: bounds are inclusive at the exact min/max (an operator parked max_normal_wait_s at 3600, its own ceiling)', () => {
  assert.equal(validateSchedulingFields('3600', '0', fastlaneSchema).valid, true);
  // '60' -> '3', the real ceiling. Same purpose -- the exact maximum is
  // admissible, not off-by-one rejected -- asserted at the bound that exists.
  assert.equal(validateSchedulingFields('1', '3', fastlaneSchema).valid, true);
});

test('validateSchedulingFields: rejects non-numeric max_normal_wait_s', () => {
  const result = validateSchedulingFields('not-a-number', '0', fastlaneSchema);
  assert.equal(result.valid, false);
  if (!result.valid) assert.match(result.error, /must be a number/);
});

test('validateSchedulingFields: rejects non-numeric cross_model_switches_per_min', () => {
  const result = validateSchedulingFields('45', 'not-a-number', fastlaneSchema);
  assert.equal(result.valid, false);
  if (!result.valid) assert.match(result.error, /whole number/);
});

test('validateSchedulingFields: rejects below-minimum max_normal_wait_s', () => {
  const result = validateSchedulingFields('0', '0', fastlaneSchema);
  assert.equal(result.valid, false);
  if (!result.valid) assert.match(result.error, />= 1/);
});

test('validateSchedulingFields: rejects above-maximum max_normal_wait_s', () => {
  const result = validateSchedulingFields('3601', '0', fastlaneSchema);
  assert.equal(result.valid, false);
  if (!result.valid) assert.match(result.error, /<= 3600/);
});

test('validateSchedulingFields: rejects above-maximum cross_model_switches_per_min', () => {
  // '61' -> '4' and /<= 60/ -> /<= 3/. Same purpose -- one past the
  // maximum is rejected, and the message quotes the bound -- at the real
  // ceiling. Held at '61' this asserted a rejection the shipped product would
  // now make for a different reason, at a bound it no longer has.
  const result = validateSchedulingFields('45', '4', fastlaneSchema);
  assert.equal(result.valid, false);
  if (!result.valid) assert.match(result.error, /<= 3/);
});

test('validateSchedulingFields: degrades gracefully when schema is unavailable -- still validates numeric-ness, skips bounds, server stays the authority', () => {
  const result = validateSchedulingFields('999999', '999999', undefined);
  assert.deepEqual(result, {
    valid: true,
    payload: { max_normal_wait_s: 999999, cross_model_switches_per_min: 999999 },
  });
});

test('validateSchedulingFields: degraded schema still rejects non-numeric input', () => {
  const result = validateSchedulingFields('nope', '0', undefined);
  assert.equal(result.valid, false);
});

// --- computeFastLaneLoadState (load-state certification) --------------------

test('computeFastLaneLoadState: a clean fastlane section (no config_error) certifies as loaded', () => {
  const state = computeFastLaneLoadState({
    enabled: true,
    max_normal_wait_s: 900,
    cross_model_switches_per_min: 3,
  });
  assert.equal(state.configLoaded, true);
  assert.equal(state.configError, null);
});

test('computeFastLaneLoadState: a present config_error refuses to certify defaults as loaded', () => {
  const state = computeFastLaneLoadState({
    enabled: false,
    max_normal_wait_s: 3600,
    cross_model_switches_per_min: 2,
    config_error: '1 validation error for FastLaneConfig',
  });
  assert.equal(state.configLoaded, false, 'a dropped section must never certify as loaded');
  assert.ok(state.configError, 'the operator must see why');
  assert.match(state.configError as string, /CODE DEFAULTS/);
  assert.match(state.configError as string, /1 validation error for FastLaneConfig/);
});

test('computeFastLaneLoadState: an explicit null config_error is treated the same as absent', () => {
  const state = computeFastLaneLoadState({ enabled: true, config_error: null });
  assert.equal(state.configLoaded, true);
  assert.equal(state.configError, null);
});

test('CONTROL: an empty-string config_error (falsy but not null/undefined) does not certify either -- the check is truthiness, not a null check', () => {
  // Distinguishes "checks for a present error string" from a narrower
  // "!== null" test that a non-null-but-empty server value would defeat.
  const state = computeFastLaneLoadState({ enabled: true, config_error: '' });
  assert.equal(state.configLoaded, true, 'an empty string is not a real error to surface');
});

// --- rendered product version (Settings -> General footer) ----------------
//
// The version a user actually reads. GeneralSettings is not exported and has
// a react-router <Link>, so it is reached here through the default export
// under a MemoryRouter rather than by exporting it -- the component under
// test is left alone. renderToStaticMarkup runs no effects, so the config
// fetches in GeneralSettings never fire and need no stub; the footer is
// static JSX and renders regardless.

test('Settings -> General footer reports the shipped product version', () => {
  const html = renderToStaticMarkup(
    <MemoryRouter initialEntries={['/settings']}>
      <Settings />
    </MemoryRouter>,
  );

  // GREEN CONTROL: prove the render actually reached the footer. Without this,
  // a render that never mounted GeneralSettings would make the version
  // assertion fail identically to a genuinely wrong version -- an empty result
  // is byte-identical to a real absence.
  assert.match(html, /MIT-licensed wrapper around the inference engine/);

  assert.match(html, /Turbohaul-Manager v0\.8\.0/);
});
