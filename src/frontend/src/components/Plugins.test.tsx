// This runner has no DOM/jsdom (same
// limitation as every other test file in this package -- see
// FastLane.test.tsx's header comment). Plugins.tsx's testable surface is
// deliberately factored into pure, exported functions (formatConfiguredIndicator,
// indicatorForEntry, toggleEnabled, validatePluginRuntimeFields) so real dynamic
// assertions can run against them without a browser; wiring those functions into
// the actual onChange/onBlur handlers is verified by direct code reading
// (Plugins.tsx), the same pattern FastLane.test.tsx already establishes.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  formatConfiguredIndicator,
  indicatorForEntry,
  toggleEnabled,
  validatePluginRuntimeFields,
} from './Plugins';
import type { ConfigSchema, PluginListEntry } from '../api';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import App from '../App';
import Plugins from './Plugins';

// --- formatConfiguredIndicator -------------------------------------------

test('formatConfiguredIndicator(true) reads "Configured", ok-tone, never red', () => {
  const r = formatConfiguredIndicator(true);
  // The text is 'Configured', never 'OK': 'OK' is the word a user reads as
  // "this works", but this fact comes from a registry lookup that never
  // opened a socket.
  assert.equal(r.text, 'Configured');
  assert.equal(r.tone, 'ok');
});

test('formatConfiguredIndicator(true) must not claim reachability in its copy', () => {
  const r = formatConfiguredIndicator(true);
  assert.ok(
    !/\bok\b|reachable|online|up\b|healthy|available|connected|live/i.test(r.text),
    `the true branch must not assert reachability -- nothing probed anything: ${r.text}`,
  );
});

test('formatConfiguredIndicator(false) is amber/"pending", not an error tone', () => {
  const r = formatConfiguredIndicator(false);
  assert.equal(r.tone, 'pending');
  assert.notEqual(r.tone, 'error' as unknown);
  assert.ok(!/error|fail|invalid/i.test(r.text), `text must not read as an error: ${r.text}`);
});

// --- the WIRE-CONTRACT control -------------------------------------------
//
// The tests above hand the pure function a boolean they wrote themselves,
// which is exactly why they CANNOT catch the failure this change is most
// likely to actually commit: rename the backend's wire key, leave the FE
// reading the old one, and `entry.resolves` is `undefined` -> falsy -> all six
// plugins render "Not configured yet" while this suite stays entirely green.
//
// So the fixture below is a literal JSON STRING, parsed at runtime exactly as
// fetch().json() would parse it -- no TypeScript annotation stands between the
// key the backend emits and the key the FE reads. It byte-matches
// api/plugins.py's list_plugins() response shape.

const WIRE_RESPONSE_JSON = `{
  "plugins": [
    {
      "model_tag": "whisperx-transcribe",
      "lane": "gpu",
      "capabilities": ["transcribe", "diarize"],
      "configured": true,
      "provides_routes": [],
      "provides_executables": [],
      "invoke": {
        "method": "POST",
        "url": "/api/plugins/whisperx-transcribe/invoke",
        "body": { "path": "<one of provides_routes>", "payload": {} }
      }
    }
  ],
  "total": 1
}`;

function firstWireEntry(): Record<string, unknown> {
  const parsed = JSON.parse(WIRE_RESPONSE_JSON) as {
    plugins: Array<Record<string, unknown>>;
    total: number;
  };
  return parsed.plugins[0];
}

test('wire contract: indicatorForEntry reads the key the backend actually emits', () => {
  // Goes RED if Plugins.tsx's key-read drifts from api/plugins.py's wire key
  // -- the whole point of this control.
  const entry = firstWireEntry() as unknown as PluginListEntry;
  const indicator = indicatorForEntry(entry);
  assert.equal(indicator.text, 'Configured');
  assert.equal(indicator.tone, 'ok');
});

test('wire contract: the reachability-named key is gone and `configured` is present', () => {
  const entry = firstWireEntry();
  assert.ok('configured' in entry, 'the backend must emit `configured`');
  assert.ok(
    !('resolves' in entry),
    'the old reachability-named key must not come back: GET /api/plugins opens no socket, ' +
      'so nothing in its response may assert reachability under any name',
  );
});

test('a key the wire does not carry fails SILENTLY to amber -- the failure mode, pinned', () => {
  // Deliberately reads a key that is NOT on the wire, to prove the failure is
  // silent: no throw, no runtime type error, just a badge that quietly lies in
  // the other direction. This is the executable justification for why the
  // rename had to land atomically across wire + FE + docs in one change.
  const entry = firstWireEntry();
  const wrong = formatConfiguredIndicator(entry.resolves as boolean);
  assert.equal(wrong.text, 'Not configured yet');
  assert.equal(wrong.tone, 'pending');
});

// --- toggleEnabled (pure, immutable map update) -------------------------

test('toggleEnabled sets only the targeted model_tag key', () => {
  const before = { 'model-a': true, 'model-b': false };
  const after = toggleEnabled(before, 'model-b', true);
  assert.equal(after['model-b'], true);
  assert.equal(after['model-a'], true, 'model-a must be untouched');
});

test('toggleEnabled does not mutate its input', () => {
  const before = { 'model-a': true };
  const before_copy = { ...before };
  toggleEnabled(before, 'model-a', false);
  assert.deepEqual(before, before_copy, 'input object must be unchanged (immutability)');
});

test('toggleEnabled can add a new key not previously present', () => {
  const after = toggleEnabled({}, 'new-plugin', false);
  assert.deepEqual(after, { 'new-plugin': false });
});

// --- validatePluginRuntimeFields -----------------------------------------

const noSchema: ConfigSchema[string] | undefined = undefined;
const schemaWithBounds: ConfigSchema[string] = {
  max_concurrent: { type: 'integer', default: 4, minimum: 1, maximum: 64 },
  no_progress_timeout_s: { type: 'number', default: 600, minimum: 0, maximum: 86400 },
};

test('validatePluginRuntimeFields accepts valid numeric input', () => {
  const r = validatePluginRuntimeFields('8', '300', schemaWithBounds);
  assert.equal(r.valid, true);
  if (r.valid) {
    assert.deepEqual(r.payload, { max_concurrent: 8, no_progress_timeout_s: 300 });
  }
});

test('validatePluginRuntimeFields rejects non-numeric max_concurrent', () => {
  const r = validatePluginRuntimeFields('not-a-number', '300', schemaWithBounds);
  assert.equal(r.valid, false);
  if (!r.valid) assert.match(r.error, /max concurrent/i);
});

test('validatePluginRuntimeFields rejects non-numeric timeout', () => {
  const r = validatePluginRuntimeFields('8', 'nope', schemaWithBounds);
  assert.equal(r.valid, false);
  if (!r.valid) assert.match(r.error, /timeout/i);
});

test('validatePluginRuntimeFields enforces the schema minimum for max_concurrent', () => {
  const r = validatePluginRuntimeFields('0', '300', schemaWithBounds);
  assert.equal(r.valid, false);
  if (!r.valid) assert.match(r.error, />=\s*1/);
});

test('validatePluginRuntimeFields enforces the schema maximum for max_concurrent', () => {
  const r = validatePluginRuntimeFields('999', '300', schemaWithBounds);
  assert.equal(r.valid, false);
  if (!r.valid) assert.match(r.error, /<=\s*64/);
});

test('validatePluginRuntimeFields enforces the schema maximum for the timeout', () => {
  const r = validatePluginRuntimeFields('8', '999999', schemaWithBounds);
  assert.equal(r.valid, false);
  if (!r.valid) assert.match(r.error, /<=\s*86400/);
});

test('validatePluginRuntimeFields degrades gracefully when schema is unavailable -- still validates numeric-ness, skips bounds, server stays the authority', () => {
  const r = validatePluginRuntimeFields('0', '999999', noSchema);
  assert.equal(r.valid, true, 'out-of-range values must not be rejected client-side without a schema');
  if (r.valid) {
    assert.deepEqual(r.payload, { max_concurrent: 0, no_progress_timeout_s: 999999 });
  }
});

test('validatePluginRuntimeFields degraded schema still rejects non-numeric input', () => {
  const r = validatePluginRuntimeFields('abc', '300', noSchema);
  assert.equal(r.valid, false);
});

// --- work-in-progress marker ---------------------------------------------
//
// renderToStaticMarkup runs no effects, so the plugin list never loads and
// this renders the page's first, empty state (same pattern as
// Settings.test.tsx). That is the point: the banner has to be there before
// any data arrives, and when nothing is configured or the list fails to
// load, not only once a plugin table exists.

const WIP_BANNER = 'Work in progress — feedback and contributions welcome';
const PLUGINS_INTRO = 'Turbohaul ships no media tooling itself';

test('Plugins page shows the work-in-progress banner once, above the intro, before any plugin has loaded', () => {
  const html = renderToStaticMarkup(<Plugins />);
  // Sanity check: the render reached the page. Without it, a render that
  // produced nothing would fail the banner assertion identically to a page
  // that really lacks the banner.
  const intro = html.indexOf(PLUGINS_INTRO);
  assert.ok(intro >= 0, 'control: the intro text must be in the static render');
  const banner = html.indexOf(WIP_BANNER);
  assert.ok(banner >= 0, `the banner text is missing from the Plugins page: ${WIP_BANNER}`);
  assert.equal(html.split(WIP_BANNER).length - 1, 1, 'the banner must appear exactly once');
  assert.ok(banner < intro, 'the banner must come before the intro box');
});

test('the nav tab for the plugins route reads "Plugins (WIP)"', () => {
  const html = renderToStaticMarkup(
    <MemoryRouter initialEntries={['/plugins']}>
      <App />
    </MemoryRouter>,
  );
  // GREEN CONTROL: the nav rendered, with a neighbouring tab that is not touched.
  assert.match(html, /<a [^>]*href="\/settings"[^>]*>Settings<\/a>/);
  assert.match(html, /<a [^>]*href="\/plugins"[^>]*>Plugins \(WIP\)<\/a>/);
});
