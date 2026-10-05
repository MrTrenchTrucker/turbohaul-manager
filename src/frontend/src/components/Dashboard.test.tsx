// Tests the request identity strip, the speculation-downgrade widget and the
// dashboard queue card.
//
// Zero new dependencies deliberately: node_modules is vendored in-repo for
// offline builds, so this avoids checking in a vitest/RTL/jsdom dependency
// tree to test an 83-line display fix. Uses only what's already vendored:
// react-dom/server (renderToStaticMarkup needs no DOM), Node's built-in
// node:test + node:assert/strict, and esbuild (already used by vite itself)
// to transpile this file before running it — see package.json's "test"
// script for the exact invocation.
import test from 'node:test';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueueCard, RequestIdentityStrip, SpecDowngradeWidget } from './Dashboard';
import { WaitingCard } from './queue/waiting';
import type { RequestIdentity, SpecDowngradeRecord } from '../api';

function makeIdentity(overrides: Partial<RequestIdentity> = {}): RequestIdentity {
  return {
    ip: '10.0.0.5',
    model_tag: 'model-a-27b',
    session_id: 'sess-abc123',
    is_main: true,
    is_sub_agent: false,
    is_curator: false,
    is_compression: false,
    resolved_class: null,
    thread_id: 'thread-xyz',
    ...overrides,
  };
}

test('RequestIdentityStrip renders the "≠ resident" chip when the request model differs from the card model', () => {
  const requestModel = 'model-c-35b';
  const cardModel = 'model-a-27b';
  const html = renderToStaticMarkup(
    <RequestIdentityStrip
      identity={makeIdentity({ model_tag: requestModel })}
      residentModelTag={cardModel}
    />,
  );
  assert.match(
    html,
    /≠ resident/,
    `expected the "≠ resident" chip when request model (${requestModel}) != card model (${cardModel}), got: ${html}`,
  );
});

test('RequestIdentityStrip does NOT render the "≠ resident" chip when the request model matches the card model', () => {
  const sharedModel = 'model-a-27b';
  const html = renderToStaticMarkup(
    <RequestIdentityStrip
      identity={makeIdentity({ model_tag: sharedModel })}
      residentModelTag={sharedModel}
    />,
  );
  assert.doesNotMatch(
    html,
    /≠ resident/,
    `expected NO "≠ resident" chip when request model == card model (${sharedModel}), got: ${html}`,
  );
});

// Feature: the operator-saved Fast Lane label rides on the identity line
// between the IP and the role/model tag — and an IP with no saved label
// renders NOTHING (no placeholder, no 'unknown', no stray separator).

test('RequestIdentityStrip renders the saved Fast Lane label between the IP and the model tag', () => {
  const html = renderToStaticMarkup(
    <RequestIdentityStrip identity={makeIdentity({ label: 'Gateway' })} />,
  );
  const ipAt = html.indexOf('>10.0.0.5<');
  const labelAt = html.indexOf('>Gateway<');
  const modelAt = html.indexOf('>model-a-27b<');
  assert.ok(ipAt >= 0, `ip missing from: ${html}`);
  assert.ok(labelAt >= 0, `label missing from: ${html}`);
  assert.ok(modelAt >= 0, `model tag missing from: ${html}`);
  assert.ok(ipAt < labelAt, `label must come AFTER the ip, got: ${html}`);
  assert.ok(labelAt < modelAt, `label must come BEFORE the model tag (between ip and tag), got: ${html}`);
  assert.match(html, /<span class="text-cyan-400">Gateway<\/span>/, `label must be distinct from the role color, got: ${html}`);
});

test('RequestIdentityStrip renders no label span and no placeholder when the label is absent, null, or empty', () => {
  const cases: Array<Partial<RequestIdentity>> = [{}, { label: null }, { label: '' }];
  for (const overrides of cases) {
    const html = renderToStaticMarkup(
      <RequestIdentityStrip identity={makeIdentity(overrides)} />,
    );
    // ip and model tag must stay adjacent with exactly ONE separator —
    // the label's separator may not appear on its own, and no placeholder text.
    assert.match(
      html,
      /10\.0\.0\.5<\/span><span class="text-slate-600">·<\/span><span>model-a-27b<\/span>/,
      `no-label identity must render ip and model tag adjacent with a single separator (overrides=${JSON.stringify(overrides)}), got: ${html}`,
    );
  }
});

// DOWNGRADED must never read as an error/failure -- these tests
// assert that directly (no rose/red classes, the label text, correct
// per-model filtering) rather than just checking the widget renders at all.

function makeSpecDowngrade(overrides: Partial<SpecDowngradeRecord> = {}): SpecDowngradeRecord {
  return {
    model_tag: 'qwen35-shiny',
    source: 'engine_log',
    arch: 'qwen35',
    component: 'draft',
    reason_code: 'draft_context_init_failed',
    detail: 'nextn tensor missing for layer 12',
    ...overrides,
  };
}

test('SpecDowngradeWidget renders nothing when there are no records for this model', () => {
  const html = renderToStaticMarkup(
    <SpecDowngradeWidget records={[]} modelTag="qwen35-shiny" />,
  );
  assert.equal(html, '', `expected empty render with no records, got: ${html}`);
});

test('SpecDowngradeWidget renders nothing when records exist but for a different model_tag', () => {
  const html = renderToStaticMarkup(
    <SpecDowngradeWidget
      records={[makeSpecDowngrade({ model_tag: 'some-other-model' })]}
      modelTag="qwen35-shiny"
    />,
  );
  assert.equal(
    html, '',
    `expected empty render when the only record is for a different model_tag, got: ${html}`,
  );
});

test('SpecDowngradeWidget renders the informational notice, with arch/component/reason, for a matching record', () => {
  const html = renderToStaticMarkup(
    <SpecDowngradeWidget records={[makeSpecDowngrade()]} modelTag="qwen35-shiny" />,
  );
  assert.match(html, /SPECULATION DOWNGRADED/, `expected the informational label, got: ${html}`);
  assert.match(html, /qwen35/, `expected the arch value surfaced, got: ${html}`);
  assert.match(html, /draft_context_init_failed/, `expected the reason_code surfaced, got: ${html}`);
});

test('SpecDowngradeWidget NEVER uses the failure/error tone (rose/red) -- it is informational only, structurally, not by convention', () => {
  const html = renderToStaticMarkup(
    <SpecDowngradeWidget records={[makeSpecDowngrade()]} modelTag="qwen35-shiny" />,
  );
  assert.doesNotMatch(
    html, /rose|red-/,
    `SpecDowngradeWidget must never render a rose/red (error) class -- that tone is reserved for LOADING_FAIL. Got: ${html}`,
  );
  assert.doesNotMatch(
    html, /FAILED/,
    `SpecDowngradeWidget must never render the literal word FAILED -- got: ${html}`,
  );
  assert.match(
    html, /cyan/,
    `expected the distinct informational (cyan) tone, got: ${html}`,
  );
});

test('SpecDowngradeWidget shows the MOST RECENT record when multiple exist for the same model', () => {
  const html = renderToStaticMarkup(
    <SpecDowngradeWidget
      records={[
        makeSpecDowngrade({ reason_code: 'stale_reason_should_not_show' }),
        makeSpecDowngrade({ reason_code: 'draft_context_init_failed' }),
      ]}
      modelTag="qwen35-shiny"
    />,
  );
  assert.match(html, /draft_context_init_failed/, `expected the newest record's reason, got: ${html}`);
  assert.doesNotMatch(
    html, /stale_reason_should_not_show/,
    `expected only the newest record, not the stale one -- got: ${html}`,
  );
});

test('SpecDowngradeWidget handles the manager_config source (manager-config tenant) with arch=null gracefully', () => {
  const html = renderToStaticMarkup(
    <SpecDowngradeWidget
      records={[makeSpecDowngrade({
        source: 'manager_config', arch: null, component: 'draft',
        reason_code: 'spec_type_mismatch',
        detail: "spec_draft_gguf_blob_sha256 set but spec_type='draft-mtp' does not use a standalone draft model",
      })]}
      modelTag="qwen35-shiny"
    />,
  );
  assert.match(html, /SPECULATION DOWNGRADED/, `expected the widget to render with arch=null, got: ${html}`);
  assert.match(html, /spec_type_mismatch/, `expected the manager-config reason surfaced, got: ${html}`);
  assert.doesNotMatch(html, /arch=/, `expected NO arch= chip when arch is null, got: ${html}`);
});

// Explicit design decision: engine arch="unknown" (the engine
// TRIED to read general.architecture and failed -- a real, useful fact) is
// a DIFFERENT state from manager arch=null (this tenant never had an
// architecture to report at all -- irrelevant, not unknown) and the two
// must render differently, not be collapsed into the same display rule.
// arch=null -> omit the clause entirely (test above). arch="unknown" (a
// truthy string) -> render it literally, same as any other real arch value,
// with NO special-casing in the component -- proven here by using the
// literal string "unknown", not a real arch name, and asserting it appears.
test('SpecDowngradeWidget renders arch="unknown" literally (engine read failed -- a real, useful fact) -- distinct from arch=null (test above)', () => {
  const html = renderToStaticMarkup(
    <SpecDowngradeWidget
      records={[makeSpecDowngrade({ source: 'engine_log', arch: 'unknown' })]}
      modelTag="qwen35-shiny"
    />,
  );
  assert.match(
    html, /arch=unknown/,
    `expected the literal engine arch="unknown" fallback to render as-is (a real fact: the engine tried and failed), got: ${html}`,
  );
});

// ===========================================================================
// The DASHBOARD queue card: the indicator must be solid when something is
// waiting; rendering the staging depth made it flap every 10 seconds.
//
// Fixture: six requests queued:
//   queue_depth_total   SOLID at 6 for 45s at 5Hz, never dropped
//   staging_queue_depth 0, blipping to 1 for ~0.2s, twice
// The backend total is correct and the Queue TAB renders it; the dashboard
// card must render that total too, not staging.
// ===========================================================================

const SLOTS = { used: 1, max: 2 };
const BUSY_QUEUE = {
  acceptance_buffer_depth: 0,
  staging_queue_depth: 0,      // transient: 0 almost always, even while busy
  staging_queue_max: 100,
  queue_depth_total: 6,        // solid: what is REALLY waiting
};

test('dashboard queue card leads with the WAITING TOTAL, not the staging depth', () => {
  const html = renderToStaticMarkup(<QueueCard queue={BUSY_QUEUE} parallelSlots={SLOTS} />);
  const headline = html.match(/data-testid="dashboard-queue-waiting"[^>]*>([^<]*)</);
  assert.ok(headline, `headline element missing from: ${html}`);
  assert.equal(headline[1], '6',
    `the card headlines "${headline[1]}" -- with six requests waiting and staging ` +
    'transiently empty, this must read 6, not the staging depth');
});

test('dashboard queue card goes amber on the WAITING TOTAL, not on staging', () => {
  // The old card keyed amber on staging_queue_depth > 0, so with six requests
  // waiting and staging at 0 it sat calm and grey -- and flashed amber for the
  // 0.2s blips. That is the flap.
  const busy = renderToStaticMarkup(<QueueCard queue={BUSY_QUEUE} parallelSlots={SLOTS} />);
  assert.ok(busy.includes('border-amber-700'),
    'six requests waiting and the card is not amber');
  const idle = renderToStaticMarkup(
    <QueueCard queue={{ ...BUSY_QUEUE, queue_depth_total: 0 }} parallelSlots={SLOTS} />);
  assert.ok(!idle.includes('border-amber-700'),
    'CONTROL: nothing waiting must NOT be amber, or the tone assertion above is ' +
    'satisfied by a card that is amber unconditionally');
});

test('an older manager that omits queue_depth_total shows an em dash and says so, never a fabricated 0', () => {
  const { queue_depth_total, ...legacy } = BUSY_QUEUE;
  const html = renderToStaticMarkup(<QueueCard queue={legacy} parallelSlots={SLOTS} />);
  const headline = html.match(/data-testid="dashboard-queue-waiting"[^>]*>([^<]*)</);
  assert.ok(headline, `headline element missing from: ${html}`);
  assert.equal(headline[1], '\u2014', 'must render an em dash, not a number');
  assert.ok(html.includes('not reported by this manager'), 'must say WHY it is blank');
  assert.ok(!html.includes('border-amber-700'), 'unknown is not a reason to alarm');
});

test('staging, acceptance buffer and parallel slots survive as secondary detail', () => {
  // Without this, every assertion above is satisfied by a card that deleted
  // the old rows outright -- which is NOT what was asked for. They are real
  // numbers and the N / max shape is the one users are used to.
  const html = renderToStaticMarkup(<QueueCard queue={BUSY_QUEUE} parallelSlots={SLOTS} />);
  for (const frag of ['staging', '0 / 100', 'acceptance buffer', 'parallel slots', '1 / 2']) {
    assert.ok(html.includes(frag),
      `secondary detail ${frag} was dropped -- the card must ADD the total, not ` +
      `replace what was there: ${html}`);
  }
});

test('THE REGRESSION GUARD: the dashboard card and the Queue tab report the SAME waiting number', () => {
  // This bug WAS the two screens disagreeing -- the Queue tab got the corrected
  // total and the dashboard kept rendering staging. The comment in QueueCard
  // promises they cannot drift because both call one imported waitingCount;
  // a promise in a comment is a test nobody wrote, so here it is.
  const read = (html: string, id: string) => {
    const m = html.match(new RegExp(`data-testid="${id}"[^>]*>([^<]*)<`));
    assert.ok(m, `${id} not found in: ${html}`);
    return m[1];
  };
  for (const q of [
    BUSY_QUEUE,                                        // busy: 6 waiting
    { ...BUSY_QUEUE, queue_depth_total: 0 },           // idle
    { ...BUSY_QUEUE, queue_depth_total: 1 },           // the "1/100" case
  ]) {
    const dash = read(renderToStaticMarkup(<QueueCard queue={q} parallelSlots={SLOTS} />),
                      'dashboard-queue-waiting');
    const tab = read(renderToStaticMarkup(<WaitingCard queue={q} />), 'queue-waiting-total');
    assert.equal(dash, tab,
      `the two screens disagree for queue_depth_total=${q.queue_depth_total}: ` +
      `dashboard says "${dash}", queue tab says "${tab}"`);
  }
  // and they must agree on the UNKNOWN case too, not just on numbers
  const { queue_depth_total, ...legacy } = BUSY_QUEUE;
  assert.equal(
    read(renderToStaticMarkup(<QueueCard queue={legacy} parallelSlots={SLOTS} />), 'dashboard-queue-waiting'),
    read(renderToStaticMarkup(<WaitingCard queue={legacy} />), 'queue-waiting-total'),
    'the two screens disagree when the manager does not report the field');
});

// ===========================================================================
// The dashboard's queue bar must follow the corrected total, not only the
// staging count; otherwise the number climbs and the bar never moves.
// The tests above cover the HEADLINE (the number). A bar fill percentage
// computed from queue.staging_queue_depth directly would, with BUSY_QUEUE's
// numbers (staging 0, corrected total 6), sit visually empty while the
// headline read 6. These tests cover that second half.
// ===========================================================================

function barWidthPct(html: string): string {
  const m = html.match(/style="width:([\d.]+)%"/);
  assert.ok(m, `no bar width style found in: ${html}`);
  return m[1];
}

test('dashboard queue card BAR fill reflects the WAITING TOTAL, not the staging depth', () => {
  // BUSY_QUEUE: staging_queue_depth=0 (transient), queue_depth_total=6 (real).
  // Before the fix this renders width:0% while the headline above it already
  // says 6 -- the exact "number climbs, bar never moves" the architecture
  // doc names.
  const html = renderToStaticMarkup(<QueueCard queue={BUSY_QUEUE} parallelSlots={SLOTS} />);
  assert.equal(barWidthPct(html), '6',
    `bar must fill to the corrected total (6/${BUSY_QUEUE.staging_queue_max}), got: ${html}`);
});

test('CONTROL: an older manager without queue_depth_total keeps the bar on staging_queue_depth, byte-identical to today', () => {
  // Backward compatibility: a manager that has never sent queue_depth_total
  // must see exactly today's bar behaviour, not a bar that goes blank the
  // moment the corrected total is unknown.
  const { queue_depth_total, ...legacy } = BUSY_QUEUE;
  const html = renderToStaticMarkup(
    <QueueCard queue={{ ...legacy, staging_queue_depth: 37 }} parallelSlots={SLOTS} />,
  );
  assert.equal(barWidthPct(html), '37',
    `an older manager's bar must still reflect staging_queue_depth (37/${BUSY_QUEUE.staging_queue_max}) ` +
    `when queue_depth_total is absent, got: ${html}`);
});

test('a real, reported waiting total of ZERO fills the bar to 0%, not a fallback to staging depth', () => {
  // Backend invariant (queue.py: queue_depth_total = staging + accepted +
  // inbox_waiting) means the real system can never report queue_depth_total
  // BELOW staging_queue_depth -- but this unit test exercises the JS
  // operator choice directly, not a realistic backend payload: `waiting ?? x`
  // must treat a real 0 as a value, not as "absent". A `waiting || x` fix
  // would be wrong here (0 is falsy) and would silently re-introduce a
  // narrower version of the same bug this fix closes.
  const html = renderToStaticMarkup(
    <QueueCard
      queue={{ ...BUSY_QUEUE, queue_depth_total: 0, staging_queue_depth: 5 }}
      parallelSlots={SLOTS}
    />,
  );
  assert.equal(barWidthPct(html), '0',
    `a reported total of 0 must render a 0% bar even when staging_queue_depth is nonzero, got: ${html}`);
});
