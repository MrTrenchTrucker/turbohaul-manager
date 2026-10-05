// The Fast Lane subtab key moved from 'fastline' to 'fastlane'
// (/queue/fastline -> /queue/fastlane). Without help, SubTabs.tsx's own
// resolveActiveKey (a pure function, read but not modified here) treats an
// unrecognized URL segment as a silent fallback to the default tab -- an old
// bookmark to /queue/fastline would land the visitor on Queue Overview with
// no explanation, not an error page. isLegacyFastLaneQueuePath is the exact
// match this repo's Queue() component uses to decide when to render a
// <Navigate> instead of the tab host; it's exported and pure specifically so
// that decision is testable without a router (this harness has no way to
// render <Navigate>/useLocation() in a real router context).
import test from 'node:test';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import { isLegacyFastLaneQueuePath, waitingCount, WaitingCard, QueueWaitingRows } from './Queue';
import { QueueClaimRows } from './queue/claims';
import type { FastlaneClaimRow, QueueWaitingRow } from '../api';

function _row(overrides: Partial<QueueWaitingRow> = {}): QueueWaitingRow {
  return {
    position: 0,
    slot_id: 'slot-abc123',
    model_tag: 'm',
    thread_id_prefix: 'thread-a',
    state: 'STAGED',
    waited_s: 1.2,
    fastlane: null,
    floor_promoted: false,
    likely_victim: null,
    ...overrides,
  };
}

test('matches the exact legacy path', () => {
  assert.equal(isLegacyFastLaneQueuePath('/queue/fastline'), true);
});

test('matches a legacy path with a stray deeper segment (SubTabs only ever looked at the first segment anyway)', () => {
  assert.equal(isLegacyFastLaneQueuePath('/queue/fastline/anything'), true);
});

test('does NOT match the new path (must not redirect the destination onto itself)', () => {
  assert.equal(isLegacyFastLaneQueuePath('/queue/fastlane'), false);
});

test('does NOT match the Overview tab or the bare /queue path', () => {
  assert.equal(isLegacyFastLaneQueuePath('/queue'), false);
  assert.equal(isLegacyFastLaneQueuePath('/queue/overview'), false);
});

test('does NOT match a coincidentally-prefixed different path (no false positive on "fastlinefoo")', () => {
  assert.equal(isLegacyFastLaneQueuePath('/queue/fastlinefoo'), false);
});

test('does NOT match an unrelated route', () => {
  assert.equal(isLegacyFastLaneQueuePath('/models'), false);
  assert.equal(isLegacyFastLaneQueuePath('/'), false);
});

// --- the waiting total ---------------------------
//
// The UI shows a plain count of requests waiting, and it has to be correct.
// A request whose model has a live resident goes
// straight to that resident's inbox and never enters _staging, so the
// staging number reads 0 while work is genuinely queued.

test('waitingCount: reports the manager total, including inbox waiters', () => {
  assert.equal(waitingCount({ queue_depth_total: 7 }), 7);
});

test('waitingCount: zero is a REAL answer, not a missing one', () => {
  // An undercount shows up as 0, which makes 0 the suspicious value, so 0 must survive as a
  // number and never be coerced into "unknown" by a falsy check.
  assert.equal(waitingCount({ queue_depth_total: 0 }), 0);
  assert.notEqual(waitingCount({ queue_depth_total: 0 }), null);
});

test('waitingCount: an older manager yields null, NOT a fallback number', () => {
  // The obvious fallback (staging + acceptance) is precisely the undercount
  // this field exists to fix, so falling back would render a confidently
  // wrong number. Unknown is honest; wrong is not.
  assert.equal(waitingCount({}), null);
  assert.equal(waitingCount({ queue_depth_total: undefined }), null);
});

test('waitingCount: a non-finite or non-numeric value is unknown, not NaN on screen', () => {
  assert.equal(waitingCount({ queue_depth_total: NaN }), null);
  assert.equal(waitingCount({ queue_depth_total: Infinity }), null);
  assert.equal(waitingCount({ queue_depth_total: '3' as unknown as number }), null);
});

// --- the RENDER arm --------------------------------------
//
// The render arm is needed because a revert of the FE change would otherwise
// leave the suite green: nothing else rendered these. Same technique ResidentCard.test.tsx uses on
// LiveOutputBox: render the exported presentational component with props and
// assert on data-testid in the markup.

test('WaitingCard renders the waiting total where an operator can read it', () => {
  const html = renderToStaticMarkup(<WaitingCard queue={{ queue_depth_total: 7 }} />);
  assert.match(html, /data-testid="queue-waiting-total"[^>]*>\s*7\s*</);
  assert.match(html, /Requests waiting/);
});

test('WaitingCard renders an em dash, not a wrong number, when unreported', () => {
  const html = renderToStaticMarkup(<WaitingCard queue={{}} />);
  assert.match(html, /data-testid="queue-waiting-total"[^>]*>\s*—\s*</);
  assert.match(html, /does not report a waiting total/);
  // control: it must NOT invent a zero -- 0 is a REAL answer elsewhere, so
  // rendering it here would be indistinguishable from "nothing is waiting".
  assert.doesNotMatch(html, /data-testid="queue-waiting-total"[^>]*>\s*0\s*</);
});

test('WaitingCard shows a real zero as 0, not as unknown', () => {
  const html = renderToStaticMarkup(<WaitingCard queue={{ queue_depth_total: 0 }} />);
  assert.match(html, /data-testid="queue-waiting-total"[^>]*>\s*0\s*</);
  assert.doesNotMatch(html, /does not report a waiting total/);
});

// --- Section 6: the waiting-request surface rows -----

test('QueueWaitingRows: an older manager (waiting undefined) says so, not a fabricated empty list', () => {
  const html = renderToStaticMarkup(<QueueWaitingRows waiting={undefined} />);
  assert.match(html, /does not report the waiting-request surface/);
  assert.doesNotMatch(html, /queue-waiting-row/);
});

test('QueueWaitingRows: a real empty list says "Nothing waiting", distinct from "not reported"', () => {
  const html = renderToStaticMarkup(<QueueWaitingRows waiting={[]} />);
  assert.match(html, /Nothing waiting/);
  assert.doesNotMatch(html, /does not report the waiting-request surface/);
});

test('QueueWaitingRows: a listed row shows the fastlane label as client and the rank as tag', () => {
  const html = renderToStaticMarkup(
    <QueueWaitingRows
      waiting={[
        _row({
          fastlane: { rule_index: 0, rank: 1, label: 'ops-box', fastlane_rule: '10.0.0.1' },
          likely_victim: 'model-a',
        }),
      ]}
    />,
  );
  assert.match(html, /data-testid="queue-waiting-row"/);
  assert.match(html, /ops-box/);
  assert.match(html, />1</);
  assert.match(html, /waiting for a turn to complete \(likely victim: model-a\)/);
});

test('QueueWaitingRows: an unlisted row falls back to thread_id_prefix and shows an em dash for tag/rank', () => {
  const html = renderToStaticMarkup(
    <QueueWaitingRows waiting={[_row({ thread_id_prefix: 'thread-z', fastlane: null })]} />,
  );
  assert.match(html, /thread-z/);
  assert.match(html, />—</);
});

test('QueueWaitingRows: likely_victim null omits the parenthetical entirely, not "(likely victim: null)"', () => {
  const html = renderToStaticMarkup(
    <QueueWaitingRows waiting={[_row({ likely_victim: null })]} />,
  );
  assert.match(html, /waiting for a turn to complete/);
  assert.doesNotMatch(html, /likely victim/);
  assert.doesNotMatch(html, /null/);
});

// --- the Fast Lane claims strip -----
//
// A DIFFERENT population from QueueWaitingRows above (the why is documented
// in ./queue/claims itself): a claim registered via _defer_unroutable can
// genuinely precede staging, so queue.waiting is blind to it. This strip is what
// makes a waiting client visible.
// NOTE: a unit pass here proves the
// component renders correctly GIVEN DATA. It does NOT prove the live render
// -- that needs a check in a running UI, not this static harness.

function _claim(overrides: Partial<FastlaneClaimRow> = {}): FastlaneClaimRow {
  return {
    slot_id: 'slot-claim1',
    model_tag: 'model-a',
    thread_id_prefix: 'thread-c',
    fastlane: null,
    reason: 'no routable slot',
    registered_at: '2026-08-31T12:00:00Z',
    waited_s: 12.3,
    ...overrides,
  };
}

test('QueueClaimRows: an older manager (claims undefined) says so, not a fabricated empty list', () => {
  const html = renderToStaticMarkup(<QueueClaimRows claims={undefined} />);
  assert.match(html, /does not report Fast Lane claims/);
  assert.doesNotMatch(html, /queue-claim-row/);
});

test('QueueClaimRows: a real empty list says "No active claims", distinct from "not reported"', () => {
  const html = renderToStaticMarkup(<QueueClaimRows claims={[]} />);
  assert.match(html, /No active claims/);
  assert.doesNotMatch(html, /does not report Fast Lane claims/);
  assert.doesNotMatch(html, /queue-claim-row/);
});

test('QueueClaimRows: a listed claim shows model, the fastlane label as client, rank, reason, waited_s', () => {
  const html = renderToStaticMarkup(
    <QueueClaimRows
      claims={[
        _claim({
          fastlane: { rule_index: 0, rank: 2, label: 'ops-box', fastlane_rule: '10.0.0.1' },
          reason: 'resident busy',
        }),
      ]}
    />,
  );
  assert.match(html, /data-testid="queue-claim-row"/);
  assert.match(html, /model-a/);
  assert.match(html, /ops-box/);
  assert.match(html, />2</);
  assert.match(html, /resident busy/);
  assert.match(html, /12\.3s/);
});

test('QueueClaimRows: a fastlane:null (unlisted/floor-only) claim falls back to thread_id_prefix and an em dash for rank', () => {
  const html = renderToStaticMarkup(
    <QueueClaimRows claims={[_claim({ thread_id_prefix: 'thread-z', fastlane: null })]} />,
  );
  assert.match(html, /data-testid="queue-claim-row"/);
  assert.match(html, /thread-z/);
  assert.match(html, />—</);
});
