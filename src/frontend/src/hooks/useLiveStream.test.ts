// useLiveStream had ZERO existing coverage, and its interesting
// behaviour (the connection-diffing effect, the fetch/SSE-reading loop) is
// not reachable via this repo's renderToStaticMarkup (SSR) test harness --
// React effects do not run under SSR, the same limitation
// useBusyTimers already had. applyFrame/markPaneDone are the pure pieces
// extracted specifically so the core behaviour (independent per-resident
// pane state) is actually testable -- this file covers only the pure pieces;
// the effects themselves are out of reach of this harness.
import test from 'node:test';
import assert from 'node:assert/strict';
import { applyFrame, markPaneDone } from './useLiveStream';
import type { GenPane } from './useLiveStream';

/* ------------------------------------------------------------------ */
/*  KEY TEST: two residents, two independently-populated                */
/*  panes -- a single-stream/shared-key implementation must fail this.  */
/* ------------------------------------------------------------------ */

test('applyFrame: two residents streaming DISTINCT frames end up as TWO independent panes, never one shared/overwritten pane', () => {
  let panes: Record<string, GenPane> = {};
  panes = applyFrame('model-alpha', panes, { generation_id: 'gen-a', text: 'alpha output', reset: true, tok_s: 12 });
  panes = applyFrame('model-beta', panes, { generation_id: 'gen-b', text: 'beta output', reset: true, tok_s: 34 });

  assert.equal(Object.keys(panes).length, 2, 'two residents streaming must produce two panes, not one shared pane key');
  assert.equal(panes['model-alpha'].text, 'alpha output');
  assert.equal(panes['model-beta'].text, 'beta output');
  assert.notEqual(
    panes['model-alpha'].text,
    panes['model-beta'].text,
    'the two panes must never converge on the same text -- that is exactly what a single shared connection produces',
  );
});

test('applyFrame: interleaved deltas from two residents append to their OWN pane only, never cross-contaminating', () => {
  let panes: Record<string, GenPane> = {};
  panes = applyFrame('model-alpha', panes, { generation_id: 'gen-a', text: 'A1', reset: true });
  panes = applyFrame('model-beta', panes, { generation_id: 'gen-b', text: 'B1', reset: true });
  panes = applyFrame('model-alpha', panes, { generation_id: 'gen-a', text: '-A2' });
  panes = applyFrame('model-beta', panes, { generation_id: 'gen-b', text: '-B2' });
  panes = applyFrame('model-alpha', panes, { generation_id: 'gen-a', text: '-A3' });

  assert.equal(panes['model-alpha'].text, 'A1-A2-A3');
  assert.equal(panes['model-beta'].text, 'B1-B2');
});

/* ------------------------------------------------------------------ */
/*  applyFrame -- reset vs delta, tag as source of truth                */
/* ------------------------------------------------------------------ */

test('applyFrame: reset frame REPLACES the pane text, does not append', () => {
  let panes: Record<string, GenPane> = {};
  panes = applyFrame('model-a', panes, { generation_id: 'gen-1', text: 'first turn', reset: true });
  panes = applyFrame('model-a', panes, { generation_id: 'gen-2', text: 'second turn (new generation)', reset: true });
  assert.equal(panes['model-a'].text, 'second turn (new generation)');
  assert.equal(panes['model-a'].generation_id, 'gen-2');
});

test('applyFrame: delta frame with no prior pane for that tag starts fresh (empty existing text) rather than throwing', () => {
  const panes = applyFrame('model-a', {}, { generation_id: 'gen-1', text: 'delta with no reset first' });
  assert.equal(panes['model-a'].text, 'delta with no reset first');
});

test('applyFrame: model_tag on the resulting pane is the connection\'s OWN tag, never read from the frame -- ' +
  'this is the core of the design: identity comes from which stream the frame arrived on, not from parsing the frame', () => {
  const panes = applyFrame('the-real-tag', {}, { generation_id: 'gen-1', text: 'x', reset: true });
  assert.equal(panes['the-real-tag'].model_tag, 'the-real-tag');
  assert.equal(panes['the-real-tag'].paneKey, 'the-real-tag');
});

test('applyFrame: generation_id null/absent is a no-op (idle frames are handled separately, by the grace-hold timer, not here)', () => {
  const result = applyFrame('model-a', { x: 1 } as unknown as Record<string, GenPane>, {
    generation_id: null,
    text: 'should be ignored',
  });
  assert.deepEqual(result, { x: 1 });
});

test('applyFrame: tok_s history accumulates per-tag and caps at TOK_HISTORY_CAP (40), oldest dropped first', () => {
  let panes: Record<string, GenPane> = {};
  panes = applyFrame('model-a', panes, { generation_id: 'gen-1', text: '', reset: true, tok_s: 0 });
  for (let i = 1; i <= 45; i++) {
    panes = applyFrame('model-a', panes, { generation_id: 'gen-1', text: '', tok_s: i });
  }
  assert.equal(panes['model-a'].tokHistory.length, 40);
  assert.equal(panes['model-a'].tokHistory[0], 6, 'oldest samples (1-5) must have rolled off a 40-cap buffer');
  assert.equal(panes['model-a'].tokHistory[39], 45);
});

/* ------------------------------------------------------------------ */
/*  markPaneDone                                                        */
/* ------------------------------------------------------------------ */

test('markPaneDone: flips only the named tag\'s pane to done, leaves other tags untouched', () => {
  let panes: Record<string, GenPane> = {};
  panes = applyFrame('model-a', panes, { generation_id: 'gen-a', text: 'a', reset: true });
  panes = applyFrame('model-b', panes, { generation_id: 'gen-b', text: 'b', reset: true });
  const result = markPaneDone('model-a', panes);
  assert.equal(result['model-a'].done, true);
  assert.equal(result['model-b'].done, false, 'marking model-a done must not affect model-b -- the old design had ONE shared idle timer that flipped every pane at once, this must not');
});

test('markPaneDone: a tag with no existing pane is a no-op, not a crash or a phantom pane', () => {
  const result = markPaneDone('nonexistent', {});
  assert.deepEqual(result, {});
});
