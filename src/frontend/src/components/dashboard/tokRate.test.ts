// The tok/s colour rule as a tested pure function. The resident card applies
// tokRateTone() in ResidentCard.tsx; this file only proves the rule itself.
import test from 'node:test';
import assert from 'node:assert/strict';
import { tokRateTone } from './tokRate';
import type { GenerationInfo } from '../../api';

function makeGeneration(overrides: Partial<GenerationInfo> = {}): GenerationInfo {
  return {
    state: 'generating',
    prompt_progress: null,
    stalled: false,
    streaming: true,
    generation_id: 'gen-1',
    measured_at_iso: '2026-08-20T00:00:00Z',
    ...overrides,
  };
}

/* ------------------------------------------------------------------ */
/*  null / undefined                                                   */
/* ------------------------------------------------------------------ */

test('tokRateTone: null generation -> idle', () => {
  assert.equal(tokRateTone(null), 'idle');
});

test('tokRateTone: undefined generation -> idle', () => {
  assert.equal(tokRateTone(undefined), 'idle');
});

/* ------------------------------------------------------------------ */
/*  every real backend state, both sides of the live/idle boundary     */
/* ------------------------------------------------------------------ */

test('tokRateTone: state="generating" with a real tok_s -> live', () => {
  const gen = makeGeneration({ state: 'generating', tok_s: 42.5 });
  assert.equal(tokRateTone(gen), 'live');
});

test(
  'tokRateTone: state="generating" with tok_s===null (EWMA-not-warmed-up transient) -> live -- ' +
    'distinguishing fixture: an active card must not look idle. ' +
    'A wrong impl that gates on tok_s truthiness instead of state would return idle here.',
  () => {
    const gen = makeGeneration({ state: 'generating', tok_s: undefined });
    assert.equal(tokRateTone(gen), 'live');
  },
);

test(
  'tokRateTone: state="prefill" (tok_s is ALWAYS null during real prefill) -> live -- ' +
    'distinguishing fixture: a resident mid-prefill is genuinely working; a tok_s-truthy-only ' +
    'implementation would render it idle, the same class of bug as the generating case above.',
  () => {
    const gen = makeGeneration({ state: 'prefill', tok_s: undefined, prompt_progress: '0.42' });
    assert.equal(tokRateTone(gen), 'live');
  },
);

test(
  'tokRateTone: state="stalled" (backend forces tok_s=0.0, a real value, not null) -> stalled, ' +
    'NOT idle -- distinguishing fixture: a tok_s>0-only rule would read this resident as calmly ' +
    'idle, exactly the "not stalled" behaviour.',
  () => {
    const gen = makeGeneration({ state: 'stalled', tok_s: 0, stalled: true });
    assert.equal(tokRateTone(gen), 'stalled');
  },
);

test('tokRateTone: state="finishing" (backend sets tok_s=0.0, a healthy wind-down) -> idle', () => {
  const gen = makeGeneration({ state: 'finishing', tok_s: 0 });
  assert.equal(tokRateTone(gen), 'idle');
});

test('tokRateTone: state="idle" (backend sets tok_s=0.0) -> idle', () => {
  const gen = makeGeneration({ state: 'idle', tok_s: 0 });
  assert.equal(tokRateTone(gen), 'idle');
});

test('tokRateTone: state="transitioning" (tok_s null) -> idle', () => {
  const gen = makeGeneration({ state: 'transitioning', tok_s: undefined });
  assert.equal(tokRateTone(gen), 'idle');
});

test('tokRateTone: state="loading" (tok_s null, no decode activity at all) -> idle', () => {
  const gen = makeGeneration({ state: 'loading', tok_s: undefined });
  assert.equal(tokRateTone(gen), 'idle');
});

test('tokRateTone: an unrecognized/future state string -> idle (safe default, never crashes)', () => {
  const gen = makeGeneration({ state: 'some-future-state', tok_s: 99 });
  assert.equal(tokRateTone(gen), 'idle');
});

/* ------------------------------------------------------------------ */
/*  gen.stalled boolean (defensive OR, mirrors residentAlarm's own      */
/*  predicate: gen.stalled || gen.state === 'stalled')                  */
/* ------------------------------------------------------------------ */

test('tokRateTone: gen.stalled===true with a non-"stalled" state string -> stalled', () => {
  const gen = makeGeneration({ state: 'generating', stalled: true, tok_s: 0 });
  assert.equal(tokRateTone(gen), 'stalled');
});

test('tokRateTone: gen.stalled===false and state==="stalled" -> stalled (state string alone is sufficient)', () => {
  const gen = makeGeneration({ state: 'stalled', stalled: false, tok_s: 0 });
  assert.equal(tokRateTone(gen), 'stalled');
});

/* ------------------------------------------------------------------ */
/*  prefill_stall_alarm fold-in                                       */
/* ------------------------------------------------------------------ */

test(
  'tokRateTone: prefill_stall_alarm===true during state="prefill" -> stalled, NOT live -- ' +
    'proves the fold-in actually overrides the prefill-is-live rule rather than being silently dropped',
  () => {
    const gen = makeGeneration({
      state: 'prefill',
      tok_s: undefined,
      prefill_stall_alarm: true,
    });
    assert.equal(tokRateTone(gen), 'stalled');
  },
);

test('tokRateTone: prefill_stall_alarm===false during state="prefill" -> live (the alarm must actually gate, not just be present)', () => {
  const gen = makeGeneration({
    state: 'prefill',
    tok_s: undefined,
    prefill_stall_alarm: false,
  });
  assert.equal(tokRateTone(gen), 'live');
});

test('tokRateTone: prefill_stall_alarm undefined (field absent) during state="prefill" -> live, never crashes on a missing optional field', () => {
  const gen = makeGeneration({ state: 'prefill', tok_s: undefined });
  delete (gen as { prefill_stall_alarm?: boolean }).prefill_stall_alarm;
  assert.equal(tokRateTone(gen), 'live');
});
