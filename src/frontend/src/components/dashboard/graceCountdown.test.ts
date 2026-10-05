// The grace countdown on the resident card.
//
// Each property below is written to fail independently when the matching branch of the
// code is changed.
//
// The property that matters most is the third one: a phase label with NO seconds
// renders NOTHING. That is why assigning `ResidentState.GRACE` alone would not
// produce a countdown -- the card draws a countdown from a NUMBER, and a state
// label carries none.
import test from 'node:test';
import assert from 'node:assert/strict';
import { residentCountdown } from './ResidentCard';
import type { GenerationInfo, ResidentModel } from '../../api';

function makeGeneration(overrides: Partial<GenerationInfo> = {}): GenerationInfo {
  return {
    state: 'generating',
    prompt_progress: null,
    stalled: false,
    streaming: true,
    generation_id: 'gen-1',
    measured_at_iso: '2026-09-10T00:00:00Z',
    ...overrides,
  };
}

function makeResident(overrides: Partial<ResidentModel> = {}): ResidentModel {
  return {
    model_tag: 'model-b-27b',
    state: 'ACTIVE',
    port: 11500,
    pid: 4242,
    spawn_seq: 1,
    reserved_need_mib: 24000,
    parallel: 1,
    main_gpu: 0,
    split_mode: 'single',
    inflight: 0,
    idle_expires_in_s: null,
    generation: makeGeneration(),
    ...overrides,
  };
}

/* The GRACE arm. */
test('a resident in GRACE renders a grace countdown from remaining_s', () => {
  assert.deepEqual(residentCountdown(makeResident({ phase: 'GRACE', remaining_s: 4 })), {
    label: 'grace',
    seconds: 4,
  });
});

/* The legacy path is untouched: guards the idle fallback. */
test('the legacy idle countdown is unchanged (backend predating the resolver)', () => {
  assert.deepEqual(residentCountdown(makeResident({ idle_expires_in_s: 7 })), {
    label: 'unload in',
    seconds: 7,
  });
});

/* A label without a number draws nothing: guards against treating a null
   remaining_s as renderable. */
test('phase=GRACE with NO seconds renders NO countdown -- a state label alone carries no number to draw', () => {
  assert.equal(residentCountdown(makeResident({ phase: 'GRACE', remaining_s: null })), null);
});

/* Precedence: a live grace window wins over a stale idle field, so the card can
   never show "unload in" while the resident is actually in grace. Guards the arm
   order. */
test('GRACE takes precedence over a stale idle_expires_in_s', () => {
  assert.deepEqual(
    residentCountdown(makeResident({ phase: 'GRACE', remaining_s: 3, idle_expires_in_s: 9 })),
    { label: 'grace', seconds: 3 },
  );
});

/* Nothing to draw stays nothing to draw. */
test('no phase and no idle field -> no countdown, exactly as before', () => {
  assert.equal(residentCountdown(makeResident()), null);
});
