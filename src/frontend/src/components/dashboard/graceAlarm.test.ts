// Why the card said "busy", and the narrow suppression that stops it.
//
// WHY IT SAID BUSY, AND WHY THAT WAS NOT A MISLABEL. `busyForS` comes from
// useBusyTimers, which detects TELEMETRY GOING QUIET. The backend starts a resident's
// grace window in `_serve_on_resident` at TURN COMPLETION -- precisely the moment
// telemetry goes quiet. So a detector whose only evidence is quiet MUST classify every
// grace window as "engine busy -- telemetry paused". Nothing in the payload
// distinguished "quiet because the turn ended" from "quiet because the engine is
// stuck". The card was not failing to show grace; it was confidently showing the one
// other thing that looks identical from the outside. `phase` is that discriminator.
//
// ⛔ THIS IS THE DANGEROUS HALF OF THE CHANGE: it deliberately hides an alarm operators
// rely on. It therefore tests the cases that must NOT be suppressed, and not only the
// suppression itself -- `busy` MUST still fire when phase is not GRACE, and the other
// three tokens MUST still fire even during GRACE. A suppression proven only to suppress
// is half-proven.
//
// NOTE ON THE RED: taken with `alarms.ts` at its pre-change content. `api.ts`'s additive
// `phase?` declaration is present, because a TypeScript field declaration is erased at
// runtime and cannot change behaviour -- reverting it would only have produced a
// compile error, and a test that cannot run is not a watched RED.
import test from 'node:test';
import assert from 'node:assert/strict';
import { residentAlarm } from './alarms';
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

/* ================================================================== */
/*  THE SUPPRESSION -- the single property this adds to alarms.ts      */
/* ================================================================== */

test('phase=GRACE suppresses the busy token -- the word operators saw', () => {
  assert.equal(residentAlarm(makeResident({ phase: 'GRACE', remaining_s: 4 }), 5), '');
});

/* ================================================================== */
/*  busy MUST STILL FIRE when phase is not GRACE. Each of these three  */
/*  tests fails on its own if the suppression is made unconditional.   */
/* ================================================================== */

test('busy STILL FIRES when phase is absent (a backend predating the resolver)', () => {
  assert.equal(residentAlarm(makeResident(), 5), 'busy');
});

test('busy STILL FIRES when phase is ACTIVE', () => {
  assert.equal(residentAlarm(makeResident({ phase: 'ACTIVE' }), 5), 'busy');
});

test('busy STILL FIRES when phase is IDLE_EVICTABLE', () => {
  assert.equal(residentAlarm(makeResident({ phase: 'IDLE_EVICTABLE', remaining_s: 9 }), 5), 'busy');
});

/* ================================================================== */
/*  The suppression is NARROW. A genuinely                            */
/*  wedged engine must still alarm even while phase says GRACE.        */
/*  These are the tests that stop the suppression hiding a real fault. */
/* ================================================================== */

test('stalled still fires during GRACE -- suppression must not hide a wedged engine', () => {
  const r = makeResident({ phase: 'GRACE', generation: makeGeneration({ stalled: true }) });
  assert.equal(residentAlarm(r, 5), 'stalled');
});

test('no-telemetry still fires during GRACE', () => {
  assert.equal(residentAlarm(makeResident({ phase: 'GRACE' }), 120), 'no-telemetry');
});

test('prefill-hang still fires during GRACE', () => {
  const r = makeResident({
    phase: 'GRACE',
    generation: makeGeneration({ prefill_stall_alarm: true }),
  });
  assert.equal(residentAlarm(r, null), 'prefill-hang');
});
