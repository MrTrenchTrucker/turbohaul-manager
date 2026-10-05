// residentAlarm is the single authored
// verdict both ResidentCard.tsx (per-card alarm) and the aggregate banner
// (per-token count) consume -- tested as a pure function, independent of
// useBusyTimers (the stateful half that produces its busyForS parameter in
// production). React effects do not run under this repo's
// renderToStaticMarkup test harness, so useBusyTimers itself is NOT covered
// here or anywhere else in this suite. These pure-function
// tests are what make the no-telemetry escalation path verifiable at all.
import test from 'node:test';
import assert from 'node:assert/strict';
import { residentAlarm, BUSY_ESCALATE_S, PREFILL_STALL_AFTER_S } from './alarms';
import type { GenerationInfo, ResidentModel } from '../../api';

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
    inflight: 1,
    idle_expires_in_s: null,
    generation: makeGeneration(),
    ...overrides,
  };
}

/* ------------------------------------------------------------------ */
/*  generation:null -> no alarm, regardless of busyForS                 */
/* ------------------------------------------------------------------ */

test('residentAlarm: generation null -> empty token even with a nonzero busyForS', () => {
  const r = makeResident({ generation: null });
  assert.equal(residentAlarm(r, 999), '');
});

/* ------------------------------------------------------------------ */
/*  stalled                                                             */
/* ------------------------------------------------------------------ */

test('residentAlarm: gen.stalled=true -> stalled', () => {
  const r = makeResident({ generation: makeGeneration({ stalled: true, state: 'generating' }) });
  assert.equal(residentAlarm(r, null), 'stalled');
});

test('residentAlarm: gen.state==="stalled" (stalled flag false) -> stalled -- distinguishing the two triggers', () => {
  const r = makeResident({ generation: makeGeneration({ stalled: false, state: 'stalled' }) });
  assert.equal(residentAlarm(r, null), 'stalled');
});

/* ------------------------------------------------------------------ */
/*  no-telemetry / busy escalation                                      */
/* ------------------------------------------------------------------ */

test('residentAlarm: busyForS >= BUSY_ESCALATE_S -> no-telemetry', () => {
  const r = makeResident();
  assert.equal(residentAlarm(r, BUSY_ESCALATE_S), 'no-telemetry');
});

test('residentAlarm: exact boundary -- one second under BUSY_ESCALATE_S is still busy, not no-telemetry', () => {
  const r = makeResident();
  assert.equal(residentAlarm(r, BUSY_ESCALATE_S - 1), 'busy');
});

test('residentAlarm: busyForS 0 (just started) -> busy, not clear -- zero is a real elapsed reading, not "not busy"', () => {
  const r = makeResident();
  assert.equal(residentAlarm(r, 0), 'busy');
});

test('residentAlarm: busyForS null -> never busy or no-telemetry regardless of state', () => {
  const r = makeResident();
  const result = residentAlarm(r, null);
  assert.notEqual(result, 'busy');
  assert.notEqual(result, 'no-telemetry');
});

/* ------------------------------------------------------------------ */
/*  prefill-hang                                                        */
/* ------------------------------------------------------------------ */

test('residentAlarm: prefill_stall_alarm true, otherwise clean -> prefill-hang', () => {
  const r = makeResident({ generation: makeGeneration({ prefill_stall_alarm: true }) });
  assert.equal(residentAlarm(r, null), 'prefill-hang');
});

/* ------------------------------------------------------------------ */
/*  clean                                                               */
/* ------------------------------------------------------------------ */

test('residentAlarm: generation present, not stalled, busyForS null, no prefill alarm -> empty token', () => {
  const r = makeResident();
  assert.equal(residentAlarm(r, null), '');
});

/* ------------------------------------------------------------------ */
/*  precedence -- stalled > no-telemetry > prefill-hang > busy          */
/*  (distinguishing fixtures: two conditions true at once, assert the   */
/*  HIGHER one wins, which a differently-ordered impl would get wrong)  */
/* ------------------------------------------------------------------ */

test('residentAlarm PRECEDENCE: stalled beats prefill-hang when both are true', () => {
  const r = makeResident({ generation: makeGeneration({ stalled: true, prefill_stall_alarm: true }) });
  assert.equal(residentAlarm(r, null), 'stalled');
});

test('residentAlarm PRECEDENCE: no-telemetry beats prefill-hang when both are true', () => {
  const r = makeResident({ generation: makeGeneration({ prefill_stall_alarm: true }) });
  const result = residentAlarm(r, BUSY_ESCALATE_S);
  assert.equal(result, 'no-telemetry', 'a wrong impl that checks prefill-hang before the escalation branch would return prefill-hang here');
});

test('residentAlarm PRECEDENCE: prefill-hang beats busy when both are true -- the case the explicit precedence order exists for', () => {
  const r = makeResident({ generation: makeGeneration({ prefill_stall_alarm: true }) });
  const result = residentAlarm(r, 5); // non-null, under threshold -> "busy" territory
  assert.equal(result, 'prefill-hang', 'a naive busy-first early-return would mask prefill-hang behind busy here');
});

test('residentAlarm PRECEDENCE: stalled beats a simultaneous no-telemetry-shaped input, even though the two are structurally exclusive in production', () => {
  // gen.stalled=true keeps useBusyTimers' internal isLiveGeneration() true,
  // which means busyForS stays null in production whenever stalled is true --
  // but the pure function must still resolve correctly if ever called with
  // both true, since it does not itself enforce that exclusion.
  const r = makeResident({ generation: makeGeneration({ stalled: true }) });
  assert.equal(residentAlarm(r, BUSY_ESCALATE_S), 'stalled');
});

/* ------------------------------------------------------------------ */
/*  PREFILL_STALL_AFTER_S is exported for callers to build the detail   */
/*  text themselves (the shared verdict is a bare token) */
/* ------------------------------------------------------------------ */

test('PREFILL_STALL_AFTER_S and BUSY_ESCALATE_S are exported with the exact values ThroughputSection.tsx uses', () => {
  assert.equal(BUSY_ESCALATE_S, 120);
  assert.equal(PREFILL_STALL_AFTER_S, 60);
});
