/* DOES THE CARD ACTUALLY RENDER THE COUNTDOWN?
 *
 * WHY THIS FILE EXISTS: it closes a wiring gap in the earlier grace-countdown tests.
 * `graceCountdown.test.ts` and `graceAlarm.test.ts` call `residentCountdown()` and
 * `residentAlarm()` as PURE FUNCTIONS. Neither one renders `ResidentCard`. So the wiring
 * — helper -> JSX -> rendered output — was UNTESTED, and nobody else asserts on
 * `resident-countdown` either (grepped: zero hits outside ResidentCard.tsx itself).
 *
 * That gap is the same kind of defect as the grace-countdown bug itself. That bug was not a wrong
 * computation; it was a branch (`state === 'GRACE'`) that could never be reached, so the
 * card confidently rendered something else. A countdown that computes perfectly and is
 * never wired into the JSX would reproduce that failure exactly, and every existing
 * test would still be green.
 *
 * WHAT THIS FILE PROVES: that with a GRACE payload on the props, the rendered markup
 * contains the countdown element, the badge reads GRACE, and the alarm token is no longer
 * `busy` -- the symptom of "busy" being shown instead of the grace timer,
 * reproduced and then closed at the render boundary.
 *
 * ⚠ WHAT IT DOES **NOT** PROVE — stated so a passing suite is not read as more than it is.
 * `renderToStaticMarkup` has no DOM, no layout engine, and DOES NOT RUN REACT EFFECTS
 * (the pre-existing ResidentCard.test.tsx discloses the same limit). Consequently:
 *   · `busyForS` is normally produced by the `useBusyTimers` EFFECT. It cannot run here, so
 *     it is supplied as a PROP. These tests prove what the card does GIVEN a busy duration;
 *     they do not prove the timer ever produces one.
 *   · Nothing here proves the countdown TICKS. A live-DOM/fake-timer test is the stronger
 *     instrument for that and is not part of this suite.
 * This file closes WIRING, not RUNTIME.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import { ResidentCard } from './ResidentCard';
import type { GenerationInfo, ResidentModel } from '../../api';

function makeGeneration(overrides: Partial<GenerationInfo> = {}): GenerationInfo {
  return {
    state: 'generating',
    prompt_progress: null,
    stalled: false,
    streaming: true,
    generation_id: 'gen-1',
    measured_at_iso: '2026-08-20T00:00:00Z',
    tok_s: 42.5,
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

function stubTokRateTone(gen: GenerationInfo | null | undefined): 'live' | 'idle' | 'stalled' {
  if (!gen) return 'idle';
  if (gen.stalled || gen.state === 'stalled' || gen.prefill_stall_alarm) return 'stalled';
  if (gen.state === 'generating' || gen.state === 'prefill') return 'live';
  return 'idle';
}

/** Same shape as the pre-existing ResidentCard.test.tsx harness (⑥c: port the
 *  assertions/idiom first, then the new logic). */
function render(overrides: Partial<Parameters<typeof ResidentCard>[0]> = {}) {
  return renderToStaticMarkup(
    <ResidentCard model={makeResident()} busyForS={null} tokRateTone={stubTokRateTone} {...overrides} />,
  );
}

const countdowns = (html: string) => (html.match(/data-testid="resident-countdown"/g) || []).length;
const alarmValue = (html: string) => html.match(/data-testid="resident-alarm" data-value="([^"]*)"/)?.[1];
const badgeText = (html: string) => html.match(/data-testid="resident-state-badge"[^>]*>([^<]*)</)?.[1];

/* ⑥b: the extractor itself needs a firing control. A regex that silently returns
 * `undefined` would make every badge assertion below vacuously comparable to
 * `undefined` and pass nothing. This asserts the extractor FINDS the badge on a
 * default card before any test relies on it. (An earlier version used the wrong testid
 * -- `resident-state` -- and because the helper was never CALLED, nothing
 * failed and the title of a test claimed a property it never checked.) */
test('EXTRACTOR CONTROL: badgeText() actually finds the state badge, so a badge assertion cannot pass vacuously', () => {
  const html = renderToStaticMarkup(
    <ResidentCard model={makeResident({ state: 'ACTIVE' })} busyForS={null} tokRateTone={stubTokRateTone} />,
  );
  assert.equal(badgeText(html), 'ACTIVE');
  assert.notEqual(badgeText(html), undefined);
});

/* ------------------------------------------------------------------ */
/*  BUSY SHOWN INSTEAD OF GRACE, REPRODUCED AT THE RENDER BOUNDARY      */
/* ------------------------------------------------------------------ */

test('RENDER: a pre-resolver backend (no phase) while busy -> alarm says "busy" and NO countdown is drawn', () => {
  const html = render({ model: makeResident(), busyForS: 5 });
  assert.equal(alarmValue(html), 'busy');
  assert.equal(countdowns(html), 0);
});

test('RENDER: with phase=GRACE the card DRAWS the countdown, the badge reads GRACE, and the busy token is GONE', () => {
  const html = render({
    model: makeResident({ phase: 'GRACE', remaining_s: 4 }),
    busyForS: 5,                       // identical busy duration to the test above
  });
  // 1. the countdown is actually in the markup -- the wiring under test
  assert.equal(countdowns(html), 1);
  assert.match(html, /data-testid="resident-countdown"[^>]*data-label="grace"/);
  assert.match(html, /data-testid="resident-countdown"[^>]*data-value="4"/);
  // 2. the seconds are VISIBLE text, not only a data attribute a user never sees
  assert.match(html, />4s</);
  // 3. the alarm label no longer occupies the slot
  assert.equal(alarmValue(html), '');
  // 4. the badge reads GRACE -- `phase` reaches `displayState`, which is the OTHER
  //    half of the unification. Without this the card could draw a countdown while
  //    still labelling itself ACTIVE, i.e. two surfaces disagreeing on one card.
  assert.equal(badgeText(html), 'GRACE');
});

test('RENDER: with NO phase the badge still falls back to model.state, so the resolver is additive on this surface too', () => {
  const html = render({ model: makeResident({ state: 'IDLE_EVICTABLE', idle_expires_in_s: 7 }) });
  assert.equal(badgeText(html), 'IDLE_EVICTABLE');
});

/* ------------------------------------------------------------------ */
/*  Each of these pins a property a regression could silently break */
/* ------------------------------------------------------------------ */

test('RENDER: the legacy idle countdown still draws for a backend predating the resolver', () => {
  const html = render({ model: makeResident({ state: 'IDLE_EVICTABLE', idle_expires_in_s: 7 }) });
  assert.equal(countdowns(html), 1);
  assert.match(html, /data-testid="resident-countdown"[^>]*data-label="unload in"/);
  assert.match(html, /data-testid="resident-countdown"[^>]*data-value="7"/);
});

test('RENDER: phase=GRACE with NO seconds draws NO countdown -- why assigning ResidentState.GRACE alone would not have fixed the card', () => {
  const html = render({ model: makeResident({ phase: 'GRACE', remaining_s: null }) });
  assert.equal(countdowns(html), 0);
});

test('RENDER: GRACE takes precedence over a stale idle_expires_in_s in the MARKUP, not just in the helper', () => {
  const html = render({ model: makeResident({ phase: 'GRACE', remaining_s: 3, idle_expires_in_s: 9 }) });
  assert.equal(countdowns(html), 1);
  assert.match(html, /data-testid="resident-countdown"[^>]*data-label="grace"/);
  assert.match(html, /data-testid="resident-countdown"[^>]*data-value="3"/);
});

test('RENDER: no phase and no idle field -> exactly zero countdown elements, card otherwise unchanged', () => {
  const html = render({ model: makeResident() });
  assert.equal(countdowns(html), 0);
  assert.match(html, /data-testid="resident-card"/);   // the card still rendered at all
});
