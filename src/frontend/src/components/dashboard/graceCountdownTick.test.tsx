/* DOES THE RENDERED COUNTDOWN FOLLOW THE SERVER?
 *
 * Requirement: the countdown must count down on both the frontend and the backend.
 *
 * THE GAP THIS CLOSES: a gap in the earlier grace-countdown render tests.
 * `graceCardRender.test.tsx` says in its own header: "Nothing here
 * proves the countdown TICKS." It renders ONE payload and asserts the markup contains
 * `data-value="4"`. `graceCountdown.test.ts` calls `residentCountdown()` as a pure
 * function on fixed inputs (4, 3, null). So every FE assertion in the tree is a STATIC
 * render from a FROZEN value — none of them can distinguish a live countdown from a
 * stuck one, which is exactly the property required.
 *
 * WHAT THIS FILE PROVES, AND WHY IT IS THE RIGHT SHAPE FOR THIS CARD:
 * A baseline check established that the card runs NO LOCAL CLOCK -- it renders the
 * SERVER value, and there is no `setInterval` near the countdown (firing control: 2 FE
 * files DO use setInterval and 19 use useEffect, so that detector is not blind). That
 * makes the render a PURE FUNCTION of its input, and gives the tick property a
 * deterministic, DOM-free formulation:
 *   1. two successive payloads with DIFFERENT remaining_s must render DIFFERENT text,
 *      and each must equal the server's number  -> it FOLLOWS the server;
 *   2. two renders of the SAME payload must render the SAME text -> it invents nothing
 *      on its own, which is the other half of "no local clock" and the half that a
 *      naive "it changed" test omits.
 * Together: the number on screen is the server's number, and it moves when and only
 * when the server's number moves. The BE half (that the server's number actually
 * decreases in real time) is pinned separately by
 * the backend test for the grace countdown decrementing -- neither half is sufficient alone.
 *
 * ⚠ WHAT THIS DOES NOT PROVE — stated so a green is not read as more than it is.
 * `renderToStaticMarkup` has no DOM and runs no React effects. This proves the card is a
 * faithful function of its props; it does NOT prove the app POLLS /status, nor that
 * React re-renders on new data. Those are wiring above this component and are not in
 * this file's reach.
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

/** Render the card as the dashboard would for one /status payload. */
function renderAtRemaining(remaining_s: number) {
  return renderToStaticMarkup(
    <ResidentCard
      model={makeResident({ phase: 'GRACE', remaining_s })}
      busyForS={5}
      tokRateTone={stubTokRateTone}
    />,
  );
}

const countdownValue = (html: string) =>
  html.match(/data-testid="resident-countdown"[^>]*data-value="([^"]*)"/)?.[1];
const countdownLabel = (html: string) =>
  html.match(/data-testid="resident-countdown"[^>]*data-label="([^"]*)"/)?.[1];

/* The extractor needs its own firing control: a regex that silently returns `undefined`
 * would make every comparison below vacuous. (An earlier test suffered exactly that defect --
 * a helper with the wrong testid that was never called, so nothing failed and a test
 * title claimed a property it never checked.) */
test('EXTRACTOR CONTROL: countdownValue() actually finds the number, so no assertion below can pass vacuously', () => {
  const html = renderAtRemaining(12);
  assert.equal(countdownValue(html), '12');
  assert.notEqual(countdownValue(html), undefined);
  assert.equal(countdownLabel(html), 'grace');
});

test('FE: the rendered countdown FOLLOWS the server value across two successive payloads', () => {
  const first = renderAtRemaining(30);
  const second = renderAtRemaining(28);

  // it moved...
  assert.notEqual(
    countdownValue(first),
    countdownValue(second),
    'the rendered countdown did not change when the server value changed 30 -> 28: ' +
      'the card is not following the server, and a frozen number on screen is exactly ' +
      'the defect this test exists to catch',
  );
  // ...and it moved TO THE SERVER'S NUMBER, not merely to some other number.
  assert.equal(countdownValue(first), '30');
  assert.equal(countdownValue(second), '28');
  // the visible text, not only the data attribute a user never sees
  assert.match(first, />30s</);
  assert.match(second, />28s</);
  // and it is still the GRACE countdown in both frames, not a different timer taking the slot
  assert.equal(countdownLabel(first), 'grace');
  assert.equal(countdownLabel(second), 'grace');
});

test('FE: the card invents NO countdown of its own -- identical payloads render identical text (no local clock)', () => {
  const a = renderAtRemaining(17);
  const b = renderAtRemaining(17);
  assert.equal(
    countdownValue(a),
    countdownValue(b),
    'two renders of the SAME payload produced DIFFERENT countdown values: the card is ' +
      'running a clock of its own, so what the user sees would drift away from the ' +
      'server and could keep counting after grace ended',
  );
  assert.equal(countdownValue(a), '17');
});
