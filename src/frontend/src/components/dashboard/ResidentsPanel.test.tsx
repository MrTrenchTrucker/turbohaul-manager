// ResidentsPanel.tsx previously had no direct coverage. Covers the
// panel's own responsibilities: rendering one card per resident (0/1/N>=2),
// stable display order, and per-resident pane resolution -- NOT alarm
// computation (useBusyTimers is a stateful hook; React effects do not run
// under this repo's renderToStaticMarkup harness, so every busyForS read
// through a live ResidentsPanel render is null here regardless of input --
// see ResidentCard.test.tsx, which exercises every alarm token by supplying
// busyForS directly as a prop, bypassing the hook).
import test from 'node:test';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import { ResidentsPanel } from './ResidentsPanel';
import type { GenerationInfo, ResidentModel } from '../../api';
import type { GenPane } from '../../hooks/useLiveStream';

// tokRateTone is injected (it lives in another module and is not
// imported here). This stub is only for supplying the required prop with
// realistic-shaped behaviour -- ResidentsPanel threads it straight through
// without calling it itself, so no test here depends on its exact logic.
function stubTokRateTone(gen: GenerationInfo | null | undefined): 'live' | 'idle' | 'stalled' {
  if (!gen) return 'idle';
  if (gen.stalled || gen.state === 'stalled' || gen.prefill_stall_alarm) return 'stalled';
  if (gen.state === 'generating' || gen.state === 'prefill') return 'live';
  return 'idle';
}

function makeGeneration(overrides: Partial<GenerationInfo> = {}): GenerationInfo {
  return {
    state: 'generating',
    prompt_progress: null,
    stalled: false,
    streaming: true,
    generation_id: 'gen-1',
    measured_at_iso: '2026-08-20T00:00:00Z',
    tok_s: 10,
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

function makePane(overrides: Partial<GenPane> = {}): GenPane {
  return {
    paneKey: 'p',
    generation_id: 'gen-1',
    model_tag: null,
    text: '',
    done: false,
    lastTokS: null,
    lastFrameAt: Date.now(),
    tokHistory: [],
    ...overrides,
  };
}

function render(residents: ResidentModel[], panes: Record<string, GenPane> = {}) {
  return renderToStaticMarkup(
    <ResidentsPanel
      residents={residents}
      vram={null}
      vramTotal={null}
      parallelSlots={{ used: 0, max: 4 }}
      panes={panes}
      tokRateTone={stubTokRateTone}
    />,
  );
}

function cardCount(html: string): number {
  return (html.match(/data-testid="resident-card"/g) || []).length;
}

function tagsInOrder(html: string): string[] {
  return [...html.matchAll(/data-testid="resident-tag"[^>]*>([^<]*)</g)].map(m => m[1]);
}

function dataValue(html: string, testId: string): string | undefined {
  const m = html.match(new RegExp(`data-testid="${testId}"[^>]*data-value="([^"]*)"`));
  return m?.[1];
}

/* ------------------------------------------------------------------ */
/*  0 / 1 / N>=2 residents                                              */
/* ------------------------------------------------------------------ */

test('ResidentsPanel: 0 residents -> panel renders, zero resident-card elements, VramPlaceholder shown', () => {
  const html = render([]);
  assert.match(html, /data-testid="residents-panel"/);
  assert.equal(cardCount(html), 0);
  assert.match(html, /VRAM/);
});

test('ResidentsPanel: 1 resident -> exactly one resident-card', () => {
  const html = render([makeResident()]);
  assert.equal(cardCount(html), 1);
});

test('ResidentsPanel: N>=2 residents -> one resident-card each, count matches input length exactly', () => {
  const html = render([
    makeResident({ model_tag: 'a' }),
    makeResident({ model_tag: 'b' }),
    makeResident({ model_tag: 'c' }),
  ]);
  assert.equal(cardCount(html), 3);
});

/* ------------------------------------------------------------------ */
/*  Grid layout: desktop 2-up, wraps to new rows                         */
/* ------------------------------------------------------------------ */

test(
  'ResidentsPanel grid layout: the grid className is EXACTLY "grid grid-cols-1 lg:grid-cols-2 gap-3" -- ' +
    'this pins the grid class exactly. ' +
    'renderToStaticMarkup has no layout engine (same disclosed limit ' +
    'as in the throughput box tests): this proves the class STRING is exactly right -- mobile stays grid-cols-1 ' +
    'unconditionally (the unprefixed, base class), lg: adds the second column, gap-3 is unchanged -- and ' +
    'that the grid still renders every resident as a child, in the correct sort order, across a count that ' +
    'forces wrapping (4, i.e. two full rows of two). It does NOT and cannot prove that two cards visually ' +
    'sit side-by-side without crowding at any real viewport width -- that needs a live browser capture ' +
    '(1680 desktop 2-up, a forced-4-children wrap check, 412px mobile unchanged), not this test.',
  () => {
    const html = render([
      makeResident({ model_tag: 'delta' }),
      makeResident({ model_tag: 'alpha' }),
      makeResident({ model_tag: 'charlie' }),
      makeResident({ model_tag: 'bravo' }),
    ]);
    assert.match(
      html,
      /<div class="grid grid-cols-1 lg:grid-cols-2 gap-3">/,
      'exact className string, not a substring/toContain match',
    );
    assert.equal(cardCount(html), 4, 'all four residents render as cards -- the grid wraps them, none are dropped');
    assert.deepEqual(
      tagsInOrder(html),
      ['alpha', 'bravo', 'charlie', 'delta'],
      'sort order (the grid change did not touch sortForDisplay/ordered.map) is unaffected by the new column count',
    );
  },
);

/* ------------------------------------------------------------------ */
/*  Stable order: model_tag primary, spawn_seq tie-break                 */
/* ------------------------------------------------------------------ */

test('ResidentsPanel ORDER: cards render alphabetically by model_tag regardless of input array order -- distinguishing fixture', () => {
  const html = render([
    makeResident({ model_tag: 'zulu-model' }),
    makeResident({ model_tag: 'alpha-model' }),
    makeResident({ model_tag: 'mike-model' }),
  ]);
  assert.deepEqual(tagsInOrder(html), ['alpha-model', 'mike-model', 'zulu-model']);
});

test('ResidentsPanel ORDER: does not mutate or reorder-in-place the caller\'s array (display order only)', () => {
  const input = [makeResident({ model_tag: 'zulu' }), makeResident({ model_tag: 'alpha' })];
  const before = input.map(r => r.model_tag);
  render(input);
  assert.deepEqual(input.map(r => r.model_tag), before, 'the prop array itself must be untouched -- set identity');
});

/* ------------------------------------------------------------------ */
/*  generation:null mixed with live residents                           */
/* ------------------------------------------------------------------ */

test('ResidentsPanel: generation:null resident mixed with a live one -- both get cards, tok-s data-value 0 vs real', () => {
  const html = render([
    makeResident({ model_tag: 'live-one', generation: makeGeneration({ tok_s: 33 }) }),
    makeResident({ model_tag: 'null-gen-one', state: 'GRACE', generation: null }),
  ]);
  assert.equal(cardCount(html), 2);
  // alphabetical: live-one, null-gen-one
  const tokValues = [...html.matchAll(/data-testid="resident-tok-s"[^>]*data-value="([^"]*)"/g)].map(m => m[1]);
  assert.deepEqual(tokValues, ['33', '0']);
});

/* ------------------------------------------------------------------ */
/*  Pane resolution -- matched once per resident by model_tag, never    */
/*  swapped between cards                                               */
/* ------------------------------------------------------------------ */

// useLiveStream now opens one connection PER RESIDENT TAG, so
// panes are always keyed by the resident's own model_tag directly -- these
// fixtures key panes that way, matching what useLiveStream actually
// produces (the old paneKey-as-arbitrary-string / null-tag-fallback
// fixtures below were testing a design that no longer exists; the
// decision was to remove the dead branch, not just stop testing it, so the
// tests that only existed to cover it are removed here, not updated).
test('ResidentsPanel PANES: each card gets its OWN matching pane by model_tag, never a sibling\'s -- distinguishing fixture', () => {
  const html = render(
    [
      makeResident({ model_tag: 'model-a', generation: makeGeneration() }),
      makeResident({ model_tag: 'model-b', generation: makeGeneration() }),
    ],
    {
      'model-a': makePane({ paneKey: 'model-a', model_tag: 'model-a', text: 'TEXT-FOR-A' }),
      'model-b': makePane({ paneKey: 'model-b', model_tag: 'model-b', text: 'TEXT-FOR-B' }),
    },
  );
  assert.match(html, /TEXT-FOR-A/);
  assert.match(html, /TEXT-FOR-B/);
  // Order in the DOM is alphabetical (model-a, model-b) -- A's text must
  // appear before B's, not swapped.
  assert.ok(html.indexOf('TEXT-FOR-A') < html.indexOf('TEXT-FOR-B'));
});

test('ResidentsPanel PANES: a pane keyed under a DIFFERENT tag than any current resident is invisible to all cards -- direct-lookup replaces the old ambiguous-guess fallback', () => {
  const html = render(
    [makeResident({ model_tag: 'only-one', generation: makeGeneration() })],
    { 'some-other-tag': makePane({ paneKey: 'some-other-tag', model_tag: 'some-other-tag', text: 'SHOULD-NOT-APPEAR' }) },
  );
  assert.doesNotMatch(html, /SHOULD-NOT-APPEAR/);
});

test('ResidentsPanel: no pane at all for a resident -- that card renders with no live-output section, not a crash', () => {
  const html = render([makeResident({ model_tag: 'lonely', generation: makeGeneration() })], {});
  assert.equal(cardCount(html), 1);
  assert.doesNotMatch(html, /live output|last output/);
});

/* ------------------------------------------------------------------ */
/*  soleResident wiring                                                 */
/* ------------------------------------------------------------------ */

test('ResidentsPanel: soleResident is true for exactly 1 resident, false for N>=2 -- verified indirectly via RequestIdentityStrip only rendering under sole-or-matching-tag', () => {
  // With 2 residents and requestIdentity.model_tag matching NEITHER
  // resident's own tag, only a sole resident would still show the strip
  // (soleResident=true bypasses the tag-match requirement). Asserting the
  // strip does NOT appear on a >=2-resident panel under a non-matching
  // identity confirms soleResident is false there, not silently stuck true.
  const html = renderToStaticMarkup(
    <ResidentsPanel
      residents={[
        makeResident({ model_tag: 'model-a' }),
        makeResident({ model_tag: 'model-b' }),
      ]}
      vram={null}
      vramTotal={null}
      parallelSlots={{ used: 0, max: 4 }}
      panes={{}}
      requestIdentity={{
        ip: '10.0.0.9',
        model_tag: 'model-c-not-resident',
        session_id: 's',
        is_main: true,
        is_sub_agent: false,
        is_curator: false,
        is_compression: false,
        resolved_class: null,
        thread_id: 't',
      }}
      tokRateTone={stubTokRateTone}
    />,
  );
  assert.doesNotMatch(html, /no request yet/, 'requestIdentity was supplied, so if it rendered anywhere it would not be the placeholder');
  assert.doesNotMatch(html, /model-c-not-resident/, 'neither card matches the identity\'s model_tag and neither is sole, so the strip must not render on either');
});

/* ------------------------------------------------------------------ */
/*  Per-resident identity -- THE NO-FLIP PROPERTY.                      */
/*  Two residents, each with its OWN request_identity,                 */
/*  rendered together in one panel -- neither card may show the other's */
/*  identity, a shared one, or a stale/leaked one.                      */
/* ------------------------------------------------------------------ */

test('ResidentsPanel NO-FLIP: two residents with DIFFERENT request_identity -- ' +
  'each card shows its OWN, never the sibling\'s, never swapped, never a shared/global one', () => {
  const html = renderToStaticMarkup(
    <ResidentsPanel
      residents={[
        makeResident({
          model_tag: 'model-alpha',
          request_identity: {
            ip: '10.0.0.1', model_tag: 'model-alpha', session_id: 'sess-alpha',
            is_main: true, is_sub_agent: false, is_curator: false, is_compression: false,
            resolved_class: null, thread_id: 't1', label: 'IDENTITY-ALPHA',
          },
        }),
        makeResident({
          model_tag: 'model-beta',
          request_identity: {
            ip: '10.0.0.2', model_tag: 'model-beta', session_id: 'sess-beta',
            is_main: true, is_sub_agent: false, is_curator: false, is_compression: false,
            resolved_class: null, thread_id: 't2', label: 'IDENTITY-BETA',
          },
        }),
      ]}
      vram={null}
      vramTotal={null}
      parallelSlots={{ used: 0, max: 4 }}
      panes={{}}
      // Deliberately supply a THIRD, different global identity -- if either
      // card fell back to it or leaked into the other, this label would
      // appear where it shouldn't, or IDENTITY-ALPHA/BETA would be missing.
      requestIdentity={{
        ip: '10.0.0.9', model_tag: 'model-alpha', session_id: 's-global',
        is_main: true, is_sub_agent: false, is_curator: false, is_compression: false,
        resolved_class: null, thread_id: 't-global', label: 'STALE-GLOBAL-SHOULD-NOT-APPEAR',
      }}
      tokRateTone={stubTokRateTone}
    />,
  );
  assert.match(html, /IDENTITY-ALPHA/, 'model-alpha\'s card must show its own identity');
  assert.match(html, /IDENTITY-BETA/, 'model-beta\'s card must show its own identity');
  assert.doesNotMatch(html, /STALE-GLOBAL-SHOULD-NOT-APPEAR/, 'neither card may fall back to the global once its own per-resident field is present');
  // Order is alphabetical (model-alpha, model-beta) -- ALPHA's identity must
  // appear before BETA's in the DOM, confirming it landed on the right card,
  // not just "somewhere on the page".
  assert.ok(
    html.indexOf('IDENTITY-ALPHA') < html.indexOf('IDENTITY-BETA'),
    'alpha\'s identity must appear on alpha\'s (earlier, alphabetically-sorted) card, not swapped onto beta\'s',
  );
});

/* ------------------------------------------------------------------ */
/*  Follow-up: make the no-flip property MACHINE-READABLE                */
/*  on the rendered page, not just assertable in a unit test.           */
/*                                                                      */
/*  Why this exists: a live probe needs a stable data-testid hook to    */
/*  read each identity strip by; without one it cannot tell a correct   */
/*  page from a broken one. These pin the hook so the live              */
/*  probe and this suite assert the SAME property.                      */
/* ------------------------------------------------------------------ */

test('ResidentsPanel NO-FLIP, machine-readable: each card carries resident-request-identity whose ' +
  'data-value is ITS OWN request model_tag, and the two are DISTINCT', () => {
  const html = renderToStaticMarkup(
    <ResidentsPanel
      residents={[
        makeResident({
          model_tag: 'model-alpha',
          request_identity: {
            ip: '10.0.0.1', model_tag: 'model-alpha', session_id: 'sess-alpha',
            is_main: true, is_sub_agent: false, is_curator: false, is_compression: false,
            resolved_class: null, thread_id: 't1', label: 'IDENTITY-ALPHA',
          },
        }),
        makeResident({
          model_tag: 'model-beta',
          request_identity: {
            ip: '10.0.0.2', model_tag: 'model-beta', session_id: 'sess-beta',
            is_main: true, is_sub_agent: false, is_curator: false, is_compression: false,
            resolved_class: null, thread_id: 't2', label: 'IDENTITY-BETA',
          },
        }),
      ]}
      vram={null}
      vramTotal={null}
      parallelSlots={{ used: 0, max: 4 }}
      panes={{}}
      tokRateTone={stubTokRateTone}
    />,
  );
  const values = [...html.matchAll(/data-testid="resident-request-identity"[^>]*?data-value="([^"]*)"/g)].map(
    (m) => m[1],
  );
  assert.equal(values.length, 2, 'both cards must carry the hook -- one strip cannot evidence no-flip');
  assert.deepEqual(values, ['model-alpha', 'model-beta'],
    'each card\'s data-value must be its OWN request model_tag, in card order -- a flip would repeat one value');
  assert.equal(new Set(values).size, 2, 'a flip-flop collapses both cards onto ONE identity; distinctness is the property');
});

test('ResidentsPanel: a resident with NO request yet still renders the identity hook with data-value="" -- ' +
  'no-request must be distinguishable from the element having been dropped', () => {
  const html = renderToStaticMarkup(
    <ResidentsPanel
      residents={[makeResident({ model_tag: 'model-alpha', request_identity: null })]}
      vram={null}
      vramTotal={null}
      parallelSlots={{ used: 0, max: 4 }}
      panes={{}}
      tokRateTone={stubTokRateTone}
    />,
  );
  assert.equal(dataValue(html, 'resident-request-identity'), '',
    'the hook must be PRESENT and empty, not absent -- absent reads identically to a bug that removed the strip');
});
