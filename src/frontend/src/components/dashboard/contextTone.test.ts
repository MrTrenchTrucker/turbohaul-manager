// The context-fullness colour rule is ONE fact shared by the
// per-resident card and the overall aggregate box, so these tests pin the
// single shared classifier both components use. The components' own suites still assert that they
// APPLY the tone to the right number; that is a different question and lives
// with them.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  contextTone,
  CONTEXT_TONE_BAR_CLASSES,
  CONTEXT_TONE_TEXT_CLASSES,
} from './contextTone';
import type { ContextTone } from './contextTone';

test(
  'contextTone PURE: thresholds are green <75 -- amber 75-90 -- red >90, ' +
    'boundaries exact',
  () => {
    assert.equal(contextTone(0), 'green');
    assert.equal(contextTone(74.9), 'green');
    assert.equal(contextTone(75), 'amber', '75 itself is amber, not green -- boundary is inclusive on the amber side');
    assert.equal(contextTone(90), 'amber', '90 itself is still amber, not red -- red is strictly greater than 90');
    assert.equal(contextTone(90.1), 'red');
    assert.equal(contextTone(100), 'red');
  },
);

// A tone with no colour renders as `undefined` inside a className template --
// silently unstyled, not a crash, so nothing else in the suite would catch it.
test('every ContextTone has BOTH a bar colour and a text colour -- a new tone cannot be added half-coloured', () => {
  const tones: ContextTone[] = ['green', 'amber', 'red'];
  for (const tone of tones) {
    assert.ok(CONTEXT_TONE_BAR_CLASSES[tone], `${tone} has no bar class`);
    assert.ok(CONTEXT_TONE_TEXT_CLASSES[tone], `${tone} has no text class`);
  }
  assert.deepEqual(Object.keys(CONTEXT_TONE_BAR_CLASSES).sort(), tones.slice().sort());
  assert.deepEqual(Object.keys(CONTEXT_TONE_TEXT_CLASSES).sort(), tones.slice().sort());
});
