/**
 * The context-fullness colour rule. PURE -- no hooks,
 * no React, no component. Both the per-resident card and the overall aggregate
 * box apply it; this file only decides.
 *
 * WHY ITS OWN MODULE rather than an export from either component: the two
 * readouts share exactly ONE fact -- "how full is too full" -- and a copy in
 * each file drifts silently the first time somebody tunes a threshold. But
 * the design keeps the aggregate box (ThroughputSection.tsx) from depending
 * on the per-resident card (ResidentCard.tsx), so the shared fact cannot live
 * in either one without pointing the dependency in a direction that design rule forbids.
 * It lives here instead, the same shape as tokRate.ts and alarms.ts: a
 * dependency-free classifier that both sides import and NEITHER side owns.
 *
 * This is deliberately NOT the SPARK_SAMPLES/CARD_SPARK_SAMPLES situation. Those
 * two are separately-pinned on purpose because they are different facts that
 * merely look alike (60 backend ticks of combined history vs 30 of one card's).
 * These thresholds are one fact used twice: if the card calls 91% red, the
 * aggregate calling it amber is a bug, never a deliberate difference.
 *
 * Thresholds: green <75% -- amber 75-90% (75 itself is amber) -- red >90%
 * (90 itself is still amber). Pinned by
 * contextTone.test.ts. The components' own suites assert that they APPLY these
 * tones to the right number, which is a different question from what the tones
 * are, and both remain necessary.
 */
export type ContextTone = 'green' | 'amber' | 'red';

export function contextTone(pct: number): ContextTone {
  if (pct > 90) return 'red';
  if (pct >= 75) return 'amber';
  return 'green';
}

/** Bar FILL colour -- the card's per-resident bar and the aggregate's both use it. */
export const CONTEXT_TONE_BAR_CLASSES: Record<ContextTone, string> = {
  green: 'bg-emerald-500',
  amber: 'bg-amber-500',
  red: 'bg-red-500',
};

/**
 * Text colour for a toned percentage FIGURE. Only the aggregate renders one
 * today (the card tones its bar alone), but it belongs beside the bar map so
 * the two colour families stay a single decision rather than drifting into
 * "amber bar, green number" the next time either is tuned.
 */
export const CONTEXT_TONE_TEXT_CLASSES: Record<ContextTone, string> = {
  green: 'text-emerald-300',
  amber: 'text-amber-300',
  red: 'text-red-400',
};
