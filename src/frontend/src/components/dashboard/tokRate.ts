import type { GenerationInfo } from '../../api';

/**
 * The tok/s colour rule. PURE -- no hooks, no time, no
 * component. ResidentCard.tsx applies this; this file only decides.
 *
 * THREE tones, not two: a genuinely STALLED resident reports tok_s === 0.0
 * (live_monitor.py _derive's stall branch), so a binary tok_s>0-vs-not rule
 * would render it calm grey -- exactly the "active and not stalled" honesty
 * the dashboard requires. This restores the red-for-stalled distinction
 * the single-generation Hero (ThroughputSection.tsx, in its earlier form) already
 * drew before the aggregate rewrite dropped it for a reason (no single
 * coherent state across N residents) that does not apply at this
 * per-resident granularity.
 *
 * 'live' fires on 'generating' OR 'prefill': during a genuine prefill,
 * tok_s is always null (live_monitor.py never sets it in the prefill
 * branch), so a resident mid-prefill is genuinely working but would render
 * with the same grey look as a truly idle resident under a
 * state==='generating'-only reading.
 *
 * 'stalled' also folds in prefill_stall_alarm: a resident
 * hung mid-prefill is not healthily "prefilling" even though gen.state
 * stays the literal string 'prefill'. This is a coarser, different question
 * from residentAlarm's own 'prefill-hang' token (which answers "which kind
 * of alarm" for the badge) -- tokRateTone answers "should this number look
 * alarmed at all" for the tok/s figure specifically. Two questions, not one
 * fact computed twice, so this does not reopen the divergence
 * risk (residentAlarm is still the single author of the alarm VERDICT; this reads
 * the same underlying signal for a narrower purpose).
 *
 * Does NOT need elapsed time: every input (state, stalled,
 * prefill_stall_alarm) is a snapshot field the backend already computed
 * using its OWN elapsed-time logic (STALL_AFTER_S / PREFILL_STALL_AFTER_S
 * timers in live_monitor.py) and exposes as a ready-made boolean/string --
 * this function reads an already-derived verdict, it never re-derives a
 * time threshold itself.
 *
 * Who remembers the LAST tok/s for the grey state: NOT this function (pure,
 * stateless by construction) -- the ResidentCard component, via the existing
 * useTokSHistory ref-based buffer already in ResidentCard.tsx.
 */
export type TokRateTone = 'live' | 'idle' | 'stalled';

export function tokRateTone(gen: GenerationInfo | null | undefined): TokRateTone {
  if (!gen) return 'idle';
  if (gen.stalled || gen.state === 'stalled' || gen.prefill_stall_alarm) return 'stalled';
  if (gen.state === 'generating' || gen.state === 'prefill') return 'live';
  return 'idle';
}
