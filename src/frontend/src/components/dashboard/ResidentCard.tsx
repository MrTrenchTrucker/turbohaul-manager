import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { GenerationInfo, LoadVerifyRecord, RequestIdentity, ResidentModel, SpecDowngradeRecord } from '../../api';
import type { GenPane } from '../../hooks/useLiveStream';
import { Card, KV } from './primitives';
import { RequestIdentityStrip } from './RequestIdentityStrip';
import { residentAlarm, PREFILL_STALL_AFTER_S, type AlarmToken } from './alarms';
import { CONTEXT_TONE_BAR_CLASSES, contextTone } from './contextTone';

// ⛔ DEAD-BRANCH CENSUS. Dead branches are marked and kept here,
// deliberately not deleted. Derived by AST over manager.py's
// ResidentState against slot.py's SlotState:
//
//   ✅ REACHABLE — real ResidentState members  5 : ACTIVE DEAD GRACE IDLE_EVICTABLE
//                                                  RESERVED_LOADING
//   ⛔ DEAD — SlotState vocabulary only        5 : GRACE_BUSY IDLE_HOT LOADING
//                                                  LOADING_FAIL POPPED
//   ⛔ DEAD — in NEITHER vocabulary            3 : IDLE_COLD PRE_LOADING READY
//
// 8 of the 13 literals below can NEVER match a resident: this switch is written
// against the SLOT vocabulary plus three invented names.
//
// ⚠ `GRACE` and `GRACE_BUSY` fail for DIFFERENT reasons and must not be conflated:
// `GRACE` IS a real ResidentState member that simply has zero assignments (fixable by
// supplying the value, which `phase` does). `GRACE_BUSY` is not a
// member at all; assigning `ResidentState.GRACE_BUSY` raises AttributeError.
function stateTone(state: string): string {
  if (state === 'ACTIVE') return 'border-emerald-700';
  if (state === 'GRACE' || state === 'GRACE_BUSY') return 'border-amber-700';
  if (state === 'LOADING' || state === 'PRE_LOADING' || state === 'RESERVED_LOADING') return 'border-blue-700';
  if (state === 'IDLE_HOT') return 'border-emerald-800';
  if (state === 'IDLE_EVICTABLE') return 'border-amber-700';
  if (state === 'DEAD') return 'border-slate-800';
  return 'border-slate-700';
}

function stateBadge(state: string): string {
  if (state === 'ACTIVE') return 'bg-emerald-700 text-emerald-100';
  if (state === 'GRACE' || state === 'GRACE_BUSY') return 'bg-amber-700 text-amber-100';
  if (state === 'LOADING' || state === 'PRE_LOADING') return 'bg-blue-700 text-blue-100';
  if (state === 'IDLE_HOT') return 'bg-emerald-800 text-emerald-200';
  if (state === 'IDLE_COLD' || state === 'POPPED') return 'bg-slate-600 text-slate-200';
  if (state === 'LOADING_FAIL') return 'bg-red-700 text-red-100';
  if (state === 'READY') return 'bg-teal-700 text-teal-100';
  if (state === 'RESERVED_LOADING') return 'bg-blue-700 text-blue-100';
  if (state === 'IDLE_EVICTABLE') return 'bg-amber-700 text-amber-100';
  if (state === 'DEAD') return 'bg-slate-700 text-slate-300';
  return 'bg-slate-600 text-slate-200';
}

/* ------------------------------------------------------------------ */
/*  Per-card alarm badge                                                */
/* ------------------------------------------------------------------ */

const ALARM_STYLES: Record<Exclude<AlarmToken, ''>, { classes: string; label: string }> = {
  stalled: { classes: 'bg-red-950/60 border-red-600 text-red-300', label: 'STALLED' },
  'no-telemetry': { classes: 'bg-red-950/60 border-red-600 text-red-300', label: 'NO TELEMETRY' },
  'prefill-hang': { classes: 'bg-amber-950/60 border-amber-600 text-amber-300', label: 'PREFILL HANG' },
  busy: { classes: 'bg-amber-950/60 border-amber-600 text-amber-300', label: 'BUSY' },
};

// Detail text is NOT part of the shared residentAlarm() contract (it is a
// bare token so the dashboard banner doesn't need to carry per-card
// copy) -- built locally here, human-facing only, from the same fields
// derivePill's own detail text used.
/**
 * The ONE countdown a resident card draws.
 *
 * Before this, the card's only time input was `idle_expires_in_s`, which the backend
 * set ONLY when `r.state is IDLE_EVICTABLE`. A resident serving out its grace window
 * still reads `state === 'ACTIVE'` (ResidentState.GRACE has zero assignments), so
 * during grace the payload carried no seconds at all and this block rendered nothing.
 *
 * `phase`/`remaining_s` come from the backend's single resolver. The
 * `idle_expires_in_s` arm is kept as the fallback so any backend that predates the
 * resolver renders exactly as it does today — additive, no regression.
 */
export function residentCountdown(
  model: ResidentModel,
): { label: string; seconds: number } | null {
  if (model.phase === 'GRACE' && model.remaining_s != null) {
    return { label: 'grace', seconds: model.remaining_s };
  }
  if (model.idle_expires_in_s != null) {
    return { label: 'unload in', seconds: model.idle_expires_in_s };
  }
  return null;
}

function alarmDetail(token: AlarmToken, busyForS: number | null): string | undefined {
  if (token === 'no-telemetry') return busyForS != null ? `engine unresponsive ${busyForS}s` : undefined;
  if (token === 'busy') {
    return busyForS != null ? `engine busy — telemetry paused ${busyForS}s` : 'engine busy — telemetry paused';
  }
  if (token === 'prefill-hang') {
    return `prefill heartbeat frozen ≥${PREFILL_STALL_AFTER_S}s — engine may be stuck`;
  }
  return undefined;
}

function AlarmBadge({ token, busyForS }: { token: AlarmToken; busyForS: number | null }) {
  // The wrapping span carries the testid/data-value UNCONDITIONALLY (the
  // rule: a missing element is ambiguous between "correctly clear" and "a
  // bug dropped it" -- same principle applied to resident-tok-s). The visible
  // pill only renders when there is something to show.
  return (
    <span data-testid="resident-alarm" data-value={token}>
      {token !== '' && (
        <span
          className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${ALARM_STYLES[token].classes}`}
          title={alarmDetail(token, busyForS)}
        >
          {ALARM_STYLES[token].label}
        </span>
      )}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  Compact per-card tok/s + graph + split progress bar                 */
/*  Built LOCAL to this file -- ThroughputSection's                     */
/*  Hero/BigSparkline/PrefillBar/Progress are private, differently      */
/*  sized for the aggregate hero panel, and not a fit to reuse; this is */
/*  a genuinely smaller, per-card version, not a duplicate.             */
/* ------------------------------------------------------------------ */

function fmtInt(n: number): string {
  return n.toLocaleString('en-US');
}

function fmtTokS(v: number | undefined | null): string {
  if (v == null) return '—';
  return v.toFixed(1);
}

/* ------------------------------------------------------------------ */
/*  tok/s colour + the grey last-known value                            */
/* ------------------------------------------------------------------ */

// dashboard/tokRate.ts owns the tone logic and its tests; this is a LOCAL type
// alias matching that contract exactly, not an import -- same injection
// pattern already established for ThroughputSection consuming alarms.ts
// via Dashboard.tsx (even though that module already existed): a
// cross-module dependency goes through Dashboard.tsx as a prop, not a
// direct cross-file import, so neither side's build depends on the other's
// file existing yet or its exact path.
export type TokRateTone = 'live' | 'idle' | 'stalled';

const TOK_RATE_TONE_CLASSES: Record<TokRateTone, string> = {
  live: 'text-emerald-300',
  idle: 'text-slate-500',
  stalled: 'text-red-400',
};

/**
 * PURE. What to show for the tok/s number and whether it's a remembered
 * value rather than the current live one. Split from useLastKnownTokS
 * below the same way residentAlarm is split from useBusyTimers: this half
 * is fully testable, the stateful half that supplies lastKnownTokS is not
 * (React effects do not run under this repo's SSR test harness).
 *
 * Deliberately NOT what feeds resident-tok-s's data-value: combinedTokS
 * (aggregate.ts) treats an absent tok_s as a 0 contribution whether the
 * cause is generation:null or mid-prefill (tok_s never populated), so
 * data-value must stay the raw current number always -- substituting the
 * remembered value there would break sum(resident-tok-s) == agg-tok-s,
 * the exact property the aggregate-consistency test asserts. This function
 * only decides what a HUMAN sees.
 */
export function resolveTokSDisplay(
  currentTokS: number | null | undefined,
  lastKnownTokS: number | null,
): { value: number | null; isLastKnown: boolean } {
  if (currentTokS != null) return { value: currentTokS, isLastKnown: false };
  if (lastKnownTokS != null) return { value: lastKnownTokS, isLastKnown: true };
  return { value: null, isLastKnown: false };
}

// Small sibling ref, deliberately NOT reusing useTokSHistory below: that
// hook tracks tok_s_instant for the sparkline, a different field from
// tok_s (what fmtTokS and the prominent number use). This reuses the
// existing memory pattern in this file rather than inventing a
// second one -- it follows the same ref+effect shape, applied to the
// right field.
function useLastKnownTokS(tokS: number | null | undefined): number | null {
  const last = useRef<number | null>(null);
  useEffect(() => {
    if (tokS != null) last.current = tokS;
  }, [tokS]);
  return last.current;
}

/* ------------------------------------------------------------------ */
/*  Context-size readout (used / capacity), PER RESIDENT                */
/* ------------------------------------------------------------------ */

// green <75% -- amber 75-90% -- red >90%. The rule
// itself lives in ./contextTone so the aggregate box applies the identical
// thresholds without a second, silently-driftable copy of the boundary
// numbers, and without either component depending on the other.

export interface ContextDisplay {
  used: number;
  capacity: number;
  pct: number;
  isLastKnown: boolean;
}

/**
 * PURE. Mirrors resolveTokSDisplay's split exactly (same reasoning: the
 * stateful half -- remembering the last real reading -- is not testable
 * under this repo's SSR harness, since React effects never run there; this
 * half is the fully-testable other half).
 *
 * currentNCtx is the gate, not currentUsed: measured data shows n_ctx
 * and n_prompt_tokens go null/0 TOGETHER on idle (they are the same
 * snapshot), so "do we have a live reading right now" is answered by n_ctx
 * alone, matching the wire's own null-together contract. lastKnown is null
 * only when this resident has NEVER had a real context reading (never
 * generated at all) -- that is the one case that legitimately falls through
 * to '-', same as resolveTokSDisplay's own "genuinely nothing to show" case.
 */
export function resolveContextDisplay(
  currentNCtx: number | null | undefined,
  currentUsed: number | null | undefined,
  lastKnown: { used: number; capacity: number } | null,
): ContextDisplay | null {
  if (currentNCtx != null) {
    const used = currentUsed ?? 0;
    return { used, capacity: currentNCtx, pct: currentNCtx > 0 ? (used / currentNCtx) * 100 : 0, isLastKnown: false };
  }
  if (lastKnown != null) {
    const { used, capacity } = lastKnown;
    return { used, capacity, pct: capacity > 0 ? (used / capacity) * 100 : 0, isLastKnown: true };
  }
  return null;
}

// Same ref+effect shape as useLastKnownTokS directly above -- reused,
// not a second mechanism -- applied to the (used, capacity) pair
// instead of a single number.
function useLastKnownContext(
  nCtx: number | null | undefined,
  used: number | null | undefined,
): { used: number; capacity: number } | null {
  const last = useRef<{ used: number; capacity: number } | null>(null);
  useEffect(() => {
    if (nCtx != null) last.current = { used: used ?? 0, capacity: nCtx };
  }, [nCtx, used]);
  return last.current;
}

// Fullness bar. No longer flex-1/min-w-0 --
// those classes existed to stretch it beside the shrink-0 numbers block in a
// shared flex row (same role MiniSparkline plays); the bar now sits on
// its own full-width line below the numbers
// instead, a plain block div, so those two flex-child classes would now be
// inert (no flex parent to size against) -- dropped rather than left as
// dead, misleading styling.
function ContextFullnessBar({ pct }: { pct: number | null }) {
  const width = pct != null ? Math.min(100, Math.max(0, pct)) : 0;
  const tone = pct != null ? contextTone(pct) : 'green';
  return (
    <div className="h-1.5 w-full overflow-hidden rounded bg-slate-800">
      <div
        data-testid="resident-context-bar"
        className={`h-full ${CONTEXT_TONE_BAR_CLASSES[tone]} transition-all`}
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

export const CARD_SPARK_SAMPLES = 30; // smaller than ThroughputSection's 60 -- sized for a compact card, not the hero panel

// Rolling tok_s_instant buffer, one per card. A proper hook call inside a
// genuine child component (ResidentCard) instantiated via .map() in the
// parent -- not a hook called inside the .map() callback itself, so this
// does not vary hook count within any single component instance.
function useTokSHistory(gen: GenerationInfo | null): number[] {
  const history = useRef<number[]>([]);
  const lastMeasured = useRef<string | null>(null);
  const [, forceTick] = useState(0);
  useEffect(() => {
    if (!gen) {
      history.current = [];
      lastMeasured.current = null;
      return;
    }
    if (gen.measured_at_iso === lastMeasured.current) return;
    lastMeasured.current = gen.measured_at_iso;
    const next = [...history.current, gen.tok_s_instant ?? 0];
    history.current = next.length > CARD_SPARK_SAMPLES ? next.slice(next.length - CARD_SPARK_SAMPLES) : next;
    forceTick(t => t + 1);
  }, [gen]);
  return history.current;
}

function MiniSparkline({ samples }: { samples: number[] }) {
  const W = 96;
  const H = 24;
  // Stretch across the row's available width instead of sitting
  // fixed-width crammed to the right.
  // flex-1 + min-w-0 (not w-24) on BOTH the placeholder and the real SVG, so
  // the row doesn't jump width once real samples arrive. preserveAspectRatio
  // ="none" already stretches the viewBox to whatever CSS width lands here --
  // the same mechanism ThroughputSection's BigSparkline already uses via
  // w-full -- so W/H/step stay untouched; this is a className-only change.
  if (samples.length < 2) {
    return <div className="h-6 flex-1 min-w-0" aria-hidden="true" />;
  }
  const max = Math.max(...samples, 1);
  const step = W / (CARD_SPARK_SAMPLES - 1);
  const points = samples
    .map((v, i) => `${(i * step).toFixed(1)},${(H - (v / max) * H).toFixed(1)}`)
    .join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-6 flex-1 min-w-0" role="img" aria-label="tok/s history">
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth={1.5} className="text-emerald-400" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// Stateless by design (unlike ThroughputSection's PrefillBar, which holds
// its numerator through /slots-starvation ticks via a ref): a compact card
// is a smaller stakes display than the hero panel, and a stateless read is
// fully exercisable by a static-render test, which a starvation-hold ref
// would not be. Documented tradeoff.
function CompactProgress({ gen }: { gen: GenerationInfo }) {
  if (gen.state === 'prefill') {
    const now = (gen.n_prompt_cache ?? 0) + (gen.n_prompt_proc ?? 0);
    const pct = gen.prefill_pct != null ? Math.min(99, Math.max(1, gen.prefill_pct)) : now > 0 ? 99 : null;
    return (
      <div>
        <div className="h-1.5 bg-slate-800 rounded overflow-hidden">
          {pct != null ? (
            <div className="h-full bg-blue-500" style={{ width: `${pct}%` }} />
          ) : (
            <div className="h-full w-1/3 bg-blue-600/70 rounded animate-pulse" />
          )}
        </div>
        <div className="mt-0.5 text-[10px] text-blue-300/70 font-mono">
          prefill · {fmtInt(now)} tokens{pct != null ? ` · ~${Math.round(pct)}%` : ''}
        </div>
      </div>
    );
  }
  const bounded = gen.max_tokens != null && gen.pct != null;
  const nDecoded = gen.n_decoded ?? 0;
  return (
    <div>
      <div className="h-1.5 bg-slate-800 rounded overflow-hidden">
        {bounded ? (
          <div className="h-full bg-emerald-500" style={{ width: `${Math.min(100, Math.max(0, gen.pct as number))}%` }} />
        ) : gen.state === 'generating' || gen.state === 'finishing' || gen.state === 'stalled' ? (
          <div className="h-full w-1/3 bg-emerald-600/70 rounded animate-pulse" />
        ) : null}
      </div>
      <div className="mt-0.5 text-[10px] text-slate-500 font-mono">
        {bounded ? `${fmtInt(nDecoded)} / ${fmtInt(gen.max_tokens as number)}` : `${fmtInt(nDecoded)} tokens`}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Half-height live-output box                                         */
/*  attached to the card; it is the ONLY live-output surface for this   */
/*  resident (Dashboard.tsx has no separate LiveOutputPanel column), so */
/*  the same output is never shown in two places.                       */
/*                                                                       */
/*  Gated on `pane` alone, NOT `gen && pane`: the box stays visible     */
/*  and greys out while the resident is idle, never                     */
/*  disappearing, so it must stay visible even when                     */
/*  model.generation is null -- idle is a real, common state for a      */
/*  still-resident model, not the same event as the resident unloading  */
/*  (which drops its pane entirely, see useLiveStream.ts). Gating on    */
/*  `gen` would hide the box on every idle tick, which is exactly the   */
/*  disappearing-mid-glance behaviour, which is worse than grey.        */
/* ------------------------------------------------------------------ */

// Expand/collapse this box between
// its default compact height and a taller one, same component, no
// duplicated logic, no modal/portal. "Close enough to the bottom to keep
// auto-following" extracted PURE from the inline onScroll check so the
// expand/collapse risk ("the scroll maths silently stops
// working at the new size") is directly testable at two different
// clientHeight values, not just asserted. The stateful half (the real
// useLayoutEffect-driven auto-scroll) is not reachable by this repo's
// renderToStaticMarkup harness -- same limitation useBusyTimers/
// useLiveStream's connection effect/useLastKnownTokS already have.
export function isNearBottom(
  scrollHeight: number,
  scrollTop: number,
  clientHeight: number,
  thresholdPx = 40,
): boolean {
  return scrollHeight - scrollTop - clientHeight < thresholdPx;
}

export function LiveOutputBox({ pane, defaultExpanded = false }: { pane: GenPane; defaultExpanded?: boolean }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  // Sticky-bottom, same pattern as the box this replaces: only auto-follow
  // when the user is already at/near the bottom.
  const stickRef = useRef(true);
  const [atBottom, setAtBottom] = useState(true);
  // Per-resident and default-collapsed both fall out of this for free:
  // LiveOutputBox is instantiated once per card, so this state can never
  // leak between residents, and useState(false) is the collapsed default.
  // defaultExpanded is a testability seam only -- ResidentCard never passes
  // it, nothing today needs a non-default initial value (expand state is
  // deliberately not persisted across a refresh).
  const [expanded, setExpanded] = useState(defaultExpanded);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const near = isNearBottom(el.scrollHeight, el.scrollTop, el.clientHeight);
    stickRef.current = near;
    setAtBottom(near);
  };

  // `expanded` is in the dependency list too: if the user was stuck to the
  // bottom when they toggle, the box should still be showing the bottom
  // immediately after, at whichever height it just became -- the same
  // re-snap new text already triggers, not left wherever it happened to
  // land at the old height.
  useLayoutEffect(() => {
    if (scrollRef.current && stickRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [pane.text, expanded]);

  const jumpToBottom = () => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    stickRef.current = true;
    setAtBottom(true);
  };

  const toggleExpanded = () => setExpanded(e => !e);

  const ago = pane.lastFrameAt ? `${Math.round((Date.now() - pane.lastFrameAt) / 1000)}s ago` : '—';

  return (
    <div className="mt-2 pt-2 border-t border-slate-800">
      <div className="flex items-center gap-2 mb-1">
        <span
          data-testid="resident-live-output-status"
          data-value={pane.done ? 'done' : 'live'}
          className={`text-[10px] font-medium px-1.5 py-0.5 rounded uppercase tracking-wide ${
            pane.done ? 'bg-slate-700 text-slate-300' : 'bg-emerald-700 text-emerald-100'
          }`}
        >
          {pane.done ? 'DONE' : 'LIVE'}
        </span>
        {!pane.done && pane.lastTokS != null && (
          <span className="text-[10px] font-mono text-emerald-400">{pane.lastTokS.toFixed(1)} tok/s</span>
        )}
        <span className="text-[10px] text-slate-500 ml-auto">last: {ago}</span>
        <button
          type="button"
          onClick={toggleExpanded}
          aria-label={expanded ? 'Retract live output' : 'Expand live output'}
          data-testid="resident-live-output-toggle"
          data-value={expanded ? 'expanded' : 'collapsed'}
          className="text-[10px] font-medium px-2 py-0.5 rounded uppercase tracking-wide bg-emerald-700 text-emerald-100 hover:bg-emerald-600"
        >
          {expanded ? 'Retract' : 'Expand'}
        </button>
      </div>
      <div className="relative">
        <div
          ref={scrollRef}
          onScroll={onScroll}
          className={`${expanded ? 'h-64' : 'h-32'} overflow-y-auto rounded p-2 text-xs font-mono whitespace-pre-wrap break-words ${
            pane.done ? 'bg-slate-900/60 text-slate-500' : 'bg-slate-900 text-slate-200'
          }`}
        >
          {pane.text || <span className="text-slate-600 italic">Waiting for tokens…</span>}
        </div>
        {!atBottom && (
          <button
            type="button"
            onClick={jumpToBottom}
            aria-label="Jump to latest output"
            className="absolute bottom-1 right-1 text-[10px] px-1.5 py-0.5 rounded bg-emerald-700 text-emerald-100 opacity-90 hover:opacity-100"
          >
            ↓ latest
          </button>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Per-resident identity, backward-compat fallback                     */
/* ------------------------------------------------------------------ */

// Three-way on whether `request_identity` exists on the model object at
// all (`in` would also work; `!== undefined` is equivalent here and reads
// clearer against the null case right below it):
//   undefined -> the field hasn't shipped on this backend yet. Fall back to
//                EXACTLY today's logic (global prop + soleResident/tag gate)
//                -- this is the backward-compat path, and it has to be
//                byte-identical to current behaviour, not just similar.
//   null      -> shipped, this resident just hasn't served a request yet.
//                Shown UNCONDITIONALLY -- RequestIdentityStrip already
//                renders "no request yet" for a null identity, no new
//                placeholder logic needed, just stop gating it.
//   present   -> shipped, this IS that resident's own last request. Shown
//                unconditionally too -- no more soleResident/tag gating,
//                the field is per-resident by construction so there is
//                nothing left to disambiguate.
// Once the per-resident field lands, the strip becomes unconditional per
// card (present-or-null, but always there) -- the same shape resident-tok-s
// and resident-alarm already have. Cards that today show nothing (N>=2, tag
// doesn't match, not sole) will show "no request yet" instead. Intentional,
// not a side effect.
function resolveCardIdentity(
  model: ResidentModel,
  globalIdentity: RequestIdentity | null | undefined,
  soleResident: boolean,
): { show: boolean; identity: RequestIdentity | null | undefined } {
  if (model.request_identity !== undefined) {
    return { show: true, identity: model.request_identity };
  }
  const show = !!globalIdentity && (soleResident || globalIdentity.model_tag === model.model_tag);
  return { show, identity: globalIdentity };
}

/* ------------------------------------------------------------------ */
/*  Resident Card                                                        */
/* ------------------------------------------------------------------ */

export function ResidentCard({
  model,
  requestIdentity,
  soleResident = false,
  loadVerify,
  specDowngrade,
  pane,
  busyForS,
  tokRateTone,
}: {
  model: ResidentModel;
  requestIdentity?: RequestIdentity | null;
  soleResident?: boolean;
  loadVerify?: LoadVerifyRecord[] | null;
  specDowngrade?: SpecDowngradeRecord[] | null;
  // This resident's own SSE pane, resolved once in ResidentsPanel
  // (mirroring how LiveOutputPanel already resolves panes -> entries).
  pane?: GenPane | null;
  // This resident's own busy-escalation
  // elapsed seconds, resolved once for the whole set by ResidentsPanel's
  // single useBusyTimers() call -- never computed inside this component.
  busyForS: number | null;
  // From dashboard/tokRate.ts, injected -- see
  // the comment above TokRateTone for why injection, not import.
  tokRateTone: (gen: GenerationInfo | null | undefined) => TokRateTone;
}) {
  const gen = model.generation;
  const engineOp = (model as any).engine_op;
  const alarm = residentAlarm(model, busyForS);
  const tokHistory = useTokSHistory(gen);
  const lastKnownTokS = useLastKnownTokS(gen?.tok_s);
  const tokSDisplay = resolveTokSDisplay(gen?.tok_s, lastKnownTokS);
  const tokTone = tokRateTone(gen);
  const lastKnownContext = useLastKnownContext(gen?.n_ctx, gen?.n_prompt_tokens);
  const contextDisplay = resolveContextDisplay(gen?.n_ctx, gen?.n_prompt_tokens, lastKnownContext);
  const cardIdentity = resolveCardIdentity(model, requestIdentity, soleResident);
  const displayState = model.phase ?? model.state;
  return (
    <Card
      title={model.model_tag}
      tone={stateTone(displayState)}
      rootProps={{ 'data-testid': 'resident-card' }}
      titleProps={{ 'data-testid': 'resident-tag' }}
    >
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        {/* `phase` is what r.state WOULD say if ResidentState.GRACE
            were assignable — it uses the ResidentState value names verbatim, so this
            branch set does not change and no fourth vocabulary is introduced. Falls
            back to the raw state on any backend that predates the resolver. */}
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded ${stateBadge(displayState)}`}
          data-testid="resident-state-badge"
        >
          {displayState}
        </span>
        {model.inflight > 0 && (
          <span className="text-xs font-medium px-2 py-0.5 rounded bg-violet-700 text-violet-100">
            {model.inflight} inflight
          </span>
        )}
        {engineOp && engineOp !== 'idle' && (
          <span className="text-xs font-medium px-2 py-0.5 rounded bg-slate-700 text-slate-100">
            {engineOp.toUpperCase()}
          </span>
        )}
        <AlarmBadge token={alarm} busyForS={busyForS} />
        <span className="text-xs text-slate-500">
          GPU{model.main_gpu} · pid {model.pid && model.pid > 0 ? model.pid : '—'} · port {model.port && model.port > 0 ? model.port : '—'}
        </span>
      </div>

      {/* CURRENT renders ABOVE
          last, so the live row is not buried under a finished session.
          Always per-resident (no soleResident/tag-gating the way the legacy
          global `request_identity` fallback needs below), unconditional,
          null when this resident is idle right now. */}
      <RequestIdentityStrip
        identity={model.current_request_identity}
        residentModelTag={model.model_tag}
        heading="current request"
        testId="resident-current-request-identity"
      />
      {cardIdentity.show && (
        <RequestIdentityStrip identity={cardIdentity.identity} residentModelTag={model.model_tag} />
      )}
      <LoadVerifyWidget records={loadVerify} modelTag={model.model_tag} />
      <SpecDowngradeWidget records={specDowngrade} modelTag={model.model_tag} />
      <KV k="reserved need" v={`${model.reserved_need_mib} MiB`} />
      <KV k="parallel" v={model.parallel} />
      <KV k="split_mode" v={model.split_mode} />
      {/* The grace countdown, previously missing from the card.
          Drawn from the backend's single phase resolver, falling back to the legacy
          idle field so a pre-resolver backend renders unchanged. */}
      {(() => {
        const countdown = residentCountdown(model);
        if (!countdown) return null;
        return (
          <div
            className="mt-2 flex items-baseline gap-2 rounded border border-amber-800 bg-amber-950/40 px-2 py-1"
            data-testid="resident-countdown"
            data-label={countdown.label}
            data-value={countdown.seconds}
          >
            <span className="text-xs uppercase tracking-wide text-amber-500">{countdown.label}</span>
            <span className="font-mono font-bold text-amber-300">{countdown.seconds}s</span>
          </div>
        );
      })()}

      {/* This resident's OWN tok/s -- ALWAYS rendered (the
          rule: a missing element is ambiguous between "correctly
          excluded" and "a bug dropped it"), data-value=0 when there is no
          live generation to report a rate for. */}
      {/* MiniSparkline is flex-1 (stretches across the freed
          width) so justify-between is dropped -- there's no leftover gap
          left to distribute once the sparkline grows to fill it; the label
          +number block gets shrink-0 so it keeps its natural width. */}
      <div className="mt-2 pt-2 border-t border-slate-800 flex items-center gap-3">
        <div className="shrink-0">
          <div className="text-[10px] uppercase tracking-wide text-slate-500">tok/s</div>
          <span
            data-testid="resident-tok-s"
            data-value={gen?.tok_s ?? 0}
            className={`text-xl font-bold tabular-nums ${TOK_RATE_TONE_CLASSES[tokTone]}`}
          >
            {tokSDisplay.value != null ? fmtTokS(tokSDisplay.value) : '—'}
          </span>
          {tokSDisplay.isLastKnown && (
            <span className="text-[10px] text-slate-600 ml-1" title="last known rate, not current">
              last
            </span>
          )}
        </div>
        <MiniSparkline samples={tokHistory} />
      </div>

      {gen && (
        <div className="mt-2">
          <CompactProgress gen={gen} />
        </div>
      )}

      {/* Context readout.
          Placed BELOW the per-turn bar above, as one
          unit -- numbers block first, then the fullness bar gets its OWN
          full-width line underneath instead of squeezing into the numbers'
          row, so it reads at the same weight as CompactProgress's bar
          above it. The numbers keep their own bar directly below them; only
          the geometry (stacked, not side-by-side) changed, not which bar
          belongs to which numbers.
          Design rule: idle shows the LAST-KNOWN reading, marked
          with the same "last" tag tok/s uses two rows up, and the number is
          deliberately NOT dimmed/tone-shifted the way tok/s's own color is --
          a stale-but-true context reading is current occupancy, not a faded
          fact, so only the small marker (not the number's color) says
          "stale". resident-context-used/-capacity stay ALWAYS present with
          data-value=0 when there has truly never been a reading (same
          "always render the testid" rule as resident-tok-s) -- never omitted,
          so a dropped row can't be mistaken for a correctly-idle one.

          Test-coverage limit: no test in
          this file can render the TRUE isLastKnown=true JSX branch below
          and inspect its className, because every test here renders via a
          single renderToStaticMarkup call and SSR never runs useEffect --
          lastKnownContext (from useLastKnownContext) is always null on
          first render, so resolveContextDisplay's lastKnown!=null branch is
          reached by the PURE-FUNCTION tests only, never through a full
          component render. A change that conditions this span's
          className on contextDisplay.isLastKnown (e.g. adding an opacity
          class when stale) would violate the "not dimmed" requirement yet
          stay fully green here. resident-tok-s's own "last" marker two rows
          up has the identical blind spot (its isLastKnown=true rendering has
          no direct test coverage in this file). Closing it would need real
          effect execution (e.g. a jsdom/testing-library harness).
          The regression guard (ResidentCard.test.tsx) is
          built on this exact boundary: the idle/last-known half is proven
          PURE (resolveContextDisplay -> contextTone), the pct->bar-colour
          half is proven by a live render, and the two compose because
          ContextFullnessBar's signature below is `{ pct }` alone -- it has
          no isLastKnown parameter to branch on without a visible
          signature change. */}
      <div className="mt-2 pt-2 border-t border-slate-800">
        <div>
          <div className="text-[10px] uppercase tracking-wide text-slate-500">context</div>
          <span
            data-testid="resident-context-used"
            data-value={contextDisplay?.used ?? 0}
            className="text-xl font-bold tabular-nums text-slate-200"
          >
            {contextDisplay != null ? fmtInt(contextDisplay.used) : '—'}
          </span>
          <span className="text-xl font-bold tabular-nums text-slate-500"> / </span>
          {/* Capacity is a reference ceiling, not the number
              that moves -- it would otherwise compete visually with
              `used` above. Dimmed to the exact " / " separator idiom
              (text-slate-500), size/weight/tabular-nums unchanged so digits
              stay aligned with `used`. */}
          <span
            data-testid="resident-context-capacity"
            data-value={contextDisplay?.capacity ?? 0}
            className="text-xl font-bold tabular-nums text-slate-500"
          >
            {contextDisplay != null ? fmtInt(contextDisplay.capacity) : '—'}
          </span>
          {contextDisplay != null && (
            <span className="ml-1 text-xs text-slate-500 tabular-nums">· {Math.round(contextDisplay.pct)}%</span>
          )}
          {contextDisplay?.isLastKnown && (
            <span className="text-[10px] text-slate-600 ml-1" title="last known context, not current">
              last
            </span>
          )}
        </div>
        <div className="mt-1.5">
          <ContextFullnessBar pct={contextDisplay != null ? contextDisplay.pct : null} />
        </div>
      </div>

      {pane && <LiveOutputBox pane={pane} />}

      {gen && (
        <div className="mt-2 pt-2 border-t border-slate-800">
          <div className="text-xs text-slate-500 mb-1">Current generation</div>
          <KV k="gen_id" v={gen.generation_id ? gen.generation_id.slice(0, 8) : '—'} />
          <KV k="state" v={gen.state} />
          {gen.eta_s != null && <KV k="ETA" v={`${gen.eta_s.toFixed(0)}s`} />}
        </div>
      )}
    </Card>
  );
}

// Load-verify observability: green/yellow/red verdict per model (re)spawn +
// KV restore, with expected-vs-actual n_past. Answers "did the model + precomputed
// KV truly load?" at a glance — the blind spot that let a dead llama-server look
// idle-hot. final_status is the backend's verdict (ok / retried_ok / failed) for
// MODEL LOAD rows. For those rows, 'unverified' is a REAL backend status (the
// verifier read /slots before the engine populated n_prompt_tokens) — it is NOT
// a failure, hence the distinct neutral tone below so it never reads as failed.
// KV-RESTORE rows do NOT use this function — see kvRestoreTone() below, which
// distinguishes "nothing to restore" (no attempt was made) from a genuine
// attempted-but-unmeasured or attempted-and-failed restore. Collapsing those
// onto this function's single 'unverified' case would label a resident that
// never had anything to restore as unverified.
function loadVerifyTone(status: string): { dot: string; text: string; label: string; state: string } {
  if (status === 'ok') return { dot: 'bg-emerald-500', text: 'text-emerald-400', label: 'OK', state: status };
  if (status === 'retried_ok')
    return { dot: 'bg-amber-500', text: 'text-amber-400', label: 'RETRIED', state: status };
  if (status === 'failed') return { dot: 'bg-rose-500', text: 'text-rose-400', label: 'FAILED', state: status };
  if (status === 'unverified')
    return { dot: 'bg-slate-400', text: 'text-slate-300', label: 'UNVERIFIED', state: status };
  return { dot: 'bg-slate-500', text: 'text-slate-400', label: (status || '—').toUpperCase(), state: status };
}

// KV restore label: a full 4-branch state machine, scoped ONLY
// to kv_restore-event rows — model-load rows keep calling loadVerifyTone()
// above, byte-identically.
//   restore_attempted === false                           -> no-data: nothing
//     was ever attempted, so there is nothing to verify. NOT the same claim as
//     "we tried and could not tell" — rendering it as UNVERIFIED would be wrong.
//   restore_attempted === true  && kv_restore_ok === true  -> ok (green — the
//     expected outcome). Will not fire on live data until the
//     engine-side fix (not touched here) starts populating
//     kv_restore_ok; this branch is proven with fixtures instead.
//   restore_attempted === true  && kv_restore_ok === false -> failed: a
//     restore was genuinely attempted and did not verify.
//   restore_attempted === true  && kv_restore_ok === null  -> pending: attempted
//     but not yet measured. Neither "nothing to restore" nor a failure —
//     collapsing this into either neighbor is its own version of the original
//     bug, so it stays visually and textually distinct from both.
//   restore_attempted is null/undefined (older records predating this field)
//     -> fall back to loadVerifyTone(final_status) rather than invent a claim
//     a record that cannot answer the question must not be given an answer.
export function kvRestoreTone(rec: LoadVerifyRecord): { dot: string; text: string; label: string; state: string } {
  if (rec.restore_attempted === false) {
    return { dot: 'bg-slate-400', text: 'text-slate-300', label: 'NOTHING TO RESTORE', state: 'no-data' };
  }
  if (rec.restore_attempted === true) {
    if (rec.kv_restore_ok === true) {
      return { dot: 'bg-emerald-500', text: 'text-emerald-400', label: 'OK', state: 'ok' };
    }
    if (rec.kv_restore_ok === false) {
      return { dot: 'bg-rose-500', text: 'text-rose-400', label: 'FAILED', state: 'failed' };
    }
    return { dot: 'bg-amber-500', text: 'text-amber-400', label: 'VERIFYING', state: 'pending' };
  }
  return loadVerifyTone(rec.final_status);
}

function LoadVerifyRow({ rec }: { rec: LoadVerifyRecord }) {
  const isKv = rec.event === 'kv_restore';
  const tone = isKv ? kvRestoreTone(rec) : loadVerifyTone(rec.final_status);
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs font-mono py-0.5">
      <span className={`inline-block w-2 h-2 rounded-full ${tone.dot}`} />
      <span className="text-slate-400">{isKv ? 'KV restore' : 'model load'}</span>
      <span className="text-slate-600">·</span>
      <span className="text-slate-500">{rec.trigger}</span>
      <span className="text-slate-600">·</span>
      <span
        {...(isKv ? { 'data-testid': 'load-verify-kv-status', 'data-value': tone.state } : {})}
        className={tone.text}
      >
        {tone.label}
      </span>
      {rec.retry_count > 0 && <span className="text-amber-500">×{rec.retry_count}</span>}
      {isKv && rec.kv_expected_tokens != null && (
        <>
          <span className="text-slate-600">·</span>
          <span className={rec.kv_restore_ok === false ? 'text-rose-400' : 'text-slate-400'}>
            n_past {rec.kv_actual_n_past?.toLocaleString() ?? '—'}/{rec.kv_expected_tokens.toLocaleString()}
          </span>
        </>
      )}
      {rec.process_alive === false && <span className="text-rose-400">· dead-pid</span>}
      {rec.model_resident === false && <span className="text-rose-400">· not-resident</span>}
      {rec.reason && (
        <span className="text-slate-500 truncate max-w-[12rem]" title={rec.reason}>· {rec.reason}</span>
      )}
    </div>
  );
}

function LoadVerifyWidget({
  records,
  modelTag,
}: {
  records?: LoadVerifyRecord[] | null;
  modelTag: string;
}) {
  const mine = (records ?? []).filter(r => r.model_tag === modelTag);
  // records arrive newest-last; find the most-recent of each event type.
  const lastLoad = [...mine].reverse().find(r => r.event === 'model_load');
  const lastRestore = [...mine].reverse().find(r => r.event === 'kv_restore');
  if (!lastLoad && !lastRestore) return null;
  return (
    <div className="mt-2 pt-2 border-t border-slate-800">
      <div className="text-xs text-slate-500 mb-1">Load / Restore verify</div>
      {lastLoad && <LoadVerifyRow rec={lastLoad} />}
      {lastRestore && <LoadVerifyRow rec={lastRestore} />}
    </div>
  );
}

// Surfaced so the user instantly sees it: a model whose
// requested speculative decoding ran WITHOUT the acceleration it asked
// for -- e.g. a D-Spark/D-Flash config on an architecture the engine's
// state-capture doesn't support for. This widget carries ONLY the
// informational case; there is no failure/error value in this record
// shape at all (see api.ts's SpecDowngradeRecord) -- a real load failure
// stays on the card's own state pill (LOADING_FAIL, rose/red), a
// completely separate field, so downgraded can never visually or
// structurally read as failed. Cyan is used deliberately because it is
// not part of the existing lifecycle palette (emerald=active,
// amber=grace/idle/unload-warning, blue=loading, rose=error,
// slate=dead/neutral) -- an unambiguous "informational" color a user has
// not already learned to associate with "something is wrong."
function SpecDowngradeRow({ rec }: { rec: SpecDowngradeRecord }) {
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs font-mono py-0.5">
      <span className="inline-block w-2 h-2 rounded-full bg-cyan-500" />
      <span className="text-cyan-400 font-semibold">SPECULATION DOWNGRADED</span>
      {rec.component && (
        <>
          <span className="text-slate-600">·</span>
          <span className="text-slate-400">{rec.component}</span>
        </>
      )}
      {rec.arch && (
        <>
          <span className="text-slate-600">·</span>
          <span className="text-slate-400">arch={rec.arch}</span>
        </>
      )}
      {rec.reason_code && (
        <>
          <span className="text-slate-600">·</span>
          <span className="text-slate-500">{rec.reason_code}</span>
        </>
      )}
      {rec.detail && (
        <span className="text-slate-500 truncate max-w-[16rem]" title={rec.detail}>
          · {rec.detail}
        </span>
      )}
    </div>
  );
}

export function SpecDowngradeWidget({
  records,
  modelTag,
}: {
  records?: SpecDowngradeRecord[] | null;
  modelTag: string;
}) {
  const mine = (records ?? []).filter(r => r.model_tag === modelTag);
  if (mine.length === 0) return null;
  // records arrive newest-last; show the most recent one only -- this is a
  // per-model current-state notice, not a history log (the full ring is
  // still on /status for anyone who wants it).
  const last = mine[mine.length - 1];
  return (
    <div className="mt-2 pt-2 border-t border-cyan-900/40">
      <div className="text-xs text-cyan-500 mb-1">Speculation status</div>
      <SpecDowngradeRow rec={last} />
    </div>
  );
}
