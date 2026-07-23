import { useEffect, useLayoutEffect, useRef, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { GenerationInfo, LoadVerifyRecord, RequestIdentity, ResidentModel, StatusSnapshot } from '../api';
import { useStatus } from '../hooks/useStatus';
import { useLiveStream } from '../hooks/useLiveStream';
import type { GenPane } from '../hooks/useLiveStream';

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/**
 * Synthesize a partial ResidentModel from legacy single-residency fields
 * (active / loading / grace / idle_hot / generation).
 *
 * Under cap<=1, residents[] is empty by design — the inference data lives
 * on the legacy fields. This bridges that gap so the dashboard panels
 * (model name, state, tok/s, tok/s graph, live output) still render.
 *
 * Priority order matches the clean FE LoadedBanner:
 *   active > loading > grace > idle_hot
 */
function synthesizeResident(data: StatusSnapshot): ResidentModel | null {
  const source =
    data.active ??
    data.loading ??
    (data.grace ? { model_tag: data.grace.model_tag, state: 'GRACE' as const } : null) ??
    (data.idle_hot ? { model_tag: data.idle_hot.model_tag, state: 'IDLE_HOT' as const } : null);

  if (!source) return null;

  return {
    model_tag: source.model_tag,
    state: source.state,
    port: (source as any).port ?? 0,
    pid: (source as any).pid ?? 0,
    spawn_seq: 0,
    reserved_need_mib: 0,
    parallel: 1,
    main_gpu: 0,
    split_mode: 'single',
    inflight: 0,
    // 'unload in Ns' is only truthful when the model is actually
    // parked — never while a serve is in flight (this badge + the stale advertised
    // grace clock was the source of a 'grace timer fired mid-prefill' illusion).
    idle_expires_in_s:
      source.state === 'GRACE' || source.state === 'IDLE_HOT'
        ? (data.grace?.remaining_s ?? data.idle_hot?.remaining_s ?? null)
        : null,
    generation: data.generation,
    // engine_op from active/loading info
    engine_op: (source as any).engine_op,
  };
}

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
/*  Small UI atoms                                                      */
/* ------------------------------------------------------------------ */

function Card({
  title,
  tone,
  children,
}: {
  title: string;
  tone: string;
  children: ReactNode;
}) {
  return (
    <div className={`rounded-lg border ${tone} bg-slate-950 p-3`}>
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
        {title}
      </div>
      {children}
    </div>
  );
}

function KV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-sm py-0.5">
      <span className="text-slate-400">{k}</span>
      <span className="font-mono text-slate-200 truncate">{v}</span>
    </div>
  );
}

/* ================================================================== */
/*  LIVE INFERENCE / THROUGHPUT section                                */
/*  Ported from the clean LiveInference.tsx, adapted to our LOOSER     */
/*  GenerationInfo types (every numeric field is `number | undefined`, */
/*  state is `string`, prompt_progress is `string | null`).           */
/* ================================================================== */

const SPARK_SAMPLES = 60;

// ── Activity phase (single source of truth, shared by Pill + Hero) ──────────
// Bursty workloads hard-cycle gen.state generating->finishing->idle->generating
// every ~20s. Without smoothing the StatePill and Hero flip on every poll tick.
// We collapse state into a 3-value PHASE with a grace HOLD on the DOWN edge only:
//   'live'   -> actively working RIGHT NOW (snap up instantly, no debounce)
//   'recent' -> quiet, but within RECENT_HOLD_MS of the last burst (shows the
//               LAST burst's REAL tok/s, captioned)
//   'idle'   -> genuinely quiet past the grace window (stark IDLE — honest)
type ActivityPhase = 'live' | 'recent' | 'idle';

// Observed inter-burst gaps run ~5-6s, so a 10s hold bridges adjacent bursts
// while still surfacing a genuine stop as IDLE within ~10s.
const RECENT_HOLD_MS = 10000;

// mid-prefill hang alarm threshold (must match the backend PREFILL_STALL_AFTER_S)
// Prefill can legitimately run 30-60s on large contexts; decoding stall clock
// (STALL_AFTER_S=10s) is too aggressive for prefill.
const PREFILL_STALL_AFTER_S = 60;

// NOTE: our GenerationInfo.state is a plain `string` (no GenerationState union),
// so this set is keyed on string.
const LIVE_STATES: ReadonlySet<string> = new Set<string>([
  'generating',
  'prefill',
  'finishing',
  'loading',
  'grace',
  'stalled',
]);

interface ActivityHold {
  phase: ActivityPhase;
  // Last non-null EWMA tok/s captured while live — surfaced (captioned) during
  // 'recent' ONLY when still attributable to the burst that just went quiet.
  heldTokS: number | null;
}

function useActivityPhase(gen: GenerationInfo | null): ActivityHold {
  const [phase, setPhase] = useState<ActivityPhase>('idle');
  const heldTokS = useRef<number | null>(null);
  const heldGenId = useRef<string | null>(null); // genId under which heldTokS was captured
  const liveGenId = useRef<string | null>(null); // genId of the most recent live burst
  const lastLiveAt = useRef<number>(0);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (timer.current !== undefined) {
      window.clearTimeout(timer.current);
      timer.current = undefined;
    }
    if (!gen) {
      setPhase('idle');
      heldTokS.current = null;
      heldGenId.current = null;
      liveGenId.current = null;
      return;
    }
    const isLive = gen.stalled || LIVE_STATES.has(gen.state);
    if (isLive) {
      // Snap UP instantly — responsiveness is never debounced.
      lastLiveAt.current = Date.now();
      if (gen.generation_id !== null) liveGenId.current = gen.generation_id;
      if (gen.tok_s != null) {
        heldTokS.current = gen.tok_s; // remember real burst speed + its owner
        heldGenId.current = gen.generation_id;
      }
      setPhase('live');
      return;
    }
    // Quiet tick: hold 'recent' until the grace window since the last live tick elapses.
    const elapsed = Date.now() - lastLiveAt.current;
    const remaining = RECENT_HOLD_MS - elapsed;
    if (lastLiveAt.current === 0 || remaining <= 0) {
      setPhase('idle');
      heldTokS.current = null;
      return;
    }
    setPhase('recent');
    // Re-arm the down-edge so we fall to stark IDLE even if polling pauses.
    timer.current = window.setTimeout(() => {
      setPhase('idle');
      heldTokS.current = null;
    }, remaining);
  }, [gen]);

  useEffect(
    () => () => {
      if (timer.current !== undefined) window.clearTimeout(timer.current);
    },
    [],
  );

  // Surface the held number only if it belongs to the burst that just went quiet.
  const attributable = heldGenId.current !== null && heldGenId.current === liveGenId.current;
  return { phase, heldTokS: attributable ? heldTokS.current : null };
}

// ── State pill ────────────────────────────────────────────────────────────
type PillTone = 'green' | 'blue' | 'slate' | 'red' | 'amber' | 'gray';

interface Pill {
  label: string;
  tone: PillTone;
  detail?: string;
}

function pillClasses(tone: PillTone): string {
  switch (tone) {
    case 'green':
      return 'bg-emerald-950/60 border-emerald-600 text-emerald-300';
    case 'blue':
      return 'bg-blue-950/60 border-blue-600 text-blue-300';
    case 'red':
      return 'bg-red-950/60 border-red-600 text-red-300';
    case 'amber':
      return 'bg-amber-950/60 border-amber-600 text-amber-300';
    case 'slate':
      return 'bg-slate-800/60 border-slate-500 text-slate-300';
    case 'gray':
    default:
      return 'bg-slate-900 border-slate-700 text-slate-400';
  }
}

const BUSY_ESCALATE_S = 120; // longest legit engine op (4.6GB KV restore) is ~30s

function derivePill(data: StatusSnapshot, gen: GenerationInfo, phase: ActivityPhase, busyForS: number | null): Pill {
  // PREFILL keys on the backend state ONLY. prefill_pct must
  // NOT gate the pill — on this fork n_prompt grows through decode while proc
  // freezes at DONE_PROMPT, so pct<100 through all of healthy decode.
  // Never paint stark IDLE while the manager reports an ACTIVE serve
  // (the /slots poll starves during engine save/restore) — but a starved poll
  // must ESCALATE: a dead engine behind an ACTIVE slot must not read as calm
  // amber forever.
  // No % detail — on this fork prompt.tokens fills as it is
  // processed, so (cache+proc)/n_prompt pegs at 100 for the whole prefill; a
  // constant "100%" is a lie. A true % needs the admission-context denominator
  // the manager knows at submit time (follow-up).
  if (gen.state === 'prefill') {
    return { label: 'PREFILL', tone: 'amber' };
  }
  // engine_op pill override — show the named engine operation
  // (kv_restore, kv_save, prefill, decode, idle, stream, unload) when active/loading.
  const activeOp = data.active?.engine_op;
  const loadingOp = data.loading?.engine_op;
  const engineOp = activeOp || loadingOp;
  // The op pill must NEVER outrank the alarms — a stalled engine
  // or a telemetry blackout paints RED even mid-op (a calm blue pill over them
  // would re-mask exactly that class).
  if (engineOp && engineOp !== 'idle' && !gen.stalled && gen.state !== 'stalled' && phase !== 'idle') {
    const opLabel = engineOp.toUpperCase().replace('_', ' ');
    return { label: opLabel, tone: 'blue' };
  }
  if (phase === 'idle' && data.active) {
    if (busyForS != null && busyForS >= BUSY_ESCALATE_S) {
      return { label: 'NO TELEMETRY', tone: 'red', detail: `engine unresponsive ${busyForS}s` };
    }
    return {
      label: 'BUSY',
      tone: 'amber',
      detail: busyForS != null ? `engine busy — telemetry paused ${busyForS}s` : 'engine busy — telemetry paused',
    };
  }
  // STALLED takes precedence among true alarm states.
  if (gen.stalled || gen.state === 'stalled') {
    return { label: 'STALLED', tone: 'red' };
  }
  // Grace HOLD: a quiet tick within the recent window keeps a soft 'RECENT'
  // badge instead of snapping to gray IDLE — same source of truth as the Hero.
  if (phase === 'recent') {
    return { label: 'RECENT', tone: 'slate', detail: 'between requests' };
  }
  switch (gen.state) {
    case 'generating':
      return { label: 'GENERATING', tone: 'green' };
    case 'prefill': {
      // prompt_progress is a STRING (or null) in our types — show it verbatim,
      // never numeric math on it.
      const p = gen.prompt_progress;
      return {
        label: 'PREFILL',
        tone: 'blue',
        detail: p != null && p !== '' ? p : undefined,
      };
    }
    case 'finishing':
      // Slate, deliberately NOT red — finishing is a healthy wind-down.
      return { label: 'FINISHING', tone: 'slate' };
    case 'loading': {
      const el = data.loading?.elapsed_s;
      return {
        label: 'LOADING',
        tone: 'amber',
        detail: el !== undefined ? `${el.toFixed(1)}s` : undefined,
      };
    }
    case 'grace': {
      const rem = data.grace?.remaining_s;
      return {
        label: 'GRACE',
        tone: 'slate',
        detail: rem !== undefined ? `${rem}s remaining` : undefined,
      };
    }
    case 'transitioning':
      return { label: 'TRANSITIONING', tone: 'gray' };
    case 'idle':
    default:
      return { label: 'IDLE', tone: 'gray' };
  }
}

function StatePill({ pill }: { pill: Pill }) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-semibold uppercase tracking-wide ${pillClasses(
        pill.tone,
      )}`}
    >
      {pill.label}
      {pill.detail && (
        <span className="font-mono text-xs font-normal normal-case opacity-80">
          {pill.detail}
        </span>
      )}
    </span>
  );
}

// ── Hero tok/s ────────────────────────────────────────────────────────────
// Honest '—' when null/undefined (first-decode pending) — never a fake low number.
function fmtTokS(v: number | undefined | null): string {
  if (v == null) return '—';
  return v.toFixed(1);
}

function Hero({
  gen,
  phase,
  heldTokS,
}: {
  gen: GenerationInfo;
  phase: ActivityPhase;
  heldTokS: number | null;
}) {
  const generating = gen.state === 'generating';
  const stalled = gen.stalled || gen.state === 'stalled';
  const prefill = gen.state === 'prefill';

  // RECENT (grace hold): just finished a burst. Show the LAST burst's REAL
  // throughput, dimmed + captioned as historical — not live, not fabricated.
  if (phase === 'recent' && !stalled && !prefill && !generating) {
    return (
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-6">
        <div className="text-xs uppercase tracking-wide text-slate-500 mb-1">Throughput</div>
        <div className="flex items-end gap-3">
          <span className="text-7xl font-bold tabular-nums leading-none text-slate-400">
            {fmtTokS(heldTokS)}
          </span>
          <span className="text-2xl font-medium text-slate-600 pb-1">tok/s</span>
        </div>
        <div className="text-xs text-slate-500 mt-2">last burst · between requests</div>
      </div>
    );
  }

  // IDLE-CLARITY: only after the grace window fully elapses do we paint the
  // stark IDLE panel — never a fake '0.0'.
  const showIdle = phase === 'idle' && !generating && !prefill && !stalled;
  if (showIdle) {
    return (
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-6">
        <div className="text-xs uppercase tracking-wide text-slate-500 mb-1">Throughput</div>
        <div className="flex items-end gap-3">
          <span className="text-7xl font-bold tabular-nums leading-none text-slate-500">
            IDLE
          </span>
        </div>
        <div className="text-xs text-slate-500 mt-2">waiting for request</div>
      </div>
    );
  }

  const pending = gen.tok_s == null;
  const numberTone = stalled
    ? 'text-red-300'
    : pending
    ? 'text-slate-500'
    : generating
    ? 'text-emerald-300'
    : 'text-slate-200';
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 p-6">
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-1">Throughput</div>
      <div className="flex items-end gap-3">
        <span className={`text-7xl font-bold tabular-nums leading-none ${numberTone}`}>
          {fmtTokS(gen.tok_s)}
        </span>
        <span className="text-2xl font-medium text-slate-500 pb-1">tok/s</span>
      </div>
      <div className="text-xs text-slate-500 mt-2">measured from llama-server /slots</div>
    </div>
  );
}

// ── Prefill indicator ─────────────────────────────────────────────────────
// prompt_progress is a STRING|null in our types — so we cannot draw a numeric
// fill bar. Surface it as an indeterminate "processing prompt" block plus the
// raw progress string if the backend supplies one.
function fmtInt(n: number): string {
  return n.toLocaleString('en-US');
}

function PrefillBar({ gen, sessionTotal }: { gen: GenerationInfo; sessionTotal: number }) {
  // A GROWING prefill bar. Numerator = cache-restored +
  // newly-processed tokens (both rise through prefill on this fork; n_prompt
  // itself fills as it is processed so it can't be the denominator mid-flight).
  // Denominator = the largest prompt seen this session (context only grows
  // within a session) — an estimate, so the fill clamps at 99% until the
  // engine flips to decode. No session history yet -> indeterminate pulse
  // with the live token count.
  const holdVal = useRef(0);
  const nowRaw = (gen.n_prompt_cache ?? 0) + (gen.n_prompt_proc ?? 0);
  if (nowRaw > 0) holdVal.current = nowRaw; // hold through /slots-starvation ticks
  const now = holdVal.current;
  // Use the backend's prefill_pct when available (correct denominator: the
  // current n_prompt from live_monitor.py). Fall back to sessionTotal — the
  // LAST COMPLETED turn's prompt total, learned by the always-mounted parent
  // (this component only exists during 'prefill' — n_prompt_tokens fills
  // DURING prefill on this fork, so learning it here pegged the % at ~99 while
  // the count grew, e.g. 24,576->42,648 both '~99%'). If sessionTotal is also 0
  // but we have live token counts (now > 0), still render the bar with the
  // growing token count and a meaningful fill — clamp at 99% since we lack the
  // true denominator. The w-1/3 pulse is ONLY for the true indeterminate state
  // (no token data at all). The 99% clamp keeps the bar from reading 100%
  // during prefill.
  const pct = gen.prefill_pct != null
      ? Math.min(99, Math.max(1, gen.prefill_pct))
      : sessionTotal > 0
        ? Math.min(99, Math.max(1, (now / sessionTotal) * 100))
        : now > 0 ? 99 : null;
  return (
    <div className="rounded-lg border border-blue-700 bg-blue-950/30 p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-xs uppercase tracking-wide text-blue-300 font-semibold">
          Processing prompt
        </div>
        <span className="font-mono text-sm text-blue-200 tabular-nums">
          {fmtInt(now)} tokens{pct != null ? ` · ~${Math.round(pct)}%` : ''}
        </span>
      </div>
      <div className="h-3 bg-slate-800 rounded overflow-hidden">
        {pct != null ? (
          <div className="h-full bg-blue-500 transition-all" style={{ width: `${pct}%` }} />
        ) : (
          <div className="h-full w-1/3 bg-blue-600/70 rounded animate-pulse" />
        )}
      </div>
      <div className="mt-2 text-xs text-blue-300/70">
        restored from KV + newly processed · % vs session size (est) · hands off to token progress at first decode
      </div>
    </div>
  );
}

// ── Progress bar ──────────────────────────────────────────────────────────
function Progress({ gen }: { gen: GenerationInfo }) {
  const nDecoded = gen.n_decoded ?? 0;
  const bounded = gen.max_tokens != null && gen.pct != null;
  // CONTEXT used / window: surface the request context against the model's
  // context-window capacity. Fall back to just the used count when n_ctx is unknown.
  const nPrompt = gen.n_prompt_tokens ?? 0;
  const contextLabel =
    gen.n_ctx != null
      ? `${fmtInt(nPrompt)} / ${fmtInt(gen.n_ctx)} window`
      : `${fmtInt(nPrompt)} tokens`;
  const riders = gen.riders ?? 0;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Progress</div>
      <div className="flex items-baseline justify-between text-sm mb-2">
        <span className="font-mono text-slate-200 tabular-nums">
          {bounded
            ? `${fmtInt(nDecoded)} / ${fmtInt(gen.max_tokens as number)} tokens`
            : `${fmtInt(nDecoded)} tokens`}
        </span>
        <span className="font-mono text-slate-400 tabular-nums">
          {bounded ? `${Math.round(gen.pct as number)}%` : 'unbounded'}
        </span>
      </div>
      <div className="h-2 bg-slate-800 rounded overflow-hidden">
        {bounded ? (
          <div
            className="h-full bg-emerald-500 transition-all"
            style={{ width: `${Math.min(100, Math.max(0, gen.pct as number))}%` }}
          />
        ) : gen.state === 'generating' || gen.state === 'finishing' || gen.state === 'stalled' ? (
          // Indeterminate barber-pole ONLY while a generation actually runs
          // unbounded. This block used to render at IDLE too — a
          // permanently-stuck ~33% bar on an idle dashboard.
          <div className="h-full w-1/3 bg-emerald-600/70 rounded animate-pulse" />
        ) : null}
      </div>
      <div className="mt-2 space-y-0.5 text-xs text-slate-500">
        <div className="font-mono">context: {contextLabel}</div>
        {gen.eta_s != null && (
          <div className="font-mono">eta: {gen.eta_s.toFixed(1)}s</div>
        )}
        {riders > 1 && (
          <div className="font-mono text-amber-400">riders: {riders} concurrent slots</div>
        )}
      </div>
    </div>
  );
}

// ── Big throughput sparkline (240x48) ─────────────────────────────────────
// Rolling tok_s_instant samples kept in a useRef array appended once per
// status.generation change.
function BigSparkline({ samples }: { samples: number[] }) {
  const W = 240;
  const H = 48;
  if (samples.length < 2) {
    return (
      <div className="flex h-12 items-center justify-center text-xs text-slate-600">
        — gathering samples —
      </div>
    );
  }
  const max = Math.max(...samples, 1);
  const step = W / (SPARK_SAMPLES - 1);
  const points = samples
    .map((v, i) => {
      const x = i * step;
      const y = H - (v / max) * H;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="h-12 w-full"
      role="img"
      aria-label="tokens per second history"
    >
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        className="text-emerald-400"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

// ── Throughput section — reads status.generation (the PRIMARY generation     */
//    block) directly, so it works in single-residency + series. Under         */
//    double-parallel it shows the primary gen + a "N concurrent" caption; the  */
//    per-generation live-output panes below render every concurrent gen.       */
function ThroughputSection({ data }: { data: StatusSnapshot }) {
  const gen = data.generation ?? null;

  // Rolling sparkline buffer of tok_s_instant, appended once per generation
  // snapshot (keyed off measured_at_iso to dedupe).
  const sparkRef = useRef<number[]>([]);
  const lastMeasured = useRef<string | null>(null);
  const lastGenId = useRef<string | null>(null);
  const [, forceTick] = useState(0);

  // Temporal phase (live/recent/idle) — shared by StatePill + Hero. Called
  // unconditionally (before any early return) to keep hook order stable.
  const { phase, heldTokS } = useActivityPhase(gen);

  // BUSY elapsed clock — hooks live ABOVE the early return
  // (rules-of-hooks; same invariant as useActivityPhase). Escalates a
  // permanently-starved /slots (dead engine behind an ACTIVE slot) to a red
  // NO TELEMETRY alarm instead of calm amber forever.
  const busySince = useRef<number | null>(null);
  // prefill-bar hold flag — hook lives above the early return too.
  const lastWasPrefill = useRef(false);
  const engineBusy = phase === 'idle' && !!data.active;
  useEffect(() => {
    if (engineBusy) {
      if (busySince.current == null) busySince.current = Date.now();
    } else {
      busySince.current = null;
    }
  }, [engineBusy]);
  const busyForS = engineBusy && busySince.current != null
    ? Math.max(0, Math.round((Date.now() - busySince.current) / 1000))
    : null;

  // PrefillBar mounts ONLY during 'prefill' (see the render
  // swap below), so it can never observe 'generating'/'finishing' and its own
  // ref would reset on every remount — learn the finished prompt total HERE,
  // in the always-mounted parent (hook above the early return, same invariant
  // as busySince), and pass it down as the bar's denominator.
  const promptTotalRef = useRef(0);
  if (gen && (gen.state === 'generating' || gen.state === 'finishing')) {
    promptTotalRef.current = gen.n_prompt_tokens ?? 0;
  }

  useEffect(() => {
    if (!gen) return;
    // A NEW generation resets the rolling
    // buffer. Otherwise peak/sparkline blend samples across generations and
    // "peak" can show a number that never occurred in the current generation.
    if (gen.generation_id !== lastGenId.current) {
      lastGenId.current = gen.generation_id;
      sparkRef.current = [];
      lastMeasured.current = null;
    }
    if (gen.measured_at_iso === lastMeasured.current) return;
    lastMeasured.current = gen.measured_at_iso;
    const next = [...sparkRef.current, gen.tok_s_instant ?? 0];
    sparkRef.current =
      next.length > SPARK_SAMPLES ? next.slice(next.length - SPARK_SAMPLES) : next;
    forceTick((t) => t + 1);
  }, [gen]);

  if (!gen) {
    return (
      <div className="space-y-3">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-semibold text-slate-300">LIVE INFERENCE</h2>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-6 text-sm italic text-slate-500">
          — no active generation —
        </div>
      </div>
    );
  }

  const pill = derivePill(data, gen, phase, busyForS);
  // PEAK tok/s — the max of the recent tok_s_instant samples in the buffer.
  // Bursty workloads dip to 0 between bursts; peak shows the true speed.
  const peak = sparkRef.current.length > 0 ? Math.max(...sparkRef.current) : 0;
  const instant = gen.tok_s_instant ?? 0;
  // Hold the prefill bar through /slots-starvation flaps (prefill <->
  // transitioning at ~batch cadence) so the bars don't alternate; any other
  // real state clears the hold.
  if (gen.state === 'prefill') lastWasPrefill.current = true;
  else if (gen.state !== 'transitioning') lastWasPrefill.current = false;
  const prefill =
    gen.state === 'prefill' || (gen.state === 'transitioning' && lastWasPrefill.current);

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <div className="flex items-baseline gap-2">
          <h2 className="text-sm font-semibold text-slate-300">LIVE INFERENCE</h2>
          {/* Honest about double-parallel — /status.generation is a
              single (primary) block, so the Hero/sparkline/peak reflect ONE gen.
              The per-generation live-output panes below show every concurrent gen. */}
          {(gen.riders ?? 0) > 1 && (
            <span className="text-xs font-mono text-amber-400">
              showing primary of {gen.riders} concurrent (see live output)
            </span>
          )}
        </div>
        <StatePill pill={pill} />
      </div>

      {/* Mid-prefill hang alarm — red banner when prefill heartbeat
          frozen for PREFILL_STALL_AFTER_S (60s). Observability-only, drives no
          FSM decision. */}
      {gen.prefill_stall_alarm && (
        <div className="mt-2 rounded border-2 border-red-700 bg-red-950/40 p-3">
          <div className="flex items-center gap-2 text-red-300">
            <span className="font-mono text-sm">⚠ MID-PREFILL HANG</span>
            <span className="text-xs">prefill heartbeat frozen ≥{PREFILL_STALL_AFTER_S}s — engine may be stuck</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <Hero gen={gen} phase={phase} heldTokS={heldTokS} />
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
            tok/s (last {SPARK_SAMPLES})
          </div>
          <BigSparkline samples={sparkRef.current} />
          <div className="mt-2 flex items-center justify-between text-xs font-mono text-slate-500 tabular-nums">
            <span>instant: {instant.toFixed(1)} tok/s</span>
            <span className="text-slate-400">peak: {peak.toFixed(1)} tok/s</span>
          </div>
        </div>
      </div>

      {/* The prefill bar REPLACES the token-progress bar during
          prompt processing, then hands off at first decode. */}
      {prefill ? <PrefillBar gen={gen} sessionTotal={promptTotalRef.current} /> : <Progress gen={gen} />}

      <div className="text-xs text-slate-500 flex items-center gap-3">
        <span>
          generation: <span className="font-mono">{gen.generation_id ?? '—'}</span>
        </span>
        <span>
          measured: <span className="font-mono">{gen.measured_at_iso}</span>
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Residents Panel                                                      */
/* ------------------------------------------------------------------ */

function ResidentCard({
  model,
  requestIdentity,
  soleResident = false,
  loadVerify,
}: {
  model: ResidentModel;
  requestIdentity?: RequestIdentity | null;
  soleResident?: boolean;
  loadVerify?: LoadVerifyRecord[] | null;
}) {
  const gen = model.generation;
  // engine_op from active/loading info (synthesized resident)
  // We get this from the data.active or data.loading engine_op via synthesis
  const engineOp = (model as any).engine_op;
  return (
    <Card title={model.model_tag} tone={stateTone(model.state)}>
      <div className="flex items-center gap-2 mb-2">
        <span className={`text-xs font-medium px-2 py-0.5 rounded ${stateBadge(model.state)}`}>
          {model.state}
        </span>
        {model.inflight > 0 && (
          <span className="text-xs font-medium px-2 py-0.5 rounded bg-violet-700 text-violet-100">
            {model.inflight} inflight
          </span>
        )}
        {/* Engine operation pill */}
        {engineOp && engineOp !== 'idle' && (
          <span className="text-xs font-medium px-2 py-0.5 rounded bg-slate-700 text-slate-100">
            {engineOp.toUpperCase()}
          </span>
        )}
        <span className="text-xs text-slate-500">
          GPU{model.main_gpu} · pid {model.pid} · port {model.port}
        </span>
      </div>
      {/* The structured-identity strip
          lives INSIDE the resident slot card (per-slot), not floating above the
          RESIDENTS box. Shown on the sole resident (single-residency deployment)
          or on the card whose model served the last request (multi-residency). */}
      {requestIdentity && (soleResident || requestIdentity.model_tag === model.model_tag) && (
        <RequestIdentityStrip identity={requestIdentity} />
      )}
      {/* Per-slot load/restore verify verdict (green/yellow/red). */}
      <LoadVerifyWidget records={loadVerify} modelTag={model.model_tag} />
      <KV k="reserved need" v={`${model.reserved_need_mib} MiB`} />
      <KV k="parallel" v={model.parallel} />
      <KV k="split_mode" v={model.split_mode} />
      {model.idle_expires_in_s != null && (
        <div className="mt-2 flex items-baseline gap-2 rounded border border-amber-800 bg-amber-950/40 px-2 py-1">
          <span className="text-xs uppercase tracking-wide text-amber-500">unload in</span>
          <span className="font-mono font-bold text-amber-300">{model.idle_expires_in_s}s</span>
        </div>
      )}
      {gen && (
        <div className="mt-2 pt-2 border-t border-slate-800">
          <div className="text-xs text-slate-500 mb-1">Current generation</div>
          <KV k="gen_id" v={gen.generation_id ? gen.generation_id.slice(0, 8) : '—'} />
          <KV k="state" v={gen.state} />
          {gen.tok_s != null && <KV k="tok/s" v={gen.tok_s.toFixed(1)} />}
          {gen.n_decoded != null && (
            <KV
              k="progress"
              v={`${gen.n_decoded.toLocaleString()} / ${gen.max_tokens?.toLocaleString() ?? '∞'}`}
            />
          )}
          {gen.eta_s != null && <KV k="ETA" v={`${gen.eta_s.toFixed(0)}s`} />}
        </div>
      )}
    </Card>
  );
}

function VramBars({ vram, vramTotal }: { vram: number[] | null; vramTotal: number[] | null }) {
  const TOTAL = vramTotal?.[0] ?? 24576;  // prefer backend-reported total; fallback to a 24 GiB GPU
  if (!vram || vram.length === 0) return null;
  return (
    <Card title="VRAM" tone="border-slate-700">
      {vram.map((freeMiB, i) => {
        const used = Math.max(0, TOTAL - freeMiB);
        return (
          <div key={i} className="mb-2 last:mb-0">
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-slate-400">GPU {i}</span>
              <span className="font-mono text-emerald-300">{used.toLocaleString()} / {TOTAL.toLocaleString()} MiB used</span>
            </div>
            <div className="h-2 bg-slate-800 rounded overflow-hidden">
              <div
                className="h-full bg-emerald-600 transition-all"
                style={{ width: `${Math.min(100, (used / TOTAL) * 100)}%` }}
              />
            </div>
          </div>
        );
      })}
    </Card>
  );
}

// VRAM honesty placeholder. Under single-residency (cap<=1) the backend
// suppresses /status.vram (null). The FE has NO real VRAM source, so we DO NOT
// fabricate numbers — we surface the gap so it's visibly accounted-for rather
// than silently missing.
// TODO: backend must populate status.vram even at cap<=1.
function VramPlaceholder() {
  return (
    <Card title="VRAM" tone="border-slate-700">
      <div className="text-sm text-slate-500">
        GPU VRAM telemetry unavailable under single-residency
        <span className="text-slate-600"> (requires backend support)</span>.
      </div>
    </Card>
  );
}

// Compact strip showing the last request's structured identity — proof
// Turbohaul reads + trusts the structured client_meta instead of guessing
// off the thread_id prefix.
// Null-safe: renders a muted placeholder when no request has landed yet.
function RequestIdentityStrip({ identity }: { identity: RequestIdentity | null | undefined }) {
  if (!identity) {
    return (
      <div className="text-xs text-slate-600 italic px-1">— no request yet —</div>
    );
  }
  // Derive the single role label from the is_* booleans (priority: curator >
  // compression > sub_agent > main; fall back to resolved_class).
  const role = identity.is_curator
    ? 'Curator'
    : identity.is_compression
      ? 'Compression'
      : identity.is_sub_agent
        ? 'Sub-Agent'
        : identity.is_main
          ? 'Main'
          : (identity.resolved_class ?? '—');
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs font-mono text-slate-400 px-1 py-1 border-b border-slate-800 mb-2">
      <span>{identity.ip ?? '—'}</span>
      <span className="text-slate-600">·</span>
      <span>{identity.model_tag ?? '—'}</span>
      <span className="text-slate-600">·</span>
      <span className="text-emerald-400">{role}</span>
      <span className="text-slate-600">·</span>
      <span className="truncate max-w-[10rem]">{identity.session_id ?? '—'}</span>
    </div>
  );
}

// Observability: green/yellow/red verdict per model (re)spawn +
// KV restore, with expected-vs-actual n_past. Answers "did the model + precomputed
// KV truly load?" at a glance — the blind spot that let a dead llama-server look
// idle-hot. final_status is the resolved verdict (ok / retried_ok / failed).
function loadVerifyTone(status: string): { dot: string; text: string; label: string } {
  if (status === 'ok') return { dot: 'bg-emerald-500', text: 'text-emerald-400', label: 'OK' };
  if (status === 'retried_ok') return { dot: 'bg-amber-500', text: 'text-amber-400', label: 'RETRIED' };
  if (status === 'failed') return { dot: 'bg-rose-500', text: 'text-rose-400', label: 'FAILED' };
  return { dot: 'bg-slate-500', text: 'text-slate-400', label: (status || '—').toUpperCase() };
}

function LoadVerifyRow({ rec }: { rec: LoadVerifyRecord }) {
  const tone = loadVerifyTone(rec.final_status);
  const isKv = rec.event === 'kv_restore';
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs font-mono py-0.5">
      <span className={`inline-block w-2 h-2 rounded-full ${tone.dot}`} />
      <span className="text-slate-400">{isKv ? 'KV restore' : 'model load'}</span>
      <span className="text-slate-600">·</span>
      <span className="text-slate-500">{rec.trigger}</span>
      <span className="text-slate-600">·</span>
      <span className={tone.text}>{tone.label}</span>
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

function ResidentsPanel({
  residents,
  vram,
  vramTotal,
  parallelSlots,
  requestIdentity,
  loadVerify,
}: {
  residents: ResidentModel[];
  vram: number[] | null;
  vramTotal: number[] | null;
  parallelSlots: { used: number; max: number };
  requestIdentity?: RequestIdentity | null;
  loadVerify?: LoadVerifyRecord[] | null;
}) {
  const hasVram = vram != null && vram.length > 0;
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-300">RESIDENTS</h2>
        <span className="text-xs text-slate-500 font-mono">
          slots {parallelSlots.used}/{parallelSlots.max}
        </span>
      </div>
      <div className="grid grid-cols-1 gap-3">
        {residents.map(m => (
          <ResidentCard
            key={m.model_tag}
            model={m}
            requestIdentity={requestIdentity}
            soleResident={residents.length === 1}
            loadVerify={loadVerify}
          />
        ))}
      </div>
      {/* Real bars when vram is a populated array (cap>=2); honest placeholder
          otherwise (cap<=1, backend suppression) — never fabricated numbers. */}
      {hasVram ? <VramBars vram={vram} vramTotal={vramTotal} /> : <VramPlaceholder />}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Tok/s sparkline (per-pane, hand-rolled inline SVG)                  */
/* ------------------------------------------------------------------ */

function Sparkline({
  samples,
  width = 120,
  height = 28,
}: {
  samples: number[];
  width?: number;
  height?: number;
}) {
  if (samples.length < 2) {
    return (
      <svg width={width} height={height} className="opacity-40">
        <line
          x1={0}
          y1={height - 1}
          x2={width}
          y2={height - 1}
          stroke="#475569"
          strokeWidth={1}
        />
      </svg>
    );
  }
  const max = Math.max(...samples);
  const min = Math.min(...samples);
  const span = max - min || 1;
  const stepX = width / (samples.length - 1);
  const pts = samples
    .map((v, i) => {
      const x = i * stepX;
      const y = height - 1 - ((v - min) / span) * (height - 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return (
    <svg width={width} height={height} aria-label="tok/s sparkline">
      <polyline
        points={pts}
        fill="none"
        stroke="#34d399"
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Live Output Panes (one per loaded model)                             */
/* ------------------------------------------------------------------ */

function LiveOutputPane({
  genId,
  modelTag,
  text,
  done,
  tokS,
  lastFrameAt,
  tokHistory,
}: {
  genId: string;
  modelTag: string;
  text: string;
  done: boolean;
  tokS: number | null;
  lastFrameAt: number;
  tokHistory: number[];
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  // Sticky-bottom: only auto-follow the
  // live output when the user is already at/near the bottom, so scrolling up to
  // free-read mid-generation is NOT yanked back down on every new token.
  const stickRef = useRef(true);
  const [atBottom, setAtBottom] = useState(true);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    stickRef.current = near;
    setAtBottom(near);
  };

  // Auto-scroll when text grows — ONLY if the user is sticking to the bottom.
  // useLayoutEffect (pre-paint) avoids a one-frame un-scrolled flicker while streaming.
  useLayoutEffect(() => {
    if (scrollRef.current && stickRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [text]);

  const jumpToBottom = () => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    stickRef.current = true;
    setAtBottom(true);
  };

  const shortId = genId.slice(0, 8);
  const ago = lastFrameAt ? `${Math.round((Date.now() - lastFrameAt) / 1000)}s ago` : '—';

  return (
    <Card
      title={`${modelTag} / ${shortId}`}
      tone={done ? 'border-slate-700' : 'border-emerald-700'}
    >
      <div className="flex items-center gap-2 mb-2">
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded ${
            done ? 'bg-slate-600 text-slate-300' : 'bg-emerald-700 text-emerald-100'
          }`}
        >
          {done ? 'DONE' : 'LIVE'}
        </span>
        {tokS != null && (
          <span className="text-xs font-mono text-emerald-400">{tokS.toFixed(1)} tok/s</span>
        )}
        <Sparkline samples={tokHistory} />
        <span className="text-xs text-slate-500 ml-auto">last: {ago}</span>
      </div>
      <div className="relative">
        <div
          ref={scrollRef}
          onScroll={onScroll}
          className="h-64 overflow-y-auto rounded bg-slate-900 p-3 text-sm font-mono text-slate-200 whitespace-pre-wrap break-words"
        >
          {text || <span className="text-slate-600 italic">Waiting for tokens…</span>}
        </div>
        {!atBottom && (
          <button
            type="button"
            onClick={jumpToBottom}
            aria-label="Jump to latest output"
            className="absolute bottom-2 right-2 text-xs px-2 py-0.5 rounded bg-emerald-700 text-emerald-100 opacity-90 hover:opacity-100"
          >
            ↓ latest
          </button>
        )}
      </div>
    </Card>
  );
}

function LiveOutputPanel({
  panes,
  residents,
}: {
  panes: Record<string, GenPane>;
  residents: ResidentModel[];
}) {
  // PRUNE stale model boxes: keep a pane only if its model_tag is a current
  // resident (true model still loaded), OR it is a generation_id-keyed pane
  // (model_tag null — the cap<=1 single-slot case) with no resident match.
  const residentTags = new Set(residents.map(r => r.model_tag));
  const entries = Object.values(panes).filter(
    p => p.model_tag == null || residentTags.has(p.model_tag),
  );

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-slate-300">
        LIVE OUTPUT ({entries.length} active)
      </h2>
      {entries.length === 0 ? (
        <Card title="No active generations" tone="border-slate-700">
          <div className="text-slate-500 text-sm italic">
            No concurrent inferences. Start a generation to see it here.
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {entries.map(p => (
            <LiveOutputPane
              key={p.paneKey}
              genId={p.generation_id}
              /* Under single-residency the SSE frame's model_tag is null (the
                 poller writes only the generation alias) → fall back to the
                 active resident's model name instead of a bare 'unknown'. */
              modelTag={p.model_tag ?? residents[0]?.model_tag ?? 'unknown'}
              text={p.text}
              done={p.done}
              tokS={p.lastTokS}
              lastFrameAt={p.lastFrameAt}
              tokHistory={p.tokHistory}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Queue + Parallel Slots mini-card (kept from old dashboard)            */
/* ------------------------------------------------------------------ */

function QueueCard({
  queue,
  parallelSlots,
}: {
  queue: { acceptance_buffer_depth: number; staging_queue_depth: number; staging_queue_max: number };
  parallelSlots: { used: number; max: number };
}) {
  const pct =
    queue.staging_queue_max > 0
      ? Math.min(100, (queue.staging_queue_depth / queue.staging_queue_max) * 100)
      : 0;
  return (
    <Card title="Queue" tone={queue.staging_queue_depth > 0 ? 'border-amber-700' : 'border-slate-700'}>
      <div className="text-lg font-semibold text-slate-200 mb-2">
        {queue.staging_queue_depth} / {queue.staging_queue_max}
      </div>
      <div className="h-2 bg-slate-800 rounded mb-3 overflow-hidden">
        <div className="h-full bg-amber-500 transition-all" style={{ width: `${pct}%` }} />
      </div>
      <KV k="acceptance buffer" v={queue.acceptance_buffer_depth} />
      <KV k="parallel slots" v={`${parallelSlots.used} / ${parallelSlots.max}`} />
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Dashboard (Split-View)                                          */
/* ------------------------------------------------------------------ */

export default function Dashboard() {
  const { data, error, lastUpdate } = useStatus();
  const { panes, connected: streamConnected } = useLiveStream();

  // When residents[] is empty (single-residency, cap<=1), synthesize a
  // partial ResidentModel from the legacy active/loading/grace/idle_hot
  // fields + the generation alias. This bridges the split-view FE
  // back to the data the operator wants to see.
  const effectiveResidents = useMemo(() => {
    if (!data) return [];
    if (data.residents.length > 0) return data.residents;
    const synthetic = synthesizeResident(data);
    return synthetic ? [synthetic] : [];
  }, [data]);

  if (!data) {
    return (
      <div className="text-slate-400">
        {error ? (
          <div className="text-amber-400">Error fetching /status: {error.message}</div>
        ) : (
          <div>Loading…</div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Top row: queue + parallel slots */}
      <QueueCard queue={data.queue} parallelSlots={data.parallel_slots} />

      {/* Full-width LIVE INFERENCE / throughput block — reads data.generation
          (the primary gen block) directly. Single-residency + series show the
          full picture; double-parallel shows the primary + a concurrency caption. */}
      <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-4">
        <ThroughputSection data={data} />
      </div>

      {/* Split-view: Residents (left) | Live Output (right) */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">
        <div className="xl:col-span-2">
          <ResidentsPanel
            residents={effectiveResidents}
            vram={data.vram}
            vramTotal={data.vram_total_mib}
            parallelSlots={data.parallel_slots}
            requestIdentity={data.request_identity}
            loadVerify={data.load_verify}
          />
        </div>
        <div className="xl:col-span-3">
          <LiveOutputPanel panes={panes} residents={effectiveResidents} />
        </div>
      </div>

      {/* Status footer */}
      <div className="text-xs text-slate-500 flex items-center gap-3">
        <span>
          last update:{' '}
          <span className="font-mono">{lastUpdate?.toISOString() ?? '—'}</span>
        </span>
        <span>
          sse:{' '}
          <span className={streamConnected ? 'text-emerald-400' : 'text-amber-400'}>
            {streamConnected ? 'connected' : 'disconnected'}
          </span>
        </span>
        {error && <span className="text-amber-400">⚠ {error.message} (retrying…)</span>}
      </div>
    </div>
  );
}
