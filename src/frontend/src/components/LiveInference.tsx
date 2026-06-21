import { useEffect, useRef, useState } from 'react';
import type { GenerationInfo, GenerationState, StatusSnapshot } from '../api';
import { useLiveOutput } from '../hooks/useLiveOutput';

// Live inference panel. Folded UNDER the Dashboard tab (no longer its own tab).
// Metrics come from the EXISTING useStatus() 2s poll (status.generation) passed
// in by the parent; the only new wire is useLiveOutput() for the SSE text, which
// now ANCHOR-FOLLOWS the current generation (one persistent connection, no gid).
//
// Honesty contract: tok/s shows '—' (never a fake low number) until the EWMA is
// populated, and shows 'IDLE' when the model is loaded-but-not-generating —
// never a stark 0.0 as if it were crawling. FINISHING is slate, NOT red;
// STALLED is the only red state; PREFILL gets its own prominent progress block.

const SPARK_SAMPLES = 60;

// ── Activity phase (single source of truth, shared by Pill + Hero) ──────────
// Bursty workloads hard-cycle gen.state generating->finishing->idle->generating
// every ~20s. Without smoothing the StatePill and Hero flip on every poll tick.
// We collapse state into a 3-value PHASE with a grace HOLD on the DOWN edge only:
//   'live'   -> model is actively working RIGHT NOW (snap up instantly, no debounce)
//   'recent' -> quiet, but within RECENT_HOLD_MS of the last burst (soft 'between
//               requests' treatment; shows the LAST burst's REAL tok/s, captioned)
//   'idle'   -> genuinely quiet past the grace window (stark IDLE — honest)
// Honesty: 'recent' is its OWN label/tone, never GENERATING; the held tok/s is a
// real measured value explicitly captioned as last-burst, never a fake/0 number,
// and it is surfaced ONLY while it is still attributable to the burst that just
// went quiet (same generation_id) — otherwise we show '—' rather than misattribute.
type ActivityPhase = 'live' | 'recent' | 'idle';

// Observed inter-burst gaps run ~5-6s (from a recording of UI flicker), so a
// 10s hold bridges adjacent bursts with margin while still surfacing a genuine
// stop as IDLE within ~10s (a real idle between delegations is minutes).
const RECENT_HOLD_MS = 10000;

const LIVE_STATES: ReadonlySet<GenerationState> = new Set<GenerationState>([
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
      if (gen.tok_s !== null) {
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

  // Surface the held number only if it belongs to the burst that just went quiet,
  // so a later short burst that never populated tok_s can't show a prior burst's rate.
  const attributable = heldGenId.current !== null && heldGenId.current === liveGenId.current;
  return { phase, heldTokS: attributable ? heldTokS.current : null };
}

// ── State pill ────────────────────────────────────────────────────────────
// Derived from status.generation.state, cross-checked against the lifecycle
// blocks (active/loading/grace) so a LOADING sidecar shows its elapsed timer.
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

function derivePill(data: StatusSnapshot, gen: GenerationInfo, phase: ActivityPhase): Pill {
  // STALLED takes precedence — it's the one alarm state.
  if (gen.stalled || gen.state === 'stalled') {
    return { label: 'STALLED', tone: 'red' };
  }
  // Grace HOLD: a quiet tick within the recent window keeps a soft 'RECENT' badge
  // instead of snapping to gray IDLE — same source of truth as the Hero.
  if (phase === 'recent') {
    return { label: 'RECENT', tone: 'slate', detail: 'between requests' };
  }
  switch (gen.state) {
    case 'generating':
      return { label: 'GENERATING', tone: 'green' };
    case 'prefill': {
      const p = gen.prompt_progress;
      return {
        label: 'PREFILL',
        tone: 'blue',
        detail: p !== null ? `${Math.round(p * 100)}% prompt` : undefined,
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
function fmtTokS(v: number | null): string {
  // null === first-decode pending — show an honest em dash, never a fake low number.
  if (v === null) return '—';
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

  // RECENT (grace hold): quiet, but just finished a burst. Show the LAST burst's
  // REAL throughput, dimmed, explicitly captioned as historical — NOT live, NOT a
  // fabricated number. This replaces the stark IDLE flip during bursty workloads.
  // (heldTokS is null when not attributable to the just-finished burst -> '—'.)
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

  // IDLE-CLARITY: only after the grace window fully elapses (phase==='idle') do we
  // paint the stark 'IDLE' panel — never a fake '0.0', and no longer flipping every
  // burst gap (that down-edge is now held by the RECENT phase above).
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

  const pending = gen.tok_s === null;
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
// Prominent, own visible block (not just a pill detail). Surfaces the "model is
// loaded but does nothing then eventually generates" gap as PROMPT PROCESSING.
// Rendered only while state === 'prefill'.
function fmtInt(n: number): string {
  return n.toLocaleString('en-US');
}

function PrefillBar({ gen }: { gen: GenerationInfo }) {
  const p = gen.prompt_progress;
  const frac = p !== null ? Math.min(1, Math.max(0, p)) : null;
  const pctLabel = frac !== null ? `${Math.round(frac * 100)}%` : '…';
  return (
    <div className="rounded-lg border border-blue-700 bg-blue-950/30 p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-xs uppercase tracking-wide text-blue-300 font-semibold">
          Processing prompt
        </div>
        <span className="font-mono text-sm text-blue-200 tabular-nums">{pctLabel}</span>
      </div>
      <div className="h-3 bg-slate-800 rounded overflow-hidden">
        {frac !== null ? (
          <div
            className="h-full bg-blue-500 transition-all"
            style={{ width: `${frac * 100}%` }}
          />
        ) : (
          <div className="h-full w-1/3 bg-blue-600/70 rounded animate-pulse" />
        )}
      </div>
      <div className="mt-2 text-xs text-blue-300/70">
        reading the prompt before the first token decodes
      </div>
    </div>
  );
}

// ── Progress bar ──────────────────────────────────────────────────────────
function Progress({ gen }: { gen: GenerationInfo }) {
  const bounded = gen.max_tokens !== null && gen.pct !== null;
  // CONTEXT used / window: surface the request context against the model's
  // context-window capacity so a 40K sub-request doesn't look wrong next to a
  // 250K window. Fall back to just the used count when n_ctx is unknown.
  const contextLabel =
    gen.n_ctx !== null
      ? `${fmtInt(gen.n_prompt_tokens)} / ${fmtInt(gen.n_ctx)} window`
      : `${fmtInt(gen.n_prompt_tokens)} tokens`;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Progress</div>
      <div className="flex items-baseline justify-between text-sm mb-2">
        <span className="font-mono text-slate-200 tabular-nums">
          {bounded
            ? `${fmtInt(gen.n_decoded)} / ${fmtInt(gen.max_tokens as number)} tokens`
            : `${fmtInt(gen.n_decoded)} tokens`}
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
        ) : (
          // Indeterminate barber-pole for unbounded generations (count-up only).
          <div className="h-full w-1/3 bg-emerald-600/70 rounded animate-pulse" />
        )}
      </div>
      <div className="mt-2 space-y-0.5 text-xs text-slate-500">
        <div className="font-mono">context: {contextLabel}</div>
        {gen.eta_s !== null && (
          <div className="font-mono">eta: {gen.eta_s.toFixed(1)}s</div>
        )}
        {gen.riders > 1 && (
          <div className="font-mono text-amber-400">riders: {gen.riders} concurrent slots</div>
        )}
      </div>
    </div>
  );
}

// ── Sparkline ─────────────────────────────────────────────────────────────
// Rolling tok_s_instant samples kept in LOCAL state (a useRef array appended on
// each status.generation change) — no separate hook, no network.
function Sparkline({ samples }: { samples: number[] }) {
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

// ── Live output pane ──────────────────────────────────────────────────────
// ALWAYS opens the anchor-follow stream (one persistent connection that follows
// the current generation). Only gated by gen.streaming for the UNAVAILABLE copy
// — a non-streaming request can't surface live text.
function LiveOutputPane({ gen }: { gen: GenerationInfo }) {
  const { text, available, ended } = useLiveOutput();
  const boxRef = useRef<HTMLPreElement | null>(null);
  const stick = useRef(true);

  // Auto-scroll to bottom while the user is parked at the bottom.
  useEffect(() => {
    const el = boxRef.current;
    if (el && stick.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [text]);

  const onScroll = () => {
    const el = boxRef.current;
    if (!el) return;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  };

  // Keep the captured tail visible across burst gaps (useLiveOutput retains it):
  // only fall back to the "unavailable" notice when there is NO text to show AND
  // the current request isn't streaming. This stops the pane flipping
  // text <-> "unavailable" on every inter-request gap (a 2nd flicker source seen
  // in a screen recording), without inventing output we never received.
  if (!gen.streaming && !text) {
    return (
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
        <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Live output</div>
        <div className="text-sm italic text-slate-500">
          live text unavailable (non-streaming request)
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-xs uppercase tracking-wide text-slate-500">Live output</div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-600">
          <span>live tail (last ~16KiB)</span>
          {available && !ended && (
            <span className="inline-flex items-center gap-1 text-emerald-400">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              live
            </span>
          )}
          {ended && <span className="text-slate-500">ended</span>}
        </div>
      </div>
      <pre
        ref={boxRef}
        onScroll={onScroll}
        className="h-72 overflow-y-auto whitespace-pre-wrap break-words rounded bg-black/40 border border-slate-800 p-3 text-xs font-mono text-slate-200"
      >
        {text || <span className="text-slate-600">— waiting for output —</span>}
      </pre>
    </div>
  );
}

// ── Panel ─────────────────────────────────────────────────────────────────
// Folded under Dashboard: takes the live status snapshot from the parent's
// useStatus() poll (no own header/tab chrome). Renders nothing heavy when there
// is no generation block, mirroring the prior tab's "no active generation" copy.
export default function LiveInference({ data }: { data: StatusSnapshot }) {
  // Rolling sparkline buffer of tok_s_instant. Appended once per generation
  // snapshot (i.e. per useStatus tick), keyed off measured_at_iso to dedupe.
  const sparkRef = useRef<number[]>([]);
  const lastMeasured = useRef<string | null>(null);
  const [, forceTick] = useState(0);

  const gen = data.generation ?? null;
  // Temporal phase (live/recent/idle) — single source of truth shared by the
  // StatePill and Hero so they smooth together and never disagree. Called
  // unconditionally (before the early return) to keep hook order stable.
  const { phase, heldTokS } = useActivityPhase(gen);

  useEffect(() => {
    if (!gen) return;
    if (gen.measured_at_iso === lastMeasured.current) return;
    lastMeasured.current = gen.measured_at_iso;
    const next = [...sparkRef.current, gen.tok_s_instant];
    sparkRef.current =
      next.length > SPARK_SAMPLES ? next.slice(next.length - SPARK_SAMPLES) : next;
    forceTick((t) => t + 1);
  }, [gen]);

  if (!gen) {
    return (
      <div className="space-y-4">
        <div className="flex items-baseline justify-between">
          <h2 className="text-xl font-bold text-slate-100">Live inference</h2>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-6 text-sm italic text-slate-500">
          — no active generation —
        </div>
      </div>
    );
  }

  const pill = derivePill(data, gen, phase);
  // PEAK tok/s — the max of the recent tok_s_instant samples already kept in the
  // sparkline buffer. Bursty workloads dip to 0 between bursts; peak shows the
  // true speed. (Buffer may be empty on the very first tick.)
  const peak = sparkRef.current.length > 0 ? Math.max(...sparkRef.current) : 0;
  const prefill = gen.state === 'prefill';

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <h2 className="text-xl font-bold text-slate-100">Live inference</h2>
        <StatePill pill={pill} />
      </div>

      {prefill && <PrefillBar gen={gen} />}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <Hero gen={gen} phase={phase} heldTokS={heldTokS} />
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
            tok/s (last {SPARK_SAMPLES})
          </div>
          <Sparkline samples={sparkRef.current} />
          <div className="mt-2 flex items-center justify-between text-xs font-mono text-slate-500 tabular-nums">
            <span>instant: {gen.tok_s_instant.toFixed(1)} tok/s</span>
            <span className="text-slate-400">peak: {peak.toFixed(1)} tok/s</span>
          </div>
        </div>
      </div>

      <Progress gen={gen} />

      <LiveOutputPane gen={gen} />

      <div className="text-xs text-slate-500 flex items-center gap-3">
        <span>
          generation:{' '}
          <span className="font-mono">{gen.generation_id ?? '—'}</span>
        </span>
        <span>
          measured: <span className="font-mono">{gen.measured_at_iso}</span>
        </span>
      </div>
    </div>
  );
}
