import type { GenerationInfo, LoadVerifyRecord, RequestIdentity, ResidentModel, SpecDowngradeRecord } from '../../api';
import type { GenPane } from '../../hooks/useLiveStream';
import { Card } from './primitives';
import { ResidentCard, type TokRateTone } from './ResidentCard';
import { useBusyTimers } from './alarms';

function VramBars({ vram, vramTotal }: { vram: number[] | null; vramTotal: number[] | null }) {
  const TOTAL = vramTotal?.[0] ?? 24576;  // prefer backend-reported total; fallback to 24 GiB
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
// TODO: backend must populate status.vram even at cap<=1
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

// Stable card order. Primary key model_tag (locale-compare
// ascending) -- the one ResidentModel field that structurally cannot change
// while a resident occupies its card, so sorting on it can't cause a reorder
// due to backend-side value drift. Tie-break spawn_seq ascending, kept as a
// defensive total-order guarantee even though it is
// unreachable in practice (the manager keys the resident registry BY
// model_tag, so two residents cannot share one -- the primary key alone is
// already total). Never mutates or filters the input array: display order
// only, the set-identity guarantee (aggregate == SUM(cards) by
// construction) is untouched.
function sortForDisplay(residents: ResidentModel[]): ResidentModel[] {
  return residents.slice().sort((a, b) => {
    const byTag = a.model_tag.localeCompare(b.model_tag);
    if (byTag !== 0) return byTag;
    return a.spawn_seq - b.spawn_seq;
  });
}

// Resolves each resident to its own SSE pane ONCE here (via the panes
// prop). This is a direct lookup --
// useLiveStream now opens one connection PER RESIDENT TAG, so `panes` is
// always keyed by a real model_tag and every pane's own model_tag equals
// its key. The old null-tag/sole-resident fallback (for the cap<=1 case
// where the SSE frame used to omit model_tag under the single shared-anchor
// stream) is dead code under the new design -- removed rather than left as
// a branch nothing can reach; a reader finding it later would reasonably
// assume it still fires.
function resolvePane(
  resident: ResidentModel,
  panes: Record<string, GenPane>,
): GenPane | null {
  return panes[resident.model_tag] ?? null;
}

export function ResidentsPanel({
  residents,
  vram,
  vramTotal,
  parallelSlots,
  requestIdentity,
  loadVerify,
  specDowngrade,
  panes,
  tokRateTone,
}: {
  residents: ResidentModel[];
  vram: number[] | null;
  vramTotal: number[] | null;
  parallelSlots: { used: number; max: number };
  requestIdentity?: RequestIdentity | null;
  loadVerify?: LoadVerifyRecord[] | null;
  specDowngrade?: SpecDowngradeRecord[] | null;
  // SSE panes, identical shape to what Dashboard.tsx's own
  // LiveOutputPanel already receives from useLiveStream().
  panes: Record<string, GenPane>;
  // dashboard/tokRate.ts's tone function, injected and
  // threaded straight through to each card -- ResidentsPanel does not
  // call it itself, same as it doesn't call residentAlarm itself.
  tokRateTone: (gen: GenerationInfo | null | undefined) => TokRateTone;
}) {
  const hasVram = vram != null && vram.length > 0;
  const ordered = sortForDisplay(residents);
  // ONE hook call for the whole set -- never one
  // useBusyTimers-shaped hook per card.
  const busyTimers = useBusyTimers(residents);
  return (
    <div className="space-y-3" data-testid="residents-panel">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-300">RESIDENTS</h2>
        <span className="text-xs text-slate-500 font-mono">
          slots {parallelSlots.used}/{parallelSlots.max}
        </span>
      </div>
      {/* Desktop 2-up, wrapping to new rows for any count
          > 2 (native CSS grid wrap, no span hack: two per row, with
          new rows being produced). lg: (1024), not md: (768) -- a
          live measurement at md: gave 347px cards with two new
          truncations, below ResidentCard's own proven-safe ~380px mobile
          content width; lg: gives 475px with zero new truncations. Also
          keeps a ~980px desktop-mode mobile browser (Chrome "request
          desktop site") at ONE column, which md: would not have. Mobile
          stays grid-cols-1 unconditionally -- non-negotiable.
          gap-3 deliberately unchanged (this file's own
          existing gap convention; growing it only shrinks the already-
          tight column width further). */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {ordered.map(m => (
          <ResidentCard
            key={m.model_tag}
            model={m}
            requestIdentity={requestIdentity}
            soleResident={residents.length === 1}
            loadVerify={loadVerify}
            specDowngrade={specDowngrade}
            pane={resolvePane(m, panes)}
            busyForS={busyTimers[m.model_tag] ?? null}
            tokRateTone={tokRateTone}
          />
        ))}
      </div>
      {/* Real bars when vram is a populated array (cap>=2); honest placeholder
          otherwise (cap<=1, backend suppression) — never fabricated numbers. */}
      {hasVram ? <VramBars vram={vram} vramTotal={vramTotal} /> : <VramPlaceholder />}
    </div>
  );
}
