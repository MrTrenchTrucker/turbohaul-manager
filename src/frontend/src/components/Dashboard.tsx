import type { ReactNode } from 'react';
import type {
  ActiveInfo,
  GraceInfo,
  IdleHotInfo,
  LoadingInfo,
  QueueInfo,
  StatusSnapshot,
} from '../api';
import { useStatus } from '../hooks/useStatus';

function Card({
  title,
  tone,
  children,
}: {
  title: string;
  tone: 'idle' | 'active' | 'warn';
  children: ReactNode;
}) {
  const borderTone =
    tone === 'active'
      ? 'border-emerald-700'
      : tone === 'warn'
      ? 'border-amber-700'
      : 'border-slate-700';
  return (
    <div className={`rounded-lg border ${borderTone} bg-slate-950 p-4`}>
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">{title}</div>
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

function ActiveCard({ active }: { active: ActiveInfo | null }) {
  if (!active) {
    return (
      <Card title="Active sidecar" tone="idle">
        <div className="text-slate-500 text-sm italic">— no in-flight request —</div>
      </Card>
    );
  }
  return (
    <Card title="Active sidecar" tone="active">
      <div className="text-lg font-semibold text-emerald-300 mb-2">{active.model_tag}</div>
      <KV k="state" v={active.state} />
      <KV k="port" v={active.port} />
      <KV k="pid" v={active.pid} />
      <KV k="thread" v={active.thread_id_prefix || '—'} />
      <KV k="slot" v={active.slot_id} />
    </Card>
  );
}

function GraceCard({ grace }: { grace: GraceInfo | null }) {
  if (!grace) {
    return (
      <Card title="Grace timer" tone="idle">
        <div className="text-slate-500 text-sm italic">— inactive —</div>
      </Card>
    );
  }
  return (
    <Card title="Grace timer" tone="warn">
      <div className="text-lg font-semibold text-amber-300 mb-2">
        {grace.remaining_s}s remaining
      </div>
      <KV k="model" v={grace.model_tag} />
      <KV k="thread" v={grace.thread_id_prefix || '—'} />
      <KV k="extensions" v={`${grace.extension_count}/${grace.max_extensions}`} />
    </Card>
  );
}

function QueueCard({ queue, used, max }: { queue: QueueInfo; used: number; max: number }) {
  const pct =
    queue.staging_queue_max > 0
      ? Math.min(100, (queue.staging_queue_depth / queue.staging_queue_max) * 100)
      : 0;
  return (
    <Card title="Queue" tone={queue.staging_queue_depth > 0 ? 'warn' : 'idle'}>
      <div className="text-lg font-semibold text-slate-200 mb-2">
        {queue.staging_queue_depth} / {queue.staging_queue_max}
      </div>
      <div className="h-2 bg-slate-800 rounded mb-3 overflow-hidden">
        <div
          className="h-full bg-amber-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <KV k="acceptance buffer" v={queue.acceptance_buffer_depth} />
      <KV k="parallel slots" v={`${used} / ${max}`} />
    </Card>
  );
}

function IdleCard({ idle }: { idle: IdleHotInfo | null }) {
  if (!idle) {
    return (
      <Card title="Idle hot-load" tone="idle">
        <div className="text-slate-500 text-sm italic">— cold —</div>
      </Card>
    );
  }
  return (
    <Card title="Idle hot-load" tone="active">
      <div className="text-lg font-semibold text-emerald-300 mb-2">{idle.model_tag}</div>
      <KV k="remaining" v={`${idle.remaining_s}s`} />
    </Card>
  );
}

function LoadingCard({ loading }: { loading: LoadingInfo | null }) {
  if (!loading) {
    return (
      <Card title="Loading" tone="idle">
        <div className="text-slate-500 text-sm italic">— idle —</div>
      </Card>
    );
  }
  return (
    <Card title="Loading" tone="warn">
      <div className="text-lg font-semibold text-amber-300 mb-2">{loading.model_tag}</div>
      <KV k="state" v={loading.state} />
      <KV k="elapsed" v={`${loading.elapsed_s.toFixed(1)}s`} />
      {loading.port !== null && <KV k="port" v={loading.port} />}
      {loading.pid !== null && <KV k="pid" v={loading.pid} />}
      <KV k="thread" v={loading.thread_id_prefix || '—'} />
      <KV k="slot" v={loading.slot_id} />
    </Card>
  );
}

function LoadedBanner({ data }: { data: StatusSnapshot }) {
  // Single at-a-glance line for "what's currently in VRAM". The cards
  // below give per-lifecycle detail; this banner avoids the user reading
  // empty Active/Loading cards as "no model loaded" during GRACE/IDLE_HOT.
  const tag =
    data.active?.model_tag ??
    data.loading?.model_tag ??
    data.grace?.model_tag ??
    data.idle_hot?.model_tag ??
    null;

  if (!tag) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-950 px-4 py-3 text-sm text-slate-500">
        <span className="text-xs uppercase tracking-wide text-slate-600 mr-2">Loaded model</span>
        <span className="italic">— none —</span>
      </div>
    );
  }

  let stateLabel = '';
  let tone = 'border-emerald-700';
  if (data.active) {
    stateLabel = data.active.state;
  } else if (data.loading) {
    stateLabel = `${data.loading.state} · ${data.loading.elapsed_s.toFixed(1)}s`;
    tone = 'border-amber-600';
  } else if (data.grace) {
    stateLabel = `GRACE · ${data.grace.remaining_s}s remaining`;
    tone = 'border-amber-700';
  } else if (data.idle_hot) {
    stateLabel = `IDLE_HOT · ${data.idle_hot.remaining_s}s remaining`;
    tone = 'border-emerald-700';
  }

  return (
    <div className={`rounded-lg border-2 ${tone} bg-slate-950 px-4 py-3`}>
      <div className="text-xs uppercase tracking-wide text-slate-500">Loaded model</div>
      <div className="flex items-baseline gap-4 mt-1 flex-wrap">
        <span className="text-2xl font-bold text-emerald-300 font-mono">{tag}</span>
        <span className="text-sm font-mono text-slate-300">{stateLabel}</span>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { data, error, lastUpdate } = useStatus();

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
      <LoadedBanner data={data} />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <ActiveCard active={data.active} />
        <LoadingCard loading={data.loading} />
        <GraceCard grace={data.grace} />
        <QueueCard
          queue={data.queue}
          used={data.parallel_slots.used}
          max={data.parallel_slots.max}
        />
        <IdleCard idle={data.idle_hot} />
      </div>
      <div className="text-xs text-slate-500 flex items-center gap-3">
        <span>
          last update: <span className="font-mono">{lastUpdate?.toISOString() ?? '—'}</span>
        </span>
        {error && (
          <span className="text-amber-400">⚠ {error.message} (retrying…)</span>
        )}
      </div>
    </div>
  );
}
