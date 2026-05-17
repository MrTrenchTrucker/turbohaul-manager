import { useEffect, useState, useCallback } from 'react';
import {
  getTags,
  getManifest,
  putManifest,
  type ModelTag,
  type Manifest,
} from '../api';

// Wave 2 Models tab — per-model manifest editor.
//
// Per Devil's Advocate scope-down: 5 primary structured fields covering the
// 90% common-edit surface (ctx_size, n_gpu_layers, temp, top_p, cache_type_k)
// plus an Advanced (raw JSON) escape hatch for the long tail. Real-world
// manifest YAMLs in this fleet use 1-2 flags; richer per-knob form is deferred
// until empirical demand surfaces.

const KV_QUANT_OPTIONS = ['f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1'];
const FLASH_ATTN_OPTIONS: (boolean | string)[] = [true, false, 'on', 'off', 'auto'];

function fmtBytes(n?: number): string {
  if (!n) return '—';
  const gb = n / 1e9;
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = n / 1e6;
  return `${mb.toFixed(1)} MB`;
}

function fmtCtx(n?: number): string {
  if (!n) return '—';
  if (n >= 1024) return `${(n / 1024).toFixed(0)}K`;
  return String(n);
}

interface PrimaryFields {
  ctx_size: number;
  n_gpu_layers: number | string;
  temp: number;
  top_p: number;
  cache_type_k: string;
  cache_type_v: string;
}

function extractPrimary(m: Manifest): PrimaryFields {
  const f = (m.llama_server_flags || {}) as Record<string, unknown>;
  return {
    ctx_size: (f.ctx_size as number) ?? m.context_size ?? 4096,
    n_gpu_layers: (f.n_gpu_layers as number | string) ?? 999,
    temp: (f.temp as number) ?? 0.8,
    top_p: (f.top_p as number) ?? 0.95,
    cache_type_k: (f.cache_type_k as string) ?? 'f16',
    cache_type_v: (f.cache_type_v as string) ?? 'f16',
  };
}

function applyPrimary(m: Manifest, p: PrimaryFields): Manifest {
  const flags = { ...(m.llama_server_flags || {}) };
  flags.ctx_size = p.ctx_size;
  flags.n_gpu_layers = p.n_gpu_layers;
  flags.temp = p.temp;
  flags.top_p = p.top_p;
  flags.cache_type_k = p.cache_type_k;
  flags.cache_type_v = p.cache_type_v;
  return {
    ...m,
    context_size: p.ctx_size,
    llama_server_flags: flags,
  };
}

function ModelEditor({
  tag,
  onClose,
  onSaved,
}: {
  tag: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [etag, setEtag] = useState<string>('');
  const [primary, setPrimary] = useState<PrimaryFields | null>(null);
  const [rawJson, setRawJson] = useState<string>('');
  const [rawMode, setRawMode] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [err, setErr] = useState<string>('');
  const [ok, setOk] = useState<string>('');

  useEffect(() => {
    (async () => {
      try {
        const { manifest: m, etag: e } = await getManifest(tag);
        setManifest(m);
        setEtag(e);
        setPrimary(extractPrimary(m));
        setRawJson(JSON.stringify(m, null, 2));
      } catch (ex: unknown) {
        setErr(String(ex));
      }
    })();
  }, [tag]);

  const onSave = useCallback(async () => {
    if (!manifest || !primary) return;
    setSaving(true);
    setErr('');
    setOk('');
    try {
      let toSave: Manifest;
      if (rawMode) {
        try {
          toSave = JSON.parse(rawJson) as Manifest;
        } catch (jx) {
          throw new Error(`Invalid JSON: ${String(jx)}`);
        }
      } else {
        toSave = applyPrimary(manifest, primary);
      }
      // Force model_tag to URL tag
      toSave.model_tag = tag;
      const res = await putManifest(tag, toSave, etag);
      setOk(`Saved revision ${res.revision}.${res.restart_required ? ' Restart required.' : ' Hot-reload on next stage.'}`);
      // Re-fetch to get new ETag
      const { manifest: m2, etag: e2 } = await getManifest(tag);
      setManifest(m2);
      setEtag(e2);
      setPrimary(extractPrimary(m2));
      setRawJson(JSON.stringify(m2, null, 2));
      onSaved();
    } catch (ex: unknown) {
      setErr(String(ex));
    } finally {
      setSaving(false);
    }
  }, [manifest, primary, rawJson, rawMode, etag, tag, onSaved]);

  if (!manifest || !primary) {
    return (
      <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-semibold text-slate-100">Editing: {tag}</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white text-sm">close</button>
        </div>
        <p className="text-sm text-slate-400 mt-3">
          {err ? <span className="text-red-400">{err}</span> : 'Loading...'}
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-slate-100">Editing: {manifest.display_name || manifest.model_tag}</h3>
          <p className="text-xs text-slate-400 font-mono">tag={manifest.model_tag} · rev={manifest.revision} · etag={etag}</p>
        </div>
        <div className="flex gap-2 items-center">
          <button
            onClick={() => setRawMode((v) => !v)}
            className="px-3 py-1 rounded text-xs font-medium border border-slate-600 text-slate-300 hover:text-white hover:border-slate-400"
          >
            {rawMode ? '← Primary fields' : 'Advanced (raw JSON) →'}
          </button>
          <button onClick={onClose} className="text-slate-400 hover:text-white text-sm">close</button>
        </div>
      </div>

      {!rawMode ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Field label="ctx_size" hint="Max context window (tokens). Bigger = more KV cache VRAM. Common: 4K/8K/32K/64K/128K">
            <input
              type="number"
              min={1}
              max={2_000_000}
              value={primary.ctx_size}
              onChange={(e) => setPrimary({ ...primary, ctx_size: parseInt(e.target.value || '0', 10) })}
              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 font-mono"
            />
          </Field>

          <Field label="n_gpu_layers" hint="Number of layers on GPU. 999 = all-on-GPU. 0 = CPU-only. -1 = auto.">
            <input
              type="number"
              min={-1}
              max={999}
              value={typeof primary.n_gpu_layers === 'number' ? primary.n_gpu_layers : 999}
              onChange={(e) => setPrimary({ ...primary, n_gpu_layers: parseInt(e.target.value || '0', 10) })}
              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 font-mono"
            />
          </Field>

          <Field label="temp" hint="Sampling temperature. 0.0 = deterministic. 0.7-1.0 common. >2.0 = chaos.">
            <input
              type="number"
              min={0}
              max={10}
              step={0.05}
              value={primary.temp}
              onChange={(e) => setPrimary({ ...primary, temp: parseFloat(e.target.value || '0') })}
              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 font-mono"
            />
          </Field>

          <Field label="top_p" hint="Nucleus sampling. 1.0 = no truncation. 0.9-0.95 common.">
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={primary.top_p}
              onChange={(e) => setPrimary({ ...primary, top_p: parseFloat(e.target.value || '0') })}
              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 font-mono"
            />
          </Field>

          <Field label="cache_type_k (K cache quant)" hint="f16 = highest quality, biggest. q4_0 = quarter size, slight quality loss. q8_0 = half size, well-tested.">
            <select
              value={primary.cache_type_k}
              onChange={(e) => setPrimary({ ...primary, cache_type_k: e.target.value })}
              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 font-mono"
            >
              {KV_QUANT_OPTIONS.map((q) => (
                <option key={q} value={q}>{q}</option>
              ))}
            </select>
          </Field>

          <Field label="cache_type_v (V cache quant)" hint="Match cache_type_k for symmetric quant. Mismatch is allowed but unusual.">
            <select
              value={primary.cache_type_v}
              onChange={(e) => setPrimary({ ...primary, cache_type_v: e.target.value })}
              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 font-mono"
            >
              {KV_QUANT_OPTIONS.map((q) => (
                <option key={q} value={q}>{q}</option>
              ))}
            </select>
          </Field>
        </div>
      ) : (
        <div className="space-y-2">
          <p className="text-xs text-slate-400">
            <span className="text-amber-300 font-semibold">Advanced mode:</span> raw JSON manifest editor.
            Schema validated server-side. All ~80 SAFE_LLAMA_FLAGS + 22 DENIED_FLAGS gated.
            Adds: rope_*, yarn_*, mirostat_*, dry_*, xtc_*, dynatemp_*, reasoning_*, embeddings, metrics, samplers...
          </p>
          <textarea
            value={rawJson}
            onChange={(e) => setRawJson(e.target.value)}
            rows={20}
            spellCheck={false}
            className="w-full bg-slate-950 border border-slate-700 rounded p-3 text-xs text-slate-100 font-mono"
          />
        </div>
      )}

      {err && (
        <div className="px-3 py-2 rounded bg-red-950/40 border border-red-700 text-sm text-red-200">
          ⚠ {err}
        </div>
      )}
      {ok && (
        <div className="px-3 py-2 rounded bg-emerald-950/40 border border-emerald-700 text-sm text-emerald-200">
          ✓ {ok}
        </div>
      )}

      <div className="flex justify-end gap-2 pt-2 border-t border-slate-800">
        <button
          onClick={onClose}
          className="px-4 py-2 rounded text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800"
          disabled={saving}
        >
          Cancel
        </button>
        <button
          onClick={onSave}
          disabled={saving}
          className="px-4 py-2 rounded text-sm font-medium bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-50"
        >
          {saving ? 'Saving…' : 'Save manifest'}
        </button>
      </div>
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-xs font-semibold text-slate-200 font-mono">{label}</span>
      </div>
      {children}
      {hint && <p className="text-[10px] text-slate-500 mt-1">{hint}</p>}
    </label>
  );
}

function ModelCard({ m, onEdit }: { m: ModelTag; onEdit: () => void }) {
  const ctx = m.details?.context_length;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4 flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <div>
          <h3 className="text-base font-semibold text-slate-100">{m.details?.display_name || m.name}</h3>
          <p className="text-xs text-slate-400 font-mono">{m.name}</p>
        </div>
        <button
          onClick={onEdit}
          className="px-3 py-1 rounded text-xs font-medium border border-slate-600 text-slate-300 hover:text-white hover:border-emerald-500"
        >
          Edit ✎
        </button>
      </div>
      {m.details?.description && (
        <p className="text-xs text-slate-400">{m.details.description}</p>
      )}
      <div className="grid grid-cols-2 gap-2 text-xs font-mono pt-1">
        <KV label="size" v={fmtBytes(m.size)} />
        <KV label="ctx" v={fmtCtx(ctx)} />
        <KV label="vram_expected" v={fmtBytes(m.details?.expected_vram_bytes)} />
        <KV label="rev" v={String(m.revision ?? '?')} />
      </div>
      <p className="text-[10px] text-slate-600 font-mono truncate" title={m.digest}>
        sha: {m.digest?.replace(/^sha256:/, '').slice(0, 16)}…
      </p>
    </div>
  );
}

function KV({ label, v }: { label: string; v: string }) {
  return (
    <div className="flex items-baseline gap-1">
      <span className="text-slate-500">{label}:</span>
      <span className="text-slate-200">{v}</span>
    </div>
  );
}

export default function Models() {
  const [models, setModels] = useState<ModelTag[] | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [err, setErr] = useState<string>('');
  const [refreshTick, setRefreshTick] = useState<number>(0);

  useEffect(() => {
    (async () => {
      try {
        const d = await getTags();
        setModels(d.models);
      } catch (ex: unknown) {
        setErr(String(ex));
      }
    })();
  }, [refreshTick]);

  return (
    <div className="space-y-6">
      <div className="flex items-baseline justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-100">Models</h2>
          <p className="text-sm text-slate-400">
            Per-model manifest editor. Edits hot-reload on the next stage; no restart required.
            BE allowlist: ~80 SAFE_LLAMA_FLAGS · 50+ DENIED_FLAGS (path/RCE-class) · suffix-pattern forward-defense.
          </p>
        </div>
        <button
          onClick={() => setRefreshTick((t) => t + 1)}
          className="px-3 py-1 rounded text-sm font-medium border border-slate-600 text-slate-300 hover:text-white"
        >
          ↻ Refresh
        </button>
      </div>

      {err && (
        <div className="px-3 py-2 rounded bg-red-950/40 border border-red-700 text-sm text-red-200">
          ⚠ {err}
        </div>
      )}

      {editing && (
        <ModelEditor
          tag={editing}
          onClose={() => setEditing(null)}
          onSaved={() => setRefreshTick((t) => t + 1)}
        />
      )}

      {!models ? (
        <p className="text-sm text-slate-500">Loading models…</p>
      ) : models.length === 0 ? (
        <p className="text-sm text-slate-500">No models. Use /api/pull or stage GGUFs to populate.</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((m) => (
            <ModelCard key={m.name} m={m} onEdit={() => setEditing(m.name)} />
          ))}
        </div>
      )}
    </div>
  );
}
