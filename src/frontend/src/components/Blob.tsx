import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ModelTag } from '../api';
import { getTags } from '../api';
import { SubTabs } from './SubTabs';
import Models from './Models';

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

// Humanize a raw GGUF parameter_count when the BE hasn't
// supplied a pre-formatted size_label (e.g. "27B").
function humanizeParams(n: number): string {
  if (n < 1e3) return `${n}`;
  if (n < 1e6) return `${(n / 1e3).toFixed(1)}K`;
  if (n < 1e9) return `${(n / 1e6).toFixed(1)}M`;
  if (n < 1e12) return `${(n / 1e9).toFixed(1)}B`;
  return `${(n / 1e12).toFixed(1)}T`;
}

function paramsDisplay(m: ModelTag): string {
  const d = m.details;
  if (d?.size_label) return d.size_label;
  if (typeof d?.parameter_count === 'number') return humanizeParams(d.parameter_count);
  return '—';
}

function TypeBadge({ isMoe }: { isMoe: boolean | null | undefined }) {
  if (isMoe === true) {
    return <span className="px-1.5 py-0.5 rounded bg-purple-900/40 text-purple-300 text-xs">MoE</span>;
  }
  if (isMoe === false) {
    return <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 text-xs">Dense</span>;
  }
  return <span className="px-1.5 py-0.5 rounded bg-slate-900 text-slate-600 text-xs">—</span>;
}

function ModalityBadge({ modality }: { modality: 'text' | 'vision' | null | undefined }) {
  if (modality === 'vision') {
    return <span className="px-1.5 py-0.5 rounded bg-sky-900/40 text-sky-300 text-xs">Vision</span>;
  }
  if (modality === 'text') {
    return <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 text-xs">Text</span>;
  }
  return <span className="px-1.5 py-0.5 rounded bg-slate-900 text-slate-600 text-xs">—</span>;
}

type SortKey = 'name' | 'size' | 'modified' | 'params' | 'type' | 'modality';
type SortDir = 'asc' | 'desc';

const SORT_LABELS: Record<SortKey, string> = {
  name: 'Name',
  size: 'Size',
  modified: 'Date loaded',
  params: 'Parameters',
  type: 'Type (MoE/Dense)',
  modality: 'Modality',
};

// Nulls always sort last, in BOTH directions — never let "unknown" masquerade
// as the smallest/earliest/first value just because a sort flipped direction.
function compareNullable<T>(
  a: T | null | undefined,
  b: T | null | undefined,
  cmp: (a: T, b: T) => number,
  dir: SortDir,
): number {
  const aNull = a === null || a === undefined;
  const bNull = b === null || b === undefined;
  if (aNull && bNull) return 0;
  if (aNull) return 1;
  if (bNull) return -1;
  return cmp(a, b) * (dir === 'asc' ? 1 : -1);
}

function compareModels(a: ModelTag, b: ModelTag, key: SortKey, dir: SortDir): number {
  switch (key) {
    case 'name':
      return compareNullable(a.name, b.name, (x, y) => x.localeCompare(y), dir);
    case 'size':
      return compareNullable(a.size, b.size, (x, y) => x - y, dir);
    case 'modified':
      return compareNullable(a.modified_at, b.modified_at, (x, y) => Date.parse(x) - Date.parse(y), dir);
    case 'params':
      return compareNullable(a.details?.parameter_count, b.details?.parameter_count, (x, y) => x - y, dir);
    case 'type':
      return compareNullable(a.details?.is_moe, b.details?.is_moe, (x, y) => Number(x) - Number(y), dir);
    case 'modality':
      return compareNullable(a.details?.modality, b.details?.modality, (x, y) => x.localeCompare(y), dir);
  }
}

type PullSource = 'url' | 'hf' | 'import';

async function postJSON(path: string, body: unknown): Promise<Response> {
  return fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

async function deleteJSON(path: string, body: unknown): Promise<Response> {
  return fetch(path, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

// Blob is the SubTabs host for the Blob browse view and Models.
export default function Blob() {
  return (
    <SubTabs
      basePath="/blob"
      defaultKey="contents"
      tabs={[
        { key: 'contents', label: 'Blob', element: <BlobContents /> },
        { key: 'models', label: 'Models', element: <Models /> },
      ]}
    />
  );
}

function BlobContents() {
  const [tags, setTags] = useState<ModelTag[] | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  const [query, setQuery] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('name');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const [source, setSource] = useState<PullSource>('url');
  const [pullUrl, setPullUrl] = useState('');
  const [pullSha, setPullSha] = useState('');
  const [pullTag, setPullTag] = useState('');
  const [hfRepo, setHfRepo] = useState('');
  const [hfFile, setHfFile] = useState('');
  const [hfTag, setHfTag] = useState('');
  const [importPath, setImportPath] = useState('');
  const [importTag, setImportTag] = useState('');

  const refresh = useCallback(async () => {
    try {
      const r = await getTags();
      setTags(r.models);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const visibleTags = useMemo(() => {
    if (!tags) return null;
    const q = query.trim().toLowerCase();
    const filtered = q ? tags.filter((m) => m.name.toLowerCase().includes(q)) : tags;
    return [...filtered].sort((a, b) => compareModels(a, b, sortKey, sortDir));
  }, [tags, query, sortKey, sortDir]);

  const runPull = useCallback(async () => {
    setBusy(true);
    setStatus(null);
    try {
      let r: Response;
      if (source === 'url') {
        r = await postJSON('/api/pull-url', {
          url: pullUrl,
          expected_sha256: pullSha || undefined,
          tag: pullTag || undefined,
        });
      } else if (source === 'hf') {
        r = await postJSON('/api/pull-hf', {
          repo: hfRepo,
          file: hfFile,
          tag: hfTag || undefined,
        });
      } else {
        r = await postJSON('/api/import', {
          path: importPath,
          tag: importTag || undefined,
        });
      }
      const text = await r.text();
      setStatus(`HTTP ${r.status}: ${text.slice(0, 300)}`);
      if (r.ok) await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }, [source, pullUrl, pullSha, pullTag, hfRepo, hfFile, hfTag, importPath, importTag, refresh]);

  const runDelete = useCallback(
    async (digest: string) => {
      if (!window.confirm(`Delete blob ${digest.slice(0, 16)}…?`)) return;
      setBusy(true);
      try {
        const r = await deleteJSON('/api/delete', { digest });
        setStatus(`DELETE HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`);
        await refresh();
      } catch (e) {
        setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-slate-200">Installed models</h2>
          <button
            onClick={() => void refresh()}
            className="px-3 py-1 rounded-md bg-slate-800 text-slate-300 text-sm hover:bg-slate-700"
            disabled={busy}
          >
            Refresh
          </button>
        </div>
        {error && (
          <div className="text-amber-400 text-sm mb-3">
            ⚠ {error.message}
          </div>
        )}
        {tags !== null && tags.length > 0 && (
          <div className="flex flex-wrap items-center gap-3 mb-4">
            <div className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by name…"
                className="w-56 rounded-md bg-slate-900 border border-slate-700 px-3 py-1.5 pr-7 text-sm font-mono text-slate-200 focus:outline-none focus:ring-1 focus:ring-emerald-600"
              />
              {query && (
                <button
                  onClick={() => setQuery('')}
                  aria-label="Clear search"
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 text-sm leading-none"
                >
                  ×
                </button>
              )}
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-slate-500">Sort:</span>
              <select
                value={sortKey}
                onChange={(e) => setSortKey(e.target.value as SortKey)}
                className="rounded-md bg-slate-900 border border-slate-700 px-2 py-1.5 text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-emerald-600"
              >
                {(Object.keys(SORT_LABELS) as SortKey[]).map((k) => (
                  <option key={k} value={k}>
                    {SORT_LABELS[k]}
                  </option>
                ))}
              </select>
              <button
                onClick={() => setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))}
                className="px-2 py-1.5 rounded-md bg-slate-800 text-slate-300 text-xs hover:bg-slate-700"
                aria-label={sortDir === 'asc' ? 'Sort ascending' : 'Sort descending'}
              >
                {sortDir === 'asc' ? '↑ Asc' : '↓ Desc'}
              </button>
            </div>
          </div>
        )}
        {tags === null ? (
          <div className="text-slate-500 text-sm italic">Loading…</div>
        ) : tags.length === 0 ? (
          <div className="text-slate-500 text-sm italic">No models installed yet. Pull one below.</div>
        ) : visibleTags && visibleTags.length === 0 ? (
          <div className="text-slate-500 text-sm italic">No models match &lsquo;{query}&rsquo;</div>
        ) : (
          <div className="rounded-lg border border-slate-700 bg-slate-950 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-900 text-xs uppercase text-slate-500">
                <tr>
                  <th className="text-left px-4 py-2">Name</th>
                  <th className="text-right px-4 py-2">Size</th>
                  <th className="text-right px-4 py-2">Params</th>
                  <th className="text-center px-4 py-2">Type</th>
                  <th className="text-center px-4 py-2">Modality</th>
                  <th className="text-left px-4 py-2">Digest</th>
                  <th className="text-left px-4 py-2">Modified</th>
                  <th className="text-right px-4 py-2"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {(visibleTags ?? []).map((m) => (
                  <tr key={m.digest} className="text-slate-300">
                    <td className="px-4 py-2 font-mono">{m.name}</td>
                    <td className="px-4 py-2 font-mono text-right">{formatBytes(m.size)}</td>
                    <td className="px-4 py-2 font-mono text-right text-slate-400">{paramsDisplay(m)}</td>
                    <td className="px-4 py-2 text-center">
                      <TypeBadge isMoe={m.details?.is_moe} />
                    </td>
                    <td className="px-4 py-2 text-center">
                      <ModalityBadge modality={m.details?.modality} />
                    </td>
                    <td className="px-4 py-2 font-mono text-slate-500">
                      {m.digest.slice(0, 16)}…
                    </td>
                    <td className="px-4 py-2 text-slate-500">{m.modified_at ?? '—'}</td>
                    <td className="px-4 py-2 text-right">
                      <button
                        className="px-2 py-1 rounded-md bg-rose-900/40 text-rose-300 text-xs hover:bg-rose-900/60"
                        disabled={busy}
                        onClick={() => void runDelete(m.digest)}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Pull model</h2>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
          <div className="flex gap-2 mb-4">
            {(['url', 'hf', 'import'] as const).map((s) => (
              <button
                key={s}
                onClick={() => setSource(s)}
                className={
                  source === s
                    ? 'px-3 py-1 rounded-md bg-emerald-700 text-white text-sm'
                    : 'px-3 py-1 rounded-md bg-slate-800 text-slate-400 text-sm hover:bg-slate-700'
                }
              >
                {s === 'url' ? 'URL' : s === 'hf' ? 'HuggingFace' : 'Local import'}
              </button>
            ))}
          </div>

          {source === 'url' && (
            <div className="space-y-2">
              <Field label="URL (https only)" value={pullUrl} onChange={setPullUrl} placeholder="https://..." />
              <Field label="Expected sha256 (optional)" value={pullSha} onChange={setPullSha} placeholder="hex 64 chars" />
              <Field label="Tag (optional)" value={pullTag} onChange={setPullTag} placeholder="e.g. my-model:8b" />
            </div>
          )}
          {source === 'hf' && (
            <div className="space-y-2">
              <Field label="HF repo" value={hfRepo} onChange={setHfRepo} placeholder="owner/repo" />
              <Field label="File in repo" value={hfFile} onChange={setHfFile} placeholder="model-q4.gguf" />
              <Field label="Tag (optional)" value={hfTag} onChange={setHfTag} placeholder="e.g. my-model:8b" />
            </div>
          )}
          {source === 'import' && (
            <div className="space-y-2">
              <Field label="Absolute path (must be under import_allowed_root)" value={importPath} onChange={setImportPath} placeholder="/var/lib/turbohaul/import-staging/foo.gguf" />
              <Field label="Tag (optional)" value={importTag} onChange={setImportTag} placeholder="e.g. my-model:8b" />
            </div>
          )}

          <div className="mt-4">
            <button
              onClick={() => void runPull()}
              disabled={busy}
              className="px-4 py-2 rounded-md bg-emerald-700 text-white text-sm font-medium hover:bg-emerald-600 disabled:bg-slate-700"
            >
              {busy ? 'Working…' : `Run ${source}`}
            </button>
          </div>
        </div>
      </div>

      {status && (
        <pre className="rounded-lg border border-slate-700 bg-slate-950 p-3 text-xs text-slate-300 whitespace-pre-wrap break-all">
{status}
        </pre>
      )}
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="block">
      <span className="block text-xs text-slate-400 mb-1">{label}</span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-1.5 text-sm font-mono text-slate-200 focus:outline-none focus:ring-1 focus:ring-emerald-600"
      />
    </label>
  );
}
