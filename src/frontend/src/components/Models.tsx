import { useEffect, useState, useCallback, useMemo } from 'react';
import {
  getTags,
  getManifest,
  putManifest,
  type ModelTag,
  type Manifest,
} from '../api';
import {
  FLAGS_SCHEMA,
  CATEGORY_ORDER,
  getFlagsByCategory,
  type FlagSpec,
  type FlagCategory,
} from '../flagsSchema';

//  v2 Models tab — comprehensive structured editor mirroring BE
// SAFE_LLAMA_FLAGS exactly. FE matches BE
// exactly". ~80 flags grouped by category. Primary flags featured at top.

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

type FlagValue = number | string | boolean | undefined;

function FlagInput({
  spec,
  value,
  enabled,
  onChange,
  onToggle,
}: {
  spec: FlagSpec;
  value: FlagValue;
  enabled: boolean;
  onChange: (v: FlagValue) => void;
  onToggle: (en: boolean) => void;
}) {
  const inputBase =
    'w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 font-mono text-xs disabled:opacity-40';

  let widget: React.ReactNode;
  switch (spec.type) {
    case 'int':
      widget = (
        <input
          type="number"
          min={spec.bounds?.[0]}
          max={spec.bounds?.[1]}
          step={1}
          disabled={!enabled}
          value={typeof value === 'number' ? value : (spec.default as number) ?? 0}
          onChange={(e) => onChange(parseInt(e.target.value || '0', 10))}
          className={inputBase}
        />
      );
      break;
    case 'float':
      widget = (
        <input
          type="number"
          min={spec.bounds?.[0]}
          max={spec.bounds?.[1]}
          step={0.01}
          disabled={!enabled}
          value={typeof value === 'number' ? value : (spec.default as number) ?? 0}
          onChange={(e) => onChange(parseFloat(e.target.value || '0'))}
          className={inputBase}
        />
      );
      break;
    case 'bool':
      widget = (
        <input
          type="checkbox"
          disabled={!enabled}
          checked={typeof value === 'boolean' ? value : (spec.default as boolean) ?? false}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 accent-emerald-500"
        />
      );
      break;
    case 'enum-string':
      widget = (
        <select
          disabled={!enabled}
          value={(value as string) ?? (spec.default as string) ?? ''}
          onChange={(e) => onChange(e.target.value)}
          className={inputBase}
        >
          {spec.enumValues?.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      );
      break;
    case 'int-or-string': {
      const isStr = typeof value === 'string';
      widget = (
        <div className="flex gap-1">
          <select
            disabled={!enabled}
            value={isStr ? (value as string) : '__int__'}
            onChange={(e) => {
              if (e.target.value === '__int__') {
                onChange(spec.default as number ?? 0);
              } else {
                onChange(e.target.value);
              }
            }}
            className={inputBase + ' w-24'}
          >
            <option value="__int__">int…</option>
            {spec.enumValues?.map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
          {!isStr && (
            <input
              type="number"
              min={spec.bounds?.[0]}
              max={spec.bounds?.[1]}
              disabled={!enabled}
              value={typeof value === 'number' ? value : (spec.default as number) ?? 0}
              onChange={(e) => onChange(parseInt(e.target.value || '0', 10))}
              className={inputBase}
            />
          )}
        </div>
      );
      break;
    }
    case 'bool-or-enum': {
      const v = value ?? spec.default;
      widget = (
        <select
          disabled={!enabled}
          value={typeof v === 'boolean' ? (v ? '__true__' : '__false__') : String(v)}
          onChange={(e) => {
            const s = e.target.value;
            if (s === '__true__') onChange(true);
            else if (s === '__false__') onChange(false);
            else onChange(s);
          }}
          className={inputBase}
        >
          <option value="__true__">true (legacy bool)</option>
          <option value="__false__">false (legacy bool)</option>
          {spec.enumValues?.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      );
      break;
    }
    case 'chat-template':
      widget = (
        <div className="flex flex-col gap-1">
          <select
            disabled={!enabled}
            value={
              spec.enumValues?.includes((value as string) ?? '')
                ? (value as string)
                : '__custom__'
            }
            onChange={(e) => {
              if (e.target.value === '__custom__') {
                onChange('');
              } else {
                onChange(e.target.value);
              }
            }}
            className={inputBase}
          >
            <option value="__custom__">— custom string —</option>
            {spec.enumValues?.map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
          <input
            type="text"
            disabled={!enabled}
            placeholder="Custom template name (no Jinja {% or {{ )"
            value={(value as string) ?? ''}
            onChange={(e) => onChange(e.target.value)}
            className={inputBase}
          />
        </div>
      );
      break;
    case 'string':
    default:
      widget = (
        <input
          type="text"
          disabled={!enabled}
          value={(value as string) ?? ''}
          onChange={(e) => onChange(e.target.value)}
          className={inputBase}
        />
      );
      break;
  }

  return (
    <div className="grid grid-cols-[24px_minmax(160px,_220px)_1fr] gap-2 items-center py-1 border-b border-slate-900 last:border-b-0">
      <input
        type="checkbox"
        checked={enabled}
        onChange={(e) => onToggle(e.target.checked)}
        className="h-3.5 w-3.5 accent-slate-500"
        title={enabled ? 'Flag SET in manifest — click to omit (use llama-server default)' : 'Flag OMITTED — click to SET'}
      />
      <div className="flex flex-col">
        <span className={`text-xs font-mono ${enabled ? 'text-slate-100' : 'text-slate-500'}`}>
          {spec.name}
        </span>
        <span className="text-[10px] text-slate-500 leading-tight">{spec.hint}</span>
      </div>
      <div>{widget}</div>
    </div>
  );
}

function CategorySection({
  cat,
  flags,
  values,
  enabledFlags,
  onChange,
  onToggle,
  defaultOpen,
}: {
  cat: FlagCategory;
  flags: FlagSpec[];
  values: Record<string, FlagValue>;
  enabledFlags: Set<string>;
  onChange: (name: string, v: FlagValue) => void;
  onToggle: (name: string, en: boolean) => void;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const setCount = flags.filter((f) => enabledFlags.has(f.name)).length;
  return (
    <div className="rounded-md border border-slate-800 bg-slate-925">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full px-3 py-2 flex justify-between items-center hover:bg-slate-800/50"
      >
        <span className="text-sm font-semibold text-slate-200">
          {open ? '▼' : '▶'} {cat}
          {setCount > 0 && (
            <span className="ml-2 text-[10px] text-emerald-400 font-mono">
              {setCount}/{flags.length} set
            </span>
          )}
          {setCount === 0 && (
            <span className="ml-2 text-[10px] text-slate-600 font-mono">
              0/{flags.length}
            </span>
          )}
        </span>
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-0">
          {flags.map((spec) => (
            <FlagInput
              key={spec.name}
              spec={spec}
              value={values[spec.name]}
              enabled={enabledFlags.has(spec.name)}
              onChange={(v) => onChange(spec.name, v)}
              onToggle={(en) => onToggle(spec.name, en)}
            />
          ))}
        </div>
      )}
    </div>
  );
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
  const [flagValues, setFlagValues] = useState<Record<string, FlagValue>>({});
  const [enabledFlags, setEnabledFlags] = useState<Set<string>>(new Set());
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
        const flags = (m.llama_server_flags || {}) as Record<string, FlagValue>;
        setFlagValues(flags);
        setEnabledFlags(new Set(Object.keys(flags)));
        setRawJson(JSON.stringify(m, null, 2));
      } catch (ex: unknown) {
        setErr(String(ex));
      }
    })();
  }, [tag]);

  const onSave = useCallback(async () => {
    if (!manifest) return;
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
        // Rebuild llama_server_flags from enabled set + values
        const newFlags: Record<string, unknown> = {};
        enabledFlags.forEach((name) => {
          const v = flagValues[name];
          if (v !== undefined && v !== '' && v !== null) {
            newFlags[name] = v;
          }
        });
        // Also: context_size at manifest top-level mirrors flags.ctx_size
        const ctxSize = (newFlags.ctx_size as number) ?? manifest.context_size ?? 4096;
        toSave = {
          ...manifest,
          context_size: ctxSize,
          llama_server_flags: newFlags,
        };
      }
      toSave.model_tag = tag;
      const res = await putManifest(tag, toSave, etag);
      setOk(
        `Saved revision ${res.revision}.${
          res.restart_required ? ' Restart required.' : ' Hot-reload on next stage.'
        }`,
      );
      const { manifest: m2, etag: e2 } = await getManifest(tag);
      setManifest(m2);
      setEtag(e2);
      const flags2 = (m2.llama_server_flags || {}) as Record<string, FlagValue>;
      setFlagValues(flags2);
      setEnabledFlags(new Set(Object.keys(flags2)));
      setRawJson(JSON.stringify(m2, null, 2));
      onSaved();
    } catch (ex: unknown) {
      setErr(String(ex));
    } finally {
      setSaving(false);
    }
  }, [manifest, flagValues, enabledFlags, rawJson, rawMode, etag, tag, onSaved]);

  const onChangeFlag = useCallback((name: string, v: FlagValue) => {
    setFlagValues((s) => ({ ...s, [name]: v }));
  }, []);

  const onToggleFlag = useCallback((name: string, en: boolean) => {
    setEnabledFlags((s) => {
      const ns = new Set(s);
      if (en) {
        ns.add(name);
        // Seed default value if missing
        const spec = FLAGS_SCHEMA.find((f) => f.name === name);
        setFlagValues((v) =>
          v[name] === undefined && spec?.default !== undefined
            ? { ...v, [name]: spec.default as FlagValue }
            : v,
        );
      } else {
        ns.delete(name);
      }
      return ns;
    });
  }, []);

  // Primary flags featured at top
  const primaryFlags = useMemo(() => FLAGS_SCHEMA.filter((f) => f.primary), []);

  if (!manifest) {
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
    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4 space-y-4 max-h-[80vh] overflow-y-auto">
      <div className="flex items-center justify-between sticky top-0 bg-slate-900 -mx-4 px-4 pb-3 border-b border-slate-700 z-10">
        <div>
          <h3 className="text-base font-semibold text-slate-100">
            Editing: {manifest.display_name || manifest.model_tag}
          </h3>
          <p className="text-xs text-slate-400 font-mono">
            tag={manifest.model_tag} · rev={manifest.revision} · etag={etag} · {enabledFlags.size}/{FLAGS_SCHEMA.length} flags set
          </p>
        </div>
        <div className="flex gap-2 items-center">
          <button
            onClick={() => setRawMode((v) => !v)}
            className="px-3 py-1 rounded text-xs font-medium border border-slate-600 text-slate-300 hover:text-white hover:border-slate-400"
          >
            {rawMode ? '← Structured' : 'Raw JSON →'}
          </button>
          <button onClick={onClose} className="text-slate-400 hover:text-white text-sm px-2">close</button>
          <button
            onClick={onSave}
            disabled={saving}
            className="px-4 py-1.5 rounded text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save manifest'}
          </button>
        </div>
      </div>

      {err && (
        <div className="px-3 py-2 rounded bg-red-950/40 border border-red-700 text-xs text-red-200 font-mono">
          ⚠ {err}
        </div>
      )}
      {ok && (
        <div className="px-3 py-2 rounded bg-emerald-950/40 border border-emerald-700 text-xs text-emerald-200">
          ✓ {ok}
        </div>
      )}

      {rawMode ? (
        <div className="space-y-2">
          <p className="text-xs text-slate-400">
            <span className="text-amber-300 font-semibold">Raw JSON manifest:</span> bypasses structured form; pure server-side validation.
          </p>
          <textarea
            value={rawJson}
            onChange={(e) => setRawJson(e.target.value)}
            rows={28}
            spellCheck={false}
            className="w-full bg-slate-950 border border-slate-700 rounded p-3 text-xs text-slate-100 font-mono"
          />
        </div>
      ) : (
        <div className="space-y-3">
          <div className="rounded-md border border-emerald-900 bg-emerald-950/20 p-3">
            <h4 className="text-xs font-semibold text-emerald-300 mb-2">★ Primary (most-edited)</h4>
            {primaryFlags.map((spec) => (
              <FlagInput
                key={spec.name}
                spec={spec}
                value={flagValues[spec.name]}
                enabled={enabledFlags.has(spec.name)}
                onChange={(v) => onChangeFlag(spec.name, v)}
                onToggle={(en) => onToggleFlag(spec.name, en)}
              />
            ))}
          </div>

          {CATEGORY_ORDER.filter((c) => c !== 'Common').map((cat) => {
            const flagsInCat = getFlagsByCategory(cat);
            if (flagsInCat.length === 0) return null;
            const hasSetInCat = flagsInCat.some((f) => enabledFlags.has(f.name));
            return (
              <CategorySection
                key={cat}
                cat={cat}
                flags={flagsInCat}
                values={flagValues}
                enabledFlags={enabledFlags}
                onChange={onChangeFlag}
                onToggle={onToggleFlag}
                defaultOpen={hasSetInCat}
              />
            );
          })}
        </div>
      )}
    </div>
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
            Per-model manifest editor — <strong>FE schema mirrors BE SAFE_LLAMA_FLAGS exactly</strong> ({FLAGS_SCHEMA.length} flags · 50+ DENIED_FLAGS path/RCE-class · suffix-pattern forward-defense · numeric bounds · chat_template Jinja-injection rejected).
            Edits hot-reload on the next stage; no restart required.
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
