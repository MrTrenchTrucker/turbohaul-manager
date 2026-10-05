import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import {
  getManifest,
  putManifest,
  patchManifestHidden,
  restoreManifestDefaults,
  getManifests,
  getBlobs,
  deleteBlobApi,
  deleteManifestApi,
  putBlobDescription,
  type Manifest,
  type ManifestSummaryRow,
} from '../api';
// The existing redacted state channel -- the same one
// useStatus already rides. NOT a new mechanism, and nothing new in api.ts.
import { subscribeWsState } from '../ws';
import {
  FLAGS_SCHEMA,
  CATEGORY_ORDER,
  getCategorySectionFlags,
  type FlagSpec,
  type FlagCategory,
} from '../flagsSchema';
// Reuse the dashboard's card/box treatment rather than inventing a new
// one -- reuse what is already in
// components/dashboard/ where possible. KV in particular is exactly the
// label/value row shape the tile stats need.
import { KV } from './dashboard/primitives';

// Models tab -- comprehensive structured editor mirroring BE
// SAFE_LLAMA_FLAGS exactly: the FE must match the BE
// exactly. ~80 flags grouped by category. Primary flags featured at top.

function fmtBytes(n?: number): string {
  if (!n) return '—';
  const gb = n / 1e9;
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = n / 1e6;
  return `${mb.toFixed(1)} MB`;
}

type FlagValue = number | string | boolean | undefined;

export function FlagInput({
  spec,
  value,
  enabled,
  onChange,
  onToggle,
  describedById,
}: {
  spec: FlagSpec;
  value: FlagValue;
  enabled: boolean;
  onChange: (v: FlagValue) => void;
  onToggle: (en: boolean) => void;
  // Optional id of a cross-field hint (e.g. the
  // reasoning_budget/n_predict conflict warning) that describes THIS
  // field. Wired to aria-describedby on the actual widget so the
  // association is programmatic, not just visual adjacency. Undefined by
  // default -- every existing caller (all ~108 flags) is unaffected.
  describedById?: string;
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
          aria-describedby={describedById}
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
          aria-describedby={describedById}
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
          aria-describedby={describedById}
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
          aria-describedby={describedById}
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
            aria-describedby={describedById}
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
          aria-describedby={describedById}
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
            aria-describedby={describedById}
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
          aria-describedby={describedById}
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

// A manifest can hold a thinking budget larger than the
// output ceiling the caller actually receives -- the model then spends its
// whole allowance inside <think> and returns nothing. Both values are
// individually legal (see flagsSchema.ts's own bounds), so nothing else
// catches this. `n_predict: -1` means unlimited and must NEVER trip this --
// it is the healthy default on most models.
//
// Gated on `enabledFlags.has(...)`, not just `flagValues[...]`: disabling a
// flag (ModelEditor's onToggleFlag) removes it from enabledFlags but does
// NOT clear its value from flagValues, so a since-disabled flag can leave a
// stale number sitting in flagValues that must not feed this check.
//
// Non-blocking by design: the
// true ceiling at request time is `min(n_predict, a caller-supplied
// max_tokens)` and the caller's value is invisible here (it is a per-request
// forwarded knob, not a manifest field -- see api/chat_completion.py's
// _COMMON_FORWARDED_KNOBS). A manifest-only check can be wrong in EITHER
// direction: it can miss a conflict a stricter caller introduces, or flag one
// a looser caller would never hit. This function only ever informs; nothing
// in Models.tsx uses its result to block Save.
export function computeReasoningBudgetConflict(
  flagValues: Record<string, FlagValue>,
  enabledFlags: Set<string>,
): { nPredict: number; reasoningBudget: number } | null {
  if (!enabledFlags.has('n_predict') || !enabledFlags.has('reasoning_budget')) {
    return null;
  }
  const nPredict = flagValues.n_predict;
  const reasoningBudget = flagValues.reasoning_budget;
  if (typeof nPredict !== 'number' || typeof reasoningBudget !== 'number') {
    return null;
  }
  if (nPredict > 0 && reasoningBudget >= nPredict) {
    return { nPredict, reasoningBudget };
  }
  return null;
}

// The id below is the aria-describedby target both FlagInput rows (n_predict,
// reasoning_budget) point to when computeReasoningBudgetConflict is non-null
// -- the programmatic association the accessibility requirement asks for,
// not just visual placement next to the fields. `aria-live="polite"` because
// this hint can appear while focus stays inside a number input as the
// operator types, not only when they tab between fields.
export const REASONING_BUDGET_HINT_ID = 'reasoning-budget-conflict-hint';

export function ReasoningBudgetConflictHint({
  nPredict,
  reasoningBudget,
}: {
  nPredict: number;
  reasoningBudget: number;
}) {
  return (
    <div
      id={REASONING_BUDGET_HINT_ID}
      aria-live="polite"
      className="mt-2 rounded-md border border-amber-700/50 bg-amber-950/30 px-3 py-2 text-xs text-amber-300"
    >
      ⚠ Reasoning budget ({reasoningBudget}) is at or above the output limit ({nPredict}).
      At these values, the model thinks past its output ceiling and returns nothing at
      all. A caller that supplies its own limit can avoid this, but as saved, it will
      fail for any caller that doesn't. Lower reasoning_budget below n_predict, or raise
      n_predict, to fix it here.
    </div>
  );
}

export function CategorySection({
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
  // display_name/description are editable here as form fields, not only
  // via raw JSON. model_tag stays out of this editor entirely: it is
  // immutable here by design (see the Rename control in the manifest
  // list), never folded into this save path.
  const [displayNameInput, setDisplayNameInput] = useState<string>('');
  const [descriptionInput, setDescriptionInput] = useState<string>('');
  const [rawJson, setRawJson] = useState<string>('');
  const [rawMode, setRawMode] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [restoring, setRestoring] = useState<boolean>(false);
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
        setDisplayNameInput(m.display_name ?? '');
        setDescriptionInput(m.description ?? '');
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
          // Only the structured path needs this -- raw mode already
          // carries whatever display_name/description the user typed into
          // the JSON textarea itself.
          display_name: displayNameInput || undefined,
          description: descriptionInput || undefined,
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
      setDisplayNameInput(m2.display_name ?? '');
      setDescriptionInput(m2.description ?? '');
      setRawJson(JSON.stringify(m2, null, 2));
      onSaved();
    } catch (ex: unknown) {
      setErr(String(ex));
    } finally {
      setSaving(false);
    }
  }, [manifest, flagValues, enabledFlags, rawJson, rawMode, etag, tag, onSaved, displayNameInput, descriptionInput]);

  const onRestore = useCallback(async () => {
    if (!manifest) return;
    setRestoring(true);
    setErr('');
    setOk('');
    try {
      const res = await restoreManifestDefaults(tag, etag);
      const clearedN = Object.keys(res.cleared).length;
      // Re-read the manifest so the panel reflects the restored values
      // immediately — closing and reopening to see success is the bug.
      const { manifest: m2, etag: e2 } = await getManifest(tag);
      setManifest(m2);
      setEtag(e2);
      const flags2 = (m2.llama_server_flags || {}) as Record<string, FlagValue>;
      setFlagValues(flags2);
      setEnabledFlags(new Set(Object.keys(flags2)));
      setDisplayNameInput(m2.display_name ?? '');
      setDescriptionInput(m2.description ?? '');
      setRawJson(JSON.stringify(m2, null, 2));
      setOk(
        `${clearedN ? `Cleared ${clearedN} override(s)` : 'Already on defaults'}. ${res.takes_effect}`,
      );
      onSaved();
    } catch (ex: unknown) {
      setErr(String(ex));
    } finally {
      setRestoring(false);
    }
  }, [manifest, etag, tag, onSaved]);

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

  // Recomputed on every render (cheap -- two map lookups), not
  // memoized on flagValues/enabledFlags identity, since both are plain
  // objects/Sets replaced wholesale on every edit anyway (see
  // onChangeFlag/onToggleFlag above) -- a memo dependency array here would
  // just restate the same two values with no real caching benefit.
  const reasoningBudgetConflict = computeReasoningBudgetConflict(flagValues, enabledFlags);

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
          <button
            onClick={onRestore}
            disabled={restoring || saving}
            className="px-3 py-1 rounded text-xs font-medium border border-slate-600 text-slate-300 hover:border-amber-500 hover:text-amber-300 disabled:opacity-50"
            title="Clear this model's KV/checkpoint overrides back to the defaults"
          >
            {restoring ? 'Restoring…' : 'Restore defaults'}
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

      {!rawMode && (
        <div className="rounded-md border border-slate-800 bg-slate-925 p-3 space-y-2">
          <label className="block">
            <span className="block text-xs text-slate-400 mb-1">Display name</span>
            <input
              type="text"
              value={displayNameInput}
              onChange={(e) => setDisplayNameInput(e.target.value)}
              placeholder="e.g. my-model-35b-q4"
              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 text-sm"
            />
          </label>
          <label className="block">
            <span className="block text-xs text-slate-400 mb-1">Description</span>
            <input
              type="text"
              value={descriptionInput}
              onChange={(e) => setDescriptionInput(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 text-sm"
            />
          </label>
          <p className="text-[10px] text-slate-500">
            model_tag ({manifest.model_tag}) is immutable here — use Rename in the manifest list.
          </p>
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
                describedById={
                  reasoningBudgetConflict && (spec.name === 'n_predict' || spec.name === 'reasoning_budget')
                    ? REASONING_BUDGET_HINT_ID
                    : undefined
                }
              />
            ))}
            {reasoningBudgetConflict && (
              <ReasoningBudgetConflictHint
                nPredict={reasoningBudgetConflict.nPredict}
                reasoningBudget={reasoningBudgetConflict.reasoningBudget}
              />
            )}
          </div>

          {CATEGORY_ORDER.map((cat) => {
            // Primary flags are featured in the Primary block above, so they must be
            // excluded here or they render twice. Filtering on the primary property
            // itself (rather than excluding the whole 'Common' category) covers the
            // primary flags in every category (cache_type_k, cache_type_v, temp,
            // top_p, spec_type), and lets a future non-primary Common flag appear
            // instead of being hidden. Empty categories are dropped just below.
            //
            // spec_type is `dualRender: true`: it is also `primary`, so without this it
            // would be filtered out of every category section and could not be found by
            // category name when looking for speculative-decode settings.
            // It renders in BOTH places -- same flagValues/enabledFlags state
            // and the same onChangeFlag/onToggleFlag callbacks passed to both spots, so
            // the two controls are one piece of state, not a copy that needs syncing.
            const flagsInCat = getCategorySectionFlags(cat);
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

// The three-level model-first layout.
// L1 = one tile per real model, grouped by gguf_blob_sha256.
// L2 = click a tile -> only that model's manifests.
// L3 = click Edit on a manifest row -> the editor expands INLINE directly
//      beneath that row (an accordion, not a page).
// Plugins (kind !== 'model') are filtered out of this page entirely,
// so they never appear in the model list.

export interface BlobEntry {
  digest: string;
  sizeBytes: number;
  // A human description is shown on
  // the model tile. null when unset (GET /api/blobs's own uniform-shape
  // rule) -- never assume a string.
  description: string | null;
}

// The summary listing carries display_name and gguf_blob_sha256
// (manifests.py's list_manifests), so L1/L2 build entirely from ONE
// getManifests() call, with no per-manifest fetch. getManifest(tag) (the
// full per-manifest read) is used ONLY for the L3 accordion when a manifest is actually opened
// (~108-flag editor) -- by design, both are available:
// list = enough to draw the page, individual = everything, on demand.
export interface ManifestDetailLite {
  model_tag: string;
  displayName: string | null;
  blobSha256: string;
  hidden: boolean;
  revision: number;
  etag: string;
  // Fallback source for the tile's file-size row when GET /api/blobs (the
  // canonical source) isn't live yet -- same disclosed-gap pattern as
  // display_name/gguf_blob_sha256. The tile shows a human description plus
  // file size rather than per-manifest context_size/expected_vram_bytes
  // (which would raise "which manifest's number do I show"), so those
  // fields are not carried here.
  fileSizeBytes: number | null;
}

export interface TileGroup {
  digest: string;
  sizeBytes: number | null;
  displayName: string | null;
  // Blob-only data (GET /api/blobs), no manifest-level
  // fallback exists or is meaningful -- a "model description" is a
  // property of the blob/model, not any one manifest.
  description: string | null;
  manifests: ManifestDetailLite[];
}

// This page is filtered to kind==='model' throughout (plugins
// are out of scope entirely). Extracted as its own pure function so the
// exclusion is a directly testable claim, not just an inline predicate
// buried in a useEffect. Rows with hidden===null are the "unreadable
// manifest" sentinel (see manifests.py) -- kind is also null there, so this
// same filter drops them for free; they were never actionable in the old
// bottom list either (every action button there was `disabled={... ||
// row.hidden === null}`).
export function selectModelManifestRows(rows: ManifestSummaryRow[]): ManifestSummaryRow[] {
  return rows.filter((r) => r.kind === 'model' && r.hidden !== null);
}

// Maps ONE summary row (already filtered by selectModelManifestRows) to
// the shape L1/L2 need, straight from the listing -- no per-manifest fetch.
// Returns null (skip, don't crash or fabricate a blob grouping) if
// gguf_blob_sha256 is missing -- that shape is only supposed to occur on
// the unreadable-row sentinel, which selectModelManifestRows already
// drops, but the type is `string | null` on the wire and this is the one
// place that narrows it, so the guard is real, not decorative. Guard for
// null display_name explicitly too (guard rather than assuming a
// string) -- both the unreadable-sentinel case AND
// a genuinely readable manifest that simply has no display_name set
// (Manifest.display_name is optional) produce null here, and both are
// legitimate, not errors.
export function summaryRowToDetail(row: ManifestSummaryRow): ManifestDetailLite | null {
  if (row.gguf_blob_sha256 == null) return null;
  return {
    model_tag: row.model_tag,
    displayName: row.display_name ?? null,
    blobSha256: row.gguf_blob_sha256,
    hidden: !!row.hidden,
    revision: row.revision,
    etag: row.etag,
    fileSizeBytes: row.gguf_size_bytes ?? null,
  };
}

// Pure, network-free -- unit-testable without a fetch mock. `blobs` is the
// GET /api/blobs result; null means the route was not reachable (the blobs route is not
// deployed yet in this tree; the gap is disclosed rather than blocking:
// the page ships with it). With blobs===null, a blob with
// zero manifests simply gets no tile -- the zero-manifest guarantee is
// only as complete as the blobs route's availability, and that is stated here, not
// silently assumed away.
export function groupIntoTiles(
  details: ManifestDetailLite[],
  blobs: BlobEntry[] | null,
): TileGroup[] {
  const byDigest = new Map<string, TileGroup>();
  for (const d of details) {
    let g = byDigest.get(d.blobSha256);
    if (!g) {
      g = { digest: d.blobSha256, sizeBytes: null, displayName: null, description: null, manifests: [] };
      byDigest.set(d.blobSha256, g);
    }
    g.manifests.push(d);
    if (!g.displayName && d.displayName) g.displayName = d.displayName;
  }
  if (blobs) {
    for (const b of blobs) {
      let g = byDigest.get(b.digest);
      if (!g) {
        g = { digest: b.digest, sizeBytes: b.sizeBytes, displayName: null, description: b.description, manifests: [] };
        byDigest.set(b.digest, g);
      } else {
        if (g.sizeBytes == null) g.sizeBytes = b.sizeBytes;
        g.description = b.description;
      }
    }
  }
  for (const g of byDigest.values()) {
    g.manifests.sort((a, b) => a.model_tag.localeCompare(b.model_tag));
  }
  return Array.from(byDigest.values()).sort(
    (a, b) => (a.displayName ?? '').localeCompare(b.displayName ?? '') || a.digest.localeCompare(b.digest),
  );
}

// Rename, extracted to a standalone function so its two safety
// properties are directly testable rather than only observable inside a
// component closure:
//   (1) ORDER -- putManifest (create the new tag) is awaited and MUST
//       complete before deleteManifestApi (remove the old tag) is ever
//       called. Inverting this is the difference between "a failed rename
//       leaves both tags" (safe) and "a failed rename leaves NEITHER"
//       (data loss) -- an ordering regression that the earlier
//       pre-extraction version's tests did not catch.
//   (2) the required "callers using the old name will get a 404"
//       warning is present on a successful rename's message -- not
//       decoration, the decision to permit rename in the first
//       place (callers are responsible for adapting) depends on the user
//       actually being told.
// Dependencies are passed in explicitly (not imported directly) so a test
// can inject spies and assert call order without touching the network.
export async function performRename(
  oldTag: string,
  newTag: string,
  deps: {
    getManifest: typeof getManifest;
    putManifest: typeof putManifest;
    deleteManifestApi: typeof deleteManifestApi;
  },
): Promise<{ ok: boolean; message: string }> {
  const { manifest } = await deps.getManifest(oldTag);
  await deps.putManifest(newTag, { ...manifest, model_tag: newTag }, null);
  try {
    await deps.deleteManifestApi(oldTag);
    return {
      ok: true,
      message: `${oldTag} renamed to ${newTag}. Callers using the old name will get a 404.`,
    };
  } catch (delEx: unknown) {
    return {
      ok: false,
      message: `${newTag} created, but deleting the old tag ${oldTag} failed: ${String(delEx)} — both now exist, clean up manually.`,
    };
  }
}

// L1 tile. NO action controls on the tile face at all (Delete was moved off
// the tile entirely so a destructive action takes several deliberate steps:
// open the card first, then Delete Manifest, then the
// actual Delete Model -- a little bit safer). Clickable anywhere -- purely
// "click to open" (one big button).
// No action belongs on the tile face any more.
// Description + file size are shown, not a per-manifest ctx/VRAM figure
// (which would raise "which manifest's number do I show"). The extra
// content rows make the tile rectangular rather than square and give it
// enough information; no padding adjustment is needed.
//
// A restrained depth treatment so the tile reads as a
// pressable object, not a flat outline -- subtle. No
// existing elevation convention in this codebase to match (only 5x ring-1,
// 4x ring-2, one shadow-lg total) -- designed, not copied, as a deliberate
// choice. `shadow-sm ring-1 ring-white/10` at rest (soft shadow + a faint
// light-ring hint of a top/edge highlight, restrained enough to survive a
// dark background). `active:scale-[0.98] active:shadow-inner`
// is the actual press response a touchscreen needs (hover alone does
// nothing there) -- the single cheapest thing that reads as "button."
// `transition` keeps the state change smooth, not jarring. Applied
// identically to ManifestRow's own root -- NOT to ModelInfoCard, which is
// not a click target. Two behaviours depend on a real browser render and
// are not asserted here: whether ring-1 sitting
// outside the existing border double-outlines, and whether CSS :active on
// an ancestor also fires when a nested button (which stopPropagation only
// stops the click/toggle handler on, not the :active pseudo-class) is
// pressed.
export function ModelTileCard({
  tile,
  onOpen,
}: {
  tile: TileGroup;
  onOpen: () => void;
}) {
  const label = tile.displayName ?? 'Unnamed model';
  const fileSize = tile.sizeBytes ?? tile.manifests.find((m) => m.fileSizeBytes != null)?.fileSizeBytes ?? null;
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onOpen();
      }}
      className="text-left w-full rounded-lg border border-slate-700 bg-slate-950 p-5 flex flex-col gap-2 shadow-sm ring-1 ring-white/10 transition active:scale-[0.98] active:shadow-inner hover:border-emerald-600 cursor-pointer"
      data-testid="model-tile"
    >
      <h3 className="text-lg font-semibold text-slate-100">{label}</h3>
      {/* Description is read-only here (editing happens only after
          clicking in, where there is a separate
          edit-description button).
          null shows NOTHING, not a placeholder
          sentence -- "no description" as visible text is noise. */}
      {tile.description && <p className="text-xs text-slate-400">{tile.description}</p>}
      {tile.manifests.length === 0 ? (
        <p className="text-xs text-slate-500">No manifests yet — click to add the first one.</p>
      ) : (
        <p className="text-xs text-slate-400">
          {tile.manifests.length} manifest{tile.manifests.length === 1 ? '' : 's'}
        </p>
      )}
      {fileSize != null && (
        <div className="pt-1 border-t border-slate-800">
          <KV k="size" v={fmtBytes(fileSize)} />
        </div>
      )}
      <p className="text-[10px] text-slate-600 font-mono truncate" title={tile.digest}>
        sha: {tile.digest.replace(/^sha256:/, '').slice(0, 16)}…
      </p>
    </div>
  );
}

// The L2 header distinguishes "acts on the MODEL" from "acts on a
// MANIFEST": this card makes clear which actions are on the model itself.
// This card holds the model's OWN identity and OWN actions and
// nothing else -- same bordered/Card styling as ModelTileCard, so the
// container itself communicates "this is about the model" without a label
// saying so.
//
// Edit description and Delete model must NOT be adjacent or share
// an alignment column (a fat-finger hazard). Edit description sits top-right, next to
// the heading; Delete model sits alone at the BOTTOM, past a divider,
// left-aligned -- different column, real vertical distance, a rule
// between them, and Delete keeps its rose color as the one meaningful
// distinction (unchanged).
export function ModelInfoCard({
  tile,
  onEditDescriptionClick,
  onDeleteModelClick,
}: {
  tile: TileGroup;
  onEditDescriptionClick: () => void;
  onDeleteModelClick: () => void;
}) {
  const label = tile.displayName ?? 'Unnamed model';
  const fileSize = tile.sizeBytes ?? tile.manifests.find((m) => m.fileSizeBytes != null)?.fileSizeBytes ?? null;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 flex flex-col gap-2" data-testid="model-info-card">
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-lg font-semibold text-slate-100">{label}</h3>
        <button
          onClick={onEditDescriptionClick}
          className="shrink-0 min-h-11 sm:min-h-0 px-3 py-1 rounded text-sm font-medium border border-slate-600 text-slate-300 hover:border-sky-500 hover:text-sky-300"
        >
          Edit description
        </button>
      </div>
      <p className="text-sm text-slate-400">
        {tile.description || <span className="text-slate-600 italic">No description yet.</span>}
      </p>
      <div className="pt-1 border-t border-slate-800">
        {fileSize != null && <KV k="size" v={fmtBytes(fileSize)} />}
        <KV k="sha" v={`${tile.digest.replace(/^sha256:/, '').slice(0, 16)}…`} />
      </div>
      <div className="pt-3 mt-1 border-t border-slate-800">
        <button
          onClick={onDeleteModelClick}
          className="min-h-11 sm:min-h-0 px-3 py-1 rounded text-sm font-medium border border-rose-900 text-rose-400 hover:bg-rose-950/40"
        >
          Delete model
        </button>
      </div>
    </div>
  );
}

// L2 row: one manifest.
// Clicking on a manifest collapses or expands
// it -- the ROW is the toggle, not only the Edit button. Every nested
// control calls e.stopPropagation() before its own action, so clicking
// Duplicate/Rename/Hide/Delete never also opens/closes the accordion.
// Styled to look clickable (border + hover treatment matching
// ModelTileCard/the dashboard's Card convention) so the affordance and the
// behavior agree: if it looks clickable, people will click it, so it
// had better be. Restore-defaults is NOT duplicated here
// -- it already lives inside the L3 editor.
//
// Row layout:
// There is no separate "✎ description" or Edit/Close button: the row
//   IS the toggle (tested below), and ModelEditor's own "close" is the
//   only close. Display name/description are the FIRST two fields inside
//   the editor the row opens, so they are easy to find.
// Duplicate/Rename/Hide/Delete share one uniform size (text-sm).
//   Delete keeps its rose color -- that distinction is meaningful. Fit at
//   normal desktop width is an estimate (about 140px spare at 1680px);
//   it cannot be rendered in unit tests.
//
// Touch targets: the repo's 44px minimum touch target applies here too
//   (as in App.tsx, SubTabs.tsx, FastLane.tsx, Plugins.tsx, Settings.tsx);
//   py-2 alone renders 36px, a mis-tap risk on a phone. It is applied via
//   `min-h-11 sm:min-h-0` (FastLane.tsx's own pattern) to
//   ModelInfoCard's Edit-description/Delete-model,
//   the Manifests-section "+ Add manifest", DeleteBlobPrompt's three
//   buttons, and InlineTextPrompt's Create/Cancel (rename and add-manifest
//   both use it). Additive class, no pinned className string disturbed.
//   The manifest-row buttons (px-5 py-3) already clear the floor.
// `flex-wrap` is set on DeleteBlobPrompt's and InlineTextPrompt's
//   button rows; it matches the dashboard's own idiom on a
//   variable-count control row (ResidentCard.tsx).
//
// This page is also read on a PHONE. At 412px a single nowrap line of
// `shrink-0` row-button chips overflows, and there would be no way to know
// what the manifest is until you click on it. So:
//   (a) the row is TWO LINES of content -- the manifest's OWN
//       displayName (GET /api/manifests returns a
//       distinct per-manifest display_name) as the primary line, model_tag
//       in mono underneath.
//       If displayName is null, the tag becomes the primary line and NO
//       second line renders (never an empty one).
//   (b) the four action buttons sit on their OWN row below, in a
//       `flex-wrap` container with no `shrink-0` -- they WRAP onto new
//       lines on a narrow viewport instead of forcing horizontal overflow.
//       Two lines of row height leave room for
//       `px-5 py-3 text-base font-semibold`, a comfortable touch target
//       (~44px tall).
// The no-horizontal-overflow check (scrollWidth === clientWidth at 412px,
// L1/L2/L3) is a real-browser measurement that cannot be run in unit
// tests; it is not assumed to pass just because the CSS uses flex-wrap.
//
// Colour: bg-slate-950 stays constant in every state (rest, hover,
// expanded), so an open row is never the single LIGHTEST thing on a dark
// page; only the BORDER color changes (border-slate-800 at rest,
// border-emerald-600 on hover/expanded) -- the same border-differentiates-
// not-fill mechanism the dashboard's own nested boxes use (ThroughputSection
// .tsx's Combined Context / SplitBar boxes are the SAME slate-950 as their
// containing panel, separated only by border). The Manifests section and
// its rows are both bg-slate-950, nested by border alone. ModelTileCard's
// own hover uses no lighter fill, for the same reason.
export function ManifestRow({
  row,
  expanded,
  busy,
  onToggleEdit,
  onToggleHide,
  onDuplicate,
  onRenameClick,
  onDelete,
}: {
  row: ManifestDetailLite;
  expanded: boolean;
  busy: boolean;
  onToggleEdit: () => void;
  onToggleHide: () => void;
  onDuplicate: () => void;
  onRenameClick: () => void;
  onDelete: () => void;
}) {
  const primaryLine = row.displayName ?? row.model_tag;
  // A manifest whose display_name happens to equal its
  // own model_tag would otherwise render the identical text twice
  // (a manifest named like its tag). Two identical lines
  // is worse than one.
  const showSecondaryLine = row.displayName != null && row.displayName !== row.model_tag;
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onToggleEdit}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onToggleEdit();
      }}
      className={`flex flex-col gap-2 py-2 px-3 rounded-lg border cursor-pointer bg-slate-950 shadow-sm ring-1 ring-white/10 transition active:scale-[0.98] active:shadow-inner ${
        expanded ? 'border-emerald-600' : 'border-slate-800 hover:border-emerald-600'
      }`}
      data-testid="manifest-row"
    >
      <div className="flex items-center gap-2">
        <span
          className={`w-2 h-2 shrink-0 rounded-full ${row.hidden ? 'bg-amber-400' : 'bg-emerald-400'}`}
          title={row.hidden ? 'hidden' : 'visible'}
        />
        <div className="min-w-0 flex-1">
          <p className="text-sm text-slate-100 truncate">{primaryLine}</p>
          {showSecondaryLine && (
            <p className="font-mono text-xs text-slate-500 truncate" title={row.model_tag}>
              {row.model_tag}
            </p>
          )}
        </div>
        <span className="text-[10px] text-slate-500 font-mono shrink-0">rev {row.revision}</span>
      </div>
      {/* The row's action buttons are right-
          aligned, not packed left. justify-content resolves PER WRAPPED
          LINE, so every line (including a short last one) stays flush to
          the same right edge -- the ragged slack lands on the left of the
          short line, not as buttons drifting away from the block. */}
      <div className="flex flex-wrap justify-end gap-2">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDuplicate();
          }}
          disabled={busy}
          className="px-5 py-3 rounded-md text-base font-semibold border border-slate-600 text-slate-300 hover:border-sky-500 hover:text-sky-300 disabled:opacity-40"
        >
          Duplicate
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRenameClick();
          }}
          disabled={busy}
          className="px-5 py-3 rounded-md text-base font-semibold border border-slate-600 text-slate-300 hover:border-sky-500 hover:text-sky-300 disabled:opacity-40"
        >
          Rename
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggleHide();
          }}
          disabled={busy}
          className="px-5 py-3 rounded-md text-base font-semibold border border-slate-600 text-slate-300 hover:border-amber-500 hover:text-amber-300 disabled:opacity-40"
        >
          {row.hidden ? 'Show' : 'Hide'}
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          disabled={busy}
          className="px-5 py-3 rounded-md text-base font-semibold border border-rose-900 text-rose-400 hover:bg-rose-950/40 disabled:opacity-40"
        >
          Delete
        </button>
      </div>
    </div>
  );
}

// Small inline text-entry control shared by Rename and Add manifest
// -- both are "type a tag, confirm" with the same shape.
// The description editor needs a multi-line
// field (the old edit description box was too small -- measured 26px
// tall, ~34 chars visible against a 2000-char server cap, no maxLength set
// so a long paste just 400s). This component is SHARED with rename and
// add-manifest, where a single-line tag input is correct and must never
// become a textarea (a model_tag can never legally contain a newline).
// `multiline` defaults to false/undefined, and the multiline branch is an
// EARLY RETURN -- the single-line branch below is the untouched-by-
// construction original (only its className strings gained a
// recolor and a `min-h-11` on the input itself, matching
// the intent that "untouched" meant TYPE, not size; the input was
// only 26px tall when the prompt was open, not just resting). Neither rename nor add-manifest pass
// `multiline` -- they hit the same code path as before.
export function InlineTextPrompt({
  label,
  initial,
  confirmLabel,
  note,
  busy,
  onConfirm,
  onCancel,
  multiline,
}: {
  label: string;
  initial: string;
  confirmLabel: string;
  note?: string;
  busy: boolean;
  onConfirm: (value: string) => void;
  onCancel: () => void;
  multiline?: boolean;
}) {
  const [value, setValue] = useState(initial);

  if (multiline) {
    return (
      <div className="flex flex-col gap-2 py-1.5 px-2 rounded bg-slate-950 border border-slate-700">
        <span className="text-[11px] text-slate-400">{label}</span>
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          rows={3}
          maxLength={2000}
          className="w-full min-h-11 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-slate-100"
        />
        <div className="flex flex-wrap justify-end gap-2">
          <button
            onClick={() => onConfirm(value)}
            disabled={busy || !value.trim()}
            className="min-h-11 sm:min-h-0 px-2 py-1 rounded text-[11px] font-medium bg-emerald-700 text-white hover:bg-emerald-600 disabled:opacity-50"
          >
            {confirmLabel}
          </button>
          <button
            onClick={onCancel}
            disabled={busy}
            className="min-h-11 sm:min-h-0 px-2 py-1 rounded text-[11px] font-medium border border-slate-600 text-slate-300"
          >
            Cancel
          </button>
        </div>
        {note && <p className="text-[10px] text-amber-400">{note}</p>}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1 py-1.5 px-2 rounded bg-slate-950 border border-slate-700">
      <span className="text-[11px] text-slate-400">{label}</span>
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className="min-h-11 sm:min-h-0 flex-1 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-slate-100"
        />
        <button
          onClick={() => onConfirm(value)}
          disabled={busy || !value.trim()}
          className="min-h-11 sm:min-h-0 shrink-0 px-2 py-1 rounded text-[11px] font-medium bg-emerald-700 text-white hover:bg-emerald-600 disabled:opacity-50"
        >
          {confirmLabel}
        </button>
        <button
          onClick={onCancel}
          disabled={busy}
          className="min-h-11 sm:min-h-0 shrink-0 px-2 py-1 rounded text-[11px] font-medium border border-slate-600 text-slate-300"
        >
          Cancel
        </button>
      </div>
      {note && <p className="text-[10px] text-amber-400">{note}</p>}
    </div>
  );
}

// Two options, not a confirm-maze, framed as the question:
// "Do you want to delete all the manifests with it or
// keep the manifests and just delete the model?" The blob delete 409s while
// any manifest still names the blob (see performBlobDelete), so "keep" only
// succeeds for a blob no manifest references. Manifests left orphaned
// re-link automatically if the same bytes are ever re-pulled, since the
// blob store is content-addressed.
export function DeleteBlobPrompt({
  manifestCount,
  busy,
  onDropManifests,
  onKeepManifests,
  onCancel,
}: {
  manifestCount: number;
  busy: boolean;
  onDropManifests: () => void;
  onKeepManifests: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="flex flex-col gap-2 mt-2 py-2 px-3 rounded bg-rose-950/20 border border-rose-800">
      <p className="text-xs text-rose-200">
        {manifestCount} manifest{manifestCount === 1 ? '' : 's'} reference this model.
      </p>
      <div className="flex flex-wrap gap-2">
        <button
          onClick={onDropManifests}
          disabled={busy}
          className="min-h-11 sm:min-h-0 px-2 py-1 rounded text-[11px] font-medium bg-rose-800 text-white hover:bg-rose-700 disabled:opacity-50"
        >
          Delete blob + {manifestCount} manifest{manifestCount === 1 ? '' : 's'}
        </button>
        <button
          onClick={onKeepManifests}
          disabled={busy}
          className="min-h-11 sm:min-h-0 px-2 py-1 rounded text-[11px] font-medium border border-rose-700 text-rose-300 hover:bg-rose-950/40 disabled:opacity-50"
        >
          Delete blob, keep manifest{manifestCount === 1 ? '' : 's'}
        </button>
        <button
          onClick={onCancel}
          disabled={busy}
          className="min-h-11 sm:min-h-0 px-2 py-1 rounded text-[11px] font-medium border border-slate-600 text-slate-300"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

// ===== Refresh when something CHANGES, not on a bare timer =====

/** Fallback tick. NOT the mechanism -- the safety net under the event channel.
 *
 * Deliberately NOT useStatus's 1000 ms. Following that pattern means the same
 * SHAPE (a ref, an interval, cleared on unmount), not the same NUMBER: the
 * interval is a property of the data's cost and change rate, not of React.
 * /status is a tiny payload -- what the 1000 ms precedent was written for --
 * while /api/manifests + /api/blobs are far heavier:
 * list_manifests re-reads and pydantic-validates EVERY yaml on EVERY call
 * with no caching, so the cost also grows linearly with the registry. That
 * is many times the server work per tick, for data that changes a few times
 * a day.
 */
export const MODELS_POLL_INTERVAL_MS = 10_000;

/** Trailing debounce, so a burst of events costs one re-fetch, not one each.
 *
 * Neither existing layer coalesces: ws.ts's subscribeWsState is a bare
 * `ws.onmessage = (e) => handler(JSON.parse(e.data))`, and useStatus calls
 * fetchOnce() unconditionally per event (its wsTimer only defers restarting
 * the INTERVAL -- that de-conflicts event-vs-tick, never event-vs-event).
 * Harmless at 0.29 ms a call; not harmless here. A worker writing twenty
 * manifests in a loop collapses to one or two re-fetches instead of twenty.
 */
export const MODELS_EVENT_DEBOUNCE_MS = 300;

/** The ONLY event names this page reacts to.
 *
 * ⛔ Filtering is not an optimisation, it is required. subscribeWsState hands
 * EVERY event on the socket to EVERY handler, and live_monitor publishes
 * `generation_tick` at ~1 Hz for the whole duration of every generation --
 * its own comment says it exists to drive useStatus's re-fetch. A subscriber
 * that copied useStatus verbatim would therefore re-parse the entire manifest
 * registry at 1 Hz throughout every inference. Inference traffic is
 * continuous; manifest writes are not.
 */
export const MODELS_REFRESH_EVENTS: ReadonlySet<string> = new Set([
  'manifest_changed',
  'blob_changed',
]);

/** Every interaction on this page that a refresh must not land in the middle of. */
export interface ModelsInteractionState {
  /** ModelEditor open (L3 accordion). */
  expandedTag: string | null;
  /** InlineTextPrompt -- rename. */
  renamingTag: string | null;
  /** InlineTextPrompt -- add manifest. */
  addingForDigest: string | null;
  /** InlineTextPrompt -- model description (multiline). */
  editingDescription: boolean;
  /** DeleteBlobPrompt. */
  showDeleteModelPrompt: boolean;
  /** A mutation is in flight. */
  busyKey: string | null;
}

/**
 * True while a refresh must be held back.
 *
 * The hazard is NOT that fresh data overwrites what you typed -- it cannot.
 * ModelEditor takes no data props at all ({tag, onClose, onSaved}) and keys
 * its own fetch on [tag], which a refresh does not change; InlineTextPrompt
 * seeds with `useState(initial)`, an initializer React ignores on later prop
 * changes. Both are immune to a data swap by construction.
 *
 * The hazard is UNMOUNT. An externally deleted or renamed manifest drops its
 * row from `selectedTile.manifests`, the keyed <div> goes, and an open editor
 * dies with every unsaved flag, display_name and description in it. An
 * externally deleted blob nulls `selectedTile` and drops the user from L2 to
 * L1 mid-edit. Suspending closes both, and it stays a pure predicate -- which
 * is the only form this test harness can actually prove.
 *
 * Six fields, covering all seven interaction surfaces: the three
 * InlineTextPrompt mount sites share one component but have three separate
 * open-states, so "InlineTextPrompt open" is the union of renamingTag,
 * addingForDigest and editingDescription rather than a field of its own.
 */
export function shouldSuspendRefresh(s: ModelsInteractionState): boolean {
  return (
    s.expandedTag !== null ||
    s.renamingTag !== null ||
    s.addingForDigest !== null ||
    s.editingDescription ||
    s.showDeleteModelPrompt ||
    s.busyKey !== null
  );
}

export type RefreshTrigger = 'event' | 'tick' | 'unsuspend';

export interface RefreshLatchInput {
  suspended: boolean;
  pendingChange: boolean;
  trigger: RefreshTrigger;
}

export interface RefreshAction {
  refreshNow: boolean;
  pendingChange: boolean;
}

/**
 * The latch. A suspended TICK is merely skipped -- the next one is 10 s away
 * -- but a suspended EVENT is GONE, so correctness cannot be left to the
 * fallback timer this change exists to demote. Anything that arrives while
 * suspended is remembered and applied the moment the last editor closes.
 *
 * A tick latches too, not just an event: the tick exists precisely because
 * events can be missed, so treating one as "may have changed" is the
 * fail-safe reading. Trusting the channel completely is the thing the tick is
 * there to distrust. The practical consequence is that any edit outliving one
 * interval refreshes on close, which is what we want anyway -- and it means
 * no call site needs its own refresh: this covers ModelEditor's own Close,
 * the row's Edit toggle, and backToTiles identically, because all three end
 * with the predicate going false.
 *
 * `unsuspend` while still suspended is contradictory input; it is defined as
 * a no-op that preserves the latch rather than left to chance.
 */
export function resolveRefreshAction({
  suspended,
  pendingChange,
  trigger,
}: RefreshLatchInput): RefreshAction {
  if (trigger === 'unsuspend') {
    if (suspended) return { refreshNow: false, pendingChange };
    return { refreshNow: pendingChange, pendingChange: false };
  }
  if (suspended) return { refreshNow: false, pendingChange: true };
  return { refreshNow: true, pendingChange: false };
}

export default function Models() {
  const [details, setDetails] = useState<Record<string, ManifestDetailLite> | null>(null);
  const [blobs, setBlobs] = useState<BlobEntry[] | null>(null);
  const [selectedDigest, setSelectedDigest] = useState<string | null>(null);
  const [expandedTag, setExpandedTag] = useState<string | null>(null);
  const [err, setErr] = useState<string>('');
  const [notice, setNotice] = useState<string>('');
  const [noticeErr, setNoticeErr] = useState(false);
  const [refreshTick, setRefreshTick] = useState<number>(0);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [addingForDigest, setAddingForDigest] = useState<string | null>(null);
  const [renamingTag, setRenamingTag] = useState<string | null>(null);
  // Delete moved OFF the L1 tile entirely (the user has
  // to click on the card first), so this only ever opens for the
  // CURRENTLY SELECTED (opened) tile -- a boolean is enough, no digest map
  // needed any more.
  const [showDeleteModelPrompt, setShowDeleteModelPrompt] = useState(false);
  // Model-level description edit, only ever relevant
  // for the currently-selected (opened) tile -- same shape as
  // showDeleteModelPrompt above.
  const [editingDescription, setEditingDescription] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const m = await getManifests();
        const modelRows = selectModelManifestRows(m.manifests);
        const details2: Record<string, ManifestDetailLite> = {};
        for (const row of modelRows) {
          const lite = summaryRowToDetail(row);
          if (lite) details2[row.model_tag] = lite;
        }
        setDetails(details2);
        setErr('');
      } catch (ex: unknown) {
        setErr(String(ex));
      }
      try {
        const b = await getBlobs();
        setBlobs(b.blobs.map((x) => ({ digest: x.digest, sizeBytes: x.size_bytes, description: x.description })));
      } catch {
        // Known gap, disclosed: GET
        // /api/blobs may not be deployed yet in this tree. Degrade to
        // deriving tiles from manifests only -- zero-manifest blobs simply
        // don't render a tile until this route ships; self-heals with no
        // code change once it does (see getBlobs in api.ts).
        setBlobs(null);
      }
    })();
  }, [refreshTick]);

  // Both triggers funnel into refreshTick -- the SAME loader
  // the Refresh button already drives. One refresh path, three triggers, so
  // the event channel adds no second way to mutate `details` (which is also
  // why the event's identifier is deliberately ignored below: reacting to
  // WHICH tag changed would be exactly that second path).
  const suspended = shouldSuspendRefresh({
    expandedTag,
    renamingTag,
    addingForDigest,
    editingDescription,
    showDeleteModelPrompt,
    busyKey,
  });
  // Latest-value refs: read only from async callbacks, never during render.
  // The subscription must NOT be re-created when suspension flips -- ws.ts
  // reconnects with exponential backoff on close, so tearing the socket down
  // every time an editor opens would be far worse than the staleness.
  const suspendedRef = useRef(suspended);
  suspendedRef.current = suspended;
  const pendingChangeRef = useRef(false);
  const prevSuspendedRef = useRef(suspended);
  const intervalRef = useRef<ReturnType<typeof window.setInterval> | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const applyTrigger = useCallback((trigger: RefreshTrigger) => {
    const { refreshNow, pendingChange } = resolveRefreshAction({
      suspended: suspendedRef.current,
      pendingChange: pendingChangeRef.current,
      trigger,
    });
    pendingChangeRef.current = pendingChange;
    if (refreshNow) setRefreshTick((t) => t + 1);
  }, []);

  // Fire the latched refresh the instant the last editor or prompt closes,
  // rather than waiting up to a full interval for the next tick.
  useEffect(() => {
    if (prevSuspendedRef.current && !suspended) applyTrigger('unsuspend');
    prevSuspendedRef.current = suspended;
  }, [suspended, applyTrigger]);

  // One subscription and one fallback interval for the page's lifetime.
  useEffect(() => {
    const sub = subscribeWsState((ev) => {
      const name = ev.event;
      if (typeof name !== 'string' || !MODELS_REFRESH_EVENTS.has(name)) return;
      if (debounceRef.current !== null) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        debounceRef.current = null;
        applyTrigger('event');
      }, MODELS_EVENT_DEBOUNCE_MS);
    });
    intervalRef.current = window.setInterval(() => {
      applyTrigger('tick');
    }, MODELS_POLL_INTERVAL_MS);
    return () => {
      sub.close();
      if (intervalRef.current !== null) clearInterval(intervalRef.current);
      if (debounceRef.current !== null) clearTimeout(debounceRef.current);
    };
  }, [applyTrigger]);

  const tiles = useMemo(
    () => (details ? groupIntoTiles(Object.values(details), blobs) : []),
    [details, blobs],
  );
  const selectedTile = tiles.find((t) => t.digest === selectedDigest) ?? null;

  const backToTiles = () => {
    setSelectedDigest(null);
    setExpandedTag(null);
    setAddingForDigest(null);
    setRenamingTag(null);
    setShowDeleteModelPrompt(false);
    setEditingDescription(false);
  };

  // PUT /api/blobs/{digest}/description. Empty string
  // clears it server-side -- InlineTextPrompt's own Save button is already
  // disabled on an empty value, so clearing needs the Cancel-then-retype
  // path today; there is no explicit "clear" affordance.
  const updateModelDescription = async (digest: string, description: string) => {
    setBusyKey(digest);
    try {
      await putBlobDescription(digest, description);
      setNoticeErr(false);
      setNotice('Description saved.');
      setEditingDescription(false);
      setRefreshTick((t) => t + 1);
    } catch (ex: unknown) {
      setNoticeErr(true);
      setNotice(`Description save failed: ${String(ex)} — hit Refresh and retry.`);
    } finally {
      setBusyKey(null);
    }
  };

  const toggleHide = async (row: ManifestDetailLite) => {
    setBusyKey(row.model_tag);
    try {
      const res = await patchManifestHidden(row.model_tag, !row.hidden, row.etag);
      setNoticeErr(false);
      setNotice(`${row.model_tag} ${res.hidden ? 'hidden' : 'visible'} (rev ${res.revision}).`);
      setRefreshTick((t) => t + 1);
    } catch (ex: unknown) {
      setNoticeErr(true);
      setNotice(`Toggle failed for ${row.model_tag}: ${String(ex)} — hit Refresh and retry.`);
    } finally {
      setBusyKey(null);
    }
  };

  const deleteManifestRow = async (row: ManifestDetailLite) => {
    setBusyKey(row.model_tag);
    try {
      await deleteManifestApi(row.model_tag);
      setNoticeErr(false);
      setNotice(`${row.model_tag} deleted.`);
      if (expandedTag === row.model_tag) setExpandedTag(null);
      setRefreshTick((t) => t + 1);
    } catch (ex: unknown) {
      setNoticeErr(true);
      setNotice(`Delete failed for ${row.model_tag}: ${String(ex)} — hit Refresh and retry.`);
    } finally {
      setBusyKey(null);
    }
  };

  const duplicateManifest = async (row: ManifestDetailLite, newTag: string) => {
    setBusyKey(row.model_tag);
    try {
      const { manifest } = await getManifest(row.model_tag);
      await putManifest(newTag, { ...manifest, model_tag: newTag }, null);
      setNoticeErr(false);
      setNotice(`${row.model_tag} duplicated to ${newTag}.`);
      setRefreshTick((t) => t + 1);
    } catch (ex: unknown) {
      setNoticeErr(true);
      setNotice(`Duplicate failed for ${row.model_tag}: ${String(ex)} — hit Refresh and retry.`);
    } finally {
      setBusyKey(null);
    }
  };

  // PUT the new tag (a create -- no If-Match), then DELETE the old one.
  // Never in place -- model_tag is the manifest's filename on disk AND the
  // KV bin name (manager.py), which is exactly why the old
  // in-place overwrite was a silent no-op rather than a real
  // rename. If the create succeeds but the delete fails, BOTH tags now exist
  // -- surfaced as an explicit notice rather than silently left half-done,
  // since Refresh will otherwise just show two rows with no explanation.
  const renameManifest = async (row: ManifestDetailLite, newTag: string) => {
    setBusyKey(row.model_tag);
    try {
      const result = await performRename(row.model_tag, newTag, {
        getManifest,
        putManifest,
        deleteManifestApi,
      });
      setNoticeErr(!result.ok);
      setNotice(result.message);
      setRenamingTag(null);
      if (expandedTag === row.model_tag) setExpandedTag(null);
      setRefreshTick((t) => t + 1);
    } catch (ex: unknown) {
      setNoticeErr(true);
      setNotice(`Rename failed for ${row.model_tag}: ${String(ex)} — hit Refresh and retry.`);
    } finally {
      setBusyKey(null);
    }
  };

  // Create under the current tile's own blob digest, or the new
  // manifest would join no tile at all (or the wrong one).
  const addManifest = async (digest: string, newTag: string) => {
    setBusyKey(digest);
    try {
      await putManifest(
        newTag,
        { model_tag: newTag, gguf_blob_sha256: digest, llama_server_flags: {} },
        null,
      );
      setNoticeErr(false);
      setNotice(`${newTag} added.`);
      setAddingForDigest(null);
      setRefreshTick((t) => t + 1);
    } catch (ex: unknown) {
      setNoticeErr(true);
      setNotice(`Add manifest failed for ${newTag}: ${String(ex)} — hit Refresh and retry.`);
    } finally {
      setBusyKey(null);
    }
  };

  // dropManifests deletes every manifest referencing this blob first
  // (so the blob doesn't just vanish out from under still-listed rows),
  // then the blob itself. Keeping manifests skips straight to the blob
  // delete.
  // The blob delete 409s if ANY manifest still names the
  // digest -- so "keep manifests" on a blob that still has some will fail,
  // by design, with the referencing tags named in the message. Note also
  // that `tile.manifests` is grouped on gguf_blob_sha256 ALONE, so it can be
  // empty for a blob that three models reference as their projector; the
  // 409's own list is authoritative, not this one.
  const performBlobDelete = async (tile: TileGroup, dropManifests: boolean) => {
    setBusyKey(tile.digest);
    try {
      if (dropManifests) {
        for (const m of tile.manifests) {
          await deleteManifestApi(m.model_tag);
        }
      }
      await deleteBlobApi(tile.digest);
      setNoticeErr(false);
      setNotice(
        `Deleted ${tile.displayName ?? tile.digest.slice(0, 16) + '…'}` +
          (dropManifests
            ? ` and ${tile.manifests.length} manifest(s).`
            : tile.manifests.length > 0
              ? `, kept ${tile.manifests.length} manifest(s) (now orphaned until re-linked).`
              : '.'),
      );
      setShowDeleteModelPrompt(false);
      if (selectedDigest === tile.digest) backToTiles();
      setRefreshTick((t) => t + 1);
    } catch (ex: unknown) {
      setNoticeErr(true);
      setNotice(`Delete failed: ${String(ex)} — hit Refresh and retry.`);
    } finally {
      setBusyKey(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Refresh is level-scoped, not page-level: a page-level one would be
          rendered unconditionally regardless of L1/L2 -- a loose, unboxed
          control on a page where everything else sits in
          bordered bg-slate-950 boxes.
          L1 gets its own boxed Refresh below; L2's box further down holds
          Back + Refresh together. One consistent rule -- every page-level
          action control lives in a bordered box, always -- rather than a
          special case for the new Back+Refresh pairing alone. */}
      <div className="flex items-baseline justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-100">Models</h2>
          <p className="text-sm text-slate-400">
            {selectedTile
              ? `${selectedTile.displayName ?? 'Unnamed model'} — its manifests only.`
              : `${tiles.length} model${tiles.length === 1 ? '' : 's'} — FE schema mirrors BE SAFE_LLAMA_FLAGS exactly (${FLAGS_SCHEMA.length} flags). Edits hot-reload on the next stage; no restart required.`}
          </p>
        </div>
      </div>

      {err && (
        <div className="px-3 py-2 rounded bg-red-950/40 border border-red-700 text-sm text-red-200">
          ⚠ {err}
        </div>
      )}

      {notice && (
        <div
          className={`px-3 py-2 rounded border text-sm ${
            noticeErr
              ? 'bg-red-950/40 border-red-700 text-red-200'
              : 'bg-emerald-950/40 border-emerald-700 text-emerald-200'
          }`}
        >
          {notice}
        </div>
      )}

      {!selectedTile ? (
        // ---- L1 ----
        // Matches ResidentsPanel's own
        // lg: gate rather than sm:, so a phone in
        // desktop-request mode (~980px) stays one column like every other
        // 2-up surface in the app -- consistency, not a content-fit need
        // of its own (a tile's own content is light enough that sm: would
        // have fit). Resident boxes follow the same rule: in desktop
        // view on mobile, one underneath the other is correct.
        // Known tradeoff: a 1023px desktop
        // window now also drops to one column, the same tradeoff
        // ResidentsPanel already ships.
        <div className="space-y-3">
          <div className="rounded-lg border border-slate-700 bg-slate-950 p-2 flex justify-end">
            <button
              onClick={() => setRefreshTick((t) => t + 1)}
              className="px-3 py-1 rounded text-sm font-medium border border-slate-600 text-slate-300 hover:text-white"
            >
              ↻ Refresh
            </button>
          </div>
          {!details ? (
            <p className="text-sm text-slate-500">Loading models…</p>
          ) : tiles.length === 0 ? (
            <p className="text-sm text-slate-500">No models. Use /api/pull or stage GGUFs to populate.</p>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3">
              {tiles.map((t) => (
                <ModelTileCard key={t.digest} tile={t} onOpen={() => setSelectedDigest(t.digest)} />
              ))}
            </div>
          )}
        </div>
      ) : (
        // ---- L2 (+ inline L3) ----
        // Back is on its OWN row, large enough to tap, and physically
        // separated from what's
        // below it. The rest of the header is two
        // distinct, bordered regions rather than loose floating buttons --
        // a MODEL CARD (ModelInfoCard: identity + model-level actions,
        // nothing else) and a separate MANIFESTS section (its own heading,
        // "+ Add manifest" on that heading row, since adding a manifest is
        // an action on the LIST, not the model). The
        // container now says what level each action is at,
        // instead of the user having to infer it. Back now
        // sits inside a bordered bg-slate-950 box alongside Refresh --
        // both page-level, non-destructive, no adjacency hazard between
        // them (unlike Delete-model's own required separation).
        <div className="space-y-3">
          <div className="rounded-lg border border-slate-700 bg-slate-950 p-2 flex items-center justify-between">
            <button
              onClick={backToTiles}
              className="px-5 py-2.5 rounded-md text-base font-semibold border border-slate-600 text-slate-200 hover:text-white hover:border-slate-400"
            >
              ← Back to models
            </button>
            <button
              onClick={() => setRefreshTick((t) => t + 1)}
              className="px-3 py-1 rounded text-sm font-medium border border-slate-600 text-slate-300 hover:text-white"
            >
              ↻ Refresh
            </button>
          </div>

          <ModelInfoCard
            tile={selectedTile}
            onEditDescriptionClick={() => setEditingDescription(true)}
            onDeleteModelClick={() => setShowDeleteModelPrompt(true)}
          />
          {editingDescription && (
            <InlineTextPrompt
              label="Model description"
              initial={selectedTile.description ?? ''}
              confirmLabel="Save"
              busy={busyKey === selectedTile.digest}
              onConfirm={(desc) => void updateModelDescription(selectedTile.digest, desc)}
              onCancel={() => setEditingDescription(false)}
              multiline
            />
          )}
          {showDeleteModelPrompt && (
            <DeleteBlobPrompt
              manifestCount={selectedTile.manifests.length}
              busy={busyKey === selectedTile.digest}
              onDropManifests={() => void performBlobDelete(selectedTile, true)}
              onKeepManifests={() => void performBlobDelete(selectedTile, false)}
              onCancel={() => setShowDeleteModelPrompt(false)}
            />
          )}

          <div className="rounded-lg border border-slate-700 bg-slate-950 p-3" data-testid="manifests-section">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-slate-300">
                Manifests ({selectedTile.manifests.length})
              </h3>
              <button
                onClick={() => setAddingForDigest(selectedTile.digest)}
                className="min-h-11 sm:min-h-0 px-3 py-1 rounded text-sm font-medium bg-sky-800 text-white hover:bg-sky-700"
              >
                + Add manifest
              </button>
            </div>
            {addingForDigest === selectedTile.digest && (
              <InlineTextPrompt
                label="New manifest tag"
                initial=""
                confirmLabel="Create"
                busy={busyKey === selectedTile.digest}
                onConfirm={(tag) => void addManifest(selectedTile.digest, tag)}
                onCancel={() => setAddingForDigest(null)}
              />
            )}
            {selectedTile.manifests.length === 0 ? (
              <p className="text-xs text-slate-500 p-2">No manifests yet. Add one above.</p>
            ) : (
              // The row's own `mb-1.5 last:mb-0`
              // NEVER worked -- each row is the sole child of its own
              // `<div key={...}>` wrapper, so `last:mb-0` (:last-child)
              // matched EVERY row against its own one-child parent and
              // zeroed the margin every time. Deleted rather than left as
              // a dead pair that looks load-bearing to the next reader.
              // Fixed with the SAME `gap-3` token the tile grid uses, on a
              // real flex container -- `display: block` (what this div was
              // before) does not support `gap` at all, so the container
              // itself had to become `flex flex-col`, not just gain a class.
              <div className="flex flex-col gap-3">
                {selectedTile.manifests.map((row) => (
                <div key={row.model_tag}>
                  <ManifestRow
                    row={row}
                    expanded={expandedTag === row.model_tag}
                    busy={busyKey === row.model_tag}
                    onToggleEdit={() =>
                      setExpandedTag((cur) => (cur === row.model_tag ? null : row.model_tag))
                    }
                    onToggleHide={() => void toggleHide(row)}
                    onDuplicate={() => void duplicateManifest(row, `${row.model_tag}-copy`)}
                    onRenameClick={() => setRenamingTag(row.model_tag)}
                    onDelete={() => void deleteManifestRow(row)}
                  />
                  {renamingTag === row.model_tag && (
                    <InlineTextPrompt
                      label="New tag"
                      initial={row.model_tag}
                      confirmLabel="Rename"
                      note="Callers using the old name will get a 404 once this completes."
                      busy={busyKey === row.model_tag}
                      onConfirm={(newTag) => void renameManifest(row, newTag)}
                      onCancel={() => setRenamingTag(null)}
                    />
                  )}
                  {expandedTag === row.model_tag && (
                    <ModelEditor
                      tag={row.model_tag}
                      onClose={() => setExpandedTag(null)}
                      onSaved={() => setRefreshTick((t) => t + 1)}
                    />
                  )}
                </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
