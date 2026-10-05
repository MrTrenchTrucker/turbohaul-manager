import { useCallback, useEffect, useState } from 'react';
import type { ConfigFieldSchema, ConfigFieldType, ConfigSchema, ConfigSectionSchema } from '../api';
import { getConfig, getConfigSchema, putConfig } from '../api';

// Schema-driven per-setting editor (defaults, bounds, reset)
// for whichever runtime sections exist. Generalized
// from two hardcoded SectionEditor mounts (queue, pull) to every section
// GET /api/config actually returns, minus BOOT_SECTIONS and a small per-field
// skip list — see deriveRenderableSections/SKIP_FIELDS below for why each
// exclusion exists. Before this, 5 of 7 RuntimeConfig sections (fastlane
// included) were PUT-able but had no UI at all.

interface ConfigShape {
  server: Record<string, unknown>;
  storage: Record<string, unknown>;
  runtime: Record<string, unknown>;
  ui: Record<string, unknown>;
  // Runtime sections are not individually named here on purpose:
  // this index signature is what lets cfg[section] work for any section
  // GET /api/config returns, TS-safely, without hand-listing each one — the
  // exact hand-listing that caused the original bug.
  [section: string]: Record<string, unknown>;
}

// Was a closed union ('queue' | 'pull') that silently hid 5 of 7
// backend-advertised RuntimeConfig sections (including fastlane) from the UI.
// Now just `string`, derived at runtime by deriveRenderableSections() below —
// not a hardcoded list.
type SectionKey = string;
type EditsMap = Partial<Record<SectionKey, Record<string, unknown>>>;
type SectionStatus = { type: 'success' | 'error'; text: string };

// Boot sections are frozen and never PUT-able (config_put.py:43
// BOOT_SECTIONS, prevents the binary-swap attack class). This FE
// list is COSMETIC, not a security control: config_put.py is the actual
// enforcement boundary, and a PUT to a boot section still 403s server-side
// regardless of what renders here. If this list ever goes stale, the failure
// mode is a visible section whose Save 400s/403s — a loud UX bug — never a
// privilege escalation or a binary swap. That asymmetry is what makes an
// EXCLUDE list acceptable here where the INCLUDE list it replaced was
// not: forgetting to add a new runtime section costs nothing (it renders
// automatically); forgetting to add a new boot section costs a papercut, not
// a security hole, because the server still blocks the write either way.
// Mirrors config_put.py's BOOT_SECTIONS name + values so the two stay
// grep-pairable across languages even without shared code. Do not "tidy"
// this into an allowlist.
const BOOT_SECTIONS = new Set(['server', 'storage', 'runtime', 'ui']);

// Fields with a dedicated, independently-owned editor elsewhere in the app —
// rendering them again here would create a second independently-fetched-and-
// saved copy of the same server value. Settings.tsx and FastLane.tsx are, like
// this component, permanently-mounted SubTabs panes that never refetch on
// revisit; editing a field via one and then saving anything else on the other
// (still frozen at whatever it fetched on first visit) silently reverts the
// first edit.
//   - persist.max_bytes -> Settings.tsx (pre-existing intent).
//   - fastlane.enabled / max_normal_wait_s / cross_model_switches_per_min ->
//     Settings.tsx's Fast Lane card.
//   - fastlane.rules -> FastLane.tsx's dedicated rules table. Also
//     structurally unsafe to render generically regardless of ownership: it
//     is list[FastLaneRule] (nested objects), but the schema endpoint reports
//     a bare `type: "array"` indistinguishable from list[str] (e.g.
//     pull.hf_host_allowlist), and the array widget's join('\n')/split('\n')
//     would turn saved rules into "[object Object]" garbage.
const SKIP_FIELDS: Record<string, Set<string> | undefined> = {
  persist: new Set(['max_bytes']),
  fastlane: new Set(['enabled', 'rules', 'max_normal_wait_s', 'cross_model_switches_per_min']),
};

// The only place "which sections render" is decided. Pure + exported so
// the drift test can prove it structurally (nothing gates inclusion
// by an allowlist of specific names) instead of asserting against a literal
// section-name array, which would itself be a THIRD hand-typed list subject
// to the exact drift this function exists to kill. `_`-prefixed keys
// (_provenance, _provenance_stamp) are GET /api/config bookkeeping, not
// sections — the leading-underscore convention is already how this codebase
// marks "not a real section" (_ALL_SECTION_MODELS, _SECTIONS,
// _PROVENANCE_STAMP_KEY all follow it).
export function deriveRenderableSections(allKeys: string[]): string[] {
  return allKeys.filter((k) => !k.startsWith('_') && !BOOT_SECTIONS.has(k)).sort();
}

// Shared by SectionEditor's render loop and resetSectionToDefaults, so a
// skipped field can never become dirty (no onChange ever fires for it) and
// never gets reset-staged either — one filter point, not two that could
// disagree. Exported so the drift test can directly cover the two data-loss
// shapes SKIP_FIELDS exists to prevent (fastlane.rules corruption, the
// Settings.tsx dual-editor revert), not just the section-level list.
export function renderableFields(section: string, sectionData: Record<string, unknown>): string[] {
  const skip = SKIP_FIELDS[section];
  return Object.keys(sectionData).filter((f) => !skip?.has(f));
}

const inputBase =
  'w-full max-w-sm bg-slate-800 border border-slate-600 rounded px-3 py-2 text-slate-100 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500';

function resolveWidgetType(
  fieldSchema: ConfigFieldSchema | undefined,
  sampleValue: unknown,
): ConfigFieldType {
  if (fieldSchema) return fieldSchema.type;
  if (Array.isArray(sampleValue)) return 'array';
  if (typeof sampleValue === 'boolean') return 'boolean';
  if (typeof sampleValue === 'number') return Number.isInteger(sampleValue) ? 'integer' : 'number';
  return 'string';
}

function formatVal(v: unknown): string {
  if (Array.isArray(v)) return v.length ? v.join(', ') : '(empty)';
  if (v === null || v === undefined) return '—';
  return String(v);
}

function defaultAsEditValue(type: ConfigFieldType, def: unknown): unknown {
  if (type === 'integer' || type === 'number') return String(def);
  if (type === 'array') return Array.isArray(def) ? [...def] : [];
  return def;
}

function valuesEqual(a: unknown, b: unknown, type: ConfigFieldType): boolean {
  if (type === 'integer' || type === 'number') {
    const na = typeof a === 'string' ? parseFloat(a) : Number(a);
    const nb = typeof b === 'string' ? parseFloat(b) : Number(b);
    return !Number.isNaN(na) && !Number.isNaN(nb) && na === nb;
  }
  if (type === 'array') {
    const aa = Array.isArray(a) ? a : [];
    const ab = Array.isArray(b) ? b : [];
    return aa.length === ab.length && aa.every((v, i) => v === ab[i]);
  }
  return a === b;
}

function coerceForPut(value: unknown, type: ConfigFieldType): unknown {
  if (type === 'integer') {
    const n = typeof value === 'string' ? parseInt(value, 10) : Number(value);
    return Number.isNaN(n) ? value : Math.trunc(n);
  }
  if (type === 'number') {
    const n = typeof value === 'string' ? parseFloat(value) : Number(value);
    return Number.isNaN(n) ? value : n;
  }
  return value;
}

// Whether a per-setting reset button should be disabled + why. Pulled out of
// SettingRow so the schema-unavailable / no-default / at-default decision
// tree (the fallback when the schema is unavailable) is independently testable.
function computeResetState(
  fieldSchema: ConfigFieldSchema | undefined,
  schemaAvailable: boolean,
  value: unknown,
  widgetType: ConfigFieldType,
): { disabled: boolean; title: string } {
  if (!schemaAvailable) {
    return { disabled: true, title: 'defaults unavailable — schema fetch failed' };
  }
  if (!fieldSchema) {
    return { disabled: true, title: 'no default known for this field' };
  }
  if (valuesEqual(value, fieldSchema.default, widgetType)) {
    return { disabled: true, title: `already at default: ${formatVal(fieldSchema.default)}` };
  }
  return { disabled: false, title: `reset to default: ${formatVal(fieldSchema.default)}` };
}

// Only the dirty fields, coerced to the type their schema (or the original
// server value, if the schema is unavailable) says they should be. Mirrors
// the shallow-merge PUT contract at config_put.py:54 — never a full section.
function buildSavePayload(
  dirty: Record<string, unknown>,
  sectionSchema: ConfigSectionSchema | undefined,
  sectionData: Record<string, unknown>,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {};
  for (const [field, value] of Object.entries(dirty)) {
    const fieldSchema = sectionSchema?.[field];
    const widgetType = resolveWidgetType(fieldSchema, sectionData[field]);
    payload[field] = coerceForPut(value, widgetType);
  }
  return payload;
}

function SettingRow({
  field,
  value,
  serverValue,
  dirty,
  fieldSchema,
  schemaAvailable,
  onChange,
  onReset,
}: {
  field: string;
  value: unknown;
  serverValue: unknown;
  dirty: boolean;
  fieldSchema: ConfigFieldSchema | undefined;
  schemaAvailable: boolean;
  onChange: (value: unknown) => void;
  onReset: () => void;
}) {
  const widgetType = resolveWidgetType(fieldSchema, serverValue);
  const { disabled: resetDisabled, title: resetTitle } = computeResetState(
    fieldSchema,
    schemaAvailable,
    value,
    widgetType,
  );

  let widget: React.ReactNode;
  switch (widgetType) {
    case 'integer':
    case 'number':
      widget = (
        <input
          type="number"
          step={widgetType === 'integer' ? 1 : 0.1}
          min={schemaAvailable ? fieldSchema?.minimum : undefined}
          max={schemaAvailable ? fieldSchema?.maximum : undefined}
          value={String(value ?? '')}
          onChange={(e) => onChange(e.target.value)}
          className={inputBase}
        />
      );
      break;
    case 'boolean':
      widget = (
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 accent-emerald-500"
        />
      );
      break;
    case 'array': {
      const lines = Array.isArray(value) ? (value as string[]) : [];
      widget = (
        <textarea
          value={lines.join('\n')}
          onChange={(e) =>
            onChange(
              e.target.value
                .split('\n')
                .map((s) => s.trim())
                .filter(Boolean),
            )
          }
          rows={Math.max(3, lines.length)}
          spellCheck={false}
          className={`${inputBase} font-mono`}
        />
      );
      break;
    }
    case 'string':
    default:
      widget = (
        <input
          type="text"
          value={String(value ?? '')}
          onChange={(e) => onChange(e.target.value)}
          className={inputBase}
        />
      );
      break;
  }

  return (
    <div
      // No per-row border here: `last:border-b-0`
      // only targets the DOM-last child, which in a 2-up grid is bottom-right,
      // not "bottom of every column" (see left-column dangling-border case
      // this avoided). Row separation comes from the parent grid's gap-y
      // instead. Also, the inner 3-col grid is NEVER active while the
      // outer grid is 2-up (`lg:grid-cols-1` and no wider override) —
      // re-promoting to 3-col at `xl` still
      // collapsed hf_host_allowlist to 63px there, because widening the
      // reset button ate into the same `1fr` the pixel math had
      // only budgeted for the hint. Once stacked, the widget is on its own
      // grid row with nothing to share space with, so it renders at exactly
      // `max-w-sm` (384px) at every width from `lg` up — a hard cap, not an
      // estimate — so no field, no matter how long its label/default/value,
      // can ever squeeze it again. Below `lg` the outer is still 1 column
      // (full page width), so 3-col here is unchanged from the earlier single-column baseline.
      className={`grid grid-cols-[minmax(160px,220px)_1fr_auto] lg:grid-cols-1 gap-x-3 gap-y-1 items-start py-2 ${
        dirty ? 'border-l-2 border-l-emerald-600 pl-2 -ml-2' : ''
      }`}
    >
      <div className="flex flex-col pt-2">
        <span className="text-sm font-mono text-slate-300">{field}</span>
        {dirty && <span className="text-[10px] text-slate-500">was: {formatVal(serverValue)}</span>}
      </div>
      <div>{widget}</div>
      <div className="flex items-center gap-2 justify-end pt-1">
        <button
          onClick={onReset}
          disabled={resetDisabled}
          title={resetTitle}
          className="px-3 py-1.5 rounded bg-slate-800 text-slate-300 text-xs hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed whitespace-nowrap"
        >
          ↻ Reset
        </button>
        {/* truncate, never whitespace-nowrap: this cell is the grid's `auto`
            column, so an un-truncated long default (pull.hf_host_allowlist is 5
            comma-joined hostnames) expands it and squeezes the `1fr` widget
            column down to an unusable sliver. Full value stays on hover. */}
        <span
          className="text-xs text-slate-500 font-mono truncate max-w-[200px]"
          title={fieldSchema ? `default: ${formatVal(fieldSchema.default)}` : 'default unknown'}
        >
          {fieldSchema ? `default: ${formatVal(fieldSchema.default)}` : 'default: —'}
        </span>
      </div>
    </div>
  );
}

function SectionEditor({
  title,
  section,
  cfg,
  schema,
  schemaAvailable,
  edits,
  onFieldChange,
  onFieldReset,
  onSave,
  onResetSection,
  saving,
  status,
}: {
  title: string;
  section: SectionKey;
  cfg: ConfigShape;
  schema: ConfigSchema | null;
  schemaAvailable: boolean;
  edits: Record<string, unknown> | undefined;
  onFieldChange: (field: string, value: unknown) => void;
  onFieldReset: (field: string, fieldSchema: ConfigFieldSchema) => void;
  onSave: () => void;
  onResetSection: () => void;
  saving: boolean;
  status: SectionStatus | undefined;
}) {
  const sectionData = cfg[section];
  const sectionSchema = schema?.[section];
  const dirtyCount = edits ? Object.keys(edits).length : 0;
  const hasKnownDefaults = schemaAvailable && !!sectionSchema && Object.keys(sectionSchema).length > 0;
  const fields = renderableFields(section, sectionData);

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-slate-200">{title}</h3>
        <div className="flex gap-2">
          <button
            onClick={onResetSection}
            disabled={!hasKnownDefaults}
            title={
              hasKnownDefaults
                ? 'Stage every field in this section back to its default (still requires Save)'
                : 'defaults unavailable — schema fetch failed'
            }
            className="px-3 py-1.5 rounded-md bg-slate-800 text-slate-300 text-xs hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Reset section to defaults
          </button>
          <button
            onClick={onSave}
            disabled={dirtyCount === 0 || saving}
            className="px-4 py-1.5 rounded-md bg-emerald-700 text-white text-xs font-medium hover:bg-emerald-600 disabled:bg-slate-700 disabled:text-slate-400"
          >
            {saving ? 'Saving…' : `Save changes (${dirtyCount})`}
          </button>
        </div>
      </div>
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 grid grid-cols-1 lg:grid-cols-2 gap-x-6 gap-y-2">
        {fields.map((field) => (
          <SettingRow
            key={field}
            field={field}
            value={edits?.[field] !== undefined ? edits[field] : sectionData[field]}
            serverValue={sectionData[field]}
            dirty={edits?.[field] !== undefined}
            fieldSchema={sectionSchema?.[field]}
            schemaAvailable={schemaAvailable}
            onChange={(v) => onFieldChange(field, v)}
            onReset={() => {
              const fs = sectionSchema?.[field];
              if (fs) onFieldReset(field, fs);
            }}
          />
        ))}
      </div>
      {status && (
        <div className={`mt-2 text-sm ${status.type === 'success' ? 'text-emerald-400' : 'text-rose-400'}`}>
          {status.text}
        </div>
      )}
    </div>
  );
}

function BootRow({ k, v }: { k: string; v: unknown }) {
  const display = typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v);
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-slate-400">{k}</span>
      <span className="font-mono text-slate-200 text-right truncate" title={display}>
        {display}
      </span>
    </div>
  );
}

function BootSection({ title, data }: { title: string; data: Record<string, unknown> }) {
  return (
    <div className="mb-4 last:mb-0 space-y-1.5">
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-1">{title}</div>
      {Object.entries(data).map(([k, v]) => (
        <BootRow key={k} k={k} v={v} />
      ))}
    </div>
  );
}

export default function Config() {
  const [cfg, setCfg] = useState<ConfigShape | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [schema, setSchema] = useState<ConfigSchema | null>(null);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [edits, setEdits] = useState<EditsMap>({});
  const [savingSection, setSavingSection] = useState<SectionKey | null>(null);
  const [sectionStatus, setSectionStatus] = useState<Partial<Record<SectionKey, SectionStatus>>>({});

  const refresh = useCallback(async () => {
    try {
      const raw = await getConfig();
      setCfg(raw as unknown as ConfigShape);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    getConfigSchema()
      .then((s) => setSchema(s))
      .catch((e) => setSchemaError(e instanceof Error ? e.message : String(e)));
  }, []);

  const reloadAll = useCallback(() => {
    setEdits({});
    setSectionStatus({});
    void refresh();
  }, [refresh]);

  // A staged value that matches what the server already has is NOT a change, and
  // must not be counted as one: editing 30 -> 77 -> back to 30 (by hand or via the
  // reset button) would otherwise leave "Save changes (1)" while nothing differs
  // from the server, and Save would PUT a no-op. Drop the field from `edits` in
  // that case. Compared against the SERVER value, not the default -- resetting a
  // field whose server value is genuinely off-default must still stage as dirty,
  // which is the whole point of the reset button.
  const setFieldValue = useCallback(
    (section: SectionKey, field: string, value: unknown) => {
      const serverValue = cfg?.[section]?.[field];
      const widgetType = resolveWidgetType(schema?.[section]?.[field], serverValue);
      const matchesServer = valuesEqual(value, serverValue, widgetType);
      setEdits((prev) => {
        const sectionEdits = { ...(prev[section] ?? {}) };
        if (matchesServer) {
          delete sectionEdits[field];
        } else {
          sectionEdits[field] = value;
        }
        return { ...prev, [section]: sectionEdits };
      });
    },
    [cfg, schema],
  );

  const resetField = useCallback(
    (section: SectionKey, field: string, fieldSchema: ConfigFieldSchema) => {
      setFieldValue(section, field, defaultAsEditValue(fieldSchema.type, fieldSchema.default));
    },
    [setFieldValue],
  );

  const resetSectionToDefaults = useCallback(
    (section: SectionKey) => {
      if (!cfg || !schema?.[section]) return;
      const sectionData = cfg[section];
      const sectionSchema = schema[section];
      const next: Record<string, unknown> = {};
      for (const field of renderableFields(section, sectionData)) {
        const fieldSchema = sectionSchema[field];
        if (!fieldSchema) continue;
        const current = edits[section]?.[field] ?? sectionData[field];
        if (!valuesEqual(current, fieldSchema.default, fieldSchema.type)) {
          next[field] = defaultAsEditValue(fieldSchema.type, fieldSchema.default);
        }
      }
      if (Object.keys(next).length === 0) return;
      setEdits((prev) => ({ ...prev, [section]: { ...(prev[section] ?? {}), ...next } }));
    },
    [cfg, schema, edits],
  );

  const saveSection = useCallback(
    async (section: SectionKey) => {
      if (!cfg) return;
      const dirty = edits[section];
      if (!dirty || Object.keys(dirty).length === 0) return;
      const sectionSchema = schema?.[section];
      const payload = buildSavePayload(dirty, sectionSchema, cfg[section]);
      setSavingSection(section);
      setSectionStatus((prev) => ({ ...prev, [section]: undefined }));
      try {
        await putConfig({ [section]: payload });
        setSectionStatus((prev) => ({
          ...prev,
          [section]: { type: 'success', text: `Saved ${Object.keys(payload).length} field(s).` },
        }));
        setEdits((prev) => ({ ...prev, [section]: {} }));
        await refresh();
      } catch (e) {
        setSectionStatus((prev) => ({
          ...prev,
          [section]: { type: 'error', text: e instanceof Error ? e.message : String(e) },
        }));
      } finally {
        setSavingSection(null);
      }
    },
    [cfg, edits, schema, refresh],
  );

  // The only call site that decides which sections actually render — see
  // deriveRenderableSections' own comment for why this isn't a hardcoded list.
  const sections = cfg ? deriveRenderableSections(Object.keys(cfg)) : [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Boot config (read-only)</h2>
        <div className="text-sm text-slate-500 mb-3">
          Boot fields are immutable at runtime — PUT /api/config returns HTTP 403 if you try to
          mutate them. They are baked in from /etc/turbohaul/turbohaul.yaml + env overrides.
        </div>
        {error && <div className="text-amber-400 text-sm mb-3">⚠ {error.message}</div>}
        {cfg && (
          <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
            <BootSection title="server" data={cfg.server} />
            <BootSection title="storage" data={cfg.storage} />
            <BootSection title="runtime (binary + port_base)" data={cfg.runtime} />
            <BootSection title="ui" data={cfg.ui} />
          </div>
        )}
      </div>

      <div>
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-xl font-semibold text-slate-200">Runtime config (editable)</h2>
          <button
            onClick={reloadAll}
            className="px-3 py-1.5 rounded-md bg-slate-800 text-slate-300 text-xs hover:bg-slate-700"
          >
            Reload (discard edits)
          </button>
        </div>
        <div className="text-sm text-slate-500 mb-3">
          Per-setting edit + reset-to-default. Save sends only the fields you changed, scoped to
          their section — matching the server&apos;s per-section merge.
        </div>
        {schemaError && (
          <div className="text-amber-400 text-sm mb-3 rounded-md border border-amber-900/50 bg-amber-950/30 px-3 py-2">
            ⚠ Per-setting defaults are unavailable (GET /api/config/schema failed: {schemaError}).
            Values below are still editable; reset-to-default is disabled and min/max bounds are
            not enforced client-side until this recovers. The server remains the bounds authority
            regardless.
          </div>
        )}
        {cfg && (
          <div className="space-y-6">
            {sections.map((section) => (
              <SectionEditor
                key={section}
                title={section}
                section={section}
                cfg={cfg}
                schema={schema}
                schemaAvailable={!schemaError}
                edits={edits[section]}
                onFieldChange={(field, value) => setFieldValue(section, field, value)}
                onFieldReset={(field, fieldSchema) => resetField(section, field, fieldSchema)}
                onSave={() => void saveSection(section)}
                onResetSection={() => resetSectionToDefaults(section)}
                saving={savingSection === section}
                status={sectionStatus[section]}
              />
            ))}
          </div>
        )}
      </div>

      {cfg && sections.length > 0 && (
        <details className="rounded-lg border border-slate-700 bg-slate-950 p-3">
          <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-400 select-none">
            Raw JSON ({sections.join(', ')})
          </summary>
          <pre className="mt-3 text-xs font-mono text-slate-400 overflow-x-auto">
            {JSON.stringify(Object.fromEntries(sections.map((s) => [s, cfg[s]])), null, 2)}
          </pre>
        </details>
      )}

      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-400">
        <strong className="text-slate-300">Per-model manifest editor:</strong> deferred to a
        later release. /api/manifests CRUD + ETag/If-Match concurrency are wired on the BE;
        a model-picker UI that consumes them is planned.
      </div>
    </div>
  );
}
