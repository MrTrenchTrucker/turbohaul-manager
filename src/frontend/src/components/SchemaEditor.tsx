import { useEffect, useMemo, useRef, useState } from 'react';
import {
  EditorMode,
  RejectionReason,
  ResponseFormat,
  SCHEMA_DEPTH_MAX,
  SCHEMA_PROPERTY_COUNT_MAX,
  SCHEMA_SIZE_MAX_BYTES,
  validateInnerSchema,
} from '../lib/responseFormatValidator';

/**
 * Schema editor for the chat-completion `response_format` envelope.
 *
 * Lets the user choose one of three modes — `text`, `json_object`, `json_schema` —
 * and (in `json_schema` mode) author the inner schema in a plain textarea. The
 * 10 BE rejection reasons are mirrored as client-side preflight checks, with the
 * server's `detail.message` always shown verbatim when a 422/400 comes back.
 *
 * Design notes:
 * - Plain `<textarea>` per HARD CONSTRAINT (no Monaco / no codemirror / no
 *   react-jsonschema-form). Monospace + line-height tweaked to make pasted
 *   JSON readable; that's the extent of editor sugar.
 * - The component is uncontrolled w.r.t. the parent: caller passes in an
 *   initial `value` and an `onChange` callback receiving the constructed
 *   `ResponseFormat`. Parent can decide whether to fire chat-completions
 *   against the schema directly or just hold the envelope.
 * - Server-error display: caller can pass `serverError` (the parsed `detail`
 *   from the 422/400 body); we render the message verbatim with a friendly
 *   headline keyed off `detail.error`. We never invent reason text.
 */

interface ServerErrorDetail {
  error?: string;
  message?: string;
  received_type?: string;
}

interface SchemaEditorProps {
  /** Optional initial envelope; defaults to `text`. */
  initial?: ResponseFormat;
  /** Fires whenever the constructed envelope or its validity changes. */
  onChange: (rf: ResponseFormat | null, valid: boolean) => void;
  /** When the parent receives a 400/422 from the server, hand the `detail` here. */
  serverError?: ServerErrorDetail | null;
}

const DEFAULT_NAME = 'response';
const STARTER_SCHEMA = `{
  "type": "object",
  "properties": {
    "answer": { "type": "string" }
  },
  "required": ["answer"],
  "additionalProperties": false
}`;

const FRIENDLY_HEADLINES: Record<RejectionReason, string> = {
  jsonschema_lib_unavailable:
    'Server-side schema validator is temporarily unavailable',
  missing_or_malformed_json_schema_field:
    'The `json_schema` envelope is missing or not an object',
  missing_or_malformed_schema_field:
    'The inner `schema` field is missing or not an object',
  schema_not_json_serializable:
    "Schema can't be serialized to JSON (cycles or non-JSON values?)",
  schema_size_exceeded: `Schema is too large — max ${SCHEMA_SIZE_MAX_BYTES} bytes when serialized`,
  schema_depth_exceeded: `Schema is too deep — max nesting level is ${SCHEMA_DEPTH_MAX}`,
  schema_property_count_exceeded: `Too many total properties — max ${SCHEMA_PROPERTY_COUNT_MAX}`,
  schema_contains_ref_unsupported:
    "Schema uses `$ref` — inline the referenced subschema instead",
  schema_missing_additionalProperties_guard:
    'Every `type: "object"` subschema must set `additionalProperties: false`',
  schema_compile_failed: 'Server failed to compile the schema',
};

/** Map a server-returned `error` code to a friendly headline. */
function friendlyHeadlineFor(error?: string): string | null {
  if (!error) return null;
  // Server emits suffixed codes like `schema_size_exceeded:65540` and
  // `schema_compile_failed:ValueError`. Strip the suffix for lookup so the
  // headline matches regardless of the dynamic payload.
  const base = error.split(':', 1)[0] as RejectionReason;
  return FRIENDLY_HEADLINES[base] ?? null;
}

export default function SchemaEditor({ initial, onChange, serverError }: SchemaEditorProps) {
  const initialMode: EditorMode = initial && initial !== null ? initial.type : 'text';
  const initialName =
    initial && initial.type === 'json_schema' ? initial.json_schema.name : DEFAULT_NAME;
  const initialSchemaText =
    initial && initial.type === 'json_schema'
      ? JSON.stringify(initial.json_schema.schema, null, 2)
      : STARTER_SCHEMA;

  const [mode, setMode] = useState<EditorMode>(initialMode);
  const [name, setName] = useState<string>(initialName);
  const [schemaText, setSchemaText] = useState<string>(initialSchemaText);

  // Parse + preflight every render (cheap; schema-text rarely > a few KB).
  // useMemo so re-renders that don't touch schemaText don't re-run the walks.
  const parsed = useMemo<
    | { kind: 'parsed'; schema: Record<string, unknown> }
    | { kind: 'parse-error'; message: string }
  >(() => {
    if (mode !== 'json_schema') {
      // mode doesn't use schemaText; return a sentinel parsed object so the
      // memo stays simple. Caller branches on mode before reading parsed.
      return { kind: 'parsed', schema: {} };
    }
    try {
      const v = JSON.parse(schemaText) as unknown;
      if (typeof v !== 'object' || v === null || Array.isArray(v)) {
        return {
          kind: 'parse-error',
          message: 'Schema must be a JSON object (got array or non-object).',
        };
      }
      return { kind: 'parsed', schema: v as Record<string, unknown> };
    } catch (e) {
      return {
        kind: 'parse-error',
        message: `JSON parse error: ${e instanceof Error ? e.message : String(e)}`,
      };
    }
  }, [mode, schemaText]);

  // Run the BE-mirror preflight (rules #4–#9) when in json_schema mode and JSON parsed cleanly.
  const preflight = useMemo(() => {
    if (mode !== 'json_schema') return null;
    if (parsed.kind !== 'parsed') return null;
    return validateInnerSchema(parsed.schema);
  }, [mode, parsed]);

  // Build the envelope + signal the parent. We push the latest envelope and
  // a validity boolean so the parent can disable a submit button until valid.
  const envelope = useMemo<ResponseFormat | null>(() => {
    if (mode === 'text') return { type: 'text' };
    if (mode === 'json_object') return { type: 'json_object' };
    if (parsed.kind !== 'parsed') return null;
    if (!name.trim()) return null;
    return {
      type: 'json_schema',
      json_schema: { name: name.trim(), schema: parsed.schema },
    };
  }, [mode, name, parsed]);

  const valid =
    mode !== 'json_schema'
      ? true
      : parsed.kind === 'parsed' && (preflight?.ok ?? false) && name.trim().length > 0;

  // Fire onChange whenever envelope or validity changes.
  // Using useEffect would also work; here we call directly in render-derived
  // useMemo via a separate effect to avoid double-fire. React 18 strict-mode
  // double-mount is fine because the parent's setter is idempotent.
  // (We use a small useEffect to keep render pure.)
  useChangeNotifier(envelope, valid, onChange);

  // ---- Render ----

  const preflightBanner = (() => {
    if (mode !== 'json_schema') return null;
    if (parsed.kind === 'parse-error') {
      return (
        <Banner tone="error" headline="JSON syntax error" detail={parsed.message} />
      );
    }
    if (preflight && !preflight.ok && preflight.reason) {
      const head = FRIENDLY_HEADLINES[preflight.reason];
      return (
        <Banner
          tone="error"
          headline={head}
          detail={preflight.detail ? `(${preflight.detail})` : undefined}
        />
      );
    }
    if (valid) {
      return (
        <Banner
          tone="ok"
          headline="Preflight passed — schema looks valid client-side"
          detail="Server-side validator will run on submit (truth-source)."
        />
      );
    }
    return null;
  })();

  const serverErrorBanner = (() => {
    if (!serverError) return null;
    const head = friendlyHeadlineFor(serverError.error) ?? 'Server rejected the request';
    // Server's detail.message is the truth-source — render verbatim, never
    // synthesize. The friendly headline is a UI affordance only.
    return (
      <Banner
        tone="error"
        headline={head}
        detail={serverError.message ?? `(error code: ${serverError.error ?? 'unknown'})`}
      />
    );
  })();

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 space-y-4">
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
            Response format
          </div>
          <div className="flex gap-2" role="radiogroup" aria-label="Response format mode">
            <ModeRadio current={mode} value="text" label="text" onSelect={setMode} />
            <ModeRadio
              current={mode}
              value="json_object"
              label="json_object"
              onSelect={setMode}
            />
            <ModeRadio
              current={mode}
              value="json_schema"
              label="json_schema"
              onSelect={setMode}
            />
          </div>
        </div>

        {mode === 'text' && (
          <p className="text-sm text-slate-500">
            Default text completion. The model returns a free-form string.
          </p>
        )}
        {mode === 'json_object' && (
          <p className="text-sm text-slate-500">
            Best-effort JSON object output. The server hints the model to return a
            JSON object; no schema validation runs. Use{' '}
            <code className="text-slate-300">json_schema</code> below if you need a
            specific shape.
          </p>
        )}
        {mode === 'json_schema' && (
          <div className="space-y-3">
            <div>
              <label
                htmlFor="schema-name"
                className="block text-xs uppercase tracking-wide text-slate-500 mb-1"
              >
                Name
              </label>
              <input
                id="schema-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="response"
                spellCheck={false}
                className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm font-mono text-slate-200 focus:outline-none focus:ring-1 focus:ring-emerald-600"
              />
              {!name.trim() && (
                <div className="mt-1 text-xs text-amber-400">
                  Name is required by the OpenAI envelope.
                </div>
              )}
            </div>
            <div>
              <label
                htmlFor="schema-body"
                className="block text-xs uppercase tracking-wide text-slate-500 mb-1"
              >
                Schema (JSON)
              </label>
              <textarea
                id="schema-body"
                value={schemaText}
                onChange={(e) => setSchemaText(e.target.value)}
                spellCheck={false}
                className="w-full h-80 rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm font-mono leading-relaxed text-slate-200 focus:outline-none focus:ring-1 focus:ring-emerald-600"
              />
            </div>
          </div>
        )}
      </div>

      {preflightBanner}
      {serverErrorBanner}

      <details className="rounded-lg border border-slate-800 bg-slate-950 p-3 text-sm">
        <summary className="cursor-pointer text-slate-400 hover:text-slate-200">
          Constructed response_format payload
        </summary>
        <pre className="mt-2 text-xs font-mono text-slate-400 bg-slate-900 rounded-md p-2 overflow-x-auto whitespace-pre-wrap break-all">
{envelope === null
  ? '(invalid — fix the schema to see the envelope)'
  : JSON.stringify(envelope, null, 2)}
        </pre>
      </details>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ModeRadio({
  current,
  value,
  label,
  onSelect,
}: {
  current: EditorMode;
  value: EditorMode;
  label: string;
  onSelect: (v: EditorMode) => void;
}) {
  const selected = current === value;
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      onClick={() => onSelect(value)}
      className={
        selected
          ? 'px-3 py-1.5 rounded-md text-sm font-mono bg-emerald-700 text-white'
          : 'px-3 py-1.5 rounded-md text-sm font-mono bg-slate-800 text-slate-300 hover:bg-slate-700'
      }
    >
      {label}
    </button>
  );
}

function Banner({
  tone,
  headline,
  detail,
}: {
  tone: 'ok' | 'error';
  headline: string;
  detail?: string;
}) {
  const cls =
    tone === 'ok'
      ? 'border-emerald-700/50 bg-emerald-950/30 text-emerald-300'
      : 'border-rose-700/50 bg-rose-950/30 text-rose-300';
  const glyph = tone === 'ok' ? '✓' : '⚠';
  return (
    <div className={`rounded-lg border px-3 py-2 text-sm ${cls}`}>
      <div className="font-medium">
        {glyph} {headline}
      </div>
      {detail && <div className="text-xs mt-1 opacity-80 break-words">{detail}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Effect helper — fires onChange when envelope/validity transitions.
// Kept separate from the main component so it stays scan-readable.
// ---------------------------------------------------------------------------

function useChangeNotifier(
  envelope: ResponseFormat | null,
  valid: boolean,
  onChange: (rf: ResponseFormat | null, valid: boolean) => void,
) {
  // Serialize for shallow change detection — the envelope object is rebuilt
  // every render via useMemo with new identity even when shape hasn't changed.
  const serialized = envelope === null ? '__null__' : JSON.stringify(envelope);
  const lastRef = useRef<{ s: string; v: boolean } | null>(null);
  useEffect(() => {
    const prev = lastRef.current;
    if (!prev || prev.s !== serialized || prev.v !== valid) {
      lastRef.current = { s: serialized, v: valid };
      onChange(envelope, valid);
    }
  }, [serialized, valid, envelope, onChange]);
}
