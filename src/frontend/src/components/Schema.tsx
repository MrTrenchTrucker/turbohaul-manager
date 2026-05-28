import { useCallback, useState } from 'react';
import SchemaEditor from './SchemaEditor';
import type { ResponseFormat } from '../lib/responseFormatValidator';

/**
 * Schema tab page — hosts `SchemaEditor` and surfaces the constructed
 * `response_format` envelope so users can copy it into other tools or
 * (in a future polish wave) fire a smoke chat-completion against the
 * sidecar directly.
 *
 * This is author-only (no live POST UI); the
 * `chatComplete` helper is intentionally deferred to a follow-on RC so
 * this  FE doesn't grow legs into the chat-execution surface.
 */
export default function Schema() {
  const [envelope, setEnvelope] = useState<ResponseFormat | null>({ type: 'text' });
  const [valid, setValid] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);

  const handleChange = useCallback((rf: ResponseFormat | null, ok: boolean) => {
    setEnvelope(rf);
    setValid(ok);
    setCopied(false);
  }, []);

  const handleCopy = useCallback(async () => {
    if (envelope === null) return;
    const text = JSON.stringify(envelope, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
    } catch {
      // Clipboard API unavailable (older browser / non-secure context).
      // Fall back to selectable preformatted text — already rendered below.
      setCopied(false);
    }
  }, [envelope]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-2">
          Response-format schema editor
        </h2>
        <p className="text-sm text-slate-500 max-w-3xl">
          Build a <code className="text-slate-300">response_format</code> envelope to
          pass to <code className="text-slate-300">POST /v1/chat/completions</code> or{' '}
          <code className="text-slate-300">POST /api/chat</code>. Choose a mode and,
          for <code className="text-slate-300">json_schema</code>, paste an inline
          JSON schema. Preflight checks mirror the server&apos;s validator so you
          catch obvious mistakes before submitting; the server is still the truth
          source on rejections.
        </p>
      </div>

      <SchemaEditor initial={{ type: 'text' }} onChange={handleChange} />

      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs uppercase tracking-wide text-slate-500">
            Copy envelope
          </div>
          <button
            type="button"
            onClick={() => void handleCopy()}
            disabled={!valid || envelope === null}
            className="px-3 py-1.5 rounded-md bg-emerald-700 text-white text-xs font-medium hover:bg-emerald-600 disabled:bg-slate-700 disabled:text-slate-500"
          >
            {copied ? 'Copied ✓' : 'Copy to clipboard'}
          </button>
        </div>
        <pre className="text-xs font-mono text-slate-400 bg-slate-900 rounded-md p-2 overflow-x-auto whitespace-pre-wrap break-all">
{envelope === null || !valid
  ? '(fix the schema preflight first; envelope unavailable)'
  : JSON.stringify(envelope, null, 2)}
        </pre>
        <p className="mt-2 text-xs text-slate-500">
          Live POST + retry-on-noncompliance behavior is deferred to a follow-on
          polish wave; this page is author-only today.
        </p>
      </div>
    </div>
  );
}
