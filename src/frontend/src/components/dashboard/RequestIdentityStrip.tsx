import type { RequestIdentity } from '../../api';

// Proof surface: compact strip showing
// the last request's structured identity — proof Turbohaul reads + trusts
// the structured client_meta instead of guessing off the thread_id prefix.
// Null-safe: renders a muted placeholder when no request has landed yet.
//
// This strip is a REQUEST RECORD (the manager's
// LAST-ADMITTED request), NOT the resident's current identity. It is labeled
// "last request" so it can never read as the resident's role, and when the
// request hit a DIFFERENT model than the resident card it sits on (e.g. a
// sub-agent's curator on a different model while main is resident), it renders in amber
// with a "≠ resident" chip instead of contradicting the card's own title.
// The role is disambiguated by model: main+curator SHARE the resident model
// (curator reuses main's cache), so a curator on a different model is a
// SUB-AGENT's curator by construction. Display-only; no backend changes.
//
// Exported so Dashboard.test.tsx can render this component in
// isolation. No other change: still called exactly as before at its one
// call site above; unexported behavior is otherwise identical.
export function RequestIdentityStrip({
  identity,
  residentModelTag,
  heading = 'last request',
  testId = 'resident-request-identity',
}: {
  identity: RequestIdentity | null | undefined;
  residentModelTag?: string;
  // NOT cosmetic -- the render below renders this literal
  // string, and without a distinguishing heading a second instance
  // (current_request_identity) would ALSO render "last request", making
  // the two rows indistinguishable by anything but position (which a
  // future layout change breaks). Defaults to the existing text so every
  // pre-existing call site's "last" row is unchanged without editing it.
  heading?: string;
  // A SECOND instance per card (current_request_identity)
  // must NOT share the "last" instance's data-testid -- ResidentsPanel.test.tsx's
  // NO-FLIP test counts exactly one `resident-request-identity` hook per
  // card via `matchAll`; two same-testid strips per card would silently
  // double that count. Defaults to the existing value so every pre-existing
  // call site is unchanged without editing it.
  testId?: string;
}) {
  if (!identity) {
    // Rendered UNCONDITIONALLY, same convention as resident-tok-s (ResidentCard.tsx):
    // a missing element is ambiguous between "correctly excluded" and "a bug
    // dropped it". data-value="" is the no-request state and is distinguishable
    // from the null a test helper returns when the element is absent entirely.
    //
    // The heading renders here too, not just in the
    // populated branch below -- without it, an idle "current request" row
    // and an empty "last request" row would render IDENTICAL placeholder
    // text with no visible label distinguishing them (caught by the
    // component's own test). The two rows mean different things when empty: "last"
    // empty means nothing has EVER landed; "current" empty means idle
    // right now (and something may well have run before). Keying the
    // placeholder text off `heading` rather than adding a third prop keeps
    // this a one-prop change; any heading other than the two shipped
    // values falls back to the original, heading-agnostic wording.
    return (
      <div
        data-testid={testId}
        data-value=""
        className="px-1 py-1 mb-2 border-b border-slate-800"
      >
        <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-0.5">
          {heading}
        </div>
        <div className="text-xs text-slate-600 italic">
          {heading === 'current request' ? '— idle —' : '— no request yet —'}
        </div>
      </div>
    );
  }
  const isDifferentModel =
    !!residentModelTag && identity.model_tag !== residentModelTag;
  // Derive the single role label from the is_* booleans (priority: curator >
  // compression > sub_agent > main; fall back to resolved_class). Curator is
  // qualified by model: a curator whose model differs from this resident card
  // CANNOT be main's curator (main+curator share a model) — it is a
  // sub-agent's curator (sub-agent+curator use a different model on purpose,
  // by architecture).
  const role = identity.is_curator
    ? isDifferentModel
      ? 'Curator (sub-agent)'
      : 'Curator'
    : identity.is_compression
      ? 'Compression'
      : identity.is_sub_agent
        ? 'Sub-Agent'
        : identity.is_main
          ? 'Main'
          : (identity.resolved_class ?? '—');
  // Role color distinct from the green ACTIVE pill: Main=emerald, Curator=
  // violet, Compression=amber, Sub-Agent=sky, unknown=slate. A green curator
  // read as an active serve.
  const roleColor = identity.is_curator
    ? 'text-violet-400'
    : identity.is_compression
      ? 'text-amber-400'
      : identity.is_sub_agent
        ? 'text-sky-400'
        : identity.is_main
          ? 'text-emerald-400'
          : 'text-slate-400';
  return (
    <div
      data-testid={testId}
      data-value={identity.model_tag ?? ''}
      className={`px-1 py-1 mb-2 border-b ${isDifferentModel ? 'border-amber-900/60' : 'border-slate-800'}`}
    >
      <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-0.5">
        {heading}
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs font-mono text-slate-400">
        {isDifferentModel && (
          <span
            className="px-1 rounded bg-amber-950/60 text-amber-400"
            title="This request ran on a different model than the resident above — it is NOT this resident's identity"
          >
            ≠ resident
          </span>
        )}
        <span>{identity.ip ?? '—'}</span>
        {identity.label ? (
          <>
            <span className="text-slate-600">·</span>
            <span className="text-cyan-400">{identity.label}</span>
          </>
        ) : null}
        <span className="text-slate-600">·</span>
        <span>{identity.model_tag ?? '—'}</span>
        <span className="text-slate-600">·</span>
        <span className={roleColor}>{role}</span>
        <span className="text-slate-600">·</span>
        <span className="truncate max-w-[10rem]">{identity.session_id ?? '—'}</span>
      </div>
    </div>
  );
}
