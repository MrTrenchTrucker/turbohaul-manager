import type { ReactNode } from 'react';

// Deliberately a plain string-keyed record, not React.HTMLAttributes<...>:
// the only real payload here is data-* test hooks, which TypeScript's
// HTMLAttributes interface does not declare (excess-property-checked object
// literals like {'data-testid': 'x'} fail against it). A record is honest
// about what this is actually for.
type PassthroughProps = Record<string, unknown>;

/* ------------------------------------------------------------------ */
/*  Small UI atoms                                                      */
/* ------------------------------------------------------------------ */

export function Card({
  title,
  tone,
  children,
  rootProps,
  titleProps,
}: {
  title: string;
  tone: string;
  children: ReactNode;
  // Optional passthrough for test hooks (data-testid
  // etc.) on the root/title elements. Undefined for every caller that
  // doesn't pass them -- zero behaviour change. Deliberately generic
  // (HTMLAttributes, not a single-purpose testId prop) so this also covers
  // aria-* or other future needs without growing more one-off props. Kept
  // OFF the shared Card component's own default rendering path so a hook
  // never applies to every Card-shaped box on the page at once -- each
  // caller opts in per-instance.
  rootProps?: PassthroughProps;
  titleProps?: PassthroughProps;
}) {
  return (
    <div className={`rounded-lg border ${tone} bg-slate-950 p-3`} {...rootProps}>
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-2" {...titleProps}>
        {title}
      </div>
      {children}
    </div>
  );
}

export function KV({ k, v }: { k: string; v: ReactNode }) {
  // A label with NO value is indistinguishable from a value of zero.
  // Render an explicit em-dash when the value is missing/empty so the label
  // never silently renders blank next to it.
  const shown =
    v === null || v === undefined || v === '' ? (
      <span className="text-slate-600">—</span>
    ) : (
      v
    );
  return (
    <div className="flex items-baseline justify-between gap-3 text-sm py-0.5">
      <span className="text-slate-400">{k}</span>
      <span className="font-mono text-slate-200 truncate">{shown}</span>
    </div>
  );
}
