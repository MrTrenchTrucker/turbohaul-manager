import { Link, useLocation } from 'react-router-dom';
import type { ReactNode } from 'react';
import { useEffect, useState } from 'react';

// Generic sub-tab primitive — the first sub-tab pattern
// in this FE. Takes only {key,label,element}; owns active-tab styling + URL
// routing, nothing caller-specific. Each pane mounts lazily on first visit,
// then stays mounted with visibility toggled rather than unmounted, so
// switching tabs never destroys in-flight state a pane is holding (e.g.
// Config's unsaved edits, Schema's authored envelope).

export interface SubTabSpec {
  key: string;
  label: string;
  element: ReactNode;
}

// Pure: which tab key a URL resolves to. Only the first path segment past
// basePath matters (a stray deeper segment, or an unknown/absent one, both
// fall back to defaultKey) — never a partial match on tab identity.
export function resolveActiveKey(
  pathname: string,
  basePath: string,
  tabKeys: string[],
  defaultKey: string,
): string {
  const prefix = basePath.endsWith('/') ? basePath : `${basePath}/`;
  const rest = pathname.startsWith(prefix) ? pathname.slice(prefix.length) : '';
  const segment = rest.split('/')[0];
  return tabKeys.includes(segment) ? segment : defaultKey;
}

// Pure: the lazy-mount-then-keep-mounted contract. A key, once added, is
// NEVER removed by this function — that monotonic growth is what guarantees
// a pane's internal state survives navigating away and back. Idempotent
// (returns the same Set reference) when activeKey is already mounted, so it
// is safe to call unconditionally on every render/navigation.
export function nextMountedSet(mounted: Set<string>, activeKey: string): Set<string> {
  if (mounted.has(activeKey)) return mounted;
  return new Set(mounted).add(activeKey);
}

export function SubTabs({
  basePath,
  tabs,
  defaultKey,
}: {
  basePath: string;
  tabs: SubTabSpec[];
  defaultKey: string;
}) {
  const location = useLocation();
  const activeKey = resolveActiveKey(
    location.pathname,
    basePath,
    tabs.map((t) => t.key),
    defaultKey,
  );

  const [mounted, setMounted] = useState<Set<string>>(() => new Set([activeKey]));
  useEffect(() => {
    setMounted((prev) => nextMountedSet(prev, activeKey));
  }, [activeKey]);

  return (
    <div className="space-y-4">
      {/* flex-wrap so a tab set wider than the viewport wraps rather than
          forcing the page to scroll sideways; min-h-11 (44px) meets the
          touch-target floor — py-1.5 alone rendered 28px. */}
      <nav className="flex flex-wrap gap-1.5 border-b border-slate-800 pb-2">
        {tabs.map((t) => {
          const href = t.key === defaultKey ? basePath : `${basePath}/${t.key}`;
          const active = t.key === activeKey;
          const base =
            'inline-flex items-center min-h-11 px-3 rounded-md text-xs font-medium';
          return (
            <Link
              key={t.key}
              to={href}
              aria-current={active ? 'page' : undefined}
              className={
                active
                  ? `${base} bg-slate-700 text-white`
                  : `${base} text-slate-400 hover:text-white hover:bg-slate-800`
              }
            >
              {t.label}
            </Link>
          );
        })}
      </nav>
      {tabs.map((t) => (
        <div key={t.key} style={{ display: t.key === activeKey ? 'block' : 'none' }}>
          {mounted.has(t.key) ? t.element : null}
        </div>
      ))}
    </div>
  );
}
