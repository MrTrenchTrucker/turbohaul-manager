import { Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
import type { ReactNode } from 'react';
import Dashboard from './components/Dashboard';
import Queue from './components/Queue';
import Blob from './components/Blob';
import Plugins from './components/Plugins';
import Settings from './components/Settings';

function Layout({ children }: { children: ReactNode }) {
  const loc = useLocation();
  const tab = (path: string, label: string) => {
    // prefix match (not just exact) so Settings stays highlighted while on
    // any of its sub-tab routes, e.g. /settings/config
    const active = loc.pathname === path || loc.pathname.startsWith(`${path}/`);
    // min-h-11 = 44px, the Apple HIG touch-target floor. py-2 alone rendered
    // 36px, which is a mis-tap risk on a phone.
    const base =
      'inline-flex items-center min-h-11 px-4 rounded-md text-sm font-medium';
    return (
      <Link
        to={path}
        aria-current={active ? 'page' : undefined}
        className={
          active
            ? `${base} bg-slate-700 text-white`
            : `${base} text-slate-400 hover:text-white hover:bg-slate-800`
        }
      >
        {label}
      </Link>
    );
  };
  return (
    <div className="min-h-screen flex flex-col">
      {/* The header stacks below `sm` so the title and the four nav links do
          not compete for one 390px row — that combination overflowed the
          viewport by ~93px and was the only source of horizontal scroll on
          the whole app. `flex-wrap` on the nav is the belt-and-braces: even
          if a longer label is added later, it wraps instead of overflowing. */}
      <header className="border-b border-slate-700 bg-slate-950 px-4 sm:px-6 py-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h1 className="text-lg font-bold tracking-tight">Turbohaul Manager</h1>
          <nav className="flex flex-wrap gap-2">
            {tab('/', 'Dashboard')}
            {tab('/queue', 'Queue')}
            {tab('/blob', 'Blob')}
            {tab('/plugins', 'Plugins (WIP)')}
            {tab('/settings', 'Settings')}
          </nav>
        </div>
      </header>
      <main className="flex-1 px-4 sm:px-6 py-6">{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/queue/*" element={<Queue />} />
        <Route path="/blob/*" element={<Blob />} />
        <Route path="/plugins" element={<Plugins />} />
        {/* legacy top-level routes now live under /settings/* or /blob/*;
            kept as redirects (not deleted) so existing bookmarks still resolve */}
        <Route path="/models" element={<Navigate to="/blob/models" replace />} />
        <Route path="/config" element={<Navigate to="/settings/config" replace />} />
        <Route path="/schema" element={<Navigate to="/settings/schema" replace />} />
        <Route path="/logs" element={<Navigate to="/settings/logs" replace />} />
        <Route path="/settings/*" element={<Settings />} />
      </Routes>
    </Layout>
  );
}
