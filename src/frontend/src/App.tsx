import { Routes, Route, Link, useLocation } from 'react-router-dom';
import type { ReactNode } from 'react';

function Layout({ children }: { children: ReactNode }) {
  const loc = useLocation();
  const tab = (path: string, label: string) => (
    <Link
      to={path}
      className={
        loc.pathname === path
          ? 'px-4 py-2 rounded-md text-sm font-medium bg-slate-700 text-white'
          : 'px-4 py-2 rounded-md text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800'
      }
    >
      {label}
    </Link>
  );
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-700 bg-slate-950 px-6 py-3">
        <div className="flex items-center justify-between">
          <h1 className="text-lg font-bold tracking-tight">Turbohaul Manager</h1>
          <nav className="flex gap-2">
            {tab('/', 'Dashboard')}
            {tab('/queue', 'Queue')}
            {tab('/blob', 'Blob')}
            {tab('/config', 'Config')}
            {tab('/logs', 'Logs')}
            {tab('/settings', 'Settings')}
          </nav>
        </div>
      </header>
      <main className="flex-1 px-6 py-6">{children}</main>
    </div>
  );
}

function Stub({ name }: { name: string }) {
  return (
    <div className="text-slate-400">
      <div className="text-xl font-semibold text-slate-300">{name}</div>
      <div className="mt-2 text-sm">
        Wave 17-20 populates this view (data-bound to /status + /api/tags + /ws/state).
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Stub name="Dashboard" />} />
        <Route path="/queue" element={<Stub name="Queue" />} />
        <Route path="/blob" element={<Stub name="Blob" />} />
        <Route path="/config" element={<Stub name="Config" />} />
        <Route path="/logs" element={<Stub name="Logs" />} />
        <Route path="/settings" element={<Stub name="Settings" />} />
      </Routes>
    </Layout>
  );
}
