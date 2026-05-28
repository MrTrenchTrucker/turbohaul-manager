import { Routes, Route, Link, useLocation } from 'react-router-dom';
import type { ReactNode } from 'react';
import Dashboard from './components/Dashboard';
import Queue from './components/Queue';
import Blob from './components/Blob';
import Config from './components/Config';
import Logs from './components/Logs';
import Settings from './components/Settings';
import Models from './components/Models';
// Schema tab — author + preflight `response_format`
// envelopes for /v1/chat/completions and /api/chat. Placed adjacent to
// /config.
import Schema from './components/Schema';

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
          <div className="flex items-baseline gap-2">
            <h1 className="text-lg font-bold tracking-tight">Turbohaul Manager</h1>
            <span className="text-xs font-semibold text-emerald-400" title="TurboQuant KV-cache quantization + MTP speculative decoding">TurboQuant + MTP</span>
          </div>
          <nav className="flex gap-2">
            {tab('/', 'Dashboard')}
            {tab('/models', 'Models')}
            {tab('/queue', 'Queue')}
            {tab('/blob', 'Blob')}
            {tab('/config', 'Config')}
            {tab('/schema', 'Schema')}
            {tab('/logs', 'Logs')}
            {tab('/settings', 'Settings')}
          </nav>
        </div>
      </header>
      <main className="flex-1 px-6 py-6">{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/models" element={<Models />} />
        <Route path="/queue" element={<Queue />} />
        <Route path="/blob" element={<Blob />} />
        <Route path="/config" element={<Config />} />
        <Route path="/schema" element={<Schema />} />
        <Route path="/logs" element={<Logs />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  );
}
