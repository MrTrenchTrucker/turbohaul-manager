import { useEffect, useState } from 'react';
import type { VersionInfo } from '../api';
import { getVersion } from '../api';

export default function Settings() {
  const [ver, setVer] = useState<VersionInfo | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;
    getVersion()
      .then((v) => {
        if (!cancelled) setVer(v);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e : new Error(String(e)));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">About</h2>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 space-y-3 text-sm">
          {error && <div className="text-amber-400">⚠ {error.message}</div>}
          {ver ? (
            <>
              <Row k="version" v={ver.version} />
              <Row k="backend" v={ver.backend} />
              <Row k="backend SHA pinned" v={String(ver.backend_sha_pinned)} />
              <Row k="api compat" v={ver.api_compat} />
              <Row k="user-agent" v={ver.user_agent} />
            </>
          ) : (
            <div className="text-slate-500 italic">Loading…</div>
          )}
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Licenses + attribution</h2>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-400 space-y-2">
          <p>
            Turbohaul-Manager v0.2 — MIT-licensed wrapper around the inference engine.
          </p>
          <p>
            Inference backend: <span className="font-mono text-slate-300">llama-server</span>{' '}
            built from Tom&apos;s TurboQuant fork of llama.cpp (MIT).
          </p>
          <p>
            See <span className="font-mono text-slate-300">THIRD_PARTY_LICENSES</span> in the
            container at <span className="font-mono text-slate-300">/usr/share/doc/turbohaul/</span>.
          </p>
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-slate-400">{k}</span>
      <span className="font-mono text-slate-200 text-right truncate">{v}</span>
    </div>
  );
}
