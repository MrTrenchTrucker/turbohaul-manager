import { useStatus } from "../hooks/useStatus";

export default function Logs() {
  const { data, error } = useStatus();

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-slate-200">Logs</h2>
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-400 space-y-3">
        <p>
          A <span className="font-mono text-slate-300">/api/logs/{"{slot_port}"}</span> endpoint
          will be added in a polish wave to stream the active sidecar&apos;s stderr (Phase 5+). For
          now, view container logs directly:
        </p>
        <pre className="text-xs font-mono bg-slate-900 rounded-md p-3 text-slate-300 whitespace-pre-wrap">
          {`docker logs -f turbohaul-manager
# or for a specific sidecar PID:
journalctl --user-unit=turbohaul.service -f`}
        </pre>
        {error && (
          <div className="text-amber-400 text-xs">/status check failed: {error.message}</div>
        )}
        {data && data.active && (
          <div className="text-xs text-slate-400 mt-3">
            Active sidecar:{" "}
            <span className="font-mono text-slate-200">{data.active.model_tag}</span> on port{" "}
            <span className="font-mono text-slate-200">{data.active.port}</span> (pid{" "}
            <span className="font-mono text-slate-200">{data.active.pid}</span>)
          </div>
        )}
      </div>
    </div>
  );
}
