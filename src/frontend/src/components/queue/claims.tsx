// The Fast Lane claims strip.
//
// WHY ITS OWN MODULE (same reasoning waiting.tsx documents): inline in the
// Queue.tsx page body this could not be rendered by the static harness
// (renderToStaticMarkup), so the entire feature could ship or be reverted
// with every other test still passing. Exported + prop-driven, specifically
// so Queue.test.tsx can pin its rendering in isolation.
//
// WHY THIS IS A SEPARATE POPULATION from QueueWaitingRows (queue.waiting):
// queue.waiting is backed by queue_snapshot(limit=50) over queue._staging and
// only sees requests that have REACHED staging. A Fast Lane claim that is
// genuinely deferring -- registered via _defer_unroutable but not yet
// staged -- is INVISIBLE to that population, so the queue tab would show
// nothing while a client is genuinely waiting. /status
// serves this population separately, as fastlane_claims_snapshot
// (manager.status_snapshot).
// Rendering it without a distinct, labeled section would present two
// populations as one.
import type { FastlaneClaimRow } from '../../api';

export function QueueClaimRows({ claims }: { claims?: FastlaneClaimRow[] }) {
  if (claims === undefined) {
    return (
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-400">
        This manager does not report Fast Lane claims.
      </div>
    );
  }
  if (claims.length === 0) {
    return (
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-400">
        No active claims.
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-slate-900 text-xs uppercase text-slate-500">
          <tr>
            <th className="text-left px-4 py-2">Model</th>
            <th className="text-left px-4 py-2">Client</th>
            <th className="text-left px-4 py-2">Tag / rank</th>
            <th className="text-left px-4 py-2">Reason</th>
            <th className="text-left px-4 py-2">Waited</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {claims.map((row) => {
            // QueueWaitingRows' own idiom: the fastlane label as the client
            // identity, thread_id_prefix when unlisted; the rank when listed,
            // an em dash otherwise.
            const client = row.fastlane?.label || row.thread_id_prefix;
            const tagRank = row.fastlane ? `${row.fastlane.rank}` : '—';
            return (
              <tr key={row.slot_id} className="text-slate-300" data-testid="queue-claim-row">
                <td className="px-4 py-2 font-mono">{row.model_tag}</td>
                <td className="px-4 py-2 font-mono">{client}</td>
                <td className="px-4 py-2 font-mono text-center">{tagRank}</td>
                <td className="px-4 py-2 text-slate-400">{row.reason}</td>
                <td className="px-4 py-2 font-mono">{row.waited_s}s</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
