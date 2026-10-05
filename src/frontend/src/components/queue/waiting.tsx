// The single definition of "how many requests are waiting", shared by the
// Queue tab's WaitingCard and the Dashboard's headline Queue card.
//
// ⛔ WHY THIS IS ITS OWN MODULE AND NOT JUST EXPORTED FROM Queue.tsx:
// Queue.tsx already imports RequestIdentityStrip from Dashboard.tsx, so having
// Dashboard.tsx import back from Queue.tsx would close an import cycle. It
// would probably work -- both are hoisted function declarations -- but "works
// because of hoisting order" is not a property to build a shared source of
// truth on. Same subdirectory convention as components/dashboard/.
//
// It is shared so the Queue tab and the Dashboard cannot disagree about what
// "waiting" means. One definition, imported twice, cannot drift.

export function waitingCount(queue: { queue_depth_total?: number }): number | null {
  // Returns null, NOT a fallback number, when the manager does not report it.
  // The obvious fallback -- staging + acceptance -- is exactly the undercount
  // this field exists to fix (a request handed to a busy resident's
  // inbox never enters either one), so falling back would render a
  // confidently wrong number that looks authoritative. "Unknown" is honest;
  // a wrong number is not.
  const n = queue.queue_depth_total;
  return typeof n === 'number' && Number.isFinite(n) ? n : null;
}

export function WaitingCard({ queue }: { queue: { queue_depth_total?: number } }) {
  // Exported and prop-driven: inline in the page body this
  // could not be rendered by the static harness, so a revert would go
  // unnoticed. Same technique the dashboard
  // tests already use on their own sub-components.
  const n = waitingCount(queue);
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Requests waiting</div>
      <div className="text-2xl font-bold text-slate-200 mb-3" data-testid="queue-waiting-total">
        {n ?? '—'}
      </div>
      <div className="text-xs text-slate-500">
        {n === null
          ? 'This manager does not report a waiting total.'
          : 'Includes requests held behind a busy resident, which never enter the staging queue.'}
      </div>
    </div>
  );
}
