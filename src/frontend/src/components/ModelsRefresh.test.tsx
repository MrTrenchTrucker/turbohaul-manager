// The models page (/ui/blob/models) never noticed the world changing. It
// reloaded only when its own mutations bumped a counter, so anything altered
// from another tab, a worker, or a manifest written on disk stayed invisible
// until someone pressed Refresh (about 60 s stale, then instant on Refresh).
//
// The fix is an event subscription with a fallback tick under it. The trap is
// that a refresh landing mid-edit must not destroy work -- and the mechanism
// is UNMOUNT, not overwrite: ModelEditor takes no data props and keys its own
// fetch on [tag]; InlineTextPrompt seeds via useState(initial), an
// initializer React ignores on later prop changes. Neither can be overwritten
// by fresh data. What kills an edit is the row or tile it lives in
// disappearing from under it.
//
// ⛔ WHAT THIS FILE CAN AND CANNOT PROVE. The harness is node --test over
// renderToStaticMarkup: no DOM events, no fetch stub, no websocket, no
// timers. It CANNOT observe a subscription firing, a socket message, a
// debounce coalescing, an interval ticking, or a component unmounting. It
// proves the PURE parts, exhaustively. Everything else is left to a manual check
// against a real browser, and is not
// implied to be covered here.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MODELS_POLL_INTERVAL_MS,
  MODELS_EVENT_DEBOUNCE_MS,
  MODELS_REFRESH_EVENTS,
  shouldSuspendRefresh,
  resolveRefreshAction,
  type ModelsInteractionState,
  type RefreshTrigger,
} from './Models';

/** The fully-idle page: nothing open, nothing in flight. */
function idle(): ModelsInteractionState {
  return {
    expandedTag: null,
    renamingTag: null,
    addingForDigest: null,
    editingDescription: false,
    showDeleteModelPrompt: false,
    busyKey: null,
  };
}

// Every interaction surface on the page, each mapped to the mount site it
// guards, so a reader can check the list against the component rather than
// trusting it. Seven named surfaces over six fields: the three
// InlineTextPrompt sites share one component but have three separate
// open-states.
const SURFACES: ReadonlyArray<{
  name: string;
  mountSite: string;
  open: Partial<ModelsInteractionState>;
}> = [
  {
    name: 'ModelEditor open',
    mountSite: 'expandedTag === row.model_tag',
    open: { expandedTag: 'model-a-27b' },
  },
  {
    name: 'InlineTextPrompt open (rename)',
    mountSite: 'renamingTag === row.model_tag',
    open: { renamingTag: 'model-a-27b' },
  },
  {
    name: 'InlineTextPrompt open (add manifest)',
    mountSite: 'addingForDigest === selectedTile.digest',
    open: { addingForDigest: 'cbb841a9ee06' },
  },
  {
    name: 'InlineTextPrompt open (model description, multiline)',
    mountSite: 'editingDescription',
    open: { editingDescription: true },
  },
  {
    name: 'DeleteBlobPrompt open',
    mountSite: 'showDeleteModelPrompt',
    open: { showDeleteModelPrompt: true },
  },
  {
    name: 'a mutation in flight',
    mountSite: 'busyKey !== null',
    open: { busyKey: 'model-a-27b' },
  },
];

test('the fully idle page does NOT suspend', () => {
  assert.equal(shouldSuspendRefresh(idle()), false);
});

// One test per surface. If a future surface is added to the page and not to
// the predicate, this list is where the omission shows up.
for (const s of SURFACES) {
  test(`suspends while: ${s.name}  [${s.mountSite}]`, () => {
    assert.equal(
      shouldSuspendRefresh({ ...idle(), ...s.open }),
      true,
      `${s.name} must suspend the refresh -- it guards ${s.mountSite}`,
    );
  });
}

test('every surface is independently sufficient — none relies on another', () => {
  // Guards against a predicate written with && instead of ||, which would
  // pass every single-surface test above only if they were checked together.
  for (const s of SURFACES) {
    const state = { ...idle(), ...s.open };
    const others = SURFACES.filter((o) => o !== s);
    for (const o of others) {
      assert.equal(
        shouldSuspendRefresh(state),
        true,
        `${s.name} alone must suspend, without ${o.name}`,
      );
    }
  }
});

test('all six fields together suspend', () => {
  const all = SURFACES.reduce((acc, s) => ({ ...acc, ...s.open }), idle());
  assert.equal(shouldSuspendRefresh(all), true);
});

test('an empty-string busyKey is not treated as busy', () => {
  // busyKey is a nullable id, not a flag: '' is not a real key. Pinned
  // because `!!busyKey` and `busyKey !== null` differ here, and the loose
  // form would be indistinguishable in every other test.
  assert.equal(shouldSuspendRefresh({ ...idle(), busyKey: '' }), true);
});

// ===== the latch =====

// All 12 combinations, written out rather than generated, so the expected
// column is a statement of intent and not a re-implementation of the function.
const LATCH_CASES: ReadonlyArray<{
  trigger: RefreshTrigger;
  suspended: boolean;
  pendingChange: boolean;
  refreshNow: boolean;
  nextPending: boolean;
  why: string;
}> = [
  { trigger: 'event', suspended: false, pendingChange: false, refreshNow: true, nextPending: false,
    why: 'idle page, something changed -> refresh now' },
  { trigger: 'event', suspended: false, pendingChange: true, refreshNow: true, nextPending: false,
    why: 'idle page with a latch set -> refresh now and clear it' },
  { trigger: 'event', suspended: true, pendingChange: false, refreshNow: false, nextPending: true,
    why: 'THE TRAP: an event during an edit is remembered, never applied' },
  { trigger: 'event', suspended: true, pendingChange: true, refreshNow: false, nextPending: true,
    why: 'a second event during an edit stays latched, does not stack' },

  { trigger: 'tick', suspended: false, pendingChange: false, refreshNow: true, nextPending: false,
    why: 'the fallback net catches a missed event on an idle page' },
  { trigger: 'tick', suspended: false, pendingChange: true, refreshNow: true, nextPending: false,
    why: 'tick on an idle page clears any latch too' },
  { trigger: 'tick', suspended: true, pendingChange: false, refreshNow: false, nextPending: true,
    why: 'A TICK LATCHES TOO: the tick exists because events can be missed, so it is read as MAY have changed' },
  { trigger: 'tick', suspended: true, pendingChange: true, refreshNow: false, nextPending: true,
    why: 'already latched, stays latched' },

  { trigger: 'unsuspend', suspended: false, pendingChange: true, refreshNow: true, nextPending: false,
    why: 'THE POINT OF THE LATCH: the editor closed and the held change lands immediately, not on the next tick' },
  { trigger: 'unsuspend', suspended: false, pendingChange: false, refreshNow: false, nextPending: false,
    why: 'closed with nothing held -> no pointless re-fetch' },
  { trigger: 'unsuspend', suspended: true, pendingChange: true, refreshNow: false, nextPending: true,
    why: 'contradictory input: still suspended, so hold the latch rather than guess' },
  { trigger: 'unsuspend', suspended: true, pendingChange: false, refreshNow: false, nextPending: false,
    why: 'contradictory input with nothing held: no refresh, latch untouched' },
];

for (const c of LATCH_CASES) {
  test(`latch: ${c.trigger} / suspended=${c.suspended} / pending=${c.pendingChange} — ${c.why}`, () => {
    const got = resolveRefreshAction({
      suspended: c.suspended,
      pendingChange: c.pendingChange,
      trigger: c.trigger,
    });
    assert.deepEqual(got, { refreshNow: c.refreshNow, pendingChange: c.nextPending });
  });
}

test('the 12 cases are exhaustive over the input space', () => {
  // Cheap guard against the table silently drifting out of sync with the type.
  const triggers: RefreshTrigger[] = ['event', 'tick', 'unsuspend'];
  const seen = new Set(LATCH_CASES.map((c) => `${c.trigger}|${c.suspended}|${c.pendingChange}`));
  for (const trigger of triggers) {
    for (const suspended of [true, false]) {
      for (const pendingChange of [true, false]) {
        assert.ok(
          seen.has(`${trigger}|${suspended}|${pendingChange}`),
          `missing case: ${trigger} suspended=${suspended} pending=${pendingChange}`,
        );
      }
    }
  }
  assert.equal(LATCH_CASES.length, 12);
});

test('a refresh is NEVER issued while suspended, whatever the trigger', () => {
  // The single invariant the whole trap reduces to, asserted directly rather
  // than left as an emergent property of the table above.
  const triggers: RefreshTrigger[] = ['event', 'tick', 'unsuspend'];
  for (const trigger of triggers) {
    for (const pendingChange of [true, false]) {
      const got = resolveRefreshAction({ suspended: true, pendingChange, trigger });
      assert.equal(got.refreshNow, false, `${trigger} refreshed while suspended`);
    }
  }
});

test('a change is never silently dropped while suspended', () => {
  // The other half: suspending must LATCH, not discard. A suspended event
  // that left pendingChange false would lose the update entirely, because an
  // event -- unlike a tick -- does not come round again.
  for (const trigger of ['event', 'tick'] as RefreshTrigger[]) {
    const got = resolveRefreshAction({ suspended: true, pendingChange: false, trigger });
    assert.equal(got.pendingChange, true, `${trigger} was dropped instead of latched`);
    // ...and the held change lands as soon as the page is interactive again.
    const onClose = resolveRefreshAction({
      suspended: false,
      pendingChange: got.pendingChange,
      trigger: 'unsuspend',
    });
    assert.equal(onClose.refreshNow, true);
    assert.equal(onClose.pendingChange, false);
  }
});

// ===== constants =====

test('the fallback interval is 10s, not useStatus\'s 1000ms', () => {
  assert.equal(MODELS_POLL_INTERVAL_MS, 10_000);
});

test('the event debounce is short enough to feel instant, long enough to coalesce a burst', () => {
  assert.equal(MODELS_EVENT_DEBOUNCE_MS, 300);
  assert.ok(MODELS_EVENT_DEBOUNCE_MS < MODELS_POLL_INTERVAL_MS);
});

test('only manifest/blob mutation events trigger a refresh', () => {
  assert.ok(MODELS_REFRESH_EVENTS.has('manifest_changed'));
  assert.ok(MODELS_REFRESH_EVENTS.has('blob_changed'));
  assert.equal(MODELS_REFRESH_EVENTS.size, 2);
});

test('generation_tick is NOT a refresh event', () => {
  // The one that matters: live_monitor publishes generation_tick at ~1Hz for
  // the whole duration of EVERY generation, expressly to drive useStatus's
  // re-fetch. subscribeWsState hands every event to every handler, so a
  // subscriber that did not filter would re-parse the entire manifest
  // registry at 1Hz throughout every inference.
  assert.equal(MODELS_REFRESH_EVENTS.has('generation_tick'), false);
});

test('no inference-lifecycle or pull event triggers a refresh', () => {
  // ws_state.py:5-7's own list, plus the import/pull families. None of these
  // says anything about a manifest or a blob's metadata.
  const notOurs = [
    'connected', 'submit', 'stage_to_loading', 'active', 'grace_enter',
    'teardown', 'idle_hot_enter', 'queue_change', 'generation_tick',
    'import_started', 'import_complete', 'import_failed',
    'pull_started', 'pull_progress', 'pull_complete',
  ];
  for (const name of notOurs) {
    assert.equal(MODELS_REFRESH_EVENTS.has(name), false, `${name} must not trigger a refresh`);
  }
});
