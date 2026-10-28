// Mounts the real default Dashboard (real useStatus, real useLiveStream, real
// children) under a hand-written fake DOM and drives it through status
// snapshots that the test controls. Nothing here waits on real time: the
// poll interval is captured from the fake timers and called by hand.
import test from 'node:test';
import assert from 'node:assert/strict';
import { act } from 'react';
import type { Root } from 'react-dom/client';
import type { ActiveInfo, GenerationInfo, ResidentModel, StatusSnapshot } from '../api';
import Dashboard from './Dashboard';
import { findAllByAttribute, findElements, installFakeDom } from '../test-support/fakeDom';
import type { FakeDomHandle, FakeElement, FakeNode } from '../test-support/fakeDom';

// The test tsconfig has no @types/node, so the one Node global used here is declared by hand.
// It is a real Node function at run time; the declaration only satisfies the type checker.
declare function setImmediate(callback: () => void): unknown;

const MAX_SETTLE_MACROTASKS = 20;

function makeGeneration(overrides: Partial<GenerationInfo> = {}): GenerationInfo {
  return {
    state: 'generating',
    prompt_progress: null,
    stalled: false,
    streaming: true,
    generation_id: 'gen-1',
    measured_at_iso: '2026-08-20T00:00:00Z',
    ...overrides,
  };
}

function makeResident(overrides: Partial<ResidentModel> = {}): ResidentModel {
  return {
    model_tag: 'qwen3.6-27b',
    state: 'ACTIVE',
    port: 11500,
    pid: 4242,
    spawn_seq: 1,
    reserved_need_mib: 24000,
    parallel: 1,
    main_gpu: 0,
    split_mode: 'single',
    inflight: 1,
    idle_expires_in_s: null,
    generation: makeGeneration(),
    ...overrides,
  };
}

function makeStatus(overrides: Partial<StatusSnapshot> = {}): StatusSnapshot {
  return {
    queue: { acceptance_buffer_depth: 0, staging_queue_depth: 0, staging_queue_max: 8 },
    active: null,
    loading: null,
    grace: null,
    idle_hot: null,
    parallel_slots: { used: 0, max: 1 },
    residents: [],
    vram: null,
    vram_total_mib: null,
    generation: null,
    ...overrides,
  };
}

// The legacy single-residency field that synthesizeResident reads first. With
// residents[] empty at cap 1 it becomes one card carrying this tag.
function makeActive(modelTag: string): ActiveInfo {
  return {
    slot_id: 's0',
    model_tag: modelTag,
    state: 'ACTIVE',
    thread_id_prefix: 'th-',
    pid: 1,
    port: 11599,
  };
}

const ALIAS_TAG = 'alias-most-recently-active';

interface DashboardView {
  /** One entry per resident card the user can see, in display order. */
  cardTags(): string[];
  /** The value of the QueueCard "parallel slots" row, for example "2 / 2". */
  slotsRow(): string;
  /** Fires the one poll interval useStatus registered, then settles its fetch. */
  tick(): Promise<void>;
  /** How many /status snapshots the stub has handed out so far. */
  statusServed(): number;
}

// QueueCard renders each KV row as a div holding a label span and a value span.
// The row is found by its label, so no other text on the page can match it.
function parallelSlotsRow(container: FakeNode): string {
  const rows = findElements(
    container,
    (el) => el.childNodes.length === 2 && el.childNodes[0].textContent === 'parallel slots',
  );
  assert.equal(
    rows.length,
    1,
    `expected exactly one "parallel slots" row (QueueCard in Dashboard.tsx), found ${rows.length}`,
  );
  return rows[0].childNodes[1].textContent;
}

// A tick is judged by what it renders. The slots row and the card tags both
// come from the snapshot just served, so a poll that never applies the
// snapshot leaves them on the previous value and fails here.
function assertGrid(view: DashboardView, when: string, slots: string, tags: string[]): void {
  assert.equal(
    view.slotsRow(),
    slots,
    `${when}: the "parallel slots" row must show "${slots}". If it still shows the previous value, ` +
      'the poll did not apply this snapshot to the rendered tree (see useStatus.ts fetchOnce).',
  );
  assert.deepEqual(
    view.cardTags(),
    tags,
    `${when}: resident cards must be ${JSON.stringify(tags)}. Each card root carries ` +
      'data-testid="resident-card" and its title data-testid="resident-tag" (ResidentCard.tsx). ' +
      `An extra card with the alias tag means the held set was not used; that tag comes from ` +
      'synthesizeResident, which reads the legacy active field first.',
  );
}

// Mounts the real Dashboard with a fetch stub that serves `script` one entry
// per /status call. Teardown always runs: the unmount in its own try, then
// the globals restore in a nested finally, so a throw in either step cannot
// leave the fake installed for the next test.
async function withDashboard(
  script: StatusSnapshot[],
  body: (view: DashboardView) => Promise<void>,
): Promise<void> {
  let served = 0;
  let jsonReads = 0;
  const dom: FakeDomHandle = installFakeDom({
    fetch: (url) => {
      // The per-resident SSE streams never answer. Nothing in these tests reads them.
      if (url !== '/status') return new Promise<never>(() => {});
      const snapshot = script[served];
      served += 1;
      if (!snapshot) {
        return Promise.reject(new Error('status script ran out: the Dashboard polled more often than the test expects'));
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => {
          jsonReads += 1;
          return snapshot;
        },
      });
    },
  });

  // Waits for real macrotasks until the expected snapshot has been fetched and
  // its body read. A macrotask only runs after the promise queue is empty, so
  // the state update is queued by the time the check passes. Bounded.
  async function settleStatus(snapshotsExpected: number): Promise<void> {
    for (let turn = 0; turn < MAX_SETTLE_MACROTASKS; turn += 1) {
      await new Promise<void>((resolve) => setImmediate(resolve));
      if (served >= snapshotsExpected && jsonReads >= snapshotsExpected) return;
    }
    throw new Error(
      `status snapshot ${snapshotsExpected} did not settle after ${MAX_SETTLE_MACROTASKS} macrotasks: ` +
        'the poll did not fetch or read it',
    );
  }

  let root: Root | null = null;
  let unmounted = false;
  const unmount = async (): Promise<void> => {
    if (!root || unmounted) return;
    unmounted = true;
    const closing = root;
    await act(async () => {
      closing.unmount();
    });
  };

  try {
    const { createRoot } = await import('react-dom/client');
    const container = dom.document.createElement('div');
    dom.document.body.appendChild(container);
    const mounted = createRoot(container as unknown as HTMLElement);
    root = mounted;
    const baseline = dom.listenerCounts();

    await act(async () => {
      mounted.render(<Dashboard />);
    });
    await act(async () => {
      await settleStatus(1);
    });

    const view: DashboardView = {
      cardTags: () =>
        findAllByAttribute(container, 'data-testid', 'resident-card').map((card: FakeElement) => {
          const [title] = findAllByAttribute(card, 'data-testid', 'resident-tag');
          return title ? title.textContent : '?';
        }),
      slotsRow: () => parallelSlotsRow(container),
      tick: async () => {
        const repeating = dom.timers.pending().filter((timer) => timer.repeat);
        assert.equal(
          repeating.length,
          1,
          'the mounted Dashboard must register exactly one repeating timer, the useStatus poll; ' +
            'if a new interval is legitimate, identify the poll by its callback instead of by count',
        );
        const [poll] = repeating;
        const expected = served + 1;
        await act(async () => {
          poll.fn();
          await settleStatus(expected);
        });
      },
      statusServed: () => served,
    };

    await body(view);

    await unmount();
    assert.deepEqual(
      dom.listenerCounts(),
      baseline,
      'window or document listeners after unmount must match the counts taken right after createRoot',
    );
    assert.equal(
      dom.timers.pending().filter((timer) => timer.repeat).length,
      0,
      'no repeating timer may be pending after unmount (useStatus clears its poll in its cleanup)',
    );
    assert.ok(dom.sockets.length > 0, 'the status subscription should have opened a WebSocket');
    for (const socket of dom.sockets) {
      assert.equal(socket.closed, true, `WebSocket ${socket.url} must be closed after unmount`);
    }
  } finally {
    try {
      await unmount();
    } finally {
      dom.restore();
    }
  }
}

test('Dashboard keeps both residents across a swap-empty tick at cap 2', async () => {
  const resA = makeResident({ model_tag: 'res-A', port: 11500 });
  const resB = makeResident({ model_tag: 'res-B', port: 11501, spawn_seq: 2 });
  const twoResidents = makeStatus({ parallel_slots: { used: 2, max: 2 }, residents: [resA, resB] });
  // The swap tick: residents[] empties at the same configured cap. The legacy
  // fields still carry a singular alias that a collapse would render as a card.
  const swapEmpty = makeStatus({
    parallel_slots: { used: 0, max: 2 },
    residents: [],
    active: makeActive(ALIAS_TAG),
    generation: makeGeneration({ tok_s: 30 }),
  });

  await withDashboard([twoResidents, swapEmpty, twoResidents], async (view) => {
    assertGrid(view, 'after mount', '2 / 2', ['res-A', 'res-B']);

    await view.tick();
    assertGrid(view, 'tick 2 (swap-empty)', '0 / 2', ['res-A', 'res-B']);

    await view.tick();
    assertGrid(view, 'tick 3', '2 / 2', ['res-A', 'res-B']);

    assert.equal(view.statusServed(), 3, 'each tick must fetch exactly one snapshot');
  });
});

test('CONTROL at cap 1: the same swap-empty tick legitimately shows exactly one synthesized card', async () => {
  // Same three ticks as the cap-2 test, with the configured cap at 1. Two
  // residents at cap 1 is not a state the backend produces; the snapshot is
  // reused only so the tick sequence is identical and the probe is the only
  // thing that differs.
  const resA = makeResident({ model_tag: 'res-A', port: 11500 });
  const resB = makeResident({ model_tag: 'res-B', port: 11501, spawn_seq: 2 });
  const twoResidents = makeStatus({ parallel_slots: { used: 1, max: 1 }, residents: [resA, resB] });
  const swapEmpty = makeStatus({
    parallel_slots: { used: 0, max: 1 },
    residents: [],
    active: makeActive(ALIAS_TAG),
    generation: makeGeneration({ tok_s: 30 }),
  });

  await withDashboard([twoResidents, swapEmpty, twoResidents], async (view) => {
    assertGrid(view, 'after mount (cap 1)', '1 / 1', ['res-A', 'res-B']);

    await view.tick();
    assertGrid(view, 'tick 2 at cap 1', '0 / 1', [ALIAS_TAG]);

    await view.tick();
    assertGrid(view, 'tick 3 at cap 1', '1 / 1', ['res-A', 'res-B']);
  });
});

// Builders for the cases below. Each call returns a new object, so a tick that
// serves one never hands the status hook the same reference as the tick before.
function emptyGridSnapshot(): StatusSnapshot {
  return makeStatus({ parallel_slots: { used: 0, max: 2 }, residents: [] });
}

function residentPairSnapshot(tagA: string, tagB: string): StatusSnapshot {
  return makeStatus({
    parallel_slots: { used: 2, max: 2 },
    residents: [
      makeResident({ model_tag: tagA, port: 11500 }),
      makeResident({ model_tag: tagB, port: 11501, spawn_seq: 2 }),
    ],
  });
}

test('Dashboard releases the held residents on the third consecutive empty snapshot at cap 2', async () => {
  // Pins the bound on the hold: the first and second consecutive empty
  // snapshots still show the held pair, and the third shows the empty grid.
  // A hold with no limit keeps the pair on the third tick and fails there.
  const script = [
    residentPairSnapshot('res-A', 'res-B'),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
  ];

  await withDashboard(script, async (view) => {
    assertGrid(view, 'after mount', '2 / 2', ['res-A', 'res-B']);

    await view.tick();
    assertGrid(view, 'tick 2 (1st empty)', '0 / 2', ['res-A', 'res-B']);
    assert.equal(view.statusServed(), 2, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 3 (2nd empty)', '0 / 2', ['res-A', 'res-B']);
    assert.equal(view.statusServed(), 3, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 4 (3rd empty)', '0 / 2', []);
    assert.equal(view.statusServed(), 4, 'each tick must fetch exactly one snapshot');
  });
});

test('Dashboard keeps the grid released through a fourth and fifth consecutive empty snapshot at cap 2', async () => {
  // Pins that the release is sticky: once the held pair is dropped, later empty
  // snapshots do not bring it back. Ticks 2 and 3 match the case above.
  const script = [
    residentPairSnapshot('res-A', 'res-B'),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
  ];

  await withDashboard(script, async (view) => {
    assertGrid(view, 'after mount', '2 / 2', ['res-A', 'res-B']);

    await view.tick();
    assertGrid(view, 'tick 2 (1st empty)', '0 / 2', ['res-A', 'res-B']);
    assert.equal(view.statusServed(), 2, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 3 (2nd empty)', '0 / 2', ['res-A', 'res-B']);
    assert.equal(view.statusServed(), 3, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 4 (3rd empty)', '0 / 2', []);
    assert.equal(view.statusServed(), 4, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 5 (4th empty)', '0 / 2', []);
    assert.equal(view.statusServed(), 5, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 6 (5th empty)', '0 / 2', []);
    assert.equal(view.statusServed(), 6, 'each tick must fetch exactly one snapshot');
  });
});

test('Dashboard restarts the empty-snapshot count after a non-empty snapshot at cap 2', async () => {
  // Pins the reset: a non-empty snapshot clears the count, so the new resident
  // pair gets its own two empty polls of hold before the release. Ticks 2 to 6
  // are controls; the final assertion fails for an unbounded hold.
  const script = [
    residentPairSnapshot('res-A', 'res-B'),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
    residentPairSnapshot('res-C', 'res-D'),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
  ];

  await withDashboard(script, async (view) => {
    assertGrid(view, 'after mount', '2 / 2', ['res-A', 'res-B']);

    await view.tick();
    assertGrid(view, 'tick 2 (1st empty)', '0 / 2', ['res-A', 'res-B']);
    assert.equal(view.statusServed(), 2, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 3 (2nd empty)', '0 / 2', ['res-A', 'res-B']);
    assert.equal(view.statusServed(), 3, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 4 (new pair)', '2 / 2', ['res-C', 'res-D']);
    assert.equal(view.statusServed(), 4, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 5 (1st empty after reset)', '0 / 2', ['res-C', 'res-D']);
    assert.equal(view.statusServed(), 5, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 6 (2nd empty after reset)', '0 / 2', ['res-C', 'res-D']);
    assert.equal(view.statusServed(), 6, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 7 (3rd empty after reset)', '0 / 2', []);
    assert.equal(view.statusServed(), 7, 'each tick must fetch exactly one snapshot');
  });
});

test('Dashboard restarts the empty-snapshot count after a cap-1 empty snapshot at cap 2', async () => {
  // Pins that an empty snapshot at cap 1 is not counted and resets the count.
  // Ticks 2 and 3 are the first two cap-2 empties, so the pair is held. Tick 4
  // is a cap-1 empty: it must restart the count, not add a third empty to it.
  // Ticks 5 and 6 are then the first two cap-2 empties again and still hold the
  // pair; tick 7 is the third and releases it. Without the reset, tick 5 would
  // already be the third empty and the grid would empty one poll early.
  const script = [
    residentPairSnapshot('res-A', 'res-B'),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
    makeStatus({
      parallel_slots: { used: 0, max: 1 },
      residents: [],
      active: makeActive(ALIAS_TAG),
      generation: makeGeneration({ tok_s: 30 }),
    }),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
    emptyGridSnapshot(),
  ];

  await withDashboard(script, async (view) => {
    assertGrid(view, 'after mount', '2 / 2', ['res-A', 'res-B']);

    await view.tick();
    assertGrid(view, 'tick 2 (1st empty at cap 2)', '0 / 2', ['res-A', 'res-B']);
    assert.equal(view.statusServed(), 2, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 3 (2nd empty at cap 2)', '0 / 2', ['res-A', 'res-B']);
    assert.equal(view.statusServed(), 3, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 4 (cap-1 empty)', '0 / 1', [ALIAS_TAG]);
    assert.equal(view.statusServed(), 4, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 5 (1st cap-2 empty after the cap-1 reset)', '0 / 2', ['res-A', 'res-B']);
    assert.equal(view.statusServed(), 5, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 6 (2nd cap-2 empty after the cap-1 reset)', '0 / 2', ['res-A', 'res-B']);
    assert.equal(view.statusServed(), 6, 'each tick must fetch exactly one snapshot');

    await view.tick();
    assertGrid(view, 'tick 7 (3rd cap-2 empty after the cap-1 reset)', '0 / 2', []);
    assert.equal(view.statusServed(), 7, 'each tick must fetch exactly one snapshot');
  });
});
