// DELETE /api/delete now refuses with 409 when any manifest
// still names the blob's digest, and the backend puts the referencing tags in
// `detail`. deleteBlobApi used to throw `DELETE /api/delete ${r.status}` and
// drop the body on the floor, so the user pressing Delete on a projector three
// models need saw exactly "DELETE /api/delete 409" -- refused, with no reason
// and no list. A guard whose explanation never reaches a human is a guard
// talking to itself.
//
// These are pure function tests over a stubbed global fetch: no DOM, no
// events, no rendering. What they prove is that the thrown Error CARRIES the
// detail. What they cannot prove is how Models.tsx's notice bar displays it --
// that is for a browser-level check, not this runner.
import test from 'node:test';
import assert from 'node:assert/strict';
import { deleteBlobApi } from './api';

type Call = { url: string; init: RequestInit | undefined };

function stubFetch(status: number, json: () => Promise<unknown>) {
  const calls: Call[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
    calls.push({ url: String(url), init });
    return {
      ok: status >= 200 && status < 300,
      status,
      json,
    } as unknown as Response;
  }) as typeof globalThis.fetch;
  return { calls, restore: () => { globalThis.fetch = original; } };
}

const REFERENCED_DETAIL =
  'blob sha256:de1e7eab is still referenced by 3 manifest(s): ' +
  'model-a-27b (mmproj_blob_sha256), model-a-27b-q4 (mmproj_blob_sha256), ' +
  'model-a-27b-video (mmproj_blob_sha256). Delete or re-point them first ' +
  '(DELETE /api/manifests/<tag>), then retry.';

test('a 409 surfaces the referencing manifests, not just the status', async () => {
  const s = stubFetch(409, async () => ({ detail: REFERENCED_DETAIL }));
  try {
    await assert.rejects(
      () => deleteBlobApi('de1e7eab'),
      (e: Error) => {
        assert.match(e.message, /DELETE \/api\/delete 409/);
        // The whole point: every referencing tag reaches the caller.
        assert.ok(e.message.includes('model-a-27b-video'), e.message);
        assert.ok(e.message.includes('mmproj_blob_sha256'), e.message);
        return true;
      },
    );
  } finally {
    s.restore();
  }
});

test('the detail is appended in the same shape as every other write path', async () => {
  const s = stubFetch(409, async () => ({ detail: 'nope' }));
  try {
    await assert.rejects(
      () => deleteBlobApi('abc'),
      (e: Error) => {
        assert.equal(e.message, 'DELETE /api/delete 409 — nope');
        return true;
      },
    );
  } finally {
    s.restore();
  }
});

test('a failure whose body is not JSON still throws with the status', async () => {
  // A proxy returning an HTML error page must not turn into a JSON parse
  // error that hides the real status from the user.
  const s = stubFetch(502, async () => {
    throw new SyntaxError('Unexpected token < in JSON');
  });
  try {
    await assert.rejects(
      () => deleteBlobApi('abc'),
      (e: Error) => {
        assert.equal(e.message, 'DELETE /api/delete 502');
        return true;
      },
    );
  } finally {
    s.restore();
  }
});

test('a failure with no detail field does not append a dangling separator', async () => {
  const s = stubFetch(404, async () => ({}));
  try {
    await assert.rejects(
      () => deleteBlobApi('abc'),
      (e: Error) => {
        assert.equal(e.message, 'DELETE /api/delete 404');
        return true;
      },
    );
  } finally {
    s.restore();
  }
});

test('a successful delete still resolves', async () => {
  const s = stubFetch(200, async () => ({ status: 'deleted' }));
  try {
    await deleteBlobApi('abc');
  } finally {
    s.restore();
  }
});

test('the request itself is unchanged', async () => {
  // Reading the response body must not alter what is sent: same method, same
  // Ollama-compat {digest} payload, same header.
  const s = stubFetch(200, async () => ({}));
  try {
    await deleteBlobApi('de1e7eab');
    assert.equal(s.calls.length, 1);
    assert.equal(s.calls[0].url, '/api/delete');
    assert.equal(s.calls[0].init?.method, 'DELETE');
    assert.equal(s.calls[0].init?.body, JSON.stringify({ digest: 'de1e7eab' }));
    assert.deepEqual(s.calls[0].init?.headers, { 'Content-Type': 'application/json' });
  } finally {
    s.restore();
  }
});
