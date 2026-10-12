"""Does `mlx_lm server` actually serve requests CONCURRENTLY?

Turbohaul's series-parallel fan-out is capped by `handle.parallel`. mlx_spawn
pins that to 1 ("mlx-lm is single-slot per process"). Modern mlx_lm.server is a
ThreadingHTTPServer with a real BatchGenerator
(completion_batch_size=--decode-concurrency), so that comment may be stale.

This measures it instead of assuming: fire N identical-shape requests at once
and compare total wall time against the measured single-request time.

  ~N x single  -> serialized (parallel=1 is correct)
  ~1 x single  -> genuinely batched (fan-out is real)

Usage:
  python tests/manual_mlx_concurrency_probe.py <port> --model-id=<path> [-n 4]
"""
import json
import sys
import threading
import time
import urllib.request

_pos = [a for a in sys.argv[1:] if not a.startswith("-")]
PORT = int(_pos[0]) if _pos else 11500
MODEL_ID = next(
    (a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--model-id=")), None
)
N = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("-n=")), 4))
if not MODEL_ID:
    sys.exit("need --model-id=<the sidecar's --model arg>")

# Distinct prompts: identical ones could be served from the prompt cache and
# fake a speedup that has nothing to do with batching.
PROMPTS = [
    "Count slowly from 1 to 30, one number per line.",
    "List 30 common English verbs, one per line.",
    "Name 30 countries, one per line.",
    "List 30 colors, one per line.",
    "List 30 animals, one per line.",
    "List 30 fruits, one per line.",
]


def one(prompt, out, idx):
    body = json.dumps({
        "model": MODEL_ID,
        "max_tokens": 120,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        usage = d.get("usage") or {}
        out[idx] = {"s": round(time.time() - t0, 2),
                    "tok": usage.get("completion_tokens")}
    except Exception as exc:
        out[idx] = {"s": round(time.time() - t0, 2), "error": repr(exc)[:120]}


print(f"sidecar :{PORT}  n={N}\n")

# Warm-up: load the model and let any first-request cost settle, so the
# baseline below measures GENERATION, not model load. Without this the
# baseline is inflated and the batching ratio is meaningless.
print("--- warm-up (model load; timing discarded) ---")
_warm = {}
one(PROMPTS[0], _warm, 0)
print(f"    {_warm[0]}\n")
if "error" in _warm[0]:
    sys.exit("warm-up failed; fix that before interpreting concurrency")

print("--- baseline: ONE request alone (warm) ---")
base = {}
one(PROMPTS[1], base, 0)
b = base[0]
print(f"    {b}\n")
if "error" in b:
    sys.exit("baseline failed; fix that before interpreting concurrency")

print(f"--- {N} requests fired SIMULTANEOUSLY ---")
res = {}
threads = [
    threading.Thread(target=one, args=(PROMPTS[i % len(PROMPTS)], res, i))
    for i in range(N)
]
t0 = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.time() - t0

for i in sorted(res):
    print(f"    req{i}: {res[i]}")

single = b["s"]
serial_est = single * N
print(f"\n    wall={wall:.2f}s   single={single:.2f}s   "
      f"serial estimate={serial_est:.2f}s")
print("=" * 64)
ratio = wall / single if single else 0
if wall < serial_est * 0.7:
    print(f"BATCHED: {N} requests in {ratio:.1f}x a single request "
          f"(serial would be ~{N}x). Fan-out is REAL -- parallel=1 is wrong.")
else:
    print(f"SERIALIZED: {N} requests took {ratio:.1f}x a single request. "
          "parallel=1 is correct for this configuration.")
