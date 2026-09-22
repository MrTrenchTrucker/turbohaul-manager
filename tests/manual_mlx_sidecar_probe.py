"""Benchmark the mlx_lm sidecar DIRECTLY, bypassing Turbohaul.

Two uses:
  1. Diagnosis -- isolate whether a hang is in the sidecar or in Turbohaul's
     proxy/event loop (a wedged sidecar won't even answer /v1/models).
  2. Benchmarking -- run with --bench against the SAME sidecar Turbohaul
     spawned, right before/after the Turbohaul-routed run, so machine load is
     comparable. The delta is Turbohaul's proxy/queue overhead; the absolute
     number is what mlx_lm itself does on this model.

Note: send the REAL model id (the sidecar's --model arg), never the Turbohaul
tag -- mlx_lm re-resolves the request's `model` field as a HF repo and 404s.

Usage: python tests/manual_mlx_sidecar_probe.py [port] [--bench]
"""
import json
import sys
import time
import urllib.request

_args = [a for a in sys.argv[1:] if not a.startswith("-")]
PORT = int(_args[0]) if _args else 11500
BENCH = "--bench" in sys.argv

with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10) as r:
    models = json.loads(r.read())
# CAUTION: mlx_lm server's /v1/models lists the HF CACHE contents, NOT the
# loaded model -- the first entry is often some other model entirely. Sending
# it makes the sidecar load THAT model, silently benchmarking the wrong thing.
# Pass the real --model arg explicitly with --model-id=<path-or-repo>.
_override = [a for a in sys.argv[1:] if a.startswith("--model-id=")]
if _override:
    MODEL_ID = _override[0].split("=", 1)[1]
else:
    MODEL_ID = models["data"][0]["id"]
    print("WARNING: using /v1/models[0] -- this may NOT be the loaded model.")
    print("         Pass --model-id=<the sidecar's --model arg> to be sure.")
print(f"sidecar :{PORT}  model_id={MODEL_ID}\n")

PROMPTS = [
    "What is 12 times 13? Answer briefly.",
    "Name three primary colors. Answer briefly.",
    "What is the capital of France? Answer briefly.",
]


def ask(prompt, max_tokens=128):
    body = json.dumps({
        "model": MODEL_ID,          # REAL id, not the Turbohaul tag
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    ans = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
    usage = d.get("usage") or {}
    ctok = usage.get("completion_tokens")
    return {
        "answer": " ".join(ans.split())[:160],
        "latency_s": round(dt, 3),
        "completion_tokens": ctok,
        "prompt_tokens": usage.get("prompt_tokens"),
        "tok_s": round(ctok / dt, 2) if (ctok and dt > 0) else None,
    }


rows = []
for i, p in enumerate(PROMPTS if BENCH else PROMPTS[:1], 1):
    r = ask(p)
    rows.append(r)
    print(f"=== REQ {i}: {p}")
    print(f"    latency={r['latency_s']}s  prompt_tok={r['prompt_tokens']}  "
          f"completion_tok={r['completion_tokens']}  tok/s={r['tok_s']}")
    print(f"    ANSWER: {r['answer']}\n")

speeds = [r["tok_s"] for r in rows if r["tok_s"]]
if speeds:
    print("=" * 64)
    print(f"DIRECT SIDECAR decode: {speeds} tok/s  "
          f"(avg {round(sum(speeds) / len(speeds), 2)})")
    print("Compare against the Turbohaul-routed number from")
    print("tests/manual_mlx_prompt_replay_check.py --no-stream (same model/load).")
