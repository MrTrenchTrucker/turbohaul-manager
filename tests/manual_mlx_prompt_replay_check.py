"""Live MLX check: prompt-replay correctness (Fix D) + decode tok/s.

Sends several DIFFERENT prompts with NO session_id -- the bare-curl shape that
triggered the warm-inherit prompt-clobber bug -- against a running Turbohaul.

BEFORE Fix D: request 2+ returns request 1's answer.
AFTER  Fix D: each request answers its own prompt.

Speed: decode tok/s = usage.completion_tokens / wall_latency (ground truth).
NEVER count SSE chunks as tokens -- reasoning + keepalive pseudo-chunks make
that number meaningless. Streaming is the default because mlx_lm buffers
non-streamed responses, so there is no mid-flight token signal for live stats.

Usage:
    python tests/manual_mlx_prompt_replay_check.py [model_tag] [--no-stream]
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:11401/v1/chat/completions"

_args = [a for a in sys.argv[1:] if not a.startswith("-")]
MODEL = _args[0] if _args else "qwen3.5-9b-mlx-8bit"
STREAM = "--no-stream" not in sys.argv

PROMPTS = [
    "What is 12 times 13? Answer briefly.",
    "Name three primary colors. Answer briefly.",
    "What is the capital of France? Answer briefly.",
]


def _extract(d):
    """Reasoning models put the answer in `reasoning`, leaving `content` absent."""
    return d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""


def ask(prompt):
    payload = {
        "model": MODEL,
        "max_tokens": 128,
        "messages": [{"role": "user", "content": prompt}],
        "stream": STREAM,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    parts = []
    usage = {}
    created = None

    with urllib.request.urlopen(req, timeout=900) as r:
        if not STREAM:
            d = json.loads(r.read())
            parts.append(_extract(d["choices"][0]["message"]))
            usage = d.get("usage") or {}
            created = d.get("created")
        else:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                created = created or ev.get("created")
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    piece = _extract(ch.get("delta") or {})
                    if piece:
                        if ttft is None:
                            ttft = time.time() - t0
                        parts.append(piece)

    dt = time.time() - t0
    ctok = usage.get("completion_tokens")
    return {
        "answer": " ".join("".join(parts).split())[:220],
        "created": created,
        "latency_s": round(dt, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "completion_tokens": ctok,
        "tok_s": round(ctok / dt, 2) if (ctok and dt > 0) else None,
    }


print(f"model={MODEL}  stream={STREAM}\n")
results = []
for i, p in enumerate(PROMPTS, 1):
    print(f"=== REQ {i}: {p}")
    r = ask(p)
    results.append(r)
    print(f"    created={r['created']}  latency={r['latency_s']}s  "
          f"ttft={r['ttft_s']}s  tokens={r['completion_tokens']}  "
          f"tok/s={r['tok_s']}")
    print(f"    ANSWER: {r['answer']}\n")

answers = [r["answer"] for r in results]
print("=" * 64)
distinct = len(set(answers))
if distinct == 1:
    print("CORRECTNESS: *** BUG PRESENT *** every prompt returned the SAME answer.")
else:
    print(f"CORRECTNESS: OK -- {distinct}/{len(answers)} distinct answers.")

if results[0]["tok_s"]:
    print(f"SPEED: first request (includes model load) {results[0]['tok_s']} tok/s")
warm = [r["tok_s"] for r in results[1:] if r["tok_s"]]
if warm:
    print(f"SPEED: warm decode avg {round(sum(warm) / len(warm), 2)} tok/s "
          f"over {len(warm)} request(s)")
    print("       [oMLX baseline ~49.85 tok/s on Qwen3.5-9B-MLX-8bit]")
