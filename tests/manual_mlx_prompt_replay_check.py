"""Live reproduction of the MLX warm-inherit prompt-clobber bug (Fix D).

Sends two DIFFERENT prompts with NO session_id (the bare-curl shape that
triggered the bug) through a running Turbohaul on 127.0.0.1:11401.

BEFORE the fix: request 2 returns request 1's answer.
AFTER the fix:  each request answers its own prompt.
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:11401/v1/chat/completions"
MODEL = "qwen2.5-0.5b"

PROMPTS = [
    "What is 12 times 13? Answer briefly.",
    "Name three primary colors. Answer briefly.",
    "What is the capital of France? Answer briefly.",
]


def ask(prompt):
    body = json.dumps({
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    ans = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
    usage = d.get("usage") or {}
    return {
        "answer": " ".join(ans.split())[:220],
        "created": d.get("created"),
        "latency_s": round(dt, 3),
        "completion_tokens": usage.get("completion_tokens"),
    }


results = []
for i, p in enumerate(PROMPTS, 1):
    print(f"=== REQ {i}: {p}")
    r = ask(p)
    results.append(r)
    print(f"    created={r['created']}  latency={r['latency_s']}s  "
          f"tokens={r['completion_tokens']}")
    print(f"    ANSWER: {r['answer']}")
    print()

answers = [r["answer"] for r in results]
print("=" * 60)
if len(set(answers)) == 1:
    print("VERDICT: BUG PRESENT — all prompts returned the SAME answer.")
else:
    print(f"VERDICT: OK — {len(set(answers))}/{len(answers)} distinct answers; "
          "each prompt answered on its own.")
