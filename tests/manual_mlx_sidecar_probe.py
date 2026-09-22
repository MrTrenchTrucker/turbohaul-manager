"""Probe the mlx_lm sidecar DIRECTLY (bypassing Turbohaul) to isolate whether a
hang is in the sidecar or in the manager's proxy/event loop."""
import json
import sys
import time
import urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 11500

# What model identity does the sidecar think it serves?
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
    models = json.loads(r.read())
print("MODELS:", json.dumps(models)[:400])

model_id = models["data"][0]["id"]
print("using model_id:", model_id)

body = json.dumps({
    "model": model_id,
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "What is 12 times 13? Answer briefly."}],
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=body, headers={"Content-Type": "application/json"},
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=120) as r:
    d = json.loads(r.read())
dt = time.time() - t0
msg = d["choices"][0]["message"]
ans = msg.get("content") or msg.get("reasoning") or ""
print(f"DIRECT SIDECAR OK in {dt:.2f}s")
print("ANSWER:", " ".join(ans.split())[:200])
print("usage:", d.get("usage"))
