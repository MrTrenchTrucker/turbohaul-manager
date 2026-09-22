"""Head-to-head MLX runner benchmark harness (v2, ground-truth tokens).

For one OpenAI-compatible server measures:
  * cold model-load-from-disk : restart server first, read usage.model_load_duration
                                 (oMLX) OR first-request TTFT (load+prefill) for mlx_lm
  * warm prefill (TTFT)       : streaming time-to-first-token
  * decode tok/s (ground truth): non-streaming usage.completion_tokens / latency
  * RSS (MB)                   : idle, peak-during-generation (via ps)

Streaming is used ONLY for TTFT; token counts come from the server's own
usage block (non-streaming), which is the authoritative token count. oMLX
emits reasoning_content + keepalive pseudo-chunks, so counting SSE chunks as
tokens is invalid.

Usage:
  python bench_runner.py --base http://127.0.0.1:8011 --key omlx-api-key \
      --model Qwen3.5-9B-MLX-8bit --out /tmp/results_omlx.json --cold
  (caller restarts the server before a --cold run to guarantee a fresh load)
"""
import argparse
import json
import subprocess
import time

import httpx


PROMPT = (
    "The history of maritime navigation is a story of incremental mastery over "
    "uncertainty. Early Polynesian voyagers crossed thousands of kilometres of open "
    "ocean using only the stars, swell patterns, and the flight of birds as guides, "
    "developing a spatial memory of the Pacific that modern charts still struggle to "
    "capture. Centuries later, Mediterranean sailors refined the astrolabe and the "
    "quadrant, while Arabian cartographers preserved and extended Greek and Indian "
    "knowledge through a network of ports stretching from Gibraltar to the spice "
    "islands. The Age of Sail turned navigation into a discipline of logarithms and "
    "lunar distances, letting captains fix their position far from any shore. Each "
    "leap reduced the ocean's indifference by a measurable margin, and each generation "
    "of instruments made the invisible geometry of the globe a little more legible to "
    "those willing to trust the mathematics. Write a thoughtful continuation of this "
    "essay, keeping the same measured tone, for roughly two paragraphs."
)


def rss_mb(pid):
    if not pid:
        return None
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return int(out) // 1024 if out else None
    except Exception:
        return None


def find_pid(port):
    try:
        out = subprocess.check_output(
            ["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"], text=True
        ).strip()
        pids = out.split()
        return int(pids[0]) if pids else None
    except Exception:
        return None


def stream_ttft(base, key, model, prompt, max_tokens, pid, timeout=240):
    """Return (ttft_s, peak_rss_mb) — streaming used only for first-token time."""
    url = f"{base}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "stream": True}
    t0 = time.monotonic()
    first_t = None
    peak = rss_mb(pid)
    with httpx.stream("POST", url, json=payload, timeout=timeout, headers=headers) as r:
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            now = time.monotonic()
            if first_t is None:
                first_t = now
            cur = rss_mb(pid)
            if cur and (peak is None or cur > peak):
                peak = cur
    return (round(first_t - t0, 3) if first_t else None, peak)


def nonstream(base, key, model, prompt, max_tokens, pid, timeout=240):
    """Return dict with latency, completion_tokens, tok/s (ground truth), load_dur, peak_rss."""
    url = f"{base}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "stream": False}
    peak = rss_mb(pid)
    t0 = time.monotonic()
    with httpx.stream("POST", url, json=payload, timeout=timeout, headers=headers) as r:
        body = r.read()
    elapsed = time.monotonic() - t0
    cur = rss_mb(pid)
    if cur and (peak is None or cur > peak):
        peak = cur
    try:
        j = json.loads(body)
        usage = j.get("usage", {}) or {}
        comp = usage.get("completion_tokens")
        load_dur = usage.get("model_load_duration")
    except Exception:
        comp, load_dur = None, None
    tok_s = (comp / elapsed) if (comp and elapsed > 0) else None
    return {
        "latency_s": round(elapsed, 3),
        "completion_tokens": comp,
        "avg_tok_s": round(tok_s, 2) if tok_s else None,
        "model_load_duration_s": load_dur,
        "gen_peak_rss_mb": peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cold", action="store_true",
                    help="first request after a fresh server start (measures load)")
    args = ap.parse_args()

    port = int(args.base.rsplit(":", 1)[1].rstrip("/"))
    pid = find_pid(port)
    idle_rss = rss_mb(pid)

    if args.cold:
        # Caller restarted the server; first non-streaming request carries the
        # cold model_load_duration (oMLX) and TTFT≈load+prefill.
        cold_ns = nonstream(args.base, args.key, args.model, PROMPT, args.max_tokens, pid)
        cold_ttft, _ = stream_ttft(args.base, args.key, args.model, PROMPT, args.max_tokens, pid)
        cold = {"load_plus_prefill_ttft_s": cold_ttft, **cold_ns}
    else:
        cold = None

    # WARM: prefill (TTFT) + decode tok/s (ground truth, non-streaming)
    warm_ttft, warm_peak = stream_ttft(args.base, args.key, args.model, PROMPT, args.max_tokens, pid)
    warm_ns = nonstream(args.base, args.key, args.model, PROMPT, args.max_tokens, pid)

    result = {
        "base": args.base, "model": args.model, "max_tokens": args.max_tokens,
        "pid": pid, "idle_rss_mb": idle_rss,
        "cold": cold,
        "warm": {
            "prefill_ttft_s": warm_ttft,
            "decode_tok_s": warm_ns["avg_tok_s"],
            "completion_tokens": warm_ns["completion_tokens"],
            "latency_s": warm_ns["latency_s"],
            "gen_peak_rss_mb": max(warm_peak or 0, warm_ns["gen_peak_rss_mb"] or 0),
        },
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
