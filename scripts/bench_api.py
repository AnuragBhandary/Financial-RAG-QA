"""End-to-end latency of POST /ask as a client sees it: time to the passages event, to the first
answer token, and to the last byte. Sequential requests against a running server (`make serve`).

    uv run python scripts/bench_api.py --n 60
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import httpx

from finrag.config import Settings
from finrag.data import split_queries
from finrag.evaluate import percentile


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    s = Settings()
    queries, _ = split_queries(s.fiqa_dir, "test")
    # A different seed from the answer-quality sample, so the LLM's prompt cache cannot help.
    qs = random.Random(99).sample(sorted(queries.values()), args.n + args.warmup)
    rows = []
    with httpx.Client(base_url=args.url, timeout=120) as client:
        for i, q in enumerate(qs):
            t0 = time.perf_counter()
            t_passages = t_token = None
            server: dict[str, float] = {}
            with client.stream("POST", "/ask", json={"question": q}) as r:
                r.raise_for_status()
                event = None
                for line in r.iter_lines():
                    now = time.perf_counter()
                    if line.startswith("event: "):
                        event = line[7:]
                        if event == "passages" and t_passages is None:
                            t_passages = now
                        if event == "token" and t_token is None:
                            t_token = now
                    elif line.startswith("data: ") and event == "done":
                        server = json.loads(line[6:])["timings_ms"]
            total = time.perf_counter() - t0
            if i < args.warmup:
                continue
            rows.append(
                {
                    "passages_ms": 1000 * ((t_passages or t0) - t0),
                    "first_token_ms": 1000 * ((t_token or t0) - t0),
                    "total_ms": 1000 * total,
                    "server_retrieval_ms": server.get("retrieval_total", 0.0),
                }
            )
    out = {"requests": len(rows)}
    for key in ("passages_ms", "first_token_ms", "total_ms", "server_retrieval_ms"):
        vals = [r[key] for r in rows]
        out[key] = {
            "p50": round(percentile(vals, 50)),
            "p95": round(percentile(vals, 95)),
            "max": round(max(vals)),
        }
    print(json.dumps(out, indent=2))
    Path(s.runs_dir, "latency-api.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
