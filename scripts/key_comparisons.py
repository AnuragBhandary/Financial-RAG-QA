"""Paired significance for the comparisons RESULTS.md quotes, from saved per-query scores.

Needs runs/retrieval-test-per-query.json (finrag eval-retrieval --split test) and the cached
fp16 corpus embeddings of the off-the-shelf model (finrag compare-embedders --split test).

    uv run python scripts/key_comparisons.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from finrag import metrics
from finrag.config import Settings
from finrag.data import iter_corpus, split_queries
from finrag.ingest import embedding_cache_path
from finrag.models import SentenceTransformerEmbedder

PAIRS = [
    ("hybrid+rerank", "bm25"),
    ("hybrid+rerank", "dense-offtheshelf"),
    ("hybrid+rerank", "dense"),
    ("hybrid+rerank", "hybrid"),
    ("hybrid+rerank", "hybrid+rerank-ft"),
    ("dense", "dense-offtheshelf"),
    ("hybrid", "dense"),
    ("hybrid+rerank-only", "hybrid"),
]


def main() -> None:
    s = Settings()
    runs = Path(s.runs_dir)
    per_query = json.loads((runs / "retrieval-test-per-query.json").read_text())
    queries, qrels = split_queries(s.fiqa_dir, "test")

    # The off-the-shelf embedder is not in the database any more; rank exactly in memory.
    base = s.base_embed_model
    ids = [d.id for d in iter_corpus(s.fiqa_dir)]
    doc_vecs = np.load(embedding_cache_path(s, f"{base}-fp16", len(ids)))
    q_vecs = SentenceTransformerEmbedder(base, s.device).embed_queries(list(queries.values()))
    sims = q_vecs @ doc_vecs.T
    run = {qid: [ids[i] for i in np.argsort(-sims[r])[:100]] for r, qid in enumerate(queries)}
    per_query["dense-offtheshelf"] = metrics.per_query_scores(run, qrels)

    qids = sorted(qrels)
    out = []
    for a, b in PAIRS:
        x = [per_query[a]["ndcg@10"][q] for q in qids]
        y = [per_query[b]["ndcg@10"][q] for q in qids]
        row = {
            "variant": a,
            "baseline": b,
            "variant_ndcg@10": round(metrics.mean(x), 4),
            "baseline_ndcg@10": round(metrics.mean(y), 4),
            "delta": round(metrics.mean(x) - metrics.mean(y), 4),
            "relative": f"{100 * (metrics.mean(x) / metrics.mean(y) - 1):+.1f}%",
            "p_value": round(metrics.paired_randomization_test(x, y), 4),
            "wins": sum(i > j for i, j in zip(x, y, strict=True)),
            "losses": sum(i < j for i, j in zip(x, y, strict=True)),
        }
        out.append(row)
        print(row)
    (runs / "key-comparisons-test.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
