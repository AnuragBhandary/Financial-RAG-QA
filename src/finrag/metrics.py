"""Ranking metrics (same definitions as trec_eval / BEIR) plus paired statistics.

Every function works on one query's ranked list so per-query scores can be kept for paired
comparisons between retrievers; the mean over queries is the reported number.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

Ranking = Sequence[str]
Judgments = dict[str, int]


def ndcg_at_k(ranking: Ranking, rels: Judgments, k: int = 10) -> float:
    """Linear gain (trec_eval's default, which BEIR reports): DCG = sum rel / log2(rank + 1)."""
    dcg = sum(rels.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranking[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranking: Ranking, rels: Judgments, k: int) -> float:
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def mrr_at_k(ranking: Ranking, rels: Judgments, k: int = 10) -> float:
    for i, doc in enumerate(ranking[:k]):
        if rels.get(doc, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def per_query_scores(
    run: dict[str, list[str]], qrels: dict[str, Judgments]
) -> dict[str, dict[str, float]]:
    """metric name -> {query id -> score}. Queries missing from the run score 0."""
    out: dict[str, dict[str, float]] = {
        "ndcg@10": {},
        "mrr@10": {},
        "recall@10": {},
        "recall@100": {},
    }
    for qid, rels in qrels.items():
        ranking = run.get(qid, [])
        out["ndcg@10"][qid] = ndcg_at_k(ranking, rels, 10)
        out["mrr@10"][qid] = mrr_at_k(ranking, rels, 10)
        out["recall@10"][qid] = recall_at_k(ranking, rels, 10)
        out["recall@100"][qid] = recall_at_k(ranking, rels, 100)
    return out


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 7
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over queries."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples) - 1]
    return lo, hi


def paired_randomization_test(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 10000, seed: int = 7
) -> float:
    """Two-sided p-value for mean(a) != mean(b) on the same queries (sign-flip permutation)."""
    if len(a) != len(b):
        raise ValueError("paired test needs the same queries in the same order")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    observed = abs(mean(diffs))
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_resamples):
        flipped = mean([d if rng.random() < 0.5 else -d for d in diffs])
        if abs(flipped) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_resamples + 1)
