"""Hybrid retrieval (reciprocal rank fusion) and cross-encoder reranking."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from finrag.data import Doc
from finrag.models import Reranker
from finrag.retrieval.base import Hit, Retriever


def rrf(
    rankings: Sequence[Sequence[Hit]], k: int = 60, weights: Sequence[float] | None = None
) -> list[Hit]:
    """score(d) = sum_i w_i / (k + rank_i(d)), rank from 1.

    Uses ranks, not scores, so BM25 scores (unbounded) and cosine similarities (-1..1) never
    have to be put on one scale. Ties break by doc id so runs are reproducible.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for w, ranking in zip(weights, rankings, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + w / (k + rank)
    return [Hit(d, s) for d, s in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


class HybridRetriever:
    """BM25 + dense, each to `depth`, fused with RRF."""

    def __init__(
        self,
        lexical: Retriever,
        dense: Retriever,
        depth: int = 100,
        rrf_k: int = 60,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.name = "hybrid"
        self.lexical = lexical
        self.dense = dense
        self.depth = depth
        self.rrf_k = rrf_k
        self.weights = weights

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_many({"q": query}, k)["q"]

    def search_many(self, queries: Mapping[str, str], k: int) -> dict[str, list[Hit]]:
        depth = max(self.depth, k)
        lex = self.lexical.search_many(queries, depth)
        den = self.dense.search_many(queries, depth)
        return {
            qid: rrf([lex[qid], den[qid]], self.rrf_k, list(self.weights))[:k] for qid in queries
        }


DocFetcher = Callable[[Sequence[str]], Mapping[str, Doc]]


def passage_text(doc: Doc) -> str:
    return f"{doc.title}\n{doc.text}".strip()


def order_head(
    head: Sequence[Hit],
    scores: Sequence[float],
    fuse_weight: float | None = None,
    rrf_k: int = 60,
) -> list[Hit]:
    """Order the reranked head: by cross-encoder score alone, or (fuse_weight set) by RRF of the
    first-stage order and the cross-encoder order, which keeps first-stage signal the
    cross-encoder lacks."""
    ranked = sorted(
        (Hit(h.doc_id, s) for h, s in zip(head, scores, strict=True)),
        key=lambda h: (-h.score, h.doc_id),
    )
    if fuse_weight is None:
        return ranked
    return rrf([list(head), ranked], rrf_k, [1.0, fuse_weight])


class RerankRetriever:
    """Re-scores the first `depth` candidates of a base retriever with a cross-encoder.

    Below `depth` the base order is kept, so recall@100 is unchanged and only the top moves.
    """

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        fetch: DocFetcher,
        depth: int = 50,
        fuse_weight: float | None = None,
    ) -> None:
        self.name = f"{base.name}+rerank"
        self.base = base
        self.reranker = reranker
        self.fetch = fetch
        self.depth = depth
        self.fuse_weight = fuse_weight

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_many({"q": query}, k)["q"]

    def search_many(self, queries: Mapping[str, str], k: int) -> dict[str, list[Hit]]:
        base = self.base.search_many(queries, max(k, self.depth))
        return {qid: self.rerank(queries[qid], base[qid])[:k] for qid in queries}

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        head, tail = list(hits[: self.depth]), list(hits[self.depth :])
        docs = self.fetch([h.doc_id for h in head])
        head = [h for h in head if h.doc_id in docs]
        scores = self.reranker.score(query, [passage_text(docs[h.doc_id]) for h in head])
        ranked = order_head(head, scores, self.fuse_weight)
        # Tail keeps base order; its scores are on another scale, so they are pushed below.
        floor = min((h.score for h in ranked), default=0.0) - 1.0
        return ranked + [Hit(h.doc_id, floor - i) for i, h in enumerate(tail)]
