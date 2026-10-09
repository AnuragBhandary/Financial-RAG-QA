"""Wires settings into retrievers, the reranker and the LLM. Everything else depends on this."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import psycopg

from finrag import db
from finrag.answer import Answer, answer
from finrag.config import Settings
from finrag.data import Doc
from finrag.llm import LLM, make_llm
from finrag.models import (
    CrossEncoderReranker,
    Embedder,
    Reranker,
    SentenceTransformerEmbedder,
)
from finrag.retrieval.base import Hit, Retriever
from finrag.retrieval.bm25 import Bm25Retriever
from finrag.retrieval.fusion import HybridRetriever, RerankRetriever
from finrag.retrieval.pg import DenseRetriever, PgFtsRetriever

# Retrieval defaults, chosen on the dev split (see `finrag tune`), never on test.
DEFAULT_DEPTH = 100
DEFAULT_RRF_K = 10
DEFAULT_RRF_WEIGHTS = (1.0, 2.0)  # (bm25, dense): BM25 is much weaker on FiQA
DEFAULT_RERANK_DEPTH = 30
# RRF of the hybrid order and the cross-encoder order, cross-encoder weighted 0.5. On FiQA the
# cross-encoder alone is worse than the hybrid ranking it reorders (docs/RESULTS.md).
DEFAULT_RERANK_FUSE: float | None = 0.5
CONTEXT_PASSAGES = 5


@dataclass
class Timings:
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0


@dataclass
class RagResult:
    question: str
    hits: list[Hit]
    passages: list[Doc]
    answer: Answer | None = None
    timings: Timings = field(default_factory=Timings)


class RagService:
    def __init__(
        self,
        settings: Settings,
        conn: psycopg.Connection[tuple[object, ...]] | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        ft_reranker: Reranker | None = None,
        llm: LLM | None = None,
        bm25: Bm25Retriever | None = None,
    ) -> None:
        self.settings = settings
        self._conn = conn
        self._embedder = embedder
        self._reranker = reranker
        self._ft_reranker = ft_reranker
        self._llm = llm
        self._bm25 = bm25

    @property
    def conn(self) -> psycopg.Connection[tuple[object, ...]]:
        if self._conn is None:
            self._conn = db.connect(self.settings.database_url)
        return self._conn

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(
                self.settings.resolved_embed_model(), self.settings.device
            )
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.settings.rerank_model, self.settings.device)
        return self._reranker

    @property
    def ft_reranker(self) -> Reranker:
        """The reranker fine-tuned on FiQA train, kept for the comparison in RESULTS.md."""
        if self._ft_reranker is None:
            self._ft_reranker = CrossEncoderReranker(
                self.settings.ft_rerank_model, self.settings.device
            )
        return self._ft_reranker

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = make_llm(self.settings)
        return self._llm

    @property
    def bm25(self) -> Bm25Retriever:
        if self._bm25 is None:
            self._bm25 = Bm25Retriever.load(self.settings.bm25_dir)
        return self._bm25

    def fetch(self, ids: Sequence[str]) -> dict[str, Doc]:
        return db.fetch_docs(self.conn, ids)

    @cached_property
    def dense(self) -> DenseRetriever:
        indexed = db.index_meta(self.conn).get("embed_model")
        if indexed != self.embedder.name:
            # Query vectors from one model against document vectors from another are noise.
            raise RuntimeError(
                f"index was built with {indexed!r} but queries use {self.embedder.name!r}; "
                "run `finrag ingest`"
            )
        return DenseRetriever(self.conn, self.embedder)

    def retriever(self, mode: str) -> Retriever:
        """Base: bm25, pg-fts, dense, dense-exact or hybrid. Optional suffix:
        "+rerank" (cross-encoder fused with the base order, the default pipeline),
        "+rerank-only" (cross-encoder order alone), "+rerank-ft" (fine-tuned cross-encoder, fused).
        """
        base_name, _, rerank = mode.partition("+")
        base: Retriever
        if base_name == "bm25":
            base = self.bm25
        elif base_name == "pg-fts":
            base = PgFtsRetriever(self.conn)
        elif base_name == "dense":
            base = self.dense
        elif base_name == "dense-exact":
            base = DenseRetriever(self.conn, self.embedder, exact=True)
        elif base_name == "hybrid":
            base = HybridRetriever(
                self.bm25, self.dense, DEFAULT_DEPTH, DEFAULT_RRF_K, DEFAULT_RRF_WEIGHTS
            )
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        if rerank == "rerank":
            r = RerankRetriever(
                base, self.reranker, self.fetch, DEFAULT_RERANK_DEPTH, DEFAULT_RERANK_FUSE
            )
        elif rerank == "rerank-only":
            r = RerankRetriever(base, self.reranker, self.fetch, DEFAULT_RERANK_DEPTH, None)
        elif rerank == "rerank-ft":
            r = RerankRetriever(
                base, self.ft_reranker, self.fetch, DEFAULT_RERANK_DEPTH, DEFAULT_RERANK_FUSE
            )
        elif rerank:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        else:
            return base
        r.name = mode
        return r

    def retrieve(self, question: str, mode: str = "hybrid+rerank", k: int = 10) -> RagResult:
        timings = Timings()
        retriever = self.retriever(mode)
        if isinstance(retriever, RerankRetriever):
            t0 = time.perf_counter()
            candidates = retriever.base.search(question, max(k, retriever.depth))
            t1 = time.perf_counter()
            hits = retriever.rerank(question, candidates)[:k]
            timings.retrieve_ms = (t1 - t0) * 1000
            timings.rerank_ms = (time.perf_counter() - t1) * 1000
        else:
            t0 = time.perf_counter()
            hits = retriever.search(question, k)
            timings.retrieve_ms = (time.perf_counter() - t0) * 1000
        docs = self.fetch([h.doc_id for h in hits])
        return RagResult(
            question=question,
            hits=hits,
            passages=[docs[h.doc_id] for h in hits if h.doc_id in docs],
            timings=timings,
        )

    def ask(
        self, question: str, mode: str = "hybrid+rerank", n_passages: int = CONTEXT_PASSAGES
    ) -> RagResult:
        result = self.retrieve(question, mode, k=max(10, n_passages))
        result.answer = answer(self.llm, question, result.passages[:n_passages])
        return result
