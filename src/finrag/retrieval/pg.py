"""Retrievers that run inside PostgreSQL: pgvector cosine search and built-in full-text search."""

from __future__ import annotations

from collections.abc import Mapping

import psycopg

from finrag.models import Embedder
from finrag.retrieval.base import Hit

Conn = psycopg.Connection[tuple[object, ...]]


class DenseRetriever:
    """Cosine kNN with pgvector. `exact=True` disables the HNSW index (the ground truth for ANN)."""

    def __init__(
        self, conn: Conn, embedder: Embedder, ef_search: int = 200, exact: bool = False
    ) -> None:
        self.name = "dense-exact" if exact else "dense"
        self._conn = conn
        self._embedder = embedder
        self._ef_search = ef_search
        self._exact = exact

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_many({"q": query}, k)["q"]

    def search_many(self, queries: Mapping[str, str], k: int) -> dict[str, list[Hit]]:
        if not queries:
            return {}
        qids = list(queries)
        vecs = self._embedder.embed_queries([queries[q] for q in qids])
        out: dict[str, list[Hit]] = {}
        with self._conn.transaction():
            # HNSW returns at most ef_search candidates, so it must be >= k.
            self._conn.execute(f"SET LOCAL hnsw.ef_search = {max(self._ef_search, k)}")
            if self._exact:
                self._conn.execute("SET LOCAL enable_indexscan = off")
            for qid, vec in zip(qids, vecs, strict=True):
                out[qid] = self.search_vector(vec, k)
        return out

    def search_vector(self, vec: object, k: int) -> list[Hit]:
        rows = self._conn.execute(
            "SELECT id, 1 - (embedding <=> %s) AS sim FROM docs ORDER BY embedding <=> %s LIMIT %s",
            (vec, vec, k),
        ).fetchall()
        return [Hit(str(r[0]), float(r[1])) for r in rows]  # type: ignore[arg-type]


class PgFtsRetriever:
    """Postgres full-text search with OR semantics over the query lexemes, ranked by ts_rank
    normalised by log document length.

    plainto_tsquery ANDs every word, which almost never matches a whole natural-language question,
    so the lexemes are re-joined with | before matching. ts_rank_cd (cover density) was tried
    first and scored 0.037 nDCG@10 on dev: with OR queries it rewards long posts that repeat
    common words. ts_rank(..., 1) scored 0.190 on the same queries.
    """

    name = "pg-fts"

    def __init__(self, conn: Conn) -> None:
        self._conn = conn

    def search(self, query: str, k: int) -> list[Hit]:
        rows = self._conn.execute(
            """
            WITH q AS (
                SELECT to_tsquery('english',
                    replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')) AS q
            )
            SELECT id, ts_rank(tsv, q.q, 1) AS score
            FROM docs, q
            WHERE q.q::text <> '' AND tsv @@ q.q
            ORDER BY score DESC, id
            LIMIT %s
            """,
            (query, k),
        ).fetchall()
        return [Hit(str(r[0]), float(r[1])) for r in rows]  # type: ignore[arg-type]

    def search_many(self, queries: Mapping[str, str], k: int) -> dict[str, list[Hit]]:
        return {qid: self.search(text, k) for qid, text in queries.items()}
