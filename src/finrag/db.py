"""PostgreSQL + pgvector: schema, bulk load, and document lookup.

One table holds everything a query needs: the text, a generated tsvector for Postgres full-text
search, and the embedding with an HNSW index for approximate cosine search.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from finrag.data import Doc


def connect(url: str, autocommit: bool = True) -> psycopg.Connection[tuple[object, ...]]:
    conn = psycopg.connect(url, autocommit=autocommit)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def schema_sql(dim: int) -> str:
    return f"""
    CREATE TABLE docs (
        id        text PRIMARY KEY,
        title     text NOT NULL DEFAULT '',
        body      text NOT NULL,
        tsv       tsvector GENERATED ALWAYS AS
                    (to_tsvector('english', title || ' ' || body)) STORED,
        embedding vector({dim}) NOT NULL
    );
    CREATE TABLE index_meta (
        key   text PRIMARY KEY,
        value text NOT NULL
    );
    """


INDEX_SQL = """
CREATE INDEX docs_tsv_gin ON docs USING gin (tsv);
CREATE INDEX docs_embedding_hnsw ON docs
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
"""


def load_corpus(
    conn: psycopg.Connection[tuple[object, ...]],
    docs: Sequence[Doc],
    embeddings: np.ndarray,
    embed_model: str,
) -> None:
    """Replace the whole index in one transaction: a failed load leaves the old index serving."""
    if len(docs) != len(embeddings):
        raise ValueError(f"{len(docs)} docs but {len(embeddings)} embeddings")
    dim = int(embeddings.shape[1])
    with conn.transaction():
        conn.execute("DROP TABLE IF EXISTS docs, index_meta")
        conn.execute(schema_sql(dim))
        with conn.cursor() as cur:
            copy_sql = "COPY docs (id, title, body, embedding) FROM STDIN WITH (FORMAT BINARY)"
            with cur.copy(copy_sql) as copy:
                copy.set_types(["text", "text", "text", "vector"])
                for doc, vec in zip(docs, embeddings, strict=True):
                    copy.write_row([doc.id, doc.title, doc.text, vec])
        # Build indexes after the load: one bulk build is much faster than per-row maintenance.
        conn.execute("SET LOCAL maintenance_work_mem = '1GB'")
        conn.execute(INDEX_SQL)
        conn.execute(
            "INSERT INTO index_meta VALUES ('embed_model', %s), ('dim', %s), ('docs', %s)",
            (embed_model, str(dim), str(len(docs))),
        )
    conn.execute("ANALYZE docs")


def index_meta(conn: psycopg.Connection[tuple[object, ...]]) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM index_meta").fetchall()
    return {str(k): str(v) for k, v in rows}


def fetch_docs(conn: psycopg.Connection[tuple[object, ...]], ids: Iterable[str]) -> dict[str, Doc]:
    ids = list(ids)
    if not ids:
        return {}
    rows = conn.execute("SELECT id, title, body FROM docs WHERE id = ANY(%s)", (ids,)).fetchall()
    return {str(r[0]): Doc(id=str(r[0]), title=str(r[1]), text=str(r[2])) for r in rows}
