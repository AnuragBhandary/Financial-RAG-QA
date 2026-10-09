"""Corpus -> embeddings (cached on disk) -> PostgreSQL/pgvector, plus the BM25 index."""

from __future__ import annotations

import re
import time
from collections.abc import Callable

import numpy as np

from finrag import db
from finrag.config import Settings
from finrag.data import Doc, iter_corpus
from finrag.models import Embedder
from finrag.retrieval.bm25 import Bm25Retriever

Log = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


def embedding_cache_path(settings: Settings, model_name: str, n_docs: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").lower()
    return str(settings.data_dir / f"embeddings-{slug}-{n_docs}.npy")


def embed_corpus(
    embedder: Embedder, docs: list[Doc], log: Log, chunk: int = 4096, batch_size: int = 64
) -> np.ndarray:
    parts = []
    t0 = time.perf_counter()
    for start in range(0, len(docs), chunk):
        batch = docs[start : start + chunk]
        parts.append(
            embedder.embed_docs([f"{d.title} {d.text}".strip() for d in batch], batch_size)
        )
        done = start + len(batch)
        rate = done / (time.perf_counter() - t0)
        log(f"  embedded {done:,}/{len(docs):,} ({rate:,.0f} docs/s)")
    return np.concatenate(parts).astype(np.float32)


def ingest(
    settings: Settings, embedder: Embedder, log: Log = _print, limit: int | None = None
) -> dict[str, float]:
    t_start = time.perf_counter()
    docs = list(iter_corpus(settings.fiqa_dir))
    if limit is not None:
        docs = docs[:limit]
    log(f"corpus: {len(docs):,} docs")

    cache = embedding_cache_path(settings, embedder.name, len(docs))
    t0 = time.perf_counter()
    try:
        vecs = np.load(cache)
        log(f"embeddings: loaded cache {cache}")
    except FileNotFoundError:
        vecs = embed_corpus(embedder, docs, log)
        np.save(cache, vecs)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25 = Bm25Retriever.build(docs)
    bm25.save(settings.bm25_dir)
    t_bm25 = time.perf_counter() - t0
    log(f"bm25: indexed in {t_bm25:.1f} s")

    t0 = time.perf_counter()
    with db.connect(settings.database_url) as conn:
        db.load_corpus(conn, docs, vecs, embedder.name)
    t_pg = time.perf_counter() - t0
    log(f"postgres: loaded + HNSW/GIN built in {t_pg:.1f} s")
    return {
        "docs": len(docs),
        "embed_s": round(t_embed, 1),
        "bm25_s": round(t_bm25, 1),
        "pg_load_and_index_s": round(t_pg, 1),
        "total_s": round(time.perf_counter() - t_start, 1),
    }
