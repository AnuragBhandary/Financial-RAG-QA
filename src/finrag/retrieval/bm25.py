"""Okapi BM25 over title + text (bm25s: sparse matrices, so all of FiQA scores in milliseconds).

Postgres' ts_rank is not BM25 (no IDF, no length saturation), so the lexical baseline lives here;
the Postgres full-text retriever is kept as its own variant to show the difference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
import Stemmer

from finrag.data import Doc
from finrag.retrieval.base import Hit


def _stemmer() -> Any:
    return Stemmer.Stemmer("english")


def _tokenize(texts: Sequence[str]) -> Any:
    return bm25s.tokenize(list(texts), stopwords="en", stemmer=_stemmer(), show_progress=False)


class Bm25Retriever:
    name = "bm25"

    def __init__(self, model: Any, doc_ids: list[str]) -> None:
        self._model = model
        self._doc_ids = doc_ids

    @classmethod
    def build(cls, docs: Sequence[Doc], k1: float = 1.2, b: float = 0.75) -> Bm25Retriever:
        model = bm25s.BM25(k1=k1, b=b)
        model.index(_tokenize([f"{d.title} {d.text}".strip() for d in docs]), show_progress=False)
        return cls(model, [d.id for d in docs])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        (path / "doc_ids.json").write_text(json.dumps(self._doc_ids))

    @classmethod
    def load(cls, path: Path) -> Bm25Retriever:
        model = bm25s.BM25.load(str(path))
        doc_ids = json.loads((path / "doc_ids.json").read_text())
        return cls(model, doc_ids)

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_many({"q": query}, k)["q"]

    def search_many(self, queries: Mapping[str, str], k: int) -> dict[str, list[Hit]]:
        if not queries:
            return {}
        qids = list(queries)
        k = min(k, len(self._doc_ids))
        idx, scores = self._model.retrieve(
            _tokenize([queries[q] for q in qids]), k=k, show_progress=False, n_threads=1
        )
        out: dict[str, list[Hit]] = {}
        for row, qid in enumerate(qids):
            hits = [
                Hit(self._doc_ids[int(i)], float(s))
                for i, s in zip(np.asarray(idx[row]), np.asarray(scores[row]), strict=True)
                if s > 0  # a query with no matching term returns arbitrary zero-score docs
            ]
            out[qid] = hits
        return out
