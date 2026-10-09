"""Embedding and reranking models behind small protocols.

The real ones (sentence-transformers) load lazily so importing finrag never needs torch; the
hashing fakes give tests and CI deterministic, lexical-ish vectors without downloading anything.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]

# bge v1.5 was trained with this instruction on the query side only (documents get none).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed_queries(self, texts: Sequence[str]) -> Vectors: ...

    def embed_docs(self, texts: Sequence[str], batch_size: int = 64) -> Vectors: ...


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


def pick_device(requested: str) -> str:  # pragma: no cover - needs torch
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder:  # pragma: no cover - downloads model weights
    def __init__(
        self, model_name: str, device: str = "auto", max_seq_length: int = 512, fp16: bool = True
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        dev = pick_device(device)
        self._model: Any = SentenceTransformer(model_name, device=dev)
        self._model.max_seq_length = max_seq_length
        if fp16 and dev == "mps":
            self._model.half()  # 3.3x the docs/s of fp32 on Apple GPUs (docs/RESULTS.md)
        self.dim = int(self._model.get_embedding_dimension() or 0)
        self._query_prefix = BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        prefixed = [self._query_prefix + t for t in texts]
        return self._encode(prefixed, batch_size=64)

    def embed_docs(self, texts: Sequence[str], batch_size: int = 64) -> Vectors:
        return self._encode(list(texts), batch_size=batch_size)

    def _encode(self, texts: list[str], batch_size: int) -> Vectors:
        vecs = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)


class CrossEncoderReranker:  # pragma: no cover - downloads model weights
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 256) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.name = model_name
        dev = pick_device(device)
        # fp16 on Apple GPUs: about 3x the pairs/s of fp32 (docs/RESULTS.md)
        kwargs = {"torch_dtype": torch.float16} if dev == "mps" else {}
        self._model: Any = CrossEncoder(
            model_name, device=dev, max_length=max_length, model_kwargs=kwargs
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict([(query, p) for p in passages], batch_size=32)
        return [float(s) for s in scores]


_WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class HashingEmbedder:
    """Feature-hashed bag of words, L2-normalised. Deterministic; for tests and CI only."""

    def __init__(self, dim: int = 64) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _one(self, text: str) -> npt.NDArray[np.float32]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokens(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._one(t) for t in texts])

    def embed_docs(self, texts: Sequence[str], batch_size: int = 64) -> Vectors:
        return self.embed_queries(texts)


class OverlapReranker:
    """Fraction of query words present in the passage. For tests and CI only."""

    name = "overlap"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(tokens(query))
        return [len(q.intersection(tokens(p))) / (len(q) or 1) for p in passages]
