"""What every retriever returns, and the batch helper the evaluation harness uses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...

    def search_many(self, queries: Mapping[str, str], k: int) -> dict[str, list[Hit]]: ...


def run_ids(results: Mapping[str, list[Hit]]) -> dict[str, list[str]]:
    """Strip scores: the metrics only look at order."""
    return {qid: [h.doc_id for h in hits] for qid, hits in results.items()}
