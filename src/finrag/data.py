"""BEIR-format loader for FiQA-2018: corpus.jsonl, queries.jsonl, qrels/{train,dev,test}.tsv."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# qrels[query_id][doc_id] = graded relevance (FiQA only uses 1)
Qrels = dict[str, dict[str, int]]


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str


def iter_corpus(fiqa_dir: Path) -> Iterator[Doc]:
    with (fiqa_dir / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield Doc(id=str(row["_id"]), title=row.get("title") or "", text=row["text"])


def load_queries(fiqa_dir: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with (fiqa_dir / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    return queries


def load_qrels(fiqa_dir: Path, split: str) -> Qrels:
    qrels: Qrels = {}
    with (fiqa_dir / "qrels" / f"{split}.tsv").open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id corpus-id score
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def split_queries(fiqa_dir: Path, split: str) -> tuple[dict[str, str], Qrels]:
    """Queries of one split (only those with judgments) and their qrels."""
    queries = load_queries(fiqa_dir)
    qrels = load_qrels(fiqa_dir, split)
    return {qid: queries[qid] for qid in qrels}, qrels
