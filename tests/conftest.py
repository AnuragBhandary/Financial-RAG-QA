from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import psycopg
import pytest

from finrag import db
from finrag.config import Settings
from finrag.data import Doc
from finrag.llm import Completion, Message, Usage
from finrag.models import HashingEmbedder, OverlapReranker
from finrag.retrieval.bm25 import Bm25Retriever
from finrag.service import RagService

TEST_DB_URL = os.environ.get(
    "FINRAG_TEST_DATABASE_URL", "postgresql://finrag:finrag@127.0.0.1:5436/finrag_test"
)

CORPUS = [
    Doc(
        "d1",
        "",
        "A Roth IRA is funded with after-tax money and qualified withdrawals are tax free.",
    ),
    Doc("d2", "", "A traditional IRA gives a tax deduction now; withdrawals are taxed as income."),
    Doc("d3", "", "Index funds track a market index and usually have very low expense ratios."),
    Doc("d4", "", "An ETF trades on an exchange like a stock during the day."),
    Doc(
        "d5", "", "Car insurance deductibles for an accident on a business trip may be deductible."
    ),
    Doc("d6", "", "Dollar cost averaging means investing a fixed amount at regular intervals."),
    Doc("d7", "", "A credit score is affected by payment history and credit utilization."),
    Doc("d8", "", "Paying off a mortgage early saves interest but reduces liquidity."),
    Doc("d9", "", "Capital gains on stocks held over a year are taxed at long-term rates."),
    Doc("d10", "", "An emergency fund should cover three to six months of expenses."),
]


class FakeLLM:
    """Replays scripted responses; a callable response sees the messages."""

    def __init__(self, responses: Sequence[str | Callable[[list[Message]], str]] = ()) -> None:
        self.model = "fake"
        self.responses = list(responses)
        self.calls: list[list[Message]] = []
        self.schemas: list[dict[str, Any] | None] = []

    def _next(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeLLM ran out of responses")
        r = self.responses.pop(0)
        return r(messages) if callable(r) else r

    def complete(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        self.schemas.append(schema)
        return Completion(
            self._next(messages), Usage(calls=1, prompt_tokens=10, completion_tokens=5)
        )

    def stream(
        self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> Iterator[str]:
        text = self._next(messages)
        for i in range(0, len(text), 7):
            yield text[i : i + 7]


@pytest.fixture
def fake_llm() -> Callable[..., FakeLLM]:
    return FakeLLM


def _ensure_test_db() -> None:
    admin = TEST_DB_URL.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as c:
            exists = c.execute("SELECT 1 FROM pg_database WHERE datname = 'finrag_test'").fetchone()
            if not exists:
                c.execute("CREATE DATABASE finrag_test")
    except psycopg.OperationalError as e:
        pytest.skip(f"PostgreSQL not reachable ({e}); run `make infra`")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    fiqa = tmp_path / "data" / "fiqa"
    (fiqa / "qrels").mkdir(parents=True)
    with (fiqa / "corpus.jsonl").open("w") as f:
        for d in CORPUS:
            f.write(json.dumps({"_id": d.id, "title": d.title, "text": d.text}) + "\n")
    queries = {
        "q1": "roth ira withdrawals tax",
        "q2": "what is an etf",
        "q3": "emergency fund size",
    }
    with (fiqa / "queries.jsonl").open("w") as f:
        for qid, text in queries.items():
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")
    qrels = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td4\t1\nq3\td10\t1\n"
    for split in ("dev", "test"):
        (fiqa / "qrels" / f"{split}.tsv").write_text(qrels)
    return Settings(
        database_url=TEST_DB_URL,
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
    )


@pytest.fixture
def pg_service(settings: Settings) -> Iterator[RagService]:
    """A RagService on the real pgvector database with hashing embeddings and fakes."""
    _ensure_test_db()
    embedder = HashingEmbedder(64)
    conn = db.connect(TEST_DB_URL)
    db.load_corpus(conn, CORPUS, embedder.embed_docs([d.text for d in CORPUS]), embedder.name)
    svc = RagService(
        settings,
        conn=conn,
        embedder=embedder,
        reranker=OverlapReranker(),
        ft_reranker=OverlapReranker(),
        llm=FakeLLM(),
        bm25=Bm25Retriever.build(CORPUS),
    )
    yield svc
    conn.close()
