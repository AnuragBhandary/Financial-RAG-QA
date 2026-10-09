"""Against the real pgvector container (make infra), with hashing embeddings and fake LLMs."""

from __future__ import annotations

import json

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

from finrag import db
from finrag.evaluate import eval_retrieval, generate_answers, judge_answers, summarize_answers
from finrag.service import RagService
from tests.conftest import CORPUS, FakeLLM

pytestmark = pytest.mark.integration


def test_load_builds_indexes_and_meta(pg_service: RagService) -> None:
    conn = pg_service.conn
    assert db.index_meta(conn) == {"embed_model": "hashing-64", "dim": "64", "docs": "10"}
    idx = {r[0] for r in conn.execute("SELECT indexname FROM pg_indexes WHERE tablename='docs'")}
    assert {"docs_tsv_gin", "docs_embedding_hnsw"} <= idx


def test_failed_reload_keeps_the_old_index(pg_service: RagService) -> None:
    conn = pg_service.conn
    with pytest.raises(ValueError):
        db.load_corpus(conn, CORPUS, np.zeros((3, 64), dtype=np.float32), "bad")
    bad = CORPUS[:2]
    with pytest.raises(psycopg.Error):  # zero-dimension vectors fail inside the COPY
        db.load_corpus(conn, bad, np.zeros((2, 64), dtype=np.float32)[:, :0], "bad")
    assert db.index_meta(conn)["docs"] == "10"


def test_dense_ann_matches_exact_on_small_corpus(pg_service: RagService) -> None:
    ann = pg_service.retriever("dense").search("roth ira tax free", 5)
    exact = pg_service.retriever("dense-exact").search("roth ira tax free", 5)
    assert [h.doc_id for h in ann] == [h.doc_id for h in exact]
    assert ann[0].doc_id == "d1"
    assert -1.0 <= ann[0].score <= 1.0


def test_pg_fts_uses_or_semantics(pg_service: RagService) -> None:
    fts = pg_service.retriever("pg-fts")
    # No document contains all of these words; AND semantics would return nothing.
    out = fts.search("roth etf mortgage spaceship", 10)
    assert {"d1", "d4", "d8"} <= {h.doc_id for h in out}
    assert fts.search("the and of", 5) == []  # only stopwords


def test_every_mode_returns_results(pg_service: RagService) -> None:
    for mode in [
        "bm25",
        "pg-fts",
        "dense",
        "hybrid",
        "hybrid+rerank",
        "dense+rerank",
        "hybrid+rerank-only",
        "hybrid+rerank-ft",
    ]:
        res = pg_service.retrieve("what is an etf", mode, k=3)
        assert res.passages, mode
        assert res.passages[0].id == "d4", mode
    with pytest.raises(ValueError):
        pg_service.retriever("hybrid+magic")
    with pytest.raises(ValueError):
        pg_service.retriever("nope")


def test_eval_retrieval_writes_reports(pg_service: RagService) -> None:
    out = pg_service.settings.runs_dir
    report = eval_retrieval(pg_service, "test", ["bm25", "dense", "hybrid+rerank"], out)
    assert report["queries"] == 3
    assert report["variants"]["bm25"]["summary"]["ndcg@10"]["mean"] == 1.0
    assert (out / "run-test-bm25.trec").read_text().startswith("q1 Q0 d1 1 ")
    assert {c["variant"] for c in report["comparisons"]} == {"dense", "hybrid+rerank"}
    per_query = json.loads((out / "retrieval-test-per-query.json").read_text())
    assert set(per_query["dense"]["ndcg@10"]) == {"q1", "q2", "q3"}


def test_answers_judging_and_resume(pg_service: RagService) -> None:
    pg_service._llm = FakeLLM([lambda m: "Grounded claim [1]. Another [2]."] * 3)
    path = pg_service.settings.runs_dir / "answers.jsonl"
    pg_service.settings.runs_dir.mkdir(parents=True, exist_ok=True)
    recs = generate_answers(pg_service, "rag:hybrid+rerank", 3, path)
    assert len(recs) == 3 and all(r["cites_gold"] in (True, False) for r in recs)
    # Resuming makes no new LLM calls (FakeLLM would raise if asked for a 4th response).
    assert len(generate_answers(pg_service, "rag:hybrid+rerank", 3, path)) == 3

    verdict = json.dumps(
        {
            "verdicts": [
                {"sentence": 1, "reason": "ok", "supported": True},
                {"sentence": 2, "reason": "no", "supported": False},
            ]
        }
    )
    judged = judge_answers(pg_service, FakeLLM([verdict] * 3), path, path.with_suffix(".j"))
    s = summarize_answers(judged)
    assert s["sentence_support_rate"] == 0.5
    assert s["mean_answer_faithfulness"] == 0.5
    assert s["context_has_gold"] == 1.0


def test_agent_answers_on_real_retrieval(pg_service: RagService) -> None:
    from finrag.agent import RagAgent

    llm = FakeLLM(
        [
            json.dumps({"queries": ["etf exchange"]}),
            json.dumps({"missing": "", "sufficient": True, "follow_up_query": ""}),
            "An ETF trades like a stock [1].",
            json.dumps({"verdicts": [{"sentence": 1, "reason": "", "supported": True}]}),
        ]
    )
    res = RagAgent(llm, pg_service.retriever("hybrid+rerank"), pg_service.fetch).run(
        "what is an etf"
    )
    assert res.evidence[0].id == "d4"
    assert res.answer.cited_doc_ids == ["d4"]


@pytest.fixture
def client(pg_service: RagService):  # type: ignore[no-untyped-def]
    from finrag import api

    api.state["service"] = pg_service
    with TestClient(api.app) as c:
        yield c
    api.state.pop("service", None)


def test_api_search_and_health(client: TestClient) -> None:
    r = client.get("/search", params={"q": "emergency fund", "mode": "hybrid+rerank", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["doc_id"] == "d10"
    assert set(body["timings_ms"]) == {"retrieve", "rerank"}
    assert client.get("/search", params={"q": "x y", "mode": "nope"}).status_code == 400
    assert client.get("/healthz").json()["index"]["docs"] == "10"


def test_api_ask_streams_sse(client: TestClient, pg_service: RagService) -> None:
    pg_service._llm = FakeLLM(["An ETF trades on an exchange [1]. Bad cite [9]."])
    with client.stream("POST", "/ask", json={"question": "what is an etf"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())
    events = [
        (b.split("\n")[0].removeprefix("event: "), json.loads(b.split("\n")[1][6:]))
        for b in raw.strip().split("\n\n")
    ]
    kinds = [e for e, _ in events]
    assert kinds[0] == "passages" and kinds[-1] == "done" and "token" in kinds
    text = "".join(d for e, d in events if e == "token")
    assert text == "An ETF trades on an exchange [1]. Bad cite [9]."
    done = events[-1][1]
    assert done["cited_doc_ids"] == ["d4"] and done["invalid_citations"] == [9]
    assert done["timings_ms"]["first_token"] <= done["timings_ms"]["total"]


def test_api_ask_reports_model_errors(client: TestClient, pg_service: RagService) -> None:
    pg_service._llm = FakeLLM([])  # raises on first call
    with client.stream("POST", "/ask", json={"question": "what is an etf"}) as r:
        raw = "".join(r.iter_text())
    assert "event: error" in raw
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_api_agent(client: TestClient, pg_service: RagService) -> None:
    pg_service._llm = FakeLLM(
        [
            json.dumps({"queries": []}),
            json.dumps({"missing": "", "sufficient": True, "follow_up_query": ""}),
            "Three to six months [1].",
            json.dumps({"verdicts": [{"sentence": 1, "reason": "", "supported": True}]}),
        ]
    )
    body = client.post("/agent", json={"question": "emergency fund size"}).json()
    assert body["cited_doc_ids"] == ["d10"] and body["llm_calls"] == 4


def test_mcp_tools(pg_service: RagService) -> None:
    from finrag import mcp_server

    mcp_server.set_service(pg_service)
    hits = mcp_server.search_finance_qa("credit score", k=50)
    assert hits[0]["doc_id"] == "d7" and len(hits) <= 20
    assert mcp_server.get_document("d7")["text"].startswith("A credit score")
    assert "error" in mcp_server.get_document("nope")


def test_mcp_over_the_protocol(pg_service: RagService) -> None:
    """A real MCP client session (in-process transport): list tools, call one."""
    import asyncio

    from mcp import Client

    from finrag import mcp_server

    mcp_server.set_service(pg_service)

    async def run() -> tuple[set[str], str]:
        async with Client(mcp_server.server) as client:
            tools = await client.list_tools()
            result = await client.call_tool("search_finance_qa", {"query": "credit score", "k": 2})
            return {t.name for t in tools.tools}, json.dumps(
                result.structured_content or [c.model_dump() for c in result.content]
            )

    names, payload = asyncio.run(run())
    assert names == {"search_finance_qa", "get_document"}
    assert "d7" in payload
