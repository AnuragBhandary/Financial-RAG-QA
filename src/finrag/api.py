"""FastAPI service: search, cited answers streamed over Server-Sent Events, and the agent."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from finrag import db
from finrag.answer import parse_answer, stream_answer
from finrag.config import Settings, load_dotenv
from finrag.service import CONTEXT_PASSAGES, RagResult, RagService

MODES = {"bm25", "dense", "hybrid", "hybrid+rerank", "dense+rerank", "bm25+rerank", "pg-fts"}
WEB_DIR = Path(__file__).resolve().parents[2] / "web"

state: dict[str, Any] = {}
# One psycopg connection and one set of models per process; torch modules and psycopg
# connections are not safe to share across threads, so retrieval is serialised. Token
# streaming (the slow part) runs outside the lock.
_retrieval_lock = threading.Lock()


def get_service() -> RagService:
    svc: RagService = state["service"]
    return svc


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if "service" not in state:
        load_dotenv()
        svc = RagService(Settings())
        svc.retrieve("warm up the models", "hybrid+rerank")  # load weights before traffic
        state["service"] = svc
    yield


app = FastAPI(title="finrag", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = "hybrid+rerank"
    passages: int = Field(default=CONTEXT_PASSAGES, ge=1, le=10)


def _retrieve(question: str, mode: str, k: int) -> RagResult:
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {sorted(MODES)}")
    with _retrieval_lock:
        return get_service().retrieve(question, mode, k)


def _passages_json(result: RagResult, n: int) -> list[dict[str, Any]]:
    scores = {h.doc_id: h.score for h in result.hits}
    return [
        {"n": i, "doc_id": p.id, "score": round(scores.get(p.id, 0.0), 4), "text": p.text}
        for i, p in enumerate(result.passages[:n], start=1)
    ]


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    svc = get_service()
    with _retrieval_lock:
        meta = db.index_meta(svc.conn)
    return {"status": "ok", "index": meta, "llm": svc.llm.model}


@app.get("/search")
def search(
    q: str = Query(min_length=2, max_length=500),
    mode: str = "hybrid+rerank",
    k: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    result = _retrieve(q, mode, k)
    return {
        "query": q,
        "mode": mode,
        "timings_ms": {
            "retrieve": round(result.timings.retrieve_ms, 1),
            "rerank": round(result.timings.rerank_ms, 1),
        },
        "hits": _passages_json(result, k),
    }


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/ask")
def ask(req: AskRequest) -> StreamingResponse:
    """SSE stream: `passages`, then `token`* , then `done` with parsed citations and timings."""
    t0 = time.perf_counter()
    result = _retrieve(req.question, req.mode, max(10, req.passages))
    passages = result.passages[: req.passages]
    t_retrieved = time.perf_counter()
    llm = get_service().llm

    def events() -> Iterator[str]:
        yield _sse("passages", _passages_json(result, req.passages))
        parts: list[str] = []
        first_token_at: float | None = None
        try:
            for delta in stream_answer(llm, req.question, passages):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                parts.append(delta)
                yield _sse("token", delta)
        except Exception as e:  # surface model errors to the client instead of a dead stream
            yield _sse("error", str(e))
            return
        ans = parse_answer("".join(parts), passages)
        end = time.perf_counter()
        yield _sse(
            "done",
            {
                "abstained": ans.abstained,
                "cited_doc_ids": ans.cited_doc_ids,
                "invalid_citations": ans.invalid_citations,
                "uncited_sentences": ans.uncited_sentences,
                "timings_ms": {
                    "retrieve": round(result.timings.retrieve_ms, 1),
                    "rerank": round(result.timings.rerank_ms, 1),
                    "first_token": round(((first_token_at or end) - t0) * 1000, 1),
                    "total": round((end - t0) * 1000, 1),
                    "retrieval_total": round((t_retrieved - t0) * 1000, 1),
                },
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/agent")
def agent(req: AskRequest) -> dict[str, Any]:
    from finrag.agent import RagAgent

    svc = get_service()

    class LockedRetriever:
        name = "hybrid+rerank"

        def __init__(self) -> None:
            self._inner = svc.retriever("hybrid+rerank")

        def search(self, query: str, k: int) -> Any:
            with _retrieval_lock:
                return self._inner.search(query, k)

        def search_many(self, queries: Any, k: int) -> Any:
            with _retrieval_lock:
                return self._inner.search_many(queries, k)

    def fetch(ids: Any) -> Any:
        with _retrieval_lock:
            return svc.fetch(ids)

    res = RagAgent(svc.llm, LockedRetriever(), fetch).run(req.question)
    return {
        "answer": res.answer.text,
        "abstained": res.answer.abstained,
        "cited_doc_ids": res.answer.cited_doc_ids,
        "searches": res.searches,
        "trace": res.trace,
        "passages": [
            {"n": i, "doc_id": d.id, "text": d.text} for i, d in enumerate(res.evidence, 1)
        ],
        "llm_calls": res.usage.calls,
        "seconds": round(res.seconds, 2),
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
