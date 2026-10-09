"""MCP server: exposes the retriever as tools, so any MCP client (Claude Desktop, an IDE, another
agent) can search the FiQA index and read documents, with citations it can check.

Run over stdio:  uv run finrag mcp
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from finrag.config import Settings, load_dotenv
from finrag.service import RagService

server = MCPServer(
    name="finrag",
    instructions=(
        "Search a corpus of 57,638 answers from a personal-finance Q&A forum (FiQA-2018). "
        "Use search_finance_qa to find passages, then cite them by doc_id."
    ),
)

_service: RagService | None = None


def service() -> RagService:
    global _service
    if _service is None:
        load_dotenv()
        _service = RagService(Settings())
    return _service


def set_service(svc: RagService) -> None:
    """Tests inject a service built on fakes."""
    global _service
    _service = svc


@server.tool()
def search_finance_qa(query: str, k: int = 5, mode: str = "hybrid+rerank") -> list[dict[str, Any]]:
    """Search the personal-finance Q&A corpus.

    Args:
        query: a natural-language question or keywords.
        k: number of passages to return (1-20).
        mode: bm25, dense, hybrid or hybrid+rerank (best quality, slowest).
    """
    k = max(1, min(k, 20))
    result = service().retrieve(query, mode, k)
    scores = {h.doc_id: h.score for h in result.hits}
    return [
        {"doc_id": p.id, "score": round(scores.get(p.id, 0.0), 4), "text": p.text}
        for p in result.passages
    ]


@server.tool()
def get_document(doc_id: str) -> dict[str, Any]:
    """Fetch one corpus document by doc_id (as returned by search_finance_qa)."""
    docs = service().fetch([doc_id])
    if doc_id not in docs:
        return {"error": f"no document {doc_id!r}"}
    d = docs[doc_id]
    return {"doc_id": d.id, "title": d.title, "text": d.text}


def main() -> None:
    server.run("stdio")
