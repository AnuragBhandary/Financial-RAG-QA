"""Agentic RAG as a LangGraph state machine: plan -> retrieve -> grade (loop) -> answer -> verify.

The single-shot pipeline retrieves once for the question as written. The agent can rewrite the
question into several searches, ask for a follow-up search when the evidence is thin, and drop
answer sentences its own check finds unsupported. Whether that is worth 3-5x the LLM calls is a
measured question (docs/RESULTS.md), not an assumption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from finrag.answer import NO_ANSWER, Answer, answer, parse_answer
from finrag.data import Doc
from finrag.judge import judge_answer
from finrag.llm import LLM, LLMError, Usage, parse_json
from finrag.retrieval.base import Hit, Retriever
from finrag.retrieval.fusion import rrf

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "missing": {"type": "string"},
        "sufficient": {"type": "boolean"},
        "follow_up_query": {"type": "string"},
    },
    "required": ["missing", "sufficient", "follow_up_query"],
}

PLAN_PROMPT = """You write search queries for a corpus of answers from a personal-finance Q&A \
forum. Turn the user's question into 1 to 3 short search queries. Use 1 query when the \
question asks one thing; use more only when it asks about several distinct things or uses \
vague wording that a rephrasing would match better. Return JSON."""

GRADE_PROMPT = """Decide whether the passages contain enough information to answer the \
question. First say briefly what is missing (or "nothing"). If something important is \
missing, give one short follow-up search query for it; otherwise give an empty string. \
Return JSON."""


class AgentState(TypedDict, total=False):
    question: str
    pending: list[str]
    searches: list[str]
    rankings: list[list[Hit]]
    evidence: list[Doc]
    rounds: int
    answer: Answer
    trace: list[str]


@dataclass
class AgentConfig:
    max_queries: int = 3
    max_rounds: int = 2  # retrieval rounds, including the first
    per_query_k: int = 10
    n_passages: int = 5
    verify: bool = True


@dataclass
class AgentResult:
    question: str
    answer: Answer
    searches: list[str]
    evidence: list[Doc]
    trace: list[str]
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0


class RagAgent:
    def __init__(
        self,
        llm: LLM,
        retriever: Retriever,
        fetch: Any,
        config: AgentConfig | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.fetch = fetch
        self.config = config or AgentConfig()
        self.usage = Usage()
        self.graph = self._build()

    # ---- nodes -------------------------------------------------------------------------

    def _plan(self, state: AgentState) -> AgentState:
        q = state["question"]
        try:
            out = self.llm.complete(
                [{"role": "system", "content": PLAN_PROMPT}, {"role": "user", "content": q}],
                schema=PLAN_SCHEMA,
                max_tokens=200,
            )
            self.usage.add(out.usage)
            queries = [s.strip() for s in parse_json(out.text).get("queries", []) if s.strip()]
        except LLMError:
            queries = []
        # The original question always runs: rewrites can only add recall, never replace it.
        queries = [q] + [s for s in queries if s.lower() != q.lower()]
        queries = queries[: self.config.max_queries]
        return {"pending": queries, "trace": [*state.get("trace", []), f"plan: {queries}"]}

    def _retrieve(self, state: AgentState) -> AgentState:
        rankings = list(state.get("rankings", []))
        searches = list(state.get("searches", []))
        for query in state["pending"]:
            rankings.append(self.retriever.search(query, self.config.per_query_k))
            searches.append(query)
        fused = rrf(rankings)[: self.config.n_passages]
        docs = self.fetch([h.doc_id for h in fused])
        evidence = [docs[h.doc_id] for h in fused if h.doc_id in docs]
        return {
            "rankings": rankings,
            "searches": searches,
            "evidence": evidence,
            "pending": [],
            "rounds": state.get("rounds", 0) + 1,
            "trace": [*state.get("trace", []), f"retrieve: {[d.id for d in evidence]}"],
        }

    def _grade(self, state: AgentState) -> AgentState:
        if state["rounds"] >= self.config.max_rounds:
            return {}
        passages = "\n\n".join(
            f"[{i}] {d.text[:600]}" for i, d in enumerate(state["evidence"], start=1)
        )
        try:
            out = self.llm.complete(
                [
                    {"role": "system", "content": GRADE_PROMPT},
                    {
                        "role": "user",
                        "content": f"Question: {state['question']}\n\nPassages:\n\n{passages}",
                    },
                ],
                schema=GRADE_SCHEMA,
                max_tokens=200,
            )
            self.usage.add(out.usage)
            verdict = parse_json(out.text)
        except LLMError:
            return {"trace": [*state.get("trace", []), "grade: unparseable, stop"]}
        follow = str(verdict.get("follow_up_query", "")).strip()
        done = bool(verdict.get("sufficient", True)) or not follow or follow in state["searches"]
        note = "sufficient" if done else f"follow-up {follow!r} ({verdict.get('missing', '')})"
        return {
            "pending": [] if done else [follow],
            "trace": [*state.get("trace", []), f"grade: {note}"],
        }

    def _route_after_grade(self, state: AgentState) -> str:
        return "retrieve" if state.get("pending") else "answer"

    def _answer(self, state: AgentState) -> AgentState:
        ans = answer(self.llm, state["question"], state["evidence"])
        self.usage.add(ans.usage)
        return {"answer": ans, "trace": [*state.get("trace", []), "answer"]}

    def _verify(self, state: AgentState) -> AgentState:
        ans = state["answer"]
        if not self.config.verify or ans.abstained or not ans.sentences:
            return {}
        try:
            judgement = judge_answer(self.llm, state["question"], ans)
        except LLMError:
            return {"trace": [*state.get("trace", []), "verify: unparseable, kept answer"]}
        self.usage.add(judgement.usage)
        kept = [v.sentence for v in judgement.verdicts if v.supported]
        dropped = len(judgement.verdicts) - len(kept)
        if dropped == 0:
            return {"trace": [*state.get("trace", []), "verify: all supported"]}
        new_text = " ".join(kept) if kept else NO_ANSWER
        new = parse_answer(new_text, ans.passages, ans.usage)
        return {
            "answer": new,
            "trace": [*state.get("trace", []), f"verify: dropped {dropped} unsupported"],
        }

    # ---- graph -------------------------------------------------------------------------

    def _build(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("answer", self._answer)
        g.add_node("verify", self._verify)
        g.add_edge(START, "plan")
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._route_after_grade, ["retrieve", "answer"])
        g.add_edge("answer", "verify")
        g.add_edge("verify", END)
        return g.compile()

    def run(self, question: str) -> AgentResult:
        self.usage = Usage()
        t0 = time.perf_counter()
        state: AgentState = self.graph.invoke({"question": question, "trace": []})
        return AgentResult(
            question=question,
            answer=state["answer"],
            searches=state.get("searches", []),
            evidence=state.get("evidence", []),
            trace=state.get("trace", []),
            usage=self.usage,
            seconds=time.perf_counter() - t0,
        )
