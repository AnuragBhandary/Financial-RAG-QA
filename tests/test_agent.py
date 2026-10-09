import json

from finrag.agent import AgentConfig, RagAgent
from finrag.answer import NO_ANSWER
from finrag.data import Doc
from finrag.retrieval.base import Hit

DOCS = {f"d{i}": Doc(f"d{i}", "", f"passage number {i}") for i in range(1, 9)}


class ScriptedRetriever:
    name = "scripted"

    def __init__(self, results: dict[str, list[str]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.queries.append(query)
        return [Hit(d, 1.0) for d in self.results.get(query, [])[:k]]

    def search_many(self, queries: dict[str, str], k: int) -> dict[str, list[Hit]]:
        return {q: self.search(t, k) for q, t in queries.items()}


def fetch(ids: list[str]) -> dict[str, Doc]:
    return {i: DOCS[i] for i in ids if i in DOCS}


def plan(*queries: str) -> str:
    return json.dumps({"queries": list(queries)})


def grade(sufficient: bool, follow: str = "") -> str:
    return json.dumps({"missing": "x", "sufficient": sufficient, "follow_up_query": follow})


def verdicts(*supported: bool) -> str:
    return json.dumps(
        {
            "verdicts": [
                {"sentence": i, "reason": "r", "supported": s}
                for i, s in enumerate(supported, start=1)
            ]
        }
    )


def test_single_round_when_evidence_is_sufficient(fake_llm) -> None:  # type: ignore[no-untyped-def]
    llm = fake_llm(
        [plan("q"), grade(True), "Answer one [1]. Answer two [2].", verdicts(True, True)]
    )
    r = ScriptedRetriever({"q": ["d1", "d2"]})
    res = RagAgent(llm, r, fetch).run("q")
    assert r.queries == ["q"]  # the original question runs; a duplicate rewrite is dropped
    assert res.answer.text == "Answer one [1]. Answer two [2]."
    assert res.usage.calls == 4
    assert res.trace[-1] == "verify: all supported"


def test_follow_up_search_then_stop_at_max_rounds(fake_llm) -> None:  # type: ignore[no-untyped-def]
    llm = fake_llm(
        [
            plan("roth tax", "roth withdrawal rules"),
            grade(False, "five year rule"),
            # round 2 = max_rounds: no further grading call
            "Ans [1].",
            verdicts(True),
        ]
    )
    r = ScriptedRetriever(
        {"q": ["d1"], "roth tax": ["d2"], "roth withdrawal rules": ["d3"], "five year rule": ["d4"]}
    )
    res = RagAgent(llm, r, fetch, AgentConfig(max_rounds=2)).run("q")
    assert r.queries == ["q", "roth tax", "roth withdrawal rules", "five year rule"]
    assert {d.id for d in res.evidence} == {"d1", "d2", "d3", "d4"}
    assert any("follow-up" in t for t in res.trace)


def test_verify_drops_unsupported_sentences(fake_llm) -> None:  # type: ignore[no-untyped-def]
    llm = fake_llm([plan(), grade(True), "Good [1]. Invented [2].", verdicts(True, False)])
    res = RagAgent(llm, ScriptedRetriever({"q": ["d1", "d2"]}), fetch).run("q")
    assert res.answer.text == "Good [1]."
    assert "dropped 1" in res.trace[-1]


def test_all_unsupported_becomes_abstention(fake_llm) -> None:  # type: ignore[no-untyped-def]
    llm = fake_llm([plan(), grade(True), "Invented [1].", verdicts(False)])
    res = RagAgent(llm, ScriptedRetriever({"q": ["d1"]}), fetch).run("q")
    assert res.answer.text == NO_ANSWER and res.answer.abstained


def test_garbage_from_planner_and_grader_degrades_to_single_shot(fake_llm) -> None:  # type: ignore[no-untyped-def]
    llm = fake_llm(["not json", "also not json", "Fine [1].", "garbage verdicts"])
    res = RagAgent(llm, ScriptedRetriever({"q": ["d1"]}), fetch).run("q")
    assert res.searches == ["q"]
    assert res.answer.text == "Fine [1]."
    assert "kept answer" in res.trace[-1]


def test_repeated_follow_up_query_stops_the_loop(fake_llm) -> None:  # type: ignore[no-untyped-def]
    llm = fake_llm([plan(), grade(False, "q"), "A [1].", verdicts(True)])
    r = ScriptedRetriever({"q": ["d1"]})
    RagAgent(llm, r, fetch, AgentConfig(max_rounds=3)).run("q")
    assert r.queries == ["q"]
