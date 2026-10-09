"""LLM judge for faithfulness: is each answer sentence supported by the passages it cites?

Sentence-level, binary labels, so the same unit can be hand-labeled and the judge's agreement
with people measured (Cohen's kappa), instead of trusting an opaque 1-10 score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from finrag.answer import Answer, truncate_words
from finrag.llm import LLM, LLMError, Usage, parse_json

JUDGE_SYSTEM = """You check whether sentences from an answer are supported by source passages.
A sentence is SUPPORTED only if the passages it cites state or directly imply everything it \
claims. It is NOT supported if any part of it is missing from, goes beyond, or contradicts \
the cited passages, even if the claim is true in general. Judge only against the cited \
passages, never against your own knowledge."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "integer"},
                    "reason": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["sentence", "reason", "supported"],
            },
        }
    },
    "required": ["verdicts"],
}


@dataclass(frozen=True)
class Verdict:
    sentence: str
    citations: tuple[int, ...]
    supported: bool
    reason: str
    by_rule: bool = False  # decided without the LLM (no citation / citation out of range)


@dataclass
class Judgement:
    verdicts: list[Verdict]
    usage: Usage

    @property
    def faithfulness(self) -> float | None:
        if not self.verdicts:
            return None
        return sum(v.supported for v in self.verdicts) / len(self.verdicts)


def _passage(ans: Answer, n: int) -> str:
    p = ans.passages[n - 1]
    return truncate_words(f"{p.title} {p.text}".strip())


def judge_answer(llm: LLM, question: str, ans: Answer) -> Judgement:
    if ans.abstained or not ans.sentences:
        return Judgement([], Usage())
    n_passages = len(ans.passages)
    verdicts: dict[int, Verdict] = {}
    to_ask: list[int] = []
    for i, s in enumerate(ans.sentences):
        valid = [c for c in s.citations if 1 <= c <= n_passages]
        if not valid:
            verdicts[i] = Verdict(s.text, s.citations, False, "no valid citation", by_rule=True)
        else:
            to_ask.append(i)
    usage = Usage()
    if to_ask:
        cited = sorted(
            {c for i in to_ask for c in ans.sentences[i].citations if 1 <= c <= n_passages}
        )
        passages = "\n\n".join(f"[{c}] {_passage(ans, c)}" for c in cited)
        listing = "\n".join(f"{j}. {ans.sentences[i].text}" for j, i in enumerate(to_ask, start=1))
        user = (
            f"Question: {question}\n\nPassages:\n\n{passages}\n\n"
            f"Sentences (each cites passages in brackets):\n{listing}\n\n"
            "For every sentence, give a short reason and then whether it is supported."
        )
        out = llm.complete(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
            schema=JUDGE_SCHEMA,
            max_tokens=200 + 120 * len(to_ask),
        )
        usage = out.usage
        raw = parse_json(out.text).get("verdicts", [])
        by_num = {int(v["sentence"]): v for v in raw if "sentence" in v}
        for j, i in enumerate(to_ask, start=1):
            if j not in by_num:
                raise LLMError(f"judge skipped sentence {j}")
            v = by_num[j]
            s = ans.sentences[i]
            verdicts[i] = Verdict(s.text, s.citations, bool(v["supported"]), str(v["reason"]))
    return Judgement([verdicts[i] for i in range(len(ans.sentences))], usage)


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
