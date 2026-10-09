"""Grounded answers with numbered citations, and the parsing that checks those citations."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from finrag.data import Doc
from finrag.llm import LLM, Message, Usage

MAX_PASSAGE_WORDS = 220
NO_ANSWER = "The retrieved passages do not answer this question."

SYSTEM_PROMPT = f"""You answer personal-finance and investing questions using ONLY the numbered \
passages provided. Rules:
- Every sentence must end with at least one citation like [1] or [2][4], naming the passages \
that support that sentence.
- Do not use facts that are not in the passages. Do not cite a passage that does not say it.
- Write 2 to 5 short sentences. Plain prose, no headings or bullet points.
- If the passages do not contain an answer, reply exactly: {NO_ANSWER}"""


@dataclass(frozen=True)
class Sentence:
    text: str
    citations: tuple[int, ...]  # passage numbers, 1-based, as written by the model


@dataclass
class Answer:
    text: str
    passages: list[Doc]
    sentences: list[Sentence]
    abstained: bool
    invalid_citations: list[int]
    usage: Usage = field(default_factory=Usage)
    # Answered, then also said the passages do not answer (the abstention line is not a claim,
    # so it is removed from `sentences` and not judged).
    hedged: bool = False

    @property
    def cited_doc_ids(self) -> list[str]:
        seen: list[str] = []
        for s in self.sentences:
            for n in s.citations:
                if 1 <= n <= len(self.passages) and self.passages[n - 1].id not in seen:
                    seen.append(self.passages[n - 1].id)
        return seen

    @property
    def uncited_sentences(self) -> int:
        return sum(1 for s in self.sentences if not s.citations)


def truncate_words(text: str, n: int = MAX_PASSAGE_WORDS) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " ..."


def build_messages(question: str, passages: Sequence[Doc]) -> list[Message]:
    blocks = "\n\n".join(
        f"[{i}] {truncate_words((p.title + ' ' + p.text).strip())}"
        for i, p in enumerate(passages, start=1)
    )
    user = f"Passages:\n\n{blocks}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Split after ., ! or ? (plus any trailing citations) when followed by space and a capital/quote.
_SENT_END = re.compile(r"(?<=[.!?])((?:\s*\[\d+(?:\s*,\s*\d+)*\])*)\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[Sentence]:
    text = " ".join(text.split())
    if not text:
        return []
    parts: list[str] = []
    last = 0
    for m in _SENT_END.finditer(text):
        parts.append(text[last : m.end(1)])
        last = m.end()
    parts.append(text[last:])
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        nums = tuple(int(n) for grp in _CITE.findall(part) for n in grp.split(","))
        out.append(Sentence(text=part, citations=nums))
    return out


def parse_answer(text: str, passages: Sequence[Doc], usage: Usage | None = None) -> Answer:
    text = text.strip()
    sentences = split_sentences(text)
    claims = [s for s in sentences if not _is_abstention(s.text)]
    abstained = not claims
    invalid = sorted({n for s in claims for n in s.citations if not 1 <= n <= len(passages)})
    return Answer(
        text=text,
        passages=list(passages),
        sentences=claims,
        abstained=abstained,
        invalid_citations=invalid,
        usage=usage or Usage(),
        hedged=bool(claims) and len(claims) < len(sentences),
    )


def _is_abstention(sentence: str) -> bool:
    bare = _CITE.sub("", sentence).strip().rstrip(".").lower()
    return bare == NO_ANSWER.rstrip(".").lower()


def answer(llm: LLM, question: str, passages: Sequence[Doc], max_tokens: int = 400) -> Answer:
    if not passages:
        return parse_answer(NO_ANSWER, [])
    out = llm.complete(build_messages(question, passages), max_tokens=max_tokens)
    return parse_answer(out.text, passages, out.usage)


def stream_answer(llm: LLM, question: str, passages: Sequence[Doc]) -> Iterator[str]:
    if not passages:
        yield NO_ANSWER
        return
    yield from llm.stream(build_messages(question, passages), max_tokens=400)
