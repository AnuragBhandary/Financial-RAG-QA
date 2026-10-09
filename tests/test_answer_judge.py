import json

import pytest

from finrag.answer import NO_ANSWER, answer, build_messages, parse_answer, split_sentences
from finrag.data import Doc
from finrag.judge import cohens_kappa, judge_answer
from finrag.llm import LLMError, parse_json

PASSAGES = [
    Doc("d1", "", "A Roth IRA is funded with after-tax money."),
    Doc("d2", "", "Qualified Roth withdrawals are tax free."),
]


def test_split_sentences_keeps_trailing_citations() -> None:
    s = split_sentences("Roth IRAs use after-tax money [1]. Withdrawals are tax free [2][1]. Done")
    assert [x.citations for x in s] == [(1,), (2, 1), ()]
    assert s[0].text == "Roth IRAs use after-tax money [1]."


def test_split_sentences_comma_citations_and_abbrev_safe() -> None:
    s = split_sentences("It is e.g. common [1, 2]. Another one [3].")
    assert [x.citations for x in s] == [(1, 2), (3,)]
    assert split_sentences("   ") == []


def test_parse_answer_flags_invalid_and_uncited() -> None:
    a = parse_answer("Roth is after-tax [1]. Something else [7]. No cite here.", PASSAGES)
    assert a.invalid_citations == [7]
    assert a.uncited_sentences == 1
    assert a.cited_doc_ids == ["d1"]
    assert not a.abstained


def test_parse_answer_detects_abstention() -> None:
    a = parse_answer(NO_ANSWER, PASSAGES)
    assert a.abstained and a.sentences == []


def test_answer_prompt_numbers_passages(fake_llm) -> None:  # type: ignore[no-untyped-def]
    llm = fake_llm(["Roth uses after-tax money [1]."])
    a = answer(llm, "what is a roth ira", PASSAGES)
    user = llm.calls[0][1]["content"]
    assert "[1] A Roth IRA" in user and "[2] Qualified" in user
    assert a.cited_doc_ids == ["d1"]
    assert answer(llm, "q", []).abstained  # no passages: no LLM call
    msgs = build_messages("q", [Doc("x", "", "word " * 1000)])
    assert msgs[1]["content"].count("word") == 220


def test_judge_rules_and_llm_verdicts(fake_llm) -> None:  # type: ignore[no-untyped-def]
    ans = parse_answer("After-tax money [1]. Free withdrawals [2]. Made up [9]. Bare.", PASSAGES)
    reply = json.dumps(
        {
            "verdicts": [
                {"sentence": 1, "reason": "stated", "supported": True},
                {"sentence": 2, "reason": "missing 'qualified'", "supported": False},
            ]
        }
    )
    llm = fake_llm([reply])
    j = judge_answer(llm, "q", ans)
    assert [v.supported for v in j.verdicts] == [True, False, False, False]
    assert [v.by_rule for v in j.verdicts] == [False, False, True, True]
    assert j.faithfulness == 0.25
    prompt = llm.calls[0][1]["content"]
    assert "[1] A Roth IRA" in prompt and "[9]" not in prompt
    assert llm.schemas[0] is not None


def test_judge_raises_when_a_sentence_is_skipped(fake_llm) -> None:  # type: ignore[no-untyped-def]
    ans = parse_answer("A [1]. B [2].", PASSAGES)
    llm = fake_llm([json.dumps({"verdicts": [{"sentence": 1, "reason": "", "supported": True}]})])
    with pytest.raises(LLMError):
        judge_answer(llm, "q", ans)


def test_judge_abstained_answer_has_no_verdicts(fake_llm) -> None:  # type: ignore[no-untyped-def]
    j = judge_answer(fake_llm([]), "q", parse_answer(NO_ANSWER, PASSAGES))
    assert j.verdicts == [] and j.faithfulness is None


def test_parse_json_tolerates_fences() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMError):
        parse_json("no json")
    with pytest.raises(LLMError):
        parse_json("{not: json}")


def test_cohens_kappa() -> None:
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohens_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohens_kappa([True, True], [True, True]) == 1.0
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_trailing_abstention_is_a_hedge_not_an_abstention() -> None:
    a = parse_answer(f"Roth is after-tax [1]. {NO_ANSWER}", PASSAGES)
    assert not a.abstained and a.hedged
    assert [s.text for s in a.sentences] == ["Roth is after-tax [1]."]
    b = parse_answer(f"{NO_ANSWER} [1]", PASSAGES)
    assert b.abstained and not b.hedged
    assert parse_answer("", PASSAGES).abstained
