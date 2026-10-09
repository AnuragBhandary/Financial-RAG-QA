import math

import pytest

from finrag import metrics


def test_ndcg_perfect_and_hand_computed() -> None:
    rels = {"a": 1, "b": 1}
    assert metrics.ndcg_at_k(["a", "b", "x"], rels) == pytest.approx(1.0)
    # relevant at ranks 2 and 3: (1/log2(3) + 1/log2(4)) / (1 + 1/log2(3))
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert metrics.ndcg_at_k(["x", "a", "b"], rels) == pytest.approx(expected)


def test_ndcg_cutoff_and_empty() -> None:
    rels = {"a": 1}
    assert metrics.ndcg_at_k(["x"] * 10 + ["a"], rels, k=10) == 0.0
    assert metrics.ndcg_at_k(["a"], {}, k=10) == 0.0
    assert metrics.ndcg_at_k([], rels) == 0.0


def test_ndcg_graded_uses_linear_gain() -> None:
    rels = {"a": 2, "b": 1}
    got = metrics.ndcg_at_k(["b", "a"], rels)
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert got == pytest.approx(expected)


def test_recall_and_mrr() -> None:
    rels = {"a": 1, "b": 1, "c": 1, "d": 1}
    assert metrics.recall_at_k(["a", "x", "b"], rels, 2) == 0.25
    assert metrics.recall_at_k(["a", "x", "b"], rels, 3) == 0.5
    assert metrics.recall_at_k(["a"], {}, 3) == 0.0
    assert metrics.mrr_at_k(["x", "x", "c"], rels) == pytest.approx(1 / 3)
    assert metrics.mrr_at_k(["x"] * 10 + ["a"], rels) == 0.0


def test_per_query_scores_counts_missing_queries_as_zero() -> None:
    scores = metrics.per_query_scores({"q1": ["a"]}, {"q1": {"a": 1}, "q2": {"b": 1}})
    assert scores["ndcg@10"] == {"q1": 1.0, "q2": 0.0}


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = [0.0, 1.0] * 100
    lo, hi = metrics.bootstrap_ci(values)
    assert lo < 0.5 < hi
    assert hi - lo < 0.2


def test_randomization_test() -> None:
    a = [0.9] * 50
    b = [0.1] * 50
    assert metrics.paired_randomization_test(a, b, n_resamples=2000) < 0.01
    same = [0.3, 0.5, 0.7] * 10
    assert metrics.paired_randomization_test(same, same, n_resamples=500) == 1.0
    with pytest.raises(ValueError):
        metrics.paired_randomization_test([1.0], [1.0, 2.0])
