from pathlib import Path

import pytest

from finrag.data import Doc
from finrag.models import HashingEmbedder, OverlapReranker
from finrag.retrieval.base import Hit, run_ids
from finrag.retrieval.bm25 import Bm25Retriever
from finrag.retrieval.fusion import HybridRetriever, RerankRetriever, order_head, rrf
from tests.conftest import CORPUS


def hits(*ids: str) -> list[Hit]:
    return [Hit(d, 1.0) for d in ids]


def test_rrf_rewards_agreement_and_is_deterministic() -> None:
    fused = rrf([hits("a", "b", "c"), hits("b", "a", "d")], k=60)
    assert [h.doc_id for h in fused][:2] == ["a", "b"]  # tie on score, broken by id
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert {h.doc_id for h in fused} == {"a", "b", "c", "d"}


def test_rrf_weights() -> None:
    fused = rrf([hits("a"), hits("b")], k=60, weights=[1.0, 2.0])
    assert fused[0].doc_id == "b"


class ListRetriever:
    def __init__(self, name: str, ranking: list[str]) -> None:
        self.name = name
        self.ranking = ranking
        self.seen_k: list[int] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.seen_k.append(k)
        return hits(*self.ranking[:k])

    def search_many(self, queries: dict[str, str], k: int) -> dict[str, list[Hit]]:
        return {q: self.search(t, k) for q, t in queries.items()}


def test_hybrid_fetches_depth_from_both_and_truncates() -> None:
    lex = ListRetriever("lex", ["a", "b", "c"])
    den = ListRetriever("den", ["c", "d"])
    h = HybridRetriever(lex, den, depth=50, rrf_k=60)
    out = h.search("q", 2)
    assert lex.seen_k == [50] and den.seen_k == [50]
    assert [x.doc_id for x in out] == ["c", "a"]


def test_rerank_reorders_head_and_keeps_tail_order() -> None:
    docs = {
        "a": Doc("a", "", "nothing relevant here"),
        "b": Doc("b", "", "roth ira tax free withdrawals"),
        "c": Doc("c", "", "roth"),
        "t1": Doc("t1", "", "x"),
        "t2": Doc("t2", "", "y"),
    }
    base = ListRetriever("base", ["a", "b", "c", "t1", "t2"])
    rr = RerankRetriever(base, OverlapReranker(), lambda ids: {i: docs[i] for i in ids}, depth=3)
    out = [h.doc_id for h in rr.search("roth ira withdrawals", 5)]
    assert out[:3] == ["b", "c", "a"]
    assert out[3:] == ["t1", "t2"]
    assert rr.name == "base+rerank"


def test_rerank_skips_ids_missing_from_store() -> None:
    base = ListRetriever("base", ["gone", "b"])
    rr = RerankRetriever(base, OverlapReranker(), lambda ids: {"b": Doc("b", "", "roth")}, depth=10)
    assert [h.doc_id for h in rr.search("roth", 5)] == ["b"]


def test_bm25_ranks_lexical_match_first_and_round_trips(tmp_path: Path) -> None:
    bm = Bm25Retriever.build(CORPUS)
    assert bm.search("roth ira", 3)[0].doc_id == "d1"
    assert bm.search("exchange traded stock", 3)[0].doc_id == "d4"
    assert bm.search("zzzz qqqq", 3) == []  # no term matches: no arbitrary zero-score docs
    bm.save(tmp_path / "idx")
    loaded = Bm25Retriever.load(tmp_path / "idx")
    assert run_ids(loaded.search_many({"q": "emergency fund"}, 2)) == run_ids(
        bm.search_many({"q": "emergency fund"}, 2)
    )
    assert bm.search_many({}, 3) == {}


def test_hashing_embedder_is_normalised_and_deterministic() -> None:
    e = HashingEmbedder(32)
    a = e.embed_queries(["roth ira", "roth ira"])
    assert a.shape == (2, 32)
    assert float((a[0] * a[0]).sum()) == pytest.approx(1.0, abs=1e-5)
    assert (a[0] == a[1]).all()
    assert e.embed_docs([]).shape == (0, 32)


def test_order_head_fusion_keeps_first_stage_signal() -> None:
    head = hits("a", "b", "c")
    # The cross-encoder alone flips the list. Fused with equal weight, a (1st, 3rd) and c (3rd,
    # 1st) tie above b (2nd, 2nd): 1/61 + 1/63 > 2/62. The first-stage favourite is not buried.
    assert [h.doc_id for h in order_head(head, [0.1, 0.5, 0.9])] == ["c", "b", "a"]
    fused = [h.doc_id for h in order_head(head, [0.1, 0.5, 0.9], fuse_weight=1.0)]
    assert fused == ["a", "c", "b"]
    assert order_head(head, [0.1, 0.5, 0.9], fuse_weight=100.0)[0].doc_id == "c"
