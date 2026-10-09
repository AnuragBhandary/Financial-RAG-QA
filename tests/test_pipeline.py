"""Ingest, tuning, ANN and latency reports, hand-label tooling, and the CLI, on the test DB."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from finrag import cli, db
from finrag.config import Settings, load_dotenv
from finrag.data import load_qrels, split_queries
from finrag.evaluate import bench_latency, hnsw_recall, tune
from finrag.ingest import embedding_cache_path, ingest
from finrag.labeling import agreement, label_loop, make_label_sample
from finrag.models import HashingEmbedder
from finrag.service import RagService
from tests.conftest import CORPUS, TEST_DB_URL, _ensure_test_db

pytestmark = pytest.mark.integration


def test_ingest_caches_embeddings_and_reloads(settings: Settings) -> None:
    _ensure_test_db()
    emb = HashingEmbedder(32)
    logs: list[str] = []
    stats = ingest(settings, emb, log=logs.append)
    assert stats["docs"] == 10
    cache = Path(embedding_cache_path(settings, emb.name, 10))
    assert np.load(cache).shape == (10, 32)
    assert (settings.bm25_dir / "doc_ids.json").exists()
    # Second run reads the cache instead of re-embedding.
    logs.clear()
    ingest(settings, emb, log=logs.append, limit=10)
    assert any("loaded cache" in line for line in logs)
    with db.connect(TEST_DB_URL) as conn:
        assert db.index_meta(conn)["dim"] == "32"


def test_data_loader_skips_zero_grades(settings: Settings) -> None:
    path = settings.fiqa_dir / "qrels" / "dev.tsv"
    path.write_text(path.read_text() + "q1\td2\t0\n")
    assert load_qrels(settings.fiqa_dir, "dev")["q1"] == {"d1": 1}
    queries, qrels = split_queries(settings.fiqa_dir, "test")
    assert set(queries) == set(qrels) == {"q1", "q2", "q3"}


def test_tune_hnsw_and_bench_reports(pg_service: RagService) -> None:
    out = pg_service.settings.runs_dir
    out.mkdir(parents=True)
    t = tune(pg_service, out, log=lambda s: None)
    assert len(t["hybrid_grid"]) == 48 and len(t["rerank_grid"]) == 40
    h = hnsw_recall(pg_service, out, log=lambda s: None)
    assert h["exact"]["recall@10_vs_exact"] == 1.0
    assert h["ef_search=40"]["recall@10_vs_exact"] == 1.0  # 10 docs: the graph sees all
    b = bench_latency(pg_service, ["bm25", "hybrid+rerank"], 3, out, log=lambda s: None)
    assert b["hybrid+rerank"]["p95_ms"] >= b["hybrid+rerank"]["rerank_p95_ms"] >= 0
    assert {p.name for p in out.iterdir()} >= {
        "tune-dev.json",
        "hnsw-test.json",
        "latency-retrieval.json",
    }


def _judged_file(path: Path) -> None:
    recs = []
    for i, qid in enumerate(["q1", "q2", "q3"]):
        recs.append(
            {
                "system": "rag:x",
                "qid": qid,
                "question": f"question {qid}",
                "passage_ids": ["d1", "d4"],
                "verdicts": [
                    {
                        "sentence": "A [1].",
                        "citations": [1],
                        "supported": True,
                        "reason": "",
                        "by_rule": False,
                    },
                    {
                        "sentence": "B [2].",
                        "citations": [2],
                        "supported": i == 0,
                        "reason": "",
                        "by_rule": False,
                    },
                    {
                        "sentence": "C.",
                        "citations": [],
                        "supported": False,
                        "reason": "",
                        "by_rule": True,
                    },
                ],
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_label_sample_loop_and_agreement(
    pg_service: RagService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    judged = tmp_path / "judged.jsonl"
    _judged_file(judged)
    sample = tmp_path / "labels" / "sample.jsonl"
    make_label_sample(pg_service, [judged], 4, sample)
    lines = [json.loads(x) for x in sample.read_text().splitlines()]
    meta, items = lines[0], lines[1:]
    assert meta["pool_supported"] == 4 and meta["pool_unsupported"] == 2
    assert len(items) == 4 and sum(not i["judge_supported"] for i in items) == 2
    assert all(i["cited_passages"] for i in items)

    # The person agrees with the judge on everything except the first item.
    answers = iter(
        ["maybe", "n" if items[0]["judge_supported"] else "y"]
        + ["y" if it["judge_supported"] else "n" for it in items[1:3]]
        + ["s", "q"]
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    labels = tmp_path / "labels" / "human.jsonl"
    label_loop(None, sample, labels)
    assert "SENTENCE" in capsys.readouterr().out
    assert len(labels.read_text().splitlines()) == 3
    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    label_loop(None, sample, labels)  # resumes, quits immediately

    a = agreement(sample, labels)
    assert a["labeled"] == 3
    assert a["raw_agreement"] == pytest.approx(2 / 3, abs=1e-3)
    assert "reweighted_agreement" in a
    with pytest.raises(SystemExit):
        empty = tmp_path / "none.jsonl"
        empty.write_text("")
        agreement(sample, empty)


def test_load_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nFINRAG_X='1'\nFINRAG_Y=2   # inline note\nnot a pair\nFINRAG_Z=from-file\n"
    )
    monkeypatch.setenv("FINRAG_Z", "from-env")
    monkeypatch.delenv("FINRAG_X", raising=False)
    monkeypatch.delenv("FINRAG_Y", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["FINRAG_X"] == "1" and os.environ["FINRAG_Y"] == "2"
    assert os.environ["FINRAG_Z"] == "from-env"
    load_dotenv(tmp_path / "missing.env")
    assert Settings(llm_model="custom").default_llm_model() == "custom"


def test_cli_commands(
    pg_service: RagService, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.conftest import FakeLLM

    monkeypatch.setattr(cli, "_service", lambda settings: pg_service)
    monkeypatch.setattr(cli, "Settings", lambda: pg_service.settings)
    cli.main(["eval-retrieval", "--split", "dev", "--modes", "bm25,dense"])
    assert "bm25" in capsys.readouterr().out

    pg_service._llm = FakeLLM(["An ETF trades like a stock [1]."])
    cli.main(["ask", "what is an etf"])
    out = capsys.readouterr().out
    assert "An ETF trades like a stock [1]." in out and "[1] doc d4" in out

    pg_service._llm = FakeLLM(
        [
            json.dumps({"queries": []}),
            json.dumps({"missing": "", "sufficient": True, "follow_up_query": ""}),
            "Three to six months [1].",
            json.dumps({"verdicts": [{"sentence": 1, "reason": "", "supported": True}]}),
        ]
    )
    cli.main(["ask", "--agent", "emergency fund size"])
    assert "· plan" in capsys.readouterr().out

    pg_service._llm = FakeLLM(["Claim [1]."] * 3)
    cli.main(["answers", "--system", "rag:bm25", "--n", "3"])
    verdict = json.dumps({"verdicts": [{"sentence": 1, "reason": "", "supported": True}]})
    monkeypatch.setattr("finrag.llm.make_llm", lambda s, m=None: FakeLLM([verdict] * 3))
    cli.main(["judge", "--system", "rag:bm25"])
    summary = json.loads(
        (pg_service.settings.runs_dir / "answers-rag-bm25-summary.json").read_text()
    )
    assert summary["sentence_support_rate"] == 1.0 and summary["judge"] == "fake"


def test_mine_pairs_skips_top_unjudged_and_never_uses_gold_as_negative(
    pg_service: RagService,
) -> None:
    from finrag.data import load_qrels
    from finrag.train import mine_pairs

    train = pg_service.settings.fiqa_dir / "qrels" / "train.tsv"
    train.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t1\nq2\td4\t1\n")
    pairs = mine_pairs(pg_service, depth=10, skip_top=2, negatives_per_query=3)
    gold = load_qrels(pg_service.settings.fiqa_dir, "train")
    texts = {d.text: d.id for d in CORPUS}
    positives = [(q, texts[p.strip()]) for q, p, y in pairs if y == 1.0]
    negatives = [(q, texts[p.strip()]) for q, p, y in pairs if y == 0.0]
    assert len(positives) == 3
    assert len(negatives) == 6  # 3 per query
    by_query = {"roth ira withdrawals tax": "q1", "what is an etf": "q2"}
    assert all(d not in gold[by_query[q]] for q, d in negatives)
    # The two highest-ranked unjudged docs per query are never used as negatives.
    hits = pg_service.retriever("hybrid").search("roth ira withdrawals tax", 10)
    top_unjudged = [h.doc_id for h in hits if h.doc_id not in gold["q1"]][:2]
    assert not {d for q, d in negatives if q == "roth ira withdrawals tax"} & set(top_unjudged)


def test_embed_model_falls_back_to_base_when_not_trained(tmp_path: Path) -> None:
    s = Settings(embed_model=str(tmp_path / "missing"), base_embed_model="base")
    assert s.resolved_embed_model() == "base"
    trained = tmp_path / "trained"
    trained.mkdir()
    assert Settings(embed_model=str(trained)).resolved_embed_model() == str(trained)
    assert Settings(embed_model="org/hub-model").resolved_embed_model() == "org/hub-model"


def test_dense_refuses_an_index_built_by_another_model(pg_service: RagService) -> None:
    other_embedder = HashingEmbedder(64)
    other_embedder.name = "some-other-model"
    other = RagService(pg_service.settings, conn=pg_service.conn, embedder=other_embedder)
    with pytest.raises(RuntimeError, match="finrag ingest"):
        other.retriever("dense")
