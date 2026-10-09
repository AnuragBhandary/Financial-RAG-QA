"""Evaluation harness: retrieval metrics for every variant, latency, and answer quality.

Tuning reads only the dev split; test numbers are produced once with the frozen defaults.
Every run writes its per-query scores so any number in RESULTS.md can be recomputed.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from finrag import metrics
from finrag.agent import AgentConfig, RagAgent
from finrag.answer import Answer, answer
from finrag.data import Qrels, split_queries
from finrag.judge import judge_answer
from finrag.llm import LLM, Usage
from finrag.retrieval.base import Hit, run_ids
from finrag.retrieval.fusion import HybridRetriever, order_head, passage_text, rrf
from finrag.service import CONTEXT_PASSAGES, RagService

Log = Callable[[str], None]
MAIN_METRICS = ["ndcg@10", "mrr@10", "recall@10", "recall@100"]


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]


def write_trec_run(path: Path, run: dict[str, list[str]], tag: str) -> None:
    with path.open("w") as f:
        for qid, docs in run.items():
            for rank, did in enumerate(docs, start=1):
                f.write(f"{qid} Q0 {did} {rank} {1.0 / rank:.6f} {tag}\n")


def score_run(run: dict[str, list[str]], qrels: Qrels) -> dict[str, Any]:
    per_query = metrics.per_query_scores(run, qrels)
    qids = sorted(qrels)
    summary: dict[str, Any] = {}
    for name in MAIN_METRICS:
        vals = [per_query[name][q] for q in qids]
        lo, hi = metrics.bootstrap_ci(vals)
        summary[name] = {"mean": round(metrics.mean(vals), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return {"summary": summary, "per_query": per_query}


def eval_retrieval(
    service: RagService, split: str, modes: Sequence[str], out_dir: Path, log: Log = print
) -> dict[str, Any]:
    queries, qrels = split_queries(service.settings.fiqa_dir, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"{split}: {len(queries)} queries, {sum(len(v) for v in qrels.values())} judgments")
    results: dict[str, Any] = {}
    qids = sorted(qrels)
    for mode in modes:
        retriever = service.retriever(mode)
        t0 = time.perf_counter()
        run = run_ids(retriever.search_many(queries, 100))
        seconds = time.perf_counter() - t0
        write_trec_run(out_dir / f"run-{split}-{mode}.trec", run, mode)
        scored = score_run(run, qrels)
        scored["seconds_total"] = round(seconds, 2)
        results[mode] = scored
        s = scored["summary"]
        log(
            f"  {mode:<16} nDCG@10 {s['ndcg@10']['mean']:.4f}  MRR@10 {s['mrr@10']['mean']:.4f}  "
            f"R@10 {s['recall@10']['mean']:.4f}  R@100 {s['recall@100']['mean']:.4f}  "
            f"({seconds:.1f} s)"
        )
    # Paired significance: each variant against BM25 and against the one listed before it.
    comparisons = []
    for i, mode in enumerate(modes):
        refs = {modes[0]} | ({modes[i - 1]} if i > 0 else set())
        for ref in sorted(refs - {mode}):
            a = [results[mode]["per_query"]["ndcg@10"][q] for q in qids]
            b = [results[ref]["per_query"]["ndcg@10"][q] for q in qids]
            comparisons.append(
                {
                    "variant": mode,
                    "baseline": ref,
                    "delta_ndcg@10": round(metrics.mean(a) - metrics.mean(b), 4),
                    "p_value": round(metrics.paired_randomization_test(a, b), 4),
                    "wins": sum(x > y for x, y in zip(a, b, strict=True)),
                    "losses": sum(x < y for x, y in zip(a, b, strict=True)),
                }
            )
    report = {
        "split": split,
        "queries": len(queries),
        "embed_model": service.settings.embed_model,
        "rerank_model": service.settings.rerank_model,
        "variants": {
            m: {k: v for k, v in r.items() if k != "per_query"} for m, r in results.items()
        },
        "comparisons": comparisons,
    }
    (out_dir / f"retrieval-{split}.json").write_text(json.dumps(report, indent=2))
    (out_dir / f"retrieval-{split}-per-query.json").write_text(
        json.dumps({m: r["per_query"] for m, r in results.items()})
    )
    return report


def tune(service: RagService, out_dir: Path, log: Log = print) -> dict[str, Any]:
    """Grid over fusion and rerank depth on the DEV split only."""
    queries, qrels = split_queries(service.settings.fiqa_dir, "dev")
    bm25_run = service.bm25.search_many(queries, 200)
    dense_run = service.dense.search_many(queries, 200)

    class Cached:
        def __init__(self, name: str, hits: dict[str, Any]) -> None:
            self.name = name
            self._hits = hits

        def search(self, query: str, k: int) -> Any:
            raise NotImplementedError

        def search_many(self, qs: Any, k: int) -> dict[str, Any]:
            return {q: self._hits[q][:k] for q in qs}

    lex, den = Cached("bm25", bm25_run), Cached("dense", dense_run)
    rows: list[dict[str, Any]] = []
    for depth in (50, 100, 200):
        for rrf_k in (10, 30, 60, 100):
            for w in ((1.0, 1.0), (1.0, 1.5), (1.0, 2.0), (1.5, 1.0)):
                h = HybridRetriever(lex, den, depth, rrf_k, w)
                ndcg = score_run(run_ids(h.search_many(queries, 100)), qrels)["summary"]
                rows.append(
                    {
                        "depth": depth,
                        "rrf_k": rrf_k,
                        "w_bm25": w[0],
                        "w_dense": w[1],
                        "ndcg@10": ndcg["ndcg@10"]["mean"],
                        "recall@100": ndcg["recall@100"]["mean"],
                    }
                )
    rows.sort(key=lambda r: -r["ndcg@10"])
    for r in rows[:5]:
        log(f"  hybrid {r}")
    # Rerank grid: score the top 100 of the frozen hybrid once per model, then every depth and
    # fusion weight is a re-sort of cached scores.
    base_hits = service.retriever("hybrid").search_many(queries, 100)
    docs = service.fetch(sorted({h.doc_id for hs in base_hits.values() for h in hs}))
    rerank_rows: list[dict[str, Any]] = []
    for label, model in (("base", service.reranker), ("fine-tuned", service.ft_reranker)):
        t0 = time.perf_counter()
        scores = {
            qid: model.score(queries[qid], [passage_text(docs[h.doc_id]) for h in hits])
            for qid, hits in base_hits.items()
        }
        log(f"  scored {label} reranker in {time.perf_counter() - t0:.0f} s")
        for depth in (10, 20, 30, 50, 100):
            for fuse in (None, 0.5, 1.0, 2.0):
                run = {
                    qid: [
                        h.doc_id
                        for h in order_head(hits[:depth], scores[qid][:depth], fuse) + hits[depth:]
                    ]
                    for qid, hits in base_hits.items()
                }
                summ = score_run(run, qrels)["summary"]
                rerank_rows.append(
                    {
                        "reranker": label,
                        "depth": depth,
                        "fuse_weight": fuse,
                        "ndcg@10": summ["ndcg@10"]["mean"],
                        "mrr@10": summ["mrr@10"]["mean"],
                    }
                )
    no_rerank = score_run(run_ids(base_hits), qrels)["summary"]["ndcg@10"]["mean"]
    log(f"  hybrid (frozen defaults), no rerank: {no_rerank}")
    for row in sorted(rerank_rows, key=lambda r: -r["ndcg@10"])[:8]:
        log(f"  rerank {row}")
    report = {
        "split": "dev",
        "queries": len(queries),
        "hybrid_grid": rows,
        "hybrid_frozen_ndcg@10": no_rerank,
        "rerank_grid": rerank_rows,
    }
    (out_dir / "tune-dev.json").write_text(json.dumps(report, indent=2))
    return report


def hnsw_recall(service: RagService, out_dir: Path, log: Log = print) -> dict[str, Any]:
    """How much the approximate index loses against an exact scan, and what it saves."""
    queries, qrels = split_queries(service.settings.fiqa_dir, "test")
    vecs = service.embedder.embed_queries(list(queries.values()))
    out: dict[str, Any] = {"queries": len(queries)}
    exact_run: dict[str, list[str]] = {}
    for label, ef in (
        ("exact", 0),
        ("ef_search=10", 10),
        ("ef_search=40", 40),
        ("ef_search=100", 100),
        ("ef_search=200", 200),
    ):
        lat: list[float] = []
        run: dict[str, list[str]] = {}
        with service.conn.transaction():
            if ef == 0:
                service.conn.execute("SET LOCAL enable_indexscan = off")
            else:
                service.conn.execute(f"SET LOCAL hnsw.ef_search = {ef}")
            for v in vecs[:50]:  # warm-up: index pages into shared buffers for every config
                service.dense.search_vector(v, 10)
            for qid, v in zip(queries, vecs, strict=True):
                t0 = time.perf_counter()
                run[qid] = [h.doc_id for h in service.dense.search_vector(v, 10)]
                lat.append((time.perf_counter() - t0) * 1000)
        if ef == 0:
            exact_run = run
        overlap = metrics.mean(
            [len(set(run[q]) & set(exact_run[q])) / max(1, len(exact_run[q])) for q in queries]
        )
        out[label] = {
            "recall@10_vs_exact": round(overlap, 4),
            "ndcg@10": score_run(run, qrels)["summary"]["ndcg@10"]["mean"],
            "p50_ms": round(percentile(lat, 50), 2),
            "p95_ms": round(percentile(lat, 95), 2),
        }
        log(f"  {label}: {out[label]}")
    (out_dir / "hnsw-test.json").write_text(json.dumps(out, indent=2))
    return out


def bench_latency(
    service: RagService, modes: Sequence[str], n: int, out_dir: Path, log: Log = print
) -> dict[str, Any]:
    """Single-query latency, as the API sees it (one query at a time, warm models)."""
    queries, _ = split_queries(service.settings.fiqa_dir, "test")
    sample = random.Random(7).sample(sorted(queries), min(n, len(queries)))
    out: dict[str, Any] = {"queries": len(sample)}
    for mode in modes:
        for qid in sample[:5]:  # warm-up: model load, MPS kernels, buffer cache
            service.retrieve(queries[qid], mode)
        total, rerank = [], []
        for qid in sample:
            res = service.retrieve(queries[qid], mode)
            total.append(res.timings.retrieve_ms + res.timings.rerank_ms)
            rerank.append(res.timings.rerank_ms)
        out[mode] = {
            "p50_ms": round(percentile(total, 50), 1),
            "p95_ms": round(percentile(total, 95), 1),
            "rerank_p95_ms": round(percentile(rerank, 95), 1),
        }
        log(f"  {mode:<16} {out[mode]}")
    (out_dir / "latency-retrieval.json").write_text(json.dumps(out, indent=2))
    return out


# ---- answers ------------------------------------------------------------------------------


def answer_record(qid: str, question: str, ans: Answer, gold: dict[str, int]) -> dict[str, Any]:
    cited = ans.cited_doc_ids
    return {
        "qid": qid,
        "question": question,
        "answer": ans.text,
        "abstained": ans.abstained,
        "hedged": ans.hedged,
        "passage_ids": [p.id for p in ans.passages],
        "cited_ids": cited,
        "sentences": [{"text": s.text, "citations": list(s.citations)} for s in ans.sentences],
        "invalid_citations": ans.invalid_citations,
        "uncited_sentences": ans.uncited_sentences,
        "context_has_gold": any(p.id in gold for p in ans.passages),
        "cites_gold": any(d in gold for d in cited),
        "prompt_tokens": ans.usage.prompt_tokens,
        "completion_tokens": ans.usage.completion_tokens,
    }


def sample_test_questions(
    service: RagService, n: int, seed: int = 11
) -> list[tuple[str, str, dict[str, int]]]:
    queries, qrels = split_queries(service.settings.fiqa_dir, "test")
    qids = random.Random(seed).sample(sorted(queries), min(n, len(queries)))
    return [(q, queries[q], qrels[q]) for q in qids]


def generate_answers(
    service: RagService, system: str, n: int, out_path: Path, log: Log = print
) -> list[dict[str, Any]]:
    """system: 'rag:<retrieval mode>' (single shot) or 'agent'. Resumable: done qids are skipped."""
    done: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["qid"]] = rec
    sample = sample_test_questions(service, n)
    agent = None
    if system == "agent":
        agent = RagAgent(
            service.llm, service.retriever("hybrid+rerank"), service.fetch, AgentConfig()
        )
    with out_path.open("a") as f:
        for i, (qid, question, gold) in enumerate(sample, start=1):
            if qid in done:
                continue
            t0 = time.perf_counter()
            if agent is not None:
                res = agent.run(question)
                rec = answer_record(qid, question, res.answer, gold)
                rec["searches"] = res.searches
                rec["trace"] = res.trace
                rec["llm_calls"] = res.usage.calls
                rec["prompt_tokens"] = res.usage.prompt_tokens
                rec["completion_tokens"] = res.usage.completion_tokens
            else:
                mode = system.removeprefix("rag:")
                r = service.retrieve(question, mode, k=10)
                ans = answer(service.llm, question, r.passages[:CONTEXT_PASSAGES])
                rec = answer_record(qid, question, ans, gold)
                rec["llm_calls"] = ans.usage.calls
            rec["seconds"] = round(time.perf_counter() - t0, 2)
            rec["system"] = system
            rec["model"] = service.llm.model
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[qid] = rec
            if i % 10 == 0:
                log(f"  {system}: {i}/{len(sample)}")
    return [done[q] for q, _, _ in sample if q in done]


def judge_answers(
    service: RagService, judge: LLM, answers_path: Path, out_path: Path, log: Log = print
) -> list[dict[str, Any]]:
    """Adds sentence verdicts from the judge model. Resumable like generate_answers."""
    from finrag.answer import parse_answer

    judged: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            rec = json.loads(line)
            judged[rec["qid"]] = rec
    records = [json.loads(line) for line in answers_path.read_text().splitlines()]
    _, qrels = split_queries(service.settings.fiqa_dir, "test")
    with out_path.open("a") as f:
        for i, rec in enumerate(records, start=1):
            if rec["qid"] in judged:
                continue
            docs = service.fetch(rec["passage_ids"])
            passages = [docs[d] for d in rec["passage_ids"]]
            ans = parse_answer(rec["answer"], passages, Usage())
            j = judge_answer(judge, rec["question"], ans)
            # Re-derive every parsed field from the raw text with the current parser.
            reparsed = answer_record(rec["qid"], rec["question"], ans, qrels.get(rec["qid"], {}))
            for key in ("prompt_tokens", "completion_tokens"):
                reparsed.pop(key)
            rec = {
                **rec,
                **reparsed,
                "judge_model": judge.model,
                "verdicts": [asdict(v) for v in j.verdicts],
                "faithfulness": j.faithfulness,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            judged[rec["qid"]] = rec
            if i % 10 == 0:
                log(f"  judged {i}/{len(records)}")
    return [judged[r["qid"]] for r in records]


def summarize_answers(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answered = [r for r in records if not r["abstained"]]
    sentences = [v for r in answered for v in r.get("verdicts", [])]
    faith = [r["faithfulness"] for r in answered if r.get("faithfulness") is not None]
    secs = [r["seconds"] for r in records]
    return {
        "questions": len(records),
        "abstain_rate": round(1 - len(answered) / len(records), 4) if records else 0,
        "sentences": len(sentences),
        "sentence_support_rate": round(sum(v["supported"] for v in sentences) / len(sentences), 4)
        if sentences
        else None,
        "mean_answer_faithfulness": round(statistics.mean(faith), 4) if faith else None,
        "fully_faithful_answers": round(sum(f == 1.0 for f in faith) / len(faith), 4)
        if faith
        else None,
        "answers_with_invalid_citation": sum(bool(r["invalid_citations"]) for r in answered),
        "uncited_sentences": sum(r["uncited_sentences"] for r in answered),
        "context_has_gold": round(sum(r["context_has_gold"] for r in records) / len(records), 4),
        "cites_gold": round(sum(r["cites_gold"] for r in answered) / len(answered), 4)
        if answered
        else 0,
        "llm_calls_mean": round(statistics.mean(r.get("llm_calls", 1) for r in records), 2),
        "tokens_mean": round(
            statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in records)
        ),
        "seconds_p50": round(percentile(secs, 50), 2),
        "seconds_p95": round(percentile(secs, 95), 2),
    }


def compare_embedders(
    service: RagService, models: Sequence[str], split: str, out_dir: Path, log: Log = print
) -> dict[str, Any]:  # pragma: no cover - embeds the full corpus once per model
    """Dense and tuned-hybrid quality per embedding model, by exact search in memory, so a
    candidate model can be judged on dev without reloading the database."""
    import numpy as np

    from finrag.data import iter_corpus
    from finrag.ingest import embed_corpus, embedding_cache_path
    from finrag.models import SentenceTransformerEmbedder
    from finrag.service import DEFAULT_RRF_K, DEFAULT_RRF_WEIGHTS

    queries, qrels = split_queries(service.settings.fiqa_dir, split)
    docs = list(iter_corpus(service.settings.fiqa_dir))
    ids = [d.id for d in docs]
    bm25 = service.bm25.search_many(queries, 100)
    out: dict[str, Any] = {"split": split, "queries": len(queries)}
    for name in models:
        emb = SentenceTransformerEmbedder(name, service.settings.device)
        cache = Path(embedding_cache_path(service.settings, f"{name}-fp16", len(docs)))
        if cache.exists():
            doc_vecs = np.load(cache)
        else:
            t0 = time.perf_counter()
            doc_vecs = embed_corpus(emb, docs, lambda s: None)
            log(f"  {name}: embedded corpus in {time.perf_counter() - t0:.0f} s")
            np.save(cache, doc_vecs)
        q_vecs = emb.embed_queries(list(queries.values()))
        sims = q_vecs @ doc_vecs.T
        dense: dict[str, list[Hit]] = {}
        for row, qid in enumerate(queries):
            top = np.argsort(-sims[row])[:100]
            dense[qid] = [Hit(ids[i], float(sims[row, i])) for i in top]
        hybrid = {
            qid: rrf([bm25[qid], dense[qid]], DEFAULT_RRF_K, list(DEFAULT_RRF_WEIGHTS))[:100]
            for qid in queries
        }
        out[name] = {
            "dense": score_run(run_ids(dense), qrels)["summary"],
            "hybrid": score_run(run_ids(hybrid), qrels)["summary"],
        }
        log(
            f"  {name}: dense nDCG@10 {out[name]['dense']['ndcg@10']['mean']}  "
            f"hybrid {out[name]['hybrid']['ndcg@10']['mean']}  "
            f"R@100 {out[name]['dense']['recall@100']['mean']}"
        )
        del emb
    (out_dir / f"embedders-{split}.json").write_text(json.dumps(out, indent=2))
    return out
