"""finrag command line: ingest, evaluate, ask, label, serve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finrag.config import Settings, load_dotenv

RETRIEVAL_MODES = [
    "bm25",
    "pg-fts",
    "dense",
    "hybrid",
    "dense+rerank",
    "hybrid+rerank-only",
    "hybrid+rerank-ft",
    "hybrid+rerank",
]


def _service(settings: Settings):  # type: ignore[no-untyped-def]
    from finrag.service import RagService

    return RagService(settings)


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.ingest import ingest
    from finrag.models import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(settings.embed_model, settings.device)
    stats = ingest(settings, embedder, limit=args.limit)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    (settings.runs_dir / "ingest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def cmd_eval_retrieval(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.evaluate import eval_retrieval

    modes = args.modes.split(",") if args.modes else RETRIEVAL_MODES
    eval_retrieval(_service(settings), args.split, modes, settings.runs_dir)


def cmd_tune(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.evaluate import tune

    tune(_service(settings), settings.runs_dir)


def cmd_hnsw(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.evaluate import hnsw_recall

    hnsw_recall(_service(settings), settings.runs_dir)


def cmd_bench(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.evaluate import bench_latency

    modes = args.modes.split(",") if args.modes else ["bm25", "dense", "hybrid", "hybrid+rerank"]
    bench_latency(_service(settings), modes, args.n, settings.runs_dir)


def _answers_path(settings: Settings, system: str) -> Path:
    return settings.runs_dir / f"answers-{system.replace(':', '-').replace('+', '-')}.jsonl"


def cmd_answers(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.evaluate import generate_answers

    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    generate_answers(_service(settings), args.system, args.n, _answers_path(settings, args.system))


def cmd_judge(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.evaluate import judge_answers, summarize_answers
    from finrag.llm import make_llm

    service = _service(settings)
    judge = make_llm(settings, settings.judge_model or "qwen3:14b")
    src = _answers_path(settings, args.system)
    out = src.with_name(src.stem + "-judged.jsonl")
    records = judge_answers(service, judge, src, out)
    summary = summarize_answers(records)
    summary_path = src.with_name(src.stem + "-summary.json")
    summary_path.write_text(
        json.dumps({"system": args.system, "judge": judge.model, **summary}, indent=2)
    )
    print(json.dumps(summary, indent=2))


def cmd_train(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.train import train_reranker

    train_reranker(
        _service(settings),
        Path(args.out),
        epochs=args.epochs,
        limit=args.limit,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


def cmd_train_embedder(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.train import train_embedder

    train_embedder(_service(settings), Path(args.out), epochs=args.epochs, limit=args.limit)


def cmd_compare_embedders(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.evaluate import compare_embedders

    compare_embedders(_service(settings), args.models.split(","), args.split, settings.runs_dir)


def cmd_ask(args: argparse.Namespace, settings: Settings) -> None:
    service = _service(settings)
    if args.agent:
        from finrag.agent import RagAgent

        agent = RagAgent(service.llm, service.retriever("hybrid+rerank"), service.fetch)
        res = agent.run(args.question)
        for step in res.trace:
            print(f"  · {step}")
        ans = res.answer
    else:
        result = service.ask(args.question, args.mode)
        assert result.answer is not None
        ans = result.answer
        t = result.timings
        print(f"  · retrieve {t.retrieve_ms:.0f} ms, rerank {t.rerank_ms:.0f} ms")
    print()
    print(ans.text)
    print()
    for i, p in enumerate(ans.passages, start=1):
        print(f"[{i}] doc {p.id}: {p.text[:140].strip()}...")


def cmd_label(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.labeling import label_loop

    label_loop(settings, Path(args.sample), Path(args.out))


def cmd_label_sample(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.labeling import make_label_sample

    make_label_sample(_service(settings), [Path(p) for p in args.judged], args.n, Path(args.out))


def cmd_agreement(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.labeling import agreement

    print(json.dumps(agreement(Path(args.sample), Path(args.labels)), indent=2))


def cmd_serve(args: argparse.Namespace, settings: Settings) -> None:
    import uvicorn

    uvicorn.run("finrag.api:app", host=args.host, port=args.port, log_level="info")


def cmd_mcp(args: argparse.Namespace, settings: Settings) -> None:
    from finrag.mcp_server import main as mcp_main

    mcp_main()


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    settings = Settings()
    p = argparse.ArgumentParser(prog="finrag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="embed the corpus, load pgvector, build BM25")
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("eval-retrieval", help="nDCG/MRR/recall for every retriever variant")
    s.add_argument("--split", default="test", choices=["dev", "test"])
    s.add_argument("--modes", default="")
    s.set_defaults(fn=cmd_eval_retrieval)

    s = sub.add_parser("tune", help="fusion + rerank-depth grid on the dev split")
    s.set_defaults(fn=cmd_tune)

    s = sub.add_parser("hnsw", help="HNSW recall and latency vs exact search")
    s.set_defaults(fn=cmd_hnsw)

    s = sub.add_parser("bench", help="single-query retrieval latency")
    s.add_argument("--modes", default="")
    s.add_argument("--n", type=int, default=200)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("train-reranker", help="fine-tune the cross-encoder on the train split")
    s.add_argument("--out", default="models/reranker-fiqa")
    s.add_argument("--epochs", type=float, default=1.0)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--max-length", type=int, default=256)
    s.add_argument("--limit", type=int, default=None, help="train queries (smoke test)")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("train-embedder", help="contrastive fine-tune of the bi-encoder")
    s.add_argument("--out", default="models/bge-base-fiqa")  # "bge" keeps the query prefix
    s.add_argument("--epochs", type=float, default=1.0)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_train_embedder)

    s = sub.add_parser("compare-embedders", help="dense/hybrid quality per embedding model")
    s.add_argument("--models", default="BAAI/bge-base-en-v1.5,models/bge-base-fiqa")
    s.add_argument("--split", default="dev", choices=["dev", "test"])
    s.set_defaults(fn=cmd_compare_embedders)

    s = sub.add_parser("answers", help="generate answers for a seeded sample of test queries")
    s.add_argument("--system", default="rag:hybrid+rerank", help="rag:<mode> or agent")
    s.add_argument("--n", type=int, default=150)
    s.set_defaults(fn=cmd_answers)

    s = sub.add_parser("judge", help="LLM-judge the generated answers, write a summary")
    s.add_argument("--system", default="rag:hybrid+rerank")
    s.set_defaults(fn=cmd_judge)

    s = sub.add_parser("ask", help="answer one question")
    s.add_argument("question")
    s.add_argument("--mode", default="hybrid+rerank")
    s.add_argument("--agent", action="store_true")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("label-sample", help="draw judged sentences for hand labeling")
    s.add_argument("judged", nargs="+")
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--out", default="labels/sample.jsonl")
    s.set_defaults(fn=cmd_label_sample)

    s = sub.add_parser("label", help="hand-label sentences in the terminal (resumable)")
    s.add_argument("--sample", default="labels/sample.jsonl")
    s.add_argument("--out", default="labels/human.jsonl")
    s.set_defaults(fn=cmd_label)

    s = sub.add_parser("agreement", help="judge vs hand labels: accuracy, kappa, confusion")
    s.add_argument("--sample", default="labels/sample.jsonl")
    s.add_argument("--labels", default="labels/human.jsonl")
    s.set_defaults(fn=cmd_agreement)

    s = sub.add_parser("serve", help="run the FastAPI app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("mcp", help="run the MCP server (stdio) exposing search tools")
    s.set_defaults(fn=cmd_mcp)

    args = p.parse_args(argv)
    args.fn(args, settings)


if __name__ == "__main__":
    main()
