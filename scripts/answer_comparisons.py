"""Paired comparison of answer faithfulness between systems on the same test questions.

Per question: the share of answer sentences the judge found supported. Only questions that both
systems answered (did not abstain) are compared.

    uv run python scripts/answer_comparisons.py
"""

from __future__ import annotations

import json
from pathlib import Path

from finrag import metrics
from finrag.config import Settings

PAIRS = [
    ("rag-hybrid-rerank", "rag-bm25"),
    ("agent", "rag-hybrid-rerank"),
]


def load(runs: Path, system: str) -> dict[str, float]:
    out = {}
    for line in (runs / f"answers-{system}-judged.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if not rec["abstained"] and rec["faithfulness"] is not None:
            out[rec["qid"]] = float(rec["faithfulness"])
    return out


def main() -> None:
    runs = Path(Settings().runs_dir)
    rows = []
    for a, b in PAIRS:
        fa, fb = load(runs, a), load(runs, b)
        qids = sorted(set(fa) & set(fb))
        x, y = [fa[q] for q in qids], [fb[q] for q in qids]
        row = {
            "variant": a,
            "baseline": b,
            "questions_both_answered": len(qids),
            "variant_faithfulness": round(metrics.mean(x), 4),
            "baseline_faithfulness": round(metrics.mean(y), 4),
            "delta": round(metrics.mean(x) - metrics.mean(y), 4),
            "p_value": round(metrics.paired_randomization_test(x, y), 4),
            "better": sum(i > j for i, j in zip(x, y, strict=True)),
            "worse": sum(i < j for i, j in zip(x, y, strict=True)),
        }
        rows.append(row)
        print(row)
    (runs / "answer-comparisons.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
