"""Hand labels for the faithfulness judge: draw a sample, label it in the terminal, score agreement.

The sample is stratified by the judge's verdict (half "supported", half "not") because a judge
that says "supported" 90% of the time would otherwise leave too few "not supported" cases to
measure. Agreement is reported per stratum and re-weighted back to the real verdict mix.
"""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path
from typing import Any

from finrag.answer import truncate_words
from finrag.judge import cohens_kappa
from finrag.service import RagService


def make_label_sample(service: RagService, judged_files: list[Path], n: int, out: Path) -> None:
    pool: list[dict[str, Any]] = []
    for path in judged_files:
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            for i, v in enumerate(rec.get("verdicts", [])):
                if v["by_rule"]:
                    continue  # rule-decided (no valid citation): nothing for a person to check
                pool.append(
                    {
                        "id": f"{rec['system']}|{rec['qid']}|{i}",
                        "system": rec["system"],
                        "question": rec["question"],
                        "sentence": v["sentence"],
                        "citations": v["citations"],
                        "passage_ids": rec["passage_ids"],
                        "judge_supported": v["supported"],
                    }
                )
    rng = random.Random(13)
    supported = [p for p in pool if p["judge_supported"]]
    unsupported = [p for p in pool if not p["judge_supported"]]
    half = n // 2
    take_u = min(len(unsupported), half)
    sample = rng.sample(unsupported, take_u) + rng.sample(
        supported, min(len(supported), n - take_u)
    )
    rng.shuffle(sample)
    for item in sample:
        docs = service.fetch(item["passage_ids"])
        item["cited_passages"] = {
            str(c): truncate_words(docs[item["passage_ids"][c - 1]].text)
            for c in item["citations"]
            if 1 <= c <= len(item["passage_ids"]) and item["passage_ids"][c - 1] in docs
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "_meta": True,
        "pool_supported": len(supported),
        "pool_unsupported": len(unsupported),
        "files": [str(p) for p in judged_files],
    }
    with out.open("w") as f:
        f.write(json.dumps(meta) + "\n")
        for item in sample:
            f.write(json.dumps(item) + "\n")
    print(
        f"{len(sample)} sentences -> {out} (pool: {len(supported)} supported, "
        f"{len(unsupported)} not, by the judge)"
    )


def _read(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    meta: dict[str, Any] = next((x for x in lines if x.get("_meta")), {})
    return meta, [x for x in lines if not x.get("_meta")]


def label_loop(settings: Any, sample_path: Path, out_path: Path) -> None:
    _, items = _read(sample_path)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text().splitlines()}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(done)} labeled, {len(todo)} to go. Judge only against the cited passages.\n")
    wrap = textwrap.TextWrapper(width=100, initial_indent="    ", subsequent_indent="    ")
    with out_path.open("a") as f:
        for n, it in enumerate(todo, start=1):
            print("=" * 100)
            print(f"[{len(done) + n}/{len(items)}] Q: {it['question']}\n")
            for num, text in it["cited_passages"].items():
                print(f"  [{num}]")
                print(wrap.fill(text))
                print()
            print(f"  SENTENCE: {it['sentence']}\n")
            while True:
                key = input("  Supported by the cited passages? [y]es / [n]o / [s]kip / [q]uit: ")
                key = key.strip().lower()[:1]
                if key in {"y", "n", "s", "q"}:
                    break
            if key == "q":
                return
            if key == "s":
                continue
            f.write(json.dumps({"id": it["id"], "supported": key == "y"}) + "\n")
            f.flush()


def agreement(sample_path: Path, labels_path: Path) -> dict[str, Any]:
    meta, items = _read(sample_path)
    labels = {
        x["id"]: x["supported"]
        for x in (json.loads(line) for line in labels_path.read_text().splitlines())
    }
    pairs = [(it["judge_supported"], labels[it["id"]]) for it in items if it["id"] in labels]
    if not pairs:
        raise SystemExit("no labeled items yet")
    judge = [j for j, _ in pairs]
    human = [h for _, h in pairs]
    tp = sum(j and h for j, h in pairs)
    tn = sum((not j) and (not h) for j, h in pairs)
    fp = sum(j and not h for j, h in pairs)  # judge said supported, person said not
    fn = sum((not j) and h for j, h in pairs)
    # Precision of each judge verdict = agreement within its stratum.
    prec_sup = tp / (tp + fp) if tp + fp else None
    prec_unsup = tn / (tn + fn) if tn + fn else None
    out: dict[str, Any] = {
        "labeled": len(pairs),
        "raw_agreement": round((tp + tn) / len(pairs), 4),
        "cohens_kappa": round(cohens_kappa(judge, human), 4),
        "confusion": {
            "judge_sup_human_sup": tp,
            "judge_sup_human_unsup": fp,
            "judge_unsup_human_sup": fn,
            "judge_unsup_human_unsup": tn,
        },
        "judge_supported_precision": prec_sup,
        "judge_unsupported_precision": prec_unsup,
    }
    ps, pu = meta.get("pool_supported"), meta.get("pool_unsupported")
    if ps and pu and prec_sup is not None and prec_unsup is not None:
        w = ps / (ps + pu)
        out["reweighted_agreement"] = round(w * prec_sup + (1 - w) * prec_unsup, 4)
        # Human-estimated support rate over the whole pool, correcting the judge's errors.
        out["human_estimated_support_rate"] = round(w * prec_sup + (1 - w) * (1 - prec_unsup), 4)
        out["judge_support_rate"] = round(w, 4)
    return out
