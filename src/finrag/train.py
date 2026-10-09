"""Fine-tune the cross-encoder reranker on FiQA's train split (never dev or test).

Off the shelf, bge-reranker-base makes FiQA rankings worse than the dense retriever alone (it was
trained on web search and Chinese/English QA, not forum answers about money). In-domain training:

- positives: the train split's relevance judgments (14,166 pairs over 5,500 questions)
- hard negatives: passages the first-stage retriever ranks highly that are not judged relevant.
  The top few unjudged hits are skipped because FiQA's labels are incomplete, so the very top
  "negatives" are often relevant answers nobody labelled (false negatives teach the wrong thing).
- loss: binary cross-entropy on (question, passage) pairs, positives up-weighted to the
  negative:positive ratio so the model is not rewarded for saying "no" to everything.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from finrag.data import split_queries
from finrag.retrieval.fusion import passage_text
from finrag.service import RagService

Log = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


def mine_pairs(
    service: RagService,
    mode: str = "hybrid",
    depth: int = 50,
    skip_top: int = 3,
    negatives_per_query: int = 7,
    seed: int = 7,
    limit: int | None = None,
) -> list[tuple[str, str, float]]:
    queries, qrels = split_queries(service.settings.fiqa_dir, "train")
    if limit is not None:
        queries = dict(list(queries.items())[:limit])
    hits = service.retriever(mode).search_many(queries, depth)
    rng = random.Random(seed)
    rows: list[tuple[str, str, str, float]] = []  # (qid, query, doc id, label)
    for qid, text in queries.items():
        gold = qrels[qid]
        for did in gold:
            rows.append((qid, text, did, 1.0))
        unjudged = [h.doc_id for h in hits[qid] if h.doc_id not in gold]
        pool = unjudged[skip_top:]
        for did in rng.sample(pool, min(negatives_per_query, len(pool))):
            rows.append((qid, text, did, 0.0))
    docs = service.fetch(sorted({r[2] for r in rows}))
    pairs = [(q, passage_text(docs[d]), label) for _, q, d, label in rows if d in docs]
    rng.shuffle(pairs)
    return pairs


def train_reranker(
    service: RagService,
    out_dir: Path,
    epochs: float = 1.0,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    limit: int | None = None,
    log: Log = _print,
) -> dict[str, Any]:  # pragma: no cover - minutes of GPU time; exercised by `make train`
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    t0 = time.perf_counter()
    pairs = mine_pairs(service, limit=limit)
    n_pos = sum(1 for p in pairs if p[2] == 1.0)
    n_neg = len(pairs) - n_pos
    log(
        f"mined {len(pairs):,} pairs ({n_pos:,} positive, {n_neg:,} hard negative) "
        f"in {time.perf_counter() - t0:.0f} s"
    )

    model = CrossEncoder(service.settings.rerank_model, num_labels=1, max_length=max_length)
    ds = Dataset.from_dict(
        {
            "query": [p[0] for p in pairs],
            "passage": [p[1] for p in pairs],
            "label": [p[2] for p in pairs],
        }
    )
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(n_neg / max(1, n_pos)))
    args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=100,
        save_strategy="no",
        seed=7,
        report_to="none",
        dataloader_drop_last=True,
    )
    from transformers import TrainerCallback

    class FreeMpsCache(TrainerCallback):
        """Dynamic padding gives nearly every batch a new shape, and the MPS allocator keeps
        buffers for each one: without this the process grew to 17 GB and swapped."""

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if state.global_step % 25 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    t1 = time.perf_counter()
    CrossEncoderTrainer(
        model=model, args=args, train_dataset=ds, loss=loss, callbacks=[FreeMpsCache()]
    ).train()
    train_s = time.perf_counter() - t1
    model.save_pretrained(str(out_dir))
    meta = {
        "base_model": service.settings.rerank_model,
        "pairs": len(pairs),
        "positives": n_pos,
        "hard_negatives": n_neg,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "train_seconds": round(train_s),
        "device": str(model.device),
    }
    (out_dir / "finrag_train.json").write_text(json.dumps(meta, indent=2))
    log(json.dumps(meta, indent=2))
    return meta


def mine_triplets(
    service: RagService,
    mode: str = "hybrid",
    depth: int = 50,
    skip_top: int = 3,
    seed: int = 7,
    limit: int | None = None,
) -> list[tuple[str, str, str, str]]:
    """(query id, query, positive passage, hard negative passage): one row per judged pair."""
    queries, qrels = split_queries(service.settings.fiqa_dir, "train")
    if limit is not None:
        queries = dict(list(queries.items())[:limit])
    hits = service.retriever(mode).search_many(queries, depth)
    rng = random.Random(seed)
    rows: list[tuple[str, str, str, str]] = []
    for qid, text in queries.items():
        gold = qrels[qid]
        pool = [h.doc_id for h in hits[qid] if h.doc_id not in gold][skip_top:]
        for did in gold:
            if pool:
                rows.append((qid, text, did, rng.choice(pool)))
    docs = service.fetch(sorted({d for r in rows for d in r[2:]}))
    out = [
        (qid, q, passage_text(docs[p]), passage_text(docs[n]))
        for qid, q, p, n in rows
        if p in docs and n in docs
    ]
    rng.shuffle(out)
    return out


def train_embedder(
    service: RagService,
    out_dir: Path,
    epochs: float = 1.0,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 256,
    limit: int | None = None,
    log: Log = _print,
) -> dict[str, Any]:  # pragma: no cover - GPU minutes; exercised by `make train`
    """Contrastive fine-tune of the bi-encoder: (cached) MultipleNegativesRankingLoss with one hard
    negative per (question, answer) pair plus every other passage in the batch as a negative.
    NO_DUPLICATES batching keeps two answers to the same question out of one batch, where each
    would otherwise be trained as a negative for the other."""
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    from transformers import TrainerCallback

    from finrag.models import BGE_QUERY_INSTRUCTION

    t0 = time.perf_counter()
    rows = mine_triplets(service, limit=limit)
    log(
        f"mined {len(rows):,} (question, answer, hard negative) triplets "
        f"in {time.perf_counter() - t0:.0f} s"
    )
    base = service.settings.base_embed_model
    # Eager attention: MPS scaled_dot_product_attention does not support dropout in training.
    model = SentenceTransformer(base, model_kwargs={"attn_implementation": "eager"})
    model.max_seq_length = max_length
    prefix = BGE_QUERY_INSTRUCTION if "bge" in base.lower() else ""
    ds = Dataset.from_dict(
        {
            "anchor": [prefix + r[1] for r in rows],
            "positive": [r[2] for r in rows],
            "negative": [r[3] for r in rows],
        }
    )

    class FreeMpsCache(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if state.global_step % 25 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50,
        save_strategy="no",
        seed=7,
        report_to="none",
    )
    t1 = time.perf_counter()
    SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        # GradCache: embed in mini-batches without gradients, compute the loss over the whole
        # batch, then backprop mini-batch by mini-batch. Same loss as plain MNRL; plain MNRL
        # needed more than 12 GB of GPU memory at batch 32 x 3 texts x 256 tokens.
        loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=8),
        callbacks=[FreeMpsCache()],
    ).train()
    train_s = time.perf_counter() - t1
    model.save_pretrained(str(out_dir))
    meta = {
        "base_model": base,
        "triplets": len(rows),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "train_seconds": round(train_s),
        "device": str(model.device),
    }
    (out_dir / "finrag_train.json").write_text(json.dumps(meta, indent=2))
    log(json.dumps(meta, indent=2))
    return meta
