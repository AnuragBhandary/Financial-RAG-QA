# Results

Everything here was measured on one laptop (Apple M5 Pro, 24 GB unified memory, macOS) with
the commands in each section. Raw outputs are in [`results/`](results/). Every knob was chosen
on the **dev** split (500 questions); the **test** split (648 questions, 1,706 relevance
judgments) was used only for the final report.

## 1. Retrieval on the 648 test questions

| Variant | nDCG@10 [95% CI] | MRR@10 | Recall@10 | Recall@100 |
|---|---|---|---|---|
| Postgres full-text (`ts_rank`, OR query) | 0.1841 [0.163, 0.206] | 0.2273 | 0.2429 | 0.4699 |
| **BM25** (bm25s, stemmed) | **0.2532** [0.229, 0.279] | 0.3133 | 0.3198 | 0.5534 |
| Dense, off-the-shelf bge-base-en-v1.5 | 0.4061 [0.378, 0.434] | 0.4858 | 0.4810 | 0.7420 |
| Dense, bge-base fine-tuned on FiQA train | 0.4136 [0.385, 0.442] | 0.4908 | 0.4861 | 0.7582 |
| Hybrid (BM25 + fine-tuned dense, RRF k=10, dense ×2) | 0.4249 [0.398, 0.453] | 0.5005 | 0.5077 | 0.7691 |
| Hybrid → off-the-shelf reranker **alone** | 0.3649 [0.339, 0.390] | 0.4227 | 0.4656 | 0.7691 |
| Hybrid → fine-tuned reranker, fused | 0.4283 [0.400, 0.456] | 0.4983 | 0.5157 | 0.7691 |
| Dense → reranker, fused | 0.4234 [0.395, 0.452] | 0.5070 | 0.4923 | 0.7582 |
| **Hybrid → reranker (top 30), fused with hybrid order** | **0.4347** [0.407, 0.463] | **0.5117** | **0.5167** | 0.7691 |

`finrag eval-retrieval --split test` and `finrag compare-embedders --split test`.

### Which differences are real (paired randomization test, 10,000 sign flips)

| Final system vs | Δ nDCG@10 | Relative | p | Queries better / worse |
|---|---|---|---|---|
| BM25 | +0.1815 | **+71.7%** | 0.0001 | 353 / 58 |
| Off-the-shelf dense retrieval | +0.0286 | **+7.0%** | 0.0003 | 207 / 139 |
| Fine-tuned dense | +0.0211 | +5.1% | 0.0005 | 190 / 138 |
| Hybrid without reranking | +0.0098 | +2.3% | 0.035 | 167 / 123 |
| Same pipeline with the fine-tuned reranker | +0.0064 | +1.5% | 0.18 (not significant) | 154 / 136 |

| Other comparisons | Δ nDCG@10 | p |
|---|---|---|
| Fine-tuned vs off-the-shelf embedder, dense alone | +0.0075 (+1.8%) | 0.28 (not significant) |
| Hybrid vs dense | +0.0113 (+2.7%) | 0.015 |
| Reranker alone vs the hybrid it reorders | −0.0599 (−14.1%) | 0.0001 |

`scripts/key_comparisons.py`.

### What the measurements changed

The textbook pipeline is BM25 + embeddings + cross-encoder. Measured stage by stage on dev,
two of the three default choices made things worse:

1. **Equal-weight hybrid hurt.** BM25 is far weaker than the embeddings on FiQA (0.25 vs
   0.41): forum questions and answers rarely share exact words. With RRF k=60 and equal
   weights, fusion pulled the dense ranking *down* (dev, off-the-shelf embedder: 0.3722 vs
   0.4225 for dense alone).
   Tuned on dev (k=10, dense weighted 2×), hybrid adds a small significant gain.
2. **The off-the-shelf reranker hurt, badly** (−14% on test). bge-reranker-base was trained on
   web search and general QA, and FiQA's labels mean "an answer posted in that thread". On its
   own its order is worse than the retrieval it reorders, and the deeper it reranks the worse it
   gets (dev: 0.41 at depth 10 → 0.31 at depth 100).
3. **Fusing the reranker with the first-stage order fixed it.** RRF of the hybrid order and the
   cross-encoder order (cross-encoder weight 0.5) keeps both signals: +2.3% over hybrid on test.
4. **Fine-tuning helped where it was not obvious.** The fine-tuned embedder is not significantly
   better on its own (+1.8%, p=0.28) but raises recall@100 from 0.742 to 0.758, and the extra
   relevant answers it pulls into the candidate list are what hybrid fusion and reranking use.
5. **The fine-tuned reranker was not worth it.** It beat the off-the-shelf reranker when used
   alone (dev, off-the-shelf embedder, depth 10: 0.4232 vs 0.4109) but, stacked on the fine-tuned embedder, lost to the
   off-the-shelf one fused (dev 0.4353 vs 0.4447; test 0.4283 vs 0.4347, p=0.18). Both
   fine-tuned models learned from the same noisy labels; the off-the-shelf reranker adds a
   *different* signal. It is kept as `+rerank-ft` and measured, not used by default.

### Tuning on dev (500 questions)

| Hybrid fusion (fine-tuned embedder) | nDCG@10 |
|---|---|
| RRF k=60, weights 1:1 (the common default) | 0.3895 |
| **RRF k=10, BM25:dense = 1:2** (chosen) | **0.4317** |
| best of 48 combinations (candidate depth 200) | 0.4320 |

| Reranker on top of hybrid, by depth (fusion weight 0.5) | 10 | 20 | **30** | 50 | 100 |
|---|---|---|---|---|---|
| off-the-shelf bge-reranker-base | 0.4385 | 0.4426 | **0.4438** | 0.4447 | 0.4412 |
| fine-tuned on FiQA train | 0.4353 | 0.4319 | 0.4313 | 0.4263 | 0.4216 |
| off-the-shelf, cross-encoder order alone | 0.4130 | 0.3884 | 0.3662 | 0.3535 | 0.3184 |

Depth 30 was chosen over 50: within 0.001 nDCG and 40% fewer pairs to score. `finrag tune`.

With the off-the-shelf embedder (before fine-tuning), the same grid gave hybrid 0.4223 and the
best fused rerank 0.4383 (fine-tuned reranker, depth 10), versus 0.4447 with the fine-tuned
embedder.

### Fine-tuning runs (FiQA train split: 5,500 questions, 14,166 judged pairs)

| | Embedder (bge-base-en-v1.5, 110M) | Reranker (bge-reranker-base, 278M) |
|---|---|---|
| Data | 14,166 (question, answer, hard negative) triplets | 52,666 pairs: 14,166 positive, 38,500 hard negative |
| Hard negatives | hybrid top 50, minus judged, minus the top 3 unjudged | same |
| Loss | Cached MultipleNegativesRankingLoss (GradCache), batch 32, mini-batch 8 | Binary cross-entropy, pos_weight = 2.72 |
| Schedule | 1 epoch, lr 2e-5, warmup 10%, max 256 tokens | 1 epoch, lr 2e-5, warmup 10%, max 256 tokens |
| Time (M5 Pro GPU) | 22 min | 43 min |

Train, dev and test questions are disjoint (checked in code). `finrag train-embedder`,
`finrag train-reranker`.

## 2. Vector index (pgvector HNSW, m=16, ef_construction=128)

| Search | Recall@10 vs exact | nDCG@10 | p50 | p95 |
|---|---|---|---|---|
| Exact scan | 1.0000 | 0.4138 | 60.7 ms | 65.3 ms |
| HNSW, ef_search=10 | 0.9205 | 0.4069 | 0.72 ms | 1.27 ms |
| HNSW, ef_search=40 (pgvector default) | 0.9841 | 0.4108 | 0.97 ms | 1.27 ms |
| **HNSW, ef_search=100** | **0.9965** | **0.4136** | **1.70 ms** | **2.50 ms** |
| HNSW, ef_search=200 | 0.9986 | 0.4136 | 2.92 ms | 4.44 ms |

At ef_search=100, HNSW finds 99.65% of the exact top 10 at **35× lower latency**, with no loss in
nDCG. 648 test queries, top 10, SQL time only (query vectors precomputed), warm buffers.
`finrag hnsw`.

## 3. Ingest

| Step | Time |
|---|---|
| Embed 57,638 documents (bge-base, fp16 on the laptop GPU) | 150 s (~385 docs/s) |
| BM25 index | 2.2 s |
| Postgres `COPY` (binary) + HNSW + GIN build, one transaction | 23 s |

fp16 vs fp32 on Apple GPUs: 422 vs 129 docs/s for the embedder, 257 vs 88 pairs/s for the
cross-encoder; dev nDCG@10 for dense retrieval 0.4213 (fp16) vs 0.4225 (fp32).

## 4. Latency

### Retrieval (single query, warm models, 300 test questions)

| Mode | p50 | p95 |
|---|---|---|
| BM25 | 0.5 ms | 0.7 ms |
| Dense (query embedding + HNSW) | 12.8 ms | 17.4 ms |
| Hybrid | 14.6 ms | 17.7 ms |
| **Hybrid + rerank (30 pairs)** | **84.7 ms** | **94.9 ms** (rerank: 75.8 ms) |

`finrag bench --n 300`.

### Answers through the API (POST /ask, SSE, 60 sequential test questions after 3 warm-up)

| Client-side | p50 | p95 | max |
|---|---|---|---|
| Passages event (retrieval done) | 694 ms | 1,195 ms | 1,378 ms |
| **First answer token** | **1.81 s** | **2.27 s** | 2.59 s |
| Full answer (qwen3:8b, ~95 tokens) | 4.17 s | 5.61 s | 5.79 s |

`scripts/bench_api.py --n 60` against `finrag serve` with Ollama on the same laptop.

**Why retrieval is 690 ms here but 85 ms in §4:** memory, not code. This 24 GB laptop also runs a
12 GB Docker VM; with qwen3:8b resident (6 GB), the embedding and reranking models get paged
out between requests. With the LLM unloaded, `/search` on the same server measured 14-34 ms
retrieval + 72-81 ms rerank. In production the LLM would run on separate hardware (or Groq,
which the client already supports), and retrieval would be back near 100 ms.

## 5. Answer quality (150 seeded test questions, same questions for every system)

Generator qwen3:8b (Ollama, temperature 0, `think=false`), top 5 passages. Judge qwen3:14b,
a different and larger model, sentence by sentence against the cited passages only.

| | Single-shot RAG (final retriever) | Single-shot, BM25 retrieval | Agent (LangGraph) |
|---|---|---|---|
| Answer sentences judged | 575 | 515 | 467 |
| **Sentences supported by their citations** | **80.2%** | 69.1% | **93.4%** |
| Answers with every sentence supported | 45.9% | 37.8% | 82.6% |
| Sentences with no citation (unsupported by rule) | 38 | 47 | 0 |
| Citations to passages that do not exist | 0 | 0 | 0 |
| Abstained ("the passages do not answer") | 2.7% | 10.0% | 4.0% |
| Context contains a human-labelled relevant passage | 65.3% | 44.0% | 64.0% |
| Answer cites a human-labelled relevant passage | 57.5% | 45.2% | 56.3% |
| LLM calls / tokens per question | 1 / 1,085 | 1 / 1,167 | 3.97 / 2,852 |
| Latency p50 / p95 (offline, warm) | 3.4 s / 4.6 s | 3.3 s / 4.3 s | 12.5 s / 15.9 s |

Per-question faithfulness, paired over questions both systems answered
(`scripts/answer_comparisons.py`):

| Comparison | Questions | Faithfulness | p | Better / worse |
|---|---|---|---|---|
| Final retriever vs BM25 retrieval | 134 | 0.813 vs 0.697 | 0.0001 | 63 / 32 |
| Agent vs single-shot | 143 | 0.936 vs 0.808 | 0.0001 | 65 / 13 |

**Reading it:**

- **Better retrieval means more grounded answers.** Same model, same prompt, only the passages
  differ: +11.6 points of faithfulness, and BM25's weaker context makes the model abstain 4×
  as often.
- **The agent's gain comes from its self-check, not from searching better.** Its evidence
  contains a relevant passage as often as single-shot's (64.0% vs 65.3%); the grader asked for
  a follow-up search on only 4 of 150 questions. What changed is that the verify step dropped
  at least one sentence in 79 of 150 answers. That costs 4× the LLM calls and 3.7× the latency.
- **Caveat on the agent's 93%:** its self-check uses the same judge prompt as the evaluation
  (with the 8B model instead of the 14B), so the two share blind spots. The hand labels below
  are what say whether the dropped sentences were really unsupported.
- **Side effect:** dropping sentences can leave an answer that starts mid-thought ("They may
  also..."). A rewrite step after verification would fix it at one more LLM call.

## 6. Is the judge right? (hand labels)

120 judged sentences drawn stratified by verdict (60 the judge called supported, 60 not; pool:
1,253 supported, 219 not, rule-decided sentences excluded), across all three systems.
`finrag label-sample` → `finrag label` → `finrag agreement`.

HAND_LABELS_PENDING

## Reproduce

```bash
make install data infra
uv run finrag train-embedder          # 22 min on an M-series GPU (optional: falls back to base)
make ingest tune eval hnsw bench
uv run finrag train-reranker          # 43 min (only for the comparison row)
uv run python scripts/key_comparisons.py
make answers judge                    # needs Ollama with qwen3:8b and qwen3:14b
uv run finrag label-sample runs/answers-*-judged.jsonl && uv run finrag label && uv run finrag agreement
```
