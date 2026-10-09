# Design

Why each piece is the way it is, what else was considered, and where it stops working.
Numbers live in [RESULTS.md](RESULTS.md).

## 1. The question this project answers

"Does my RAG system work?" is two questions:

1. **Does retrieval find the right passages?** FiQA-2018 has human relevance judgments for 648
   test questions, so this is measured exactly (nDCG@10, MRR@10, recall@10/100). No LLM involved.
2. **Does the answer say only what those passages say?** There is no gold answer to compare
   against, so this needs a judge. The judge is an LLM, so the judge itself is measured against
   hand labels before any of its numbers are trusted.

Everything below serves one of those two measurements.

## 2. Dataset: FiQA-2018 (BEIR)

57,638 answers from a personal-finance Q&A forum (StackExchange-style), 6,648 questions split
train/dev/test, test = 648 questions with 1,706 relevance judgments (about 2.6 relevant answers
per question).

Why FiQA: real user questions with messy wording ("Is it wise to..."), relevance labelled by
people, a standard BEIR benchmark so numbers can be compared with published ones (BM25 0.236
nDCG@10 in the BEIR paper), and small enough to embed on a laptop.

Known limits of the labels: only answers that the original asker's thread contained are marked
relevant, so a retrieved passage from another thread that answers the question equally well
counts as a miss. This **underestimates** every retriever, the strong ones most (they find more
of these unlabelled good passages). Reported numbers are therefore lower bounds and are only
compared with each other, on the same queries.

## 3. Storage: one PostgreSQL table

```sql
docs(id, title, body,
     tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || body)) STORED,
     embedding vector(768))
-- HNSW (m=16, ef_construction=128, cosine) on embedding, GIN on tsv
```

**Why pgvector and not a vector database:** at 57k vectors (176 MB of float32) any store works.
Postgres means one system to run, transactions (the whole index is swapped atomically), real
SQL for looking at the data, and the text, the lexical index and the vectors in one row so a
hit never points at a missing document. A dedicated vector DB starts paying for itself at tens
of millions of vectors or when filtering + ANN need to scale out.

**Atomic reload:** `load_corpus` drops and recreates the table inside one transaction and builds
the indexes after the bulk `COPY ... FORMAT BINARY` (one index build is far faster than
maintaining HNSW row by row). If anything fails, the old index keeps serving. Tested by feeding
mismatched embeddings.

**HNSW vs exact:** pgvector's HNSW returns at most `ef_search` candidates, so `ef_search` is set
per transaction to at least `k` (the default 40 would silently cap a top-100 search at 40).
`finrag hnsw` measures recall@10 of HNSW against an exact scan (`enable_indexscan = off`) at
several `ef_search` values.

## 4. Retrievers

| Variant | What it is | Why it is in the comparison |
|---|---|---|
| `bm25` | Okapi BM25 (k1=1.2, b=0.75), Snowball stemming, English stopwords, via `bm25s` | The lexical baseline every paper reports |
| `pg-fts` | Postgres `tsvector` + `ts_rank(..., 1)`, query lexemes OR-ed | To show that Postgres full-text search is **not** BM25 (no IDF) |
| `dense` | bge-base-en-v1.5 **fine-tuned on FiQA train**, cosine, HNSW | Semantic matching ("pay down debt" ~ "reduce loans") |
| `hybrid` | BM25 top-100 and dense top-100 fused with RRF (k=10, dense weighted 2x) | Lexical and semantic errors are different errors |
| `+rerank` | bge-reranker-base on the top 30, its order **fused** with the hybrid order (RRF, weight 0.5) | The default pipeline |
| `+rerank-only` | the same cross-encoder, its order alone | To show why fusion is needed |
| `+rerank-ft` | a cross-encoder fine-tuned on FiQA train, fused | Measured, not used (RESULTS.md) |

**Why BM25 lives outside Postgres:** `ts_rank` has no inverse document frequency, so "the" and
"amortization" weigh the same once matched. ParadeDB's `pg_search` adds real BM25 to Postgres but
is another extension to install; `bm25s` scores the whole corpus with sparse matrices in a few
milliseconds per query and its index is a few MB on disk.

**Why `plainto_tsquery` is rewritten to OR:** it ANDs every word, and a 12-word question almost
never has all 12 words in one answer, so AND returns nothing for most queries. With OR, the
ranking function matters: `ts_rank_cd` (cover density) scored 0.037 nDCG@10 on dev because it
rewards long posts that repeat common words; `ts_rank` normalised by log length scored 0.190.

**bge query instruction:** bge v1.5 was trained with the prefix
`"Represent this sentence for searching relevant passages: "` on queries only. Leaving it off
or adding it to documents costs accuracy.

**Why RRF and not a weighted sum of scores:** BM25 scores are unbounded and their scale changes
per query; cosine similarities sit in a narrow band. Normalising them (min-max, z-score) is
fragile. RRF uses only ranks: `score(d) = Σ w/(k + rank)`. The paper's `k=60` with equal weights
made hybrid *worse* than dense alone here (BM25 is much weaker on FiQA); on dev, `k=10` with the
dense list weighted 2x was best. A small k makes the top ranks of each list count much more than
the rest, so a strong dense top 5 is not diluted by BM25's long tail.

**Why the reranker's order is fused instead of trusted:** the off-the-shelf cross-encoder alone
was worse than the hybrid ranking it reordered (−14% on test), and worse the deeper it went.
It still carries signal the retrievers lack (it reads query and passage together), so its order
is fused with the hybrid order by RRF, cross-encoder weighted 0.5: +2.3% over hybrid (p=0.035).
Only the top 30 are rescored (one forward pass per pair, so cost is linear in depth); below
that the hybrid order is kept, so recall@100 does not change. Depth and weight chosen on dev.

**Fine-tuning (finrag train-embedder / train-reranker):** both models are trained on FiQA's
train split (5,500 questions, 14,166 judged answers, disjoint from dev and test) with hard
negatives mined from this system's own hybrid retriever: highly ranked answers that are not
labelled relevant. The top 3 unjudged hits are skipped because FiQA's labels are incomplete and
the very top "negatives" are often relevant answers nobody labelled.
- *Embedder:* MultipleNegativesRankingLoss (each question's answer vs every other answer in the
  batch plus its hard negative), batch 32, with **GradCache** (cached MNRL, mini-batch 8): plain
  MNRL needed more than 12 GB of GPU memory at this batch size; GradCache computes the same loss
  in mini-batches. Batches never contain two answers to the same question (each would be
  trained as the other's negative). Result: recall@100 0.742 → 0.758 and the best hybrid; on its
  own, not significantly better (+1.8%, p=0.28).
- *Reranker:* binary cross-entropy on 52,666 pairs, positives weighted by the 2.7:1 negative
  ratio. Better than the off-the-shelf reranker on its own, but once the embedder was
  fine-tuned, stacking it lost to the off-the-shelf one fused (both learned from the same noisy
  labels). Kept as `+rerank-ft` for the comparison.
- *Apple-GPU specifics:* fp16 inference (3x faster); an `empty_cache` callback during training,
  because dynamic padding gives almost every batch a new shape and the MPS allocator kept every
  one (the process grew to 17 GB and swapped); eager attention for the embedder, because MPS
  `scaled_dot_product_attention` has no dropout support.

**Tuning discipline:** every knob (RRF k, weights, depth, rerank depth and fusion weight, which
models to fine-tune and use) was chosen on the dev split (500 queries). The test split was run
once with the frozen configuration; the only later change was the pg-fts ranking function,
re-chosen on dev, which affects only that baseline row. Tuning on test would report a number the
system will not reproduce on new questions.

**Guard:** the index records which embedding model built it, and the dense retriever refuses to
run if the query model differs (query vectors from one model against document vectors from
another are silently meaningless).

## 5. Statistics

Each metric is reported with a **95% bootstrap confidence interval** over queries (2,000
resamples), and every comparison has a **paired randomization test** (10,000 sign flips of the
per-query differences) plus win/loss counts. With 648 queries a 0.01 nDCG difference is often
noise; the p-value says which differences are real.

Per-query scores and TREC-format run files are written to `runs/`, so any number can be
recomputed (or checked with `trec_eval`) without rerunning retrieval.

## 6. Generation

- Top 5 passages after reranking, each truncated to 220 words, numbered `[1]`..`[5]`.
- The prompt requires a citation on every sentence, forbids outside facts, and gives an exact
  abstention sentence for when the passages do not answer.
- The output is parsed into sentences with their citations. Citations to passages that do not
  exist (`[7]` when there are 5) are **flagged**, not silently dropped, and counted in results.
- An answer is "abstained" only if the abstention sentence is all it says. The model sometimes
  answers and *then* appends the abstention line; the first parser treated any occurrence as an
  abstention and threw the answer away. Now that line is removed from the judged sentences and
  the answer is flagged `hedged`.
- `temperature=0`, fixed seed; `think=false` for qwen3 (its default chain-of-thought adds
  hundreds of tokens of latency and leaks into the answer).

**Local by default:** Ollama with qwen3:8b on the laptop GPU, no API key, no rate limits, ₹0.
`LLM_PROVIDER=groq` switches to Groq's OpenAI-compatible API (Llama 3.3 70B) with 429 retry.
Both clients are plain `httpx`, so the whole contract (request body, streaming format, token
counts) is visible in `llm.py`.

## 7. Measuring faithfulness

**Unit = sentence**, label = binary: "is everything this sentence claims stated or directly
implied by the passages it cites?" Binary sentence labels can be hand-labeled quickly and
consistently; a 1-10 "faithfulness score" cannot.

- Sentences with no citation, or only out-of-range citations, are unsupported **by rule** (no
  LLM call, and excluded from hand labeling since there is nothing to check).
- The judge sees only the cited passages, writes a short reason **before** its verdict (reasoning
  first improves accuracy; verdict-first lets the model rationalise), and returns JSON constrained
  by a schema.
- **Judge != generator.** qwen3:14b judges qwen3:8b. Same family is a known risk
  (self-preference); the hand labels are what measure whether it matters.

**Checking the judge:** `finrag label-sample` draws judged sentences **stratified** by verdict
(half supported, half not), because a judge that says "supported" 90% of the time would leave
almost no "not supported" cases to check. `finrag label` shows each sentence with its cited
passages in the terminal; `finrag agreement` reports raw agreement, Cohen's kappa, the confusion
matrix, the precision of each verdict, and agreement **re-weighted** to the judge's real verdict
mix (the stratified sample over-represents "not supported").

**Objective cross-check that needs no judge:** FiQA's relevance labels say which passages are
good, so for every answer we also record whether its context contained a relevant passage and
whether it **cited** one.

## 8. The agent

A LangGraph state machine:

```
plan -> retrieve -> grade --(follow-up query, < max rounds)--> retrieve
                         \--(sufficient)--> answer -> verify -> END
```

- **plan:** rewrite the question into 1-3 searches. The original question always runs too, so
  rewriting can only add candidates, never lose the ones single-shot would have found.
- **retrieve:** hybrid+rerank per search, results fused with RRF, top 5 kept as evidence.
- **grade:** "is anything important missing? if so, one follow-up search". Capped at 2 rounds;
  a follow-up identical to an earlier search ends the loop.
- **answer:** the same prompt as single-shot, so differences come from evidence, not wording.
- **verify:** the generator checks its own sentences with the judge prompt and drops unsupported
  ones; if none survive, it abstains.
- Every LLM step degrades gracefully: unparseable planner/grader output falls back to
  single-shot behaviour, an unparseable self-check keeps the answer. Tested.

**Why LangGraph:** the control flow (a loop with an exit condition, then a linear tail) is
explicit and testable node by node; state is a typed dict. The nodes call this project's own
retriever and LLM client, not LangChain wrappers, so nothing is hidden.

**Is it worth it?** Measured on the same 150 questions (RESULTS.md §5): faithfulness 0.81 →
0.94 (p < 0.001), at 4x the LLM calls and 3.7x the latency. The gain comes almost entirely from
the verify step (it dropped a sentence in 79 of 150 answers); query rewriting did not find
better evidence (relevant passage in context 64% vs 65%), and the grader asked for a follow-up
search on 4 of 150 questions. So the cheap version of the agent is "single-shot + verify"; the
planning and grading nodes are where to cut cost.

## 9. Serving

- `GET /search`: any retrieval mode, with timings.
- `POST /ask`: **Server-Sent Events** — `passages` first (the UI shows sources immediately),
  then `token` deltas, then `done` with parsed citations, invalid citations and timings
  (retrieval, rerank, first token, total). Model errors become an `error` event instead of a
  silently closed stream.
- `POST /agent`: the agent with its trace.
- `GET /`: a one-file demo page that renders citations as they stream and flags invalid ones.
- **MCP server** (`finrag mcp`, stdio): `search_finance_qa` and `get_document` tools, so Claude
  Desktop, an IDE or another agent can use the retriever. Tested through a real MCP client
  session.

**Concurrency:** one process holds one Postgres connection and one copy of each model. PyTorch
modules and psycopg connections are not safe to share across threads, so retrieval is
serialised by a lock; token streaming, the slow part, runs outside it. Scaling out means more
processes (each with its own models) behind a load balancer, or moving embedding and reranking
to a batching inference server.

## 10. What I would do next

- A larger reranker (bge-reranker-v2-m3) and listwise training for the in-domain one; a
  query-embedding cache.
- After the agent's verify step, a short rewrite so a dropped sentence never leaves the answer
  starting mid-thought.
- Serve the LLM on separate hardware: on one 24 GB laptop it pushes the retrieval models into
  swap (retrieval 85 ms alone, ~690 ms inside /ask).
- Judge with a stronger, different-family model (Llama 3.3 70B on Groq) and compare its
  agreement with the hand labels against qwen3:14b's.
- Chunking: FiQA answers are short (median ~90 words), so whole-document retrieval is fine here;
  long documents would need chunking with overlap and parent-document expansion.
- Filtered search (by date, source) with pgvector iterative scans.
