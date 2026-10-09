.PHONY: install data infra down ingest tune eval hnsw bench answers judge results serve test lint typecheck check

install:            ## Python deps incl. torch + sentence-transformers (needs uv)
	uv sync --extra models --extra train

data:               ## download FiQA-2018 in BEIR format (~18 MB) into data/
	mkdir -p data && curl -sSL -o data/fiqa.zip \
		https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip \
		&& cd data && unzip -o -q fiqa.zip && rm fiqa.zip

infra:              ## PostgreSQL 17 + pgvector on localhost:5436
	docker compose up -d --wait postgres

down:
	docker compose down

ingest:             ## embed 57,638 docs (cached), load pgvector + HNSW, build BM25
	uv run finrag ingest

tune:               ## fusion and rerank-depth grid, dev split only
	uv run finrag tune

eval:               ## every retriever variant on the 648 test queries
	uv run finrag eval-retrieval --split test

hnsw:               ## HNSW recall and latency vs an exact scan
	uv run finrag hnsw

bench:              ## single-query retrieval latency, p50/p95
	uv run finrag bench

answers:            ## 150 seeded test questions: single-shot RAG and the agent (needs Ollama)
	uv run finrag answers --system rag:hybrid+rerank
	uv run finrag answers --system rag:bm25
	uv run finrag answers --system agent

judge:              ## LLM judge (JUDGE_MODEL, default qwen3:14b) over each answer set
	uv run finrag judge --system rag:hybrid+rerank
	uv run finrag judge --system rag:bm25
	uv run finrag judge --system agent

serve:              ## API + demo page on http://127.0.0.1:8000
	uv run finrag serve

test:
	uv run pytest --cov --cov-report=term

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy

check: lint typecheck test
