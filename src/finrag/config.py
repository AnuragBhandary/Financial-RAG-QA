"""Settings, read once from the environment (a .env file is loaded by the CLI and the API)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader: KEY=VALUE lines; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0]  # inline comment
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: _env(
            "DATABASE_URL", "postgresql://finrag:finrag@127.0.0.1:5436/finrag"
        )
    )
    data_dir: Path = field(default_factory=lambda: Path(_env("FINRAG_DATA_DIR", "data")))
    runs_dir: Path = field(default_factory=lambda: Path(_env("FINRAG_RUNS_DIR", "runs")))

    # Fine-tuned on FiQA train (`finrag train-embedder`); falls back to the base model if absent.
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "models/bge-base-fiqa"))
    base_embed_model: str = field(
        default_factory=lambda: _env("BASE_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    )
    rerank_model: str = field(
        default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-base")
    )
    # Fine-tuned on FiQA train (`finrag train-reranker`): measured, and not used by default.
    ft_rerank_model: str = field(
        default_factory=lambda: _env("FT_RERANK_MODEL", "models/reranker-fiqa")
    )
    device: str = field(default_factory=lambda: _env("FINRAG_DEVICE", "auto"))

    # "ollama" (local, default) or "groq" (needs GROQ_API_KEY).
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", ""))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://127.0.0.1:11434"))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))

    def resolved_embed_model(self) -> str:
        """The configured embedder, or the base one when a local fine-tuned model is missing
        (a fresh clone before `finrag train-embedder`)."""
        local = self.embed_model.startswith(("models/", "./", "/"))
        if local and not Path(self.embed_model).exists():
            return self.base_embed_model
        return self.embed_model

    @property
    def fiqa_dir(self) -> Path:
        return self.data_dir / "fiqa"

    @property
    def bm25_dir(self) -> Path:
        return self.data_dir / "bm25_index"

    def default_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return "llama-3.3-70b-versatile" if self.llm_provider == "groq" else "qwen3:8b"
