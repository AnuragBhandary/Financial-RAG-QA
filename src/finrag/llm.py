"""Chat clients for Ollama (local) and Groq (OpenAI-compatible), over plain httpx.

Both return text plus token counts, stream token deltas, and can be asked for JSON that matches
a schema. No vendor SDKs: the whole contract is visible in this file.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from finrag.config import Settings

Message = dict[str, str]


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, other: Usage) -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.seconds += other.seconds


@dataclass
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion: ...

    def stream(
        self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> Iterator[str]: ...


def parse_json(text: str) -> Any:
    """Models sometimes wrap JSON in prose or fences; take the outermost object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise LLMError(f"no JSON object in model output: {text[:200]!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"bad JSON from model: {e}: {text[:200]!r}") from e


class OllamaLLM:
    def __init__(
        self,
        model: str,
        base_url: str,
        num_ctx: int = 8192,
        timeout: float = 300,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._num_ctx = num_ctx
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def _body(
        self, messages: list[Message], temperature: float, max_tokens: int, stream: bool
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": False,  # qwen3 reasons out loud by default; answers here must be direct
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self._num_ctx,
                "seed": 7,
            },
        }

    def complete(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        body = self._body(messages, temperature, max_tokens, stream=False)
        if schema is not None:
            body["format"] = schema  # Ollama constrains decoding to this JSON schema
        t0 = time.perf_counter()
        r = self._http.post("/api/chat", json=body)
        if r.status_code != 200:
            raise LLMError(f"ollama {r.status_code}: {r.text[:300]}")
        data = r.json()
        return Completion(
            text=data["message"]["content"],
            usage=Usage(
                calls=1,
                prompt_tokens=int(data.get("prompt_eval_count", 0)),
                completion_tokens=int(data.get("eval_count", 0)),
                seconds=time.perf_counter() - t0,
            ),
        )

    def stream(
        self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> Iterator[str]:
        body = self._body(messages, temperature, max_tokens, stream=True)
        with self._http.stream("POST", "/api/chat", json=body) as r:
            if r.status_code != 200:
                raise LLMError(f"ollama {r.status_code}: {r.read()[:300]!r}")
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    yield delta
                if chunk.get("done"):
                    break


class GroqLLM:
    """Groq's OpenAI-compatible endpoint. Retries 429s using the server's retry-after."""

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout: float = 120,
        max_retries: int = 6,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set (put it in .env)")
        self.model = model
        self._max_retries = max_retries
        self._http = httpx.Client(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            r = self._http.post("/chat/completions", json=body)
            if r.status_code != 429 or attempt == self._max_retries:
                return r
            time.sleep(float(r.headers.get("retry-after", 2**attempt)))
        raise AssertionError("unreachable")

    def complete(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": 7,
        }
        if schema is not None:
            body["response_format"] = {"type": "json_object"}
            body["messages"] = [
                *messages[:-1],
                {
                    "role": messages[-1]["role"],
                    "content": messages[-1]["content"]
                    + "\n\nReturn only JSON matching this schema:\n"
                    + json.dumps(schema),
                },
            ]
        t0 = time.perf_counter()
        r = self._post(body)
        if r.status_code != 200:
            raise LLMError(f"groq {r.status_code}: {r.text[:300]}")
        data = r.json()
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"],
            usage=Usage(
                calls=1,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                seconds=time.perf_counter() - t0,
            ),
        )

    def stream(
        self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> Iterator[str]:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        with self._http.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise LLMError(f"groq {r.status_code}: {r.read()[:300]!r}")
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                if delta:
                    yield delta


def make_llm(settings: Settings, model: str | None = None) -> LLM:
    model = model or settings.default_llm_model()
    if settings.llm_provider == "groq":
        return GroqLLM(model, settings.groq_api_key)
    if settings.llm_provider == "ollama":
        return OllamaLLM(model, settings.ollama_url)
    raise LLMError(f"unknown LLM_PROVIDER {settings.llm_provider!r}")
