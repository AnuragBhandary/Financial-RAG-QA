import json

import httpx
import pytest

from finrag.config import Settings
from finrag.llm import GroqLLM, LLMError, OllamaLLM, make_llm


def test_ollama_complete_sends_schema_and_disables_thinking() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"content": "hi"}, "prompt_eval_count": 12, "eval_count": 3},
        )

    llm = OllamaLLM("qwen3:8b", "http://x", transport=httpx.MockTransport(handler))
    out = llm.complete([{"role": "user", "content": "q"}], schema={"type": "object"})
    assert out.text == "hi"
    assert (out.usage.prompt_tokens, out.usage.completion_tokens, out.usage.calls) == (12, 3, 1)
    assert seen["think"] is False and seen["format"] == {"type": "object"}
    assert seen["stream"] is False


def test_ollama_stream_and_errors() -> None:
    lines = [
        {"message": {"content": "Hel"}},
        {"message": {"content": "lo"}},
        {"message": {"content": ""}, "done": True},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == "missing":
            return httpx.Response(404, text="model not found")
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines) + "\n")

    t = httpx.MockTransport(handler)
    assert "".join(OllamaLLM("m", "http://x", transport=t).stream([])) == "Hello"
    with pytest.raises(LLMError):
        OllamaLLM("missing", "http://x", transport=t).complete([])
    with pytest.raises(LLMError):
        list(OllamaLLM("missing", "http://x", transport=t).stream([]))


def test_groq_retries_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("finrag.llm.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "1"})
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert "schema" in body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"a": 1}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    llm = GroqLLM("m", "key", transport=httpx.MockTransport(handler))
    out = llm.complete([{"role": "user", "content": "q"}], schema={"type": "object"})
    assert out.text == '{"a": 1}' and calls["n"] == 3


def test_groq_stream_parses_sse() -> None:
    sse = "".join(
        f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in ["A", "B"]
    )
    t = httpx.MockTransport(lambda r: httpx.Response(200, text=sse + "data: [DONE]\n\n"))
    assert "".join(GroqLLM("m", "key", transport=t).stream([])) == "AB"


def test_groq_needs_a_key_and_errors_surface() -> None:
    with pytest.raises(LLMError):
        GroqLLM("m", "")
    t = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError):
        GroqLLM("m", "k", transport=t, max_retries=0).complete([{"role": "user", "content": "q"}])


def test_make_llm_picks_provider() -> None:
    assert make_llm(Settings(llm_provider="ollama")).model == "qwen3:8b"
    assert make_llm(Settings(llm_provider="groq", groq_api_key="k")).model.startswith("llama")
    with pytest.raises(LLMError):
        make_llm(Settings(llm_provider="nope"))
