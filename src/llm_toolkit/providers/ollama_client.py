"""Ollama (yerel model) sağlayıcısı."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import ollama
from pydantic import BaseModel

from llm_toolkit.datatypes import LLMResponse, Message, Tool, ToolCall, Usage
from llm_toolkit.errors import PermanentLLMError, TransientLLMError
from llm_toolkit.pricing import cost_usd
from llm_toolkit.retry import with_retries


def to_ollama_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    """Ortak Message listesini Ollama formatına çevirir. Sistem mesajı listenin başına girer."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": m.content}
            if m.tool_calls:
                item["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
                ]
            out.append(item)
        else:  # tool: Ollama'da araç sonucu kendi 'tool' rolüyle gönderilir
            out.append({"role": "tool", "content": m.content, "tool_name": m.tool_name})
    return out


def to_ollama_tool(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_json_schema(),
        },
    }


@contextmanager
def translate_errors() -> Iterator[None]:
    try:
        yield
    except ollama.ResponseError as exc:
        if exc.status_code == 429 or exc.status_code >= 500:
            raise TransientLLMError(f"Ollama {exc.status_code}: {exc.error}") from exc
        raise PermanentLLMError(f"Ollama {exc.status_code}: {exc.error}") from exc
    except ConnectionError as exc:
        # Yerel sunucu kapalıysa tekrar denemek işe yaramaz; kullanıcıya ne yapacağını söyle
        raise PermanentLLMError(
            "Ollama'ya bağlanılamadı. Ollama uygulaması açık mı? ('ollama serve')"
        ) from exc


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        max_tokens: int = 4096,
        timeout_s: float = 120,
        max_attempts: int = 3,
        sdk: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._sdk = sdk or ollama.Client(host=host, timeout=timeout_s)

    def complete(self, messages, *, system=None, tools=None, max_tokens=None) -> LLMResponse:
        kwargs = self._request(messages, system, max_tokens)
        if tools:
            kwargs["tools"] = [to_ollama_tool(t) for t in tools]
        return self._send(kwargs)

    def complete_json(
        self, messages, schema: type[BaseModel], *, system=None, max_tokens=None
    ) -> LLMResponse:
        kwargs = self._request(messages, system, max_tokens)
        kwargs["format"] = schema.model_json_schema()  # Ollama'nın yerleşik şema kısıtlaması
        return self._send(kwargs)

    def stream(self, messages, *, system=None, max_tokens=None) -> Iterator[str | LLMResponse]:
        kwargs = self._request(messages, system, max_tokens)
        start = time.perf_counter()
        parts: list[str] = []
        last = None
        with translate_errors():
            for chunk in self._sdk.chat(**kwargs, stream=True):
                piece = chunk.message.content or ""
                if piece:
                    parts.append(piece)
                    yield piece
                last = chunk  # Token sayıları sadece son parçada gelir
        usage = Usage(
            input_tokens=(last.prompt_eval_count or 0) if last else 0,
            output_tokens=(last.eval_count or 0) if last else 0,
        )
        yield self._build("".join(parts), [], usage, time.perf_counter() - start)

    # --- yardımcılar ---

    def _request(self, messages, system, max_tokens) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": to_ollama_messages(messages, system),
            # num_predict, Ollama'da max_tokens'ın karşılığıdır
            "options": {"num_predict": max_tokens or self.max_tokens},
        }

    def _send(self, kwargs: dict[str, Any]) -> LLMResponse:
        def call():
            with translate_errors():
                return self._sdk.chat(**kwargs)

        start = time.perf_counter()
        raw = with_retries(call, max_attempts=self.max_attempts, sleep=self._sleep)
        calls = [
            # Ollama araç çağrılarına kimlik vermez; eşleştirme için biz üretiyoruz
            ToolCall(id=f"call_{i}", name=c.function.name, arguments=dict(c.function.arguments))
            for i, c in enumerate(raw.message.tool_calls or [])
        ]
        usage = Usage(input_tokens=raw.prompt_eval_count or 0, output_tokens=raw.eval_count or 0)
        return self._build(raw.message.content or "", calls, usage, time.perf_counter() - start)

    def _build(self, text, calls, usage, latency_s) -> LLMResponse:
        return LLMResponse(
            text=text,
            tool_calls=calls,
            usage=usage,
            provider=self.provider,
            model=self.model,
            latency_s=round(latency_s, 3),
            cost_usd=cost_usd(self.provider, self.model, usage),
        )
