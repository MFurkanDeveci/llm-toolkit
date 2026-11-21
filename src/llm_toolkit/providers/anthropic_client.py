"""Claude API sağlayıcısı."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import anthropic
from anthropic import transform_schema
from pydantic import BaseModel

from llm_toolkit.datatypes import LLMResponse, Message, Tool, ToolCall, Usage
from llm_toolkit.errors import PermanentLLMError, TransientLLMError
from llm_toolkit.pricing import cost_usd
from llm_toolkit.retry import with_retries


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Ortak Message listesini Claude API formatına çevirir."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            if m.provider_blocks:
                # Claude'un kendi ürettiği bloklar (düşünme + araç çağrısı) aynen geri gider
                out.append({"role": "assistant", "content": m.provider_blocks})
                continue
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for call in m.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
            out.append({"role": "assistant", "content": blocks})
        else:  # tool: Claude'da araç sonucu, user rolünde bir tool_result bloğudur
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            previous = out[-1] if out else None
            if previous and previous["role"] == "user" and isinstance(previous["content"], list):
                previous["content"].append(block)  # Aynı turdaki sonuçlar tek mesajda toplanır
            else:
                out.append({"role": "user", "content": [block]})
    return out


def to_anthropic_tool(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters.model_json_schema(),
    }


@contextmanager
def translate_errors() -> Iterator[None]:
    """Anthropic SDK hatalarını bizim iki hata türümüze çevirir."""
    try:
        yield
    except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
        raise TransientLLMError(f"{type(exc).__name__}: {exc}") from exc
    except anthropic.APIConnectionError as exc:  # Zaman aşımı da bunun alt sınıfı
        raise TransientLLMError(f"Bağlantı hatası: {exc}") from exc
    except anthropic.APIError as exc:
        raise PermanentLLMError(f"{type(exc).__name__}: {exc}") from exc


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        effort: str = "low",
        max_tokens: int = 4096,
        timeout_s: float = 60,
        max_attempts: int = 3,
        sdk: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self._sleep = sleep
        # SDK'nın kendi tekrar denemesini kapatıyoruz; tüm sağlayıcılarda aynı mantık bizim olsun
        self._sdk = sdk or anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout_s)

    def complete(self, messages, *, system=None, tools=None, max_tokens=None) -> LLMResponse:
        kwargs = self._request(messages, system, max_tokens)
        if tools:
            kwargs["tools"] = [to_anthropic_tool(t) for t in tools]
        return self._send(kwargs)

    def complete_json(
        self, messages, schema: type[BaseModel], *, system=None, max_tokens=None
    ) -> LLMResponse:
        kwargs = self._request(messages, system, max_tokens)
        # Yapılandırılmış çıktı: model sadece bu şemaya uyan JSON üretebilir.
        # transform_schema, API'nin desteklemediği kuralları (min/max gibi) açıklamaya taşır.
        kwargs["output_config"]["format"] = {
            "type": "json_schema",
            "schema": transform_schema(schema),
        }
        return self._send(kwargs)

    def stream(self, messages, *, system=None, max_tokens=None) -> Iterator[str | LLMResponse]:
        # Streaming'de otomatik tekrar deneme yok: yarıda kesilen akışı baştan almak,
        # kullanıcının ekranda aynı metni iki kez görmesine yol açar.
        kwargs = self._request(messages, system, max_tokens)
        start = time.perf_counter()
        with translate_errors(), self._sdk.messages.stream(**kwargs) as s:
            yield from s.text_stream  # Sadece metin parçaları; düşünme parçaları atlanır
            final = s.get_final_message()
        yield self._to_response(final, time.perf_counter() - start)

    # --- yardımcılar ---

    def _request(self, messages, system, max_tokens) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": to_anthropic_messages(messages),
            "output_config": {"effort": self.effort},
        }
        if system:
            kwargs["system"] = system
        return kwargs

    def _send(self, kwargs: dict[str, Any]) -> LLMResponse:
        def call():
            with translate_errors():
                return self._sdk.messages.create(**kwargs)

        start = time.perf_counter()
        raw = with_retries(call, max_attempts=self.max_attempts, sleep=self._sleep)
        return self._to_response(raw, time.perf_counter() - start)

    def _to_response(self, raw: Any, latency_s: float) -> LLMResponse:
        # Yanıt birden fazla bloktan oluşur: thinking (düşünme), text (cevap), tool_use (araç).
        # Kullanıcıya sadece text bloklarını gösteririz.
        text = "".join(b.text for b in raw.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in raw.content
            if b.type == "tool_use"
        ]
        usage = Usage(input_tokens=raw.usage.input_tokens, output_tokens=raw.usage.output_tokens)
        return LLMResponse(
            text=text,
            tool_calls=calls,
            usage=usage,
            provider=self.provider,
            model=self.model,
            latency_s=round(latency_s, 3),
            cost_usd=cost_usd(self.provider, self.model, usage),
            provider_blocks=[b.model_dump(mode="json", exclude_none=True) for b in raw.content],
        )
