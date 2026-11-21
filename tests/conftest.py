"""Ortak test yardımcıları: ağa çıkmayan, önceden yazılmış cevaplar veren sahte LLM."""

from collections.abc import Iterator

import pytest

from llm_toolkit.datatypes import LLMResponse, Message, ToolCall, Usage


def make_response(text: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=10, output_tokens=5),
        provider="fake",
        model="fake-model",
        latency_s=0.01,
        cost_usd=0.001,
    )


class FakeLLM:
    """LLMClient arayüzüne uyan sahte istemci.

    Sırayla verilecek cevapları alır ve kendisine gönderilen her mesaj listesini kaydeder.
    Böylece testte hem 'model ne cevap verdi' hem 'modele ne gönderildi' kontrol edilebilir.
    """

    provider = "fake"
    model = "fake-model"

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def _next(self, messages: list[Message]) -> LLMResponse:
        self.calls.append([m.model_copy() for m in messages])
        return self._responses.pop(0)

    def complete(self, messages, *, system=None, tools=None, max_tokens=None):
        return self._next(messages)

    def complete_json(self, messages, schema, *, system=None, max_tokens=None):
        return self._next(messages)

    def stream(self, messages, *, system=None, max_tokens=None) -> Iterator:
        response = self._next(messages)
        yield from response.text.split(" ")
        yield response


@pytest.fixture
def no_sleep():
    """Tekrar deneme testlerinde gerçekten beklememek için beklenen süreleri kaydeder."""
    delays: list[float] = []
    return delays, delays.append
