"""Her sağlayıcının uyması gereken ortak arayüz (sözleşme).

max_tokens verilmezse istemcinin kendi varsayılanı (Settings.max_tokens) kullanılır.
"""

from collections.abc import Iterator
from typing import Protocol

from pydantic import BaseModel

from llm_toolkit.datatypes import LLMResponse, Message, Tool


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Tek seferde cevap üretir. Model araç çağırmak isterse tool_calls dolu gelir."""
        ...

    def complete_json(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Cevabı verilen şemaya uyan bir JSON metni olarak üretir (response.text)."""
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str | LLMResponse]:
        """Metni parça parça (str) verir; en sonda toplam bilgiyle bir LLMResponse verir."""
        ...
