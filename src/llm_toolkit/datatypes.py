"""Sağlayıcıdan bağımsız ortak veri tipleri.

Uygulamanın geri kalanı sadece bu tipleri bilir. Anthropic ve Ollama'nın kendi
formatlarına çeviri yalnızca providers/ klasöründe yapılır.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant", "tool"]


class ToolCall(BaseModel):
    """Modelin 'şu aracı şu argümanlarla çalıştır' isteği."""

    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # sadece assistant mesajlarında
    tool_call_id: str | None = None  # sadece tool mesajlarında: hangi çağrının sonucu
    tool_name: str | None = None  # sadece tool mesajlarında
    # Sağlayıcının ham içerik blokları (örn. Claude'un düşünme blokları). Claude, tool döngüsünde
    # kendi düşünme bloklarının aynen geri gönderilmesini ister; ortak formatta karşılıkları yok.
    provider_blocks: list[dict[str, Any]] | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage
    provider: str
    model: str
    latency_s: float
    cost_usd: float
    provider_blocks: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class Tool:
    """Modelin çağırabileceği bir Python fonksiyonu.

    parameters: Argümanları tanımlayan Pydantic modeli. Hem modele gönderilen
    şemayı üretir hem de modelden gelen argümanları doğrular.
    function: Doğrulanmış argümanları alıp metin sonuç döndürür.
    """

    name: str
    description: str
    parameters: type[BaseModel]
    function: Callable[[Any], str]
