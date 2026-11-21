"""Tool calling döngüsü: model araç ister, biz çalıştırırız, sonucu modele geri veririz."""

import logging

from pydantic import ValidationError

from llm_toolkit.datatypes import LLMResponse, Message, Tool, ToolCall
from llm_toolkit.errors import ToolLoopError
from llm_toolkit.providers.base import LLMClient

logger = logging.getLogger(__name__)


def execute_tool(tools: dict[str, Tool], call: ToolCall) -> str:
    """Tek bir araç çağrısını çalıştırır. Hata olursa programı çökertmez, hatayı metin
    olarak döndürür: model hatayı görüp düzeltebilir ya da kullanıcıya açıklayabilir."""
    tool = tools.get(call.name)
    if tool is None:
        return f"HATA: '{call.name}' adında bir araç yok. Mevcut araçlar: {', '.join(tools)}"
    try:
        args = tool.parameters.model_validate(call.arguments)
    except ValidationError as exc:
        return f"HATA: Argümanlar geçersiz: {exc}"
    try:
        return tool.function(args)
    except Exception as exc:  # Araç bir dış sisteme bağlı olabilir; her hata modele iletilir
        logger.warning("Araç %s hata verdi: %s", call.name, exc)
        return f"HATA: Araç çalışırken hata oluştu ({type(exc).__name__}): {exc}"


def run_with_tools(
    client: LLMClient,
    prompt: str,
    tools: list[Tool],
    *,
    system: str | None = None,
    max_steps: int = 5,
) -> tuple[str, list[LLMResponse], list[Message]]:
    """Model araç istemeyi bırakana kadar döngüyü çalıştırır.

    Döndürür: son cevap metni, tüm model yanıtları (maliyet için), tüm konuşma (inceleme için).
    """
    by_name = {t.name: t for t in tools}
    messages = [Message(role="user", content=prompt)]
    responses: list[LLMResponse] = []

    for step in range(1, max_steps + 1):
        response = client.complete(messages, system=system, tools=tools)
        responses.append(response)
        messages.append(
            Message(
                role="assistant",
                content=response.text,
                tool_calls=response.tool_calls,
                provider_blocks=response.provider_blocks,
            )
        )
        if not response.tool_calls:  # Araç istemiyorsa cevap hazır demektir
            return response.text, responses, messages

        for call in response.tool_calls:
            logger.info("Adım %d: %s(%s)", step, call.name, call.arguments)
            result = execute_tool(by_name, call)
            messages.append(
                Message(role="tool", content=result, tool_call_id=call.id, tool_name=call.name)
            )

    # Sınırsız döngü hem para yakar hem de takılı kalmış bir modelin işaretidir
    raise ToolLoopError(f"{max_steps} adımda cevap üretilemedi.")
