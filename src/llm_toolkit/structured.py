"""Yapılandırılmış çıktı: metinden şemaya uyan, doğrulanmış Python nesnesi çıkarma."""

import logging

from pydantic import BaseModel, ValidationError

from llm_toolkit.datatypes import LLMResponse, Message
from llm_toolkit.errors import StructuredOutputError
from llm_toolkit.providers.base import LLMClient

logger = logging.getLogger(__name__)


def extract[T: BaseModel](
    client: LLMClient,
    prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
    max_attempts: int = 2,
) -> tuple[T, list[LLMResponse]]:
    """Modelden şemaya uyan JSON ister ve Pydantic ile doğrular.

    Sağlayıcı JSON biçimini garanti etse bile (örneğin 1-5 arası puan gibi) bazı kurallar
    sadece bizim tarafta kontrol edilir. Doğrulama başarısız olursa hatayı modele
    geri gönderip düzeltmesini isteriz. Döndürülen liste maliyet hesabı içindir.
    """
    messages = [Message(role="user", content=prompt)]
    responses: list[LLMResponse] = []
    for attempt in range(1, max_attempts + 1):
        response = client.complete_json(messages, schema, system=system)
        responses.append(response)
        try:
            return schema.model_validate_json(response.text), responses
        except ValidationError as exc:
            logger.warning("Deneme %d: çıktı şemaya uymadı: %s", attempt, exc.errors())
            messages += [
                Message(role="assistant", content=response.text),
                Message(
                    role="user",
                    content=(
                        "Çıktın şu kurallara uymadı:\n"
                        f"{exc}\n"
                        "Hataları düzelterek sadece geçerli JSON döndür."
                    ),
                ),
            ]
    raise StructuredOutputError(f"{max_attempts} denemede geçerli çıktı alınamadı.")
