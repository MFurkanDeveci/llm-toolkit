"""Ayarlara göre doğru sağlayıcıyı kuran fabrika fonksiyonu."""

from llm_toolkit.config import Settings
from llm_toolkit.errors import PermanentLLMError
from llm_toolkit.providers.anthropic_client import AnthropicClient
from llm_toolkit.providers.base import LLMClient
from llm_toolkit.providers.ollama_client import OllamaClient

__all__ = ["AnthropicClient", "LLMClient", "OllamaClient", "create_client"]


def create_client(settings: Settings) -> LLMClient:
    if settings.provider == "anthropic":
        key = settings.anthropic_api_key
        if key is None:
            # Hızlı başarısız ol: eksik ayarı ilk istekte değil, açılışta yakala
            raise PermanentLLMError(
                "ANTHROPIC_API_KEY tanımlı değil. "
                ".env dosyasına ekle ya da --provider ollama kullan."
            )
        return AnthropicClient(
            settings.anthropic_model,
            api_key=key.get_secret_value(),
            effort=settings.anthropic_effort,
            max_tokens=settings.max_tokens,
            timeout_s=settings.timeout_s,
            max_attempts=settings.max_attempts,
        )
    return OllamaClient(
        settings.ollama_model,
        host=settings.ollama_host,
        max_tokens=settings.max_tokens,
        timeout_s=settings.timeout_s,
        max_attempts=settings.max_attempts,
    )
