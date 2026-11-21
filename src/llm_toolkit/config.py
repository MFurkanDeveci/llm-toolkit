"""Ayarlar: ortam değişkenlerinden ve .env dosyasından okunur."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    provider: Literal["anthropic", "ollama"] = "ollama"
    anthropic_model: str = "claude-haiku-5-5"
    # Haiku 5.5 cevap vermeden önce düşünür (thinking). effort ne kadar düşüneceğini belirler:
    # düşük effort = daha hızlı ve ucuz, yüksek effort = zor sorularda daha isabetli.
    anthropic_effort: Literal["low", "medium", "high"] = "low"
    max_tokens: int = 4096  # Düşünme tokenları da bu bütçeden yer; çok küçük tutma
    ollama_model: str = "llama3.2:3b"
    ollama_host: str | None = None  # None = http://localhost:11434
    timeout_s: float = 60
    max_attempts: int = 3
    ml_service_url: str = "http://localhost:8000"  # Modül 1'deki tahmin servisi

    # Önek kuralının istisnası: anahtar standart adıyla okunur. SecretStr loglarda görünmez.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
