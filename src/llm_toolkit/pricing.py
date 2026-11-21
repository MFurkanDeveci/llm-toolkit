"""Token kullanımından dolar cinsinden maliyet hesabı."""

import logging

from llm_toolkit.datatypes import Usage

logger = logging.getLogger(__name__)

# USD / 1 milyon token: (girdi, çıktı)
# Kaynak: https://platform.claude.com/docs/en/about-claude/pricing (21 Kasım 2026'da kontrol edildi)
# Not: Haiku 5.5'te 100.000 tokendan uzun promptlar daha pahalıdır; bu tablo kısa promptlar içindir.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def cost_usd(provider: str, model: str, usage: Usage) -> float:
    if provider == "ollama":
        return 0.0  # Yerel model: token başına ücret yok (elektrik ve donanım hariç)
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        logger.warning("Fiyatı bilinmeyen model: %s. Maliyet 0 gösterilecek.", model)
        return 0.0
    input_price, output_price = prices
    return (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000
