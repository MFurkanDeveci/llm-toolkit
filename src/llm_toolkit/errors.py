"""Hata sınıfları. Sağlayıcıya özel hatalar bunlara çevrilir."""


class LLMError(Exception):
    """Tüm LLM hatalarının ortak atası."""


class TransientLLMError(LLMError):
    """Geçici hata: biraz bekleyip tekrar denemek işe yarayabilir (rate limit, aşırı yük, ağ)."""


class PermanentLLMError(LLMError):
    """Kalıcı hata: tekrar denemek işe yaramaz (yanlış anahtar, geçersiz istek, olmayan model)."""


class StructuredOutputError(LLMError):
    """Model, denemelere rağmen şemaya uyan çıktı üretemedi."""


class ToolLoopError(LLMError):
    """Tool döngüsü izin verilen adım sayısını aştı."""
