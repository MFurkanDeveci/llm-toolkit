"""Embedding: metni, anlamını temsil eden bir sayı vektörüne çevirir (yerel Ollama modeliyle).

Anlamca yakın metinlerin vektörleri de birbirine yakın olur. Arama bunun üzerine kurulur.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import ollama

from llm_toolkit.providers.ollama_client import translate_errors
from llm_toolkit.retry import with_retries

# Çok dilli (Türkçe dahil) model. İlk kullanımdan önce: ollama pull bge-m3
DEFAULT_MODEL = "bge-m3"


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray  # Satır sayısı = metin sayısı, sütun sayısı = vektör boyutu
    model: str
    input_tokens: int
    latency_s: float
    cost_usd: float = 0.0  # Yerel model: token başına ücret yok


class OllamaEmbedder:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str | None = None,
        timeout_s: float = 120,
        max_attempts: int = 3,
        sdk: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._sdk = sdk or ollama.Client(host=host, timeout=timeout_s)

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Metinlerin hepsini tek istekte vektöre çevirir."""
        if not texts:
            raise ValueError("Metin listesi boş.")

        def call():
            with translate_errors():
                return self._sdk.embed(model=self.model, input=texts)

        start = time.perf_counter()
        raw = with_retries(call, max_attempts=self.max_attempts, sleep=self._sleep)
        return EmbeddingResult(
            vectors=np.array(raw.embeddings, dtype=np.float32),
            model=self.model,
            input_tokens=raw.prompt_eval_count or 0,
            latency_s=round(time.perf_counter() - start, 3),
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a'daki her vektörün b'deki her vektörle kosinüs benzerliği.

    Sonuç (len(a), len(b)) boyutunda bir matristir. 1 = aynı yön (anlamca çok yakın),
    0 = ilgisiz, -1 = ters yön. Vektörlerin uzunluğu sonucu etkilemez, sadece yönü önemlidir.
    """
    # Her vektörü uzunluğuna bölerek boyunu 1 yap; 1e-12 sıfır vektörde bölme hatasını önler
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T
