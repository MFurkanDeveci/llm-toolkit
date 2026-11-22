"""Embedding testleri. Gerçek ollama yanıt tipleri kullanılır, ağa çıkılmaz."""

import numpy as np
import ollama
import pytest

from llm_toolkit.embeddings import OllamaEmbedder, cosine_similarity
from llm_toolkit.errors import PermanentLLMError


class FakeOllama:
    """Sırayla verilen sonuçları döndürür; sonuç bir hata ise onu fırlatır."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[dict] = []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def embed_response(vectors: list[list[float]]) -> ollama.EmbedResponse:
    return ollama.EmbedResponse(model="bge-m3", embeddings=vectors, prompt_eval_count=9)


def test_embed_returns_matrix_tokens_and_zero_cost():
    sdk = FakeOllama(embed_response([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    result = OllamaEmbedder(sdk=sdk).embed(["kedi", "köpek"])
    assert result.vectors.shape == (2, 3)
    assert (result.model, result.input_tokens, result.cost_usd) == ("bge-m3", 9, 0.0)
    assert sdk.calls == [{"model": "bge-m3", "input": ["kedi", "köpek"]}]


def test_cosine_similarity_values_and_shape():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([[3.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sim = cosine_similarity(a, b)
    assert sim.shape == (2, 3)
    # Aynı yön = 1, dik = 0, ters yön = -1. Vektörün uzunluğu sonucu değiştirmez.
    np.testing.assert_allclose(sim, [[1, 0, -1], [0, 1, 0]], atol=1e-6)


def test_zero_vector_does_not_give_nan():
    sim = cosine_similarity(np.zeros((1, 3)), np.ones((1, 3)))
    assert not np.isnan(sim).any()


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="boş"):
        OllamaEmbedder(sdk=FakeOllama()).embed([])


def test_missing_model_gives_pull_hint():
    error = ollama.ResponseError('model "bge-m3" not found, try pulling it first', 404)
    with pytest.raises(PermanentLLMError, match="404"):
        OllamaEmbedder(sdk=FakeOllama(error)).embed(["x"])


def test_server_down_gives_clear_message():
    sdk = FakeOllama(ConnectionError("Failed to connect"))
    with pytest.raises(PermanentLLMError, match="ollama serve"):
        OllamaEmbedder(sdk=sdk).embed(["x"])


def test_busy_server_is_retried(no_sleep):
    delays, sleep = no_sleep
    sdk = FakeOllama(ollama.ResponseError("busy", 503), embed_response([[1.0, 0.0]]))
    result = OllamaEmbedder(sdk=sdk, sleep=sleep).embed(["x"])
    assert result.vectors.shape == (1, 2)
    assert len(delays) == 1  # Bir kez bekleyip tekrar denedi
