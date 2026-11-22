"""Vektör deposu testleri. Embedding için sahte bir model kullanılır, ağa çıkılmaz."""

import numpy as np
import pytest

from llm_toolkit.chunking import Chunk
from llm_toolkit.embeddings import EmbeddingResult
from llm_toolkit.vector_store import VectorStore, build_index


class KeywordEmbedder:
    """Sahte embedding modeli: her boyut bir anahtar kelimenin metinde geçip geçmediğidir."""

    model = "keyword-test"
    WORDS = ["iade", "kargo", "puan"]

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> EmbeddingResult:
        self.calls.append(texts)
        vectors = np.array([[float(w in t.lower()) for w in self.WORDS] for t in texts])
        return EmbeddingResult(vectors=vectors, model=self.model, input_tokens=7, latency_s=0.01)


def make_store() -> VectorStore:
    chunks = [
        Chunk(source="iade.md", index=0, text="İade 14 gün içinde yapılır."),
        Chunk(source="kargo.md", index=0, text="Kargo 250 TL üstü ücretsiz."),
        Chunk(source="uyelik.md", index=0, text="Her alışverişte puan kazanılır."),
    ]
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return VectorStore(chunks, vectors, model="keyword-test")


def test_search_returns_best_match_first():
    hits = make_store().search(np.array([0.1, 0.9, 0.0]), k=2)
    assert [h.chunk.source for h in hits] == ["kargo.md", "iade.md"]
    assert hits[0].score > hits[1].score


def test_k_larger_than_store_returns_all():
    assert len(make_store().search(np.array([1.0, 0.0, 0.0]), k=10)) == 3


def test_vectors_and_chunks_must_match():
    with pytest.raises(ValueError, match="eşit"):
        VectorStore(make_store().chunks, np.zeros((2, 3)), model="x")


def test_save_and_load_roundtrip(tmp_path):
    store = make_store()
    store.save(tmp_path / "index")
    loaded = VectorStore.load(tmp_path / "index")
    assert loaded.model == "keyword-test"
    assert loaded.chunks == store.chunks
    np.testing.assert_allclose(loaded.vectors, store.vectors)


def test_build_index_chunks_all_documents_in_one_call():
    documents = {"iade.md": "İade kolaydır. İade ücretsizdir.", "kargo.md": "Kargo hızlıdır."}
    embedder = KeywordEmbedder()
    store, result = build_index(documents, embedder, max_chars=200, overlap_chars=0)
    assert [c.source for c in store.chunks] == ["iade.md", "kargo.md"]
    assert len(embedder.calls) == 1  # Tüm parçalar tek istekte gönderildi
    assert store.model == "keyword-test"
    assert result.input_tokens == 7


def test_build_index_rejects_empty_documents():
    with pytest.raises(ValueError, match="parça"):
        build_index({"bos.md": "   "}, KeywordEmbedder())
