"""Vektör deposu: parçaları ve vektörlerini saklar, bir soruya en yakın parçaları bulur.

Bu sürüm bilerek basit: tüm vektörler bellekte bir numpy matrisinde durur ve her aramada
hepsiyle karşılaştırılır. Birkaç on bin parçaya kadar bu yeterince hızlıdır.
"""

import json
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from llm_toolkit.chunking import Chunk, chunk_text
from llm_toolkit.embeddings import EmbeddingResult, cosine_similarity


class Embedder(Protocol):
    """Metinleri vektöre çeviren her şey (gerçek Ollama modeli ya da testteki sahte model)."""

    model: str

    def embed(self, texts: list[str]) -> EmbeddingResult: ...


class SearchHit(BaseModel):
    chunk: Chunk
    score: float  # Kosinüs benzerliği: 1'e ne kadar yakınsa soruya o kadar yakın


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, model: str):
        if len(chunks) != len(vectors):
            raise ValueError("Parça sayısı ile vektör sayısı eşit olmalı.")
        self.chunks = chunks
        self.vectors = vectors
        self.model = model  # Soru da aynı modelle vektöre çevrilmeli, yoksa karşılaştırma anlamsız

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[SearchHit]:
        """Soru vektörüne en benzer k parçayı, en benzerden başlayarak döndürür."""
        scores = cosine_similarity(query_vector.reshape(1, -1), self.vectors)[0]
        best = np.argsort(-scores)[:k]  # Eksi işareti: büyükten küçüğe sırala
        return [SearchHit(chunk=self.chunks[i], score=float(scores[i])) for i in best]

    def save(self, folder: Path) -> None:
        """Vektörleri .npy, parçaları okunabilir .json olarak kaydeder."""
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        data = {"model": self.model, "chunks": [c.model_dump() for c in self.chunks]}
        text = json.dumps(data, ensure_ascii=False, indent=2)
        (folder / "chunks.json").write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "VectorStore":
        data = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk.model_validate(c) for c in data["chunks"]]
        return cls(chunks, np.load(folder / "vectors.npy"), model=data["model"])


def build_index(
    documents: dict[str, str],
    embedder: Embedder,
    *,
    max_chars: int = 400,
    overlap_chars: int = 80,
) -> tuple[VectorStore, EmbeddingResult]:
    """Dokümanları parçalar, tüm parçaları tek istekte vektöre çevirir ve depo kurar.

    documents: {kaynak adı: metin}. EmbeddingResult, maliyet ve süreyi göstermek için döner.
    """
    chunks = [
        chunk
        for source, text in documents.items()
        for chunk in chunk_text(text, source, max_chars=max_chars, overlap_chars=overlap_chars)
    ]
    if not chunks:
        raise ValueError("Dokümanlardan hiç parça çıkmadı; metinler boş olabilir.")
    result = embedder.embed([c.text for c in chunks])
    return VectorStore(chunks, result.vectors, model=embedder.model), result
