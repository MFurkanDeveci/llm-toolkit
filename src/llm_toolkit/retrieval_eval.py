"""Arama ölçümü: altın soru setiyle her yöntemin doğru parçayı ne kadar iyi bulduğunu ölçer.

Bir parça "ilgili" sayılır, eğer sorunun kanıt ifadelerinden birini birebir içeriyorsa.
İki ölçüt:
- recall@k: kanıtların ne kadarı ilk k sonuçta? (1 = hepsi, 0 = hiçbiri)
- MRR (Mean Reciprocal Rank): ilk ilgili parça kaçıncı sırada? 1. sıra = 1, 2. = 0.5, 3. = 0.33
"""

from collections import defaultdict

import numpy as np
from pydantic import BaseModel

from llm_toolkit.chunking import Chunk
from llm_toolkit.embeddings import cosine_similarity
from llm_toolkit.evalset import EvalQuestion
from llm_toolkit.hybrid import BM25, hybrid_search
from llm_toolkit.vector_store import VectorStore

METHODS = ("vektör", "bm25", "hibrit")


class RetrievalResult(BaseModel):
    question_id: str
    category: str
    method: str
    first_rank: int | None  # İlk ilgili parçanın sırası (1 = en üstte); None = hiç bulunamadı
    recall: float  # İlk k sonuçta bulunan kanıtların oranı

    @property
    def reciprocal_rank(self) -> float:
        return 1 / self.first_rank if self.first_rank else 0.0


def rank_chunks(
    method: str, store: VectorStore, bm25: BM25, query_vector: np.ndarray, query_text: str
) -> list[int]:
    """Parça numaralarını, yöntemin en iyi bulduğundan başlayarak sıralar."""
    if method == "vektör":
        scores = cosine_similarity(query_vector.reshape(1, -1), store.vectors)[0]
        return [int(i) for i in np.argsort(-scores, kind="stable")]
    if method == "bm25":
        scores = bm25.scores(query_text)
        # Hiç kelime eşleşmeyen parçalar BM25 sıralamasına girmez
        return [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0]
    if method == "hibrit":
        position = {(c.source, c.index): i for i, c in enumerate(store.chunks)}
        hits = hybrid_search(store, bm25, query_vector, query_text, k=len(store.chunks))
        return [position[(h.chunk.source, h.chunk.index)] for h in hits]
    raise ValueError(f"Bilinmeyen yöntem: {method}")


def score_ranking(
    ranking: list[int], chunks: list[Chunk], evidence: list[str], k: int
) -> tuple[int | None, float]:
    """Bir sıralama için (ilk ilgili parçanın sırası, recall@k) hesaplar."""
    first_rank = next(
        (
            rank
            for rank, i in enumerate(ranking, start=1)
            if any(e in chunks[i].text for e in evidence)
        ),
        None,
    )
    top_texts = [chunks[i].text for i in ranking[:k]]
    found = sum(any(e in text for text in top_texts) for e in evidence)
    return first_rank, found / len(evidence)


def evaluate_retrieval(
    questions: list[EvalQuestion],
    store: VectorStore,
    bm25: BM25,
    query_vectors: np.ndarray,
    k: int,
) -> list[RetrievalResult]:
    """Her cevaplanabilir soru ve her yöntem için bir sonuç döndürür.

    query_vectors: questions listesiyle aynı sırada, her soru için bir vektör.
    Cevaplanamayan ("yok") sorular atlanır: bulunacak doğru parça yoktur.
    """
    if len(query_vectors) != len(questions):
        raise ValueError("Her soru için bir vektör olmalı (sıra da aynı olmalı).")
    results = []
    for question, vector in zip(questions, query_vectors, strict=True):
        if not question.answerable:
            continue
        for method in METHODS:
            ranking = rank_chunks(method, store, bm25, vector, question.question)
            first_rank, recall = score_ranking(ranking, store.chunks, question.evidence, k)
            results.append(
                RetrievalResult(
                    question_id=question.id,
                    category=question.category,
                    method=method,
                    first_rank=first_rank,
                    recall=recall,
                )
            )
    return results


def summarize(results: list[RetrievalResult]) -> list[dict]:
    """Yöntem ve soru türüne göre ortalamalar. "tümü" satırı tüm soruların ortalamasıdır."""
    groups: dict[tuple[str, str], list[RetrievalResult]] = defaultdict(list)
    for r in results:
        groups[(r.method, "tümü")].append(r)
        groups[(r.method, r.category)].append(r)
    return [
        {
            "yöntem": method,
            "tür": category,
            "soru": len(group),
            "recall": sum(r.recall for r in group) / len(group),
            "mrr": sum(r.reciprocal_rank for r in group) / len(group),
        }
        for (method, category), group in groups.items()
    ]
