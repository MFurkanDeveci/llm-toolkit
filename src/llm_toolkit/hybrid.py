"""Hibrit arama: kelime tabanlı arama (BM25) ile vektör aramayı birleştirir.

Vektör arama anlamı yakalar ama "39,90 TL" gibi kesin ifadeleri kaçırabilir.
BM25 kesin kelimeleri yakalar ama eş anlamlıları ("geri gönder" / "iade") bilmez.
İkisinin sıralamalarını birleştirince genellikle ikisinden de iyi sonuç çıkar.
"""

import math
import re
from collections import Counter

import numpy as np
from pydantic import BaseModel

from llm_toolkit.chunking import Chunk
from llm_toolkit.embeddings import cosine_similarity
from llm_toolkit.vector_store import VectorStore

# Türkçe eklemeli bir dil: "iade", "iadeler", "iadenizi" aynı kelime.
# Kelimenin ilk 5 harfini kök saymak basit ama Türkçede şaşırtıcı derecede işe yarar.
STEM_LENGTH = 5


def tokenize(text: str) -> list[str]:
    """Metni küçük harfli kelime köklerine ayırır."""
    # Python'un lower() fonksiyonu Türkçe İ/I'yı doğru çevirmez; önce elle çeviriyoruz
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return [word[:STEM_LENGTH] for word in re.findall(r"\w+", text)]


class BM25:
    """Klasik kelime tabanlı sıralama.

    Bir kelime bir parçada ne kadar çok geçerse puan o kadar artar (ama azalarak).
    Az parçada geçen (nadir) kelimeler, her yerde geçen kelimelerden daha değerlidir (idf).
    Uzun parçalar, sırf uzun oldukları için avantajlı olmasın diye cezalandırılır (b).
    """

    def __init__(self, texts: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        doc_freq = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def scores(self, query: str) -> np.ndarray:
        """Her parça için bir puan. 0 = sorgudaki hiçbir kelime geçmiyor."""
        out = np.zeros(len(self.docs))
        for i, (doc, length) in enumerate(zip(self.docs, self.lengths, strict=True)):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            for term in tokenize(query):
                tf = doc.get(term, 0)
                if tf:
                    out[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Birden fazla sıralamayı birleştirir (Reciprocal Rank Fusion).

    Her listede r. sıradaki belge 1 / (k + r) puan alır; puanlar toplanır. Skorların kendisine
    değil sadece sıraya bakıldığı için, ölçekleri farklı yöntemler sorunsuz birleşir.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1 / (k + rank)
    return fused


class HybridHit(BaseModel):
    chunk: Chunk
    score: float  # RRF puanı
    vector_rank: int | None  # Vektör aramasındaki sırası (1 = en iyi)
    bm25_rank: int | None  # BM25'teki sırası; None = hiç kelime eşleşmedi


def hybrid_search(
    store: VectorStore,
    bm25: BM25,
    query_vector: np.ndarray,
    query_text: str,
    *,
    k: int = 3,
    rrf_k: int = 60,
) -> list[HybridHit]:
    """Vektör ve BM25 sıralamalarını RRF ile birleştirip en iyi k parçayı döndürür."""
    vector_scores = cosine_similarity(query_vector.reshape(1, -1), store.vectors)[0]
    vector_ranking = [int(i) for i in np.argsort(-vector_scores)]
    bm25_scores = bm25.scores(query_text)
    bm25_ranking = [int(i) for i in np.argsort(-bm25_scores) if bm25_scores[i] > 0]

    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking], k=rrf_k)
    vector_pos = {doc: r for r, doc in enumerate(vector_ranking, start=1)}
    bm25_pos = {doc: r for r, doc in enumerate(bm25_ranking, start=1)}
    best = sorted(fused, key=lambda doc: fused[doc], reverse=True)[:k]
    return [
        HybridHit(
            chunk=store.chunks[doc],
            score=fused[doc],
            vector_rank=vector_pos.get(doc),
            bm25_rank=bm25_pos.get(doc),
        )
        for doc in best
    ]
