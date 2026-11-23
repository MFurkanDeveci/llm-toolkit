"""Hibrit arama testleri: BM25, sıralama birleştirme (RRF) ve Türkçe kelime ayırma."""

import numpy as np

from llm_toolkit.chunking import Chunk
from llm_toolkit.hybrid import BM25, hybrid_search, reciprocal_rank_fusion, tokenize
from llm_toolkit.vector_store import VectorStore

TEXTS = [
    "İade süresi teslimattan itibaren 14 gündür.",
    "Kargo ücreti 39,90 TL'dir; 250 TL üstü ücretsizdir.",
    "Puanlar teslimattan 14 gün sonra hesaba geçer.",
]


def test_tokenize_handles_turkish_capitals_and_suffixes():
    # Python'da "İ".lower() görünmez bir nokta ekler; tokenize bunu düzeltmeli
    assert tokenize("İADE") == tokenize("iade") == ["iade"]
    assert tokenize("IĞDIR") == ["ığdır"]
    # İlk 5 harf kök kabul edilir: ekler farklı olsa da aynı kelime eşleşir
    assert tokenize("teslimattan teslimatı") == ["tesli", "tesli"]


def test_bm25_prefers_documents_with_rare_query_words():
    scores = BM25(TEXTS).scores("39,90 TL kargo")
    assert int(np.argmax(scores)) == 1


def test_bm25_gives_zero_when_no_word_matches():
    assert BM25(TEXTS).scores("şifre yenileme").tolist() == [0.0, 0.0, 0.0]


def test_rrf_rewards_documents_ranked_high_in_both_lists():
    fused = reciprocal_rank_fusion([[0, 1], [2, 1]], k=60)
    # 1 numaralı belge iki listede de 2. sırada; sadece tek listede 1. olanları geçer
    assert max(fused, key=fused.get) == 1
    assert fused[0] == 1 / 61  # Sadece ilk listede, 1. sırada


def test_hybrid_search_reports_rank_in_each_method():
    chunks = [Chunk(source=f"d{i}.md", index=0, text=t) for i, t in enumerate(TEXTS)]
    vectors = np.eye(3)  # Her belge kendi yönünde
    store = VectorStore(chunks, vectors, model="test")
    # Vektör araması 2. belgeyi, kelime araması 1. belgeyi en üste koyuyor
    hits = hybrid_search(store, BM25(TEXTS), np.array([0.0, 0.2, 1.0]), "39,90 TL", k=3)
    by_source = {h.chunk.source: h for h in hits}
    assert by_source["d1.md"].bm25_rank == 1
    assert by_source["d2.md"].vector_rank == 1
    assert by_source["d0.md"].bm25_rank is None  # Kelime eşleşmesi yok
    assert hits[0].score >= hits[1].score >= hits[2].score
