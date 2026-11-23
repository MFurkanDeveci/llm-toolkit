"""Arama kalitesi testi: CI'da her PR'da çalışır, kalite eşiğin altına düşerse kırmızı yanar.

CI'da Ollama (embedding modeli) yok; bu yüzden sadece BM25 ölçülür. BM25 deterministiktir:
aynı belge ve sorularla her zaman aynı sonucu verir, bu yüzden test kararsız (flaky) olmaz.

Ne yakalar? Belgeleri, soru setini, parçalamayı ya da BM25/Türkçe kelime ayırmayı bozan
değişiklikleri. Ne yakalamaz? Embedding modelinin ve LLM cevaplarının kalitesini; onlar
arayüzdeki Değerlendirme ve Cevap Ölçümü sayfalarında ölçülür.
"""

from pathlib import Path

import numpy as np

from llm_toolkit.chunking import chunk_text
from llm_toolkit.evalset import load_documents, load_questions
from llm_toolkit.hybrid import BM25
from llm_toolkit.retrieval_eval import RetrievalResult, evaluate_retrieval, summarize
from llm_toolkit.vector_store import VectorStore

ROOT = Path(__file__).parents[1]
MAX_CHARS, OVERLAP = 400, 80  # Arayüzdeki varsayılan parçalama ayarları
K = 3

# Eşikler bugünkü değerin biraz altında: küçük oynamalara izin verir, ciddi düşüşü yakalar.
# Bugün (Ders 4.6): recall@3 = 1.00, MRR = 0.97
MIN_RECALL = 0.95
MIN_MRR = 0.90


def bm25_results() -> list[RetrievalResult]:
    questions = [q for q in load_questions(ROOT / "data/eval/questions.jsonl") if q.answerable]
    documents = load_documents(ROOT / "data/docs")
    chunks = [
        chunk
        for source, text in documents.items()
        for chunk in chunk_text(text, source, max_chars=MAX_CHARS, overlap_chars=OVERLAP)
    ]
    # CI'da embedding yok. Sıfır vektörler sadece fonksiyonun istediği biçimi sağlamak için;
    # vektör ve hibrit sonuçlarını kullanmıyoruz, yalnızca BM25 satırlarını alıyoruz.
    store = VectorStore(chunks, np.zeros((len(chunks), 1)), model="yok")
    bm25 = BM25([c.text for c in chunks])
    results = evaluate_retrieval(questions, store, bm25, np.zeros((len(questions), 1)), k=K)
    return [r for r in results if r.method == "bm25"]


def test_bm25_retrieval_quality_does_not_regress():
    results = bm25_results()
    overall = next(row for row in summarize(results) if row["tür"] == "tümü")
    missed = [f"{r.question_id} (ilk ilgili sıra: {r.first_rank})" for r in results if r.recall < 1]
    # Test kırmızı yanarsa bu mesaj CI çıktısında görünür: neyin bozulduğunu hemen söyler
    detail = (
        f"recall@{K}={overall['recall']:.2f} (eşik {MIN_RECALL}), "
        f"MRR={overall['mrr']:.2f} (eşik {MIN_MRR}). "
        f"İlk {K} sonuçta tam bulunamayan sorular: {missed or 'yok'}"
    )
    assert overall["recall"] >= MIN_RECALL, detail
    assert overall["mrr"] >= MIN_MRR, detail
