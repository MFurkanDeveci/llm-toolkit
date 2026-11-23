"""Arama ölçümü testleri: sonucu elle hesaplanabilen küçük bir örnekle."""

import numpy as np
import pytest

from llm_toolkit.chunking import Chunk
from llm_toolkit.evalset import EvalQuestion
from llm_toolkit.hybrid import BM25
from llm_toolkit.retrieval_eval import RetrievalResult, evaluate_retrieval, summarize
from llm_toolkit.vector_store import VectorStore

TEXTS = [
    "İade süresi 14 gündür.",
    "Kargo ücreti 39,90 TL'dir.",
    "Puanlar 1 yıl sonra silinir.",
]
CHUNKS = [Chunk(source=f"d{i}.md", index=0, text=t) for i, t in enumerate(TEXTS)]
STORE = VectorStore(CHUNKS, np.eye(3), model="test")  # Her parça kendi yönünde
BM = BM25(TEXTS)


def q(qid: str, text: str, evidence: list[str], category: str = "olgu") -> EvalQuestion:
    keywords = ["x"] if evidence else []
    return EvalQuestion(
        id=qid, question=text, category=category, evidence=evidence, answer_keywords=keywords
    )


def by(results: list[RetrievalResult], method: str, qid: str) -> RetrievalResult:
    return next(r for r in results if r.method == method and r.question_id == qid)


def test_vector_finds_right_chunk_bm25_misses_paraphrase():
    # Anlam sorusu: vektör doğru yöne bakıyor, ama sorudaki kelimeler parçada geçmiyor
    questions = [q("anlam", "geri gönderme kaç gün", ["14 gündür"], "anlam")]
    results = evaluate_retrieval(questions, STORE, BM, np.array([[1.0, 0.1, 0.0]]), k=1)
    assert by(results, "vektör", "anlam").first_rank == 1
    assert by(results, "vektör", "anlam").recall == 1.0
    assert by(results, "bm25", "anlam").first_rank is None  # Hiç kelime eşleşmedi
    assert by(results, "bm25", "anlam").reciprocal_rank == 0.0


def test_bm25_finds_exact_amount_vector_ranks_it_second():
    questions = [q("tutar", "kargo 39,90 TL", ["39,90 TL"], "kesin")]
    results = evaluate_retrieval(questions, STORE, BM, np.array([[1.0, 0.5, 0.0]]), k=1)
    assert by(results, "bm25", "tutar").first_rank == 1
    assert by(results, "vektör", "tutar").first_rank == 2
    assert by(results, "vektör", "tutar").recall == 0.0  # k=1: ilk sonuçta yok
    assert by(results, "vektör", "tutar").reciprocal_rank == 0.5


def test_multi_evidence_question_gets_partial_recall():
    questions = [q("coklu", "iade ve kargo", ["14 gündür", "1 yıl sonra"], "coklu")]
    results = evaluate_retrieval(questions, STORE, BM, np.array([[1.0, 0.0, 0.2]]), k=1)
    assert by(results, "vektör", "coklu").recall == 0.5  # İki kanıttan biri ilk sonuçta


def test_unanswerable_questions_are_skipped():
    questions = [q("var", "iade", ["14 gündür"]), q("yok", "hediye paketi", [], "yok")]
    results = evaluate_retrieval(questions, STORE, BM, np.zeros((2, 3)), k=3)
    assert {r.question_id for r in results} == {"var"}
    assert {r.method for r in results} == {"vektör", "bm25", "hibrit"}


def test_vectors_must_match_questions():
    with pytest.raises(ValueError, match="vektör"):
        evaluate_retrieval([q("a", "iade", ["14"])], STORE, BM, np.zeros((2, 3)), k=1)


def test_summarize_averages_per_method_and_category():
    results = [
        RetrievalResult(question_id="a", category="olgu", method="bm25", first_rank=1, recall=1),
        RetrievalResult(
            question_id="b", category="anlam", method="bm25", first_rank=None, recall=0
        ),
    ]
    rows = {(r["yöntem"], r["tür"]): r for r in summarize(results)}
    assert rows[("bm25", "tümü")]["recall"] == 0.5
    assert rows[("bm25", "tümü")]["mrr"] == 0.5
    assert rows[("bm25", "anlam")]["recall"] == 0.0
    assert rows[("bm25", "tümü")]["soru"] == 2
