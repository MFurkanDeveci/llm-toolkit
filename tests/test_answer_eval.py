"""Cevap ölçümü testleri: sahte cevap modeli ve sahte yargıçla, ağa çıkmadan."""

import numpy as np
from conftest import FakeLLM, make_response

from llm_toolkit.answer_eval import (
    evaluate_answer,
    is_refusal,
    keyword_score,
    summarize_answers,
)
from llm_toolkit.chunking import Chunk
from llm_toolkit.embeddings import EmbeddingResult
from llm_toolkit.evalset import EvalQuestion
from llm_toolkit.vector_store import VectorStore


class KeywordEmbedder:
    model = "keyword-test"
    WORDS = ["iade", "kargo", "puan"]

    def embed(self, texts: list[str]) -> EmbeddingResult:
        vectors = np.array([[float(w in t.lower()) for w in self.WORDS] for t in texts])
        return EmbeddingResult(vectors=vectors, model=self.model, input_tokens=4, latency_s=0.01)


TEXTS = ["İade süresi 14 gündür.", "Kargo ücreti 39,90 TL'dir.", "Her alışverişte puan kazanılır."]
CHUNKS = [Chunk(source=f"d{i}.md", index=0, text=t) for i, t in enumerate(TEXTS)]
STORE = VectorStore(CHUNKS, KeywordEmbedder().embed(TEXTS).vectors, "keyword-test")

IADE = EvalQuestion(
    id="iade",
    question="İade kaç gün?",
    category="olgu",
    evidence=["İade süresi 14 gündür"],
    answer_keywords=["14 gün"],
)
HEDIYE = EvalQuestion(
    id="hediye", question="Hediye paketi var mı?", category="yok", evidence=[], answer_keywords=[]
)
VERDICT_OK = '{"correct": true, "faithful": true, "reason": "Kaynakla uyumlu."}'


def test_keyword_score_handles_alternatives_and_turkish_case():
    assert keyword_score("İADE EDİLEMEZ.", ["iade edilemez"]) == 1.0
    assert keyword_score("Puanın %5'i kadardır.", ["yüzde 5|%5"]) == 1.0
    assert keyword_score("14 gün içinde, ücret yok.", ["14 gün", "39,90|39.90"]) == 0.5


def test_refusal_detection():
    assert is_refusal("Bu konuda elimdeki belgelerde BİLGİ YOK.")
    assert not is_refusal("İade süresi 14 gündür [d0.md #0].")


def test_correct_answer_with_judge():
    answer = FakeLLM([make_response("İade süresi 14 gündür [d0.md #0].")])
    judge = FakeLLM([make_response(VERDICT_OK)])
    result = evaluate_answer(IADE, STORE, KeywordEmbedder(), answer, judge=judge, k=1)
    assert result.keyword_score == 1.0
    assert result.keyword_pass
    assert result.verdict.correct and result.verdict.faithful
    assert result.answer_cost == 0.001 and result.judge_cost == 0.001
    # Yargıç soruyu, beklenen bilgiyi, kaynağı ve cevabı görmeli
    judge_prompt = judge.calls[0][0].content
    for expected in ["İade kaç gün?", "İade süresi 14 gündür", "d0.md #0", "[d0.md #0]."]:
        assert expected in judge_prompt


def test_refusing_an_answerable_question_fails():
    answer = FakeLLM([make_response("Bu konuda elimdeki belgelerde bilgi yok. 14 gün olabilir.")])
    result = evaluate_answer(IADE, STORE, KeywordEmbedder(), answer, k=1)
    assert result.refused
    assert not result.keyword_pass  # İfade geçse bile "bilgi yok" demek başarısızlık


def test_unanswerable_question_passes_only_with_refusal():
    refuse = FakeLLM([make_response("Bu konuda elimdeki belgelerde bilgi yok.")])
    assert evaluate_answer(HEDIYE, STORE, KeywordEmbedder(), refuse, k=1).keyword_pass
    invent = FakeLLM([make_response("Evet, hediye paketi ücretsizdir.")])
    assert not evaluate_answer(HEDIYE, STORE, KeywordEmbedder(), invent, k=1).keyword_pass


def test_without_judge_there_is_no_verdict_or_judge_cost():
    answer = FakeLLM([make_response("14 gün [d0.md #0].")])
    result = evaluate_answer(IADE, STORE, KeywordEmbedder(), answer, k=1)
    assert result.verdict is None
    assert result.judge_cost == 0.0


def test_summarize_answers():
    answers = FakeLLM([make_response("14 gün [d0.md #0]."), make_response("Hediye var.")])
    results = [
        evaluate_answer(IADE, STORE, KeywordEmbedder(), answers, k=1),
        evaluate_answer(HEDIYE, STORE, KeywordEmbedder(), answers, k=1),
    ]
    rows = {r["tür"]: r for r in summarize_answers(results)}
    assert rows["tümü"]["soru"] == 2
    assert rows["tümü"]["ifade_başarısı"] == 0.5
    assert rows["yok"]["ifade_başarısı"] == 0.0
    assert rows["tümü"]["maliyet"] == 0.002
