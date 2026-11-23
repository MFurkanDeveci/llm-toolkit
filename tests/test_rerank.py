"""Reranking testleri: sahte LLM'in puanlarıyla sıralamanın nasıl değiştiğini kontrol eder."""

import json

import numpy as np
from conftest import FakeLLM, make_response

from llm_toolkit.chunking import Chunk
from llm_toolkit.evalset import EvalQuestion
from llm_toolkit.hybrid import BM25
from llm_toolkit.rerank import evaluate_reranking, rerank
from llm_toolkit.vector_store import VectorStore

TEXTS = ["Kargo 1-2 iş günü sürer.", "İade süresi 14 gündür.", "Puan 1 yıl geçerlidir."]
CHUNKS = [Chunk(source=f"d{i}.md", index=0, text=t) for i, t in enumerate(TEXTS)]


def scores(*pairs: tuple[int, int]) -> str:
    """Sahte modelin JSON cevabı: [(parça id, puan), ...]"""
    return json.dumps({"scores": [{"id": i, "score": s} for i, s in pairs]})


def test_rerank_orders_by_score_and_shows_ids_to_model():
    llm = FakeLLM([make_response(scores((0, 1), (1, 3), (2, 0)))])
    order, responses = rerank("İade kaç gün?", CHUNKS, llm)
    assert order == [1, 0, 2]
    assert len(responses) == 1
    prompt = llm.calls[0][0].content
    assert "İade kaç gün?" in prompt and '<parca id="2">' in prompt


def test_ties_keep_original_order_and_missing_ids_go_last():
    # 2 numaralı parça hiç puanlanmamış; 99 diye var olmayan bir id uydurulmuş
    llm = FakeLLM([make_response(scores((1, 2), (0, 2), (99, 3)))])
    order, _ = rerank("soru", CHUNKS, llm)
    assert order == [0, 1, 2]


def test_empty_candidate_list_makes_no_call():
    llm = FakeLLM([])
    assert rerank("soru", [], llm) == ([], [])
    assert llm.calls == []


def test_reranking_moves_relevant_chunk_up_but_only_within_candidates():
    store = VectorStore(CHUNKS, np.eye(3), model="test")
    question = EvalQuestion(
        id="iade",
        question="geri gönderme",
        category="anlam",
        evidence=["14 gündür"],
        answer_keywords=["14 gün"],
    )
    vector = np.array([[1.0, 0.5, 0.0]])  # Vektör sıralaması: d0, d1, d2 (doğru parça 2.)

    # Aday sayısı 2: doğru parça adaylar içinde, reranker onu 1. sıraya taşıyor
    llm = FakeLLM([make_response(scores((0, 0), (1, 3)))])
    run = evaluate_reranking(
        [question], store, BM25(TEXTS), vector, llm, base_method="vektör", candidates=2, k=1
    )
    assert run.results[0].first_rank == 1
    assert run.results[0].method == "vektör+rerank"
    assert run.cost_usd == 0.001

    # Aday sayısı 1: doğru parça adaylara hiç girmiyor, reranker onu göremez
    llm = FakeLLM([make_response(scores((0, 3)))])
    run = evaluate_reranking(
        [question], store, BM25(TEXTS), vector, llm, base_method="vektör", candidates=1, k=1
    )
    assert run.results[0].first_rank == 2
