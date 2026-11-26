"""Agent testleri: sahte LLM'in araç çağrılarıyla agent'ın adımlarını kontrol eder."""

import numpy as np
import pytest
from conftest import FakeLLM, make_response

from llm_toolkit.agent import FALLBACK_ANSWER, Limits, run_agent
from llm_toolkit.chunking import Chunk
from llm_toolkit.datatypes import ToolCall
from llm_toolkit.embeddings import EmbeddingResult
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


def search(query: str, call_id: str = "c1"):
    """Sahte modelin 'belgelerde_ara' aracını çağırdığı bir cevap."""
    call = ToolCall(id=call_id, name="belgelerde_ara", arguments={"query": query})
    return make_response(tool_calls=[call])


def test_agent_searches_then_answers_with_citation():
    llm = FakeLLM([search("iade süresi"), make_response("İade süresi 14 gündür [d0.md #0].")])
    run = run_agent("Kaç gün içinde iade edebilirim?", STORE, KeywordEmbedder(), llm, k=1)
    assert [s.query for s in run.searches] == ["iade süresi"]
    assert run.searches[0].hits[0].chunk.source == "d0.md"
    assert run.answer == "İade süresi 14 gündür [d0.md #0]."
    assert run.citations == [("d0.md", 0)]
    assert run.unknown_citations == []
    assert run.model_calls == 2
    assert run.cost_usd == pytest.approx(0.002)
    # Arama sonucu, kimlik etiketiyle birlikte modele geri gönderilmeli
    tool_result = llm.calls[1][-1]
    assert tool_result.role == "tool" and '<kaynak id="d0.md #0">' in tool_result.content


def test_agent_can_search_more_than_once():
    llm = FakeLLM(
        [
            search("iade süresi", "c1"),
            search("kargo ücreti", "c2"),
            make_response("İade 14 gün [d0.md #0], kargo 39,90 TL [d1.md #0]."),
        ]
    )
    run = run_agent("İade süresi ve kargo ücreti?", STORE, KeywordEmbedder(), llm, k=1)
    assert [s.query for s in run.searches] == ["iade süresi", "kargo ücreti"]
    assert run.model_calls == 3
    assert run.unknown_citations == []  # İki kaynak da aramalarda bulunmuştu


def test_answer_without_searching_is_visible():
    llm = FakeLLM([make_response("Bu konuda elimdeki belgelerde bilgi yok.")])
    run = run_agent("Hediye paketi var mı?", STORE, KeywordEmbedder(), llm)
    assert run.searches == []
    assert run.model_calls == 1


def test_citation_not_seen_in_any_search_is_flagged():
    llm = FakeLLM([search("iade"), make_response("Kargo ücretsiz [d1.md #0].")])
    run = run_agent("İade?", STORE, KeywordEmbedder(), llm, k=1)
    assert run.unknown_citations == [("d1.md", 0)]  # Bu parça hiçbir aramada gelmedi


def test_normal_answer_has_stop_reason_cevap():
    llm = FakeLLM([search("iade"), make_response("14 gün [d0.md #0].")])
    run = run_agent("İade?", STORE, KeywordEmbedder(), llm, k=1)
    assert run.stop_reason == "cevap"


def test_step_limit_asks_for_best_answer_instead_of_crashing():
    # Model durmadan arıyor; 2 adımdan sonra durdurulup eldekilerle cevap vermesi istenir
    llm = FakeLLM([search("a", "c1"), search("b", "c2"), make_response("Kısmi cevap.")])
    run = run_agent("?", STORE, KeywordEmbedder(), llm, limits=Limits(max_steps=2))
    assert run.stop_reason == "adım"
    assert run.answer == "Kısmi cevap."
    assert run.model_calls == 3  # 2 araçlı adım + 1 kapanış çağrısı
    wrap_up = llm.calls[2][-1]  # Kapanış çağrısında modele giden son mesaj
    assert wrap_up.role == "user" and "adım sınırına" in wrap_up.content


def test_budget_limit():
    # Her sahte çağrı 0.001 $; 2 çağrıdan sonra harcama 0.002 >= 0.0015 sınırı
    llm = FakeLLM([search("a", "c1"), search("b", "c2"), make_response("Kısmi.")])
    run = run_agent("?", STORE, KeywordEmbedder(), llm, limits=Limits(max_cost_usd=0.0015))
    assert run.stop_reason == "bütçe"
    assert run.model_calls == 3


def test_time_limit():
    # Sahte saat: başlangıç 0 sn, ilk kontrol 1 sn, ikinci kontrol 100 sn (sınır 60)
    times = iter([0.0, 1.0, 100.0])
    llm = FakeLLM([search("a"), make_response("Kısmi.")])
    run = run_agent(
        "?", STORE, KeywordEmbedder(), llm, limits=Limits(max_seconds=60), clock=lambda: next(times)
    )
    assert run.stop_reason == "süre"
    assert run.model_calls == 2


def test_wrap_up_that_still_wants_tools_gets_fallback_answer():
    llm = FakeLLM([search("a", "c1"), search("b", "c2")])  # Kapanışta bile arama istiyor
    run = run_agent("?", STORE, KeywordEmbedder(), llm, limits=Limits(max_steps=1))
    assert run.answer == FALLBACK_ANSWER
    assert len(run.searches) == 1  # Kapanış çağrısındaki arama isteği çalıştırılmadı
