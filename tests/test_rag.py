"""RAG cevap testleri: sahte embedding modeli ve sahte LLM ile, ağa çıkmadan."""

import numpy as np
from conftest import FakeLLM, make_response

from llm_toolkit.chunking import Chunk
from llm_toolkit.embeddings import EmbeddingResult
from llm_toolkit.rag import answer_question, build_prompt, extract_citations
from llm_toolkit.vector_store import SearchHit, VectorStore


class KeywordEmbedder:
    """Sahte embedding: her boyut bir anahtar kelimenin geçip geçmediği."""

    model = "keyword-test"
    WORDS = ["iade", "kargo", "puan"]

    def embed(self, texts: list[str]) -> EmbeddingResult:
        vectors = np.array([[float(w in t.lower()) for w in self.WORDS] for t in texts])
        return EmbeddingResult(vectors=vectors, model=self.model, input_tokens=4, latency_s=0.01)


def make_store() -> VectorStore:
    chunks = [
        Chunk(source="iade.md", index=0, text="İade süresi 14 gündür."),
        Chunk(source="kargo.md", index=0, text="Kargo 250 TL üstü ücretsizdir."),
        Chunk(source="uyelik.md", index=0, text="Her alışverişte puan kazanılır."),
    ]
    return VectorStore(chunks, KeywordEmbedder().embed([c.text for c in chunks]).vectors, "x")


def test_prompt_contains_question_and_labeled_sources():
    hits = [SearchHit(chunk=make_store().chunks[0], score=0.9)]
    prompt = build_prompt("İade kaç gün?", hits)
    assert '<kaynak id="iade.md #0">' in prompt
    assert "İade süresi 14 gündür." in prompt
    assert prompt.rstrip().endswith("Soru: İade kaç gün?")


def test_extract_citations_in_order_without_duplicates():
    text = "Süre 14 gün [iade.md #0]. Kargo ücretsiz [kargo.md#0]. Yine [iade.md #0]."
    assert extract_citations(text) == [("iade.md", 0), ("kargo.md", 0)]


def test_answer_uses_retrieved_chunks_and_reports_costs():
    llm = FakeLLM([make_response("İade süresi 14 gündür [iade.md #0].")])
    result = answer_question("İade kaç gün içinde yapılır?", make_store(), KeywordEmbedder(), llm)
    assert result.hits[0].chunk.source == "iade.md"  # En yakın parça ilk sırada
    assert result.citations == [("iade.md", 0)]
    assert result.unknown_citations == []
    assert "İade süresi 14 gündür." in llm.calls[0][0].content  # Parça modele gönderildi
    assert result.search.input_tokens == 4
    assert result.response.cost_usd == 0.001


def test_citation_to_unretrieved_source_is_flagged():
    llm = FakeLLM([make_response("Kargo ücretsiz [kargo.md #7].")])
    result = answer_question("iade", make_store(), KeywordEmbedder(), llm, k=1)
    assert result.unknown_citations == [("kargo.md", 7)]