"""RAG: soruya en yakın parçaları bul, modele bağlam olarak ver, kaynak göstererek cevaplat."""

import re

from pydantic import BaseModel, ConfigDict

from llm_toolkit.datatypes import LLMResponse, Message
from llm_toolkit.embeddings import EmbeddingResult
from llm_toolkit.providers.base import LLMClient
from llm_toolkit.vector_store import Embedder, SearchHit, VectorStore

SYSTEM = """Sen bir müşteri destek asistanısın. Soruyu sadece <kaynak> etiketleri içindeki \
bilgilere dayanarak, kısa ve Türkçe cevapla.
Kurallar:
- Her bilginin hemen ardından kaynağını köşeli parantezle yaz, örnek: [iade.md #0]
- Sadece sana verilen kaynak kimliklerini kullan; kimlik uydurma.
- Kaynaklarda cevap yoksa "Bu konuda elimdeki belgelerde bilgi yok." de; tahmin etme."""

# [iade.md #0] ya da [iade.md#0] biçimindeki kaynak etiketleri
CITATION = re.compile(r"\[([\w.\-]+)\s*#(\d+)\]")


class RAGAnswer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # EmbeddingResult numpy içerir

    question: str
    prompt: str  # Modele gönderilen tam metin (arayüzde göstermek için)
    hits: list[SearchHit]
    search: EmbeddingResult  # Sorunun vektöre çevrilmesinin maliyeti ve süresi
    response: LLMResponse  # Cevap üretmenin maliyeti ve süresi
    citations: list[tuple[str, int]]  # Cevapta geçen kaynaklar

    @property
    def unknown_citations(self) -> list[tuple[str, int]]:
        """Cevapta geçen ama modele verilmemiş kaynaklar: model bunları uydurmuş demektir."""
        given = {(h.chunk.source, h.chunk.index) for h in self.hits}
        return [c for c in self.citations if c not in given]


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    """Bulunan parçaları kimlikleriyle etiketleyip sorunun önüne koyar."""
    sources = "\n\n".join(
        f'<kaynak id="{h.chunk.source} #{h.chunk.index}">\n{h.chunk.text}\n</kaynak>' for h in hits
    )
    return f"{sources}\n\nSoru: {question}"


def extract_citations(text: str) -> list[tuple[str, int]]:
    """Cevaptaki kaynak etiketlerini ilk geçtikleri sırayla, tekrarsız döndürür."""
    found = [(source, int(index)) for source, index in CITATION.findall(text)]
    return list(dict.fromkeys(found))  # dict, sırayı koruyarak tekrarları atar


def answer_question(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    client: LLMClient,
    *,
    k: int = 3,
) -> RAGAnswer:
    """RAG'ın üç adımı: ara (retrieve), bağlamı ekle (augment), cevap üret (generate)."""
    search = embedder.embed([question])
    hits = store.search(search.vectors[0], k=k)
    prompt = build_prompt(question, hits)
    response = client.complete([Message(role="user", content=prompt)], system=SYSTEM)
    return RAGAnswer(
        question=question,
        prompt=prompt,
        hits=hits,
        search=search,
        response=response,
        citations=extract_citations(response.text),
    )