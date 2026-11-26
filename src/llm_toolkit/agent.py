"""Agent: aramayı modele araç olarak verir; ne zaman, kaç kez, ne arayacağına model karar verir.

Sabit RAG (rag.py): her soruda tam olarak 1 arama, ardından cevap. Akışı biz belirleriz.
Agent: model soruyu okur, gerekirse arar, sonucu okur, eksik kalırsa başka bir sorguyla
tekrar arar, yeterli bilgi toplayınca cevaplar. Akışı model belirler.

Döngünün kendisi Modül 2'deki run_with_tools; burada arama aracını ve kayıt tutmayı ekliyoruz.
"""

from pydantic import BaseModel, ConfigDict, Field

from llm_toolkit.datatypes import LLMResponse, Message, Tool
from llm_toolkit.providers.base import LLMClient
from llm_toolkit.rag import extract_citations
from llm_toolkit.tools import run_with_tools
from llm_toolkit.vector_store import Embedder, SearchHit, VectorStore

AGENT_SYSTEM = """Sen bir müşteri destek asistanısın. Soruları sadece belgelerde_ara aracıyla \
bulduğun bilgilere dayanarak, kısa ve Türkçe cevapla.
Kurallar:
- Cevaplamadan önce mutlaka ara. Soru birden fazla konu içeriyorsa her konu için ayrı ara.
- Sonuçlar soruyu cevaplamıyorsa farklı kelimelerle tekrar ara.
- Her bilginin hemen ardından kaynağını köşeli parantezle yaz, örnek: [iade.md #0]
- Sadece arama sonuçlarında gördüğün kaynak kimliklerini kullan; kimlik uydurma.
- Aramalarda cevap yoksa "Bu konuda elimdeki belgelerde bilgi yok." de; tahmin etme."""


class SearchArgs(BaseModel):
    query: str = Field(
        description="Aranacak ifade. Belgelerde geçebilecek kelimeleri kullan; kısa ve net olsun."
    )


class SearchStep(BaseModel):
    """Agent'ın yaptığı bir arama: ne aradı, ne buldu, ne kadar sürdü."""

    query: str
    hits: list[SearchHit]
    latency_s: float


class AgentRun(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    answer: str
    searches: list[SearchStep]  # Yapılan aramalar, sırasıyla
    responses: list[LLMResponse]  # Her model çağrısı (maliyet ve süre için)
    messages: list[Message]  # Modelle yapılan konuşmanın tamamı (inceleme için)
    citations: list[tuple[str, int]]

    @property
    def model_calls(self) -> int:
        return len(self.responses)

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.responses)  # Arama yerel ve ücretsiz

    @property
    def latency_s(self) -> float:
        return sum(r.latency_s for r in self.responses) + sum(s.latency_s for s in self.searches)

    @property
    def unknown_citations(self) -> list[tuple[str, int]]:
        """Cevapta geçen ama hiçbir aramada bulunmamış kaynaklar: model bunları uydurmuş."""
        seen = {(h.chunk.source, h.chunk.index) for s in self.searches for h in s.hits}
        return [c for c in self.citations if c not in seen]


def make_search_tool(store: VectorStore, embedder: Embedder, k: int, log: list[SearchStep]) -> Tool:
    """Vektör aramasını modelin çağırabileceği bir araca çevirir. Her aramayı log'a yazar."""

    def search(args: SearchArgs) -> str:
        result = embedder.embed([args.query])
        hits = store.search(result.vectors[0], k=k)
        log.append(SearchStep(query=args.query, hits=hits, latency_s=result.latency_s))
        if not hits:
            return "Sonuç bulunamadı."
        return "\n\n".join(
            f'<kaynak id="{h.chunk.source} #{h.chunk.index}">\n{h.chunk.text}\n</kaynak>'
            for h in hits
        )

    return Tool(
        name="belgelerde_ara",
        description=(
            "Şirketin yardım belgelerinde anlamca arama yapar ve en ilgili parçaları "
            "kaynak kimlikleriyle döndürür. Farklı konular için ayrı ayrı çağrılabilir."
        ),
        parameters=SearchArgs,
        function=search,
    )


def run_agent(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    client: LLMClient,
    *,
    k: int = 3,
    max_steps: int = 5,
) -> AgentRun:
    """Soruyu agent'a sorar. max_steps aşılırsa ToolLoopError fırlatır (run_with_tools'tan)."""
    searches: list[SearchStep] = []
    tool = make_search_tool(store, embedder, k, searches)
    answer, responses, messages = run_with_tools(
        client, question, [tool], system=AGENT_SYSTEM, max_steps=max_steps
    )
    return AgentRun(
        question=question,
        answer=answer,
        searches=searches,
        responses=responses,
        messages=messages,
        citations=extract_citations(answer),
    )
