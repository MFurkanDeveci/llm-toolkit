"""Agent: aramayı modele araç olarak verir; ne zaman, kaç kez, ne arayacağına model karar verir.

Sabit RAG (rag.py): her soruda tam olarak 1 arama, ardından cevap. Akışı biz belirleriz.
Agent: model soruyu okur, gerekirse arar, sonucu okur, eksik kalırsa başka bir sorguyla
tekrar arar, yeterli bilgi toplayınca cevaplar. Akışı model belirler.

Korumalar (Ders 5.2): adım, bütçe ve süre sınırı. Sınıra takılan agent hata vermez; son bir
çağrıyla o ana kadar bulduklarıyla en iyi cevabı vermesi istenir ve neden durduğu kaydedilir.

Hafıza (Ders 5.3): önceki turların sadece soru ve cevapları modele verilir (aramalar değil).
Böylece "peki e-kitaplar için?" gibi devam soruları anlaşılır, token maliyeti de sınırlı kalır.

İnsan onayı (Ders 5.5): arama sadece okur; işlem araçları (actions) ise dünyayı değiştirir
(örn. iade başlatmak). Model bir işlem aracı isteyince agent durur (stop_reason "onay") ve
isteği pending'e koyar. İşlem, ancak kullanıcı onaylarsa resume_agent ile çalıştırılır.
"""

import time
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_toolkit.datatypes import LLMResponse, Message, Tool, ToolCall
from llm_toolkit.providers.base import LLMClient
from llm_toolkit.rag import extract_citations
from llm_toolkit.tools import execute_tool
from llm_toolkit.vector_store import Embedder, SearchHit, VectorStore

AGENT_SYSTEM = """Sen bir müşteri destek asistanısın. Soruları sadece belgelerde_ara aracıyla \
bulduğun bilgilere dayanarak, kısa ve Türkçe cevapla.
Kurallar:
- Cevaplamadan önce mutlaka ara. Soru birden fazla konu içeriyorsa her konu için ayrı ara.
- Sonuçlar soruyu cevaplamıyorsa farklı kelimelerle tekrar ara.
- Her bilginin hemen ardından kaynağını köşeli parantezle yaz, örnek: [iade.md #0]
- Sadece arama sonuçlarında gördüğün kaynak kimliklerini kullan; kimlik uydurma.
- Aramalarda cevap yoksa "Bu konuda elimdeki belgelerde bilgi yok." de; tahmin etme."""

# Sadece işlem araçları verildiğinde sisteme eklenir; ölçümlerdeki agent davranışı değişmez.
ACTION_RULES = """
İşlem araçları (örn. iade_baslat) gerçek bir işlem yapar ve kullanıcı onaylamadan çalışmaz:
- İşlem aracını sadece kullanıcı açıkça istediğinde çağır.
- Gerekli bilgileri (sipariş numarası gibi) kullanıcıdan öğren; asla uydurma. Eksikse sor.
- Kullanıcı reddederse işlemi tekrar deneme; yapılmadığını söyle."""
REJECTED = "Kullanıcı bu işlemi reddetti; işlem yapılmadı. Tekrar deneme."


StopReason = Literal["cevap", "onay", "adım", "bütçe", "süre"]

REASONS = {
    "adım": "adım sınırına",
    "bütçe": "bütçe sınırına",
    "süre": "süre sınırına",
}
WRAP_UP = """Agent {reason} ulaştı; artık arama yapamazsın. Şimdiye kadar bulduğun bilgilerle \
soruyu cevapla. Cevaplayamadığın kısım varsa bunu açıkça söyle; tahmin etme."""
FALLBACK_ANSWER = "Sınıra ulaşıldı ve eldeki bilgilerle bir cevap üretilemedi."


class Limits(BaseModel):
    """Agent'ın harcayabileceği kaynaklar. Her araçlı model çağrısından önce kontrol edilir."""

    max_steps: int = 5  # Araç kullanabilen en fazla model çağrısı
    max_cost_usd: float = 0.05  # Bu tutara ulaşınca yeni arama yapılmaz
    max_seconds: float = 60.0  # Bu süre dolunca yeni arama yapılmaz


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
    # "cevap" = model kendisi bitirdi; "onay" = işlem için kullanıcıyı bekliyor; diğerleri = sınır
    stop_reason: StopReason
    history_citations: list[tuple[str, int]] = []  # Önceki cevaplarda geçen kaynaklar
    pending: list[ToolCall] = []  # Onay bekleyen işlem çağrıları (stop_reason "onay" ise)

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
        """Cevapta geçen ama ne bu turun aramalarında ne de önceki cevaplarda bulunan kaynaklar:
        model bunları uydurmuş demektir."""
        seen = {(h.chunk.source, h.chunk.index) for s in self.searches for h in s.hits}
        seen |= set(self.history_citations)
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


def build_history(runs: list[AgentRun], max_turns: int) -> list[Message]:
    """Önceki turlardan hafıza üretir: son max_turns turun soru ve cevabı (kayan pencere).

    Aramalar ve araç sonuçları hafızaya alınmaz: cevaplar zaten bulunan bilgiyi kaynak
    etiketleriyle taşır. Tüm konuşmayı göndermek her turda token maliyetini katlar.
    """
    recent = runs[-max_turns:] if max_turns > 0 else []
    history: list[Message] = []
    for run in recent:
        history.append(Message(role="user", content=run.question))
        history.append(Message(role="assistant", content=run.answer))
    return history


def check_limits(
    responses: list[LLMResponse], elapsed_s: float, limits: Limits
) -> StopReason | None:
    """Sınırlardan biri aşıldıysa sebebini, aşılmadıysa None döndürür."""
    if len(responses) >= limits.max_steps:
        return "adım"
    if sum(r.cost_usd for r in responses) >= limits.max_cost_usd:
        return "bütçe"
    if elapsed_s >= limits.max_seconds:
        return "süre"
    return None


def needs_approval(call: ToolCall, actions: dict[str, Tool]) -> bool:
    """Bu çağrı insana sorulmalı mı? İşlem aracıysa ve argümanları geçerliyse evet.

    Argümanları geçersiz bir istek insana gösterilmez: execute_tool hatayı modele döndürür,
    model düzeltir ya da eksik bilgiyi kullanıcıya sorar.
    """
    tool = actions.get(call.name)
    if tool is None:
        return False
    try:
        tool.parameters.model_validate(call.arguments)
    except ValidationError:
        return False
    return True


def run_agent(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    client: LLMClient,
    *,
    k: int = 3,
    limits: Limits | None = None,
    history: list[Message] | None = None,
    actions: list[Tool] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AgentRun:
    """Soruyu agent'a sorar. Sınıra takılırsa hata vermez, eldekilerle cevap ister.

    En fazla limits.max_steps araçlı çağrı + 1 kapanış çağrısı yapılır. Bütçe ve süre "yumuşak"
    sınırdır: aşıldıkları an yeni arama başlamaz, ama kapanış çağrısı biraz daha harcar.
    history: önceki turlar (build_history ile); yeni sorudan önce modele gönderilir.
    actions: onay gerektiren işlem araçları. İstenirse agent "onay" ile durur (resume_agent).
    """
    history = history or []
    return _agent_loop(
        question,
        # Yeni liste: çağıranın history listesi değişmesin
        messages=[*history, Message(role="user", content=question)],
        responses=[],
        searches=[],
        history_citations=[
            c for m in history if m.role == "assistant" for c in extract_citations(m.content)
        ],
        store=store,
        embedder=embedder,
        client=client,
        k=k,
        limits=limits or Limits(),
        actions=actions or [],
        clock=clock,
    )


def resume_agent(
    run: AgentRun,
    approved: bool,
    store: VectorStore,
    embedder: Embedder,
    client: LLMClient,
    *,
    k: int = 3,
    limits: Limits | None = None,
    actions: list[Tool] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AgentRun:
    """Onay bekleyen agent'ı kullanıcının kararıyla sürdürür.

    Onaylandıysa bekleyen işlemler çalışır, reddedildiyse çalışmaz ve model bunu öğrenir.
    Adım ve bütçe sınırı önceki çağrılarla birlikte sayılır; süre yeniden başlar
    (kullanıcının düşünme süresi agent'ın süresinden sayılmaz).
    """
    if run.stop_reason != "onay":
        raise ValueError("Bu agent onay beklemiyor.")
    actions = actions or []
    by_name = {t.name: t for t in actions}
    messages = list(run.messages)
    for call in run.pending:
        result = execute_tool(by_name, call) if approved else REJECTED
        messages.append(
            Message(role="tool", content=result, tool_call_id=call.id, tool_name=call.name)
        )
    return _agent_loop(
        run.question,
        messages=messages,
        responses=list(run.responses),
        searches=list(run.searches),
        history_citations=run.history_citations,
        store=store,
        embedder=embedder,
        client=client,
        k=k,
        limits=limits or Limits(),
        actions=actions,
        clock=clock,
    )


def _agent_loop(
    question: str,
    *,
    messages: list[Message],
    responses: list[LLMResponse],
    searches: list[SearchStep],
    history_citations: list[tuple[str, int]],
    store: VectorStore,
    embedder: Embedder,
    client: LLMClient,
    k: int,
    limits: Limits,
    actions: list[Tool],
    clock: Callable[[], float],
) -> AgentRun:
    """Agent döngüsü: run_agent (yeni soru) ve resume_agent (onaydan sonra) bunu kullanır."""
    search_tool = make_search_tool(store, embedder, k, searches)
    action_tools = {t.name: t for t in actions}
    tools = {search_tool.name: search_tool, **action_tools}
    system = AGENT_SYSTEM + ACTION_RULES if actions else AGENT_SYSTEM
    start = clock()

    def finish(
        answer: str, stop_reason: StopReason, pending: list[ToolCall] | None = None
    ) -> AgentRun:
        return AgentRun(
            question=question,
            answer=answer,
            searches=searches,
            responses=responses,
            messages=messages,
            citations=extract_citations(answer),
            stop_reason=stop_reason,
            history_citations=history_citations,
            pending=pending or [],
        )

    def ask_model() -> LLMResponse:
        response = client.complete(messages, system=system, tools=list(tools.values()))
        responses.append(response)
        messages.append(
            Message(
                role="assistant",
                content=response.text,
                tool_calls=response.tool_calls,
                provider_blocks=response.provider_blocks,
            )
        )
        return response

    while (stop := check_limits(responses, clock() - start, limits)) is None:
        response = ask_model()
        if not response.tool_calls:  # Model aramayı bıraktı: cevap hazır
            return finish(response.text, "cevap")
        pending = []
        for call in response.tool_calls:
            if needs_approval(call, action_tools):
                pending.append(call)  # İşlem: çalıştırma, kullanıcıya sor
                continue
            result = execute_tool(tools, call)
            messages.append(
                Message(role="tool", content=result, tool_call_id=call.id, tool_name=call.name)
            )
        if pending:  # Aynı adımdaki aramalar çalıştı; işlemler onay bekliyor
            return finish(response.text, "onay", pending)

    # Sınıra takıldı: son bir çağrıyla, yeni arama yapmadan cevap vermesini iste.
    # Araç tanımı yine gönderilir (geçmişte araç çağrıları var), ama talimat aramayı yasaklar.
    messages.append(Message(role="user", content=WRAP_UP.format(reason=REASONS[stop])))
    response = ask_model()
    if response.tool_calls:  # Talimata rağmen araç istedi: hiçbir aracı çalıştırmıyoruz
        return finish(FALLBACK_ANSWER, stop)
    return finish(response.text, stop)
