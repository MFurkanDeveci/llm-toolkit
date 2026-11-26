"""Cevap ölçümü: RAG cevaplarını altın soru setine göre puanlar.

İki bağımsız görüş:
1. Anahtar ifade kontrolü (ücretsiz, anında): doğru cevapta geçmesi gereken ifadeler var mı?
   Cevaplanamayan ("yok") sorularda ise doğru davranış "bilgi yok" demektir.
2. LLM yargıç (ücretli): başka bir model cevabı okuyup karar verir. Cevap doğru mu? Cevaptaki
   her bilgi verilen kaynaklarda geçiyor mu (sadakat)? İfade kontrolünün yakalayamadığı
   durumları (doğru ama farklı kelimeyle yazılmış cevap, uydurulmuş ek bilgi) yakalar.

Aynı ölçüm hem sabit RAG'a (evaluate_answer) hem agent'a (evaluate_agent_answer, Ders 5.4)
uygulanır: soru, ifade kontrolü ve yargıç aynı; sadece cevabı üreten yöntem değişir.
Agent'ta yargıç, agent'ın tüm aramalarında bulduğu parçaları kaynak olarak görür.
"""

from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from llm_toolkit.agent import Limits, run_agent
from llm_toolkit.evalset import EvalQuestion
from llm_toolkit.providers.base import LLMClient
from llm_toolkit.rag import answer_question
from llm_toolkit.structured import extract
from llm_toolkit.vector_store import Embedder, SearchHit, VectorStore

REFUSAL = "bilgi yok"  # rag.SYSTEM, cevap yoksa "...belgelerde bilgi yok." demesini istiyor

JUDGE_SYSTEM = """Sen bir müşteri destek asistanının cevaplarını değerlendiren titiz bir \
denetçisin. Sana soru, beklenen bilgi, asistana verilen kaynaklar ve asistanın cevabı verilecek.
- correct: Cevap soruyu beklenen bilgiyle uyumlu şekilde cevaplıyor mu? Beklenen bilgi \
"cevap yok" ise, doğru cevap bilgi olmadığını söylemektir.
- faithful: Cevaptaki her bilgi verilen kaynaklarda geçiyor mu? Kaynakta olmayan tek bir \
ek bilgi bile varsa false.
Kaynak etiketlerinin biçimini değerlendirme; sadece içeriğe bak."""


def normalize(text: str) -> str:
    """Türkçe büyük/küçük harf farkını ve fazla boşlukları yok eder."""
    return " ".join(text.replace("İ", "i").replace("I", "ı").lower().split())


def keyword_score(answer: str, keywords: list[str]) -> float:
    """Beklenen ifadelerin cevapta geçme oranı. "a|b": a ya da b yeterli."""
    if not keywords:
        return 1.0
    text = normalize(answer)
    found = sum(any(normalize(alt) in text for alt in kw.split("|")) for kw in keywords)
    return found / len(keywords)


def is_refusal(answer: str) -> bool:
    return REFUSAL in normalize(answer)


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct: bool = Field(description="Cevap, beklenen bilgiyle uyumlu ve soruyu cevaplıyor mu?")
    faithful: bool = Field(description="Cevaptaki her bilgi verilen kaynaklarda geçiyor mu?")
    reason: str = Field(max_length=300, description="Kararın tek cümlelik Türkçe gerekçesi")


class AnswerResult(BaseModel):
    question_id: str
    category: str
    question: str
    answer: str
    keyword_score: float
    refused: bool
    verdict: JudgeVerdict | None  # Yargıç kullanılmadıysa None
    answer_cost: float
    judge_cost: float
    latency_s: float  # Arama + cevap süresi (yargıç hariç)
    # Ders 5.4: varsayılanlar sabit RAG'ın değerleri; eski kayıtlar bu sayede okunmaya devam eder
    searches: int = 1  # Kaç arama yapıldı
    model_calls: int = 1  # Kaç cevap modeli çağrısı yapıldı
    stop_reason: str = "cevap"  # Agent sınıra takıldıysa "adım", "bütçe" ya da "süre"

    @property
    def keyword_pass(self) -> bool:
        """Cevaplanamayan soruda "bilgi yok" demek, diğerlerinde tüm ifadeleri içermek."""
        if self.category == "yok":
            return self.refused
        return self.keyword_score == 1.0 and not self.refused


def build_judge_prompt(question: EvalQuestion, hits: list[SearchHit], answer: str) -> str:
    if question.answerable:
        expected = "\n".join(f"- {e}" for e in question.evidence)
    else:
        expected = "Belgelerde bu sorunun cevabı yok."
    sources = "\n\n".join(
        f'<kaynak id="{h.chunk.source} #{h.chunk.index}">\n{h.chunk.text}\n</kaynak>' for h in hits
    )
    return (
        f"<soru>{question.question}</soru>\n\n"
        f"<beklenen_bilgi>\n{expected}\n</beklenen_bilgi>\n\n"
        f"<kaynaklar>\n{sources}\n</kaynaklar>\n\n"
        f"<cevap>\n{answer}\n</cevap>"
    )


def score_answer(
    question: EvalQuestion,
    answer: str,
    hits: list[SearchHit],
    judge: LLMClient | None,
    *,
    answer_cost: float,
    latency_s: float,
    searches: int = 1,
    model_calls: int = 1,
    stop_reason: str = "cevap",
) -> AnswerResult:
    """Cevabı kim üretmiş olursa olsun aynı şekilde puanlar: ifade kontrolü + isteğe bağlı yargıç.

    hits: cevabı üreten modelin gördüğü parçalar; yargıç sadakati bunlara göre değerlendirir.
    """
    verdict, judge_cost = None, 0.0
    if judge is not None:
        prompt = build_judge_prompt(question, hits, answer)
        verdict, responses = extract(judge, prompt, JudgeVerdict, system=JUDGE_SYSTEM)
        judge_cost = sum(r.cost_usd for r in responses)
    return AnswerResult(
        question_id=question.id,
        category=question.category,
        question=question.question,
        answer=answer,
        keyword_score=keyword_score(answer, question.answer_keywords),
        refused=is_refusal(answer),
        verdict=verdict,
        answer_cost=answer_cost,
        judge_cost=judge_cost,
        latency_s=latency_s,
        searches=searches,
        model_calls=model_calls,
        stop_reason=stop_reason,
    )


def evaluate_answer(
    question: EvalQuestion,
    store: VectorStore,
    embedder: Embedder,
    client: LLMClient,
    *,
    judge: LLMClient | None = None,
    k: int = 3,
) -> AnswerResult:
    """Bir soruyu sabit RAG ile cevaplatır (1 arama + 1 cevap) ve puanlar."""
    rag = answer_question(question.question, store, embedder, client, k=k)
    return score_answer(
        question,
        rag.response.text,
        rag.hits,
        judge,
        answer_cost=rag.search.cost_usd + rag.response.cost_usd,
        latency_s=rag.search.latency_s + rag.response.latency_s,
    )


def evaluate_agent_answer(
    question: EvalQuestion,
    store: VectorStore,
    embedder: Embedder,
    client: LLMClient,
    *,
    judge: LLMClient | None = None,
    k: int = 3,
    limits: Limits | None = None,
) -> AnswerResult:
    """Bir soruyu agent'a cevaplatır ve sabit RAG'la aynı şekilde puanlar."""
    run = run_agent(question.question, store, embedder, client, k=k, limits=limits)
    # Yargıç, agent'ın gördüğü tüm parçaları görmeli; aynı parça iki aramada gelmişse bir kez
    unique: dict[tuple[str, int], SearchHit] = {}
    for step in run.searches:
        for hit in step.hits:
            unique.setdefault((hit.chunk.source, hit.chunk.index), hit)
    return score_answer(
        question,
        run.answer,
        list(unique.values()),
        judge,
        answer_cost=run.cost_usd,
        latency_s=run.latency_s,
        searches=len(run.searches),
        model_calls=run.model_calls,
        stop_reason=run.stop_reason,
    )


def summarize_answers(results: list[AnswerResult]) -> list[dict]:
    """Soru türüne göre ve toplamda başarı oranları, maliyet ve süre."""
    groups: dict[str, list[AnswerResult]] = defaultdict(list)
    for r in results:
        groups["tümü"].append(r)
        groups[r.category].append(r)
    rows = []
    for category, group in groups.items():
        judged = [r.verdict for r in group if r.verdict is not None]
        rows.append(
            {
                "tür": category,
                "soru": len(group),
                "ifade_başarısı": sum(r.keyword_pass for r in group) / len(group),
                "yargıç_doğru": sum(v.correct for v in judged) / len(judged) if judged else None,
                "yargıç_sadık": sum(v.faithful for v in judged) / len(judged) if judged else None,
                "maliyet": sum(r.answer_cost + r.judge_cost for r in group),
                "ort_süre": sum(r.latency_s for r in group) / len(group),
            }
        )
    return rows
