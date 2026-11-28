"""İnsan onayı testleri: işlem aracı onaysız çalışmamalı; onay ve ret doğru işlenmeli."""

import numpy as np
import pytest
from conftest import FakeLLM, make_response

from llm_toolkit.actions import RefundDesk
from llm_toolkit.agent import REJECTED, resume_agent, run_agent
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
QUESTION = "KK-123456 numaralı siparişimi iade etmek istiyorum, kitap hasarlı geldi."


def refund(order_id: str = "KK-123456", call_id: str = "r1"):
    """Sahte modelin 'iade_baslat' aracını çağırdığı bir cevap."""
    args = {"order_id": order_id, "reason": "Kitap hasarlı geldi"}
    call = ToolCall(id=call_id, name="iade_baslat", arguments=args)
    return make_response("İadenizi başlatıyorum.", tool_calls=[call])


def start(llm: FakeLLM, desk: RefundDesk):
    return run_agent(QUESTION, STORE, KeywordEmbedder(), llm, k=1, actions=[desk.as_tool()])


def resume(run, approved: bool, llm: FakeLLM, desk: RefundDesk):
    return resume_agent(run, approved, STORE, KeywordEmbedder(), llm, k=1, actions=[desk.as_tool()])


def test_action_waits_for_approval_and_does_not_run():
    desk = RefundDesk()
    run = start(FakeLLM([refund()]), desk)
    assert run.stop_reason == "onay"
    assert [c.name for c in run.pending] == ["iade_baslat"]
    assert run.pending[0].arguments["order_id"] == "KK-123456"
    assert desk.requests == []  # Onay gelmeden hiçbir işlem yapılmadı


def test_approved_action_runs_and_agent_continues():
    desk = RefundDesk()
    llm = FakeLLM([refund(), make_response("İade talebiniz oluşturuldu: IADE-0001.")])
    run = resume(start(llm, desk), True, llm, desk)
    assert [r.order_id for r in desk.requests] == ["KK-123456"]
    assert run.stop_reason == "cevap"
    assert run.answer == "İade talebiniz oluşturuldu: IADE-0001."
    assert run.pending == []
    assert run.model_calls == 2  # Onaydan önceki ve sonraki çağrılar birlikte sayılır
    tool_result = llm.calls[1][-1]  # Modele işlemin sonucu gönderildi
    assert tool_result.role == "tool" and "IADE-0001" in tool_result.content


def test_rejected_action_does_not_run_and_model_is_told():
    desk = RefundDesk()
    llm = FakeLLM([refund(), make_response("Anlaşıldı, iade başlatılmadı.")])
    run = resume(start(llm, desk), False, llm, desk)
    assert desk.requests == []
    assert run.stop_reason == "cevap"
    assert llm.calls[1][-1].content == REJECTED


def test_invalid_arguments_go_back_to_model_without_asking_human():
    desk = RefundDesk()
    # Sipariş numarası biçimi yanlış: insana gösterilmez, hata modele döner, model kullanıcıya sorar
    llm = FakeLLM([refund(order_id="123"), make_response("Sipariş numaranızı yazar mısınız?")])
    run = start(llm, desk)
    assert run.stop_reason == "cevap"
    assert desk.requests == []
    assert llm.calls[1][-1].content.startswith("HATA: Argümanlar geçersiz")


def test_search_in_same_step_runs_immediately_only_action_waits():
    desk = RefundDesk()
    search = ToolCall(id="s1", name="belgelerde_ara", arguments={"query": "iade"})
    action = refund().tool_calls[0]
    run = start(FakeLLM([make_response(tool_calls=[search, action])]), desk)
    assert run.stop_reason == "onay"
    assert [s.query for s in run.searches] == ["iade"]  # Arama beklemeden çalıştı
    assert [c.name for c in run.pending] == ["iade_baslat"]


def test_resume_requires_a_waiting_run():
    desk = RefundDesk()
    llm = FakeLLM([make_response("Merhaba.")])
    run = start(llm, desk)
    with pytest.raises(ValueError):
        resume(run, True, llm, desk)


def test_refund_desk_refuses_duplicate_requests():
    desk = RefundDesk()
    tool = desk.as_tool()
    args = tool.parameters(order_id="KK-123456", reason="Hasarlı")
    assert "IADE-0001" in tool.function(args)
    assert "zaten" in tool.function(args)  # Aynı sipariş için ikinci talep açılmaz
    assert len(desk.requests) == 1
