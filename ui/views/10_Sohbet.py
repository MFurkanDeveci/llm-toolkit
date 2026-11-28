"""Sohbet sayfası: hafızalı agent. Devam soruları önceki konuşmaya göre anlaşılır.

Ders 5.5b: agent iade başlatabilir, ama sadece kullanıcı onay kartında "Onayla"ya basarsa.
Onay beklerken yeni soru yazılamaz; böylece yarım kalmış bir işlem unutulmaz.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from llm_toolkit.actions import RefundDesk
from llm_toolkit.agent import REASONS, AgentRun, Limits, build_history, resume_agent, run_agent
from llm_toolkit.config import Settings
from llm_toolkit.datatypes import ToolCall
from llm_toolkit.embeddings import OllamaEmbedder
from llm_toolkit.errors import LLMError
from llm_toolkit.providers import LLMClient, create_client
from llm_toolkit.vector_store import VectorStore

INDEX_DIR = Path("data/index")  # Arama sayfasında oluşturulan indeks
ANTHROPIC_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]


@st.cache_resource
def get_client(provider: str, model: str) -> LLMClient:
    field = "anthropic_model" if provider == "anthropic" else "ollama_model"
    return create_client(Settings().model_copy(update={"provider": provider, field: model}))


def describe(call: ToolCall) -> str:
    """Bir işlem isteğini kullanıcının okuyacağı hale getirir."""
    args = ", ".join(f"{key}: **{value}**" for key, value in call.arguments.items())
    return f"`{call.name}` → {args}"


def show_turn(turn: dict) -> None:
    """Bir soru-cevap turunu sohbet balonları olarak gösterir."""
    run: AgentRun = turn["run"]
    with st.chat_message("user"):
        st.markdown(run.question)
    with st.chat_message("assistant"):
        if run.stop_reason in REASONS:  # Adım, bütçe ya da süre sınırı
            st.warning(f"Agent {REASONS[run.stop_reason]} takıldı; cevap eksik olabilir.")
        for action in turn.get("actions", []):  # Bu turda karar verilmiş işlemler
            icon = "✅" if turn["decision"] == "onaylandı" else "🚫"
            st.caption(f"{icon} İşlem {turn['decision']}: {action}")
        if run.stop_reason == "onay":
            st.markdown(run.answer or "_Bu işlem için onayın gerekiyor._")
        else:
            st.markdown(run.answer)
        if run.unknown_citations:
            st.warning(f"Doğrulanamayan kaynak: {run.unknown_citations}")
        input_tokens = sum(r.usage.input_tokens for r in run.responses)
        st.caption(
            f"{run.model_calls} çağrı · {len(run.searches)} arama · girdi {input_tokens} token · "
            f"${run.cost_usd:.6f} · hafızada {turn['memory_turns']} önceki tur"
        )
        if run.searches:
            with st.expander("Aramalar"):
                for step in run.searches:
                    sources = ", ".join(f"{h.chunk.source} #{h.chunk.index}" for h in step.hits)
                    st.markdown(f"🔎 «{step.query}» → {sources}")


st.set_page_config(page_title="AI Lab · Sohbet", page_icon="🗨️", layout="wide")
st.title("🗨️ Sohbet: hafızalı agent")

with st.sidebar:
    st.header("Agent modeli")
    provider = st.radio("Sağlayıcı", ["anthropic", "ollama"], horizontal=True)
    if provider == "anthropic":
        model = st.selectbox("Model", ANTHROPIC_MODELS)
    else:
        model = st.text_input("Model", value=Settings().ollama_model)
    st.header("Hafıza")
    max_turns = st.slider("Kaç önceki tur hatırlansın (0 = kapalı)", 0, 10, 3)
    st.header("İşlemler")
    use_actions = st.toggle("Agent iade başlatabilsin (onayla)", value=True)
    if st.button("Sohbeti temizle"):
        st.session_state.chat = []
        st.rerun()

if not (INDEX_DIR / "chunks.json").exists():
    st.info("Önce Arama sayfasında 'İndeksi oluştur'a bas.")
    st.stop()

store = VectorStore.load(INDEX_DIR)
settings = Settings()
embedder = OllamaEmbedder(store.model, host=settings.ollama_host, timeout_s=settings.timeout_s)
# Her eleman: {"run", "memory_turns"}; işlem kararı verilmiş turlarda "actions" ve "decision"
chat: list[dict] = st.session_state.setdefault("chat", [])
# Sahte iade sistemi sayfa yenilense de yaşasın; "Sohbeti temizle" iadeleri silmez (gerçek işlem)
desk: RefundDesk = st.session_state.setdefault("desk", RefundDesk())
actions = [desk.as_tool()] if use_actions else []

if not chat:
    st.caption(
        "Şöyle bir sohbet dene: 1) İade süresi ne kadar? → 2) Peki e-kitaplar için? → "
        "3) İade ederken kargo ücreti öder miyim? İşlem denemek için: "
        "KK-123456 numaralı siparişimi iade etmek istiyorum, kitap hasarlı geldi."
    )
for turn in chat:
    show_turn(turn)

waiting = bool(chat) and chat[-1]["run"].stop_reason == "onay"
if waiting:
    turn = chat[-1]
    with st.chat_message("assistant", avatar="⚠️"), st.container(border=True):
        st.markdown("**Agent bir işlem yapmak istiyor. Onaylıyor musun?**")
        for call in turn["run"].pending:
            st.markdown(describe(call))
        col_yes, col_no, _ = st.columns([1, 1, 4])
        approve = col_yes.button("Onayla", type="primary")
        reject = col_no.button("Reddet")
    if approve or reject:
        try:
            with st.spinner("Agent devam ediyor..."):
                client = get_client(provider, model)
                run = resume_agent(turn["run"], approve, store, embedder, client, actions=actions)
        except LLMError as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            st.stop()
        turn["actions"] = [describe(call) for call in turn["run"].pending]
        turn["decision"] = "onaylandı" if approve else "reddedildi"
        turn["run"] = run
        st.rerun()

question = st.chat_input(
    "Önce yukarıdaki işlemi onayla ya da reddet" if waiting else "Bir soru yaz...", disabled=waiting
)
if question:
    history = build_history([turn["run"] for turn in chat], max_turns)
    try:
        with st.spinner("Agent çalışıyor..."):
            client = get_client(provider, model)
            run = run_agent(
                question,
                store,
                embedder,
                client,
                limits=Limits(),
                history=history,
                actions=actions,
            )
    except LLMError as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()
    chat.append({"run": run, "memory_turns": len(history) // 2})
    st.rerun()  # Yeni turu yukarıdaki listede göstermek için sayfayı baştan çiz

if chat:
    with st.sidebar:
        st.header("Hafızanın maliyeti")
        tokens = [sum(r.usage.input_tokens for r in t["run"].responses) for t in chat]
        per_turn = pd.DataFrame({"girdi token": tokens}, index=range(1, len(tokens) + 1))
        st.bar_chart(per_turn, x_label="Tur", y_label="Girdi token", height=180)
        st.caption("Her tur, hatırlanan önceki turları da tekrar gönderir; token sayısı büyür.")
        total = sum(t["run"].cost_usd for t in chat)
        st.metric("Sohbetin toplam maliyeti", f"${total:.6f}")

if desk.requests:
    with st.sidebar:
        st.header("Oluşturulan iadeler")
        st.dataframe(
            pd.DataFrame([r.model_dump() for r in desk.requests]),
            hide_index=True,
            column_config={"refund_id": "takip no", "order_id": "sipariş", "reason": "sebep"},
        )
