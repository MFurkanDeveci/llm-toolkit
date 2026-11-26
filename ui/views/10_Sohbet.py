"""Sohbet sayfası: hafızalı agent. Devam soruları önceki konuşmaya göre anlaşılır."""

from pathlib import Path

import pandas as pd
import streamlit as st

from llm_toolkit.agent import REASONS, AgentRun, Limits, build_history, run_agent
from llm_toolkit.config import Settings
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


def show_turn(run: AgentRun, memory_turns: int) -> None:
    """Bir soru-cevap turunu sohbet balonları olarak gösterir."""
    with st.chat_message("user"):
        st.markdown(run.question)
    with st.chat_message("assistant"):
        if run.stop_reason != "cevap":
            st.warning(f"Agent {REASONS[run.stop_reason]} takıldı; cevap eksik olabilir.")
        st.markdown(run.answer)
        if run.unknown_citations:
            st.warning(f"Doğrulanamayan kaynak: {run.unknown_citations}")
        input_tokens = sum(r.usage.input_tokens for r in run.responses)
        st.caption(
            f"{run.model_calls} çağrı · {len(run.searches)} arama · girdi {input_tokens} token · "
            f"${run.cost_usd:.6f} · hafızada {memory_turns} önceki tur"
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
    if st.button("Sohbeti temizle"):
        st.session_state.chat = []
        st.rerun()

if not (INDEX_DIR / "chunks.json").exists():
    st.info("Önce Arama sayfasında 'İndeksi oluştur'a bas.")
    st.stop()

store = VectorStore.load(INDEX_DIR)
settings = Settings()
embedder = OllamaEmbedder(store.model, host=settings.ollama_host, timeout_s=settings.timeout_s)
chat: list[dict] = st.session_state.setdefault("chat", [])  # Her eleman: {"run", "memory_turns"}

if not chat:
    st.caption(
        "Şöyle bir sohbet dene: 1) İade süresi ne kadar? → 2) Peki e-kitaplar için? → "
        "3) İade ederken kargo ücreti öder miyim?"
    )
for turn in chat:
    show_turn(turn["run"], turn["memory_turns"])

question = st.chat_input("Bir soru yaz...")
if question:
    history = build_history([turn["run"] for turn in chat], max_turns)
    try:
        with st.spinner("Agent çalışıyor..."):
            client = get_client(provider, model)
            run = run_agent(question, store, embedder, client, limits=Limits(), history=history)
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
