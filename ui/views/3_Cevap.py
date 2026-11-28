"""Cevap sayfası: soruya en yakın parçaları bul, modele kaynak göstererek cevaplat."""

from pathlib import Path

import streamlit as st

from llm_toolkit.config import Settings
from llm_toolkit.embeddings import OllamaEmbedder
from llm_toolkit.errors import LLMError
from llm_toolkit.providers import LLMClient, create_client
from llm_toolkit.rag import SYSTEM, answer_question
from llm_toolkit.vector_store import VectorStore

INDEX_DIR = Path("data/index")  # Arama sayfasında oluşturulan indeks
ANTHROPIC_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]


@st.cache_resource
def get_client(provider: str, model: str) -> LLMClient:
    """İstemciyi bir kez kurar; aynı seçimle tekrar çağrılınca hazır olanı döndürür."""
    field = "anthropic_model" if provider == "anthropic" else "ollama_model"
    return create_client(Settings().model_copy(update={"provider": provider, field: model}))


st.set_page_config(page_title="AI Lab · Cevap", page_icon="💬", layout="wide")
st.title("💬 Kaynak göstererek cevap")

settings = Settings()  # .env'deki varsayılanlar
with st.sidebar:
    st.header("Cevap modeli")
    provider = st.radio(
        "Sağlayıcı",
        ["anthropic", "ollama"],
        index=0 if settings.provider == "anthropic" else 1,
        horizontal=True,
    )
    if provider == "anthropic":
        default = settings.anthropic_model
        model = st.selectbox(
            "Model",
            ANTHROPIC_MODELS,
            index=ANTHROPIC_MODELS.index(default) if default in ANTHROPIC_MODELS else 0,
        )
    else:
        model = st.text_input("Model", value=settings.ollama_model)
    k = st.slider("Modele verilecek parça sayısı (k)", 1, 6, 3)

if not (INDEX_DIR / "chunks.json").exists():
    st.info("Önce Arama sayfasında 'İndeksi oluştur'a bas.")
    st.stop()

store = VectorStore.load(INDEX_DIR)
# Soru, indeksle aynı embedding modeliyle vektöre çevrilmeli
embedder = OllamaEmbedder(store.model, host=settings.ollama_host, timeout_s=settings.timeout_s)
st.caption(f"İndeks: {len(store.chunks)} parça · embedding modeli: {store.model}")

with st.form("ask"):
    question = st.text_input("Soru", value="Kitabı iade edersem param ne zaman hesabıma geçer?")
    submitted = st.form_submit_button("Cevapla", type="primary")

if submitted and question.strip():
    try:
        with st.spinner("Parçalar aranıyor ve cevap yazılıyor..."):
            client = get_client(provider, model)
            st.session_state.last_answer = answer_question(question, store, embedder, client, k=k)
    except LLMError as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

if "last_answer" not in st.session_state:
    st.stop()

answer = st.session_state.last_answer
st.subheader("Cevap")
st.markdown(answer.response.text)
if answer.unknown_citations:
    st.warning(f"Model, kendisine verilmeyen kaynak gösterdi: {answer.unknown_citations}")
elif not answer.citations:
    st.warning("Cevapta hiç kaynak etiketi yok.")

st.subheader("Maliyet ve süre")
col_search, col_answer, col_in, col_out, col_cost = st.columns(5)
col_search.metric("Arama süresi", f"{answer.search.latency_s:.2f} sn")
col_answer.metric("Cevap süresi", f"{answer.response.latency_s:.2f} sn")
col_in.metric("Girdi token", answer.response.usage.input_tokens)
col_out.metric("Çıktı token", answer.response.usage.output_tokens)
col_cost.metric("Toplam maliyet", f"${answer.search.cost_usd + answer.response.cost_usd:.6f}")
st.caption(
    f"Cevap: {answer.response.provider} / {answer.response.model} · "
    f"Arama: {answer.search.model} (yerel, ücretsiz)"
)

st.subheader("Modele verilen kaynaklar")
cited = set(answer.citations)
for hit in answer.hits:
    used = (hit.chunk.source, hit.chunk.index) in cited
    with st.container(border=True):
        mark = "✅ Cevapta kullanıldı" if used else "⬜ Kullanılmadı"
        st.caption(f"{mark} · {hit.chunk.source} #{hit.chunk.index} · skor {hit.score:.3f}")
        st.text(hit.chunk.text)

with st.expander("Modele gönderilen tam metin"):
    st.code(f"[system]\n{SYSTEM}\n\n[user]\n{answer.prompt}", language="text")
