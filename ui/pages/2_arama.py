"""Arama sayfası: dokümanları indeksle, bir soruya anlamca en yakın parçaları bul."""

from pathlib import Path

import streamlit as st

from llm_toolkit.config import Settings
from llm_toolkit.embeddings import DEFAULT_MODEL, EmbeddingResult, OllamaEmbedder
from llm_toolkit.errors import LLMError
from llm_toolkit.vector_store import VectorStore, build_index

DOCS_DIR = Path("data/docs")  # Aranacak dokümanlar (.md, .txt)
INDEX_DIR = Path("data/index")  # İndeksin kaydedildiği yer; Git'e girmez


def show_usage(result: EmbeddingResult) -> None:
    col_tokens, col_time, col_cost = st.columns(3)
    col_tokens.metric("Girdi token", result.input_tokens)
    col_time.metric("Süre", f"{result.latency_s:.2f} sn")
    col_cost.metric("Maliyet", f"${result.cost_usd:.6f}")


st.set_page_config(page_title="AI Lab · Arama", page_icon="🔎", layout="wide")
st.title("🔎 Vektör arama")

with st.sidebar:
    st.header("Ayarlar")
    model = st.text_input("Embedding modeli (Ollama)", value=DEFAULT_MODEL)
    max_chars = st.slider("Parça boyu (karakter)", 200, 2000, 400, step=50)
    overlap_chars = st.slider("Örtüşme (karakter)", 0, 500, 80, step=10)
    k = st.slider("Sonuç sayısı (k)", 1, 10, 3)

settings = Settings()  # Ollama adresi ve zaman aşımı .env'den
embedder = OllamaEmbedder(model, host=settings.ollama_host, timeout_s=settings.timeout_s)

# --- 1. İndeks: dokümanları parçala, vektöre çevir, diske kaydet ---
st.subheader("1. İndeks")
paths = sorted([*DOCS_DIR.glob("*.md"), *DOCS_DIR.glob("*.txt")])
st.caption(f"{DOCS_DIR} klasöründe {len(paths)} doküman: {', '.join(p.name for p in paths)}")

if overlap_chars >= max_chars:
    st.warning("Örtüşme, parça boyundan küçük olmalı.")
elif st.button("İndeksi oluştur", disabled=not paths):
    documents = {p.name: p.read_text(encoding="utf-8") for p in paths}
    try:
        with st.spinner("Parçalar vektöre çevriliyor..."):
            store, result = build_index(
                documents, embedder, max_chars=max_chars, overlap_chars=overlap_chars
            )
    except LLMError as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()
    store.save(INDEX_DIR)
    st.success(f"{len(store.chunks)} parça indekslendi ve {INDEX_DIR} klasörüne kaydedildi.")
    show_usage(result)

if not (INDEX_DIR / "chunks.json").exists():
    st.info("Henüz indeks yok. 'İndeksi oluştur'a bas.")
    st.stop()

store = VectorStore.load(INDEX_DIR)
st.caption(f"Yüklü indeks: {len(store.chunks)} parça · model: {store.model}")
if store.model != model:
    st.warning(f"İndeks '{store.model}' ile oluşturulmuş, seçili model '{model}'. Yeniden oluştur.")
    st.stop()

# --- 2. Arama: soruyu aynı modelle vektöre çevir, en yakın k parçayı getir ---
st.subheader("2. Ara")
with st.form("search"):
    query = st.text_input("Soru", value="Kitabı geri göndermek için kaç günüm var?")
    submitted = st.form_submit_button("Ara", type="primary")

if submitted and query.strip():
    try:
        st.session_state.last_search = (query, embedder.embed([query]))
    except LLMError as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

if "last_search" in st.session_state:
    query, result = st.session_state.last_search
    if result.model != store.model:
        st.info("Model değişti; soruyu tekrar ara.")
        st.stop()
    show_usage(result)
    # Soru vektörü saklı olduğu için k değişince model tekrar çağrılmadan yeniden sıralanır
    for rank, hit in enumerate(store.search(result.vectors[0], k=k), start=1):
        with st.container(border=True):
            st.caption(
                f"Sonuç {rank} · skor {hit.score:.3f} · {hit.chunk.source} · "
                f"parça #{hit.chunk.index}"
            )
            st.text(hit.chunk.text)
