"""Hibrit arama sayfası: vektör, BM25 ve ikisinin birleşimini yan yana karşılaştır."""

import time
from pathlib import Path

import numpy as np
import streamlit as st

from llm_toolkit.chunking import Chunk
from llm_toolkit.config import Settings
from llm_toolkit.embeddings import OllamaEmbedder, cosine_similarity
from llm_toolkit.errors import LLMError
from llm_toolkit.hybrid import BM25, hybrid_search, tokenize
from llm_toolkit.vector_store import VectorStore

INDEX_DIR = Path("data/index")  # Arama sayfasında oluşturulan indeks
EXAMPLES = [
    "Kargo ücreti 39,90 TL mi?",
    "Kitabı geri göndermek için kaç günüm var?",
    "14 gün",
    "Şifremi unuttum",
]


@st.cache_resource
def get_bm25(texts: tuple[str, ...]) -> BM25:
    """BM25 indeksini parça metinlerinden bir kez kurar (birkaç milisaniye sürer)."""
    return BM25(list(texts))


def show_card(rank: int, chunk: Chunk, detail: str) -> None:
    with st.container(border=True):
        st.caption(f"Sıra {rank} · {chunk.source} #{chunk.index} · {detail}")
        st.text(chunk.text)


st.set_page_config(page_title="AI Lab · Hibrit arama", page_icon="⚖️", layout="wide")
st.title("⚖️ Hibrit arama")

with st.sidebar:
    st.header("Ayarlar")
    k = st.slider("Sonuç sayısı (k)", 1, 6, 3)

if not (INDEX_DIR / "chunks.json").exists():
    st.info("Önce Arama sayfasında 'İndeksi oluştur'a bas.")
    st.stop()

store = VectorStore.load(INDEX_DIR)
bm25 = get_bm25(tuple(c.text for c in store.chunks))
settings = Settings()  # Ollama adresi ve zaman aşımı .env'den
embedder = OllamaEmbedder(store.model, host=settings.ollama_host, timeout_s=settings.timeout_s)
st.caption(f"İndeks: {len(store.chunks)} parça · embedding modeli: {store.model}")

with st.form("hybrid"):
    query = st.text_input("Soru", value=EXAMPLES[0])
    submitted = st.form_submit_button("Karşılaştır", type="primary")
st.caption("Bunları da dene: " + " · ".join(EXAMPLES[1:]))

if submitted and query.strip():
    try:
        st.session_state.last_hybrid = (query, embedder.embed([query]))
    except LLMError as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

if "last_hybrid" not in st.session_state:
    st.stop()

query, result = st.session_state.last_hybrid
if result.model != store.model:
    st.info("İndeks değişti; soruyu tekrar gönder.")
    st.stop()

start = time.perf_counter()
bm25_scores = bm25.scores(query)
bm25_ms = (time.perf_counter() - start) * 1000
vector_scores = cosine_similarity(result.vectors[0].reshape(1, -1), store.vectors)[0]
hybrid_hits = hybrid_search(store, bm25, result.vectors[0], query, k=k)

col_embed, col_bm25, col_cost = st.columns(3)
col_embed.metric("Embedding süresi", f"{result.latency_s:.2f} sn")
col_bm25.metric("BM25 süresi", f"{bm25_ms:.2f} ms")
col_cost.metric("Maliyet", f"${result.cost_usd:.6f}")
st.caption(f"BM25'in gördüğü kelime kökleri: {tokenize(query)}")

col_vector, col_keyword, col_hybrid = st.columns(3)
with col_vector:
    st.markdown("**Vektör (anlam)**")
    for rank, i in enumerate(np.argsort(-vector_scores)[:k], start=1):
        show_card(rank, store.chunks[i], f"benzerlik {vector_scores[i]:.3f}")
with col_keyword:
    st.markdown("**BM25 (kelime)**")
    matched = [i for i in np.argsort(-bm25_scores) if bm25_scores[i] > 0][:k]
    if not matched:
        st.info("Sorudaki hiçbir kelime parçalarda geçmiyor.")
    for rank, i in enumerate(matched, start=1):
        show_card(rank, store.chunks[i], f"puan {bm25_scores[i]:.2f}")
with col_hybrid:
    st.markdown("**Hibrit (RRF)**")
    for rank, hit in enumerate(hybrid_hits, start=1):
        bm25_rank = f"{hit.bm25_rank}." if hit.bm25_rank else "yok"
        show_card(rank, hit.chunk, f"vektörde {hit.vector_rank}. · BM25'te {bm25_rank}")
