"""Reranking sayfası: ilk aşamanın bulduğu adayları LLM ile yeniden sırala, farkı ölç."""

from pathlib import Path

import pandas as pd
import streamlit as st

from llm_toolkit.config import Settings
from llm_toolkit.embeddings import EmbeddingResult, OllamaEmbedder
from llm_toolkit.errors import LLMError
from llm_toolkit.evalset import load_questions
from llm_toolkit.hybrid import BM25
from llm_toolkit.providers import LLMClient, create_client
from llm_toolkit.rerank import evaluate_reranking
from llm_toolkit.retrieval_eval import METHODS, evaluate_retrieval, summarize
from llm_toolkit.vector_store import VectorStore

INDEX_DIR = Path("data/index")  # Arama sayfasında oluşturulan indeks
ANTHROPIC_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]


@st.cache_resource
def get_client(provider: str, model: str) -> LLMClient:
    field = "anthropic_model" if provider == "anthropic" else "ollama_model"
    return create_client(Settings().model_copy(update={"provider": provider, field: model}))


@st.cache_resource(show_spinner="Sorular vektöre çevriliyor...")
def embed_questions(model: str, texts: tuple[str, ...]) -> EmbeddingResult:
    settings = Settings()
    embedder = OllamaEmbedder(model, host=settings.ollama_host, timeout_s=settings.timeout_s)
    return embedder.embed(list(texts))


def overall(results) -> tuple[float, float]:
    """(recall@k, MRR) ortalaması."""
    row = next(r for r in summarize(results) if r["tür"] == "tümü")
    return row["recall"], row["mrr"]


st.set_page_config(page_title="AI Lab · Reranking", page_icon="🔀", layout="wide")
st.title("🔀 Reranking deneyi")

with st.sidebar:
    st.header("İlk aşama")
    base_method = st.selectbox("Arama yöntemi", METHODS, index=METHODS.index("hibrit"))
    candidates = st.slider("Aday sayısı (N)", 2, 10, 6)
    k = st.slider("k (ilk kaç sonuca bakılsın)", 1, 6, 1)
    st.header("Reranker")
    provider = st.radio("Sağlayıcı", ["anthropic", "ollama"], horizontal=True)
    if provider == "anthropic":
        model = st.selectbox("Model", ANTHROPIC_MODELS)
    else:
        model = st.text_input("Model", value=Settings().ollama_model)

if not (INDEX_DIR / "chunks.json").exists():
    st.info("Önce Arama sayfasında 'İndeksi oluştur'a bas.")
    st.stop()

store = VectorStore.load(INDEX_DIR)
bm25 = BM25([c.text for c in store.chunks])
answerable = [q for q in load_questions() if q.answerable]
try:
    query = embed_questions(store.model, tuple(q.question for q in answerable))
except LLMError as exc:
    st.error(f"{type(exc).__name__}: {exc}")
    st.stop()

# İlk aşama ücretsiz ve hızlı: her zaman hesaplanır
base = [
    r
    for r in evaluate_retrieval(answerable, store, bm25, query.vectors, k=k)
    if r.method == base_method
]
ceiling = [
    r
    for r in evaluate_retrieval(answerable, store, bm25, query.vectors, k=candidates)
    if r.method == base_method
]
st.caption(
    f"İndeks: {len(store.chunks)} parça · {len(answerable)} soru · ilk aşama: {base_method}, "
    f"{candidates} aday · reranker her soru için 1 çağrı yapar ({len(answerable)} çağrı)"
)

setup = (base_method, candidates, k, provider, model)
if st.button("Reranking'i çalıştır", type="primary"):
    with st.spinner(f"{len(answerable)} soru için adaylar yeniden sıralanıyor..."):
        try:
            client = get_client(provider, model)
            st.session_state.rerank_run = (
                setup,
                evaluate_reranking(
                    answerable,
                    store,
                    bm25,
                    query.vectors,
                    client,
                    base_method=base_method,
                    candidates=candidates,
                    k=k,
                ),
            )
        except LLMError as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            st.stop()

base_recall, base_mrr = overall(base)
ceiling_recall, _ = overall(ceiling)
saved = st.session_state.get("rerank_run")
run = saved[1] if saved and saved[0] == setup else None

st.subheader(f"Karşılaştırma (k = {k})")
rows = [
    {
        "yöntem": base_method,
        f"recall@{k}": base_recall,
        "MRR": base_mrr,
        "ek maliyet": 0.0,
        "ek süre (sn)": 0.0,
    }
]
if run:
    rerank_recall, rerank_mrr = overall(run.results)
    rows.append(
        {
            "yöntem": f"{base_method} + rerank ({model})",
            f"recall@{k}": rerank_recall,
            "MRR": rerank_mrr,
            "ek maliyet": run.cost_usd,
            "ek süre (sn)": run.latency_s,
        }
    )
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    column_config={
        f"recall@{k}": st.column_config.NumberColumn(format="%.2f"),
        "MRR": st.column_config.NumberColumn(format="%.2f"),
        "ek maliyet": st.column_config.NumberColumn(format="$%.6f"),
        "ek süre (sn)": st.column_config.NumberColumn(format="%.1f"),
    },
)
st.caption(
    f"Tavan: ilk aşamanın recall@{candidates} değeri **{ceiling_recall:.2f}**. Reranker sadece "
    f"ilk {candidates} adayı görür; doğru parça orada yoksa onu bulamaz."
)
if saved and not run:
    st.info("Ayarlar değişti. Yeni ayarlarla görmek için 'Reranking'i çalıştır'a bas.")

if run:
    st.subheader("Soru soru: doğru parçanın sırası")
    after = {r.question_id: r.first_rank for r in run.results}
    table = []
    for r in base:
        before, now = r.first_rank, after[r.question_id]
        change = "" if before == now else ("⬆️" if (now or 99) < (before or 99) else "⬇️")
        table.append(
            {
                "soru": r.question_id,
                "tür": r.category,
                "önce": before,
                "sonra": now,
                "değişim": change,
            }
        )
    st.dataframe(
        pd.DataFrame(table),
        hide_index=True,
        height=35 * (len(table) + 1) + 3,
        column_config={c: st.column_config.NumberColumn(format="%d") for c in ["önce", "sonra"]},
    )
    st.caption("Boş = doğru parça hiç bulunamadı. ⬇️ = reranker doğru parçayı aşağı itmiş.")
