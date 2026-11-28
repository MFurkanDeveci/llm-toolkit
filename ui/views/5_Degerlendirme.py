"""Değerlendirme sayfası: altın soru setiyle vektör, BM25 ve hibrit aramayı ölç."""

import pandas as pd
import streamlit as st

from llm_toolkit.config import Settings
from llm_toolkit.embeddings import DEFAULT_MODEL, EmbeddingResult, OllamaEmbedder
from llm_toolkit.errors import LLMError
from llm_toolkit.evalset import load_documents, load_questions
from llm_toolkit.hybrid import BM25
from llm_toolkit.retrieval_eval import METHODS, evaluate_retrieval, summarize
from llm_toolkit.vector_store import VectorStore, build_index


def make_embedder(model: str) -> OllamaEmbedder:
    settings = Settings()  # Ollama adresi ve zaman aşımı .env'den
    return OllamaEmbedder(model, host=settings.ollama_host, timeout_s=settings.timeout_s)


@st.cache_resource(show_spinner="Belgeler parçalanıp vektöre çevriliyor...")
def get_index(model: str, max_chars: int, overlap_chars: int) -> tuple[VectorStore, BM25, float]:
    """Seçilen ayarlarla indeksi bellekte kurar; aynı ayarlar için tekrar hesaplamaz."""
    store, result = build_index(
        load_documents(), make_embedder(model), max_chars=max_chars, overlap_chars=overlap_chars
    )
    return store, BM25([c.text for c in store.chunks]), result.latency_s


@st.cache_resource(show_spinner="Sorular vektöre çevriliyor...")
def embed_questions(model: str, texts: tuple[str, ...]) -> EmbeddingResult:
    return make_embedder(model).embed(list(texts))


st.set_page_config(page_title="AI Lab · Değerlendirme", page_icon="📏", layout="wide")
st.title("📏 Arama değerlendirmesi")

with st.sidebar:
    st.header("Ayarlar")
    model = st.text_input("Embedding modeli (Ollama)", value=DEFAULT_MODEL)
    max_chars = st.slider("Parça boyu (karakter)", 200, 2000, 400, step=50)
    overlap_chars = st.slider("Örtüşme (karakter)", 0, 500, 80, step=10)
    k = st.slider("k (ilk kaç sonuca bakılsın)", 1, 6, 3)

questions = load_questions()
answerable = [q for q in questions if q.answerable]
st.caption(
    f"Soru seti: {len(questions)} soru · {len(answerable)} tanesi ölçülüyor "
    f"(cevabı belgelerde olmayan {len(questions) - len(answerable)} soru Ders 4.3'te ölçülecek)"
)

if overlap_chars >= max_chars:
    st.warning("Örtüşme, parça boyundan küçük olmalı.")
    st.stop()
if not st.session_state.get("eval_started"):
    if not st.button("Ölçümü başlat", type="primary"):
        st.stop()
    st.session_state.eval_started = True  # Bundan sonra ayar değişince kendiliğinden ölçer

try:
    store, bm25, index_seconds = get_index(model, max_chars, overlap_chars)
    query = embed_questions(model, tuple(q.question for q in answerable))
except LLMError as exc:
    st.error(f"{type(exc).__name__}: {exc}")
    st.stop()

results = evaluate_retrieval(answerable, store, bm25, query.vectors, k=k)
summary = pd.DataFrame(summarize(results))

col_chunks, col_index, col_query, col_cost = st.columns(4)
col_chunks.metric("Parça sayısı", len(store.chunks))
col_index.metric("İndeks süresi", f"{index_seconds:.2f} sn")
col_query.metric("Soru embedding süresi", f"{query.latency_s:.2f} sn")
col_cost.metric("Maliyet", f"${query.cost_usd:.6f}")

st.subheader(f"Genel sonuç (k = {k})")
overall = summary[summary["tür"] == "tümü"].set_index("yöntem").loc[list(METHODS)]
col_table, col_chart = st.columns([2, 3])
col_table.dataframe(
    overall[["recall", "mrr"]].rename(columns={"recall": f"recall@{k}", "mrr": "MRR"}),
    column_config={c: st.column_config.NumberColumn(format="%.2f") for c in [f"recall@{k}", "MRR"]},
)
by_category = summary[summary["tür"] != "tümü"].pivot(
    index="tür", columns="yöntem", values="recall"
)
# Yöntem başına sabit renk: vektör turuncu, BM25 mavi, hibrit yeşil
col_chart.bar_chart(
    by_category[list(METHODS)],
    stack=False,
    color=["#e4572e", "#4c78a8", "#54a24b"],
    y_label=f"recall@{k}",
    height=260,
)
col_chart.caption("Soru türüne göre recall: hangi yöntem nerede zorlanıyor?")

st.subheader("Soru soru")
st.caption(
    f"Sayılar: ilk ilgili parçanın sırası. Boş = hiç bulunamadı · {k}'dan büyük = ilk {k}'da yok"
)
ranks = pd.DataFrame(
    [
        {"soru": r.question_id, "tür": r.category, "yöntem": r.method, "sıra": r.first_rank}
        for r in results
    ]
).pivot(index=["soru", "tür"], columns="yöntem", values="sıra")[list(METHODS)]
texts = {q.id: q.question for q in answerable}
ranks.insert(0, "metin", [texts[qid] for qid, _ in ranks.index])
st.dataframe(
    ranks.reset_index(),
    hide_index=True,
    height=35 * (len(ranks) + 1) + 3,  # Kaydırmadan tüm sorular görünsün
    column_config={m: st.column_config.NumberColumn(format="%d") for m in METHODS},
)
