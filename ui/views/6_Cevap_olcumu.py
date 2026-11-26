"""Cevap ölçümü sayfası: soru setini sabit RAG ya da agent ile cevaplat, ifade kontrolü ve LLM
yargıçla puanla. Sonuçlar deney kaydına yazılır; Karşılaştırma sayfası onları yan yana gösterir.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from llm_toolkit.agent import Limits
from llm_toolkit.answer_eval import evaluate_agent_answer, evaluate_answer, summarize_answers
from llm_toolkit.config import Settings
from llm_toolkit.embeddings import OllamaEmbedder
from llm_toolkit.errors import LLMError
from llm_toolkit.evalset import load_questions
from llm_toolkit.providers import LLMClient, create_client
from llm_toolkit.runs import save_run
from llm_toolkit.vector_store import VectorStore

INDEX_DIR = Path("data/index")  # Arama sayfasında oluşturulan indeks
ANTHROPIC_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]


@st.cache_resource
def get_client(provider: str, model: str) -> LLMClient:
    field = "anthropic_model" if provider == "anthropic" else "ollama_model"
    return create_client(Settings().model_copy(update={"provider": provider, field: model}))


def pick_model(label: str, key: str, default_model: str) -> tuple[str, str]:
    """Kenar çubuğunda sağlayıcı ve model seçimi; (sağlayıcı, model) döndürür."""
    provider = st.radio(label, ["anthropic", "ollama"], horizontal=True, key=f"{key}_provider")
    if provider == "anthropic":
        index = ANTHROPIC_MODELS.index(default_model)
        return provider, st.selectbox("Model", ANTHROPIC_MODELS, index=index, key=f"{key}_model")
    return provider, st.text_input("Model", value=Settings().ollama_model, key=f"{key}_ollama")


st.set_page_config(page_title="AI Lab · Cevap ölçümü", page_icon="🧑‍⚖️", layout="wide")
st.title("🧑‍⚖️ Cevap ölçümü")

with st.sidebar:
    st.header("Cevap modeli")
    method = st.radio("Yöntem", ["sabit RAG", "agent"], horizontal=True)
    answer_provider, answer_model = pick_model("Sağlayıcı", "answer", "claude-haiku-5-5")
    k = st.slider("Arama başına parça sayısı (k)", 1, 6, 3)
    if method == "agent":
        st.caption("Agent varsayılan korumalarla çalışır: 5 adım, $0.05, 60 sn (soru başına).")
    st.header("Yargıç")
    use_judge = st.toggle("LLM yargıç kullan", value=True)
    if use_judge:
        judge_provider, judge_model = pick_model("Sağlayıcı", "judge", "claude-sonnet-5-5")

if not (INDEX_DIR / "chunks.json").exists():
    st.info("Önce Arama sayfasında 'İndeksi oluştur'a bas.")
    st.stop()

store = VectorStore.load(INDEX_DIR)
settings = Settings()
embedder = OllamaEmbedder(store.model, host=settings.ollama_host, timeout_s=settings.timeout_s)
questions = load_questions()

calls = len(questions) * (2 if use_judge else 1)
at_least = "en az " if method == "agent" else ""  # Agent soru başına 2+ çağrı yapabilir
st.caption(
    f"İndeks: {len(store.chunks)} parça · {len(questions)} soru · {at_least}{calls} model çağrısı "
    f"(cevap{' + yargıç' if use_judge else ''})"
)
if "anthropic" in (answer_provider, judge_provider if use_judge else ""):
    st.warning("Bu ölçüm ücretli Claude çağrıları yapar. Maliyet çalışırken aşağıda güncellenir.")

if st.button("Ölçümü çalıştır", type="primary"):
    client = get_client(answer_provider, answer_model)
    judge = get_client(judge_provider, judge_model) if use_judge else None
    progress = st.progress(0.0, text="Başlıyor...")
    live_cost = st.empty()
    results, spent = [], 0.0
    try:
        for i, question in enumerate(questions, start=1):
            progress.progress(i / len(questions), text=f"{i}/{len(questions)} · {question.id}")
            if method == "agent":
                result = evaluate_agent_answer(
                    question, store, embedder, client, judge=judge, k=k, limits=Limits()
                )
            else:
                result = evaluate_answer(question, store, embedder, client, judge=judge, k=k)
            results.append(result)
            spent += result.answer_cost + result.judge_cost
            live_cost.metric("Şu ana kadarki maliyet", f"${spent:.6f}")
    except LLMError as exc:
        st.error(f"{question.id} sorusunda durdu. {type(exc).__name__}: {exc}")
        st.stop()
    progress.empty()
    live_cost.empty()
    st.session_state.answer_eval = {
        "results": results,
        "method": method,
        "setup": (
            f"Yöntem: {method} · Cevap: {answer_model} · "
            f"Yargıç: {judge_model if use_judge else 'yok'} · k={k}"
        ),
    }
    # Deney kaydı: Karşılaştırma sayfası bu dosyaları okur
    path = save_run(
        results,
        {
            "yöntem": method,
            "cevap_modeli": answer_model,
            "yargıç": judge_model if use_judge else "yok",
            "k": k,
            "parça_sayısı": len(store.chunks),
            "embedding": store.model,
        },
    )
    st.success(f"Çalıştırma kaydedildi: {path}")

if "answer_eval" not in st.session_state:
    st.stop()

run = st.session_state.answer_eval
results = run["results"]
summary = pd.DataFrame(summarize_answers(results)).set_index("tür")
total = summary.loc["tümü"]
st.subheader("Genel sonuç")
st.caption(run["setup"])

judged = total["yargıç_doğru"] is not None and not pd.isna(total["yargıç_doğru"])
cols = st.columns(5)
cols[0].metric("İfade kontrolü", f"%{100 * total['ifade_başarısı']:.0f}")
cols[1].metric("Yargıç: doğru", f"%{100 * total['yargıç_doğru']:.0f}" if judged else "–")
cols[2].metric("Yargıç: kaynağa sadık", f"%{100 * total['yargıç_sadık']:.0f}" if judged else "–")
cols[3].metric("Toplam maliyet", f"${total['maliyet']:.6f}")
cols[4].metric("Ort. cevap süresi", f"{total['ort_süre']:.2f} sn")

st.dataframe(
    summary.drop(index="tümü"),
    column_config={
        "ifade_başarısı": st.column_config.NumberColumn("ifade", format="percent"),
        "yargıç_doğru": st.column_config.NumberColumn("doğru", format="percent"),
        "yargıç_sadık": st.column_config.NumberColumn("sadık", format="percent"),
        "maliyet": st.column_config.NumberColumn(format="$%.6f"),
        "ort_süre": st.column_config.NumberColumn("ort. süre (sn)", format="%.2f"),
    },
)

if run.get("method") == "agent":
    st.subheader("Agent nasıl çalıştı?")
    searches = pd.Series([r.searches for r in results])
    col_chart, col_stats = st.columns([2, 1])
    with col_chart:
        counts = searches.value_counts().reindex(range(searches.max() + 1), fill_value=0)
        st.bar_chart(counts, x_label="Arama sayısı", y_label="Soru sayısı", height=220)
        st.caption("0 = hiç aramadan cevap verdi. Sabit RAG'da her soru tam 1 arama yapar.")
    with col_stats:
        st.metric("Ort. arama", f"{searches.mean():.1f}")
        st.metric("Ort. model çağrısı", f"{sum(r.model_calls for r in results) / len(results):.1f}")
        limited = [r.question_id for r in results if r.stop_reason != "cevap"]
        st.metric("Sınıra takılan", len(limited))
        if limited:
            st.caption(", ".join(limited))

st.subheader("Soru soru")
only_problems = st.toggle("Sadece sorunlu soruları göster", value=False)
rows = []
for r in results:
    disagree = r.verdict is not None and r.verdict.correct != r.keyword_pass
    problem = not r.keyword_pass or (
        r.verdict is not None and not (r.verdict.correct and r.verdict.faithful)
    )
    if only_problems and not problem:
        continue
    rows.append(
        {
            "soru": r.question_id,
            "tür": r.category,
            "arama": r.searches,
            "ifade": "✅" if r.keyword_pass else "❌",
            "doğru": ("✅" if r.verdict.correct else "❌") if r.verdict else "–",
            "sadık": ("✅" if r.verdict.faithful else "❌") if r.verdict else "–",
            "anlaşmazlık": "⚠️" if disagree else "",
            "cevap": r.answer,
            "yargıç gerekçesi": r.verdict.reason if r.verdict else "",
        }
    )
st.caption(
    "⚠️ = ifade kontrolü ile yargıç farklı karar vermiş. "
    "Bu satırların cevabını okuyup kimin haklı olduğuna bak."
)
st.dataframe(pd.DataFrame(rows), hide_index=True, height=35 * (len(rows) + 1) + 3)
