"""Agent sayfası: model aramayı bir araç olarak kullanıyor; her adımını izle."""

from pathlib import Path

import streamlit as st

from llm_toolkit.agent import AGENT_SYSTEM, run_agent
from llm_toolkit.config import Settings
from llm_toolkit.embeddings import OllamaEmbedder
from llm_toolkit.errors import LLMError
from llm_toolkit.providers import LLMClient, create_client
from llm_toolkit.vector_store import VectorStore

INDEX_DIR = Path("data/index")  # Arama sayfasında oluşturulan indeks
ANTHROPIC_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
EXAMPLES = [
    "İade gönderirken kargo ücreti öder miyim, normal siparişte kargo ne kadar?",
    "Kitabı geri göndermek için kaç günüm var?",
    "Kapıda ödeme seçeneği var mı?",
]


@st.cache_resource
def get_client(provider: str, model: str) -> LLMClient:
    field = "anthropic_model" if provider == "anthropic" else "ollama_model"
    return create_client(Settings().model_copy(update={"provider": provider, field: model}))


st.set_page_config(page_title="AI Lab · Agent", page_icon="🤖", layout="wide")
st.title("🤖 Agent: aramayı model yönetiyor")

with st.sidebar:
    st.header("Agent modeli")
    provider = st.radio("Sağlayıcı", ["anthropic", "ollama"], horizontal=True)
    if provider == "anthropic":
        model = st.selectbox("Model", ANTHROPIC_MODELS)
    else:
        model = st.text_input("Model", value=Settings().ollama_model)
    k = st.slider("Her aramada kaç parça (k)", 1, 6, 3)
    max_steps = st.slider("En fazla model çağrısı (adım)", 2, 10, 5)

if not (INDEX_DIR / "chunks.json").exists():
    st.info("Önce Arama sayfasında 'İndeksi oluştur'a bas.")
    st.stop()

store = VectorStore.load(INDEX_DIR)
settings = Settings()
embedder = OllamaEmbedder(store.model, host=settings.ollama_host, timeout_s=settings.timeout_s)
st.caption(f"İndeks: {len(store.chunks)} parça · embedding modeli: {store.model}")

with st.form("agent"):
    question = st.text_input("Soru", value=EXAMPLES[0])
    submitted = st.form_submit_button("Sor", type="primary")
st.caption("Bunları da dene: " + " · ".join(EXAMPLES[1:]))

if submitted and question.strip():
    try:
        with st.spinner("Agent çalışıyor..."):
            client = get_client(provider, model)
            st.session_state.agent_run = run_agent(
                question, store, embedder, client, k=k, max_steps=max_steps
            )
    except LLMError as exc:
        # ToolLoopError da bir LLMError: agent adım sınırına takıldıysa burada görünür
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

if "agent_run" not in st.session_state:
    st.stop()

run = st.session_state.agent_run
st.subheader("Cevap")
st.markdown(run.answer)
if run.unknown_citations:
    st.warning(f"Model, hiçbir aramada bulunmayan kaynak gösterdi: {run.unknown_citations}")
elif not run.searches:
    st.warning("Agent hiç arama yapmadan cevap verdi.")

cols = st.columns(5)
cols[0].metric("Model çağrısı", run.model_calls)
cols[1].metric("Arama", len(run.searches))
cols[2].metric("Girdi token", sum(r.usage.input_tokens for r in run.responses))
cols[3].metric("Maliyet", f"${run.cost_usd:.6f}")
cols[4].metric("Toplam süre", f"{run.latency_s:.2f} sn")
st.caption(f"Model: {run.responses[0].provider} / {run.responses[0].model}")

st.subheader("Adım adım")
cited = set(run.citations)
for i, step in enumerate(run.searches, start=1):
    with st.container(border=True):
        st.markdown(f"**🔎 Arama {i}:** «{step.query}»")
        st.caption(f"{len(step.hits)} parça · {step.latency_s:.2f} sn")
        for hit in step.hits:
            used = "✅" if (hit.chunk.source, hit.chunk.index) in cited else "⬜"
            label = f"{used} {hit.chunk.source} #{hit.chunk.index} · skor {hit.score:.3f}"
            with st.expander(label):
                st.text(hit.chunk.text)
with st.container(border=True):
    st.markdown("**✍️ Cevap yazıldı**")
    st.caption("✅ = bu parça cevapta kaynak olarak gösterildi")

with st.expander("Modelle yapılan konuşmanın tamamı"):
    st.code(f"[system]\n{AGENT_SYSTEM}", language="text", wrap_lines=True)
    for m in run.messages:
        if m.role == "assistant" and m.tool_calls:
            calls = ", ".join(f"{c.name}({c.arguments})" for c in m.tool_calls)
            st.code(
                f"[assistant → araç]\n{m.content}\n{calls}".strip(),
                language="text",
                wrap_lines=True,
            )
        else:
            st.code(f"[{m.role}]\n{m.content}", language="text", wrap_lines=True)
