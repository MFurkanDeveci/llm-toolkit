"""Embedding sayfası: metinleri vektöre çevir, hangilerinin anlamca yakın olduğunu gör."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from llm_toolkit.config import Settings
from llm_toolkit.embeddings import DEFAULT_MODEL, OllamaEmbedder, cosine_similarity
from llm_toolkit.errors import LLMError

# Üç konu, ikişer cümle. Doğru çalışıyorsa ısı haritasında üç koyu kare görünür.
SAMPLE = """Model eğitim verisini ezberleyince yeni veride kötü sonuç verir.
Aşırı öğrenmeyi önlemek için düzenlileştirme ve çapraz doğrulama kullanılır.
Makarna suyuna tuz atıp kaynayınca makarnayı ekleyin.
Sosu hazırlamak için domatesleri zeytinyağında kısık ateşte pişirin.
Takım son dakikada attığı golle maçı kazandı.
Hakem penaltı kararını video incelemesinden sonra verdi."""

st.set_page_config(page_title="AI Lab · Embedding", page_icon="🧭", layout="wide")
st.title("🧭 Embedding ve benzerlik")

with st.sidebar:
    st.header("Ayarlar")
    model = st.text_input("Embedding modeli (Ollama)", value=DEFAULT_MODEL)

with st.form("embed"):
    raw_text = st.text_area("Her satıra bir metin", SAMPLE, height=180)
    submitted = st.form_submit_button("Vektörlere çevir", type="primary")

if submitted:
    texts = [line.strip() for line in raw_text.splitlines() if line.strip()]
    settings = Settings()  # Ollama adresi ve zaman aşımı .env'den
    embedder = OllamaEmbedder(model, host=settings.ollama_host, timeout_s=settings.timeout_s)
    try:
        with st.spinner("Vektörler hesaplanıyor... (ilk çağrıda model belleğe yüklenir)"):
            result = embedder.embed(texts)
    except LLMError as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.caption(f"Model indirilmemişse terminalde: ollama pull {model}")
        st.stop()
    # Sonucu sakla: sayfa başka bir sebeple yeniden çalışınca kaybolmasın
    st.session_state.embedding = (texts, result)

if "embedding" not in st.session_state:
    st.info("Metinleri girip 'Vektörlere çevir'e bas.")
    st.stop()

texts, result = st.session_state.embedding
col_model, col_dim, col_tokens, col_time, col_cost = st.columns(5)
col_model.metric("Model", result.model)
col_dim.metric("Vektör boyutu", result.vectors.shape[1])
col_tokens.metric("Girdi token", result.input_tokens)
col_time.metric("Süre", f"{result.latency_s:.2f} sn")
col_cost.metric("Maliyet", f"${result.cost_usd:.6f}")

sim = cosine_similarity(result.vectors, result.vectors)
labels = [f"{i + 1}. {t[:28]}…" if len(t) > 28 else f"{i + 1}. {t}" for i, t in enumerate(texts)]
cells = pd.DataFrame(
    [
        {"satır": labels[i], "sütun": labels[j], "benzerlik": float(sim[i, j])}
        for i in range(len(texts))
        for j in range(len(texts))
    ]
)
base = alt.Chart(cells).encode(
    x=alt.X("sütun:N", sort=labels, title=None, axis=alt.Axis(labelAngle=-40)),
    y=alt.Y("satır:N", sort=labels, title=None),
)
heatmap = base.mark_rect().encode(
    color=alt.Color("benzerlik:Q", scale=alt.Scale(scheme="blues"), title="Benzerlik")
)
numbers = base.mark_text(fontSize=11).encode(
    text=alt.Text("benzerlik:Q", format=".2f"),
    color=alt.condition("datum.benzerlik > 0.7", alt.value("white"), alt.value("black")),
)
st.subheader("Kosinüs benzerliği")
st.altair_chart(heatmap + numbers, height=420)

if len(texts) > 1:
    # Köşegen her metnin kendisiyle benzerliği (hep 1); en benzer/en farklı çifti ararken atla
    off_diagonal = np.where(np.eye(len(texts), dtype=bool), np.nan, sim)
    i, j = np.unravel_index(np.nanargmax(off_diagonal), sim.shape)
    k, m = np.unravel_index(np.nanargmin(off_diagonal), sim.shape)
    st.success(f"En benzer çift ({sim[i, j]:.2f}): **{texts[i]}** ↔ **{texts[j]}**")
    st.warning(f"En farklı çift ({sim[k, m]:.2f}): **{texts[k]}** ↔ **{texts[m]}**")

with st.expander("Bir vektör neye benziyor?"):
    first = result.vectors[0]
    st.write(f"1. metnin vektörü {first.shape[0]} sayıdan oluşuyor. İlk 10 tanesi:")
    st.code(np.array2string(first[:10], precision=3, separator=", "))
