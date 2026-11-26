"""Parçalama sayfası: bir dokümanın cümle sınırında nasıl parçalara bölündüğünü gör."""

import streamlit as st

from llm_toolkit.chunking import chunk_text

SAMPLE = """Büyük dil modelleri eğitildikleri tarihe kadar gördükleri metinleri bilir. \
Şirketinizin iç belgeleri, dün yayımlanan bir rapor ya da size özel notlar bu bilginin \
içinde yoktur. Model böyle bir soruyla karşılaşınca ya bilmediğini söyler ya da kulağa \
doğru gelen ama yanlış bir cevap uydurur.

RAG (Retrieval-Augmented Generation) bu sorunu modele soruyla birlikte ilgili belgeleri \
vererek çözer. Sistem önce belgeler arasında arama yapar. Bulduğu parçaları soruya ekler. \
Model cevabını bu parçalara dayanarak yazar ve hangi belgeden yararlandığını gösterebilir.

Arama yapabilmek için belgeler önceden hazırlanır. Uzun bir belge olduğu gibi modele \
verilmez, çünkü hem pahalıdır hem de ilgisiz kısımlar cevabı bulandırır. Bunun yerine \
belge birkaç yüz karakterlik parçalara bölünür. Bu işleme parçalama (chunking) denir.

Parça boyu önemli bir ayardır. Çok küçük parçalar bağlamı kaybeder: bir cümle tek başına \
anlamsız kalabilir. Çok büyük parçalar ise ilgisiz bilgiyi de beraberinde getirir ve \
maliyeti artırır. İki parçanın sınırına denk gelen bilgi kaybolmasın diye parçalar \
birbiriyle biraz örtüşür. Doğru ayar belgeye ve sorulara göre değişir; bunu ileride \
değerlendirme (eval) ile ölçeceğiz.
"""

st.set_page_config(page_title="AI Lab", page_icon="🧪", layout="wide")
st.title("✂️ Parçalama (chunking)")

with st.sidebar:
    st.header("Ayarlar")
    max_chars = st.slider("Parça boyu (karakter)", 200, 2000, 400, step=50)
    overlap_chars = st.slider("Örtüşme (karakter)", 0, 500, 80, step=10)

uploaded = st.file_uploader("Doküman yükle (.txt veya .md)", type=["txt", "md"])
if uploaded:
    source, text = uploaded.name, uploaded.getvalue().decode("utf-8", errors="replace")
else:
    source = "ornek.txt"
    text = st.text_area("Ya da metni buraya yapıştır", SAMPLE, height=200)

if overlap_chars >= max_chars:
    st.warning("Örtüşme, parça boyundan küçük olmalı.")
    st.stop()

chunks = chunk_text(text, source, max_chars=max_chars, overlap_chars=overlap_chars)
if not chunks:
    st.info("Metin boş.")
    st.stop()

lengths = [len(c.text) for c in chunks]
clean_length = len(" ".join(text.split()))  # Boşluklar sadeleştirilmiş doküman uzunluğu
# Örtüşme yüzünden aynı metin birden fazla parçada yer alır; bu, saklama ve embedding maliyetidir
extra = sum(lengths) + len(chunks) - 1 - clean_length

col_doc, col_count, col_avg, col_extra = st.columns(4)
col_doc.metric("Doküman (karakter)", clean_length)
col_count.metric("Parça sayısı", len(chunks))
col_avg.metric("Ortalama parça (karakter)", sum(lengths) // len(chunks))
col_extra.metric("Örtüşme payı", f"%{100 * extra / clean_length:.0f}")

st.bar_chart({"karakter": lengths}, x_label="Parça no", y_label="Karakter", height=200)

for chunk in chunks:
    previous = chunks[chunk.index - 1].text if chunk.index > 0 else ""
    # Bu parçanın başındaki kaç karakter önceki parçanın sonunda da var?
    shared = next(
        (
            k
            for k in range(len(chunk.text), 0, -1)
            if previous.endswith(chunk.text[:k]) and chunk.text[k : k + 1] in ("", " ")
        ),
        0,
    )
    with st.container(border=True):
        note = f" · ilk {shared} karakter önceki parçayla aynı" if shared else ""
        st.caption(f"#{chunk.index} · {source} · {len(chunk.text)} karakter{note}")
        st.text(chunk.text)
