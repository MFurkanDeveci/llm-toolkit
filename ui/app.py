"""AI Lab arayüzü. Çalıştırma (proje kökünden): uv run streamlit run ui/app.py

Bu dosya sadece sayfa listesini tutar. Yeni sayfa: ui/views altına dosya ekle, buraya bir satır yaz.
Not: Klasörün adı bilerek 'pages' değil; Streamlit o adı otomatik menü için ayırıyor.
"""

import streamlit as st

pages = {
    "Modül 3 · RAG": [
        st.Page("views/0_Parcalama.py", title="Parçalama", icon="✂️"),
        st.Page("views/1_Embedding.py", title="Embedding", icon="🧭"),
        st.Page("views/2_arama.py", title="Arama", icon="🔎"),
        st.Page("views/3_Cevap.py", title="Cevap", icon="💬"),
        st.Page("views/4_Hibrit.py", title="Hibrit Arama", icon="⚖️"),
    ],
    "Modül 4 · Değerlendirme": [
        st.Page("views/5_Degerlendirme.py", title="Değerlendirme", icon="📏"),
        st.Page("views/6_Cevap_olcumu.py", title="Cevap Ölçümü", icon="🧑‍⚖️"),
        st.Page("views/7_Rerank.py", title="Reranking", icon="🔢"),
        st.Page("views/8_Karsilastirma.py", title="Karşılaştırma", icon="📊"),
    ],
    "Modül 5 · Agent": [
        st.Page("views/9_Agent.py", title="Agent", icon="🤖"),
        st.Page("views/10_Sohbet.py", title="Sohbet", icon="🗨️"),
    ],
}
st.navigation(pages).run()
