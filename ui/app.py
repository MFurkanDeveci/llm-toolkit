"""AI Lab arayüzü. Çalıştırma (proje kökünden): uv run streamlit run ui/app.py

Bu dosya sadece sayfa listesini tutar. Yeni sayfa: ui/pages altına dosya ekle, buraya bir satır yaz.
"""

import streamlit as st

pages = {
    "Modül 3 · RAG": [
        st.Page("pages/0_Parcalama.py", title="Parçalama", icon="✂️"),
        st.Page("pages/1_Embedding.py", title="Embedding", icon="🧭"),
        st.Page("pages/2_arama.py", title="Arama", icon="🔎"),
        st.Page("pages/3_Cevap.py", title="Cevap", icon="💬"),
        st.Page("pages/4_Hibrit.py", title="Hibrit Arama", icon="⚖️"),
    ],
    "Modül 4 · Değerlendirme": [
        st.Page("pages/5_Degerlendirme.py", title="Değerlendirme", icon="📏"),
        st.Page("pages/6_Cevap_olcumu.py", title="Cevap Ölçümü", icon="🧑‍⚖️"),
        st.Page("pages/7_Rerank.py", title="Reranking", icon="🔢"),
        st.Page("pages/8_Karsilastirma.py", title="Karşılaştırma", icon="📊"),
    ],
}
st.navigation(pages).run()
