"""AI Lab arayüzü. Çalıştırma (proje kökünden): uv run streamlit run ui/app.py

Bu dosya sadece sayfa listesini tutar. Yeni sayfa: ui/pages altına dosya ekle, buraya bir satır yaz.
"""

import streamlit as st

pages = {
    "Modül 3 · RAG": [
        st.Page("pages/0_Parcalama.py", title="Parçalama", icon="✂️"),
        st.Page("pages/1_Embedding.py", title="Embedding", icon="🧭"),
        st.Page("pages/2_arama.py", title="Arama", icon="🔎"),
    ],
}
st.navigation(pages).run()
