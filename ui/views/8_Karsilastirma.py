"""Karşılaştırma sayfası: kayıtlı cevap ölçümlerini kalite, maliyet ve süreyle karşılaştır.

Sabit RAG ve agent çalıştırmaları aynı tabloda ve grafikte; grafikte yöntem şekille ayrılır.
"""

import altair as alt
import pandas as pd
import streamlit as st

from llm_toolkit.runs import RUNS_DIR, compare_runs, load_runs, runs_table

st.set_page_config(page_title="AI Lab · Karşılaştırma", page_icon="📊", layout="wide")
st.title("📊 Çalıştırmaları karşılaştır")

runs = load_runs()
if not runs:
    st.info(f"{RUNS_DIR} klasöründe kayıt yok. Önce Cevap Ölçümü sayfasında bir ölçüm çalıştır.")
    st.stop()

table = pd.DataFrame(runs_table(runs))
best = table.loc[table["kalite"].idxmax()]
cheapest = table.loc[table["soru_başı_cevap_maliyeti"].idxmin()]
col_count, col_best, col_cheap = st.columns(3)
col_count.metric("Kayıtlı çalıştırma", len(runs))
col_best.metric("En yüksek kalite", f"%{100 * best['kalite']:.0f}")
col_best.caption(f"{best['çalıştırma']} · {best['yöntem']} · {best.get('cevap_modeli', '?')}")
col_cheap.metric("En ucuz (soru başı)", f"${cheapest['soru_başı_cevap_maliyeti']:.5f}")
col_cheap.caption(
    f"{cheapest['çalıştırma']} · {cheapest['yöntem']} · {cheapest.get('cevap_modeli', '?')}"
)

st.subheader("Tüm çalıştırmalar")
st.caption(
    "Kalite: yargıç varsa onun 'doğru' oranı, yoksa ifade kontrolü. "
    "⭐ = Pareto sınırında: soru başına daha ucuz ve aynı ya da daha kaliteli başka bir "
    "çalıştırma yok. Yargıç maliyeti ölçümün maliyetidir; kullanıcıya hizmet verirken ödenmez."
)
table["pareto"] = table["pareto"].map({True: "⭐", False: ""})
st.dataframe(
    table,
    hide_index=True,
    column_config={
        "kalite": st.column_config.NumberColumn(format="percent"),
        "ifade_başarısı": st.column_config.NumberColumn("ifade", format="percent"),
        "yargıç_sadık": st.column_config.NumberColumn("sadık", format="percent"),
        "soru_başı_cevap_maliyeti": st.column_config.NumberColumn(
            "soru başı cevap", format="$%.5f"
        ),
        "yargıç_maliyeti": st.column_config.NumberColumn("yargıç (ölçüm)", format="$%.4f"),
        "ort_süre": st.column_config.NumberColumn("ort. süre (sn)", format="%.2f"),
        "ort_arama": st.column_config.NumberColumn("ort. arama", format="%.1f"),
    },
)

st.subheader("Kalite ve maliyet")
chart = table.assign(grup=table["pareto"].map({"⭐": "Pareto sınırında", "": "diğer"}))
points = (
    alt.Chart(chart)
    .mark_point(size=220, opacity=0.85, filled=True)  # mark_circle şekli yok sayar
    .encode(
        x=alt.X(
            "soru_başı_cevap_maliyeti:Q",
            title="Soru başına cevap maliyeti ($)",
            scale=alt.Scale(padding=20),
        ),
        y=alt.Y("kalite:Q", title="Kalite", scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color("grup:N", title=None),
        shape=alt.Shape("yöntem:N", title="Yöntem"),
        tooltip=[
            "çalıştırma",
            "yöntem",
            "cevap_modeli",
            "k",
            alt.Tooltip("kalite:Q", format=".0%"),
            alt.Tooltip("ort_arama:Q", title="ort. arama", format=".1f"),
        ],
    )
)
st.altair_chart(points, height=320)
st.caption("Sol üst köşe en iyisi: ucuz ve kaliteli. Sağ alttaki bir nokta hem pahalı hem kötü.")

if len(runs) >= 2:
    st.subheader("İki çalıştırmayı soru soru karşılaştır")
    rows_by_id = {row["çalıştırma"]: row for row in runs_table(runs)}
    labels = {
        run_id: (
            f"{run_id} · {row['yöntem']} · {row.get('cevap_modeli', '?')} · k={row.get('k', '?')}"
        )
        for run_id, row in rows_by_id.items()
    }
    ids = list(labels)
    col_old, col_new = st.columns(2)
    old_id = col_old.selectbox("Eski", ids, index=len(ids) - 2, format_func=labels.get)
    new_id = col_new.selectbox("Yeni", ids, index=len(ids) - 1, format_func=labels.get)
    by_id = {r.run_id: r for r in runs}
    diff = pd.DataFrame(compare_runs(by_id[old_id], by_id[new_id]))
    fixed, broken = (diff["değişim"] == "düzeldi").sum(), (diff["değişim"] == "bozuldu").sum()
    st.markdown(f"**{fixed}** soru düzeldi, **{broken}** soru bozuldu.")
    old_row, new_row = rows_by_id[old_id], rows_by_id[new_id]

    def ratio(column: str) -> str:
        """Yeni / eski oranı. Eski değer 0 ise (örn. ücretsiz Ollama) oran anlamsız."""
        old_value = old_row[column]
        return f"{new_row[column] / old_value:.1f}x" if old_value else "–"

    col_cost, col_time, col_search = st.columns(3)
    col_cost.metric("Soru başı maliyet (yeni / eski)", ratio("soru_başı_cevap_maliyeti"))
    col_time.metric("Ort. süre (yeni / eski)", ratio("ort_süre"))
    col_search.metric(
        "Ort. arama (eski → yeni)", f"{old_row['ort_arama']:.1f} → {new_row['ort_arama']:.1f}"
    )
    only_changed = st.toggle("Sadece değişen soruları göster", value=True)
    if only_changed:
        diff = diff[diff["değişim"] != ""]
    diff[["önce", "sonra"]] = diff[["önce", "sonra"]].replace({True: "✅", False: "❌"})
    st.dataframe(diff, hide_index=True, height=35 * (len(diff) + 1) + 3)
