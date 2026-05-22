import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import joblib
import os
import json
import numpy as np
from src.preprocess import sql_to_graph, extract_graph_features

# Sayfa Yapılandırması
st.set_page_config(
    page_title="🛡️ AI-SQLi Detektör | SQL Injection Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #f0f2f6; }
    .stTextArea textarea {
        background-color: #161b22 !important;
        color: #00ff41 !important;
        font-family: 'Fira Code', monospace !important;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-bottom: 10px;
    }
    .status-safe { color: #00ff66; background-color: rgba(0, 255, 102, 0.1); border: 1px solid #00ff66; border-radius: 8px; padding: 8px; font-weight: bold; }
    .status-danger { color: #ff3333; background-color: rgba(255, 51, 51, 0.1); border: 1px solid #ff3333; border-radius: 8px; padding: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {"tfidf": None, "lr_text": None, "rf_graph": None, "if_anomaly": None, "comparison": None}
    if os.path.exists('models/tfidf_vectorizer.pkl') and os.path.exists('models/lr_text_model.pkl'):
        models["tfidf"] = joblib.load('models/tfidf_vectorizer.pkl')
        models["lr_text"] = joblib.load('models/lr_text_model.pkl')
    if os.path.exists('models/rf_graph_model.pkl'):
        models["rf_graph"] = joblib.load('models/rf_graph_model.pkl')
    if os.path.exists('models/if_anomaly_model.pkl'):
        models["if_anomaly"] = joblib.load('models/if_anomaly_model.pkl')
    if os.path.exists('models/model_comparison.json'):
        with open('models/model_comparison.json', 'r', encoding='utf-8') as f:
            models["comparison"] = json.load(f)
    return models

models = load_models()

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=70)
    st.title("🛡️ AI-SQLi Detektör")
    st.markdown("---")
    st.subheader("📊 Performans Kıyaslaması")
    if models["comparison"]:
        metrics_data = []
        for key, m in models["comparison"].items():
            metrics_data.append({"Model": m["name"].split(" (")[0], "Accuracy": f"{m['accuracy']:.2%}", "F1-Score": f"{m['f1']:.2%}"})
        st.table(pd.DataFrame(metrics_data))

# MAIN
st.title("🛡️ SQL Injection Çoklu Yapay Zeka Karşılaştırma Paneli")
tab1, tab2 = st.tabs(["🔍 SQLi Detektör", "📖 Proje Metodolojisi & Bilgi"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 SQL Sorgu Analiz Sahası")
        uploaded_file = st.file_uploader("📂 SQL/Metin dosyası yükleyin:", type=["sql", "txt"])
        initial_value = uploaded_file.read().decode("utf-8") if uploaded_file else ""
        query_input = st.text_area("SQL Sorgusunu Girin:", value=initial_value, height=150)
        analyze_clicked = st.button("🔍 Sorguyu Analiz Et")

    with col2:
        st.subheader("🌳 Sorgu Graf Yapısı (AST)")
        if query_input:
            G = sql_to_graph(query_input)
            if G.number_of_nodes() > 0:
                fig, ax = plt.subplots(figsize=(10, 7))
                pos = nx.spring_layout(G, k=0.3)
                nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), 
                        node_color='#1f6feb', node_size=1000, font_size=6, font_color='white', ax=ax)
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                st.pyplot(fig)
            else: st.info("Graf oluşturulamadı.")
        else: st.info("Sorgu girildiğinde graf burada görünecektir.")

    if query_input and (analyze_clicked or query_input != ""):
        st.markdown("---")
        st.subheader("🧠 Yapay Zeka Tahmin Sonuçları")
        features = extract_graph_features(query_input)
        
        # Tahminler
        res = {"LR": "N/A", "RF": "N/A", "IF": "N/A"}
        if models["tfidf"] and models["lr_text"]:
            res["LR"] = "⚠️ SALDIRI" if models["lr_text"].predict(models["tfidf"].transform([query_input]))[0] == 1 else "✅ GÜVENLİ"
        if models["rf_graph"]:
            res["RF"] = "⚠️ SALDIRI" if models["rf_graph"].predict(pd.DataFrame([features]))[0] == 1 else "✅ GÜVENLİ"
        if models["if_anomaly"]:
            res["IF"] = "⚠️ SALDIRI" if models["if_anomaly"].predict(pd.DataFrame([features]))[0] == -1 else "✅ GÜVENLİ"

        c1, c2, c3 = st.columns(3)
        for c, (m, r) in zip([c1, c2, c3], [("Model A: Metin ML", res["LR"]), ("Model B: Graf ML", res["RF"]), ("Model C: Anomali", res["IF"])]):
            status_class = "status-danger" if "SALDIRI" in r else "status-safe"
            c.markdown(f'<div class="metric-card"><b>{m}</b><div style="margin-top:10px;" class="{status_class}">{r}</div></div>', unsafe_allow_html=True)

with tab2:
    st.header("📖 Proje Metodolojisi ve Teknik Detaylar")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("🌳 Abstract Syntax Tree (AST) Nedir?")
        st.write("""
        SQL sorguları sadece düz metin değildir; her sorgunun mantıksal bir hiyerarşisi vardır. 
        Bu projede, sorguları **AST graf yapılarına** dönüştürerek analiz ediyoruz. 
        Saldırganlar 'admin' yerine 'a' + 'dmin' yazarak metin filtrelerini aşabilirler, 
        ancak sorgunun **mantıksal graf yapısı** (mantıksal 'OR' veya '1=1' düğümleri) değişmez.
        """)
        st.info("💡 **Graf Özelliği:** Düğüm sayısı, derinlik ve merkeziyet (centrality) değerleri, sorgunun 'anormal' olup olmadığını belirleyen temel metriklerdir.")

    with col_m2:
        st.subheader("🤖 Hibrit Model Yaklaşımı")
        st.write("""
        Sistem 3 farklı katmanda koruma sağlar:
        1. **Model A (Metin ML):** Bilinen saldırı kelimelerini (UNION, SELECT, DROP) kelime frekansıyla yakalar.
        2. **Model B (Graf ML):** Sorgunun yapısal graf özelliklerini (düğüm/kenar ilişkisi) sınıflandırır.
        3. **Model C (Anomali):** Daha önce hiç görülmemiş (Zero-day) saldırıları, normal sorgu yapısından sapma miktarına göre yakalar.
        """)

    st.markdown("---")
    st.subheader("🚀 Bulut Bilişim ve Yaygınlaştırma")
    st.write("""
    Bu proje, bulut tabanlı bir mikroservis mimarisine uygun şekilde tasarlanmıştır:
    - **Konteynerizasyon:** `Dockerfile` ile her ortamda (Azure, AWS vb.) aynı performansta çalışabilir.
    - **API Uyumluluğu:** Model çıktıları JSON formatında olduğu için bir güvenlik duvarı (WAF) servisi olarak API üzerinden çağrılabilir.
    """)

st.markdown("---")
st.caption("🛡️ Bulut Bilişim Projesi | Beyzanur KAYA")
