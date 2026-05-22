import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import joblib
import os
import json
import numpy as np
from src.preprocess import sql_to_graph, extract_graph_features

# Sayfa Yapılandırması (Premium Geniş Ekran)
st.set_page_config(
    page_title="🛡️ AI-SQLi Detektör | SQL Injection Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS ile Premium Karanlık/Cam Tasarım (Glassmorphism & High-end UI)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #161b22;
    }
    .stTextArea textarea {
        background-color: #161b22 !important;
        color: #00ff41 !important;
        font-family: 'Fira Code', 'Courier New', Courier, monospace !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        border-color: rgba(255, 255, 255, 0.15);
    }
    .status-safe {
        color: #00ff66;
        background-color: rgba(0, 255, 102, 0.1);
        border: 1px solid #00ff66;
        border-radius: 8px;
        padding: 8px;
        font-weight: bold;
        font-size: 16px;
    }
    .status-danger {
        color: #ff3333;
        background-color: rgba(255, 51, 51, 0.1);
        border: 1px solid #ff3333;
        border-radius: 8px;
        padding: 8px;
        font-weight: bold;
        font-size: 16px;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1f6feb, #0972d3) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0972d3, #1f6feb) !important;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.4) !important;
        transform: scale(1.02) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# MODELLERİN YÜKLENMESİ
# ----------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "tfidf": None,
        "lr_text": None,
        "rf_graph": None,
        "if_anomaly": None,
        "comparison": None
    }
    
    # Model A: Text ML
    if os.path.exists('models/tfidf_vectorizer.pkl') and os.path.exists('models/lr_text_model.pkl'):
        models["tfidf"] = joblib.load('models/tfidf_vectorizer.pkl')
        models["lr_text"] = joblib.load('models/lr_text_model.pkl')
        
    # Model B: Graph ML
    if os.path.exists('models/rf_graph_model.pkl'):
        models["rf_graph"] = joblib.load('models/rf_graph_model.pkl')
    elif os.path.exists('models/sql_model.pkl'):
        models["rf_graph"] = joblib.load('models/sql_model.pkl')
        
    # Model C: Anomaly Detection
    if os.path.exists('models/if_anomaly_model.pkl'):
        models["if_anomaly"] = joblib.load('models/if_anomaly_model.pkl')
        
    # Kıyaslama Raporu
    if os.path.exists('models/model_comparison.json'):
        with open('models/model_comparison.json', 'r', encoding='utf-8') as f:
            models["comparison"] = json.load(f)
            
    return models

models = load_models()

# ----------------------------------------------------
# SOL MENÜ (SIDEBAR): Performans & Kıyaslama Tablosu
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=70)
    st.title("🛡️ AI-SQLi Detektör")
    st.markdown("Bulut Bilişim ve Yapay Zeka tabanlı SQL Injection saldırı tespit arayüzü.")
    st.markdown("---")
    
    st.subheader("📊 Modellerin Performans Kıyaslaması")
    if models["comparison"]:
        # Metrikleri dataframe yapalım
        metrics_data = []
        for key, m in models["comparison"].items():
            metrics_data.append({
                "Model": m["name"].split(" (")[0],
                "Accuracy": f"{m['accuracy']:.2%}",
                "F1-Score": f"{m['f1']:.2%}"
            })
        st.table(pd.DataFrame(metrics_data))
        st.info("💡 **Yorum:** Metin tabanlı ML yüksek başarı gösterse de, statik kalıpların dışındaki bilinmeyen saldırıları (Sıfırıncı Gün) yakalamada **Graf tabanlı Anomali Tespiti** en güvenilir yaklaşımdır.")
    else:
        st.warning("⚠️ Modeller henüz eğitilmemiş. Lütfen `src/train.py` dosyasını çalıştırın.")

# ----------------------------------------------------
# ANA EKRAN DÜZENİ
# ----------------------------------------------------
st.title("🛡️ SQL Injection Çoklu Yapay Zeka Karşılaştırma Paneli")
st.markdown("Sorguların hem **kelime dizilimlerini (TF-IDF)** hem de yapısal **Abstract Syntax Tree (AST)** graf özelliklerini analiz eden hibrit tespit arayüzü.")
st.markdown("---")

col1, col2 = st.columns([11, 9])

with col1:
    st.subheader("📝 SQL Sorgu Analiz Sahası")
    
    # Dosya yükleme desteği (Hocanın PDF'deki fonksiyonel dosya yükleme gereksinimi)
    uploaded_file = st.file_uploader("📂 Veya bir SQL/Metin dosyası yükleyin (.sql / .txt):", type=["sql", "txt"])
    
    initial_value = ""
    if uploaded_file is not None:
        try:
            initial_value = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")
            
    query_input = st.text_area(
        "SQL Sorgusunu Buraya Girin:",
        value=initial_value,
        height=150,
        placeholder="Örn: SELECT * FROM users WHERE username = 'admin' AND password = '1' OR '1'='1'"
    )
    
    analyze_clicked = st.button("🔍 Sorguyu Analiz Et")
    
    if query_input and (analyze_clicked or query_input != ""):
        # Graf özelliklerini çıkaralım
        features = extract_graph_features(query_input)
        
        st.markdown("### 📊 Sorgu Graf Özellikleri (AST Metrics)")
        df_features = pd.DataFrame([features])
        # Tabloyu daha şık gösterelim
        st.dataframe(df_features.style.format(precision=4), use_container_width=True)
        
        # ----------------------------------------------------
        # TAHMİNLERİN ALINMASI
        # ----------------------------------------------------
        pred_lr_label = "Model Yüklenemedi"
        pred_rf_label = "Model Yüklenemedi"
        pred_if_label = "Model Yüklenemedi"
        
        # 1. Model A: Metin ML
        if models["tfidf"] and models["lr_text"]:
            tfidf_vec = models["tfidf"].transform([query_input])
            pred_lr = models["lr_text"].predict(tfidf_vec)[0]
            pred_lr_label = "⚠️ SALDIRI (SQLi)" if pred_lr == 1 else "✅ GÜVENLİ"
            
        # 2. Model B: Graf ML (Supervised)
        if models["rf_graph"]:
            X_graph_in = pd.DataFrame([features])
            pred_rf = models["rf_graph"].predict(X_graph_in)[0]
            pred_rf_label = "⚠️ SALDIRI (SQLi)" if pred_rf == 1 else "✅ GÜVENLİ"
            
        # 3. Model C: Anomali Tespiti (Unsupervised Isolation Forest)
        if models["if_anomaly"]:
            X_graph_in = pd.DataFrame([features])
            raw_pred_if = models["if_anomaly"].predict(X_graph_in)[0]
            # -1 anomali (saldırı), 1 normal (güvenli)
            pred_if_label = "⚠️ SALDIRI (SQLi)" if raw_pred_if == -1 else "✅ GÜVENLİ"

        # ----------------------------------------------------
        # TAHMİN KARTLARININ GÖSTERİLMESİ
        # ----------------------------------------------------
        st.markdown("### 🧠 Yapay Zeka Karşılaştırmalı Tahmin Sonuçları")
        
        c_a, c_b, c_c = st.columns(3)
        
        with c_a:
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size:13px; color:#8b949e; font-weight:bold;">Model A: Metin tabanlı ML</span>
                <div style="font-size:18px; margin-top:10px; font-weight:bold;">Logistic Regression</div>
                <div style="margin-top:12px;" class="{'status-danger' if 'SALDIRI' in pred_lr_label else 'status-safe'}">
                    {pred_lr_label}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_b:
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size:13px; color:#8b949e; font-weight:bold;">Model B: Yapısal Graf ML</span>
                <div style="font-size:18px; margin-top:10px; font-weight:bold;">Random Forest</div>
                <div style="margin-top:12px;" class="{'status-danger' if 'SALDIRI' in pred_rf_label else 'status-safe'}">
                    {pred_rf_label}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_c:
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size:13px; color:#8b949e; font-weight:bold;">Model C: Anomali Tespiti</span>
                <div style="font-size:18px; margin-top:10px; font-weight:bold;">Isolation Forest</div>
                <div style="margin-top:12px;" class="{'status-danger' if 'SALDIRI' in pred_if_label else 'status-safe'}">
                    {pred_if_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.subheader("🌳 Sorgunun Graf Yapısı (AST Visualization)")
    if query_input:
        G = sql_to_graph(query_input)
        if G.number_of_nodes() > 0:
            fig, ax = plt.subplots(figsize=(10, 8.5))
            
            # Graf tasarımı: Spring Layout
            pos = nx.spring_layout(G, k=0.3, iterations=50)
            labels = nx.get_node_attributes(G, 'label')
            
            # Nodes
            nx.draw_networkx_nodes(
                G, pos, 
                node_color='#1f6feb', 
                node_size=1600, 
                alpha=0.9, 
                ax=ax
            )
            
            # Edges
            nx.draw_networkx_edges(
                G, pos, 
                edge_color='#30363d', 
                width=1.5, 
                arrowstyle='->', 
                arrowsize=12, 
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )
            
            # Labels
            nx.draw_networkx_labels(
                G, pos, 
                labels=labels, 
                font_size=7, 
                font_color='#ffffff', 
                font_family='sans-serif', 
                font_weight='bold',
                ax=ax
            )
            
            # Arka plan ve kenarlık temizleme
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            plt.tight_layout()
            ax.axis('off')
            
            st.pyplot(fig)
        else:
            st.info("Graf oluşturmak için geçerli bir SQL sorgusu girin.")
    else:
        st.info("Sorgu girildiğinde, o sorgunun Abstract Syntax Tree (AST) grafiği burada canlı olarak görüntülenecektir.")

st.markdown("---")
st.caption("🛡️ Bulut Bilişim ve Makine Öğrenmesi Projesi | SQL Injection Detection System using AST & Anomaly Modeling")
