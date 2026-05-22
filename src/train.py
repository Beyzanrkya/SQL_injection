import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os
import sys

# Çıktı kodlamasını UTF-8 olarak ayarla (Windows terminal emoji/karakter uyumluluğu için)
sys.stdout.reconfigure(encoding='utf-8')

from src.preprocess import SQLQueryPreprocessor

class SQLModelTrainer:
    """
    SQL Injection tespiti için 3 farklı yapay zeka modelini (Metin ML, Graf ML, Anomali Tespiti)
    eğitmek, değerlendirmek ve kaydetmek için kullanılan model yönetici sınıfı.
    """
    def __init__(self, data_path: str = 'data/SQLiV3.csv', sample_size: int = 5000):
        self.data_path = data_path
        self.sample_size = sample_size
        self.preprocessor = SQLQueryPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
        # Model tanımlamaları
        self.text_model = LogisticRegression(max_iter=1000)
        self.graph_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        
    def load_and_preprocess_data(self) -> tuple:
        """
        Veri setini yükler, dengeler (stratified) ve hem metin hem graf özelliklerini çıkarır.
        
        Returns:
            tuple: Model eğitimi için hazır hale getirilmiş veri çerçeveleri (X_text, X_graph, y).
        """
        print("1. Veri seti yukleniyor...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Veri seti bulunamadi: {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        print("Veri setindeki mevcut sütunlar:", df.columns.tolist())
        
        # --- DÜZELTME BURADA ---
        # Veri setini temizle ('Query' yerine 'Sentence' yazıldı)
        df = df.dropna(subset=['Sentence', 'Label'])
        
        # Dengeli orneklem al (Her siniftan esit sayida)
        safe_df = df[df['Label'] == 0]
        malicious_df = df[df['Label'] == 1]
        
        half_sample = self.sample_size // 2
        safe_sample = safe_df.sample(n=min(half_sample, len(safe_df)), random_state=42)
        malicious_sample = malicious_df.sample(n=min(half_sample, len(malicious_df)), random_state=42)
        
        balanced_df = pd.concat([safe_sample, malicious_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Dengeli veri boyutu: {len(balanced_df)} (Guvenli: {len(safe_sample)}, Saldiri: {len(malicious_sample)})")
        
        # Graf ozelliklerini cikar
        print("2. Graf ozellikleri cikariliyor (AST Metrikleri)...")
        graph_features_list = []
        
        # --- DÜZELTME BURADA ---
        # Veri setinden sorguları çekerken 'Query' yerine 'Sentence' kullanıldı
        for idx, query in enumerate(balanced_df['Sentence']):
            feats = self.preprocessor.extract_graph_features(query)
            graph_features_list.append(feats)
            if (idx + 1) % 1000 == 0:
                print(f"  {idx + 1} sorgu islendi...")
                
        graph_df = pd.DataFrame(graph_features_list)
        
        # --- DÜZELTME BURADA ---
        # Geri dönüş değerinde de 'Query' yerine 'Sentence' kullanıldı
        return balanced_df['Sentence'], graph_df, balanced_df['Label']

    def train_and_evaluate(self):
        """
        3 modeli de eğitir, performans metriklerini hesaplar, 
        modelleri kaydeder ve performans raporunu JSON olarak yazar.
        """
        # Veriyi yukle ve hazirla
        queries, X_graph, y = self.load_and_preprocess_data()
        
        # Egitim ve test bolmesi
        indices = np.arange(len(y))
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(indices, y, test_size=0.2, random_state=42, stratify=y)
        
        print("\n3. Model A Eğitiliyor (Metin ML: TF-IDF + Logistic Regression)...")
        # Metin TF-IDF donusumu
        train_queries = queries.iloc[X_train_idx]
        test_queries = queries.iloc[X_test_idx]
        
        X_train_text = self.vectorizer.fit_transform(train_queries)
        X_test_text = self.vectorizer.transform(test_queries)
        
        self.text_model.fit(X_train_text, y_train)
        y_pred_text = self.text_model.predict(X_test_text)
        
        print("4. Model B Eğitiliyor (Graf ML: AST + Random Forest)...")
        X_train_graph = X_graph.iloc[X_train_idx]
        X_test_graph = X_graph.iloc[X_test_idx]
        
        self.graph_model.fit(X_train_graph, y_train)
        y_pred_graph = self.graph_model.predict(X_test_graph)
        
        print("5. Model C Eğitiliyor (Anomali Tespiti: AST + Isolation Forest)...")
        # Isolation Forest sadece normal sorgularla (Label == 0) egitilir
        normal_train_graph = X_train_graph[y_train.reset_index(drop=True) == 0]
        self.anomaly_model.fit(normal_train_graph)
        
        # Test seti tahmini (-1 anomali/saldiri, 1 normal/guvenli)
        raw_pred_anomaly = self.anomaly_model.predict(X_test_graph)
        # Biyerlesik formata cevir (0: Guvenli, 1: Saldiri)
        y_pred_anomaly = np.where(raw_pred_anomaly == -1, 1, 0)
        
        # Metriklerin Hesaplanması (Hocanın PDF'deki Accuracy, Precision, Recall, F1 kriterleri)
        metrics = {
            "Text_ML": {
                "name": "Metin ML (TF-IDF + Logistic Regression)",
                **self._calculate_metrics(y_test, y_pred_text)
            },
            "Graph_ML": {
                "name": "Graf ML (AST + Random Forest)",
                **self._calculate_metrics(y_test, y_pred_graph)
            },
            "Anomaly_Detection": {
                "name": "Anomali Tespiti (AST + Isolation Forest)",
                **self._calculate_metrics(y_test, y_pred_anomaly)
            }
        }
        
        print("\n=== Model Performans Sonucları ===")
        for key, m_dict in metrics.items():
            print(f"\n{m_dict['name']}:")
            print(f"  Accuracy:  {m_dict['accuracy']:.4f}")
            print(f"  Precision: {m_dict['precision']:.4f}")
            print(f"  Recall:    {m_dict['recall']:.4f}")
            print(f"  F1-Score:  {m_dict['f1']:.4f}")
            
        # Kaydetme islemleri
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(self.text_model, 'models/lr_text_model.pkl')
        joblib.dump(self.graph_model, 'models/rf_graph_model.pkl')
        # Geriye donuk uyumluluk icin eski dosya ismini de kaydet
        joblib.dump(self.graph_model, 'models/sql_model.pkl')
        joblib.dump(self.anomaly_model, 'models/if_anomaly_model.pkl')
        
        with open('models/model_comparison.json', 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print("\nModeller ve karsılastırma metrikleri 'models/' dizinine basarıyla kaydedildi.")

    def _calculate_metrics(self, y_true, y_pred) -> dict:
        """Yardımcı fonksiyon: Sklearn metriklerini hesaplar."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0))
        }

if __name__ == "__main__":
    trainer = SQLModelTrainer()
    trainer.train_and_evaluate()