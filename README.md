# YAPAY ZEKA VE GRAF TEORİSİ İLE SQL INJECTİON SALDIRI TESPİT SİSTEMİ
Bu proje, bulut bilişim dersifinal projesi için geliştirlmiştir. Amacı, web uygulamalarına yönelik en tehlikeli siber saldırılardan biri olan SQL Injection (SQLi) girişimlerini yapay zeka ve grafik modelleme teknikleriyle tespit etmektir.
 
 ---

 ## Geliştirici Bilgileri
* **Adı Soyadı:** Beyzanur KAYA 
* **Öğrenci No:** 2311012073
* **Ders:** Bulut Bilişim(Cloud Computing)

---

##  Canlı Uygulama Linki (Live Demo)
Projeye herhangi bir kurulum yapmadan doğrudan internet üzerinden erişmek ve test etmek için:
👉 **[https://beyzanur-sql-injection.streamlit.app/](https://beyzanur-sql-injection.streamlit.app/)**

---

## 🎥 Proje Tanıtım Videosu (Presentation Video)
Projenin detaylı anlatımını, kod yapısını ve canlı arayüzün test senaryolarını içeren sunum videosuna buradan erişebilirsiniz:
👉 **[https://www.youtube.com/watch?v=HLTm1E2WNdI](https://www.youtube.com/watch?v=HLTm1E2WNdI)**

---

##  Projenin Amacı ve Önemi
SQL Injection saldırıları web siteleri için oldukça önemlidir.Geleneksel imza tabanlı güvenlik duvarları(WAF), bilinen SQLi kalıplarını engellesede saldırganların bu filtreleri aşmak için geliştirdiği karmaşık mantıksal sorguları ve yeni nesil saldırı türlerini tespit edememektedir.SQL sorgularını sadece düz metin olarak incelemek yerine onları birer soyut sözdizilimi ağacı (AST ) graf yapısına dönüştürdük.Proje kapsamında metin tabanlı ve grafik tabanlı özellikler çıkararak makine öğrenmesi modelleri ile SQLi saldırılarını tespit etmeye çalıştık.Klasik filtreler '1=1' gibi yapıları yakalasa da daha karmaşık enjeksiyonları gözden kaçırabilir, bu yüzden yapay zekaya ihtiyaç duyarız.

---


##  Kullanılan Yapay Zeka Yaklaşımları

Bu projede tek bir model kullanmak yerine, 3 farklı yapay zeka yaklaşımını yan yana eğiterek performanslarını karşılaştırdık:

1. **Metin ML (TF-IDF + Logistic Regression):** Sorgu metinlerindeki kelime dizilimlerini analiz eden geleneksel yöntem.
2. **Graf ML (AST + Random Forest):** SQL sorgusunu bir soyut sözdizimi ağacına (AST) dönüştürüp, düğüm sayısı ve derece merkeziyeti (Degree Centrality) gibi graf metrikleriyle çalışan gözetimli yöntem.
3. **Anomali Tespiti (AST + Isolation Forest):** Sadece güvenli SQL sorgularıyla eğitilen ve normalin dışındaki tüm yapısal sapmaları anomali (saldırı) olarak yakalayan yarı-gözetimsiz yöntem.

---

##  Model Performans Karşılaştırma Sonuçları
Veri setimiz üzerindeki testler sonucunda modellerimizin elde ettiği başarı oranları şu şekildedir:

* **Geleneksel Metin ML (Logistic Regression):** %99.20 Doğruluk
* **Graf Tabanlı ML (Random Forest):** %96.40 Doğruluk
* **Graf Tabanlı Anomali Tespiti (Isolation Forest):** %68.20 Doğruluk

---

## Proje Dosya Yapısı
* `app.py`: Tarayıcı üzerinden sorguları test etmemizi sağlayan Streamlit web arayüzü kodu.
* `requirements.txt`: Projenin çalışması için gereken Python kütüphaneleri.
* `Dockerfile` & `.dockerignore`: Projeyi bulut uyumlu hale getiren Docker konteyner yapılandırmaları.
* `src/`: 
  * `preprocess.py`: SQL sorgusunu grafa çeviren ve merkeziyet özelliklerini çıkaran önişleme kodu.
  * `train.py`: 3 farklı modeli eğiten ve kaydeden eğitim betiği.
* `models/`: Eğitilmiş modellerimizin (`.pkl`) ve kıyaslama istatistiklerinin tutulduğu dizin.
* `data/`: Model eğitiminde kullanılan SQL veri setleri.

---

## Kurulum ve Çalıştırma Yolları

### Yol A: Docker ile Çalıştırma (Önerilen )
Bilgisayarınızda Docker yüklüyse, hiçbir kütüphane kurmadan tek komutla uygulamayı ayağa kaldırabilirsiniz:
1. Terminalde proje dizinine gidin.
2. Docker imajını derleyin:
   ```bash
   docker build -t sql-injection-detector .
   ```
3. Konteyneri başlatın:
   ```bash
   docker run -p 8501:8501 sql-injection-detector
   ```
4. Tarayıcınızda **http://localhost:8501** adresine gidin.

### Yol B: Geleneksel Python ile Çalıştırma 
1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. Streamlit arayüzünü çalıştırın:
   ```bash
   streamlit run app.py
   ```
3. Tarayıcınızda açılan **http://localhost:8501** adresinden sorguları analiz edin.


