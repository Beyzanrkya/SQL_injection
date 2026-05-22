-- ========================================================
-- BULUT BİLİŞİM VE YAPAY ZEKA DERSİ FİNAL PROJESİ
-- TEST SORGUSU (SQL INJECTION DOSYA YÜKLEME DENEMESİ)
-- ========================================================

SELECT username, email FROM members WHERE id = 1 UNION SELECT password, credit_card_number FROM payments --
