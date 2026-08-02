-- Seed demo data for Grafana dashboard verification
INSERT INTO ojk.conversations (query, answer, prompt_version, model, docs, usage_tokens, feedback, created_at) VALUES
('Apa itu QRIS?', 'QRIS adalah standar kode QR pembayaran Indonesia.', 'v1', 'gpt-5.4-mini', '[{"doc_id":"PADG_3_2025_QRIS_TUNTAS"}]', 1200, 'up', now() - interval '6 days'),
('Kewajiban bank AI?', 'Bank wajib menerapkan tata kelola AI.', 'v1', 'gpt-5.4-mini', '[{"doc_id":"OJK_AI_Governance_Banking_2025"}]', 4052, 'up', now() - interval '5 days'),
('Keamanan siber bank?', 'Bank wajib menerapkan keamanan siber.', 'v1', 'gpt-5.4-mini', '[{"doc_id":"SEOJK_29_2022_Keamanan_Siber_Bank_Umum"}]', 3671, 'up', now() - interval '4 days'),
('Syarat ITSK?', 'ITSK wajib memiliki izin dari OJK.', 'v2', 'gpt-5.4-mini', '[{"doc_id":"POJK_30_2025_Tata_Kelola_ITSK"}]', 1890, 'down', now() - interval '3 days'),
('Transaksi valas?', 'Transaksi valas wajib memiliki underlying.', 'v2', 'gpt-5.4-mini', '[{"doc_id":"PADG_24_2022_Pasar_Valuta_Asing"}]', 1835, 'up', now() - interval '2 days'),
('Strategi anti fraud?', 'LJK wajib punya strategi anti fraud.', 'v1', 'gpt-5.4-mini', '[{"doc_id":"POJK_12_2024_Strategi_Anti_Fraud"}]', 3392, NULL, now() - interval '1 day'),
('Ketentuan BI-FAST?', 'BI-FAST adalah sistem pembayaran cepat BI.', 'v1', 'gpt-5.4-mini', '[{"doc_id":"PBI_10_2025_Industri_Sistem_Pembayaran"}]', 2100, 'up', now() - interval '12 hours'),
('Apa itu PJP?', 'PJP adalah penyedia jasa pembayaran.', 'v2', 'gpt-5.4-mini', '[{"doc_id":"PADG_32_2025_Industri_Sistem_Pembayaran"}]', 1500, NULL, now() - interval '6 hours'),
('Maturitas digital?', 'SEOJK 24 mengatur maturitas digital.', 'v1', 'gpt-5.4-mini', '[{"doc_id":"SEOJK_24_2023_Maturitas_Digital_Bank_Umum"}]', 1750, 'up', now() - interval '3 hours'),
('Perlindungan konsumen?', 'PUJK wajib melindungi konsumen.', 'v1', 'gpt-5.4-mini', '[{"doc_id":"POJK_22_2023_Pelindungan_Konsumen"}]', 2600, 'up', now() - interval '1 hour'),
('KPMM bank umum?', 'Bank wajib memenuhi KPMM.', 'v1', 'gpt-5.4-mini', '[{"doc_id":"UU_4_2023_P2SK"}]', 3100, 'down', now() - interval '30 minutes'),
('Tata kelola teknologi?', 'PADK 1/2026 mengatur TI bank umum.', 'v2', 'gpt-5.4-mini', '[{"doc_id":"PADK_1_2026_Teknologi_Informasi_Bank_Umum"}]', 1450, 'up', now() - interval '5 minutes');
