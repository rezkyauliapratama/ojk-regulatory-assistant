# OJK Regulatory Intelligence Assistant
# LLM Zoomcamp 2026 — Final Project
# RAG-based Q&A over Indonesian banking regulations (OJK/BI)

> **Disclaimer:** This is a personal learning project (LLM Zoomcamp 2026 Final
> Project). Not affiliated with PT Bank Sinarmas Tbk or any financial
> institution. All regulatory documents are public domain from ojk.go.id and
> bi.go.id (Indonesian Copyright Law No. 28/2014, Article 42).

## Problem

Indonesian banking professionals need quick, accurate answers from regulatory
documents (POJK, SEOJK, PBI, PADG) — but these are long PDFs that are hard to
search manually. This project builds a RAG assistant over a curated set of
regulations focused on **AI/technology governance** and **payment systems**.

## Tech Stack

- **UI:** Streamlit
- **Ingestion:** dlt (data load tool) — Python pipeline, incremental loading
- **Vector DB:** PostgreSQL + pgvector (HNSW index)
- **Embeddings:** multilingual-e5-base (768-dim, Bahasa Indonesia + English)
- **LLM:** OpenRouter (OpenAI-compatible, `gpt-5.4-mini`)
- **Memory:** Nyawa — offline-first memory engine (MCP over stdio)
- **Monitoring:** Grafana (5+ charts from `conversations` table)
- **Deploy:** Docker Compose (pgvector + grafana + streamlit)

## Quick Start (Local)

```bash
# 1. Clone & configure
git clone https://github.com/rezkyauliapratama/ojk-regulatory-assistant.git
cd ojk-regulatory-assistant
cp .env.example .env        # fill in API keys

# 2. Start infrastructure (Postgres + Grafana + Streamlit)
docker compose up -d

# 3. Install Python deps (Python 3.11+)
pip install -r requirements.txt

# 4. Ingest regulations (PDFs already in data/pdfs/ — no download needed)
python scripts/ingest.py        # loads chunks + embeddings into PGVector
python scripts/setup_vector.py  # creates vector(384) column + HNSW index

# 5. Verify retrieval works
python scripts/verify.py

# 6. Open the chat UI
streamlit run app/app.py
open http://localhost:8501
```

### Ingesting from scratch (re-download PDFs)

PDFs are committed under `data/pdfs/`, but you can re-fetch from the
canonical OJK/BI URLs at any time:

```bash
python scripts/fetch_pdfs.py     # 15 docs -> data/pdfs/
python scripts/extract_text.py   # PDF -> data/extracted/*.txt
python scripts/chunk.py          # -> data/chunks/*.jsonl (~3.7k chunks)
python scripts/embed.py          # -> data/embeddings/*.npy (multilingual-e5-small)
python scripts/ingest.py         # dlt -> PostgreSQL
python scripts/setup_vector.py   # pgvector HNSW index
python scripts/verify.py         # smoke test
```

> **Note:** `embed.py` runs on CPU by default (~45 min for 3.7k chunks on a
> 2-core VM). Set `EMBEDDING_MODEL=intfloat/multilingual-e5-base` in `.env`
> for higher quality (768-dim) if you have a GPU or more CPU.

## Project Structure

```
├── app/               # Streamlit UI + RAG engine + memory layer
├── scripts/           # Ingestion pipeline (fetch → extract → chunk → embed → load)
├── data/
│   ├── pdfs/          # Source PDFs (committed — public domain)
│   └── sources.yaml   # Document manifest (15 regulations)
├── evaluation/        # Retrieval & LLM evaluation results
├── tests/             # Smoke tests
└── docker-compose.yml
```

## Evaluation Criteria Mapping

*(Detailed in `evaluation/results.md`)*

| Criterion | Status |
|-----------|--------|
| Problem description | README + this doc |
| Retrieval flow | PGVector + LLM RAG |
| Retrieval evaluation | 3 strategies (vector / text / hybrid) |
| LLM evaluation | 2 prompt versions, LLM-as-a-Judge |
| Interface | Streamlit UI |
| Ingestion pipeline | dlt (automated) |
| Monitoring | Grafana dashboard (5+ charts) |
| Containerization | Full docker-compose |
| Reproducibility | PDFs committed + pinned versions |

## License

MIT — educational project. Regulatory documents remain property of their
issuers (OJK / Bank Indonesia), reproduced for educational purposes under
Article 42 of Law No. 28/2014.
