# OJK Regulatory Intelligence Assistant

**LLM Zoomcamp 2026 — Final Project**

A RAG-based Q&A assistant over Indonesian banking and payment regulations
(OJK/BI). Ask questions in Bahasa Indonesia about AI governance, cyber
security, payment systems (QRIS, BI-FAST), consumer protection, and more —
the system retrieves relevant regulation passages and answers with citations.

> **Disclaimer:** Personal learning project (LLM Zoomcamp 2026 Final Project).
> Not affiliated with PT Bank Sinarmas Tbk or any financial institution. All
> regulatory documents are public domain from ojk.go.id and bi.go.id
> (Indonesian Copyright Law No. 28/2014, Article 42).

---

## 1. Problem

Indonesian banking and fintech professionals need fast, accurate answers from
regulatory documents — POJK, SEOJK, PBI, PADG — which are long PDFs that are
hard to search manually. Finding the right pasal (article) often means
reading hundreds of pages.

This project builds an end-to-end RAG assistant over a curated corpus of
**15 regulations** focused on two hot topics:

- **AI / technology governance** — AI governance for banks, IT risk
  management, cyber security, digital maturity, consumer protection
- **Payment systems** — QRIS, BI-FAST, payment service providers (PJP),
  foreign exchange, P2SK financial sector law

## 2. Dataset

15 public-domain regulatory documents from ojk.go.id and bi.go.id:

| # | Document | Topic |
|---|----------|-------|
| 1 | OJK AI Governance for Indonesian Banking 2025 | AI governance |
| 2 | POJK 11/2022 | Bank IT |
| 3 | SEOJK 29/2022 | Bank cyber security |
| 4 | POJK 12/2024 | Anti-fraud strategy |
| 5 | POJK 17/2023 | Bank governance |
| 6 | POJK 22/2023 | Consumer protection |
| 7 | POJK 30/2025 | ITSK (fintech innovation) |
| 8 | PADK 1/2026 | Bank IT (new) |
| 9 | SEOJK 24/2023 | Digital maturity |
| 10 | PBI 10/2025 | Payment systems industry |
| 11 | PADG 32/2025 | Payment systems (PJP) |
| 12 | PADG 3/2025 | QRIS |
| 13 | PBI 24/2022 | Foreign exchange |
| 14 | PADG 24/2022 | Foreign exchange market |
| 15 | UU 4/2023 | P2SK (financial sector law) |

The PDFs are committed under `data/pdfs/` (public domain, so this is legal),
making the project **fully reproducible without any download step**. The
full manifest with source URLs is in `data/sources.yaml`.

## 3. Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│  Streamlit  │ ──► │                 RAG Engine                   │
│  UI (8501)  │     │                                              │
└─────────────┘     │  1. Query rewriting (LLM expands acronyms)   │
                    │     KPMM → "Kewajiban Penyediaan Modal       │
                    │            Minimum"                          │
                    │  2. Hybrid retrieval:                        │
                    │     • Dense: Jina embeddings → pgvector      │
                    │       cosine (HNSW index)                    │
                    │     • Lexical: Postgres FTS (tsvector,       │
                    │       stopword-filtered OR semantics)        │
                    │     • Fusion: Reciprocal Rank Fusion (RRF)   │
                    │  3. Rerank: Jina cross-encoder (top-10→5)    │
                    │  4. LLM: DeepSeek answers with citations     │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │          PostgreSQL (pgvector)               │
                    │  regulatory.regulation_chunks (3,670 chunks)   │
                    │  regulatory.conversations (Q&A log + feedback) │
                    └───────────────┬──────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   ┌──────────────────┐                          ┌──────────────────┐
   │  Grafana (3000)  │                          │  Nyawa (optional)│
   │  Monitoring:     │                          │  memory engine   │
   │  6 charts        │                          │  cross-session   │
   └──────────────────┘                          │  context recall  │
                                                 └──────────────────┘
```

### Data flow (ingestion pipeline)

```
data/pdfs/*.pdf
  → extract_text.py   → data/extracted/*.txt      (raw text)
  → chunk.py          → data/chunks/*.jsonl       (3,670 chunks, pasal-aware)
  → embed.py          → data/embeddings/*.npy     (Jina API, 1024-dim)
  → ingest.py         → PostgreSQL via dlt        (text + embedding_json)
  → setup_vector.py   → vector(1024) cast + HNSW + FTS tsvector
```

## 4. Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Interface | Streamlit 1.60 | Fast UI with chat, citations, feedback |
| Ingestion | dlt 1.29 (postgres) | Incremental loading, schema inference |
| Vector DB | PostgreSQL 16 + pgvector | HNSW cosine index |
| Embeddings | Jina AI `jina-embeddings-v3` (API, 1024-dim) | Multilingual, no local model |
| Hybrid search | pgvector dense + Postgres FTS + RRF | Best-practice hybrid retrieval |
| Reranker | Jina `jina-reranker-v2-base-multilingual` | Multilingual cross-encoder |
| LLM | DeepSeek `deepseek-v4-flash` (OpenAI-compatible API) | Cheap, strong citations |
| Monitoring | Grafana 11 | 6-chart dashboard from conversations |
| Memory (bonus) | Nyawa v1.0.0 (offline memory engine, MCP) | Cross-session Q&A context |

All APIs are OpenAI-compatible — swap providers by editing `.env`:

```bash
# Default (DeepSeek):
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-...          # DeepSeek key
LLM_MODEL=deepseek-v4-flash

# Alternative (OpenRouter):
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-...       # OpenRouter key
LLM_MODEL=gpt-5.4-mini
```

## 5. Quick Start

### Prerequisites

| Requirement | Version / Notes |
|-------------|-----------------|
| Python | 3.11+ (tested on 3.13) |
| Docker + Docker Compose | for Postgres + Grafana |
| Jina API key | https://jina.ai (free tier: 1M tokens/mo) |
| DeepSeek API key | https://platform.deepseek.com (cheap, ~$0.02/M tokens) |
| RAM | ~2 GB for the app + Postgres (fine on a small VM) |

```bash
# 1. Clone & configure
git clone https://github.com/rezkyauliapratama/ojk-regulatory-assistant.git
cd ojk-regulatory-assistant
cp .env.example .env            # fill in: JINA_API_KEY, OPENAI_API_KEY, APP_PASSWORD

# 2. Start infrastructure (Postgres + Grafana + Streamlit)
#    Grafana & Streamlit use custom Dockerfiles, so build is required
#    (first run or after any change to grafana/ or Dockerfile):
docker compose up -d --build

# 3. Install Python deps (Python 3.11+)
pip install -r requirements.txt

# 4. Ingest regulations (PDFs already in data/pdfs/ — no download needed)
python scripts/ingest.py         # dlt loads 3,670 chunks into PGVector
python scripts/setup_vector.py   # vector(1024) + HNSW + FTS indexes

# 5. Verify retrieval works
python scripts/verify.py

# 6. Open the chat UI
streamlit run app/app.py
open http://localhost:8501       # login with APP_PASSWORD
```

### Run from local (dev mode — no Streamlit container)

Prefer running the app directly on your machine instead of the Docker
Streamlit container? Only start the *infrastructure* containers, then run
the app from your local Python:

```bash
# 1. Start only Postgres + Grafana (infrastructure)
#    Grafana has a custom Dockerfile (provisioning/dashboards baked in),
#    so build it first:
docker compose up -d --build pgvector grafana

# 2. Local Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Ingest (one-time) — PDFs already committed, no download needed
python scripts/ingest.py
python scripts/setup_vector.py

# 4. Verify retrieval
python scripts/verify.py

# 5. Run the UI locally
streamlit run app/app.py --server.port 8501
```

The app connects to Postgres via `DATABASE_URL` in `.env` (default:
`postgresql://rag:change_me@localhost:5432/rag_db`). If your
Postgres runs in Docker, use host `172.17.0.1` (Docker gateway) instead
of `localhost` — see `DATABASE_URL` in `.env`.

### Ports

| Service | Port | URL |
|---------|------|-----|
| Streamlit UI | 8501 | http://localhost:8501 |
| Grafana | 3000 | http://localhost:3000 |
| PostgreSQL | 5432 | (internal, not exposed) |

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `dlt` load stuck / stale state | `dlt pipeline rag_regulations drop-pending-packages`, then re-run `ingest.py` |
| FTS query error `syntax error in tsquery` | Only happens with old code — pull latest (`git pull`) so `fts_query()` sanitizes input |
| `vector(1024)` column missing | Re-run `python scripts/setup_vector.py` after ingest |
| Jina 429 rate limit | `embed.py` retries with backoff automatically (10s×attempt) |
| Grafana shows "no data" | Run `docker exec -i rag-pgvector psql -U rag -d rag_db < scripts/seed_grafana_demo.sql` |
| Memory toggle does nothing | Nyawa binary missing — check `nyawa/nyawa` exists, or it degrades gracefully (app still works) |

### Re-ingest from scratch (re-download PDFs)

```bash
python scripts/fetch_pdfs.py     # 15 docs -> data/pdfs/  (OJK/BI URLs)
python scripts/extract_text.py   # PDF -> data/extracted/*.txt
python scripts/chunk.py          # -> data/chunks/*.jsonl (~3.7k chunks)
python scripts/embed.py          # -> data/embeddings/*.npy (Jina API)
python scripts/ingest.py         # dlt -> PostgreSQL (text + embeddings)
python scripts/setup_vector.py   # pgvector + HNSW + FTS indexes
python scripts/verify.py         # smoke test
```

> If you hit a stale DB state after a failed run, reset:
> `dlt pipeline rag_regulations drop-pending-packages` then
> `DROP SCHEMA regulatory CASCADE` in Postgres, then re-run.

## 6. Project Structure

```
├── app/
│   ├── app.py              # Streamlit UI (login, chat, citations, feedback)
│   ├── rag_engine.py       # Hybrid search + RRF + Jina reranker + rewrite
│   ├── llm_flow.py         # 2 prompt versions (V1 strict citation / V2 structured)
│   ├── conversations.py    # Postgres logging (for Grafana)
│   └── memory_layer.py     # Nyawa memory (optional bonus feature)
├── scripts/
│   ├── fetch_pdfs.py       # download from OJK/BI (browser UA + verify=False)
│   ├── extract_text.py     # PDF → text
│   ├── chunk.py            # pasal-aware chunking (3,670 chunks)
│   ├── embed.py            # Jina embeddings (1024-dim)
│   ├── ingest.py           # dlt pipeline → PostgreSQL
│   ├── setup_vector.py     # vector(1024) + HNSW + FTS indexes
│   ├── verify.py           # smoke tests
│   ├── rag_ask.py          # end-to-end CLI: retrieve → answer → log
│   ├── retrieval_test.py   # 10-query retrieval smoke test
│   ├── gen_ground_truth.py # generate eval Q&A pairs (30)
│   ├── eval_retrieval.py   # HitRate@5 + MRR@5 (3 strategies)
│   ├── eval_llm.py         # LLM-as-a-Judge (v1 vs v2)
│   └── seed_grafana_demo.sql  # demo data for the monitoring dashboard
├── data/
│   ├── pdfs/               # 15 source PDFs (committed — public domain)
│   ├── sources.yaml        # document manifest with canonical URLs
│   └── (extracted/, chunks/, embeddings/ — generated, gitignored)
├── evaluation/
│   ├── ground_truth.json   # 30 Q&A pairs (2 per document)
│   └── results/
│       ├── retrieval_eval.json  # HitRate@5 / MRR@5 per strategy
│       └── llm_eval.json        # judge scores per prompt version
├── grafana/
│   ├── Dockerfile              # custom image with provisioning baked in
│   ├── provisioning/           # datasource + dashboard provider
│   └── dashboards/rag-monitoring.json
├── nyawa/nyawa             # Nyawa v1.0.0 binary (bonus memory feature)
├── docker-compose.yml      # pgvector + grafana + streamlit
├── Dockerfile              # Streamlit app image
└── requirements.txt        # pinned versions
```

## 7. Evaluation

Evaluation artifacts live in `evaluation/` (reproducible via the `eval_*`
scripts). This section is also reflected in the README per the course's
"mention the evaluation criteria" recommendation.

### 7.1 Retrieval evaluation — 3 strategies

Ground truth: 30 Q&A pairs generated from the regulation chunks (2 per
document, `scripts/gen_ground_truth.py`). Metrics: Hit Rate@5 and MRR@5.

| Strategy | HitRate@5 | MRR@5 |
|----------|-----------|-------|
| Dense (pgvector cosine) | 0.800 | 0.631 |
| FTS (Postgres full-text) | 0.700 | 0.386 |
| **Hybrid (dense + FTS + RRF)** | **0.800** | **0.667** |

**The hybrid strategy is used in production** — RRF fusion improves ranking
(MRR 0.667 vs 0.631 for dense-only) by combining semantic and lexical
signals. Reproduce with:

```bash
python scripts/eval_retrieval.py
```

### 7.2 LLM evaluation — 2 prompt versions, LLM-as-a-Judge

Two prompt templates:

- **V1 — strict citation**: answers every claim with `[document pasal]`
- **V2 — structured**: answer summary / key points / legal basis sections

The judge LLM scores each answer on 4 criteria (1-5): relevance,
groundedness, completeness, citations.

| Prompt | Relevance | Groundedness | Completeness | Citations | **Overall** |
|--------|-----------|--------------|--------------|-----------|-------------|
| **V1 strict citation** | 4.67 | 3.10 | 4.27 | 3.63 | **3.92** |
| V2 structured | 4.60 | 3.20 | 4.20 | 3.37 | 3.84 |

Result (judge: deepseek-v4-flash, 30 questions): **V1 wins and is the
default** in the UI. The result is consistent with an earlier run using a
different judge model (gpt-5.4-mini: V1 3.95 vs V2 3.69). Reproduce with:

```bash
python scripts/eval_llm.py --sample 30
```

## 8. Monitoring (Grafana)

The Grafana dashboard **"RAG Monitoring"** is provisioned automatically
(custom image, `grafana/`). It has **6 charts** fed by the
`regulatory.conversations` table (every chat interaction + 👍/👎 feedback):

1. Pertanyaan per Hari — queries per day (time series)
2. Total Pertanyaan — total count (stat)
3. Feedback Pengguna (👍/👎) — feedback distribution (bar)
4. Token per Pertanyaan — LLM usage per query (time series)
5. Prompt Version — v1 vs v2 usage (pie)
6. Dokumen Paling Dikutip — top cited documents (bar)

Open `http://localhost:3000` (admin / GRAFANA_PASSWORD from `.env`). To see
data immediately, seed the demo set:

```bash
docker exec -i rag-pgvector psql -U rag -d rag_db \
  < scripts/seed_grafana_demo.sql
```

## 9. Evaluation Criteria Mapping

| Criterion | Implementation | Points |
|-----------|----------------|--------|
| Problem description | README §1 + §2 | 2 |
| Retrieval flow | PGVector KB + DeepSeek LLM (§3) | 2 |
| Retrieval evaluation | 3 strategies, best (hybrid) used (§7.1) | 2 |
| LLM evaluation | 2 prompts, LLM-as-a-Judge, best (V1) used (§7.2) | 2 |
| Interface | Streamlit UI: login, chat, citations, feedback | 2 |
| Ingestion pipeline | dlt (automated, incremental) | 2 |
| Monitoring | User feedback + Grafana dashboard 6 charts | 2 |
| Containerization | Full docker-compose (pgvector + grafana + streamlit) | 2 |
| Reproducibility | PDFs committed, pinned versions, clear steps | 2 |
| Best practice: hybrid search | Dense + FTS + RRF, evaluated | +1 |
| Best practice: reranking | Jina cross-encoder, top-10 → top-5 | +1 |
| Best practice: query rewriting | LLM expands acronyms (KPMM, ITSK, PJP...) | +1 |
| Bonus: Nyawa memory layer | Offline memory engine, cross-session recall | +1 |

## 10. Costs & API usage

| Service | Usage | Key in `.env` |
|---------|-------|---------------|
| DeepSeek API | LLM (rewrite, answer, judge) | `OPENAI_API_KEY` |
| Jina AI | Embeddings + reranker | `JINA_API_KEY` |
| (optional) OpenRouter | Alternative LLM provider | swap in `.env` |

Free tiers: Jina offers 1M free embedding tokens/month; DeepSeek pricing is
~$0.02/M tokens (a full evaluation run costs a few cents). The embedding
step (~3.7k chunks) runs on Jina's free tier.

## 11. License

MIT — educational project. Regulatory documents remain property of their
issuers (OJK / Bank Indonesia), reproduced for educational purposes under
Article 42 of Indonesian Copyright Law No. 28/2014.
