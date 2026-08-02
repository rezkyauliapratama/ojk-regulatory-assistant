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

# 3. Create venv & install Python deps (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Generate chunks & embeddings from the PDFs (one-time).
#    data/extracted/, data/chunks/, data/embeddings/ are gitignored,
#    so a fresh clone must build them first:
python scripts/extract_text.py     # PDFs -> data/extracted/*.txt
python scripts/chunk.py            # -> data/chunks/*.jsonl (section-based)
python scripts/embed.py            # -> data/embeddings/*.npy (Jina API, needs JINA_API_KEY)

# 5. Ingest regulations (loads 3,670 chunks into PGVector)
python scripts/ingest.py           # dlt loads chunks into Postgres
python scripts/setup_vector.py     # vector(1024) + HNSW + FTS indexes

# 6. Verify retrieval works
python scripts/verify.py

# 7. Open the chat UI
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

# 3. Generate chunks & embeddings from the PDFs (one-time, gitignored dirs)
python scripts/extract_text.py
python scripts/chunk.py
python scripts/embed.py

# 4. Ingest (one-time) — loads chunks into PGVector
python scripts/ingest.py
python scripts/setup_vector.py

# 5. Verify retrieval
python scripts/verify.py

# 6. Run the UI locally
streamlit run app/app.py --server.port 8501
```

The app connects to Postgres via `DATABASE_URL` in `.env` (default:
`postgresql://rag:***@localhost:5432/rag_db`).

- **Streamlit in Docker** (`docker compose up -d --build`): compose
  overrides `DATABASE_URL` with host `pgvector` (the service name), so
  `localhost` in `.env` is ignored — nothing to change.
- **Local dev mode**: use host `localhost`. If your Postgres runs in
  Docker on a remote VM, use host `172.17.0.1` (Docker gateway)
  instead — see `DATABASE_URL` in `.env`.

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
├── scripts/
│   ├── setup_nyawa.sh      # one-command Nyawa installer (see §9)
│   └── ...                 # fetch/extract/chunk/embed/ingest/verify/eval
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

## 9. Nyawa Memory Layer (Bonus Feature)

### What is Nyawa?

[Nyawa](https://github.com/rezkyauliapratama/nyawa) ("soul" in Indonesian)
is an **offline-first AI memory engine** written in Go. It is a single
~8 MB binary with zero runtime dependencies (just SQLite): semantic
memory recall (HNSW + FTS5 hybrid search), a built-in RAG engine,
namespaces, and an MCP server so any AI agent can talk to it.

```
go build + one binary -> memory that lasts, no cloud, no Docker, no vector DB
```

### Why Nyawa here?

- **Cross-session conversation memory** — the chat UI stores every Q&A
  pair and recalls related past conversations, so users get continuity
  across sessions (the `Use session memory (Nyawa)` checkbox in the
  sidebar).
- **Offline & private** — all memory lives in a local SQLite file
  (`data/nyawa_memory.db`); nothing leaves the machine.
- **Zero infrastructure** — no separate service, no API key, no network.
- **Evaluated as a bonus** — it adds the *cross-session recall* bonus
  point in the rubric (see §10) without complicating the architecture.
- **Open source** — MIT-licensed, so reviewers can audit exactly how the
  memory layer behaves.

### Source

| | |
|---|---|
| Repository | https://github.com/rezkyauliapratama/nyawa |
| Language | Go 1.23+ |
| License | MIT |
| Latest release | v1.0.0 (80 commits, stable) |
| Build size | ~8.1 MB (single binary) |

### Install — easiest: one command (recommended)

```bash
bash scripts/setup_nyawa.sh              # local run (macOS/Linux host)
bash scripts/setup_nyawa.sh --for-docker # Docker container (Linux binary)
```

The script installs a **working** binary and verifies it:
- **Local run** → builds from source natively (CGO stays enabled, which
  go-sqlite3 and the BGE embedder require)
- **`--for-docker`** → builds inside a `golang:1.23` Docker container so
  the Linux binary is compiled **on Linux with CGO enabled**, then copies
  it to `./nyawa/nyawa`

> **Why not cross-compile?** Building with `GOOS=linux` on macOS silently
> sets `CGO_ENABLED=0`, which produces a broken binary that fails with
> `go-sqlite3 requires cgo to work` and `BGE unavailable`. The binary
> must be built on the same OS the app runs on — that is exactly what
> `--for-docker` does (build inside a Linux container). Requires Docker;
> without it the script falls back to the prebuilt `linux/amd64` release
> binary (only correct if your container is amd64).

### Install — Option A: download a release binary (Linux x86_64)

```bash
curl -L -o nyawa.gz https://github.com/rezkyauliapratama/nyawa/releases/download/v1.0.0/nyawa-linux-amd64.gz
gunzip nyawa.gz && chmod +x nyawa && mv nyawa ./nyawa/nyawa
```

### Install — Option B: build from source (macOS / any platform)

Requires [Go 1.23+](https://go.dev/dl/). Cloning a specific release
tag works the same as cloning `main` — checkout `v1.0.0` for the
stable, tested version:

```bash
git clone --branch v1.0.0 --depth 1 https://github.com/rezkyauliapratama/nyawa.git
cd nyawa
make build                # -> ./nyawa  (uses sqlite_fts5 build tag)

# macOS Apple Silicon: cross-compile target for your OS
GOOS=darwin GOARCH=arm64 go build -tags "sqlite_fts5" -ldflags="-s -w" -o nyawa ./cmd/nyawa/

# verify
./nyawa --version         # Nyawa — Offline-First AI Memory Engine v1.0.0

# place it where this project expects it
mkdir -p ../ojk-regulatory-assistant/nyawa
cp nyawa ../ojk-regulatory-assistant/nyawa/nyawa
```

### Wire it into this project

```bash
# 1. .env (defaults already point here, adjust if you moved things)
NYAWA_BINARY=./nyawa/nyawa
NYAWA_DB=./data/nyawa_memory.db

# 2. Start the app — the memory layer auto-detects the binary
docker compose up -d --build streamlit
# or locally: streamlit run app/app.py

# 3. In the sidebar: tick "Use session memory (Nyawa)"
#    - green "Available" = binary found
#    - amber warning with the expected path = binary missing
```

The integration lives in `app/memory_layer.py` (thin MCP-over-stdio
wrapper): each Q&A is stored via `nyawa_store` and related conversations
are recalled via `nyawa_recall` on every query. Recall metrics
(n_results, avg/max relevance score) are also logged to
`regulatory.nyawa_recalls` and shown in the Grafana dashboard (§8).

> Nyawa is fully optional. Without the binary the app degrades
> gracefully — the memory checkbox simply has no effect and the chat
> keeps working normally.

## 10. Evaluation Criteria Mapping

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

## 11. Costs & API usage

| Service | Usage | Key in `.env` |
|---------|-------|---------------|
| DeepSeek API | LLM (rewrite, answer, judge) | `OPENAI_API_KEY` |
| Jina AI | Embeddings + reranker | `JINA_API_KEY` |
| (optional) OpenRouter | Alternative LLM provider | swap in `.env` |

Free tiers: Jina offers 1M free embedding tokens/month; DeepSeek pricing is
~$0.02/M tokens (a full evaluation run costs a few cents). The embedding
step (~3.7k chunks) runs on Jina's free tier.

## 12. Glossary

Terminology used throughout this project, explained in plain language.

### Core RAG concepts

- **RAG (Retrieval-Augmented Generation)** — a pattern where the LLM's
  answer is grounded in documents retrieved from a knowledge base, instead
  of relying only on its training data. Steps: retrieve relevant chunks →
  feed them to the LLM as context → generate an answer with citations.
- **Chunk / Chunking** — splitting a long document into smaller pieces
  (here: per *BAB* / *Pasal* section). Each chunk is embedded and stored
  separately so retrieval can find the exact relevant part of a 50-page
  regulation.
- **Embedding** — a numerical vector (list of numbers) that captures the
  *meaning* of a text. Similar texts get similar vectors, which is what
  enables semantic search. This project uses **Jina AI
  (`jina-embeddings-v3`)**, 1024 dimensions, multilingual (supports
  Indonesian).
- **Vector database** — a store optimized for similarity search over
  embeddings. Here: **PGVector**, a PostgreSQL extension (not a separate
  database — it lives inside Postgres).
- **HNSW index** — *Hierarchical Navigable Small World*: the graph-based
  algorithm PGVector uses to answer "give me the 5 most similar vectors"
  quickly, even with thousands of chunks.
- **Cosine similarity** — the distance metric used by the HNSW index to
  compare embeddings (angle between vectors, 0..1). `vector_cosine_ops`
  in the index definition refers to this.

### Retrieval strategies

- **Dense retrieval** — semantic search over embeddings (PGVector cosine).
  Understands meaning, synonyms, and paraphrases.
- **Sparse / FTS (Full-Text Search)** — keyword search over the text using
  PostgreSQL `tsvector` + `GIN` index. Exact terms, good for codes like
  "POJK", "QRIS", "KPMM".
- **Hybrid search** — running dense + FTS together and merging results.
- **RRF (Reciprocal Rank Fusion)** — the merging method: for each
  candidate, sum `1 / (k + rank)` from each strategy (k=60 default).
  Simple, robust, no score normalization needed. Best retrieval strategy
  in this project's evaluation (MRR 0.667).
- **Reranking** — after retrieval returns top-10 candidates, a second,
  more precise model (Jina `jina-reranker-v2-base-multilingual`,
  a cross-encoder) re-scores them to pick the top-5. Improves precision
  at the cost of one extra API call.
- **Query rewriting** — before retrieval, the LLM rewrites the user's
  query: expands acronyms (KPMM → Kewajiban Penyediaan Modal Minimum),
  adds context, fixes typos. Improves recall on jargon-heavy queries.

### Pipeline & infrastructure

- **dlt (data load tool)** — the Python library used for the ingestion
  pipeline: reads chunk files, infers a schema, and loads them into
  PostgreSQL with incremental merge on `chunk_id`. Re-runs are safe
  (no duplicate chunks).
- **`dataset_name`** — dlt's term for the Postgres schema where it
  creates tables (here: `regulatory`).
- **Pipeline** — the full ingestion flow: PDF → extract text → chunk →
  embed → load into PGVector. Each step is a script in `scripts/`.
- **PGVector** — see *Vector database* above.
- **Docker Compose** — the tool that runs the 3 services
  (`pgvector`, `grafana`, `streamlit`) together with one command.
- **Containerization** — packaging an app with its dependencies into a
  container image so it runs identically anywhere.

### LLM & evaluation

- **LLM-as-a-Judge** — using an LLM to score another LLM's answers
  (instead of a human). Two prompt versions (v1: strict citations,
  v2: structured) were compared; the judge scored v1 higher (3.92 vs
  3.84 with DeepSeek as judge), so v1 is the default.
- **Hit Rate** — fraction of test queries where the correct document was
  retrieved in the top-k results.
- **MRR (Mean Reciprocal Rank)** — for each query, `1 / rank` of the
  first correct result, averaged. Higher = relevant docs appear earlier.
- **Ground truth** — a curated set of (query → expected document)
  pairs used to measure retrieval quality.
- **Prompt version** — a variation of the system prompt given to the LLM.
  This project evaluates 2 versions and picks the better one.

### Monitoring

- **Grafana** — the open-source dashboard tool that visualizes data from
  PostgreSQL (query volume, feedback, token usage, per-conversation
  detail).
- **Datasource** — Grafana's connection to a data source (here: the
  `PostgreSQL` datasource, UID `PG`, pointing at the `pgvector` service).
- **Provisioning** — configuring Grafana (datasources + dashboards) via
  files baked into the custom image, so a fresh `docker compose up`
  gets a fully configured Grafana with no manual clicks.

## 13. License

MIT — educational project. Regulatory documents remain property of their
issuers (OJK / Bank Indonesia), reproduced for educational purposes under
Article 42 of Indonesian Copyright Law No. 28/2014.
