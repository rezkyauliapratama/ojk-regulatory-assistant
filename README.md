# OJK Regulatory Intelligence Assistant

**LLM Zoomcamp 2026 - Final Project**

A RAG-based Q&A assistant over Indonesian banking and payment regulations
(OJK/BI). Ask questions about AI governance, cyber security, payment
systems (QRIS, BI-FAST), consumer protection, and more. The system
retrieves the relevant regulation passages and answers with citations.

> **Disclaimer:** Personal learning project (LLM Zoomcamp 2026 Final
> Project). Not affiliated with PT Bank Sinarmas Tbk or any financial
> institution. All regulatory documents are public domain from ojk.go.id
> and bi.go.id (Indonesian Copyright Law No. 28/2014, Article 42).

---

## 🚀 Live Demo

Try the running app right now - no setup needed.

| | URL | Login |
|---|---|---|
| **Chat App** | https://banking-comp-assistant.rezkyaulia.dev/ | Password: `whatisit?` |
| **Monitoring (Grafana)** | https://banking-comp-assistant.rezkyaulia.dev/grafana | Username: `reviewer` / Password: `grafana_llmzoomcamp_123` |

Sample questions to try: "What are the QRIS requirements?" or
"Apa ketentuan QRIS?" (Indonesian works too). See
[docs/sample-questions.md](docs/sample-questions.md) for 10 more.
The Grafana dashboard tracks queries, tokens, feedback, and cited
documents over time.

---

## 1. Problem

Indonesian banking and fintech professionals need fast, accurate answers
from regulatory documents: POJK, SEOJK, PBI, PADG. These are long PDFs
that are hard to search manually. Finding the right pasal (article) often
means reading hundreds of pages.

This project builds an end-to-end RAG assistant over a curated corpus of
**15 regulations** focused on two hot topics:

- **AI / technology governance** - AI governance for banks, IT risk
  management, cyber security, digital maturity, consumer protection
- **Payment systems** - QRIS, BI-FAST, payment service providers (PJP),
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

The PDFs are committed under `data/pdfs/` (public domain, so this is
legal), making the project **fully reproducible without any download
step**. The full manifest with source URLs is in `data/sources.yaml`.

## 3. Architecture

```
Streamlit UI (8501)
  |
  v
RAG Engine
  1. Query rewriting (LLM expands acronyms)
     KPMM -> "Kewajiban Penyediaan Modal Minimum"
  2. Hybrid retrieval
     - Dense: Jina embeddings -> pgvector cosine (HNSW index)
     - Lexical: Postgres FTS (tsvector, stopword-filtered OR)
     - Fusion: Reciprocal Rank Fusion (RRF)
  3. Rerank: Jina cross-encoder (top-10 -> top-5)
  4. LLM: DeepSeek answers with citations
  |
  v
PostgreSQL (pgvector)
  - regulatory.regulation_chunks (3,670 chunks)
  - regulatory.conversations (Q&A log + feedback)
  - regulatory.nyawa_recalls (memory recall metrics)
  |
  +---> Grafana (3000): monitoring dashboard
  +---> Nyawa (optional): cross-session memory engine
```

### Data flow (ingestion pipeline)

```
data/pdfs/*.pdf
  -> extract_text.py -> data/extracted/*.txt      (raw text)
  -> chunk.py        -> data/chunks/*.jsonl       (3,670 chunks, pasal-aware)
  -> embed.py        -> data/embeddings/*.npy     (Jina API, 1024-dim)
  -> ingest.py       -> PostgreSQL via dlt        (text + embedding_json)
  -> setup_vector.py -> vector(1024) cast + HNSW + FTS tsvector
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
| LLM | DeepSeek `deepseek-v4-flash` (OpenAI-compatible) | Cheap, strong citations |
| Monitoring | Grafana 11 | Dashboard from conversations table |
| Memory (bonus) | Nyawa v1.0.0 (offline memory engine, MCP) | Cross-session Q&A context |

All APIs are OpenAI-compatible. Swap providers by editing `.env`:

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

### Requirements

| Requirement | Version / Notes |
|-------------|-----------------|
| Python | 3.11+ (tested on 3.13) |
| Docker + Docker Compose | for Postgres + Grafana + Streamlit |
| Jina API key | https://jina.ai (free tier: 1M tokens/mo) |
| DeepSeek API key | https://platform.deepseek.com (cheap, ~$0.02/M tokens) |
| Go 1.23+ | only if you build Nyawa from source (local mode) |
| RAM | ~2 GB for the app + Postgres (fine on a small VM) |

## 5. Quick Start (fresh clone to running app)

All commands below run from the project root
(`cd ojk-regulatory-assistant`). Follow the order: each step depends on
the previous one.

### Step 1: Clone & configure

```bash
git clone https://github.com/rezkyauliapratama/ojk-regulatory-assistant.git
cd ojk-regulatory-assistant

# create .env from the template, then fill in 3 required keys:
cp .env.example .env
# - JINA_API_KEY   (https://jina.ai, free tier: 1M tokens/month)
# - OPENAI_API_KEY (DeepSeek: https://platform.deepseek.com)
# - APP_PASSWORD   (UI login password, anything you like)
```

`.env.example` already ships correct defaults (database URL, ports,
NYAWA_BINARY, NYAWA_DB). You only need to fill in the 3 keys above.

### Step 2: Setup Nyawa (optional, but recommended)

Nyawa is the bonus memory engine (cross-session Q&A recall). If you
skip this step, everything else still works - only the memory checkbox
in the sidebar stays inactive.

```bash
# macOS / Linux: build the binary + download BGE embedder files automatically
bash scripts/setup_nyawa.sh

# If Streamlit runs in DOCKER (the default), use this flag instead:
# (builds a LINUX binary inside a golang container so CGO stays enabled)
bash scripts/setup_nyawa.sh --for-docker
```

The script does it all: builds `./nyawa/nyawa`, downloads the
all-MiniLM model (~23MB) into `./nyawa/bge/`, and verifies with
`nyawa version`. It needs `git` + Go 1.23+ (local mode) or Docker
(`--for-docker`).

**Important - do not cross-compile manually.** If you build with
`GOOS=linux` from macOS without `--for-docker`, Go silently sets
`CGO_ENABLED=0` and the binary is broken: `go-sqlite3 requires cgo` +
`BGE unavailable`. See section 9 for details.

### Step 3: Start infrastructure (Postgres + Grafana + Streamlit)

```bash
docker compose up -d --build
```

`--build` is required on the first run (Grafana & Streamlit use custom
Dockerfiles with provisioning/image dependencies). If you change files
under `grafana/` or the `Dockerfile`, run it again with `--build` so
the image gets rebuilt.

Verify all containers are healthy:

```bash
docker compose ps
# Name          Status
# rag-pgvector  Up (healthy)
# rag-grafana   Up
# rag-streamlit Up
```

### Step 4: Python environment (for the ingestion scripts)

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

### Step 5: Generate chunks & embeddings (one-time)

The folders `data/extracted/`, `data/chunks/`, `data/embeddings/` are
gitignored, so a fresh clone must build them from the PDFs already
present in `data/pdfs/`:

```bash
python scripts/extract_text.py   # PDFs -> data/extracted/*.txt
python scripts/chunk.py          # -> data/chunks/*.jsonl (section-based)
python scripts/embed.py          # -> data/embeddings/*.npy (needs JINA_API_KEY)
```

`embed.py` calls the Jina API for 3,670 chunks. The Jina free tier
(1M tokens/month) is enough. It retries automatically on rate limits.

### Step 6: Ingest into Postgres (one-time)

```bash
python scripts/ingest.py         # dlt loads 3,670 chunks into Postgres
python scripts/setup_vector.py   # vector(1024) + HNSW + FTS indexes
```

### Step 7: Verify retrieval

```bash
python scripts/verify.py
```

You should see relevant retrieval results (e.g. for the query
"QRIS"). If you get `relation regulatory.regulation_chunks does not
exist`, Step 6 did not finish or failed.

### Step 7b: Run unit tests (optional)

```bash
pip install pytest          # not in requirements.txt (dev dependency)
python -m pytest tests/ -v
# 6 passed - pure-function tests (no API keys, no DB, no network)
```

### Step 8: Open the UI

```bash
streamlit run app/app.py
open http://localhost:8501
# log in with APP_PASSWORD from .env
```

### Sample questions to try

10 ready-to-use questions (English + Bahasa Indonesia) with the
regulations they target. Full list with tips: see
[`docs/sample-questions.md`](docs/sample-questions.md).

| # | English | Bahasa Indonesia | Regulation(s) covered |
|---|---------|------------------|-----------------------|
| 1 | What is QRIS and who can be a QRIS payment service provider? | Apa itu QRIS dan siapa saja yang dapat menjadi penyelenggara jasa pembayaran QRIS? | PADG 3/2025, PADG 32/2025 |
| 2 | What are the key requirements for AI governance in Indonesian banks? | Apa saja persyaratan utama tata kelola AI di perbankan Indonesia? | OJK AI Governance 2025 |
| 3 | What cybersecurity obligations do banks have under SEOJK 29/2022? | Apa saja kewajiban keamanan siber bank berdasarkan SEOJK 29/2022? | SEOJK 29/2022 |
| 4 | How must banks handle customer complaints under POJK 22/2023? | Bagaimana bank harus menangani pengaduan nasabah berdasarkan POJK 22/2023? | POJK 22/2023 |
| 5 | What is the minimum IT risk management framework a bank needs? | Apa saja komponen minimum kerangka manajemen risiko TI yang wajib dimiliki bank? | POJK 11/2022, PADK 1/2026 |
| 6 | What anti-fraud strategies does POJK 12/2024 require? | Strategi anti-fraud apa yang diwajibkan oleh POJK 12/2024? | POJK 12/2024 |
| 7 | What is BI-FAST and how does it affect payment services? | Apa itu BI-FAST dan bagaimana pengaruhnya terhadap layanan pembayaran? | PBI 10/2025 |
| 8 | What are the digital maturity levels in SEOJK 24/2023? | Apa saja tingkat kematangan digital dalam SEOJK 24/2023? | SEOJK 24/2023 |
| 9 | What does the P2SK law (UU 4/2023) change for financial sector supervision? | Apa yang diubah UU 4/2023 (P2SK) terhadap pengawasan sektor keuangan? | UU 4/2023 |
| 10 | What are the requirements for banks using fintech innovation (ITSK) under POJK 30/2025? | Apa saja persyaratan bank dalam menggunakan inovasi teknologi sektor keuangan (ITSK) berdasarkan POJK 30/2025? | POJK 30/2025 |

### App screenshots

Six screenshots of the running app, in order: login, home, search
flow, answer with citations, memory recall, and the knowledge base.

**1. Login page**

![Login page](docs/screenshots/01-login-page.png)

Password-protected entry point (APP_PASSWORD from `.env`).

**2. Home - knowledge base overview**

![Home - knowledge base overview](docs/screenshots/02-home-knowledge-base.png)

Landing view: app title, subtitle, and the collapsible knowledge base
panel listing all 15 regulations.

**3. Knowledge base detail**

![Knowledge base detail](docs/screenshots/03-knowledge-base-detail.png)

The knowledge base expanded: regulations grouped by topic
(AI / Technology, Payment Systems, Banking).

**4. Search in progress**

![Search in progress](docs/screenshots/04-searching-regulations.png)

The engine searching across all 15 regulations (query rewriting +
hybrid retrieval + reranking).

**5. Answer with citations & sources**

![Answer with citations and sources](docs/screenshots/05-answer-with-sources.png)

A generated answer with inline citations (PADG 3/2025 pasal),
expandable sources, and the 👍 / 👎 feedback buttons.

**6. Related past conversations (Nyawa memory)**

![Related past conversations](docs/screenshots/06-related-conversations.png)

Cross-session memory recall: related past Q&A shown above the current
answer (requires the "Use session memory (Nyawa)" toggle).

### Run from local (dev mode, no Streamlit container)

If you prefer running the app directly on your machine (not in a
container), only start Postgres + Grafana as infrastructure:

```bash
# 1. Infrastructure only (Grafana custom image, --build once)
docker compose up -d --build pgvector grafana

# 2. Python env + ingestion (same as Steps 4-7 above)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/extract_text.py
python scripts/chunk.py
python scripts/embed.py
python scripts/ingest.py
python scripts/setup_vector.py
python scripts/verify.py

# 3. Run the UI locally
streamlit run app/app.py --server.port 8501
```

**Database connection.** The app connects via `DATABASE_URL` in `.env`
(default: `postgresql://rag:***@localhost:5432/rag_db`):

- **Streamlit in Docker** (`docker compose up -d --build`): compose
  overrides `DATABASE_URL` with host `pgvector` (the service name), so
  you do not need to change anything.
- **Local dev mode**: use host `localhost`. If Postgres runs in Docker
  on a remote VM, use `172.17.0.1` (Docker gateway) instead.

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
| FTS error `syntax error in tsquery` | Old code - `git pull` so `fts_query()` sanitizes input |
| `vector(1024)` column missing | Re-run `python scripts/setup_vector.py` after ingest |
| Jina 429 rate limit | `embed.py` retries automatically (backoff 10s x attempt) |
| Grafana "no data" | `docker exec -i rag-pgvector psql -U rag -d rag_db < scripts/seed_grafana_demo.sql` |
| Memory toggle has no effect | Check the sidebar: amber warning = Nyawa binary missing, run `bash scripts/setup_nyawa.sh --for-docker` then `docker compose up -d --build streamlit` |
| Grafana "Datasource PG was not found" | Old image - rebuild: `docker compose up -d --build grafana` |

### Re-ingest from scratch (re-download PDFs)

```bash
python scripts/fetch_pdfs.py     # 15 docs -> data/pdfs/ (OJK/BI URLs)
python scripts/extract_text.py   # PDF -> data/extracted/*.txt
python scripts/chunk.py          # -> data/chunks/*.jsonl (~3.7k chunks)
python scripts/embed.py          # -> data/embeddings/*.npy (Jina API)
python scripts/ingest.py         # dlt -> PostgreSQL (text + embeddings)
python scripts/setup_vector.py   # pgvector + HNSW + FTS indexes
python scripts/verify.py         # smoke test
```

> If the DB is in a stale state after a failed run, reset:
> `dlt pipeline rag_regulations drop-pending-packages`, then
> `DROP SCHEMA regulatory CASCADE` in Postgres, then re-run.

## 6. Project Structure

```
├── app/
│   ├── app.py              # Streamlit UI (login, chat, citations, feedback, EN/ID toggle)
│   ├── rag_engine.py       # Hybrid search + RRF + Jina reranker + rewrite
│   ├── llm_flow.py         # 2 prompt versions (V1 strict citation / V2 structured)
│   ├── conversations.py    # Postgres logging (for Grafana)
│   └── memory_layer.py     # Nyawa memory (optional bonus feature)
├── tests/
│   └── test_rag_engine.py  # unit tests for pure functions (pytest)
├── scripts/
│   ├── fetch_pdfs.py       # download from OJK/BI (browser UA + verify=False)
│   ├── extract_text.py     # PDF -> text
│   ├── chunk.py            # pasal-aware chunking (3,670 chunks)
│   ├── embed.py            # Jina embeddings (1024-dim)
│   ├── ingest.py           # dlt pipeline -> PostgreSQL
│   ├── setup_vector.py     # vector(1024) + HNSW + FTS indexes
│   ├── verify.py           # smoke tests
│   ├── rag_ask.py          # end-to-end CLI: retrieve -> answer -> log
│   ├── retrieval_test.py   # 10-query retrieval smoke test
│   ├── gen_ground_truth.py # generate eval Q&A pairs (30)
│   ├── eval_retrieval.py   # HitRate@5 + MRR@5 (3 strategies)
│   ├── eval_llm.py         # LLM-as-a-Judge (v1 vs v2)
│   ├── setup_nyawa.sh      # one-command Nyawa installer (binary + BGE files)
│   ├── bge_server.py       # BGE embedder helper used by Nyawa (bundled)
│   └── seed_grafana_demo.sql  # demo data for the monitoring dashboard
├── data/
│   ├── pdfs/               # 15 source PDFs (committed, public domain)
│   ├── sources.yaml        # document manifest with canonical URLs
│   └── (extracted/, chunks/, embeddings/ - generated, gitignored)
├── evaluation/
│   ├── ground_truth.json   # 30 Q&A pairs (2 per document)
│   └── results/
│       ├── retrieval_eval.json  # HitRate@5 / MRR@5 per strategy
│       └── llm_eval.json        # judge scores per prompt version
├── docs/
│   ├── sample-questions.md # 10 bilingual sample questions + tips
│   └── screenshots/        # 7 screenshots (app UI + Grafana dashboard)
├── grafana/
│   ├── Dockerfile              # custom image with provisioning baked in
│   ├── provisioning/           # datasource + dashboard provider
│   └── dashboards/rag-monitoring.json
├── nyawa/                  # created by setup_nyawa.sh (gitignored binary)
│   ├── nyawa               # Nyawa v1.0.0 binary
│   └── bge/                # BGE embedder files (bge_server.py + ONNX model)
├── docker-compose.yml      # pgvector + grafana + streamlit
├── Dockerfile              # Streamlit app image
└── requirements.txt        # pinned versions
```

## 7. Evaluation

Evaluation artifacts live in `evaluation/` and are reproducible via the
`eval_*` scripts. This section also maps to the course's evaluation
criteria (see section 10).

### 7.1 Retrieval evaluation - 3 strategies

Ground truth: 30 Q&A pairs generated from the regulation chunks (2 per
document, `scripts/gen_ground_truth.py`). Metrics: Hit Rate@5 and MRR@5.

| Strategy | HitRate@5 | MRR@5 |
|----------|-----------|-------|
| Dense (pgvector cosine) | 0.800 | 0.631 |
| FTS (Postgres full-text) | 0.700 | 0.386 |
| **Hybrid (dense + FTS + RRF)** | **0.800** | **0.667** |

**The hybrid strategy is used in production.** RRF fusion improves
ranking (MRR 0.667 vs 0.631 for dense-only) by combining semantic and
lexical signals. Reproduce with:

```bash
python scripts/eval_retrieval.py
```

### 7.2 LLM evaluation - 2 prompt versions, LLM-as-a-Judge

Two prompt templates:

- **V1 - strict citation**: answers every claim with `[document pasal]`
- **V2 - structured**: answer summary / key points / legal basis sections

The judge LLM scores each answer on 4 criteria (1-5): relevance,
groundedness, completeness, citations.

| Prompt | Relevance | Groundedness | Completeness | Citations | **Overall** |
|--------|-----------|--------------|--------------|-----------|-------------|
| **V1 strict citation** | 4.67 | 3.10 | 4.27 | 3.63 | **3.92** |
| V2 structured | 4.60 | 3.20 | 4.20 | 3.37 | 3.84 |

Result (judge: deepseek-v4-flash, 30 questions): **V1 wins and is the
default** in the UI. This is consistent with an earlier run using a
different judge model (gpt-5.4-mini: V1 3.95 vs V2 3.69). Reproduce:

```bash
python scripts/eval_llm.py --sample 30
```

## 8. Monitoring (Grafana)

The Grafana dashboard **"RAG Monitoring"** is provisioned automatically
(custom image, `grafana/`). It has 13 panels fed by the
`regulatory.conversations` table (every chat interaction + feedback)
and `regulatory.nyawa_recalls` (memory metrics):

- Queries per Day (time series)
- Total Queries (stat)
- User Feedback (bar chart)
- Tokens per Query (time series)
- Prompt Version (pie)
- Most Cited Documents (bar)
- Avg Tokens / Query, Avg Docs Cited, Positive Feedback (%) (stats)
- Recent Conversations (Detail) - table of the last 50 Q&A: query,
  answer preview, prompt version, model, tokens, feedback, cited docs
- Nyawa Avg Relevance, Nyawa Recall Relevance over Time, Nyawa Recall
  Results per Query (memory metrics)

![Grafana RAG Monitoring dashboard](docs/screenshots/07-grafana-monitoring.png)

Open `http://localhost:3000` (admin / GRAFANA_PASSWORD from `.env`).
To see data immediately, seed the demo set:

```bash
docker exec -i rag-pgvector psql -U rag -d rag_db \
  < scripts/seed_grafana_demo.sql
```

Any change under `grafana/` requires rebuilding the image:

```bash
docker compose up -d --build grafana
```

## 9. Nyawa Memory Layer (Bonus Feature)

### What is Nyawa?

[Nyawa](https://github.com/rezkyauliapratama/nyawa) ("soul" in
Indonesian) is an **offline-first AI memory engine** written in Go. It
is a single ~8 MB binary with zero runtime dependencies (just SQLite):
semantic memory recall (HNSW + FTS5 hybrid search), a built-in RAG
engine, namespaces, and an MCP server so any AI agent can talk to it.

```bash
go build + one binary -> memory that lasts, no cloud, no Docker, no vector DB
```

### Why Nyawa?

- **Cross-session conversation memory** - the chat UI stores every Q&A
  pair and recalls related past conversations, so users get continuity
  across sessions (the "Use session memory (Nyawa)" checkbox in the
  sidebar)
- **Offline & private** - all memory lives in a local SQLite file
  (`data/nyawa_memory.db`); nothing leaves the machine
- **Zero infrastructure** - no separate service, no API key, no network
- **Bonus rubric point** - adds the *cross-session recall* bonus point
  (see section 10)
- **Open source** - MIT license, so reviewers can audit exactly how the
  memory layer behaves

### Source

| | |
|---|---|
| Repository | https://github.com/rezkyauliapratama/nyawa |
| Language | Go 1.23+ |
| License | MIT |
| Latest release | v1.0.0 (80 commits, stable) |
| Build size | ~8.1 MB (single binary) |

### Install - easiest: one command (recommended)

```bash
# macOS / Linux, for local runs:
bash scripts/setup_nyawa.sh

# macOS / Linux, if Streamlit runs in DOCKER (this project's default):
bash scripts/setup_nyawa.sh --for-docker
```

The script automatically:
1. Detects your OS + architecture (macOS arm64, macOS amd64, Linux
   arm64, Linux amd64 are all supported)
2. Builds the Nyawa v1.0.0 binary from source (native build, CGO
   enabled)
   - `--for-docker` builds inside a `golang:1.23` container so the
     binary is LINUX (matches the streamlit container)
3. Downloads the BGE embedder files (bge_server.py + all-MiniLM ONNX
   model, ~23MB from Hugging Face) into `./nyawa/bge/`
4. Verifies with `nyawa version`

Requires `git` + Go 1.23+ for local mode; Docker for `--for-docker`.

> **Why not cross-compile manually?** If you build with `GOOS=linux`
> from macOS, Go silently sets `CGO_ENABLED=0`. The resulting binary is
> broken: `go-sqlite3 requires cgo to work` and `BGE unavailable`. The
> binary must be built on the same OS where the app runs. That is
> exactly what `--for-docker` does (build inside a Linux container).
> Without Docker, the script falls back to the prebuilt `linux/amd64`
> release binary (only correct if your container is amd64).

### Install - Option A: download release binary (Linux x86_64 only)

```bash
curl -L -o nyawa.gz https://github.com/rezkyauliapratama/nyawa/releases/download/v1.0.0/nyawa-linux-amd64.gz
gunzip nyawa.gz && chmod +x nyawa && mv nyawa ./nyawa/nyawa
```

### Install - Option B: build manually from source (macOS / Linux)

Requires [Go 1.23+](https://go.dev/dl/). Clone the release tag
`v1.0.0` (not `main`, to get the stable & tested version):

```bash
git clone --branch v1.0.0 --depth 1 https://github.com/rezkyauliapratama/nyawa.git
cd nyawa
make build                # -> ./nyawa  (uses sqlite_fts5 build tag)

# verify
./nyawa version           # nyawa v1.0.0

# place it where this project expects it
mkdir -p ../ojk-regulatory-assistant/nyawa
cp nyawa ../ojk-regulatory-assistant/nyawa/nyawa
```

### Integration (once the binary + BGE files are in place)

```bash
# 1. .env (defaults already point here, adjust if you moved things)
NYAWA_BINARY=./nyawa/nyawa
NYAWA_DB=./data/nyawa_memory.db

# 2. REBUILD the streamlit image (required - the new image installs
#    onnxruntime & mounts ./nyawa/bge to the path Nyawa looks for)
docker compose up -d --build streamlit

# 3. Open the UI and tick "Use session memory (Nyawa)" in the sidebar
#    - green "Available" = binary + embedder working
#    - amber warning = binary/BGE files incomplete
```

**Mount structure.** Nyawa v1.0.0 hardcodes the BGE embedder path at
`/opt/data/nyawa/internal/embedder`. docker-compose.yml mounts
`./nyawa/bge` to that path (read-only), so the embedder files from the
host are directly visible inside the container. The Dockerfile also
installs `onnxruntime`, `numpy`, `tokenizers` (Python dependencies of
bge_server.py).

**How it works.** `app/memory_layer.py` is a thin MCP-over-stdio
wrapper: every Q&A is stored via `nyawa_store` and related past
conversations are recalled via `nyawa_recall` on every query. Recall
metrics (n_results, avg/max relevance score) are also logged to
`regulatory.nyawa_recalls` and shown in the Grafana dashboard
(section 8).

**Nyawa is fully optional.** Without the binary, the app degrades
gracefully - the memory checkbox has no effect and the chat keeps
working normally.

## 10. Evaluation Criteria Mapping

| Criterion | Implementation | Points |
|-----------|----------------|--------|
| Problem description | README section 1 + 2 | 2 |
| Retrieval flow | PGVector KB + DeepSeek LLM (section 3) | 2 |
| Retrieval evaluation | 3 strategies, best (hybrid) used (section 7.1) | 2 |
| LLM evaluation | 2 prompts, LLM-as-a-Judge, best (V1) used (section 7.2) | 2 |
| Interface | Streamlit UI: login, chat, citations, feedback, EN/ID toggle | 2 |
| Ingestion pipeline | dlt (automated, incremental) | 2 |
| Monitoring | User feedback + Grafana dashboard 13 panels | 2 |
| Containerization | Full docker-compose (pgvector + grafana + streamlit) | 2 |
| Reproducibility | PDFs committed, pinned versions, clear steps | 2 |
| Best practice: hybrid search | Dense + FTS + RRF, evaluated | +1 |
| Best practice: reranking | Jina cross-encoder, top-10 -> top-5 | +1 |
| Best practice: query rewriting | LLM expands acronyms (KPMM, ITSK, PJP...) | +1 |
| Bonus: Nyawa memory layer | Offline memory engine, cross-session recall | +1 |

## 11. Costs & API usage

| Service | Usage | Key in `.env` |
|---------|-------|---------------|
| DeepSeek API | LLM (rewrite, answer, judge) | `OPENAI_API_KEY` |
| Jina AI | Embeddings + reranker | `JINA_API_KEY` |
| (optional) OpenRouter | Alternative LLM provider | swap in `.env` |

Free tiers: Jina offers 1M free embedding tokens/month; DeepSeek pricing
is ~$0.02/M tokens (a full evaluation run costs a few cents). The
embedding step (~3.7k chunks) runs on Jina's free tier.

## 12. Glossary

Terminology used throughout this project, explained in plain language.

### Core RAG concepts

- **RAG (Retrieval-Augmented Generation)** - a pattern where the LLM's
  answer is grounded in documents retrieved from a knowledge base,
  instead of relying only on its training data. Steps: retrieve relevant
  chunks, feed them to the LLM as context, generate an answer with
  citations.
- **Chunk / Chunking** - splitting a long document into smaller pieces
  (here: per *BAB* / *Pasal* section). Each chunk is embedded and stored
  separately so retrieval can find the exact relevant part of a 50-page
  regulation.
- **Embedding** - a numerical vector (list of numbers) that captures the
  *meaning* of a text. Similar texts get similar vectors, which enables
  semantic search. This project uses **Jina AI (`jina-embeddings-v3`)**
  with 1024 dimensions, multilingual (supports Indonesian).
- **Vector database** - a store optimized for similarity search over
  embeddings. Here: **PGVector**, a PostgreSQL extension (not a separate
  database, it lives inside Postgres).
- **HNSW index** - *Hierarchical Navigable Small World*: the graph-based
  algorithm PGVector uses to answer "give me the 5 most similar vectors"
  quickly, even with thousands of chunks.
- **Cosine similarity** - the distance metric used by the HNSW index to
  compare embeddings (angle between vectors, 0..1). `vector_cosine_ops`
  in the index definition refers to this.

### Retrieval strategies

- **Dense retrieval** - semantic search over embeddings (PGVector
  cosine). Understands meaning, synonyms, and paraphrases.
- **Sparse / FTS (Full-Text Search)** - keyword search over the text
  using PostgreSQL `tsvector` + `GIN` index. Exact terms, good for codes
  like "POJK", "QRIS", "KPMM".
- **Hybrid search** - running dense + FTS together and merging results.
- **RRF (Reciprocal Rank Fusion)** - the merging method: for each
  candidate, sum `1 / (k + rank)` from each strategy (k=60 default).
  Simple, robust, no score normalization needed. Best retrieval strategy
  in this project's evaluation (MRR 0.667).
- **Reranking** - after retrieval returns top-10 candidates, a second,
  more precise model (Jina `jina-reranker-v2-base-multilingual`, a
  cross-encoder) re-scores them to pick the top-5. Improves precision at
  the cost of one extra API call.
- **Query rewriting** - before retrieval, the LLM rewrites the user's
  query: expands acronyms (KPMM -> Kewajiban Penyediaan Modal Minimum),
  adds context, fixes typos. Improves recall on jargon-heavy queries.

### Pipeline & infrastructure

- **dlt (data load tool)** - the Python library used for the ingestion
  pipeline: reads chunk files, infers a schema, and loads them into
  PostgreSQL with incremental merge on `chunk_id`. Re-runs are safe
  (no duplicate chunks).
- **`dataset_name`** - dlt's term for the Postgres schema where it
  creates tables (here: `regulatory`).
- **Pipeline** - the full ingestion flow: PDF -> extract text -> chunk
  -> embed -> load into PGVector. Each step is a script in `scripts/`.
- **PGVector** - see *Vector database* above.
- **Docker Compose** - the tool that runs the 3 services (`pgvector`,
  `grafana`, `streamlit`) together with one command.
- **Containerization** - packaging an app with its dependencies into a
  container image so it runs identically anywhere.

### LLM & evaluation

- **LLM-as-a-Judge** - using an LLM to score another LLM's answers
  (instead of a human). Two prompt versions (v1: strict citations,
  v2: structured) were compared; the judge scored v1 higher (3.92 vs
  3.84 with DeepSeek as judge), so v1 is the default.
- **Hit Rate** - fraction of test queries where the correct document was
  retrieved in the top-k results.
- **MRR (Mean Reciprocal Rank)** - for each query, `1 / rank` of the
  first correct result, averaged. Higher = relevant docs appear earlier.
- **Ground truth** - a curated set of (query -> expected document) pairs
  used to measure retrieval quality.
- **Prompt version** - a variation of the system prompt given to the
  LLM. This project evaluates 2 versions and picks the better one.

### Monitoring

- **Grafana** - the open-source dashboard tool that visualizes data from
  PostgreSQL (query volume, feedback, token usage, per-conversation
  detail).
- **Datasource** - Grafana's connection to a data source (here: the
  `PostgreSQL` datasource, UID `PG`, pointing at the `pgvector` service).
- **Provisioning** - configuring Grafana (datasources + dashboards) via
  files baked into the custom image, so a fresh `docker compose up` gets
  a fully configured Grafana with no manual clicks.

### Nyawa memory layer

- **Nyawa** - an offline-first AI memory engine written in Go. Stores
  Q&A pairs in SQLite and recalls related past conversations via hybrid
  search (HNSW vector + FTS5 keyword).
- **MCP (Model Context Protocol)** - the protocol Nyawa uses to expose
  tools (`nyawa_store`, `nyawa_recall`) over stdio, so the Streamlit app
  can call it like a small local server.
- **BGE embedder** - the local embedding model (all-MiniLM-L6-v2, ONNX,
  384-dim) Nyawa uses to vectorize text. Needs `bge_server.py` + model
  files at `/opt/data/nyawa/internal/embedder` (mounted from
  `./nyawa/bge/`).
- **CGO** - a Go mechanism to call C code; go-sqlite3 requires it.
  Cross-compiling with `GOOS=linux` from macOS disables it, producing a
  broken Nyawa binary.
- **Namespace** - a label that separates memories into groups (here:
  `rag_qa` for chat Q&A).

## 13. License

MIT - educational project. Regulatory documents remain property of their
issuers (OJK / Bank Indonesia), reproduced for educational purposes
under Article 42 of Indonesian Copyright Law No. 28/2014.

