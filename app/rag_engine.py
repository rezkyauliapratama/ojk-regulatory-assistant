"""RAG engine — hybrid retrieval (dense + FTS + RRF), Jina reranker, query rewriting.

Components:
1. QueryRewriter   — LLM expands abbreviations (KPMM, NPL, ITSK, PJP...)
2. HybridSearch    — PGVector dense (cosine) + Postgres FTS, fused with RRF
3. JinaReranker    — cross-encoder rerank top-N -> top-K
4. RagEngine       — full flow: rewrite -> hybrid retrieve -> rerank

Run standalone (CLI demo):
    python -m app.rag_engine "Apa kewajiban bank dalam penerapan tata kelola?"
"""

import json
import os
import pathlib
import re
import sys
from typing import Any

import psycopg2
import requests
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(ROOT / ".env")

JINA_BASE = os.environ.get("JINA_BASE_URL", "https://api.jina.ai/v1")
JINA_EMBED_MODEL = os.environ.get("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
JINA_RERANK_MODEL = os.environ.get("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")
EMBED_DIM = int(os.environ.get("JINA_EMBEDDING_DIM", "1024"))
EMBED_BATCH = 8

# LLM (OpenRouter / OpenAI-compatible)
LLM_BASE = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")

# RRF fusion constant
RRF_K = 60

STOPWORDS_ID = {
    "apa", "yang", "untuk", "dengan", "bagaimana", "dalam", "pada", "dari",
    "ke", "di", "dan", "atau", "adalah", "itu", "ini", "tersebut", "wajib",
    "bank", "umum", "ketentuan", "mengenai", "tentang", "apakah", "saja",
    "dokumen", "pasal", "dengan", "suatu", "sebuah", "para", "oleh", "akan",
    "tidak", "juga", "saat", "lebih", "paling", "harus", "bisa", "dapat",
}


def fts_query(query: str) -> str:
    """Build an OR-semantics tsquery from query keywords (stopwords removed)."""
    words = [w.lower() for w in re.findall(r"[a-z0-9]+", query.lower())
             if w not in STOPWORDS_ID and len(w) > 3]
    if not words:
        return query
    return " | ".join(words)


# ---------------------------------------------------------------- embedding
def embed_query(query: str) -> list[float]:
    """Embed a user query (task=retrieval.query — asymmetric with stored passages)."""
    resp = requests.post(
        f"{JINA_BASE}/embeddings",
        json={"model": JINA_EMBED_MODEL, "task": "retrieval.query", "dimensions": EMBED_DIM, "input": [query]},
        headers={"Authorization": f"Bearer {os.environ['JINA_API_KEY']}", "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


# ------------------------------------------------------------- query rewrite
class QueryRewriter:
    """LLM-based query rewriting: expand abbreviations + add context."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.enabled = bool(self.api_key)

    def rewrite(self, query: str) -> str:
        if not self.enabled:
            return query
        try:
            resp = requests.post(
                f"{LLM_BASE}/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You rewrite user queries about Indonesian financial "
                                "regulations (OJK/BI). Expand abbreviations (KPMM, NPL, "
                                "BMPK, ITSK, PJP, PIP, SNAP, QRIS, BI-FAST, PUJK, LJK) "
                                "into full terms and add clarifying context. "
                                "Reply ONLY with the rewritten query, no explanation."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                },
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            rewritten = resp.json()["choices"][0]["message"]["content"].strip()
            return rewritten if rewritten else query
        except Exception as e:  # noqa: BLE001 — fail-open to original query
            print(f"  [rewrite] fallback to original ({e})", flush=True)
            return query


# ------------------------------------------------------------- hybrid search
class HybridSearch:
    """Dense (pgvector cosine) + FTS (tsvector) retrieval fused with RRF."""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        self.conn = psycopg2.connect(os.environ["DATABASE_URL"])

    def _dense(self, query_vec: list[float], k: int) -> list[tuple[str, float]]:
        vec_str = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT chunk_id, (embedding <=> %s::vector) AS dist
            FROM regulatory.regulation_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vec_str, vec_str, k),
        )
        rows = cur.fetchall()
        cur.close()
        return [(r[0], r[1]) for r in rows]

    def _fts(self, query: str, k: int) -> list[tuple[str, float]]:
        tq = fts_query(query)
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT chunk_id, ts_rank(text_tsv, to_tsquery('simple', %s)) AS score
            FROM regulatory.regulation_chunks
            WHERE text_tsv @@ to_tsquery('simple', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (tq, tq, k),
        )
        rows = cur.fetchall()
        cur.close()
        return [(r[0], r[1]) for r in rows]

    def search(self, query: str, query_vec: list[float], top_k: int | None = None) -> list[dict]:
        k = top_k or self.top_k
        dense = self._dense(query_vec, k)
        fts = self._fts(query, k)

        # RRF fusion
        scores: dict[str, float] = {}
        for rank, (chunk_id, _) in enumerate(dense):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (chunk_id, _) in enumerate(fts):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        fused = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [{"chunk_id": cid, "rrf_score": s} for cid, s in fused]

    def fetch(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT chunk_id, doc_id, pasal, ayat, text
            FROM regulatory.regulation_chunks
            WHERE chunk_id = ANY(%s)
            """,
            (chunk_ids,),
        )
        rows = cur.fetchall()
        cur.close()
        by_id = {r[0]: {"chunk_id": r[0], "doc_id": r[1], "pasal": r[2], "ayat": r[3], "text": r[4]} for r in rows}
        return [by_id[c] for c in chunk_ids if c in by_id]

    def close(self) -> None:
        self.conn.close()


# ------------------------------------------------------------- reranker
class JinaReranker:
    """Jina cross-encoder reranker (multilingual)."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        if not docs:
            return []
        docs = docs[:30]  # cap rerank input
        resp = requests.post(
            f"{JINA_BASE}/rerank",
            json={
                "model": JINA_RERANK_MODEL,
                "query": query,
                "documents": [d["text"][:2000] for d in docs],
                "top_n": self.top_k,
            },
            headers={"Authorization": f"Bearer {os.environ['JINA_API_KEY']}", "Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        ranked = resp.json()["results"]
        out: list[dict] = []
        for item in sorted(ranked, key=lambda x: x["index"]):
            doc = docs[item["index"]]
            out.append({**doc, "rerank_score": item.get("relevance_score", 0.0)})
        out.sort(key=lambda x: -x["rerank_score"])
        return out


# ------------------------------------------------------------- engine
class RagEngine:
    def __init__(self, top_k: int = 10, rerank_k: int = 5) -> None:
        self.rewriter = QueryRewriter()
        self.search = HybridSearch(top_k=top_k)
        self.reranker = JinaReranker(top_k=rerank_k)

    def retrieve(self, query: str, rewrite: bool = True) -> list[dict]:
        original = query
        if rewrite:
            query = self.rewriter.rewrite(query)
            if query != original:
                print(f"  [rewrite] '{original}' -> '{query}'", flush=True)

        query_vec = embed_query(query)
        fused = self.search.search(query, query_vec)
        docs = self.search.fetch([d["chunk_id"] for d in fused])
        ranked = self.reranker.rerank(query, docs)
        return ranked

    def close(self) -> None:
        self.search.close()


# ------------------------------------------------------------- CLI demo
def _main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m app.rag_engine \"<query>\"")
        return 1
    query = " ".join(sys.argv[1:])
    engine = RagEngine()
    results = engine.retrieve(query)
    print(f"\nQuery: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['rerank_score']:.3f}] {r['doc_id']} | Pasal {r['pasal'] or '-'}")
        print(f"   {r['text'][:120].replace(chr(10), ' ')}...\n")
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
