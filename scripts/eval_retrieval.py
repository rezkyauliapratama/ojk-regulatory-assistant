"""Phase 5 — retrieval evaluation: compare 3 strategies.

Strategies:
1. dense  — pgvector cosine only
2. fts    — Postgres full-text search only
3. hybrid — RRF fusion (dense + fts)

Metrics: Hit Rate @5, MRR @5. Ground truth: evaluation/ground_truth.json
(each item has question + chunk_ids of the correct chunk).

Usage: python scripts/eval_retrieval.py
Output: evaluation/results/retrieval_eval.json
"""

import json
import os
import pathlib
import re
import sys
import time

import psycopg2
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.rag_engine import embed_query  # noqa: E402

load_dotenv(ROOT / ".env")

GT_FILE = ROOT / "evaluation" / "ground_truth.json"
OUT_FILE = ROOT / "evaluation" / "results" / "retrieval_eval.json"
RRF_K = 60
TOP_K = 5


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


class EvalSearch:
    """Bare retrieval for eval (no reranker — measuring raw retrieval)."""

    def __init__(self) -> None:
        self.conn = psycopg2.connect(os.environ["DATABASE_URL"])

    def dense(self, vec: list[float], k: int = TOP_K) -> list[str]:
        vec_str = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
        cur = self.conn.cursor()
        cur.execute(
            "SELECT chunk_id FROM regulatory.regulation_chunks "
            "WHERE embedding IS NOT NULL ORDER BY embedding <=> %s::vector LIMIT %s",
            (vec_str, k),
        )
        ids = [r[0] for r in cur.fetchall()]
        cur.close()
        return ids

    def fts(self, query: str, k: int = TOP_K) -> list[str]:
        tq = fts_query(query)
        cur = self.conn.cursor()
        cur.execute(
            "SELECT chunk_id FROM regulatory.regulation_chunks "
            "WHERE text_tsv @@ to_tsquery('simple', %s) "
            "ORDER BY ts_rank(text_tsv, to_tsquery('simple', %s)) DESC LIMIT %s",
            (tq, tq, k),
        )
        ids = [r[0] for r in cur.fetchall()]
        cur.close()
        return ids

    def hybrid(self, query: str, vec: list[float], k: int = TOP_K) -> list[str]:
        dense_ids = self.dense(vec, k * 2)
        fts_ids = self.fts(query, k * 2)
        scores: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, cid in enumerate(fts_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [cid for cid, _ in ranked]

    def close(self) -> None:
        self.conn.close()


def hit_rate_at_k(retrieved: list[str], relevant: list[str], k: int = TOP_K) -> float:
    return 1.0 if any(cid in relevant for cid in retrieved[:k]) else 0.0


def mrr_at_k(retrieved: list[str], relevant: list[str], k: int = TOP_K) -> float:
    for rank, cid in enumerate(retrieved[:k], 1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def main() -> int:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    print(f"Ground truth: {len(gt)} questions\n")

    search = EvalSearch()
    strategies = {"dense": [], "fts": [], "hybrid": []}

    for item in gt:
        q = item["question"]
        relevant = item.get("chunk_ids", [])
        if not relevant:
            continue
        vec = embed_query(q)
        for name in strategies:
            if name == "dense":
                retrieved = search.dense(vec)
            elif name == "fts":
                retrieved = search.fts(q)
            else:
                retrieved = search.hybrid(q, vec)
            strategies[name].append(
                {"hit": hit_rate_at_k(retrieved, relevant), "mrr": mrr_at_k(retrieved, relevant)}
            )
        time.sleep(0.2)  # Jina rate limit courtesy

    search.close()

    results = {}
    for name, scores in strategies.items():
        n = len(scores)
        hit = sum(s["hit"] for s in scores) / n if n else 0
        mrr = sum(s["mrr"] for s in scores) / n if n else 0
        results[name] = {"hit_rate_at_5": round(hit, 4), "mrr_at_5": round(mrr, 4), "n": n}
        print(f"  {name:8s}  HitRate@5={hit:.3f}  MRR@5={mrr:.3f}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
