"""Phase 1 — inject embeddings into PGVector + create HNSW index.

Reads data/chunks/*.jsonl (row order) + data/embeddings/*.npy and writes
into the `ojk.regulation_chunks` table loaded by dlt (ingest.py).

Run AFTER ingest.py.
"""

import json
import os
import pathlib
import sys

import numpy as np
import psycopg2
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CHUNKS_DIR = ROOT / "data" / "chunks"
EMB_DIR = ROOT / "data" / "embeddings"
DIM = 384  # intfloat/multilingual-e5-small

load_dotenv(ROOT / ".env")


def main() -> int:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("ALTER TABLE ojk.regulation_chunks ADD COLUMN IF NOT EXISTS embedding vector(%d)" % DIM)
    print("vector(%d) column ready" % DIM)

    n_updated = 0
    for jsonl in sorted(CHUNKS_DIR.glob("*.jsonl")):
        doc_id = jsonl.stem
        emb_path = EMB_DIR / f"{doc_id}.npy"
        if not emb_path.exists():
            print(f"  [skip] {doc_id}: no embeddings file")
            continue
        vectors = np.load(emb_path)
        chunks = [json.loads(line) for line in jsonl.open(encoding="utf-8")]
        assert len(vectors) == len(chunks), f"{doc_id}: {len(vectors)} vectors vs {len(chunks)} chunks"

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}-{i}"
            vec_str = "[" + ",".join(f"{v:.6f}" for v in vectors[i]) + "]"
            cur.execute(
                "UPDATE ojk.regulation_chunks SET embedding = %s::vector WHERE chunk_id = %s",
                (vec_str, chunk_id),
            )
        n_updated += len(chunks)
        print(f"  [ok] {doc_id}: {len(chunks)} embeddings injected", flush=True)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON ojk.regulation_chunks USING hnsw (embedding vector_cosine_ops)")
    print("HNSW index created")
    print(f"Total embeddings injected: {n_updated}")

    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
