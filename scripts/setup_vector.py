"""Phase 1 — cast embedding_json to pgvector column + create HNSW index.

Run AFTER ingest.py. Reads the JSON-string embeddings loaded by dlt,
casts them to vector(1024) and builds a cosine HNSW index.
"""

import os
import pathlib
import sys

import psycopg2
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DIM = 1024  # jina-embeddings-v3

load_dotenv(ROOT / ".env")


def main() -> int:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("ALTER TABLE ojk.regulation_chunks ADD COLUMN IF NOT EXISTS embedding vector(%d)" % DIM)

    # JSON string -> float[] -> vector
    cur.execute("""
        UPDATE ojk.regulation_chunks
        SET embedding = (embedding_json::jsonb)::text::vector
        WHERE embedding IS NULL AND embedding_json IS NOT NULL
    """)
    print(f"Embeddings cast to vector({DIM})")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
        ON ojk.regulation_chunks USING hnsw (embedding vector_cosine_ops)
    """)
    print("HNSW cosine index created")

    # FTS index for hybrid search (Bahasa Indonesia config if available)
    cur.execute("""
        ALTER TABLE ojk.regulation_chunks
        ADD COLUMN IF NOT EXISTS text_tsv tsvector
    """)
    cur.execute("""
        UPDATE ojk.regulation_chunks
        SET text_tsv = to_tsvector('simple', coalesce(text, ''))
        WHERE text_tsv IS NULL
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_fts
        ON ojk.regulation_chunks USING gin (text_tsv)
    """)
    print("FTS tsvector + GIN index created")

    cur.execute("SELECT count(*) FROM ojk.regulation_chunks WHERE embedding IS NOT NULL")
    n = cur.fetchone()[0]
    print(f"Chunks with embeddings: {n}")

    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
