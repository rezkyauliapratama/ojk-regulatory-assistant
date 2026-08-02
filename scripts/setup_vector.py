"""Phase 1 — create pgvector column + HNSW index on the dlt-loaded table."""

import os
import pathlib
import sys

import psycopg2
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def main() -> int:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # dlt stores embedding as a float[] array; convert to vector(768) column
    cur.execute("""
        ALTER TABLE regulation_chunks
        ADD COLUMN IF NOT EXISTS embedding_vec vector(384)
    """)
    cur.execute("""
        UPDATE regulation_chunks
        SET embedding_vec = embedding::vector
        WHERE embedding_vec IS NULL AND embedding IS NOT NULL
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
        ON regulation_chunks USING hnsw (embedding_vec vector_cosine_ops)
    """)
    print("pgvector column + HNSW index ready.")
    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
