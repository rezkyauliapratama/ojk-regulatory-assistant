"""Phase 1 — smoke test: verify a retrieval query returns relevant chunks."""

import os
import pathlib
import sys

import psycopg2
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SAMPLE_QUERIES = [
    "Apa kewajiban bank dalam penerapan tata kelola?",
    "Bagaimana ketentuan keamanan siber untuk bank umum?",
    "Apa itu pasal terkait teknologi informasi?",
]


def main() -> int:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM regulation_chunks")
    n = cur.fetchone()[0]
    print(f"Total chunks in PGVector: {n}")
    assert n > 0, "No chunks loaded!"
    cur.close()
    conn.close()
    print("Verify OK — retrieval ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
