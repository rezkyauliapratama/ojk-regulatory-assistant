"""Phase 3 — conversation logging to PostgreSQL (used by Grafana monitoring).

Table `conversations`:
- id, query, answer, prompt_version, model
- docs (JSON list of cited chunks), usage_tokens
- feedback (thumbs up/down), created_at
"""

import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any

import psycopg2
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(ROOT / ".env")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ojk.conversations (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    rewritten_query TEXT,
    answer TEXT NOT NULL,
    prompt_version TEXT DEFAULT 'v1',
    model TEXT,
    docs JSONB,
    usage_tokens INT,
    feedback TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
)
"""


def init_db() -> None:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(SCHEMA)
    cur.close()
    conn.close()


def log_conversation(
    query: str,
    answer: str,
    docs: list[dict],
    *,
    rewritten_query: str | None = None,
    prompt_version: str = "v1",
    model: str | None = None,
    usage_tokens: int | None = None,
    feedback: str | None = None,
) -> int:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ojk.conversations
            (query, rewritten_query, answer, prompt_version, model, docs, usage_tokens, feedback)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            query,
            rewritten_query,
            answer,
            prompt_version,
            model,
            json.dumps(docs, ensure_ascii=False),
            usage_tokens,
            feedback,
        ),
    )
    conv_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return conv_id


def set_feedback(conv_id: int, feedback: str) -> None:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "UPDATE ojk.conversations SET feedback = %s WHERE id = %s",
        (feedback, conv_id),
    )
    cur.close()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("conversations table ready")
