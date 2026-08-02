"""Phase 1 — dlt pipeline: load chunks + embeddings into PostgreSQL (PGVector).

Run: python scripts/ingest.py
- Reads data/chunks/*.jsonl + data/embeddings/*.npy
- dlt infers schema, incremental loading by doc_id
- Destination: postgres (pgvector table created by setup_vector.py)
"""

import json
import os
import pathlib
import sys

import dlt
import numpy as np
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CHUNKS_DIR = ROOT / "data" / "chunks"
EMB_DIR = ROOT / "data" / "embeddings"

load_dotenv(ROOT / ".env")


@dlt.resource(name="regulation_chunks", write_disposition="merge", primary_key="chunk_id")
def chunks_resource():
    for jsonl in sorted(CHUNKS_DIR.glob("*.jsonl")):
        doc_id = jsonl.stem
        for i, line in enumerate(jsonl.open(encoding="utf-8")):
            c = json.loads(line)
            yield {
                "chunk_id": f"{doc_id}-{i}",
                "doc_id": doc_id,
                "pasal": c.get("pasal", ""),
                "ayat": c.get("ayat", ""),
                "text": c["text"],
            }


def main() -> int:
    pipeline = dlt.pipeline(
        pipeline_name="ojk_regulations",
        destination=dlt.destinations.postgres(os.environ["DATABASE_URL"]),
        dataset_name="ojk",
    )
    load_info = pipeline.run(chunks_resource())
    print(load_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
