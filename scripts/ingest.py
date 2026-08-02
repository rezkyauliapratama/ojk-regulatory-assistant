"""Phase 1 — dlt pipeline: load chunks + embeddings into PostgreSQL (PGVector).

Run: python scripts/ingest.py
- Reads data/chunks/*.jsonl + data/embeddings/*.npy
- Embeddings are stored as JSON strings (avoids dlt array -> child table),
  then cast to vector(1024) by setup_vector.py
- dlt schema inference + incremental loading (merge on chunk_id)
- Destination: postgres
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
        emb_path = EMB_DIR / f"{doc_id}.npy"
        vectors = np.load(emb_path) if emb_path.exists() else None
        for i, line in enumerate(jsonl.open(encoding="utf-8")):
            c = json.loads(line)
            emb = vectors[i].tolist() if vectors is not None else None
            yield {
                "chunk_id": f"{doc_id}-{i}",
                "doc_id": doc_id,
                "pasal": c.get("pasal", ""),
                "ayat": c.get("ayat", ""),
                "text": c["text"],
                # JSON string -> stays in the same row (no child table)
                "embedding_json": json.dumps(emb) if emb else None,
            }


def main() -> int:
    pipeline = dlt.pipeline(
        pipeline_name="rag_regulations",
        destination=dlt.destinations.postgres(os.environ["DATABASE_URL"]),
        dataset_name="regulatory",
    )
    load_info = pipeline.run(chunks_resource())
    print(load_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
