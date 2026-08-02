"""Phase 1 — embed chunks with Jina AI API (jina-embeddings-v3, 1024-dim).

Uses the Jina API (multilingual, no local model download).
Requires JINA_API_KEY in .env (see .env.example).
"""

import json
import os
import pathlib
import sys
import time

import numpy as np
import requests
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CHUNKS_DIR = ROOT / "data" / "chunks"
OUT_DIR = ROOT / "data" / "embeddings"

load_dotenv(ROOT / ".env")

API_KEY = os.environ["JINA_API_KEY"]
BASE_URL = os.environ.get("JINA_BASE_URL", "https://api.jina.ai/v1")
MODEL = os.environ.get("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
BATCH = 32  # embeddings per API request


def embed_batch(texts: list[str], retries: int = 6) -> list[list[float]]:
    """Embed one batch via Jina API. Handles rate limits with backoff."""
    payload = {
        "model": MODEL,
        "task": "retrieval.passage",  # retrieval task type
        "dimensions": 1024,
        "input": texts,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(retries):
        resp = requests.post(f"{BASE_URL}/embeddings", json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        if resp.status_code == 429:  # rate limit
            wait = 10 * (attempt + 1)
            print(f"  [429] rate limited, waiting {wait}s (attempt {attempt+1}/{retries})...", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Jina embed failed after {retries} retries")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonls = sorted(CHUNKS_DIR.glob("*.jsonl"))
    print(f"Jina model: {MODEL} | {len(jsonls)} chunk files", flush=True)

    for jsonl in jsonls:
        chunks = [json.loads(line) for line in jsonl.open(encoding="utf-8")]
        if not chunks:
            continue
        # skip docs already embedded
        out = OUT_DIR / f"{jsonl.stem}.npy"
        if out.exists():
            print(f"  [skip] {jsonl.stem} already embedded", flush=True)
            continue
        texts = [c["text"][:2000] for c in chunks]

        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            vectors.extend(embed_batch(batch))
            if (i // BATCH) % 10 == 0:
                print(f"  {jsonl.stem}: {min(i+BATCH, len(texts))}/{len(texts)}", flush=True)

        np.save(out, np.array(vectors, dtype=np.float32))
        print(f"  [ok] {jsonl.stem}: {len(vectors)} vectors (1024-dim) -> {out.name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
