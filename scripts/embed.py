"""Phase 1 — embed chunks with multilingual-e5-base (local, no API key)."""

import json
import pathlib
import sys

from sentence_transformers import SentenceTransformer

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CHUNKS_DIR = ROOT / "data" / "chunks"
OUT_DIR = ROOT / "data" / "embeddings"
MODEL = "intfloat/multilingual-e5-small"
PREFIX = "query: " if False else "passage: "  # e5 requires 'passage:' prefix for stored docs


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL)
    print(f"Model loaded: {MODEL}")

    for jsonl in sorted(CHUNKS_DIR.glob("*.jsonl")):
        chunks = [json.loads(line) for line in jsonl.open(encoding="utf-8")]
        if not chunks:
            continue
        texts = [PREFIX + c["text"][:2000] for c in chunks]
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
        out = OUT_DIR / f"{jsonl.stem}.npy"
        import numpy as np

        np.save(out, vectors)
        print(f"  {jsonl.stem}: {len(vectors)} vectors -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
