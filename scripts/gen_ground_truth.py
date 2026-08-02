"""Phase 5 — generate ground truth Q&A pairs from regulation chunks.

For each document, sample a few chunks and ask the LLM to produce
question-answer pairs grounded in those chunks. Output: evaluation/ground_truth.json

Usage: python scripts/gen_ground_truth.py [--per-doc 2] [--docs ...]
"""

import json
import os
import pathlib
import random
import sys

import requests
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CHUNKS_DIR = ROOT / "data" / "chunks"
OUT = ROOT / "evaluation" / "ground_truth.json"

load_dotenv(ROOT / ".env")

LLM_BASE = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")

SYSTEM = (
    "You create question-answer pairs about Indonesian financial regulations "
    "(OJK/BI) for evaluating a RAG system. Given a document ID and its text "
    "chunks, produce EXACTLY 1 question per chunk. Rules:\n"
    "- The question must be answerable ONLY from that chunk.\n"
    "- Write the question as a real user would ask (Bahasa Indonesia, natural).\n"
    "- Answer: 1-3 sentence summary grounded in the chunk.\n"
    "- Return JSON list: [{\"question\": \"...\", \"answer\": \"...\", "
    "\"doc_id\": \"...\", \"chunk_ids\": [\"...\"]}]\n"
    "- No extra text outside the JSON."
)


def generate_pairs(doc_id: str, chunks: list[dict], per_doc: int, offset: int = 0) -> list[dict]:
    """Ask LLM to produce Q&A pairs for sampled chunks of one document."""
    # sample evenly-spaced chunks, offset shifts the starting point
    step = max(1, len(chunks) // per_doc)
    sampled = chunks[offset::step][:per_doc]
    prompt_chunks = []
    for idx, c in enumerate(sampled):
        cid = f"{doc_id}-{offset + idx * step}"  # chunk_id convention: {doc_id}-{i}
        prompt_chunks.append(f"[{cid}] {c['text'][:800]}")
    user_msg = (
        f"Document: {doc_id}\n"
        f"Chunks:\n" + "\n".join(prompt_chunks)
    )
    resp = requests.post(
        f"{LLM_BASE}/chat/completions",
        json={
            "model": LLM_MODEL,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {"type": "json_object"},
        },
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        timeout=90,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    # strip markdown fences if present
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    try:
        data = json.loads(content)
        # json_object mode can return either a list or a single-pair object
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "questions" in data and isinstance(data["questions"], list):
                return data["questions"]
            if "question" in data:  # single pair object
                return [data]
        return []
    except json.JSONDecodeError as e:
        print(f"  [warn] {doc_id}: JSON parse failed ({e}); raw={content[:100]}")
        return []


def main() -> int:
    per_doc = int(sys.argv[sys.argv.index("--per-doc") + 1]) if "--per-doc" in sys.argv else 2
    offset = int(sys.argv[sys.argv.index("--offset") + 1]) if "--offset" in sys.argv else 0
    out_file = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else str(OUT)
    doc_filter = None
    if "--docs" in sys.argv:
        doc_filter = sys.argv[sys.argv.index("--docs") + 1 :]

    jsonls = sorted(CHUNKS_DIR.glob("*.jsonl"))
    if doc_filter:
        jsonls = [j for j in jsonls if any(d in j.stem for d in doc_filter)]

    all_pairs: list[dict] = []
    for jsonl in jsonls:
        doc_id = jsonl.stem
        chunks = [json.loads(line) for line in jsonl.open(encoding="utf-8")]
        if not chunks:
            continue
        print(f"  {doc_id}: {len(chunks)} chunks, sampling {per_doc} (offset {offset})...", flush=True)
        pairs = generate_pairs(doc_id, chunks, per_doc, offset)
        all_pairs.extend(pairs)
        print(f"    -> {len(pairs)} pairs", flush=True)

    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    print(f"\nTotal ground truth: {len(all_pairs)} pairs -> {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
