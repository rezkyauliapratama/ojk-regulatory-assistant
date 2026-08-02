"""Phase 3 — LLM client with two prompt versions (OpenRouter / OpenAI-compatible).

Prompt versions:
- V1 strict citation: answer strictly from provided passages, cite pasal + doc
- V2 structured: same grounding, but structured output (summary, points, citations)
"""

import json
import os
import pathlib
import sys
from typing import Any

import requests
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(ROOT / ".env")

LLM_BASE = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")

SYSTEM_V1 = (
    "You are an Indonesian financial regulation assistant (OJK/BI). "
    "Answer the user's question STRICTLY based on the provided regulatory "
    "passages. If the passages do not contain the answer, say so clearly. "
    "Always cite the source document and pasal for every claim, like "
    "[POJK 11/2022 Pasal 3]. Never invent regulations or pasal numbers."
)

SYSTEM_V2 = (
    "You are an Indonesian financial regulation assistant (OJK/BI). "
    "Answer the user's question based on the provided regulatory passages. "
    "Structure your answer as: (1) Jawaban singkat — 2-3 sentences; "
    "(2) Poin-poin kunci — bullet list; (3) Dasar hukum — list of cited "
    "documents with pasal. If the passages lack the answer, say so. "
    "Never invent regulations or pasal numbers."
)


def _render_context(docs: list[dict]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        pasal = f"Pasal {d['pasal']}" if d.get("pasal") else "Pasal -"
        parts.append(f"[{i}] Sumber: {d['doc_id']} ({pasal})\n{d['text'][:2000]}")
    return "\n\n".join(parts)


def answer(
    query: str,
    docs: list[dict],
    prompt_version: str = "v1",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call LLM with retrieved docs. Returns {answer, model, prompt_version}."""
    system = SYSTEM_V1 if prompt_version == "v1" else SYSTEM_V2
    context = _render_context(docs)

    resp = requests.post(
        f"{LLM_BASE}/chat/completions",
        json={
            "model": LLM_MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"PERTANYAAN:\n{query}\n\n"
                        f"PASAL-PASAL YANG RELEVAN:\n{context}"
                    ),
                },
            ],
        },
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "answer": data["choices"][0]["message"]["content"].strip(),
        "model": data.get("model", LLM_MODEL),
        "prompt_version": prompt_version,
        "usage": data.get("usage", {}),
    }


# ------------------------------------------------------------- CLI demo
def _main() -> int:
    import sys

    from app.rag_engine import RagEngine

    if len(sys.argv) < 2:
        print("Usage: python -m app.llm_flow \"<query>\" [v1|v2]")
        return 1
    args = sys.argv[1:]
    version = "v1"
    if args and args[-1] in ("v1", "v2"):
        version = args.pop()
    query = " ".join(args)

    engine = RagEngine()
    docs = engine.retrieve(query)
    result = answer(query, docs, prompt_version=version)
    print(f"\n=== Prompt {version.upper()} ===")
    print(result["answer"])
    print(f"\n--- model: {result['model']} | tokens: {result['usage'].get('total_tokens')}")
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
