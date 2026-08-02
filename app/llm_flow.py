"""Phase 3 — LLM client with two prompt versions (OpenRouter / OpenAI-compatible).

Prompt versions:
- V1 strict citation: answer strictly from provided passages, cite pasal + doc
- V2 structured: same grounding, but structured output (summary, points, citations)

Language support:
- answer(..., language="en"|"id") — system prompt instructs the LLM to
  answer in the requested language (default: English)
- translate_docs(docs, "en") — batch-translates cited chunk texts so the
  UI can show English citations while retrieval stays on the original
  Indonesian text.
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

# Language instruction appended to the system prompt.
LANG_INSTRUCTION = {
    "en": "Answer in English. Keep legal terminology precise and cite sources.",
    "id": "Jawab dalam Bahasa Indonesia. Pertahankan istilah hukum yang tepat dan kutip sumbernya.",
}


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
    language: str = "en",
) -> dict[str, Any]:
    """Call LLM with retrieved docs. Returns {answer, model, prompt_version, usage}."""
    system = (SYSTEM_V1 if prompt_version == "v1" else SYSTEM_V2)
    system += " " + LANG_INSTRUCTION.get(language, LANG_INSTRUCTION["en"])
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


def translate_docs(docs: list[dict], target_lang: str = "en") -> list[dict]:
    """Batch-translate cited chunk texts to target_lang (one LLM call).

    Only the retrieved chunks are translated (3-5 per query), not the whole
    knowledge base. Returns copies of docs with 'text' replaced by the
    translation; original kept in '_original'. Falls back to original text
    if the translation call fails.
    """
    if target_lang == "id" or not docs:
        return docs

    texts = [(d.get("text") or "")[:2000] for d in docs]
    numbered = "\n\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
    prompt = (
        "Translate each numbered passage below from Indonesian to English.\n"
        "Keep legal terminology precise and faithful. Do not omit any part.\n"
        "Return ONLY a JSON object mapping the number to its translation, "
        'e.g. {"1": "...", "2": "..."}.\n\n'
        f"{numbered}"
    )

    try:
        resp = requests.post(
            f"{LLM_BASE}/chat/completions",
            json={
                "model": LLM_MODEL,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional legal translator for "
                            "Indonesian financial regulations (OJK/BI)."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "Content-Type": "application/json",
            },
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        mapping = json.loads(content)
    except Exception as e:  # noqa: BLE001
        print(f"  [translate] fallback to original ({e})", flush=True)
        return docs

    translated = []
    for i, d in enumerate(docs, 1):
        dd = dict(d)
        if str(i) in mapping and mapping[str(i)]:
            dd["_original"] = d.get("text", "")
            dd["text"] = mapping[str(i)]
        translated.append(dd)
    return translated


# ------------------------------------------------------------- CLI demo
def _main() -> int:
    from app.rag_engine import RagEngine

    args = sys.argv[1:]
    version = "v1"
    if args and args[-1] in ("v1", "v2"):
        version = args.pop()
    if not args:
        print("Usage: python -m app.llm_flow \"<query>\" [v1|v2] [en|id]")
        return 1
    language = "en"
    if args and args[-1] in ("en", "id"):
        language = args.pop()
    query = " ".join(args)

    engine = RagEngine()
    docs = engine.retrieve(query)
    result = answer(query, docs, prompt_version=version, language=language)
    print(f"\n=== Prompt {version.upper()} / lang {language} ===")
    print(result["answer"])
    print(f"\n--- model: {result['model']} | tokens: {result['usage'].get('total_tokens')}")
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
