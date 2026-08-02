"""Phase 5 — LLM evaluation: LLM-as-a-Judge comparing prompt versions v1 vs v2.

For each ground-truth question:
- retrieve docs (hybrid) -> generate answer with prompt v1 and v2
- judge LLM scores both answers (relevance, groundedness, completeness, citations)
Output: evaluation/results/llm_eval.json with per-version mean scores.

Usage: python scripts/eval_llm.py [--sample 10]
"""

import json
import os
import pathlib
import sys
import time

import requests
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.llm_flow import answer  # noqa: E402
from app.rag_engine import RagEngine  # noqa: E402

load_dotenv(ROOT / ".env")

GT_FILE = ROOT / "evaluation" / "ground_truth.json"
OUT_FILE = ROOT / "evaluation" / "results" / "llm_eval.json"
LLM_BASE = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")

JUDGE_SYSTEM = (
    "You are an evaluator for a RAG system answering questions about Indonesian "
    "financial regulations (OJK/BI). Score the assistant answer on 4 criteria, "
    "each 1-5:\n"
    "1. relevance: does it answer the question directly?\n"
    "2. groundedness: is every claim supported by the provided passages?\n"
    "3. completeness: does it cover all key points in the passages?\n"
    "4. citations: are citations (document + pasal) present and correct?\n"
    'Return JSON: {"relevance": n, "groundedness": n, "completeness": n, '
    '"citations": n, "notes": "short"}'
)


def judge(query: str, answer_text: str, passages: str) -> dict:
    resp = requests.post(
        f"{LLM_BASE}/chat/completions",
        json={
            "model": LLM_MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION: {query}\n\nPASSAGES:\n{passages}\n\n"
                        f"ASSISTANT ANSWER:\n{answer_text}\n\nScore the answer."
                    ),
                },
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
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"relevance": 0, "groundedness": 0, "completeness": 0, "citations": 0, "notes": content[:100]}


def main() -> int:
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 10
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))[:sample]
    print(f"Evaluating {len(gt)} questions x 2 prompt versions\n")

    engine = RagEngine()
    scores = {"v1": [], "v2": []}
    details: list[dict] = []

    for item in gt:
        q = item["question"]
        docs = engine.retrieve(q)
        passages = "\n\n".join(f"[{i}] {d['doc_id']} Pasal {d['pasal'] or '-'}: {d['text'][:600]}" for i, d in enumerate(docs, 1))
        row = {"question": q, "doc_id": item["doc_id"]}
        for version in ("v1", "v2"):
            try:
                result = answer(q, docs, prompt_version=version)
                j = judge(q, result["answer"], passages)
                row[version] = j
                scores[version].append(j)
                print(f"  [{version}] {q[:45]} -> rel={j.get('relevance')} grd={j.get('groundedness')} "
                      f"cmp={j.get('completeness')} cit={j.get('citations')}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [{version}] {q[:45]} -> ERROR: {e}", flush=True)
                scores[version].append({"relevance": 0, "groundedness": 0, "completeness": 0, "citations": 0})
            time.sleep(0.3)

    engine.close()

    results = {}
    for version, vals in scores.items():
        n = len(vals)
        mean = {k: round(sum(v.get(k, 0) for v in vals) / n, 2) for k in ("relevance", "groundedness", "completeness", "citations")}
        mean["overall"] = round(sum(mean.values()) / 4, 2)
        results[version] = {"mean": mean, "n": n}
        print(f"\n{version.upper()} mean: {mean}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"results": results, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
