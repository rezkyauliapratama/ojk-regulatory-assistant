"""Phase 3 — end-to-end RAG flow: retrieve -> answer -> log conversation.

CLI: python scripts/rag_ask.py "<query>" [v1|v2]
Test batch: python scripts/rag_ask.py --test
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.conversations import init_db, log_conversation  # noqa: E402
from app.llm_flow import answer  # noqa: E402
from app.rag_engine import RagEngine  # noqa: E402

TEST_QUERIES = [
    ("Bagaimana kewajiban bank dalam penerapan tata kelola kecerdasan artifisial?", "v1"),
    ("Bagaimana ketentuan keamanan siber untuk bank umum?", "v1"),
    ("Apa syarat menjadi penyelenggara inovasi teknologi sektor keuangan?", "v2"),
    ("Bagaimana standar QRIS ditetapkan?", "v2"),
    ("Bagaimana strategi anti fraud untuk lembaga jasa keuangan?", "v1"),
]


def ask(query: str, version: str = "v1", log: bool = True) -> dict:
    engine = RagEngine()
    docs = engine.retrieve(query)
    rewritten = query
    result = answer(query, docs, prompt_version=version)
    conv_id = None
    if log:
        init_db()
        conv_id = log_conversation(
            query=query,
            answer=result["answer"],
            docs=docs,
            rewritten_query=rewritten,
            prompt_version=version,
            model=result["model"],
            usage_tokens=result.get("usage", {}).get("total_tokens"),
        )
    engine.close()
    return {"result": result, "docs": docs, "conv_id": conv_id}


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--test":
        print(f"Running {len(TEST_QUERIES)} end-to-end tests...\n")
        for q, v in TEST_QUERIES:
            try:
                out = ask(q, v)
                print(f"Q: {q} [{v.upper()}] -> conv_id={out['conv_id']}")
                print(f"   {out['result']['answer'][:200].replace(chr(10), ' ')}...")
                print(f"   cited docs: {[d['doc_id'] for d in out['docs'][:3]]}\n")
            except Exception as e:  # noqa: BLE001
                print(f"Q: {q} -> ERROR: {e}\n")
        return 0

    if not args:
        print('Usage: python scripts/rag_ask.py "<query>" [v1|v2] | --test')
        return 1
    version = "v1"
    if args[-1] in ("v1", "v2"):
        version = args.pop()
    query = " ".join(args)
    out = ask(query, version)
    print(f"\n=== Prompt {version.upper()} (conv_id={out['conv_id']}) ===")
    print(out["result"]["answer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
