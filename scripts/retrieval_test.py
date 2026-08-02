"""Phase 2 — retrieval smoke test: 10 queries, check top-1 relevance.

Run: python scripts/retrieval_test.py
Prints per-query top results with rerank scores.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.rag_engine import RagEngine  # noqa: E402

QUERIES = [
    "Apa kewajiban bank dalam penerapan tata kelola kecerdasan artifisial?",
    "Bagaimana ketentuan keamanan siber untuk bank umum?",
    "Apa syarat menjadi penyelenggara inovasi teknologi sektor keuangan (ITSK)?",
    "Bagaimana ketentuan KPMM untuk bank umum?",
    "Apa itu BI-FAST dan bagaimana ketentuannya?",
    "Bagaimana standar QRIS ditetapkan?",
    "Apa kewajiban bank dalam pelindungan konsumen?",
    "Bagaimana ketentuan transaksi valuta asing?",
    "Apa itu penyedia jasa pembayaran (PJP)?",
    "Bagaimana strategi anti fraud untuk lembaga jasa keuangan?",
]


def main() -> int:
    engine = RagEngine()
    for q in QUERIES:
        try:
            results = engine.retrieve(q, rewrite=False)
            print(f"\nQ: {q}")
            for r in results[:3]:
                print(f"   [{r['rerank_score']:.3f}] {r['doc_id']} | Pasal {r['pasal'] or '-'}")
        except Exception as e:  # noqa: BLE001
            print(f"\nQ: {q}\n   ERROR: {e}")
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
