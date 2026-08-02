"""Phase 1 — section-based chunking per BAB/Pasal (legal-optimized).

Handles multiple regulatory formats:
1. Pasal headings: "Pasal 5", "Pasal 5 Ayat (2)"  (POJK/PBI/UU)
2. Roman sections: "I.", "II.", "I.a."              (SEOJK)
3. "BAB N" / "BAB I - TITLE"                        (UU, P2SK)
4. Fallback: paragraph-based chunking               (guidance docs, bilingual)
"""

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
EXTRACTED_DIR = ROOT / "data" / "extracted"
OUT_DIR = ROOT / "data" / "chunks"

# Pasal headings: "Pasal 5", "Pasal 5 Ayat (2)", "Pasal 15A"
PASAL_RE = re.compile(r"^Pasal\s+(\d+[A-Za-z]?)(?:\s+Ayat\s*\((\d+)\))?", re.IGNORECASE | re.MULTILINE)
# BAB headings: "BAB I", "BAB 1", "BAB II - KETENTUAN UMUM"
BAB_RE = re.compile(r"^BAB\s+([IVXLCDM]+|\d+)(?:\s*[-–—]\s*(.*))?$", re.IGNORECASE | re.MULTILINE)
# Roman sections: "I.", "II.", "I.a.", "III." at line start
ROMAN_RE = re.compile(r"^([IVXLCDM]+)\.\s*$", re.MULTILINE)
# Numbered sections: "1. ", "2. " at line start (guidance docs)
NUM_SECTION_RE = re.compile(r"^(\d{1,2})\.\s+", re.MULTILINE)

MAX_CHUNK_CHARS = 2500


def _emit(chunks: list, doc_id: str, pasal: str, ayat: str, text: str) -> None:
    text = text.strip()
    if len(text) < 50:
        return
    # split oversized pasal-level chunks further (some pasal are very long)
    while len(text) > MAX_CHUNK_CHARS:
        split_at = text.rfind("\n", 0, MAX_CHUNK_CHARS)
        if split_at < MAX_CHUNK_CHARS // 2:
            split_at = MAX_CHUNK_CHARS
        chunks.append({"doc_id": doc_id, "pasal": pasal, "ayat": ayat, "text": text[:split_at].strip()})
        text = text[split_at:].strip()
    if text:
        chunks.append({"doc_id": doc_id, "pasal": pasal, "ayat": ayat, "text": text})


def chunk_document(text: str, doc_id: str) -> list[dict]:
    chunks: list[dict] = []

    # Strategy 1: Pasal-level (POJK/PBI/UU — primary format)
    pasal_matches = list(PASAL_RE.finditer(text))
    if len(pasal_matches) >= 2:
        for i, m in enumerate(pasal_matches):
            start = m.start()
            end = pasal_matches[i + 1].start() if i + 1 < len(pasal_matches) else len(text)
            _emit(chunks, doc_id, m.group(1), m.group(2) or "", text[start:end])
        return chunks

    # Strategy 2: BAB-level
    bab_matches = list(BAB_RE.finditer(text))
    if len(bab_matches) >= 2:
        for i, m in enumerate(bab_matches):
            start = m.start()
            end = bab_matches[i + 1].start() if i + 1 < len(bab_matches) else len(text)
            _emit(chunks, doc_id, "", "", text[start:end])
        return chunks

    # Strategy 3: Roman sections (SEOJK: "I.", "II.", "I.a.")
    roman_matches = list(ROMAN_RE.finditer(text))
    if len(roman_matches) >= 3:
        for i, m in enumerate(roman_matches):
            start = m.start()
            end = roman_matches[i + 1].start() if i + 1 < len(roman_matches) else len(text)
            _emit(chunks, doc_id, "", "", text[start:end])
        return chunks

    # Strategy 4: numbered sections (guidance / bilingual docs)
    num_matches = list(NUM_SECTION_RE.finditer(text))
    if len(num_matches) >= 3:
        for i, m in enumerate(num_matches):
            start = m.start()
            end = num_matches[i + 1].start() if i + 1 < len(num_matches) else len(text)
            _emit(chunks, doc_id, "", "", text[start:end])
        return chunks

    # Strategy 5: paragraph fallback (split on double newline)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 80]
    if paragraphs:
        buf = ""
        for p in paragraphs:
            if len(buf) + len(p) > MAX_CHUNK_CHARS and buf:
                chunks.append({"doc_id": doc_id, "pasal": "", "ayat": "", "text": buf})
                buf = ""
            buf += p + "\n"
        if buf.strip():
            chunks.append({"doc_id": doc_id, "pasal": "", "ayat": "", "text": buf.strip()})
        return chunks

    # Last resort: whole doc (truncated)
    chunks.append({"doc_id": doc_id, "pasal": "", "ayat": "", "text": text.strip()[:MAX_CHUNK_CHARS]})
    return chunks


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for txt in sorted(EXTRACTED_DIR.glob("*.txt")):
        text = txt.read_text(encoding="utf-8")
        doc_id = txt.stem
        chunks = chunk_document(text, doc_id)
        out = OUT_DIR / f"{doc_id}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        total += len(chunks)
        print(f"  {doc_id}: {len(chunks)} chunks")
    print(f"\nTotal chunks: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
