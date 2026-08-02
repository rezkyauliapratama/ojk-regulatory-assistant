"""Phase 1 — extract text from PDFs (PyMuPDF) to data/extracted/."""

import pathlib
import sys

import fitz  # PyMuPDF

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PDF_DIR = ROOT / "data" / "pdfs"
OUT_DIR = ROOT / "data" / "extracted"


def extract(pdf_path: pathlib.Path) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs")
    for pdf in pdfs:
        text = extract(pdf)
        out = OUT_DIR / f"{pdf.stem}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"  {pdf.name}: {len(text)} chars -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
