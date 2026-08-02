"""Phase 1 — fetch regulatory PDFs from OJK/BI.

Handles OJK WAF (requires browser User-Agent) + self-signed SSL.
Downloads all documents in data/sources.yaml to data/pdfs/.
"""

import pathlib
import sys
import time

import requests
import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = ROOT / "data" / "sources.yaml"
PDF_DIR = ROOT / "data" / "pdfs"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
HEADERS = {"User-Agent": UA}


def load_manifest() -> list[dict]:
    with open(SOURCES, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["documents"]


def fetch_pdf(doc: dict, timeout: int = 60) -> pathlib.Path:
    """Download one PDF. Returns saved path. Raises on failure."""
    url = doc["url"]
    if not url:
        raise ValueError(f"No URL for {doc['id']} — resolve it first")
    out = PDF_DIR / doc["filename"]
    if out.exists() and out.stat().st_size > 100_000:
        print(f"  [skip] {doc['id']} already exists ({out.stat().st_size} bytes)")
        return out

    resp = requests.get(url, headers=HEADERS, verify=False, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    # WAF/HTML guard: a PDF must start with %PDF
    if not resp.content.startswith(b"%PDF"):
        raise RuntimeError(f"Not a PDF (got {len(resp.content)} bytes, starts {resp.content[:20]!r})")
    out.write_bytes(resp.content)
    print(f"  [ok]   {doc['id']} -> {out.name} ({len(resp.content)} bytes)")
    return out


def main() -> int:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_manifest()
    print(f"Manifest: {len(docs)} documents -> {PDF_DIR}")

    ok, fail = 0, 0
    for doc in docs:
        try:
            fetch_pdf(doc)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {doc['id']}: {e}")
            fail += 1
        time.sleep(1)  # polite crawl

    print(f"\nDone: {ok} ok, {fail} failed.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
