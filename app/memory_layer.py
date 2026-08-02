"""Nyawa memory layer — conversation memory via Nyawa MCP CLI (optional bonus).

Wraps the Nyawa binary (MCP over stdio) for storing Q&A pairs and recalling
related past conversations. If the Nyawa binary is not present, the layer
degrades gracefully (memory is skipped, app keeps working).

Requires:
- nyawa binary (public repo: github.com/rezkyauliapratama/nyawa) at ./nyawa/nyawa
- NYAWA_DB path in .env (default ./data/nyawa_memory.db)
"""

import json
import os
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


class MemoryLayer:
    def __init__(self, binary: str | None = None, db: str | None = None) -> None:
        # Resolve relative paths against the project root (ROOT), not the
        # current working directory — streamlit's CWD differs in Docker.
        def _resolve(p: str) -> pathlib.Path:
            path = pathlib.Path(p)
            return path if path.is_absolute() else ROOT / path

        self.binary = str(_resolve(binary or os.environ.get("NYAWA_BINARY", "nyawa/nyawa")))
        self.db = str(_resolve(db or os.environ.get("NYAWA_DB", "data/nyawa_memory.db")))
        self.available = pathlib.Path(self.binary).exists()

    def _call(self, method: str, params: dict) -> dict | None:
        if not self.available:
            return None
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": method, "arguments": params},
            }
            proc = subprocess.run(
                [self.binary, "mcp", self.db],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=30,
            )
            out = json.loads(proc.stdout)
            result = out.get("result", {})
            content = result.get("content", [])
            if content and isinstance(content[0], dict):
                return json.loads(content[0].get("text", "{}"))
            return result
        except Exception:  # noqa: BLE001 — memory is best-effort
            return None

    def store(self, content: str, namespace: str = "ojk_qa", type_: str = "chat") -> str | None:
        """Store a Q&A pair. Returns memory id or None."""
        res = self._call("nyawa_store", {"content": content, "namespace": namespace, "type": type_})
        if res and isinstance(res, dict):
            return res.get("id")
        return None

    def recall(self, query: str, namespace: str = "ojk_qa", limit: int = 3) -> list[dict]:
        """Recall related past Q&A. Returns list of {content, score} or [].

        Normalizes Nyawa's capitalized keys (ID/Content/Score/Type/
        Namespace/CreatedAt) to lowercase for the app layer.
        """
        res = self._call(
            "nyawa_recall",
            {"query": query, "namespace": namespace, "limit": limit},
        )
        if not (res and isinstance(res, dict)):
            return []
        results = res.get("results", []) or res.get("memories", [])
        normalized = []
        for item in results:
            if not isinstance(item, dict):
                continue
            normalized.append({k.lower(): v for k, v in item.items()})
        return normalized
