"""Nyawa memory layer — conversation memory via Nyawa MCP CLI (optional bonus).

Wraps the Nyawa binary (MCP over stdio) for storing Q&A pairs and recalling
related past conversations. If the Nyawa binary is not present or cannot
execute, the layer degrades gracefully (memory is skipped, app keeps
working) and surfaces a human-readable `error` for the UI.

Requires:
- nyawa binary (public repo: github.com/rezkyauliapratama/nyawa) at ./nyawa/nyawa
- NYAWA_DB path in .env (default ./data/nyawa_memory.db)

Platform note: the binary must run where the app runs. If Streamlit runs
in a Docker container (Linux), a macOS-built binary exists but will not
execute — `available` will be False and `error` explains why.
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
        self.error: str | None = None
        self.last_error: str | None = None

        bin_path = pathlib.Path(self.binary)
        if not bin_path.exists():
            self.available = False
            self.error = f"binary not found at {self.binary}"
            return

        # Existence is not enough — verify it actually executes on this
        # platform (a macOS binary inside a Linux container exists but
        # raises OSError on exec).
        try:
            proc = subprocess.run(
                [self.binary, "version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0 and "nyawa" in (proc.stdout + proc.stderr).lower():
                self.available = True
            else:
                self.available = False
                self.error = (
                    f"binary at {self.binary} exists but does not run "
                    f"(rc={proc.returncode}). Wrong platform? If Streamlit runs "
                    f"in Docker, rebuild with: bash scripts/setup_nyawa.sh --for-docker"
                )
        except OSError as e:
            self.available = False
            self.error = (
                f"binary at {self.binary} exists but cannot execute: {e}. "
                f"Platform mismatch — if Streamlit runs in Docker, rebuild with: "
                f"bash scripts/setup_nyawa.sh --for-docker"
            )
        except Exception as e:  # noqa: BLE001
            self.available = False
            self.error = f"binary check failed: {e}"

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
            if proc.returncode != 0:
                self.last_error = f"{method}: mcp exited rc={proc.returncode}: {proc.stderr.strip()[:200]}"
                return None
            out = json.loads(proc.stdout)
            result = out.get("result", {})
            content = result.get("content", [])
            if content and isinstance(content[0], dict):
                return json.loads(content[0].get("text", "{}"))
            return result
        except Exception as e:  # noqa: BLE001 — memory is best-effort
            self.last_error = f"{method}: {type(e).__name__}: {e}"
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

    def recent_questions(self, namespace: str = "rag_qa", limit: int = 20) -> list[str]:
        """Return stored question strings (most recent first, best-effort).

        Nyawa has no 'list all' MCP tool, so we recall with a generic
        query that matches everything, then parse the 'Q: <question>'
        prefix that app.py stores. Returns [] on any failure.
        """
        results = self.recall("conversation history Q&A", namespace=namespace, limit=limit)
        questions = []
        for r in results:
            content = r.get("content", "") or ""
            first_line = content.split("\n", 1)[0]
            if first_line.startswith("Q: "):
                questions.append(first_line[3:].strip())
        return questions
