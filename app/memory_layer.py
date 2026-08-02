"""Nyawa memory layer (Phase 4) — placeholder.

Wraps the Nyawa MCP stdio interface for conversation memory + feedback.
"""


class MemoryLayer:
    def __init__(self, binary: str = "./nyawa/nyawa", db: str = "./data/nyawa_memory.db"):
        self.binary = binary
        self.db = db

    def store(self, content: str, namespace: str = "hermes", type_: str = "chat") -> str:
        raise NotImplementedError("Nyawa memory layer lands in Phase 4.")
