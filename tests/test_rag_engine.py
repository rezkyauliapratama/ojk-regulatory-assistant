"""Unit tests for pure functions (no API keys, no DB, no network).

Run: pytest tests/ -v
"""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag_engine import fts_query  # noqa: E402


def test_fts_query_basic():
    q = fts_query("QRIS requirements for payment providers")
    assert "qris" in q
    assert "requirements" in q
    assert "|" in q  # OR semantics


def test_fts_query_mixed_id():
    # Indonesian stopwords removed; 'bank' is also a stopword in this list
    q = fts_query("apa itu qris dan bank")
    assert q == "qris"


def test_fts_query_all_stopwords_falls_back():
    # all-stopword query falls back to the raw query (by design)
    q = fts_query("apa yang untuk dan")
    assert q == "apa yang untuk dan"


def test_fts_query_short_words_falls_back():
    q = fts_query("qr is ok")
    assert q == "qr is ok"


def test_fts_query_empty_falls_back_to_raw():
    q = fts_query("")
    assert q == ""


def test_fts_query_english_only_regex():
    # non-alphanumeric tokens are stripped
    q = fts_query("POJK 11/2022 (IT risk)!")
    assert "/" not in q
    assert "pojk" in q
    assert "risk" in q
