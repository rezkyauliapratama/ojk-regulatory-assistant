"""Regulatory Intelligence Assistant — Streamlit UI.

Login (password) -> chat -> answer with citations -> thumbs feedback.
Bilingual UI (default English, toggle to Bahasa Indonesia):
- UI strings follow the selected language
- LLM answers in the selected language
- Cited chunk texts are shown translated to English when UI is English

Optional Nyawa memory layer for cross-session context.

Run: streamlit run app/app.py
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st
import yaml

from conversations import init_db, log_conversation, log_nyawa_recall, set_feedback
from llm_flow import answer, translate_docs
from memory_layer import MemoryLayer
from rag_engine import RagEngine

APP_PASSWORD = os.environ.get("APP_PASSWORD", "admin")
ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- i18n
T = {
    "en": {
        "page_title": "Regulatory Intelligence Assistant",
        "login_title": "🏦 Regulatory Intelligence Assistant",
        "login_caption": "LLM Zoomcamp 2026 — Final Project",
        "password_label": "Password",
        "login_btn": "Login",
        "login_error": "Incorrect password.",
        "welcome_caption": "RAG over 15 OJK/BI regulations — hybrid search + reranking",
        "sidebar_settings": "⚙️ Settings",
        "sidebar_language": "Language",
        "lang_en": "English",
        "lang_id": "Bahasa Indonesia",
        "sidebar_prompt": "Answer style",
        "prompt_v1": "Standard (cited)",
        "prompt_v2": "Structured",
        "sidebar_memory": "Use session memory (Nyawa)",
        "logout": "Logout",
        "sources": "📄 Sources ({n})",
        "article": "Article",
        "general": "General",
        "chat_placeholder": "Ask about OJK/BI regulations... (e.g. 'QRIS requirements?')",
        "searching": "Searching 15 regulations...",
        "error_prefix": "⚠️ Error",
        "memory_note": "📌 *Context from previous sessions:* {n} related conversation(s).",
        "feedback_helpful": "Helpful",
        "feedback_not_helpful": "Not helpful",
        "feedback_thanks": "Thank you for your feedback!",
        "feedback_noted": "Feedback recorded, we will improve.",
        "original_note": "Original (Indonesian): {text}",
        "logged_out": "You have been logged out.",
        "kb_expander": "📚 Knowledge Base ({n} regulations)",
        "kb_intro": "The assistant answers from these OJK/BI regulations:",
        "kb_cat_ai": "AI / Technology",
        "kb_cat_payment": "Payment Systems",
        "kb_cat_banking": "Banking",
        "kb_cat_other": "Other",
        "memory_title": "🧠 Session memory (Nyawa)",
        "memory_available": "Available",
        "memory_missing": "Not available — Nyawa binary not found at `{path}`. Download it from github.com/rezkyauliapratama/nyawa/releases and place it in `nyawa/nyawa`.",
        "memory_related": "🧠 Related past conversations:",
        "memory_empty": "No related past conversations found yet — start chatting and past Q&A will be recalled here.",
        "memory_stored": "Q&A stored to memory.",
    },
    "id": {
        "page_title": "Asisten Intelijen Regulasi",
        "login_title": "🏦 Asisten Intelijen Regulasi",
        "login_caption": "LLM Zoomcamp 2026 — Proyek Akhir",
        "password_label": "Kata Sandi",
        "login_btn": "Masuk",
        "login_error": "Kata sandi salah.",
        "welcome_caption": "RAG atas 15 regulasi OJK/BI — hybrid search + reranking",
        "sidebar_settings": "⚙️ Pengaturan",
        "sidebar_language": "Bahasa",
        "lang_en": "English",
        "lang_id": "Bahasa Indonesia",
        "sidebar_prompt": "Gaya jawaban",
        "prompt_v1": "Standar (dengan kutipan)",
        "prompt_v2": "Terstruktur",
        "sidebar_memory": "Gunakan memori sesi (Nyawa)",
        "logout": "Keluar",
        "sources": "📄 Sumber ({n})",
        "article": "Pasal",
        "general": "Umum",
        "chat_placeholder": "Tanya tentang regulasi OJK/BI... (mis. 'ketentuan QRIS?')",
        "searching": "Mencari di 15 regulasi...",
        "error_prefix": "⚠️ Error",
        "memory_note": "📌 *Konteks dari sesi sebelumnya:* {n} percakapan terkait.",
        "feedback_helpful": "Jawaban membantu",
        "feedback_not_helpful": "Jawaban kurang tepat",
        "feedback_thanks": "Terima kasih atas feedbacknya!",
        "feedback_noted": "Feedback dicatat, kami akan perbaiki.",
        "original_note": "Asli (Bahasa Indonesia): {text}",
        "logged_out": "Kamu telah keluar.",
        "kb_expander": "📚 Basis Pengetahuan ({n} regulasi)",
        "kb_intro": "Asisten menjawab dari regulasi OJK/BI berikut:",
        "kb_cat_ai": "AI / Teknologi",
        "kb_cat_payment": "Sistem Pembayaran",
        "kb_cat_banking": "Perbankan",
        "kb_cat_other": "Lainnya",
        "memory_title": "🧠 Memori sesi (Nyawa)",
        "memory_available": "Tersedia",
        "memory_missing": "Tidak tersedia — binary Nyawa tidak ditemukan di `{path}`. Unduh dari github.com/rezkyauliapratama/nyawa/releases dan letakkan di `nyawa/nyawa`.",
        "memory_related": "🧠 Percakapan terkait sebelumnya:",
        "memory_empty": "Belum ada percakapan terkait — mulai ngobrol dan Q&A sebelumnya akan muncul di sini.",
        "memory_stored": "Q&A tersimpan ke memori.",
    },
}


def tr(key: str, lang: str, **fmt) -> str:
    s = T[lang].get(key, T["en"].get(key, key))
    return s.format(**fmt) if fmt else s


# ---------------------------------------------------------------- auth
def check_login() -> bool:
    if "authed" not in st.session_state:
        st.session_state.authed = False
    return st.session_state.authed


def login(lang: str) -> None:
    st.title(tr("login_title", lang))
    st.caption(tr("login_caption", lang))
    pwd = st.text_input(tr("password_label", lang), type="password", key="login_pwd")
    if st.button(tr("login_btn", lang)):
        if pwd == APP_PASSWORD:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error(tr("login_error", lang))


# ---------------------------------------------------------------- helpers
@st.cache_resource
def get_engine() -> RagEngine:
    return RagEngine()


@st.cache_resource
def get_memory() -> MemoryLayer:
    return MemoryLayer()


@st.cache_resource
def get_sources() -> list[dict]:
    """Load the regulation manifest from data/sources.yaml."""
    path = ROOT / "data" / "sources.yaml"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("documents", [])


CATEGORY_LABEL = {
    "ai": "kb_cat_ai",
    "payment": "kb_cat_payment",
    "banking": "kb_cat_banking",
}


def render_kb(lang: str) -> None:
    """Show which documents the RAG knowledge base contains."""
    docs = get_sources()
    if not docs:
        return
    with st.expander(tr("kb_expander", lang, n=len(docs))):
        st.write(tr("kb_intro", lang))
        by_cat: dict[str, list[dict]] = {}
        for d in docs:
            cat = d.get("category", "other")
            by_cat.setdefault(cat, []).append(d)
        for cat, items in by_cat.items():
            label_key = CATEGORY_LABEL.get(cat, "kb_cat_other")
            st.markdown(f"**{tr(label_key, lang)}**")
            for d in items:
                title = d.get("title", "")
                number = d.get("number", "")
                year = d.get("year", "")
                label = f"{number} — {title}" if number else title
                if year:
                    label += f" ({year})"
                st.markdown(f"- {label}")


@st.cache_data(show_spinner=False)
def _translate_cached(texts: tuple[str, ...]) -> tuple[str, ...]:
    """Cache layer for translate_docs: keyed by chunk texts, one call per batch."""
    docs = [{"text": t} for t in texts]
    out = translate_docs(docs, target_lang="en")
    return tuple(d["text"] for d in out)


def translated_texts(texts: list[str], lang: str) -> list[str]:
    if lang == "id" or not texts:
        return texts
    try:
        return list(_translate_cached(tuple(texts)))
    except Exception:
        return texts


def run_query(query: str, version: str, lang: str) -> tuple[str, list[dict], int | None]:
    """Retrieve -> answer -> log. Returns (answer, docs, conv_id)."""
    engine = get_engine()
    docs = engine.retrieve(query)
    result = answer(query, docs, prompt_version=version, language=lang)
    init_db()
    conv_id = log_conversation(
        query=query,
        answer=result["answer"],
        docs=docs,
        prompt_version=version,
        model=result["model"],
        usage_tokens=result.get("usage", {}).get("total_tokens"),
    )
    return result["answer"], docs, conv_id


def render_citation(docs: list[dict], lang: str) -> None:
    st.subheader(tr("sources", lang, n=len(docs)))
    texts = [d["text"] for d in docs]
    translated = translated_texts(texts, lang)
    for i, d in enumerate(docs, 1):
        pasal = f"{tr('article', lang)} {d['pasal']}" if d.get("pasal") else tr("general", lang)
        with st.expander(f"{i}. {d['doc_id']} — {pasal}"):
            st.write(translated[i - 1] if lang == "en" else d["text"])
            if lang == "en" and translated[i - 1] != d["text"]:
                st.caption(tr("original_note", lang, text=d["text"][:200] + ("..." if len(d["text"]) > 200 else "")))


def render_feedback(conv_id: int | None, lang: str) -> None:
    if conv_id is None:
        return
    col1, col2 = st.columns(2)
    if col1.button("👍", key=f"up_{conv_id}", help=tr("feedback_helpful", lang)):
        set_feedback(conv_id, "up")
        st.success(tr("feedback_thanks", lang))
    if col2.button("👎", key=f"down_{conv_id}", help=tr("feedback_not_helpful", lang)):
        set_feedback(conv_id, "down")
        st.warning(tr("feedback_noted", lang))


# ---------------------------------------------------------------- main
def main() -> None:
    if "lang" not in st.session_state:
        st.session_state.lang = "en"  # default English
    lang = st.session_state.lang

    if not check_login():
        login(lang)
        return

    st.title(tr("page_title", lang))
    st.caption(tr("welcome_caption", lang))

    # sidebar
    with st.sidebar:
        st.header(tr("sidebar_settings", lang))
        lang_label = st.selectbox(
            tr("sidebar_language", lang),
            options=["en", "id"],
            index=0 if lang == "en" else 1,
            format_func=lambda x: T[lang]["lang_en"] if x == "en" else T[lang]["lang_id"],
        )
        if lang_label != lang:
            st.session_state.lang = lang_label
            st.rerun()

        version = st.radio(
            tr("sidebar_prompt", lang),
            ["v1", "v2"],
            index=0,
            format_func=lambda x: T[lang]["prompt_v1"] if x == "v1" else T[lang]["prompt_v2"],
        )
        use_memory = st.checkbox(tr("sidebar_memory", lang), value=False)
        # memory availability status
        memory = get_memory()
        if use_memory:
            if memory.available:
                st.success(f"🧠 {tr('memory_available', lang)}")
            elif memory.error:
                st.warning(f"🧠 {tr('memory_missing', lang, path=memory.binary)}\n\n{memory.error}")
            else:
                st.warning(tr("memory_missing", lang, path=str(memory.binary)))
        if st.button(tr("logout", lang)):
            st.session_state.authed = False
            st.rerun()

    # knowledge base scope info
    render_kb(lang)

    # chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("docs"):
                    render_citation(msg["docs"], msg.get("lang", lang))
                if msg.get("conv_id"):
                    render_feedback(msg["conv_id"], msg.get("lang", lang))

    # query input
    query = st.chat_input(tr("chat_placeholder", lang))

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner(tr("searching", lang)):
                try:
                    answer_text, docs, conv_id = run_query(query, version, lang)
                except Exception as e:  # noqa: BLE001
                    answer_text = f"{tr('error_prefix', lang)}: {e}"
                    docs, conv_id = [], None

                # optional memory: store Q&A + show related past Q&A
                memory = get_memory()
                if use_memory and memory.available:
                    mem_id = memory.store(
                        content=f"Q: {query}\nA: {answer_text[:500]}",
                        namespace="rag_qa",
                        type_="chat",
                    )
                    related = memory.recall(query)
                    if related:
                        st.markdown(f"**{tr('memory_related', lang)}**")
                        for r in related[:3]:
                            c = r.get("content", "")
                            if c:
                                st.markdown(f"- {c[:300]}")
                    elif memory.last_error:
                        st.caption(f"⚠️ {memory.last_error}")
                    else:
                        st.caption(tr("memory_empty", lang))
                    # log recall metrics to Postgres for Grafana (best-effort)
                    log_nyawa_recall(query, related)
                    if mem_id is None and memory.last_error:
                        st.caption(f"⚠️ store failed: {memory.last_error}")

                st.markdown(answer_text)
                if docs:
                    render_citation(docs, lang)
                    render_feedback(conv_id, lang)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "docs": docs, "lang": lang, "conv_id": conv_id}
        )


if __name__ == "__main__":
    init_db()
    main()
