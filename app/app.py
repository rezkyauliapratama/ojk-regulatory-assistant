"""OJK Regulatory Intelligence Assistant — Streamlit UI.

Login (password) -> chat -> answer with citations -> thumbs feedback.
Optional Nyawa memory layer for cross-session context.

Run: streamlit run app/app.py
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st

from conversations import init_db, log_conversation, set_feedback
from llm_flow import answer
from memory_layer import MemoryLayer
from rag_engine import RagEngine

APP_PASSWORD = os.environ.get("APP_PASSWORD", "admin")

st.set_page_config(page_title="OJK Regulatory Assistant", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------- auth
def check_login() -> bool:
    if "authed" not in st.session_state:
        st.session_state.authed = False
    return st.session_state.authed


def login() -> None:
    st.title("🏦 OJK Regulatory Intelligence Assistant")
    st.caption("LLM Zoomcamp 2026 — Final Project")
    pwd = st.text_input("Password", type="password", key="login_pwd")
    if st.button("Masuk"):
        if pwd == APP_PASSWORD:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Password salah.")


# ---------------------------------------------------------------- helpers
@st.cache_resource
def get_engine() -> RagEngine:
    return RagEngine()


@st.cache_resource
def get_memory() -> MemoryLayer:
    return MemoryLayer()


def run_query(query: str, version: str) -> tuple[str, list[dict], int | None]:
    """Retrieve -> answer -> log. Returns (answer, docs, conv_id)."""
    engine = get_engine()
    docs = engine.retrieve(query)
    result = answer(query, docs, prompt_version=version)
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


def render_citation(docs: list[dict]) -> None:
    st.subheader(f"📄 Sumber ({len(docs)})")
    for i, d in enumerate(docs, 1):
        pasal = f"Pasal {d['pasal']}" if d.get("pasal") else "Umum"
        with st.expander(f"{i}. {d['doc_id']} — {pasal}"):
            st.write(d["text"])


def render_feedback(conv_id: int | None) -> None:
    if conv_id is None:
        return
    col1, col2 = st.columns(2)
    if col1.button("👍", key=f"up_{conv_id}", help="Jawaban membantu"):
        set_feedback(conv_id, "up")
        st.success("Terima kasih atas feedbacknya!")
    if col2.button("👎", key=f"down_{conv_id}", help="Jawaban kurang tepat"):
        set_feedback(conv_id, "down")
        st.warning("Feedback dicatat, kami akan perbaiki.")


# ---------------------------------------------------------------- main
def main() -> None:
    if not check_login():
        login()
        return

    st.title("🏦 OJK Regulatory Intelligence Assistant")
    st.caption("RAG atas 15 regulasi OJK/BI — hybrid search + Jina reranker + gpt-5.4-mini")

    # sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        version = st.radio("Prompt version", ["v1 (citasi ketat)", "v2 (terstruktur)"], index=0)
        st.caption("v1 dipilih default — menang LLM-as-a-Judge (3.95 vs 3.69)")
        prompt_version = "v1" if version.startswith("v1") else "v2"
        use_memory = st.checkbox("Gunakan memori sesi (Nyawa)", value=False)
        if st.button("Logout"):
            st.session_state.authed = False
            st.rerun()

    # chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("docs"):
                with st.expander(f"📄 Sumber ({len(msg['docs'])})"):
                    for i, d in enumerate(msg["docs"], 1):
                        pasal = f"Pasal {d['pasal']}" if d.get("pasal") else "Umum"
                        st.markdown(f"**{i}. {d['doc_id']} — {pasal}**")
                        st.write(d["text"][:400] + ("..." if len(d["text"]) > 400 else ""))

    # query input
    query = st.chat_input("Tanya tentang regulasi OJK/BI... (mis. 'ketentuan QRIS?')")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Mencari di 15 regulasi..."):
                try:
                    answer_text, docs, conv_id = run_query(query, prompt_version)
                except Exception as e:  # noqa: BLE001
                    answer_text = f"⚠️ Error: {e}"
                    docs, conv_id = [], None

                # optional memory: store Q&A + show related past Q&A
                memory_note = ""
                memory = get_memory()
                if use_memory and memory.available:
                    memory.store(
                        content=f"Q: {query}\nA: {answer_text[:500]}",
                        namespace="ojk_qa",
                        type_="chat",
                    )
                    related = memory.recall(query)
                    if related:
                        memory_note = f"\n\n📌 *Konteks dari sesi sebelumnya:* {len(related)} percakapan terkait."

                st.markdown(answer_text + memory_note)
                if docs:
                    with st.expander(f"📄 Sumber ({len(docs)})"):
                        for i, d in enumerate(docs, 1):
                            pasal = f"Pasal {d['pasal']}" if d.get("pasal") else "Umum"
                            st.markdown(f"**{i}. {d['doc_id']} — {pasal}**")
                            st.write(d["text"][:400] + ("..." if len(d["text"]) > 400 else ""))
                    render_feedback(conv_id)

        st.session_state.messages.append({"role": "assistant", "content": answer_text, "docs": docs})


if __name__ == "__main__":
    init_db()
    main()
