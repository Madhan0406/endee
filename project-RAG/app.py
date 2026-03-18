"""Streamlit RAG PDF Chatbot — powered by Endee + Groq."""

import html
import time

import streamlit as st

from config import ENDEE_URL, INDEX_NAME, GROQ_API_KEY
from endee_client import EndeeClient
from pipeline import (
    load_model,
    extract_text_from_pdf,
    chunk_text,
    ingest,
    search,
    generate_answer,
)

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon=":material/smart_toy:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── styling ──────────────────────────────────────────────────

def inject_theme():
    st.markdown(
        """
<style>
:root{--bg:#0b1220;--surface:#0f1b31;--card:#16243a;--border:#2b3b57;
--text:#f1f5f9;--muted:#cbd5e1;--accent:#22c55e;--accent-alt:#3b82f6}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
background:radial-gradient(1100px 520px at 0% 0%,#1e3a8a 0%,var(--bg) 42%) no-repeat,
linear-gradient(180deg,#0a1120,var(--bg));color:var(--text);
font-family:"Inter","Segoe UI",sans-serif}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1427,#0c172b);
border-right:1px solid var(--border)}
[data-testid="stSidebar"] *{color:var(--text)!important}
[data-testid="stHeader"]{background:transparent}
[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(22,36,58,.92);
border:1px solid var(--border);border-radius:16px;
box-shadow:0 10px 30px rgba(2,6,23,.45)}
[data-testid="stChatMessage"]{background:rgba(15,27,49,.75);
border:1px solid var(--border);border-radius:14px;padding:.5rem .75rem}
.stButton>button{width:100%;border-radius:12px;border:0;color:#fff;
background:linear-gradient(90deg,var(--accent-alt),var(--accent));
box-shadow:0 8px 20px rgba(34,197,94,.25);font-weight:600}
.stButton>button:hover{transform:translateY(-1px);
box-shadow:0 12px 30px rgba(99,102,241,.35)}
.header-wrap{padding:8px 0 16px}
.header-title{font-size:2rem;font-weight:700;margin:0;color:var(--text)}
.header-subtitle{margin:4px 0 12px;color:var(--muted);font-size:.95rem}
.header-divider{height:2px;border-radius:999px;
background:linear-gradient(90deg,var(--accent-alt),var(--accent));opacity:.85}
.status-item{display:flex;align-items:center;justify-content:space-between;
padding:8px 10px;margin-bottom:8px;border:1px solid var(--border);
border-radius:10px;background:rgba(15,23,42,.75)}
.status-left{display:flex;align-items:center;gap:8px;font-size:.88rem}
.dot{width:9px;height:9px;border-radius:50%}
.pill{font-size:.72rem;font-weight:600;padding:2px 8px;border-radius:999px}
.pill-ok{color:#bbf7d0;background:rgba(22,163,74,.2);border:1px solid rgba(34,197,94,.5)}
.pill-off{color:#fecaca;background:rgba(220,38,38,.2);border:1px solid rgba(239,68,68,.5)}
.source-card{border:1px solid var(--border);border-radius:12px;
padding:10px 12px;margin-bottom:10px;background:rgba(15,23,42,.7)}
.source-head{display:flex;justify-content:space-between;font-size:.9rem;
font-weight:600;margin-bottom:4px}
.source-doc{color:#cbd5e1;font-size:.82rem;margin-bottom:4px}
.source-text{color:var(--muted);font-size:.84rem;line-height:1.4}
</style>""",
        unsafe_allow_html=True,
    )


# ── state ────────────────────────────────────────────────────

def init_state():
    defaults = {
        "docs_indexed": 0,
        "chunks_stored": 0,
        "last_query_time": None,
        "chat_history": [],
        "last_sources": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── small helpers ────────────────────────────────────────────

def status_badge(label: str, ok: bool) -> str:
    color = "#22c55e" if ok else "#ef4444"
    cls = "pill-ok" if ok else "pill-off"
    txt = "OK" if ok else "OFF"
    return (
        f"<div class='status-item'><div class='status-left'>"
        f"<span class='dot' style='background:{color}'></span>"
        f"<span>{html.escape(label)}</span></div>"
        f"<span class='pill {cls}'>{txt}</span></div>"
    )


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_client():
    return EndeeClient()


# ── layout ───────────────────────────────────────────────────

def render_header():
    st.markdown(
        '<div class="header-wrap">'
        '<p class="header-title">AI Document Assistant</p>'
        '<p class="header-subtitle">RAG Pipeline — Endee Vector DB + Groq LLM</p>'
        '<div class="header-divider"></div></div>',
        unsafe_allow_html=True,
    )


def render_sidebar(endee_ok: bool):
    with st.sidebar:
        st.markdown("## System Status")
        st.markdown(status_badge("Vector DB", endee_ok), unsafe_allow_html=True)
        st.markdown(
            status_badge("LLM API", bool(GROQ_API_KEY)), unsafe_allow_html=True
        )
        st.markdown(
            status_badge("Documents", st.session_state.docs_indexed > 0),
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("## Settings")
        top_k = st.slider("Top-K results", 1, 8, 3)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        chunk_size = st.slider("Chunk size (words)", 200, 900, 500, 100)

        st.markdown("---")
        st.markdown("## Index Management")
        if st.button("Initialize / Reset Index"):
            client = get_client()
            client.delete_index()
            ok = client.create_index()
            if ok:
                st.session_state.docs_indexed = 0
                st.session_state.chunks_stored = 0
                st.success("Index created.")
            else:
                st.error("Failed to create index.")

        st.markdown("---")
        st.caption(f"Index: **{INDEX_NAME}**")
        st.caption(f"Endee: `{ENDEE_URL}`")

    return {"top_k": top_k, "temperature": temperature, "chunk_size": chunk_size}


def render_metrics():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Documents", st.session_state.docs_indexed)
    with c2:
        st.metric("Chunks", st.session_state.chunks_stored)
    with c3:
        qt = st.session_state.last_query_time
        st.metric("Query Time", f"{qt:.2f}s" if isinstance(qt, float) else "—")


def render_sources(sources):
    with st.container(border=True):
        st.subheader("Retrieved Sources")
        if not sources:
            st.caption("No sources yet.")
            return
        with st.expander("View Sources", expanded=True):
            for i, src in enumerate(sources, 1):
                score = src.get("score")
                score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                doc = html.escape(str(src.get("doc", "Document")))
                snippet = html.escape(str(src.get("text", ""))[:400])
                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="source-head"><span>Source {i}</span>'
                    f"<span>Score: {score_text}</span></div>"
                    f'<div class="source-doc">{doc}</div>'
                    f'<div class="source-text">{snippet}</div></div>',
                    unsafe_allow_html=True,
                )


# ── main ─────────────────────────────────────────────────────

def main():
    inject_theme()
    init_state()

    client = get_client()
    endee_ok = client.is_healthy()
    settings = render_sidebar(endee_ok)

    render_header()
    render_metrics()

    # ── Upload ───────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Upload Document")
        uploaded = st.file_uploader(
            "Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=False
        )

        if uploaded:
            if uploaded.name.lower().endswith(".pdf"):
                file_text = extract_text_from_pdf(uploaded)
            else:
                file_text = uploaded.getvalue().decode("utf-8", errors="replace")

            est_chunks = len(chunk_text(file_text, settings["chunk_size"]))
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption(f"File: {uploaded.name}")
            with c2:
                st.caption(f"Characters: {len(file_text):,}")
            with c3:
                st.caption(f"≈ Chunks: {est_chunks}")

            if st.button("Ingest Document", type="primary", disabled=not endee_ok):
                model = get_model()
                with st.status("Ingesting…", expanded=True) as status:
                    status.write("Ensuring index exists…")
                    client.ensure_index()

                    status.write("Chunking & embedding…")
                    count = ingest(
                        file_text,
                        uploaded.name,
                        model,
                        client,
                        settings["chunk_size"],
                    )

                    if count > 0:
                        st.session_state.docs_indexed += 1
                        st.session_state.chunks_stored += count
                        status.update(label="Done", state="complete", expanded=False)
                        st.success(
                            f"**{uploaded.name}** — {count} chunks stored."
                        )
                    else:
                        status.update(label="Failed", state="error")
        else:
            st.caption("Upload a PDF or TXT to get started.")

    # ── Question ─────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Ask a Question")
        question = st.text_area(
            "Your question",
            height=110,
            placeholder="Ask something about your uploaded documents…",
        )

        disabled = (not question.strip()) or (not endee_ok)
        if st.button("Ask", type="primary", disabled=disabled):
            st.session_state.chat_history.append(
                {"role": "user", "content": question.strip()}
            )

            t0 = time.perf_counter()
            with st.status("Searching & generating…", expanded=True) as status:
                model = get_model()

                status.write("Searching vector database…")
                sources = search(question.strip(), model, client, settings["top_k"])

                status.write("Generating answer with LLM…")
                answer = generate_answer(
                    question.strip(), sources, settings["temperature"]
                )

                elapsed = time.perf_counter() - t0
                st.session_state.last_query_time = elapsed
                st.session_state.last_sources = sources
                status.update(label="Done", state="complete", expanded=False)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

        if not endee_ok:
            st.info("Vector DB unavailable. Check that Endee is running.")

    # ── Chat history ─────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Conversation")
        if not st.session_state.chat_history:
            st.caption("No conversation yet.")
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    render_sources(st.session_state.last_sources)


if __name__ == "__main__":
    main()