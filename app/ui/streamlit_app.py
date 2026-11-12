"""
Offline Document Intelligence Chatbot — UI refresh.

This file is a visual-only redesign of the existing Streamlit interface.
Existing API endpoints, request payloads, session-state keys, upload rules,
job handling, document deletion, chat history, citations, confidence checks,
verification, and error handling are retained.
"""

import os
import requests
import httpx
import streamlit as st

from app.core.config import validate_loopback_url


API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
if not validate_loopback_url(API):
    raise RuntimeError("API_URL must be an HTTP loopback address for offline-only operation.")

st.set_page_config(
    page_title="Offline Document Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Visual layer only — no backend, database, retrieval, or model logic here.
#
# Design notes
#   * One accent colour (cornflower) is used only for actions and focus.
#     Green / amber / red are reserved for status so they always mean something.
#   * Three surface levels: page < panel < raised. Borders do the separating,
#     not shadows or glows.
#   * System fonts only, so nothing is fetched from the internet (offline-safe).
#   * Two corner radii: 12px for panels, 8px for controls.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --page: #0f141c;
        --panel: #151b25;
        --raised: #1c2431;
        --line: #263040;
        --line-strong: #364357;
        --text: #e8edf5;
        --text-2: #b6c1d1;
        --muted: #8795a9;
        --accent: #7f9cff;
        --accent-hover: #9db3ff;
        --accent-ink: #0b1220;
        --accent-soft: rgba(127, 156, 255, .12);
        --accent-line: rgba(127, 156, 255, .32);
        --ok: #4cc9a0;
        --warn: #e8b04f;
        --bad: #ef7b86;
        --radius-panel: 12px;
        --radius-control: 8px;
    }

    /* ---------- Base ---------- */
    .stApp,
    .stApp button,
    .stApp input,
    .stApp textarea {
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
    }

    .stApp {
        background: var(--page);
        color: var(--text);
    }

    * {
        scrollbar-width: thin;
        scrollbar-color: var(--line-strong) transparent;
    }

    /* Hide Streamlit chrome that is not useful in a local tool */
    #MainMenu,
    footer,
    [data-testid="stDeployButton"],
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }

    [data-testid="stHeader"] {
        background: rgba(15, 20, 28, .85);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid var(--line);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-weight: 650 !important;
        letter-spacing: -.02em;
    }

    .stApp h3 {
        font-size: 1.1rem !important;
        line-height: 1.3 !important;
        padding: .25rem 0 .1rem !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: var(--text-2);
        line-height: 1.6;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
        font-size: .82rem;
        line-height: 1.45;
    }

    a { color: var(--accent); }
    a:hover { color: var(--accent-hover); }

    code {
        padding: .1em .35em;
        border-radius: 5px;
        background: var(--raised);
        color: #c9d6ff;
        font-size: .88em;
    }

    hr {
        margin: 2rem 0 !important;
        border-color: var(--line) !important;
    }

    /* ---------- Page header ---------- */
    .app-hero {
        padding: 24px 28px 22px;
        margin: 0 0 8px 0;
        border: 1px solid var(--line);
        border-radius: var(--radius-panel);
        background: var(--panel);
    }

    .hero-title {
        margin: 0 0 6px 0;
        color: var(--text);
        font-size: clamp(1.5rem, 2.6vw, 1.95rem);
        line-height: 1.2;
        font-weight: 700;
        letter-spacing: -.03em;
    }

    .hero-description {
        max-width: 64ch;
        margin: 0;
        color: var(--text-2);
        font-size: .95rem;
        line-height: 1.6;
    }

    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        color: var(--text-2);
        font-size: .8rem;
        font-weight: 500;
    }

    .hero-tag::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--ok);
    }

    .hero-steps {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-top: 20px;
        padding-top: 18px;
        border-top: 1px solid var(--line);
    }

    .hero-step {
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }

    .step-num {
        flex: none;
        display: grid;
        place-items: center;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: var(--accent-soft);
        border: 1px solid var(--accent-line);
        color: var(--accent);
        font-size: .78rem;
        font-weight: 700;
    }

    .step-title {
        color: var(--text);
        font-size: .9rem;
        font-weight: 650;
        line-height: 1.35;
    }

    .step-text {
        margin-top: 2px;
        color: var(--muted);
        font-size: .82rem;
        line-height: 1.45;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] h3 {
        font-size: .98rem !important;
        padding-bottom: 0 !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 2px 2px 18px 2px;
        margin-bottom: 14px;
        border-bottom: 1px solid var(--line);
    }

    .brand-mark {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: var(--accent);
        color: var(--accent-ink);
        font-size: 1.1rem;
        font-weight: 700;
    }

    .brand-title {
        color: var(--text);
        font-size: .98rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .brand-caption {
        margin-top: 3px;
        color: var(--muted);
        font-size: .74rem;
    }

    [data-testid="stSidebar"] [data-testid="stAlertContainer"] {
        padding: .5rem .75rem;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        font-size: .78rem;
    }

    /* ---------- Status messages ---------- */
    [data-testid="stAlert"] {
        border-radius: var(--radius-control);
        font-size: .88rem;
    }

    [data-testid="stAlertContainer"] p {
        color: var(--text) !important;
        line-height: 1.5;
    }

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {
        padding: 14px 18px;
        border: 1px solid var(--line);
        border-radius: var(--radius-panel);
        background: var(--panel);
    }

    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: .8rem !important;
        font-weight: 500 !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        font-variant-numeric: tabular-nums;
    }

    /* ---------- Bordered containers (library rows) ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 10px !important;
        background: var(--panel);
    }

    [data-testid="stVerticalBlockBorderWrapper"] p {
        margin: 0;
        font-size: .9rem;
        overflow-wrap: anywhere;
    }

    /* ---------- Buttons ---------- */
    [data-testid^="stBaseButton"] {
        min-height: 2.5rem;
        border-radius: var(--radius-control);
        font-weight: 600;
        white-space: nowrap;
        transition: background-color .12s ease, border-color .12s ease, color .12s ease;
    }

    [data-testid="stBaseButton-primary"] {
        border: 1px solid transparent;
        background: var(--accent);
        color: var(--accent-ink);
    }

    [data-testid="stBaseButton-primary"]:hover:not(:disabled) {
        background: var(--accent-hover);
        color: var(--accent-ink);
    }

    [data-testid="stBaseButton-primary"]:disabled {
        background: var(--raised);
        border-color: var(--line);
        color: var(--muted);
        cursor: not-allowed;
    }

    [data-testid="stBaseButton-secondary"] {
        border: 1px solid var(--line-strong);
        background: transparent;
        color: var(--text-2);
    }

    [data-testid="stBaseButton-secondary"]:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
        color: var(--text);
    }

    /* Compact buttons inside library rows and job panels */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stBaseButton-secondary"],
    [data-testid="stExpander"] [data-testid="stBaseButton-secondary"] {
        min-height: 2rem;
        padding: .15rem .75rem;
        font-size: .82rem;
    }

    /* Destructive actions look neutral until hovered */
    [class*="st-key-delete_"] button:hover,
    [class*="st-key-stop_"] button:hover {
        border-color: var(--bad);
        background: rgba(239, 123, 134, .1);
        color: var(--bad);
    }

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploaderDropzone"] {
        padding: 14px;
        border: 1.5px dashed var(--line-strong);
        border-radius: 10px;
        background: var(--page);
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent);
    }

    /* ---------- Expanders (jobs, sources, answer details) ---------- */
    [data-testid="stExpander"] {
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        background: var(--panel) !important;
    }

    [data-testid="stExpander"] summary {
        padding: .6rem .85rem;
        font-size: .88rem;
        color: var(--text-2);
    }

    [data-testid="stExpander"] summary:hover {
        background: var(--raised);
        color: var(--text);
    }

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: .25rem .9rem .9rem;
    }

    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        background: var(--page) !important;
    }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        gap: .9rem;
        padding: 14px 18px;
        margin-bottom: 10px;
        border: 1px solid var(--line);
        border-radius: var(--radius-panel);
        background: var(--panel);
    }

    /* Your questions are tinted so the conversation is easy to scan */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        border-color: var(--accent-line);
        background: var(--accent-soft);
    }

    [data-testid="stChatMessageAvatarUser"] {
        background: var(--accent);
        color: var(--accent-ink);
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background: var(--raised);
        color: var(--text);
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        max-width: 78ch;
        color: var(--text);
        line-height: 1.7;
    }

    [data-testid="stBottom"] > div {
        background: var(--page);
    }

    [data-testid="stChatInput"] {
        border: 1px solid var(--line-strong) !important;
        border-radius: var(--radius-panel) !important;
        background: var(--panel) !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft);
    }

    .chat-hint {
        margin: 6px 0 18px;
        padding: 20px 22px;
        border: 1px dashed var(--line-strong);
        border-radius: var(--radius-panel);
        background: var(--panel);
    }

    .chat-hint-title {
        color: var(--text);
        font-size: .98rem;
        font-weight: 650;
    }

    .chat-hint-text {
        margin-top: 4px;
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.55;
    }

    .chat-examples {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }

    .chat-examples span {
        padding: 6px 12px;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: var(--raised);
        color: var(--text-2);
        font-size: .82rem;
    }

    /* ---------- Progress ---------- */
    [data-testid="stProgress"] div[role="progressbar"] {
        height: 6px;
        border-radius: 999px;
        background: var(--raised);
    }

    [data-testid="stProgress"] div[role="progressbar"] > div,
    [data-testid="stProgress"] > div > div > div {
        border-radius: 999px;
        background: var(--accent);
    }

    /* ---------- Accessibility ---------- */
    .stApp :focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }

    @media (prefers-reduced-motion: reduce) {
        * {
            transition: none !important;
            animation: none !important;
        }
    }

    /* ---------- Small screens ---------- */
    @media (max-width: 800px) {
        .block-container { padding-top: 1rem; }
        .app-hero { padding: 20px; }
        .hero-steps { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def api_get(path, timeout=10):
    r = requests.get(API + path, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(path, **kwargs):
    r = requests.post(API + path, **kwargs)
    r.raise_for_status()
    return r.json()

def api_post_stream(path, **kwargs):
    """Send multipart uploads with httpx's chunked file streaming."""
    timeout = kwargs.pop("timeout", 30)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(API + path, **kwargs)
        response.raise_for_status()
        return response.json()


def citation_line(sources):
    return " · ".join(
        f"[{source.get('rank', index)}] {source.get('filename', 'document')} — {source.get('source_location', 'location unavailable')}"
        for index, source in enumerate(sources, 1)
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-hero">
        <div class="hero-title">Offline Document Intelligence</div>
        <p class="hero-description">
            Upload your documents, follow the processing pipeline, and ask
            evidence-grounded questions using your local AI models.
        </p>
        <div class="hero-tag">Everything runs on this computer</div>
        <div class="hero-steps">
            <div class="hero-step">
                <div class="step-num">1</div>
                <div>
                    <div class="step-title">Upload</div>
                    <div class="step-text">Add files from the sidebar.</div>
                </div>
            </div>
            <div class="hero-step">
                <div class="step-num">2</div>
                <div>
                    <div class="step-title">Wait for indexing</div>
                    <div class="step-text">Track progress in the processing queue.</div>
                </div>
            </div>
            <div class="hero-step">
                <div class="step-num">3</div>
                <div>
                    <div class="step-title">Ask</div>
                    <div class="step-text">Get answers with source passages.</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — same health request and upload workflow as the original UI.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark">✦</div>
            <div>
                <div class="brand-title">Doc Intelligence</div>
                <div class="brand-caption">Offline knowledge workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### System status")
    health_limits = {}
    try:
        health_response = requests.get(API + "/health", timeout=5)
        health_response.raise_for_status()
        h = health_response.json()
        health_limits = h.get("limits") or {}
        if h.get("status") == "ok":
            st.success("Local API online", icon=":material/check_circle:")
        else:
            st.warning(f"API status: {h.get('status', 'unknown')}")
        ollama = h.get("ollama", {})
        if ollama.get("available") and ollama.get("generation_model_available") and ollama.get("embedding_model_available"):
            st.success("Local models ready", icon=":material/memory:")
        else:
            st.warning("One or more local models are unavailable.")
        st.caption(f"Answer model: {ollama.get('generation_model', 'unknown')}")
        st.caption(f"Embedding model: {ollama.get('embedding_model', 'unknown')}")
        st.caption(f"Indexed chunks: {h.get('chromadb', {}).get('count', 0)}")
        st.caption(f"File limit: {health_limits.get('max_file_mb',100)} MB | Batch limit: {health_limits.get('max_batch_mb',500)} MB")
    except Exception as e:
        st.warning(f"Backend not reachable. Start FastAPI first. {e}")

    st.divider()
    st.markdown("### Upload documents")
    st.caption("Add one or more files to the local ingestion queue.")

    if "uploader_id" not in st.session_state:
        st.session_state.uploader_id = 0

    up = st.file_uploader(
        "Select multiple documents",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_id}",
    )

    if st.button("Upload and process", disabled=not up, type="primary"):
        try:
            total = sum(f.size for f in up)
            max_batch_mb = int(health_limits.get("max_batch_mb", 500) or 500)
            if total > max_batch_mb * 1024 * 1024:
                st.error(f"Batch exceeds {max_batch_mb} MB.")
            else:
                files = []
                for uploaded in up:
                    uploaded.seek(0)
                    files.append(("files", (uploaded.name, uploaded, uploaded.type or "application/octet-stream")))
                with st.spinner("Saving files and adding ingestion jobs to the local queue..."):
                    data = api_post_stream("/documents/upload", files=files, timeout=600)
                st.success(f"Accepted {data['files']} file(s), {data['total_bytes'] / (1024*1024):.1f} MB. Staging: {data['staging_seconds']} sec.")
                st.info("The job list updates automatically with the current extraction, chunking, embedding, and indexing stage.")
                st.session_state.uploader_id += 1
                st.rerun()
        except Exception as e:
            st.error(f"Upload failed: {e}")


# ---------------------------------------------------------------------------
# Jobs and documents — original fragment cadence and API operations retained.
# ---------------------------------------------------------------------------
@st.fragment(run_every="3s")
def show_jobs_and_documents():
    jobs_column, documents_column = st.columns([1, 1], gap="large")

    with jobs_column:
        st.markdown("### Processing queue")
        st.caption("Live progress for document extraction, chunking, embedding and indexing.")

        try:
            job_data = api_get("/jobs", timeout=10).get("jobs", [])
            # Successful jobs are automatically removed by the backend. Keep only
            # active work and unsuccessful terminal jobs visible for diagnosis/retry.
            visible_jobs = [j for j in job_data if j.get("status") in
                            ("queued", "processing", "cancel_requested", "failed", "cancelled", "delete_failed")]
            active = [j for j in visible_jobs if j.get("status") in ("queued", "processing", "cancel_requested")]
            if active:
                st.caption(f"Auto-refreshing every 3 seconds · {len(active)} job(s) active")
            if not visible_jobs:
                st.info("No active or failed ingestion jobs. Successfully indexed documents are listed under Your documents.")
            for j in visible_jobs[:50]:
                label = f"{j['filename']} · {j['status']} · {j.get('progress',0)}%"
                with st.expander(label, expanded=j["status"] in ("queued", "processing", "failed")):
                    st.progress(max(0, min(100, int(j.get("progress", 0)))) / 100)
                    st.write("Stage:", j.get("stage") or j["status"])
                    st.write("Updated:", j.get("updated_at", "—"))
                    st.write("Size:", f"{j['size_bytes']/(1024*1024):.1f} MB")
                    if j.get("document_id"):
                        st.write("Document ID:", j["document_id"])
                    if j.get("error"):
                        st.error(j["error"])
                    # Deletion is available only after the worker is confirmed stopped.
                    state = j["status"]
                    can_stop = state in ("queued", "processing")
                    if state == "cancel_requested":
                        st.info("Stopping safely. The worker may finish its current local model request; deletion will become available after it exits.")
                    if can_stop and st.button("Stop", key="stop_" + j["job_id"]):
                        try:
                            response = api_post("/jobs/" + j["job_id"] + "/cancel", timeout=10)
                            st.info(response.get("message", "Cancellation requested."))
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                    if j["status"] == "failed" and st.button("Retry", key="retry_" + j["job_id"]):
                        try:
                            api_post("/jobs/" + j["job_id"] + "/retry", timeout=10)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
        except Exception as e:
            st.info(f"Jobs are not available: {e}")

    with documents_column:
        st.markdown("### Your library")
        st.caption("Indexed files and their current processing status.")

        try:
            docs = api_get("/documents", timeout=10).get("documents", [])
            indexed_docs = [
                doc for doc in docs
                if doc.get("processing_status") == "indexed"
                and doc.get("indexing_status", "complete") == "complete"
            ]
            metric_docs, metric_chunks = st.columns(2)
            metric_docs.metric("Indexed files", len(indexed_docs))
            metric_chunks.metric(
                "Searchable passages",
                sum(int(doc.get("chunk_count", 0) or 0) for doc in indexed_docs),
            )
            if not docs:
                st.info("No documents have been indexed yet.")
            for d in docs:
                with st.container(border=True):
                    a, b, c = st.columns([5, 2, 1])
                    a.write(d["filename"])
                    b.write(f"{d.get('processing_status','unknown')} · {d.get('chunk_count',0)} chunks")
                    if d.get("processing_error"):
                        a.caption("Issue: " + d["processing_error"])
                    if c.button("Delete", key="delete_" + d["document_id"]):
                        r = requests.delete(API + "/documents/" + d["document_id"], timeout=30)
                        r.raise_for_status()
                        st.rerun()
        except Exception as e:
            st.info(f"Document inventory unavailable: {e}")


# ---------------------------------------------------------------------------
# Chat — same message history, payload, response parsing, and evidence checks.
# ---------------------------------------------------------------------------
st.divider()
chat_title, clear_column = st.columns([5, 1], vertical_alignment="center")
chat_title.markdown("### Ask your documents")
chat_title.caption("Answers use passages retrieved from your indexed files, with source references.")
if clear_column.button("Clear chat", icon=":material/delete_sweep:"):
    st.session_state.messages = []
    st.rerun()

if not st.session_state.messages:
    st.markdown(
        """
        <div class="chat-hint">
            <div class="chat-hint-title">No questions yet</div>
            <div class="chat-hint-text">Once a document is indexed, ask about it in the box at the bottom of the page. For example:</div>
            <div class="chat-examples">
                <span>What are the key points in this report?</span>
                <span>What does the contract say about termination?</span>
                <span>Compare the two proposals on cost.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.caption("Sources: " + citation_line(m["sources"]))
        for src in m.get("sources", []):
            with st.expander(f"{src.get('filename')} — {src.get('source_location')}"):
                st.write("Relevance:", src.get("relevance_score", "—"))
                st.write(src.get("supporting_text", ""))

with st.bottom:
    q = st.chat_input("Ask a question about your uploaded documents")

if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Resolving context, retrieving supporting passages, and generating a local answer..."):
                # The API caps each history message at 2000 characters. Answers can
                # be longer, so trim here instead of failing the next request with 422.
                history = [
                    {"role": m["role"], "content": str(m["content"])[:2000]}
                    for m in st.session_state.messages[:-1][-8:]
                    if m.get("role") in ("user", "assistant")
                ]
                data = api_post("/chat", json={"question": q, "history": history}, timeout=600)

            st.markdown(data.get("answer", "No answer returned."))
            retrieval = data.get("retrieval", {})

            with st.expander("Answer details", expanded=False):
                if data.get("question_resolved"):
                    st.caption("Interpreted follow-up as: " + data["question_resolved"])
                if data.get("question_type"):
                    st.caption("Question type: " + data["question_type"])

                st.caption(f"Status: {data.get('status', 'unknown')} | Response time: {data.get('response_seconds', '—')} seconds")
                confidence = data.get("confidence") or {}
                if confidence.get("label"):
                    score = confidence.get("score")
                    label = confidence["label"]
                    heading = f"Confidence: {label}" + (f" ({score:.0%})" if isinstance(score, (int, float)) else "")
                    (st.success if label == "high" else st.info if label == "medium" else st.warning)(heading)
                    signals = confidence.get("signals") or {}
                    if signals:
                        st.caption(f"Retrieval similarity: {signals.get('retrieval_similarity', '—')} | Term coverage: {signals.get('term_coverage', '—')} | Citation coverage: {signals.get('citation_coverage', '—')} | Verifier: {signals.get('verified', '—')}")

                if retrieval:
                    st.caption(f"Evidence: {retrieval.get('selected_chunks', 0)} passage(s) selected from {retrieval.get('candidate_chunks', 0)} candidates | Best similarity: {retrieval.get('best_similarity', '—')} | Retrieval passes: {retrieval.get('retrieval_passes', 1)}")

            citation_check = data.get("citation_validation")
            # An abstention also carries an invalid citation block (the refusal has
            # no [n] markers). Reporting that as a citation failure contradicts the
            # real status, which is "no answer in the evidence".
            if data.get("status") == "citation_validation_failed" and citation_check and not citation_check.get("valid", False):
                st.warning("Citation validation did not pass. The response may have been withheld because its source reference was missing or invalid.")

            verification = data.get("verification") or {}
            if verification.get("checked") and not verification.get("passed"):
                st.warning("The local verifier could not confirm this answer against the cited passages. Treat it as unverified and check the sources below.")

            conflicts = data.get("potential_conflicts") or []
            if conflicts:
                st.warning(f"Potential evidence disagreement detected in {len(conflicts)} passage pair(s). Review the cited passages; this is a heuristic warning, not a confirmed contradiction.")

            if retrieval.get("generation_fallback"):
                st.info("Ollama generation did not complete. The application returned a verbatim sentence from the retrieved PDF instead of inventing an answer.")
                if retrieval.get("generation_error"):
                    st.caption("Generation diagnostic: " + str(retrieval["generation_error"]))
            if data.get("error"):
                st.error(data["error"])

            if data.get("sources"):
                st.markdown("**Evidence used**")
                st.caption(citation_line(data["sources"]))

            for src in data.get("sources", []):
                with st.expander(f"{src.get('filename')} — {src.get('source_location')}"):
                    st.write("Relevance:", src.get("relevance_score", "—"))
                    st.write(src.get("supporting_text", ""))

            st.session_state.messages.append({
                "role": "assistant",
                "content": data.get("answer", "No answer returned."),
                "sources": data.get("sources", []),
            })

        except Exception as e:
            st.error(f"Chat request failed: {e}")
            st.session_state.messages.append({"role": "assistant", "content": str(e)})


st.divider()
st.markdown("### Library and processing")
st.caption("Manage uploaded files and monitor extraction, embedding, and indexing.")
show_jobs_and_documents()