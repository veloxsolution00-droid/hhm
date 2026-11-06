"""
Offline Document Intelligence Chatbot — UI refresh.

This file is a visual-only redesign of the existing Streamlit interface.
Existing API endpoints, request payloads, session-state keys, upload rules,
job handling, document deletion, chat history, citations, confidence checks,
verification, and error handling are retained.
"""

import os
import requests
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
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --page: #0b1020;
        --surface: #111a2d;
        --surface-2: #151f35;
        --border: rgba(148, 163, 184, .16);
        --text: #edf2ff;
        --muted: #93a2bd;
        --accent: #8798ff;
        --accent-2: #6379ef;
        --green: #37c99a;
        --amber: #f1b85b;
        --red: #ef7f89;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', 'Segoe UI', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(99, 121, 239, .13), transparent 34%),
            radial-gradient(ellipse at 95% 8%, rgba(55, 201, 154, .07), transparent 28%),
            var(--page);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, .88);
        border-bottom: 1px solid rgba(148, 163, 184, .08);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10192c 0%, #0d1424 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.1rem;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.7rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', 'DM Sans', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -.035em;
    }

    h1 { font-weight: 800 !important; }
    h2, h3 { font-weight: 750 !important; }
    p, li, label, [data-testid="stMarkdownContainer"] {
        color: #d8e1f2;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    .app-hero {
        position: relative;
        overflow: hidden;
        padding: 27px 30px;
        margin: 0 0 22px 0;
        border: 1px solid rgba(135, 152, 255, .24);
        border-radius: 22px;
        background:
            radial-gradient(circle at 88% 12%, rgba(135, 152, 255, .22), transparent 29%),
            linear-gradient(115deg, rgba(27, 41, 72, .98), rgba(17, 26, 45, .98));
        box-shadow: 0 18px 48px rgba(0, 0, 0, .17);
    }

    .hero-kicker {
        color: #aab7ff;
        font-size: .72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .15em;
        margin-bottom: 9px;
    }

    .hero-title {
        color: #f5f7ff;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.65rem, 3vw, 2.35rem);
        line-height: 1.18;
        font-weight: 800;
        letter-spacing: -.045em;
        margin: 0 0 9px 0;
    }

    .hero-description {
        color: #b5c2dc;
        max-width: 760px;
        font-size: .94rem;
        line-height: 1.65;
        margin: 0;
    }

    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 11px;
        margin-top: 15px;
        border: 1px solid rgba(55, 201, 154, .25);
        border-radius: 999px;
        color: #a6ead4;
        background: rgba(55, 201, 154, .08);
        font-size: .74rem;
        font-weight: 700;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        color: #eef3ff;
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: -.025em;
        margin: 2px 0 4px 0;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: .84rem;
        margin: 0 0 14px 0;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 4px 3px 17px 3px;
        margin-bottom: 16px;
        border-bottom: 1px solid var(--border);
    }

    .brand-mark {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border-radius: 13px;
        color: #fff;
        font-size: 1.25rem;
        background: linear-gradient(145deg, #91a0ff, #5d73e8);
        box-shadow: 0 8px 22px rgba(99, 121, 239, .27);
    }

    .brand-title {
        color: #f2f5ff;
        font-family: 'Manrope', sans-serif;
        font-size: .96rem;
        font-weight: 800;
        line-height: 1.15;
    }

    .brand-caption {
        color: #8292ae;
        font-size: .69rem;
        margin-top: 4px;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(23, 34, 57, .96), rgba(17, 26, 45, .96));
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 13px 16px;
        box-shadow: 0 10px 26px rgba(0, 0, 0, .11);
    }

    [data-testid="stMetricLabel"] p {
        color: #9eacc5 !important;
        font-size: .78rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f4f7ff !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border) !important;
        border-radius: 16px !important;
        background: rgba(17, 26, 45, .55);
    }

    .stButton > button {
        min-height: 2.5rem;
        border-radius: 10px;
        border: 1px solid rgba(135, 152, 255, .28);
        font-weight: 700;
        transition: all .15s ease;
    }

    .stButton > button[kind="primary"] {
        color: #fff;
        border: 0;
        background: linear-gradient(135deg, #8798ff, #6379ef);
        box-shadow: 0 7px 18px rgba(99, 121, 239, .2);
    }

    .stButton > button:hover {
        border-color: rgba(170, 183, 255, .75);
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"] {
        padding: 8px;
        border: 1px dashed rgba(135, 152, 255, .38);
        border-radius: 13px;
        background: rgba(20, 31, 53, .5);
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, .15) !important;
        border-radius: 12px !important;
        background: rgba(17, 26, 45, .5) !important;
    }

    [data-testid="stChatMessage"] {
        padding: 13px 16px;
        margin-bottom: 12px;
        border: 1px solid rgba(148, 163, 184, .12);
        border-radius: 16px;
        background: rgba(17, 26, 45, .66);
    }

    [data-testid="stChatInput"] {
        border-color: rgba(135, 152, 255, .3) !important;
        border-radius: 14px !important;
        background: rgba(17, 26, 45, .92) !important;
    }

    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #8798ff, #37c99a);
    }

    .chat-hint {
        margin: 10px 0 20px;
        padding: 18px 20px;
        border: 1px solid var(--border);
        border-radius: 14px;
        color: var(--muted);
        background: rgba(17, 26, 45, .58);
        line-height: 1.6;
    }

    hr {
        border-color: rgba(148, 163, 184, .13);
    }

    @media (max-width: 800px) {
        .block-container { padding-top: 1rem; }
        .app-hero { padding: 21px 20px; border-radius: 17px; }
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
        <div class="hero-kicker">PRIVATE · LOCAL · AI-POWERED</div>
        <div class="hero-title">Offline Document Intelligence</div>
        <p class="hero-description">
            Upload your documents, follow the processing pipeline, and ask
            evidence-grounded questions using your local AI models.
        </p>
        <div class="hero-tag">● Local processing workspace</div>
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
                <div class="brand-caption">OFFLINE KNOWLEDGE WORKSPACE</div>
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

    if st.button("Upload and process", disabled=not up, width="stretch", type="primary"):
        try:
            total = sum(f.size for f in up)
            max_batch_mb = int(health_limits.get("max_batch_mb", 500) or 500)
            if total > max_batch_mb * 1024 * 1024:
                st.error(f"Batch exceeds {max_batch_mb} MB.")
            else:
                files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in up]
                with st.spinner("Saving files and adding ingestion jobs to the local queue..."):
                    data = api_post("/documents/upload", files=files, timeout=600)
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
if clear_column.button("Clear chat", icon=":material/delete_sweep:", width="stretch"):
    st.session_state.messages = []
    st.rerun()

if not st.session_state.messages:
    st.markdown(
        """
        <div class="chat-hint">
            Your conversation will appear here. Start with a question about a
            document, a key fact, a comparison, or a specific passage.
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
