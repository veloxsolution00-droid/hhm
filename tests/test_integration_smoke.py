# Integration smoke test for the full local RAG path: real Ollama generation and
# embeddings plus a real ChromaDB collection, answering a handful of questions
# whose answers are known from a small ingested corpus.
#
# It is opt-in because it needs a running Ollama, downloads nothing, and takes
# minutes; the mocked unit suite must stay fast. Run it explicitly:
#
#   PowerShell:  $env:RND_INTEGRATION="1"; python -m pytest tests/test_integration_smoke.py -q
#   bash:        RND_INTEGRATION=1 python -m pytest tests/test_integration_smoke.py -q
#
# It skips (never fails) when the opt-in flag is absent, when Ollama is
# unreachable, or when a pinned model is missing, so a machine without a local
# model stack still gets a green suite.
import json
import os
import urllib.request
from datetime import datetime, timezone

import pytest

from app.core.config import settings

if os.getenv("RND_INTEGRATION") != "1":
    pytest.skip("set RND_INTEGRATION=1 to run the live Ollama + Chroma smoke test",
                allow_module_level=True)

DOCUMENT_ID = "smoke-employee-handbook"
FILENAME = "employee-handbook.txt"

# One fact per section, so retrieval precision is part of what is checked.
SECTIONS = [
    ("Section 1 - Working hours",
     "The standard working week is 35 hours, Monday to Friday."),
    ("Section 2 - Notice period",
     "An employee who resigns must give 30 days of written notice."),
    ("Section 3 - Annual leave",
     "Every employee receives 25 days of paid annual leave plus public holidays."),
    ("Section 4 - Expenses",
     "Travel expenses are reimbursed within 14 days of an approved claim."),
    ("Section 5 - Equipment",
     "The company issues each employee one laptop, the Aurora X2."),
]

# Known-answer questions. Each expected token is copied verbatim from the corpus
# above, so a correct answer must contain it; nothing here depends on paraphrase.
QUESTIONS = [
    ("How many hours is the standard working week?", "35"),
    ("What notice period must an employee who resigns give?", "30"),
    ("How many days of paid annual leave do employees receive?", "25"),
    ("Within how many days are travel expenses reimbursed?", "14"),
    ("Which laptop model does the company issue to each employee?", "aurora x2"),
]

# A control the corpus cannot answer: the app must abstain, not invent a fact.
UNANSWERABLE = "What is the company's policy on sabbaticals?"


def _ollama_models():
    try:
        with urllib.request.urlopen(
                settings.ollama_base_url.rstrip("/") + "/api/tags", timeout=5) as resp:
            return {m.get("name", "") for m in json.loads(resp.read()).get("models", [])}
    except Exception:
        return set()


def _require_local_stack():
    models = _ollama_models()
    if not models:
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}")
    for required in (settings.generation_model, settings.embedding_model):
        if not any(name == required or name.split(":")[0] == required.split(":")[0]
                   for name in models):
            pytest.skip(f"Ollama is missing the pinned model {required!r}")


@pytest.fixture(scope="module", autouse=True)
def _local_stack():
    _require_local_stack()


@pytest.fixture(scope="module")
def live_corpus(tmp_path_factory):
    """Ingest the handbook into an isolated Chroma/ SQLite store using real embeddings."""
    import chromadb
    if not hasattr(chromadb, "PersistentClient"):
        pytest.skip("the real chromadb package is not importable in this session")

    root = tmp_path_factory.mktemp("smoke_data")
    patch = pytest.MonkeyPatch()
    patch.setattr(settings, "data_path", root)
    patch.setattr(settings, "uploads_path", root / "uploads")
    patch.setattr(settings, "extracted_path", root / "extracted")
    patch.setattr(settings, "chroma_path", root / "chroma_db")
    patch.setattr(settings, "metadata_path", root / "metadata")
    patch.setattr(settings, "logs_path", root / "logs")
    patch.setattr(settings, "jobs_path", root / "jobs")
    patch.setattr(settings, "sqlite_path", root / "metadata" / "documents.sqlite3")
    patch.setattr(settings, "cache_enabled", False)

    from app.retrieval import store
    from app.storage import database as db
    store._client = None
    store._collection = None

    db.initialize()
    now = datetime.now(timezone.utc).isoformat()
    db.save_document({
        "document_id": DOCUMENT_ID, "owner_id": "local", "filename": FILENAME,
        "stored_filename": FILENAME, "original_path": str(root / FILENAME),
        "content_hash": "smoke-handbook", "file_format": "txt",
        "uploaded_at": now, "updated_at": now,
        "processing_status": "indexed", "extraction_status": "complete",
        "indexing_status": "complete", "chunk_count": len(SECTIONS),
        "source_metadata": {"document_type": "txt"}, "extracted_path": None,
        "processing_error": None, "last_reconciled_at": None,
    })
    store.add([
        {"id": f"{DOCUMENT_ID}::{index}",
         "text": f"{title}. {body}",
         "metadata": {"document_id": DOCUMENT_ID, "owner_id": "local",
                      "filename": FILENAME, "source_location": title,
                      "chunk_index": index}}
        for index, (title, body) in enumerate(SECTIONS, 1)
    ])

    yield {"root": root, "store": store, "db": db}

    store._client = None
    store._collection = None
    patch.undo()


def test_retrieval_finds_the_ingested_document(live_corpus):
    hits = live_corpus["store"].search("How many days of paid annual leave?", "local")
    assert hits, "real dense + lexical retrieval returned no chunks"
    assert any(hit["metadata"].get("document_id") == DOCUMENT_ID for hit in hits)


def test_known_answers_are_grounded_and_cited(live_corpus):
    from app.rag.chat import answer
    failures = []
    for question, expected in QUESTIONS:
        result = answer(question, "local")
        text = str(result.get("answer") or "")
        if result.get("status") != "success":
            failures.append(f"{question!r}: status={result.get('status')}")
            continue
        if expected.casefold() not in text.casefold():
            failures.append(f"{question!r}: answer {text!r} does not contain {expected!r}")
            continue
        if not result.get("sources") or "[" not in text:
            failures.append(f"{question!r}: answer carries no evidence citation: {text!r}")
    assert not failures, "live smoke failures:\n" + "\n".join(failures)


def test_unanswerable_question_abstains(live_corpus):
    from app.rag.chat import answer
    result = answer(UNANSWERABLE, "local")
    assert result.get("abstained") is True
    assert result.get("status") != "success"
