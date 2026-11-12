"""Regression coverage for the approved reliability/security hardening."""
from datetime import datetime, timezone
from pathlib import Path
import types
import pytest


def _document(db, tmp_path, document_id="doc-a", status="indexed", chunk_count=2):
    now = datetime.now(timezone.utc).isoformat()
    original = tmp_path / f"{document_id}.txt"
    original.write_text("source evidence", encoding="utf-8")
    extracted = tmp_path / f"{document_id}.jsonl"
    extracted.write_text('{"text":"source evidence"}\\n', encoding="utf-8")
    db.save_document({
        "document_id": document_id, "owner_id": "local", "filename": original.name,
        "stored_filename": original.name, "original_path": str(original),
        "content_hash": f"hash-{document_id}", "file_format": "txt",
        "uploaded_at": now, "updated_at": now, "processing_status": status,
        "extraction_status": "complete", "indexing_status": "complete" if status == "indexed" else "pending",
        "chunk_count": chunk_count, "source_metadata": {}, "extracted_path": str(extracted),
        "processing_error": None, "last_reconciled_at": None,
    })
    return original, extracted


def test_citation_validation_rejects_uncited_factual_sentence():
    from app.rag.citations import validate_citation_references

    result = validate_citation_references(
        "Revenue was 10 crore [1]. The company had 25 employees.",
        1,
        ["annual-report.pdf"],
    )
    assert result["citation_present"] is True
    assert result["citation_coverage"] < 1.0
    assert result["valid"] is False
    assert result["uncited_sentences"]


def test_citation_validation_accepts_fully_cited_sentences():
    from app.rag.citations import validate_citation_references

    result = validate_citation_references(
        "Revenue was 10 crore [1]. The company had 25 employees [1].",
        1,
        ["annual-report.pdf"],
    )
    assert result["valid"] is True
    assert result["citation_coverage"] == 1.0


def test_citation_validation_separates_legal_references_from_full_coverage():
    """An answer whose citations are legal but incomplete is still grounded.

    A submitted answer with one legal citation and one uncited factual sentence
    must remain distinguishable from a broken answer: references_valid stays
    True (so the answer can be returned) while fully_cited/valid stay False.
    """
    from app.rag.citations import validate_citation_references

    result = validate_citation_references(
        "Revenue was 10 crore [1]. The company had 25 employees.",
        1,
        ["annual-report.pdf"],
    )
    assert result["citation_present"] is True
    assert result["citation_coverage"] < 1.0
    assert result["references_valid"] is True
    assert result["fully_cited"] is False
    assert result["valid"] is False


def test_citation_validation_rejects_illegal_references_even_when_incomplete():
    """A partial answer that also cites an out-of-range number is not returned."""
    from app.rag.citations import validate_citation_references

    result = validate_citation_references(
        "Revenue was 10 crore [7]. The company had 25 employees.",
        1,
        ["annual-report.pdf"],
    )
    assert result["references_valid"] is False
    assert result["valid"] is False


def test_citation_validation_keeps_bare_domain_and_path_in_one_sentence():
    from app.rag.citations import validate_citation_references

    result = validate_citation_references(
        "Visit library.example.org/free-guides/ for free books [1].", 1)
    assert result["valid"] is True
    assert result["citation_coverage"] == 1.0
    assert result["uncited_sentences"] == []


def test_markdown_url_is_not_mistaken_for_phantom_source_label():
    from app.rag.citations import validate_citation_references

    result = validate_citation_references(
        "Scan the code or visit [library.example.org/free-guides/] [1] "
        "to download free books.", 1)
    assert result["phantom_labels"] == []
    assert result["valid"] is True


def test_verbatim_evidence_url_gets_citation_but_unmatched_url_does_not():
    from app.rag.citations import cite_verbatim_urls, validate_citation_references

    answer = "Visit [library.example.org/free-guides/] to download books."
    sources = [{"text": "Scan the code or go to library.example.org/free-guides/"}]
    cited = cite_verbatim_urls(answer, sources)
    assert cited.endswith("[1].")
    assert validate_citation_references(cited, 1)["valid"] is True

    unrelated = cite_verbatim_urls(
        "Visit [other.example.org/free-books/] for details.", sources)
    assert "[1]" not in unrelated


def test_uncited_sentence_gets_reference_only_when_its_words_match_evidence():
    from app.rag.citations import cite_supported_uncited_sentences, validate_citation_references

    sources = [{"text": "The Science of Getting Rich explains that thinking in a certain way "
                        "and focused thought help a person create value."}]
    answer = cite_supported_uncited_sentences(
        "The Science of Getting Rich says thinking in a certain way helps create value.", sources)
    assert "[1]" in answer
    assert validate_citation_references(answer, 1)["valid"] is True

    unrelated = cite_supported_uncited_sentences(
        "The author was born in London and became a famous actor.", sources)
    assert "[1]" not in unrelated
    assert validate_citation_references(unrelated, 1)["valid"] is False


def test_out_of_range_page_number_reference_is_reselected_from_matching_evidence():
    from app.rag.citations import cite_supported_uncited_sentences, validate_citation_references

    sources = [{"text": "Dracula was first published in 1897 as an epistolary novel."}]
    repaired = cite_supported_uncited_sentences(
        "Dracula was first published in 1897 [29].", sources)
    assert "[29]" not in repaired
    assert "[1]" in repaired
    assert validate_citation_references(repaired, 1)["valid"] is True


def test_semantic_cache_anchors_capitalized_acronyms():
    from app.cache.manager import _anchors

    assert "acronym:gst" in _anchors("GST return for FY 2025")
    assert "acronym:gstr" in _anchors("GSTR return for FY 2025")
    assert _anchors("GST return 2025") != _anchors("GSTR return 2025")


def test_multi_question_reports_partial_success(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    import app.rag.chat as chat

    monkeypatch.setattr(chat, "list_docs", lambda owner_id=None: [
        {"document_id": "d1", "processing_status": "indexed", "indexing_status": "complete"}
    ])
    def fake_answer(question, indexed, owner_id, doc_types=None):
        if "First" in question:
            return {"question": question, "status": "success", "answer": "Found [1].",
                    "sources": [{"filename": "a.txt", "source_location": "page 1",
                                 "supporting_text": "Found"}], "abstained": False}
        return {"question": question, "status": "insufficient_evidence",
                "answer": "Cannot find it.", "sources": [], "abstained": True}
    monkeypatch.setattr(chat, "_answer_one", fake_answer)

    result = chat.answer("First question? Second question?")
    assert result["status"] == "partial_success"
    assert len(result["partial_failures"]) == 1
    assert result["abstained"] is False


def _verifier_outage_stack(monkeypatch, draft):
    """Minimal retrieval stack whose model verifier raises, for fail-closed checks."""
    import sys
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    import app.rag.chat as chat
    from app.core.config import settings

    hit = {"id": "c1", "text": "Revenue was 10 crore.", "similarity": 0.9,
           "metadata": {"document_id": "d1", "filename": "annual-report.pdf",
                        "source_location": "page 1"}}
    monkeypatch.setattr(chat, "list_docs", lambda owner_id=None: [
        {"document_id": "d1", "processing_status": "indexed", "indexing_status": "complete"}
    ])
    monkeypatch.setattr(chat.cache_manager, "lookup_result", lambda *a, **k: None)
    monkeypatch.setattr(chat.cache_manager, "put_result", lambda *a, **k: None)
    monkeypatch.setattr(chat, "embed_query", lambda question: [1.0, 0.0])
    monkeypatch.setattr(chat, "search", lambda *a, **k: [hit])
    monkeypatch.setattr(chat, "query_variants", lambda question: [question])
    monkeypatch.setattr(chat, "indexed_vocabulary", lambda: [])
    monkeypatch.setattr(chat, "rank_candidates", lambda question, hits: hits)
    monkeypatch.setattr(chat, "select_evidence", lambda *a, **k: {
        "hits": [hit], "term_coverage": 1.0, "missing_terms": []
    })
    monkeypatch.setattr(chat, "generate", lambda prompt: draft)
    monkeypatch.setattr(chat, "_verify_answer",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("verifier offline")))
    monkeypatch.setattr(settings, "verify_answers", True)
    return chat


def test_verifier_exception_fails_closed(monkeypatch):
    # The draft is not supported by the passage it cites, so the deterministic
    # arbiter cannot decide it and the model verifier is required. An unavailable
    # verifier must then withhold the unchecked draft rather than promote it.
    chat = _verifier_outage_stack(monkeypatch, "The company hired fifty new engineers [1].")
    result = chat._answer_one("What was revenue?", {"d1"}, "local")
    assert result["status"] == "answer_validation_failed"
    assert result["abstained"] is True
    assert result["verification"]["checked"] is False
    assert "verifier offline" in result["verification"]["error"]


def test_verifier_outage_withholds_even_lexically_overlapping_draft(monkeypatch):
    # Word overlap is not entailment. If the configured verifier is unavailable,
    # even a plausible-looking draft must not be promoted to a successful answer.
    chat = _verifier_outage_stack(monkeypatch, "Revenue was 10 crore [1].")
    result = chat._answer_one("What was revenue?", {"d1"}, "local")
    assert result["status"] == "answer_validation_failed"
    assert result["abstained"] is True
    assert result["verification"]["checked"] is False


def test_reconcile_resets_partial_vectors_instead_of_promoting(monkeypatch, tmp_path):
    from tests.test_persistence import configure_test_storage
    from app.services import documents

    db = configure_test_storage(monkeypatch, tmp_path)
    _document(db, tmp_path, status="processing", chunk_count=500)
    deleted = []
    fake_store = types.SimpleNamespace(
        list_chunk_document_ids=lambda: {"doc-a"},
        document_chunk_count=lambda document_id: 120,
        delete=lambda document_id: deleted.append(document_id),
        rebuild_fts_for_document=lambda document_id: None,
    )
    monkeypatch.setattr(documents, "store", fake_store)
    events = documents.reconcile()

    doc = db.get_document("doc-a")
    assert doc["processing_status"] == "failed"
    assert doc["indexing_status"] == "failed"
    assert doc["chunk_count"] == 0
    assert deleted == ["doc-a"]
    assert any("expected 500 chunks, found 120" in event for event in events)


def test_document_delete_is_blocked_while_ingestion_owns_it(monkeypatch, tmp_path):
    from tests.test_persistence import configure_test_storage
    from app.services import documents

    db = configure_test_storage(monkeypatch, tmp_path)
    _document(db, tmp_path, status="processing", chunk_count=0)
    monkeypatch.setattr(documents.store, "delete", lambda document_id: None)
    with pytest.raises(documents.DocumentBusyError):
        documents.remove_document_record("doc-a")

    db.update_document("doc-a", processing_status="indexed")
    now = datetime.now(timezone.utc).isoformat()
    db.create_job({"job_id": "active-job", "owner_id": "local", "filename": "doc-a.txt",
                   "staged_path": str(tmp_path / "staged"), "size_bytes": 10,
                   "status": "processing", "created_at": now, "updated_at": now,
                   "document_id": "doc-a"})
    with pytest.raises(documents.DocumentBusyError):
        documents.remove_document_record("doc-a")


def test_queue_admission_limits_are_enforced_atomically(monkeypatch, tmp_path):
    from tests.test_persistence import configure_test_storage

    db = configure_test_storage(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    job = {"job_id": "queue-1", "owner_id": "local", "filename": "one.txt",
           "staged_path": str(tmp_path / "one.txt"), "size_bytes": 8,
           "status": "queued", "created_at": now, "updated_at": now}
    db.create_job(job, max_pending=1, max_pending_bytes=10)
    second = dict(job, job_id="queue-2", filename="two.txt", size_bytes=1)
    with pytest.raises(RuntimeError, match="queue is full"):
        db.create_job(second, max_pending=1, max_pending_bytes=10)


def test_archive_expansion_limit_is_enforced(monkeypatch, tmp_path):
    import zipfile
    from app.core.config import settings
    from app.ingestion.extractor import _check_archive_limits

    archive_path = tmp_path / "large.docx"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "x" * 5000)
    monkeypatch.setattr(settings, "max_archive_uncompressed_mb", 0)
    with pytest.raises(ValueError, match="expands beyond"):
        _check_archive_limits(archive_path, ".docx")


def test_retry_admission_obeys_queue_capacity(monkeypatch, tmp_path):
    from tests.test_persistence import configure_test_storage

    db = configure_test_storage(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    for job_id, status in (("retry-me", "failed"), ("already-queued", "queued")):
        db.create_job({"job_id": job_id, "owner_id": "local", "filename": f"{job_id}.txt",
                       "staged_path": str(tmp_path / f"{job_id}.txt"), "size_bytes": 5,
                       "status": status, "created_at": now, "updated_at": now})
    with pytest.raises(RuntimeError, match="queue is full"):
        db.requeue_failed_job("retry-me", max_pending=1, max_pending_bytes=20)
    assert db.get_job("retry-me")["status"] == "failed"


def test_multi_question_does_not_hide_mixed_processing_failure(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    import app.rag.chat as chat

    monkeypatch.setattr(chat, "list_docs", lambda owner_id=None: [
        {"document_id": "d1", "processing_status": "indexed", "indexing_status": "complete"}
    ])
    def fake_answer(question, indexed, owner_id, doc_types=None):
        status = "processing_error" if "First" in question else "insufficient_evidence"
        return {"question": question, "status": status, "answer": "Not available.",
                "sources": [], "abstained": True}
    monkeypatch.setattr(chat, "_answer_one", fake_answer)
    result = chat.answer("First question? Second question?")
    assert result["status"] == "partial_failure"
    assert len(result["partial_failures"]) == 2


def test_ingestion_reads_staged_file_without_loading_content_argument(monkeypatch, tmp_path):
    import sys
    from app.core.config import settings
    from app.storage import database as db

    monkeypatch.setattr(settings, "uploads_path", tmp_path / "data" / "uploads")
    monkeypatch.setattr(settings, "extracted_path", tmp_path / "data" / "extracted")
    monkeypatch.setattr(settings, "sqlite_path", tmp_path / "data" / "metadata" / "documents.sqlite3")
    monkeypatch.setattr(settings, "chunk_size", 80)
    monkeypatch.setattr(settings, "chunk_overlap", 10)
    monkeypatch.setattr(settings, "embedding_batch_size", 2)
    db.initialize()

    saved_batches = []
    fake_store = types.ModuleType("app.retrieval.store")
    fake_store.add = lambda records, progress_callback=None: saved_batches.append(list(records))
    fake_store.delete = lambda document_id: None
    monkeypatch.setitem(sys.modules, "app.retrieval.store", fake_store)

    staged = tmp_path / "jobs" / "upload.txt.upload"
    staged.parent.mkdir(parents=True, exist_ok=True)
    payload = ("A durable disk-backed ingestion test with enough text to create several chunks. " * 20)
    staged.write_text(payload, encoding="utf-8")

    from app.ingestion.extractor import ingest
    result = ingest(str(staged), None, filename="upload.txt")
    doc = db.get_document(result["document_id"])
    assert result["status"] == "indexed"
    assert doc["chunk_count"] == sum(len(batch) for batch in saved_batches)
    assert Path(doc["original_path"]).read_text(encoding="utf-8") == payload
    assert not list(Path(settings.extracted_path).glob("*.chunks.jsonl"))


def test_pdf_ocr_failure_marks_document_incomplete(monkeypatch, tmp_path):
    import sys
    fitz = pytest.importorskip("fitz")
    from app.core.config import settings
    from app.ingestion.extractor import iter_extract

    pdf_path = tmp_path / "scan.pdf"
    image_path = tmp_path / "tiny.png"
    from PIL import Image
    Image.new("RGB", (32, 32), "white").save(image_path)
    document = fitz.open()
    page = document.new_page()
    page.insert_image(fitz.Rect(0, 0, 32, 32), filename=str(image_path))
    document.save(pdf_path)
    document.close()

    fake_tesseract = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd=None),
        image_to_string=lambda *a, **k: (_ for _ in ()).throw(TimeoutError("OCR timed out")),
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract)
    monkeypatch.setattr(settings, "ocr_enabled", True)
    with pytest.raises(RuntimeError, match="OCR failed"):
        list(iter_extract(pdf_path))


def test_pdf_image_page_requires_ocr_when_disabled(monkeypatch, tmp_path):
    fitz = pytest.importorskip("fitz")
    from app.core.config import settings
    from app.ingestion.extractor import NoReadableTextError, iter_extract
    from PIL import Image

    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    pdf_path = tmp_path / "scan.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_image(fitz.Rect(0, 0, 32, 32), filename=str(image_path))
    document.save(pdf_path)
    document.close()

    monkeypatch.setattr(settings, "ocr_enabled", False)
    with pytest.raises(NoReadableTextError, match="Enable local OCR"):
        list(iter_extract(pdf_path))


def test_verifier_receives_only_the_passages_cited_by_each_claim(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    import app.rag.chat as chat
    from app.core.config import settings

    prompts = []
    monkeypatch.setattr(settings, "verify_answers", True)
    monkeypatch.setattr(chat, "generate",
                        lambda prompt, num_predict=None: prompts.append(prompt) or "SUPPORTED ANSWERED")
    evidence = "[1] report.pdf (page 1): Revenue was 10 crore.\n\n[2] report.pdf (page 2): The company had 25 employees."
    result = chat._verify_answer(
        "What was revenue?", evidence, "Revenue was 10 crore [2].", {})
    assert result is True
    assert len(prompts) == 1
    assert "The company had 25 employees" in prompts[0]
    assert "Revenue was 10 crore." not in prompts[0]


def test_extractive_fallback_returns_only_a_verbatim_relevant_sentence():
    from app.rag.chat import _extractive_answer_for_question

    hits = [{"text": "Benjamin Franklin was a printer, writer, and public official."}]
    answer = _extractive_answer_for_question("Who was Benjamin Franklin?", hits)
    assert answer == "“Benjamin Franklin was a printer, writer, and public official.” [1]"
    life_answer = _extractive_answer_for_question("What does Benjamin do in his life?", hits)
    assert life_answer == "“Benjamin Franklin was a printer, writer, and public official.” [1]"
    assert _extractive_answer_for_question("What was the weather?", hits) is None


def test_failed_original_copy_does_not_leave_orphan_upload(monkeypatch, tmp_path):
    from app.core.config import settings
    import app.ingestion.extractor as extractor

    uploads = tmp_path / "uploads"
    extracted = tmp_path / "extracted"
    staged = tmp_path / "staged" / "broken.txt.upload"
    staged.parent.mkdir(parents=True)
    staged.write_text("source content", encoding="utf-8")
    monkeypatch.setattr(settings, "uploads_path", uploads)
    monkeypatch.setattr(settings, "extracted_path", extracted)
    monkeypatch.setattr(extractor, "_copy_file_durable",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        extractor.ingest(str(staged), None, filename="broken.txt")
    assert not list(uploads.glob("*"))
