"""Document and ingestion-job service facade over the canonical SQLite/vector stores."""
from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime, timezone
from app.core.config import settings
from app.storage import database as db
from app.retrieval import store

logger = logging.getLogger(__name__)
LOCAL_OWNER_ID = "local"

def now(): return datetime.now(timezone.utc).isoformat()

def list_docs(owner_id="local"):
    docs = db.list_documents(owner_id)
    result=[]
    for doc in docs:
        item=dict(doc)
        item.update({"document_type":item.get("file_format"), "storage_path":item.get("original_path"),
                     "upload_date":item.get("uploaded_at"), "status":item.get("processing_status"),
                     "metadata":item.get("source_metadata") or {},
                     "source_metadata":item.get("source_metadata") or {}})
        result.append(item)
    return result

def save_document(document_id, filename, document_type, storage_path, file_size=0):
    """Compatibility API; ingestion uses database.save_document with full metadata."""
    import hashlib
    p=Path(storage_path); digest=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""
    db.save_document({"document_id":document_id,"owner_id":LOCAL_OWNER_ID,"filename":filename,
        "stored_filename":p.name,"original_path":str(p),"content_hash":digest,"file_format":document_type,
        "uploaded_at":now(),"updated_at":now(),"processing_status":"uploaded","extraction_status":"pending",
        "indexing_status":"pending","chunk_count":0,"source_metadata":{},"extracted_path":None,"processing_error":None})

def reconcile():
    """Reconcile metadata, vectors and FTS without promoting partial indexes."""
    events = []
    documents = db.list_documents()
    known_ids = {doc["document_id"] for doc in documents}
    vector_ids = None
    try:
        vector_ids = store.list_chunk_document_ids()
    except Exception as exc:
        events.append(f"ChromaDB orphan scan unavailable: {exc}")
        db.log_recovery_event("warning", "orphan_vector_scan_unavailable", exc, None)
    fts_ids = db.indexed_document_ids()

    for doc in documents:
        document_id = doc["document_id"]
        if doc.get("processing_status") == "deleting":
            try:
                remove_document_record(document_id, resume=True)
                events.append(f"Completed interrupted deletion: {doc.get('filename')}")
            except Exception as exc:
                message = f"Interrupted deletion still needs recovery: {doc.get('filename')}: {exc}"
                events.append(message)
                db.log_recovery_event("error", "document_deletion_recovery_failed", message, document_id)
            continue
        original = Path(doc.get("original_path") or "")
        if not original.is_file():
            message = f"Original missing: {doc.get('filename')} ({document_id})"
            db.update_document(document_id, processing_status="recovery_required",
                               indexing_status="recovery_required", processing_error=message,
                               updated_at=now())
            events.append(message)
            db.log_recovery_event("error", "missing_original", message, document_id)
            continue

        actual = store.document_chunk_count(document_id)
        expected = int(doc.get("chunk_count") or 0)
        fts_actual = db.indexed_chunk_count_for_document(document_id)

        # A vector count is not proof of completion. Only a non-zero exact match
        # against the durable expected count can be promoted to indexed.
        if expected > 0 and actual == expected:
            if fts_actual != expected:
                db.remove_indexed_chunks(document_id)
                store.rebuild_fts_for_document(document_id)
                fts_actual = db.indexed_chunk_count_for_document(document_id)
            if fts_actual == expected and (
                doc.get("processing_status") != "indexed" or doc.get("indexing_status") != "complete"
            ):
                db.update_document(document_id, indexing_status="complete", processing_status="indexed",
                                   extraction_status="complete", updated_at=now(), processing_error=None)
                events.append(f"Restored verified indexed status: {doc['filename']}")
            elif fts_actual != expected:
                message = (f"FTS index mismatch: {doc['filename']} expected {expected}, "
                           f"found {fts_actual}")
                db.update_document(document_id, processing_status="recovery_required",
                                   indexing_status="recovery_required", processing_error=message,
                                   updated_at=now())
                events.append(message)
        elif actual or expected or (
            doc.get("processing_status") == "indexed" and doc.get("indexing_status") == "complete"
        ):
            # Remove partial vectors/FTS so incomplete documents cannot appear
            # searchable. Keep original + registry for a controlled retry.
            store.delete(document_id)
            message = (f"Incomplete index reset: {doc['filename']} expected {expected} "
                       f"chunks, found {actual}")
            db.update_document(document_id, processing_status="failed", indexing_status="failed",
                               chunk_count=0, processing_error=message, updated_at=now())
            events.append(message)
            db.log_recovery_event("warning", "partial_index", message, document_id)

    orphan_vectors = sorted(vector_ids - known_ids) if vector_ids is not None else []
    orphan_fts = sorted(fts_ids - known_ids)
    if orphan_vectors:
        message = f"Orphan ChromaDB document IDs detected: {len(orphan_vectors)}"
        events.append(message)
        db.log_recovery_event("warning", "orphaned_vectors", message, None)
    if orphan_fts:
        message = f"Orphan SQLite FTS document IDs detected: {len(orphan_fts)}"
        events.append(message)
        db.log_recovery_event("warning", "orphaned_fts", message, None)
    return events

class JobNotFoundError(Exception):
    pass


class JobConflictError(Exception):
    pass


class JobDeletionError(Exception):
    pass


class DocumentBusyError(Exception):
    """Raised when an ingestion worker still owns a document."""


def remove_document_record(document_id, allow_active_job_id=None, resume=False):
    """Delete vectors, FTS rows and files before removing the durable registry row.

    The order is intentional: if cleanup fails, the metadata row remains so the
    operation can be retried instead of losing the paths needed for recovery.
    """
    doc = db.get_document(document_id)
    if not doc:
        return False
    if not resume and not db.claim_document_deletion(document_id, exclude_job_id=allow_active_job_id):
        raise DocumentBusyError(
            "Document is being ingested, retried, or deleted. Wait for that operation to finish first."
        )
    try:
        from app.cache.manager import invalidate_document
        invalidate_document(document_id)
        store.delete(document_id)
        for key in ("original_path", "extracted_path"):
            value = doc.get(key)
            if value:
                path = Path(value)
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    raise JobDeletionError(f"Could not remove {key}: {path}: {exc}") from exc
        db.remove_document_record(document_id)
        return True
    except Exception as exc:
        db.mark_document_deletion_failed(document_id, exc)
        raise

def remove_indexed_chunks(document_id):
    from app.cache.manager import invalidate_document
    invalidate_document(document_id)
    store.collection().delete(where={"document_id":document_id})
    db.remove_indexed_chunks(document_id)

def remove(document_id):
    # Let the API distinguish a missing document, an active worker conflict,
    # and a genuine storage cleanup failure.
    return remove_document_record(document_id)

def get_job(job_id): return db.get_job(job_id)
def list_jobs(owner_id="local"): return db.list_jobs(owner_id)
def submit_ingestion_job(job_id, filename, document_id, staged_path, size_bytes):
    db.create_job({"job_id":job_id,"owner_id":LOCAL_OWNER_ID,"filename":filename,"staged_path":staged_path,
                   "size_bytes":size_bytes,"status":"queued","created_at":now(),"updated_at":now(),"document_id":document_id})

def cancel_job(job_id):
    """Cancel queued work or request a cooperative stop from an active worker."""
    job = db.get_job(job_id)
    result = db.request_job_cancellation(job_id)
    if result == "cancelled" and job and job.get("status") == "queued":
        staged = job.get("staged_path")
        if staged:
            try:
                Path(staged).unlink(missing_ok=True)
            except OSError:
                logger.exception("Could not remove staged upload for cancelled job %s", job_id)
    return result


def delete_ingestion_job(job_id, resume=False):
    """Delete only a terminal ingestion-history record.

    The indexed document is independent of its ingestion job. Its original file,
    extracted artifact, metadata, FTS rows, and vector chunks remain untouched;
    document deletion is handled exclusively by the documents endpoint.
    """
    job = db.get_job(job_id)
    if not job:
        raise JobNotFoundError(job_id)
    state = db.transition_job_to_deleting(job_id, resume=resume)
    if state is None:
        raise JobNotFoundError(job_id)
    if state in ("processing", "cancel_requested"):
        raise JobConflictError("The ingestion worker is still active. Wait until cancellation is confirmed.")
    if state == "already_deleting":
        raise JobConflictError("Deletion is already in progress for this job.")
    if state != "deleting":
        raise JobConflictError(f"Job cannot be deleted in state '{state}'.")

    # Re-read after acquiring the deleting state so we know the final document
    # reference, but never delete or mutate that document from this operation.
    job = db.get_job(job_id)
    if not job:
        raise JobNotFoundError(job_id)

    try:
        staged = Path(job["staged_path"])
        staged.unlink(missing_ok=True)
        if not db.delete_job_record(job_id):
            raise JobDeletionError("The job changed while deletion was being finalized.")
        return bool(job.get("document_id"))
    except Exception as exc:
        logger.exception("Ingestion-history deletion failed for job %s", job_id)
        try:
            db.update_job(job_id, status="delete_failed", stage="History deletion incomplete; retry is safe", error=str(exc), updated_at=now())
        except Exception:
            logger.exception("Could not persist deletion failure for job %s", job_id)
        if isinstance(exc, JobDeletionError):
            raise
        raise JobDeletionError(str(exc)) from exc


def recover_deleting_jobs():
    """Resume interrupted deletion transactions during application startup."""
    recovered = []
    for job in db.jobs_with_status("deleting"):
        try:
            delete_ingestion_job(job["job_id"], resume=True)
            recovered.append(job["job_id"])
        except Exception:
            logger.exception("Could not recover deletion for job %s", job["job_id"])
    return recovered

