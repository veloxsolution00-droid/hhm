"""Persistent, bounded background ingestion queue. Staged originals remain on disk until successful."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import logging
from app.core.config import settings
from app.storage import database as db
from app.ingestion.extractor import ingest

logger = logging.getLogger(__name__)
_executor = None

def now(): return datetime.now(timezone.utc).isoformat()


def _remove_completed_history(job_id, staged_path=None):
    """Purge successful job history without touching its indexed document."""
    try:
        if staged_path:
            Path(staged_path).unlink(missing_ok=True)
        db.delete_completed_job_record(job_id)
    except Exception:
        # Preserve the completed row so startup can retry cleanup.
        logger.exception("Could not automatically remove completed job history: %s", job_id)


def _purge_old_completed_history():
    """Clean up completed history left behind by a previous interrupted shutdown."""
    for job in db.completed_jobs():
        _remove_completed_history(job["job_id"], job.get("staged_path"))

class IngestionCancelled(Exception):
    """Raised at a safe checkpoint after the user requests cancellation."""


def _run(job_id):
    # Claim the job atomically first: a queued job must never be processed by
    # two workers at the same time (double retry, or recovery racing an upload).
    if not db.claim_job(job_id):
        return
    job = db.get_job(job_id)
    if not job:
        return
    path = Path(job["staged_path"])
    # A retry replaces its previous failed attempt cleanly while preserving the
    # staged upload. Remove the failed attempt's partial index and artifacts.
    if job.get("document_id"):
        previous = db.get_document(job["document_id"])
        if previous and previous.get("processing_status") != "indexed":
            try:
                from app.services.documents import remove_document_record
                remove_document_record(job["document_id"], allow_active_job_id=job_id)
            except Exception:
                logger.exception("Could not clean previous failed attempt for job %s", job_id)
                db.update_job(job_id, status="failed", stage="Retry cleanup failed", updated_at=now(), error="Could not clean previous failed attempt")
                return
    if not db.update_job_if_status(job_id, "processing", stage="Starting", progress=1, updated_at=now(), error=None):
        current = db.get_job(job_id)
        if current and current.get("status") == "cancel_requested":
            db.update_job_if_status(job_id, "cancel_requested", status="cancelled",
                                    stage="Cancelled before extraction", progress=100,
                                    updated_at=now(), error=None)
            path.unlink(missing_ok=True)
        return
    try:
        # The ingestion pipeline reads from disk incrementally; do not load a
        # potentially large staged upload into worker memory.
        def progress(stage, percent, message):
            current = db.get_job(job_id)
            if not current or current.get("status") in ("cancel_requested", "cancelled"):
                raise IngestionCancelled("Cancellation requested; worker stopped at a safe checkpoint")
            if current.get("status") != "processing":
                raise RuntimeError(f"Unexpected ingestion job state: {current.get('status')}")
            if not db.update_job_if_status(job_id, "processing", stage=f"{stage}: {message}", progress=max(1, min(99, int(percent))), updated_at=now()):
                raise IngestionCancelled("Cancellation requested; worker stopped at a safe checkpoint")
        result = ingest(str(path), None, owner_id="local", progress_callback=progress, filename=job["filename"], job_id=job_id)
        doc_id = result.get("document_id")
        doc = db.get_document(doc_id) if doc_id else None
        if doc_id:
            db.update_job(job_id, document_id=doc_id, document_created=0 if result.get("duplicate") else 1, updated_at=now())
        if not doc or doc.get("processing_status") != "indexed" or doc.get("indexing_status") != "complete":
            detail = (doc or {}).get("processing_error") or "Ingestion returned without a fully indexed document."
            raise RuntimeError(detail)
        completed = db.update_job_if_status(job_id, "processing", status="complete", stage="Searchable",
                                             progress=100, updated_at=now(), document_id=doc_id,
                                             document_created=0 if result.get("duplicate") else 1, error=None)
        if not completed:
            current = db.get_job(job_id)
            if current and current.get("status") == "cancel_requested":
                if doc_id and not result.get("duplicate"):
                    from app.services.documents import remove_document_record
                    remove_document_record(doc_id, allow_active_job_id=job_id)
                db.update_job_if_status(job_id, "cancel_requested", status="cancelled", stage="Worker stopped; safe to delete",
                                        progress=100, updated_at=now())
                path.unlink(missing_ok=True)
                return
            if current and current.get("status") == "cancelled":
                path.unlink(missing_ok=True)
                return
            raise RuntimeError("Job state changed before completion could be committed")
        _remove_completed_history(job_id, path)
    except Exception as exc:
        current = db.get_job(job_id)
        if current and current.get("status") == "cancel_requested":
            # The extractor has already rolled back any partial vectors/FTS rows.
            # Mark cancelled only after control has returned from the worker.
            db.update_job_if_status(job_id, "cancel_requested", status="cancelled",
                                    stage="Worker stopped; safe to delete", progress=100,
                                    updated_at=now(), error=None)
            path.unlink(missing_ok=True)
            logger.info("Ingestion cancellation completed: %s", job_id)
            return
        if isinstance(exc, IngestionCancelled) or (current and current.get("status") == "cancelled"):
            path.unlink(missing_ok=True)
            return
        logger.exception("Ingestion job failed: %s", job_id)
        if current and current.get("status") == "processing":
            db.update_job_if_status(job_id, "processing", status="failed", stage="Failed",
                                    progress=100, updated_at=now(), error=str(exc))
        # Staged file intentionally retained for diagnosis/retry; never silently discard it.

def start_queue():
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=max(1, min(settings.ingestion_workers, 2)), thread_name_prefix="doc-ingest")
        # A previous process may have died mid-ingestion. Requeue those jobs
        # before listing recoverable work so they are not stuck at 'processing'.
        db.reset_processing_jobs()
        _purge_old_completed_history()
    for job in db.recoverable_jobs():
        _executor.submit(_run, job["job_id"])

def submit(job_id):
    if _executor is None: start_queue()
    _executor.submit(_run, job_id)

def shutdown_queue():
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=False)
        _executor = None
