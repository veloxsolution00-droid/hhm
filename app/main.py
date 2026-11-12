import logging, os, threading, time, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from app.core.config import settings
from app.storage import database as db
from app.retrieval.ollama import status
from app.retrieval.store import collection, backfill_owner_metadata, backfill_text_search
from app.services.documents import (list_docs, remove, reconcile, remove_document_record, remove_indexed_chunks,
    cancel_job, delete_ingestion_job, recover_deleting_jobs, JobNotFoundError, JobConflictError, JobDeletionError)
from app.ingestion.extractor import SUPPORTED
from app.ingestion.ocr_health import local_ocr_status
from app.services.job_queue import submit, start_queue, shutdown_queue
from app.rag.chat import answer
from app.cache.manager import stats as cache_stats

LOCAL_OWNER_ID = "local"
Path(settings.logs_path).mkdir(parents=True, exist_ok=True)
from logging.handlers import RotatingFileHandler
_handler = RotatingFileHandler(Path(settings.logs_path) / "application.log", maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
logging.basicConfig(level=logging.INFO, handlers=[_handler, logging.StreamHandler()], format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2000)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)

def _warm_models():
    """Load both Ollama models before the first chat request.

    Without this the first question pays model load on top of generation.
    Failures are non-fatal; /health still reports real Ollama status.
    """
    from app.retrieval.ollama import generate, embed
    try:
        generate("Reply with one word: ready", num_predict=4)
        embed(["warmup"])
        logger.info("Ollama generation and embedding models warmed")
    except Exception:
        logger.warning("Ollama warm-up skipped (service may be unavailable)", exc_info=True)


def startup_recovery():
    db.initialize()
    # This build is single-user and loopback-only. Preserve all existing records,
    # but normalize old authenticated installs to one local owner so old documents
    # remain visible and searchable after login is removed.
    db.migrate_to_local_owner()
    try:
        backfill_owner_metadata(db.list_documents())
        backfill_text_search()
    except Exception:
        logger.exception("Could not migrate legacy ChromaDB owner metadata; retrieval diagnostics will show if no chunks match.")
    if settings.recovery_on_startup:
        try:
            events = reconcile()
            logger.info("Startup reconciliation complete: %s event(s)", len(events))
            for event in events:
                logger.warning("Reconciliation: %s", event)
        except Exception:
            logger.exception("Startup reconciliation encountered an error; data was not cleared.")
    try:
        recovered_deletions = recover_deleting_jobs()
        if recovered_deletions:
            logger.info("Recovered %d interrupted ingestion deletion(s)", len(recovered_deletions))
    except Exception:
        logger.exception("Could not recover interrupted ingestion deletions")
    start_queue()
    # Warm in the background so startup stays responsive and the first question
    # does not also pay model load time.
    threading.Thread(target=_warm_models, daemon=True).start()

def shutdown():
    shutdown_queue()

@asynccontextmanager
async def lifespan(_app):
    startup_recovery()
    try:
        yield
    finally:
        shutdown()

app = FastAPI(title="Offline Document Intelligence Chatbot", version="0.5.0-local",
              lifespan=lifespan)

@app.get("/")
def root():
    return {"app": app.title, "docs": "/docs", "health": "/health", "mode": "local-single-user"}

@app.get("/health")
def health():
    try:
        c = collection()
        db_health = {"available": True, "count": c.count(), "path": str(settings.chroma_path)}
    except Exception as e:
        db_health = {"available": False, "error": str(e)}
    try:
        document_count = len(list_docs())
        metadata_health = {"available": True}
    except Exception as e:
        # A locked or unreadable metadata database should report a degraded
        # service, not turn the whole health endpoint into a 500.
        document_count = None
        metadata_health = {"available": False, "error": str(e)}
    try:
        cache = cache_stats()
    except Exception as e:
        cache = {"enabled": None, "error": str(e)}
    o = status()
    models_ready = bool(o.get("generation_model_available")) and bool(o.get("embedding_model_available"))
    ocr = local_ocr_status()
    ocr_ready = not ocr["enabled"] or ocr["available"]
    healthy = (bool(o.get("available")) and models_ready and db_health["available"]
               and metadata_health["available"] and ocr_ready)
    return {"status": "ok" if healthy else "degraded",
            "ollama": o, "models_ready": models_ready, "ocr": ocr,
            "chromadb": db_health, "metadata": metadata_health,
            "documents": document_count,
            "cache": cache,
            "sqlite_path": str(settings.sqlite_path), "data_path": str(settings.data_path), "uploads_path": str(settings.uploads_path),
            "extracted_path": str(settings.extracted_path), "jobs_path": str(settings.jobs_path),
            "logs_path": str(settings.logs_path), "limits": {"max_file_mb": settings.max_file_mb,
            "max_batch_mb": settings.max_batch_mb, "max_upload_files": settings.max_upload_files}}

def _directory_size_bytes(directory: Path) -> int:
    """Best-effort bounded-by-filesystem accounting for local storage quota."""
    total = 0
    if not directory.exists():
        return 0
    for current, _dirs, names in os.walk(directory):
        for name in names:
            try:
                total += (Path(current) / name).stat().st_size
            except OSError:
                continue
    return total


@app.post("/documents/upload", status_code=202)
async def upload(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "Choose at least one file.")
    if len(files) > settings.max_upload_files:
        raise HTTPException(413, f"Too many files. Maximum is {settings.max_upload_files} per batch.")
    max_file = settings.max_file_mb * 1024 * 1024
    max_batch = settings.max_batch_mb * 1024 * 1024
    sizes, total = [], 0
    for upload_file in files:
        try:
            upload_file.file.seek(0, 2)
            size = upload_file.file.tell()
            upload_file.file.seek(0)
        except Exception:
            size = -1
        filename = Path((upload_file.filename or "upload").replace("\\", "/")).name
        if Path(filename).suffix.lower() not in SUPPORTED:
            raise HTTPException(415, f"{filename}: unsupported file type. Supported extensions: {', '.join(sorted(SUPPORTED))}")
        if size < 0 or size > max_file:
            raise HTTPException(413, f"{filename}: individual file limit is {settings.max_file_mb} MB.")
        total += size
        sizes.append(size)
    if total > max_batch:
        raise HTTPException(413, f"Batch exceeds {settings.max_batch_mb} MB total.")
    current_storage = _directory_size_bytes(Path(settings.data_path))
    storage_limit = int(settings.max_data_storage_mb) * 1024 * 1024
    if current_storage + total > storage_limit:
        raise HTTPException(507, f"Local document storage limit ({settings.max_data_storage_mb} MB) would be exceeded.")

    batch_started = time.perf_counter()
    Path(settings.jobs_path).mkdir(parents=True, exist_ok=True)
    staged_items = []
    created_jobs = []
    try:
        # Stage every file before creating any job. A staging failure is therefore
        # all-or-nothing instead of leaving earlier files queued after HTTP 500.
        for upload_file, size in zip(files, sizes):
            filename = Path((upload_file.filename or "upload").replace("\\", "/")).name
            job_id = str(uuid.uuid4())
            ext = Path(filename).suffix.lower()
            staged = Path(settings.jobs_path) / f"{job_id}{ext}.upload"
            staged_items.append({"job_id": job_id, "filename": filename,
                                 "staged_path": staged, "size_bytes": size})
            copied = 0
            with staged.open("wb") as out:
                while True:
                    block = await upload_file.read(1024 * 1024)
                    if not block:
                        break
                    copied += len(block)
                    if copied > max_file or copied > size:
                        raise HTTPException(413, f"{filename}: upload exceeded configured size limit.")
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            if copied != size:
                raise HTTPException(400, f"{filename}: uploaded size did not match the staged file size.")
            staged_items[-1]["size_bytes"] = copied
        max_pending_bytes = settings.max_pending_upload_mb * 1024 * 1024
        for item in staged_items:
            now = datetime.now(timezone.utc).isoformat()
            db.create_job({
                "job_id": item["job_id"], "owner_id": LOCAL_OWNER_ID,
                "filename": item["filename"], "staged_path": str(item["staged_path"]),
                "size_bytes": item["size_bytes"], "status": "queued",
                "stage": "Queued for extraction", "progress": 0,
                "created_at": now, "updated_at": now, "document_id": None, "error": None,
            }, max_pending=settings.max_pending_jobs, max_pending_bytes=max_pending_bytes)
            created_jobs.append(item["job_id"])
    except HTTPException:
        for job_id in created_jobs:
            db.rollback_queued_job(job_id)
        for item in staged_items:
            item["staged_path"].unlink(missing_ok=True)
        raise
    except RuntimeError as exc:
        for job_id in created_jobs:
            db.rollback_queued_job(job_id)
        for item in staged_items:
            item["staged_path"].unlink(missing_ok=True)
        raise HTTPException(429, str(exc)) from exc
    except Exception as exc:
        for job_id in created_jobs:
            db.rollback_queued_job(job_id)
        for item in staged_items:
            item["staged_path"].unlink(missing_ok=True)
        logger.exception("Batch upload staging failed")
        raise HTTPException(500, f"Could not stage upload: {exc}") from exc
    finally:
        for upload_file in files:
            await upload_file.close()

    results = []
    for item in staged_items:
        try:
            submit(item["job_id"])
            status_value = "queued"
            submit_error = None
        except Exception as exc:
            # The durable queued job is retained; startup recovery can dispatch it.
            logger.exception("Could not dispatch queued job %s", item["job_id"])
            status_value = "queued_pending_worker"
            submit_error = str(exc)
        results.append({"job_id": item["job_id"], "filename": item["filename"],
                        "status": status_value, "size_bytes": item["size_bytes"],
                        **({"dispatch_error": submit_error} if submit_error else {})})
    return {"status": "accepted", "files": len(results), "total_bytes": total,
            "staging_seconds": round(time.perf_counter() - batch_started, 3),
            "message": "All files were staged durably and persisted as jobs. Job status reports processing progress.",
            "results": results}

@app.get("/jobs")
def jobs():
    return {"jobs": db.list_jobs(LOCAL_OWNER_ID)}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = db.get_job(job_id, LOCAL_OWNER_ID)
    if not job: raise HTTPException(404, "Job not found.")
    return job

@app.post("/jobs/{job_id}/retry")
def retry_job(job_id: str):
    job = db.get_job(job_id, LOCAL_OWNER_ID)
    if not job: raise HTTPException(404, "Job not found.")
    if job["status"] != "failed": raise HTTPException(409, "Only failed jobs can be retried.")
    if not Path(job["staged_path"]).exists(): raise HTTPException(410, "Staged file is missing; upload it again.")
    # Atomic transition: two simultaneous retry clicks must not both queue the job.
    try:
        queued = db.requeue_failed_job(
            job_id,
            max_pending=settings.max_pending_jobs,
            max_pending_bytes=settings.max_pending_upload_mb * 1024 * 1024,
        )
    except RuntimeError as exc:
        raise HTTPException(429, str(exc)) from exc
    if not queued:
        raise HTTPException(409, "Only failed jobs with a non-deleting document can be retried.")
    try:
        submit(job_id)
    except Exception:
        # The job is already durably queued, so a dispatch failure must not
        # surface as a 500 that hides the successful requeue. Startup recovery
        # will dispatch the retained queued job.
        logger.exception("Could not dispatch retried job %s", job_id)
        return {"job_id": job_id, "status": "queued_pending_worker",
                "message": "Job is queued; the worker will pick it up on the next dispatch."}
    return {"job_id": job_id, "status": "queued"}


@app.post("/jobs/{job_id}/cancel")
def cancel_ingestion_job(job_id: str):
    """Cancel queued work immediately or request a safe cooperative worker stop."""
    state = cancel_job(job_id)
    if state is None:
        raise HTTPException(404, "Ingestion job not found.")
    if state in ("complete", "failed", "cancelled", "deleting", "delete_failed"):
        raise HTTPException(409, f"Job cannot be stopped in state '{state}'.")
    return {"job_id": job_id, "status": state,
            "worker_stopped": state == "cancelled",
            "message": "Worker has stopped; deletion is safe." if state == "cancelled" else
                       "Cancellation requested. Wait for the worker to acknowledge before deleting."}


def _delete_ingestion_response(job_id: str):
    try:
        document_preserved = delete_ingestion_job(job_id)
    except JobNotFoundError:
        raise HTTPException(404, "Ingestion job not found.")
    except JobConflictError as exc:
        raise HTTPException(409, str(exc))
    except JobDeletionError as exc:
        raise HTTPException(500, f"Ingestion deletion failed: {exc}")
    return {"job_id": job_id, "status": "deleted", "document_preserved": bool(document_preserved)}


@app.post("/jobs/{job_id}/delete")
@app.delete("/jobs/{job_id}/delete")
def delete_ingestion_endpoint(job_id: str):
    """Delete only the ingestion history entry; preserve the indexed document and chunks."""
    return _delete_ingestion_response(job_id)

@app.get("/documents")
def documents():
    return {"documents": list_docs(LOCAL_OWNER_ID)}

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    doc = db.get_document(document_id)
    if not doc or doc.get("owner_id", LOCAL_OWNER_ID) != LOCAL_OWNER_ID:
        raise HTTPException(404, "Document not found.")
    try:
        if not remove(document_id):
            raise HTTPException(404, "Document not found.")
        return {"status": "deleted", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        from app.services.documents import DocumentBusyError
        if isinstance(e, DocumentBusyError):
            raise HTTPException(409, str(e)) from e
        raise HTTPException(500, f"Deletion incomplete and recorded for recovery: {e}") from e

@app.post("/admin/reconcile")
def admin_reconcile():
    try: return {"events": reconcile()}
    except Exception as e:
        logger.exception("Manual reconciliation failed")
        raise HTTPException(500, f"Reconciliation failed: {e}")

@app.get("/admin/recovery-events")
def get_recovery_events():
    return {"events": db.recovery_events()}

@app.post("/chat")
def chat(req: ChatRequest):
    started = time.perf_counter()
    result = answer(req.question, owner_id=LOCAL_OWNER_ID, history=[item.model_dump() for item in req.history])
    result["response_seconds"] = round(time.perf_counter()-started, 3)
    return result
