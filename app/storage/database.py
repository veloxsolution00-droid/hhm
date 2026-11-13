import sqlite3
from functools import lru_cache
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from app.core.config import settings

class ClosingConnection(sqlite3.Connection):
    """SQLite connection whose context manager also closes the handle."""
    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT 'local',
    filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    original_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    file_format TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    processing_status TEXT NOT NULL,
    extraction_status TEXT NOT NULL,
    indexing_status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    source_metadata_json TEXT NOT NULL DEFAULT '{}',
    extracted_path TEXT,
    processing_error TEXT,
    last_reconciled_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    staged_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    document_id TEXT,
    error TEXT,
    stage TEXT NOT NULL DEFAULT 'Queued',
    progress INTEGER NOT NULL DEFAULT 0,
    document_created INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_jobs_owner_created ON ingestion_jobs(owner_id, created_at DESC);
CREATE TABLE IF NOT EXISTS deletion_operations (
    operation_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS recovery_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    severity TEXT NOT NULL,
    document_id TEXT,
    event_type TEXT NOT NULL,
    details TEXT NOT NULL
);
"""

def ensure_directories():
    for path in (settings.data_path, settings.uploads_path, settings.extracted_path,
                 settings.chroma_path, settings.metadata_path, settings.logs_path, settings.jobs_path):
        Path(path).mkdir(parents=True, exist_ok=True)
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)

def connect():
    ensure_directories()
    conn = sqlite3.connect(str(settings.sqlite_path), timeout=30, factory=ClosingConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def initialize():
    with connect() as conn:
        conn.executescript(SCHEMA)
        # FTS5 is bundled with standard SQLite builds. Keep the application
        # usable on an unusual build without it; vector retrieval remains the
        # safe fallback in that case.
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(
                    chunk_id UNINDEXED, document_id UNINDEXED, owner_id UNINDEXED,
                    filename UNINDEXED, source_location UNINDEXED, text
                )
            """)
        except sqlite3.OperationalError as exc:
            if "fts5" not in str(exc).lower():
                raise
        # Safe, additive migration for databases created by earlier versions.
        columns = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "owner_id" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'local'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash_owner ON documents(owner_id, content_hash)")
        job_columns = {r[1] for r in conn.execute("PRAGMA table_info(ingestion_jobs)").fetchall()}
        if "stage" not in job_columns:
            conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN stage TEXT NOT NULL DEFAULT 'Queued'")
        if "progress" not in job_columns:
            conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN progress INTEGER NOT NULL DEFAULT 0")
        if "document_created" not in job_columns:
            # Existing completed jobs own their associated document; jobs without
            # a document_id are unaffected. Duplicate uploads set this to 0.
            conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN document_created INTEGER NOT NULL DEFAULT 1")

@contextmanager
def transaction():
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def row_to_document(row):
    if row is None:
        return None
    d = dict(row)
    import json
    d["source_metadata"] = json.loads(d.pop("source_metadata_json") or "{}")
    return d

def get_document(document_id):
    with connect() as conn:
        return row_to_document(conn.execute(
            "SELECT * FROM documents WHERE document_id=?", (document_id,)
        ).fetchone())

def get_by_hash(content_hash, owner_id="local"):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE content_hash=? AND owner_id=? ORDER BY uploaded_at LIMIT 1",
            (content_hash, owner_id)
        ).fetchone()
        return row_to_document(row)

def list_documents(owner_id=None):
    with connect() as conn:
        if owner_id is None:
            rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        else:
            rows = conn.execute("SELECT * FROM documents WHERE owner_id=? ORDER BY uploaded_at DESC", (owner_id,)).fetchall()
        return [row_to_document(r) for r in rows]

def save_document(doc):
    import json
    with transaction() as conn:
        conn.execute("""
        INSERT INTO documents (
          document_id,owner_id,filename,stored_filename,original_path,content_hash,file_format,
          uploaded_at,updated_at,processing_status,extraction_status,indexing_status,
          chunk_count,source_metadata_json,extracted_path,processing_error,last_reconciled_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(document_id) DO UPDATE SET
          owner_id=excluded.owner_id, filename=excluded.filename, stored_filename=excluded.stored_filename,
          original_path=excluded.original_path, content_hash=excluded.content_hash,
          file_format=excluded.file_format, uploaded_at=excluded.uploaded_at,
          updated_at=excluded.updated_at, processing_status=excluded.processing_status,
          extraction_status=excluded.extraction_status, indexing_status=excluded.indexing_status,
          chunk_count=excluded.chunk_count, source_metadata_json=excluded.source_metadata_json,
          extracted_path=excluded.extracted_path, processing_error=excluded.processing_error,
          last_reconciled_at=excluded.last_reconciled_at
        """, (
            doc["document_id"], doc.get("owner_id", "local"), doc["filename"], doc["stored_filename"], doc["original_path"],
            doc["content_hash"], doc["file_format"], doc["uploaded_at"], doc["updated_at"],
            doc["processing_status"], doc["extraction_status"], doc["indexing_status"],
            doc.get("chunk_count", 0), json.dumps(doc.get("source_metadata", {})),
            doc.get("extracted_path"), doc.get("processing_error"), doc.get("last_reconciled_at")
        ))

def update_document(document_id, **fields):
    import json
    allowed = {"updated_at","processing_status","extraction_status","indexing_status",
               "chunk_count","source_metadata","extracted_path","processing_error","last_reconciled_at"}
    values = []
    assignments = []
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"Unsupported document field: {key}")
        column = "source_metadata_json" if key == "source_metadata" else key
        if key == "source_metadata":
            value = json.dumps(value)
        assignments.append(f"{column}=?")
        values.append(value)
    if not assignments:
        return
    values.append(document_id)
    with transaction() as conn:
        conn.execute(f"UPDATE documents SET {', '.join(assignments)} WHERE document_id=?", values)

def remove_document_record(document_id):
    with transaction() as conn:
        conn.execute("DELETE FROM documents WHERE document_id=?", (document_id,))

def log_recovery_event(severity, event_type, details, document_id=None):
    from datetime import datetime, timezone
    with transaction() as conn:
        conn.execute(
            "INSERT INTO recovery_events(created_at,severity,document_id,event_type,details) VALUES(?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), severity, document_id, event_type, str(details))
        )

def create_deletion_operation(operation_id, document_id):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with transaction() as conn:
        conn.execute("INSERT INTO deletion_operations VALUES(?,?,?,?,?,?)",
                     (operation_id, document_id, "pending", now, now, None))

def update_deletion_operation(operation_id, status, error=None):
    from datetime import datetime, timezone
    with transaction() as conn:
        conn.execute("UPDATE deletion_operations SET status=?,updated_at=?,error=? WHERE operation_id=?",
                     (status, datetime.now(timezone.utc).isoformat(), error, operation_id))

def pending_deletions():
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM deletion_operations WHERE status!='complete' ORDER BY created_at"
        ).fetchall()]

def recovery_events(limit=100):
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM recovery_events ORDER BY event_id DESC LIMIT ?", (limit,)
        ).fetchall()]


def create_job(job, max_pending=None, max_pending_bytes=None):
    """Persist a job only when configured queue admission limits permit it."""
    with transaction() as conn:
        if max_pending is not None:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM ingestion_jobs "
                "WHERE status IN ('queued','processing','cancel_requested')"
            ).fetchone()
            if int(row[0]) >= int(max_pending):
                raise RuntimeError("Ingestion queue is full; wait for pending jobs to finish.")
            if max_pending_bytes is not None and int(row[1]) + int(job["size_bytes"]) > int(max_pending_bytes):
                raise RuntimeError("Pending upload storage limit reached; wait for queued files to finish.")
        conn.execute("INSERT INTO ingestion_jobs(job_id,owner_id,filename,staged_path,size_bytes,status,created_at,updated_at,document_id,error,stage,progress,document_created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (job["job_id"], job.get("owner_id", "local"), job["filename"], job["staged_path"], job["size_bytes"], job["status"], job["created_at"], job["updated_at"], job.get("document_id"), job.get("error"), job.get("stage", "Queued"), job.get("progress", 0), int(job.get("document_created", 1))))
    return True

def update_job(job_id, **fields):
    allowed = {"status", "updated_at", "document_id", "error", "stage", "progress", "document_created"}
    if not fields or not set(fields).issubset(allowed):
        raise ValueError("Unsupported job fields")
    with transaction() as conn:
        conn.execute("UPDATE ingestion_jobs SET " + ",".join(f"{k}=?" for k in fields) + " WHERE job_id=?", (*fields.values(), job_id))


def update_job_if_status(job_id, expected_status, **fields):
    allowed = {"status", "updated_at", "document_id", "error", "stage", "progress", "document_created"}
    if not fields or not set(fields).issubset(allowed):
        raise ValueError("Unsupported job fields")
    with transaction() as conn:
        cursor = conn.execute(
            "UPDATE ingestion_jobs SET " + ",".join(f"{k}=?" for k in fields) + " WHERE job_id=? AND status=?",
            (*fields.values(), job_id, expected_status),
        )
        return cursor.rowcount == 1

def get_job(job_id, owner_id=None):
    with connect() as conn:
        if owner_id is None:
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_id=?", (job_id,)).fetchone()
        else:
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_id=? AND owner_id=?", (job_id, owner_id)).fetchone()
        return dict(row) if row else None

def request_job_cancellation(job_id):
    """Atomically cancel queued work or request cooperative stop of a worker."""
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).isoformat()
    with transaction() as conn:
        row = conn.execute("SELECT status FROM ingestion_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        status = row["status"]
        if status == "queued":
            conn.execute("UPDATE ingestion_jobs SET status='cancelled',stage='Cancelled before start',updated_at=? WHERE job_id=? AND status='queued'", (stamp, job_id))
            return "cancelled"
        if status == "processing":
            conn.execute("UPDATE ingestion_jobs SET status='cancel_requested',stage='Waiting for worker to stop safely',updated_at=? WHERE job_id=? AND status='processing'", (stamp, job_id))
            return "cancel_requested"
        return status


def transition_job_to_deleting(job_id, resume=False):
    """Claim terminal job deletion; a worker can no longer claim this job."""
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).isoformat()
    allowed = ("queued", "cancelled", "failed", "complete", "delete_failed", "deleting")
    with transaction() as conn:
        row = conn.execute("SELECT status FROM ingestion_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        status = row["status"]
        if status in ("processing", "cancel_requested"):
            return status
        if status == "deleting":
            return "deleting" if resume else "already_deleting"
        if status not in allowed:
            return status
        cursor = conn.execute("UPDATE ingestion_jobs SET status='deleting',stage='Deleting document and ingestion data',updated_at=? WHERE job_id=? AND status=?", (stamp, job_id, status))
        return "deleting" if cursor.rowcount == 1 else (conn.execute("SELECT status FROM ingestion_jobs WHERE job_id=?", (job_id,)).fetchone() or {"status": "missing"})["status"]


def jobs_with_status(status):
    with connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM ingestion_jobs WHERE status=? ORDER BY created_at", (status,)).fetchall()]


def delete_job_record(job_id):
    with transaction() as conn:
        cursor = conn.execute("DELETE FROM ingestion_jobs WHERE job_id=? AND status='deleting'", (job_id,))
        return cursor.rowcount == 1





def rollback_queued_job(job_id):
    """Remove a newly staged queued job during an all-or-nothing batch rollback."""
    with transaction() as conn:
        cursor = conn.execute(
            "DELETE FROM ingestion_jobs WHERE job_id=? AND status='queued'", (job_id,)
        )
        return cursor.rowcount == 1


def delete_completed_job_record(job_id):
    """Remove only a successfully completed job-history row; preserve its document."""
    with transaction() as conn:
        cursor = conn.execute(
            "DELETE FROM ingestion_jobs WHERE job_id=? AND status='complete'",
            (job_id,),
        )
        return cursor.rowcount == 1


def completed_jobs():
    """Return completed jobs awaiting automatic history cleanup."""
    with connect() as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM ingestion_jobs WHERE status='complete' ORDER BY created_at"
        ).fetchall()]





def claim_document_deletion(document_id, exclude_job_id=None):
    """Atomically reserve a document for deletion when no worker owns it."""
    with transaction() as conn:
        row = conn.execute(
            "SELECT processing_status FROM documents WHERE document_id=?", (document_id,)
        ).fetchone()
        if row is None or row["processing_status"] in ("processing", "deleting"):
            return False
        sql = ("SELECT COUNT(*) FROM ingestion_jobs WHERE document_id=? "
               "AND status IN ('queued','processing','cancel_requested','deleting')")
        params = [document_id]
        if exclude_job_id:
            sql += " AND job_id!=?"
            params.append(exclude_job_id)
        active = int(conn.execute(sql, params).fetchone()[0])
        if active:
            return False
        cursor = conn.execute(
            "UPDATE documents SET processing_status='deleting',updated_at=? "
            "WHERE document_id=? AND processing_status!='processing'",
            (datetime.now(timezone.utc).isoformat(), document_id),
        )
        return cursor.rowcount == 1


def mark_document_deletion_failed(document_id, error):
    with transaction() as conn:
        conn.execute(
            "UPDATE documents SET processing_status='recovery_required',"
            "indexing_status='recovery_required',processing_error=?,updated_at=? "
            "WHERE document_id=?",
            (str(error), datetime.now(timezone.utc).isoformat(), document_id),
        )


def active_jobs_for_document(document_id, exclude_job_id=None):
    """Return active jobs referencing a document; used to serialize deletion."""
    with connect() as conn:
        sql = ("SELECT job_id,status FROM ingestion_jobs WHERE document_id=? "
               "AND status IN ('queued','processing','cancel_requested','deleting')")
        params = [document_id]
        if exclude_job_id:
            sql += " AND job_id!=?"
            params.append(exclude_job_id)
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]


def indexed_document_ids():
    try:
        with connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT document_id FROM chunk_search WHERE document_id IS NOT NULL"
            ).fetchall()
            return {str(row[0]) for row in rows}
    except sqlite3.OperationalError as exc:
        if "chunk_search" in str(exc) or "fts5" in str(exc).lower():
            return set()
        raise


def indexed_chunk_count_for_document(document_id):
    try:
        with connect() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM chunk_search WHERE document_id=?", (document_id,)
            ).fetchone()[0])
    except sqlite3.OperationalError as exc:
        if "chunk_search" in str(exc) or "fts5" in str(exc).lower():
            return 0
        raise


def count_other_jobs_for_document(document_id, exclude_job_id):
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM ingestion_jobs WHERE document_id=? AND job_id!=?",
            (document_id, exclude_job_id),
        ).fetchone()
        return int(row[0]) if row else 0


def list_jobs(owner_id="local"):
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM ingestion_jobs WHERE owner_id=? ORDER BY created_at DESC LIMIT 200", (owner_id,)).fetchall()]

def migrate_to_local_owner():
    """Convert this local install to one owner and remove obsolete auth tables."""
    with transaction() as conn:
        conn.execute("UPDATE documents SET owner_id='local' WHERE owner_id IS NULL OR owner_id!='local'")
        conn.execute("UPDATE ingestion_jobs SET owner_id='local' WHERE owner_id IS NULL OR owner_id!='local'")
        # The FTS index stores owner_id in its own table, so it must be migrated
        # alongside the Chroma metadata. Otherwise upgraded installs keep a
        # keyword index scoped to an owner that no longer exists, and exact-term
        # retrieval silently returns nothing.
        try:
            conn.execute("UPDATE chunk_search SET owner_id='local' WHERE owner_id IS NULL OR owner_id!='local'")
        except sqlite3.OperationalError as exc:
            if "chunk_search" not in str(exc) and "fts5" not in str(exc).lower():
                raise
        # Login was explicitly removed from this local-only build. Drop its old
        # credential/session tables; document, job, extraction, and vector data remain.
        conn.execute("DROP TABLE IF EXISTS sessions")
        conn.execute("DROP TABLE IF EXISTS users")

def recoverable_jobs():
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM ingestion_jobs WHERE status IN ('queued','processing') ORDER BY created_at").fetchall()]


def claim_job(job_id):
    """Atomically move one queued/failed job to processing.

    Returns True only for the caller that won the transition. This prevents two
    submissions of the same job (for example a double-clicked retry, or a
    startup recovery racing an upload) from extracting and indexing the same
    document concurrently.
    """
    with transaction() as conn:
        cursor = conn.execute(
            "UPDATE ingestion_jobs SET status='processing', stage='Starting' WHERE job_id=? AND status='queued'",
            (job_id,),
        )
        return cursor.rowcount == 1


def reset_processing_jobs():
    """Requeue jobs left in processing by a previous process.

    Called once when the worker pool is created. Without this, a crash during
    ingestion would leave a job stuck at 'processing' forever.
    """
    from datetime import datetime, timezone
    with transaction() as conn:
        stamp = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE ingestion_jobs SET status='queued', stage='Resuming after restart', updated_at=? "
            "WHERE status='processing'", (stamp,)
        )
        # A process restart proves the old worker no longer exists. Complete any
        # pending cancellation rather than leaving it permanently non-deletable.
        conn.execute(
            "UPDATE ingestion_jobs SET status='cancelled', stage='Cancelled after worker shutdown', updated_at=? "
            "WHERE status='cancel_requested'", (stamp,)
        )


def requeue_failed_job(job_id, max_pending=None, max_pending_bytes=None):
    """Atomically requeue a failed job while enforcing queue/storage admission."""
    from datetime import datetime, timezone
    with transaction() as conn:
        job = conn.execute(
            "SELECT size_bytes,document_id,status FROM ingestion_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if job is None or job["status"] != "failed":
            return False
        if job["document_id"]:
            doc = conn.execute(
                "SELECT processing_status FROM documents WHERE document_id=?", (job["document_id"],)
            ).fetchone()
            if doc and doc["processing_status"] == "deleting":
                return False
        row = conn.execute(
            "SELECT COUNT(*),COALESCE(SUM(size_bytes),0) FROM ingestion_jobs "
            "WHERE status IN ('queued','processing','cancel_requested')"
        ).fetchone()
        if max_pending is not None and int(row[0]) >= int(max_pending):
            raise RuntimeError("Ingestion queue is full; wait for pending jobs to finish.")
        if max_pending_bytes is not None and int(row[1]) + int(job["size_bytes"]) > int(max_pending_bytes):
            raise RuntimeError("Pending upload storage limit reached; wait for queued files to finish.")
        cursor = conn.execute(
            "UPDATE ingestion_jobs SET status='queued', stage='Queued for retry', progress=0, "
            "updated_at=?, error=NULL WHERE job_id=? AND status='failed'",
            (datetime.now(timezone.utc).isoformat(), job_id),
        )
        return cursor.rowcount == 1


def index_chunks(records):
    """Mirror searchable chunk text in SQLite FTS5 for exact-term retrieval."""
    if not records:
        return
    indexed_vocabulary.cache_clear()
    try:
        with transaction() as conn:
            for record in records:
                metadata = record.get("metadata", {})
                conn.execute("DELETE FROM chunk_search WHERE chunk_id=?", (record["id"],))
                conn.execute(
                    "INSERT INTO chunk_search(chunk_id,document_id,owner_id,filename,source_location,text) VALUES(?,?,?,?,?,?)",
                    (record["id"], metadata.get("document_id"), metadata.get("owner_id", "local"),
                     metadata.get("filename"), metadata.get("source_location"), record["text"]),
                )
    except sqlite3.OperationalError as exc:
        if "chunk_search" not in str(exc) and "fts5" not in str(exc).lower():
            raise


def remove_indexed_chunks(document_id):
    indexed_vocabulary.cache_clear()
    try:
        with transaction() as conn:
            conn.execute("DELETE FROM chunk_search WHERE document_id=?", (document_id,))
    except sqlite3.OperationalError as exc:
        if "chunk_search" not in str(exc) and "fts5" not in str(exc).lower():
            raise


def search_indexed_chunks(query, owner_id="local", limit=20):
    """Return exact keyword matches without loading all document text into RAM."""
    if not query:
        return []
    try:
        with connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id,document_id,filename,source_location,text,bm25(chunk_search) AS rank "
                "FROM chunk_search WHERE chunk_search MATCH ? AND owner_id=? ORDER BY rank LIMIT ?",
                (query, owner_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.OperationalError as exc:
        if "chunk_search" in str(exc) or "fts5" in str(exc).lower() or "syntax error" in str(exc).lower():
            return []
        raise


def indexed_chunk_count():
    try:
        with connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunk_search").fetchone()[0])
    except sqlite3.OperationalError as exc:
        if "chunk_search" in str(exc) or "fts5" in str(exc).lower():
            return 0
        raise


@lru_cache(maxsize=8)
def indexed_vocabulary(limit=20000):
    """Return a bounded vocabulary from the persistent FTS index for typo recovery."""
    try:
        with connect() as conn:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.chunk_vocab USING fts5vocab(main, chunk_search, 'row')")
            rows = conn.execute(
                "SELECT term FROM temp.chunk_vocab WHERE length(term) >= 3 ORDER BY doc DESC, term LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            return [str(row[0]) for row in rows]
    except sqlite3.OperationalError as exc:
        if "chunk_search" in str(exc) or "fts5" in str(exc).lower():
            return []
        raise


def indexed_lexical_statistics(terms):
    """Return (indexed_chunk_count, {term: number-of-chunks-containing-it}).

    Both values come from one connection so a search does not pay two separate
    index round-trips. The vocabulary lookup is bounded to the supplied query
    terms, so it never scans the whole vocabulary. Used to weight rare query
    terms more heavily than ubiquitous ones. Returns (0, {}) when the FTS index
    or its vocabulary is unavailable, which lets the caller fall back to
    uniform weighting.
    """
    wanted = sorted({str(term) for term in terms if term})
    try:
        with connect() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM chunk_search").fetchone()[0])
            if not wanted:
                return total, {}
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.chunk_vocab USING fts5vocab(main, chunk_search, 'row')")
            placeholders = ",".join("?" for _ in wanted)
            rows = conn.execute(
                f"SELECT term, doc FROM temp.chunk_vocab WHERE term IN ({placeholders})",
                tuple(wanted),
            ).fetchall()
            return total, {str(row[0]): int(row[1]) for row in rows}
    except sqlite3.OperationalError as exc:
        if "chunk_search" in str(exc) or "fts5" in str(exc).lower():
            return 0, {}
        raise
