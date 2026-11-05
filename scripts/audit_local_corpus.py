"""Read-only audit of RND2's local document, extraction and index inventory.

Run from the project root:
    python scripts/audit_local_corpus.py
Optional:
    python scripts/audit_local_corpus.py --output evaluation/local_corpus_audit.json

This script never changes, repairs, deletes, or rebuilds any database or file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readonly_connection(path: Path):
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def quick_check(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with readonly_connection(path) as conn:
            return str(conn.execute("PRAGMA quick_check").fetchone()[0])
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def main() -> dict:
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app.core.config import settings

    uploads = Path(settings.uploads_path)
    extracted = Path(settings.extracted_path)
    sqlite_path = Path(settings.sqlite_path)
    chroma_sqlite = Path(settings.chroma_path) / "chroma.sqlite3"

    docs = []
    with readonly_connection(sqlite_path) as conn:
        rows = conn.execute(
            "SELECT document_id, filename, stored_filename, content_hash, file_format, "
            "processing_status, extraction_status, indexing_status, chunk_count, "
            "source_metadata_json, extracted_path FROM documents ORDER BY filename"
        ).fetchall()
        fts_count = conn.execute("SELECT count(*) FROM chunk_search").fetchone()[0]
        doc_table_quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]

    registered_hashes = set()
    registered_files = set()
    registered_ids = set()
    metadata_chunk_total = 0

    for row in rows:
        (document_id, filename, stored_filename, content_hash, file_format,
         processing_status, extraction_status, indexing_status, chunk_count,
         source_metadata_json, extracted_path) = row
        registered_ids.add(document_id)
        registered_hashes.add(content_hash)
        registered_files.add(stored_filename)
        metadata_chunk_total += int(chunk_count or 0)

        original = uploads / stored_filename
        original_exists = original.exists()
        actual_hash = sha256(original) if original_exists else None
        extracted_file = Path(extracted_path) if extracted_path else extracted / f"{document_id}.jsonl"
        # A copied project may retain an absolute path from another Windows
        # machine. Prefer that path only when it exists; otherwise use the
        # current project's ID-based extracted-text location.
        if not extracted_file.exists():
            extracted_file = extracted / f"{document_id}.jsonl"

        records = 0
        extracted_chars = 0
        locations = set()
        malformed_lines = 0
        if extracted_file.exists():
            with extracted_file.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        records += 1
                        value = str(item.get("text") or "")
                        extracted_chars += len(value)
                        if item.get("source_location"):
                            locations.add(str(item["source_location"]))
                    except Exception:
                        malformed_lines += 1

        pdf_pages = None
        if original_exists and original.suffix.lower() == ".pdf":
            try:
                import fitz
                with fitz.open(original) as pdf:
                    pdf_pages = len(pdf)
            except Exception:
                pdf_pages = "unavailable"

        try:
            source_meta = json.loads(source_metadata_json or "{}")
        except Exception:
            source_meta = {}

        docs.append({
            "document_id": document_id,
            "filename": filename,
            "file_format": file_format,
            "document_type": source_meta.get("document_type"),
            "processing_status": processing_status,
            "extraction_status": extraction_status,
            "indexing_status": indexing_status,
            "metadata_chunk_count": int(chunk_count or 0),
            "original_file_exists": original_exists,
            "original_hash_matches_metadata": (actual_hash == content_hash) if actual_hash else False,
            "pdf_page_count": pdf_pages,
            "extracted_jsonl_exists": extracted_file.exists(),
            "extracted_record_count": records,
            "extracted_character_count": extracted_chars,
            "distinct_source_locations": len(locations),
            "malformed_jsonl_lines": malformed_lines,
        })

    uploads_inventory = []
    upload_hashes: dict[str, list[str]] = {}
    for path in sorted(uploads.iterdir()) if uploads.exists() else []:
        if not path.is_file() or path.name == ".gitkeep":
            continue
        digest = sha256(path)
        upload_hashes.setdefault(digest, []).append(path.name)
        uploads_inventory.append({
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "registered_by_filename": path.name in registered_files,
            "registered_by_content_hash": digest in registered_hashes,
            "duplicate_content_names": [],
        })
    for item in uploads_inventory:
        digest = sha256(uploads / item["filename"])
        item["duplicate_content_names"] = upload_hashes.get(digest, [])

    extracted_orphans = []
    for path in sorted(extracted.glob("*.jsonl")) if extracted.exists() else []:
        if path.stem not in registered_ids:
            extracted_orphans.append(path.name)

    chroma_info = {
        "database_exists": chroma_sqlite.exists(),
        "quick_check": quick_check(chroma_sqlite),
        "embedding_rows": None,
        "collections": [],
    }
    if chroma_sqlite.exists():
        try:
            with readonly_connection(chroma_sqlite) as conn:
                chroma_info["embedding_rows"] = conn.execute(
                    "SELECT count(*) FROM embeddings"
                ).fetchone()[0]
                chroma_info["collections"] = [
                    {"name": row[0], "collection_id": row[1]}
                    for row in conn.execute("SELECT name, id FROM collections")
                ]
        except Exception as exc:
            chroma_info["read_error"] = f"{type(exc).__name__}: {exc}"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_mode": "read_only",
        "project_root": ".",
        "metadata_database": {
            "exists": sqlite_path.exists(),
            "quick_check": doc_table_quick_check,
            "registered_document_count": len(docs),
            "metadata_chunk_total": metadata_chunk_total,
            "sqlite_fts_chunk_count": int(fts_count),
        },
        "chroma": chroma_info,
        "documents": docs,
        "uploads": uploads_inventory,
        "unregistered_uploads": [
            item["filename"] for item in uploads_inventory
            if not item["registered_by_filename"]
        ],
        "duplicate_upload_groups": [
            names for names in upload_hashes.values() if len(names) > 1
        ],
        "orphan_extracted_jsonl": extracted_orphans,
        "integrity_comparison": {
            "metadata_chunks_equal_fts_rows": metadata_chunk_total == int(fts_count),
            "metadata_chunks_equal_chroma_embeddings": (
                metadata_chunk_total == chroma_info["embedding_rows"]
                if chroma_info["embedding_rows"] is not None else None
            ),
        },
        "safety_note": "Read-only audit; no database or document was modified.",
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/local_corpus_audit.json")
    args = parser.parse_args()
    result = main()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "registered_documents": result["metadata_database"]["registered_document_count"],
        "metadata_chunks": result["metadata_database"]["metadata_chunk_total"],
        "fts_chunks": result["metadata_database"]["sqlite_fts_chunk_count"],
        "chroma_embeddings": result["chroma"]["embedding_rows"],
        "unregistered_uploads": result["unregistered_uploads"],
        "duplicate_upload_groups": result["duplicate_upload_groups"],
        "integrity_comparison": result["integrity_comparison"],
    }, indent=2))
