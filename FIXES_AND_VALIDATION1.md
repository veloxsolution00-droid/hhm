# Fixes and Validation Notes

This package contains a repair pass based on the supplied repository audit. It is not a production certification.

## Changes made

- Restored multi-format extraction for PDF, DOCX, PPTX, XLSX/XLSM/ODS, CSV, text/Markdown/logs, HTML, XML, JSON, YAML, RTF, ODT, EML, and images with optional local Tesseract OCR.
- Restored `extract()` / `iter_extract()` compatibility and table-aware extraction helpers.
- Connected the background worker to the actual ingestion pipeline.
- Added durable original-file and extracted-text writes, SHA-256 duplicate detection, document classification, chunking, embedding/index calls, and document/job linkage.
- Added cleanup of partial vector/FTS writes on ingestion failure while retaining the original and failed metadata record for diagnosis.
- Replaced the document service's incompatible row adapter with the canonical SQLite document API.
- Corrected document deletion to delete vectors through ChromaDB and FTS rows through SQLite before removing metadata and persisted artifacts.
- Corrected reconciliation to preserve missing-original registry records and report recovery-required state.
- Implemented bounded process-local cache APIs with corpus/config scoping, TTL, entry caps, semantic retrieval reuse, model-scoped embedding cache, and successful-cited-answer-only caching. Cache is intentionally disposable and not persistent across backend restarts.
- Removed question-mark-based early abstention for ordinary user questions.
- Applied question-type retrieval limits, retaining top-1 for factual lookups.
- Changed answer verification failure to withhold the answer rather than returning it as a successful unverified response.
- Made the combined launcher bind Streamlit to loopback by default. Network exposure now requires explicit `--share-network` opt-in and prints a warning.
- Raised the default individual upload limit to 200 MB.
- Standardized SQLite context-managed connections to close handles after use.
- Renamed ad-hoc root-level scripts so pytest does not mistake them for test modules; removed an obsolete broken script that rewrote application source files.

## Validation performed

- `pytest -q tests`: **99 passed, 1 skipped** in the provided execution environment.
- Python AST parsing: run on the packaged Python source files.
- A new mocked ingestion regression test verifies permanent original storage, extracted-text persistence, document indexing metadata, chunk records, and ingestion-job linkage without requiring a running Ollama service.

## Not validated in this environment

- Live ChromaDB integration and persistent vector writes (the execution environment did not have the `chromadb` package installed).
- Live Ollama model calls, embeddings, and grounded answer quality (Ollama was not running).
- Full API/UI startup and browser workflows.
- OCR against a real Tesseract installation.
- Target Windows machine performance, 200 MB memory usage, and long-running ingestion.
- Network-disabled operation and external network traffic inspection.
- Recovery/deletion against a real populated ChromaDB collection.

Install dependencies from `requirements.txt`, confirm the configured Ollama models are present, and run the application on the target Windows machine before relying on it. Semantic verification remains probabilistic; no LLM-based checker can guarantee factual entailment.


## Ingestion job lifecycle (v3)

- `Stop` immediately cancels queued jobs; for active workers it changes the state to `cancel_requested`.
- Active workers check cancellation at extraction, chunking, and embedding batch checkpoints. A model request already in progress is allowed to return; deletion remains disabled until the worker acknowledges stop.
- `cancelled` means the worker has exited the ingestion call and it is safe to delete.
- `Delete ingestion` is offered only for terminal states (`cancelled`, `failed`, `complete`, or `delete_failed`).
- Deleting a completed job removes its associated indexed document, ChromaDB vectors, FTS rows, original file, and extracted artifact, unless another ingestion job still references the same deduplicated document. The last referencing job performs the deletion.
- Deletion uses a durable `deleting` state and is retried at application startup if the process stops mid-cleanup. Cleanup failures are shown as `delete_failed` and can be retried.
- The API supports both `POST` and `DELETE` for `/jobs/{job_id}/delete`; a running worker returns HTTP 409, an unknown job returns 404, and a cleanup failure returns 500.
