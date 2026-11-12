# Production Readiness: Offline Document Intelligence

## Product boundary

This deployment is a **single-user, local-computer application**. FastAPI, Streamlit, and Ollama must bind to loopback interfaces. No cloud LLM, hosted embedding, remote OCR, telemetry service, or internet API is part of the runtime data path. Authentication is intentionally out of scope for this local-only build; do not expose the services to a LAN or public network.

Development/setup may require downloading Python packages, Ollama binaries, model weights, and OCR binaries. For an air-gapped deployment, provision and verify those assets before disconnecting the machine. Runtime must not download missing models or packages.

## Existing architecture (retain and harden)

- FastAPI API and Streamlit local UI.
- Ollama for local generation and embeddings.
- Persistent ChromaDB collection `document_chunks`.
- SQLite document/job metadata and FTS5 lexical index.
- Durable original uploads, extracted artifacts, metadata, jobs, and logs under configurable `DATA_PATH`.
- Format-aware extraction for PDF, Office/OpenDocument, text, structured data, email, and raster images; local Tesseract OCR for image-based pages.
- Hybrid semantic + lexical retrieval, deterministic reranking/evidence selection, citation validation, and local-model verification.
- Startup reconciliation, bounded ingestion queue, explicit document deletion, and cache invalidation.

Do not replace persistent indexes, re-embed the corpus, clear data directories, or migrate schemas without a tested migration and a recovery path.

## Engineering workstreams

### 1. Runtime reliability and observability
- Bound inference queue wait independently from Ollama HTTP timeout.
- Log queue wait and request duration without logging prompts, document contents, or secrets.
- Validate embedding response count, dimensions, and finite numeric values before ChromaDB writes.
- Report each failure stage separately: extraction, embedding, indexing, retrieval, generation, citation integrity, semantic verification, and persistence.
- Preserve useful source passages and diagnostics when a later stage fails.

### 2. Ingestion and document fidelity
- Keep extraction streaming for large spreadsheets and file-backed uploads.
- Preserve page/slide/sheet/row provenance through extraction, chunking, retrieval, and citation display.
- Treat OCR errors and unreadable pages as incomplete extraction, never as successful empty text.
- Add fixtures for scanned PDFs, multi-column PDFs, tables, DOCX, XLSX, CSV, PPTX, EML, HTML, and images.
- Add file hash deduplication, archive expansion limits, upload limits, and interrupted-job recovery tests.

### 3. Retrieval and answer correctness
- Measure candidate recall before changing retrieval algorithms.
- Keep dense and FTS lexical retrieval as distinct retrieval signals and fuse by stable chunk identity.
- Evaluate query expansion and spelling correction against the original query; never silently replace exact identifiers, names, dates, or numeric tokens.
- Keep answer generation evidence-bounded.
- Validate citation references and claim-level source mappings deterministically before optional semantic verification.
- Do not weaken abstention because a model verifier rejects an answer.
- Do not treat a model's self-verification as mathematical proof.

### 4. Performance
- Measure ingestion, embedding, retrieval, reranking, prompt construction, generation, and verification separately.
- Report p50/p95 latency and throughput on the actual target CPU/RAM profile.
- Bound prompt tokens and evidence count; do not assume larger TOP_K improves correctness.
- Benchmark cold start, warm model, cache hit, cache miss, concurrent ingestion, and concurrent chat.
- Prefer fewer model calls and deterministic local checks where they preserve the same evidence guarantees.

### 5. Persistence, recovery, and local security
- Test restart persistence and backup/restore of the complete data directory.
- Test interrupted ingestion, partial Chroma writes, SQLite/FTS reconciliation, deletion, and cache invalidation.
- Keep uploads private to the local machine and never include user documents, `.env`, runtime databases, or logs in source archives.
- Validate loopback binding for every service and disable optional usage statistics.
- Treat uploaded documents as untrusted data; document text must never override system instructions or trigger tools.

## Release gates (must pass before calling this production-ready)

1. Unit/regression suite passes on Python 3.11 on Windows and Linux.
2. Live integration suite passes with the supported local Ollama version, both configured models, and a real ChromaDB persistence directory.
3. Air-gap test: after all assets are provisioned, disconnect Wi-Fi/Ethernet and cold-start, ingest, query, restart, and query again successfully. Capture outbound connection attempts; do not infer offline compliance merely from loopback URLs.
4. Persistence test: original files, extracted records, SQLite metadata/FTS, Chroma vectors, and document counts reconcile after restart and forced interruption.
5. Retrieval benchmark reports Recall@K, Precision@K, and MRR on a labelled corpus.
6. Answer benchmark reports human-adjudicated factual correctness, citation precision/coverage, unsupported-claim rate, and abstention precision/recall. Heuristic confidence must be labelled uncalibrated until calibration is measured.
7. Performance benchmark reports p50/p95 response latency, per-stage timings, ingestion throughput, peak RAM, and model memory on the target machine.
8. Security review covers upload validation, archive bombs, prompt injection in documents, local API exposure, path traversal, data deletion, and backup handling.
9. Packaging test installs from a clean machine using only the prepared offline wheel/model/OCR bundle.

## Current implementation status

This document is an engineering target and release checklist, not a claim that every gate has passed. Current unit tests and static checks do not replace live Ollama, real-corpus, Windows, backup/restore, or disconnected-network tests. Each completed workstream must link its test evidence and measured results in the pull request.
