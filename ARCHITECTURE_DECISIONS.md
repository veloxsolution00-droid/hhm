# Architecture Decisions — Offline Document Intelligence

## Architecture objective

Keep the runtime fully local and single-user while making ingestion, retrieval, answer generation, verification, persistence, and UI independently testable. The application must remain usable without network access after its Python packages, Ollama binaries/models, and optional OCR runtime have been provisioned.

This document records architectural boundaries and release invariants. It is not a claim that every planned refactor or production gate is complete.

## Current runtime boundaries

| Boundary | Current modules | Responsibility |
|---|---|---|
| Local API and lifecycle | `app/main.py`, `run.py` | Loopback-only API, lifespan, durable job dispatch, health and recovery endpoints |
| Configuration | `app/core/config.py` | Validated local paths, model endpoints, resource limits and retrieval settings |
| Ingestion | `app/ingestion/*` | Format detection, extraction, OCR, chunk preparation and provenance |
| Persistence | `app/storage/*`, `app/retrieval/store.py` | SQLite metadata/FTS5 and persistent Chroma vectors |
| Model gateway | `app/retrieval/ollama.py` | Bounded local inference, embedding validation, generation and model diagnostics |
| Retrieval | `app/retrieval/*` | Dense and lexical candidate retrieval, fusion, reranking and evidence selection |
| RAG orchestration | `app/rag/chat.py`, `app/rag/*` | Query planning, evidence-bounded prompts, citations, recovery and answer assembly |
| Reasoning | `app/reasoning/*` | Explicit calculation, conflict hints and answer coverage |
| Cache | `app/cache/*` | Corpus/config-scoped query, retrieval and answer reuse |
| UI | `app/ui/*` | Local Streamlit presentation only; no business logic or persistence ownership |

## Architectural decisions and invariants

### 1. The model gateway is the only Ollama boundary

Application code must not make ad-hoc HTTP calls to Ollama. All inference passes through `app/retrieval/ollama.py`, which owns timeouts, bounded concurrency, response validation, model names and safe diagnostics.

Embedding vectors must be validated for batch count, non-empty shape, consistent dimensions, numeric values and finite values before any persistent write.

### 2. Retrieval routes must fail independently

Dense retrieval and FTS5 are separate candidate-generation routes. Failure of the local embedding model must not disable an already-built FTS index. FTS results remain owner-scoped and bounded.

Hybrid score semantics must be consistent:
- Dense candidates: `(1 - lexical_weight) * semantic_score + lexical_weight * lexical_score`.
- FTS-only candidates: `lexical_weight * lexical_score`.

Do not describe multiple rankings of the same candidate list as independent recall routes. Measure candidate recall before changing the fusion strategy.

### 3. Citation syntax and evidence support are separate gates

A source label or filename is not a stable claim-to-passage reference. The verifier consumes numbered evidence blocks, so every factual sentence must carry one or more valid numeric passage references.

Citation validation checks reference integrity and per-sentence coverage. Semantic verification checks whether each claim is directly supported by only the passages cited for that claim. Neither check alone proves correctness:
- Lexical overlap is not entailment.
- A model verifier is not independent ground truth.
- Confidence values remain explicitly uncalibrated until evaluated against human labels.

A partially cited answer must not be returned as a successful fully grounded answer. If a safe extractive answer cannot be produced, fail closed and show the retrieved source passages.

### 4. Verification is a separate application service

Claim-level verification now lives in `app/rag/verification.py`. It receives the generation callable as a dependency, allowing deterministic unit tests without Ollama. `app/rag/chat.py` retains a compatibility adapter while orchestration is progressively decomposed.

### 5. Persistence is authoritative and non-destructive

- Original uploads, extracted artifacts, SQLite metadata/FTS and Chroma vectors are durable source data.
- Normal startup must never clear, recreate or re-embed a collection.
- Deleting a document must remove its indexed chunks and invalidate its answer/retrieval caches.
- Deleting an ingestion-history record must preserve the indexed document and its caches.
- Schema or embedding-model changes require an explicit migration, backup, validation and rollback plan.

### 6. Uploaded content is untrusted input

Document text is evidence, never instructions. It must not override system prompts, authorize actions, or change application configuration. Local APIs remain loopback-bound; no authentication is provided because this product is intentionally single-user and local-only. Do not expose it to a LAN or public network.

### 7. Latency is measured by stage

Record durations for queue wait, embedding, dense search, FTS search, fusion/reranking, evidence selection, prompt construction, generation, verification and persistence. Report p50/p95, not just average response time. Do not remove verification or increase concurrency simply to improve a latency number.

## Refactor roadmap

The following work is intentionally incremental so persistent user data and existing behavior are not put at risk:

1. **Complete the service boundaries:** extract retrieval orchestration, answer candidate selection/recovery, and answer assembly from the large `chat.py` module. Preserve public API response shapes during each extraction.
2. **Introduce typed internal contracts:** define stable structures for `RetrievedChunk`, `EvidenceBundle`, `CitationValidation`, `VerificationResult`, and `AnswerResult`; validate external/model payloads at boundaries.
3. **Make retrieval diagnostics route-specific:** expose dense/FTS candidate counts, overlap, unique recall, fusion ranks and selection reasons without claiming that correlated rank views are independent routes.
4. **Add durable integration tests:** use temporary SQLite/Chroma paths for restart, interruption, deletion, reconciliation, FTS backfill and cache invalidation. Keep tests away from the user's live data directory.
5. **Add a labelled evaluation harness:** track Recall@K, MRR, evidence coverage, factual claim precision, citation precision/coverage, unsupported-claim rate, abstention precision/recall and p50/p95 stage latency.
6. **Add target-environment release checks:** Windows, configured Ollama models, actual corpus, constrained RAM/CPU, restart/persistence, and disconnected-network execution.

## Changes in this engineering branch

- FTS5 retrieval can continue when Ollama embedding is unavailable.
- FTS-only scores now use the configured lexical weight rather than the dense weight.
- Citation validation now requires numeric evidence references compatible with the claim verifier; filename labels alone cannot satisfy citation presence or coverage.
- Incomplete citation coverage no longer advances to semantic verification as if it were a valid answer.
- Claim verification was extracted from the chat orchestrator into an injectable, independently testable module.
- Regression tests cover lexical fallback, hybrid score weighting, filename-only citations, per-sentence numeric citation coverage, and verifier isolation.

## Release status

These changes are source-level engineering work. The branch is not certified production-ready until the automated suite passes and the release gates in `PRODUCTION_OFFLINE_READINESS.md` have been run on the target Windows/Ollama environment with a labelled real corpus and an offline cold-start test.
