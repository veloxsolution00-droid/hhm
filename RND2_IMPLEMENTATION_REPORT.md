# RND2 Quality Upgrade — Implementation Report

## Scope and safety

Implementation was performed in a separate working copy. The supplied `RND2_TOPK4_balanced.zip` was not modified. No project source was changed in the original archive. The upgrade package omits the empty snapshot SQLite database and empty application log so they cannot overwrite live deployment state; `.gitkeep` placeholders remain.

The project remains local-first:
- Ollama endpoint validation still requires loopback HTTP.
- ChromaDB remains the persistent vector store.
- SQLite FTS5 remains the exact-term index.
- No cloud LLM, hosted embedding, hosted reranker, or telemetry dependency was added.
- Existing embedding model and collection name were not changed.
- Existing document IDs, metadata keys, owner filters, deletion and recovery pathways were retained.

## Implemented changes

### Retrieval and ranking
- Added explicit route/candidate diagnostics to the ranking pipeline while retaining the existing `rank_candidates()` API.
- Improved deterministic reranking with query-term coverage, semantic relevance, hybrid score, fusion score, exact-number cues, answer-bearing cues, and boilerplate down-weighting.
- Improved evidence selection using relevance, incremental query-term coverage, and text redundancy suppression.
- Added selected-character and selection-strategy diagnostics.
- Kept dense and keyword ranking as views over the current shared candidate pool. Diagnostics explicitly report `candidate_pool_shared=true`; they do not claim independent retrieval pools.

### Query expansion and performance
- Query expansion and vocabulary typo recovery now run conditionally when the initial owner-scoped result set is too small or weak.
- Original spelling is still searched when normalization changes the query.
- FTS vocabulary lookup is cached in-process and invalidated when indexed chunks are added or removed.
- Added timing fields for retrieval, ranking, evidence selection, answer generation, retry generation, and verification when those stages execute.

### Context and answer safety
- Added configurable `MODEL_CONTEXT_TOKENS`, `CONTEXT_SAFETY_TOKENS`, and `CONTEXT_CHARS_PER_TOKEN`.
- Ollama generation now reads the configured context window instead of a hard-coded value.
- Evidence character allocation is capped to a conservative estimate of the available context after reserving space for instructions and output.
- Existing evidence-only prompting, citation validation, answer recovery, deterministic calculation validation, and fail-closed verification were retained.
- Cache fingerprints were versioned so old cached retrieval/answer results are not reused under the changed ranking/context behavior.

### Multi-question diagnostics
- Multi-question responses now aggregate candidate counts, selected counts, retrieval passes, best similarity, route counts and stage timings.
- Per-question diagnostics are retained in the aggregate response.
- This addresses the UI mismatch where the multi-question result previously exposed only question count and indexed document count while the UI looked for chunk metrics.

### Evaluation support
- Added `scripts/evaluate_retrieval.py` for offline Recall@K, Precision@K and MRR against a user-labelled JSONL evaluation set.
- Added focused regression tests for ranking diagnostics, evidence coverage, multi-question diagnostic aggregation and context budgeting.

## Files changed

- `.env.example`
- `app/core/config.py`
- `app/rag/chat.py`
- `app/retrieval/evidence_selector.py`
- `app/retrieval/ollama.py`
- `app/retrieval/pipeline.py`
- `app/retrieval/reranker.py`
- `app/storage/database.py`
- `tests/test_rnd2_quality_upgrade.py`

New:
- `scripts/evaluate_retrieval.py`

## Test results

Baseline from the original project copy:
- 116 passed
- 1 skipped
- 0 failed

After implementation:
- 120 passed
- 1 skipped
- 0 failed

The skipped test is the ChromaDB persistence integration test because the audit runtime did not have the `chromadb` package installed. The test was not counted as a pass.

All 77 existing Python source files in the original snapshot compiled without syntax errors. The modified workspace also passed the complete pytest suite. No live Ollama or ChromaDB performance benchmark was possible in this environment.

## Remaining validation required with the real local corpus

The following must be measured on the deployment machine before claiming quality or speed gains:
1. Install the project's pinned/runtime requirements in the target offline environment.
2. Run the ChromaDB persistence integration test.
3. Use a labelled question set with gold relevant chunk IDs and run `scripts/evaluate_retrieval.py`.
4. Measure 100–200 PDF ingestion time, storage growth, query latency and memory usage on the target machine.
5. Compare answer correctness, citation support, abstention quality and response latency against the saved baseline.
6. Inspect any regression in multi-document comparisons and long financial tables.

The initial plan's numerical targets (Precision@4 >= 85%, Recall@10 >= 90%, citation accuracy >= 95%) are acceptance goals, not measured outcomes.

## Run the offline retrieval evaluator

Create a UTF-8 JSONL file with one record per labelled question:

```json
{"question":"What was the reported revenue for FY 2025?","relevant_chunk_ids":["<document-id>:<chunk-index>"]}
```

Then run from the project root:

```bash
python scripts/evaluate_retrieval.py evaluation.jsonl local 10
```

The script uses the existing local ChromaDB and SQLite FTS indexes. It does not contact external services.

## Deployment notes

- Keep the existing `.env` values unless deliberately changing a setting.
- `MODEL_CONTEXT_TOKENS` defaults to 4096 to match the existing Ollama generation context.
- Existing ChromaDB data can be retained; no re-embedding is required by these code changes.
- Do not delete `data/chroma_db`, `data/metadata`, `data/uploads`, or `data/extracted` during deployment.
- Take a backup of the deployment's data directory before replacing application code.
- Do not overwrite the deployment's existing `data/` directory or `.env` file when extracting the ZIP. The package intentionally omits runtime SQLite/log files; preserve all existing uploads, extracted text, vectors, metadata and logs.
