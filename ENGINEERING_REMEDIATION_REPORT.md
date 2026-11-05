# RND2 Engineering Remediation & Verification Report

**Scope:** Supplied RND2 source archive  
**Implementation mode:** Changes applied to a separate working copy; original upload preserved.

## Summary

The supplied archive already contained implementations for most of the issues listed in the earlier audit. Direct source inspection confirmed that the current code includes recovery count checks, deletion/ingestion ownership checks, sentence-level citation coverage, fail-closed verification, streamed CSV/XLSX extraction, partial batch handling, health model checks, orphan-vector detection, upload fsync, queue admission limits, archive expansion limits, and explicit OCR failure handling.

The earlier audit summary overstated several of these as outstanding. They were not rewritten unnecessarily.

## Change made

- Repaired the malformed final lines in `.env.example`. The archive contained literal `\\n` characters appended to `MAX_DATA_STORAGE_MB` and `MAX_ARCHIVE_ENTRIES`, which made the sample configuration invalid for a clean setup.
- Validated that the corrected sample configuration loads successfully and resolves `MAX_DATA_STORAGE_MB=20480`, `MAX_ARCHIVE_ENTRIES=20000`, and `MAX_ARCHIVE_UNCOMPRESSED_MB=1024`.
- Kept the user's machine-specific `.env`, live runtime databases, uploaded documents, logs, Python bytecode, and pytest cache out of the distributable source ZIP. `.env.example` and empty data-directory placeholders are retained.

## Verified implementation areas

| Area | Source verification |
|---|---|
| Partial-index recovery | Reconciliation compares actual vector and FTS counts with durable expected chunk count; partial vectors are removed and document marked failed |
| Delete/ingestion race | Document deletion claims a deletion state and rejects active ingestion ownership |
| Citation coverage | Citation validator rejects uncited factual sentences |
| Verifier failure | Enabled verifier exceptions withhold the draft rather than returning it as successful |
| Spreadsheet extraction | XLSX uses openpyxl read-only iteration; CSV reads line-by-line for file-backed uploads; blank header columns and first source row are retained |
| Batch upload | All files are staged before jobs are created; queued records are rolled back on job-creation failure |
| Deletion errors | API distinguishes missing documents, busy documents, and cleanup failures |
| Health checks | Health state includes Ollama availability and configured generation/embedding model availability |
| Orphan detection | Reconciliation detects vector and FTS document IDs without registry records |
| Queue limits | Queue count and pending byte admission limits are enforced in SQLite transactions |
| Upload durability | Staged and permanent files are flushed and fsynced |
| Archive safety | Office archive entry count, expanded size, and compression ratio are checked |
| OCR errors | OCR failures are raised with page context; they are not silently treated as complete extraction |
| Offline launch | Supplied launcher commands disable Streamlit usage statistics and bind to loopback |

## Test results

Executed in the available environment:

- `python -m pytest -q`: **114 passed, 1 skipped**
- `python -m compileall -q app run.py evaluation`: passed
- AST parsing: **54 Python files** passed
- `.env.example` load check: passed

The skipped test is the ChromaDB persistence integration test. ChromaDB, Streamlit, and `striprtf` are not installed in this audit runtime. Therefore, full application startup, live ChromaDB persistence, Streamlit UI startup, RTF extraction, Ollama inference, real Windows execution, and network-disconnected runtime verification were not performed.

Static security scanners (Bandit, pip-audit) and type/lint tools (Ruff, mypy) were unavailable and were not represented as having passed.

## Remaining verification risks

1. Install the declared requirements in the target Windows environment and run the full application there.
2. Verify Ollama has the configured generation and embedding models installed.
3. Run end-to-end ingestion, retrieval, citation, deletion, retry, and recovery tests against a real local ChromaDB instance.
4. Run security and dependency scans in an environment with the relevant tools.
5. Benchmark large PDF, XLSX, CSV, OCR, and multi-file workloads on the target machine.

## Integrity and packaging

- The uploaded original ZIP was not modified.
- Runtime user data and `.env` were excluded from the distributable ZIP to avoid packaging private documents, databases, logs, and machine-specific configuration.
- Copy `.env.example` to `.env` and adjust local model/data settings before running.
- No claim is made that the project is bug-free or fully production-secure.

---

# Balanced TOP_K Configuration

## Active defaults in this build

- Base `TOP_K`: **4**
- `RETRIEVAL_CANDIDATE_MULTIPLIER`: **5**
- Dense ChromaDB candidate pool: **20** (`4 × 5`, bounded by collection size)
- `LEXICAL_CANDIDATE_LIMIT`: **40** (unchanged)
- `HYBRID_LEXICAL_WEIGHT`: **0.35** (unchanged)

## Question-type evidence limits

| Question type | Final evidence chunk limit |
|---|---:|
| Factual | 4 |
| Explanation | 6 |
| Calculation | 5 |
| Summary | 7 |
| Comparison | 7 |

Complex question types can use more than the base four chunks. The maximum remains bounded by the existing application cap of 10.

## Important deployment step

The project archive intentionally does not include a machine-specific `.env`. If you already have an `.env` file, edit its existing setting:

```env
TOP_K=4
```

An existing `.env` value takes precedence over the Python default and `.env.example`. Restart the FastAPI backend after changing it.

## Expected performance effect

The dense candidate request decreases from 40 to 20 with the default multiplier, which reduces the amount of dense candidate data passed into ranking. Factual answers also pass at most four selected chunks to the LLM. Explanations, calculations, summaries, and comparisons retain larger evidence budgets.

This does **not** guarantee a 50% reduction in end-to-end response time: query embedding, keyword retrieval, Ollama generation, verification, and disk/database overhead remain. A real latency/quality benchmark requires the target machine, its Ollama models, and the actual indexed corpus.

## Validation performed

- Regression test asserts the five question-type limits above.
- Regression test asserts the existing hard maximum remains enforced.
- Full test suite: **116 passed, 1 skipped**.
- Python compile check: passed.
