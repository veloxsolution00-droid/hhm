# RND2 — Document-Agnostic RAG Reliability Audit & Improvement Report

**Audit date:** 5 November 2026  
**Scope:** Uploaded `RND2_1.zip` snapshot and its existing source code, extracted text, SQLite metadata/FTS database, and persisted ChromaDB SQLite database.  
**Implementation approach:** Minimal targeted changes in a separate working copy. No architecture replacement, model switch, dependency addition, database rebuild, or document deletion.

## 1. Executive summary

The project already contains a substantial local RAG pipeline. The most useful improvements were not a wholesale rewrite: they were corrections to answer prompting, refusal recovery, evidence selection, citation verification, ranking, cache test isolation, and the real-corpus evaluation tools.

The updated test suite reports **168 passed, 1 skipped**. The uploaded snapshot's initial test run reported **159 passed, 6 failed, 1 skipped**. The six initial failures comprised:
- Three retrieval/citation tests contaminated by the process-local cache shared across tests.
- One real ranking defect: InfoBooks download-store text was still penalized when the user directly asked about InfoBooks.
- Two recovery-test expectations/diagnostic issues: a verifier-rejected retry was reported as selected before verification passed, and the old test expected no controlled retry after a verifier rejection.

The live model and retrieval quality were **not** certified in this environment: the Ollama service was not running and the `chromadb` Python package was unavailable. The persisted ChromaDB SQLite database was inspected read-only and passed its SQLite integrity check. The 19-case real-corpus API evaluation has been made portable, but it still needs to be run on the target Windows machine with Ollama and the backend running.

## 2. Existing configuration and architecture

The supplied `.env` resolves to:
- Generation model: `qwen2.5:3b`
- Embedding model: `nomic-embed-text`
- Model context: 4096 tokens
- `TOP_K=4`, minimum similarity `0.25`
- Hybrid lexical weight `0.35`
- SQLite FTS candidate limit `40`
- Answer verification enabled
- OCR enabled
- Character chunk size `1200`, overlap `180`
- Inference concurrency `1`

The existing local stack remains:
- Streamlit UI and FastAPI backend
- Ollama for local embeddings and generation
- One ChromaDB collection named `document_chunks`
- SQLite metadata and FTS5 lexical index
- Page/section-labelled extraction, character-based chunking, deterministic reranking, evidence selection, citation validation, and local-model verification

ChromaDB stores chunks in one collection, while each chunk carries document and owner metadata. Retrieval filters by `owner_id` and the indexed `document_id` set; it does not create a separate physical collection per document. Dense Chroma retrieval and FTS5 lexical retrieval are both performed by the store, then combined into a shared candidate set. The later dense/keyword ranking adapters are views over that union; they are not independent candidate pools.

## 3. Actual corpus inventory and data integrity

The uploaded project snapshot contains **four registered/indexed documents** and five PDF files in the uploads directory. The four registered documents are:

| Registered document | PDF pages | Extracted page records | Indexed chunks |
|---|---:|---:|---:|
| `the-autobiography-of-benjamin-franklin-benjamin-franklin.pdf` | 128 | 128 | 569 |
| `the-enchiridion-epictetus.pdf` | 73 | 72 | 77 |
| `the-science-of-getting-rich.pdf` | 75 | 75 | 154 |
| `think-and-grow-rich-napoleon-hill.pdf` | 495 | 494 | 745 |
| **Total** | **771** | **769** | **1,545** |

The missing extracted records correspond to blank PDF pages: Enchiridion PDF page 8 and Think and Grow Rich PDF page 10 each contain zero text and no embedded images. They are blank pages, not evidence that readable page content was silently dropped.

Read-only integrity checks found:
- Metadata database `PRAGMA quick_check`: `ok`
- ChromaDB SQLite `PRAGMA quick_check`: `ok`
- Document metadata chunk total: `1,545`
- SQLite FTS5 rows: `1,545`
- ChromaDB embedding rows: `1,545`
- All four registered original PDFs exist and their SHA-256 hashes match the document metadata.

### Duplicate/orphan item requiring awareness

The uploads directory also contains `f2b4cc7f-e166-48b9-87fe-bbac7b95acb0.pdf`, which is **byte-for-byte identical** to the registered Franklin PDF `6573ac31-366a-46dd-9b96-f9112cee61fd.pdf`. The duplicate file has no separate `documents` row, and its matching extracted JSONL file is also orphaned. The recovery log contains two historical `missing_original` events for that same orphan ID.

This is a duplicate/orphan snapshot item, not a fifth distinct indexed document. It was **not deleted or reconciled**. Keep it intact until you review why it exists on the source machine.

The machine-readable snapshot is in `evaluation/local_corpus_audit.json`. Re-run the read-only audit with:

```powershell
python scripts/audit_local_corpus.py
```

## 4. Confirmed issues and implemented corrections

### A. Answer prompt was too restrictive for explanation and paraphrase questions
**Finding:** The generation prompt prohibited interpretation/synthesis and required the model to use only words appearing in the evidence. Those constraints can suppress valid why/how explanations, paraphrases, and answers where the document expresses the same idea using different wording.

**Change:** The prompt now permits narrowly supported synthesis and explanation, allows accurate paraphrases, asks the model to answer supported parts, and requires it to identify only the missing details. It still prohibits outside knowledge and speculation. It no longer directs the model to turn a partially answerable question into a generic refusal.

### B. A pure refusal could be returned even when selected evidence was relevant
**Finding:** A refusal-only first draft did not trigger a recovery attempt, even when selected evidence overlapped meaningfully with the question.

**Change:** Added one bounded evidence-based recovery attempt when selected evidence passes a conservative relevance signal. The signal uses evidence term coverage and semantic similarity together; a retrieval hit alone is not enough. A recovered answer must still pass citation validation and the existing verifier. If the evidence does not pass the relevance signal, or the recovery still refuses, the system can still abstain.

### C. Short answer-bearing chunks could be discarded
**Finding:** The evidence selector discarded every candidate shorter than 150 characters whenever any longer candidate existed. This could remove short but decisive table rows, values, labels, or concise definitions.

**Change:** Removed that unconditional short-chunk filter. Short chunks now compete using the same relevance, coverage, and redundancy scoring as other evidence. Added a regression test for a short approval-limit/table-value passage alongside a longer unrelated passage.

### D. InfoBooks boilerplate was penalized even for a question about InfoBooks
**Finding:** Publisher/download boilerplate was down-weighted for unrelated questions, but the query-aware exception did not recognize “How can I order more books from InfoBooks?”

**Change:** The reranker now avoids the boilerplate penalty when the question is explicitly about InfoBooks, a bookstore/publisher, downloading, or ordering books. The passage remains penalized for unrelated questions; it is not globally excluded.

### E. Verifier context selection missed comma-separated citation groups
**Finding:** Citation validation accepted `[1, 2]`, but verifier evidence focusing parsed only single-number citations. A grouped citation could therefore make the verifier receive more evidence than necessary.

**Change:** The verifier now parses both `[1]` and `[1, 2]` and focuses on every cited evidence block. Added a regression test confirming that both cited blocks are included and an uncited third block is omitted.

### F. A verifier-rejected retry could be reported as the selected draft
**Finding:** The verification retry was marked selected before the verifier had accepted it.

**Change:** Retry selection is now provisional. The draft becomes selected only after verification passes. If verification is unavailable or rejects the retry, `selected_draft` is `none` and recovery is marked unsuccessful; the answer remains withheld.

### G. Process-local cache contaminated test cases
**Finding:** The cache is intentionally process-local, but a test fixture changed the metadata path without clearing the in-memory cache. This let one test reuse another test's cached result and caused three unrelated retrieval/citation tests to fail when the entire suite ran.

**Change:** Added a small `cache_manager.clear()` helper and reset the process-local cache in the affected test fixture. Runtime cache behavior and data persistence are unchanged.

### H. Cache fingerprints
The retrieval pipeline and answer prompt version identifiers were incremented. Existing cached retrieval/answer results from the previous ranking and prompt versions therefore cannot be mistaken for results from this implementation. Existing document embeddings remain valid; no re-embedding is required.

## 5. Files changed

| File | Change |
|---|---|
| `app/rag/chat.py` | Evidence-friendly prompt; bounded refusal recovery; provisional verifier retry selection; grouped citation parsing; cache fingerprint version bumps |
| `app/retrieval/evidence_selector.py` | Retain short answer-bearing evidence chunks |
| `app/retrieval/reranker.py` | Query-aware exception for InfoBooks/store/download questions |
| `app/cache/manager/__init__.py` | Added process-local cache clear helper for isolated tests |
| `tests/test_chat_retrieval.py` | Reset global process-local cache per test |
| `tests/test_evidence_aware_recovery.py` | Updated verifier-retry expectations and added false-refusal/grouped-citation regressions |
| `tests/test_rnd2_quality_upgrade.py` | Added short answer-bearing chunk regression |
| `tests/eval_live_corpus.py` | Removed machine-specific Windows paths; added CLI configuration and answer/source expectation checks |
| `scripts/audit_local_corpus.py` | New read-only audit of documents, hashes, extracted records, FTS, ChromaDB and orphan files |
| `evaluation/local_corpus_audit.json` | Snapshot audit output |
| `README.md` | Added audit/live-evaluation commands and corrected factual TOP_K description |
| `RND2_RELIABILITY_AUDIT_REPORT.md` | This report |

Original versions of all eight modified existing source/test files are preserved in:

```text
backup_20261105_rag_reliability/
```

The new audit script and report do not replace existing modules.

## 6. Validation results

### Unit and regression tests

| Run | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Uploaded snapshot baseline | 159 | 6 | 1 |
| Updated working copy | **168** | **0** | **1** |

The skipped test is the ChromaDB persistence integration test; the Python `chromadb` package is not installed in this audit runtime. This is not counted as a pass.

The OCR extraction path was smoke-tested on the two image-heavy cover pages using the local Tesseract binary in this audit runtime. The original `.env` was not changed. The target Windows machine should still verify its configured `TESSERACT_CMD` path.

### Actual persisted database checks

The raw SQLite files inside the supplied ChromaDB and metadata directories were inspected read-only. Their integrity checks passed and their indexed chunk counts agree at 1,545. No ChromaDB collection was recreated and no database was rebuilt.

### Live model/API evaluation

Not run in this environment:
- Ollama was not running at `127.0.0.1:11434`.
- The `chromadb` Python package was unavailable for a live collection query.
- No live answer correctness, citation entailment accuracy, response latency, or model memory measurements can therefore be claimed.

The 19-case real-corpus evaluation script has been made portable and now includes manually labelled expected answer terms and source-document/page checks for selected cases. It must be run against the live local application before release. Its lexical checks are regression signals; they are not a semantic proof.

## 7. Remaining limitations

1. **No universal-support claim:** the snapshot contains PDFs only. DOCX, XLSX, CSV, PPTX, images, EML, XML, YAML, ODT and other supported formats have not all been validated against real user files in this run.
2. **Extraction support:** the current `SUPPORTED` extension set includes PDF, DOCX, TXT, MD, RTF, ODT, HTML/HTM, LOG, YAML/YML, XLSX/XLSM, ODS, CSV, PPTX, JPG/JPEG/PNG/TIFF/BMP/WEBP, JSON, XML and EML. Legacy `.xls` and `.ppt` are not natively listed as supported.
3. **Chunking:** chunking is character-based and structure-aware, not token-aware or semantic. Page/section labels are retained, but automatic neighboring-chunk expansion is not implemented.
4. **Reranking:** the reranker is deterministic and heuristic, not a trained cross-encoder.
5. **Citations:** reference validation checks citation syntax, bounds, labels and sentence coverage. The local LLM verifier is a second model judgement, not mathematical proof of entailment.
6. **Conflict detection:** conflict detection remains heuristic and may miss or over-report disagreements.
7. **Confidence:** confidence is a weighted diagnostic score, not a statistically calibrated probability.
8. **OCR:** the local OCR path was smoke-tested, but scanned documents with dense tables, rotated pages, handwriting, or poor image quality need separate target-machine tests.
9. **Performance:** before/after latency and RAM usage were not measured because live Ollama was unavailable.
10. **Retrieval pool:** dense and FTS routes are both used, but downstream ranking diagnostics correctly describe their merged candidate pool as shared. Independent dense-vs-lexical Recall@K still requires a manually labelled chunk-ID dataset.

## 8. Restart and validation on the Windows machine

Do not delete or rebuild `data/chroma_db`, `data/metadata`, `data/uploads`, or `data/extracted`. Do not overwrite a newer `.env` or newer runtime `data/` directory with an older snapshot copy.

From the project root, activate the existing environment and start the backend:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Run the offline corpus inventory:

```powershell
python scripts/audit_local_corpus.py
```

Run the quick real-corpus API checks:

```powershell
python tests/eval_live_corpus.py --quick
```

Run the complete 19-case real-corpus matrix:

```powershell
python tests/eval_live_corpus.py
```

Run all unit/regression tests:

```powershell
python -m pytest -q
```

The live evaluation writes timestamped JSON files under `evaluation/live_results/` by default. Review every failed case, especially multi-part, cross-document, comparison, missing-information, and cache-consistency cases, before treating the build as production-ready.

## 9. Final status

**Verified in this audit runtime:** targeted source changes, regression suite, Python compilation, read-only database integrity and chunk-count consistency, duplicate/orphan inventory, and a Tesseract OCR smoke test.

**Partially tested:** real-corpus extraction inventory and manually labelled 19-case evaluation design.

**Not yet verified:** live Qwen2.5:3b answers, actual ChromaDB query retrieval, semantic citation entailment, all non-PDF formats, full OCR workload, Windows end-to-end UI/API behavior, and before/after latency or memory.

The build is a targeted reliability improvement, not a claim that every possible question or every supported file format is now guaranteed to work. Final local validation completed with 168 passing tests, one skipped ChromaDB integration test, successful Python compilation, and no changes to the uploaded `.env` or any file under `data/`.
