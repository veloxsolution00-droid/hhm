# Update Notes — Offline Document Chatbot

## Implemented in this build

- Added bounded recent conversation history to `POST /chat` and passed recent user/assistant turns from Streamlit.
- Added local-model follow-up rewriting with a safe fallback to the original question.
- Added deterministic question routing for factual, scenario, comparison, summary, explanation, calculation, and follow-up categories.
- Added conservative query variants and a second retrieval pass when the initial pass returns no evidence above the configured threshold. The original user spelling is also searched when common-typo normalization changes the query.
- Improved vocabulary spelling correction to fix multiple likely typos in one query; preserves acronyms, title-case names, and numeric/hyphenated identifiers, and avoids ambiguous fuzzy matches. The corrected query is searched even if the initial query returned some hits; the original query remains a separate route.
- Preserved hybrid ChromaDB semantic retrieval plus SQLite FTS5 exact-term retrieval; candidate retrieval base default is `TOP_K=4` (expanded by question type for complex questions), `RETRIEVAL_CANDIDATE_MULTIPLIER=5`, `HYBRID_LEXICAL_WEIGHT=0.35`, and `LEXICAL_CANDIDATE_LIMIT=40`.
- Added a safe arithmetic evaluator for explicit arithmetic/percentage expressions using a restricted AST (no Python `eval`). For document-derived calculations, the model proposes a formula/source references; the code checks source indices and verifies numeric operands appear in the question/evidence before evaluating the expression. Formula selection remains a semantic risk.
- Added a configuration guard that rejects non-loopback or HTTPS Ollama endpoints, enforcing the local-only runtime configuration.
- Added numeric/filename citation-reference validation; generated answers with missing or invalid citations are withheld. Source passages remain available in the API/UI.
- Added a conservative potential-conflict detector for retrieved passages with similar terms and differing numbers or polarity signals. This is only a warning signal; the model must inspect the passages and cite both sides.
- Kept grounded instructions for scenario facts, user assumptions, derived conclusions, missing conditions, comparisons, summaries, calculations, and explanations.
- Added offline evaluation metric helpers and an API evaluation runner with configurable case checks.
- Updated the Streamlit diagnostics to show retrieval passes, citation validation warnings, and potential evidence conflicts.
- Reworked chunking to prefer paragraph/heading/sentence boundaries, preserve bounded overlap, and enforce a hard chunk-size cap; added long-paragraph and structure tests.
- Removed the unreferenced `queue_ingest` and `process_document` duplicate ingestion entry points from `app/services/documents.py`; the active durable job queue uses `ingest` directly. See `DEAD_CODE_AUDIT.md`.
- Added `FEATURE_COVERAGE.md` to make implemented, partial, blocked, and not-run requirements explicit.
- Updated README, `.env.example`, and regression tests.
- Added bounded process-local query-embedding, exact/semantic retrieval, and exact successful-cited-answer caching; cache is disposable and cleared on backend restart.
- Preserved existing storage and ingestion architecture; no code path was added to clear the user's persistent data at startup.

## Validation performed in this environment

- Python AST parsing: passed for packaged Python source files.
- `python -m pytest -q tests`: **99 passed, 1 skipped** in the available Python 3.13 environment.
- A mocked ingestion regression test verifies permanent original storage, extracted-text persistence, indexing metadata, chunk records, and job linkage without live Ollama/ChromaDB.
- Live service startup, actual Ollama inference, persistent ChromaDB indexing, OCR, and target-machine performance were not tested in this environment.

## Not claimed as solved

- This is not a formal policy/rule engine and cannot guarantee correct scenario reasoning for every possible case.
- The arithmetic engine safely evaluates expressions. Identifying the right figures/formula from arbitrary documents remains an evidence-selection and interpretation problem; the code validates numeric operands but cannot prove the formula is conceptually correct.
- The conflict detector is heuristic and may flag unrelated numbers; it may also miss semantic contradictions.
- The LLM verifier is another model judgment, not an independent proof of truth.
- The evaluation runner only measures the checks that you define in your own case file; it is not a pre-filled benchmark of your private documents.
- Live Ollama generation/embedding, ChromaDB indexing/retrieval, FastAPI startup, and a loopback Streamlit launch were not exercised here. OCR, firewall-level network isolation, and latency/memory profiling must be validated on the target Windows machine.
- The second UI was not included as a separate artifact in the current `RND` archive, so a true merge of two distinct UI implementations cannot be completed from this archive alone.


## Local caching added

- Cache is bounded, process-local memory (not a persistent SQLite database).
- Query embeddings are scoped by embedding model.
- Retrieval cache entries are scoped by owner, corpus/configuration fingerprints, and question workflow.
- Answer cache stores only successful cited answers and exact normalized-question matches by default.
- Cache TTL and entry count are bounded; backend restart clears cache contents without affecting documents or indexes.

## Cache correctness safeguard

Semantic answer reuse is now disabled by default (`SEMANTIC_ANSWER_CACHE_ENABLED=false`). A semantic retrieval-cache hit may reuse candidate chunks, but the application generates a fresh answer for the current question. Exact normalized-question answer hits remain available when corpus/config fingerprints match. This is intentionally conservative: embedding similarity is not proof that two questions request the same fact. Enabling semantic answer reuse should be done only after evaluating a labeled set of paraphrase and hard-negative question pairs.

## Follow-up and citation recovery fixes

- Standalone questions are no longer sent to the follow-up resolver just because history exists. Only elliptical/context-dependent forms such as "What about his role?" are resolved using conversation history.
- Query normalization removes an accidental comma inserted between letters (for example, `bra,m` becomes `bram`) before retrieval. Vocabulary-based correction now starts from the normalized query and uses a slightly more permissive cutoff; the original query is still searched.
- If the first generated answer is an abstention or fails citation-reference validation, the app retries once using the same retrieved evidence and strict numeric citation bounds.
- If citation validation still fails, the status is `citation_validation_failed`. These are no longer mislabeled as a genuine lack of evidence.
- The generation prompt forbids added analysis, interpretation, motives, symbolism, or conclusions, and forbids copying a term from the question that the evidence never uses. The model otherwise padded correct answers with invented commentary and echoed question vocabulary (for example answering "Transylvania" where the evidence says "Carpathian mountains").
- The answer verifier is a conservative gate: if verification explicitly fails, the draft is withheld with `answer_validation_failed`. This reduces unsupported answers but can produce false negatives with small local models.
- The answer verifier prompt is anchored by concrete examples rather than an abstract definition of "supported". With the strict wording the local model rejected grounded, correctly cited drafts (a false negative).
- Generation calls send `keep_alive` so the model stays resident across the closely spaced generate/retry/verify calls of one request. On CPU-only hosts Ollama otherwise unloads the model between calls and each reload is paid as latency.
- `TIMEOUT` is a per-Ollama-call limit and defaults to 300 s. CPU-only inference of a grounded answer can exceed 90 s, so the previous 90 s default aborted valid generations.
- The answer-cache prompt fingerprint was bumped so cached results from the previous answer-generation rules are not reused.
- Added regression tests for standalone-question preservation, elliptical follow-ups, punctuation typo normalization, and citation recovery.


- Balanced retrieval update: base `TOP_K=4`; factual=4, explanation=6, calculation=5, summary=7, comparison=7. Dense candidate pool remains `TOP_K × RETRIEVAL_CANDIDATE_MULTIPLIER` (20 by default); lexical candidate limit remains 40.
