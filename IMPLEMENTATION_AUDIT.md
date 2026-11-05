# Implementation Audit — Offline Document Chatbot

## Included engineering work

- Existing local-only FastAPI + Streamlit + Ollama + ChromaDB + SQLite architecture retained.
- Chat API accepts a bounded recent history; UI sends prior user/assistant turns for follow-up resolution.
- Question planner routes factual, scenario, comparison, summary, explanation, calculation, and follow-up intents.
- Retrieval retains dense semantic + FTS5 keyword matching; it searches original and normalized query forms, adds conservative query variants after weak retrieval, and can correct likely typos toward terms found in indexed document vocabulary.
- Evidence selection filters to indexed documents and uses multiple chunks within a configured context limit.
- Answer prompts treat uploaded content as untrusted data, separate scenario facts/assumptions/conclusions, request citations, and instruct the model to report document conflicts instead of silently selecting one.
- Local-model answer verification is enforced conservatively when enabled; failed verification withholds the draft, though false negatives remain possible.
- Citation validation checks numeric reference bounds, explicit filename references, and missing citations; invalid answers are withheld.
- Potential conflict heuristics identify likely differing numbers or polarity signals for model review.
- Explicit arithmetic uses a restricted AST evaluator. For evidence-based calculations, the local model proposes a formula and source references; source indices and numeric operands are checked before deterministic evaluation.
- Runtime Ollama endpoint is constrained to plain HTTP loopback hosts.
- Added structure-aware paragraph/heading/sentence chunking with bounded overlap and regression tests. This is heuristic chunking, not embedding-based semantic segmentation.
- Added retrieval metrics helpers, a local API evaluation runner, sample evaluation cases, regression tests, and updated documentation.
- Added `FEATURE_COVERAGE.md`, a requirement-by-requirement matrix that explicitly marks partial, blocked, and not-run items.
- Added a bounded process-local in-memory cache for query embeddings, exact/semantic retrieval results, and successful cited answers; cache scope includes owner, corpus fingerprint, and relevant model/config fingerprints.
- Existing ingestion, persistence, recovery, and data folders retained. No startup data-clearing behavior was introduced.

## Verification

- Python AST parsing and targeted test collection were used during the repair pass.
- `python -m pytest -q tests`: **99 passed, 1 skipped** in the available Python 3.13 environment.
- A mocked ingestion regression test validates permanent original-file storage, extracted-text persistence, indexing metadata, chunk creation, and job/document linkage.
- Live Ollama and ChromaDB integration was **not** available in this execution environment; no live FastAPI/Streamlit startup or real vector indexing is claimed.

## Post-review fixes

- Multi-question citations of the form `[1, 2]` are remapped to the combined source list.
- `LEXICAL_CANDIDATE_LIMIT` is enforced as a real cap; ChromaDB telemetry is disabled for this local-only app.
- FTS `owner_id` rows are migrated with the rest of a legacy install; retry submission and job claiming are atomic so a job cannot be processed twice.
- Direct arithmetic is allowed with no indexed documents, but year ranges and identifiers (for example `2024-2025`) are no longer evaluated as subtraction.
- Statements after a question mark remain part of that question instead of becoming extra sub-answers.
- Extraction handles HTML as visible text, cp1252/latin-1 text, BOM-prefixed JSON, and both PyMuPDF module names.
- Relative storage overrides are anchored to the project root; `/health` reports degraded (and returns 200) when metadata is unavailable instead of raising.
- Streamlit reads the batch limit from `/health` and truncates history to the API's 2,000-character limit.

## Limits requiring real-machine validation

- ChromaDB, Ollama generation/embedding models, document indexing, and a loopback FastAPI + Streamlit launch were exercised on this machine. OCR was not validated because no image-only/OCR fixture was available.
- Operation with the network fully disabled (firewall-level) was not tested; all configured endpoints are loopback-only.
- Structure-aware chunking is not embedding-based semantic segmentation; ranking is not a calibrated confidence model.
- Scenario reasoning is not a formal policy engine. LLM-based formula selection can still be wrong even when numeric operands and arithmetic syntax validate.
- Citation checks prove reference bounds, not semantic entailment. Model verification is not independent proof.
- Conflict detection is heuristic and can both over-flag and miss contradictions.
- The evaluation case file is a template; it must be populated with ground truth from your own documents.
- Only one UI implementation was present in the supplied archive. A true merge of two distinct UIs requires the other UI source as well.
