# Offline Document AI (RND2) — Revised Implementation Plan

**Status:** PRE-IMPLEMENTATION. Awaiting explicit approval. No code, configuration, dependency, index, or production document has been modified.
**Date:** 13 November 2026
**Audited revision:** `57872830ebaacc49aaab32273c9965351c62a359` ("new version fixed"), branch `main`
**Proposed engineering branch (not yet created):** `fix/rag-answer-reliability-2`, from `5787283`
**Target configuration:** `qwen2.5:3b` generation · `nomic-embed-text` embeddings · programmatic evidence verification by default · conditional same-model verification only when demonstrably necessary · one inference at a time · CPU, 8 GB RAM, offline.
**Supersedes:** the plan in `EVALUATION_REPORT_3B_REVALIDATION.md` §G (preserved, not overwritten).

---

## 1. Baseline and Git verification

| Item | Verified value | Evidence |
|---|---|---|
| Working tree | Tracked files clean | `git status --porcelain` → only `?? evaluation/audit/` |
| HEAD commit | `57872830ebaacc49aaab32273c9965351c62a359` | `git rev-parse HEAD` |
| Branch | `main` | `git branch --show-current` |
| Diff vs HEAD | Empty | `git diff`, `git diff --cached` empty |
| Model actually loaded (report's live runs) | `qwen2.5:3b` | Ollama log `2dc4cbc144f63bd5bf4322fb7fc581da\output.log`: 10× `model=qwen2.5:3b`, 0× `qwen2.5:7b`, 2026-11-13 13:02–13:09 |
| Model-proof cross-check | Timing match | `live_cfgA_3b_programmatic_rerun.log` gen times == log elapsed_seconds within ~50 ms |
| Config override mechanism | env > `.env` | `run_configs.ps1` lines 8/48–50 set `GENERATION_MODEL`/`VERIFICATION_MODEL`/`VERIFY_ANSWERS`; health line `gen=qwen2.5:3b ver=qwen2.5:3b` |
| Deployed `.env` (NOT used in those runs) | `GENERATION_MODEL=qwen2.5:7b`, `VERIFICATION_MODEL=qwen2.5:7b`, `VERIFY_ANSWERS=true` | `.env` lines 3, 6, 39 |
| Code default (disagrees with `.env`) | `generation_model="qwen2.5:3b"`, `verification_model="qwen2.5:7b"`, `verify_answers=True` | `app/core/config.py` lines 40, 47, 166 |

**Baseline metrics (frozen; from `retrieval_eval_2.log` + `live_cfgA_3b_programmatic_rerun.log`):**

| Metric | Baseline |
|---|---|
| Recall@1 / @3 / @4 / @8 / @20 | 0.215 / 0.431 / 0.477 / 0.585 / 0.662 |
| MRR | 0.348 |
| recall_selected | 0.538 |
| precision_selected | 0.149 |
| page_coverage_selected | 0.423 |
| term_coverage_selected | 0.683 |
| Live cases (n=9) clearly correct | 1 (H05) |
| Live latency | mean 40.5 s · P50 31.7 s · max 70.6 s |
| Unit tests | 203 passed, 1 skipped (16.9 s) |

**Protected resources (must remain untouched throughout):** production source PDFs and `data/extracted/*.jsonl`; ChromaDB collection; SQLite `chunk_search` FTS5 index; `.env`; `main`.

---

## 2. Confirmed root causes

Verification key: **[S]** source-verified, **[R]** reproduced/tested, **[H]** hypothesis (see §3).

| ID | Finding | Basis |
|---|---|---|
| RC-1 | `_named_document_scope` collapses a comparison to one document (`return named if len(named)==1 else set()`) | [S] `chat.py#L697-L721`; [R] D01 refused live |
| RC-2 | Reranking can favour commentary/back-matter over relevant content; boilerplate penalty only fires when `coverage < 0.25` | [S] `reranker.py#L7-L14,#L170-L189`; [R] A02 Page 7 exists in `data/extracted/*.jsonl` but never retrieved; rank #1 = Page-4 decoy |
| RC-3 | FTS5 OR-joins all terms; FTS-only hits scored `lexical_weight × coverage ≤ 0.35` against gate `min_similarity=0.25` ⇒ ≥71% term match needed | [S] `store.py#L35-L37,#L168-L174`; `database.py#L580-L595`; `chat.py#L1292,#L1339`; [R] metric arithmetic |
| RC-4 | Evidence selection is greedy with no adaptive diversity; `max_chunks` = `top_k`, `max_chars` budget only | [S] `evidence_selector.py#L15-L86`; [H] causal quality impact (one observation, F01) |
| RC-5 | Completeness not enforced (dropped "Friday" A01; partial 6-step list B02) | [S] prompts/answer-planner; [H] prompt vs model-capacity split |
| RC-6 | Early abstention is dead code — `is_question_bank_query=False`, `lacks_document_anchor=False` ⇒ `abstain_early` always `False` | [S] `chat.py#L762-L783`; [R] J01 hallucinated live |
| RC-7 | `fully_cited = coverage==1.0`; `valid = references_valid and fully_cited`; only `valid` drafts advance ⇒ valid partial/paraphrased answers rejected | [S] `citations.py#L243-L246`; `chat.py#L1659-L1665`; [R] D01 refused |
| RC-8 | Spelling correction can change meaning; joined words unsplit. **Correction: default cutoff is 0.86, not 0.80** | [S] `planner.py#L125-L177`; [R] `feers→fees` |
| RC-9 | Latency is generation-bound (all non-generation stages < 2 s) | [S]/[R] `score_live.py` stage output |
| RC-10 | `.pptx`/`.rtf` advertised in `SUPPORTED` but `python-pptx`/`striprtf` not installed; imports unguarded | [S] `extractor.py#L13,#L221-L226,#L323-L325`; `requirements.txt`; [R] format probe |
| RC-11 | Extractive fallback accepts ≥50% overlap and can bypass the verifier via `EXACT_PASSAGE_MATCH` | [S] `chat.py#L333-L448,#L1760-L1768`; [H] whether it ever misfires in practice |
| RC-12 | Up to 3 generations (+2 verifications) per question (`for attempt in (1, 2)`) | [S] `chat.py#L1777-L1810` |

---

## 3. Unresolved hypotheses (each with a resolution test)

| ID | Hypothesis | Resolution test (read-only / test build) |
|---|---|---|
| H1 | RC-4: single-document crowding causally degrades quality, vs. one incidental observation | Instrument `selected` doc distribution across the full 70-question harness; correlate with `recall_selected` |
| H2 | RC-5: incompleteness is prompt-engineering vs 3B capacity | A/B two prompt variants on identical evidence, fixed seed, k=3 |
| H3 | RC-11: extractive fallback actually misfires | Count `extractive_*` / `EXACT_PASSAGE_MATCH` firings across the full harness; adjudicate any firing |
| H4 | RC-9: latency split among prompt length, `num_predict`, CPU scheduling | Log `prompt_eval_count`, `eval_count`, `load_duration` from Ollama responses |
| H5 | Whether conditional same-model verification is ever required at 3B | Run programmatic-only vs conditional-verify on ambiguous/paraphrase cases; keep programmatic default |
| H6 | RC-10: the failing `.pptx`/`.rtf` path is an unhandled ImportError vs another fault | Trace the exact exception through the job queue in a test process |

No hypothesis is treated as a fix target until its resolution test passes.

---

## 4. Corrections to the original plan

1. **Sample size (§3.1):** the prior plan's "≥6 of 9" is withdrawn. Acceptance is redefined on the expanded dataset (§9, §11). The 9-question set is retained only as regression anchors.
2. **Citation vs sufficiency (§3.2):** the prior plan treated "relax the citation gate" and "restore abstention" as adjacent. They are now **four separate concepts** with separate code paths, metrics, and tests:
   - **A. Citation integrity** (structural): cited doc exists, chunk exists, page metadata matches actual source, reference resolves to a real retrieved record.
   - **B. Claim-level evidence support**: does the cited passage support the claim (direct / paraphrased / distributed / contradictory / unsupported).
   - **C. Answer completeness**: all requested components answered (list items, numbers, steps).
   - **D. Evidence sufficiency**: is the available evidence sufficient to answer at all.
   Behaviour contract: preserve supported claims; remove/qualify unsupported claims; attach citations using **actual retrieved metadata only**; never let the model invent doc/chunk/page ids; never reject a useful answer over one unsupported sentence; never accept an answer merely because its citation points to a real document; abstain only when remaining evidence cannot support a reliable answer. **Do not weaken evidence validation merely to reduce citation failures.**
3. **FTS5 (§3.3):** the prior plan's "rare-term weighting / AND-of-rare-terms" is replaced by a measured investigation (BM25, IDF, phrase, exact entity/identifier match, query-term coverage, controlled variants, candidate limits, strict-match fallback). **The original user query must always remain available.** OR must not be blindly replaced by AND. Measure precision *and* recall, not candidate counts.
4. **Evidence selection (§3.4):** the prior plan's "per-document cap" is withdrawn. Selection must be **adaptive** to question type (single-doc, multi-doc comparison, multi-page, procedural, numerical) balancing diversity, redundancy, and relevance; preserve single-document context when one document genuinely holds the answer.
5. **Abstention (§3.5):** implement as an independent deterministic sufficiency mechanism; measure false abstentions and unsupported answers **separately**; never lower thresholds merely to answer more questions.
6. **Performance (§3.6):** measurement is expanded to full stage breakdown, TTFT vs total generation, model-call counts, prompt/generated tokens, CPU, and all three memory figures; P90/P95 only reported where n supports it.
7. **Config hazard:** `config.py` default `generation_model=qwen2.5:3b` vs `.env` `qwen2.5:7b`. Runtime model for all evaluation is pinned by env override and asserted in the harness (see §12).

---

## 5. Revised implementation order (dependency-aware)

**Phase 1 — Retrieval candidate quality:** RC-3 (FTS5/lexical) → RC-2 (reranking) → RC-4 (adaptive selection).
**Phase 2 — Document scope:** RC-1 (comparison/scoping).
**Phase 3 — Evidence reliability:** RC-7 (citation integrity **A** + claim support **B**) → RC-6 (sufficiency **D**). A/B and D stay separate.
**Phase 4 — Answer quality:** RC-5 (completeness **C**) → RC-8 (meaning-preserving spelling).
**Phase 5 — Ingestion & orchestration:** RC-10 → RC-11 → RC-12.
**Phase 6 — Performance:** RC-9.
**Phase 7 — Final regression & acceptance.** Correctness precedes performance.

---

## 6. Exact files and functions involved

| RC | Files | Functions / lines |
|---|---|---|
| RC-1 | `app/rag/chat.py` | `_named_document_scope` (697-721); scope application + caller in `_answer_one`; `classify_question` consumer |
| RC-2 | `app/retrieval/reranker.py` | `_BOILERPLATE` (7-14); blend + penalty (170-189) |
| RC-3 | `app/retrieval/store.py`; `app/storage/database.py` | `_query_terms` (20-22), `_lexical_similarity` (24-33), `_fts_query` (35-37), FTS branch (150-174); `search_indexed_chunks` (580-595) |
| RC-4 | `app/retrieval/evidence_selector.py` | `select_evidence` (15-86) |
| RC-5 | `app/rag/prompts.py`; `app/reasoning/answer_planner.py`; `app/rag/chat.py` | `answer_instructions`; `build_answer_plan`, `coverage_summary`; completion pass |
| RC-6 | `app/rag/chat.py` | `_analyze_question` (753-800); new sufficiency gate near (1292/1339) |
| RC-7 | `app/rag/citations.py`; `app/rag/chat.py` | `validate_citations` (230-256); `_drop_unsupported_sentences` (487); selection (1659-1665); failure branch (1700-1746) |
| RC-8 | `app/rag/planner.py` | `correct_query_from_vocabulary` (125-177) |
| RC-9 | `app/rag/chat.py`; `app/rag/prompts.py`; `app/core/config.py` | prompt sizing, `answer_max_tokens`, stage timing |
| RC-10 | `app/ingestion/extractor.py`; `requirements.txt` | `SUPPORTED` (13); pptx (221-226); rtf (323-325) |
| RC-11 | `app/rag/chat.py` | `_extractive_recovery_from_evidence` (333-392), `_extractive_answer_for_question` (395-431), `_is_exact_passage_extract` (433-448) |
| RC-12 | `app/rag/chat.py` | verification loop (1777-1810); retrieval passes (1292-1380) |

---

## 7. Detailed changes proposed for each root cause

Each change states: **why · what is wrong now · how corrected · how verified · what could break · rollback.**

### RC-3 — FTS5 and lexical retrieval (Phase 1)
- **Why:** rare-term / identifier / exact-entity queries are crowded out; FTS candidates are effectively killed at the gate.
- **Wrong now:** `_fts_query` returns `" OR ".join(...)` over all terms; FTS-only similarity = `0.35 × coverage`, so ≥71% coverage is required to pass `min_similarity=0.25`.
- **Correction (proposal, to be measured):** introduce a **layered** query strategy — (1) exact/phrase attempt for multi-word entities and identifiers; (2) rare-term (higher-IDF) OR fallback; (3) current OR as final fallback when strict attempts return nothing. Weight FTS contribution using BM25 rank rather than raw coverage, and evaluate a separate FTS gate instead of the shared `min_similarity`. **The original normalized query is always issued and its results always retained** (no query loss). Candidate limits (`lexical_candidate_limit`, dense multiplier) unchanged.
- **Verify:** unit tests on `_fts_query`/scoring; retrieval eval on K/A/H categories; measure precision *and* recall.
- **Could break:** query syntax errors → mitigated by existing `syntax error` fallback; over-broad recall → bounded by candidate limits and reranking.
- **Rollback:** strategy behind a config flag defaulting to current behaviour.

### RC-2 — Reranking (Phase 1)
- **Why:** the true passage (A02 Page 7) never outranks commentary.
- **Wrong now:** boilerplate penalty applies only when `coverage < 0.25`; commentary can carry high cosine similarity and full lexical coverage.
- **Correction:** make boilerplate/back-matter detection effective independently of coverage (bounded penalty with a floor), and add a content-vs-commentary signal derived from the existing `_BOILERPLATE` vocabulary. Keep the blend deterministic.
- **Verify:** unit test (commentary vs content chunk ordering); retrieval eval — correct page enters top-4 for A02/I01/K02/H05.
- **Could break:** penalising legitimate prefaces → bounded penalty + per-category eval.
- **Rollback:** penalty weight behind a flag (default current).

### RC-4 — Adaptive evidence selection (Phase 1)
- **Why:** one document can fill the context, or required multi-document evidence is not represented.
- **Wrong now:** greedy `0.78◦base + 0.27◦coverage_gain − 0.20◦redundancy`, no diversity term.
- **Correction (adaptive, **not** an unconditional cap): derive selection policy from question type — single-document questions keep deep single-doc context; multi-document/comparison questions require representation of each named/relevant document; procedural/numeric questions prioritise contiguous steps/values. Diversity is a soft adaptive term, not a hard cap.
- **Verify:** unit tests per question class; retrieval eval measuring **recall_selected and precision_selected** on D/E/F categories.
- **Could break:** dropping needed single-doc context → guarded by the single-document policy and its regression set.
- **Rollback:** selection policy behind a flag (default current).

### RC-1 — Multi-document comparison and scoping (Phase 2)
- **Why:** D01-class comparison questions are scoped to one file and refused.
- **Wrong now:** `named if len(named)==1 else set()` — a comparison naming two entities matches one filename token.
- **Correction:** do not scope when the question is a comparison/multi-document type or names ≥2 distinct entities; scope only for single-document, non-comparison questions where the matched token covers the distinctive title tokens. When scoping is disabled, keep all documents that contain named entities.
- **Verify:** unit tests; D01 live returns a two-source answer.
- **Could break:** cross-document bleed for single-doc factual questions → scoping retained for that class; regression set.
- **Rollback:** scoping rule behind a flag.

### RC-7 — Citation integrity + claim-level support (Phase 3, concept A+B)
- **Why:** valid, partially-supported answers are rejected wholesale, and citation correctness is conflated with evidence sufficiency.
- **Wrong now:** `fully_cited = coverage==1.0` gates acceptance; only `valid` drafts advance.
- **Correction (two separate checks):**
  - **A. Citation integrity (hard, structural):** every label ∈ `[1, |sources|]`; every label resolves to a **retrieved evidence record**; the cited page/chunk metadata matches the actual source record. Phantom/fabricated identifiers are never accepted. Citations are attached programmatically from retrieved metadata, not trusted from model text.
  - **B. Claim-level support (soft, per sentence):** supported sentences kept; unsupported sentences **removed or explicitly qualified** via `_drop_unsupported_sentences`; a useful answer is **not** rejected because one sentence failed support.
  - Evidence sufficiency (D / RC-6) remains a separate gate and is **not** weakened.
- **Verify:** unit tests — partial-but-reference-valid draft accepted after unsupported-sentence removal; phantom label rejected; paraphrase verified by entailment. Live — D01 no longer `citation_validation_failed`.
- **Could break:** weakly grounded sentences slipping through → they are removed, not kept; entailment check remains for paraphrases.
- **Rollback:** new acceptance rule behind a flag (default current) until the phase passes.

### RC-6 — Evidence sufficiency and abstention (Phase 3, concept D)
- **Why:** unanswerable questions are answered with fabricated facts.
- **Wrong now:** `abstain_early` is hardcoded `False`.
- **Correction:** implement a **deterministic, measurable** sufficiency function over retrieved evidence (max hybrid similarity, salient-entity presence, term/requirement coverage, contradiction detection) producing one of {sufficient, partially sufficient, insufficient, contradictory, irrelevant}. Abstain before generation on insufficient/irrelevant; qualify on partial; resolve/qualify on contradictory.
- **Verify:** the seven evidence classes in §3.5 (strong/weak/missing/contradictory/partial/unanswerable/misleading). **False abstentions and unsupported answers measured separately.**
- **Could break:** over-abstention → thresholds conservative and validated against the full answerable set.
- **Rollback:** sufficiency gate behind a flag (default off until phase passes).

### RC-5 — Answer completeness (Phase 4, concept C)
- **Why:** dropped details and truncated lists.
- **Wrong now:** no enforced completeness check.
- **Correction:** build an answer plan / requirement checklist, require all list items, numbers, and steps, and run at most **one** bounded completion pass; verify against `evidence_requirements.json` groups.
- **Verify:** A01 includes "Friday"; B02 all six steps.
- **Could break:** extra latency → bounded to one pass; correctness precedes latency.
- **Rollback:** completion pass behind a flag.

### RC-8 — Meaning-preserving spelling correction (Phase 4)
- **Why:** corrected queries can change meaning.
- **Wrong now:** default `cutoff=0.86` (report said 0.80 — corrected); fuzzy replacement can alter meaning.
- **Correction:** never replace a token when the substitution changes meaning; fix only unambiguous typos; guard joined words conservatively; **always retain the original query** as an additional (never exclusive) query.
- **Verify:** labelled typo set — `feers` unchanged; `benjmin frankln`→`benjamin franklin`; ambiguous left unchanged.
- **Could break:** under-correction → original query always present.
- **Rollback:** cutoff/guard behind a flag.

### RC-10 — Format support and dependency handling (Phase 5)
- **Why:** advertised formats fail silently at ingestion.
- **Wrong now:** `.pptx`/`.rtf` in `SUPPORTED`; imports unguarded; deps absent.
- **Correction:** either install the declared dependencies **or** remove the extensions from `SUPPORTED` and report an accurate capability list; wrap the imports so a missing dependency yields a clear, controlled "unsupported (missing dependency)" result rather than an unhandled error. Do not advertise unverified support.
- **Verify:** format probe green for supported, clear message for unsupported; negative test for missing dep.
- **Could break:** none material.
- **Rollback:** revert `SUPPORTED` and import guards.

### RC-11 — Extractive fallback (Phase 5)
- **Why:** a non-answer could be surfaced as an answer.
- **Wrong now:** ≥50% overlap acceptance; `EXACT_PASSAGE_MATCH` bypasses the verifier.
- **Correction:** require strict contiguous exact extracts for the verifier bypass; tighten the overlap acceptance; never bypass verification for non-exact extracts.
- **Verify:** unit tests + harness firing audit (H3).
- **Could break:** missing valid extracts → exact-extract path retained.
- **Rollback:** flag for the fallback threshold.

### RC-12 — Generation and recovery call budget (Phase 5)
- **Why:** up to 3 generations (+2 verifications) inflate tail latency.
- **Wrong now:** `for attempt in (1, 2)`.
- **Correction:** default one generation; recovery only on a **conclusive** verifier rejection; cap model calls per question; record calls/question.
- **Verify:** unit test on call count; perf measurement.
- **Could break:** missing one needed recovery → only conclusive verdicts trigger recovery.
- **Rollback:** flag restoring the current retry loop.

### RC-9 — Performance (Phase 6, last)
- **Why:** median 31.7 s, generation-bound.
- **Correction:** evidence-first extractive answers for factual/numeric/list questions where demonstrably more reliable; hard token caps; shorter prompts; initial context target **2048 tokens** (larger only if measured RAM permits); optional streaming for earlier first token. **Correctness must not be traded for latency.**
- **Verify:** full stage-latency + memory + CPU + token measurement (§12).
- **Could break:** accuracy → correctness gates first; all P0–P1 metrics re-measured after.
- **Rollback:** each optimisation behind its own flag.

---

## 8. Unit, integration and regression tests

**Unit (per RC):** `_named_document_scope` (comparison/multi-entity/single-doc); `_fts_query` + FTS scoring + fallback; reranker ordering (commentary vs content); `select_evidence` per question class; citation integrity (valid/invalid/phantom/missing-page); claim support (direct/paraphrase/distributed/unsupported); sufficiency classes; completeness (list/number/step); spelling (typo/joined/ambiguous); extractor dependency handling; call-budget counters.

**Integration:** end-to-end retrieval harness (`retrieval_eval.py`) per phase; live harness (`live_eval.py`) per phase; ingestion job path for RC-10; cache + deletion lifecycle tests (§9).

**Negative tests:** fabricated citation id; citation to an irrelevant-but-real chunk; missing page metadata; empty document; missing ingestion dependency; contradictory passages; nondeterministic repeat.

**Regression:** the 25 scenarios in §7-mapping below (mandatory), all existing 203 unit tests, and the 9 original live anchors.

**Mandatory regression scenarios (§7 of the role spec) mapped:**

| # | Scenario | Primary RC / test type |
|---|---|---|
| 1 | Exact factual | RC-2 (retrieval eval + live) |
| 2 | Paraphrased factual | RC-7B (claim support) |
| 3 | Misspelled entity | RC-8, RC-3 |
| 4 | Joined/missing spaces | RC-8 |
| 5 | Rare technical term | RC-3 |
| 6 | Numerical | RC-4, RC-5 |
| 7 | Multi-page answer | RC-4 |
| 8 | Multi-document comparison | RC-1, RC-4 |
| 9 | Multi-step procedural | RC-4, RC-5 |
| 10 | List-based answer | RC-5 |
| 11 | Partially supported answer | RC-7B |
| 12 | Unsupported question | RC-6 (sufficiency) |
| 13 | Contradictory passages | RC-6 |
| 14 | Valid citation → irrelevant content | RC-7A/B |
| 15 | Missing page metadata | RC-7A |
| 16 | Invalid/fabricated citation | RC-7A (negative) |
| 17 | Exact extractive answer | RC-11 |
| 18 | Non-contiguous extractive passages | RC-11 |
| 19 | Empty document | ingestion |
| 20 | Scanned PDF | ingestion/OCR |
| 21 | Missing optional ingestion dependency | RC-10 |
| 22 | Large-document ingestion | ingestion |
| 23 | Repeated question | cache |
| 24 | Concurrent requests | queue (1 at a time) |
| 25 | Cache invalidation after deletion | lifecycle |

---

## 9. Evaluation dataset expansion strategy

**Current (`questions.json`, frozen, do not modify):** 70 questions — 65 answerable, 5 unanswerable (J01–J05). Categories: A 10 · B 8 · C 6 · D 6 · E 6 · F 6 · G 5 · H 5 · I 5 · J 5 · K 8.

**Expansion (new files only; originals preserved):**
- Evaluate **all 65 answerable** questions live (not a 9-question subset).
- **Expand the unanswerable set** with new genuinely-unsupported questions (target +10–15) spanning: entity absent from the corpus; plausible-but-absent fact about a present entity; cross-document contradiction with no resolution; out-of-scope/future question; question containing a misleading term.
- **Retain every existing regression question** (the 9 original live anchors + all 70).
- **3 repetitions** on a fixed representative subset (~15 cases spanning all 11 categories) to detect nondeterminism.
- Report **category-level and aggregate** results, separating correctness, completeness, citation accuracy, and abstention accuracy.
- Flag **nondeterministic outcomes** (status or answer changes across the 3 reps).
- New expected answers read directly from `data/extracted/*.jsonl`; no synthetic labels; the existing dataset and its expected answers are never silently changed.

Proposed new artifacts: `evaluation/audit/questions_expanded.json`, `evidence_requirements_expanded.json`, `retrieval_results_expanded.json`, `live_results/` per-run JSON. Baseline artifacts stay as-is.

---

## 10. Metric definitions and formulas

All metrics use the **same frozen dataset, same model config, same formula** for before/after. Different datasets/configs/formulas are never compared.

**Relevance predicate** (per `retrieval_eval.py`): `rel(h,q)=1` iff `doc_key(h.filename) ∈ allowed_docs(q)` **and** `str(h.source_location) ∈ q.pages`. `ranked` = post-rerank order; `selected` = post-selection hits.

**Retrieval (over answerable set Q_A):**
- `Recall@k = (1/|Q_A|) Σ_q 1[ ∃ h ∈ ranked_q[:k] : rel(h,q) ]`
- `MRR = (1/|Q_A|) Σ_q 1 / rank_q(first relevant)` (0 if none)
- `Selected-evidence recall = (1/|Q_A|) Σ_q 1[ ∃ h ∈ selected_q : rel(h,q) ]`
- `Selected-evidence precision = (1/|Q_A|) Σ_q |{h∈selected_q:rel}| / |selected_q|`
- `Page coverage = (1/|Q_A|) Σ_q |pages_found_q ∩ q.pages| / |q.pages|`
- `Term coverage = (1/|Q_A|) Σ_q |terms_present_q| / |q.evidence_terms|`
- **NEW — Evidence-requirement coverage** (from `evidence_requirements.json`): `(1/|Q_A|) Σ_q (satisfied_groups_q / total_groups_q)`; a group is satisfied if any of its alternatives appears in the union of selected text.

**Answer quality (live, adjudicated where noted):**
- `Factual correctness` = adjudicated rate (0/0.5/1 per case).
- `Answer completeness` = (requirement groups present in the answer) / (total groups).
- `Citation integrity` = fraction of citations where label ∈ `[1,|sources|]` **and** label resolves to a retrieved record **and** cited page metadata == actual source record.
- `Citation support accuracy` = adjudicated: supported claims / cited claims.
- `Citation page accuracy` = `|cited_pages ∩ expected_pages| / |expected_pages|`.
- `Abstention accuracy` = abstained / labelled-unanswerable.
- `False-abstention rate` = abstained / labelled-answerable (measured separately).
- `Unsupported-claim rate` = unsupported claims / total claims.

**Performance:**
- `P50/P90/P95` latency — **P90/P95 only reported where n supports it (state n)**; ≥20 samples per percentile band. Never claim reliable P95 from 9 observations.
- Stage latency: preprocessing, embedding, dense, FTS5, fusion, rerank, selection, prompt construction, generation (total **and** time-to-first-token), verification, citation validation, recovery attempts.
- `Generation calls/question`, `Verification calls/question`.
- Memory: application working set, Ollama working set, total system, available system. `CPU utilization`. `Prompt tokens`, `Generated tokens`.

---

## 11. Before/after acceptance criteria (expanded dataset — no "6/9")

All thresholds are **proposed targets** to be confirmed with the user; none is measured yet.

| Dimension | Baseline | Proposed acceptance |
|---|---|---|
| Recall@1 / @4 / @8 / @20 | 0.215 / 0.477 / 0.585 / 0.662 | @1 ≥ 0.35 · @4 ≥ 0.70 · @8 ≥ 0.80 · @20 ≥ 0.85 |
| MRR | 0.348 | ≥ 0.45 |
| Selected-evidence recall | 0.538 | ≥ 0.75 |
| Selected-evidence precision | 0.149 | not worse (+ improvement where achievable) |
| Page coverage | 0.423 | ≥ 0.70 |
| Evidence-requirement coverage | (unmeasured) | ≥ 0.75 |
| Factual correctness (all 65 answerable) | 1/9 live | ≥ 0.80 aggregate |
| Answer completeness | 2/9 partial | ≥ 0.75 aggregate |
| Citation integrity | high (structural) | 100% of returned citations valid |
| Citation support accuracy | (unmeasured) | ≥ 0.85 |
| Abstention accuracy (expanded unanswerable) | 0/1 | ≥ 0.90 |
| False-abstention rate (answerable) | (unmeasured) | ≤ 0.05 |
| Unsupported-claim rate | (unmeasured) | ≤ 0.10 |
| Generation calls/question | up to 3 | ≤ 1.2 mean |
| Latency | P50 31.7 s | P50 materially lower; **no regression** in correctness |
| Memory/CPU | (unmeasured) | within 8 GB; recorded before/after |

Category-level thresholds are set per category in Phase 7 with the same dataset and formula. Any metric not measurable at baseline is measured first (pre-change) to establish the reference.

---

## 12. Performance measurement methodology

- **Runtime pinning:** every run asserts `health.ollama.generation_model == "qwen2.5:3b"` and `verification_model` per config, and the harness rejects a run whose observed model differs.
- **Isolation:** fresh process per config; models unloaded between configs (as in `run_configs.ps1`); caches cleared.
- **Instrumentation to add (test build only):** per-stage timers already emitted in `retrieval`; add prompt/generation token counts and `load_duration` from Ollama responses; sample CPU and the three memory figures around each request; record TTFT when streaming is tested.
- **Percentiles:** computed with the existing `percentile()` helper; report n; do not present P90/P95 for n<20.
- **Repetition:** 3 reps on a fixed subset for nondeterminism; single pass over all 65 answerable.
- **Fixed comparison contract:** baseline vs after use identical dataset, config, and formula; results recorded in new artifacts.

---

## 13. Hardware and RAM risk analysis (8 GB, CPU, offline)

- Generation is the dominant cost (~19.8–68.7 s/case at 3B). Optimisation must not add model calls.
- Context is capped at `model_context_tokens=4096` with `context_safety_tokens=900`; the spec's initial target is **2048** — testing larger only when measured memory permits.
- Sequential single inference (`inference_concurrency=1`) must be preserved; concurrent-request test (#24) verifies the queue, not parallelism.
- Risks: larger context or streaming buffers raising peak RSS; loading a second model. Mitigation: one model at a time, unload between configs, measure application + Ollama + system memory.
- No hardware upgrades proposed; no larger models introduced.

---

## 14. Security implications

- Verify loopback binding (`127.0.0.1`) and no external calls; **do not claim air-gap compliance unless network activity is measured**.
- Path traversal / unsafe file handling / archive limits on ingestion (RC-10 path).
- Prompt injection from uploaded documents (evidence is interpolated verbatim) — low risk single-user, but the citation-integrity check (RC-7A) prevents fabricated identifiers from being trusted.
- SQLite query construction (FTS5 `MATCH` uses quoted terms — verify no injection via term quoting).
- ChromaDB metadata integrity; cache isolation; resource exhaustion; dependency security.
- New retrieval/sufficiency code must not introduce any network access.

---

## 15. Rollback strategy

- Isolated branch `fix/rag-answer-reliability-2` from `5787283`; `main` untouched.
- One commit per RC ⇒ `git revert <sha>` per fix.
- Every behavioural change behind a config flag defaulting to **current behaviour**; reverting behaviour needs no code change.
- No index rebuild, no migration, no `.env` change; existing reports never overwritten.
- If a phase fails its acceptance criteria, revert that phase before proceeding.

---

## 16. Expected benefits and limitations

**Expected benefits:** better recall (esp. rare-term and comparison questions), fewer false refusals with intact citation integrity, real abstention on unanswerable questions, more complete answers, meaning-preserving query cleanup, accurate format support, lower tail latency, and a measurement framework that separates the four evidence concepts.

**Limitations:** 3B on CPU remains slow; some quality gains depend on hypotheses H1–H5; live evaluation of 65 questions × reps is costly in wall-clock time; external validity is limited to the five-document corpus; no claim of improvement is made until measured on the frozen dataset.

---

## 17. Dependencies between engineering changes

- RC-3 → RC-2 → RC-4 (candidate quality → ranking → selection).
- RC-1 independent (Phase 2; parallelisable with Phase 1).
- RC-7 (A+B) → RC-6 (D): citation semantics stabilise before sufficiency gating.
- RC-5 (C) depends on Phase 1–3.
- RC-11, RC-12 depend on the RC-6/RC-7 accept/reject path.
- RC-9 last; re-measures all P0–P1 metrics.
- All Phase-1 changes share one frozen retrieval harness and dataset.

---

## 18. Explicit regression risks

| RC | Risk of regression | Mitigation |
|---|---|---|
| RC-1 | Cross-document bleed for single-doc factual questions | Keep scoping for single-doc non-comparison questions |
| RC-2 | Penalising legitimate preface content | Bounded penalty; per-category eval |
| RC-3 | Over-broad recall / query syntax errors | Candidate limits; strict-match fallback; syntax-error guard |
| RC-4 | Losing needed single-document context | Adaptive policy; single-doc regression set |
| RC-5 | Extra latency / hallucinated completeness | One bounded pass; verify items against evidence |
| RC-6 | Over-abstention | Conservative thresholds; answerable-set measurement |
| RC-7 | Weakly grounded sentences accepted | Remove/qualify unsupported; keep entailment for paraphrases; never weaken validation |
| RC-8 | Under-correction | Original query always retained |
| RC-9 | Accuracy traded for speed | Correctness gates precede optimisation; re-measure |
| RC-10 | Removing a truly supported format | Verify dependency availability before restricting |
| RC-11 | Missing a valid extract | Preserve strict exact-extract path |
| RC-12 | Missing a needed recovery | Recovery only on conclusive verdicts |

---

## Approval gate

This document is the revised implementation plan. **No files have been edited, no dependencies changed, no migrations run, no indexes rebuilt, no production documents or `.env` modified, and the Git branch is unchanged.** Per the role spec §12, work stops here pending explicit approval. After approval, changes proceed in the Phase 1→7 order, in isolated independently-testable commits, with per-phase tests, diff review, regression checks, and metric comparison against the frozen baseline.
