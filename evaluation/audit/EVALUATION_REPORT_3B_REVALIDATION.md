# Offline Document AI (RND2) — Re-Validation Report (3B Configuration)

**Date:** 13 November 2026
**Configuration under test:** `GENERATION_MODEL=qwen2.5:3b`, verification = programmatic (role-file mandate: 3B generation, no separate larger verifier model).
**Application root:** `C:\Users\Admin\Downloads\RND2_2\RND2_1\RND`
**Method:** read-only audit + black-box measurement of the running application. No application file, index, or uploaded document was modified. New artifacts live under `evaluation/audit/`.
**Relationship to prior report:** this re-validates `evaluation/audit/EVALUATION_REPORT.md` against the current code and the mandated 3B configuration. Retrieval metrics reproduce the prior report exactly, which confirms the code is unchanged since the prior evaluation.

---

## A. Executive summary

**Verdict: NOT production-ready for accuracy-critical document Q&A.**

The architecture is sound and engineering hygiene is good (203 unit tests pass in 16.9 s). But, measured on the real 5-document corpus with the mandated 3B model:

- Retrieval is the primary failure domain: only **54%** of answerable questions have a relevant selected chunk, and the top-ranked result is relevant only **21.5%** of the time.
- End-to-end, only **1 of 9** live cases produced a clearly correct, complete answer.
- Median response time is **31.7 s** (target < 20 s); the model must generate for **20–70 s** per question on this CPU. Latency is *generation-bound*, not retrieval-bound (retrieval + rank + select < 2 s).
- The application still **hallucinates on unanswerable questions** (J01 answered a year that is not in the source) and **refuses answerable comparison questions** (D01 failed citation validation).

### Critical blockers
1. **RC-1 — Multi-document comparison is broken** by `_named_document_scope` collapsing the corpus to one file (D01).
2. **RC-2 — Retrieval prefers commentary/back-matter over content.** The correct passage for A02 exists in the PDF (Page 7) but is never retrieved; rank #1 is a decoy from Page 4.
3. **RC-6 — No working abstention gate.** Early abstention is dead code; the model answers unanswerable questions with fabricated facts (J01).
4. **RC-9 — Latency** is ~20–70 s/question with 3B on CPU; the 10–20 s target is not met by the current design.

---

## B. Test execution summary

| Item | Count |
|---|---|
| Questions generated from the corpus (`questions.json`) | 70 |
| Answerable questions | 65 |
| Unanswerable questions (abstention cases) | 5 |
| Retrieval cases executed (production components, no LLM) | 70 |
| Live end-to-end cases executed (`POST /chat`) | 9 |
| Live cases: clearly correct | 1 (H05) |
| Live cases: partially correct / incomplete | 2 (B02, F01) |
| Live cases: incorrect / wrong-source | 4 (A01, A02, I01, K02) |
| Live cases: refused (citation-gate failure) | 1 (D01) |
| Live cases: hallucinated (unanswerable answered) | 1 (J01) |
| Unit test suite | 203 passed, 1 skipped (16.9 s) |
| Skipped / not executed | PPTX & RTF ingestion (missing deps); scanned-PDF OCR on a real scan; 200 MB / batch limits; air-gap packet capture |

---

## C. Accuracy report

### Retrieval (65 answerable questions, production pipeline order)

| Metric | Value | Target |
|---|---|---|
| Recall@1 | 0.215 | — |
| Recall@3 | 0.431 | — |
| **Recall@4** | **0.477** | ≥ 0.90 |
| Recall@8 | 0.585 | — |
| Recall@20 | 0.662 | — |
| Recall (selected evidence) | 0.538 | ≥ 0.90 |
| **MRR** | **0.348** | — |
| Precision@selected | 0.149 | — |
| Page coverage (selected) | 0.423 | ≥ 0.90 |
| Term coverage (selected) | 0.683 | — |
| Fetch→gate emptiness | 0 of 65 | 0 |

`recall_selected` by category: A basic 0.700 · B detailed 0.625 · C multi-hop 0.667 · **D comparison 0.167** · E multi-document 0.500 · F summarization 0.667 · G scenario 0.400 · H numerical 0.800 · I ambiguous 0.400 · **K spelling 0.375**.

### Live answer quality (9 cases, adjudicated against source)

| ID | Category | Outcome | Case (Phase 5) |
|---|---|---|---|
| A01 | basic | "Junto met on evenings" — dropped **"Friday"** | Case 5 (incomplete) |
| A02 | basic | "not present in the provided evidence" — but it **is** present (Page 7) | Case 1 (retrieval) |
| B02 | detailed | listed some of the six steps; page 70 missing | Case 5 (incomplete) |
| D01 | comparison | refused: "did not cite them correctly" | Case 3 (over-strict gate) + Case 1 (scoping) |
| F01 | summarization | reasonable summary of the right document | Pass (partial) |
| H05 | numerical | "1897 … 27 chapters" | **Correct** |
| I01 | ambiguous | "information about 'him' is not … in the documents" | Case 1/2 (resolution/retrieval) |
| J01 | unanswerable | answered "1729" with citation `[4]` | **Case 4 (hallucination)** |
| K02 | spelling | Franklin description (off-target) | Case 1 (retrieval) |

- Faithfulness: 1 clear hallucination of 9 live cases (J01).
- Abstention accuracy: **0 of 1** (J01 should have abstained).
- Citation syntax validity: high (structural validator passes/fails cleanly); citation **page accuracy** is low (only 3 of 9 live cases had a matching source page).

---

## D. Performance report

**Latency is generation-bound.** Every stage other than generation is < 2 s.

| Stage | Measured |
|---|---|
| Retrieval (embed + search + gate) | 0.34 – 1.86 s |
| Ranking | 0.0003 – 0.038 s |
| Evidence selection | 0.002 – 1.23 s |
| **Generation (3B, CPU)** | **19.8 – 68.7 s** |
| Programmatic verification | not a separate LLM call (by mandate) |
| **Total end-to-end** | **mean 40.5 s · P50 31.7 s · max 70.6 s** |
| Requests > 10 s | 9 / 9 |
| Requests > 20 s | 9 / 9 |
| Timeout rate | 0 |

The 10–20 s target is **not achievable** on this hardware with a 3B generative model answering multi-sentence questions. Realistic options are (a) evidence-first extractive answers for factual/quantity/number questions, (b) hard token caps + shorter prompts, (c) streaming so first token appears earlier. Correctness must not be traded for latency.

---

## E. Failure analysis (root causes with evidence)

**RC-1 — Named-document scoping breaks comparisons.** `_named_document_scope` ([chat.py:697-721](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/rag/chat.py#L697-L721)) returns a single document when only one *filename* token appears in the question. For D01 ("Compare Franklin's Junto with Hill's Master Mind") only `franklin` matched a title token, so scope = {Franklin}; Hill evidence was discarded (`selected_chunks=2`, context 458 chars) and the answer failed citation validation. **Case 1 + Case 3.**

**RC-2 — Similarity prefers commentary/back-matter over content.** For A02 the correct passage is on **Page 7** (verified in `data/extracted/*.jsonl`: *"At ten years old I was taken home to assist my father…"*), yet ranked hits were Pages 4/105/109/107 — including the decoy *"Benjamin was bred a silk dyer"* (Page 4, lexical 1.0, sim 0.772). Same pattern in I01, K02 (top hits are "about the author" commentary) and H05 (rank #1 = "ABOUT THIS EDITION"). Driver: dense window + reranker scoring in [store.py](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/retrieval/store.py) and [reranker.py](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/retrieval/reranker.py). **Case 1.**

**RC-3 — FTS/lexical path.** `_fts_query` OR-joins terms, so a rare term is crowded out by common ones; FTS-only hits are scored `0.35 × coverage` against `MIN_SIMILARITY=0.25`, requiring ≥71% term match to survive ([store.py](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/retrieval/store.py)). **Case 1.**

**RC-4 — No document diversity in evidence selection.** [evidence_selector.py](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/retrieval/evidence_selector.py) greedily picks by relevance/coverage/redundancy with no per-document cap, so one document can fill the context (seen in F01: 7 chunks). **Case 1.**

**RC-5 — Generation incompleteness.** A01 retrieved the correct page (Page 37, relevant) yet dropped "Friday"; B02 missed part of a 6-item list. Prompt/answer-planning weakness in [chat.py](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/rag/chat.py) + [prompts.py](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/rag/prompts.py). **Case 5.**

**RC-6 — Abstention is dead code; unanswerable questions are answered.** `is_question_bank_query = False` and `lacks_document_anchor = False` are hardcoded ([chat.py:762-783](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/rag/chat.py#L762-L783)), so `abstain_early` is always `False`. J01 returned a fabricated year (1729) citing a passage that never states the arrival year. **Case 4.**

**RC-7 — Over-strict citation gate.** `validate_citation_references` requires `coverage == 1.0` (`fully_cited`) ([citations.py](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/rag/citations.py)); a correct-but-partially-cited comparison answer is rejected wholesale (D01). **Case 3.**

**RC-8 — Spelling correction is partial and can corrupt meaning.** `correct_query_from_vocabulary(cutoff=0.80)` ([planner.py](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/rag/planner.py)) fixes "responsibilites→responsibilities" but (a) does not split joined words ("benjaminfranklin" unchanged), (b) leaves "wht/abot/hpw" uncorrected, and (c) confidently corrupts "**feers → fees**" (K08). The original query is retained as a fallback, which limits damage. **Case 1.**

**RC-9 — Latency (generation-bound).** See §D.

**RC-10 — Declared-but-missing dependencies.** `python-pptx` and `striprtf` are declared in `requirements.txt` but not installed, so `.pptx` and `.rtf` uploads fail at extraction ([extractor.py](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/ingestion/extractor.py)). **Ingestion failure.**

**RC-11 — Extractive fallback risk.** `_extractive_answer_for_question` accepts sentences with ≥50% question-term overlap and can be promoted via `EXACT_PASSAGE_MATCH`, bypassing the verifier ([chat.py:395-431](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/rag/chat.py#L395-L431), [433](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/rag/chat.py#L433)). Risk of non-answers being presented as answers.

**RC-12 — Unbounded per-question cost.** `for attempt in (1, 2)` ([chat.py:1777](file:///c:/Users/Admin/Downloads/RND2_2/RND2_1/RND/app/rag/chat.py#L1777)) permits up to 3 generations (+2 verifications when enabled) for one question.

---

## F. Architecture review

- **Pipeline:** upload → job queue (ThreadPool, max 2) → extractor (26 formats) → classify → chunk (1200/180) → `nomic-embed-text` → Chroma (dense) + SQLite FTS5 (lexical) → hybrid + RRF → deterministic rerank → evidence select → qwen2.5 generate → citation validate → verify → answer. Coherent and well-factored.
- **Duplicated work:** retrieval is executed up to 4 passes (original + corrected + variants) per question; ranking/selection re-score the same candidates.
- **Dead code:** early-abstention branch is unreachable (RC-6); `_has_specific_entities`/`has_document_anchor_in_question` are unused in the decision path.
- **Config hazard:** `app/core/config.py` default `generation_model=qwen2.5:3b` disagrees with the deployed `.env` (`qwen2.5:7b`). The `.env` `TOP_K` comment says "Eight" while the value is 4.
- **Security:** loopback-only Ollama enforced (`validate_loopback_url`); single-user `local` owner; no remote calls. No obvious injection path; prompts interpolate retrieved text verbatim (prompt-injection surface exists in principle but is low-risk in a single-user offline product).

---

## G. Improvement roadmap

**P0 — critical correctness / integrity**
1. Stop the D01 class of failure: make `_named_document_scope` skip scoping when the question is a comparison or names fewer than all sources (RC-1).
2. Restore a real abstention path for unanswerable questions; gate answer generation on evidence sufficiency (RC-6).
3. Fix retrieval so the true content passage outranks commentary/back-matter (RC-2, RC-3).

**P1 — major quality / reliability**
4. Relax the citation gate from "100% coverage" to "every sentence cited OR explicitly marked inferred" (RC-7).
5. Add per-document diversity to evidence selection (RC-4).
6. Harden spelling correction to prevent semantic corruption and handle joined words (RC-8).
7. Install `python-pptx` and `striprtf` (or stop declaring `.pptx`/`.rtf` as supported) (RC-10).
8. Constrain the extractive fallback and the verification loop (RC-11, RC-12).

**P2 — performance**
9. Make answers evidence-first/extractive for factual, numerical and list questions; cap `ANSWER_MAX_TOKENS`; consider streaming (RC-9).

**P3 — enhancements**
10. Remove dead code; align `config.py`/`.env`; fix stale `TOP_K` comment (RC-12).

---

## Appendix — artifacts produced this run

`retrieval_eval_2.log`, `retrieval_results.json` (regenerated), `live_cfgA_3b_programmatic.json`, `live_cfgA_3b_programmatic_rerun.log`, `format_probe.json`.
