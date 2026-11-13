# Offline Document AI (RND2) — Independent Production-Readiness Evaluation

**Date:** 13 November 2026
**Evaluator:** read-only audit + black-box measurement of the running application
**Application root:** `C:\Users\Admin\Downloads\RND2_2\RND2_1\RND`
**Status of application code/config during this evaluation:** unchanged. No application file, index, or uploaded document was modified. All new files live under `evaluation/audit/`.

---

## 1. Verdict

**Not production-ready for accuracy-critical document Q&A. Do not hand this build to a client as a system whose answers can be trusted.**

Confirmed by measurement, not inspection:

| Dimension | Measured result | Target | Verdict |
|---|---|---|---|
| Correct, complete answers (best config) | **1 of 9** end-to-end cases | ≥ 8 of 10 | Fail |
| Evidence actually sufficient after retrieval | **38 of 65** answerable questions (58.5%) | ≥ 90% | Fail |
| Citation *syntax* validity | 100% of returned answers on the live subset | 100% | Pass |
| Citation *page* accuracy | 0–50% per case; 6 of 9 cases cited no expected page | ≥ 90% | Fail |
| Hallucination | **2 of 9** answers in the unverified config asserted unsupported facts | 0 | Fail |
| Latency (3B generation) | mean **45.6 s**, p90 68.9 s, max 77.0 s | < 10 s preferred, < 20 s acceptable | Fail |
| Latency (3B + 3B verifier) | mean **110.9 s**, max 185.8 s | < 20 s | Fail |
| Latency (3B + 7B verifier) | mean **370.2 s**, p90 518.6 s, max **755.5 s** | < 20 s | Fail |
| Test suite | 203 passed, 1 skipped (131.65 s) | green | Pass — but see §12 |

The unit-suite green light is real but does not indicate end-to-end quality: the repository's own `FEATURE_COVERAGE.md` already labels dense retrieval, fusion, abstention, multi-document reasoning, citations, and verification as *partial*, and states "Do not interpret a green unit-test suite as proof of end-to-end RAG correctness." This evaluation confirms that warning with numbers.

**One sentence:** the architecture is sound and the engineering hygiene is unusually good, but retrieval quality and answer acceptance are not yet at a deliverable standard, and the current default configuration (7B generation *and* 7B verification) is 20–50× slower than the acceptable latency budget on the target hardware.

---

## 2. Scope, method, and what was *not* tested

### 2.1 Method
1. **Static audit** of the real request path: `app/rag/chat.py`, `app/retrieval/*`, `app/rag/{citations,verification,planner}.py`, `app/storage/database.py`, `app/ingestion/*`, `app/services/*`, `app/cache/*`, `app/main.py`, UI.
2. **Corpus-grounded question set**: 70 questions built only from text actually present in the five uploaded documents, with page-level evidence read from `data/extracted/*.jsonl`.
3. **Retrieval measurement** driving the production components in production order (normalize → vocabulary correction → hybrid search → similarity gate → rank → evidence select), 70 cases.
4. **Evidence adjudication**: a separate source-derived checklist per question, so "wrong page" is not confused with "wrong fact".
5. **Live end-to-end runs** through `POST /chat` with stage timings and memory capture: 3 model configurations × 9 fixed questions.
6. **Isolated verifier benchmark**: identical claim/evidence pairs, 3B vs 7B, production verifier prompt.
7. **Format extraction probe**: per-format extraction in a temp directory (no index writes).

### 2.2 Not tested (and therefore not claimed)
- Non-PDF **ingestion end-to-end** (DOCX/XLSX/PPTX/CSV/HTML/OCR were exercised at the extraction layer only; the queue → chunk → embed → index path was not run with those formats to avoid writing to the production index).
- **OCR accuracy** on real scans (binary and `eng`/`osd` language data exist; no scanned document is in the corpus).
- **200 MB / batch / 5000-page** workload limits, disk pressure, restart-under-load.
- **Air-gapped proof** (no packet capture was performed during this evaluation).
- **UI interaction session** (UI code was read; the browser flow was not driven).
- The shipped **7B generation + 7B verification** configuration on the full set (measured on a 2-question sample only, because each request costs 6–9 minutes; see §8.4).

---

## 3. Environment and configuration as shipped

### 3.1 Hardware (measured)
- CPU: Intel Core i3-8100, 4 cores / 4 threads (no AVX-512, no usable GPU — Intel UHD 630 only)
- RAM: **7.8 GB total**
- Ollama runs **CPU-only** (`size_vram: 0` for every model observed)

This matters: every latency number below is CPU-bound and cannot be extrapolated to a GPU machine, but it *is* the machine the client will run on unless specified otherwise.

### 3.2 Installed models
| Model | Size on disk | Observed resident RAM |
|---|---|---|
| `qwen2.5:7b` | 4,466 MB | 4,835 MB |
| `qwen2.5:3b` | 1,841 MB | 2,068 MB |
| `nomic-embed-text` | 262 MB | 371 MB |
| `qwen3:30b` | 17,697 MB | not loadable in 7.8 GB |

### 3.3 Configuration discrepancy (must be resolved before delivery)
`.env` as shipped:

```
GENERATION_MODEL=qwen2.5:7b
VERIFICATION_MODEL=qwen2.5:7b
```

but `app/core/config.py` defaults and all project documentation (`FEATURE_COVERAGE.md`, README) state **3B generation** with a stronger verifier. The brief also assumes 3B generation. The deployed configuration is therefore:
- **~3–4× slower** than the documented one (7B generation dominates),
- **not** the configuration that was benchmarked or documented by the vendor,
- still labelled in the UI sidebar as "Answer model: qwen2.5:7b".

**Action:** decide the intended default explicitly; `.env` currently silently overrides the documented design.

### 3.4 Retrieval knobs
`TOP_K=4`, `RETRIEVAL_CANDIDATE_MULTIPLIER=5` (→ 20 dense candidates), `HYBRID_LEXICAL_WEIGHT=0.35`, `LEXICAL_CANDIDATE_LIMIT=40`, `MIN_SIMILARITY=0.25`, `MAX_CONTEXT_CHARS=16000`, `MODEL_CONTEXT_TOKENS=4096`, `VERIFY_ANSWERS=true`.

### 3.5 Corpus integrity
5 documents, all `indexed`/`complete`; 2,725 FTS rows and 2,725 vectors (1:1, no orphans):

| Document | Chunks | Words | Pages |
|---|---|---|---|
| Dracula (Bram Stoker) | 1,180 | 163,708 | 1–727 |
| The Autobiography of Benjamin Franklin | 569 | 78,228 | 1–128 |
| Think and Grow Rich (Napoleon Hill) | 745 | 95,682 | 1–495 |
| The Science of Getting Rich (Wallace Wattles) | 154 | 22,978 | 1–75 |
| The Enchiridion (Epictetus) | 77 | 10,155 | 1–73 |

---

## 4. Retrieval results (70 questions, production components)

### 4.1 Headline metrics (65 answerable, page-level ground truth)

| Metric | Value |
|---|---|
| Recall@1 | 0.215 |
| Recall@3 | 0.431 |
| Recall@4 | 0.477 |
| Recall@8 | 0.585 |
| Recall@20 | 0.662 |
| MRR | 0.348 |
| recall_selected (evidence contains a correct page) | 0.538 |
| precision_selected | 0.149 |
| expected-page coverage of selected evidence | 0.423 |
| query-term coverage of selected evidence | 0.683 |
| cases with **no** evidence after the similarity gate | 0 |

Only **47.7%** of questions have a correct passage inside the four chunks that `TOP_K=4` allows, and the selector only reaches **53.8%**. Precision 0.149 means roughly one of every seven selected chunks is on a page that answers the question — the rest is context noise the generator must ignore.

### 4.2 By category (share of questions with sufficient evidence selected)

| Category | OK | n | Note |
|---|---|---|---|
| A basic factual | 8 | 10 | Works when the answer sentence is short and unique |
| B detailed | 6 | 8 | Multi-sentence answers usually lose one element |
| C multi-hop | 4 | 6 | Needs two passages; usually gets one |
| D comparison | **2** | 6 | Worst structural failure (§6.4) |
| E multi-document | **2** | 6 | Second document often absent |
| F summarization | **0** | 6 | No summary case had complete evidence |
| G scenario | 2 | 5 | Scenario wording drifts away from source vocabulary |
| H numerical | **5** | 5 | Exact-number questions are the strength |
| I ambiguous/conversational | 2 | 5 | Follow-up resolution is unreliable |
| K spelling | 7 | 8 | Correction helps 4, corrupts 2, one total miss |

### 4.3 Query-expansion cost
43 of 70 questions triggered variant expansion; mean **2.91** retrieval passes (max 5), each a separate embedding + vector + FTS search. Retrieval alone costs mean 0.97 s (p90 2.00 s, max 3.97 s) once warm — small next to generation, but it triples CPU work for no measured gain in the failing categories.

---

## 5. Evidence sufficiency and failure classification

Page-level recall understates quality (a correct fact can appear on a different page), so each case was also adjudicated against a source-derived checklist of required evidence groups.

| Outcome | Count | Share |
|---|---|---|
| Fully sufficient evidence selected | 38 | 58.5% |
| Partially sufficient | 16 | 24.6% |
| Insufficient | 11 | 16.9% |

Mapped to the requested failure classes (retrieval stage):

| Class | Count | Share | Meaning |
|---|---|---|---|
| Case 1 — not retrieved (absent from top-20) | 15 | 23.1% | The answering passage never entered the candidate pool |
| Case 2 — retrieved but not selected | 5 | 7.7% | It was ranked, the selector dropped it |
| Case 5 — incomplete | 7 | 10.8% | Some required facts present, others missing |
| OK | 38 | 58.5% | Evidence sufficient before generation |

Case 3 (retrieved, then abstained) and Case 4 (wrong answer) are generation-stage classes and are scored in §7.

---

## 6. Root causes (exact code locations, with evidence)

### 6.1 FTS OR-query + 40-candidate cap crowds out the rare term — **Case 1**
`app/retrieval/store.py::_fts_query` builds `"term1" OR "term2" …`, and `app/storage/database.py::search_indexed_chunks` returns `bm25 ORDER BY rank LIMIT 40` (`LEXICAL_CANDIDATE_LIMIT`). A query containing one common word and one rare word is scored by bm25 across the OR-set, so chunks matching only the common word fill the 40 slots.

Evidence: for *"Summarize Franklin's account of the Junto."* the FTS query is `"account" OR "franklin" OR "junto" OR "summarize"`; page 37 — the only page that describes the Junto — does **not** appear in the top 45 rows, while generic "Franklin" pages do. Page 37 *does* match `"junto"` alone (rank 10 of 13). The rare term is present in the index and is being outvoted.

### 6.2 FTS-only hits are capped below the gate — **Case 1/2**
`store.py::_lexical_similarity` gives lexical-only hits `0.35 × lexical_coverage`, then compares to `MIN_SIMILARITY=0.25`. A lexical hit therefore needs **≥ 71.4%** of the query's terms to survive. Two verified examples:
- D03 (competitive vs creative plane, p25): 8 query terms, 5 matched → score 0.2188 → discarded, although the page contains the answer.
- D01 (Master Mind definition, p333): 6 terms, 2 matched → 0.117 → discarded.

A long, specific question — exactly the kind that needs lexical help — is the kind most likely to be filtered out.

### 6.3 Dense candidate window is too narrow — **Case 1**
`chat.py` computes `candidate_k = top_k × retrieval_candidate_multiplier` = **20** chunks per query across 3,022 chunks / 5 books. For entity or definition questions the embedding neighbourhood is dominated by scenes that mention the entity, not passages that define it: A05 (Van Helsing) never retrieved p218; B06 (Franklin's schooling) never retrieved p6–8; F04 (Junto summary) never retrieved p37.

### 6.4 Evidence selection has no document or entity diversity — **Case 2**
`app/retrieval/evidence_selector.py` greedily scores `0.78×base + 0.27×coverage_gain − 0.20×redundancy` with no per-document or per-entity constraint. Comparison questions therefore fill the window from one book: D02 selected 7 Hill pages about faith and **zero** Wattles pages about gratitude. D05, E03, F03, I05 fail the same way.

### 6.5 Named-document scoping silently deletes one side of a comparison — **Case 1, severe**
`chat.py::_named_document_scope` narrows candidates to any document named in the question. For *"Compare Franklin's Junto with Hill's Master Mind"* it keys on "Franklin" and restricts retrieval to the autobiography. Live evidence: `document_scope=['the-autobiography-of-benjamin-franklin…']`, only **2** eligible chunks, both Franklin, and the answer fails with `citation_validation_failed`. A comparison question can never succeed under this rule.

### 6.6 Titles, tables of contents and boilerplate win similarity — **Case 2**
B07 selected page 367 (the running header "Think and Grow Rich", similarity 0.946) instead of p41–42. E05 selected Hill "Think…" pages instead of Science p74. H03 selected the p7 table of contents. High-similarity front/back matter is not filtered by chunk role.

### 6.7 Query variants degrade into fragments
`app/rag/planner.py::query_variants` adds `" ".join(content[-2:])` and a compact variant built from `re.findall(r"[\w.-]+")`, which splits possessives. Live variants for D01 were:
```
"Master Mind."
"Compare Franklin s Junto Hill s Master Mind."
```
The first is a two-word fragment; the second injects stray `s` tokens into FTS and the embedding. Both consume full retrieval passes.

### 6.8 Spelling correction is confidently wrong on real typos — **Case 4 contributor**
`correct_query_from_vocabulary(cutoff=0.80)` rewrites any alphabetic token of length ≥ 5 that is absent from the corpus vocabulary. Measured corrections:

| Case | Original | Corrected | Assessment |
|---|---|---|---|
| K06 | responsibilites | responsibilities | correct |
| K07 | chaptrs / encheeridion | chapters / **encheiridion** | first correct; second invents a non-word |
| K08 | six basik feers | six basik **fees** | **wrong** — "fears" destroyed |
| K03 | explane the contrubutions | **explained** the contributions | **wrong** — imperative becomes past tense |
| K05 | abot the importnt events | abot the important events | half-corrected; retrieval still failed |
| K02 | whaat is benjmin frankln | what is benjamin franklin | correct, yet page-level retrieval still failed |
| F05 | recommends | recommend | harmless |
| I02 | disadvantages | disadvantage | harmless |

The module comments claim ambiguous fuzzy matches are left unchanged; "feers → fees" and "explane → explained" show that guard is not effective. Corrections *helped* four cases (K06–K08, K02) and *damaged* two.

### 6.9 Early abstention is dead code
`chat.py::_analyze_question` hardcodes `is_question_bank_query = False` and `lacks_document_anchor = False`, so `abstain_early` can never be true; `has_general_anchor` and `general_terms` are computed and unused. No retrieval-side abstention exists at all.

### 6.10 The similarity gate does not separate answerable from unanswerable
All five unanswerable questions passed the gate with 20–38 eligible chunks and best similarity **0.577–0.693** — higher than many answerable questions (several answered cases sit at 0.51–0.57). Abstention depends entirely on the generator noticing the absence, which it does unreliably (§7).

### 6.11 The extractive fallback answers with a 50%-word-overlap quote — **Case 4, severe**
`chat.py::_extractive_answer_for_question` accepts any evidence sentence sharing ≥ 50% of the question's content words, and `_is_exact_passage_extract` then marks it `EXACT_PASSAGE_MATCH`, bypassing the verifier entirely.

Live victims (cfgB, all `verified=true`):
- A02 "father's trade" → a quote about **apprenticeship indentures**.
- I01 "Tell me more about him" → *"A little more than thirty years ago, Edwin C [2]."*
- J01 "year Franklin arrived in Philadelphia" → a passage about an **elegy for Aquila Rose**, accepted because it contains "Franklin", "Philadelphia" and "first".

### 6.12 The verifier checks entailment, not responsiveness — **Case 4, severe**
`app/rag/verification.py::verify_claims` asks one yes/no question: is every claim entailed by its own cited passage (plus a completeness clause). A **verbatim but irrelevant** sentence is entailed by definition. Measured outcome: six non-responsive answers passed with `verified=true`, including the "Edwin C" fragment and the apprenticeship quote. Meanwhile a workable draft (A01) was rejected twice and returned `answer_validation_failed`.

The verifier prompt also concatenates **all** claim/evidence pairs into one string with no length bound; with `MODEL_CONTEXT_TOKENS=4096` the tail pairs are silently truncated, so late claims are effectively unverified.

### 6.13 The citation gate requires 100% sentence coverage
`citations.py::validate_citation_references` returns `valid = references_valid and fully_cited`, i.e. *every* factual sentence (≥3 content words) must carry a numeric reference. One uncited sentence fails the answer. This is why D01 returns `citation_validation_failed` rather than a partially cited comparison, and it is the single largest cause of hard failures in live runs.

### 6.14 Status and confidence do not mean what a client will read them to mean
- A refusal ("the requested information … is not present in the provided evidence") is returned as **`status: success`, `abstained: false`**, with `confidence 0.85 / "high"` (cfgA A02, cfgA I01).
- A hallucinated year is returned as **`success`, `abstained: false`, confidence 0.749 / "medium"** (cfgA J01).
- `_confidence_score` blends similarity, term coverage, citation validity, verification and citation coverage — it never measures whether the answer addresses the question. That is why an irrelevant verbatim quote scores **1.0 / "high"**.

### 6.15 Verification is unbounded and runs up to twice
`chat.py` verification loop `for attempt in (1, 2)` performs generation + verification, then a full regeneration + second verification, then an extractive fallback. Worst case: **3 generations + 2 verifications per question**. The verification prompt grows with every claim, which is why measured verification time ranges from 0.5 s to **286 s**.

---

## 7. End-to-end answer accuracy (live, 9 fixed questions)

Configuration A = `qwen2.5:3b`, verification off. Configuration B = `qwen2.5:3b` + `qwen2.5:3b` verification. Configuration C = `qwen2.5:3b` + `qwen2.5:7b` verification. Same questions, same corpus, fresh process and unloaded models per configuration, 9 sequential requests per configuration.

| Case | A: 3B, no verify | B: 3B + 3B verify | C: 3B + 7B verify |
|---|---|---|---|
| A01 Junto founder + evening | Partial — "met on evenings", no Friday | **Failed** (`answer_validation_failed`) | **Failed** (`answer_validation_failed`, 518.6 s) |
| A02 Father's trade | Refusal ("not in evidence") — status success | **Wrong** — apprenticeship quote, `verified=true`, confidence 1.0 | **Wrong** — same apprenticeship quote, `verified=true`, confidence 1.0 |
| B02 Six steps | Steps 2–6, step 1 missing | **Truncated** — "…six steps, viz: First.", `verified=true` | Partial — steps 2–6, step 1 missing, 2/3 pages |
| D01 Junto vs Master Mind | Failed (scoped to Franklin) | Failed (same cause) | Failed (same cause) |
| F01 Science summary | Generic, misses thinking-stuff/gratitude | Identical answer, `verified=true`, confidence 0.9 | Identical answer, `verified=true`, confidence 0.9 |
| H05 Dracula year + chapters | **Correct** (1897, 27) | Partial — year only, `verified=true` | Partial — year only, `verified=true` |
| I01 "Tell me more about him" | Refusal — status success, confidence 0.826 | **Nonsense** — "Edwin C", `verified=true`, confidence 0.976 | **Nonsense** — same fragment, `verified=true`, confidence 0.976 |
| J01 Franklin arrival year | **Hallucinated 1729** — status success | Irrelevant elegy passage, `verified=true` | **Hallucinated 1729** — status success |
| K02 "whaat is benjmin frankln" | **False facts** ("Samuel Franklin") | Tangential Mather quote, `verified=true` | Tangential Mather quote, `verified=true`, confidence 1.0 |

| Metric | A (3B) | B (3B + 3B verify) | C (3B + 7B verify) |
|---|---|---|---|
| `status: success` | 8/9 | 7/9 | 7/9 |
| **Fully correct** | **1/9 (11%)** | **0/9 (0%)** | **0/9 (0%)** |
| Partially correct | 3/9 | 4/9 | 3/9 |
| Non-responsive / wrong | 4/9 | 4/9 | 4/9 |
| Hard failure | 1/9 | 2/9 | 2/9 |
| Unsupported factual assertions | 2/9 | 0/9 (but irrelevant, not correct) | 1/9 |
| Answers passed as `verified=true` that are wrong/non-responsive | n/a | 6/9 | 6/9 |
| Mean latency | 45.6 s | 110.9 s | **370.2 s** |
| Max latency | 77.0 s | 185.8 s | **755.5 s** |

**Reading of this table:** enabling the verifier removed the two clean hallucinations but replaced them with confident non-answers and destroyed one workable draft. Net correctness went **down**, latency went **up 2.4×** (3B verifier) or **8.1×** (7B verifier). That is the opposite of the intended trade.

The mechanism is visible in the diagnostics: **7 of 9 first drafts were rejected by the 3B verifier** (`verification_retry_attempted: true` on A01, A02, B02, H05, I01, J01, K02). Every rejection costs a full second generation. Only F01 was accepted on the first pass, and that accepted answer was itself incomplete. The 7B verifier behaves the same way — 6 of 9 cfgC answers were passed as `verified=true` while being wrong or non-responsive (A02, F01, H05, I01, J01, K02), and it still rejected two drafts. So the verifier is simultaneously too strict on acceptable drafts and too lenient on non-responsive ones — it is evaluating a different property (entailment of the quoted text) than the property the product needs (does this answer the user's question).

**Important nuance:** config C does *not* inherit config B's behaviour. The 7B verifier accepted the irrelevant verbatim quotes that 3B also accepted, and rejected different drafts. Swapping in a larger verifier changed *which* answers pass, not *how many* are correct: 0/9 in both. That is the central finding of the model-optimization phase.

`n=9` is small; these are directional, not statistically stable. The mechanism, however, is not small-sample noise — §6.11/§6.12 explain deterministically why a verbatim extract must pass and a refusal must be reported as success.

---

## 8. Latency and resource measurements

### 8.1 Per-stage (production path, warm)
| Stage | Measured |
|---|---|
| Retrieval (incl. up to 5 passes) | mean 0.97 s, p50 0.79 s, p90 2.00 s, max 3.97 s |
| Ranking | 0.0004–0.09 s |
| Evidence selection | 0.13–2.29 s |
| Embedding (standalone) | 0.48 s cold, 0.18 s warm |
| Embedding **under 7B load** | 7.5–9.9 s (observed in application log) |
| Generation 3B | mean 42.5 s (cfgA) / 71.6 s (cfgB, includes retry) |
| Verification 3B | mean 23.0 s, max 82.0 s |
| Verification 7B | 101–286 s |

### 8.2 End-to-end by configuration
| Config | n | mean | p50 | p90 | max | min | timeout |
|---|---|---|---|---|---|---|---|
| A: 3B, no verify | 9 | **45.6 s** | 50.2 s | 68.9 s | 77.0 s | 21.8 s | 0 |
| B: 3B + 3B verify | 9 | **110.9 s** | 93.3 s | 164.2 s | 185.8 s | 71.9 s | 0 |
| C: 3B + 7B verify | 9 | **370.2 s** | 359.4 s | 518.6 s | **755.5 s** | 115.0 s | 0 |

Component means for config C: generation 104.3 s (max 190.5 s), verification 175.1 s (max 285.8 s, n=8). Verification alone is longer than the entire config-A request budget by 3.5–14×.

Cold start adds ~19 s for API warm-up (`_warm_models`) and a cold model load (measured 3B 36.5 s, 7B 135.5 s for the first verifier call). Every configuration was measured with the model already warm except for the first case of each run.

**No configuration comes within an order of magnitude of the 10 s preferred / 20 s acceptable target.** Even the fastest measured request (21.8 s) exceeds the acceptable budget, and the shipped configuration (7B generation) is slower still.

### 8.3 Memory
- `qwen2.5:3b` + embedding: 2.07 GB + 0.37 GB resident; API process RSS 101–233 MB.
- `qwen2.5:7b` + embedding: 4.83 GB + 0.37 GB on a 7.8 GB machine — this is why embedding degrades to 7.5–9.9 s during 7B work (the OS is paging/competing), and why the 7B verifier is far slower than its raw size suggests.

### 8.4 As-shipped 7B + 7B configuration
Since `.env` runs 7B for both roles, the shipped path is strictly slower than config C (7B generation on top of 7B verification). Application log evidence from the shipped configuration: one generation of `qwen2.5:7b` took **224.9 s**. Config C measured 115–755 s per question (mean 370.2 s) with only a 3B generator. The shipped configuration is therefore in the **6–12 minute per question** range on this hardware.

---

## 9. Model-configuration experiments (A / B / C)

### 9.1 Isolated verifier benchmark (9 labelled claim/evidence pairs, production prompt)
| Model | Verdict accuracy | Mean latency | Warm latency | Errors |
|---|---|---|---|---|
| `qwen2.5:3b` | **7/9 (78%)** | 6.0 s (incl. 36.5 s cold) | ~2.2 s | 2 false negatives (V04 paraphrase, V09 step) |
| `qwen2.5:7b` | **8/9 (89%)** | 20.7 s (incl. 135.5 s cold) | ~6.9 s | 1 false negative (V04 paraphrase) |

Neither model produced a **false positive** on this set (all four unsupported drafts — wrong number, unsupported addition, wrong passage, incomplete set — were rejected). 7B's single advantage is one supported-step case that 3B wrongly rejected. Cost: 3.1× the warm latency and 3.7× the cold load.

### 9.2 End-to-end configuration comparison
| | A: 3B gen, programmatic checks only | B: 3B gen + 3B verify | C: 3B gen + 7B verify |
|---|---|---|---|
| Mean latency | 45.6 s | 110.9 s | **370.2 s** |
| p90 / max latency | 68.9 s / 77.0 s | 164.2 s / 185.8 s | 518.6 s / **755.5 s** |
| Generation mean | 42.5 s | 71.6 s | 104.3 s |
| Verification mean | n/a | 23.0 s | 175.1 s |
| Fully correct | 1/9 | 0/9 | 0/9 |
| Wrong answers passed as verified | n/a (no verifier) | 6/9 | 6/9 |
| Peak RAM | ~2.4 GB | ~2.4 GB | ~5.2 GB + 2.07 GB generator resident |
| Availability failures | 1/9 | 2/9 | 2/9 |
| Requests over 20 s | **9/9** | **9/9** | **9/9** |

Config C's fastest request (115.0 s) is still 5.75× the acceptable budget, and its slowest (755.5 s) is 12.6 minutes. Verification is the bottleneck: 175.1 s mean, 285.8 s max, on a CPU-only 7.8 GB machine.

### 9.3 Recommendation
**Do not ship any of the three tested configurations.** The evidence specifically does not support "3B generation + conditional 7B verification":

1. The verifier's isolated accuracy advantage is one borderline case out of nine (8/9 vs 7/9), and end-to-end it produces **the same 0/9 fully-correct result** as the 3B verifier.
2. Its measured cost is +65 s (3B) or **+325 s (7B)** mean per request, plus a second generation whenever it rejects — and it rejects 7 of 9 first drafts.
3. It cannot catch the dominant failure mode (non-responsive but entailed text), because that is outside its question. Config C passed the "Edwin C" fragment and the apprenticeship quote as `verified=true`, exactly as config B did.
4. 7B verification on 7.8 GB RAM degrades the whole pipeline (embedding 7.5–9.9 s, both models resident) — it is not a free quality upgrade.
5. The dominant fix is upstream: better retrieval and a responsiveness check. No verifier model can recover an answer that the evidence never supported.

**Recommended target design (to implement after approval):**
- Generation: `qwen2.5:3b` (the documented default) — 7B generation is 3–4× slower for no measured accuracy gain on these document questions.
- Verification: **deterministic checks first** (the answer covers the question's anchors; every cited sentence's numbers/names appear in its own cited passage; a list question's required length is satisfied; the answer is not a near-duplicate of a chunk that lacks the questioned attribute), then call the **3B verifier only** when those checks pass but risk remains (numeric answers, comparison answers, low evidence coverage). No 7B verifier on any request path on this hardware.
- If the client requires 7B verification, it requires a GPU or ≥16 GB RAM; state that as a hardware prerequisite, not a configuration toggle.
- Even with verification off, the remaining latency (45.6 s mean) fails the 20 s budget; the P0 work must also reduce generation cost (shorter prompt, fewer chunks, `num_predict` cap, streaming) or accept a relaxed latency target explicitly agreed with the client.

---

## 10. Spelling correction (Phase 4)

Original query is always searched as a fallback (`chat.py` adds it when corrected ≠ original), so the safety property "original query remains a retrieval path" **holds**. The quality of the correction itself does not.

| Case | Original | Corrected | Retrieval outcome |
|---|---|---|---|
| K02 | whaat is benjmin frankln | what is benjamin franklin | correct text, page-level recall still 0 |
| K03 | explane the contrubutions of benjamin | **explained** the contributions of benjamin | wrong mood; retrieval missed |
| K05 | tell me abot the importnt events | tell me **abot** the important events | partial; retrieval failed |
| K06 | responsibilites of the junto | responsibilities of the junto | success |
| K07 | hpw many chaptrs … encheeridion | hpw many chapters … encheiridion | success (second token still wrong) |
| K08 | teh six basik feers | the six basik **fees** | success by luck ("six" + Hill scope) |
| F05 | recommends | recommend | success |
| I02 | disadvantages | disadvantage | success |

Extra processing time is not measurable separately (the correction runs against an `lru_cache`d vocabulary and costs well under 100 ms); the cost of correction is **quality risk**, not latency. Meaning is preserved in 4/8 cases, damaged in 2/8, and ambiguously repaired in 2/8.

**Fix:** require a minimum similarity margin between the best and second-best candidate, never rewrite a token that is a valid English word (use a small stoplist/dictionary), preserve verb morphology, and log every rewrite to the response (it already returns `spelling_correction`).

---

## 11. Citations and abstention

### 11.1 Citation syntax — good
Every live answer passed `references_valid`; no phantom labels, no out-of-range references, no fabricated filenames. The citation *plumbing* (`citations.py`) is well built.

### 11.2 Citation semantics — weak
- Page accuracy on the live subset: 6 of 9 cases cited **no** expected page; the best case was 1/2 and one was 0/3.
- Because the generator only sees numbered evidence blocks, a citation proves "this chunk", not "this page answers the question". With precision_selected at 0.149, a syntactically perfect citation frequently points at a page that does not contain the answer.
- The strict `fully_cited` gate converts one missing bracket into a user-visible failure; the loose `references_valid` gate lets a wholly irrelevant answer through. Both extremes are live-observed.

### 11.3 Abstention — unreliable in both directions
| Expected | Observed |
|---|---|
| J01 unanswerable | Hallucinated "1729", `status: success` |
| J02–J05 unanswerable (retrieval-only) | 20–38 chunks above the gate; only the generator can abstain |
| A02 answerable (p7 in corpus) | Refused, `status: success` |
| I01 answerable (after resolution) | Refused, `status: success`, confidence 0.826 |

Abstention accuracy on the live subset: **1 of 6** correct decisions (J01 wrong, A02 wrong, I01 wrong; J-only questions were not in the live subset beyond J01). The deterministic early-abstention path is dead code (§6.9).

---

## 12. Formats, ingestion, security, and code quality

### 12.1 Format support (measured at the extraction layer)
`SUPPORTED` declares 26 extensions. Extractor code exists for PDF (PyMuPDF), DOCX (python-docx, incl. tables), PPTX (python-pptx), XLSX/XLSM (openpyxl read-only), ODS, CSV, images (Tesseract), TXT/MD/log/YAML, EML (with attachments), RTF, HTML, XML, JSON, ODT.

`format_probe.py` extracted a synthetic file of each available type in a temp directory (no index writes). Measured result:

| Format | Result | Blocks | Time | Note |
|---|---|---|---|---|
| `.txt` | OK | 2 | 29 ms | paragraph split |
| `.csv` | OK | 1 | 11 ms | header/cell labelled |
| `.docx` | OK | 2 | 36 ms | heading, paragraph **and table** extracted |
| `.xlsx` | OK | 1 | 30 ms | sheet name + rows |
| `.html` | OK | 1 | 61 ms | style/script stripped |
| `.md` | OK | 2 | 8 ms | |
| `.json` | OK | 1 | 13 ms | |
| `.png` (OCR) | OK | 1 | 5.4 s | "JUNTO MEETS FRIDAY EVENING" read correctly |

**Dependency status:** `requirements.txt` correctly declares `python-pptx>=1.0`, `striprtf>=0.0.26` and `odfpy>=1.4`, but **`python-pptx` and `striprtf` are not installed in this environment**. Consequences:
- `.pptx` and `.rtf` are advertised in the upload UI and in `SUPPORTED`, and will raise `ImportError` at extraction time (queued job fails, not upload validation).
- The failure is an **environment/install gap, not a code gap** — a clean `pip install -r requirements.txt` fixes it. But the delivered machine does not currently satisfy its own manifest, so a client PPTX/RTF upload will fail. Verify with `pip check` / `pip install -r requirements.txt` before handover.
- `.odt`/`.ods` were not probed; `odfpy` import status was not confirmed.

Present and working: `pymupdf` 1.28.2, `python-docx` 1.2.0, `openpyxl` 3.1.5, `Pillow` 11.2.1, `pytesseract` 0.3.13. Tesseract binary present at `C:\Program Files\Tesseract-OCR\tesseract.exe` with `eng` + `osd` data; `OCR_ENABLED=true`; image pixel and page limits configured.

Archive-bomb guards (`_check_archive_limits`: entry count, uncompressed size, compression ratio) are implemented and applied before extraction — genuinely good.

### 12.2 Ingestion robustness
- Atomic copy + fsync before DB commit; SHA-256 duplicate detection; staged-file retention on failure.
- Spooled extraction/chunking to disk (`*.jsonl.tmp` → `os.replace`) so a 200 MB PDF is never fully in RAM — good.
- Job claiming is atomic (`claim_job`, `update_job_if_status`), cancellation is checkpointed, partial index cleanup on failure.
- Worker count clamped to 2 (`min(settings.ingestion_workers, 2)`) even though config says 4 — an undocumented cap.

### 12.3 Security / deployment posture
- Binds to `127.0.0.1`; Ollama URL validation rejects non-loopback; Chroma telemetry explicitly disabled; no outbound HTTP in the request path.
- **No authentication, no CORS policy, no rate limiting** — by design for single-user loopback, but this must be stated as a hard deployment constraint in the handover. `FEATURE_COVERAGE.md` already marks multi-user/network as out of scope.
- Upload path sanitized (`Path(...).name`), extension + size + quota validated, ownership checked on delete, cache invalidated on document delete.
- Prompt-injection defence is present in the generation and verification prompts ("treat passages as untrusted data") — reasonable, not a guarantee.

### 12.4 Test suite
203 passed, 1 skipped in 131.65 s. Coverage is broad and genuinely useful (retrieval degradation, cache, persistence, hardening, citation contract). But:
- There is **no test asserting retrieval recall on the real corpus**; the live harness (`tests/eval_live_corpus.py`) is manual and not collected.
- `test_lexical_candidate_limit_is_respected_as_a_real_cap` **locks in** the 40-candidate behaviour that causes §6.1 — changing the cap requires updating this test deliberately.
- No test covers "answer must be responsive to the question", which is the dominant live failure.

**Conclusion:** the suite protects the code from regressions in behaviour it already has; it does not protect the product from being wrong.

---

## 13. Scorecard (0–100)

Bands: 90–100 deliverable; 75–89 deliverable with minor gaps; 60–74 needs remediation; 40–59 not acceptable; <40 fail.

| Dimension | Score | Basis |
|---|---|---|
| Retrieval quality | **42** | Recall@4 0.477, recall_selected 0.538, precision 0.149 |
| Answer correctness | **22** | 1/9 fully correct; 4/9 non-responsive or wrong |
| Answer completeness | **35** | 7 incomplete cases; summaries 0/6 sufficient |
| Faithfulness / hallucination | **45** | 2/9 fabricated facts in the unverified config |
| Citation accuracy (doc + page) | **50** | Syntax 100%, page accuracy ≤ 50% |
| Abstention accuracy | **20** | 1/6 correct decisions; dead early-abstention |
| Latency / throughput | **8** | 21.8–755.5 s against a 10–20 s budget |
| Resource efficiency | **45** | Fits 7.8 GB with 3B; 7B starves the pipeline |
| Robustness / honesty of status | **30** | Refusals and hallucinations reported as success |
| Security posture (loopback single-user) | **65** | Good local hygiene; no auth by design |
| Code quality / maintainability | **78** | Clear modules, comments, guards; some dead code |
| **Overall readiness** | **38 / 100** | **Fail** |

---

## 14. Remediation roadmap

### P0 — blocking (must fix before any client delivery)
1. **Retrieval recall.** Raise the dense candidate window for long documents (e.g. `top_k × 8–10`, or a per-document quota), and stop OR-ing the FTS query blindly: use `AND` for rare terms / bm25 term weighting, and score lexical hits on their own scale instead of `0.35 × coverage` against a semantic threshold.
2. **Remove document scoping from comparison questions** (`_named_document_scope` must not narrow when ≥2 documents or comparison intent is present).
3. **Add per-document diversity to `select_evidence`** (at least one chunk from each named document/entity before filling by score).
4. **Stop the extractive fallback from returning non-answers.** It must require an answerability signal (the sentence must contain the questioned attribute/entity relation), not 50% word overlap. Until then, disable it.
5. **Make status honest.** A refusal must be `abstained: true` and must not carry high confidence; `status: success` must mean "answered", not "produced text".
6. **Disable LLM verification in the default config** until it can test responsiveness, not just entailment. As implemented it costs 2.4–11× latency and lowers measured correctness.

### P1 — required for a credible pilot
7. Filter title/TOC/running-header/colophon chunks out of retrieval (chunk-role metadata at ingestion).
8. Fix spelling correction: margin-based ambiguity rejection, dictionary guard, morphology preservation.
9. Fix query variants: drop 2-word fragments and possessive-split `s` tokens; cap variants at 2.
10. Bound and batch the verifier prompt (per-claim calls or a length cap) so late claims are never silently truncated.
11. Calibrate or relabel confidence; include answer-question relevance.
12. Install `python-pptx` and `striprtf`, or remove those extensions from `SUPPORTED` and the upload UI.

### P2 — quality
13. Two-pass retrieval for multi-hop/comparison: retrieve per entity, then merge.
14. Page-level citation verification (the cited page must contain the cited claim's key terms).
15. Re-run the 70-question set as a regression gate in CI against a frozen index snapshot.

### P3 — performance
16. Precompute/cache embeddings for the 70-question set; keep model warm; consider a GPU or a smaller quantized generator.
17. Cache verification verdicts by (question, evidence hash).

---

## 15. Artifacts produced (all read-only w.r.t. the application)

| File | Contents |
|---|---|
| `questions.json` | 70 source-verified questions (A–K) with pages, evidence terms, expected answers |
| `retrieval_results.json` | Per-question candidates, ranks, selection, metrics, timings |
| `evidence_requirements.json` | Source-derived evidence checklist per question |
| `sufficiency.json` / `case_classification.json` | Adjudicated sufficiency and Case 1/2/5 classification |
| `verifier_results.json` / `verifier_bench.log` | 3B vs 7B verifier accuracy and latency |
| `live_cfgA_3b_programmatic.json/.log` | 9 live cases, 3B, no verification |
| `live_cfgB_3b_3b.json/.log` | 9 live cases, 3B + 3B verification |
| `live_cfgC_3b_7b.json/.log` | 3B + 7B verification |
| `adjudication.json` | Per-case manual scoring (correctness, completeness, faithfulness, hallucination) |
| `format_probe.py` | Isolated per-format extraction test (no index writes) |

---

## 16. What would change the verdict

The verdict is not a judgement about the team or the architecture; it is a measurement of the current build. It would move to "deliverable with caveats" if, after P0:
- evidence-sufficiency on the 70-question set rises from 58.5% to ≥ 85%,
- fully-correct live answers rise from 1/9 to ≥ 7/9 on the same fixed set,
- no answer returns `status: success` with a refusal or an irrelevant quote,
- p90 latency on the target machine is < 20 s (which likely requires ≤ 3B generation and no LLM verifier, or a GPU).

---

## 17. Approval requested

No application code or configuration has been changed. I am asking for approval to proceed with the P0 fixes, each with a regression test, followed by a re-run of the same 70-question retrieval set and the same 9 live questions, reported before/after.

Proposed first change set (smallest set that should move the numbers):
1. `app/rag/chat.py` — comparison-aware document scoping; extractive-fallback relevance gate; honest `abstained`/status semantics.
2. `app/retrieval/evidence_selector.py` — per-document diversity floor.
3. `app/retrieval/store.py` — lexical scoring and candidate widening.
4. `app/rag/planner.py` — variant and spelling-correction guards.
5. `app/core/config.py` + `.env` — default to 3B generation, verification off, documented.

Each with a test that fails on the current code.
