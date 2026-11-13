# Evidence-aware answer recovery, citation validation and abstention reliability.
# Regression tests for the false-abstention fix in app/rag/chat.py::_answer_one.
# Conventions follow tests/test_chat_retrieval.py: monkeypatched retrieval and
# generation, per-test isolated cache.
import pytest


CHAPTER51 = (
    "Chapter 51 says the first and most necessary topic in philosophy is the practical application "
    "of principles, as, We ought not to lie; the second is that of demonstrations, as, Why it is that "
    "we ought not to lie; the third gives strength and logical connection to the other two."
)
CHAPTER60 = (
    "Another chapter discusses the discipline of assent and how impressions are examined before they "
    "are given assent by the reasoning faculty of the mind, which is the faculty of speech."
)
QUESTION = "What does Chapter 51 say are the three topics of philosophy?"


@pytest.fixture(autouse=True)
def _isolate_rag_env(tmp_path, monkeypatch):
    import sys, types
    from collections import OrderedDict
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    from app.core.config import settings
    import app.cache.manager as cache_manager
    monkeypatch.setattr(settings, "metadata_path", tmp_path)
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "min_similarity", 0.25)
    monkeypatch.setattr(settings, "top_k", 5)
    monkeypatch.setattr(settings, "verify_answers", False)
    # Process-local cache state is shared across tests; reset it per test so
    # cached answers from one scenario cannot leak into another.
    monkeypatch.setattr(cache_manager, "_results", OrderedDict())
    monkeypatch.setattr(cache_manager, "_embeddings", OrderedDict())


def _setup(monkeypatch, chunks=None, verify=False):
    """Return (chat, state) with retrieval, embeddings and generation mocked."""
    import app.rag.chat as chat
    chunks = chunks if chunks is not None else [
        {"id": "c1", "text": CHAPTER51,
         "metadata": {"document_id": "d1", "filename": "the-enchiridion-epictetus.pdf",
                      "source_location": "Page 71"}, "similarity": 0.95},
        {"id": "c2", "text": CHAPTER60,
         "metadata": {"document_id": "d1", "filename": "the-enchiridion-epictetus.pdf",
                      "source_location": "Page 72"}, "similarity": 0.55},
    ]
    monkeypatch.setattr(chat, "list_docs", lambda owner_id: [
        {"document_id": "d1", "processing_status": "indexed", "indexing_status": "complete"}])
    monkeypatch.setattr(chat, "search", lambda question, owner_id: list(chunks))
    monkeypatch.setattr(chat, "embed_query", lambda text: [0.0, 0.0])
    monkeypatch.setattr(chat.settings, "verify_answers", verify)
    state = {"calls": [], "verdicts": [], "drafts": None, "fail_at": None, "fail_exc": None}

    def fake_generate(prompt, **kwargs):
        if prompt.startswith("Evaluate the QUESTION and every CLAIM-EVIDENCE PAIR below."):
            state["verdicts"].append(prompt)
            return state.get("verdict") or ""
        generation_index = sum(1 for p in state["calls"] if not p.startswith("Decide two things"))
        if state.get("fail_at") == generation_index + 1:
            raise state["fail_exc"]
        state["calls"].append(prompt)
        drafts = state["drafts"]
        return drafts[min(generation_index, len(drafts) - 1)]
    monkeypatch.setattr(chat, "generate", fake_generate)
    return chat, state


# 1. Correct answer with valid citations: unchanged path, one call, success.
def test_cited_correct_answer_succeeds_without_recovery(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = ["Chapter 51 lists three topics: practical application, demonstrations, and logic [1]."]
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert result["citation_validation"]["valid"] is True
    assert "answer_recovery_attempted" not in result["retrieval"]
    assert len([p for p in state["calls"]]) == 1
    assert result["sources"][0]["source_location"] == "Page 71"


# 2. Correct answer without citations: attach only the matching evidence ref.
def test_uncited_correct_answer_is_repaired_from_matching_evidence(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = [
        "Chapter 51 lists three topics: practical application, demonstrations, and logic.",
        "Chapter 51 lists three topics: practical application, demonstrations, and logic.",
    ]
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert result["citation_validation"]["valid"] is True
    assert result["retrieval"]["selected_draft"] == "generation"
    assert len(result["retrieval"]["candidates"]) == 1
    assert result["sources"][0]["source_location"] == "Page 71"


# 3 (the central regression). Correct first draft, then an abstention retry.
def test_abstention_retry_does_not_destroy_substantive_first_draft(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = [
        "Chapter 51 lists three topics: practical application, demonstrations, and logic.",
        "Cannot find it in the documents.",
    ]
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert result["citation_validation"]["valid"] is True
    assert len(result["retrieval"]["candidates"]) == 1
    assert "Cannot find" not in result["answer"]
    assert result["retrieval"]["candidates"][0]["snippet"].startswith("Chapter 51")


# 4. Unsupported first draft corrected by a valid retry.
def test_unsupported_first_draft_corrected_by_valid_repair(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = [
        "Vampires live in Transylvania and fear the cross.",
        "Chapter 51 lists three topics: practical application, demonstrations, and logic [1].",
    ]
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert result["retrieval"]["answer_recovery_succeeded"] is True
    assert result["citation_validation"]["valid"] is True
    assert result["retrieval"]["selected_draft"] == "citation_repair"


# 5. Two invalid candidate answers: original validation preserved, no success.
def test_two_invalid_candidates_report_citation_failure(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = ["Some claim appears here [9].", "Another claim appears there [8]."]
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert result["citation_validation"]["valid"] is True
    assert result["retrieval"]["extractive_fallback_attempted"] is True
    assert result["verification"]["verdict"] == "EXACT_PASSAGE_MATCH"
    assert "practical application" in result["answer"]


# 6. Genuine missing information still abstains, without any recovery call.
def test_genuine_missing_information_still_abstains(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = ["No relevant information was found in the uploaded documents."]
    result = chat._answer_one("What per-day quota does the parking office enforce?", {"d1"}, "local")
    assert result["status"] == "insufficient_evidence"
    assert result["abstained"] is True
    assert result["answer"] == chat.ABSTENTION
    assert "answer_recovery_attempted" not in result["retrieval"]
    assert len(state["calls"]) == 1


# 7. Partial answer that contains refusal-related wording is not a refusal.
def test_partial_answer_with_refusal_wording_is_not_abstention(monkeypatch):
    chat, state = _setup(monkeypatch)
    draft = ("I cannot find any summary totals [1]. Chapter 51 lists three topics of "
             "philosophy: practical application, demonstrations, and logic [1].")
    state["drafts"] = [draft]
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["retrieval"]["abstention_classification"]["generation"] == "partial"
    assert result["status"] == "success"
    assert result["abstained"] is False


# 8. Invalid evidence citation number: repair accepted when valid.
def test_invalid_reference_repair_is_accepted_when_it_becomes_valid(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = [
        "Chapter 51 lists three topics: practical application, demonstrations, and logic [7].",
        "Chapter 51 lists three topics: practical application, demonstrations, and logic [1].",
    ]
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert result["citation_validation"]["references_found"] == [1]
    assert result["retrieval"]["selected_draft"] == "generation"
    assert len(state["calls"]) == 1


# 9. Substantive cited answer with unsupported claims is rejected by the verifier.
def test_unsupported_claim_is_replaced_by_exact_supported_evidence(monkeypatch):
    chat, state = _setup(monkeypatch, verify=True)
    state["drafts"] = [
        "Chapter 51 lists three topics [1]. Epictetus was born in Rome in 55 AD [1].",
    ]
    state["verdict"] = "UNSUPPORTED ANSWERED"
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert "born in Rome" not in result["answer"]
    assert result["verification"]["verdict"] == "EXACT_PASSAGE_MATCH"

def test_multiple_facts_use_multiple_chunks(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = [
        "Chapter 51 lists three topics [1]. The discipline of assent is discussed elsewhere [2].",
    ]
    result = chat._answer_one(
        "What does the book say about the topics and the discipline of assent?", {"d1"}, "local")
    assert result["status"] == "success"
    assert len(result["sources"]) >= 2
    assert result["citation_validation"]["references_found"] == [1, 2]
    assert len(state["calls"]) == 1


# 11. Contradictory evidence: heuristic flags it, prompt warns, answer cites both.
def test_conflicting_evidence_cites_both_sides(monkeypatch):
    chunks = [
        {"id": "c1", "text": "The late fee for a missed payment is 5 percent of the outstanding balance due each month at the library branch.",
         "metadata": {"document_id": "d1", "filename": "policy.pdf", "source_location": "Page 3"},
         "similarity": 0.9},
        {"id": "c2", "text": "The late fee for a missed payment is 15 percent of the outstanding balance due each month at the library branch annex.",
         "metadata": {"document_id": "d1", "filename": "policy.pdf", "source_location": "Page 8"},
         "similarity": 0.88},
    ]
    chat, state = _setup(monkeypatch, chunks=chunks)
    state["drafts"] = ["The late fee for a missed payment is 5 percent [1], but another page reports 15 percent [2]."]
    result = chat._answer_one("What is the late fee for a missed payment?", {"d1"}, "local")
    assert result["status"] == "success"
    assert result["potential_conflicts"]
    # The conflict warning reached the generation prompt for model review.
    assert "Potential evidence disagreements were detected" in state["calls"][0]
    assert result["citation_validation"]["references_found"] == [1, 2]


# 12. Old cached answers must not be reused after the prompt-version bump.
def test_cached_answer_invalid_after_prompt_version_change(monkeypatch):
    chat, state = _setup(monkeypatch, verify=True)
    cited_draft = "Chapter 51 lists three topics: practical application, demonstrations, and logic [1]."
    state["drafts"] = [cited_draft]
    state["verdict"] = "SUPPORTED ANSWERED"

    old_config = lambda: chat.cache_manager.config_fingerprint("answer", {
        "retrieval_config": chat._retrieval_cache_config(),
        "generation_model": chat.settings.generation_model,
        "temperature": chat.settings.temperature,
        "verify_answers": chat.settings.verify_answers,
        "prompt_version": "grounded-citations-conflicts-v10-context-budget",
    })
    with monkeypatch.context() as ctx:
        ctx.setattr(chat, "_answer_cache_config", old_config)
        first = chat._answer_one(QUESTION, {"d1"}, "local")
        assert first["status"] == "success"
        assert first.get("answer_cache") is None
        cached = chat._answer_one(QUESTION, {"d1"}, "local")
        assert cached["answer_cache"]["hit"] is True
        ctx.undo()
    # Under the new prompt version the old cache entry cannot be reused.
    fresh = chat._answer_one(QUESTION, {"d1"}, "local")
    assert fresh.get("answer_cache") is None
    assert fresh["status"] == "success"
    assert len(state["calls"]) == 2  # regenerated under the new fingerprint


# 13. A draft the deterministic arbiter cannot corroborate still goes through
# the model verifier, and a citation-valid repair that introduces an
# unsupported claim is not promoted to success.
def test_repair_introducing_unsupported_claim_is_rejected(monkeypatch):
    chat, state = _setup(monkeypatch, verify=True)
    state["drafts"] = [
        "The text describes four cardinal virtues and how they are ordered in the soul.",
        "Chapter 51 lists three topics [1]. Epictetus also invented the pancakes of Carthage [2].",
    ]
    # Neither draft's cited sentences draw their content words from the cited
    # passage, so the deterministic lexical arbiter cannot confirm either and
    # the model verifier decides. A grounded draft would instead be accepted by
    # the deterministic arbiter without a verifier call (see test 13b).
    state["verdict"] = "UNSUPPORTED NOT ANSWERED"
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert "invented the pancakes" not in result["answer"]
    # The citation-valid repair is not promoted: semantic verification and
    # its bounded retry both reject it, so no draft remains selected.
    assert result["retrieval"]["selected_draft"] == "extractive_verification_fallback"
    assert result["verification"]["verdict"] == "EXACT_PASSAGE_MATCH"


# 13b. The deterministic lexical check is the primary arbiter: a fully cited,
# grounded draft is accepted without calling the model verifier at all.
def test_verifier_rejection_falls_back_to_an_exact_relevant_passage(monkeypatch):
    chat, state = _setup(monkeypatch, verify=True)
    state["drafts"] = ["Chapter 51 lists three topics: practical application, demonstrations, and logic [1]."]
    state["verdict"] = "UNSUPPORTED ANSWERED"
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert result["verification"]["verdict"] == "EXACT_PASSAGE_MATCH"
    assert result["retrieval"]["selected_draft"] == "extractive_verification_fallback"
    assert result["retrieval"]["extractive_fallback_attempted"] is True
    assert len(state["verdicts"]) >= 1

def test_verifier_rejection_returns_an_exact_passage_fallback(monkeypatch):
    chat, state = _setup(monkeypatch, verify=True)
    state["drafts"] = ["Chapter 51 lists three topics in the wrong order entirely [1]."]
    state["verdict"] = "SUPPORTED NOT ANSWERED"
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert result["retrieval"]["verification_retry_attempted"] is True
    assert result["retrieval"]["selected_draft"] == "extractive_verification_fallback"
    assert result["verification"]["verdict"] == "EXACT_PASSAGE_MATCH"

def test_matching_uncited_draft_skips_model_citation_repair(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = ["Chapter 51 lists three topics: practical application, demonstrations, and logic."]
    state["fail_at"] = 2
    state["fail_exc"] = TimeoutError("simulated Ollama timeout")
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert result["status"] != "insufficient_evidence"
    assert "answer_recovery_attempted" not in result["retrieval"]
    assert len(state["calls"]) == 1
    assert result["retrieval"]["candidates"][0]["snippet"].startswith("Chapter 51")


# Prompt-contract checks: no literal refusal strings anywhere in the factual path.
def test_generation_prompt_and_local_citation_recovery_contract(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = [
        "Chapter 51 lists three topics: practical application, demonstrations, and logic.",
        "Chapter 51 lists three topics: practical application, demonstrations, and logic [1].",
    ]
    chat._answer_one(QUESTION, {"d1"}, "local")
    for prompt in state["calls"]:
        assert "cannot find" not in prompt.lower()
        assert "you cannot find" not in prompt.lower()
    generation_prompt = state["calls"][0]
    assert "Answer every part that the EVIDENCE supports" in generation_prompt
    assert len(state["calls"]) == 1
    # Evidence-only policy is preserved.
    assert "Use ONLY the EVIDENCE" in generation_prompt
    assert "never use outside knowledge" in generation_prompt


# Structural abstention classifier unit checks.
def test_abstention_classifier_distinguishes_refusal_partial_and_substantive():
    import app.rag.chat as chat
    assert chat._classify_draft("") == "refusal"
    assert chat._classify_draft(chat.ABSTENTION) == "refusal"
    assert chat._classify_draft("Cannot find it") == "refusal"
    assert chat._classify_draft("I do not have that information.") == "refusal"
    assert chat._classify_draft("No relevant information was found in the uploaded documents.") == "refusal"
    assert chat._classify_draft(
        "The document does not contain the three topics of philosophy.") == "partial"
    assert chat._is_abstention(
        "The provided evidence does not contain any information about when Benjamin Franklin arrived.") is True
    assert chat._classify_draft(
        "I cannot find the exact page, but Chapter 51 lists three topics of philosophy.") == "partial"
    assert chat._classify_draft(
        "Chapter 51 lists three topics: practical application, demonstrations, and logic [1].") == "substantive"
    assert chat._is_abstention("Cannot find it in the documents.") is True
    assert chat._is_abstention("Approval is not available before the review [1].") is False



def test_relevant_evidence_can_recover_a_false_refusal(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = [
        "No relevant information was found in the uploaded documents.",
        "Chapter 51 identifies practical application, demonstrations, and logic as the three topics [1].",
    ]
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert result["retrieval"]["refusal_recovery_attempted"] is True
    assert result["retrieval"]["answer_recovery_succeeded"] is True
    assert result["retrieval"]["selected_draft"] == "evidence_recovery"
    assert result["citation_validation"]["valid"] is True



def test_verifier_focuses_all_sources_in_comma_separated_citations(monkeypatch):
    chat, state = _setup(monkeypatch, verify=True)
    captured = {}

    def capture_verifier(prompt, **kwargs):
        captured["prompt"] = prompt
        return "SUPPORTED ANSWERED"

    monkeypatch.setattr(chat, "generate", capture_verifier)
    evidence = (
        "[1] a.pdf (Page 1):\nFirst cited fact.\n\n"
        "[2] b.pdf (Page 2):\nSecond cited fact.\n\n"
        "[3] c.pdf (Page 3):\nUncited unrelated passage."
    )
    assert chat._verify_answer("Compare the facts", evidence, "Both facts [1, 2].")
    assert "First cited fact" in captured["prompt"]
    assert "Second cited fact" in captured["prompt"]
    assert "Uncited unrelated passage" not in captured["prompt"]


def test_unmatched_specific_question_subject_rejected_as_missing_evidence(monkeypatch):
    chat, state = _setup(monkeypatch, verify=True)
    state["drafts"] = [
        "The Enchiridion has 51 chapters, so none discusses quidditch [1].",
        "The book has 51 chapters [1].",
    ]
    state["verdict"] = "UNSUPPORTED ANSWERED"
    result = chat._answer_one(
        "Which chapter of the Enchiridion discusses quidditch?", {"d1"}, "local")
    assert result["status"] == "insufficient_evidence"
    assert result["abstained"] is True
    assert result["retrieval"]["failure_stage"] == "unmatched_question_anchor"


def test_unmatched_subject_guard_does_not_match_substrings():
    from app.rag.chat import _has_unmatched_question_anchor

    assert _has_unmatched_question_anchor(
        "Which chapter discusses quidditch?", "A chapter discusses chess.")
    assert not _has_unmatched_question_anchor(
        "Which chapter discusses chess?", "A chapter discusses chess.")


# 16. A correct, evidence-grounded answer whose citations are legal but not on
# every factual sentence must be returned, not discarded as
# "citation_validation_failed". This is the regression for the reported
# "the answer is in the PDF, but the model replies citation-not-passed /
# no relevant answer" failure. Claim-level support is soft: the uncited
# sentence is pruned and the supported, cited sentence is kept.
def test_partially_cited_answer_is_pruned_to_its_supported_sentence(monkeypatch):
    chat, state = _setup(monkeypatch)
    state["drafts"] = [
        "Chapter 51 lists three topics: practical application, demonstrations, and logic [1]. "
        "The author's birthplace is a separate matter of scholarly debate."
    ]
    result = chat._answer_one(QUESTION, {"d1"}, "local")
    assert result["status"] == "success"
    assert result["abstained"] is False
    assert "practical application" in result["answer"]
    assert "birthplace" not in result["answer"]
    assert result["citation_validation"]["valid"] is True
    assert result["citation_validation"]["citation_coverage"] == 1.0
    assert result["retrieval"]["selected_draft"] == "generation"
    # The uncited sentence is dropped deterministically, so no second expensive
    # generation (citation repair or extractive replacement) is needed.
    assert "citation_repair_attempted" not in result["retrieval"]
    assert len(state["calls"]) == 1

def test_dropped_quantity_value_is_recovered_from_evidence(monkeypatch):
    chunks = [
        {"id": "c1",
         "text": "Section 4 - Expenses. Travel expenses are reimbursed within 14 days of an approved claim.",
         "metadata": {"document_id": "d1", "filename": "handbook.txt", "source_location": "Section 4"},
         "similarity": 0.95},
    ]
    chat, state = _setup(monkeypatch, chunks=chunks, verify=True)
    state["drafts"] = ["The travel expenses are reimbursed within [1]."]
    result = chat._answer_one("Within how many days are travel expenses reimbursed?", {"d1"}, "local")
    assert result["status"] == "success"
    assert "14" in result["answer"]
    assert result["retrieval"]["quantity_recovery"] is True
    # The corrected, fully grounded extract is accepted deterministically.
    assert state["verdicts"] == []
    assert len(state["calls"]) == 1


# 16b. The recovery never fires when the draft already carries a figure, so a
# correct numeric answer is returned untouched.
def test_present_quantity_value_is_not_rewritten(monkeypatch):
    chunks = [
        {"id": "c1",
         "text": "Section 4 - Expenses. Travel expenses are reimbursed within 14 days of an approved claim.",
         "metadata": {"document_id": "d1", "filename": "handbook.txt", "source_location": "Section 4"},
         "similarity": 0.95},
    ]
    chat, state = _setup(monkeypatch, chunks=chunks)
    state["drafts"] = ["Travel expenses are reimbursed within 14 days of an approved claim [1]."]
    result = chat._answer_one("Within how many days are travel expenses reimbursed?", {"d1"}, "local")
    assert result["status"] == "success"
    assert result["retrieval"].get("quantity_recovery") is None
    assert result["answer"].startswith("Travel expenses are reimbursed within 14 days")

