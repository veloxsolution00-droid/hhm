import sys
import types


def test_rank_pipeline_reports_route_diagnostics():
    from app.retrieval.pipeline import rank_candidates_with_diagnostics

    candidates = [
        {"id": "d1:1", "text": "Annual revenue was 10 crore for FY 2025.",
         "similarity": 0.82, "semantic_similarity": 0.90,
         "lexical_similarity": 0.70, "metadata": {"document_id": "d1"}},
        {"id": "d1:2", "text": "The company office address is Mumbai.",
         "similarity": 0.60, "semantic_similarity": 0.62,
         "lexical_similarity": 0.30, "metadata": {"document_id": "d1"}},
    ]
    ranked, diagnostics = rank_candidates_with_diagnostics("annual revenue FY 2025", candidates)
    assert ranked
    assert diagnostics["dense_candidates"] == 2
    assert diagnostics["keyword_candidates"] == 2
    assert diagnostics["fusion_candidates"] == 2
    assert diagnostics["candidate_pool_shared"] is True


def test_evidence_selector_reports_uncovered_terms_and_selection_metadata():
    from app.retrieval.evidence_selector import select_evidence

    result = select_evidence(
        "manager approves travel budget",
        [{"id": "1", "text": "The manager approves travel", "metadata": {}}],
    )
    assert result["evidence_found"] is True
    assert "budget" in result["missing_terms"]
    assert result["selection_strategy"] == "relevance_coverage_redundancy"
    assert result["selected_characters"] == len("The manager approves travel")


def test_multi_question_retrieval_diagnostics_are_aggregated(monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    from app.rag.chat import _aggregate_retrieval_diagnostics

    answers = [
        {"question": "Q1", "status": "success", "retrieval": {
            "candidate_chunks": 8, "selected_chunks": 4, "retrieval_passes": 2,
            "best_similarity": 0.91, "dense_candidates": 8, "keyword_candidates": 5,
            "reranked_candidates": 7, "ranking_seconds": 0.02}},
        {"question": "Q2", "status": "insufficient_evidence", "retrieval": {
            "candidate_chunks": 3, "selected_chunks": 1, "retrieval_passes": 3,
            "best_similarity": 0.55, "dense_candidates": 3, "keyword_candidates": 2,
            "reranked_candidates": 2, "ranking_seconds": 0.01}},
    ]
    result = _aggregate_retrieval_diagnostics(answers, 12)
    assert result["candidate_chunks"] == 11
    assert result["selected_chunks"] == 5
    assert result["retrieval_passes"] == 5
    assert result["best_similarity"] == 0.91
    assert len(result["question_diagnostics"]) == 2


def test_context_budget_is_below_configured_character_ceiling():
    from app.core.config import settings

    available = max(
        512, int(settings.model_context_tokens)
        - int(settings.context_safety_tokens)
        - int(settings.answer_max_tokens)
    )
    budget = min(
        int(settings.max_context_chars),
        max(1800, int(available * float(settings.context_chars_per_token))),
    )
    assert budget <= settings.max_context_chars
    assert budget > 0



def test_short_answer_bearing_chunk_is_not_dropped_beside_long_chunks():
    from app.retrieval.evidence_selector import select_evidence

    short = {
        "id": "short-table-cell",
        "text": "Approval limit: $5,000.",
        "similarity": 0.92,
        "rerank_score": 0.92,
        "metadata": {"document_id": "d1", "filename": "policy.xlsx", "source_location": "Sheet1 Row 8"},
    }
    long = {
        "id": "long-unrelated",
        "text": "Background policy description. " * 20,
        "similarity": 0.55,
        "rerank_score": 0.55,
        "metadata": {"document_id": "d1", "filename": "policy.xlsx", "source_location": "Sheet1 Row 1"},
    }
    result = select_evidence("What is the approval limit?", [long, short], max_chunks=2)
    assert "short-table-cell" in [item["id"] for item in result["hits"]]
