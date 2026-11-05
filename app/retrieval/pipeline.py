"""In-memory retrieval orchestration over the existing local indexes.

The persistent store remains unchanged. Dense and lexical views are ranked from
one bounded candidate pool, fused, then reranked. Diagnostics expose the limits
of the current shared-candidate architecture instead of implying independent
retrieval calls.
"""
from app.retrieval.dense_search import rank_dense_candidates
from app.retrieval.keyword_search import rank_keyword_candidates
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank


def rank_candidates_with_diagnostics(query: str, candidates: list[dict],
                                     limit: int | None = None) -> tuple[list[dict], dict]:
    dense = rank_dense_candidates(candidates)
    keyword = rank_keyword_candidates(candidates)
    # Include the store's hybrid order as a third signal. RRF deduplicates IDs.
    fused = reciprocal_rank_fusion([dense, keyword, candidates])
    ranked = rerank(query, fused, limit=limit)
    diagnostics = {
        "dense_candidates": len(dense),
        "keyword_candidates": len(keyword),
        "fusion_candidates": len(fused),
        "reranked_candidates": len(ranked),
        "retrieval_routes": ["dense_view", "keyword_view", "hybrid_store_order"],
        "candidate_pool_shared": True,
    }
    return ranked, diagnostics


def rank_candidates(query: str, candidates: list[dict], limit: int | None = None) -> list[dict]:
    """Backward-compatible ranking API."""
    return rank_candidates_with_diagnostics(query, candidates, limit=limit)[0]
