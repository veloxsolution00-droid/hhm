"""Offline retrieval evaluation against the real indexed corpus.

This script calls the SAME production components the chat path calls, in the
same order (normalize -> vocabulary correction -> hybrid search -> min-similarity
gate -> rank_candidates -> select_evidence). It never generates text, so it
isolates retrieval/ranking/evidence-selection from LLM quality.

Outputs evaluation/audit/retrieval_results.json and a printed summary.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.rag import chat  # noqa: E402
from app.rag.planner import classify_question, correct_query_from_vocabulary, query_variants  # noqa: E402
from app.retrieval.pipeline import rank_candidates  # noqa: E402
from app.retrieval.evidence_selector import select_evidence  # noqa: E402
from app.retrieval.store import search  # noqa: E402
from app.services.documents import list_docs  # noqa: E402
from app.storage.database import indexed_vocabulary  # noqa: E402

QUESTIONS = Path(__file__).with_name("questions.json")
OUT = Path(__file__).with_name("retrieval_results.json")
KS = (1, 3, 4, 8, 20)


def doc_key(filename: str) -> str:
    name = (filename or "").lower()
    if "dracula" in name:
        return "dracula"
    if "franklin" in name:
        return "franklin"
    if "enchiridion" in name:
        return "enchiridion"
    if "science-of-getting-rich" in name:
        return "science"
    if "think-and-grow" in name:
        return "hill"
    return name


def allowed_docs(item):
    if item.get("docs"):
        return set(item["docs"])
    return {item["doc"]}


def is_relevant(hit, item):
    meta = hit.get("metadata") or {}
    if doc_key(meta.get("filename")) not in allowed_docs(item):
        return False
    pages = item.get("pages") or []
    if not pages:
        return False
    return str(meta.get("source_location")) in pages


def term_hits(hits, item):
    text = " ".join(str((h.get("text") or "")).lower() for h in hits)
    terms = [t.lower() for t in (item.get("evidence_terms") or [])]
    present = [t for t in terms if t in text]
    return len(present), len(terms), present


def run_case(item):
    original = item["question"]
    # Conversational items are evaluated on their resolved standalone query,
    # because that is what the follow-up resolver produces in production.
    query = item.get("resolved_query") or original
    normalized = chat.normalize_question(query)
    vocabulary = []
    try:
        vocabulary = indexed_vocabulary()
    except Exception:
        pass
    corrected = None
    misspelled = None
    if vocabulary:
        corrected = correct_query_from_vocabulary(normalized, vocabulary, cutoff=0.80)
        if corrected and corrected.strip().casefold() != normalized.strip().casefold():
            misspelled = normalized
            normalized = corrected

    task_type = classify_question(normalized)
    analysis = chat._analyze_question(normalized, task_type, query)
    params = chat._get_retrieval_params(task_type, analysis["analysis"])
    indexed = {
        doc["document_id"]
        for doc in list_docs("local")
        if doc.get("processing_status") == "indexed" and doc.get("indexing_status") == "complete"
    }

    started = time.perf_counter()
    raw = search(normalized, owner_id="local")
    all_hits = list(raw)
    passes = 1
    if normalized != query:
        all_hits.extend(search(query, owner_id="local"))
        passes += 1

    dedup = {}
    for hit in all_hits:
        key = hit.get("id", ((hit.get("metadata") or {}).get("document_id"), hit.get("text")))
        if key not in dedup or hit.get("similarity", 0) > dedup[key].get("similarity", 0):
            dedup[key] = hit
    all_hits = list(dedup.values())

    eligible = [
        h for h in all_hits
        if h.get("similarity", 0) >= params["min_similarity"]
        and (h.get("metadata") or {}).get("document_id") in indexed
    ]
    best_before_gate = max((float(h.get("similarity") or 0) for h in all_hits), default=0.0)
    lexical_coverage = max((float(h.get("lexical_similarity") or 0) for h in eligible), default=0.0)
    ordered = sorted((float(h.get("similarity") or 0) for h in eligible), reverse=True)
    close_competitors = len(ordered) > 1 and ordered[0] - ordered[1] < 0.035 and len(normalized.split()) >= 5
    expansion = (
        len(eligible) < max(1, params["top_k"])
        or (ordered[0] if ordered else 0.0) < min(1.0, params["min_similarity"] + 0.10)
        or (len(normalized.split()) >= 4 and lexical_coverage < 0.35)
        or close_competitors
        or (re.search(r"\b(?:club|alliance|group|society|association)\b", normalized, re.I)
            and re.search(r"\b(?:found|founded|purpose|serve|served|organized|formed|describe|described)\b", normalized, re.I))
    )
    variants = []
    if expansion:
        original_forms = {normalized.strip().lower(), query.strip().lower()}
        variants = [v for v in query_variants(normalized)[1:] if v.strip().lower() not in original_forms]
        if vocabulary:
            c2 = correct_query_from_vocabulary(normalized, vocabulary, cutoff=0.80)
            if c2.strip().lower() not in original_forms and c2.lower() not in {v.lower() for v in variants}:
                variants.append(c2)
    by_id = {h.get("id", ((h.get("metadata") or {}).get("document_id"), h.get("text"))): h for h in all_hits}
    for variant in variants:
        passes += 1
        for hit in search(variant, owner_id="local"):
            key = hit.get("id", ((hit.get("metadata") or {}).get("document_id"), hit.get("text")))
            if key not in by_id or hit.get("similarity", 0) > by_id[key].get("similarity", 0):
                by_id[key] = hit
    all_hits = list(by_id.values())
    eligible = [
        h for h in all_hits
        if h.get("similarity", 0) >= params["min_similarity"]
        and (h.get("metadata") or {}).get("document_id") in indexed
    ]
    scope = chat._named_document_scope(normalized, eligible)
    scoped = None
    if scope:
        kept = [h for h in eligible if (h.get("metadata") or {}).get("document_id") in scope]
        if kept:
            eligible = kept
            scoped = sorted({str((h.get("metadata") or {}).get("filename")) for h in kept})

    ranked = rank_candidates(normalized, eligible)
    context_budget = min(
        int(settings.max_context_chars),
        max(1800, int((settings.model_context_tokens - settings.context_safety_tokens
                       - settings.answer_max_tokens) * settings.context_chars_per_token)),
    )
    selection = select_evidence(normalized, ranked, max_chunks=max(1, params["top_k"]), max_chars=context_budget)
    selected = selection["hits"]
    elapsed = time.perf_counter() - started

    ranks = [rank for rank, hit in enumerate(ranked, 1) if is_relevant(hit, item)]
    recall = {}
    for k in KS:
        recall[f"recall@{k}"] = 1.0 if any(r <= k for r in ranks) else 0.0
    mrr = 1.0 / ranks[0] if ranks else 0.0
    selected_matches = [h for h in selected if is_relevant(h, item)]
    pages_found = {str((h.get("metadata") or {}).get("source_location")) for h in selected_matches}
    expected_pages = set(item.get("pages") or [])
    n_present, n_terms, present_terms = term_hits(selected, item)

    return {
        "id": item["id"],
        "category": item["category"],
        "question": original,
        "query_used": query,
        "normalized": normalized,
        "spelling_correction": {"from": misspelled, "to": normalized} if misspelled else None,
        "task_type": task_type,
        "top_k": params["top_k"],
        "retrieval_passes": passes,
        "expansion_triggered": expansion,
        "variants": variants,
        "document_scope": scoped,
        "candidates_total": len(all_hits),
        "eligible_after_min_similarity": len(eligible),
        "best_similarity_before_gate": round(best_before_gate, 4),
        "min_similarity": params["min_similarity"],
        "ranked": [
            {
                "rank": rank,
                "doc": doc_key((h.get("metadata") or {}).get("filename")),
                "page": (h.get("metadata") or {}).get("source_location"),
                "filename": (h.get("metadata") or {}).get("filename"),
                "similarity": round(float(h.get("similarity") or 0), 4),
                "semantic": None if h.get("semantic_similarity") is None else round(float(h["semantic_similarity"]), 4),
                "lexical": round(float(h.get("lexical_similarity") or 0), 4),
                "relevant": is_relevant(h, item),
                "snippet": " ".join(str(h.get("text") or "").split())[:180],
            }
            for rank, h in enumerate(ranked[:20], 1)
        ],
        "selected": [
            {
                "doc": doc_key((h.get("metadata") or {}).get("filename")),
                "page": (h.get("metadata") or {}).get("source_location"),
                "similarity": round(float(h.get("similarity") or 0), 4),
                "relevant": is_relevant(h, item),
                "snippet": " ".join(str(h.get("text") or "").split())[:180],
                "text": " ".join(str(h.get("text") or "").split()),
            }
            for h in selected
        ],
        "metrics": {
            **recall,
            "mrr": round(mrr, 4),
            "recall_selected": 1.0 if selected_matches else 0.0,
            "precision_selected": round(len(selected_matches) / len(selected), 4) if selected else 0.0,
            "page_coverage_selected": round(len(pages_found & expected_pages) / len(expected_pages), 4) if expected_pages else None,
            "pages_found": sorted(pages_found),
            "expected_pages": sorted(expected_pages),
            "term_coverage_selected": round(n_present / n_terms, 4) if n_terms else None,
            "terms_present": present_terms,
            "terms_expected": n_terms,
            "no_evidence_after_gate": len(eligible) == 0,
        },
        "retrieval_seconds": round(elapsed, 3),
    }


def main():
    payload = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    results = []
    for item in payload["questions"]:
        try:
            result = run_case(item)
        except Exception as exc:  # noqa: BLE001
            result = {"id": item["id"], "category": item["category"], "question": item["question"],
                      "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        metrics = result.get("metrics") or {}
        print(f'{result["id"]:<4} {result["category"]:<26} '
              f'rec@sel={metrics.get("recall_selected")} mrr={metrics.get("mrr")} '
              f'prec={metrics.get("precision_selected")} '
              f'cov={metrics.get("page_coverage_selected")} '
              f'sim={result.get("best_similarity_before_gate")} '
              f't={result.get("retrieval_seconds")}s', flush=True)

    OUT.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "config": {"top_k": settings.top_k,
                                          "min_similarity": settings.min_similarity,
                                          "lexical_weight": settings.hybrid_lexical_weight,
                                          "embedding_model": settings.embedding_model},
                               "results": results}, indent=2), encoding="utf-8")

    answerable = [r for r in results if "error" not in r and r["metrics"]["expected_pages"]]
    unanswerable = [r for r in results if "error" not in r and not r["metrics"]["expected_pages"]]
    print("\n=== SUMMARY ===")
    print(f"cases={len(results)} answerable={len(answerable)} unanswerable={len(unanswerable)}")
    for key in ("recall@1", "recall@3", "recall@4", "recall@8", "recall@20", "recall_selected"):
        value = statistics.mean(r["metrics"][key] for r in answerable) if answerable else 0
        print(f"{key:<18} {value:.3f}")
    print(f'MRR                {statistics.mean(r["metrics"]["mrr"] for r in answerable):.3f}')
    print(f'precision@sel      {statistics.mean(r["metrics"]["precision_selected"] for r in answerable):.3f}')
    cov = [r["metrics"]["page_coverage_selected"] for r in answerable if r["metrics"]["page_coverage_selected"] is not None]
    print(f'page_coverage@sel  {statistics.mean(cov):.3f}' if cov else "page_coverage n/a")
    tc = [r["metrics"]["term_coverage_selected"] for r in answerable if r["metrics"]["term_coverage_selected"] is not None]
    print(f'term_coverage@sel  {statistics.mean(tc):.3f}' if tc else "term_coverage n/a")
    secs = sorted(r["retrieval_seconds"] for r in answerable)
    if secs:
        print(f'retrieval secs     mean={statistics.mean(secs):.2f} p50={statistics.median(secs):.2f} '
              f'p90={secs[int(len(secs)*0.9)-1]:.2f} max={secs[-1]:.2f}')
    print(f'empty-after-gate   {sum(1 for r in answerable if r["metrics"]["no_evidence_after_gate"])}')
    by_cat = defaultdict(list)
    for r in answerable:
        by_cat[r["category"]].append(r["metrics"]["recall_selected"])
    print("\nrecall_selected by category:")
    for cat, vals in sorted(by_cat.items()):
        print(f"  {cat:<26} {statistics.mean(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
