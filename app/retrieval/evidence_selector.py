"""Evidence selection balancing relevance, coverage and redundancy."""
import re
from difflib import SequenceMatcher

_STOP = {"the", "and", "for", "with", "what", "when", "where", "which", "who",
         "does", "from", "that", "this", "are", "was", "how", "about", "into",
         "you", "document", "documents", "is", "a", "an", "of", "to", "in"}


def _terms(text):
    return {t.lower() for t in re.findall(r"[\w-]+", text or "", flags=re.UNICODE)
            if len(t) > 2 and t.lower() not in _STOP}


def select_evidence(requirement: str, hits: list[dict], max_chunks: int = 5,
                    max_chars: int = 9000) -> dict:
    required = _terms(requirement)
    pool = []
    seen = set()
    for hit in hits:
        key = hit.get("id") or ((hit.get("metadata") or {}).get("document_id"), hit.get("text"))
        if key in seen:
            continue
        seen.add(key)
        text = (hit.get("text") or "").strip()
        if not text:
            continue
        item = dict(hit)
        item["_terms"] = _terms(text)
        # Prefer the reranked score, falling back to raw similarity only when
        # the reranker did not run. Test for None explicitly: a legitimate
        # rerank_score of 0.0 must not fall through to a high `similarity` via
        # `or`, which would silently reorder evidence selection.
        base = item.get("rerank_score")
        if base is None:
            base = item.get("similarity")
        item["_base"] = float(base) if base is not None else 0.0
        pool.append(item)

    selected, selected_ids, covered, chars = [], set(), set(), 0
    max_chunks = max(1, int(max_chunks))
    while pool and len(selected) < max_chunks:
        best, best_score = None, float("-inf")
        for item in pool:
            text = item.get("text", "")
            key = item.get("id") or ((item.get("metadata") or {}).get("document_id"), text)
            if key in selected_ids:
                continue
            if len(text) > max_chars or chars + len(text) > max_chars:
                continue
            new_terms = (item["_terms"] & required) - covered
            coverage_gain = len(new_terms) / max(1, len(required))
            redundancy = max(
                (SequenceMatcher(None, text[:1200].lower(), old.get("text", "")[:1200].lower()).ratio()
                 for old in selected), default=0.0
            )
            # Relevance remains dominant; coverage helps select complementary evidence.
            score = 0.78 * item["_base"] + 0.27 * coverage_gain - 0.20 * redundancy
            if score > best_score:
                best, best_score = item, score
        if best is None:
            break
        selected.append(best)
        selected_ids.add(best.get("id") or ((best.get("metadata") or {}).get("document_id"), best.get("text")))
        covered.update(best["_terms"])
        chars += len(best.get("text", ""))
        pool.remove(best)

    for item in selected:
        item.pop("_terms", None)
        item.pop("_base", None)
    missing = sorted(required - covered)
    return {
        "hits": selected, "required_terms": sorted(required),
        "covered_terms": sorted(required & covered), "missing_terms": missing,
        "term_coverage": round(len(required & covered) / max(1, len(required)), 4),
        "evidence_found": bool(selected), "selected_characters": chars,
        "selection_strategy": "relevance_coverage_redundancy",
    }
