"""Evidence selection balancing relevance, coverage, redundancy and diversity."""
import re
from difflib import SequenceMatcher

_STOP = {"the", "and", "for", "with", "what", "when", "where", "which", "who",
         "does", "from", "that", "this", "are", "was", "how", "about", "into",
         "you", "document", "documents", "is", "a", "an", "of", "to", "in"}

# Questions that ask to compare, or to cover several documents/entities, benefit
# from spreading the selected evidence across documents instead of taking the
# highest-scoring chunks from a single one.
_MULTI_DOCUMENT = re.compile(
    r"\b(compare|comparison|contrast|differences?|differ|versus|vs\.?|each|"
    r"respectively|both|across|which documents?|every)\b", re.I)


def _terms(text):
    return {t.lower() for t in re.findall(r"[\w-]+", text or "", flags=re.UNICODE)
            if len(t) > 2 and t.lower() not in _STOP}


def _doc_key(item):
    meta = item.get("metadata") or {}
    return str(meta.get("document_id") or meta.get("filename") or "")


def _page_key(item):
    return str((item.get("metadata") or {}).get("source_location") or "")


def select_evidence(requirement: str, hits: list[dict], max_chunks: int = 5,
                    max_chars: int = 9000, multi_document: bool | None = None) -> dict:
    required = _terms(requirement)
    # Whether the answer must span documents is a property of the question, so
    # the policy is derived here (or supplied by the caller) rather than being a
    # fixed per-document cap.
    if multi_document is None:
        multi_document = bool(_MULTI_DOCUMENT.search(requirement or ""))
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

    max_base = max((item["_base"] for item in pool), default=0.0)
    selected, selected_ids, covered, chars = [], set(), set(), 0
    covered_docs, covered_pages = set(), set()
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
            # Document diversity: only for comparison / multi-document questions,
            # and scaled by how relevant the candidate is, so an unrelated chunk
            # is never pulled in merely to tick a document off. This is a soft
            # preference, not a per-document cap.
            doc = _doc_key(item)
            if multi_document and doc and doc not in covered_docs and max_base > 0:
                score += 0.30 * (item["_base"] / max_base)
            # Page diversity: a mild nudge to cover distinct pages of a
            # multi-page answer without overriding relevance.
            page = _page_key(item)
            if page and page not in covered_pages:
                score += 0.04
            if score > best_score:
                best, best_score = item, score
        if best is None:
            break
        selected.append(best)
        selected_ids.add(best.get("id") or ((best.get("metadata") or {}).get("document_id"), best.get("text")))
        covered.update(best["_terms"])
        covered_docs.add(_doc_key(best))
        covered_pages.add(_page_key(best))
        chars += len(best.get("text", ""))
        pool.remove(best)

    # The selection decides *which* passages are supplied; their presentation
    # order must stay the relevance order the reranker produced. The model cites
    # the small numbers in the order it reads the passages, so returning the
    # greedy coverage order here renumbered the evidence per question and made
    # the same draft's references point at different passages.
    selected.sort(key=lambda item: item["_base"], reverse=True)

    for item in selected:
        item.pop("_terms", None)
        item.pop("_base", None)
    missing = sorted(required - covered)
    return {
        "hits": selected, "required_terms": sorted(required),
        "covered_terms": sorted(required & covered), "missing_terms": missing,
        "term_coverage": round(len(required & covered) / max(1, len(required)), 4),
        "evidence_found": bool(selected), "selected_characters": chars,
        "multi_document": multi_document,
        "selection_strategy": "relevance_coverage_redundancy_diversity",
    }
