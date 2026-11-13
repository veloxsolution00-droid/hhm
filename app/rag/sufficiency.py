"""Deterministic, measurable evidence sufficiency (evidence concept D).

The pipeline keeps four evidence concepts separate:

* citation integrity   -> :mod:`app.rag.citations`
* claim-level support  -> :mod:`app.rag.verification`
* completeness         -> :mod:`app.reasoning.answer_planner`
* sufficiency          -> this module

:func:`assess_evidence_sufficiency` answers one pre-generation question: *is the
retrieved evidence strong enough to attempt an answer at all, and how much
qualification does the answer need?*  It is deterministic and model-free, so it
runs on every question inside the fixed 8 GB / CPU / single-concurrency budget
and never adds a generation call.  It returns exactly one state:

``sufficient``
    the evidence covers the question; answer normally.
``partially_sufficient``
    the evidence covers only part of the question; answer the supported part and
    state what is not established.
``insufficient``
    the asked subject is essentially absent from the evidence; abstain before
    generation instead of letting the model guess.
``contradictory``
    usable evidence disagrees; resolve or qualify it.
``irrelevant``
    nothing usable was retrieved; abstain before generation.

Four signals are combined, per the phase plan: maximum hybrid similarity,
salient-entity (anchor) presence, term/requirement coverage and contradiction
detection.  No single signal is decisive on its own.  An unanswerable question
often retrieves topically coherent passages, so a high similarity score alone
cannot license an answer; and a question whose wording differs from the document
(a plural, a past tense) must not be judged absent merely because the surface
form differs, so term matching is stem-aware.
"""
from __future__ import annotations

import re

from app.reasoning.conflict_resolver import inspect_conflicts
from app.retrieval.evidence_selector import _terms

# Deterministic thresholds. They are module constants rather than environment
# flags so the behaviour is identical in every deployment and reproducible
# offline from the frozen evaluation results.
_RELEVANCE_FLOOR = 0.32        # below this a hit is not treated as related
_INSUFFICIENT_COVERAGE = 0.20  # the question's content terms are essentially absent
_WEAK_COVERAGE = 0.40          # partial coverage only counts when relevance is weak too
_WEAK_SIMILARITY = 0.45
_PARTIAL_COVERAGE = 0.55       # uncovered asks above this need qualification
_ANCHOR_FLOOR = 0.50           # missing salient anchors need qualification

SUFFICIENT = "sufficient"
PARTIALLY_SUFFICIENT = "partially_sufficient"
INSUFFICIENT = "insufficient"
CONTRADICTORY = "contradictory"
IRRELEVANT = "irrelevant"

# Interrogatives, request verbs and generic scaffolding are not the subject of a
# question. They are removed from the question before coverage is measured, so
# "why" or "many" never has to appear in the evidence, and a capitalised word
# only counts as a salient anchor when it is not one of these.
_GATE_STOP = frozenset({
    "about", "according", "again", "also", "among", "answer", "any", "because",
    "been", "before", "being", "between", "book", "books", "both", "came",
    "cannot", "chapter", "chapters", "could", "describe", "did", "does",
    "doing", "done", "during", "each", "either", "else", "explain", "first",
    "from", "give", "given", "gives", "into", "its", "just", "kind", "kinds",
    "many", "mention", "might", "more", "most", "much", "must", "need",
    "needs", "neither", "next", "once", "only", "other", "others", "please",
    "provide", "rather", "really", "said", "same", "say", "says", "second",
    "shall", "should", "since", "some", "state", "stated", "states", "still",
    "such", "summary", "summarise", "summarize", "tell", "than", "that",
    "their", "them", "then", "there", "these", "they", "thing", "things",
    "third", "this", "those", "though", "thus", "very", "want", "wants",
    "well", "were", "what", "when", "where", "whether", "which", "while",
    "who", "whom", "whose", "why", "will", "with", "within", "without",
    "would", "your",
})

_SPLIT_TERMS = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _stem(token: str) -> str:
    """Light, deterministic suffix normalisation for term matching.

    It only ever makes two surface forms of the same word compare equal (plural,
    past tense, gerund, trailing ``e``); it never merges unrelated words.
    """
    token = token.casefold()
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def _normalise_token(token: str) -> str:
    token = token.strip().strip("'\"\u201c\u201d\u2018\u2019().,;:!?").casefold()
    if token.endswith("'s") or token.endswith("\u2019s"):
        token = token[:-2]
    return token


def _content_terms(question: str) -> set[str]:
    """The question's subject terms: content words, stemming-normalised."""
    return {_stem(term) for term in _terms(question or "") if term not in _GATE_STOP}


def _salient_anchors(text: str) -> set[str]:
    """Proper nouns, numbers and quoted terms that anchor a question."""
    anchors: set[str] = set()
    for match in re.findall(r"\b[A-Z][A-Za-z'\u2019\-]{2,}(?:\s+[A-Z][A-Za-z'\u2019\-]{2,})*", text or ""):
        for token in match.split():
            token = _normalise_token(token)
            if token and token not in _GATE_STOP:
                anchors.add(token)
    for number in re.findall(r"\d[\d,\.]*", text or ""):
        anchors.add(number.rstrip(".").replace(",", ""))
    for quoted in re.findall(r'"([^"]{3,40})"', text or ""):
        anchors.update(_terms(quoted))
    return anchors


def _contains(term: str, text: str) -> bool:
    """Whole-word match, so a term never matches inside a longer word."""
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _present(anchor: str, text: str, numbers_only_text: str) -> bool:
    if _NUMBER.fullmatch(anchor or ""):
        return re.search(rf"(?<!\d){re.escape(anchor)}(?!\d)", numbers_only_text) is not None
    return _contains(anchor, text)


def _verdict(state, best_similarity, term_coverage, anchor_coverage, conflicts, reasons) -> dict:
    return {
        "state": state,
        "abstain": state in (INSUFFICIENT, IRRELEVANT),
        "qualify": state in (PARTIALLY_SUFFICIENT, CONTRADICTORY),
        "best_similarity": None if best_similarity is None else round(float(best_similarity), 4),
        "term_coverage": None if term_coverage is None else round(float(term_coverage), 4),
        "anchor_coverage": None if anchor_coverage is None else round(float(anchor_coverage), 4),
        "conflicts": int(conflicts or 0),
        "reasons": reasons,
    }


def assess_evidence_sufficiency(question: str, hits: list[dict] | None) -> dict:
    """Classify retrieved evidence into one deterministic sufficiency state.

    ``hits`` must be the passages actually selected for the answer; they are the
    same records the generator receives. The function performs no retrieval,
    indexing or model call, so it is cheap and exactly reproducible.
    """
    hits = [hit for hit in (hits or []) if str(hit.get("text") or "").strip()]
    if not hits:
        return _verdict(IRRELEVANT, None, None, None, 0, ["no retrieved evidence"])

    evidence = " ".join(str(hit.get("text") or "") for hit in hits)
    lowered = evidence.casefold()
    digits_only = lowered.replace(",", "")

    best_similarity = max((_as_float(hit.get("similarity")) for hit in hits), default=0.0)

    required = _content_terms(question)
    evidence_stems = {_stem(token) for token in _SPLIT_TERMS.findall(lowered)}
    covered = {term for term in required if term in evidence_stems}
    term_coverage = len(covered) / len(required) if required else None

    anchors = _salient_anchors(question or "")
    anchor_coverage = None
    if anchors:
        anchor_coverage = sum(1 for a in anchors if _present(a, lowered, digits_only)) / len(anchors)

    conflicts = len(inspect_conflicts(hits).get("potential_conflicts") or [])

    reasons: list[str] = []
    if best_similarity < _RELEVANCE_FLOOR and (term_coverage or 0.0) < _INSUFFICIENT_COVERAGE:
        state = IRRELEVANT
        reasons.append("best similarity and question-term coverage are both below the usable floor")
    elif term_coverage is not None and term_coverage < _INSUFFICIENT_COVERAGE:
        state = INSUFFICIENT
        reasons.append(f"only {term_coverage:.0%} of the question's content terms appear in the evidence")
    elif (term_coverage is not None and term_coverage < _WEAK_COVERAGE
          and best_similarity < _WEAK_SIMILARITY):
        state = INSUFFICIENT
        reasons.append(
            f"weak relevance ({best_similarity:.2f}) with only {term_coverage:.0%} term coverage")
    elif conflicts:
        state = CONTRADICTORY
        reasons.append(f"{conflicts} potential evidence disagreement(s)")
    elif ((term_coverage is not None and term_coverage < _PARTIAL_COVERAGE)
          or (anchor_coverage is not None and anchor_coverage < _ANCHOR_FLOOR)):
        state = PARTIALLY_SUFFICIENT
        if term_coverage is not None and term_coverage < _PARTIAL_COVERAGE:
            reasons.append(
                f"{term_coverage:.0%} question-term coverage leaves part of the question open")
        if anchor_coverage is not None and anchor_coverage < _ANCHOR_FLOOR:
            reasons.append(
                f"only {anchor_coverage:.0%} of the question's salient anchors appear in the evidence")
    else:
        state = SUFFICIENT
        reasons.append("evidence covers the question's content terms and salient anchors")

    return _verdict(state, best_similarity, term_coverage, anchor_coverage, conflicts, reasons)
