"""Cheap, deterministic document-type detection for type-aware ingestion.

The RAG pipeline was tuned for flowing prose. Invoices, contracts, spreadsheets
and question sheets need different extraction, chunking and answer guidance.
This module only classifies; it never rewrites the text.
"""
from __future__ import annotations

import re
from pathlib import Path

DOCUMENT_TYPES = (
    "invoice",
    "legal",
    "question_bank",
    "spreadsheet",
    "report",
    "prose",
    "generic",
)

_SPREADSHEET_EXT = {".xlsx", ".xlsm", ".ods", ".csv"}

_INVOICE_PATTERNS = (
    r"\binvoice\b", r"\btax invoice\b", r"\bbill(?:ed)? to\b", r"\bship to\b",
    r"\bamount due\b", r"\bbalance due\b", r"\bsubtotal\b", r"\btotal due\b",
    r"\binvoice (?:no|number|#)\b", r"\bpo (?:no|number)\b", r"\bpurchase order\b",
    r"\bdue date\b", r"\bpayment terms\b", r"\bvat\b", r"\bunit price\b",
    r"\bquantity\b", r"\bqty\b", r"\bremit to\b", r"\bcurrency\b",
)
_LEGAL_PATTERNS = (
    r"\bagreement\b", r"\bwhereas\b", r"\bhereinafter\b", r"\bhereby\b",
    r"\bwitnesseth\b", r"\bparty\b", r"\bparties\b", r"\bclause\b",
    r"\bsection\b", r"\bgoverning law\b", r"\bjurisdiction\b", r"\btermination\b",
    r"\bindemnif(?:y|ication)\b", r"\bliabilit(?:y|ies)\b", r"\bconfidentiality\b",
    r"\beffective date\b", r"\bforce majeure\b", r"\bnotice period\b",
    r"\bterms and conditions\b", r"\bbreach\b",
)
_REPORT_PATTERNS = (
    r"\bexecutive summary\b", r"\bmethodology\b", r"\bfindings\b",
    r"\bconclusion\b", r"\brecommendations?\b", r"\bappendix\b",
    r"\bbackground\b", r"\bresults\b",
)

_INVOICE_NAME = re.compile(r"invoice|receipt|statement|bill|quote|purchase|order|ledger", re.I)
_LEGAL_NAME = re.compile(r"contract|agreement|legal|terms|policy|nda|lease|deed|statute", re.I)

_SENTENCE_SPLIT = re.compile(r"[.!?]+")


def _count_patterns(text: str, patterns) -> tuple[int, list[str]]:
    hits = []
    for pattern in patterns:
        if re.search(pattern, text):
            hits.append(pattern)
    return len(hits), hits


def _question_ratio(text: str) -> tuple[float, int]:
    """Share of sentence terminators that are question marks.

    A question bank ("Which systems are in scope? How are alerts triaged?") has
    almost no declarative sentences, so no evidence can ever answer from it.
    """
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return 0.0, 0
    markers = re.findall(r"[.!?]+", text)
    if not markers:
        return 0.0, len(sentences)
    questions = sum(1 for marker in markers if "?" in marker)
    return questions / len(markers), len(sentences)


def detect_document_type(text: str, filename: str = "") -> dict:
    """Return {"type", "confidence", "signals"} without raising on odd input."""
    sample = (text or "")[:60000]
    low = sample.lower()
    name = Path(filename or "").name
    suffix = Path(name).suffix.lower()

    if suffix in _SPREADSHEET_EXT:
        return {"type": "spreadsheet", "confidence": 0.99,
                "signals": {"reason": "spreadsheet extension"}}

    question_ratio, sentence_count = _question_ratio(sample)
    if question_ratio >= 0.6 and sentence_count >= 4:
        return {"type": "question_bank", "confidence": round(min(0.99, question_ratio), 3),
                "signals": {"question_ratio": round(question_ratio, 3),
                            "sentences": sentence_count}}

    invoice_score, invoice_hits = _count_patterns(low, _INVOICE_PATTERNS)
    legal_score, legal_hits = _count_patterns(low, _LEGAL_PATTERNS)
    report_score, report_hits = _count_patterns(low, _REPORT_PATTERNS)

    # Filenames are a strong prior for user-named files ("Invoice-2041.pdf").
    if _INVOICE_NAME.search(name):
        invoice_score += 2
    if _LEGAL_NAME.search(name):
        legal_score += 2

    scores = {"invoice": invoice_score, "legal": legal_score, "report": report_score}
    best_type, best_score = max(scores.items(), key=lambda item: item[1])

    tabular_lines = [line for line in sample.splitlines()
                     if line.count(" | ") >= 2 or len(re.findall(r"\S\s{2,}\S", line)) >= 2]
    non_empty = [line for line in sample.splitlines() if line.strip()]
    tabular_ratio = len(tabular_lines) / len(non_empty) if non_empty else 0.0

    # Book-length narrative metric, computed for both the guard below and the
    # prose classification that follows it.
    words = re.findall(r"[A-Za-z']+", sample)
    words_per_sentence = len(words) / max(1, sentence_count) if sentence_count else 0
    narrative = len(words) >= 200 and words_per_sentence >= 8 and tabular_ratio < 0.3

    if best_score >= 2 and (not narrative or best_score >= 4):
        # A narrative document (an autobiography, a novel, a long essay) uses
        # words that also occur in contracts and invoices -- "party"/"parties"
        # are ordinary narrative words and "hereby"/"recommendations" appear in
        # formal writing. Two or three such hits must not turn an autobiography
        # into a legal document, because the answer stage would then demand
        # governing clauses that do not exist in prose. Only a text that is not
        # clearly narrative, or one that shows a dense cluster of >= 4 pattern
        # hits, is allowed to carry the invoice/legal/report type.
        # Ties are common on "terms and conditions"; prefer the richer signal.
        if invoice_score == legal_score and invoice_score >= 2:
            best_type = "invoice" if _INVOICE_NAME.search(name) else "legal"
        confidence = min(0.99, 0.55 + 0.09 * best_score)
        return {"type": best_type, "confidence": round(confidence, 3),
                "signals": {"invoice_score": invoice_score, "legal_score": legal_score,
                            "report_score": report_score, "tabular_ratio": round(tabular_ratio, 3),
                            "invoice_hits": invoice_hits, "legal_hits": legal_hits,
                            "report_hits": report_hits}}

    if tabular_ratio >= 0.5:
        return {"type": "spreadsheet", "confidence": 0.7,
                "signals": {"reason": "tabular layout", "tabular_ratio": round(tabular_ratio, 3)}}

    # Prose is the safe default for books and long narrative documents: it keeps
    # the existing behaviour rather than inventing structure that is not there.
    if len(words) >= 200 and words_per_sentence >= 8:
        return {"type": "prose", "confidence": 0.6,
                "signals": {"words": len(words), "sentences": sentence_count,
                            "words_per_sentence": round(len(words) / max(1, sentence_count), 2)}}

    return {"type": "generic", "confidence": 0.4,
            "signals": {"words": len(words), "sentences": sentence_count}}
