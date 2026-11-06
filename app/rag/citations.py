"""Citation reference validation against the exact evidence supplied to the model."""
from __future__ import annotations
import re

_SENTENCE = re.compile(r"[^.!?]+[.!?]?")
_NUMERIC_REF = re.compile(r"\[\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*\]")
_ANY_BRACKET = re.compile(r"\[([^\]\n]+)\]")
_FILENAME = re.compile(r"\.(?:pdf|docx?|txt|md|xlsx?|pptx?|csv|html?)\b", re.I)
_BRACKET_SEGMENT = re.compile(r"\[[^\]\n]*\]")
_URL = re.compile(
    r"\b(?:https?://|www\.)[^\s<>()]+"
    r"|\b(?:[A-Z0-9-]+\.)+[A-Z]{2,}(?:/[^\s<>()]*)?",
    re.I,
)
_CITATION = re.compile(
    r"\[\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*\]"
    r"|\[[^\]\n]*\.(?:pdf|docx?|txt|md|xlsx?|pptx?|csv|html?)[^\]\n]*\]",
    re.I,
)

# Bracketed text that is formatting rather than a source claim.
_BRACKET_ALLOW = {"sic", "note", "table", "page", "figure", "excerpt", "emphasis added"}


def _sentences(text: str) -> list[str]:
    """Split into sentences without breaking citations or URLs.

    A filename citation such as "[policy.pdf, page 2]" contains a full stop that
    is not a sentence boundary.
    """
    text = text or ""
    stashed: dict[str, str] = {}

    def stash(match):
        key = f"\x00{len(stashed)}\x00"
        stashed[key] = match.group(0)
        return key

    masked = _BRACKET_SEGMENT.sub(stash, text)

    def stash_url(match):
        value = match.group(0)
        trailing = ""
        while value and value[-1] in ".,;:!?":
            trailing = value[-1] + trailing
            value = value[:-1]
        key = f"\x00{len(stashed)}\x00"
        stashed[key] = value
        return key + trailing

    # Domain periods are part of URLs, not sentence boundaries.
    masked = _URL.sub(stash_url, masked)
    parts = [part.strip() for part in _SENTENCE.findall(masked) if part.strip()]
    restored = []
    for part in parts:
        for key, original in stashed.items():
            part = part.replace(key, original)
        restored.append(part)
    return restored


def _content_words(sentence: str) -> int:
    """Count substantive words, ignoring bracketed citations and digits."""
    stripped = _BRACKET_SEGMENT.sub(" ", sentence or "")
    return len(re.findall(r"[A-Za-z]{2,}", stripped))


def cite_verbatim_urls(answer: str, sources: list[dict]) -> str:
    """Attach evidence refs to URL sentences only when the URL is verbatim evidence."""
    if not answer or not sources:
        return answer or ""
    sentences = _sentences(answer)
    repaired = []
    for sentence in sentences:
        urls = [m.group(0).rstrip(".,;:!?]}").casefold() for m in _URL.finditer(sentence)]
        if urls and not _NUMERIC_REF.search(sentence):
            refs = []
            for index, source in enumerate(sources, 1):
                source_text = (source.get("text") or "").casefold()
                if any(url in source_text for url in urls):
                    refs.append(index)
            if refs:
                citation = "[" + ", ".join(map(str, refs)) + "]"
                sentence = re.sub(r"([.!?])([\"'”’)]*)$", citation + r"\1\2", sentence)
                if not re.search(r"[.!?][\"'”’)]*$", sentence):
                    sentence += " " + citation
        repaired.append(sentence)
    return " ".join(repaired)


_CITATION_MATCH_STOP = {
    "about", "after", "again", "also", "and", "are", "because", "been", "being",
    "between", "both", "but", "can", "could", "does", "each", "for", "from", "had",
    "has", "have", "into", "its", "more", "most", "not", "only", "other", "our",
    "over", "same", "some", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "under", "was", "were",
    "what", "when", "where", "which", "who", "will", "with", "would", "your",
}


def cite_supported_uncited_sentences(answer: str, sources: list[dict]) -> str:
    """Attach a source only when an uncited factual sentence matches its text.

    This is a formatting recovery for omitted citations, not a support verdict:
    low lexical matches are left untouched and the normal verifier still has to
    accept the cited claim. It never invents a new claim or a source number.
    """
    if not answer or not sources:
        return answer or ""

    repaired = []
    for sentence in _sentences(answer):
        existing_refs = [int(value) for group in re.findall(
            r"\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*\]", sentence)
                         for value in group.split(",")]
        invalid_refs = [value for value in existing_refs if value < 1 or value > len(sources)]
        if invalid_refs:
            # Models sometimes copy page numbers as reference IDs. Discard only
            # out-of-range numeric refs; the source is reselected below from
            # sentence-to-passage lexical support.
            sentence = re.sub(r"\[\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*\]", "", sentence)
        elif _CITATION.search(sentence) or _content_words(sentence) < 3:
            repaired.append(sentence)
            continue
        claim_terms = {
            term.casefold() for term in re.findall(r"[A-Za-z]{3,}",
                                                     _BRACKET_SEGMENT.sub(" ", sentence))
            if term.casefold() not in _CITATION_MATCH_STOP
        }
        if len(claim_terms) < 2:
            repaired.append(sentence)
            continue
        ranked_sources = []
        for index, source in enumerate(sources, 1):
            text = source.get("text") or source.get("supporting_text") or ""
            source_terms = {term.casefold() for term in re.findall(r"[A-Za-z]{3,}", text)}
            overlap = claim_terms & source_terms
            coverage = len(overlap) / len(claim_terms)
            ranked_sources.append((index, overlap, coverage))
        remaining = set(claim_terms)
        selected = []
        while remaining and len(selected) < 4:
            best = max(ranked_sources,
                       key=lambda item: (len(item[1] & remaining), item[2]),
                       default=None)
            if best is None:
                break
            source_number, overlap, _ = best
            useful = overlap & remaining
            if len(useful) < 2:
                break
            selected.append(source_number)
            remaining.difference_update(useful)
            ranked_sources = [item for item in ranked_sources if item[0] != source_number]
            if (len(claim_terms) - len(remaining)) / len(claim_terms) >= 0.7:
                break
        if selected and (len(claim_terms) - len(remaining)) / len(claim_terms) >= 0.25:
            citation = "[" + ", ".join(map(str, selected)) + "]"
            sentence = re.sub(r"([.!?])([\"'â€â€™)]*)$",
                              f" {citation}\\1\\2", sentence)
            if not _NUMERIC_REF.search(sentence):
                sentence += f" {citation}"
        repaired.append(sentence)
    return " ".join(repaired)


def validate_citation_references(answer: str, source_count: int, source_labels: list[str] | None = None) -> dict:
    """Validate numeric evidence references [1], [2], etc.; report, do not guess."""
    answer = answer or ""
    # Accept both individual references ([1] [2]) and a comma-separated list
    # ([1, 2]). Small local models often vary this formatting despite the prompt.
    refs = []
    for match in re.finditer(r"\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*\]", answer):
        refs.extend(int(value.strip()) for value in match.group(1).split(","))
    invalid = sorted({n for n in refs if n < 1 or n > source_count})
    valid = sorted({n for n in refs if 1 <= n <= source_count})
    labels = source_labels or []
    # Catch explicit filename-style citations that name a source not in the evidence.
    known = {str(x).lower() for x in labels if x}
    filename_cites = re.findall(
        r"(?:\[|\()([^\]\()\n]+\.(?:pdf|docx?|txt|md|xlsx?|pptx?|csv|html?))(?:,|\]|\))",
        answer, flags=re.I)
    unknown_labels = sorted({name.strip() for name in filename_cites if name.strip().lower() not in known})
    # Bracketed text that is neither a numeric reference nor a known filename is
    # a phantom citation: small models sometimes copy editorial markers from the
    # source (for example "[UVAstudstaff]") and present them as sources.
    phantom = []
    for match in _ANY_BRACKET.finditer(answer):
        content = match.group(1).strip()
        if not content or _NUMERIC_REF.fullmatch(match.group(0)):
            continue
        # Markdown links use brackets for their visible URL label; they are
        # links, not fabricated source labels. Their factual claims still need
        # a numeric evidence citation, checked independently below.
        after = answer[match.end():]
        if (_URL.fullmatch(content) or re.match(r"\s*\]\(", after)):
            continue
        if content.lower() in _BRACKET_ALLOW or _FILENAME.search(content):
            continue
        phantom.append(content)
    phantom = sorted(set(phantom))
    # Report citation density: a long factual sentence with no reference at all
    # is a coverage gap even when the answer contains one valid citation.
    factual = [s for s in _sentences(answer) if _content_words(s) >= 3]
    uncited = [s[:120] for s in factual if not _CITATION.search(s)]
    coverage = round(1 - (len(uncited) / len(factual)), 3) if factual else 0.0
    present = bool(refs or filename_cites)
    return {
        "valid": not invalid and not unknown_labels and not phantom and present and coverage == 1.0,
        "references_found": valid,
        "invalid_reference_numbers": invalid,
        "unknown_source_labels": unknown_labels,
        "phantom_labels": phantom,
        "uncited_sentences": uncited,
        "citation_present": present,
        "citation_coverage": coverage,
    }
