"""Deterministic local reranker using relevance, coverage and answer-bearing cues."""
import re

_STOP = {"the", "and", "for", "with", "what", "when", "where", "which", "who", "does",
         "from", "that", "this", "are", "was", "how", "about", "into", "you", "is",
         "a", "an", "of", "to", "in", "on", "it", "be", "by", "as"}
# Editorial, publisher and back-matter apparatus. This is not the source
# passage itself, so it must not outrank it merely because it repeats the
# subject's name; it is demoted unless the question is actually about it.
_APPARATUS = re.compile(
    r"(about this edition|copyright|isbn|transcription, correction|"
    r"editorial commentary|\[UVAstudstaff\]|^\s*glossary\s*$|^\s*index\s*$|"
    r"all rights reserved|read and download free books|downloaded from [a-z0-9.-]+)",
    re.I | re.M
)
# Structural front matter. A table of contents legitimately answers chapter or
# title questions, so it is demoted only when it barely relates to the query
# rather than merely because it is structural.
_STRUCTURE = re.compile(r"table of contents|^\s*contents\s*$", re.I | re.M)
# Vocabulary of the apparatus detected by _APPARATUS. A question that actually
# asks about the apparatus (its edition, commentary, download instructions, ...)
# is recognised by these words and must not have the apparatus suppressed.
_BOILERPLATE_TOPIC = {
    "edition", "transcription", "correction", "corrections", "commentary",
    "editorial", "glossary", "index", "contents", "copyright", "isbn", "markup",
    "download", "downloads", "downloaded", "free", "catalogue", "catalog",
    "publisher", "publishers", "order", "orders", "buy", "purchase", "scan", "code",
}
_NUMBER = re.compile(r"(?<!\w)(?:₹|rs\.?\s*)?\d[\d,]*(?:\.\d+)?%?(?!\w)", re.I)


def _tokens(text):
    return [t.lower() for t in re.findall(r"[\w-]+", text or "", flags=re.UNICODE)]


def _boilerplate(text):
    return bool(_APPARATUS.search(text or ""))


def _query_about_boilerplate(query):
    """True when the question itself concerns the boilerplate/apparatus.

    The previous gate keyed on the chunk's overall query-term coverage, which let
    high-overlap commentary (for example an editor's notes that repeat the
    subject's name) escape the penalty and outrank the real source passage. This
    asks the question-topic directly instead.
    """
    if _APPARATUS.search(query or "") or _STRUCTURE.search(query or ""):
        return True
    return bool({t for t in _tokens(query) if len(t) > 2} & _BOILERPLATE_TOPIC)


def extract_answer_lists(text: str) -> list[list[str]]:
    """Extract explicit positive lists from prose, ordinals, and numbered lines."""
    cue = re.compile(
        r"\b(?:are|include|includes|consist of|consists of|comprise|comprises|following)\s+"
        r"([^.!?]{0,320})", re.I
    )
    filler = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
              "in", "is", "of", "on", "or", "the", "to", "with", "one", "word"}
    text = text or ""
    lists = []

    # Numbered and bulleted lists are strong structure signals independent of
    # document type or punctuation style.
    marked = re.findall(r"(?m)^\s*(?:\d{1,3}[.)]|[-*•])\s+([^\r\n]+)", text)
    marked = [re.sub(r"\s+", " ", item).strip(" \t;:-") for item in marked]
    if len(marked) >= 2:
        lists.append(marked)

    # Ordered prose lists often use "first ...; second ...; third ..." rather
    # than commas. Capture consecutive ordinals and the phrase following each.
    ordinals = {"first": 1, "second": 2, "third": 3, "fourth": 4,
                "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
                "ninth": 9, "tenth": 10}
    ordinal_matches = list(re.finditer(
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
        text, re.I))
    for start_index, first_match in enumerate(ordinal_matches):
        sequence = [first_match]
        expected = ordinals[first_match.group(1).casefold()] + 1
        for candidate in ordinal_matches[start_index + 1:]:
            value = ordinals[candidate.group(1).casefold()]
            if value == expected and candidate.start() - sequence[-1].end() < 500:
                sequence.append(candidate)
                expected += 1
            elif value > expected or candidate.start() - sequence[-1].end() >= 500:
                break
        if len(sequence) < 2:
            continue
        items = []
        for index, marker in enumerate(sequence):
            end = sequence[index + 1].start() if index + 1 < len(sequence) else min(len(text), marker.end() + 350)
            phrase = text[marker.end():end].split(".", 1)[0]
            relation = re.search(r"\b(?:is|are)\b", phrase, re.I)
            if relation:
                phrase = phrase[relation.end():]
            phrase = re.split(r",\s*as,?\s", phrase, maxsplit=1, flags=re.I)[0]
            phrase = re.sub(r"\s+", " ", phrase).strip(" \t\r\n,;:-")
            phrase = re.sub(r"^(?:that of|the)\s+", "", phrase, flags=re.I)
            if phrase:
                items.append(phrase)
        if len(items) >= 2:
            lists.append(items)
            break

    for match in cue.finditer(text):
        raw = match.group(1)
        if re.match(r"\s*(?:not|never|beyond|outside)\b", raw, re.I):
            continue
        # Stop at a contrast that introduces an excluded or opposite group.
        contrast = re.search(
            r",\s+(?:and\s+)?(?:the\s+)?(?:other|others|remaining|rest)\s+"
            r"(?:are\s+)?(?:not|beyond|outside|except)\b|,\s+and\s+not\b", raw, re.I)
        if contrast:
            raw = raw[:contrast.start()]
        raw = re.sub(r"^\s*(?:a|the)\s+(?:collection|set|group|series)\s+of\s+",
                     "", raw, flags=re.I)
        # Keep a candidate within one clause; semicolons commonly separate
        # opposing properties or a new explanation.
        raw = raw.split(";", 1)[0]
        # In phrases such as "items include, for example, A, B, and C", the
        # items begin after "such as" rather than with the surrounding clause.
        if re.search(r"\bsuch as\b", raw, re.I):
            raw = re.split(r"\bsuch as\b", raw, maxsplit=1, flags=re.I)[1]
        if raw.count(",") < 2 and not (raw.count(",") and re.search(r"\band\b", raw, re.I)):
            continue
        sentence_start = max(text.rfind(mark, 0, match.start()) for mark in ".?!") + 1
        clause_prefix = text[sentence_start:match.start()]
        if (re.search(r"\b(?:not|never|neither|nor|beyond|outside|except)\b", clause_prefix, re.I)
                or re.match(r"\s*(?:not|never|beyond|outside)\b", raw, re.I)):
            continue
        members = []
        for part in re.split(r",|;|\band\b", raw, flags=re.I):
            item = re.sub(r"\s+", " ", part).strip(" \t\r\n:;-—")
            terms = [term for term in re.findall(r"[\w'-]+", item)
                     if term.casefold() not in filler and len(term) > 1]
            if terms and not re.fullmatch(r"in one word|in short|in other words", item, re.I):
                members.append(item)
        if len(members) >= 2:
            lists.append(members)
    return lists


def rerank(query: str, hits: list[dict], limit: int | None = None) -> list[dict]:
    terms = {t for t in _tokens(query) if len(t) > 2 and t not in _STOP}
    try:
        from app.rag.planner import query_variants
        variant_terms = [
            {t for t in _tokens(variant) if len(t) > 2 and t not in _STOP}
            for variant in query_variants(query)[1:]
        ]
    except Exception:
        variant_terms = []
    # "name" is both a verb ("name the members") and a noun ("the name of").
    # A bare noun use must not flag a single-value question as a list question:
    # doing so inflated the score of any list-bearing chunk by +0.14 and pushed
    # the genuinely answering passage out of the window.
    asks_for_items = bool(re.search(
        r"\b(?:list|enumerate|identify)\b|\bname\s+(?:the|all|each|every|any|both)\b"
        r"|\b(?:what|which)\b.{0,40}\b(?:are|include|consist|comprise)\b",
        query or "", re.I
    ) or re.search(r"\b(?:what|which)\b.{0,35}\b(?:kinds?|types?|forms?)\b", query or "", re.I))
    asks_about_group = bool(re.search(
        r"\b(?:club|alliance|group|society|association|organization)\b", query or "", re.I))
    query_numbers = set(_NUMBER.findall(query or ""))
    # Whether the question concerns apparatus is a property of the question, so
    # it is decided once here rather than from each candidate's coverage.
    about_boilerplate = _query_about_boilerplate(query)
    ranked = []
    for hit in hits:
        text = hit.get("text", "") or ""
        text_terms = set(_tokens(text))
        coverage = max(
            [len(terms & text_terms) / max(1, len(terms))]
            + [0.85 * len(variant & text_terms) / max(1, len(variant))
               for variant in variant_terms]
        )
        semantic = hit.get("semantic_similarity")
        hybrid = max(0.0, min(1.0, float(hit.get("similarity", 0) or 0)))
        fusion = float(hit.get("fusion_score") or 0)
        query_phrase = " ".join(_tokens(query))
        normalized_text = " ".join(_tokens(text))
        phrase = 1.0 if len(terms) > 1 and query_phrase and query_phrase in normalized_text else 0.0
        # Exact requested figures/identifiers should be prominent when present.
        number_match = bool(query_numbers and query_numbers.intersection(_NUMBER.findall(text)))
        answer_cue = bool(re.search(
            r"\b(is|are|was|were|has|had|amount|total|revenue|date|dated|signed|"
            r"appointed|located|requires|required|means|defined as|includes?|consists of|"
            r"comprises|refers to|within .{0,50} are)\b", text, re.I
        ))
        list_answer_cue = asks_for_items and bool(extract_answer_lists(text))
        group_answer_cue = asks_about_group and bool(
            re.search(r"\b(?:club|alliance|group|society|association|organization)\b", text, re.I)
            and re.search(r"\b(?:form(?:ed)?|found|founded|establish(?:ed)?|organize(?:d)?|undertook|purpose|improvement)\b", text, re.I)
        )
        semantic_signal = float(semantic) if semantic is not None else hybrid
        item = dict(hit)
        item.update({
            "query_term_coverage": round(coverage, 4),
            "exact_number_match": number_match,
            "answer_bearing_cue": answer_cue,
            "list_answer_cue": list_answer_cue,
            "group_answer_cue": group_answer_cue,
            "rerank_score": (0.50 * hybrid + 0.22 * coverage
                             + 0.12 * semantic_signal + 0.10 * min(1.0, fusion * 60)
                             + 0.02 * phrase + 0.02 * number_match + 0.06 * answer_cue
                             + 0.14 * list_answer_cue + 0.18 * group_answer_cue),
        })
        # Penalize apparatus (front matter, editorial notes, back matter,
        # publisher pages) whenever the question is not itself about that
        # apparatus. This no longer depends on the candidate's term coverage,
        # which previously let high-overlap commentary escape the penalty.
        if _boilerplate(text) and not about_boilerplate:
            item["rerank_score"] *= 0.60
            item["boilerplate"] = True
        elif _STRUCTURE.search(text) and not about_boilerplate and coverage < 0.25:
            # Structural front matter keeps the previous, milder behaviour:
            # demoted only when it barely relates to the query.
            item["rerank_score"] *= 0.65
            item["boilerplate"] = True
        ranked.append(item)
    ranked.sort(key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:limit] if limit else ranked
