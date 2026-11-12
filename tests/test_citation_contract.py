"""Citation integrity contracts shared by answer selection and verification."""

from app.rag.citations import (
    cite_supported_uncited_sentences,
    validate_citation_references,
)


def test_filename_label_alone_does_not_satisfy_numeric_evidence_contract():
    answer = "Benjamin Franklin was born in Boston [franklin.pdf]."
    result = validate_citation_references(answer, 1, ["franklin.pdf"])

    assert result["unknown_source_labels"] == []
    assert result["references_valid"] is False
    assert result["fully_cited"] is False
    assert result["citation_present"] is False
    assert result["citation_coverage"] == 0.0
    assert result["valid"] is False


def test_filename_label_is_not_mistaken_for_a_verifier_compatible_citation():
    answer = "Benjamin Franklin was born in Boston [franklin.pdf]."
    sources = [{"text": "Benjamin Franklin was born in Boston, Massachusetts."}]

    repaired = cite_supported_uncited_sentences(answer, sources)
    result = validate_citation_references(repaired, 1, ["franklin.pdf"])

    assert "[1]" in repaired
    assert result["references_valid"] is True
    assert result["fully_cited"] is True
    assert result["valid"] is True


def test_every_factual_sentence_requires_its_own_numeric_reference():
    answer = (
        "Benjamin Franklin was born in Boston [1]. "
        "He later became a printer and inventor."
    )
    result = validate_citation_references(answer, 1, ["franklin.pdf"])

    assert result["references_valid"] is True
    assert result["fully_cited"] is False
    assert result["citation_coverage"] == 0.5
    assert len(result["uncited_sentences"]) == 1
    assert result["valid"] is False
