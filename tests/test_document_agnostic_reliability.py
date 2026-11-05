# Document-agnostic reliability fixes proven against the real corpus defects:
# 1) a book-length narrative must not be classified "legal" because it contains
#    the ordinary narrative words "party"/"parties"/"hereby";
# 2) genuine contracts remain "legal"; invoices stay "invoice";
# 3) publisher boilerplate ("READ AND DOWNLOAD FREE BOOKS ... infobooks.org")
#    is down-weighted by the reranker instead of occupying an evidence slot.
import pytest

from app.ingestion.classify import detect_document_type
from app.retrieval.reranker import rerank


FRANKLIN_STYLE = (
    "It was a fine autumn evening in Boston when our party reached themeeting house, and everyone in "
    "the party agreed that the debate had gone unexpectedly well. ".replace("themeeting", "the meeting")
    + ("I hereby resolved to set down this account of my life so that my posterity may resolve to profit "
       "by it as I have done, and thus join the party of those who carefully observe events. " * 6)
    + "My father, who was a tallow chandler by trade, sent me to school with the intention that "
    "one of his sons might serve the church. "
)


def test_narrative_with_narrative_party_words_is_not_legal():
    # Franklin-style autobiography: two to three legal-cue hits scattered in
    # long narrative sentences must classify as prose, not legal.
    result = detect_document_type(FRANKLIN_STYLE, "the-autobiography-of-benjamin-franklin.pdf")
    assert result["type"] == "prose"


def test_realistic_contract_is_still_legal():
    contract = ("AGREEMENT\nWHEREAS the parties have agreed as follows.\n"
                "1. Termination. Either party may terminate this agreement with 30 days notice.\n"
                "2. Governing Law. This agreement is governed by the laws of England.\n"
                "3. Indemnification. Each party shall indemnify the other against liability.\n"
                "4. Confidentiality. The parties shall keep these terms and conditions confidential.\n"
                "5. Notice period: 60 days. Breach permits immediate termination."
                ) * 8  # book-length so the narrative guard is active
    result = detect_document_type(contract, "service-contract.pdf")
    assert result["type"] == "legal"


def test_invoice_classification_unchanged_by_narrative_guard():
    text = ("Invoice No: 2041\nBill To: Acme Ltd\nSubtotal 100.00\n"
            "VAT 20.00\nTotal Due 120.00\nDue Date: 2026-01-31")
    result = detect_document_type(text, "invoice-2041.pdf")
    assert result["type"] == "invoice"
    assert result["confidence"] >= 0.7


def test_question_bank_classification_unchanged():
    text = ("Which systems are in scope? How are alerts triaged? "
            "Who owns the SIEM? What is the retention period? How often is it reviewed?")
    assert detect_document_type(text, "siem.docx")["type"] == "question_bank"


def _chunk(text, filename="book.pdf", page="Page 3", chunk_id="c1"):
    return {"id": chunk_id, "text": text,
            "metadata": {"document_id": "d1", "filename": filename, "source_location": page},
            "similarity": 0.6, "semantic_similarity": 0.6, "lexical_similarity": 0.5}


INFOBOOKS_AD = ("READ AND DOWNLOAD FREE BOOKS Scan the code or go to\ninfobooks.org "
                "Thousands of PDF books, free and with no sign-up: infobooks.org/collection")


def test_publisher_boilerplate_is_downweighted_by_reranker():
    query = "What are the first principles of getting rich?"
    same_prefix = ("The first principle of getting rich is to think in a certain way. "
                   "Tell a certain way and a certain manner helps the mind form an idea.")
    ranked = rerank(query, [
        _chunk(same_prefix, page="Page 10", chunk_id="content"),
        _chunk(INFOBOOKS_AD, page="Page 3", chunk_id="ad"),
    ])
    scores = {h["id"]: h for h in ranked}
    assert scores["ad"].get("boilerplate") is True
    assert scores["content"].get("boilerplate", False) is False
    # The ad remains a retrievable candidate, but its rerank score must be lower.
    assert scores["ad"]["rerank_score"] < scores["content"]["rerank_score"]


def test_deliberate_infobooks_question_can_still_retrieve_the_ad():
    # Boilerplate is down-weighted, never excluded: the ad stays retrievable
    # for a question that is genuinely about it.
    ranked = rerank("Where can I download more free books from infobooks.org?", [
        _chunk("The Declaration was signed in 1776 in Philadelphia Congress.",
               page="Page 40", chunk_id="other"),
        _chunk(INFOBOOKS_AD, page="Page 3", chunk_id="ad"),
    ])
    assert any(h["id"] == "ad" for h in ranked)


def test_boilerplate_penalty_skipped_when_query_asks_about_it():
    # Asking about the download store itself is asking about the boilerplate:
    # no penalty, so the ad keeps its merits and can reach the evidence slots.
    same_prefix = ("The first principle of getting rich is to think in a certain way. "
                   "Tell a certain way and a certain manner helps the mind form an idea.")
    unrelated = rerank("What are the first principles of getting rich?", [
        _chunk(same_prefix, page="Page 10", chunk_id="content"),
        _chunk(INFOBOOKS_AD, page="Page 3", chunk_id="ad"),
    ])
    about_store = rerank("How can I order more books from InfoBooks?", [
        _chunk(same_prefix, page="Page 10", chunk_id="content"),
        _chunk(INFOBOOKS_AD, page="Page 3", chunk_id="ad"),
    ])
    scores_unrelated = {h["id"]: h["rerank_score"] for h in unrelated}
    scores_about = {h["id"]: h["rerank_score"] for h in about_store}
    # Penalized for the unrelated query, unpenalized for the store query.
    assert scores_unrelated["ad"] < scores_unrelated["content"]
    assert scores_about["ad"] > scores_about["content"]
    assert [h["id"] for h in about_store][:1] == ["ad"]
