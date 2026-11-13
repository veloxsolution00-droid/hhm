# Document-agnostic reliability regression coverage:
# 1) a book-length narrative must not be classified "legal" because it contains
#    the ordinary narrative words "party"/"parties"/"hereby";
# 2) genuine contracts remain "legal"; invoices stay "invoice";
# 3) generic publisher boilerplate is down-weighted unless the query asks about it.
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


PUBLISHER_NOTICE = ("READ AND DOWNLOAD FREE BOOKS. Scan the code or visit\n"
                    "the publisher's catalogue for free reading material.")


def test_publisher_boilerplate_is_downweighted_by_reranker():
    query = "What are the first principles of getting rich?"
    same_prefix = ("The first principle of getting rich is to think in a certain way. "
                   "Tell a certain way and a certain manner helps the mind form an idea.")
    ranked = rerank(query, [
        _chunk(same_prefix, page="Page 10", chunk_id="content"),
        _chunk(PUBLISHER_NOTICE, page="Page 3", chunk_id="notice"),
    ])
    scores = {h["id"]: h for h in ranked}
    assert scores["notice"].get("boilerplate") is True
    assert scores["content"].get("boilerplate", False) is False
    assert scores["notice"]["rerank_score"] < scores["content"]["rerank_score"]


def test_question_about_download_instructions_can_retrieve_the_notice():
    # Boilerplate is down-weighted, never excluded: the ad stays retrievable
    # for a question that is genuinely about it.
    ranked = rerank("Where can I download free reading material?", [
        _chunk("The Declaration was signed in 1776 in Philadelphia Congress.",
               page="Page 40", chunk_id="other"),
        _chunk(PUBLISHER_NOTICE, page="Page 3", chunk_id="notice"),
    ])
    assert ranked[0]["id"] == "notice"


def test_boilerplate_penalty_skipped_when_query_asks_about_downloads():
    same_prefix = ("The first principle of getting rich is to think in a certain way. "
                   "Tell a certain way and a certain manner helps the mind form an idea.")
    unrelated = rerank("What are the first principles of getting rich?", [
        _chunk(same_prefix, page="Page 10", chunk_id="content"),
        _chunk(PUBLISHER_NOTICE, page="Page 3", chunk_id="notice"),
    ])
    about_store = rerank("Where can I download free books?", [
        _chunk(same_prefix, page="Page 10", chunk_id="content"),
        _chunk(PUBLISHER_NOTICE, page="Page 3", chunk_id="notice"),
    ])
    scores_unrelated = {h["id"]: h["rerank_score"] for h in unrelated}
    scores_about = {h["id"]: h["rerank_score"] for h in about_store}
    assert scores_unrelated["notice"] < scores_unrelated["content"]
    assert scores_about["notice"] > scores_about["content"]
    assert [h["id"] for h in about_store][:1] == ["notice"]


def test_editorial_apparatus_is_demoted_even_with_high_query_coverage():
    # The old gate only demoted apparatus when the chunk had little lexical
    # relation to the query. Editorial notes that repeat the subject's name have
    # high coverage, so they escaped and outranked the real source passage.
    query = "In which city was Benjamin Franklin born?"
    content = ("I was born in Boston in New England. My elder brothers were put "
               "apprentices to different trades, and my father intended to put me "
               "to the business of a tallow chandler.")
    commentary = ("Benjamin Franklin was born in Boston. Franklin's birth in Boston "
                  "is recorded by several later writers. - [UVAstudstaff] Notes on "
                  "Franklin's birth in Boston, Massachusetts.")
    ranked = rerank(query, [
        _chunk(content, page="Page 6", chunk_id="content"),
        _chunk(commentary, page="Page 106", chunk_id="commentary"),
    ])
    scores = {h["id"]: h for h in ranked}
    assert scores["commentary"]["query_term_coverage"] > 0.25
    assert scores["commentary"]["boilerplate"] is True
    assert scores["content"]["rerank_score"] > scores["commentary"]["rerank_score"]


def test_contents_page_is_kept_for_a_chapter_title_question():
    # A table of contents is a legitimate source for a chapter-title question,
    # so structural front matter is not demoted when it relates to the query.
    query = "What is the title of Chapter 17 of The Science of Getting Rich?"
    toc = ("CONTENTS Preface Chapter 1. The Right To Be Rich Chapter 17. Summary "
           "of the Science of Getting Rich")
    ranked = rerank(query, [
        _chunk("A general passage about the science of getting rich.", page="Page 9", chunk_id="other"),
        _chunk(toc, page="Page 2", chunk_id="toc"),
    ])
    scores = {h["id"]: h for h in ranked}
    assert not scores["toc"].get("boilerplate", False)
    assert ranked[0]["id"] == "toc"


def test_answer_bearing_statement_ranks_above_incidental_mention():
    ranked = rerank("What notice period does the contract require?", [
        _chunk("The contract mentions notice in its standard terms.", page="Page 2", chunk_id="mention"),
        _chunk("The agreement requires a notice period of 30 days.", page="Page 8", chunk_id="answer"),
    ])
    assert ranked[0]["id"] == "answer"
    assert ranked[0]["answer_bearing_cue"] is True


def test_group_founder_question_prioritizes_the_passage_naming_the_created_group():
    ranked = rerank("What club did Benjamin Franklin found in Philadelphia?", [
        _chunk("Benjamin Franklin founded a hospital in Philadelphia.", page="Page 17", chunk_id="wrong"),
        _chunk("The project was approved, and every member undertook to form his club, "
               "which was called the Junto.", page="Page 61", chunk_id="junto"),
    ])
    assert ranked[0]["id"] == "junto"
    assert ranked[0]["group_answer_cue"] is True


def test_enumerated_evidence_is_prioritized_for_a_list_question():
    ranked = rerank("What are the required submission items?", [
        _chunk("The policy mentions submission items in its standard terms.",
               page="Page 2", chunk_id="mention"),
        _chunk("Required submission items are the signed form, proof of address, "
               "and a current account statement.", page="Page 8", chunk_id="list"),
    ])
    assert ranked[0]["id"] == "list"
    assert ranked[0]["list_answer_cue"] is True


def test_list_extractor_ignores_negated_list_and_keeps_positive_items():
    from app.retrieval.reranker import extract_answer_lists

    text = ("Other items are not, such as the body, property, reputation. "
            "Within our power are opinion, aim, desire, aversion, and, in one word, "
            "whatever affairs are our own.")
    assert extract_answer_lists(text) == [
        ["opinion", "aim", "desire", "aversion", "whatever affairs are our own"]
    ]


def test_list_extractor_handles_such_as_and_contrastive_items():
    from app.retrieval.reranker import extract_answer_lists

    text = ("Some things are within our power, such as opinion, desire and aversion, "
            "and others are not, such as the body, property and reputation.")
    assert extract_answer_lists(text) == [["opinion", "desire", "aversion"]]


def test_definition_query_variant_is_generic_for_any_term():
    from app.rag.planner import query_variants

    assert "depreciation definition meaning" in query_variants("What is depreciation?")
    assert "indemnity definition meaning" in query_variants("What is indemnity?")


def test_omitted_evidence_list_member_is_restored_with_its_citation():
    from app.rag.chat import _complete_list_from_evidence

    query = "What are the required submission items?"
    evidence = _chunk(
        "Required submission items are the signed form, proof of address, "
        "and a current account statement.", chunk_id="list")
    evidence = rerank(query, [evidence])[0]
    completed = _complete_list_from_evidence(
        query, "The items are the signed form and a current account statement [1].", [evidence])
    assert "proof of address [1]" in completed
    assert _complete_list_from_evidence("What is the deadline?", "30 days [1].", [evidence]) == "30 days [1]."


def test_document_form_question_restores_members_from_a_relevant_collection_passage():
    from app.rag.chat import _complete_list_from_evidence
    from app.retrieval.reranker import rerank

    query = "What kinds of records make up Dracula's narrative?"
    evidence = _chunk(
        "Dracula is a novel made of documents. The 27 chapters are a collection of journals, "
        "diaries, letters, telegrams and newspaper cuttings, and not the voice of a single narrator.",
        page="Page 5", chunk_id="dracula-forms")
    evidence = rerank(query, [evidence])[0]
    assert evidence["list_answer_cue"] is True
    completed = _complete_list_from_evidence(
        query, "Dracula's narrative is made up of records [1].", [evidence])
    for item in ("journals", "diaries", "letters", "telegrams", "newspaper cuttings"):
        assert item in completed


def test_false_no_information_draft_can_recover_a_cited_extract_from_direct_evidence():
    from app.rag.chat import _extractive_recovery_from_evidence
    from app.rag.chat import _is_exact_passage_extract
    from app.rag.citations import validate_citation_references

    query = ("What does Chapter 49 of the Enchiridion say about people who interpret "
             "the works of Chrysippus?")
    passage = _chunk(
        "Chapter 49. When anyone shows himself vain on being able to understand and interpret "
        "the works of Chrysippus, say to yourself: Unless Chrysippus had written obscurely, "
        "this person would have had nothing to be vain of.",
        filename="the-enchiridion-epictetus.pdf", page="Page 67")
    recovery = _extractive_recovery_from_evidence(
        query,
        "Chapter 49 does not say anything specific about people who interpret the works of Chrysippus [1].",
        [passage],
    )
    assert recovery is not None
    assert "written obscurely" in recovery
    assert "nothing to be vain of" in recovery
    assert validate_citation_references(recovery, 1, ["the-enchiridion-epictetus.pdf"])["valid"]
    assert _is_exact_passage_extract(recovery, [passage])
    assert not _is_exact_passage_extract(recovery.replace("vain", "proud"), [passage])
    assert _extractive_recovery_from_evidence(query, "The passage says the person is vain [1].", [passage]) is None
