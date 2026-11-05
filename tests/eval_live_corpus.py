# Reusable live evaluation of the complete RND2 workflow against the REAL
# uploaded corpus (no synthetic documents). Run with the backend live:
#   python tests/eval_live_corpus.py [--quick]
#
# It records, per question: document name/type, question, expected evidence,
# expected answer or abstention, retrieved chunks + pages, the actual answer,
# citation correctness, final status, response time and failure reason.
# Results are written to:
#   evaluation/live_results/rnd_live_eval_<stamp>.json (configurable)
import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings

API = os.getenv("RND2_EVAL_API", "http://127.0.0.1:8000")
OUT_DIR = Path(os.getenv("RND2_EVAL_OUT_DIR", str(PROJECT_ROOT / "evaluation" / "live_results")))


def call_chat(question, history=None, timeout=1200, api=None):
    body = json.dumps({"question": question, "history": history or []}).encode()
    started = time.time()
    with urlopen(Request((api or API).rstrip("/") + "/chat", data=body,
                         headers={"Content-Type": "application/json"}), timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    data["wall_seconds"] = round(time.time() - started, 3)
    return data


def citation_check(answer, sources):
    refs = []
    for group in re.findall(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]", answer or ""):
        refs.extend(int(n) for n in group.replace(" ", "").split(","))
    refs = sorted(set(refs))
    cited = []
    for n in refs:
        if 1 <= n <= len(sources):
            src = sources[n - 1]
            cited.append({"ref": n, "filename": src.get("filename"),
                          "page": src.get("source_location")})
    invalid = [n for n in refs if n < 1 or n > len(sources)]
    return {"refs": refs, "all_refs_valid": not invalid, "cited_sources": cited,
            "invalid_refs": invalid}


def doc_types():
    db_path = Path(settings.sqlite_path).resolve()
    if not db_path.exists():
        return {}
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return {row[0]: json.loads(row[1] or "{}").get("document_type")
                for row in con.execute("SELECT filename, source_metadata_json FROM documents")}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# The evaluation matrix. `expect` fields are manually verified against the
# corpus during the diagnosis phase (see rnd_corpus_probe.json).
# ---------------------------------------------------------------------------
CASES = [
    {"id": "C1_factual_direct", "category": "direct_factual_retrieval",
     "question": "According to the Enchiridion, what things are in our control?",
     "expect_status": "success",
     "expect": "Enchiridion page 4: opinion, aim, desire",
     "answer_must_include_any_of": [["opinion"], ["aim"], ["desire"]],
     "source_filename_contains": ["the-enchiridion"], "source_page_contains": ["9"]},
    {"id": "C2_keyword_exact", "category": "exact_keyword_retrieval",
     "question": "What does Chapter 49 of the Enchiridion say about people who interpret the works of Chrysippus?",
     "expect_status": "success",
     "expect": "Enchiridion Chapter 49, page 67: passage about interpreting Chrysippus",
     "answer_must_include_any_of": [["Chrysippus"]],
     "source_filename_contains": ["the-enchiridion"], "source_page_contains": ["67"]},
    {"id": "C3_semantic_concept", "category": "semantic_retrieval",
     "question": "What does the Science of Getting Rich mean by thinking in a certain way?",
     "expect_status": "success",
     "expect": "Science of Getting Rich, Chapter 4: thinking in the Certain Way",
     "answer_must_include_any_of": [["certain way"]],
     "source_filename_contains": ["the-science-of-getting-rich"], "source_page_contains": ["9"]},
    {"id": "C4_chapter_specific", "category": "chapter_section_questions",
     "question": "What does Chapter 51 say are the three topics of philosophy, and which does Epictetus consider most necessary?",
     "expect_status": "success",
     "expect": "Enchiridion Chapter 51, page 71: practical application, demonstrations, logic; practical application most necessary",
     "answer_must_include_any_of": [["practical application"], ["demonstrations"], ["logic"]],
     "source_filename_contains": ["the-enchiridion"], "source_page_contains": ["71"]},
    {"id": "C5_multi_part", "category": "multi_part_questions",
     "question": "What are the three topics of Chapter 51? What things in the book are in our control?",
     "expect_status": "success",
     "expect": "Chapter 51 topics (page 71) and things in our control (page 4)",
     "answer_must_include_any_of": [["practical application"], ["demonstrations"], ["logic"], ["opinion"], ["aim"], ["desire"]],
     "source_filename_contains": ["the-enchiridion"]},
    {"id": "C6_cross_document", "category": "cross_document_questions",
     "question": "According to the documents, what did Benjamin Franklin and Napoleon Hill each describe as a club or alliance for improvement?",
     "expect_status": "success",
     "expect": "Franklin's Junto (PDF page 37) and Hill's Master Mind alliance (PDF page 13 or 323)",
     "answer_must_include_any_of": [["junto"], ["master mind", "mastermind"]],
     "source_filename_contains": ["autobiography-of-benjamin-franklin", "think-and-grow-rich"],
     "source_page_contains": ["37", "13"]},
    {"id": "C7_numeric", "category": "numerical_questions",
     "question": "How many chapters make up the Enchiridion?",
     "expect_status": "success",
     "expect": "The final numbered chapter in the supplied edition is Chapter 51 (PDF page 71)",
     "answer_must_include_any_of": [["51", "fifty-one"]],
     "source_filename_contains": ["the-enchiridion"], "source_page_contains": ["71"]},
    {"id": "C8_date_identifier", "category": "date_identifier_questions",
     "question": "What year did Benjamin Franklin arrive in Philadelphia?",
     "expect_status": "abstain",
     "expect": "The supplied autobiography describes the journey but does not state the arrival year; abstain rather than import outside historical knowledge"},
    {"id": "C9_long_document", "category": "long_document_retrieval",
     "question": "What are the six steps Think and Grow Rich describes to turn desire into gold?",
     "expect_status": "success",
     "expect": "Think and Grow Rich, PDF page 68: six definite, practical steps",
     "answer_must_include_any_of": [["six"], ["desire"]],
     "source_filename_contains": ["think-and-grow-rich"], "source_page_contains": ["68"]},
    {"id": "C10_short_document", "category": "short_document_retrieval",
     "question": "According to the Enchiridion, when do people consider things good or bad?",
     "expect_status": "success",
     "expect": "Enchiridion PDF page 44: good or evil should consist only in things within our power",
     "source_filename_contains": ["the-enchiridion"], "source_page_contains": ["44"]},
    {"id": "C11_summary", "category": "summary_generation",
     "question": "Summarize what the Enchiridion teaches about dealing with other people.",
     "expect_status": "success",
     "expect": "Summary grounded in multiple Enchiridion passages"},
    {"id": "C12_comparison", "category": "comparison_questions",
     "question": "How do the Science of Getting Rich and Think and Grow Rich each describe the way a person gets rich?",
     "expect_status": "success",
     "expect": "Each side grounded and cited: Wattles' Certain Way; Hill's method / definiteness of purpose",
     "answer_must_include_any_of": [["certain way"], ["desire", "purpose"]],
     "source_filename_contains": ["the-science-of-getting-rich", "think-and-grow-rich"]},
    {"id": "C13_calculation", "category": "calculation_questions",
     "question": "What is 10% of 250?",
     "expect_status": "success",
     "expect": "Deterministic result 25",
     "answer_must_include_any_of": [["25"]]},
    {"id": "C14_followup", "category": "followup_questions",
     "turns": [("What club did Benjamin Franklin found in Philadelphia?", ["junto"]),
               ("What was the purpose of it?", ["mutual improvement"])],
     "expect_status": "success",
     "expect": "Follow-up resolves 'it' to the Junto and answers from evidence",
     "source_filename_contains": ["autobiography-of-benjamin-franklin"],
     "source_page_contains": ["37"]},
    {"id": "C15_abstention_missing", "category": "missing_information_abstention",
     "question": "Which chapter of the Enchiridion discusses quidditch?",
     "expect_status": "abstain",
     "expect": "Abstention: no quidditch content exists in the uploaded corpus"},
    {"id": "C16_cache_consistency", "category": "cache_consistency",
     "question": "What does Chapter 51 say are the three topics of philosophy, and which does Epictetus consider most necessary?",
     "repeat_of": "C4_chapter_specific",
     "expect_status": "success", "expect_cache_hit": True,
     "expect": "Repeat should return the same grounded answer and report an answer-cache hit"},
    {"id": "C17_similar_terminology", "category": "similar_terminology_documents",
     "question": "Which book uses the phrase 'Certain Way' and what does it define?",
     "expect_status": "success",
     "expect": "The Science of Getting Rich defines the Certain Way",
     "answer_must_include_any_of": [["certain way"]],
     "source_filename_contains": ["the-science-of-getting-rich"]},
    {"id": "C18_boilerplate", "category": "noisy_boilerplate_chunks",
     "question": "How can I order more books from InfoBooks?",
     "expect_status": "any",
     "expect": "Answer only if the InfoBooks pages contain actionable ordering information; otherwise abstain"},
    {"id": "C19_franklin_substance", "category": "answer_completeness",
     "question": "What trade did Benjamin Franklin apprentice in as a boy?",
     "expect_status": "success",
     "expect": "The named trade from the autobiography (PDF page 7: tallow-chandler and soap-boiler business; cutting wick for candles)",
     "answer_must_include_any_of": [["candle", "tallow chandler"]],
     "source_filename_contains": ["autobiography-of-benjamin-franklin"], "source_page_contains": ["7"]},
]


def answer_quality_check(case, answer, sources, answer_cache=None):
    """Check manually labelled answer terms and source-document expectations."""
    expected_status = case.get("expect_status")
    actual_status = case.get("_actual_status")
    if expected_status == "success_or_abstain":
        status_ok = actual_status == "success" or actual_status == "insufficient_evidence"
    elif expected_status == "any":
        status_ok = None
    elif expected_status:
        status_ok = (
            actual_status == expected_status
            or (expected_status == "abstain" and actual_status == "insufficient_evidence")
        )
    else:
        status_ok = None

    normalized_answer = (answer or "").casefold()
    accepted_abstention = (
        expected_status in {"abstain", "success_or_abstain"}
        and actual_status == "insufficient_evidence"
    )
    term_groups = [] if accepted_abstention else (case.get("answer_must_include_any_of") or [])
    term_checks = [
        any(str(term).casefold() in normalized_answer for term in group)
        for group in term_groups
    ]
    terms_ok = all(term_checks) if term_groups else None

    filenames = [str(source.get("filename") or "").casefold() for source in sources]
    expected_files = [] if accepted_abstention else (case.get("source_filename_contains") or [])
    file_checks = [any(fragment.casefold() in name for name in filenames)
                   for fragment in expected_files]
    files_ok = all(file_checks) if expected_files else None

    pages = [str(source.get("source_location") or "").casefold() for source in sources]
    expected_pages = [] if accepted_abstention else (case.get("source_page_contains") or [])
    page_checks = [any(fragment.casefold() in page for page in pages)
                   for fragment in expected_pages]
    pages_ok = all(page_checks) if expected_pages else None

    cache_expected = case.get("expect_cache_hit")
    cache_actual = bool((answer_cache or {}).get("hit"))
    cache_ok = cache_actual if cache_expected is not None else None
    checks = [value for value in (status_ok, terms_ok, files_ok, pages_ok, cache_ok) if value is not None]
    return {
        "status_match": status_ok,
        "answer_term_groups": term_checks,
        "answer_terms_match": terms_ok,
        "source_filename_checks": file_checks,
        "source_filenames_match": files_ok,
        "source_page_checks": page_checks,
        "source_pages_match": pages_ok,
        "quality_checkable": bool(checks),
        "cache_hit_expected": cache_expected,
        "cache_hit_actual": cache_actual if cache_expected is not None else None,
        "cache_hit_match": cache_ok,
        "quality_pass": all(checks) if checks else None,
    }


def run_case(case, types, api=None, timeout=1200):
    turns = case.get("turns") or [(case["question"], None)]
    history = []
    records = []
    for turn_index, (question, turn_terms) in enumerate(turns, 1):
        answer = call_chat(question, history=history, timeout=timeout, api=api)
        turn_case = dict(case)
        if isinstance(turn_terms, list):
            turn_case["answer_must_include_any_of"] = [[term] for term in turn_terms]
        history = history + [{"role": "user", "content": question[:2000]},
                             {"role": "assistant", "content": (answer.get("answer") or "")[:2000]}]
        record = {
            "case_id": case["id"], "category": case["category"], "turn": turn_index,
            "question": question, "expected_evidence": case.get("expect"),
            "expected_status": case.get("expect_status"),
            "status": answer.get("status"),
            "abstained": answer.get("abstained"),
            "actual_answer": answer.get("answer"),
            "question_type": answer.get("question_type"),
            "confidence": (answer.get("confidence") or {}).get("score"),
            "response_seconds": answer.get("response_seconds"),
            "wall_seconds": answer.get("wall_seconds"),
            "retrieval": {
                "indexed_documents": (answer.get("retrieval") or {}).get("indexed_documents"),
                "candidate_chunks": (answer.get("retrieval") or {}).get("candidate_chunks"),
                "selected_chunks": (answer.get("retrieval") or {}).get("selected_chunks"),
                "best_similarity": (answer.get("retrieval") or {}).get("best_similarity"),
                "document_types": (answer.get("retrieval") or {}).get("document_types"),
                "answer_cache": answer.get("answer_cache"),
            },
            "sources": [{"rank": s.get("rank"), "filename": s.get("filename"),
                         "page": s.get("source_location"),
                         "relevance": s.get("relevance_score"),
                         "supporting_text": (s.get("supporting_text") or "")[:200]}
                        for s in (answer.get("sources") or [])],
            "citation": citation_check(answer.get("answer"), answer.get("sources") or []),
            "answer_quality": answer_quality_check(
                {**turn_case, "_actual_status": answer.get("status")},
                answer.get("answer"), answer.get("sources") or [], answer.get("answer_cache")),
            "potential_conflicts": answer.get("potential_conflicts"),
            "failure_reason": answer.get("status") if answer.get("status") == "success" else
            ((answer.get("retrieval") or {}).get("abstention_classification") or
             (answer.get("retrieval") or {}).get("failure_stage") or answer.get("error") or
             answer.get("status")),
        }
        for src in record["sources"]:
            src["document_type"] = types.get(src.get("filename"))
        records.append(record)
        print("%-22s t%d %-28s %6.1fs conf=%s refs=%s" % (
            case["id"], turn_index, record["status"],
            record["wall_seconds"], record["confidence"],
            record["citation"]["refs"]))
    return records


def run(selected, api=None, output_dir=None, timeout=1200):
    types = doc_types()
    results = []
    for case in selected:
        records = run_case(case, types, api=api, timeout=timeout)
        results.extend(records)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination = Path(output_dir) if output_dir else OUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    out_path = destination / ("rnd_live_eval_%s.json" % stamp)
    totals = {"success": 0, "abstain": 0, "other": 0}
    for record in results:
        status = record["status"]
        expect = record.get("expected_status")
        if status == "success":
            totals["success"] += 1
        elif expect == "abstain" and record.get("abstained"):
            totals["abstain"] += 1
        else:
            totals["other"] += 1
    summary = {"started": stamp, "api": api or API, "results": results, "totals": totals}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print("saved", out_path)
    print("totals:", totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--api", default=API, help="Loopback RND2 API base URL")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--cases", default="",
                        help="Comma-separated case ids to re-run "
                             "(e.g. C3_semantic_concept,C8_date_identifier)")
    args = parser.parse_args()
    if args.cases:
        keep = {item.strip() for item in args.cases.split(",") if item.strip()}
        unknown = keep - {case["id"] for case in CASES}
        assert not unknown, unknown
        selected = [case for case in CASES if case["id"] in keep]
    else:
        quick_set = {"C1_factual_direct", "C4_chapter_specific", "C13_calculation",
                     "C15_abstention_missing", "C18_boilerplate"}
        selected = [case for case in CASES
                    if case["id"] in quick_set] if args.quick else list(CASES)
    run(selected, api=args.api, output_dir=args.output_dir, timeout=args.timeout)
