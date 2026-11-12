from tests.eval_live_corpus import page_fragment_match


def test_page_expectation_matches_full_numeric_page_only():
    assert page_fragment_match("7", "Page 7")
    assert not page_fragment_match("7", "Page 71")
    assert not page_fragment_match("7", "Page 17")


def test_page_expectation_keeps_text_fragments_case_insensitive():
    assert page_fragment_match("appendix", "Appendix A")


def test_answer_quality_summary_reports_labelled_accuracy_and_abstention():
    from tests.eval_live_corpus import summarize_answer_quality

    records = [
        {
            "case_id": "direct-fact",
            "expected_status": "success",
            "status": "success",
            "answer_quality": {
                "quality_checkable": True,
                "quality_pass": True,
                "answer_term_groups": [True, True],
                "status_match": True,
            },
        },
        {
            "case_id": "missing-fact",
            "expected_status": "abstain",
            "status": "insufficient_evidence",
            "abstained": True,
            "answer_quality": {
                "quality_checkable": True,
                "quality_pass": True,
                "answer_term_groups": [],
                "status_match": True,
            },
        },
        {
            "case_id": "wrong-answer",
            "expected_status": "success",
            "status": "success",
            "answer_quality": {
                "quality_checkable": True,
                "quality_pass": False,
                "answer_term_groups": [False],
                "status_match": True,
            },
        },
    ]

    summary = summarize_answer_quality(records)

    assert summary["labelled_cases_checked"] == 3
    assert summary["labelled_cases_passed"] == 2
    assert summary["labelled_regression_pass_rate"] == 0.667
    assert summary["failed_case_ids"] == ["wrong-answer"]
    assert summary["answer_term_groups_checked"] == 3
    assert summary["answer_term_groups_passed"] == 2
    assert summary["expected_abstentions"] == 1
    assert summary["correct_abstentions"] == 1
    assert summary["abstention_recall_on_labelled_unanswerable_cases"] == 1.0
    assert "not a calibrated semantic" in summary["interpretation"]


def test_answer_quality_summary_handles_no_labelled_cases():
    from tests.eval_live_corpus import summarize_answer_quality

    summary = summarize_answer_quality([])
    assert summary["labelled_cases_checked"] == 0
    assert summary["labelled_regression_pass_rate"] is None
    assert summary["answer_term_group_pass_rate"] is None
    assert summary["abstention_recall_on_labelled_unanswerable_cases"] is None
