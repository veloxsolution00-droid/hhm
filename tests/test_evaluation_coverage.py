from tests.eval_live_corpus import page_fragment_match


def test_page_expectation_matches_full_numeric_page_only():
    assert page_fragment_match("7", "Page 7")
    assert not page_fragment_match("7", "Page 71")
    assert not page_fragment_match("7", "Page 17")


def test_page_expectation_keeps_text_fragments_case_insensitive():
    assert page_fragment_match("appendix", "Appendix A")
