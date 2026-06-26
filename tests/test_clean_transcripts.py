"""Unit tests for ES-05 transcript cleaning.

All fixtures are built inline as strings — no file I/O and no dependency on the
real (gitignored) dataset, so these run in CI.
"""

from src.data.clean_transcripts import (
    classify_role,
    extract_prepared_remarks,
    extract_speaker_blocks,
    is_attribution_line,
    normalize_transcript,
    parse_transcript,
)


def test_normalize_replaces_nbsp():
    """normalize_transcript turns \\xa0 into a regular space."""
    assert normalize_transcript("Jane\xa0--\xa0CEO") == "Jane -- CEO"
    assert "\xa0" not in normalize_transcript("a\xa0b\xa0c")


def test_extract_prepared_remarks_boundaries():
    """The slice runs from the Prepared Remarks header to the Q&A header."""
    text = (
        "Prepared Remarks:\n"
        "Jane Doe -- Chief Executive Officer\n"
        "Hello there.\n"
        "Questions and Answers:\n"
        "Analyst chatter we do not want.\n"
    )
    prepared = extract_prepared_remarks(text)
    assert "Hello there." in prepared
    assert "Questions and Answers" not in prepared
    assert "Analyst chatter" not in prepared


def test_extract_prepared_remarks_no_qa_runs_to_end():
    """With no Q&A header the slice extends to the end of the string."""
    text = "Prepared Remarks:\nJane Doe -- Chief Executive Officer\nClosing thoughts."
    prepared = extract_prepared_remarks(text)
    assert prepared.endswith("Closing thoughts.")


def test_extract_prepared_remarks_missing_header_returns_none():
    """No Prepared Remarks header -> None."""
    assert extract_prepared_remarks("Some call with no header at all.") is None


def test_is_attribution_line_accepts_valid():
    """A well-formed Name -- Title line is accepted."""
    assert is_attribution_line("Jane Doe -- Chief Executive Officer")


def test_is_attribution_line_filters():
    """Lines failing any one rule are rejected."""
    # No separator.
    assert not is_attribution_line("Just a sentence of prose.")
    # Ends in sentence punctuation.
    assert not is_attribution_line("We grew margins -- a lot this year.")
    # Too long (>= 100 chars).
    assert not is_attribution_line("A" + " -- " + "B" * 200)
    # Name side does not start with a capital.
    assert not is_attribution_line("lowercase -- Chief Executive Officer")
    # Corporate-name tail on the title side.
    assert not is_attribution_line("John Smith -- Morgan & Co.")
    assert not is_attribution_line("John Smith -- Baker & Associates")


def test_classify_role_basic():
    """Each role bucket is reachable with a representative title."""
    assert classify_role("Jane Doe -- Chief Executive Officer") == "ceo"
    assert classify_role("Bob Roe -- President and CEO") == "ceo"
    assert classify_role("Sue Lin -- Chairman") == "ceo"
    assert classify_role("Sue Lin -- Vice Chairman") != "ceo"
    assert classify_role("Al Ray -- Chief Financial Officer") == "cfo"
    assert classify_role("Pat Kim -- Chief Operating Officer") == "other_exec"
    assert classify_role("Lee Ann -- Analyst, Goldman Sachs") == "exclude"
    assert classify_role("No Title Here") == "exclude"


def test_classify_role_ordering_regression():
    """Ordering bug regression: 'EVP and CFO' must resolve to cfo, not exclude.

    The CFO check must run before the EVP exclusion; if the order were reversed
    the 'EVP' token would wrongly drop a genuine CFO.
    """
    line = "Chris Vale -- EVP and Chief Financial Officer"
    # Sanity: the exclusion token really is present in the line.
    assert "evp" in line.lower()
    assert classify_role(line) == "cfo"


def test_classify_role_two_dashes_excluded():
    """A line with 2+ '--' separators is malformed and excluded."""
    assert classify_role("Jane Doe -- Title -- Chief Executive Officer") == "exclude"


def test_extract_speaker_blocks_multi_turn_ceo_concatenation():
    """Multiple CEO turns are concatenated; excluded speakers are dropped."""
    prepared = (
        "Prepared Remarks:\n"
        "Jane Doe -- Chief Executive Officer\n"
        "First CEO remark.\n"
        "Operator -- Operator\n"
        "Please stand by.\n"
        "Jane Doe -- Chief Executive Officer\n"
        "Second CEO remark.\n"
        "Al Ray -- Chief Financial Officer\n"
        "CFO walks through numbers.\n"
    )
    blocks = extract_speaker_blocks(prepared)
    assert "First CEO remark." in blocks["ceo"]
    assert "Second CEO remark." in blocks["ceo"]
    assert "Please stand by." not in blocks["ceo"]
    assert blocks["cfo"] == "CFO walks through numbers."
    assert blocks["other_exec"] is None


def test_parse_transcript_non_string():
    """Non-string input yields all-None columns and section_parse_ok=False."""
    result = parse_transcript(None)
    assert result == {
        "text_prepared_ceo": None,
        "text_prepared_cfo": None,
        "text_prepared_other_exec": None,
        "section_parse_ok": False,
    }


def test_parse_transcript_section_parse_ok_true():
    """A transcript with a CEO turn parses ok even if the CFO is absent."""
    text = (
        "Prepared Remarks:\n"
        "Jane Doe\xa0--\xa0Chief Executive Officer\n"
        "Welcome everyone.\n"
        "Questions and Answers:\n"
    )
    result = parse_transcript(text)
    assert result["text_prepared_ceo"] == "Welcome everyone."
    assert result["text_prepared_cfo"] is None
    assert result["section_parse_ok"] is True


def test_parse_transcript_section_parse_ok_false_when_no_ceo_or_cfo():
    """section_parse_ok is False when neither CEO nor CFO is found."""
    text = (
        "Prepared Remarks:\n"
        "Pat Kim -- Chief Operating Officer\n"
        "Only an other-exec spoke.\n"
    )
    result = parse_transcript(text)
    assert result["text_prepared_ceo"] is None
    assert result["text_prepared_cfo"] is None
    assert result["text_prepared_other_exec"] == "Only an other-exec spoke."
    assert result["section_parse_ok"] is False


def test_parse_transcript_missing_header_false():
    """A string with no Prepared Remarks header parses to all-None / not ok."""
    result = parse_transcript("No header anywhere in this body.")
    assert result["section_parse_ok"] is False
    assert result["text_prepared_ceo"] is None
