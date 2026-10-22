"""Pure parsing tests for scripts/maintainer/postmerge_issue_check.py (no network)."""

from __future__ import annotations

import pytest

from scripts.maintainer.postmerge_issue_check import main, parse_issue_refs


def test_one_keyword_with_a_list_targets_every_listed_issue():
    assert parse_issue_refs("Fixes #461, #463 and #464").closing == {461, 463, 464}


def test_lowercase_close_keyword():
    assert parse_issue_refs("closes #5").closing == {5}


@pytest.mark.parametrize("text", ["Part of #7", "see #8", "Follow-up: #9", "Related to #10"])
def test_non_closing_references_are_not_targets(text):
    parsed = parse_issue_refs(text)
    assert parsed.closing == set()
    assert parsed.refs  # still listed as a reference


def test_part_of_is_recorded_separately():
    parsed = parse_issue_refs("Part of #7; see #8")
    assert parsed.part_of == {7}
    assert parsed.closing == set()
    assert parsed.refs == [7, 8]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("FIXES #1", {1}),
        ("Fixed #2", {2}),
        ("fix #3", {3}),
        ("Closed #4", {4}),
        ("CLOSE #5", {5}),
        ("Resolves #6", {6}),
        ("resolved #7", {7}),
        ("ReSoLvE #8", {8}),
    ],
)
def test_mixed_case_keywords(text, expected):
    assert parse_issue_refs(text).closing == expected


def test_mixed_body_separates_closing_from_mentions():
    text = "Fixes #461, #463 and #464\n\nPart of #400. See #12 for context.\nCloses #20"
    parsed = parse_issue_refs(text)
    assert parsed.closing == {461, 463, 464, 20}
    assert parsed.part_of == {400}
    assert parsed.refs == [12, 20, 400, 461, 463, 464]


def test_pr_number_is_excluded_from_refs():
    parsed = parse_issue_refs("Merge pull request #776 from fork\n\nFixes #772", pr_number=776)
    assert parsed.refs == [772]
    assert parsed.closing == {772}


def test_keyword_inside_a_word_is_not_a_target():
    assert parse_issue_refs("prefixes #5 and suffixed #6").closing == set()


@pytest.mark.parametrize("argv", [["check"], ["check", "abc"], ["check", "#12"]])
def test_usage_errors_exit_2_before_any_github_call(argv):
    assert main(argv) == 2
