"""Recall's private OR query uses the harness's content-term spelling."""
import pytest


@pytest.mark.parametrize(("query", "expected"), [
    ("beagle, puppy!", "beagle OR puppy"),
    ("What name did we give the beagle puppy?", "name OR we OR give OR beagle OR puppy"),
    ("The BEAGLE Puppy", "BEAGLE OR Puppy"),
    ("beagle puppy beagle puppy?", "beagle OR puppy"),
    ("art-related art", "artrelated OR art OR related"),
    ("", ""),
    ("what did the?", ""),
    ("?!,", ""),
])
def test_build_recall_or_query(query, expected):
    from hermes_lcm.search_query import build_recall_or_query

    assert build_recall_or_query(query) == expected
