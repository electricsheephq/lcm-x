"""Regression for PR 1689 split."""
import sys
import pytest
from .bot1_support import DECISION


@pytest.mark.parametrize("source", ["versions", "correction_source", "stale", "stale_history"])
def test_split_v2_rejects_all_superseded_values_preserving_v1(source):
    sys.path.insert(0, str(DECISION))
    from split_scoring import fact_verdicts
    fact = dict(id="F1", fixture="alpha", value="current-id", **{"class": "name"})
    if source == "versions":
        fact[source] = [dict(value="obsolete-id"), dict(value="current-id")]
    elif source == "correction_source":
        fact[source] = dict(value="obsolete-id")
    elif source == "stale_history":
        fact[source] = [dict(value="obsolete-id")]
    else:
        fact[source] = "obsolete-id"
    answer = "current-id and obsolete-id"
    v2 = fact_verdicts(fact, answer, "", [fact], rule="c1b-terms-v2")
    assert v2["association"] is False and v2["semantic"] is False
    v1 = fact_verdicts(fact, answer, "", [fact], rule="c1b-terms-v1")
    assert v1["association"] is (source != "stale")
    assert fact_verdicts(fact, "current-id", "", [fact], rule="c1b-terms-v2")["association"] is True
