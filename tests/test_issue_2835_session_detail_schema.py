"""#2835: under session delivery, lcm_recall's `detail` text names no hits-mode cap."""

import copy

from hermes_lcm.recall_session import session_schema
from hermes_lcm.tool_registry import TOOL_REGISTRY


def _recall_schema():
    return next(spec.schema for spec in TOOL_REGISTRY if spec.name == "lcm_recall")


def test_session_detail_text_matches_session_delivery():
    hits = _recall_schema()
    before = copy.deepcopy(hits)
    detail = session_schema(hits)["parameters"]["properties"]["detail"]
    assert hits == before  # the hits-delivery schema is left as it was
    text = detail["description"]
    for hits_only in ("five hits", "2400", "first eight", "compact hit response"):
        assert hits_only not in text, hits_only
    assert "either value" in text and "seen_refs" in text and "include_occurrence_time" in text
    hits_detail = before["parameters"]["properties"]["detail"]
    assert (detail["enum"], detail["default"]) == (hits_detail["enum"], hits_detail["default"])
