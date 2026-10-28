"""A failed refit retry must leave the encoded payload unchanged."""
import inspect
import json
import time
from types import ModuleType

import pytest

from hermes_lcm import recall_session, tokens
from tests.test_issue_1828_refit import RefitFixture


@pytest.mark.parametrize("sizes,cap,uncertain,nonconverging_retry", [
    ((1600, 100), 1500, False, False),
    ((1600, 200, 100), 1500, False, False),
    ((100, 1600, 100), 2520, True, False),
    ((1600, 100), 3240, False, True),
])
def test_failed_retry_preserves_encoded_payload(monkeypatch, sizes, cap, uncertain, nonconverging_retry):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(recall_session, "CHAR_BUDGET", cap)
    # Same implementation and input, with only the retry pass disabled.
    source = inspect.getsource(recall_session)
    guard = 'if delivery.get("stop_reason") != "deadline":'
    assert source.count(guard) == 1
    baseline = ModuleType("hermes_lcm.retry_disabled")
    baseline.__package__ = "hermes_lcm"
    exec(compile(source.replace(guard, "if False:"), "<retry-disabled>", "exec"), baseline.__dict__)
    baseline.CHAR_BUDGET = cap
    fixture = RefitFixture(sizes)
    count = tokens.count_tokens_with_encoder
    oscillations = 0

    def retry_count(raw, *, encoder):
        nonlocal oscillations
        payload = json.loads(raw)
        d = payload["provenance"]["delivery"]
        # Only the tentative retry has B admitted with complete metadata.
        if nonconverging_retry and any(h["store_id"] == 2 for h in payload["hits"]) and d["sessions"][1]["complete"]:
            oscillations += 1
            return 1 + d["response_tokens"] % 2
        return count(raw, encoder=encoder)

    monkeypatch.setattr(tokens, "count_tokens_with_encoder", retry_count)

    def run(module):
        return module.deliver_session(
            fixture.ordered, dict(query="needle", limit=len(sizes), provenance=dict(arms_run=[])),
            engine=fixture.engine, deadline=time.monotonic() + 30,
            excluded=set(), allowed=None, seen_refs=set(), delta_requested=True,
            event_time_fields=(lambda row: dict(time_uncertain="carried", event_time="2020-01-01",
                                                event_time_source="synthetic")) if uncertain else lambda row: {},
            occurrence=False, summary_leads=[dict(session_id="lead", node_id=999,
                expand_hint="lcm_expand(node_id=999)")] if uncertain else [])

    try:
        expected, actual = run(baseline), run(recall_session)
    finally:
        fixture.close()
    before, after = json.loads(expected), json.loads(actual)
    if nonconverging_retry:
        assert oscillations >= 10
    assert after["provenance"]["delivery"].get("stop_reason") == before["provenance"]["delivery"].get("stop_reason")
    assert [h["store_id"] for h in before["hits"]] == [h["store_id"] for h in after["hits"]]
    if uncertain:
        assert any(h.get("event_time") for h in before["hits"])
    elif not nonconverging_retry:
        assert any("expand_hint" in s for s in before["provenance"]["delivery"]["sessions"])
    assert actual == expected
