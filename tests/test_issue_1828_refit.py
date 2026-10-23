"""Final-refit retries use whole, already-read groups in admission order."""
import copy
import inspect
import json
import sys
import time
from types import SimpleNamespace

import pytest

from hermes_lcm import recall_session, tokens
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore


class RefitFixture:
    def __init__(self, sizes, session_ids=None):
        self.store = MessageStore(":memory:")
        self.dag = SummaryDAG(":memory:")
        self.engine = SimpleNamespace(_store=self.store, _dag=self.dag, current_session_id="current")
        self.ordered = []
        for i, size in enumerate(sizes):
            sid = session_ids[i] if session_ids else f"session-{i}"
            row_id = self.store.append(sid, {"role": "user", "content": "x" * size, "timestamp": 1})
            self.ordered.append(dict(hit=dict(session_id=sid, store_id=row_id), ranks={}, _final_score=1))

    def close(self):
        self.dag.close()
        self.store.close()

    def run(self, *, seen=(), deadline=None, trace=False, query="needle"):
        response = dict(query=query, limit=len(self.ordered), provenance=dict(arms_run=[]))
        self.skipped = []
        if trace:
            source, start = inspect.getsourcelines(recall_session.deliver_session)
            admission_line = start + next(i for i, line in enumerate(source) if "if evidence_fits():" in line)

        def profile(frame, event, arg):
            if event == "return" and frame.f_code.co_name == "evidence_fits" and arg is False:
                caller = frame.f_back
                if caller.f_code is recall_session.deliver_session.__code__ and caller.f_lineno == admission_line:
                    if caller.f_locals["fit_failure"] == "budget":
                        self.skipped.append(copy.deepcopy(caller.f_locals["pending"]))

        prior = sys.getprofile()
        if trace:
            sys.setprofile(profile)
        try:
            return json.loads(recall_session.deliver_session(
                self.ordered, response, engine=self.engine,
                deadline=time.monotonic() + 30 if deadline is None else deadline,
                excluded=set(), allowed=None, seen_refs=set(seen), delta_requested=True,
                event_time_fields=lambda row: {}, occurrence=False, summary_leads=[]))
        finally:
            if trace:
                sys.setprofile(prior)


def fits_addition(result, hit, cap):
    """Independent final-envelope oracle, including truthful session metadata."""
    trial = copy.deepcopy(result)
    trial["hits"].append(copy.deepcopy(hit))
    trial["hits"].sort(key=lambda h: h["store_id"])
    trial["total_results"] = len(trial["hits"])
    refs = [h["exact_ref"] for h in trial["hits"] if h.get("exact_ref")]
    trial["delta"].update(novel_ref_count=len(refs), novel_refs=refs, progress=bool(refs),
                          termination_reason=None if refs else "no_novel_exact_ref")
    d = trial["provenance"]["delivery"]
    d.update(message_count=len(trial["hits"]), session_count=len({h["session_id"] for h in trial["hits"]}))
    for meta in d["sessions"]:
        if meta["session_id"] == hit["session_id"]:
            meta.update(complete=True, omission_reason=None, returned_id_ranges=[[hit["store_id"], hit["store_id"]]])
            meta.pop("expand_hint", None)
    d["truncated"] = bool(d.get("stop_reason")) or any(not m["complete"] for m in d["sessions"])
    while True:
        for _ in range(10):
            raw = json.dumps(trial, ensure_ascii=False)
            size = (tokens.count_tokens_with_encoder(raw, encoder=None), len(raw))
            if size == (d["response_tokens"], d["response_chars"]):
                break
            d["response_tokens"], d["response_chars"] = size
        else:
            return False
        if len(raw) <= cap and size[0] <= recall_session.TOKEN_BUDGET:
            return True
        hints = [m for m in d["sessions"] if "expand_hint" in m]
        if not hints:
            return False
        hints[-1].pop("expand_hint")


@pytest.fixture
def refit_fixture(monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    fixtures = []

    def make(sizes):
        fixture = RefitFixture(sizes)
        fixtures.append(fixture)
        return fixture

    yield make
    for fixture in fixtures:
        fixture.close()


@pytest.mark.parametrize("sizes", [(1600, 100), (1600, 200, 100)])
def test_refit_reconsiders_skipped_groups(refit_fixture, monkeypatch, sizes):
    fixture = refit_fixture(sizes)
    monkeypatch.setattr(recall_session, "CHAR_BUDGET", 9000)
    full = fixture.run()
    # A was admitted, B (and C) skipped, then final omission metadata
    # forced A out. Base is empty at these caps, including the two-to-zero shape.
    monkeypatch.setattr(recall_session, "CHAR_BUDGET", 3240 if len(sizes) == 2 else 3370)
    refitted = fixture.run()
    assert [h["store_id"] for h in refitted["hits"]] == list(range(2, len(sizes) + 1))
    recovered = []
    for cap in range(1500, 9001, 10):
        monkeypatch.setattr(recall_session, "CHAR_BUDGET", cap)
        result = fixture.run(trace=True)
        ids = {h["store_id"] for h in result["hits"]}
        for group in fixture.skipped:
            hit = group[0]
            if hit["store_id"] not in ids:
                assert not fits_addition(result, hit, cap), (cap, ids, hit["store_id"])
        if ids and 1 not in ids:
            recovered.append(cap)
        for meta in result["provenance"]["delivery"]["sessions"]:
            delivered = meta["seed_ids"][0] in ids
            assert meta["complete"] == delivered
            if delivered:
                assert meta["omission_reason"] is None
                assert "expand_hint" not in meta
    assert recovered
    assert len(full["hits"]) == len(sizes)
    monkeypatch.setattr(recall_session, "CHAR_BUDGET", 9000)
    seen = full["hits"][1]["exact_ref"]
    delta = fixture.run(seen=[seen], trace=True)
    assert 2 not in {h["store_id"] for h in delta["hits"]}
    assert all(h["exact_ref"] != seen for group in fixture.skipped for h in group)
    expired = fixture.run(deadline=-1)
    assert expired["timeout"] and not expired["hits"]
    neighbor = fixture.store.append("session-1", {"role": "user", "content": "unwalked", "timestamp": 1})
    monkeypatch.setattr(recall_session, "CHAR_BUDGET", 3240 if len(sizes) == 2 else 3370)
    partial = fixture.run()
    assert 2 in {h["store_id"] for h in partial["hits"]}
    assert neighbor not in {h["store_id"] for h in partial["hits"]}
    meta = next(m for m in partial["provenance"]["delivery"]["sessions"] if m["session_id"] == "session-1")
    assert not meta["complete"] and meta["omission_reason"] == "budget"
    assert meta["expand_hint"] == f"lcm_expand(store_id={neighbor})"
    if len(sizes) == 3:
        original = fixture.store.load_session_window

        def expire(sid, **kwargs):
            if sid == "session-2":
                raise TimeoutError
            return original(sid, **kwargs)

        monkeypatch.setattr(fixture.store, "load_session_window", expire)
        monkeypatch.setattr(recall_session, "CHAR_BUDGET", 3370)
        stopped = fixture.run()
        assert stopped["timeout"]
        assert 2 not in {h["store_id"] for h in stopped["hits"]}
