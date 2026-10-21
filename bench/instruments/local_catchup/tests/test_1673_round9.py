"""Reject transient safety gaps and non-exact real-turn trace persistence."""
from argparse import Namespace as NS
import hashlib

import pytest

from bench.instruments.local_catchup import audit, common, measurement

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


def test_transient_covered_retries_each_check_safety(tmp_path, monkeypatch):
    transient = dict(complete=False, raw_unchanged=True, changed_vectors=[],
        changed_metadata=[], unexpected_rows=[], integrity="ok",
        classes={"chunks": dict(missing_count=0)}, residue={"dispatched": 1})
    audits = iter([transient, transient, transient, dict(complete=True)])
    monkeypatch.setattr(measurement, "audit", lambda *a: next(audits))
    monkeypatch.setattr(audit, "vector_counts", lambda *a: dict(chunks=1))
    calls = []
    monkeypatch.setattr(measurement, "safety", lambda a: calls.append("safety"))
    monkeypatch.setattr(measurement.time, "sleep", lambda _: calls.append("sleep"))
    a = NS(db=tmp_path/"unused.db", root=tmp_path, max_seconds=2)
    measurement.fill_covered(a, None, None, dict(expected=dict(chunks=["1:0"])))
    assert calls == ["safety", "sleep", "safety", "sleep"]


@pytest.mark.parametrize("change,kind", [
    ("exact", None), ("duplicate_last", "extra/duplicate"),
    ("altered_extra", "extra/duplicate"), ("missing", "missing"),
    ("altered", "altered"), ("reordered", "out-of-order"),
    ("system", None), ("tool", None), ("legacy", None),
])
def test_integrity_requires_exact_real_turns(corpus, tmp_path, change, kind):
    db, manifest, mod = corpus
    messages = [dict(role=role, content=f"trace input {i}", timestamp=1700000000.+i)
                for i, role in enumerate(("user", "assistant", "user", "assistant"))]
    manifest["trace_rows"] = [["trace-session", "trace-conversation", m["role"],
        m["timestamp"], hashlib.sha256(m["content"].encode()).hexdigest()] for m in messages]
    actual = list(messages)
    if change == "duplicate_last":
        actual.append(messages[-1])
    elif change == "altered_extra":
        actual.insert(1, dict(messages[0], content="altered extra input"))
    elif change == "missing":
        actual.pop(1)
    elif change == "altered":
        actual[1] = dict(messages[1], content="altered input")
    elif change == "reordered":
        actual[0], actual[1] = actual[1], actual[0]
    elif change in ("system", "tool"):
        actual.insert(1, dict(role=change, content="product addition"))
    elif change == "legacy":
        del manifest["trace_rows"]
        actual.append(messages[-1])
    store = mod("store").MessageStore(str(db))
    try:
        for message in actual:
            store.append("trace-session", message, conversation_id="trace-conversation")
    finally:
        store.close()
    a = NS(db=db, root=tmp_path)
    if kind:
        with pytest.raises(RuntimeError, match=f"trace row corruption: {kind}") as error:
            measurement.integrity(a, manifest)
        assert "input" not in str(error.value)
    else:
        measurement.integrity(a, manifest)
        if change != "legacy":
            expected_other = {change: 1} if change in ("system", "tool") else {}
            assert common.load(tmp_path/"final-integrity.json")["other_role_rows"] == expected_other
