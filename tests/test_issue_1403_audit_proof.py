"""#1403: NULL-stamp audit consumption requires occurrence-bound evidence."""
from copy import deepcopy

import pytest

from tests.test_host_uid_shadow import _engine, _m, _state_db


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("proof", ["canonical", "absent", "other-uid", "other-lineage", "version", "engine", "read-error"])
def test_null_copy_needs_its_canonical_uid(tmp_path, monkeypatch, role, proof):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        original = _m(role, "same text", uid="old")
        engine.ingest([original])
        old = engine._store.get_session_messages("S")[0]
        assert old["observed_at"] is None
        lineage = engine._host_uid_lineage_key()[0]
        assert engine._store.host_uid_bindings_for(lineage, ["old"])["old"] == [(old["store_id"], "canonical")]
        uid = "old" if proof in {"canonical", "read-error"} else "fresh"
        if proof in {"other-lineage", "version", "engine"}:
            engine._store.add_host_uid_bindings(
                "other" if proof == "other-lineage" else lineage,
                [(old["store_id"], uid, "canonical" if proof == "other-lineage" else proof, "test")])
        if proof == "read-error":
            def unavailable(*args, **kwargs):
                raise RuntimeError("synthetic unreadable binding table")
            monkeypatch.setattr(engine._store, "host_uid_bindings_for", unavailable)
        host = [_m(role, "same text", 10., None if proof == "absent" else uid)]
        plan = engine._identity_anchor_prematch(host, host, 1, 0)
        proven = proof == "canonical"
        assert plan["cursor"] == (1 if proven else 0)
        assert plan["backfill"] == ([(old["store_id"], 10.)] if proven else [])
        assert plan.get("matched", {}) == ({0: [old]} if proven else {})
        # Exercise the ingest consumers of the audit, not only the returned plan.
        prematch = engine._identity_anchor_prematch
        monkeypatch.setattr(engine, "_identity_anchor_prematch",
                            lambda messages, identities, cursor, audit_from=None:
                            prematch(messages, identities, cursor, 0))
        engine.ingest(deepcopy(host))
        rows = engine._store.get_session_messages("S")
        assert len(rows) == (1 if proven else 2)
        assert rows[0]["observed_at"] == (10. if proven else None)
        engine.ingest(deepcopy(host))
        assert len(engine._store.get_session_messages("S")) == len(rows)
    finally:
        engine.shutdown()


def test_same_stamp_rule_and_one_use_per_copy(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m("user", "alpha", uid="one")])
        original = engine._store.get_session_messages("S")[0]
        host = [_m("user", "alpha", 10., "one"), _m("user", "alpha", 11., "one")]
        plan = engine._identity_anchor_prematch(host, host, 2, 0)
        assert plan["cursor"] == 1
        assert plan["backfill"] == [(original["store_id"], 10.)]
        assert set(plan["matched"]) == {0}
        engine._identity_anchor_commit(plan)
        # Even a matching UID cannot override #1395's different-known-stamp rule.
        later = [host[1]]
        assert engine._identity_anchor_prematch(later, later, 1, 0)["cursor"] == 0
    finally:
        engine.shutdown()
