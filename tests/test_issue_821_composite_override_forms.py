"""#821: composite constituents use recorded replay forms, never whitespace guesses."""
import hashlib
import json

import pytest

from tests.test_issue_436_identity_anchor import SYSTEM, _a, _engine, _relations, _rows, _turns, _u


def _override(engine, row, content):
    payload = {"version": 1, "content": content,
               "stored_sha256": hashlib.sha256(row["content"].encode()).hexdigest()}
    engine._store.write_metadata_json([f"host_rewrite_identity:{row['store_id']}"], json.dumps(payload))
    engine._host_rewrite_state()[1].pop(int(row["store_id"]), None)


def test_r3_override_head_keeps_exact_remainder_and_adoption(tmp_path):
    engine = _engine(tmp_path)
    head = [SYSTEM, *_turns(1, 2, 0.0)]
    r = _u("held R with  inner spaces\n", 500.0)
    rest = "  new U\n\nsecond  paragraph\n\t"
    try:
        engine.ingest([*head, r])
        row = _rows(engine)[-1]
        _override(engine, row, r["content"].strip())
        engine.shutdown()
        engine = _engine(tmp_path)
        composite = _u(r["content"].strip() + "\n\n" + rest, 500.0)
        engine.ingest([*head, composite, _a("reply", 511.0)])
        texts = [row["content"] for row in _rows(engine)]
        assert texts.count(r["content"]) == texts.count(rest) == 1
        assert composite["content"] not in texts
        remainder = next(row for row in _rows(engine) if row["content"] == rest)
        assert remainder["observed_at"] is None
        assert [rel[2] for rel in _relations(engine) if rel[1] == "composite"] == [
            row["store_id"], remainder["store_id"]]
        before = len(_rows(engine))
        engine.ingest([*head, _u(r["content"].strip(), 500.0), _u(rest, 510.0), _a("reply", 511.0)])
        assert len(_rows(engine)) == before
        assert [row["content"] for row in _rows(engine)].count(rest) == 1
    finally:
        engine.shutdown()


def test_r2_two_override_constituents_replay_and_witness(tmp_path):
    engine = _engine(tmp_path)
    head = [SYSTEM, *_turns(1, 2, 0.0)]
    r, u = _u("held R\n", 500.0), _u("held U\t", 510.0)
    try:
        engine.ingest([*head, r, u])
        rows = _rows(engine)[-2:]
        for row in rows:
            _override(engine, row, row["content"].strip())
        engine.shutdown()
        engine = _engine(tmp_path)
        composite = _u(r["content"].strip() + "\n\n" + u["content"].strip(), 500.0)
        live = [*head, composite, _a("reply", 511.0)]
        engine.ingest(live)
        assert composite["content"] not in [row["content"] for row in _rows(engine)]
        assert [rel[2] for rel in _relations(engine) if rel[1] == "composite"] == [
            row["store_id"] for row in rows]
        before = len(_rows(engine))
        engine.ingest([*head, dict(composite), _a("reply", 511.0)])
        assert len(_rows(engine)) == before
        engine._identity_anchor_text_memo = {}
        engine._last_compacted_store_id = rows[0]["store_id"] - 1
        assert engine._identity_anchor_summary_input([composite], {}, view=[composite])[0][1] == [
            row["store_id"] for row in rows]
        engine._last_compacted_store_id = rows[-1]["store_id"]
        assert engine._identity_anchor_covered_view(composite, {})
    finally:
        engine.shutdown()


def test_existing_raw_duplicate_constituents_keep_donor_order(tmp_path):
    engine = _engine(tmp_path)
    head = [SYSTEM, *_turns(1, 2, 0.0)]
    r, u = _u("identical raw occurrence", 500.0), _u("identical raw occurrence", 510.0)
    try:
        engine.ingest([*head, r, u])
        rows = _rows(engine)[-2:]
        engine.shutdown()
        engine = _engine(tmp_path)
        composite = _u(r["content"] + "\n\n" + u["content"], 500.0)
        engine.ingest([*head, composite, _a("reply", 511.0)])
        assert composite["content"] not in [row["content"] for row in _rows(engine)]
        assert [rel[2] for rel in _relations(engine) if rel[1] == "composite"] == [
            row["store_id"] for row in rows]
        engine._identity_anchor_text_memo = {}
        engine._last_compacted_store_id = rows[-1]["store_id"]
        assert engine._identity_anchor_covered_view(composite, {})
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", ["inner-whitespace", "no-override", "collision", "lossy"])
def test_unproven_or_ambiguous_constituent_stores_whole(tmp_path, case):
    engine = _engine(tmp_path)
    head = [SYSTEM, *_turns(1, 2, 0.0)]
    raw = "held R with  spaces\n"
    if case == "lossy":
        raw = "password=abcdefgh\n"
        engine._config.sensitive_patterns_enabled = True
        engine._config.sensitive_patterns = ["password_assignment"]
    r = _u(raw, 500.0)
    try:
        engine.ingest([*head, r])
        row = _rows(engine)[-1]
        form = raw.strip()
        if case == "inner-whitespace":
            _override(engine, row, form)
            form = form.replace("  ", " ")
        elif case == "collision":
            other = _u(raw.rstrip() + "\t", 510.0)
            engine.ingest([*head, r, other])
            for candidate in _rows(engine)[-2:]:
                _override(engine, candidate, form)
        elif case == "lossy":
            # Equal digest-less redactions are not identity, even with recorded metadata.
            form = row["content"].strip()
            from hermes_lcm.reconcile import _has_lossy_redacted_identity
            assert _has_lossy_redacted_identity(engine._message_replay_identity(_u(form, 500.0)))
            _override(engine, row, form)
        engine.shutdown()
        engine = _engine(tmp_path)
        composite = _u(form + "\n\nnew U", 500.0)
        engine.ingest([*head, composite, _a("reply", 511.0)])
        stored = engine._message_replay_identity(composite)[1]
        assert stored in [row["content"] for row in _rows(engine)]
        assert not [rel for rel in _relations(engine) if rel[1] == "composite"]
    finally:
        engine.shutdown()
