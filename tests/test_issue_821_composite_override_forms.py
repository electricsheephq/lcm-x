"""#821: composite constituents use recorded replay forms, never whitespace guesses."""
import hashlib
import json

import pytest

from tests.test_issue_436_identity_anchor import SYSTEM, _a, _engine, _relations, _rows, _state_db, _turns, _u


def _override(engine, row, content):
    payload = {"version": 1, "content": content,
               "stored_sha256": hashlib.sha256(row["content"].encode()).hexdigest()}
    engine._store.write_metadata_json([f"host_rewrite_identity:{row['store_id']}"], json.dumps(payload))
    engine._host_rewrite_state()[1].pop(int(row["store_id"]), None)


@pytest.mark.parametrize("host_form", ["r34.4", "upstream", "recorded-override"])
def test_new_absorbed_turn_cannot_bind_older_stored_occurrence(tmp_path, host_form):
    """F1: an out-of-view 'continue' is not the new occurrence absorbed after a crash."""
    engine = _engine(tmp_path)
    head = [SYSTEM, *_turns(1, 2, 0.0)]
    try:
        engine.ingest([*head, _u("continue", 300.0), _a("ok old", 301.0), _u("held R\n", 500.0)])
        row = _rows(engine)[-1]
        if host_form == "recorded-override":
            engine._record_ws_host_rewrite(row, _u("held R", 500.0))
        engine.shutdown()
        engine = _engine(tmp_path)
        composite = _u("held R\n\ncontinue", 500.0)
        if host_form != "recorded-override":
            composite["_merged_turn_prefix"] = "held R" + ("\n\n" if host_form == "r34.4" else "")
        engine.ingest([*head, composite, _a("reply", 511.0)])
        rows = _rows(engine)
        assert [r["content"] for r in rows].count("continue") == 2
        assert composite["content"] not in [r["content"] for r in rows]
        remainder = rows[-2]
        assert remainder["content"] == "continue" and remainder["observed_at"] is None
        assert [rel[2] for rel in _relations(engine) if rel[1] == "composite"] == [
            row["store_id"], remainder["store_id"]]
        engine.ingest([*head, _u("held R", 500.0), _u("continue", 510.0), _a("reply", 511.0)])
        assert len(_rows(engine)) == len(rows)
        assert next(r for r in _rows(engine) if r["store_id"] == remainder["store_id"])["observed_at"] == 510.0
    finally:
        engine.shutdown()


@pytest.mark.parametrize("host_form", ["r34.4", "upstream", "recorded-override"])
def test_rotation_child_stores_new_absorbed_occurrence(tmp_path, host_form):
    """The same F1 loss shape with the dangling donor in the rotation parent."""
    _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None)])
    engine = _engine(tmp_path, "P")
    head = [SYSTEM, *_turns(1, 2, 0.0)]
    try:
        engine.ingest([*head, _u("continue", 300.0), _a("ok old", 301.0),
                       *_turns(40, 2, 350.0), _u("held R\n", 500.0)])
        if host_form == "recorded-override":
            engine._record_ws_host_rewrite(_rows(engine)[-1], _u("held R", 500.0))
        engine.shutdown()
        engine = _engine(tmp_path, "C")
        composite = _u("held R\n\ncontinue", 500.0)
        if host_form != "recorded-override":
            composite["_merged_turn_prefix"] = "held R" + ("\n\n" if host_form == "r34.4" else "")
        engine.ingest([SYSTEM, *_turns(41, 1, 360.0), composite, _a("reply", 511.0)])
        assert [r["content"] for r in _rows(engine, "P")].count("continue") == 1
        assert [r["content"] for r in _rows(engine, "C")].count("continue") == 1
        assert composite["content"] not in [r["content"] for r in _rows(engine, "C")]
    finally:
        engine.shutdown()


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


@pytest.mark.parametrize("host_form", ["r34.4", "upstream"])
@pytest.mark.parametrize("recorded", [False, True], ids=["first-composite", "same-override"])
def test_merge_witness_records_head_before_r3_and_adoption(tmp_path, host_form, recorded):
    engine = _engine(tmp_path)
    head = [SYSTEM, *_turns(1, 2, 0.0)]
    raw, rest = "held R with  inner spaces\n", "  new U\n\nsecond  paragraph\n\t"
    prefix = raw.strip()
    try:
        engine.ingest([*head, _u(raw, 500.0)])
        row = _rows(engine)[-1]
        if recorded:
            engine._record_ws_host_rewrite(row, _u(prefix, 500.0))
        engine.shutdown()
        engine = _engine(tmp_path)
        composite = _u(prefix + "\n\n" + rest, 500.0)
        composite["_merged_turn_prefix"] = prefix + ("\n\n" if host_form == "r34.4" else "")
        plan = engine._identity_anchor_prematch([*head, composite], [*head, composite], 0)
        assert plan["remainders"][len(head)] == (rest, 500.0, [row], [prefix])
        key = f"host_rewrite_identity:{row['store_id']}"
        payload = engine._store.read_metadata_json(key)
        assert payload["content"] == prefix
        engine.ingest([*head, composite, _a("reply", 511.0)])
        texts = [r["content"] for r in _rows(engine)]
        assert texts.count(raw) == texts.count(rest) == 1
        assert composite["content"] not in texts
        remainder = next(r for r in _rows(engine) if r["content"] == rest)
        assert remainder["observed_at"] is None
        assert [rel[2] for rel in _relations(engine) if rel[1] == "composite"] == [
            row["store_id"], remainder["store_id"]]
        before = len(_rows(engine))
        engine.ingest([*head, _u(prefix, 500.0), _u(rest, 510.0), _a("reply", 511.0)])
        assert len(_rows(engine)) == before
        assert engine._store.read_metadata_json(key) == payload
        # Adoption's identical capture uses the same payload and skip_unchanged writer.
        changes = engine._store._conn.total_changes
        engine._record_ws_host_rewrite(row, _u(prefix, 500.0))
        assert engine._store._conn.total_changes == changes
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", [
    "no-marker", "non-string", "inner-whitespace", "two-donors", "no-stamp", "wrong-prefix",
    "both-forms", "multi-row-head", "lossy-head", "lossy-stored", "different-override",
])
def test_unbound_merge_witness_stores_whole(tmp_path, case):
    engine = _engine(tmp_path)
    head = [SYSTEM, *_turns(1, 2, 0.0)]
    raw, prefix = "held R with  spaces\n", "held R with  spaces"
    if case == "both-forms":
        raw = "\t" + raw  # the raw form must not independently explain the composite
    if case == "lossy-stored":
        prefix = "held [LCM sensitive redaction: name=password_assignment; length=8] R"
        raw = prefix + "\n"
    try:
        engine.ingest([*head, _u(raw, 500.0)])
        row = _rows(engine)[-1]
        key = f"host_rewrite_identity:{row['store_id']}"
        if case == "different-override":
            _override(engine, row, " " + prefix)
        elif case == "two-donors":
            engine.ingest([*head, _u(raw, 500.0), _u("other donor\n", 500.0)])
        elif case == "multi-row-head":
            engine.ingest([*head, _u(raw, 500.0), _u("second stored part\n", 510.0)])
        engine.shutdown()
        engine = _engine(tmp_path)
        before = engine._store.read_metadata_json(key)
        if case == "inner-whitespace":
            prefix = prefix.replace("  ", " ")
        elif case == "lossy-head":
            prefix = "held [LCM sensitive redaction: name=password_assignment; length=8] R"
        elif case == "multi-row-head":
            prefix += "\n\nsecond stored part"
        composite = _u(prefix + "\n\nnew U", None if case == "no-stamp" else 500.0)
        marker = prefix + "\n\n"
        if case == "wrong-prefix":
            marker = "different head\n\n"
        elif case == "both-forms":
            composite["content"] = marker + "\n\nnew U"
        elif case == "non-string":
            marker = 12
        if case != "no-marker":
            composite["_merged_turn_prefix"] = marker
        engine.ingest([*head, composite, _a("reply", 511.0)])
        assert composite["content"] in [r["content"] for r in _rows(engine)]
        assert not [rel for rel in _relations(engine) if rel[1] == "composite"]
        assert engine._store.read_metadata_json(key) == before
    finally:
        engine.shutdown()


def test_merge_witness_preserves_different_override_even_if_replay_identity_agrees(tmp_path):
    engine = _engine(tmp_path)
    raw, prefix = "held R\n", "held R"
    override = "[Note: model was just switched from X to Y.]\n\n" + prefix
    try:
        engine.ingest([SYSTEM, _u(raw, 500.0)])
        row = _rows(engine)[-1]
        _override(engine, row, override)
        composite = _u(prefix + "\n\nnew U", 500.0)
        composite["_merged_turn_prefix"] = prefix + "\n\n"
        assert engine._message_replay_identity(row, stored_row=True, with_host_rewrite=True)[1] == prefix
        engine._capture_merged_user_head(composite, [(row, engine._stored_row_forms(row))], set())
        assert engine._host_rewrite_override_content(row) == override
    finally:
        engine.shutdown()


def test_merge_witness_write_failure_keeps_ingest_and_redacts_log(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path)
    head = [SYSTEM, *_turns(1, 2, 0.0)]
    raw, prefix = "held R\n", "held R"
    try:
        engine.ingest([*head, _u(raw, 500.0)])
        row = _rows(engine)[-1]
        engine.shutdown()
        engine = _engine(tmp_path)
        write = engine._store.write_metadata_json

        def fail(keys, *args, **kwargs):
            if keys == [f"host_rewrite_identity:{row['store_id']}"]:
                raise RuntimeError("private exception details")
            return write(keys, *args, **kwargs)

        monkeypatch.setattr(engine._store, "write_metadata_json", fail)
        composite = _u(prefix + "\n\nnew U", 500.0)
        composite["_merged_turn_prefix"] = prefix + "\n\n"
        engine.ingest([*head, composite, _a("reply", 511.0)])
        assert composite["content"] in [r["content"] for r in _rows(engine)]
        assert not [rel for rel in _relations(engine) if rel[1] == "composite"]
        assert engine._store.read_metadata_json(f"host_rewrite_identity:{row['store_id']}") is None
        assert f"store_id {row['store_id']}" in caplog.text and "RuntimeError" in caplog.text
        assert "private exception details" not in caplog.text
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
