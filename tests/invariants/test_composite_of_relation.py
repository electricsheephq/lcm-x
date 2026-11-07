"""Stored todo composites gain provenance without changing replay or stored bytes."""
import pytest

from hermes_lcm.reconcile import _PRESERVED_TODO_CONTEXT_PREFIX as HEADER
from tests.test_issue_436_identity_anchor import _a, _engine, _relations, _rows, _u

TODO = HEADER + "\n- [ ] retain context (pending)"
P1, P2 = "first stored prompt\n\nwith whole paragraphs", "second stored prompt"


@pytest.mark.parametrize("parts,stored_todo", [(1, False), (2, False), (1, True), (2, True)])
def test_stored_composite_provenance(tmp_path, parts, stored_todo):
    engine = _engine(tmp_path)
    try:
        texts = ([TODO] if stored_todo else []) + [P1, P2][:parts]
        history = [_u(text, 500.0) for text in texts] + [_a("reply", 501.0)]
        engine.ingest(history)
        ids = {r["content"]: r["store_id"] for r in _rows(engine)}
        composite = _u("\n\n".join([TODO, *[P1, P2][:parts]]), 500.0 if stored_todo else None)
        engine.ingest([*history, composite])
        row = _rows(engine)[-1]
        assert row["content"] == composite["content"]
        targets = ([ids[TODO]] if stored_todo else [None]) + [ids[p] for p in [P1, P2][:parts]]
        relations = [r for r in _relations(engine) if r[1] == "composite_of"]
        assert relations == [(row["store_id"], "composite_of", target, i) for i, target in enumerate(targets)]
        assert all(r["observed_at"] == row["observed_at"] for r in engine._store.get_message_relations(
            [row["store_id"]], "composite_of"))
        engine.ingest([*history, composite])
        assert [r for r in _relations(engine) if r[1] == "composite_of"] == relations
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", ["no-header", "two-leftovers", "duplicate", "byte", "separator",
                                  "two-splits", "plain", "other-session", "other-conversation"])
def test_no_relation_without_unique_allowed_decomposition(tmp_path, case):
    engine = _engine(tmp_path)
    try:
        history = [_u(P1, 500.0), _a("reply", 501.0)]
        if case == "duplicate":
            history += [_u(P1, 502.0), _a("reply twice", 503.0)]
        if case == "two-splits":
            history += [_u(P2, 502.0), _a("reply two", 503.0), _u(P1 + "\n\n" + P2, 504.0)]
        if case == "plain":
            history += [_u(P2, 502.0)]
        if case in {"other-session", "other-conversation"}:
            engine.ingest(history)
            engine.on_session_start("B", platform="cli", context_length=200_000,
                                    conversation_id="other" if case == "other-conversation" else "conv")
            history = [_u(TODO, 500.0), _a("new reply", 501.0)]
        engine.ingest(history)
        content = TODO + "\n\n" + P1
        if case == "no-header":
            content = "unknown scaffold\n\n" + P1
        elif case == "two-leftovers":
            content += "\n\nunknown tail"
        elif case == "byte":
            content += "!"
        elif case == "separator":
            content = TODO + "\n" + P1
        elif case == "two-splits":
            content += "\n\n" + P2
        elif case == "plain":
            content = P1 + "\n\n" + P2
        engine.ingest([*history, _u(content, None)])
        assert _rows(engine)[-1]["content"] == content
        assert not [r for r in _relations(engine) if r[1] == "composite_of"]
    finally:
        engine.shutdown()


def test_fully_decomposable_plain_replay_is_unchanged(tmp_path):
    engine = _engine(tmp_path)
    try:
        history = [_u(P1, 500.0), _u(P2, 500.0)]
        engine.ingest(history)
        before = _rows(engine)
        engine.ingest([_u(P1 + "\n\n" + P2, 500.0)])
        assert _rows(engine) == before
        assert [r[2] for r in _relations(engine) if r[1] == "composite"] == [r["store_id"] for r in before]
        assert not [r for r in _relations(engine) if r[1] == "composite_of"]
    finally:
        engine.shutdown()


def test_relation_failure_preserves_ingest(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path)
    try:
        history = [_u(P1, 500.0), _a("reply", 501.0)]
        engine.ingest(history)
        add = engine._store.add_message_relations

        def fail(groups):
            if any(r[1] == "composite_of" for group in groups for r in group):
                add(groups)  # prove the evidence savepoint rolls back a partial write too
                raise RuntimeError("synthetic relation failure")
            return add(groups)

        monkeypatch.setattr(engine._store, "add_message_relations", fail)
        engine.ingest([*history, _u(TODO + "\n\n" + P1, None)])
        assert [r["content"] for r in _rows(engine)] == [P1, "reply", TODO + "\n\n" + P1]
        assert not [r for r in _relations(engine) if r[1] == "composite_of"]
        assert "LCM composite-of relation write failed (RuntimeError)" in caplog.text
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", ["both-rules", "trailing-scaffold", "stored-todo-second", "gap"])
def test_no_relation_when_rules_overlap_or_scaffold_is_not_leading(tmp_path, case):
    """Uniqueness is across both rules, and the scaffold must lead (the host folds the todo list in first)."""
    engine = _engine(tmp_path)
    try:
        stored = {"both-rules": [P1, TODO + "\n\n" + P1, P2], "trailing-scaffold": [P1],
                  "stored-todo-second": [P1, TODO], "gap": [P1, P2]}[case]
        content = {"both-rules": TODO + "\n\n" + P1 + "\n\n" + P2,  # (i) [scaffold, P1, P2] and (ii) [TODO+P1, P2]
                   "trailing-scaffold": P1 + "\n\n" + TODO,
                   "stored-todo-second": P1 + "\n\n" + TODO,
                   "gap": TODO + "\n\n" + P1 + "\n\nunknown middle\n\n" + P2}[case]
        history = []
        for i, text in enumerate(stored):
            history += [_u(text, 500.0 + 2 * i), _a(f"reply {i}", 501.0 + 2 * i)]
        engine.ingest(history)
        assert [r["content"] for r in _rows(engine) if r["role"] == "user"] == stored
        engine.ingest([*history, _u(content, None)])
        row = _rows(engine)[-1]
        assert row["content"] == content
        assert not [r for r in _relations(engine) if r[1] == "composite_of" and r[0] == row["store_id"]]
    finally:
        engine.shutdown()


def test_scaffold_never_holds_a_whole_stored_row(tmp_path):
    """A stored paragraph after the todo block is a part, so the only allowed split is [scaffold, note, P1]."""
    engine = _engine(tmp_path)
    try:
        note = "a stored note"
        history = [_u(note, 500.0), _a("reply", 501.0), _u(P1, 502.0), _a("reply two", 503.0)]
        engine.ingest(history)
        ids = {r["content"]: r["store_id"] for r in _rows(engine)}
        engine.ingest([*history, _u(TODO + "\n\n" + note + "\n\n" + P1, None)])
        row = _rows(engine)[-1]
        assert [r for r in _relations(engine) if r[1] == "composite_of"] == [
            (row["store_id"], "composite_of", target, i) for i, target in enumerate([None, ids[note], ids[P1]])]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", ["cross-session-copy", "overlapping-separator", "header-not-leading",
                                  "blank-scaffold"])
def test_writer_finds_every_split(tmp_path, case):
    """Writer-level shapes: a rule (ii) split still counts when a part also has a copy outside C's session; cuts
    at overlapping separators ("\\n\\n\\n") are all tried; text before the header is never a scaffold."""
    engine = _engine(tmp_path)
    try:
        store, conversation = engine._store, engine._conversation_id

        def put(session, text):
            return store.append(session, _u(text, None), conversation_id=conversation)

        if case == "cross-session-copy":  # (i) [scaffold, P1, P2] and (ii) [R in B, P2]: two splits
            composite = TODO + "\n\n" + P1
            put("A", P1), put("A", composite), put("B", composite), put("B", P2)
            target = put("B", composite + "\n\n" + P2)
        elif case == "overlapping-separator":  # [TODO, "\nP"] and [TODO + "\n", "P"]: two splits
            put("S", "\nP"), put("S", "P")
            target = put("S", TODO + "\n\n\nP")
        elif case == "blank-scaffold":  # cut 1 of "\n\n\n" + TODO leaves the scaffold "\n", which is no todo block
            put("S", TODO)
            target = put("S", "\n\n\n" + TODO)
        else:
            put("S", P1)
            target = put("S", "intro\n\n" + TODO + "\n\n" + P1)
        engine._identity_anchor_record_composite_of([target])
        assert not [r for r in _relations(engine) if r[1] == "composite_of"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", ["scaffold-holds-leading-todo", "overlap-after-stored-todo"])
def test_scaffold_bounds_at_the_writer(tmp_path, case):
    """A scaffold that starts with a whole stored todo row is not unstored; with an overlapping separator after a
    stored todo row, the next cut ("TODO\\n") is still tried, so the one allowed split is found."""
    engine = _engine(tmp_path)
    try:
        store, conversation = engine._store, engine._conversation_id
        todo = store.append("S", _u(TODO, None), conversation_id=conversation)
        if case == "scaffold-holds-leading-todo":
            store.append("S", _u(P1, None), conversation_id=conversation)
            target = store.append("S", _u(TODO + "\n\nunstored note\n\n" + P1, None), conversation_id=conversation)
            expected = []
        else:
            prompt = store.append("S", _u("P", None), conversation_id=conversation)
            target = store.append("S", _u(TODO + "\n\n\nP", None), conversation_id=conversation)
            expected = [(target, "composite_of", None, 0), (target, "composite_of", prompt, 1)]
        assert todo < target
        engine._identity_anchor_record_composite_of([target])
        assert [r for r in _relations(engine) if r[1] == "composite_of"] == expected
    finally:
        engine.shutdown()
