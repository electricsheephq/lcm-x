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
    """Uniqueness is across both rules, and the scaffold must lead: the only todo-first composite the host makes is
    its consecutive-user merge of a standalone snapshot row with the next prompt. A block the host appends after a
    real user turn ("U\\n\\nBLOCK") is matched by the #516 identity cut instead (see the trailing test below)."""
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


NOTICE = "[Skills pruned during compression — reload before acting on these tasks]\n- skill-a"
# case: (earlier stored user rows [(session, text)], composite text stored in session S,
#        expected parts in order: an index into the rows, None for the scaffold; [] means no relation)
WRITER_CASES = {
    "own-session-parts-only": ([("A", P1), ("A", TODO + "\n\n" + P1), ("S", TODO + "\n\n" + P1), ("S", P2)],
                               TODO + "\n\n" + P1 + "\n\n" + P2, [2, 3]),
    "sibling-todo-row-does-not-block": ([("A", TODO), ("S", P1)], TODO + "\n\n" + P1, [None, 1]),
    "overlapping-separator": ([("S", "\nP"), ("S", "P")], TODO + "\n\n\nP", [None, 0]),
    "stored-todo-then-overlap": ([("S", TODO), ("S", "P")], TODO + "\n\n\nP", []),
    "blank-scaffold": ([("S", TODO)], "\n\n\n" + TODO, []),
    "header-not-leading": ([("S", P1)], "intro\n\n" + TODO + "\n\n" + P1, []),
    "scaffold-holds-leading-todo": ([("S", TODO), ("S", P1)], TODO + "\n\nunstored note\n\n" + P1, []),
    "absorbed-unstored-text": ([("S", P1)], TODO + "\n\nreal user words\n\n" + P1, []),
    "absorbed-stored-text": ([("S", "real user words"), ("S", P1)], TODO + "\n\nreal user words\n\n" + P1,
                             [None, 0, 1]),
    "header-only": ([("S", P1)], HEADER + "\n\n" + P1, []),
    "header-prefixed-user-text": ([("S", P1)], HEADER + " and my own words\n\n" + P1, []),
    "words-on-the-header-line": ([("S", P1)], HEADER + " and my own words\n- [ ] 1. task (pending)\n\n" + P1, []),
    "reload-notice": ([("S", P1)], TODO + "\n\n" + NOTICE + "\n\n" + P1, [None, 0]),
    "item-with-blank-line": ([("S", P1)], HEADER + "\n- [ ] 1. first line\n\nsecond line (pending)\n\n" + P1,
                             [None, 0]),
    "open-item-without-status": ([("S", P1)], HEADER + "\n- [ ] 1. open item\n\n" + P1, []),
}


@pytest.mark.parametrize("case", list(WRITER_CASES))
def test_writer_scaffold_and_scope(tmp_path, case):
    """The scaffold is exactly the rendered todo block (#516 span: header, item lines, optional reload notice);
    parts and the uniqueness count are C's own session; overlapping separators and blank cuts are handled."""
    rows, content, expected = WRITER_CASES[case]
    engine = _engine(tmp_path)
    try:
        store, conversation = engine._store, engine._conversation_id
        ids = [store.append(session, _u(text, None), conversation_id=conversation) for session, text in rows]
        target = store.append("S", _u(content, None), conversation_id=conversation)
        engine._identity_anchor_record_composite_of([target])
        assert [r for r in _relations(engine) if r[1] == "composite_of"] == [
            (target, "composite_of", None if part is None else ids[part], ordinal)
            for ordinal, part in enumerate(expected)]
    finally:
        engine.shutdown()


def test_trailing_block_on_a_stored_turn_is_not_stored_again(tmp_path):
    """Characterization: the host appends its todo block after a real trailing user turn ("U\\n\\nBLOCK"); the
    #516 identity cut matches it to the stored U, so no new row (and no composite) is written."""
    engine = _engine(tmp_path)
    try:
        engine.ingest([_u(P2, 500.0), _a("reply", 501.0)])
        before = _rows(engine)
        engine.ingest([_u(P2 + "\n\n" + TODO, 500.0), _a("reply", 501.0)])
        assert _rows(engine) == before
        assert not [r for r in _relations(engine) if r[1] == "composite_of"]
    finally:
        engine.shutdown()


def test_middle_block_characterization(tmp_path):
    """Characterization: a cancel or error before the reply, then the host merges the next prompt V behind
    "U\\n\\nBLOCK". Today the merged row is stored as a new row carrying U's stamp (carried_stamp -> U), with
    no composite_of: the writer does not cover this shape (it is not todo-first)."""
    engine = _engine(tmp_path)
    try:
        engine.ingest([_u(P1, 500.0), _a("reply", 501.0), _u(P2, 502.0)])
        merged = P2 + "\n\n" + TODO + "\n\nthe next prompt"
        engine.ingest([_u(P1, 500.0), _a("reply", 501.0), _u(merged, 502.0)])
        # V is stored only in this row, so it must stay (release-manager ruling: not a loss, no writer change).
        assert [(r["store_id"], r["content"]) for r in _rows(engine)] == [
            (1, P1), (2, "reply"), (3, P2), (4, merged)]
        assert _relations(engine) == [(4, "carried_stamp", 3, None)]
    finally:
        engine.shutdown()


R, Q = "a repeated driver prompt", "another repeated prompt"
# case: (earlier rows of session S [(role, text)], composite text, expected parts: index into rows / None)
HOLDER_CASES = {
    # Single part, text held twice: link only the row immediately before C.
    "single-holder-just-before": ([("user", R), ("assistant", "a"), ("user", R)], TODO + "\n\n" + R, [None, 2]),
    "single-holder-not-just-before": ([("user", R), ("assistant", "a"), ("user", R), ("assistant", "b")],
                                      TODO + "\n\n" + R, []),
    # Several parts: exactly one assignment of consecutive rows (no row of any role between), in text order.
    "run-picks-the-second-occurrence": ([("user", TODO), ("assistant", "a"), ("user", TODO), ("user", R),
                                         ("user", R)], TODO + "\n\n" + R, [2, 3]),
    "two-runs-write-nothing": ([("user", TODO), ("user", R), ("assistant", "a"), ("user", TODO), ("user", R)],
                               TODO + "\n\n" + R, []),
    "holders-out-of-order": ([("user", Q), ("user", R), ("user", R)], TODO + "\n\n" + R + "\n\n" + Q, []),
    "scaffold-then-run": ([("user", R), ("user", Q), ("assistant", "a"), ("user", R)],
                          TODO + "\n\n" + R + "\n\n" + Q, [None, 0, 1]),
    "assistant-between-parts": ([("user", R), ("assistant", "a"), ("user", Q), ("user", R)],
                                TODO + "\n\n" + R + "\n\n" + Q, []),
    # One stored occurrence never stands for two parts: a repeated part needs as many distinct rows.
    "one-row-two-equal-parts": ([("user", R)], TODO + "\n\n" + R + "\n\n" + R, []),
}


@pytest.mark.parametrize("case", list(HOLDER_CASES))
def test_holder_rule_for_identical_texts(tmp_path, case):
    """Release-manager holder rule: when a part's text is held by several rows of C's session, the host's
    merge of consecutive turns decides; anything else stays fail-closed."""
    rows, content, expected = HOLDER_CASES[case]
    engine = _engine(tmp_path)
    try:
        store, conversation = engine._store, engine._conversation_id
        ids = [store.append("S", {"role": role, "content": text}, conversation_id=conversation) for role, text in rows]
        target = store.append("S", _u(content, None), conversation_id=conversation)
        engine._identity_anchor_record_composite_of([target])
        assert [r for r in _relations(engine) if r[1] == "composite_of"] == [
            (target, "composite_of", None if part is None else ids[part], ordinal)
            for ordinal, part in enumerate(expected)]
    finally:
        engine.shutdown()


def test_holder_search_is_linear_in_session_history(tmp_path, monkeypatch):
    """NeonDiff: N rows each of R and Q, an assistant row after every R. Resolving TODO + R + Q must not
    compare every R holder against every Q holder (holder membership is a set lookup)."""
    import hermes_lcm.identity_anchor as anchor
    n, calls = 300, {"n": 0}
    original = anchor._composite_of_holders

    class CountingList(list):
        def __contains__(self, item):
            calls["n"] += len(self)  # a linear scan costs one comparison per element
            return list.__contains__(self, item)

    def counting(choices, order):
        return original([CountingList(c) for c in choices], order)

    monkeypatch.setattr(anchor, "_composite_of_holders", counting)
    engine = _engine(tmp_path)
    try:
        store, conversation = engine._store, engine._conversation_id
        for _ in range(n):
            store.append("S", {"role": "user", "content": R}, conversation_id=conversation)
            store.append("S", {"role": "assistant", "content": "a"}, conversation_id=conversation)
            store.append("S", {"role": "user", "content": Q}, conversation_id=conversation)
        target = store.append("S", _u(TODO + "\n\n" + R + "\n\n" + Q, None), conversation_id=conversation)
        engine._identity_anchor_record_composite_of([target])
        assert calls["n"] <= 8 * n, calls["n"]
    finally:
        engine.shutdown()
