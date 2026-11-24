"""Compare the three copy predicates with the release line's text-only proof."""

from functools import cache
from types import SimpleNamespace

import pytest

from hermes_lcm.head_copy import (
    _head_copy_candidate,
    _read_head_copy_candidates,
    is_known_head_copy,
    known_head_copies_in_rows,
    known_head_copy_ids,
)
from hermes_lcm.store import MessageStore
from hermes_lcm.summary_prefix import summary_coverage_end, verify_summary_prefix


# The a63ddbb9c predicate, kept here as the reference for the subset check.
def _base_verify_marked_head_copy(conn, row) -> bool:
    content = row["content"]
    store_id, session_id = row["store_id"], row["session_id"]

    def get_node(node_id):
        node = conn.execute(
            "SELECT session_id, depth, summary, expand_hint FROM summary_nodes WHERE node_id=?", (node_id,)
        ).fetchone()
        return SimpleNamespace(node_id=node_id, session_id=node[0], depth=node[1],
                               summary=node[2], expand_hint=node[3] or "") if node else None

    @cache
    def predecessors():
        lineage = conn.execute(
            "WITH RECURSIVE lineage(session, hops) AS (VALUES (?, 0) UNION ALL "
            "SELECT CASE WHEN json_valid(m.value) THEN "
            "CASE WHEN json_type(m.value)='text' THEN json_extract(m.value, '$') END END, hops+1 "
            "FROM lineage JOIN metadata m ON m.key='rotation_predecessor_session:' || session WHERE hops<32) "
            "SELECT session FROM lineage WHERE hops>0 ORDER BY hops", (session_id,),
        )
        found = []
        for (predecessor,) in lineage:
            if not predecessor or predecessor == session_id or predecessor in found:
                break
            found.append(predecessor)
        return found

    def owns_node(owner):
        return owner == session_id or owner in predecessors()

    end, node_ids = verify_summary_prefix(content, get_node, owns_node)
    if end is None or end == len(content):
        return end is not None
    suffix = content[end:]
    for separator in ("\n\n---\n\n", "\n\n"):
        if suffix.startswith(separator):
            remainder = suffix[len(separator):]
            if not remainder.strip():
                return False
            covered_through = summary_coverage_end(conn, node_ids)
            sessions = [session_id] if covered_through is None else [session_id, *predecessors()]
            prior_users = conn.execute(
                "SELECT content, conversation_id FROM messages "
                f"WHERE session_id IN ({','.join('?' * len(sessions))}) AND role='user' "
                "AND store_id>? AND store_id<? ORDER BY store_id", (*sessions, covered_through or 0, store_id),
            )
            conversation = (row.get("conversation_id") or "").strip()
            for prior_content, prior_conversation in prior_users:
                if (prior_conversation or "").strip() not in ("", conversation):
                    continue
                if prior_content == remainder:
                    return True
                if covered_through is None:
                    return False
            return False
    return False


def _base_is_known_head_copy(conn, row) -> bool:
    """Prove this stored user occurrence is fully represented by earlier LCM context."""
    if conn is None or not _head_copy_candidate(row):
        return False
    relation = _read_head_copy_candidates(
        conn, "SELECT 1 FROM message_relations WHERE store_id=? AND kind='generated_unproven'", (row["store_id"],)
    )
    return relation is not None and relation.fetchone() is not None and _base_verify_marked_head_copy(conn, row)


def _audit_head_copies(store):
    conn = store.connection
    if conn is None:
        return
    marked = _read_head_copy_candidates(conn, """
        SELECT DISTINCT r.store_id FROM message_relations r JOIN messages m USING (store_id)
        WHERE r.kind='generated_unproven' AND m.role='user' AND m.content GLOB '[[]*'
    """, ())
    if marked is None or not (ids := [store_id for store_id, in marked]):
        return
    rows = store.get_batch(ids)
    base = {store_id for store_id, row in rows.items() if _base_is_known_head_copy(conn, row)}
    copies = {store_id for store_id, row in rows.items() if is_known_head_copy(conn, row)}
    assert copies <= base, copies - base
    assert known_head_copies_in_rows(conn, rows.values()) == copies
    assert known_head_copy_ids(conn, store_ids=ids) == copies
    return base, copies


@pytest.fixture
def head_copy_subset(monkeypatch):
    close = MessageStore.close

    def audited_close(store):
        try:
            _audit_head_copies(store)
        finally:
            close(store)

    monkeypatch.setattr(MessageStore, "close", audited_close)


def test_head_copy_subset_is_strict_on_unbound_carriers(tmp_path):
    from hermes_lcm.dag import SummaryDAG, SummaryNode
    from hermes_lcm.summary_prefix import _summary_part_text

    store = MessageStore(tmp_path / "subset.db")
    dag = SummaryDAG(store.db_path)
    try:
        covered, original = store.append_batch("subset", [
            {"role": "assistant", "content": "Past reply."},
            {"role": "user", "content": "continue", "timestamp": 20},
        ])
        node_id = dag.add_node(SummaryNode(session_id="subset", summary="Past turns.",
                                           expand_hint="past", source_ids=[covered]))
        head = _summary_part_text(dag.get_node(node_id))
        pure, replay, fresh = store.append_batch("subset", [
            {"role": "user", "content": head},
            {"role": "user", "content": head + "\n\ncontinue", "timestamp": 20},
            {"role": "user", "content": head + "\n\ncontinue", "timestamp": 30},
        ])
        store.add_message_relations([[(i, "generated_unproven", None, None, None) for i in (pure, replay, fresh)]])
        base, copies = _audit_head_copies(store)
        assert original not in base
        assert base == {pure, replay, fresh}
        assert copies == {pure, replay}
        incomplete = store.get(replay)
        del incomplete["observed_at"]
        assert not is_known_head_copy(store.connection, incomplete)
        assert known_head_copies_in_rows(store.connection, [incomplete]) == set()
    finally:
        dag.close()
        store.close()
