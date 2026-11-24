"""Preserve coverage while limiting copy verification to rows that can change it."""

from hermes_lcm import lifecycle_state
from hermes_lcm.dag import SummaryNode
from hermes_lcm.summary_prefix import _summary_part_text
from tests.test_issue_671_stub_first_exit import make_engine  # noqa: F401


def test_publication_keeps_covered_carrier_without_verifying_it(make_engine, monkeypatch):  # noqa: F811
    engine = make_engine(embeddings_enabled=False, temporal_rollups_enabled=False)
    store = engine._store
    session, conversation = engine._session_id, engine._conversation_id
    first, original = store.append_batch(session, [
        {"role": "assistant", "content": "Previous reply."},
        {"role": "user", "content": "Continue the work.", "timestamp": 20},
    ], conversation_id=conversation)

    def publish(expected, covered, excluded=()):
        staged = []

        def stage(conn, node_id):
            staged.append(engine._lifecycle.stage_compaction_publication(
                conn, conversation, session, node_id, expected, covered, list(excluded),
            ))

        node_id = engine._dag.add_node(SummaryNode(
            session_id=session, summary="Published turns.", expand_hint="turns", source_ids=covered,
        ), before_commit=stage)
        return engine._dag.get_node(node_id), staged

    prior, staged = publish(0, [first])
    assert staged == [first]
    carrier = {"role": "user", "content": _summary_part_text(prior) + "\n\nContinue the work.", "timestamp": 20}
    covered_copy, omitted_copy, excluded_id, last = store.append_batch(session, [
        carrier, carrier,
        {"role": "assistant", "content": "An explicitly excluded reply."},
        {"role": "assistant", "content": "Last covered reply."},
    ], conversation_id=conversation)
    store.add_message_relations([[(i, "generated_unproven", None, None, None) for i in (covered_copy, omitted_copy)]])
    assert lifecycle_state.known_head_copy_ids(store.connection, session) == {covered_copy, omitted_copy}

    verified_ids = []
    verify = lifecycle_state.known_head_copy_ids

    def record(conn, *, store_ids):
        verified_ids.append(store_ids)
        return verify(conn, store_ids=store_ids)

    monkeypatch.setattr(lifecycle_state, "known_head_copy_ids", record)
    covered = [original, covered_copy, last]
    published, staged = publish(first, covered, [excluded_id])
    assert staged == [last]
    assert engine._lifecycle.get_by_conversation(conversation).current_frontier_store_id == last
    assert published.source_ids == covered
    assert store.get(covered_copy)["content"] == store.get(omitted_copy)["content"] == carrier["content"]
    assert verified_ids == [[omitted_copy]]
