"""The budget pass reserves the final sanitizer's missing-result stubs (#1048).

A tail whose assistant turn has tool calls without results gets one plain stub per call from
the final sanitizer. The forced-recovery path already reserved them; the normal path did not,
so a full summary budget overflowed after sanitizing and the over-cap pass dropped the
preserved user objective first. With the reservation, the selector drops its lowest-priority
summary part instead and the objective stays. On main, the first test fails at the objective
assertion (red-proof recorded on PR #1049).
"""

import logging

import hermes_lcm.engine as lcm_engine
from hermes_lcm.dag import SummaryNode
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens
from tests.test_active_tool_stubbing import make_engine as make_engine  # noqa: F401 (fixture)
from tests.test_active_tool_stubbing import tool_pair


SEPARATOR = "\n\n---\n\n"
OBJECTIVE = "Deliver the current user's requested objective: repair the orchard map."
SYSTEM = {"role": "system", "content": "system"}
WARNING = "dropping the preserved user objective"


def add_summaries(engine, words=100):
    nodes = []
    for index, (depth, tag) in enumerate(((2, "DEEPEST"), (0, "SHALLOW_OLD"), (0, "SHALLOW_NEW")), 1):
        node = SummaryNode(
            session_id=engine._session_id,
            depth=depth,
            summary=tag + " " + "orchard " * words,
            token_count=words + 1,
            source_token_count=words + 1,
            source_ids=[],
            source_type="messages",
            created_at=float(index),
            earliest_at=float(index),
            latest_at=float(index),
            expand_hint="recover stored context",
        )
        node.node_id = engine._dag.add_node(node)
        nodes.append(node)
    return nodes


def assistant_calls(*call_ids):
    message = tool_pair(call_ids[0], "unused")[0]
    message["tool_calls"] = [{**message["tool_calls"][0], "id": call_id} for call_id in call_ids]
    return message


def result(call_id):
    return {"role": "tool", "tool_call_id": call_id, "content": "result body"}


def full_cap(engine, nodes, tail):
    """A cap that admits the objective, every summary part and the raw tail, nothing more."""
    engine._pending_context_anchor_messages = [{"role": "user", "content": OBJECTIVE}]
    prefix = SEPARATOR.join([
        engine._build_preserved_objective_summary_part({"role": "user", "content": OBJECTIVE}),
        *(lcm_engine._summary_part_text(node) for node in nodes),
    ])
    return count_messages_tokens([SYSTEM, {"role": "user", "content": prefix}, *tail])


def assemble(engine, tail, cap):
    return lcm_engine.LCMEngine._assemble_context(
        engine, SYSTEM, tail, assembly_cap_override=cap, include_lcm_note=False)


def view(messages):
    return "\n".join(str(message.get("content", "")) for message in messages)


def test_objective_survives_when_calls_lack_results(make_engine, caplog):  # noqa: F811
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    nodes = add_summaries(engine)
    tail = [assistant_calls("call-0", "call-1", "call-2")]
    cap = full_cap(engine, nodes, tail)
    with caplog.at_level(logging.WARNING):
        out = assemble(engine, tail, cap)

    assert OBJECTIVE in view(out)
    assert "DEEPEST" in view(out)
    assert count_messages_tokens(out) <= cap
    # The selector, not the over-cap pass, made room: one depth-0 part is gone.
    assert ("SHALLOW_OLD" in view(out)) + ("SHALLOW_NEW" in view(out)) == 1
    assert WARNING not in caplog.text


def test_complete_tool_pairs_keep_every_part(make_engine, caplog):  # noqa: F811
    """No missing results: nothing is reserved, so every part and the objective stay (as on main)."""
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    nodes = add_summaries(engine)
    tail = [assistant_calls("call-0", "call-1"), result("call-0"), result("call-1")]
    cap = full_cap(engine, nodes, tail)
    with caplog.at_level(logging.WARNING):
        out = assemble(engine, tail, cap)

    for marker in (OBJECTIVE, "DEEPEST", "SHALLOW_OLD", "SHALLOW_NEW"):
        assert marker in view(out)
    assert WARNING not in caplog.text


def test_reused_call_id_counts_only_this_turns_results(make_engine):  # noqa: F811
    """An older turn's missing `same` result is not satisfied by a newer turn's `same` result."""
    engine = make_engine(large_output_active_replay_stubbing_enabled=False, fresh_tail_count=10)
    nodes = add_summaries(engine)
    tail = [
        assistant_calls("same"),
        {"role": "user", "content": "continue"},
        assistant_calls("same"),
        result("same"),
    ]
    cap = full_cap(engine, nodes, tail)
    out = assemble(engine, tail, cap)

    assert OBJECTIVE in view(out)
    assert count_messages_tokens(out) <= cap


def test_rejected_turn_releases_its_kept_results(make_engine):  # noqa: F811
    """A turn rejected by the reservation stops charging its already-kept result to the tail.

    The result fits on its own and is kept first (newest-first walk); its turn plus the stub for
    the missing call does not fit. Charged as an orphan, it would starve the objective.
    """
    engine = make_engine(large_output_active_replay_stubbing_enabled=False, fresh_tail_count=10)
    engine._pending_context_anchor_messages = [{"role": "user", "content": OBJECTIVE}]
    objective_msg = {"role": "user", "content": engine._build_preserved_objective_summary_part(
        {"role": "user", "content": OBJECTIVE})}
    base = count_messages_tokens([SYSTEM, objective_msg])
    objective_tokens = base - count_messages_tokens([SYSTEM])
    words = 1
    while count_message_tokens({"role": "tool", "tool_call_id": "kept", "content": "payload " * words}) \
            < objective_tokens + 10:
        words += 1
    kept = {"role": "tool", "tool_call_id": "kept", "content": "payload " * words}
    tail = [assistant_calls("kept", "missing"), kept]
    cap = base + 20  # the kept result fits alone; the whole turn and its stub do not
    assert count_messages_tokens([SYSTEM, kept]) <= cap
    out = assemble(engine, tail, cap)

    assert OBJECTIVE in view(out)
    assert "payload" not in view(out)


def test_last_resort_strip_still_happens_and_warns(make_engine, monkeypatch, caplog):  # noqa: F811
    """An overflow from a source the budget cannot see keeps today's last resort, now logged."""
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    nodes = add_summaries(engine)
    tail = [assistant_calls("call-0", "call-1"), result("call-0"), result("call-1")]
    cap = full_cap(engine, nodes, tail)
    original = lcm_engine.LCMEngine._sanitize_active_context_messages

    def inflating(self, messages, **options):
        cleaned = original(self, messages, **options)
        if options.get("insert_missing_tool_stubs", True) and any(
            lcm_engine._PRESERVED_OBJECTIVE_CONTEXT_PREFIX in str(m.get("content", "")) for m in cleaned
        ):
            # Simulate an unbudgeted final-pass addition on top of a full budget.
            cleaned = [*cleaned, {"role": "user", "content": "unbudgeted " * 400}]
        return cleaned

    monkeypatch.setattr(lcm_engine.LCMEngine, "_sanitize_active_context_messages", inflating)
    with caplog.at_level(logging.WARNING):
        out = assemble(engine, tail, cap)

    assert OBJECTIVE not in view(out)
    assert WARNING in caplog.text
    assert "unbudgeted" not in caplog.text  # the warning carries counts, never content


def test_no_warning_when_no_objective_was_assembled(make_engine, caplog):  # noqa: F811
    """An oversized fixed prefix with no room for the objective must not report losing it."""
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    engine._pending_context_anchor_messages = [{"role": "user", "content": OBJECTIVE}]
    big_system = {"role": "system", "content": "system " * 200}
    with caplog.at_level(logging.WARNING):
        out = lcm_engine.LCMEngine._assemble_context(
            engine, big_system, [{"role": "user", "content": "hi"}],
            assembly_cap_override=1, include_lcm_note=False)

    assert OBJECTIVE not in view(out)
    assert WARNING not in caplog.text


def test_repeated_call_id_reserves_one_stub_per_unmatched_occurrence(make_engine):  # noqa: F811
    """Two `same` calls with one `same` result: the sanitizer stubs the second occurrence."""
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    nodes = add_summaries(engine)
    tail = [assistant_calls("same", "same"), result("same")]
    cap = full_cap(engine, nodes, tail)
    out = assemble(engine, tail, cap)

    assert OBJECTIVE in view(out)
    assert count_messages_tokens(out) <= cap
