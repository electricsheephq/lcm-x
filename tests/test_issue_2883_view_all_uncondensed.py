"""#2883, I2: a summary's source content stays covered when the frontier grows."""

import hashlib
import time

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
from hermes_lcm.dag import SummaryNode
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens
from tests.harness import engine_context
from tests.test_issue_652_653_no_fragments_bounded_prefix import (
    _depth_0_nodes,
    _pin_lcm_char_counter as _pin_lcm_char_counter,
    _rendered,
    _turn,
)

pytestmark = pytest.mark.issue(2883)
TURN_BODY = "synthetic conversation content " * 64
PART_BODY = "synthetic summary text " * 35


def _covered_content(engine, nodes, result):
    def digest(content):
        return hashlib.sha256(content.encode()).digest()

    contents = [message["content"] for message in result]
    covered = {digest(content) for content in contents}
    rows = {row["store_id"]: digest(row["content"])
            for row in engine._store.get_session_messages("S", limit=-1)}
    summaries = {node.node_id: node for node in nodes}
    rendered = [node for node in nodes if any(lcm_engine._summary_part_text(node) in text for text in contents)]
    pending = rendered.copy()
    visited = set()
    while pending:
        node = pending.pop()
        if node.node_id in visited:
            continue
        visited.add(node.node_id)
        if node.source_type == "messages":
            covered.update(rows[source] for source in node.source_ids)
        else:
            pending.extend(summaries[source] for source in node.source_ids)
    return len(rendered), sum(content not in covered for content in rows.values())


@pytest.mark.parametrize("window,leaf_tokens,ratio", [(524_000, 200, None), (1_000_000, 400, 0.05)],
                         ids=["A-default", "B-explicit-ratio"])
def test_sweeps_cover_all_stored_content(tmp_path, monkeypatch, window, leaf_tokens, ratio):
    # Disable unrelated workers and payload handling, as in the original reproducer.
    settings = dict(temporal_rollups_enabled=False, embeddings_enabled=False,
                    large_output_externalization_enabled=False, large_output_active_replay_stubbing_enabled=False)
    if ratio is not None:
        settings["summary_prefix_target_ratio"] = ratio

    elapsed = 0.0
    monkeypatch.setattr(time, "monotonic", lambda: elapsed)
    budget_drops = []
    select = lcm_engine.select_prefix

    def record_selection(parts, facts):
        result = select(parts, facts)
        if facts.summary_budget is not None:
            budget_drops.append(len(parts) - len(result.kept_indexes))
        return result

    monkeypatch.setattr(lcm_engine, "select_prefix", record_selection)
    with engine_context(tmp_path, window=window, conversation_id="conv", platform="telegram", **settings) as engine:
        condensing = False
        condense = engine._condense_summary_nodes

        def summarize_group(*args, **kwargs):
            nonlocal condensing
            condensing = True
            try:
                return condense(*args, **kwargs)
            finally:
                condensing = False

        def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
            nonlocal elapsed
            if timeout is not None and timeout < 2.0:
                elapsed += timeout
                raise TimeoutError("synthetic summary timed out")
            elapsed += 2.0
            words = "condensed arc " if condensing else "lorem ipsum dolor sit amet "
            return (words * leaf_tokens)[:leaf_tokens * 4]

        monkeypatch.setattr(engine, "_condense_summary_nodes", summarize_group)
        monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
        # Start with the note already present, so adding it on first compaction cannot
        # change the stored system prompt's bytes during this coverage check.
        view = [{"role": "system", "content": engine._append_lcm_note_to_content("system prompt")}]
        turn = 0
        for sweep in range(1, 15):
            # Grow the real replay view to the same pressure as the reference probes.
            chars = sum(len(message["content"]) for message in view)
            while chars // 4 + 1 < engine.threshold_tokens + 1000:
                pair = [{"role": "user", "content": f"[U{turn}] {TURN_BODY}", "timestamp": 10.0 * (turn + 1)},
                        {"role": "assistant", "content": f"[A{turn}] {TURN_BODY}"}]
                view.extend(pair)
                chars += sum(len(message["content"]) for message in pair)
                turn += 1
            budget_drops.clear()
            engine.ingest(view)
            view = engine.compress(view)
            nodes = engine._dag.get_session_nodes("S", limit=-1)
            consumed = {source for node in nodes if node.source_type == "nodes" for source in node.source_ids}
            leaves = sum(node.depth == 0 and node.node_id not in consumed for node in nodes)
            rendered, uncovered = _covered_content(engine, nodes, view)
            details = (f"window={window} sweep={sweep} depth_0={leaves} rendered={rendered} "
                       f"budget_dropped={sum(budget_drops)} uncovered={uncovered}")
            print(details)
            assert uncovered == 0, details
            if leaves > 120:
                break
            engine.note_turn_complete()
            elapsed += 700.0
        else:
            pytest.fail(f"sweeps did not reach more than 120 uncondensed leaves: {details}")


@pytest.mark.parametrize("leaf_count", [134, 20], ids=["C-order", "D-complete-under-cap"])
def test_depth_groups_render_in_chronological_order(tmp_path, leaf_count):
    with engine_context(tmp_path, window=200_000) as engine:
        leaf_ids = _depth_0_nodes(engine, leaf_count)
        arc_ids = [engine._dag.add_node(SummaryNode(
            session_id="S", depth=1, summary=f"arc {index}", source_ids=leaf_ids[index:index + 1],
            source_type="nodes", created_at=float(100 - index),
        )) for index in range(4)]
        durable_ids = [engine._dag.add_node(SummaryNode(
            session_id="S", depth=2, summary=f"durable {index}", source_ids=arc_ids[index:index + 1],
            source_type="nodes", created_at=float(200 - index),
        )) for index in range(2)]
        result = _rendered(engine._assemble_context(None, _turn("tail", 5000.0)))

    by_depth = [(2, durable_ids[::-1]), (1, list(reversed(arc_ids[2:]))), (0, leaf_ids[4:])]
    full = [(depth, node_id) for depth, ids in by_depth for node_id in ids]
    if leaf_count < 100:
        assert result == full
    else:
        rendered = set(result)
        assert result == [part for part in full if part in rendered]
        for depth, ids in by_depth:
            assert (depth, ids[-1]) in rendered
        assert sum(depth == 0 for depth, _ in result) >= 100


def _cap_for_newest_forty(nodes, tail):
    parts = "\n\n---\n\n".join(lcm_engine._summary_part_text(node) for node in nodes[-40:])
    return count_messages_tokens(tail) + count_message_tokens({"role": "user", "content": parts})


def test_budget_keeps_the_same_newest_forty_parts(tmp_path):
    with engine_context(tmp_path, window=200_000) as engine:
        ids = _depth_0_nodes(engine, 130, token_count=200, text=PART_BODY)
        tail = _turn("tail", 5000.0)
        nodes = engine._dag.get_session_nodes("S", limit=-1)
        cap = _cap_for_newest_forty(nodes, tail)
        result = engine._assemble_context(None, tail, assembly_cap_override=cap)
    assert _rendered(result) == [(0, node_id) for node_id in ids[-40:]]


@pytest.mark.parametrize("node_count", [100, 500, 2_000])
def test_assembly_cost(tmp_path, node_count):
    # An unknown window allows a genuinely unset budget. The second call uses the same
    # fixed-size budget as the preceding test. Measure only assembly, after seeding.
    with engine_context(tmp_path, window=0) as engine:
        _depth_0_nodes(engine, node_count, token_count=200, text=PART_BODY)
        tail = _turn("tail", 5000.0)
        nodes = engine._dag.get_session_nodes("S", limit=-1)
        cap = _cap_for_newest_forty(nodes, tail)
        for budget, override in [("unset", None), ("set", cap)]:
            started = time.perf_counter()
            engine._assemble_context(None, tail, assembly_cap_override=override)
            elapsed = time.perf_counter() - started
            print(f"assembly nodes={node_count} budget={budget} ms={elapsed * 1000:.3f}")
            if node_count == 2_000 and override is not None:
                assert elapsed <= 1.0, "2,000-node budgeted assembly exceeded 1 second"
