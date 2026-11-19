"""#2883, I2: a summary's source content stays covered when the frontier grows."""

import hashlib
import json
import random
import time

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
from hermes_lcm.dag import SummaryNode
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens
from tests.behaviour.compress_trace_harness import Clock, install_clock
from tests.harness import engine_context, host_call
from tests.test_issue_652_653_no_fragments_bounded_prefix import (
    _depth_0_nodes,
    _pin_lcm_char_counter as _pin_lcm_char_counter,
    _rendered,
    _turn,
)

pytestmark = pytest.mark.issue(2883)
TURN_BODY = "synthetic conversation content " * 64
PART_BODY = "synthetic summary text " * 35
QUIET = dict(temporal_rollups_enabled=False, embeddings_enabled=False,
             large_output_externalization_enabled=False, large_output_active_replay_stubbing_enabled=False)
SYSTEM = {"role": "system", "content": "system prompt"}


def _prefix_tokens(view):
    return sum(count_message_tokens(row) for row in view if _rendered([row]))


def _view_bytes(view):
    return json.dumps(view, ensure_ascii=False, separators=(",", ":")).encode()


def _with_main_reader(engine, monkeypatch, tail, cap=None):
    fetch = engine._dag.get_uncondensed_at_depth
    with monkeypatch.context() as patch:
        patch.setattr(engine._dag, "get_uncondensed_at_depth",
                      lambda session, depth, limit=100, newest=False: fetch(session, depth, limit=100, newest=newest))
        return engine._assemble_context(SYSTEM, tail, cap, include_lcm_note=False, persist=False)


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
    settings = QUIET.copy()
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
        band = (24_000 if ratio is None else int(ratio * window)) + engine._config.leaf_chunk_tokens
        peak = 0
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
            prefix_tokens = _prefix_tokens(view)
            peak = max(peak, prefix_tokens)
            details = (f"window={window} sweep={sweep} depth_0={leaves} rendered={rendered} "
                       f"budget_dropped={sum(budget_drops)} uncovered={uncovered} "
                       f"band={band} prefix={prefix_tokens} peak={peak}")
            print(details)
            assert uncovered == 0, details
            assert prefix_tokens <= band, details
            assert engine._last_threshold_full_sweep["summary_prefix_target_tokens"] + engine._config.leaf_chunk_tokens == band
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


@pytest.mark.parametrize("batch", range(10))
def test_randomized_main_selection(tmp_path, monkeypatch, batch):
    rng = random.Random(2883 + batch)
    oracle = hashlib.sha256()
    observed = hashlib.sha256()
    with engine_context(tmp_path, window=200_000, leaf_chunk_tokens=3000,
                        summary_prefix_target_tokens=8_000, **QUIET) as engine:
        tail = _turn("tail", 5000.0)
        for case in range(20):
            # Include below-cap cases, above-cap cases at several depths, and empty levels.
            nodes = {depth: [SummaryNode(node_id=depth * 300 + index + 1, depth=depth,
                                        summary=rng.choice(["word ", "漢字"]) * rng.randint(1, 45))
                             for index in range(rng.randint(0, 99 if case < 4 else 300))]
                     for depth in range(rng.randint(1, 4))}

            def fetch(session, depth, limit=100, newest=False):
                assert session == "S" and newest
                return nodes[depth] if limit is None else nodes[depth][-limit:]

            order = [(depth, node.node_id) for depth in reversed(nodes) for node in nodes[depth]]
            older_priority = [(depth, node.node_id) for depth in reversed(nodes)
                              for node in reversed(nodes[depth][:-100])]
            full_tokens = count_message_tokens({"role": "user", "content": "\n\n---\n\n".join(
                lcm_engine._summary_part_text(node) for depth in reversed(nodes) for node in nodes[depth])})
            fixed_tokens = count_messages_tokens([SYSTEM, *tail])
            with monkeypatch.context() as patch:
                patch.setattr(engine._dag, "get_session_depths", lambda session: list(nodes))
                patch.setattr(engine._dag, "get_uncondensed_at_depth", fetch)
                for cap in (None, fixed_tokens + 750, fixed_tokens + full_tokens // 2,
                            fixed_tokens + max(0, full_tokens - 100)):
                    baseline = _with_main_reader(engine, patch, tail, cap)
                    view = engine._assemble_context(SYSTEM, tail, cap, include_lcm_note=False, persist=False)
                    baseline_ids, rendered = set(_rendered(baseline)), set(_rendered(view))
                    assert baseline_ids <= rendered
                    assert _rendered(view) == [part for part in order if part in rendered]
                    extras = rendered.intersection(older_priority)
                    assert extras == set(older_priority[:len(extras)])
                    if case < 4:
                        assert _view_bytes(view) == _view_bytes(baseline)
                    if baseline_ids < rendered:
                        limit = 11_000 if cap is None else min(11_000, cap - fixed_tokens)
                        assert _prefix_tokens(view) <= limit
                    oracle.update(_view_bytes(baseline))
                    observed.update(_view_bytes(view))
    print(f"oracle {batch} control={oracle.hexdigest()} actual={observed.hexdigest()}")


@pytest.mark.parametrize("capped", [True, False], ids=["F1-cap", "F1-default"])
def test_deep_extras_do_not_displace_shallow_parts(tmp_path, monkeypatch, capped):
    with engine_context(tmp_path, window=200_000, conversation_id="conv", **QUIET) as engine:
        n_deep, n_shallow = (120, 40) if capped else (150, 50)
        children = _depth_0_nodes(engine, n_deep, text="condensed source")
        for index, child_id in enumerate(children):
            if capped and index < 20:
                body = "large synthetic arc text " * 300
            elif capped:
                body = "small synthetic note " * 8
            else:
                body = "medium synthetic summary words " * 70
            engine._dag.add_node(SummaryNode(session_id="S", depth=1, summary=body,
                                            source_type="nodes", source_ids=[child_id], created_at=float(index)))
        shallow = _depth_0_nodes(engine, n_shallow, text=body)
        tail = [{"role": "user", "content": "tail question", "timestamp": 5000.0}] if capped else [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"tail {i} " + "tail synthetic chat " * 250}
            for i in range(48)]
        cap = 12_000 if capped else None
        baseline = _with_main_reader(engine, monkeypatch, tail, cap)
        view = engine._assemble_context(SYSTEM, tail, cap, include_lcm_note=False, persist=False)
    baseline_ids = set(_rendered(baseline))
    assert len(baseline_ids) == 100 + n_shallow
    assert {(0, node_id) for node_id in shallow} <= baseline_ids <= set(_rendered(view))
    print(f"F1 cap={cap} main_parts={len(baseline_ids)} candidate_parts={len(_rendered(view))}")


def test_stalled_condensation_host_calls(tmp_path, monkeypatch):
    clock = Clock()
    install_clock(monkeypatch, clock)
    provider_calls = 0

    def reject(*args, **kwargs):
        raise lcm_engine.SummaryResultRejected("synthetic stall")

    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        nonlocal provider_calls
        if timeout is not None and timeout < 2.0:
            clock.now += timeout
            raise TimeoutError("synthetic summary timed out")
        provider_calls += 1
        clock.now += 2.0
        return (f"SUMMARY-MARK-{provider_calls}-END " + "synthetic recap words " * 504)[:4000]

    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    rng = random.Random(201_000)
    digest = hashlib.sha256()
    compressed = []
    with engine_context(tmp_path, window=200_000, conversation_id="conv", platform="telegram",
                        leaf_chunk_tokens=3000, **QUIET) as engine:
        monkeypatch.setattr(engine, "_condense_summary_nodes", reject)
        view = [{"role": "system", "content": engine._append_lcm_note_to_content(SYSTEM["content"])}]
        turn = 0
        for call in range(200):
            grow = 2 if call % 3 else 10_000
            pressure = count_messages_tokens(view)
            while grow > 0 and (grow <= 100 or pressure < engine.threshold_tokens + 1000):
                grow -= 1
                size = rng.randint(40, 160)
                pair = [{"role": "user", "content": f"[U{turn}] " + "user words here " * size,
                         "timestamp": 10.0 * (turn + 1)},
                        {"role": "assistant", "content": f"[A{turn}] " + "reply words there " * size}]
                view.extend(pair)
                pressure += count_messages_tokens(pair)
                turn += 1
            result = host_call(engine, view)
            if result.compressed:
                compressed.append(call)
            view = result.view
            digest.update(_view_bytes(view))
            engine.note_turn_complete()
            clock.now += 700.0
        frontier = len(engine._dag.get_session_nodes("S", limit=-1))
    small = [call for call in compressed if call % 3]
    print(f"F2 calls=200 compressed={len(compressed)} small={len(small)} frontier={frontier} "
          f"views={digest.hexdigest()} calls_compressed={compressed}")
    assert frontier > 100
    assert compressed and not small


@pytest.mark.parametrize("window", [0, 200_000], ids=["unknown-window", "known-window"])
def test_summary_band_boundary(tmp_path, monkeypatch, window):
    with engine_context(tmp_path, window=window, leaf_chunk_tokens=400, **QUIET) as engine:
        ids = _depth_0_nodes(engine, 101, text=PART_BODY)
        parts = [lcm_engine._summary_part_text(node) for node in engine._dag.get_session_nodes("S", limit=-1)]
        full = count_message_tokens({"role": "user", "content": "\n\n---\n\n".join(parts)})
        tail = _turn("tail", 5000.0)
        baseline = _with_main_reader(engine, monkeypatch, tail)
        if window == 0:
            assert engine._assemble_context(SYSTEM, tail, include_lcm_note=False, persist=False) == baseline
        for band in (full - 1, full, full + 1, _prefix_tokens(baseline) - 1):
            monkeypatch.setattr(engine._config, "summary_prefix_target_tokens", band - 400)
            view = engine._assemble_context(SYSTEM, tail, include_lcm_note=False, persist=False)
            if band >= full:
                assert _prefix_tokens(view) == full
                assert _rendered(view) == [(0, node_id) for node_id in ids]
            else:
                assert _view_bytes(view) == _view_bytes(baseline)


def test_mixed_script_band_uses_exact_joined_count(tmp_path, monkeypatch):
    with engine_context(tmp_path, window=200_000, leaf_chunk_tokens=400, **QUIET) as engine:
        for index, body in enumerate(["oldest", "a" * 5_000, "漢" * 9_000] + ["seed"] * 100, start=1):
            engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=body, created_at=float(index)))
        parts = [lcm_engine._summary_part_text(node) for node in engine._dag.get_session_nodes("S", limit=-1)]
        counts = [count_message_tokens({"role": "user", "content": "\n\n---\n\n".join(parts[-100 - k:])})
                  for k in range(4)]
        assert counts[0] < counts[2] <= counts[3] < counts[1]
        tail = _turn("tail", 5000.0)
        for band, extras in ((9500, 3), (8000, 0)):
            monkeypatch.setattr(engine._config, "summary_prefix_target_tokens", band - 400)
            view = engine._assemble_context(SYSTEM, tail, include_lcm_note=False, persist=False)
            assert _rendered(view) == [(0, node_id) for node_id in range(4 - extras, 104)]
            assert _prefix_tokens(view) == counts[extras] <= band
        print(f"mixed-script prefix counts={counts} bands=9500,8000")


@pytest.mark.parametrize("node_count", [2_000, 10_000])
def test_older_summary_backoff_cost(tmp_path, monkeypatch, node_count):
    with engine_context(tmp_path, window=0, leaf_chunk_tokens=400, **QUIET) as engine:
        _depth_0_nodes(engine, node_count, token_count=200, text=PART_BODY)
        tail = _turn("tail", 5000.0)
        parts = [lcm_engine._summary_part_text(node) for node in engine._dag.get_session_nodes("S", limit=-1)]
        full = count_message_tokens({"role": "user", "content": "\n\n---\n\n".join(parts)})
        baseline = _with_main_reader(engine, monkeypatch, tail)
        cap = full + count_messages_tokens([SYSTEM, *tail])
        for mode, band, expected in (("first", full, node_count), ("exhausted", _prefix_tokens(baseline), 100)):
            monkeypatch.setattr(engine._config, "summary_prefix_target_tokens", band - 400)
            for budget in (None, cap):
                started = time.perf_counter()
                view = engine._assemble_context(SYSTEM, tail, budget, include_lcm_note=False, persist=False)
                elapsed = time.perf_counter() - started
                print(f"r9 assembly nodes={node_count} mode={mode} budget={budget is not None} "
                      f"ms={elapsed * 1000:.3f}")
                assert len(_rendered(view)) == expected
                if node_count == 2_000 and budget is not None:
                    assert elapsed <= 1.0, "2,000-node budgeted assembly exceeded 1 second"
