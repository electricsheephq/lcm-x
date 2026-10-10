"""Post-sanitize overflow must spend summaries before the user's objective."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.dag import SummaryNode
from hermes_lcm.host_uid_emit import engine_uid
from hermes_lcm.reconcile import _emission_identity, _finalize_emission_descriptors
from hermes_lcm.tokens import count_messages_tokens
from tests.test_active_tool_stubbing import make_engine as make_engine
from tests.test_active_tool_stubbing import tool_pair


SEPARATOR = "\n\n---\n\n"
OBJECTIVE = "Deliver the current user's requested objective: repair the orchard map."
SYSTEM = {"role": "system", "content": "system"}


@pytest.fixture
def main_engine_module(tmp_path):
    """Load the exact main implementation without changing the worktree."""
    source = subprocess.run(
        ["git", "show", "origin/main:engine.py"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = tmp_path / "main_engine.py"
    path.write_text(source, encoding="utf-8")
    name = "hermes_lcm._overflow_main_engine"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def setup_case(engine, *, calls=1, words=100, summaries=True, retained=False, carrier=False):
    engine._pending_context_anchor_messages = [{"role": "user", "content": OBJECTIVE}]
    nodes = []
    if summaries:
        # As in the compression-boundary fixture, real persisted nodes remain
        # independently expandable; no node is condensed into another here.
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
    # Reuse the active-tool fixture's host-valid call shape, deliberately omit
    # its results so only the final sanitizer adds their provider-visible cost.
    tail = tool_pair("missing-0", "unused")[0:1]
    tail[0]["tool_calls"] = [
        {**tail[0]["tool_calls"][0], "id": f"missing-{index}"}
        for index in range(calls)
    ]
    if carrier:
        engine._pending_context_anchor_messages = []
        tail = [{"role": "user", "content": "historical carrier tail", "message_uid": "tail-uid"}, *tail,
                {"role": "user", "content": "fresh user tail"}]
    if carrier or retained:
        engine._store.append_batch(engine._session_id, tail)
    retained_message = {"role": "user", "content": "retained user"} if retained else None
    prefix = SEPARATOR.join([
        *([] if carrier else [engine._build_preserved_objective_summary_part({"role": "user", "content": OBJECTIVE})]),
        *(lcm_engine._summary_part_text(node) for node in nodes),
    ])
    # The early selector can afford every part. Sanitizer-added stubs alone
    # cause the final overflow, exercising the actual defect rather than the
    # earlier budget selector.
    raw = [*([] if carrier else [SYSTEM]), *([retained_message] if retained else []),
           {"role": "assistant" if retained else "user", "content": prefix}, *tail]
    cap = count_messages_tokens(raw)
    kwargs = dict(assembly_cap_override=cap, include_lcm_note=False, retained_user_message=retained_message)
    return nodes, tail, cap, kwargs


def view(result):
    return "\n".join(str(message.get("content", "")) for message in result)


def test_objective_survives_and_shallow_old_part_goes(make_engine, main_engine_module):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    nodes, tail, cap, kwargs = setup_case(engine)
    assemble = (main_engine_module.LCMEngine._assemble_context
                if os.environ.get("LCMX_OVERFLOW_ENGINE_BASELINE") == "1"
                else lcm_engine.LCMEngine._assemble_context)
    result = assemble(engine, SYSTEM, tail, **kwargs)

    assert OBJECTIVE in view(result)
    assert "SHALLOW_OLD" not in view(result)
    assert "SHALLOW_NEW" in view(result)
    assert "DEEPEST" in view(result)
    assert count_messages_tokens(result) <= cap
    assert f"{SEPARATOR}[1 older summary part(s) omitted for space; " in view(result)
    assert len(engine._dag.get_session_nodes(engine._session_id)) == len(nodes)


def test_two_removals_use_inverse_depth_and_age_priority(make_engine):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    _, tail, cap, kwargs = setup_case(engine, calls=14)
    result = engine._assemble_context(SYSTEM, tail, **kwargs)

    assert OBJECTIVE in view(result)
    assert "SHALLOW_OLD" not in view(result)
    assert "SHALLOW_NEW" not in view(result)
    assert "DEEPEST" in view(result)
    assert count_messages_tokens(result) <= cap
    assert "[2 older summary part(s) omitted for space;" in view(result)


def test_objective_is_still_stripped_as_last_resort(make_engine, main_engine_module):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    _, tail, _, kwargs = setup_case(engine, calls=8, summaries=False)
    result = engine._assemble_context(SYSTEM, tail, **kwargs)
    expected = main_engine_module.LCMEngine._assemble_context(engine, SYSTEM, tail, **kwargs)

    assert OBJECTIVE not in view(result)
    assert result == expected


def test_under_cap_output_is_byte_identical_to_main(make_engine, main_engine_module):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    _, tail, _, kwargs = setup_case(engine)
    kwargs["assembly_cap_override"] = 10_000
    result = engine._assemble_context(SYSTEM, tail, **kwargs)
    expected = main_engine_module.LCMEngine._assemble_context(engine, SYSTEM, tail, **kwargs)

    assert json.dumps(result, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_final_node_ids_allow_recall_of_dropped_summary(make_engine, monkeypatch):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    nodes, tail, cap, kwargs = setup_case(engine, words=200)
    query = {"role": "user", "content": "retrieve orchard context"}
    tail = [query, *tail]
    kwargs["assembly_cap_override"] = cap = cap + count_messages_tokens([query])
    engine._config.proactive_recall_enabled = True
    engine._config.embeddings_enabled = True
    hits = [{"node_id": node.node_id, "score": 1.0, "snippet": f"RECALLED_{node.node_id}"}
            for node in nodes]
    monkeypatch.setattr(lcm_engine.lcm_tools, "lcm_recall", lambda *args, **kw: json.dumps({"hits": hits}))
    observed = []
    build_recall = engine._build_proactive_recall_message

    def record_recall(messages, role, active):
        observed.append(set(active))
        return build_recall(messages, role, active)

    monkeypatch.setattr(engine, "_build_proactive_recall_message", record_recall)
    result = engine._assemble_context(SYSTEM, tail, **kwargs)

    assert observed[-1] == {nodes[0].node_id, nodes[2].node_id}
    assert f"RECALLED_{nodes[1].node_id}" in view(result)
    assert f"RECALLED_{nodes[0].node_id}" not in view(result)
    assert count_messages_tokens(result) <= cap
    candidate = engine._pending_emission_candidates[0]
    assert "SHALLOW_OLD" not in candidate["span"]
    assert candidate["full_identity"] == _emission_identity(candidate["row"])
    assert any(candidate["row"] is message for message in result)


@pytest.mark.parametrize("retained,carrier", [(False, False), (False, True), (True, False)])
def test_emission_and_engine_identity_use_final_carrier(make_engine, monkeypatch, retained, carrier):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    monkeypatch.setattr(lcm_engine, "identity_emit_enabled", lambda: True)
    import hermes_lcm.host_uid as host_uid
    monkeypatch.setattr(host_uid, "identity_emit_enabled", lambda: True)
    monkeypatch.setattr(engine, "_host_uid_lineage_key", lambda: ("synthetic-lineage", None))
    _, tail, cap, kwargs = setup_case(engine, retained=retained, carrier=carrier)
    result = engine._assemble_context(None if carrier else SYSTEM, tail, **kwargs)

    if not carrier:
        assert OBJECTIVE in view(result)
    assert "SHALLOW_OLD" not in view(result)
    assert count_messages_tokens(result) <= cap
    candidate = engine._pending_emission_candidates[0]
    row = candidate["row"]
    assert any(row is message for message in result)
    assert candidate["full_identity"] == _emission_identity(row)
    assert "SHALLOW_OLD" not in candidate["span"]
    if not retained:
        prefix = candidate["span"].removesuffix("\n\n") if carrier else row["content"]
        basis = hashlib.sha256(prefix.encode()).hexdigest()
        assert row["message_uid"] == engine_uid("synthetic-lineage", "summary" if carrier else "objective", basis, 0)
        assert candidate["engine_uid"] == row["message_uid"]
    if carrier:
        assert "historical carrier tail" in row["content"]
        assert "tail-uid" in row["_absorbed_message_uids"]
        descriptor = _finalize_emission_descriptors(result, [candidate], engine._emission_binding())[0]
        suffix = "historical carrier tail".encode()
        assert descriptor["suffix_sha256"] == hashlib.sha256(suffix).hexdigest()
        assert descriptor["suffix_length"] == len(suffix)
        assert descriptor["retained_source"]["store_id"] > 0
        engine._ingest_cursor = len(result)
        engine._ingest_cursor_needs_reconcile = False
        engine._record_compress_commit_proof(tail, result)
        assert engine._proof_replay_identity(tail[0], strip_carrier=False) in engine._compress_commit_proof["output_effective"]


def test_notice_is_omitted_when_it_would_break_cap(make_engine):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    _, tail, cap, kwargs = setup_case(engine, calls=3, words=20)
    result = engine._assemble_context(SYSTEM, tail, **kwargs)

    assert OBJECTIVE in view(result)
    assert "SHALLOW_OLD" not in view(result)
    assert "omitted for space" not in view(result)
    assert count_messages_tokens(result) <= cap


def test_nonpersistent_trim_writes_no_emission_candidates(make_engine):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    _, tail, cap, kwargs = setup_case(engine)
    prior = list(getattr(engine, "_pending_emission_candidates", []))
    result = engine._assemble_context(SYSTEM, tail, persist=False, **kwargs)

    assert OBJECTIVE in view(result)
    assert count_messages_tokens(result) <= cap
    assert getattr(engine, "_pending_emission_candidates", []) == prior


@pytest.mark.parametrize("quote_location", ["system", "historical"])
@pytest.mark.parametrize("quote_kind", ["part", "prefix"])
def test_overflow_trims_only_generated_occurrence(make_engine, quote_location, quote_kind):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    nodes, tail, cap, kwargs = setup_case(engine)
    prefix = SEPARATOR.join([
        engine._build_preserved_objective_summary_part({"role": "user", "content": OBJECTIVE}),
        *(lcm_engine._summary_part_text(node) for node in nodes),
    ])
    quote = prefix if quote_kind == "prefix" else lcm_engine._summary_part_text(nodes[-1])
    quoted_message = {"role": "system" if quote_location == "system" else "user",
                      "content": "Quoted historical snapshot:\n" + quote}
    if quote_location == "system":
        system = quoted_message.copy()
        cap += count_messages_tokens([system]) - count_messages_tokens([SYSTEM])
    else:
        system = SYSTEM
        tail = [quoted_message.copy(), *tail]
        cap += count_messages_tokens([quoted_message])
        engine._store.append_batch(engine._session_id, tail)
    kwargs["assembly_cap_override"] = cap

    result = engine._assemble_context(system, tail, **kwargs)

    assert quoted_message in result
    candidate = engine._pending_emission_candidates[0]
    carrier = candidate["row"]
    assert carrier != quoted_message
    assert any(message is carrier for message in result)
    assert OBJECTIVE in carrier["content"]
    assert "SHALLOW_OLD" not in carrier["content"]
    assert "DEEPEST" in carrier["content"]
    assert "[1 older summary part(s) omitted for space;" in carrier["content"]
    assert count_messages_tokens(result) <= cap
    assert candidate["full_identity"] == _emission_identity(carrier)
    descriptor = _finalize_emission_descriptors(result, [candidate], engine._emission_binding())[0]
    assert result[descriptor["output_occurrence"]["index"]] is carrier
    assert descriptor["generated_span_sha256"] == hashlib.sha256(candidate["span"].encode()).hexdigest()
    assert all("_lcm_assembly_generated" not in message for message in result)
